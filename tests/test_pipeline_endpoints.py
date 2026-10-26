"""Real tiny native endpoint stages and sealed-handoff failure gates."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import weakref

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai.adapter import tiny_model
from openbonsai import pipeline_endpoints as stages
from openbonsai import pipeline_sources as sources
from openbonsai.embedding_calibration import EmbeddingTrainingConfig
from openbonsai.eval_backends import _byte_decoder, local_directory_identity
from openbonsai.head_calibration import HeadTrainingConfig
from openbonsai.mixed_formats import MixedEmbedding, MixedLinear, candidate_nbytes
from openbonsai.rotations import fixed_signs
from openbonsai.trainer import TrainingConfig


def recipes():
    quantization = dict(group_size=128, row_chunk_size=31, block_size=64, lora_rank=2,
        factor_mapping='sigmoid', hard_phase='frozen_jacobian', soft_fraction=.8,
        max_sharpness=30., rotation_seed=41, initialization_seed=42)
    head = dict(quantization=copy.deepcopy(quantization), packing='auto',
        training=asdict(HeadTrainingConfig(epochs=1, batch_size=1, token_microbatch=2,
            vocabulary_chunk=31, learning_rate=.005, lora_learning_rate=.001)),
        cache={'max_positions': None, 'shard_positions': 2})
    embedding = dict(quantization=copy.deepcopy(quantization), packing='auto',
        training=asdict(EmbeddingTrainingConfig(steps=2)))
    return head, embedding


def make_context(root, *, multimodal=True, head=None, embedding=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = tiny_model(seed=112, multimodal=multimodal).bfloat16().eval().requires_grad_(False)
    with torch.no_grad():
        model.get_input_embeddings().weight[0].fill_(.125)  # Padding row semantics matter.
    model.save_pretrained(root / 'source')
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(_byte_decoder())}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(root / 'tokenizer')
    source = local_directory_identity(root / 'source')
    tokenizer = sources.tokenizer_identity(root / 'tokenizer')
    traces = {}
    for role, start in [('calibration', 4), ('search_validation', 40), ('evaluation', 80)]:
        path = root / (role + '.jsonl')
        rows = [{'input_ids': [start, start + 1, start + 2, 2, 0], 'attention_mask': [1, 1, 1, 1, 0]},
                {'input_ids': [start + 3, start + 4, start + 5, start + 6, 2]}]
        path.write_text(''.join(json.dumps(value) + '\n' for value in rows))
        producer = root / (role + '.producer.json')
        producer.write_text(json.dumps({'kind': 'authored-random-fixture',
            'description': 'Authored token fixture, not generated pretrained data or quality evidence',
            'source_content_sha256': source['sha256'], 'tokenizer_content_sha256': tokenizer['sha256'],
            'source_jsonl_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
        traces[role] = {'path': str(path), 'sequence_length': 5, 'max_samples': None,
            'producer': {'path': str(producer), 'sha256': hashlib.sha256(producer.read_bytes()).hexdigest()}}
    defaults = recipes()
    decoder_quantization = {key: value for key, value in defaults[0]['quantization'].items()
                            if key not in {'row_chunk_size', 'initialization_seed'}}
    decoder_quantization['aggressive_gates'] = False
    context = sources.seal_job(model_directory=root / 'source', tokenizer_directory=root / 'tokenizer',
        traces_by_role=traces, decoder={'quantization': decoder_quantization, 'packing': 'auto',
            'training': asdict(TrainingConfig(epochs=1, batch_size=1, window_size=1, stride=1))},
        head=defaults[0] if head is None else head, embedding=defaults[1] if embedding is None else embedding,
        assembly={'coverage': 'all', 'output_scope': 'language-only', 'budget_bytes': 10**8,
                  'row_chunk_size': 31, 'cache_decoded_weights': False}, output=root / 'job')
    del model
    return context


@pytest.mark.parametrize('multimodal', [False, True])
def test_real_cache_head_fit_and_full_embedding_fit_publish_bound_candidates(tmp_path, multimodal):
    context = make_context(tmp_path, multimodal=multimodal)
    before = local_directory_identity(context.job['source']['model_directory']['path'])
    events, operations = [], []
    def operation(name, function):
        operations.append(name)
        return function()
    cache_receipts = {}
    for role in sources.ROLES:
        cache_receipts[role] = stages.cache_head_stage(context, role, tmp_path / role,
            operation=operation, progress_callback=events.append)
        cache, receipt = stages.load_head_cache_stage(context, tmp_path / role, role=role)
        assert receipt == cache_receipts[role]
        assert cache.manifest['source_provenance'] == context.source_provenance
        assert cache.positions == 7
    head_receipt = stages.fit_head_stage(context, tmp_path / 'calibration', tmp_path / 'head',
        search_cache=tmp_path / 'search_validation', evaluation_cache=tmp_path / 'evaluation',
        operation=operation, progress_callback=events.append)
    embedding_receipt = stages.fit_embedding_stage(context, tmp_path / 'embedding',
        operation=operation, progress_callback=events.append)
    assert head_receipt['report']['evaluation_cache_used_for_scoring'] is False
    assert head_receipt['report']['search_validation']['role'] == 'search_validation'
    assert head_receipt['report']['cache_stage_sha256s'] == {role: row['sha256'] for role, row in cache_receipts.items()}
    assert len(head_receipt['report']['steps']) == 2
    assert len(head_receipt['report']['final_hard_state_sha256']) == 64
    assert [row['phase'] for row in embedding_receipt['report']['steps']] == ['soft', 'hard']
    assert embedding_receipt['report']['text_records_used_for_fitting'] == 0
    prefix = 'model.language_model' if multimodal else 'model'
    for stage, name in [('head', 'lm_head'), ('embedding', prefix + '.embed_tokens')]:
        mapping, receipt = sources.load_candidate_stage(context, tmp_path / stage, expected_stage=stage)
        assert set(mapping) == {name}
        candidate = mapping[name]
        provenance = candidate.metadata['provenance']
        assert provenance['original_module_name'] == name
        assert provenance['pipeline_job_sha256'] == context.job_sha256
        assert provenance['source_content_sha256'] == before['sha256']
        assert provenance['training_report_sha256'] == sources.json_sha256(receipt['report'])
        assert torch.equal(candidate.tensors['rotation_signs'], fixed_signs(128, seed=41, name=name))
        assert candidate_nbytes(candidate) == receipt['candidates'][name]['bytes']
        assert not {'weight', 'lora_A', 'lora_B'} & candidate.tensors.keys()
        if stage == 'embedding':
            certificate = provenance['calibration_certificate']
            assert certificate == receipt['report']['coverage_certificate']
            assert certificate['source_shape'] == [256, 128]
            assert all(certificate['coverage'][key] == 256 for key in
                       ['soft_rows', 'hard_rows', 'last_soft_rows', 'final_hard_rows'])
            loaded = MixedEmbedding(candidate, padding_idx=0, row_chunk_size=31)
            rows = loaded(torch.tensor([0, 7, 7, 255]))
            assert torch.equal(rows[1], rows[2]) and torch.isfinite(rows).all()
            assert rows[0].count_nonzero() > 0
        else:
            loaded = MixedLinear(candidate, row_chunk_size=31)
            assert loaded(torch.ones(1, 2, 128, dtype=torch.bfloat16)).shape == (1, 2, 256)
    assert local_directory_identity(context.job['source']['model_directory']['path']) == before
    assert {'move-only-head', 'move-only-embedding', 'export-head-candidate', 'export-embedding-candidate'} <= set(operations)
    assert not any('evaluate' in label and 'evaluation' in label for label in operations)
    assert any(event.get('phase') == 'hard' for event in events)


def test_decoder_is_released_before_endpoint_device_transfer(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    real_load = sources.load_original
    observed = []
    def load(context, device='cpu'):
        assert device == 'cpu'
        model = real_load(context, device=device)
        observed.append(weakref.ref(model.model.language_model.layers[0]))
        return model
    monkeypatch.setattr(sources, 'load_original', load)
    def operation(name, function):
        if name == 'move-only-embedding':
            assert observed and observed[-1]() is None
        return function()
    stages.fit_embedding_stage(context, tmp_path / 'embedding', operation=operation)
    assert observed[0]() is None


@pytest.mark.parametrize('change', ['missing-quantization', 'missing-training', 'missing-cache', 'seed-bool'])
def test_missing_or_invalid_explicit_recipe_rejected_before_model_load(tmp_path, monkeypatch, change):
    head, embedding = recipes()
    if change == 'missing-quantization':
        del head['quantization']['lora_rank']
    elif change == 'missing-training':
        del head['training']['weight_decay']
    elif change == 'missing-cache':
        del head['cache']['max_positions']
    else:
        head['quantization']['initialization_seed'] = True
    monkeypatch.setattr(sources, 'load_original', lambda *args, **kwargs: pytest.fail('must fail before model load'))
    with pytest.raises(ValueError):
        context = make_context(tmp_path, head=head, embedding=embedding)
        stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')


def test_cache_position_limit_and_order_are_bound(tmp_path):
    head, embedding = recipes()
    head['cache']['max_positions'] = 4
    context = make_context(tmp_path, head=head, embedding=embedding)
    receipt = stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')
    assert receipt['record_binding']['positions'] == 4
    assert [row['positions'] for row in receipt['record_binding']['records']] == [3, 1]
    with pytest.raises(ValueError, match='binding'):
        stages.load_head_cache_stage(context, tmp_path / 'cache', role='search_validation')
    path = tmp_path / 'cache' / 'stage.json'
    receipt['record_binding']['records'].reverse()
    receipt['sha256'] = sources.json_sha256({k: v for k, v in receipt.items() if k != 'sha256'})
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='accounting'):
        stages.load_head_cache_stage(context, tmp_path / 'cache', role='calibration')


def test_changed_in_memory_records_or_cached_files_cannot_fit(tmp_path):
    context = make_context(tmp_path)
    stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')
    context.records_by_role['calibration'][0]['input_ids'][0, 0] += 1
    with pytest.raises(ValueError, match='record'):
        stages.fit_head_stage(context, tmp_path / 'cache', tmp_path / 'fit')
    context = sources.load_context(context.job_path)
    (tmp_path / 'cache' / 'cache' / 'extra').write_text('unexpected')
    with pytest.raises(ValueError, match='files changed'):
        stages.fit_head_stage(context, tmp_path / 'cache', tmp_path / 'fit')
    assert not (tmp_path / 'fit').exists()


@pytest.mark.parametrize('stage', ['head', 'embedding'])
def test_interrupted_fit_does_not_publish_completed_candidate(tmp_path, stage):
    context = make_context(tmp_path)
    if stage == 'head':
        stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')
    source = local_directory_identity(context.job['source']['model_directory']['path'])
    def fail(event):
        if 'step' in event:
            raise RuntimeError('stop-after-one-update')
    with pytest.raises(RuntimeError, match='stop-after-one-update'):
        if stage == 'head':
            stages.fit_head_stage(context, tmp_path / 'cache', tmp_path / 'fit', progress_callback=fail)
        else:
            stages.fit_embedding_stage(context, tmp_path / 'fit', progress_callback=fail)
    assert not (tmp_path / 'fit').exists()
    assert local_directory_identity(context.job['source']['model_directory']['path']) == source


def test_partial_cache_and_existing_destination_are_refused(tmp_path):
    context = make_context(tmp_path)
    def stop(event):
        raise RuntimeError('intentional-record-stop')
    with pytest.raises(RuntimeError, match='intentional-record-stop'):
        stages.cache_head_stage(context, 'calibration', tmp_path / 'cache', progress_callback=stop)
    assert not (tmp_path / 'cache' / 'stage.json').exists()
    with pytest.raises(FileNotFoundError):
        stages.load_head_cache_stage(context, tmp_path / 'cache', role='calibration')
    with pytest.raises(FileExistsError):
        stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')


def test_candidate_recipe_is_bound_before_actual_byte_selection(tmp_path):
    head, embedding = recipes()
    embedding['packing'] = 'base3'
    context = make_context(tmp_path, head=head, embedding=embedding)
    receipt = stages.fit_embedding_stage(context, tmp_path / 'fit')
    mapping, _ = sources.load_candidate_stage(context, tmp_path / 'fit')
    candidate = next(mapping[name] for name in mapping)
    assert candidate.encoding == 'ternary-base3'
    assert candidate.metadata['provenance']['pipeline_stage_config_sha256'] == sources.json_sha256(embedding)
    path = tmp_path / 'fit' / next(iter(receipt['candidates'].values()))['file']
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        sources.load_candidate_stage(context, tmp_path / 'fit')


@pytest.mark.parametrize('phase', ['before-export', 'before-publication'])
def test_final_head_state_changes_cannot_publish_success(tmp_path, monkeypatch, phase):
    context = make_context(tmp_path)
    stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')
    construct, owned = stages._construct, []
    def capture(*args, **kwargs):
        operator = construct(*args, **kwargs)
        owned.append(operator)
        return operator
    monkeypatch.setattr(stages, '_construct', capture)
    def mutation(event):
        if event.get('phase') == phase:
            owned[0].raw_scale.data.add_(.01)
    with pytest.raises(ValueError, match='Final head state changed'):
        stages.fit_head_stage(context, tmp_path / 'cache', tmp_path / 'fit', progress_callback=mutation)
    assert not (tmp_path / 'fit').exists()


@pytest.mark.parametrize('stage', ['cache', 'head', 'embedding'])
def test_output_inside_sealed_source_is_rejected_before_model_work(tmp_path, monkeypatch, stage):
    context = make_context(tmp_path)
    output = Path(context.job['source']['model_directory']['path']) / 'forbidden-output'
    before = local_directory_identity(output.parent)
    monkeypatch.setattr(sources, 'load_original', lambda *args, **kwargs: pytest.fail('must reject before loading'))
    with pytest.raises(ValueError, match='outside sealed'):
        if stage == 'cache':
            stages.cache_head_stage(context, 'calibration', output)
        elif stage == 'head':
            stages.fit_head_stage(context, tmp_path / 'missing-cache', output)
        else:
            stages.fit_embedding_stage(context, output)
    assert not output.exists()
    assert local_directory_identity(output.parent) == before
