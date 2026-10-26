"""Actual supervised CPU chain; random native fixtures establish no 9B quality."""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai.adapter import tiny_model
from openbonsai import pipeline, pipeline_sources as sources
from openbonsai.embedding_calibration import EmbeddingTrainingConfig
from openbonsai.eval_backends import _byte_decoder, local_directory_identity
from openbonsai.head_calibration import HeadTrainingConfig
from openbonsai.mixed_export import load_mixed
from openbonsai.mixed_formats import MixedEmbedding, MixedLinear
from openbonsai.text_export import text_module_name
from openbonsai.trainer import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def invoke(root, name, stage, parameters, *, expected_success=True):
    request = write_json(root / (name + '.request.json'),
        {'format': pipeline.REQUEST_FORMAT, 'stage': stage, 'parameters': parameters})
    output = root / name
    command = [sys.executable, '-m', 'openbonsai', 'pipeline', '--run',
        '--request', str(request), '--expected-request-sha256', sha(request),
        '--output', str(output), '--device', 'cpu', '--threads', '2',
        '--max-seconds', '120', '--minimum-free-cuda-bytes', '0']
    process = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=150)
    # Retain subprocess evidence independently of its own stage completion.
    (root / (name + '.stdout.txt')).write_text(process.stdout)
    (root / (name + '.stderr.txt')).write_text(process.stderr)
    result = read_json(output / 'result.json') if (output / 'result.json').exists() else None
    if expected_success:
        child = read_json(output / 'child-status.json') if (output / 'child-status.json').exists() else None
        assert process.returncode == 0, (process.stdout, process.stderr, result, child)
        assert result['status'] == 'complete' and result['quality_claim'] is False
        assert result['supervision']['status'] == 'complete'
        assert result['supervision']['exit_code'] == 0
        assert result['supervision']['cleanup']['no_live_owned_processes'] is True
        assert child['status'] == 'complete'
        assert child['environment']['device'] == 'cpu' and child['environment']['threads'] == 2
        assert result['request'] == reference(request)
        assert (output / 'request.json').read_bytes() == request.read_bytes()
        assert result['code'] == sources.code_identity()
        ref = reference(output / 'result.json')
        payload, _ = pipeline.completed_stage(ref, stage)
        assert result['payload']['bytes'] == payload.stat().st_size
        return ref
    assert process.returncode != 0
    assert result is None or result['status'] == 'failed'
    if result is not None:
        assert result['supervision']['cleanup']['no_live_owned_processes'] is True
    return output, process, result


def explicit_recipes():
    q = dict(group_size=128, row_chunk_size=31, block_size=64, lora_rank=2,
             factor_mapping='sigmoid', hard_phase='frozen_jacobian', soft_fraction=.8,
             max_sharpness=30., rotation_seed=41, initialization_seed=42)
    decoder_q = {key: value for key, value in q.items()
                 if key not in {'row_chunk_size', 'initialization_seed'}}
    decoder_q['aggressive_gates'] = False
    return dict(
        decoder={'quantization': decoder_q, 'packing': 'auto',
            'training': asdict(TrainingConfig(epochs=1, batch_size=1, window_size=4,
                stride=4, teacher_prefix='original', seed=42))},
        head={'quantization': copy.deepcopy(q), 'packing': 'auto',
            'training': asdict(HeadTrainingConfig(epochs=1, batch_size=1,
                token_microbatch=8, vocabulary_chunk=31)),
            'cache': {'max_positions': None, 'shard_positions': 8}},
        embedding={'quantization': copy.deepcopy(q), 'packing': 'auto',
            'training': asdict(EmbeddingTrainingConfig(steps=2))},
        assembly={'coverage': 'all', 'output_scope': 'language-only',
            'budget_bytes': 10**8, 'row_chunk_size': 31, 'cache_decoded_weights': False})


@pytest.fixture(scope='module')
def full_chain(tmp_path_factory):
    root = tmp_path_factory.mktemp('supervised-pipeline')
    torch.set_num_threads(2)
    native = tiny_model(seed=112, multimodal=True).bfloat16().eval().requires_grad_(False)
    byte = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(_byte_decoder())}, merges=[]))
    byte.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    byte.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=byte,
        pad_token='<pad>', eos_token='<eos>', additional_special_tokens=['<eot>'])
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
    # Every generated vocabulary ID has a real tokenizer entry. Media sentinel
    # IDs lie outside this text-only random fixture's vocabulary.
    with torch.random.fork_rng():
        torch.manual_seed(113)
        native.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<eot>')]
    native.config.text_config.pad_token_id = tokenizer.pad_token_id
    native.config.text_config.eos_token_id = eos
    native.get_input_embeddings().padding_idx = tokenizer.pad_token_id
    native.generation_config.pad_token_id = tokenizer.pad_token_id
    native.generation_config.eos_token_id = eos
    native.generation_config.use_cache = False
    for offset, key in enumerate(('image_token_id', 'video_token_id',
                                  'vision_start_token_id', 'vision_end_token_id')):
        setattr(native.config, key, len(tokenizer) + 10 + offset)
    # Preserve a real saved native FP32 exception with non-BF16 values. The
    # source loader and every control must preserve it, not merely its shape.
    for name, parameter in native.named_parameters():
        if name.endswith('A_log'):
            parameter.data = parameter.data.float().fill_(.1234567)
    native.eval().requires_grad_(False).save_pretrained(root / 'source')
    tokenizer.save_pretrained(root / 'tokenizer')
    before = local_directory_identity(root / 'source')
    tokenizer_before = local_directory_identity(root / 'tokenizer')
    prepared, prompts = {}, {}
    for role, values in [('calibration', ['two', 'üç']), ('search_validation', ['four', 'beş'])]:
        prompt_path = root / (role + '.prompts.jsonl')
        prompt_path.write_text(''.join(json.dumps({'prompt': text}, ensure_ascii=False) + '\n' for text in values))
        prompts[role] = {'file': prompt_path, 'sha256': sha(prompt_path), 'values': values}
        prepared[role] = invoke(root, 'prepare-' + role, 'prepare', {
            'model_directory': str(root / 'source'), 'expected_model_sha256': before['sha256'],
            'tokenizer_directory': str(root / 'tokenizer'),
            'expected_tokenizer_sha256': sources.tokenizer_identity(root / 'tokenizer')['sha256'],
            'prompts': str(prompt_path), 'expected_prompts_sha256': sha(prompt_path), 'role': role,
            'generation': {'seed': 43, 'max_new_tokens': 2, 'max_prompt_tokens': 64,
                'max_samples': 2, 'enable_thinking': False, 'do_sample': False,
                'temperature': 1., 'top_p': 1.}})
    recipes = explicit_recipes()
    sealed = invoke(root, 'seal', 'seal', {
        'model_directory': str(root / 'source'), 'tokenizer_directory': str(root / 'tokenizer'),
        'traces_by_role': {role: {'prepared': ref, 'sequence_length': 64, 'max_samples': None}
                          for role, ref in prepared.items()}, **recipes})
    decoder = invoke(root, 'decoder', 'decoder', {'job': sealed})
    caches = {role: invoke(root, 'cache-' + role, 'cache-head', {'job': sealed, 'role': role})
              for role in prepared}
    head = invoke(root, 'head', 'head', {'job': sealed,
        'calibration_cache': caches['calibration'], 'search_cache': caches['search_validation'],
        'evaluation_cache': None})
    embedding = invoke(root, 'embedding', 'embedding', {'job': sealed})
    stages = [decoder, head, embedding]
    assembled = invoke(root, 'assemble', 'assemble', {'job': sealed, 'candidate_stages': stages})
    return dict(root=root, native=native, tokenizer=tokenizer, before=before,
        tokenizer_before=tokenizer_before, prepared=prepared, prompts=prompts, job=sealed,
        stages=stages, assembled=assembled, recipes=recipes, eos=eos)


def test_supervised_generated_records_and_two_update_full_coverage(full_chain):
    c, seen = full_chain, {}
    for role, ref in c['prepared'].items():
        path, producer = pipeline.completed_stage(ref, 'prepare')
        records = [json.loads(line) for line in (path.parent / 'traces.jsonl').read_text().splitlines()]
        assert producer['kind'] == 'openbonsai-original-ayot.v1'
        assert producer['counts']['records'] == len(records) == 2
        assert producer['source_content_sha256'] == c['before']['sha256']
        assert producer['file']['sha256'] == sha(path.parent / 'traces.jsonl')
        seen[role] = set()
        for record, prompt in zip(records, c['prompts'][role]['values']):
            ids = c['tokenizer'].apply_chat_template([{'role': 'user', 'content': prompt}],
                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            assert record['prompt'] == prompt and record['input_ids'][:len(ids)] == ids
            assert record['prompt_token_count'] == len(ids)
            assert 1 <= record['generated_token_count'] <= 2
            assert record['attention_mask'] == [1] * len(record['input_ids'])
            # Independent replay proves actual original native greedy generation,
            # not merely receipt fields or hand-written token fixtures.
            with torch.no_grad():
                c['native'].model.rope_deltas = None
                replay = c['native'].generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    max_new_tokens=2, do_sample=False, use_cache=False,
                    eos_token_id=c['eos'], pad_token_id=c['tokenizer'].pad_token_id)
            assert replay[0].tolist() == record['input_ids']
            seen[role].add(tuple(record['input_ids']))
    assert not seen['calibration'] & seen['search_validation']
    job_path, job = pipeline.completed_stage(c['job'], 'seal')
    context = sources.load_context(job_path)
    for role in seen:
        receipt = context.record_receipts_by_role[role]
        assert receipt['truncated_tokens'] == 0 and receipt['padding_tokens'] > 0
        assert receipt['num_records'] == 2
    for key, recipe in c['recipes'].items():
        assert job[key] == recipe
    decoder = read_json(Path(c['stages'][0]['path']).parent / 'stage' / 'stage.json')
    assert len(decoder['candidates']) == 25
    assert len(decoder['report']['windows']) == 1
    assert decoder['report']['windows'][0]['steps'] == 2
    head = read_json(Path(c['stages'][1]['path']).parent / 'stage' / 'stage.json')
    assert len(head['report']['steps']) == 2
    assert head['report']['search_validation']['role'] == 'search_validation'
    embedding = read_json(Path(c['stages'][2]['path']).parent / 'stage' / 'stage.json')
    certificate = embedding['report']['coverage_certificate']
    assert [row['phase'] for row in embedding['report']['steps']] == ['soft', 'hard']
    assert all(certificate['coverage'][key] == len(c['tokenizer'])
               for key in ('soft_rows', 'hard_rows', 'last_soft_rows', 'final_hard_rows'))


def test_all27_actual_bytes_independent_mm_composition_and_fresh_process_reload(full_chain):
    c = full_chain
    job_path, _ = pipeline.completed_stage(c['job'], 'seal')
    context = sources.load_context(job_path)
    assembled_path, report = pipeline.completed_stage(c['assembled'], 'assemble')
    artifact = assembled_path.parent / 'artifact'
    manifest = read_json(artifact / 'manifest.json')
    assert report['module_count'] == len(manifest['modules']) == 27
    assert report['candidate_payloads_unchanged'] is True and report['quality_claim'] is False
    total = sum(path.stat().st_size for path in artifact.rglob('*') if path.is_file())
    assert int(manifest['storage']['artifact_bytes']) == total
    assert report['storage'] == manifest['storage']
    assert int(manifest['storage']['vision_file_bytes']) == 0
    assert not (artifact / 'stage.json').exists()
    original = sources.load_original(context)
    for ref in c['stages']:
        stage = Path(ref['path']).parent / 'stage'
        candidates, receipt = sources.load_candidate_stage(context, stage)
        for name in candidates:
            candidate, previous = candidates[name], original.get_submodule(name)
            replacement = (MixedEmbedding(candidate, padding_idx=previous.padding_idx, row_chunk_size=31)
                           if candidate.module_kind == 'embedding' else MixedLinear(candidate, row_chunk_size=31))
            original.set_submodule(name, replacement)
            target = text_module_name(name)
            assert report['original_to_artifact_module_names'][name] == target
            assert (artifact / manifest['modules'][target]['file']).read_bytes() == (
                stage / receipt['candidates'][name]['file']).read_bytes()
            assert candidate.metadata['provenance']['source_content_sha256'] == c['before']['sha256']
    ids = c['tokenizer'].apply_chat_template([{'role': 'user', 'content': 'fresh'}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    sample = {'input_ids': torch.tensor([ids + [c['tokenizer'].pad_token_id] * 3]),
              'attention_mask': torch.tensor([[1] * len(ids) + [0] * 3])}
    restored = load_mixed(artifact)
    with torch.no_grad():
        expected = original(**sample, use_cache=False).logits
        actual = restored(**sample, use_cache=False).logits
    assert torch.equal(expected, actual)
    assert restored.config.model_type == 'qwen3_5_text'
    assert restored.generation_config.eos_token_id == c['eos']
    sample_file = c['root'] / 'fresh-sample.pt'
    torch.save(sample, sample_file)
    fresh_file = c['root'] / 'fresh-logits.pt'
    script = '''import sys, torch
from openbonsai.mixed_export import load_mixed
torch.set_num_threads(2)
model = load_mixed(sys.argv[1])
sample = torch.load(sys.argv[2], weights_only=True)
with torch.no_grad():
    logits = model(**sample, use_cache=False).logits
torch.save({'logits': logits, 'eos': model.generation_config.eos_token_id}, sys.argv[3])
'''
    fresh = subprocess.run([sys.executable, '-c', script, str(artifact), str(sample_file), str(fresh_file)],
        cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60)
    assert fresh.returncode == 0, fresh.stderr
    result = torch.load(fresh_file, weights_only=True)
    assert torch.equal(expected, result['logits']) and result['eos'] == c['eos']
    assert local_directory_identity(c['root'] / 'source') == c['before']
    assert local_directory_identity(c['root'] / 'tokenizer') == c['tokenizer_before']
    assert all(sha(row['file']) == row['sha256'] for row in c['prompts'].values())


def test_failed_outer_cannot_reuse_an_existing_valid_inner_stage(full_chain):
    c = full_chain
    real_path = Path(c['stages'][0]['path'])
    failed = read_json(real_path)
    assert Path(failed['payload']['path']).is_file()
    failed['status'] = 'failed'
    # Same owned directory and untouched valid candidate payload: completion is
    # rejected because the *outer* resource/supervision gate failed.
    failed_path = write_json(real_path.with_name('failed-result.json'), failed)
    bad_ref = reference(failed_path)
    with pytest.raises(ValueError, match='complete, cleanly supervised'):
        pipeline.completed_stage(bad_ref, 'decoder')
    out, process, result = invoke(c['root'], 'reject-failed-outer', 'assemble',
        {'job': c['job'], 'candidate_stages': [bad_ref, *c['stages'][1:]]}, expected_success=False)
    assert 'complete, cleanly supervised' in process.stderr
    assert not out.exists() and result is None


def test_changed_candidate_payload_fails_supervised_assembly(full_chain):
    c = full_chain
    original_attempt = Path(c['stages'][2]['path']).parent
    altered = c['root'] / 'altered-embedding-attempt'
    shutil.copytree(original_attempt, altered)
    result = read_json(altered / 'result.json')
    result['payload']['path'] = str(altered / 'stage' / 'stage.json')
    write_json(altered / 'result.json', result)
    changed_ref = reference(altered / 'result.json')
    # The stage index and outer receipt still agree. Its actual candidate bytes
    # no longer do, so the consuming child must fail without an artifact.
    path, index = pipeline.completed_stage(changed_ref, 'embedding')
    candidate_path = path.parent / next(iter(index['candidates'].values()))['file']
    values = bytearray(candidate_path.read_bytes())
    values[-1] ^= 1
    candidate_path.write_bytes(values)
    out, _, failed = invoke(c['root'], 'reject-changed-candidate', 'assemble',
        {'job': c['job'], 'candidate_stages': [*c['stages'][:2], changed_ref]}, expected_success=False)
    assert failed is not None and failed['status'] == 'failed'
    assert not (out / 'stage' / 'artifact' / 'manifest.json').exists()
    child = read_json(out / 'child-status.json')
    assert 'SHA256 mismatch' in child['error']
    assert local_directory_identity(c['root'] / 'source') == c['before']


def test_changed_inner_handoff_hash_is_rejected_before_child_launch(full_chain):
    c = full_chain
    original_attempt = Path(c['stages'][1]['path']).parent
    altered = c['root'] / 'altered-head-attempt'
    shutil.copytree(original_attempt, altered)
    result = read_json(altered / 'result.json')
    result['payload']['path'] = str(altered / 'stage' / 'stage.json')
    write_json(altered / 'result.json', result)
    with (altered / 'stage' / 'stage.json').open('ab') as stream:
        stream.write(b' ')
    out, process, result = invoke(c['root'], 'reject-changed-handoff', 'assemble',
        {'job': c['job'], 'candidate_stages': [c['stages'][0], reference(altered / 'result.json'), c['stages'][2]]},
        expected_success=False)
    assert result is None and not out.exists()
    assert 'SHA256 mismatch' in process.stderr


def test_supervised_endpoint_control_reuses_sealed_job_and_fresh_reload(full_chain):
    c = full_chain
    job_path, sealed_job = pipeline.completed_stage(c['job'], 'seal')
    job_bytes = job_path.read_bytes()
    producer_bytes = {ref['path']: Path(ref['path']).read_bytes() for ref in c['stages']}
    control = {'quantized_components': ['embedding'],
               'retained_original_components': ['decoder', 'head'],
               'output_scope': 'language-only', 'evaluation_label': 'E-only',
               'budget_bytes': 100_000_000}
    assembled = invoke(c['root'], 'embedding-control', 'assemble',
        {'job': c['job'], 'candidate_stages': [c['stages'][2]], 'assembly_control': control})
    path, report = pipeline.completed_stage(assembled, 'assemble')
    artifact = path.parent / 'artifact'
    manifest = read_json(artifact / 'manifest.json')
    assert report['coverage'] == 'control' and report['delivery_target_complete'] is False
    assert report['pipeline_job_sha256'] == sealed_job['job_sha256']
    assert report['sealed_pipeline_assembly'] == sealed_job['assembly']
    assert report['assembly_control'] == control
    assert set(manifest['modules']) == {'model.embed_tokens'}
    context = sources.load_context(job_path)
    expected = sources.load_original(context)
    retained_fp32 = {text_module_name(name): parameter.detach().clone()
                     for name, parameter in expected.named_parameters() if name.endswith('A_log')}
    assert retained_fp32 and all(value.dtype == torch.float32 for value in retained_fp32.values())
    assert all(not torch.equal(value, value.bfloat16().float()) for value in retained_fp32.values())
    stage = Path(c['stages'][2]['path']).parent / 'stage'
    candidates, receipt = sources.load_candidate_stage(context, stage)
    name = next(iter(candidates))
    source = expected.get_input_embeddings()
    expected.set_submodule(name, MixedEmbedding(candidates[name], padding_idx=source.padding_idx,
                                              row_chunk_size=sealed_job['assembly']['row_chunk_size']))
    saved = manifest['modules']['model.embed_tokens']['file']
    assert (artifact / saved).read_bytes() == (stage / receipt['candidates'][name]['file']).read_bytes()
    sample = {'input_ids': torch.tensor([[4, 7, 4, 11, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1, 1, 0]])}
    with torch.no_grad():
        expected_logits = expected(**sample, use_cache=False).logits
    sample_path = c['root'] / 'control-fresh-input.pt'
    logits_path = c['root'] / 'control-fresh-logits.pt'
    torch.save(sample, sample_path)
    script = '''import sys, torch
from openbonsai.mixed_export import load_mixed
torch.set_num_threads(2)
m = load_mixed(sys.argv[1], cache_decoded_weights=False)
x = torch.load(sys.argv[2], weights_only=True)
with torch.no_grad():
    logits = m(**x, use_cache=False).logits
torch.save({'logits': logits, 'eos': m.generation_config.eos_token_id,
            'retained_fp32': {name: p.detach().cpu() for name, p in m.named_parameters()
                             if name.endswith('A_log')}}, sys.argv[3])
'''
    process = subprocess.run([sys.executable, '-c', script, str(artifact), str(sample_path), str(logits_path)],
                             cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    fresh = torch.load(logits_path, weights_only=True)
    assert torch.equal(expected_logits, fresh['logits']) and fresh['eos'] == c['eos']
    assert set(fresh['retained_fp32']) == set(retained_fp32)
    assert all(fresh['retained_fp32'][name].dtype == value.dtype and torch.equal(fresh['retained_fp32'][name], value)
               for name, value in retained_fp32.items())
    assert job_path.read_bytes() == job_bytes
    assert all(Path(path).read_bytes() == data for path, data in producer_bytes.items())
    assert local_directory_identity(c['root'] / 'source') == c['before']
    assert local_directory_identity(c['root'] / 'tokenizer') == c['tokenizer_before']

    # Sealed source/context checks also apply to all other requested controls.
    from openbonsai.pipeline_decoder import assemble_stage
    roles = ('decoder', 'embedding', 'head')
    refs = {'decoder': c['stages'][0], 'head': c['stages'][1], 'embedding': c['stages'][2]}
    for label, selected in [('H-only', ['head']), ('D-only', ['decoder']),
                            ('D+E', ['decoder', 'embedding']), ('D+H', ['decoder', 'head'])]:
        request = {**control, 'quantized_components': selected, 'evaluation_label': label,
                   'retained_original_components': [role for role in roles if role not in selected]}
        directories = [pipeline.completed_stage(refs[role], role)[0].parent for role in selected]
        result = assemble_stage(context, directories, c['root'] / ('sealed-control-' + label),
                                assembly_control=request)
        assert result['pipeline_job_sha256'] == sealed_job['job_sha256']
        assert result['candidate_payloads_unchanged'] and result['delivery_target_complete'] is False
    assert job_path.read_bytes() == job_bytes


def test_supervised_paired_head_rotation_same_source_and_fresh_reload(full_chain):
    """Exercise O/UH/RH stage composition, not a random-model quality contest."""
    from openbonsai.export import _all_tensors
    from openbonsai.head_calibration import load_head_input_cache
    from openbonsai.pipeline_endpoints import _construct
    from openbonsai.rotations import fixed_signs
    from openbonsai.text_export import extract_text_model

    c, root = full_chain, full_chain['root']
    rotated_job_path, rotated_job = pipeline.completed_stage(c['job'], 'seal')
    protected = {str(path): sha(path) for directory in
                 [root / 'source', root / 'tokenizer', root / 'seal', root / 'head',
                  root / 'prepare-calibration', root / 'prepare-search_validation',
                  root / 'cache-calibration', root / 'cache-search_validation']
                 for path in directory.rglob('*') if path.is_file()}
    # Omitting the new option must preserve the old rotated job and its payload.
    assert 'rotation_mode' not in rotated_job['head']['quantization']
    recipes = copy.deepcopy(c['recipes'])
    recipes['head']['quantization']['rotation_mode'] = 'none'
    unrotated_job_ref = invoke(root, 'head-none-seal', 'seal', {
        'model_directory': str(root / 'source'), 'tokenizer_directory': str(root / 'tokenizer'),
        'traces_by_role': {role: {'prepared': ref, 'sequence_length': 64, 'max_samples': None}
                          for role, ref in c['prepared'].items()}, **recipes})
    unrotated_job_path, unrotated_job = pipeline.completed_stage(unrotated_job_ref, 'seal')
    normalized_head = copy.deepcopy(unrotated_job['head'])
    assert normalized_head['quantization'].pop('rotation_mode') == 'none'
    assert normalized_head == rotated_job['head']
    assert unrotated_job['job_sha256'] != rotated_job['job_sha256']
    for field in ('source_provenance', 'decoder', 'embedding', 'assembly', 'code'):
        assert unrotated_job[field] == rotated_job[field]
    for field in ('model_directory', 'checkpoint_tensors'):
        assert unrotated_job['source'][field] == rotated_job['source'][field]
    assert unrotated_job['source']['tokenizer_directory']['sha256'] == rotated_job['source']['tokenizer_directory']['sha256']
    for role in c['prepared']:
        # Sealing copies the same authenticated bytes to different owned paths.
        records = [copy.deepcopy(job['traces_by_role'][role]) for job in (rotated_job, unrotated_job)]
        for record in records:
            record.pop('sha256')
            for field in ('file', 'producer_file'):
                record[field].pop('path')
        assert records[0] == records[1]

    contexts = {'RH': sources.load_context(rotated_job_path),
                'UH': sources.load_context(unrotated_job_path)}
    unrotated_caches = {role: invoke(root, 'head-none-cache-' + role, 'cache-head',
        {'job': unrotated_job_ref, 'role': role}) for role in c['prepared']}
    for role, ref in unrotated_caches.items():
        rotated_path, rotated_receipt = pipeline.completed_stage(
            reference(root / ('cache-' + role) / 'result.json'), 'cache-head')
        unrotated_path, unrotated_receipt = pipeline.completed_stage(ref, 'cache-head')
        assert rotated_receipt['job_sha256'] == contexts['RH'].job_sha256
        assert unrotated_receipt['job_sha256'] == contexts['UH'].job_sha256
        caches = [load_head_input_cache(path.parent / 'cache')
                  for path in (rotated_path, unrotated_path)]
        assert caches[0].manifest == caches[1].manifest
        assert len(caches[0]) == len(caches[1]) == 2
        for index in range(len(caches[0])):
            batches = [list(cache.iter_record(index, token_microbatch=8)) for cache in caches]
            assert len(batches[0]) == len(batches[1])
            for left, right in zip(*batches):
                assert set(left) == set(right) == {'hidden', 'labels', 'positions'}
                assert all(left[key].dtype == right[key].dtype and torch.equal(left[key], right[key])
                           for key in left)
    unrotated_head = invoke(root, 'head-none-fit', 'head', {
        'job': unrotated_job_ref, 'calibration_cache': unrotated_caches['calibration'],
        'search_cache': unrotated_caches['search_validation'], 'evaluation_cache': None})
    heads = {'RH': c['stages'][1], 'UH': unrotated_head}
    jobs = {'RH': c['job'], 'UH': unrotated_job_ref}
    reports = {arm: pipeline.completed_stage(ref, 'head')[1]['report'] for arm, ref in heads.items()}
    for field in ('config', 'source_provenance', 'source_weight_sha256', 'positions', 'records',
                  'initial_trainable_state_sha256', 'objective'):
        assert reports['RH'][field] == reports['UH'][field]
    for report in reports.values():
        assert [(row['step'], row['total_steps'], row['progress']) for row in report['steps']] == [
            (1, 2, .5), (2, 2, 1.)]
        assert report['search_validation']['role'] == 'search_validation'
        assert report['evaluation_cache_used_for_scoring'] is False
    assert reports['RH']['head_configuration']['rotation'] is not None
    assert reports['UH']['head_configuration']['rotation'] is None

    samples = [
        {'input_ids': torch.tensor([[4, 7, 4, 11]]), 'attention_mask': torch.ones(1, 4, dtype=torch.long)},
        {'input_ids': torch.tensor([[4, 7, 4, 11, c['tokenizer'].pad_token_id]]),
         'attention_mask': torch.tensor([[1, 1, 1, 1, 0]])},
    ]
    # O is the native original, not an artifact with a quantized head. Extraction
    # and unquantized operator bypass are checked independently of packing.
    original = sources.load_original(contexts['RH'])
    extracted = extract_text_model(original, source_id=contexts['RH'].source_provenance['source_checkpoint'],
                                   source_revision=contexts['RH'].source_provenance['source_revision']).model
    with torch.no_grad():
        for sample in samples:
            assert torch.equal(original(**sample, use_cache=False).logits,
                               extracted(**sample, use_cache=False).logits)
    hidden = torch.arange(2 * original.lm_head.in_features, dtype=torch.bfloat16).reshape(2, -1) / 128
    initial_states = []
    for arm in ('RH', 'UH'):
        operator = _construct(original.lm_head, 'head', 'lm_head', contexts[arm].job['head'])
        initial_states.append({name: value.detach().clone() for name, value in operator.named_parameters()})
        operator.quant_enabled = False
        with torch.no_grad():
            for start in range(0, operator.out_features, 31):
                end = min(start + 31, operator.out_features)
                reference_logits = torch.nn.functional.linear(hidden, original.lm_head.weight[start:end])
                assert torch.equal(operator.teacher_rows(hidden, start, end), reference_logits)
                assert torch.equal(operator.forward_rows(hidden, start, end), reference_logits)
    assert set(initial_states[0]) == set(initial_states[1])
    assert all(torch.equal(initial_states[0][key], initial_states[1][key]) for key in initial_states[0])
    del operator, original, extracted, initial_states

    sample_path = root / 'paired-head-inputs.pt'
    torch.save(samples, sample_path)
    control = {'quantized_components': ['head'], 'retained_original_components': ['decoder', 'embedding'],
               'output_scope': 'language-only', 'evaluation_label': 'H-only', 'budget_bytes': 100_000_000}
    for arm in ('RH', 'UH'):
        context = contexts[arm]
        assembled = invoke(root, 'paired-head-' + arm, 'assemble',
            {'job': jobs[arm], 'candidate_stages': [heads[arm]], 'assembly_control': control})
        path, report = pipeline.completed_stage(assembled, 'assemble')
        artifact = path.parent / 'artifact'
        manifest = read_json(artifact / 'manifest.json')
        assert set(manifest['modules']) == {'lm_head'} and report['module_count'] == 1
        assert report['coverage'] == 'control' and report['delivery_target_complete'] is False
        assert report['assembly_control'] == control and report['quality_claim'] is False
        assert report['pipeline_job_sha256'] == context.job_sha256
        assert int(manifest['storage']['artifact_bytes']) == sum(
            p.stat().st_size for p in artifact.rglob('*') if p.is_file())
        assert int(manifest['storage']['vision_file_bytes']) == 0
        head_stage = pipeline.completed_stage(heads[arm], 'head')[0].parent
        candidates, receipt = sources.load_candidate_stage(context, head_stage, expected_stage='head')
        candidate, descriptor = candidates['lm_head'], receipt['candidates']['lm_head']
        assert candidate.metadata['provenance']['pipeline_job_sha256'] == context.job_sha256
        assert candidate.metadata['provenance']['source_weight_sha256'] == reports[arm]['source_weight_sha256']
        if arm == 'UH':
            assert 'rotation' not in candidate.metadata and 'rotation_signs' not in candidate.tensors
            assert descriptor['rotation'] is None and descriptor['rotation_signs_sha256'] is None
        else:
            assert candidate.metadata['rotation'] == descriptor['rotation']
            assert torch.equal(candidate.tensors['rotation_signs'], fixed_signs(128, seed=41, name='lm_head'))
        assert (artifact / manifest['modules']['lm_head']['file']).read_bytes() == (
            head_stage / descriptor['file']).read_bytes()

        expected = sources.load_original(context)
        retained = {text_module_name(name): (value.detach().clone(), kind, persistent)
                    for name, value, kind, persistent in _all_tensors(expected)
                    if not name.startswith(('lm_head.', 'model.visual.'))}
        expected.lm_head = MixedLinear(candidate, row_chunk_size=31)
        loaded = load_mixed(artifact, cache_decoded_weights=False)
        actual_tensors = {name: (value, kind, persistent) for name, value, kind, persistent in _all_tensors(loaded)}
        for name, (value, kind, persistent) in retained.items():
            actual, actual_kind, actual_persistent = actual_tensors[name]
            assert (actual.dtype, actual_kind, actual_persistent) == (value.dtype, kind, persistent)
            assert torch.equal(actual, value)
        expected_fp32 = {name: value for name, (value, _, _) in retained.items() if name.endswith('A_log')}
        assert expected_fp32 and all(value.dtype == torch.float32 for value in expected_fp32.values())
        assert all(not torch.equal(value, value.bfloat16().float()) for value in expected_fp32.values())
        with torch.no_grad():
            expected_logits = [expected(**sample, use_cache=False).logits for sample in samples]
            assert all(torch.equal(logits, loaded(**sample, use_cache=False).logits)
                       for sample, logits in zip(samples, expected_logits))
        fresh_path = root / ('paired-head-' + arm + '-fresh.pt')
        script = '''import sys, torch
from openbonsai.mixed_export import load_mixed
torch.set_num_threads(2)
model = load_mixed(sys.argv[1], cache_decoded_weights=False)
samples = torch.load(sys.argv[2], weights_only=True)
with torch.no_grad():
    logits = [model(**sample, use_cache=False).logits for sample in samples]
torch.save({'logits': logits, 'eos': model.generation_config.eos_token_id,
            'retained_fp32': {name: p.detach().cpu() for name, p in model.named_parameters()
                             if name.endswith('A_log')}}, sys.argv[3])
'''
        process = subprocess.run([sys.executable, '-c', script, str(artifact), str(sample_path), str(fresh_path)],
            cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60)
        (root / ('paired-head-' + arm + '-fresh.stdout.txt')).write_text(process.stdout)
        (root / ('paired-head-' + arm + '-fresh.stderr.txt')).write_text(process.stderr)
        assert process.returncode == 0, process.stderr
        fresh = torch.load(fresh_path, weights_only=True)
        assert len(fresh['logits']) == len(expected_logits)
        assert all(torch.equal(left, right) for left, right in zip(fresh['logits'], expected_logits))
        assert fresh['eos'] == loaded.generation_config.eos_token_id == c['eos']
        assert set(fresh['retained_fp32']) == set(expected_fp32)
        assert all(fresh['retained_fp32'][name].dtype == value.dtype
                   and torch.equal(fresh['retained_fp32'][name], value) for name, value in expected_fp32.items())
    assert all(sha(path) == digest for path, digest in protected.items())
    assert local_directory_identity(root / 'source') == c['before']
    assert local_directory_identity(root / 'tokenizer') == c['tokenizer_before']
    assert all(sha(row['file']) == row['sha256'] for row in c['prompts'].values())
