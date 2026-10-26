"""Authenticated tokenizer padding fallback; no default IDs or source rewriting."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai import pipeline, pipeline_sources as sources
from openbonsai.eval_backends import local_directory_identity
from test_pipeline_sources import make_source, make_tokenizer, producer, recipes


_MISSING = object()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n')


def real_tokenizer(root, *, pad=5, declaration='config', vocabulary=256):
    backend = Tokenizer(models.WordLevel({f't{i}': i for i in range(vocabulary)}, unk_token='t3'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='t3', eos_token='t2',
        **({} if pad is None else {'pad_token': f't{pad}'}))
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}\r\n"
    tokenizer.save_pretrained(root)
    if declaration != 'config':
        config = read(root / 'tokenizer_config.json')
        config.pop('pad_token', None)
        write(root / 'tokenizer_config.json', config)
        if declaration == 'special_map':
            write(root / 'special_tokens_map.json', {'pad_token': f't{pad}'})
        elif declaration == 'backend':
            state = read(root / 'tokenizer.json')
            state['padding'] = {'strategy': 'BatchLongest', 'direction': 'Right',
                'pad_to_multiple_of': None, 'pad_id': pad, 'pad_type_id': 0, 'pad_token': f't{pad}'}
            write(root / 'tokenizer.json', state)
        else:
            raise AssertionError(declaration)
    return root


def inputs(root, *, multimodal=False, model_pad=_MISSING, tokenizer_pad=5,
           declaration='config', byte_only=False, vocabulary=256, sequence_length=6):
    model = root / 'source'
    make_source(model, multimodal=multimodal)
    config = read(model / 'config.json')
    text = config['text_config'] if multimodal else config
    if model_pad is _MISSING:
        text.pop('pad_token_id', None)
    else:
        text['pad_token_id'] = model_pad
    write(model / 'config.json', config)
    token_path = root / 'tokens'
    if byte_only:
        make_tokenizer(token_path)
    else:
        real_tokenizer(token_path, pad=tokenizer_pad, declaration=declaration, vocabulary=vocabulary)
    trace = root / 'traces.jsonl'
    rows = [
        {'id': 'kept', 'input_ids': [10, 2, 11, 0], 'attention_mask': [1, 1, 1, 0],
         'eos_ids': [2, 4], 'historical': 'attributed, not regenerated'},
        {'id': 'long', 'input_ids': [12, 13, 4, 14, 15, 16, 17],
         'attention_mask': [1, 1, 0, 1, 1, 1, 1], 'eos_ids': [2, 4]},
    ]
    trace.write_bytes(b'\r\n' + b''.join((json.dumps(row) + '\r\n').encode() for row in rows))
    producer_file = producer(root / 'producer.json', model, token_path, trace,
                             role='calibration', historical_dtype='bf16-cast')
    params = dict(model_directory=model, tokenizer_directory=token_path,
        traces_by_role={'calibration': dict(path=trace, sequence_length=sequence_length,
            max_samples=None, producer=producer_file)}, **recipes())
    return params, rows


def originals(params):
    return {str(root): local_directory_identity(root) for root in
            (params['model_directory'], params['tokenizer_directory'])}


def assert_preserved(before):
    assert {root: local_directory_identity(root) for root in before} == before


def reseal_job(path, mutate):
    job = read(path)
    mutate(job['traces_by_role']['calibration'])
    receipt = job['traces_by_role']['calibration']
    receipt['sha256'] = sources.json_sha256({k: v for k, v in receipt.items() if k != 'sha256'})
    job['job_sha256'] = sources.json_sha256({k: v for k, v in job.items() if k != 'job_sha256'})
    write(path, job)


@pytest.mark.parametrize('multimodal', [False, True])
@pytest.mark.parametrize('model_pad', [_MISSING, None])
def test_real_tokenizer_fallback_preserves_exact_source_masks_and_eos(tmp_path, monkeypatch, multimodal, model_pad):
    params, rows = inputs(tmp_path, multimodal=multimodal, model_pad=model_pad)
    before = originals(params)
    raw = Path(params['traces_by_role']['calibration']['path']).read_bytes()
    # Sealing must never load or generate from an original model.
    monkeypatch.setattr(sources, 'load_original_source', lambda *_a, **_k: pytest.fail('model loaded during seal'))
    context = sources.seal_job(**params, output=tmp_path / 'job')
    receipt = context.calibration_receipt
    assert receipt['pad_token_id'] == 5 and receipt['padding_source'] == 'tokenizer'
    assert receipt['vocab_size'] == 256 and receipt['padding_tokens'] == 2 and receipt['truncated_tokens'] == 1
    assert context.calibration_records[0]['input_ids'].tolist() == [[10, 2, 11, 0, 5, 5]]
    assert context.calibration_records[0]['attention_mask'].tolist() == [[1, 1, 1, 0, 0, 0]]
    assert context.calibration_records[1]['input_ids'].tolist() == [[12, 13, 4, 14, 15, 16]]
    assert context.calibration_records[1]['attention_mask'].tolist() == [[1, 1, 0, 1, 1, 1]]
    assert Path(receipt['file']['path']).read_bytes() == raw
    assert receipt['consumed_jsonl_sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['raw_token_record_sha256s'] == [sources.json_sha256({k: row[k]
        for k in ('input_ids', 'attention_mask')}) for row in rows]
    assert receipt['producer']['historical_dtype'] == 'bf16-cast'
    assert sources.load_context(context.job_path).job == context.job
    assert_preserved(before)
    # Saved model and generation EOS/pad fields are not aligned or rewritten.
    saved = read(Path(params['model_directory']) / 'config.json')
    text = saved['text_config'] if multimodal else saved
    assert text.get('pad_token_id') is None
    assert text['eos_token_id'] == 2
    assert read(Path(params['model_directory']) / 'generation_config.json')['pad_token_id'] == 0


@pytest.mark.parametrize('declaration', ['config', 'special_map', 'backend'])
@pytest.mark.parametrize('model_pad', [_MISSING, 5])
def test_supported_explicit_tokenizer_declarations_agree(tmp_path, declaration, model_pad):
    params, _ = inputs(tmp_path, declaration=declaration, model_pad=model_pad)
    context = sources.seal_job(**params, output=tmp_path / 'job')
    assert context.calibration_receipt['pad_token_id'] == 5
    assert context.calibration_receipt['padding_source'] == ('tokenizer' if model_pad is _MISSING else 'model_text_config')


def test_valid_model_pad_without_tokenizer_declaration_keeps_byte_only_path(tmp_path, monkeypatch):
    params, _ = inputs(tmp_path, model_pad=0, byte_only=True)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',
                        lambda *_a, **_k: pytest.fail('legacy byte-only tokenizer was loaded'))
    context = sources.seal_job(**params, output=tmp_path / 'job')
    assert context.calibration_receipt['pad_token_id'] == 0
    assert context.calibration_receipt['padding_source'] == 'model_text_config'
    assert context.calibration_records[0]['input_ids'].tolist() == [[10, 2, 11, 0, 0, 0]]
    # Legacy field omission is accepted only with a concrete original model ID.
    reseal_job(context.job_path, lambda receipt: receipt.pop('padding_source'))
    assert sources.load_context(context.job_path).calibration_receipt['pad_token_id'] == 0


@pytest.mark.parametrize('invalid', [True, False, -1, 256, 1.5, '0'])
def test_invalid_explicit_model_pad_never_falls_back(tmp_path, monkeypatch, invalid):
    params, _ = inputs(tmp_path, model_pad=invalid)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',
                        lambda *_a, **_k: pytest.fail('invalid model pad tried tokenizer fallback'))
    with pytest.raises(ValueError):
        sources.seal_job(**params, output=tmp_path / 'job')
    assert not (tmp_path / 'job' / 'job.json').exists()


@pytest.mark.parametrize('declaration', ['config', 'special_map', 'backend'])
def test_conflicting_model_and_declared_tokenizer_pad_rejected(tmp_path, declaration):
    params, _ = inputs(tmp_path, model_pad=0, declaration=declaration)
    before = originals(params)
    with pytest.raises(ValueError):
        sources.seal_job(**params, output=tmp_path / 'job')
    assert not (tmp_path / 'job' / 'job.json').exists()
    assert_preserved(before)


@pytest.mark.parametrize('case', ['missing-pad', 'out-of-range', 'invented-token', 'custom-code', 'slow'])
def test_unsupported_fallback_fails_closed_without_eos_default(tmp_path, monkeypatch, case):
    params, _ = inputs(tmp_path, tokenizer_pad=None if case == 'missing-pad' else 5,
                       vocabulary=300 if case == 'out-of-range' else (128 if case == 'invented-token' else 256))
    tokens = Path(params['tokenizer_directory'])
    if case == 'out-of-range':
        config = read(tokens / 'tokenizer_config.json'); config['pad_token'] = 't299'; write(tokens / 'tokenizer_config.json', config)
    elif case == 'invented-token':
        config = read(tokens / 'tokenizer_config.json'); config['pad_token'] = '<brand-new-pad>'; write(tokens / 'tokenizer_config.json', config)
    elif case == 'custom-code':
        config = read(tokens / 'tokenizer_config.json'); config['auto_map'] = {'AutoTokenizer': ['evil.Tokenizer', None]}; write(tokens / 'tokenizer_config.json', config)
    elif case == 'slow':
        monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',
                            lambda *_a, **_k: SimpleNamespace(is_fast=False, pad_token_id=5, pad_token='t5'))
    # Bind the mutated tokenizer honestly; rejection is semantic, not a stale SHA shortcut.
    spec = params['traces_by_role']['calibration']
    if case != 'custom-code':
        spec['producer'] = producer(tmp_path / 'producer.json', params['model_directory'], tokens, spec['path'])
    before = originals(params)
    with pytest.raises((ValueError, TypeError)):
        sources.seal_job(**params, output=tmp_path / 'job')
    assert not (tmp_path / 'job' / 'job.json').exists()
    assert_preserved(before)


@pytest.mark.parametrize('field,value', [('pad_token_id', 0), ('vocab_size', 257),
                                         ('padding_source', 'model_text_config'), ('padding_source', 'unknown')])
def test_resealed_receipt_cannot_change_resolved_policy(tmp_path, field, value):
    params, _ = inputs(tmp_path)
    context = sources.seal_job(**params, output=tmp_path / 'job')
    reseal_job(context.job_path, lambda receipt: receipt.__setitem__(field, value))
    with pytest.raises(ValueError):
        sources.load_context(context.job_path)


def test_fallback_receipt_cannot_claim_legacy_missing_source(tmp_path):
    params, _ = inputs(tmp_path)
    context = sources.seal_job(**params, output=tmp_path / 'job')
    reseal_job(context.job_path, lambda receipt: receipt.pop('padding_source'))
    with pytest.raises(ValueError):
        sources.load_context(context.job_path)


def test_tokenizer_mutation_after_seal_is_rejected(tmp_path):
    params, _ = inputs(tmp_path)
    context = sources.seal_job(**params, output=tmp_path / 'job')
    tokens = context.tokenizer_directory
    config = read(tokens / 'tokenizer_config.json'); config['pad_token'] = 't6'; write(tokens / 'tokenizer_config.json', config)
    with pytest.raises(ValueError):
        sources.load_context(context.job_path)


def test_supervised_missing_model_pad_historical_seal_to_native_cache(tmp_path):
    """Real subprocess handoffs preserve frozen IDs and original native outputs."""
    import torch
    from openbonsai.pipeline_endpoints import load_head_cache_stage
    from test_pipeline_command import invoke, reference, read_json

    torch.set_num_threads(2)
    params, _ = inputs(tmp_path, multimodal=True)
    history = tmp_path / 'history'
    history.mkdir()
    trace = history / 'traces.jsonl'
    rows = [
        {'id': 'materialized-0', 'input_ids': [10, 11, 2, 0, 5, 5],
         'attention_mask': [1, 1, 1, 0, 0, 0], 'eos_ids': [2, 4], 'claim': 'already padded'},
        {'id': 'materialized-1', 'input_ids': [12, 13, 14, 4, 5, 5],
         'attention_mask': [1, 1, 1, 1, 0, 0], 'eos_ids': [2, 4], 'claim': 'historical stop'},
    ]
    raw = b'\r\n' + b''.join((json.dumps(row) + '\r\n').encode() for row in rows)
    trace.write_bytes(raw)
    receipt_ref = producer(history / 'producer.json', params['model_directory'], params['tokenizer_directory'], trace,
                           role='calibration', historical_dtype='bf16-cast', generated_by_import=False)
    expected_rows = [{key: row[key] for key in ('input_ids', 'attention_mask')} for row in rows]
    hashes = [sources.json_sha256(row) for row in expected_rows]
    selection = dict(num_records=2, consumed_jsonl_sha256=hashlib.sha256(raw).hexdigest(),
                     record_sha256s=hashes, raw_token_record_sha256s=hashes)
    params['traces_by_role'] = {'calibration': {
        'historical': {'traces': reference(trace), 'producer': receipt_ref, 'expected_selection': selection},
        'sequence_length': 6, 'max_samples': 2}}
    params['model_directory'] = str(params['model_directory'])
    params['tokenizer_directory'] = str(params['tokenizer_directory'])
    params['head']['cache']['shard_positions'] = 2
    before = originals(params)
    producer_bytes = Path(receipt_ref['path']).read_bytes()
    sealed = invoke(tmp_path, 'sealed-command', 'seal', params)
    job_path, _ = pipeline.completed_stage(sealed, 'seal')
    cached = invoke(tmp_path, 'cache-command', 'cache-head', {'job': sealed, 'role': 'calibration'})
    context = sources.load_context(job_path)
    receipt = context.calibration_receipt
    assert receipt['padding_source'] == 'tokenizer' and receipt['pad_token_id'] == 5
    assert receipt['padding_tokens'] == 0 and receipt['truncated_tokens'] == 0
    assert {key: receipt[key] for key in selection} == selection
    assert Path(receipt['file']['path']).read_bytes() == trace.read_bytes() == raw
    assert Path(receipt['producer_file']['path']).read_bytes() == producer_bytes
    assert receipt['producer']['historical_dtype'] == 'bf16-cast'
    assert receipt['producer']['generated_by_import'] is False
    assert [{key: tensor[0].tolist() for key, tensor in record.items()}
            for record in context.calibration_records] == expected_rows
    stage_path, _ = pipeline.completed_stage(cached, 'cache-head')
    cache, stage = load_head_cache_stage(context, stage_path.parent, role='calibration')
    assert len(cache) == 2 and cache.positions == 5
    assert [record['positions'] for record in cache.manifest['records']] == [2, 3]
    assert stage['record_receipt_sha256'] == receipt['sha256']
    native = sources.load_original(context)
    assert native.config.text_config.pad_token_id is None
    assert native.get_input_embeddings().padding_idx is None
    assert native.config.text_config.eos_token_id == 2
    assert native.generation_config.pad_token_id == 0
    native_config_before = copy.deepcopy(native.config.to_dict())
    generation_before = copy.deepcopy(native.generation_config.to_dict())
    state_before = {key: tensor.clone() for key, tensor in native.state_dict().items()}
    for index, record in enumerate(context.calibration_records):
        captured = []
        hook = native.lm_head.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                native(**record, use_cache=False, logits_to_keep=0)
        finally:
            hook.remove()
        assert len(captured) == 1
        mask, ids = record['attention_mask'].bool(), record['input_ids']
        positions = (mask[:, :-1] & mask[:, 1:]).nonzero(as_tuple=False)
        batch, token = positions.unbind(-1)
        shards = list(cache.iter_record(index, token_microbatch=1))
        actual = {key: torch.cat([part[key] for part in shards]) for key in shards[0]}
        torch.testing.assert_close(actual['hidden'], captured[0][batch, token], rtol=0, atol=0)
        torch.testing.assert_close(actual['positions'], positions, rtol=0, atol=0)
        torch.testing.assert_close(actual['labels'], ids[batch, token + 1], rtol=0, atol=0)
        assert actual['labels'][-1].item() == (2 if index == 0 else 4)
    assert native.config.to_dict() == native_config_before
    assert native.generation_config.to_dict() == generation_before
    for key, tensor in native.state_dict().items():
        assert tensor.dtype == state_before[key].dtype
        torch.testing.assert_close(tensor, state_before[key], rtol=0, atol=0)
    assert_preserved(before)
    completed = [read_json(path)['stage'] for path in tmp_path.glob('*/result.json')]
    assert sorted(completed) == ['cache-head', 'seal']
