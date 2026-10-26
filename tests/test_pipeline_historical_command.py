"""Real CPU historical import -> original-head cache, with no preparation run."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai import pipeline, pipeline_sources as sources
from openbonsai.adapter import tiny_model
from openbonsai.eval_backends import local_directory_identity
from openbonsai.pipeline_endpoints import load_head_cache_stage
from test_pipeline_command import explicit_recipes, invoke, read_json, reference, sha, write_json


@pytest.fixture(scope='module')
def historical_chain(tmp_path_factory):
    root = tmp_path_factory.mktemp('historical-pipeline')
    torch.set_num_threads(2)
    native = tiny_model(seed=910, multimodal=True).bfloat16().eval().requires_grad_(False)
    for name, parameter in native.named_parameters():
        if name.endswith('A_log'):
            parameter.data = parameter.data.float().fill_(.1234567)
    native.save_pretrained(root / 'source')
    backend = Tokenizer(models.WordLevel({f't{i}': i for i in range(256)}, unk_token='t3'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='t0',
        eos_token='t2', unk_token='t3', additional_special_tokens=['t4'])
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}\r\n"
    tokenizer.save_pretrained(root / 'tokenizer')
    source_before = local_directory_identity(root / 'source')
    tokenizer_before = local_directory_identity(root / 'tokenizer')
    token_identity = sources.tokenizer_identity(root / 'tokenizer')
    history = root / 'history'
    history.mkdir()
    specs, expected_records, raw_files = {}, {}, {}
    for role, start in [('calibration', 10), ('search_validation', 30)]:
        rows = [
            {'record_id': role + '-0', 'prompt': 'historical fixture first prefix',
             'input_ids': [start, start + 1, start + 2, start + 3, 2],
             'attention_mask': [1, 1, 1, 1, 1], 'prompt_token_count': 3,
             'generated_token_count': 2, 'stop_reason': 'eos', 'eos_ids': [2, 4]},
            {'record_id': role + '-1', 'prompt': 'historical fixture second prefix',
             'input_ids': [start + 5, start + 6, start + 7, 4, 0],
             'attention_mask': [1, 1, 1, 1, 0], 'prompt_token_count': 2,
             'generated_token_count': 2, 'stop_reason': 'eos', 'eos_ids': [2, 4]},
            {'record_id': role + '-unused', 'input_ids': [start + 8, start + 9, 2],
             'attention_mask': [1, 1, 1], 'unused_suffix': True},
        ]
        # Preserve noncanonical bytes, line endings, metadata, and an unconsumed
        # third row. The selected-prefix digest is independently constructed.
        lines = [(json.dumps(row, ensure_ascii=False, indent=None) + '\r\n').encode()
                 for row in rows]
        consumed = lines[0] + b'\r\n' + lines[1]
        trace = history / (role + '.jsonl')
        trace.write_bytes(consumed + lines[2])
        producer = write_json(history / (role + '.producer.json'), {
            'kind': 'historical-synthetic-token-fixture', 'role': role,
            'description': 'Authored historical fixture; import does not prove original generation.',
            'source_content_sha256': source_before['sha256'],
            'tokenizer_content_sha256': token_identity['sha256'],
            'source_jsonl_sha256': sha(trace),
            'historical_claims': {'source_run': 'tiny-native-fixture',
                'generation_executed_by_this_import': False, 'retained_metadata': True}})
        normalized = [{'input_ids': row['input_ids'] + [0] * 3,
                       'attention_mask': row['attention_mask'] + [0] * 3} for row in rows[:2]]
        selection = {'num_records': 2,
            'consumed_jsonl_sha256': hashlib.sha256(consumed).hexdigest(),
            'record_sha256s': [sources.json_sha256(row) for row in normalized],
            'raw_token_record_sha256s': [sources.json_sha256({key: row[key]
                for key in ('input_ids', 'attention_mask')}) for row in rows[:2]]}
        specs[role] = {'historical': {'traces': reference(trace), 'producer': reference(producer),
                                   'expected_selection': selection},
                       'sequence_length': 8, 'max_samples': 2}
        expected_records[role] = normalized
        raw_files[role] = {'trace': trace.read_bytes(), 'producer': producer.read_bytes(), 'rows': rows}
    history_before = local_directory_identity(history)
    recipes = explicit_recipes()
    recipes['head']['cache']['shard_positions'] = 3
    parameters = {'model_directory': str(root / 'source'),
        'tokenizer_directory': str(root / 'tokenizer'), 'traces_by_role': specs, **recipes}
    sealed = invoke(root, 'historical-seal', 'seal', parameters)
    job_path, job = pipeline.completed_stage(sealed, 'seal')
    caches = {role: invoke(root, 'historical-cache-' + role, 'cache-head',
                          {'job': sealed, 'role': role}) for role in specs}
    return dict(root=root, native=native, parameters=parameters, job=job, job_path=job_path,
        sealed=sealed, caches=caches, expected_records=expected_records, raw_files=raw_files,
        source_before=source_before, tokenizer_before=tokenizer_before, history_before=history_before)


def _preserved(fixture):
    root = fixture['root']
    assert local_directory_identity(root / 'source') == fixture['source_before']
    assert local_directory_identity(root / 'tokenizer') == fixture['tokenizer_before']
    assert local_directory_identity(root / 'history') == fixture['history_before']


def test_historical_exact_selection_native_cache_and_attribution(historical_chain):
    c = historical_chain
    context = sources.load_context(c['job_path'])
    # The declared source is the saved native checkpoint. A blanket .bfloat16()
    # on the fixture constructor also casts nonpersistent RoPE buffers, whereas
    # native from_pretrained recreates those buffers in FP32. Compare to a fresh
    # native load under the same explicit source policy, without relaxing exact
    # hidden-state parity or the saved persistent tensor checks.
    native = sources.load_original(context)
    for name, expected in c['native'].state_dict().items():
        actual = native.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state_before = {name: tensor.clone() for name, tensor in native.state_dict().items()}
    role_tokens = {}
    for role, spec in c['parameters']['traces_by_role'].items():
        receipt = context.record_receipts_by_role[role]
        assert {key: receipt[key] for key in spec['historical']['expected_selection']} == spec['historical']['expected_selection']
        assert receipt['source_tokens'] == 10 and receipt['active_tokens'] == 9
        assert receipt['padding_tokens'] == 6 and receipt['truncated_tokens'] == 0
        assert Path(receipt['file']['path']).read_bytes() == c['raw_files'][role]['trace']
        assert Path(receipt['producer_file']['path']).read_bytes() == c['raw_files'][role]['producer']
        assert receipt['producer']['kind'] == 'historical-synthetic-token-fixture'
        assert receipt['producer']['historical_claims']['generation_executed_by_this_import'] is False
        assert 'not independent generation proof' in receipt['provenance_status']
        records = context.records_by_role[role]
        assert [{key: value[0].tolist() for key, value in row.items()} for row in records] == c['expected_records'][role]
        role_tokens[role] = set(receipt['raw_token_record_sha256s'])
        stage_path, _ = pipeline.completed_stage(c['caches'][role], 'cache-head')
        cache, stage = load_head_cache_stage(context, stage_path.parent, role=role)
        assert cache.manifest['role'] == role and len(cache) == 2 and cache.positions == 7
        assert cache.manifest['records_seen'] == 2
        assert [row['positions'] for row in cache.manifest['records']] == [4, 3]
        assert stage['record_receipt_sha256'] == receipt['sha256']
        assert cache.manifest['source_provenance'] == context.source_provenance
        assert cache.manifest['source_parameter_dtypes']['float32']['numel'] > 0
        for index, record in enumerate(records):
            captured = []
            hook = native.lm_head.register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach().clone()))
            try:
                with torch.no_grad():
                    native(**record, use_cache=False, logits_to_keep=0)
            finally:
                hook.remove()
            assert len(captured) == 1
            ids, mask = record['input_ids'], record['attention_mask'].bool()
            positions = (mask[:, :-1] & mask[:, 1:]).nonzero(as_tuple=False)
            batch, token = positions.unbind(-1)
            shards = list(cache.iter_record(index, token_microbatch=2))
            actual = {key: torch.cat([shard[key] for shard in shards]) for key in shards[0]}
            torch.testing.assert_close(actual['hidden'], captured[0][batch, token], rtol=0, atol=0)
            torch.testing.assert_close(actual['positions'], positions, rtol=0, atol=0)
            torch.testing.assert_close(actual['labels'], ids[batch, token + 1], rtol=0, atol=0)
            assert actual['labels'][-1].item() == (2 if index == 0 else 4)
            assert actual['hidden'].dtype == torch.bfloat16 and not actual['hidden'].requires_grad
        result = read_json(c['caches'][role]['path'])
        assert result['stage'] == 'cache-head' and result['quality_claim'] is False
    assert role_tokens['calibration'].isdisjoint(role_tokens['search_validation'])
    assert read_json(c['sealed']['path'])['stage'] == 'seal'
    # This branch never invoked preparation/generation and never fitted a head.
    results = [read_json(path)['stage'] for path in c['root'].glob('*/result.json')
               if read_json(path)['status'] == 'complete']
    assert sorted(results) == ['cache-head', 'cache-head', 'seal']
    for name, expected in state_before.items():
        actual = native.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not module._forward_pre_hooks for module in native.modules())
    _preserved(c)


@pytest.mark.parametrize('failure', ['selection-order', 'trace-hash'])
def test_historical_mismatch_has_no_successful_handoff(historical_chain, failure):
    c = historical_chain
    parameters = copy.deepcopy(c['parameters'])
    historical = parameters['traces_by_role']['calibration']['historical']
    if failure == 'selection-order':
        historical['expected_selection']['record_sha256s'].reverse()
    else:
        historical['traces']['sha256'] = '0' * 64
    output, _process, result = invoke(c['root'], 'rejected-' + failure, 'seal', parameters,
                                      expected_success=False)
    if result is not None:
        assert result['status'] == 'failed'
        with pytest.raises(ValueError, match='complete, cleanly supervised'):
            pipeline.completed_stage(reference(output / 'result.json'), 'seal')
    else:
        assert not (output / 'result.json').exists()
    if failure == 'selection-order':
        # Even a written, sealed inner job cannot cross the failed outer gate.
        assert (output / 'stage' / 'job.json').is_file()
        assert result is not None and result['status'] == 'failed'
        assert 'Historical selection differs' in read_json(output / 'child-status.json')['error']
        with pytest.raises(ValueError, match='complete, cleanly supervised'):
            pipeline.completed_stage(reference(output / 'stage' / 'job.json'), 'seal')
    _preserved(c)
