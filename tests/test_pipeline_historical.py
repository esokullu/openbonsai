"""Historical admission is byte/selection validation, never model generation."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from openbonsai import pipeline
from openbonsai import pipeline_sources as sources
from openbonsai.eval_backends import _identity, local_directory_identity
from test_pipeline_sources import make_source, make_tokenizer, recipes


def ref(path):
    value = _identity(path)
    return {key: value[key] for key in ('path', 'sha256')}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    return ref(path)


def inputs(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    make_source(source)
    tokens = make_tokenizer(tmp_path / 'tokenizer')
    history = tmp_path / 'history'
    history.mkdir()
    model_id, token_id = local_directory_identity(source), sources.tokenizer_identity(tokens)
    config = sources._read_json(source / 'config.json')
    traces = {}
    for role, offset in (('calibration', 0), ('search_validation', 20)):
        rows = [
            {'input_ids': [4 + offset, 5 + offset, 2], 'attention_mask': [1, 1, 1],
             'historical': {'policy': 'bf16-cast', 'literal': 'é', 'generation_eos': [2, 3]}},
            {'input_ids': [8 + offset, 9 + offset, 3, 2, 17 + offset, 18 + offset],
             'attention_mask': [1, 1, 0, 1, 1, 1]},
            # The whole original file is pinned/copied, but this suffix is not
            # selected or presented as valid training data.
            {'input_ids': [999999], 'suffix': 'preserve original unused bytes'}]
        raw = b'\r\n' + b''.join((json.dumps(row, ensure_ascii=False) + '\r\n').encode() for row in rows)
        path = history / (role + '.jsonl')
        path.write_bytes(raw)
        _, selection = sources._trace_records(path, sequence_length=5, max_samples=2,
            pad_token_id=config['pad_token_id'], vocab_size=config['vocab_size'])
        producer = history / (role + '.producer.json')
        write_json(producer, dict(kind='historical-caller-attribution', status='not-new-generation',
            description='Imported fixed IDs; original execution is an attributed historical claim',
            role=role, source_content_sha256=model_id['sha256'],
            tokenizer_content_sha256=token_id['sha256'], source_jsonl_sha256=ref(path)['sha256'],
            historical_execution={'dtype_policy': 'bf16-cast', 'revision': 'a' * 40},
            unknown_historical_metadata={'keep': ['exact', 'claims']}))
        traces[role] = dict(historical=dict(traces=ref(path), producer=ref(producer),
            expected_selection={key: selection[key] for key in pipeline._SELECTION_FIELDS}),
            sequence_length=5, max_samples=2)
    request = dict(format=pipeline.REQUEST_FORMAT, stage='seal', parameters=dict(
        model_directory=str(source), tokenizer_directory=str(tokens), traces_by_role=traces, **recipes()))
    # Sealing may inspect headers and hashes; model load/generation is forbidden.
    monkeypatch.setattr(sources, 'load_original_source', lambda *_a, **_k: pytest.fail('must not load a model'))
    monkeypatch.setattr('openbonsai.pipeline_prepare.prepare_bound',
                        lambda *_a, **_k: pytest.fail('must not generate'))
    return request


def execute(request, output):
    pipeline.validate_request(request)
    return pipeline.execute_stage(request, output, 'cpu', lambda _label, fn: fn(), lambda _event: None)


def test_historical_seal_exact_bytes_selection_claims_and_no_generation(tmp_path, monkeypatch):
    request = inputs(tmp_path, monkeypatch)
    before = copy.deepcopy(request)
    paths = [tmp_path / 'source', tmp_path / 'tokenizer', tmp_path / 'history']
    identities = {str(path): local_directory_identity(path) for path in paths}
    job_path = execute(request, tmp_path / 'sealed')
    context = sources.load_context(job_path)
    for role, spec in request['parameters']['traces_by_role'].items():
        record = context.record_receipts_by_role[role]
        for kind, field in (('traces', 'file'), ('producer', 'producer_file')):
            assert Path(record[field]['path']).read_bytes() == Path(spec['historical'][kind]['path']).read_bytes()
        assert {key: record[key] for key in pipeline._SELECTION_FIELDS} == spec['historical']['expected_selection']
        assert record['producer']['status'] == 'not-new-generation'
        assert record['producer']['historical_execution']['dtype_policy'] == 'bf16-cast'
        assert 'not independent generation proof' in record['provenance_status']
    actual = context.calibration_records
    assert actual[0]['input_ids'].tolist() == [[4, 5, 2, 0, 0]]
    assert actual[0]['attention_mask'].tolist() == [[1, 1, 1, 0, 0]]
    assert actual[1]['input_ids'].tolist() == [[8, 9, 3, 2, 17]]
    assert actual[1]['attention_mask'].tolist() == [[1, 1, 0, 1, 1]]
    assert context.calibration_receipt['truncated_tokens'] == 1
    assert context.calibration_receipt['padding_tokens'] == 2
    assert request == before
    assert all(local_directory_identity(path) == identities[str(path)] for path in paths)


def test_two_head_recipes_reuse_same_historical_selection_without_retag(tmp_path, monkeypatch):
    request = inputs(tmp_path, monkeypatch)
    rotated = sources.load_context(execute(request, tmp_path / 'rotated'))
    unrotated_request = copy.deepcopy(request)
    unrotated_request['parameters']['head']['quantization']['rotation_mode'] = 'none'
    unrotated = sources.load_context(execute(unrotated_request, tmp_path / 'unrotated'))
    assert rotated.job_sha256 != unrotated.job_sha256
    assert rotated.source_provenance == unrotated.source_provenance
    assert rotated.job['source']['tokenizer_directory']['sha256'] == unrotated.job['source']['tokenizer_directory']['sha256']
    for role in rotated.records_by_role:
        a, b = (context.record_receipts_by_role[role] for context in (rotated, unrotated))
        assert {key: a[key] for key in pipeline._SELECTION_FIELDS} == {key: b[key] for key in pipeline._SELECTION_FIELDS}
        assert a['file']['sha256'] == b['file']['sha256']
        assert a['producer'] == b['producer']


@pytest.mark.parametrize('mutation', ['both', 'neither', 'extra', 'missing', 'historical-extra', 'traces-extra'])
def test_selector_schema_is_exact(tmp_path, monkeypatch, mutation):
    request = inputs(tmp_path, monkeypatch)
    spec = request['parameters']['traces_by_role']['calibration']
    if mutation == 'both':
        spec['prepared'] = {'path': '/unused', 'sha256': '0' * 64}
    elif mutation == 'neither':
        spec.pop('historical')
    elif mutation == 'extra':
        spec['offset'] = 1
    elif mutation == 'missing':
        spec.pop('max_samples')
    elif mutation == 'historical-extra':
        spec['historical']['offset'] = 1
    else:
        spec['historical']['traces']['extra'] = 1
    with pytest.raises(ValueError):
        pipeline.validate_request(request)


@pytest.mark.parametrize('field,value', [
    ('num_records', True), ('num_records', 0), ('num_records', 2.0),
    ('consumed_jsonl_sha256', None), ('consumed_jsonl_sha256', 'A' * 64),
    ('record_sha256s', []), ('record_sha256s', ['f' * 64]),
    ('record_sha256s', ['x' * 64] * 2), ('record_sha256s', 'f' * 64),
    ('raw_token_record_sha256s', ['f' * 64, None]), ('unknown', 1)])
def test_expected_selection_schema(tmp_path, monkeypatch, field, value):
    request = inputs(tmp_path, monkeypatch)
    request['parameters']['traces_by_role']['calibration']['historical']['expected_selection'][field] = value
    with pytest.raises(ValueError):
        pipeline.validate_request(request)


@pytest.mark.parametrize('field,value', [('sequence_length', True), ('sequence_length', 0),
    ('max_samples', True), ('max_samples', 1), ('max_samples', 2.0)])
def test_historical_selection_bound_types(tmp_path, monkeypatch, field, value):
    request = inputs(tmp_path, monkeypatch)
    request['parameters']['traces_by_role']['calibration'][field] = value
    with pytest.raises(ValueError):
        pipeline.validate_request(request)


@pytest.mark.parametrize('kind', ['traces', 'producer'])
def test_independent_original_file_pins_checked_before_output(tmp_path, monkeypatch, kind):
    request = inputs(tmp_path, monkeypatch)
    request['parameters']['traces_by_role']['calibration']['historical'][kind]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256'):
        pipeline.protected_paths(request)
    assert not (tmp_path / 'sealed').exists()


@pytest.mark.parametrize('field,value', [('source_content_sha256', 'f' * 64),
    ('tokenizer_content_sha256', 'f' * 64), ('source_jsonl_sha256', 'f' * 64), ('role', 'evaluation')])
def test_existing_producer_binding_still_applies(tmp_path, monkeypatch, field, value):
    request = inputs(tmp_path, monkeypatch)
    historical = request['parameters']['traces_by_role']['calibration']['historical']
    path = Path(historical['producer']['path'])
    producer = sources._read_json(path)
    producer[field] = value
    historical['producer'] = write_json(path, producer)
    with pytest.raises(ValueError, match='bind|role'):
        execute(request, tmp_path / 'sealed')
    assert not (tmp_path / 'sealed').exists()


@pytest.mark.parametrize('field', ['num_records', 'consumed_jsonl_sha256', 'record_sha256s', 'raw_token_record_sha256s'])
def test_exact_expected_selection_comparison(tmp_path, monkeypatch, field):
    request = inputs(tmp_path, monkeypatch)
    spec = request['parameters']['traces_by_role']['calibration']
    expected = spec['historical']['expected_selection']
    if field == 'num_records':
        # Valid declaration expects a third selected record; actual suffix is
        # absent. max_samples alone used to accept the short file.
        path = Path(spec['historical']['traces']['path'])
        path.write_bytes(b''.join(path.read_bytes().splitlines(keepends=True)[:3]))
        spec['historical']['traces'] = ref(path)
        producer_path = Path(spec['historical']['producer']['path'])
        producer = sources._read_json(producer_path)
        producer['source_jsonl_sha256'] = ref(path)['sha256']
        spec['historical']['producer'] = write_json(producer_path, producer)
        spec['max_samples'] = expected['num_records'] = 3
        for key in ('record_sha256s', 'raw_token_record_sha256s'):
            expected[key].append('f' * 64)
    elif field == 'consumed_jsonl_sha256':
        expected[field] = 'f' * 64
    else:
        expected[field].reverse()
    with pytest.raises(ValueError, match='Historical selection'):
        execute(request, tmp_path / 'sealed')
    # Inner artifacts alone are not a supervised-success handoff.
    assert (tmp_path / 'sealed' / 'job.json').is_file()
    with pytest.raises(ValueError, match='complete, cleanly supervised'):
        pipeline.completed_stage(ref(tmp_path / 'sealed' / 'job.json'), 'seal')


@pytest.mark.parametrize('kind', ['traces', 'producer', 'copied-traces', 'copied-producer'])
def test_original_and_copy_pins_rechecked_after_seal(tmp_path, monkeypatch, kind):
    request = inputs(tmp_path, monkeypatch)
    original = sources.seal_job
    def changed(**kwargs):
        context = original(**kwargs)
        if kind.startswith('copied-'):
            field = 'file' if kind == 'copied-traces' else 'producer_file'
            path = Path(context.calibration_receipt[field]['path'])
        else:
            path = Path(request['parameters']['traces_by_role']['calibration']['historical'][kind]['path'])
        with path.open('ab') as stream:
            stream.write(b' ')
        return context
    monkeypatch.setattr(sources, 'seal_job', changed)
    with pytest.raises(RuntimeError, match='Historical input changed'):
        execute(request, tmp_path / 'sealed')


def test_exact_role_overlap_remains_rejected(tmp_path, monkeypatch):
    request = inputs(tmp_path, monkeypatch)
    calibration = request['parameters']['traces_by_role']['calibration']
    producer_path = Path(calibration['historical']['producer']['path'])
    producer = sources._read_json(producer_path)
    producer.pop('role')  # Optional under the unchanged low-level attribution contract.
    calibration['historical']['producer'] = write_json(producer_path, producer)
    request['parameters']['traces_by_role']['search_validation'] = copy.deepcopy(calibration)
    with pytest.raises(ValueError, match='overlap'):
        execute(request, tmp_path / 'sealed')


@pytest.mark.parametrize('field,value', [('input_ids', [4, True, 2]),
    ('input_ids', [4, 999999, 2]), ('attention_mask', [1, 2, 1]), ('attention_mask', [0, 0, 0])])
def test_raw_ids_and_masks_reach_existing_validator(tmp_path, monkeypatch, field, value):
    request = inputs(tmp_path, monkeypatch)
    historical = request['parameters']['traces_by_role']['calibration']['historical']
    path = Path(historical['traces']['path'])
    row = dict(input_ids=[4, 5, 2], attention_mask=[1, 1, 1])
    row[field] = value
    path.write_text(json.dumps(row) + '\n')
    historical['traces'] = ref(path)
    producer_path = Path(historical['producer']['path'])
    producer = sources._read_json(producer_path)
    producer['source_jsonl_sha256'] = ref(path)['sha256']
    historical['producer'] = write_json(producer_path, producer)
    with pytest.raises(ValueError, match='Trace'):
        execute(request, tmp_path / 'sealed')


def test_parent_input_roots_are_protected_before_supervisor_output(tmp_path, monkeypatch):
    request = inputs(tmp_path, monkeypatch)
    path = tmp_path / 'request.json'
    request_ref = write_json(path, request)
    before = local_directory_identity(tmp_path / 'history')
    output = tmp_path / 'history' / 'must-not-create'
    args = SimpleNamespace(max_seconds=60., threads=2, device='cpu', minimum_free_cuda_bytes=0,
        request=path, expected_request_sha256=request_ref['sha256'], output=output)
    monkeypatch.setattr('openbonsai.evaluation_supervisor.supervise',
                        lambda *_a, **_k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='Output cannot modify'):
        pipeline.supervisor_main(args)
    assert not output.exists() and local_directory_identity(tmp_path / 'history') == before


def test_selection_mismatch_marks_child_failed_not_complete(tmp_path, monkeypatch):
    request = inputs(tmp_path, monkeypatch)
    request['parameters']['traces_by_role']['calibration']['historical']['expected_selection']['consumed_jsonl_sha256'] = 'f' * 64
    output = tmp_path / 'supervised'
    output.mkdir()
    request_ref = write_json(output / 'request.json', request)
    now = time.monotonic()
    attempt = dict(output=str(output), stage='seal', supervisor_pid=os.getppid(),
        started_monotonic=now, deadline_monotonic=now + 60, request=request_ref,
        code=sources.code_identity(), threads=2, device='cpu', minimum_free_cuda_bytes=0)
    job_path = output / 'attempt.json'
    write_json(job_path, attempt)
    assert pipeline.child_main(job_path) == 1
    status = sources._read_json(output / 'child-status.json')
    assert status['status'] == 'failed' and 'Historical selection' in status['error']
    assert 'payload' not in status
    assert (output / 'stage' / 'job.json').is_file()


def test_existing_prepared_spec_and_resolution_unchanged(tmp_path, monkeypatch):
    request = inputs(tmp_path, monkeypatch)
    prepare_root = tmp_path / 'completed-prepare'
    prepare_root.mkdir()
    calibration = request['parameters']['traces_by_role']['calibration']['historical']
    (prepare_root / 'traces.jsonl').write_bytes(Path(calibration['traces']['path']).read_bytes())
    producer = sources._read_json(calibration['producer']['path'])
    producer['kind'], producer['status'] = 'openbonsai-original-ayot.v1', 'complete'
    producer_path = prepare_root / 'producer.json'
    write_json(producer_path, producer)
    reference = {'path': '/existing-completed-prepare-result', 'sha256': 'e' * 64}
    request['parameters']['traces_by_role']['calibration'] = dict(prepared=reference, sequence_length=5, max_samples=2)
    calls = []
    def completed(value, stage):
        calls.append((value, stage))
        assert value == reference and stage == 'prepare'
        return producer_path, producer
    monkeypatch.setattr(pipeline, 'completed_stage', completed)
    context = sources.load_context(execute(request, tmp_path / 'mixed-input-kinds'))
    assert calls == [(reference, 'prepare')]
    assert context.calibration_receipt['producer']['kind'] == 'openbonsai-original-ayot.v1'
    assert context.record_receipts_by_role['search_validation']['producer']['kind'] == 'historical-caller-attribution'
