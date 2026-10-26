"""Final hard head KL is a post-fit measurement, not a training-loss proxy."""
import copy
import hashlib
import random

import pytest
import torch
from torch.nn import functional as F

from openbonsai import pipeline_endpoints as stages
from openbonsai import pipeline_sources as sources
from openbonsai.hadamard import rotate
from test_pipeline_endpoints import make_context, recipes


def _snapshot(head):
    values = {**dict(head.named_parameters()), **dict(head.named_buffers())}
    return {
        'tensors': {name: (id(value), value.data_ptr(), value._version, value.requires_grad,
                          value.detach().clone()) for name, value in values.items()},
        'gradients': {name: None if p.grad is None else
                      (id(p.grad), p.grad._version, p.grad.detach().clone())
                      for name, p in head.named_parameters()},
        'flags': {name: getattr(head, name) for name in (
            'progress', 'quant_enabled', 'rotation_only', 'training', '_step_number',
            '_snapshot_step', '_hard_snapshot_sharpness')},
        'snapshots': {name: getattr(head, name) for name in head._snapshot_names()},
        'torch_rng': torch.random.get_rng_state().clone(), 'python_rng': random.getstate()}


def _unchanged(head, before):
    after = _snapshot(head)
    assert after.keys() == before.keys()
    assert after['flags'] == before['flags'] and after['snapshots'] == before['snapshots']
    assert after['python_rng'] == before['python_rng']
    assert torch.equal(after['torch_rng'], before['torch_rng'])
    assert after['tensors'].keys() == before['tensors'].keys()
    for name, value in before['tensors'].items():
        actual = after['tensors'][name]
        assert actual[:4] == value[:4]
        assert torch.equal(actual[4], value[4])
    for name, value in before['gradients'].items():
        actual = after['gradients'][name]
        if value is None:
            assert actual is None
        else:
            assert actual[:2] == value[:2] and torch.equal(actual[2], value[2])


@torch.no_grad()
def _dense_final_kl(head, cache):
    """Tiny full-matrix/log-softmax oracle independent of the streaming reducer.

    Reconstruct current FP32-scale ternary values, then perform the same native
    BF16 cast. This deliberately precedes exported FP16-scale rounding.
    """
    assert head.progress == 1. and head.quant_enabled and not head.rotation_only
    pieces = []
    for start in range(0, head.out_features, head.row_chunk_size):
        end = min(start + head.row_chunk_size, head.out_features)
        codes, scales = head.hard_codes_rows(start, end)
        pieces.append((codes.reshape(-1, head.group_size).float() * scales.reshape(-1, 1))
                      .reshape(end - start, head.in_features).to(head.weight.dtype))
    weight = torch.cat(pieces)
    total, count = 0., 0
    for index in range(len(cache)):
        for batch in cache.iter_record(index, token_microbatch=64):
            hidden = batch['hidden']
            transformed = hidden if head.rotation_signs is None else rotate(
                hidden, head.rotation_signs, head.block_size)
            teacher = F.linear(hidden, head.weight, head.bias).double()
            student = F.linear(transformed, weight, head.bias).double()
            log_p, log_q = teacher.log_softmax(-1), student.log_softmax(-1)
            total += float((log_p.exp() * (log_p - log_q)).sum())
            count += len(hidden)
    assert count == cache.positions
    return total / count


@pytest.mark.parametrize('mode', ['none', 'hadamard'])
@pytest.mark.parametrize('with_search', [False, True])
def test_final_hard_calibration_dense_oracle_roles_and_read_only(tmp_path, monkeypatch, mode, with_search):
    recipe, _ = recipes()
    recipe['quantization']['rotation_mode'] = mode
    context = make_context(tmp_path, head=recipe)
    receipts = {role: stages.cache_head_stage(context, role, tmp_path / role)
                for role in sources.ROLES}
    before_files = {role: stages._file_tree(tmp_path / role) for role in sources.ROLES}
    original_directory = copy.deepcopy(context.job['source']['model_directory'])
    evaluation, construct = stages.evaluate_head_from_cache, stages._construct
    operators, scored, operations, begin_steps = [], [], [], []
    oracles = {}
    def capture(*args, **kwargs):
        head = construct(*args, **kwargs)
        operators.append(head)
        begin = head.begin_step
        def begin_step(progress):
            begin_steps.append(progress)
            return begin(progress)
        head.begin_step = begin_step
        return head
    monkeypatch.setattr(stages, '_construct', capture)
    def measure(head, cache, **kwargs):
        role = cache.manifest['role']
        assert role in {'calibration', 'search_validation'}
        assert head.progress == 1. and head.quant_enabled and not head.rotation_only
        assert begin_steps == [.5, 1.]
        assert all(getattr(head, name) is None for name in head._snapshot_names())
        snapshot = _snapshot(head)
        result = evaluation(head, cache, **kwargs)
        _unchanged(head, snapshot)
        oracles[role] = _dense_final_kl(head, cache)
        _unchanged(head, snapshot)
        scored.append(role)
        return result
    monkeypatch.setattr(stages, 'evaluate_head_from_cache', measure)
    def operation(label, function):
        operations.append(label)
        return function()
    result = stages.fit_head_stage(context, tmp_path / 'calibration', tmp_path / 'head',
        search_cache=tmp_path / 'search_validation' if with_search else None,
        evaluation_cache=tmp_path / 'evaluation', operation=operation)
    report = result['report']
    expected_roles = ['calibration', 'search_validation'] if with_search else ['calibration']
    assert scored == expected_roles
    assert report['evaluation_cache_used_for_scoring'] is False and 'evaluation' not in report
    assert ('search_validation' in report) is with_search
    assert len(report['steps']) == 2 and begin_steps == [.5, 1.]
    assert report['config'] == recipe['training']
    assert report['final_hard_state_sha256'] == stages._head_state_sha256(operators[0])
    for role in expected_roles:
        measured = report[role]
        assert measured['role'] == role and measured['head_progress'] == 1.
        assert measured['positions'] == 7 and measured['normalization'] == 'FP64'
        assert measured['kl'] == pytest.approx(oracles[role], rel=1e-10, abs=1e-12)
        assert measured['source_provenance'] == context.source_provenance
        assert measured['cache_manifest_sha256'] == hashlib.sha256(
            (tmp_path / role / 'cache' / 'manifest.json').read_bytes()).hexdigest()
        assert report['cache_stage_sha256s'][role] == receipts[role]['sha256']
        assert receipts[role]['head_config_sha256'] == sources.json_sha256(recipe)
    assert operations.index('fit-head') < operations.index('evaluate-head-calibration') < operations.index('export-head-candidate')
    assert operations.count('evaluate-head-calibration') == 1
    assert operations.count('evaluate-head-search') == int(with_search)
    assert not any('evaluate' in name and 'evaluation' in name for name in operations)
    if with_search:
        assert operations.index('evaluate-head-calibration') < operations.index('evaluate-head-search')
    candidates, _ = sources.load_candidate_stage(context, tmp_path / 'head')
    assert candidates['lm_head'].metadata['provenance']['training_report_sha256'] == sources.json_sha256(report)
    assert sources.local_directory_identity(original_directory['path']) == original_directory
    assert {role: stages._file_tree(tmp_path / role) for role in sources.ROLES} == before_files


@pytest.mark.parametrize('changed', ['progress', 'quant_enabled', 'rotation_only', 'missing-step'])
def test_incomplete_or_bypassed_final_state_cannot_be_measured(tmp_path, monkeypatch, changed):
    context = make_context(tmp_path)
    stages.cache_head_stage(context, 'calibration', tmp_path / 'calibration')
    fit = stages.fit_head_from_cache
    def invalid(head, *args, **kwargs):
        result = fit(head, *args, **kwargs)
        if changed == 'progress':
            head.progress = .5
        elif changed == 'quant_enabled':
            head.quant_enabled = False
        elif changed == 'rotation_only':
            head.rotation_only = True
        else:
            result.report['steps'].pop()
        return result
    monkeypatch.setattr(stages, 'fit_head_from_cache', invalid)
    monkeypatch.setattr(stages, 'evaluate_head_from_cache', lambda *_a, **_k: pytest.fail('must not evaluate'))
    with pytest.raises(RuntimeError, match='did not complete'):
        stages.fit_head_stage(context, tmp_path / 'calibration', tmp_path / 'head')
    assert not (tmp_path / 'head').exists()


def test_final_calibration_failure_cannot_export(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    stages.cache_head_stage(context, 'calibration', tmp_path / 'calibration')
    original = copy.deepcopy(context.job['source']['model_directory'])
    def fail(head, cache, **kwargs):
        assert cache.manifest['role'] == 'calibration' and head.progress == 1.
        raise RuntimeError('measurement failed')
    monkeypatch.setattr(stages, 'evaluate_head_from_cache', fail)
    monkeypatch.setattr(stages, 'export_head_candidate', lambda *_a, **_k: pytest.fail('must not export'))
    with pytest.raises(RuntimeError, match='measurement failed'):
        stages.fit_head_stage(context, tmp_path / 'calibration', tmp_path / 'head')
    assert not (tmp_path / 'head').exists()
    assert sources.local_directory_identity(original['path']) == original


def test_calibration_cannot_mutate_final_state_before_export(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    stages.cache_head_stage(context, 'calibration', tmp_path / 'calibration')
    evaluate = stages.evaluate_head_from_cache
    def mutate(head, cache, **kwargs):
        result = evaluate(head, cache, **kwargs)
        head.raw_scale.data.add_(.01)
        return result
    monkeypatch.setattr(stages, 'evaluate_head_from_cache', mutate)
    monkeypatch.setattr(stages, 'export_head_candidate', lambda *_a, **_k: pytest.fail('must not export'))
    with pytest.raises(ValueError, match='Final head state changed'):
        stages.fit_head_stage(context, tmp_path / 'calibration', tmp_path / 'head')
    assert not (tmp_path / 'head').exists()
