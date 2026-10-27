"""Independent vector arithmetic and masking/streaming contracts for diagnostics."""
import json
import math

import pytest
import torch

from openbonsai.activation_metrics import ActivationMetrics


def tensor(values):
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, -1)


def paired(a, ref, *, chunk=65536):
    return ActivationMetrics(chunk).update(a, ref, a, ref, a, torch.ones(a.shape[:2])).summary()


def test_literal_vectors_and_cross_term_prevent_additive_attribution():
    result = ActivationMetrics(1).update(
        tensor([1, 2]), tensor([2, 1]), tensor([2, 0]), tensor([0, 2]), tensor([1, 1]),
        torch.tensor([[True]]), record_ids=['literal']).summary()
    pair = result['pooled']['pairs']['xq/x0']
    assert pair['valid_elements'] == 2
    assert pair['rms_ratio'] == 1
    assert pair['cosine'] == pytest.approx(.8)
    assert pair['mse'] == 1
    assert pair['relative_rmse'] == pytest.approx(math.sqrt(.4))
    assert pair['best_nonnegative_scale'] == pytest.approx(.8)
    assert pair['rescaled_mse'] == pytest.approx(.9)
    assert pair['rescaled_relative_rmse'] == pytest.approx(.6)
    assert result['records'][0]['record_id'] == 'literal'
    decomposition = result['pooled']['decomposition']
    assert decomposition['sum_terms'] == {
        'total_error': 2., 'conditional_error': 2., 'target_gap': 8., 'cross_term': -8.}
    assert decomposition['terms_divided_by_original_target_energy'] == {
        'total_error': .5, 'conditional_error': .5, 'target_gap': 2., 'cross_term': -2.}
    assert decomposition['terms_divided_by_total_error'] == {
        'total_error': 1., 'conditional_error': 1., 'target_gap': 4., 'cross_term': -4.}
    assert decomposition['target_gap_over_total_error_rms'] == 2
    assert decomposition['closure_residual_sum'] == 0
    assert decomposition['relative_closure_residual'] == 0
    assert 'prevents causal attribution' in decomposition['interpretation']


@pytest.mark.parametrize('a,ref,ratio,cosine,scale,rescaled_mse', [
    ([2, -4], [1, -2], 2, 1, .5, 0),
    ([-2, 4], [1, -2], 2, -1, 0, 2.5),
    ([2, 0], [0, 3], 2/3, 0, 0, 4.5),
])
def test_amplitude_orthogonality_and_nonnegative_scale_constraint(a, ref, ratio, cosine, scale, rescaled_mse):
    p = paired(tensor(a), tensor(ref))['pooled']['pairs']['xq/x0']
    assert p['rms_ratio'] == pytest.approx(ratio)
    assert p['cosine'] == pytest.approx(cosine)
    assert p['best_nonnegative_scale'] == pytest.approx(scale)
    assert p['rescaled_mse'] == pytest.approx(rescaled_mse)
    # The relative MSE identity distinguishes amplitude from direction.
    assert p['relative_rmse']**2 == pytest.approx(ratio**2 + 1 - 2*ratio*cosine)


def test_near_identical_large_vectors_retain_small_scalar_residual():
    p = paired(tensor([1e8, 1e8]), tensor([1e8+1, 1e8-1]))['pooled']['pairs']['xq/x0']
    assert p['mse'] == 1
    assert p['best_nonnegative_scale'] == 1
    assert p['rescaled_mse'] == 1  # A naive ref2-dot**2/a2 loses this residual.


@pytest.mark.parametrize('a,ref,status,ratio,scale,residual', [
    ([0, 0], [0, 0], 'both_zero', None, 0, 0),
    ([0, 0], [1, 2], 'activation_zero', 0, 0, 2.5),
    ([1, 2], [0, 0], 'reference_zero', None, 0, 0),
])
def test_zero_norms_have_explicit_json_safe_meanings(a, ref, status, ratio, scale, residual):
    summary = paired(tensor(a), tensor(ref))
    p = summary['pooled']['pairs']['xq/x0']
    assert p['norm_status'] == status
    assert p['rms_ratio'] == ratio
    assert p['cosine'] is None
    assert p['best_nonnegative_scale'] == scale
    assert p['rescaled_mse'] == residual
    assert p['scale_is_unique'] == any(value != 0 for value in a)
    if not any(ref):
        assert p['relative_rmse'] is p['rescaled_relative_rmse'] is None
    json.dumps(summary, allow_nan=False)


def test_zero_total_error_can_have_nonzero_terms_but_no_error_normalization():
    zero = tensor([0, 0])
    summary = ActivationMetrics().update(zero, zero, tensor([2, -2]), zero, zero, torch.ones(1, 1)).summary()
    d = summary['pooled']['decomposition']
    assert d['sum_terms'] == {'total_error': 0., 'conditional_error': 8., 'target_gap': 8., 'cross_term': -16.}
    assert d['terms_divided_by_total_error'] is None
    assert d['terms_divided_by_original_target_energy'] is None
    assert d['target_gap_over_total_error_rms'] is None
    assert d['total_error_normalization_status'] == 'undefined_zero_total_error'
    assert d['closure_residual_sum'] == 0


def test_padding_is_removed_before_arithmetic_and_inputs_are_detached():
    values = torch.tensor([[[1., 2.], [float('nan'), float('inf')], [3., 4.]]], requires_grad=True)
    reference = torch.tensor([[[2., 1.], [float('-inf'), float('nan')], [4., 3.]]])
    before = values.detach().clone()
    summary = ActivationMetrics(3).update(values, reference, values, reference, values,
                                          torch.tensor([[1, 0, 1]])).summary()
    pair = summary['pooled']['pairs']['xq/x0']
    assert pair['mse'] == 1
    assert pair['valid_elements'] == 4
    assert summary['pooled']['valid_tokens'] == 2
    assert values.grad is None
    torch.testing.assert_close(values, before, rtol=0, atol=0, equal_nan=True)
    json.dumps(summary, allow_nan=False)


def test_pooled_results_weight_elements_across_records_and_update_lengths():
    metrics = ActivationMetrics(2)
    metrics.update(tensor([0]), tensor([1]), tensor([0]), tensor([1]), tensor([0]),
                   torch.ones(1, 1), record_ids=['short'])
    a = torch.zeros(1, 3, 1, dtype=torch.float32)
    ref = torch.full_like(a, 3.)
    metrics.update(a, ref, a, ref, a, torch.ones(1, 3), record_ids=['long'])
    summary = metrics.summary()
    assert summary['record_count'] == 2
    assert [p['pairs']['xq/x0']['mse'] for p in summary['records']] == [1, 9]
    assert summary['pooled']['valid_elements'] == 4
    assert summary['pooled']['pairs']['xq/x0']['mse'] == 7  # Not record mean 5.
    assert summary['pooled']['pairs']['xq/x0']['reference_rms'] == pytest.approx(math.sqrt(7))
    summary['records'][0]['pairs']['xq/x0']['mse'] = -100
    assert metrics.summary()['records'][0]['pairs']['xq/x0']['mse'] == 1


@pytest.mark.parametrize('chunk', [1, 3, 7, 32, 65536])
def test_noncontiguous_chunk_boundaries_match_independent_python_oracle(chunk):
    raw = torch.arange(56, dtype=torch.float64).reshape(2, 7, 4)
    a = (raw/13 - 1).transpose(1, 2)  # [2,4,7], deliberately noncontiguous.
    ref = (raw.flip(1)/17 + .3).transpose(1, 2)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)
    summary = ActivationMetrics(chunk).update(a, ref, a, ref, a, mask).summary()
    av = [value for record in range(2) for token in range(4) if mask[record, token]
          for value in a[record, token].tolist()]
    bv = [value for record in range(2) for token in range(4) if mask[record, token]
          for value in ref[record, token].tolist()]
    aa = math.fsum(x*x for x in av)
    bb = math.fsum(x*x for x in bv)
    dot = math.fsum(x*y for x,y in zip(av, bv))
    error = math.fsum((x-y)**2 for x,y in zip(av, bv))
    p = summary['pooled']['pairs']['xq/x0']
    assert p['valid_elements'] == len(av) == 35
    assert p['mse'] == pytest.approx(error / len(av), rel=1e-14)
    assert p['relative_rmse'] == pytest.approx(math.sqrt(error / bb), rel=1e-14)
    assert p['cosine'] == pytest.approx(dot / math.sqrt(aa*bb), rel=1e-14)
    assert p['best_nonnegative_scale'] == pytest.approx(max(0, dot/aa), rel=1e-14)


def test_cpu_fp64_scratch_is_bounded_even_when_hidden_width_exceeds_chunk():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    a = torch.arange(154, dtype=torch.float32).reshape(2, 7, 11)
    ref = a / 2 + 1

    class BoundDoubleScratch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                    assert value.device.type == 'cpu'
                    assert value.numel() <= 5, f'unbounded FP64 scratch through {func}'
            return result

    with BoundDoubleScratch():
        summary = paired(a, ref, chunk=5)
    assert summary['pooled']['valid_elements'] == a.numel()


@pytest.mark.parametrize('problem', ['shape', 'rank', 'integer', 'mask-shape', 'mask-two',
                                    'mask-nan', 'empty', 'nan', 'inf', 'overflow'])
def test_invalid_records_are_rejected_atomically(problem):
    metrics = ActivationMetrics(2)
    a = torch.ones(2, 2, 3, dtype=torch.float64)
    streams = [a.clone() for _ in range(5)]
    mask = torch.ones(2, 2)
    if problem == 'shape':
        streams[1] = streams[1][:, :, :2]
    elif problem == 'rank':
        streams[1] = streams[1][0]
    elif problem == 'integer':
        streams[1] = streams[1].long()
    elif problem == 'mask-shape':
        mask = mask[:, :1]
    elif problem == 'mask-two':
        mask[1, 0] = 2
    elif problem == 'mask-nan':
        mask[1, 0] = float('nan')
    elif problem == 'empty':
        mask[1].zero_()
    elif problem in ('nan', 'inf'):
        streams[4][1, 1, 1] = float(problem)
    else:
        streams[0][1, 1, 1] = 1e200
    with pytest.raises(ValueError):
        metrics.update(*streams, mask)
    with pytest.raises(ValueError, match='No valid'):
        metrics.summary()


def test_failed_later_update_and_duplicate_ids_do_not_corrupt_previous_results():
    metrics = ActivationMetrics().update(*([tensor([1, 2])] * 5), torch.ones(1, 1), record_ids=['kept'])
    initial = metrics.summary()
    with pytest.raises(ValueError, match='unique'):
        metrics.update(*([tensor([1, 2])] * 5), torch.ones(1, 1), record_ids=['kept'])
    with pytest.raises(ValueError, match='finite'):
        metrics.update(*([tensor([float('nan'), 2])] * 5), torch.ones(1, 1), record_ids=['bad'])
    assert metrics.summary() == initial


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5])
def test_invalid_chunk_limit_rejected(chunk):
    with pytest.raises(ValueError, match='positive integer'):
        ActivationMetrics(chunk)
