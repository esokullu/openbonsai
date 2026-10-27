"""Actual budget constraints, manifold geometry, and model interaction tests."""
import itertools
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openbonsai.allocation import (
    AssembledCandidateModel, ByteBudgetDecoder, CandidateTable, budget_normal,
    build_candidate_table, causal_lm_loss, expected_cost, optimize_rco,
    project_gradient, retract, transport_momentum,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_exact_decoder_matches_exhaustive_known_solution_with_unequal_sizes():
    costs = torch.tensor([[7, 15, 31], [3, 13, 23], [11, 21, 42]], dtype=torch.int64)
    scores = torch.tensor([[0., 2., 4.], [1., 8., 10.], [0., 3., 9.]])
    fixed, budget = 9, 70
    possibilities = [(sum(float(scores[i, j]) for i, j in enumerate(choice)), choice)
                     for choice in itertools.product(range(3), repeat=3)
                     if fixed + sum(int(costs[i, j]) for i, j in enumerate(choice)) <= budget]
    expected_score, expected_choice = max(possibilities)
    result = ByteBudgetDecoder(costs, budget, fixed_bytes=fixed).decode(scores)
    assert result.exact_optimization
    assert result.indices == expected_choice
    assert result.score == expected_score
    assert result.total_bytes == fixed + sum(int(costs[i, j]) for i, j in enumerate(expected_choice))


def test_decoder_does_not_prefer_filling_budget_over_maximum_score():
    result = ByteBudgetDecoder(torch.tensor([[1, 2]]), 2).decode(torch.tensor([[9., 1.]]))
    assert result.indices == (0,)
    assert result.total_bytes == 1


def test_conservative_decoder_never_repeats_nearest_grid_budget_violation():
    # The preferred mixed assignment costs 280024 bytes, above 280000.
    costs = torch.tensor([[80016, 200040], [79984, 199960]], dtype=torch.int64)
    decoder = ByteBudgetDecoder(costs, 280000, max_bins=2)
    result = decoder.decode(torch.tensor([[0., 10.], [1., 0.]]))
    assert not result.exact_optimization
    assert result.total_bytes <= 280000
    assert result.indices != (1, 0)


def test_ragged_candidates_zero_cost_and_minimum_budget_remain_reachable():
    costs = torch.tensor([[0, 7, 0], [13, 0, 0], [5, 20, 21]], dtype=torch.int64)
    valid = torch.tensor([[True, True, False], [True, False, False], [True, True, True]])
    scores = torch.tensor([[1., 9., float('nan')], [4., float('nan'), float('nan')], [0., 8., 9.]])
    result = ByteBudgetDecoder(costs, 21, fixed_bytes=3, valid=valid, max_bins=1).decode(scores)
    assert result.indices == (0, 0, 0)
    assert result.total_bytes == 21


def test_infeasible_budget_fails_before_calling_objective():
    costs = torch.tensor([[4, 8], [7, 10]], dtype=torch.int64)
    with pytest.raises(ValueError, match="minimum 14"):
        ByteBudgetDecoder(costs, 13, fixed_bytes=3)


def test_gcd_keeps_large_byte_counts_exact_without_huge_workspace():
    costs = torch.tensor([[10**9, 2 * 10**9], [3 * 10**9, 5 * 10**9]], dtype=torch.int64)
    decoder = ByteBudgetDecoder(costs, 6 * 10**9, max_bins=4)
    assert decoder.byte_unit == 10**9
    assert decoder.exact_optimization
    assert decoder.decode(torch.tensor([[0., 1.], [0., 4.]])).indices == (0, 1)


def test_budget_normal_matches_autograd_for_tensor_specific_ragged_costs():
    alpha = torch.tensor([[.1, -.2, .5], [.2, .3, .9]], dtype=torch.float64, requires_grad=True)
    costs = torch.tensor([[1., 3., 6.], [2., 7., 0.]], dtype=torch.float64)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    actual, = torch.autograd.grad(expected_cost(alpha, costs, valid), alpha)
    torch.testing.assert_close(actual, budget_normal(alpha, costs, valid), rtol=1e-13, atol=1e-13)


def test_retraction_projection_and_adam_momentum_transport():
    alpha = nn.Parameter(torch.tensor([[.1, -.2], [.4, .2]], dtype=torch.float64))
    costs = torch.tensor([[1., 3.], [2., 8.]], dtype=torch.float64)
    valid = torch.ones_like(costs, dtype=torch.bool)
    assert retract(alpha, costs, valid, 7., tolerance=1e-11) == pytest.approx(7, abs=1e-11)
    alpha.grad = torch.tensor([[1., 3.], [-1., 8.]], dtype=torch.float64)
    raw, projected = project_gradient(alpha, costs, valid)
    assert projected <= raw
    assert float((alpha.grad * budget_normal(alpha, costs, valid)).sum().detach()) == pytest.approx(0, abs=1e-12)
    optimizer = torch.optim.Adam([alpha], lr=.03)
    optimizer.step()
    retract(alpha, costs, valid, 7., tolerance=1e-11)
    transport_momentum(optimizer, alpha, costs, valid)
    moment = optimizer.state[alpha]['exp_avg']
    assert float((moment * budget_normal(alpha, costs, valid)).sum().detach()) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('target', [2., 12., float('nan')])
def test_retraction_rejects_unattainable_expected_budgets(target):
    costs = torch.tensor([[1., 3.], [2., 8.]], dtype=torch.float64)
    with pytest.raises(ValueError, match='outside'):
        retract(torch.zeros_like(costs), costs, torch.ones_like(costs, dtype=torch.bool), target)


def test_rco_forward_is_hard_feasible_and_gradients_finite_on_interaction_loss():
    # Multiplicative two-layer model: errors cannot be ranked independently.
    # Feasible outcomes are 1, 2, 3; target 3 makes upgrading the second layer
    # optimal even though its effect depends on the first layer's selection.
    costs = torch.tensor([[10, 20], [10, 20]], dtype=torch.int64)
    table = CandidateTable(('a', 'b'), ((None, None), (None, None)), costs, torch.ones_like(costs, dtype=torch.bool))
    seen = []
    def objective(c):
        torch.testing.assert_close(c.detach().sum(dim=1), torch.ones(2, dtype=c.dtype))
        assert torch.all((c.detach() == 0) | (c.detach() == 1))
        assert float((c.detach().cpu() * costs).sum()) <= 30
        first = c[0] @ c.new_tensor([1., 2.])
        second = c[1] @ c.new_tensor([1., 3.])
        loss = (first * second - 3).square()
        seen.append(float(loss.detach()))
        return loss
    result = optimize_rco(table, 30, objective, steps=80, learning_rate=.15, seed=4, tau_end=.1)
    assert result.allocation.indices == (0, 1)
    assert result.allocation.total_bytes == 30
    assert len(seen) == 80
    assert all(math.isfinite(h['raw_gradient_norm']) for h in result.history)
    assert all(h['expected_bytes'] == pytest.approx(30, abs=1e-5) for h in result.history)
    assert result.method.startswith('rco-')


def test_causal_loss_masks_padding_and_chunks_match_independent_reference():
    torch.manual_seed(8)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3, 4, 0], [0, 0, 1, 3, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
    positions = mask[:, 1:].bool() & mask[:, :-1].bool()
    reference = F.cross_entropy(logits[:, :-1][positions], tokens[:, 1:][positions])
    actual = causal_lm_loss(logits, tokens, mask, token_chunk_size=2)
    torch.testing.assert_close(actual, reference)
    gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
    reference_gradient, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(gradient, reference_gradient)


def test_assembled_context_interaction_gradients_and_exception_restoration():
    from openbonsai.mixed_formats import bf16_candidate
    model = nn.Sequential(nn.Linear(1, 1, bias=True), nn.Linear(1, 1, bias=False)).to(torch.bfloat16)
    with torch.no_grad():
        model[0].weight.fill_(7)
        model[0].bias.fill_(.5)
        model[1].weight.fill_(11)
    originals = tuple(model.children())
    table = build_candidate_table({
        '0': [bf16_candidate(torch.tensor([[1.]], dtype=torch.bfloat16), originals[0].bias),
              bf16_candidate(torch.tensor([[2.]], dtype=torch.bfloat16), originals[0].bias)],
        '1': [bf16_candidate(torch.tensor([[1.]], dtype=torch.bfloat16)),
              bf16_candidate(torch.tensor([[3.]], dtype=torch.bfloat16))],
    })
    coefficients = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    with pytest.raises(RuntimeError, match='intentional'):
        with AssembledCandidateModel(model, table) as assembled:
            result = assembled(coefficients, torch.tensor([[2.]], dtype=torch.bfloat16))
            torch.testing.assert_close(result.float(), torch.tensor([[7.5]]))
            result.sum().backward()
            # d[(2*a + .5)*b]/d[a0,a1,b0,b1] = [6,12,2.5,7.5]
            torch.testing.assert_close(coefficients.grad, torch.tensor([[6., 12.], [2.5, 7.5]]))
            assert all(p.grad is None for original in originals for p in original.parameters())
            assert not model.training
            raise RuntimeError('intentional')
    assert tuple(model.children()) == originals
    assert model.training
    assert all(p.requires_grad for p in model.parameters())


def test_candidate_table_cost_is_serialized_size_not_nominal_density():
    from openbonsai.mixed_formats import bf16_candidate, candidate_nbytes, encode_candidate
    table = build_candidate_table({
        'small': [bf16_candidate(torch.zeros(2, 3, dtype=torch.bfloat16))],
        'large': [bf16_candidate(torch.zeros(9, 7, dtype=torch.bfloat16)),
                  bf16_candidate(torch.ones(9, 7, dtype=torch.bfloat16))],
    }, fixed_bytes=137)
    assert table.valid.tolist() == [[True, True], [True, False]]
    for i, row in enumerate(table.candidates):
        for j, candidate in enumerate(row):
            assert int(table.costs_bytes[i, j]) == candidate_nbytes(candidate) == len(encode_candidate(candidate))
    assert table.minimum_bytes == 137 + sum(int(table.costs_bytes[i, 0]) for i in range(2))


def test_real_ternary_density_affine_widths_and_ragged_candidate_costs():
    from openbonsai.mixed_formats import affine_candidate, bf16_candidate, encode_candidate, ternary_candidate
    from openbonsai.packing import pack_grouped_weight
    def ternary(nonzero):
        return ternary_candidate(pack_grouped_weight(
            torch.full((8, 64), nonzero, dtype=torch.int8), torch.ones(8),
            shape=(8, 64), group_size=64, packing='bitcos'), packing='bitcos')
    sparse, dense = ternary(0), ternary(1)
    def affine(bits, shape):
        rows, width = shape
        groups = (width + 7) // 8
        return affine_candidate(torch.zeros(shape, dtype=torch.int32), torch.ones(rows, groups),
                                torch.zeros(rows, groups, dtype=torch.int32), bits=bits, group_size=8)
    table = build_candidate_table({
        'a': [sparse, dense, affine(4, (8, 64)), affine(6, (8, 64)),
              bf16_candidate(torch.zeros(8, 64, dtype=torch.bfloat16))],
        'b': [affine(4, (3, 13)), affine(6, (3, 13)),
              bf16_candidate(torch.zeros(3, 13, dtype=torch.bfloat16))],
    }, fixed_bytes=101)
    assert table.costs_bytes[0, 0] < table.costs_bytes[0, 1]
    assert table.valid[1].tolist() == [True, True, True, False, False]
    result = ByteBudgetDecoder(table.costs_bytes, table.minimum_bytes + 101,
                               valid=table.valid, fixed_bytes=table.fixed_bytes).decode(torch.zeros_like(table.costs_bytes).float())
    selected = table.selected_candidates(result.indices)
    assert result.total_bytes == 101 + sum(len(encode_candidate(c)) for c in selected.values())
    assert result.total_bytes <= table.minimum_bytes + 101


def test_assembled_embedding_backpropagates_only_to_candidate_coefficients():
    from openbonsai.mixed_formats import bf16_candidate
    model = nn.Sequential(nn.Embedding(3, 2)).to(torch.bfloat16)
    first = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    second = first * 2
    table = build_candidate_table({'0': [bf16_candidate(first, module_kind='embedding'),
                                       bf16_candidate(second, module_kind='embedding')]})
    coefficients = torch.tensor([[1., 0.]], requires_grad=True)
    with AssembledCandidateModel(model, table) as assembled:
        result = assembled(coefficients, torch.tensor([[0, 2]]))
        torch.testing.assert_close(result, first[torch.tensor([[0, 2]])])
        result.sum().backward()
    torch.testing.assert_close(coefficients.grad, torch.tensor([[10., 20.]]))
    assert model[0].weight.grad is None


def test_assembled_context_rejects_bias_mismatch_without_mutating_model():
    from openbonsai.mixed_formats import bf16_candidate
    model = nn.Sequential(nn.Linear(2, 2)).to(torch.bfloat16)
    original = model[0]
    table = build_candidate_table({'0': [bf16_candidate(original.weight)]})
    with pytest.raises(ValueError, match='bias'):
        with AssembledCandidateModel(model, table):
            pass
    assert model[0] is original and model.training
    assert original.weight.requires_grad
