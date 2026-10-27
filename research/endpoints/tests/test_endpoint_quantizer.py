"""Dense-vs-bounded math and allocation contracts, not 9B quality claims."""
import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import openbonsai.endpoint_quantizer as eq
from openbonsai.endpoint_quantizer import BoundedTernaryHead
from openbonsai.quantizer import TernaryLinear
from openbonsai.rotations import RotatedTernaryLinear, fixed_signs


def paired(*, rank=3, mapping="sigmoid", rotated=False, hard_phase="frozen_jacobian", rows=7, width=16, chunk=3):
    with torch.random.fork_rng():
        torch.manual_seed(5090)
        source = nn.Linear(width, rows, bias=True, dtype=torch.bfloat16)
        kwargs = dict(group_size=4, lora_rank=rank, factor_mapping=mapping, hard_phase=hard_phase)
        signs = fixed_signs(width, seed=5, name="head") if rotated else None
        dense = (RotatedTernaryLinear(source, signs=signs, block_size=8, **kwargs)
                 if rotated else TernaryLinear(source, **kwargs))
        bounded = BoundedTernaryHead(source, row_chunk_size=chunk, signs=signs, block_size=8, **kwargs)
        with torch.no_grad():
            # Nonzero B exercises derivatives through BOTH A/B and statistics.
            for p in dense.factor_parameters():
                p.copy_(torch.randn_like(p) * .08)
            if rank:
                dense.lora_B.copy_(torch.randn_like(dense.lora_B) * .05)
            for target, original in zip(bounded.trainable_parameters(), dense.trainable_parameters()):
                target.copy_(original)
        hidden = torch.randn(2, 3, width, dtype=torch.bfloat16)
        upstream = torch.randn(2, 3, rows, dtype=torch.bfloat16) * .03
    return source, dense, bounded, hidden, upstream


def spans(head):
    return [(s, min(s + head.row_chunk_size, head.out_features)) for s in range(0, head.out_features, head.row_chunk_size)]


def bounded_outputs(head, hidden):
    return torch.cat([head.forward_rows(hidden, start, end) for start, end in spans(head)], dim=-1)


def backward_rows(head, hidden, upstream):
    for start, end in spans(head):
        # Fresh graph for each callback, as in a memory-bounded calibration loop.
        (head.forward_rows(hidden, start, end).float() * upstream[..., start:end].float()).sum().backward()


def assert_grads(dense, bounded):
    for a, b in zip(dense.trainable_parameters(), bounded.trainable_parameters()):
        assert a.grad is not None and b.grad is not None
        assert a.grad.dtype == b.grad.dtype == torch.float32
        assert bool(torch.count_nonzero(a.grad))
        torch.testing.assert_close(b.grad, a.grad, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("rotated", [False, True])
def test_soft_values_all_gradients_and_one_adam_update_match_dense(mapping, rank, rotated):
    source, dense, bounded, hidden, upstream = paired(mapping=mapping, rank=rank, rotated=rotated)
    original = source.weight.detach().clone()
    dense.set_progress(.23)
    bounded.begin_step(.23)
    expected = dense(hidden)
    actual = bounded_outputs(bounded, hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (expected.float() * upstream.float()).sum().backward()
    backward_rows(bounded, hidden, upstream)
    assert_grads(dense, bounded)
    optimizers = [torch.optim.Adam(model.trainable_parameters(), lr=.003, eps=1e-4) for model in (dense, bounded)]
    for optimizer in optimizers:
        optimizer.step()
    for a, b in zip(dense.trainable_parameters(), bounded.trainable_parameters()):
        torch.testing.assert_close(b, a, atol=2e-6, rtol=2e-5)
    assert torch.equal(source.weight, original)
    assert source.weight.grad is None
    assert bounded.weight.data_ptr() == source.weight.data_ptr()
    bounded.assert_source_unchanged(verify_hash=True)


@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("rotated", [False, True])
def test_hard_values_and_preupdate_saved_jacobian_match_dense(mapping, rank, rotated):
    source, dense, bounded, hidden, upstream = paired(mapping=mapping, rank=rank, rotated=rotated)
    dense.set_progress(.37)
    bounded.begin_step(.37)
    # The dense operator captures its reference during soft forward. Bounded
    # captures before any rows, including a no-grad first-pass use case.
    dense(hidden)
    snapshots = {name: None if getattr(bounded, name) is None else getattr(bounded, name).clone()
                 for name in bounded._snapshot_names()}
    with torch.no_grad():
        bounded_outputs(bounded, hidden)
        for i, (a, b) in enumerate(zip(dense.trainable_parameters(), bounded.trainable_parameters())):
            a.add_(.015 * (i + 1))
            b.copy_(a)
    # Every factor and both LoRA factors differ from the derivative point.
    dense.set_progress(.95)
    bounded.begin_step(.95)
    expected = dense(hidden)
    actual = bounded_outputs(bounded, hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (expected.float() * upstream.float()).sum().backward()
    backward_rows(bounded, hidden, upstream)
    assert_grads(dense, bounded)
    for model in (dense, bounded):
        torch.optim.Adam(model.trainable_parameters(), lr=.001, eps=1e-4).step()
    for a, b in zip(dense.trainable_parameters(), bounded.trainable_parameters()):
        torch.testing.assert_close(b, a, atol=2e-6, rtol=2e-5)
    for name, expected_snapshot in snapshots.items():
        actual_snapshot = getattr(bounded, name)
        if expected_snapshot is None:
            assert actual_snapshot is None
        else:
            assert torch.equal(actual_snapshot, expected_snapshot)
    assert bounded._hard_snapshot_sharpness == pytest.approx(30 * .37 / .8)
    assert source.weight.grad is None


@pytest.mark.parametrize("rank", [0, 3])
def test_current_soft_surrogate_matches_dense(rank):
    _, dense, bounded, hidden, upstream = paired(rank=rank, hard_phase="soft_surrogate", rotated=True)
    dense.set_progress(1)
    bounded.begin_step(1)
    expected = dense(hidden)
    torch.testing.assert_close(bounded_outputs(bounded, hidden), expected, atol=0, rtol=0)
    (expected.float() * upstream.float()).sum().backward()
    backward_rows(bounded, hidden, upstream)
    assert_grads(dense, bounded)


def test_snapshot_captured_once_by_begin_step_not_row_callbacks(monkeypatch):
    _, _, bounded, hidden, _ = paired()
    with torch.no_grad():
        bounded.begin_step(.5)
    snapshot = bounded._hard_snapshot_lora_B.clone()
    marker = bounded._snapshot_step
    bounded.forward_rows(hidden, 0, 3)
    with torch.no_grad():
        bounded.lora_B.add_(1)
    bounded.forward_rows(hidden, 3, 6)
    assert torch.equal(bounded._hard_snapshot_lora_B, snapshot)
    assert bounded._snapshot_step == marker
    bounded.begin_step(.6)
    assert bounded._snapshot_step != marker
    assert torch.equal(bounded._hard_snapshot_lora_B, bounded.lora_B)
    bounded.begin_step(.4)
    assert bounded._hard_snapshot_sharpness == pytest.approx(15)
    bounded.begin_step(0)
    assert bounded._hard_snapshot_sharpness is None
    assert all(getattr(bounded, name) is None for name in bounded._snapshot_names())


def test_hard_inference_without_soft_point_and_training_requires_it():
    _, _, bounded, hidden, _ = paired()
    bounded.begin_step(1)
    with pytest.raises(RuntimeError, match="preceding soft begin_step"):
        bounded.forward_rows(hidden, 0, 3)
    with torch.no_grad():
        assert torch.isfinite(bounded.forward_rows(hidden, 0, 3)).all()
    bounded.requires_grad_(False)
    assert torch.isfinite(bounded.forward_rows(hidden, 0, 3)).all()


@pytest.mark.parametrize("rotated", [False, True])
def test_teacher_and_zero_progress_ignore_adapters_and_rotations(rotated):
    source, dense, bounded, hidden, _ = paired(rotated=rotated)
    for start, end in spans(bounded):
        expected = F.linear(hidden, source.weight[start:end], source.bias[start:end])
        assert torch.equal(bounded.teacher_rows(hidden, start, end), expected)
        assert torch.equal(bounded.forward_rows(hidden, start, end), expected)
    if not rotated:
        bounded_outputs(bounded, hidden).sum().backward()
        assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in bounded.trainable_parameters())
    bounded.begin_step(.3)
    bounded.quant_enabled = False
    torch.testing.assert_close(bounded_outputs(bounded, hidden), F.linear(hidden, source.weight, source.bias), atol=0, rtol=0)


@pytest.mark.parametrize("rotated", [False, True])
def test_export_codes_scales_exact_dense_match_with_tail_rows(rotated):
    _, dense, bounded, _, _ = paired(rotated=rotated)
    codes, scales = dense._hard_codes_and_scales()
    codes = codes.reshape(bounded.out_features, bounded.in_features).to(torch.int8)
    scales = scales.reshape(bounded.out_features, bounded.groups_per_row)
    for start, end in spans(bounded):
        actual_codes, actual_scales = bounded.hard_codes_rows(start, end)
        assert actual_codes.dtype == torch.int8
        assert actual_scales.dtype == torch.float32
        assert torch.equal(actual_codes, codes[start:end])
        torch.testing.assert_close(actual_scales, scales[start:end], atol=1e-8, rtol=1e-6)
    bounded.validate_parameters()


def test_dtype_moves_keep_original_bf16_and_all_compact_state_fp32():
    source, _, bounded, hidden, upstream = paired(rotated=True)
    bounded.begin_step(.3)
    backward_rows(bounded, hidden, upstream)
    tensors = {name: value.detach().clone() for name, value in bounded.named_parameters()}
    tensors.update({name: value.clone() for name, value in bounded.named_buffers()
                    if name in ("mu0", "alpha0", *bounded._snapshot_names())})
    source_before = source.weight.clone()
    bounded.half().float().bfloat16()
    assert bounded.weight.dtype == torch.bfloat16
    assert torch.equal(bounded.weight, source_before)
    assert torch.equal(source.weight, source_before)
    for name, expected in tensors.items():
        actual = getattr(bounded, name)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
    assert all(p.grad.dtype == torch.float32 for p in bounded.trainable_parameters())
    bounded.assert_source_unchanged(verify_hash=True)


def test_external_source_mutations_rejected_including_data_bypass():
    source, _, bounded, hidden, _ = paired()
    expected = hashlib.sha256(source.weight.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert bounded.source_weight_sha256 == expected
    with torch.no_grad():
        source.weight[0, 0].add_(1)
    with pytest.raises(RuntimeError, match="source was modified"):
        bounded.teacher_rows(hidden, 0, 1)
    source, _, bounded, _, _ = paired()
    source.weight.data[0, 0].add_(1)
    with pytest.raises(RuntimeError, match="source hash changed"):
        bounded.assert_source_unchanged(verify_hash=True)


def test_bad_mapped_factors_and_nonfinite_parameters_rejected():
    _, _, head, _, _ = paired()
    with torch.no_grad():
        head.raw_scale[0] = -10000
    with pytest.raises(ValueError, match="mapped"):
        head.hard_codes_rows(0, 1)
    with torch.no_grad():
        head.raw_scale[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        head.validate_parameters()
    with pytest.raises(ValueError, match="nonfinite"):
        head.hard_codes_rows(0, 1)


def test_no_full_matrix_temporary_init_or_row_reconstruction(monkeypatch):
    # A mechanism probe: every source FP32 promotion/rotation and every BA
    # product is confined to <=chunk rows; persistent compact factors may span
    # all rows. This is not a real 9B peak-memory measurement.
    source = nn.Linear(4096, 11, dtype=torch.bfloat16)
    observations = []
    original_rotate = eq.rotate
    def checked_rotate(value, signs, block_size):
        assert value.shape[-1] == 4096
        assert value.shape[0] <= 3
        observations.append(tuple(value.shape))
        return original_rotate(value, signs, block_size)
    monkeypatch.setattr(eq, "rotate", checked_rotate)
    head = BoundedTernaryHead(source, row_chunk_size=3, lora_rank=2,
                             signs=fixed_signs(4096, seed=2))
    original_components = head._components
    def checked_components(start, end, parameters, **kwargs):
        assert end - start <= 3
        assert parameters[4].shape[0] <= 3
        return original_components(start, end, parameters, **kwargs)
    monkeypatch.setattr(head, "_components", checked_components)
    head.begin_step(.2)
    hidden = torch.randn(2, 4096, dtype=torch.bfloat16)
    for start, end in spans(head):
        head.forward_rows(hidden, start, end).float().sum().backward()
        head.hard_codes_rows(start, end)
    assert observations
    assert head.weight.data_ptr() == source.weight.data_ptr()
    with pytest.raises(ValueError, match="row_chunk_size"):
        head.forward_rows(hidden, 0, 11)
    with pytest.raises(RuntimeError, match="no full-head"):
        head(hidden)


@pytest.mark.parametrize("kwargs", [{"group_size": 3}, {"row_chunk_size": 0}, {"lora_rank": True},
    {"soft_fraction": float("nan")}, {"soft_fraction": 1}, {"max_sharpness": 0},
    {"factor_mapping": "bad"}, {"hard_phase": "bad"}])
def test_invalid_constructor_contract(kwargs):
    with pytest.raises(ValueError):
        BoundedTernaryHead(nn.Linear(16, 7, dtype=torch.bfloat16), **kwargs)


def test_invalid_source_rows_hidden_and_progress():
    with pytest.raises(ValueError, match="BF16"):
        BoundedTernaryHead(nn.Linear(128, 7))
    _, _, head, hidden, _ = paired()
    for start, end in ((-1, 1), (0, 0), (0, 8), (True, 2), (0, 4)):
        with pytest.raises(ValueError): head.forward_rows(hidden, start, end)
    for progress in (-1, 1.1, float("nan")):
        with pytest.raises(ValueError): head.begin_step(progress)
    with pytest.raises(ValueError): head.forward_rows(hidden.float(), 0, 1)
