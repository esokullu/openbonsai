import io
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openbonsai.packing import unpack_ternary_codes
from openbonsai.quantizer import PackedTernaryLinear, TernaryLinear, hard_ternary, restore_ternary_linear, soft_ternary


def linear_with_weight(values, bias=None):
    weight = torch.tensor(values, dtype=torch.float32)
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(torch.tensor(bias))
    return linear


def test_soft_formula_analytic_gradient_and_gradcheck():
    z = torch.tensor([-1.1, -0.3, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    threshold = torch.tensor(0.43, dtype=torch.float64, requires_grad=True)
    k = 2.7
    out = soft_ternary(z, threshold, k)
    gz, gt = torch.autograd.grad(out.sum(), (z, threshold))
    minus, plus = torch.tanh(k * (z - threshold)), torch.tanh(k * (z + threshold))
    expected_gz = k * (2 - minus.square() - plus.square()) / (2 * math.tanh(k))
    expected_gt = (k * (minus.square() - plus.square()) / (2 * math.tanh(k))).sum()
    torch.testing.assert_close(gz, expected_gz)
    torch.testing.assert_close(gt, expected_gt)
    assert torch.autograd.gradcheck(lambda a, b: soft_ternary(a, b, k), (z, threshold))
    torch.testing.assert_close(soft_ternary(z, threshold, 0), z, rtol=0, atol=0)


def test_hard_threshold_boundaries_are_zero():
    z = torch.tensor([-0.6, -0.5, 0.0, 0.5, 0.6])
    assert hard_ternary(z, 0.5).tolist() == [-1, 0, 0, 0, 1]


def test_original_weights_frozen_teacher_and_dense_endpoint():
    torch.manual_seed(12)
    linear = nn.Linear(7, 3)
    layer = TernaryLinear.from_linear(linear, group_size=4, lora_rank=0)
    assert set(dict(layer.named_parameters())) == {"raw_mu", "raw_scale", "raw_threshold"}
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert len(tuple(layer.trainable_parameters())) == 3
    x = torch.randn(2, 7)
    torch.testing.assert_close(layer(x), linear(x), rtol=0, atol=0)
    layer(x).sum().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in layer.trainable_parameters())
    layer.set_progress(1)
    layer.quant_enabled = False
    torch.testing.assert_close(layer(x), linear(x), rtol=0, atol=0)
    layer.quant_enabled = True
    with torch.no_grad():
        assert not torch.equal(layer(x), linear(x))


def test_group_stats_ignore_padding_and_no_mean_addback():
    layer = TernaryLinear.from_linear(linear_with_weight([[2, 3, 5]]), group_size=4)
    torch.testing.assert_close(layer.mu0, torch.tensor([[10 / 3]]))
    torch.testing.assert_close(layer.alpha0, torch.tensor([[10 / 9]]))
    expected = torch.tensor([[-10 / 9, 0, 10 / 9]])
    torch.testing.assert_close(layer.hard_weight(), expected)
    state = layer.packed_state()
    assert unpack_ternary_codes(state["codes"], state["padded_numel"]).tolist() == [-1, 0, 1, 0]


def test_soft_factor_gradients_reach_only_factors():
    torch.manual_seed(8)
    layer = TernaryLinear.from_linear(nn.Linear(8, 4), group_size=8, lora_rank=0)
    layer.set_progress(0.25)
    loss = (layer(torch.randn(3, 8)) - torch.randn(3, 4)).square().mean()
    loss.backward()
    assert layer.weight.grad is None
    assert layer.bias.grad is None
    for p in layer.trainable_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_frozen_jacobian_matches_last_soft_derivative_after_factor_updates():
    torch.manual_seed(19)
    layer = TernaryLinear.from_linear(nn.Linear(8, 4, bias=False), group_size=8, lora_rank=0)
    with torch.no_grad():
        layer.raw_mu.uniform_(-0.1, 0.1)
        layer.raw_scale.uniform_(-0.1, 0.1)
        layer.raw_threshold.uniform_(-0.1, 0.1)
    coeff = torch.randn_like(layer.weight)
    layer.set_progress(0.8)
    expected = torch.autograd.grad((layer.quantized_weight() * coeff).sum(), tuple(layer.trainable_parameters()))
    layer.set_progress(0.81)
    with torch.no_grad():
        layer.raw_mu.add_(0.15)
        layer.raw_scale.sub_(0.12)
        layer.raw_threshold.add_(0.11)
    actual = torch.autograd.grad((layer.quantized_weight() * coeff).sum(), tuple(layer.trainable_parameters()))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(layer.quantized_weight(), layer.hard_weight(), rtol=0, atol=0)
    snapshot = layer._hard_snapshot_mu.clone()
    layer.set_progress(1)
    torch.testing.assert_close(layer._hard_snapshot_mu, snapshot, rtol=0, atol=0)
    layer.reset_annealing()
    assert layer.progress == 0 and layer._hard_snapshot_mu is None
    layer.set_progress(1)
    assert layer._hard_snapshot_mu is None
    with pytest.raises(RuntimeError, match="preceding soft forward"):
        layer.quantized_weight()


def test_dynamic_soft_surrogate_has_hard_forward_and_current_soft_gradient():
    torch.manual_seed(31)
    layer = TernaryLinear.from_linear(nn.Linear(7, 3, bias=False), group_size=4, hard_phase="soft_surrogate")
    layer.set_progress(1)
    actual = torch.autograd.grad(layer.quantized_weight().sum(), tuple(layer.trainable_parameters()))
    layer.set_progress(0.8)
    expected = torch.autograd.grad(layer.quantized_weight().sum(), tuple(layer.trainable_parameters()))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    layer.set_progress(1)
    torch.testing.assert_close(layer.quantized_weight(), layer.hard_weight(), rtol=0, atol=0)


@pytest.mark.parametrize("packing", ["2bit", "bitcos", "auto"])
def test_packed_restore_matches_hard_inference_with_fp16_rounding(packing):
    torch.manual_seed(4)
    layer = TernaryLinear.from_linear(nn.Linear(7, 5), group_size=4)
    with torch.no_grad():
        layer.raw_mu.uniform_(-0.1, 0.1)
        layer.raw_scale.uniform_(-0.1, 0.1)
    layer.set_progress(1)
    state = layer.packed_state(packing=packing)
    assert set(state).isdisjoint({"weight", "raw_mu", "raw_scale", "raw_threshold", "mu0", "alpha0"})
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    packed = TernaryLinear.from_packed_state(loaded, dtype=torch.float32)
    assert isinstance(packed, PackedTernaryLinear)
    assert not list(packed.parameters())
    x = torch.randn(3, 7)
    with torch.no_grad():
        torch.testing.assert_close(packed(x), layer(x), rtol=0.003, atol=0.0006)
    torch.testing.assert_close(packed(x), F.linear(x, packed.weight, packed.bias.float()), rtol=0, atol=0)
    restored_again = restore_ternary_linear(packed.packed_state())
    torch.testing.assert_close(restored_again(x), packed(x), rtol=0, atol=0)
    original_scales, original_bias = packed.scales.clone(), packed.bias.clone()
    packed.to(dtype=torch.bfloat16)
    assert packed.weight.dtype == torch.bfloat16
    assert packed.scales.dtype == packed.bias.dtype == torch.float16
    torch.testing.assert_close(packed.scales, original_scales, rtol=0, atol=0)
    torch.testing.assert_close(packed.bias, original_bias, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("zeros", [False, True])
def test_bitcos_linear_is_exactly_equal_to_two_bit_reference(dtype, zeros):
    torch.manual_seed(24)
    linear = nn.Linear(31, 9)
    if zeros:
        with torch.no_grad():
            linear.weight.zero_()
    layer = TernaryLinear(linear, group_size=13)
    old = restore_ternary_linear(layer.packed_state(packing="2bit"), dtype=dtype)
    new = restore_ternary_linear(layer.packed_state(packing="bitcos"), dtype=dtype)
    assert new.codes is None and new.presence.dtype == new.signs.dtype == torch.uint8
    if zeros:
        assert new.signs.numel() == 0
    # Equal numeric values are insufficient for signed zeros: compare bytes too.
    assert torch.equal(old.weight.view(torch.uint8), new.weight.view(torch.uint8))
    assert torch.equal(old.scales.view(torch.uint8), new.scales.view(torch.uint8))
    assert torch.equal(old.bias.view(torch.uint8), new.bias.view(torch.uint8))
    x = torch.randn(2, 31).to(dtype)
    torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)
    old.to(dtype=torch.bfloat16)
    new.to(dtype=torch.bfloat16)
    assert new.presence.dtype == new.signs.dtype == torch.uint8
    torch.testing.assert_close(new(x.to(torch.bfloat16)), old(x.to(torch.bfloat16)), rtol=0, atol=0)
    again = restore_ternary_linear(new.packed_state(packing="2bit"), dtype=torch.bfloat16)
    assert torch.equal(again.weight.view(torch.uint8), new.weight.view(torch.uint8))


def test_hard_evaluation_skips_backward_surrogate(monkeypatch):
    layer = TernaryLinear.from_linear(nn.Linear(4, 4), group_size=4)
    layer.set_progress(1)
    def unexpected():
        raise AssertionError("unused backward surrogate evaluated")
    monkeypatch.setattr(layer, "_frozen_jacobian_surrogate", unexpected)
    with torch.no_grad():
        torch.testing.assert_close(layer.quantized_weight(), layer.hard_weight(), rtol=0, atol=0)
    for p in layer.trainable_parameters():
        p.requires_grad_(False)
    torch.testing.assert_close(layer.quantized_weight(), layer.hard_weight(), rtol=0, atol=0)


def test_export_rejects_nonfinite_and_overflowed_factors():
    layer = TernaryLinear.from_linear(nn.Linear(4, 4))
    with torch.no_grad():
        layer.raw_mu.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        layer.packed_state()
    with torch.no_grad():
        layer.raw_mu.zero_()
        layer.raw_threshold.fill_(1000)
    with pytest.raises(ValueError, match="mapped"):
        layer.packed_state()


@pytest.mark.parametrize("value", [0.0, 2.0])
def test_degenerate_groups_stay_finite(value):
    layer = TernaryLinear.from_linear(linear_with_weight([[value] * 3]), group_size=4)
    for p in (0, 0.1, 0.8, 1):
        layer.set_progress(p)
        assert torch.isfinite(layer(torch.ones(1, 3))).all()
    assert torch.count_nonzero(layer.hard_weight()) == 0
    assert torch.isfinite(restore_ternary_linear(layer.packed_state()).weight).all()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_reject_bad_progress(value):
    with pytest.raises(ValueError, match="progress"):
        TernaryLinear.from_linear(nn.Linear(2, 2)).set_progress(value)


def test_default_lora_initialization_parameter_groups_and_dense_teacher():
    torch.manual_seed(107)
    source = nn.Linear(7, 3, bias=True, dtype=torch.bfloat16)
    layer = TernaryLinear(source)
    assert layer.lora_rank == 64
    assert layer.lora_A.shape == (64, 7)
    assert layer.lora_B.shape == (3, 64)
    assert torch.count_nonzero(layer.lora_A) > 0
    assert torch.count_nonzero(layer.lora_B) == 0
    assert layer.lora_A.abs().max() <= 1 / math.sqrt(7)
    assert len(layer.factor_parameters()) == 3
    assert layer.lora_parameters() == (layer.lora_A, layer.lora_B)
    assert layer.trainable_parameters() == layer.factor_parameters() + layer.lora_parameters()
    assert all(p.dtype == torch.float32 for p in layer.trainable_parameters())
    assert layer.mu0.dtype == layer.alpha0.dtype == torch.float32
    x = torch.randn(2, 7, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.lora_B.normal_(0, 2)
    torch.testing.assert_close(layer(x), source(x), rtol=0, atol=0)
    layer.set_progress(1)
    layer.quant_enabled = False
    torch.testing.assert_close(layer(x), source(x), rtol=0, atol=0)
    assert not layer.weight.requires_grad and not layer.bias.requires_grad


@pytest.mark.parametrize("rank", [-1, True, 1.5, None])
def test_reject_invalid_lora_rank(rank):
    with pytest.raises(ValueError, match="nonnegative integer"):
        TernaryLinear(nn.Linear(2, 2), lora_rank=rank)


def test_rank_zero_keeps_legacy_hard_formula_and_no_lora_parameters():
    source = linear_with_weight([[-0.8, -0.1, 0.3, 0.9], [0.2, -0.4, 0.7, -0.6]])
    layer = TernaryLinear(source, group_size=4, lora_rank=0)
    assert layer.lora_A is None and layer.lora_B is None
    assert layer.lora_parameters() == ()
    assert len(layer.trainable_parameters()) == 3
    groups = source.weight.detach().reshape(-1, 4)
    mean = groups.mean(1, keepdim=True)
    scale = (groups - mean).abs().mean(1, keepdim=True)
    normalized = (groups - mean) / scale
    expected = ((normalized > .5).float() - (normalized < -.5).float()) * scale
    torch.testing.assert_close(layer.hard_weight(), expected.reshape_as(source.weight), rtol=0, atol=0)


@pytest.mark.parametrize("packing", ["2bit", "bitcos"])
def test_low_rank_update_changes_codes_and_is_merged_before_export(packing):
    source = linear_with_weight([[-2, -1, 1, 2]], bias=[.125])
    layer = TernaryLinear(source, group_size=4, lora_rank=1)
    original_codes = layer._hard_codes_and_scales()[0].clone()
    with torch.no_grad():
        layer.lora_A.copy_(torch.tensor([[0., 4., 0., 0.]]))
        layer.lora_B.fill_(1)
    effective = torch.tensor([[-2., 3., 1., 2.]])
    mean = effective.mean(1, keepdim=True)
    scale = (effective - mean).abs().mean(1, keepdim=True)
    z = (effective - mean) / scale
    expected_codes = (z > .5).float() - (z < -.5).float()
    expected_weight = expected_codes * scale
    codes, actual_scale = layer._hard_codes_and_scales()
    assert not torch.equal(original_codes, codes)
    torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(layer.hard_weight(), expected_weight, rtol=0, atol=0)
    state = layer.packed_state(packing=packing)
    assert not any("lora" in key for key in state)
    restored = restore_ternary_linear(state)
    assert list(restored.parameters()) == []
    torch.testing.assert_close(restored.weight, expected_weight, rtol=0, atol=0)
    x = torch.tensor([[1., 2., -1., .5]])
    torch.testing.assert_close(restored(x), F.linear(x, expected_weight, source.bias), rtol=0, atol=0)


def test_lora_gradients_start_in_b_then_reach_both_matrices():
    torch.manual_seed(31)
    layer = TernaryLinear(nn.Linear(16, 5), group_size=8, lora_rank=3)
    x, target = torch.randn(7, 16), torch.randn(7, 5)
    layer.set_progress(.2)
    loss = (layer(x) - target).square().mean()
    loss.backward()
    assert layer.lora_A.grad is not None and torch.count_nonzero(layer.lora_A.grad) == 0
    assert layer.lora_B.grad is not None and layer.lora_B.grad.norm() > 0
    with torch.no_grad():
        layer.lora_B.add_(layer.lora_B.grad, alpha=-.1)
    layer.zero_grad(set_to_none=True)
    (layer(x) - target).square().mean().backward()
    for parameter in layer.trainable_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert layer.weight.grad is None and layer.bias.grad is None


def _independent_soft_weight(weight, parameters, rank, group_size, sharpness):
    """Direct paper equations, independent of the module's factor/snapshot helpers."""
    rm, rs, rt = parameters[:3]
    if rank:
        a, b = parameters[3:]
        effective = weight + b @ a / rank
    else:
        effective = weight
    padded_numel = math.ceil(weight.numel() / group_size) * group_size
    grouped = F.pad(effective.flatten(), (0, padded_numel - weight.numel())).reshape(-1, group_size)
    valid = (torch.arange(padded_numel).reshape_as(grouped) < weight.numel()).to(grouped.dtype)
    count = valid.sum(1, keepdim=True)
    mu0 = (grouped * valid).sum(1, keepdim=True) / count
    alpha0 = ((grouped - mu0).abs() * valid).sum(1, keepdim=True) / count
    alpha0 = alpha0.clamp_min(1e-8)
    mu, alpha, threshold = mu0 + rm.tanh() * alpha0, rs.exp() * alpha0, .5 * rt.exp()
    z = (grouped - mu) / alpha
    soft = alpha * (torch.tanh(sharpness * (z - threshold)) + torch.tanh(sharpness * (z + threshold))) / (2 * math.tanh(sharpness))
    return soft.flatten()[:weight.numel()].reshape_as(weight)


@pytest.mark.parametrize("rank", [0, 3])
def test_hard_backward_retains_actual_last_soft_point_before_all_parameter_updates(rank):
    torch.manual_seed(61)
    layer = TernaryLinear(nn.Linear(17, 5, bias=False), group_size=13, lora_rank=rank)
    with torch.no_grad():
        for parameter in layer.trainable_parameters():
            parameter.uniform_(-.12, .12)
    snapshot = tuple(p.detach().clone().requires_grad_(True) for p in layer.trainable_parameters())
    coeff = torch.randn_like(layer.weight)
    actual_k = 30 * (12 / 16) / .8
    independent = _independent_soft_weight(layer.weight.float(), snapshot, rank, 13, actual_k)
    expected = torch.autograd.grad((independent * coeff).sum(), snapshot)
    layer.set_progress(12 / 16)
    soft = layer.quantized_weight()
    torch.testing.assert_close(soft, independent, rtol=1e-5, atol=1e-7)
    assert layer._hard_snapshot_sharpness == actual_k
    # Update every parameter after the last soft forward, including both LoRA
    # matrices. Hard-stage derivatives must still use that pre-update point.
    for step in range(2):
        with torch.no_grad():
            for index, parameter in enumerate(layer.trainable_parameters()):
                parameter.add_(.04 * (index + 1))
        layer.set_progress((13 + step) / 16)
        hard = layer.quantized_weight()
        actual = torch.autograd.grad((hard * coeff).sum(), layer.trainable_parameters())
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=2e-4, atol=3e-6)
        torch.testing.assert_close(hard, layer.hard_weight(), rtol=0, atol=0)
    assert layer._hard_snapshot_sharpness == actual_k
    original_values = (*snapshot[:3], *(snapshot[3:] if rank else (None, None)))
    for saved, original in zip(layer._snapshot_names(), original_values):
        if original is not None:
            torch.testing.assert_close(getattr(layer, saved), original, rtol=0, atol=0)


def test_snapshot_is_compact_not_replaced_by_eval_and_can_be_released():
    layer = TernaryLinear(nn.Linear(128, 32), group_size=128, lora_rank=2)
    layer.set_progress(.75)
    layer.quantized_weight()
    snapshot = layer._hard_snapshot_mu.clone()
    stored = [getattr(layer, name) for name in layer._snapshot_names()]
    assert sum(x.numel() for x in stored) == 3 * layer.num_groups + 2 * (128 + 32)
    assert all(x.shape != layer.weight.shape for x in stored)
    assert not any("snapshot" in key for key in layer.state_dict())
    with torch.no_grad():
        layer.raw_mu.add_(.2)
        layer.set_progress(.79)
        layer.quantized_weight()
    torch.testing.assert_close(layer._hard_snapshot_mu, snapshot, rtol=0, atol=0)
    assert layer._hard_snapshot_sharpness == 30 * .75 / .8
    layer.set_progress(.81)
    layer.quantized_weight()
    layer.clear_soft_snapshot()
    assert all(getattr(layer, name) is None for name in layer._snapshot_names())
    with pytest.raises(RuntimeError, match="preceding soft forward"):
        layer.quantized_weight()
    with torch.no_grad():
        torch.testing.assert_close(layer.quantized_weight(), layer.hard_weight(), rtol=0, atol=0)
    for parameter in layer.trainable_parameters():
        parameter.requires_grad_(False)
    torch.testing.assert_close(layer.quantized_weight(), layer.hard_weight(), rtol=0, atol=0)


def test_decreasing_progress_discards_stale_window_snapshot():
    layer = TernaryLinear(nn.Linear(8, 4), group_size=8, lora_rank=2)
    layer.set_progress(.6)
    layer.quantized_weight()
    layer.set_progress(.5)
    assert layer._hard_snapshot_sharpness is None
    layer.set_progress(.7)
    layer.quantized_weight()
    layer.reset_annealing()
    assert layer._hard_snapshot_sharpness is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dtype_moves_preserve_fp32_parameters_statistics_gradients_and_snapshots(dtype):
    torch.manual_seed(103)
    layer = TernaryLinear(nn.Linear(16, 4), group_size=8, lora_rank=2)
    with torch.no_grad():
        for parameter in layer.trainable_parameters():
            parameter.uniform_(-.123456, .234567)
    layer.set_progress(.4)
    layer.quantized_weight().square().sum().backward()
    parameters = {name: p.detach().clone() for name, p in layer.named_parameters()}
    grads = {name: p.grad.clone() for name, p in layer.named_parameters()}
    buffers = {name: getattr(layer, name).clone() for name in ("mu0", "alpha0", *layer._snapshot_names())}
    layer.to(dtype=dtype)
    assert layer.weight.dtype == layer.bias.dtype == dtype
    for name, value in parameters.items():
        actual = getattr(layer, name)
        assert actual.dtype == actual.grad.dtype == torch.float32
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
        torch.testing.assert_close(actual.grad, grads[name], rtol=0, atol=0)
    for name, value in buffers.items():
        assert getattr(layer, name).dtype == torch.float32
        torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)
    layer.set_progress(.9)
    assert torch.isfinite(layer(torch.randn(2, 16, dtype=dtype))).all()
