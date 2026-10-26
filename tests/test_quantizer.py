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
    layer = TernaryLinear.from_linear(linear, group_size=4)
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
    layer = TernaryLinear.from_linear(nn.Linear(8, 4), group_size=8)
    layer.set_progress(0.25)
    loss = (layer(torch.randn(3, 8)) - torch.randn(3, 4)).square().mean()
    loss.backward()
    assert layer.weight.grad is None
    assert layer.bias.grad is None
    for p in layer.trainable_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_frozen_jacobian_matches_last_soft_derivative_after_factor_updates():
    torch.manual_seed(19)
    layer = TernaryLinear.from_linear(nn.Linear(8, 4, bias=False), group_size=8)
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
    torch.testing.assert_close(layer._hard_snapshot_mu, layer.raw_mu, rtol=0, atol=0)


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
