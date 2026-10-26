"""Full-matrix equation oracles and allocation checks for bounded LoRA quantization."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import checkpoint

import openbonsai.quantizer as quantizer
from openbonsai.packing import pack_grouped_weight, unpack_grouped_weight
from openbonsai.quantizer import TernaryLinear, PackedTernaryLinear


def dense_equations(layer, parameters, sharpness=None):
    """Independent, intentionally unbounded equations including valid padding."""
    rm, rs, rt = parameters[:3]
    effective = layer.weight.float()
    if layer.lora_rank:
        a, b = parameters[3:]
        effective = effective + b @ a / layer.lora_rank
    grouped = F.pad(effective.flatten(), (0, layer.padded_numel - layer.numel)).reshape(-1, layer.group_size)
    valid = torch.arange(layer.padded_numel, device=grouped.device).reshape_as(grouped) < layer.numel
    count = valid.sum(1, keepdim=True)
    mu0 = (grouped * valid).sum(1, keepdim=True) / count
    alpha0 = ((grouped - mu0).abs() * valid).sum(1, keepdim=True) / count
    if layer.factor_mapping == "exp":
        alpha0 = alpha0.clamp_min(1e-8)
        mu, alpha, threshold = mu0 + rm.tanh() * alpha0, rs.exp() * alpha0, .5 * rt.exp()
    else:
        alpha0 = alpha0 + 1e-6
        mu, alpha, threshold = mu0 + (2 * rm.sigmoid() - 1) * alpha0, 2 * rs.sigmoid() * alpha0, rt.sigmoid()
    z = (grouped - mu) / alpha
    if sharpness is None:
        codes = (z > threshold).float() - (z < -threshold).float()
        values = alpha * codes
    else:
        values = alpha * (torch.tanh(sharpness * (z - threshold)) + torch.tanh(sharpness * (z + threshold))) / (2 * math.tanh(sharpness))
    return values.flatten()[:layer.numel].reshape_as(layer.weight)


@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("phase", ["soft", "hard"])
@pytest.mark.parametrize("hard_phase", ["frozen_jacobian", "soft_surrogate"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_outputs_and_all_parameter_gradients_match_dense_oracle(monkeypatch, mapping, rank, phase, hard_phase, dtype):
    torch.manual_seed(712)
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", 39)
    layer = TernaryLinear(nn.Linear(17, 11, dtype=dtype), group_size=13, lora_rank=rank,
                          factor_mapping=mapping, hard_phase=hard_phase)
    with torch.no_grad():
        for parameter in layer.trainable_parameters():
            parameter.uniform_(-.12, .12)
    # A real last-soft forward captures all five pre-update variables and k.
    layer.set_progress(.75)
    layer.quantized_weight()
    saved = tuple(p.detach().clone() for p in layer.trainable_parameters())
    if phase == "hard":
        with torch.no_grad():
            for index, parameter in enumerate(layer.trainable_parameters()):
                parameter.add_(.025 * (index + 1))
        layer.set_progress(.9)
    current = tuple(p.detach().clone().requires_grad_() for p in layer.trainable_parameters())
    point = current
    sharpness = 30 * .75 / .8
    if phase == "hard":
        if hard_phase == "frozen_jacobian":
            point = tuple(old + (now - now.detach()) for old, now in zip(saved, current))
        else:
            sharpness = 30
    soft = dense_equations(layer, point, sharpness)
    if phase == "hard":
        with torch.no_grad():
            hard = dense_equations(layer, current)
        reference_weight = hard + (soft - soft.detach())
    else:
        reference_weight = soft
    x = torch.randn(3, 17, dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    coefficient = torch.randn(3, 11, dtype=dtype)
    actual = layer(x)
    expected = F.linear(xr, reference_weight.to(dtype), layer.bias)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grads = torch.autograd.grad((actual * coefficient).sum(), (*layer.trainable_parameters(), x))
    reference_grads = torch.autograd.grad((expected * coefficient).sum(), (*current, xr))
    for got, want in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, want, rtol=4e-4, atol=4e-6)
        assert torch.isfinite(got).all()
    assert all(g.abs().sum() > 0 for g in actual_grads[:-1])
    assert layer.weight.grad is None


@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
@pytest.mark.parametrize("hard_phase", ["frozen_jacobian", "soft_surrogate"])
@pytest.mark.parametrize("reentrant", [False, True])
def test_nested_checkpoint_retains_soft_snapshot_and_matches_uncheckpointed_training(monkeypatch, mapping, hard_phase, reentrant):
    torch.manual_seed(44)
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", 26)
    ordinary = TernaryLinear(nn.Linear(17, 7, dtype=torch.bfloat16), group_size=13,
                             lora_rank=3, factor_mapping=mapping, hard_phase=hard_phase)
    with torch.no_grad():
        ordinary.lora_B.normal_(std=.15)
    nested = copy.deepcopy(ordinary)
    for progress in (.75, .9, 1.):
        for layer in (ordinary, nested):
            layer.set_progress(progress)
            layer.zero_grad(set_to_none=True)
        x = torch.randn(2, 17, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual = checkpoint(nested, x, use_reentrant=reentrant)
        expected = ordinary(xr)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        for p, q in zip(nested.trainable_parameters(), ordinary.trainable_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
        if hard_phase == "frozen_jacobian":
            assert nested._hard_snapshot_sharpness == ordinary._hard_snapshot_sharpness == 30 * .75 / .8
            for name in nested._snapshot_names():
                torch.testing.assert_close(getattr(nested, name), getattr(ordinary, name), rtol=0, atol=0)
        with torch.no_grad():
            for layer in (ordinary, nested):
                for parameter in layer.trainable_parameters():
                    parameter.add_(parameter.grad, alpha=-.02)


class FloatAllocationSizes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.sizes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.sizes.extend(t.numel() for t in tree_flatten(result)[0]
                          if isinstance(t, torch.Tensor) and t.dtype == torch.float32)
        return result


@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
def test_training_initialization_export_and_audit_keep_fp32_arrays_chunk_bounded(monkeypatch, mapping):
    torch.manual_seed(19)
    limit = 4096
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", limit)
    source = nn.Linear(256, 512, dtype=torch.bfloat16)
    saved = []
    with FloatAllocationSizes() as allocations:
        layer = TernaryLinear(source, group_size=128, lora_rank=4, factor_mapping=mapping, copy_weight=False)
        with torch.no_grad():
            layer.lora_B.normal_(std=.1)
        for progress in (.75, .9):
            layer.set_progress(progress)
            with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
                weight = layer.quantized_weight()
            weight.square().mean().backward()
            layer.zero_grad(set_to_none=True)
        stats = layer.audit_weight_statistics()
        packed = layer.packed_state(packing="bitcos")
        with torch.no_grad():
            for codec in ("2bit", "bitcos"):
                restored = PackedTernaryLinear(layer.packed_state(packing=codec), dtype=torch.bfloat16)
                restored(torch.randn(1, 256, dtype=torch.bfloat16))
    assert max(allocations.sizes) <= limit + 2 * source.in_features
    # Checkpoint keeps compact inputs, not a dense effective/tanh/statistics graph.
    assert all(t.numel() <= limit for t in saved if t.dtype == torch.float32)
    assert layer.weight.data_ptr() == source.weight.data_ptr()
    assert sum(stats["code_counts"].values()) == source.weight.numel()
    assert all(math.isfinite(stats[key]) for key in ("original_rms", "effective_rms", "hard_rms"))
    assert packed["numel"] == source.weight.numel()


@pytest.mark.parametrize("codec", ["2bit", "bitcos", "auto"])
@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
def test_chunked_export_and_packed_decode_with_cross_byte_groups(monkeypatch, codec, mapping):
    torch.manual_seed(721)
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", 21)
    layer = TernaryLinear(nn.Linear(17, 11, dtype=torch.bfloat16), group_size=7, lora_rank=3,
                          factor_mapping=mapping)
    with torch.no_grad():
        for parameter in layer.trainable_parameters():
            parameter.normal_(std=.15)
        codes, scales = layer._hard_codes_and_scales()
        reference = pack_grouped_weight(codes, scales, shape=[11, 17], group_size=7,
                                        bias=layer.bias, packing=codec)
        bounded = layer.packed_state(packing=codec)
    for key in reference:
        if torch.is_tensor(reference[key]):
            torch.testing.assert_close(bounded[key], reference[key], rtol=0, atol=0)
        else:
            assert bounded[key] == reference[key]
    packed = PackedTernaryLinear(bounded, dtype=torch.bfloat16)
    expected = unpack_grouped_weight(reference, dtype=torch.bfloat16)
    torch.testing.assert_close(packed.weight, expected, rtol=0, atol=0)
    stats = layer.audit_weight_statistics()
    with torch.no_grad():
        effective = layer.weight.float() + layer.lora_B @ layer.lora_A / layer.lora_rank
        hard = dense_equations(layer, layer.trainable_parameters())
    assert stats["effective_rms"] == pytest.approx(float(effective.square().double().mean().sqrt()), rel=1e-6)
    assert stats["hard_rms"] == pytest.approx(float(hard.square().double().mean().sqrt()), rel=1e-6)


def test_teacher_endpoints_ignore_large_correction_and_default_copy_is_independent(monkeypatch):
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", 13)
    source = nn.Linear(17, 7)
    layer = TernaryLinear(source, group_size=13, lora_rank=3)
    assert layer.weight.data_ptr() != source.weight.data_ptr()
    with torch.no_grad():
        layer.lora_A.fill_(20)
        layer.lora_B.fill_(30)
    x = torch.randn(2, 17)
    layer.set_progress(0)
    torch.testing.assert_close(layer(x), source(x), rtol=0, atol=0)
    layer(x).sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in layer.trainable_parameters())
    layer.set_progress(1)
    layer.quant_enabled = False
    torch.testing.assert_close(layer(x), source(x), rtol=0, atol=0)


def test_noncontiguous_source_rows_preserve_bounded_reconstruction(monkeypatch):
    torch.manual_seed(12)
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", 39)
    source = nn.Linear(17, 11, dtype=torch.bfloat16)
    source.weight = nn.Parameter(source.weight.T.contiguous().T)
    assert not source.weight.is_contiguous()
    layer = TernaryLinear(source, group_size=13, lora_rank=3, copy_weight=False)
    with torch.no_grad():
        layer.lora_B.normal_(std=.2)
    layer.set_progress(.4)
    expected = dense_equations(layer, layer.trainable_parameters(), 15).to(torch.bfloat16)
    actual = layer.quantized_weight()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coefficient = torch.randn_like(actual)
    a = torch.autograd.grad((actual * coefficient).sum(), layer.trainable_parameters(), retain_graph=True)
    b = torch.autograd.grad((expected * coefficient).sum(), layer.trainable_parameters())
    for got, want in zip(a, b):
        torch.testing.assert_close(got, want, rtol=3e-4, atol=3e-6)
    assert layer.weight.data_ptr() == source.weight.data_ptr()


@pytest.mark.parametrize("codec", ["2bit", "bitcos"])
def test_packed_input_dtype_override_retains_configured_rounding(monkeypatch, codec):
    monkeypatch.setattr(quantizer, "_GROUP_CHUNK_ELEMENTS", 7)
    codes = torch.tensor([1, -1, 0, 1, 1, 0, -1, 1, -1, 0, 1, -1])
    scales = torch.tensor([.12345, .23456, .34567, .45678])
    state = pack_grouped_weight(codes, scales, shape=[3, 4], group_size=3, packing=codec)
    layer = PackedTernaryLinear(state, dtype=torch.bfloat16)
    x = torch.randn(2, 4, dtype=torch.float32)
    expected = F.linear(x, layer.weight.float())
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
