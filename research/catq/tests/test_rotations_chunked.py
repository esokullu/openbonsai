"""Independent grouped equations and bounded-memory checks for rotated CAT-Q."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import openbonsai.quantizer as quantizer
import openbonsai.rotations as rotations
from openbonsai.hadamard import rotate, inverse_rotate
from openbonsai.rotations import RotatedTernaryLinear, RotatedTernaryEmbedding, fixed_signs
from test_quantizer_chunked import FloatAllocationSizes


def dense_equations(layer, parameters, sharpness=None):
    """Full-matrix oracle, separate from all chunk/statistic/factor helpers.

    The transform primitive is shared; a separate explicit Sylvester matrix
    oracle below verifies its convention and native 4096-wide block algebra.
    """
    rm, rs, rt = parameters[:3]
    effective = rotate(layer.weight.float(), layer.rotation_signs, layer.block_size)
    if layer.lora_rank:
        a, b = parameters[3:]
        effective = effective + b @ a / layer.lora_rank
    grouped = F.pad(effective.flatten(), (0, layer.padded_numel - layer.numel)).reshape(-1, layer.group_size)
    valid = torch.arange(layer.padded_numel).reshape_as(grouped) < layer.numel
    count = valid.sum(1, keepdim=True)
    mu0 = (grouped * valid).sum(1, keepdim=True) / count
    alpha0 = ((grouped - mu0).abs() * valid).sum(1, keepdim=True) / count
    if layer.factor_mapping == 'exp':
        alpha0 = alpha0.clamp_min(1e-8)
        mu, alpha, threshold = mu0 + rm.tanh() * alpha0, rs.exp() * alpha0, .5 * rt.exp()
    else:
        alpha0 = alpha0 + 1e-6
        mu, alpha, threshold = mu0 + (2 * rm.sigmoid() - 1) * alpha0, 2 * rs.sigmoid() * alpha0, rt.sigmoid()
    z = (grouped - mu) / alpha
    codes = ((z > threshold).float() - (z < -threshold).float()) * valid
    if sharpness is None:
        values = alpha * codes
    else:
        values = alpha * (torch.tanh(sharpness * (z - threshold)) + torch.tanh(sharpness * (z + threshold))) / (2 * math.tanh(sharpness))
    return values.flatten()[:layer.numel].reshape_as(layer.weight), codes, alpha


@pytest.mark.parametrize('mapping', ['exp', 'sigmoid'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('phase', ['soft', 'hard'])
@pytest.mark.parametrize('fixture', [
    (24, 11, 16, 8, 48, 3),        # row/group crossings, final partial group
    (12288, 5, 8192, 4096, 24576, 64),  # three native blocks per row, crossing groups
    (4096, 9, 128, 4096, 8192, 64),  # native hidden width, actual group size
])
def test_rotated_dense_chunked_values_codes_and_all_gradients(monkeypatch, mapping, dtype, phase, fixture):
    width, height, group, block, limit, rank = fixture
    torch.manual_seed(311)
    monkeypatch.setattr(quantizer, '_GROUP_CHUNK_ELEMENTS', limit)
    layer = RotatedTernaryLinear(nn.Linear(width, height, dtype=dtype), group,
                                signs=fixed_signs(width, seed=5090), block_size=block,
                                lora_rank=rank, factor_mapping=mapping, copy_weight=False)
    with torch.no_grad():
        for parameter in layer.trainable_parameters():
            parameter.uniform_(-.12, .12)
    layer.set_progress(.75)
    layer.quantized_weight()
    saved = tuple(p.detach().clone() for p in layer.trainable_parameters())
    if phase == 'hard':
        with torch.no_grad():
            for index, parameter in enumerate(layer.trainable_parameters()):
                parameter.add_(.025 * (index + 1))
        layer.set_progress(.9)
    current = tuple(p.detach().clone().requires_grad_() for p in layer.trainable_parameters())
    point = tuple(old + (now - now.detach()) for old, now in zip(saved, current)) if phase == 'hard' else current
    reference, _, _ = dense_equations(layer, point, 30 * .75 / .8)
    with torch.no_grad():
        hard, codes, _ = dense_equations(layer, current)
        actual_codes = torch.cat([layer._hard_chunk(start, end, (*layer.factor_parameters(), layer.lora_A, layer.lora_B))[0]
                                  for start, end in layer._group_ranges()])
        torch.testing.assert_close(actual_codes, codes, rtol=0, atol=0)
        # The independent validity-mask reductions can differ by one FP32 ULP
        # from the chunk reduction. Codes must still agree exactly above.
        torch.testing.assert_close(layer.hard_weight(), hard.to(dtype), rtol=2e-6, atol=1e-8)
    if phase == 'hard':
        reference = hard + (reference - reference.detach())
    x = torch.randn(2, width, dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    coefficient = torch.randn(2, height, dtype=dtype)
    actual = layer(x)
    expected = F.linear(rotate(xr, layer.rotation_signs, block), reference.to(dtype), layer.bias)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grads = torch.autograd.grad((actual * coefficient).sum(), (*layer.trainable_parameters(), x))
    expected_grads = torch.autograd.grad((expected * coefficient).sum(), (*current, xr))
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=5e-4, atol=5e-6)
        assert torch.isfinite(got).all()
    assert all(g.abs().sum() > 0 for g in actual_grads[:-1])
    assert layer.weight.grad is None
    if phase == 'hard':
        assert layer._hard_snapshot_sharpness == 28.125
        for name, value in zip(layer._snapshot_names(), saved):
            torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)


@pytest.mark.parametrize('mapping', ['exp', 'sigmoid'])
@pytest.mark.parametrize('reentrant', [False, True])
@pytest.mark.parametrize('hard_phase', ['frozen_jacobian', 'soft_surrogate'])
def test_rotated_nested_checkpoint_preserves_actual_soft_snapshot(monkeypatch, mapping, reentrant, hard_phase):
    torch.manual_seed(41)
    monkeypatch.setattr(quantizer, '_GROUP_CHUNK_ELEMENTS', 48)
    plain = RotatedTernaryLinear(nn.Linear(24, 11, dtype=torch.bfloat16), 16,
                                signs=fixed_signs(24, seed=99), block_size=8,
                                lora_rank=3, factor_mapping=mapping, hard_phase=hard_phase)
    with torch.no_grad():
        plain.lora_B.normal_(std=.15)
    nested = copy.deepcopy(plain)
    for progress in (.75, .9, 1.):
        for layer in (plain, nested):
            layer.set_progress(progress)
            layer.zero_grad(set_to_none=True)
        x = torch.randn(2, 24, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual = checkpoint(nested, x, use_reentrant=reentrant)
        expected = plain(xr)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        for p, q in zip(nested.trainable_parameters(), plain.trainable_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
        if hard_phase == 'frozen_jacobian':
            assert nested._hard_snapshot_sharpness == plain._hard_snapshot_sharpness == 28.125
            for name in nested._snapshot_names():
                torch.testing.assert_close(getattr(nested, name), getattr(plain, name), rtol=0, atol=0)
        with torch.no_grad():
            for layer in (plain, nested):
                for parameter in layer.trainable_parameters():
                    parameter.add_(parameter.grad, alpha=-.02)


@pytest.mark.parametrize('mapping', ['exp', 'sigmoid'])
@pytest.mark.parametrize('rank', [0, 3])
def test_rotated_rank_zero_statistics_padding_and_noncontiguous_rows(monkeypatch, mapping, rank):
    torch.manual_seed(12)
    monkeypatch.setattr(quantizer, '_GROUP_CHUNK_ELEMENTS', 48)
    source = nn.Linear(24, 11, dtype=torch.bfloat16)
    source.weight = nn.Parameter(source.weight.T.contiguous().T)
    layer = RotatedTernaryLinear(source, 16, signs=fixed_signs(24, seed=3), block_size=8,
                                factor_mapping=mapping, lora_rank=rank, copy_weight=False)
    assert layer.weight.data_ptr() == source.weight.data_ptr()
    assert not layer.weight.is_contiguous()
    for start, end in layer._group_ranges():
        assert start * 16 % 24 == 0
        assert end == layer.num_groups or end * 16 % 24 == 0
    if rank:
        with torch.no_grad():
            layer.lora_B.normal_(std=.1)
    layer.set_progress(.4)
    expected, _, _ = dense_equations(layer, layer.trainable_parameters(), 15)
    actual = layer.quantized_weight()
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
    coefficient = torch.randn_like(actual)
    ag = torch.autograd.grad((actual * coefficient).sum(), layer.trainable_parameters())
    eg = torch.autograd.grad((expected.to(torch.bfloat16) * coefficient).sum(), layer.trainable_parameters())
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=5e-6)
    old = {name: value.clone() for name, value in layer.named_parameters()}
    stats = {name: getattr(layer, name).clone() for name in ('mu0', 'alpha0', *layer._snapshot_names())
             if getattr(layer, name) is not None}
    layer.bfloat16()
    for name, value in old.items():
        assert getattr(layer, name).dtype == torch.float32
        torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)
    for name, value in stats.items():
        assert getattr(layer, name).dtype == torch.float32
        torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)


def test_rotated_native_width_scratch_is_chunk_bounded_through_backward_and_export(monkeypatch):
    torch.manual_seed(7)
    limit = 24576
    monkeypatch.setattr(quantizer, '_GROUP_CHUNK_ELEMENTS', limit)
    source = nn.Linear(12288, 17, dtype=torch.bfloat16)
    saved = []
    with FloatAllocationSizes() as allocations:
        layer = RotatedTernaryLinear(source, 128, signs=fixed_signs(12288, seed=3), block_size=4096,
                                    lora_rank=2, factor_mapping='sigmoid', copy_weight=False)
        with torch.no_grad():
            layer.lora_B.normal_(std=.1)
        for progress in (.75, .9):
            layer.set_progress(progress)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                weight = layer.quantized_weight()
            weight.square().mean().backward()
            layer.zero_grad(set_to_none=True)
        statistics = layer.audit_weight_statistics()
        state = layer.mixed_candidate(packing='auto')
    assert max(allocations.sizes) <= limit
    assert all(t.numel() <= limit for t in saved if t.dtype == torch.float32)
    assert layer.weight.data_ptr() == source.weight.data_ptr()
    assert sum(statistics['code_counts'].values()) == source.weight.numel()
    assert all(math.isfinite(statistics[key]) for key in ('original_rms', 'effective_rms', 'hard_rms'))
    assert state.metadata['rotation']['block_size'] == 4096
    assert not any(t.numel() == source.weight.numel() and t.dtype == torch.float32 for t in layer.buffers())
    with pytest.raises(ValueError, match='legacy export'):
        layer.packed_state()


@pytest.fixture(scope='module')
def native_sylvester_matrix():
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < 4096:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix / 64


def test_native_block4096_transform_matches_explicit_matrix_and_inverse(native_sylvester_matrix):
    torch.manual_seed(18)
    x = torch.randn(2, 12288, requires_grad=True)
    signs = fixed_signs(12288, seed=5090)
    expected = ((x * signs).reshape(-1, 4096) @ native_sylvester_matrix).reshape_as(x)
    actual = rotate(x, signs, 4096)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(inverse_rotate(actual, signs, 4096), x, rtol=2e-5, atol=2e-6)
    coefficient = torch.randn_like(x)
    ag, = torch.autograd.grad((actual * coefficient).sum(), x)
    eg, = torch.autograd.grad((expected * coefficient).sum(), x)
    torch.testing.assert_close(ag, eg, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('embedding', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_block4096_float_endpoint_path_is_exercised_separately_from_teacher(monkeypatch, embedding, dtype):
    torch.manual_seed(84)
    monkeypatch.setattr(quantizer, '_GROUP_CHUNK_ELEMENTS', 8192)
    source = nn.Embedding(17, 4096, dtype=dtype) if embedding else nn.Linear(4096, 17, dtype=dtype)
    cls = RotatedTernaryEmbedding if embedding else RotatedTernaryLinear
    layer = cls(source, 128, signs=fixed_signs(4096, seed=3), block_size=4096, lora_rank=2, copy_weight=False)
    inputs = torch.tensor([[1, 3, 1, 8]]) if embedding else torch.randn(2, 4096, dtype=dtype)
    with torch.no_grad():
        layer.lora_A.fill_(20)
        layer.lora_B.fill_(30)
        expected = source(inputs)
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        layer.set_progress(1)
        layer.quant_enabled = False
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
    layer.quant_enabled = True
    layer.rotation_only = True
    calls = []
    original = rotations.rotate
    def observe(x, signs, block):
        calls.append((tuple(x.shape), x.dtype, block))
        return original(x, signs, block)
    monkeypatch.setattr(rotations, 'rotate', observe)
    with torch.no_grad():
        actual = layer(inputs)
    assert layer.progress > 0 and layer.rotation_only and layer.quant_enabled
    assert calls and all(block == 4096 and work_dtype == torch.float32 for _, work_dtype, block in calls)
    if embedding:
        # Only requested rows are transformed, even for repeated IDs.
        assert all(shape[-2] != source.num_embeddings for shape, _, _ in calls)
        reference = expected
    else:
        assert all(shape[0] <= 2 for shape, _, _ in calls)
        reference = F.linear(inputs.float(), source.weight.float(), source.bias.float()).to(dtype)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)


def test_rotated_replacement_transfers_source_storage_and_keeps_endpoint_progress_explicit():
    from openbonsai.adapter import tiny_model, iter_ternary_modules
    model = tiny_model(seed=13)
    pointers = {name: module.weight.data_ptr() for name, module in model.named_modules()
                if isinstance(module, (nn.Linear, nn.Embedding))}
    names = rotations.replace_rotated_projections(model, endpoints=True, block_size=64, lora_rank=0)
    for name in names:
        module = model.get_submodule(name)
        assert module.weight.data_ptr() == pointers[name]
    # ALL wrappers, including embedding/head, must enter the float transform
    # path explicitly; decoder-only set_progress is insufficient for this test.
    modules = list(iter_ternary_modules(model))
    for _, module in modules:
        module.set_progress(1)
        module.rotation_only = True
    assert all(m.progress > 0 and m.rotation_only for _, m in modules)
