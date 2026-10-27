"""Bounded frozen-row scoring; no pretrained quality or acceleration claims."""
import math

import pytest
import torch
from torch.nn import functional as F

from openbonsai import mixed_formats as mf
from openbonsai.hadamard import rotate
from openbonsai.packing import pack_grouped_weight
from openbonsai.rotations import fixed_signs


ENCODINGS = ('bf16', 'affine4', 'affine6', '2bit', 'bitcos', 'base3')


def candidate(encoding, *, rotated=False, bias=True, zero=False):
    generator = torch.Generator().manual_seed(619)
    rows, width, group = 13, 16, 6
    b = torch.randn(rows, generator=generator) if bias else None
    if encoding == 'bf16':
        result = mf.bf16_candidate(torch.randn(rows, width, generator=generator).bfloat16(),
                                   None if b is None else b.bfloat16())
    elif encoding.startswith('affine'):
        bits = int(encoding[-1])
        codes = torch.randint(0, 1 << bits, (rows, width), generator=generator)
        scales = torch.rand(rows, math.ceil(width / group), generator=generator) + .01317
        zeros = torch.randint(-2, (1 << bits) + 2, scales.shape, generator=generator)
        result = mf.affine_candidate(codes, scales, zeros, bits=bits, group_size=group, bias=b)
    else:
        codes = torch.randint(-1, 2, (rows, width), generator=generator, dtype=torch.int8)
        codes.reshape(-1)[::3] = 0
        if zero:
            codes.zero_()
        scales = torch.rand(math.ceil(rows * width / group), generator=generator) + .01317
        state = pack_grouped_weight(codes, scales, shape=codes.shape, group_size=group, bias=b)
        result = mf.ternary_candidate(state, packing=encoding)
    return (mf.with_rotation(result, fixed_signs(width, seed=69), 8) if rotated else result)


def transformed(x, value):
    rotation = value.metadata.get('rotation')
    return x if rotation is None else rotate(x, value.tensors['rotation_signs'], rotation['block_size'])


def forbidden(*_args, **_kwargs):
    raise AssertionError('bounded row scoring must not construct a complete weight or output')


@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('rotated,bias', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('warm', [False, True])
def test_one_row_gemm_exact_values_gradients_and_bounded_decode(monkeypatch, encoding, rotated, bias, warm):
    value = candidate(encoding, rotated=rotated, bias=bias)
    runtime = mf.MixedLinear(value, row_chunk_size=4)
    generator = torch.Generator().manual_seed(92)
    x = torch.randn(2, 3, 16, generator=generator).bfloat16().requires_grad_()
    start, end = 3, 7  # Unaligned rows exercise cross-row groups and packed offsets.
    weight = mf.candidate_weight(value, dtype=x.dtype)[start:end]
    b = value.tensors.get('bias')
    expected = F.linear(transformed(x, value), weight, None if b is None else b[start:end].to(x.dtype))
    if warm:
        runtime.prepare_decoded_weight_cache()
    payload_before = mf.encode_candidate(runtime._candidate())
    cache_bytes_before = runtime.decoded_weight_cache_bytes
    decode_calls, gemms = [], []
    rows, linear = mf._rows, mf.F.linear

    def bounded_rows(value, ids, dtype, device):
        decode_calls.append(ids.tolist())
        assert ids.tolist() == list(range(start, end))
        return rows(value, ids, dtype, device)

    def row_gemm(work, selected, selected_bias):
        gemms.append((tuple(work.shape), tuple(selected.shape), selected.dtype,
                      None if selected_bias is None else tuple(selected_bias.shape)))
        return linear(work, selected, selected_bias)

    monkeypatch.setattr(mf, 'candidate_weight', forbidden)
    monkeypatch.setattr(runtime, 'forward', forbidden)
    monkeypatch.setattr(mf, '_rows', bounded_rows)
    monkeypatch.setattr(mf.F, 'linear', row_gemm)
    actual = runtime.forward_rows(x, start, end)
    assert torch.equal(actual, expected)
    assert actual.shape == (2, 3, end - start)
    assert decode_calls == ([] if warm else [list(range(start, end))])
    assert gemms == [((2, 3, 16), (4, 16), torch.bfloat16, (4,) if bias else None)]
    upstream = torch.randn(actual.shape, generator=generator).to(actual.dtype)
    actual_gradient, = torch.autograd.grad(actual, x, upstream)
    expected_gradient, = torch.autograd.grad(expected, x, upstream)
    assert torch.equal(actual_gradient, expected_gradient)
    assert runtime.decoded_weight_cache_bytes == cache_bytes_before
    assert mf.encode_candidate(runtime._candidate()) == payload_before
    assert not list(runtime.parameters())


@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('rotated', [False, True])
def test_existing_forward_keeps_single_rotation_fixed_gemms_and_gradients(monkeypatch, encoding, rotated):
    value = candidate(encoding, rotated=rotated)
    runtime = mf.MixedLinear(value, row_chunk_size=4)
    generator = torch.Generator().manual_seed(316)
    x = torch.randn(2, 3, 16, generator=generator).requires_grad_()
    latent = mf.candidate_weight(value, dtype=x.dtype)
    work = transformed(x, value)
    bias = value.tensors['bias'].to(x.dtype)
    ranges = [(0, 4), (4, 8), (8, 12), (12, 13)]
    # Independent reconstruction of the preexisting fixed-row forward graph.
    expected = torch.cat([F.linear(work, latent[start:end], bias[start:end])
                          for start, end in ranges], -1)
    rotations, calls = [], []
    original_rotate, original_linear = mf.rotate, mf.F.linear

    def observe_rotate(*args, **kwargs):
        rotations.append(True)
        return original_rotate(*args, **kwargs)

    def observe_linear(a, b, c):
        calls.append(tuple(b.shape))
        return original_linear(a, b, c)

    monkeypatch.setattr(mf, 'rotate', observe_rotate)
    monkeypatch.setattr(mf.F, 'linear', observe_linear)
    actual = runtime(x)
    assert torch.equal(actual, expected)
    assert len(rotations) == int(rotated)
    assert calls == [(4, 16), (4, 16), (4, 16), (1, 16)]
    upstream = torch.randn(actual.shape, generator=generator)
    actual_gradient, = torch.autograd.grad(actual, x, upstream)
    expected_gradient, = torch.autograd.grad(expected, x, upstream)
    assert torch.equal(actual_gradient, expected_gradient)
    for start, end in ranges:
        assert torch.equal(runtime.forward_rows(x.detach(), start, end), actual.detach()[..., start:end])


@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('rotated', [False, True])
def test_row_gemm_survives_file_reload_exactly(tmp_path, encoding, rotated):
    value = candidate(encoding, rotated=rotated)
    path = tmp_path / 'candidate.obc'
    mf.save_candidate(value, path)
    serialized = path.read_bytes()
    restored = mf.load_candidate(path)
    original = mf.MixedLinear(value, row_chunk_size=4)
    loaded = mf.MixedLinear(restored, row_chunk_size=4, copy_payload=False)
    x = torch.arange(3 * 16).reshape(3, 16).bfloat16() / 19
    for start, end in [(0, 4), (3, 7), (12, 13)]:
        assert torch.equal(loaded.forward_rows(x, start, end), original.forward_rows(x, start, end))
    loaded.prepare_decoded_weight_cache()
    assert '_decoded_weight_cache' not in loaded.state_dict()
    assert torch.equal(loaded.forward_rows(x, 3, 7), original.forward_rows(x, 3, 7))
    assert path.read_bytes() == serialized
    assert mf.encode_candidate(loaded._candidate()) == serialized


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_row_cache_dtype_fallback_reconstructs_original_scale_precision(monkeypatch, dtype):
    value = candidate('affine6', rotated=True)
    runtime = mf.MixedLinear(value, row_chunk_size=4)
    x = (torch.arange(32).reshape(2, 16) / 13).to(dtype)
    weight = mf.candidate_weight(value, dtype=dtype)[3:7]
    expected = F.linear(transformed(x, value), weight, value.tensors['bias'][3:7].to(dtype))
    runtime.prepare_decoded_weight_cache()
    pointer, cache_bytes = runtime._decoded_weight_cache.data_ptr(), runtime.decoded_weight_cache_bytes
    calls, original = [], mf._rows

    def observed(value, ids, observed_dtype, device):
        calls.append((ids.tolist(), observed_dtype, device))
        return original(value, ids, observed_dtype, device)

    monkeypatch.setattr(mf, '_rows', observed)
    monkeypatch.setattr(mf, 'candidate_weight', forbidden)
    assert torch.equal(runtime.forward_rows(x, 3, 7), expected)
    assert calls == [([3, 4, 5, 6], dtype, x.device)]
    assert runtime._decoded_weight_cache.data_ptr() == pointer
    assert runtime.decoded_weight_cache_bytes == cache_bytes


@pytest.mark.parametrize('mutation', ['scale', 'bias', 'rotation_sign', 'code', 'new_storage', 'state_dict'])
def test_rows_invalidate_mutated_cache_and_preserve_exact_updated_behavior(mutation):
    runtime = mf.MixedLinear(candidate('affine6', rotated=True), row_chunk_size=4)
    runtime.prepare_decoded_weight_cache()
    with torch.no_grad():
        if mutation == 'scale':
            runtime.payload_scales.mul_(1.25)
        elif mutation == 'bias':
            runtime.payload_bias.add_(.125)
        elif mutation == 'rotation_sign':
            runtime.payload_rotation_signs.neg_()
        elif mutation == 'code':
            runtime.payload_codes[0].bitwise_xor_(1)
        elif mutation == 'new_storage':
            runtime.payload_scales = runtime.payload_scales + .125
        else:
            runtime.load_state_dict(runtime.state_dict())
    cold = mf.MixedLinear(runtime._candidate(), row_chunk_size=4)
    x = (torch.arange(16).reshape(1, 16) / 9).bfloat16()
    assert torch.equal(runtime.forward_rows(x, 0, 4), cold.forward_rows(x, 0, 4))
    assert runtime.decoded_weight_cache_bytes == 0


@pytest.mark.parametrize('start,end', [(-1, 3), (1, 1), (2, 1), (12, 14), (0, 5),
    (True, 3), (0, False), (1., 3), (0, 3.), (None, 3), (torch.tensor(1), 3)])
def test_invalid_row_range_is_rejected_before_any_runtime_work(monkeypatch, start, end):
    runtime = mf.MixedLinear(candidate('base3'), row_chunk_size=4)
    monkeypatch.setattr(runtime, '_prepare_forward', forbidden)
    with pytest.raises(ValueError, match='row range'):
        runtime.forward_rows(torch.ones(1, 16), start, end)


@pytest.mark.parametrize('x', [torch.ones(2, 15), torch.ones(2, 16, dtype=torch.int64)])
def test_rows_keep_existing_input_width_and_dtype_validation(x):
    runtime = mf.MixedLinear(candidate('2bit'), row_chunk_size=4)
    with pytest.raises(ValueError, match='input has incompatible'):
        runtime.forward_rows(x, 0, 4)


@pytest.mark.parametrize('shape', [(16,), (0, 16), (2, 0, 16)])
@pytest.mark.parametrize('rotated', [False, True])
def test_zero_bitcos_rows_vector_and_empty_batches(shape, rotated):
    value = candidate('bitcos', rotated=rotated, bias=False, zero=True)
    runtime = mf.MixedLinear(value, row_chunk_size=4)
    x = torch.ones(shape, dtype=torch.bfloat16, requires_grad=True)
    actual = runtime.forward_rows(x, 12, 13)
    assert actual.shape == (*shape[:-1], 1)
    assert torch.count_nonzero(actual).item() == 0
    actual.sum().backward()
    assert x.grad.shape == x.shape and torch.count_nonzero(x.grad).item() == 0
