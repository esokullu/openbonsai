"""Storage/import/runtime contracts; these fixtures make no model-quality claim."""
import hashlib
import json
import struct

import pytest
import torch
from torch.nn import functional as F

from openbonsai import mixed_formats as mf
from openbonsai.packing import pack_grouped_weight, unpack_grouped_weight
from openbonsai.hadamard import rotate, inverse_rotate


def affine(bits=4, *, kind="linear", rows=7, width=13, group_size=5, bias=True):
    generator = torch.Generator().manual_seed(123)
    q = torch.randint(0, 1 << bits, (rows, width), generator=generator)
    scale = torch.rand(rows, (width + group_size - 1) // group_size, generator=generator) + .01
    zeros = torch.randint(0, 1 << bits, scale.shape, generator=generator)
    b = torch.randn(rows, generator=generator) if bias and kind == "linear" else None
    c = mf.affine_candidate(q, scale, zeros, bits=bits, group_size=group_size, bias=b,
                            module_kind=kind, provenance={"source_revision": "a" * 40})
    return c, q, scale, zeros, b


def ternary(*, kind="linear", packing="auto", rows=9, width=15, group_size=8, zero=False):
    generator = torch.Generator().manual_seed(45)
    codes = torch.randint(-1, 2, (rows, width), generator=generator)
    if zero:
        codes.zero_()
    scales = torch.rand((rows * width + group_size - 1) // group_size, generator=generator) + .01
    bias = torch.randn(rows, generator=generator) if kind == "linear" else None
    state = pack_grouped_weight(codes, scales, shape=(rows, width), group_size=group_size, bias=bias)
    return mf.ternary_candidate(state, packing=packing, module_kind=kind), state


def assert_equal_candidates(a, b):
    assert a.metadata == b.metadata
    assert a.tensors.keys() == b.tensors.keys()
    for name in a.tensors:
        assert a.tensors[name].dtype == b.tensors[name].dtype
        assert torch.equal(a.tensors[name], b.tensors[name])


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("width", [1, 2, 3, 5, 13, 128])
def test_affine_codes_match_independent_bit_and_weight_oracle(bits, width):
    c, q, scales, zeros, _ = affine(bits, width=width)
    expected_bytes = bytearray((q.numel() * bits + 7) // 8)
    for i, code in enumerate(q.flatten().tolist()):
        for bit in range(bits):
            if code & (1 << bit):
                location = i * bits + bit
                expected_bytes[location // 8] |= 1 << (location % 8)
    assert bytes(c.tensors["codes"].tolist()) == expected_bytes
    oracle = torch.empty_like(q, dtype=torch.float32)
    for row in range(q.shape[0]):
        for col in range(q.shape[1]):
            group = col // 5
            oracle[row, col] = (int(q[row, col]) - int(zeros[row, group])) * scales[row, group]
    assert torch.equal(mf.candidate_weight(c), oracle)
    assert torch.equal(c.tensors["scales"], scales)
    assert c.tensors["scales"].dtype == torch.float32


@pytest.mark.parametrize("packing", ["2bit", "bitcos", "base3", "auto"])
@pytest.mark.parametrize("zero", [False, True])
def test_ternary_preserves_existing_learned_values_and_cross_row_groups(packing, zero):
    c, state = ternary(packing=packing, zero=zero)
    assert torch.equal(mf.candidate_weight(c), unpack_grouped_weight(state))
    assert torch.equal(c.tensors["scales"], state["scales"])
    assert torch.equal(c.tensors["bias"], state["bias"])


@pytest.mark.parametrize("factory", [
    lambda: affine(4)[0], lambda: affine(6)[0],
    lambda: ternary(packing="2bit")[0], lambda: ternary(packing="bitcos")[0],
    lambda: ternary(packing="base3")[0],
    lambda: ternary(kind="embedding", packing="bitcos", zero=True)[0],
    lambda: mf.bf16_candidate(torch.randn(3, 5).bfloat16(), torch.randn(3).bfloat16()),
])
def test_deterministic_serialization_exact_file_size_and_roundtrip(factory, tmp_path):
    c = factory()
    first = mf.encode_candidate(c)
    assert first == mf.encode_candidate(c)
    assert len(first) == mf.candidate_nbytes(c)
    parts = mf.candidate_size_breakdown(c)
    assert sum(parts[k] for k in ("prefix_bytes", "header_bytes", "payload_bytes", "checksum_bytes")) == len(first)
    assert sum(parts["tensor_bytes"].values()) == parts["payload_bytes"]
    decoded = mf.decode_candidate(first)
    assert_equal_candidates(decoded, c)
    assert torch.equal(decoded.dequantize(), c.dequantize())
    path = tmp_path / "candidate.obm"
    assert mf.save_candidate(c, path) == len(first)
    assert path.stat().st_size == len(first)
    assert path.read_bytes() == first
    assert_equal_candidates(mf.load_candidate(path), c)
    with pytest.raises(FileExistsError):
        mf.save_candidate(c, path)
    assert path.read_bytes() == first


def test_auto_uses_whole_serialized_cost_including_row_index():
    # Small rows have a proportionately expensive BITCOS row index.
    _, state = ternary(rows=64, width=2, group_size=8, zero=True)
    auto = mf.ternary_candidate(state, packing="auto")
    candidates = [mf.ternary_candidate(state, packing=p) for p in ("2bit", "bitcos", "base3")]
    assert mf.candidate_nbytes(auto) == min(map(mf.candidate_nbytes, candidates))
    assert auto.encoding == "ternary-base3"
    _, wide = ternary(rows=3, width=4096, group_size=128, zero=True)
    assert mf.ternary_candidate(wide, packing="auto").encoding == "ternary-bitcos"


def test_provenance_is_preserved_and_costed():
    c, *_ = affine()
    before = mf.candidate_nbytes(c)
    c.metadata["provenance"]["experiment"] = "additional evidence"
    assert mf.candidate_nbytes(c) > before
    assert mf.decode_candidate(mf.encode_candidate(c)).metadata == c.metadata


@pytest.mark.parametrize("zero,expected_dtype", [(-15, torch.int8), (31, torch.uint8),
    (-400, torch.int16), (70000, torch.int32), (2**40, torch.int64)])
def test_affine_preserves_out_of_code_range_signed_zero_points(zero, expected_dtype):
    q = torch.tensor([[0, 7, 15]], dtype=torch.int64)
    scales = torch.tensor([[1 / 15]], dtype=torch.float32)
    zeros = torch.tensor([[zero]], dtype=torch.int64)
    c = mf.affine_candidate(q, scales, zeros, bits=4, group_size=128)
    assert c.tensors["zeros"].dtype == expected_dtype
    assert int(c.tensors["zeros"][0, 0]) == zero
    expected = (q.float() - zeros.float()) * scales
    assert torch.equal(c.dequantize(), expected)
    assert torch.equal(mf.decode_candidate(mf.encode_candidate(c)).dequantize(), expected)


@pytest.mark.parametrize("zero", [-15, -400, 70001])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_affine_reconstructs_before_rounding_metadata_to_compute_dtype(zero, dtype):
    q = torch.tensor([[0, 7, 15]], dtype=torch.int64)
    scales = torch.tensor([[0.0030171]], dtype=torch.float32)
    zeros = torch.tensor([[zero]], dtype=torch.int64)
    candidate = mf.affine_candidate(q, scales, zeros, bits=4, group_size=128)
    arithmetic_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    expected = ((q.to(arithmetic_dtype) - zeros.to(arithmetic_dtype)) * scales.to(arithmetic_dtype)).to(dtype)
    assert torch.equal(mf.candidate_weight(candidate, dtype=dtype), expected)
    restored = mf.decode_candidate(mf.encode_candidate(candidate))
    assert torch.equal(mf.candidate_weight(restored, dtype=dtype), expected)
    if dtype == torch.bfloat16:
        prematurely_rounded = (q.to(dtype) - zeros.to(dtype)) * scales.to(dtype)
        assert not torch.equal(prematurely_rounded, expected)
        assert torch.equal(mf.MixedLinear(candidate).weight, expected)


@pytest.mark.parametrize("factory", [
    lambda: affine(4)[0], lambda: affine(6)[0],
    lambda: ternary(packing="2bit")[0], lambda: ternary(packing="bitcos")[0],
    lambda: ternary(packing="base3")[0],
    lambda: mf.bf16_candidate(torch.randn(9, 15).bfloat16(), torch.randn(9).bfloat16()),
])
def test_linear_forward_input_gradients_and_exact_reload(factory):
    c = factory()
    rows, width = c.shape
    generator = torch.Generator().manual_seed(81)
    x = torch.randn(2, 3, width, generator=generator).requires_grad_()
    runtime = mf.MixedLinear(c, dtype=torch.float32, row_chunk_size=2)
    bias = c.tensors.get("bias")
    oracle = F.linear(x, c.dequantize(), None if bias is None else bias.float())
    actual = runtime(x)
    # GEMM shape changes alter FP32 summation order. Check both against an
    # independent FP64 result with a dot-product roundoff bound, rather than
    # a brittle absolute tolerance near cancellation to zero.
    dense64 = c.dequantize().double()
    exact = F.linear(x.double(), dense64, None if bias is None else bias.double())
    magnitude = F.linear(x.double().abs(), dense64.abs(), None if bias is None else bias.double().abs())
    bound = 4 * torch.finfo(torch.float32).eps * width * magnitude
    assert bool(((actual.double() - exact).abs() <= bound).all())
    assert bool(((oracle.double() - exact).abs() <= bound).all())
    upstream = torch.randn(actual.shape, generator=generator)
    grad_a, = torch.autograd.grad(actual, x, upstream, retain_graph=True)
    grad_b, = torch.autograd.grad(oracle, x, upstream)
    exact_grad = upstream.double() @ dense64
    gradient_bound = 4 * torch.finfo(torch.float32).eps * rows * (upstream.double().abs() @ dense64.abs())
    assert bool(((grad_a.double() - exact_grad).abs() <= gradient_bound).all())
    assert bool(((grad_b.double() - exact_grad).abs() <= gradient_bound).all())
    restored = mf.MixedLinear(mf.decode_candidate(mf.encode_candidate(c)), dtype=torch.float32, row_chunk_size=2)
    assert torch.equal(restored(x.detach()), actual.detach())
    assert torch.equal(runtime.weight, c.dequantize())
    assert not list(runtime.parameters())


@pytest.mark.parametrize("factory", [
    lambda: affine(4, kind="embedding")[0], lambda: affine(6, kind="embedding")[0],
    lambda: ternary(kind="embedding", packing="2bit")[0],
    lambda: ternary(kind="embedding", packing="bitcos")[0],
    lambda: ternary(kind="embedding", packing="bitcos", zero=True)[0],
    lambda: mf.bf16_candidate(torch.randn(9, 15).bfloat16(), module_kind="embedding"),
])
def test_embedding_decodes_only_unique_requested_rows_and_reload(factory, monkeypatch):
    c = factory()
    ids = torch.tensor([[c.shape[0] - 1, 1, 1], [0, 1, 0]])
    oracle = F.embedding(ids, c.dequantize())
    runtime = mf.MixedEmbedding(c, dtype=torch.float32, row_chunk_size=2, padding_idx=0)
    observed = []
    original_rows = mf._rows
    def capture_rows(candidate, rows, dtype, device):
        observed.extend(rows.tolist())
        assert rows.numel() <= 2
        return original_rows(candidate, rows, dtype, device)
    monkeypatch.setattr(mf, "_rows", capture_rows)
    assert torch.equal(runtime(ids), oracle)
    assert observed == [0, 1, c.shape[0] - 1]
    restored = mf.MixedEmbedding(mf.decode_candidate(mf.encode_candidate(c)), dtype=torch.float32, row_chunk_size=2)
    assert torch.equal(restored(ids), oracle)
    assert runtime(torch.empty((2, 0), dtype=torch.int64)).shape == (2, 0, c.shape[1])
    assert runtime(torch.tensor(1)).shape == (c.shape[1],)
    with pytest.raises(IndexError):
        runtime(torch.tensor([-1]))
    with pytest.raises(IndexError):
        runtime(torch.tensor([c.shape[0]]))
    with pytest.raises(ValueError):
        runtime(torch.tensor([1.0]))


def test_dtype_moves_preserve_storage_and_default_is_bf16():
    c, *_ = affine(6)
    model = mf.MixedLinear(c)
    assert model.weight.dtype == torch.bfloat16
    model.to(dtype=torch.float16)
    assert model.weight.dtype == torch.float16
    for name in c.tensors:
        assert torch.equal(getattr(model, "payload_" + name), c.tensors[name])
        assert getattr(model, "payload_" + name).dtype == c.tensors[name].dtype
    bf = mf.bf16_candidate(torch.randn(3, 4).bfloat16())
    m = mf.MixedLinear(bf).float()
    assert m.payload_weight.dtype == torch.bfloat16
    assert torch.equal(m.payload_weight, bf.tensors["weight"])


@pytest.mark.parametrize("mutation", [
    lambda b: b[:7], lambda b: b[:-1], lambda b: b + b"extra",
    lambda b: bytes([b[0] ^ 1]) + b[1:],
    lambda b: b[:len(b)//2] + bytes([b[len(b)//2] ^ 1]) + b[len(b)//2+1:],
    lambda b: b[:-1] + bytes([b[-1] ^ 1]),
])
def test_corruption_and_truncation_rejected(mutation):
    blob = mf.encode_candidate(affine()[0])
    with pytest.raises(ValueError):
        mf.decode_candidate(mutation(blob))


def resign(blob, mutate_header=None, mutate_payload=None):
    prefix = len(mf.MAGIC) + 8
    size, = struct.unpack("<Q", blob[len(mf.MAGIC):prefix])
    header = json.loads(blob[prefix:prefix + size])
    payload = bytearray(blob[prefix + size:-32])
    if mutate_header:
        mutate_header(header)
    if mutate_payload:
        mutate_payload(header, payload)
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    body = mf.MAGIC + struct.pack("<Q", len(raw_header)) + raw_header + payload
    return body + hashlib.sha256(body).digest()


@pytest.mark.parametrize("mutator", [
    lambda h: h["metadata"].update(bits=True),
    lambda h: h["metadata"].update(group_axis="flattened"),
    lambda h: h["metadata"].update(shape=[0, 13]),
    lambda h: h["metadata"].update(unknown=True),
    lambda h: h["tensors"]["codes"].update(offset=1),
    lambda h: h["tensors"]["codes"].update(dtype="pickle"),
    lambda h: h["tensors"]["codes"].update(shape=[2**63]),
    lambda h: h.update(payload_bytes=True),
])
def test_valid_checksum_does_not_bypass_schema_validation(mutator):
    with pytest.raises(ValueError):
        mf.decode_candidate(resign(mf.encode_candidate(affine()[0]), mutate_header=mutator))


def test_valid_checksum_reserved_ternary_code_and_bitcos_index_rejected():
    c, _ = ternary(packing="2bit")
    def reserved(header, payload):
        payload[header["tensors"]["codes"]["offset"]] |= 3
    with pytest.raises(ValueError, match="ternary codes"):
        mf.decode_candidate(resign(mf.encode_candidate(c), mutate_payload=reserved))
    c, _ = ternary(packing="bitcos")
    def bad_index(header, payload):
        payload[header["tensors"]["row_sign_offsets"]["offset"] + 8] ^= 1
    with pytest.raises(ValueError, match="row index"):
        mf.decode_candidate(resign(mf.encode_candidate(c), mutate_payload=bad_index))


def test_nonzero_final_padding_rejected_even_with_valid_checksum():
    c, *_ = affine(6, rows=1, width=1)
    def padding(header, payload):
        payload[header["tensors"]["codes"]["offset"]] |= 128
    with pytest.raises(ValueError, match="padding"):
        mf.decode_candidate(resign(mf.encode_candidate(c), mutate_payload=padding))


def test_bf16_no_silent_rounding_and_no_embedding_bias():
    with pytest.raises(ValueError, match="original BF16"):
        mf.bf16_candidate(torch.randn(3, 4))
    with pytest.raises(ValueError, match="original BF16 bias"):
        mf.bf16_candidate(torch.randn(3, 4).bfloat16(), torch.randn(3))
    with pytest.raises(ValueError, match="cannot have bias"):
        mf.bf16_candidate(torch.randn(3, 4).bfloat16(), torch.randn(3).bfloat16(), module_kind="embedding")
    with pytest.raises(ValueError):
        mf.bf16_candidate(torch.full((2, 2), float("nan"), dtype=torch.bfloat16))


@pytest.mark.parametrize("which", ["negative_codes", "large_codes", "fractional_zero", "bad_scale", "zero_scale", "bad_shape"])
def test_affine_invalid_import_rejected(which):
    _, q, scales, zeros, _ = affine()
    if which == "negative_codes": q[0, 0] = -1
    if which == "large_codes": q[0, 0] = 16
    if which == "fractional_zero": zeros = zeros.float() + .5
    if which == "bad_scale": scales[0, 0] = float("nan")
    if which == "zero_scale": scales[0, 0] = 0
    if which == "bad_shape": scales = scales[:, :-1]
    with pytest.raises(ValueError):
        mf.affine_candidate(q, scales, zeros, bits=4, group_size=5)


def test_wrong_runtime_kind_and_invalid_configuration_rejected():
    c, *_ = affine(kind="embedding")
    with pytest.raises(ValueError): mf.MixedLinear(c)
    with pytest.raises(ValueError): mf.MixedEmbedding(c, padding_idx=1000)
    with pytest.raises(ValueError): mf.MixedEmbedding(c, row_chunk_size=0)
    with pytest.raises(ValueError): mf.MixedEmbedding(affine()[0])


@pytest.mark.parametrize("kind", ["linear", "embedding"])
@pytest.mark.parametrize("codec", ["affine4", "affine6", "2bit", "bitcos", "base3", "bf16"])
def test_rotation_payload_runtime_math_exact_reload_and_size(kind, codec):
    generator = torch.Generator().manual_seed(22)
    rows, width = 7, 32
    if codec.startswith("affine"):
        base, *_ = affine(int(codec[-1]), kind=kind, rows=rows, width=width)
    elif codec == "bf16":
        base = mf.bf16_candidate(torch.randn(rows, width, generator=generator).bfloat16(), module_kind=kind)
    else:
        base, _ = ternary(kind=kind, packing=codec, rows=rows, width=width)
    signs = torch.randint(0, 2, (width,), generator=generator) * 2 - 1
    candidate = mf.with_rotation(base, signs, 8)
    assert "rotation" not in base.metadata
    assert candidate.tensors["rotation_signs"].dtype == torch.int8
    # Stored-weight reconstruction remains LATENT. Caller owns W -> W R.
    assert torch.equal(candidate.dequantize(), base.dequantize())
    assert mf.candidate_nbytes(candidate) > mf.candidate_nbytes(base) + width
    blob = mf.encode_candidate(candidate)
    assert len(blob) == mf.candidate_nbytes(candidate)
    restored = mf.decode_candidate(blob)
    effective = inverse_rotate(base.dequantize(), signs, 8)
    if kind == "linear":
        runtime = mf.MixedLinear(candidate, dtype=torch.float32, row_chunk_size=3)
        x = torch.randn(2, width, generator=generator).requires_grad_()
        bias = base.tensors.get("bias")
        expected = F.linear(x, effective, None if bias is None else bias.float())
        actual = runtime(x)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        expected_grad, = torch.autograd.grad(expected.sum(), x)
        actual_grad, = torch.autograd.grad(actual.sum(), x)
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)
        assert torch.equal(mf.MixedLinear(restored, dtype=torch.float32, row_chunk_size=3)(x.detach()), actual.detach())
    else:
        runtime = mf.MixedEmbedding(candidate, dtype=torch.float32, row_chunk_size=2)
        ids = torch.tensor([6, 1, 1, 0])
        expected = F.embedding(ids, effective)
        actual = runtime(ids)
        assert torch.equal(actual, expected)
        assert torch.equal(mf.MixedEmbedding(restored, dtype=torch.float32, row_chunk_size=2)(ids), actual)


def test_bf16_rotated_storage_rounding_is_not_mistaken_for_exact_function_preservation():
    generator = torch.Generator().manual_seed(52)
    original = torch.randn(7, 32, generator=generator).bfloat16()
    signs = torch.randint(0, 2, (32,), generator=generator) * 2 - 1
    latent = rotate(original.float(), signs, 8)
    candidate = mf.with_rotation(mf.bf16_candidate(latent.bfloat16(), source="rotated-bf16-storage"), signs, 8)
    x = torch.randn(4, 32, generator=generator)
    baseline = F.linear(x, original.float())
    actual = mf.MixedLinear(candidate, dtype=torch.float32)(x)
    assert torch.isfinite(actual).all()
    assert (actual - baseline).square().mean() > 0  # storage rounding is real
    # The exact control uses FP32 latent weights; this is no quantization test.
    exact_control = F.linear(rotate(x, signs, 8), latent)
    torch.testing.assert_close(exact_control, baseline, atol=2e-5, rtol=2e-5)


def test_rotation_payload_contract_rejects_wrong_signs_orientation_and_duplicate_attachment():
    base, *_ = affine(width=16)
    with pytest.raises(ValueError): mf.with_rotation(base, torch.zeros(16), 8)
    with pytest.raises(ValueError): mf.with_rotation(base, torch.ones(16), 3)
    candidate = mf.with_rotation(base, torch.ones(16), 8)
    with pytest.raises(ValueError): mf.with_rotation(candidate, torch.ones(16), 8)
    blob = mf.encode_candidate(candidate)
    with pytest.raises(ValueError, match="rotation contract"):
        mf.decode_candidate(resign(blob, mutate_header=lambda h: h["metadata"]["rotation"].update(convention="row-HD")))
    def bad_sign(header, payload):
        payload[header["tensors"]["rotation_signs"]["offset"]] = 0
    with pytest.raises(ValueError, match="rotation signs"):
        mf.decode_candidate(resign(blob, mutate_payload=bad_sign))
