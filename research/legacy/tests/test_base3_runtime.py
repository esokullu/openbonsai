"""Lossless codec integration, not pretrained quantization-quality evidence."""
import pytest
import torch
from torch.nn import functional as F

from openbonsai import mixed_formats as mf
from openbonsai.packing import pack_grouped_weight, unpack_grouped_weight
from openbonsai.rotations import fixed_signs


def candidate(rows=9, width=17, group_size=8, *, kind="linear", zero=False):
    rng = torch.Generator().manual_seed(71)
    codes = torch.randint(-1, 2, (rows, width), generator=rng, dtype=torch.int8)
    if zero:
        codes.zero_()
    scales = torch.rand((rows * width + group_size - 1) // group_size, generator=rng) + .02
    bias = torch.randn(rows, generator=rng) if kind == "linear" else None
    state = pack_grouped_weight(codes, scales, shape=(rows, width), group_size=group_size, bias=bias)
    return mf.ternary_candidate(state, module_kind=kind, packing="base3",
        provenance={"source_checkpoint": "fixture", "source_weight_sha256": "a" * 64}), state


@pytest.mark.parametrize("shape,group", [((3, 7), 8), ((2, 4096), 128), ((7, 1), 3), ((129, 32), 128)])
def test_arbitrary_embedding_rows_cross_radix_bytes_and_scale_groups(shape, group):
    c, state = candidate(*shape, group, kind="embedding")
    expected = unpack_grouped_weight(state)
    ids = torch.tensor([[shape[0] - 1, 0, 1, 1]])
    runtime = mf.MixedEmbedding(c, dtype=torch.float32, row_chunk_size=1)
    torch.testing.assert_close(runtime(ids), F.embedding(ids, expected), atol=0, rtol=0)
    torch.testing.assert_close(mf.candidate_weight(c, row_chunk_size=1), expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["linear", "embedding"])
@pytest.mark.parametrize("source_codec", ["2bit", "bitcos", "base3"])
def test_repack_preserves_codes_metadata_scales_signs_and_runtime(kind, source_codec, monkeypatch):
    c, _ = candidate(width=32, kind=kind)
    c = mf.with_rotation(c, fixed_signs(32, seed=4), 8)
    c = mf.repack_ternary_candidate(c, packing=source_codec)
    source_blob = mf.encode_candidate(c)
    expected = c.dequantize()

    def forbid_dense(*_args, **_kwargs):
        raise AssertionError("Repacking must not reconstruct floating weights")

    with monkeypatch.context() as m:
        m.setattr(mf, 'candidate_weight', forbid_dense)
        changed = mf.repack_ternary_candidate(c, packing="base3")
    assert mf.encode_candidate(c) == source_blob
    assert changed.metadata == {**c.metadata, "encoding": "ternary-base3"}
    for name in ("scales", "bias", "rotation_signs"):
        if name in c.tensors:
            assert torch.equal(c.tensors[name], changed.tensors[name])
            assert c.tensors[name].dtype == changed.tensors[name].dtype
    assert torch.equal(changed.dequantize(), expected)
    restored = mf.decode_candidate(mf.encode_candidate(changed))
    assert torch.equal(restored.dequantize(), expected)
    if kind == "linear":
        x = torch.randn(2, 32, requires_grad=True)
        a = mf.MixedLinear(c, dtype=torch.float32, row_chunk_size=2)(x)
        b = mf.MixedLinear(restored, dtype=torch.float32, row_chunk_size=2)(x)
        assert torch.equal(a, b)
        assert torch.equal(torch.autograd.grad(a.sum(), x)[0], torch.autograd.grad(b.sum(), x)[0])
    else:
        ids = torch.tensor([8, 0, 3, 3])
        assert torch.equal(mf.MixedEmbedding(c)(ids), mf.MixedEmbedding(restored)(ids))


@pytest.mark.parametrize("corruption", ["reserved", "tail", "group", "index", "wrong_length", "wrong_dtype"])
def test_base3_candidate_rejects_noncanonical_payload(corruption):
    c, _ = candidate(2, 7, 8, zero=True)  # 14 values, padded to 16, stored in four bytes.
    if corruption == "reserved":
        c.tensors["codes"][0] = 243
    elif corruption == "tail":
        c.tensors["codes"][-1] = 3  # Position 16 is outside padded count.
    elif corruption == "group":
        c.tensors["codes"][2] = 81  # Position 14 is group padding.
    elif corruption == "index":
        c.tensors["row_sign_offsets"] = torch.zeros(3, dtype=torch.int64)
    elif corruption == "wrong_length":
        c.tensors["codes"] = c.tensors["codes"][:-1]
    else:
        c.tensors["codes"] = c.tensors["codes"].to(torch.int16)
    with pytest.raises(ValueError):
        mf.encode_candidate(c)


def test_base3_row_runtime_does_not_expand_full_matrix(monkeypatch):
    c, _ = candidate(137, 4096, 128, kind="embedding")
    runtime = mf.MixedEmbedding(c, row_chunk_size=2)
    sizes = []
    read = mf._base3_at

    def observed(stream, positions):
        sizes.append(positions.numel())
        return read(stream, positions)

    monkeypatch.setattr(mf, "_base3_at", observed)
    runtime(torch.tensor([136, 1, 1]))
    assert sizes and max(sizes) <= 2 * 4096


def test_auto_codec_selection_includes_full_candidate_bytes():
    dense, _ = candidate(16, 4096, 128)
    sparse, _ = candidate(16, 4096, 128, zero=True)
    for original, expected in ((dense, "ternary-base3"), (sparse, "ternary-bitcos")):
        choices = [mf.repack_ternary_candidate(original, packing=codec) for codec in ("2bit", "bitcos", "base3")]
        selected = mf.repack_ternary_candidate(original)
        assert selected.encoding == expected
        assert mf.candidate_nbytes(selected) == min(mf.candidate_nbytes(c) for c in choices)
        assert mf.encode_candidate(selected) == mf.encode_candidate(min(choices, key=mf.candidate_nbytes))


def test_integer_or_bf16_candidates_are_not_silently_ternarized():
    original = mf.bf16_candidate(torch.ones(2, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="ternary"):
        mf.repack_ternary_candidate(original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_repack_preserves_imported_bias_dtype_and_values(dtype):
    original, _ = candidate()
    original.tensors["bias"] = torch.linspace(.10001, .90001, 9).to(dtype)
    if dtype != torch.float16:
        original.tensors["bias"][0] = 70000  # No implicit FP16 conversion is valid.
    mf.validate_candidate(original)
    choices = []
    for codec in ("2bit", "bitcos", "base3", "auto"):
        result = mf.repack_ternary_candidate(original, packing=codec)
        assert result.tensors["bias"].dtype == dtype
        assert torch.equal(result.tensors["bias"], original.tensors["bias"])
        choices.append(result)
    assert mf.candidate_nbytes(choices[-1]) == min(mf.candidate_nbytes(c) for c in choices[:-1])
