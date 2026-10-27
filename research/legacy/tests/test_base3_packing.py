"""Exact base-3 storage semantics; no training or inference-speed claim."""
import copy
import io
import math

import pytest
import torch

from openbonsai.packing import (
    BASE3_FORMAT, BITCOS_FORMAT, FORMAT, pack_base3_codes, pack_grouped_weight,
    packed_state_nbytes, repack_grouped_weight, unpack_base3_codes,
    unpack_grouped_codes, unpack_grouped_weight, validate_packed_state,
)


def test_all_243_full_byte_patterns_match_independent_integer_oracle():
    # Enumerate every valid byte without using the implementation's arithmetic.
    lookup = (0, 1, -1)
    codes = torch.tensor([[lookup[(byte // (3 ** lane)) % 3] for lane in range(5)]
                          for byte in range(243)], dtype=torch.int8)
    expected = torch.arange(243, dtype=torch.uint8)
    assert torch.equal(pack_base3_codes(codes), expected)
    assert torch.equal(unpack_base3_codes(expected, 243 * 5), codes.flatten())
    assert pack_base3_codes(torch.tensor([0, 1, -1, 1, 0, -1])).tolist() == [48, 2]
    assert pack_base3_codes(torch.ones(5)).item() == 121
    assert pack_base3_codes(-torch.ones(5)).item() == 242
    assert pack_base3_codes(torch.zeros(5)).item() == 0


@pytest.mark.parametrize("count", [*range(21), 127, 128, 129, 4095, 4096, 4097])
@pytest.mark.parametrize("pattern", ["zeros", "positive", "negative", "random"])
def test_exact_roundtrip_tail_and_actual_bytes(count, pattern):
    if pattern == "random":
        codes = torch.randint(-1, 2, (count,), dtype=torch.int8,
                              generator=torch.Generator().manual_seed(451))
    else:
        value = {"zeros": 0, "positive": 1, "negative": -1}[pattern]
        codes = torch.full((count,), value, dtype=torch.int8)
    packed = pack_base3_codes(codes)
    assert packed.dtype == torch.uint8 and packed.ndim == 1
    assert packed.numel() == (count + 4) // 5
    decoded = unpack_base3_codes(packed, count)
    assert decoded.dtype == torch.int8 and torch.equal(decoded, codes)
    if count % 5:
        assert int(packed[-1]) < 3 ** (count % 5)
    # Independent per-byte integer oracle verifies boundaries and tail zeros.
    digit = {0: 0, 1: 1, -1: 2}
    expected = [sum(digit[int(codes[start + lane])] * 3 ** lane
                    for lane in range(min(5, count - start)))
                for start in range(0, count, 5)]
    assert packed.tolist() == expected


def test_noncontiguous_multidimensional_inputs_use_row_major_logical_order():
    codes = torch.tensor([[0, -1, 1, 0], [1, 1, -1, -1], [-1, 0, 0, 1]]).T
    assert not codes.is_contiguous()
    before = codes.clone()
    packed = pack_base3_codes(codes)
    assert torch.equal(unpack_base3_codes(packed, codes.numel()), codes.reshape(-1).to(torch.int8))
    assert torch.equal(codes, before)
    floating = codes.to(torch.float32).requires_grad_()
    assert torch.equal(pack_base3_codes(floating), packed)
    assert floating.grad is None


@pytest.mark.parametrize("invalid", [None, [0, 1], torch.tensor([1j]), torch.tensor([2]),
                                     torch.tensor([.5]), torch.tensor([float("nan")]),
                                     torch.tensor([float("inf")])])
def test_reject_invalid_codes(invalid):
    with pytest.raises(ValueError, match="real tensor|ternary codes"):
        pack_base3_codes(invalid)


@pytest.mark.parametrize("count", [-1, True, 1., None])
def test_reject_invalid_count(count):
    with pytest.raises(ValueError, match="nonnegative integer"):
        unpack_base3_codes(torch.empty(0, dtype=torch.uint8), count)


@pytest.mark.parametrize("packed", [None, [0], torch.tensor([0], dtype=torch.int8),
                                    torch.tensor([[0]], dtype=torch.uint8), torch.tensor([0.])])
def test_reject_invalid_stream(packed):
    with pytest.raises(ValueError, match="flat uint8"):
        unpack_base3_codes(packed, 1)


@pytest.mark.parametrize("value", range(243, 256))
def test_every_reserved_byte_rejected(value):
    with pytest.raises(ValueError, match="reserved base-3"):
        unpack_base3_codes(torch.tensor([value], dtype=torch.uint8), 5)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_noncanonical_final_trits_rejected_even_for_valid_byte(count):
    with pytest.raises(ValueError, match="padding trits"):
        unpack_base3_codes(torch.tensor([3 ** count], dtype=torch.uint8), count)


@pytest.mark.parametrize("bytes_,count", [(0, 1), (1, 0), (2, 5), (1, 6), (3, 9)])
def test_inconsistent_lengths_rejected(bytes_, count):
    with pytest.raises(ValueError, match="byte count"):
        unpack_base3_codes(torch.zeros(bytes_, dtype=torch.uint8), count)


@pytest.mark.parametrize("group_size", [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 128])
@pytest.mark.parametrize("shape", [(3, 7), (2, 128), (7, 4096)])
def test_grouped_repacking_preserves_exact_codes_scale_bias_bits_and_weight(group_size, shape):
    count = math.prod(shape)
    generator = torch.Generator().manual_seed(454)
    codes = torch.randint(-1, 2, shape, generator=generator, dtype=torch.int8)
    scales = torch.linspace(.12345, 1.23456, math.ceil(count / group_size))
    bias = torch.linspace(-.45678, 1.12345, shape[0])
    bias[0] = -0.
    original = pack_grouped_weight(codes, scales, shape=shape, group_size=group_size, bias=bias)
    original_copy = copy.deepcopy(original)
    base3 = repack_grouped_weight(original, "base3")
    assert base3["format"] == BASE3_FORMAT
    assert base3.keys() == original.keys()
    assert base3["codes"].numel() == (base3["padded_numel"] + 4) // 5
    expected_payload = base3["codes"].numel() + 2 * (base3["num_groups"] + shape[0])
    assert packed_state_nbytes(base3) == expected_payload
    assert base3["storage"]["tensor_payload_bytes"] == expected_payload
    assert base3["storage"]["weight_payload_bits_per_weight"] == (
        8 * (base3["codes"].numel() + 2 * base3["num_groups"]) / count)
    assert base3["storage"]["includes_container_overhead"] is False
    assert torch.equal(unpack_grouped_codes(original), unpack_grouped_codes(base3))
    for field in ("scales", "bias"):
        assert torch.equal(base3[field].view(torch.int16), original[field].view(torch.int16))
    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        assert torch.equal(unpack_grouped_weight(original, dtype=dtype), unpack_grouped_weight(base3, dtype=dtype))
    for codec in ("2bit", "bitcos", "base3"):
        other = repack_grouped_weight(base3, codec)
        back = repack_grouped_weight(other, "base3")
        assert torch.equal(back["codes"], base3["codes"])
        for field in ("scales", "bias"):
            assert torch.equal(back[field].view(torch.int16), base3[field].view(torch.int16))
    restored_legacy = repack_grouped_weight(base3, "2bit")
    assert torch.equal(restored_legacy["codes"], original["codes"])
    assert restored_legacy["storage"] == original["storage"]
    assert torch.equal(original["codes"], original_copy["codes"])


def test_grouped_tensor_serialization_and_default_legacy_format():
    args = dict(shape=[2, 7], group_size=4, bias=torch.tensor([-0., .01]))
    codes, scales = torch.arange(14) % 3 - 1, torch.tensor([.1, .2, .3, .4])
    assert pack_grouped_weight(codes, scales, **args)["format"] == FORMAT
    state = pack_grouped_weight(codes, scales, packing="base3", **args)
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    validate_packed_state(restored)
    assert torch.equal(unpack_grouped_codes(restored), unpack_grouped_codes(state))
    assert stream.getbuffer().nbytes > packed_state_nbytes(state)
    for field in ("scales", "bias"):
        assert torch.equal(restored[field].view(torch.int16), state[field].view(torch.int16))


@pytest.mark.parametrize("count,nonzero,expected", [
    (1, 0, FORMAT), (1, 1, FORMAT),  # All one-byte ties retain the original codec.
    (8, 0, BITCOS_FORMAT), (8, 8, FORMAT),
    (9, 1, BASE3_FORMAT), (10, 10, BASE3_FORMAT),
    (16, 8, BITCOS_FORMAT), (16, 9, FORMAT),
    (100, 0, BITCOS_FORMAT), (100, 50, BITCOS_FORMAT),
    (100, 56, BITCOS_FORMAT),  # BITCOS/base3 tie preserves BITCOS.
    (100, 57, BASE3_FORMAT), (100, 69, BASE3_FORMAT),
])
def test_auto_minimizes_actual_stream_bytes_and_preserves_older_tie_choices(count, nonzero, expected):
    codes = torch.zeros(count, dtype=torch.int8)
    codes[:nonzero] = -1
    args = dict(shape=[1, count], group_size=count)
    candidate = pack_grouped_weight(codes, torch.tensor([.123]), packing="auto", **args)
    forced = [pack_grouped_weight(codes, torch.tensor([.123]), packing=codec, **args)
              for codec in ("2bit", "bitcos", "base3")]
    assert candidate["format"] == expected
    assert packed_state_nbytes(candidate) == min(map(packed_state_nbytes, forced))
    assert candidate["storage"]["code_bytes"] == min(item["storage"]["code_bytes"] for item in forced)
    assert torch.equal(unpack_grouped_codes(candidate), codes)


def test_auto_includes_group_padding_in_all_three_costs():
    # One real nonzero, 19 group-padding zeros: BITCOS and base3 tie at 4 bytes.
    state = pack_grouped_weight(torch.ones(1), torch.ones(1), shape=[1, 1], group_size=20, packing="auto")
    assert state["format"] == BITCOS_FORMAT
    assert state["storage"]["code_bytes"] == 4
    assert unpack_grouped_codes(state).tolist() == [1] + [0] * 19


def test_reject_hidden_group_codes_and_invalid_stream_combination():
    with pytest.raises(ValueError, match="group padding"):
        pack_grouped_weight(torch.tensor([1, 0, 0, -1]), torch.ones(1),
                            shape=[1, 3], group_size=4, packing="base3")
    state = pack_grouped_weight(torch.zeros(14), torch.ones(5), shape=[2, 7], group_size=3, packing="base3")
    state["codes"][-1] = 81  # Canonical byte, but a nonzero hidden group-padding digit.
    with pytest.raises(ValueError, match="group padding"):
        validate_packed_state(state)
    state["codes"][-1] = 0
    for field in ("presence", "signs"):
        invalid = copy.deepcopy(state)
        invalid[field] = torch.empty(0, dtype=torch.uint8)
        with pytest.raises(ValueError, match="cannot contain"):
            validate_packed_state(invalid)
    state["unreported"] = torch.zeros(8)
    with pytest.raises(ValueError, match="unexpected tensor"):
        packed_state_nbytes(state)


@pytest.mark.parametrize("field", ["code_bytes", "scale_bytes", "bias_bytes", "tensor_payload_bytes",
                                  "weight_payload_bits_per_weight", "includes_container_overhead"])
def test_invalid_accounting_is_rejected(field):
    state = pack_grouped_weight(torch.zeros(14), torch.ones(5), shape=[2, 7], group_size=3, packing="base3")
    state["storage"][field] += 1
    with pytest.raises(ValueError, match="storage"):
        validate_packed_state(state)


def test_scale_bias_rounding_contract_unchanged_and_invalid_metadata_rejected():
    for scale in (float("nan"), float("inf"), -1., 1e10, 1e-12):
        with pytest.raises(ValueError, match="scale|FP16"):
            pack_grouped_weight(torch.ones(4), torch.tensor([scale]),
                                shape=[2, 2], group_size=4, packing="base3")
    with pytest.raises(ValueError, match="bias"):
        pack_grouped_weight(torch.ones(4), torch.ones(1), shape=[2, 2], group_size=4,
                            bias=torch.tensor([0., float("inf")]), packing="base3")
    state = pack_grouped_weight(torch.ones(4), torch.ones(1), shape=[2, 2], group_size=4, packing="base3")
    state["scales"] = state["scales"].float()
    with pytest.raises(ValueError, match="FP16"):
        validate_packed_state(state)
    state["scales"] = state["scales"].half()
    state["numel"] += 1
    with pytest.raises(ValueError, match="numel"):
        validate_packed_state(state)
