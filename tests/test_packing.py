import copy
import io

import pytest
import torch

from openbonsai.packing import (
    BITCOS_FORMAT, FORMAT, pack_bitcos_codes, pack_grouped_weight,
    pack_ternary_codes, packed_state_nbytes, repack_grouped_weight,
    unpack_bitcos_codes, unpack_grouped_codes, unpack_grouped_weight,
    unpack_ternary_codes, validate_packed_state,
)


def test_bit_layout_and_non_multiple_of_four():
    codes = torch.tensor([-1, 0, 1, -1, 1], dtype=torch.int8)
    packed = pack_ternary_codes(codes)
    assert packed.tolist() == [146, 1]
    assert torch.equal(unpack_ternary_codes(packed, 5), codes)
    assert pack_ternary_codes(torch.empty(0)).numel() == 0
    assert unpack_ternary_codes(torch.empty(0, dtype=torch.uint8), 0).numel() == 0


@pytest.mark.parametrize("codes", [torch.tensor([2]), torch.tensor([0.5]), torch.tensor([float('nan')])])
def test_reject_nonternary_values(codes):
    with pytest.raises(ValueError, match="ternary codes"):
        pack_ternary_codes(codes)


def test_reject_reserved_code_and_noncanonical_padding():
    with pytest.raises(ValueError, match="reserved"):
        unpack_ternary_codes(torch.tensor([3], dtype=torch.uint8), 1)
    with pytest.raises(ValueError, match="padding"):
        unpack_ternary_codes(torch.tensor([4], dtype=torch.uint8), 1)
    with pytest.raises(ValueError, match="byte count"):
        unpack_ternary_codes(torch.tensor([0, 0], dtype=torch.uint8), 1)


def test_group_padding_bytes_and_serialization():
    codes = torch.tensor([-1, 0, 1, 1, 0, -1, 1, -1, 0, 1])
    scales = torch.tensor([0.125, 0.3333, 1.75])
    state = pack_grouped_weight(codes, scales, shape=[2, 5], group_size=4, bias=torch.tensor([0.1, -0.2]))
    assert state["padded_numel"] == 12
    assert state["num_groups"] == 3
    assert state["codes"].dtype == torch.uint8
    assert state["scales"].dtype == torch.float16
    assert state["bias"].dtype == torch.float16
    assert packed_state_nbytes(state) == 3 + 6 + 4
    assert state["storage"]["tensor_payload_bytes"] == 13
    assert state["storage"]["weight_payload_bits_per_weight"] == 7.2
    assert not state["storage"]["includes_container_overhead"]
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    weight = unpack_grouped_weight(restored)
    expected = (torch.cat([codes, torch.zeros(2)]).reshape(3, 4) * scales.half().float()[:, None]).flatten()[:10].reshape(2, 5)
    torch.testing.assert_close(weight, expected, rtol=0, atol=0)
    assert stream.getbuffer().nbytes > packed_state_nbytes(state)


def test_padding_does_not_contain_hidden_codes():
    with pytest.raises(ValueError, match="group padding"):
        pack_grouped_weight(torch.tensor([1, 0, 0, 1]), torch.tensor([1.0]), shape=[1, 3], group_size=4)
    state = pack_grouped_weight(torch.tensor([1, 0, 0]), torch.tensor([1.0]), shape=[1, 3], group_size=4)
    state["codes"][0] |= 64
    with pytest.raises(ValueError, match="group padding"):
        validate_packed_state(state)


def test_reject_bad_metadata_and_unrepresentable_fp16():
    state = pack_grouped_weight(torch.ones(4), torch.ones(1), shape=[2, 2], group_size=4)
    bad = copy.deepcopy(state)
    bad["padded_numel"] = 8
    with pytest.raises(ValueError, match="padded_numel"):
        validate_packed_state(bad)
    bad = copy.deepcopy(state)
    bad["scales"] = bad["scales"].float()
    with pytest.raises(ValueError, match="FP16"):
        validate_packed_state(bad)
    for scale in (1e10, 1e-12):
        with pytest.raises(ValueError, match="representable"):
            pack_grouped_weight(torch.ones(4), torch.tensor([scale]), shape=[2, 2], group_size=4)


def test_default_group_payload_is_two_bits_plus_scale_overhead():
    state = pack_grouped_weight(torch.ones(256), torch.ones(2), shape=[2, 128], group_size=128)
    assert packed_state_nbytes(state) == 64 + 4
    assert state["storage"]["weight_payload_bits_per_weight"] == 2.125


def test_bitcos_known_bytes_and_compact_sign_order():
    # Presence follows all coordinates; signs follow only nonzero coordinates.
    codes = torch.tensor([[-1, 0, 1, -1, 1], [0, 0, -1, 1, -1]])
    presence, signs = pack_bitcos_codes(codes)
    assert presence.tolist() == [0b10011101, 0b00000011]
    assert signs.tolist() == [0b01010101]
    assert presence.dtype == signs.dtype == torch.uint8
    assert presence.ndim == signs.ndim == 1
    assert torch.equal(unpack_bitcos_codes(presence, signs, codes.numel()), codes.flatten().to(torch.int8))


@pytest.mark.parametrize("numel", [0, 1, 7, 8, 9, 17, 33])
@pytest.mark.parametrize("pattern", ["zero", "positive", "negative", "alternating", "random"])
def test_bitcos_stream_roundtrip_and_byte_counts(numel, pattern):
    if pattern == "zero":
        codes = torch.zeros(numel, dtype=torch.int8)
    elif pattern == "positive":
        codes = torch.ones(numel, dtype=torch.int8)
    elif pattern == "negative":
        codes = -torch.ones(numel, dtype=torch.int8)
    elif pattern == "alternating":
        codes = (torch.arange(numel) % 3 - 1).to(torch.int8)
    else:
        codes = torch.randint(-1, 2, (numel,), generator=torch.Generator().manual_seed(42), dtype=torch.int8)
    presence, signs = pack_bitcos_codes(codes)
    assert presence.numel() == (numel + 7) // 8
    assert signs.numel() == (int(codes.count_nonzero()) + 7) // 8
    decoded = unpack_bitcos_codes(presence, signs, numel)
    assert decoded.dtype == torch.int8
    assert torch.equal(decoded, codes)


@pytest.mark.parametrize("codes", [torch.tensor([2]), torch.tensor([0.5]), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_bitcos_rejects_nonternary_codes(codes):
    with pytest.raises(ValueError, match="ternary codes"):
        pack_bitcos_codes(codes)


@pytest.mark.parametrize("codes", [None, [0, 1], torch.tensor([1j])])
def test_bitcos_rejects_nonreal_tensors(codes):
    with pytest.raises(ValueError, match="real tensor"):
        pack_bitcos_codes(codes)


@pytest.mark.parametrize("numel", [-1, True, 1.0, None])
def test_bitcos_rejects_invalid_numel(numel):
    with pytest.raises(ValueError, match="nonnegative integer"):
        unpack_bitcos_codes(torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8), numel)


@pytest.mark.parametrize("which", ["presence", "signs"])
@pytest.mark.parametrize("bad", [None, [1], torch.tensor([1], dtype=torch.int8), torch.tensor([[1]], dtype=torch.uint8)])
def test_bitcos_rejects_bad_stream_dtype_or_shape(which, bad):
    streams = {"presence": torch.tensor([1], dtype=torch.uint8), "signs": torch.tensor([0], dtype=torch.uint8)}
    streams[which] = bad
    with pytest.raises(ValueError, match=f"{which} must be a flat uint8 tensor"):
        unpack_bitcos_codes(**streams, numel=1)


@pytest.mark.parametrize("presence,signs,numel,message", [
    ([], [], 1, "presence byte count"),
    ([0, 0], [], 1, "presence byte count"),
    ([1], [], 1, "signs byte count"),
    ([1], [0, 0], 1, "signs byte count"),
    ([0], [0], 1, "signs byte count"),
    ([2], [], 1, "presence final-byte padding"),
    ([1], [2], 1, "signs final-byte padding"),
    ([], [0], 0, "signs byte count"),
])
def test_bitcos_rejects_lengths_and_padding(presence, signs, numel, message):
    with pytest.raises(ValueError, match=message):
        unpack_bitcos_codes(torch.tensor(presence, dtype=torch.uint8), torch.tensor(signs, dtype=torch.uint8), numel)


@pytest.mark.parametrize("group_size", [1, 3, 5, 7, 8, 9, 128])
@pytest.mark.parametrize("pattern", ["zero", "dense", "random"])
def test_bitcos_grouped_serialization_and_repack_preserve_exact_values(group_size, pattern):
    if pattern == "zero":
        codes = torch.zeros(15, dtype=torch.int8)
    elif pattern == "dense":
        codes = (torch.arange(15) % 2 * 2 - 1).to(torch.int8)
    else:
        codes = torch.randint(-1, 2, (15,), generator=torch.Generator().manual_seed(24), dtype=torch.int8)
    num_groups = (15 + group_size - 1) // group_size
    scales = torch.linspace(0.12345, 1.23456, num_groups)
    bias = torch.tensor([-0.0, 0.12345, -1.23456])
    original = pack_grouped_weight(codes, scales, shape=[3, 5], group_size=group_size, bias=bias)
    original_copy = copy.deepcopy(original)
    bitcos = repack_grouped_weight(original, "bitcos")
    assert original["format"] == FORMAT
    assert bitcos["format"] == BITCOS_FORMAT
    assert "codes" not in bitcos
    for field in ("scales", "bias"):
        assert torch.equal(original[field].view(torch.int16), bitcos[field].view(torch.int16))
    assert torch.equal(unpack_grouped_codes(original), unpack_grouped_codes(bitcos))
    assert torch.equal(unpack_grouped_weight(original), unpack_grouped_weight(bitcos))
    assert torch.equal(original["codes"], original_copy["codes"])
    assert original["storage"] == original_copy["storage"]
    expected_bytes = sum(bitcos[key].numel() * bitcos[key].element_size() for key in ("presence", "signs", "scales", "bias"))
    assert packed_state_nbytes(bitcos) == expected_bytes == bitcos["storage"]["tensor_payload_bytes"]
    assert bitcos["storage"]["code_bytes"] == bitcos["storage"]["presence_bytes"] + bitcos["storage"]["sign_bytes"]
    stream = io.BytesIO()
    torch.save(bitcos, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    roundtrip = repack_grouped_weight(restored, "2bit")
    assert torch.equal(roundtrip["codes"], original["codes"])
    assert roundtrip["storage"] == original["storage"]
    for field in ("scales", "bias"):
        assert torch.equal(roundtrip[field].view(torch.int16), original[field].view(torch.int16))
    assert torch.equal(unpack_grouped_weight(restored, dtype=torch.float64), unpack_grouped_weight(original, dtype=torch.float64))


@pytest.mark.parametrize("numel,nonzero,expected_format,expected_code_bytes", [
    (1, 0, FORMAT, 1),  # Tie: the presence stream alone takes one byte.
    (1, 1, FORMAT, 1),  # BITCOS can be larger on short dense tensors.
    (8, 0, BITCOS_FORMAT, 1),
    (8, 1, FORMAT, 2),
    (8, 8, FORMAT, 2),
    (9, 1, FORMAT, 3),
    (16, 8, BITCOS_FORMAT, 3),
    (16, 9, FORMAT, 4),
])
def test_auto_compares_actual_byte_counts_with_legacy_tie_break(numel, nonzero, expected_format, expected_code_bytes):
    codes = torch.zeros(numel, dtype=torch.int8)
    codes[:nonzero] = -1
    state = pack_grouped_weight(codes, torch.ones(1), shape=[1, numel], group_size=numel, packing="auto")
    assert state["format"] == expected_format
    assert state["storage"]["code_bytes"] == expected_code_bytes
    assert torch.equal(unpack_grouped_codes(state), codes)
    forced_bitcos = pack_grouped_weight(codes, torch.ones(1), shape=[1, numel], group_size=numel, packing="bitcos")
    assert forced_bitcos["format"] == BITCOS_FORMAT
    assert repack_grouped_weight(forced_bitcos)["format"] == expected_format


def test_auto_accounts_for_group_padding_before_choosing_format():
    state = pack_grouped_weight(torch.ones(1), torch.ones(1), shape=[1, 1], group_size=17, packing="auto")
    assert state["format"] == BITCOS_FORMAT
    assert state["storage"]["code_bytes"] == 4
    assert unpack_grouped_codes(state).tolist() == [1] + [0] * 16


def test_bitcos_rejects_hidden_group_codes_even_with_valid_stream_padding():
    state = pack_grouped_weight(torch.tensor([1, 0, 0]), torch.ones(1), shape=[1, 3], group_size=4, packing="bitcos")
    state["presence"][0] |= 8  # Two valid nonzeros in the byte, one outside the shape.
    with pytest.raises(ValueError, match="group padding"):
        validate_packed_state(state)
    with pytest.raises(ValueError, match="group padding"):
        pack_grouped_weight(torch.tensor([1, 0, 0, 1]), torch.ones(1), shape=[1, 3], group_size=4, packing="bitcos")


@pytest.mark.parametrize("packing", ["2bit", "bitcos"])
def test_validate_rejects_wrong_streams_and_unaccounted_tensors(packing):
    state = pack_grouped_weight(torch.zeros(8), torch.ones(1), shape=[1, 8], group_size=8, packing=packing)
    bad = copy.deepcopy(state)
    bad["codes" if packing == "bitcos" else "presence"] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="cannot contain"):
        validate_packed_state(bad)
    bad = copy.deepcopy(state)
    bad["unreported"] = torch.zeros(8)
    with pytest.raises(ValueError, match="unexpected tensor"):
        packed_state_nbytes(bad)


@pytest.mark.parametrize("packing", ["2bit", "bitcos"])
@pytest.mark.parametrize("field", ["code_bytes", "scale_bytes", "bias_bytes", "tensor_payload_bytes", "weight_payload_bits_per_weight", "includes_container_overhead"])
def test_storage_accounting_cannot_disagree_with_payloads(packing, field):
    state = pack_grouped_weight(torch.zeros(8), torch.ones(1), shape=[1, 8], group_size=8, packing=packing)
    state["storage"][field] += 1
    with pytest.raises(ValueError, match="storage"):
        validate_packed_state(state)


def test_storage_is_optional_for_old_handmade_states_and_formats_are_strict():
    state = pack_grouped_weight(torch.zeros(8), torch.ones(1), shape=[1, 8], group_size=8)
    del state["storage"]
    assert packed_state_nbytes(state) == 4
    state["format"] = "unknown"
    with pytest.raises(ValueError, match="expected packed format"):
        unpack_grouped_weight(state)
    with pytest.raises(ValueError, match="packing must be"):
        pack_grouped_weight(torch.zeros(8), torch.ones(1), shape=[1, 8], group_size=8, packing="unknown")
