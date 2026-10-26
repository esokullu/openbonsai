import copy
import io

import pytest
import torch

from openbonsai.packing import (
    pack_grouped_weight, pack_ternary_codes, packed_state_nbytes,
    unpack_grouped_weight, unpack_ternary_codes, validate_packed_state,
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
