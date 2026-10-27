"""Portable 2-bit, BITCOS and base-3 ternary payloads.

Four codes occupy a byte, least significant code first: 0 -> 00,
+1 -> 01, -1 -> 10. 11 is invalid. Groups and final bytes are zero padded.
BITCOS stores one presence bit per code, followed by a separate stream with
one sign bit per nonzero code (1 means negative), both least significant first.
Base-3 stores five trits per byte, least significant trit first, using digits
0 -> 0, +1 -> 1, -1 -> 2. Bytes 243 through 255 are invalid; unused final trits
are zero. This independent format is not Prism PTQ1 or a GGUF encoding.
The reported bytes count tensor payloads, excluding container/metadata overhead.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

# Stable on-disk identifier: preserve compatibility with pre-OpenBonsai artifacts.
FORMAT = "scaleq.ternary-grouped.v1"
BITCOS_FORMAT = "openbonsai.ternary-bitcos.v1"
BASE3_FORMAT = "openbonsai.ternary-base3.v1"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ternary_flat(codes: Tensor) -> Tensor:
    if not isinstance(codes, Tensor) or codes.is_complex():
        raise ValueError("codes must be a real tensor")
    flat = codes.detach().reshape(-1)
    if not bool(((flat == -1) | (flat == 0) | (flat == 1)).all()):
        raise ValueError("ternary codes must be -1, 0, or +1")
    return flat


def pack_ternary_codes(codes: Tensor) -> Tensor:
    """Pack a tensor of {-1, 0, +1} values into a flat uint8 tensor."""
    flat = _ternary_flat(codes)
    encoded = torch.where(flat < 0, 2, flat).to(torch.uint8)
    if encoded.numel() % 4:
        encoded = torch.cat((encoded, encoded.new_zeros(4 - encoded.numel() % 4)))
    slots = encoded.reshape(-1, 4)
    return slots[:, 0] | (slots[:, 1] << 2) | (slots[:, 2] << 4) | (slots[:, 3] << 6)


def unpack_ternary_codes(packed: Tensor, numel: int) -> Tensor:
    """Unpack codes, rejecting reserved codes and nonzero final-byte padding."""
    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("numel must be a nonnegative integer")
    if not isinstance(packed, Tensor) or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed codes must be a flat uint8 tensor")
    if packed.numel() != math.ceil(numel / 4):
        raise ValueError("packed byte count does not match numel")
    slots = torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), dim=1).flatten()
    if bool((slots == 3).any()):
        raise ValueError("reserved 2-bit code 11 is invalid")
    if bool((slots[numel:] != 0).any()):
        raise ValueError("final-byte padding must be zero")
    slots = slots[:numel].to(torch.int8)
    return torch.where(slots == 2, -1, slots)


def pack_base3_codes(codes: Tensor) -> Tensor:
    """Losslessly pack five ternary codes per uint8, with zero tail digits.

    The flattened code at position i occupies digit i % 5 of byte i // 5.
    The digit mapping matches the existing 2-bit format: 0, +1, -1 -> 0, 1, 2.
    This is storage packing only; it does not change codes, scales or quality.
    """
    flat = _ternary_flat(codes)
    digits = torch.where(flat < 0, 2, flat).to(torch.uint8)
    if digits.numel() % 5:
        digits = torch.cat((digits, digits.new_zeros(5 - digits.numel() % 5)))
    slots = digits.reshape(-1, 5)
    # Every partial sum is at most 242, so uint8 arithmetic is exact here.
    return slots[:, 0] + 3 * slots[:, 1] + 9 * slots[:, 2] + 27 * slots[:, 3] + 81 * slots[:, 4]


def unpack_base3_codes(packed: Tensor, numel: int) -> Tensor:
    """Decode base-3, rejecting reserved bytes and nonzero unused trits."""
    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("numel must be a nonnegative integer")
    if not isinstance(packed, Tensor) or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed codes must be a flat uint8 tensor")
    if packed.numel() != (numel + 4) // 5:
        raise ValueError("packed byte count does not match numel")
    if bool((packed >= 243).any()):
        raise ValueError("reserved base-3 byte values 243 through 255 are invalid")
    digits = torch.stack(tuple((packed // power) % 3 for power in (1, 3, 9, 27, 81)), dim=1).flatten()
    if bool((digits[numel:] != 0).any()):
        raise ValueError("final-byte padding trits must be zero")
    digits = digits[:numel].to(torch.int8)
    return torch.where(digits == 2, -1, digits)


def _pack_bits(bits: Tensor) -> Tensor:
    bits = bits.to(torch.uint8)
    if bits.numel() % 8:
        bits = torch.cat((bits, bits.new_zeros(8 - bits.numel() % 8)))
    slots = bits.reshape(-1, 8)
    packed = bits.new_zeros(slots.shape[0])
    for shift in range(8):
        packed |= slots[:, shift] << shift
    return packed


def _unpack_bits(packed: Tensor, numel: int, name: str) -> Tensor:
    if not isinstance(packed, Tensor) or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError(f"{name} must be a flat uint8 tensor")
    if packed.numel() != (numel + 7) // 8:
        raise ValueError(f"{name} byte count does not match bit count")
    bits = torch.stack(tuple((packed >> shift) & 1 for shift in range(8)), dim=1).flatten()
    if bool((bits[numel:] != 0).any()):
        raise ValueError(f"{name} final-byte padding must be zero")
    return bits[:numel]


def pack_bitcos_codes(codes: Tensor) -> tuple[Tensor, Tensor]:
    """Pack row-major ternary codes as presence and compact negative-sign bits.

    Sign bits follow nonzero codes in flattened row-major order. Final unused
    bits in both streams are zero. Empty codes produce two empty uint8 tensors.
    """
    flat = _ternary_flat(codes)
    nonzero = flat != 0
    return _pack_bits(nonzero), _pack_bits(flat[nonzero] < 0)


def unpack_bitcos_codes(presence: Tensor, signs: Tensor, numel: int) -> Tensor:
    """Decode BITCOS, deriving the sign count from the validated presence bits."""
    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("numel must be a nonnegative integer")
    present = _unpack_bits(presence, numel, "presence")
    negative = _unpack_bits(signs, int(present.sum().item()), "signs")
    if signs.device != presence.device:
        raise ValueError("presence and signs must be on the same device")
    codes = torch.zeros(numel, dtype=torch.int8, device=presence.device)
    codes[present.bool()] = 1 - 2 * negative.to(torch.int8)
    return codes


def _shape_and_counts(shape: Any, group_size: Any) -> tuple[tuple[int, int], int, int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("shape must contain [out_features, in_features]")
    dims = tuple(_positive_int(v, "shape dimension") for v in shape)
    group_size = _positive_int(group_size, "group_size")
    numel = dims[0] * dims[1]
    num_groups = math.ceil(numel / group_size)
    return dims, numel, num_groups, num_groups * group_size


def pack_grouped_weight(
    codes: Tensor,
    scales: Tensor,
    *,
    shape: tuple[int, int] | list[int],
    group_size: int,
    bias: Tensor | None = None,
    packing: str = "2bit",
) -> dict[str, Any]:
    """Make a canonical CPU payload with per-group FP16 scales and FP16 bias.

    Codes may contain original elements only or include zero group padding.
    Casting scales/bias to FP16 is an additional, explicitly tested rounding step.
    ``auto`` minimizes actual code-stream bytes, breaking ties in the order
    2-bit, BITCOS, base-3 to preserve earlier choices. Scales/bias cost the same
    for each codec. Container overhead is not included at this low-level API.
    """
    if packing not in ("2bit", "bitcos", "base3", "auto"):
        raise ValueError("packing must be '2bit', 'bitcos', 'base3', or 'auto'")
    dims, numel, num_groups, padded_numel = _shape_and_counts(shape, group_size)
    flat = _ternary_flat(codes).cpu()
    if flat.numel() == numel:
        flat = torch.cat((flat, flat.new_zeros(padded_numel - numel)))
    elif flat.numel() != padded_numel:
        raise ValueError("code count must match original or padded weight count")
    if bool((flat[numel:] != 0).any()):
        raise ValueError("group padding must be zero")
    scales = scales.detach().reshape(-1).cpu()
    if scales.numel() != num_groups or not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("one finite nonnegative scale is required per group")
    half_scales = scales.to(torch.float16)
    active = (flat.reshape(num_groups, group_size) != 0).any(dim=1)
    if not bool(torch.isfinite(half_scales).all()) or bool(((half_scales == 0) & active & (scales != 0)).any()):
        raise ValueError("active group scales are not representable in FP16")
    half_bias = None
    if bias is not None:
        if bias.ndim != 1 or bias.numel() != dims[0]:
            raise ValueError("bias shape must match out_features")
        half_bias = bias.detach().cpu().to(torch.float16).contiguous()
        if not bool(torch.isfinite(half_bias).all()):
            raise ValueError("bias must be finite and representable in FP16")
    if packing == "auto":
        bitcos_bytes = (padded_numel + 7) // 8 + (int((flat != 0).sum().item()) + 7) // 8
        packing = min((("2bit", (padded_numel + 3) // 4), ("bitcos", bitcos_bytes),
                       ("base3", (padded_numel + 4) // 5)), key=lambda choice: choice[1])[0]
    state: dict[str, Any] = {
        "format": {"2bit": FORMAT, "bitcos": BITCOS_FORMAT, "base3": BASE3_FORMAT}[packing],
        "shape": list(dims),
        "group_size": group_size,
        "numel": numel,
        "padded_numel": padded_numel,
        "num_groups": num_groups,
        "scales": half_scales.contiguous(),
        "bias": half_bias,
    }
    if packing == "bitcos":
        state["presence"], state["signs"] = pack_bitcos_codes(flat)
        code_bytes = state["presence"].numel() + state["signs"].numel()
    else:
        pack_codes = pack_base3_codes if packing == "base3" else pack_ternary_codes
        state["codes"] = pack_codes(flat).contiguous()
        code_bytes = state["codes"].numel()
    scale_bytes = half_scales.numel() * half_scales.element_size()
    bias_bytes = 0 if half_bias is None else half_bias.numel() * half_bias.element_size()
    state["storage"] = {
        "code_bytes": code_bytes,
        "scale_bytes": scale_bytes,
        "bias_bytes": bias_bytes,
        "tensor_payload_bytes": code_bytes + scale_bytes + bias_bytes,
        "weight_payload_bits_per_weight": 8 * (code_bytes + scale_bytes) / numel,
        "includes_container_overhead": False,
    }
    if packing == "bitcos":
        state["storage"]["presence_bytes"] = state["presence"].numel()
        state["storage"]["sign_bytes"] = state["signs"].numel()
    return state


def _validated_grouped_codes(state: dict[str, Any]) -> Tensor:
    if not isinstance(state, dict) or state.get("format") not in (FORMAT, BITCOS_FORMAT, BASE3_FORMAT):
        raise ValueError(f"expected packed format {FORMAT}, {BITCOS_FORMAT} or {BASE3_FORMAT}")
    dims, numel, num_groups, padded_numel = _shape_and_counts(state.get("shape"), state.get("group_size"))
    for name, expected in (("numel", numel), ("num_groups", num_groups), ("padded_numel", padded_numel)):
        if state.get(name) != expected or type(state.get(name)) is not int:
            raise ValueError(f"inconsistent {name}")
    scales = state.get("scales")
    if not isinstance(scales, Tensor) or scales.dtype != torch.float16 or scales.shape != (num_groups,):
        raise ValueError("scales must be a flat FP16 tensor with one value per group")
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("scales must be finite and nonnegative")
    bias = state.get("bias")
    if bias is not None and (
        not isinstance(bias, Tensor) or bias.dtype != torch.float16 or bias.shape != (dims[0],)
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("bias must be a finite FP16 tensor matching out_features")
    if state["format"] in (FORMAT, BASE3_FORMAT):
        if "presence" in state or "signs" in state:
            name = "2-bit" if state["format"] == FORMAT else "base-3"
            raise ValueError(f"{name} format cannot contain BITCOS streams")
        unpack_codes = unpack_ternary_codes if state["format"] == FORMAT else unpack_base3_codes
        codes = unpack_codes(state.get("codes"), padded_numel)
    else:
        if "codes" in state:
            raise ValueError("BITCOS format cannot contain 2-bit codes")
        codes = unpack_bitcos_codes(state.get("presence"), state.get("signs"), padded_numel)
    if bool((codes[numel:] != 0).any()):
        raise ValueError("group padding must be zero")
    streams = ("presence", "signs") if state["format"] == BITCOS_FORMAT else ("codes",)
    allowed_tensors = {*streams, "scales", "bias"}
    if any(isinstance(value, Tensor) and key not in allowed_tensors for key, value in state.items()):
        raise ValueError("unexpected tensor in packed state")
    # Old hand-built states may omit accounting, but supplied accounting must
    # describe the actual tensors rather than a nominal number of bits.
    if "storage" in state:
        storage = state["storage"]
        if not isinstance(storage, dict):
            raise ValueError("storage must be an accounting dictionary")
        code_bytes = sum(state[key].numel() * state[key].element_size() for key in streams)
        scale_bytes = scales.numel() * scales.element_size()
        bias_bytes = 0 if bias is None else bias.numel() * bias.element_size()
        expected_bytes = {
            "code_bytes": code_bytes, "scale_bytes": scale_bytes, "bias_bytes": bias_bytes,
            "tensor_payload_bytes": code_bytes + scale_bytes + bias_bytes,
        }
        if state["format"] == BITCOS_FORMAT:
            expected_bytes.update(presence_bytes=state["presence"].numel(), sign_bytes=state["signs"].numel())
        for key, expected in expected_bytes.items():
            if type(storage.get(key)) is not int or storage[key] != expected:
                raise ValueError(f"inconsistent storage {key}")
        weight_bits = storage.get("weight_payload_bits_per_weight")
        if isinstance(weight_bits, bool) or not isinstance(weight_bits, (int, float)) or weight_bits != 8 * (code_bytes + scale_bytes) / numel:
            raise ValueError("inconsistent storage weight_payload_bits_per_weight")
        if storage.get("includes_container_overhead") is not False:
            raise ValueError("storage must exclude container overhead")
    return codes


def validate_packed_state(state: dict[str, Any]) -> None:
    """Validate all formats, including metadata, streams, and zero padding."""
    _validated_grouped_codes(state)


def unpack_grouped_codes(state: dict[str, Any]) -> Tensor:
    """Return validated flat int8 codes, including final group padding."""
    return _validated_grouped_codes(state)


def repack_grouped_weight(state: dict[str, Any], packing: str = "auto") -> dict[str, Any]:
    """Change only the code representation, preserving FP16 scales and bias.

    Returns a canonical CPU state. No weight reconstruction, fitting, or new
    rounding is performed: stored codes and existing FP16 factors are reused.
    """
    codes = unpack_grouped_codes(state)
    return pack_grouped_weight(
        codes, state["scales"], shape=state["shape"], group_size=state["group_size"],
        bias=state.get("bias"), packing=packing,
    )


def unpack_grouped_weight(
    state: dict[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Reconstruct a full dense reference weight; no low-bit matmul is supplied."""
    codes = unpack_grouped_codes(state)
    target_device = codes.device if device is None else device
    codes = codes.to(device=target_device, dtype=dtype)
    scales = state["scales"].to(device=target_device, dtype=dtype)
    grouped = codes.reshape(state["num_groups"], state["group_size"]) * scales[:, None]
    return grouped.flatten()[: state["numel"]].reshape(state["shape"])


def packed_state_nbytes(state: dict[str, Any]) -> int:
    """Exact bytes of code, scale, and bias tensors (not serialized file size)."""
    validate_packed_state(state)
    streams = ("presence", "signs") if state["format"] == BITCOS_FORMAT else ("codes",)
    return sum(state[key].numel() * state[key].element_size() for key in (*streams, "scales", "bias") if state.get(key) is not None)
