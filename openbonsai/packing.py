"""Portable 2-bit ternary payloads; this is not a 1.58-bit entropy codec.

Four codes occupy a byte, least significant code first: 0 -> 00,
+1 -> 01, -1 -> 10. 11 is invalid. Groups and final bytes are zero padded.
The reported bytes count tensor payloads, excluding container/metadata overhead.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

# Stable on-disk identifier: preserve compatibility with pre-OpenBonsai artifacts.
FORMAT = "scaleq.ternary-grouped.v1"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def pack_ternary_codes(codes: Tensor) -> Tensor:
    """Pack a tensor of {-1, 0, +1} values into a flat uint8 tensor."""
    if not isinstance(codes, Tensor) or codes.is_complex():
        raise ValueError("codes must be a real tensor")
    flat = codes.detach().reshape(-1)
    if not bool(((flat == -1) | (flat == 0) | (flat == 1)).all()):
        raise ValueError("ternary codes must be -1, 0, or +1")
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
) -> dict[str, Any]:
    """Make a canonical CPU payload with per-group FP16 scales and FP16 bias.

    Codes may contain original elements only or include zero group padding.
    Casting scales/bias to FP16 is an additional, explicitly tested rounding step.
    """
    dims, numel, num_groups, padded_numel = _shape_and_counts(shape, group_size)
    flat = codes.detach().reshape(-1).cpu()
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
    state: dict[str, Any] = {
        "format": FORMAT,
        "shape": list(dims),
        "group_size": group_size,
        "numel": numel,
        "padded_numel": padded_numel,
        "num_groups": num_groups,
        "codes": pack_ternary_codes(flat).contiguous(),
        "scales": half_scales.contiguous(),
        "bias": half_bias,
    }
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
    return state


def validate_packed_state(state: dict[str, Any]) -> None:
    """Validate metadata and payload before allocating a reconstructed weight."""
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError(f"expected packed format {FORMAT}")
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
    codes = unpack_ternary_codes(state.get("codes"), padded_numel)
    if bool((codes[numel:] != 0).any()):
        raise ValueError("group padding must be zero")


def unpack_grouped_weight(
    state: dict[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Reconstruct a full dense reference weight; no low-bit matmul is supplied."""
    validate_packed_state(state)
    target_device = state["codes"].device if device is None else device
    codes = unpack_ternary_codes(state["codes"], state["padded_numel"]).to(device=target_device, dtype=dtype)
    scales = state["scales"].to(device=target_device, dtype=dtype)
    grouped = codes.reshape(state["num_groups"], state["group_size"]) * scales[:, None]
    return grouped.flatten()[: state["numel"]].reshape(state["shape"])


def packed_state_nbytes(state: dict[str, Any]) -> int:
    """Exact bytes of code, scale, and bias tensors (not serialized file size)."""
    validate_packed_state(state)
    return sum(state[key].numel() * state[key].element_size() for key in ("codes", "scales", "bias") if state[key] is not None)
