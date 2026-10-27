"""Local normalized sign/Hadamard transforms, not a quantization algorithm.

Row vectors use R = D H. Forward: signs then H; inverse: H then signs.
FP16/BF16 butterflies accumulate in FP32 and cast back once. Rounding remains
possible and is separate from the algebraic full-precision identity.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_rotation(signs: Tensor, block_size: int, width: int):
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if type(width) is not int or width <= 0 or width % block_size:
        raise ValueError("input width must be a positive multiple of block_size")
    if not isinstance(signs, Tensor) or signs.ndim != 1 or signs.numel() != width:
        raise ValueError("one rotation sign is required per input coordinate")
    if signs.is_complex() or not bool(((signs == -1) | (signs == 1)).all()):
        raise ValueError("rotation signs must be exactly -1 or +1")


def normalized_hadamard(x: Tensor, block_size: int) -> Tensor:
    """Differentiable block FWHT with orthonormal scaling and bounded scratch."""
    if not isinstance(x, Tensor) or x.ndim == 0 or not x.is_floating_point():
        raise ValueError("Hadamard input must be a floating tensor with a last axis")
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if x.shape[-1] == 0 or x.shape[-1] % block_size:
        raise ValueError("input width must be a positive multiple of block_size")
    original_shape, original_dtype = x.shape, x.dtype
    if x.numel() == 0:
        return x.clone()
    y = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y = y.reshape(-1, block_size)
    width = 1
    while width < block_size:
        groups = y.reshape(-1, block_size // (2 * width), 2, width)
        left, right = groups[..., 0, :], groups[..., 1, :]
        y = torch.stack((left + right, left - right), dim=-2).reshape(-1, block_size)
        width *= 2
    return (y / math.sqrt(block_size)).reshape(original_shape).to(original_dtype)


def rotate(x: Tensor, signs: Tensor, block_size: int) -> Tensor:
    """Apply row-vector R = D H (signs then normalized Hadamard)."""
    if not isinstance(x, Tensor) or x.ndim == 0:
        raise ValueError("rotation input must have a last axis")
    validate_rotation(signs, block_size, x.shape[-1])
    return normalized_hadamard(x * signs.to(device=x.device, dtype=x.dtype), block_size)


def inverse_rotate(x: Tensor, signs: Tensor, block_size: int) -> Tensor:
    """Apply row-vector R^T = H D (normalized Hadamard then signs)."""
    if not isinstance(x, Tensor) or x.ndim == 0:
        raise ValueError("rotation input must have a last axis")
    validate_rotation(signs, block_size, x.shape[-1])
    return normalized_hadamard(x, block_size) * signs.to(device=x.device, dtype=x.dtype)
