"""Prism PTQ1_0 block codec (encode and decode), numpy only.

Layout checked against Prism llama.cpp c9c34bc (ggml-common.h, ggml-quants.c):

- type id 143; one block = 128 consecutive rotated input weights of one row
  in 28 bytes: ``uint8 qs[24]; uint8 qh[2]; fp16 d`` (little-endian);
- digit = code + 1 in {0, 1, 2}; a byte packs base-3 digits most significant
  first, q = sum(d_n * 3**(4 - n)), and stores ceil(q * 256 / 243);
- ``qs[m]``, m < 16, holds weights m + 16n (n = 0..4);
  ``qs[16 + m]``, m < 8, holds weights 80 + m + 8n;
  ``qh[h]``, h < 2, holds four weights 120 + h + 2m (m = 0..3), q *= 3 first;
- decode digit n of byte b: ((uint8)(b * 3**n) * 3) >> 8; weight = (digit - 1) * d.

An all-zero group encodes as qs = 128, qh = 127. The encoder keeps the caller's
scale for such a group (Prism's own quantizer writes d = 0 there). Negative
scales are normalized by negating the group's codes; -0.0 becomes +0.0.
GGUF writing (P2) builds on this codec.
"""
from __future__ import annotations

import numpy as np

GGML_TYPE_PTQ1_0 = 143
LLAMA_FTYPE_MOSTLY_PTQ1_0 = 143
QK = 128
BLOCK_BYTES = 28

_POW5 = np.array([81, 27, 9, 3, 1], dtype=np.int32)
_POW4 = np.array([27, 9, 3, 1], dtype=np.int32)
_POW3_DEC = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
# C assigns the multiplication to uint8 BEFORE multiplying by three.
DECODE_LUT = ((((np.arange(256, dtype=np.uint16)[:, None] * _POW3_DEC) & 255) * 3) >> 8).astype(np.int8) - 1


def _as_numpy(x, dtype=None):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None and x.dtype != dtype else x


def normalize_scales(codes, scales):
    """Return (codes int8, scales fp16) with non-negative scales and no -0.0."""
    codes = _as_numpy(codes).astype(np.int8, copy=True)
    scales = _as_numpy(scales).astype(np.float16, copy=True)
    if not np.isfinite(scales).all():
        raise ValueError("PTQ1_0 scales must be finite")
    rows, cols = codes.shape
    neg = np.signbit(scales)
    if neg.any():
        flip = np.repeat(neg, QK, axis=1)
        codes[flip] = -codes[flip]
        scales = np.abs(scales)
    return codes, (scales + np.float16(0)).astype(np.float16)


def encode(codes, scales) -> np.ndarray:
    """codes int8 {-1,0,1} [rows, cols], scales fp16 [rows, cols/128] -> uint8 [rows, cols/128*28]."""
    codes = _as_numpy(codes)
    scales = _as_numpy(scales)
    if codes.ndim != 2 or codes.shape[1] % QK:
        raise ValueError("PTQ1_0 needs 2-D codes whose width is a multiple of 128")
    rows, cols = codes.shape
    if scales.shape != (rows, cols // QK):
        raise ValueError(f"scales must be [rows, cols/128] = {(rows, cols // QK)}, got {scales.shape}")
    if not np.isin(codes, (-1, 0, 1)).all():
        raise ValueError("PTQ1_0 codes must be -1, 0 or +1")
    codes, scales = normalize_scales(codes, scales)
    d = codes.reshape(-1, QK).astype(np.int32) + 1
    out = np.empty((d.shape[0], BLOCK_BYTES), np.uint8)
    out[:, 0:16] = (np.einsum("bnm,n->bm", d[:, :80].reshape(-1, 5, 16), _POW5) * 256 + 242) // 243
    out[:, 16:24] = (np.einsum("bnm,n->bm", d[:, 80:120].reshape(-1, 5, 8), _POW5) * 256 + 242) // 243
    out[:, 24:26] = (np.einsum("bmh,m->bh", d[:, 120:].reshape(-1, 4, 2), _POW4) * 3 * 256 + 242) // 243
    out[:, 26:28] = scales.astype("<f2").reshape(-1, 1).view(np.uint8)
    return out.reshape(rows, cols // QK * BLOCK_BYTES)


def decode(blocks, cols: int | None = None):
    """uint8 [rows, nb*28] (or [nb, 28]) -> (codes int8 [rows, nb*128], scales fp16 [rows, nb])."""
    blocks = _as_numpy(blocks, np.uint8)
    if blocks.ndim == 1:
        blocks = blocks.reshape(1, -1)
    rows = blocks.shape[0]
    if blocks.shape[1] % BLOCK_BYTES:
        raise ValueError("PTQ1_0 rows must be a whole number of 28-byte blocks")
    nb = blocks.shape[1] // BLOCK_BYTES
    if cols is not None and cols != nb * QK:
        raise ValueError(f"expected {cols // QK} blocks per row, found {nb}")
    b = blocks.reshape(-1, BLOCK_BYTES)
    codes = np.concatenate((DECODE_LUT[b[:, :16]].transpose(0, 2, 1).reshape(-1, 80),
                            DECODE_LUT[b[:, 16:24]].transpose(0, 2, 1).reshape(-1, 40),
                            DECODE_LUT[b[:, 24:26], :4].transpose(0, 2, 1).reshape(-1, 8)), axis=1)
    scales = np.ascontiguousarray(b[:, 26:28]).view("<f2").reshape(rows, nb).astype(np.float16)
    return codes.reshape(rows, nb * QK), scales


def dequantize(blocks) -> np.ndarray:
    """FP32 weights codes * d, exactly what ``dequantize_row_ptq1_0`` returns."""
    codes, scales = decode(blocks)
    rows, cols = codes.shape
    return (codes.reshape(rows, -1, QK).astype(np.float32)
            * scales.astype(np.float32)[..., None]).reshape(rows, cols)


def nbytes(rows: int, cols: int) -> int:
    if cols % QK:
        raise ValueError("width must be a multiple of 128")
    return rows * cols // QK * BLOCK_BYTES
