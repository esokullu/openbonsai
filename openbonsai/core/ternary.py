"""Ternary modules with trainable group scales (codes frozen).

``TernaryLinear`` and ``TernaryEmbedding`` are the modules of
research/experiments/e2e_kd.py, unchanged in arithmetic: BF16 signs, a shared
BF16 dense Hadamard block, and scales ``s0 * exp(log_mult)``. The forward is
what Prism's runtime computes: y = (C * s) (R x) with R = blockdiag(H) diag(signs)
in column-vector form (signs first, then H; core/hadamard.py writes the same
rotation for row vectors as x D H). The one addition: a Linear's bias, if it
has one, is kept as a frozen buffer and added (the scripts' Qwen3.5 has none).

``dense_weight`` is the experiments' ``apply_state`` arithmetic: it writes
the dequantized weight (C * s) R^T back into an ordinary Linear, which is
how the decoder-only and endpoint results were scored.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .hadamard import hadamard_matrix, rotation_matrix


class TernaryLinear(torch.nn.Module):
    """y = (C * s) (R x) + b with R = blockdiag(H) diag(signs); s = s0 * exp(u)."""

    def __init__(self, codes, scales, signs, block, h, bias=None):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.group = self.in_features // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h  # shared [block, block] BF16 Hadamard, not a parameter
        self.register_buffer("bias", None if bias is None else bias.detach().clone(), persistent=False)
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def weight_deq(self):
        s = self.scales().to(torch.bfloat16)
        c = self.codes.view(self.out_features, -1, self.group).to(torch.bfloat16)
        return (c * s[..., None]).view(self.out_features, self.in_features)

    def forward(self, x):
        shape = x.shape
        xr = (x * self.signs).view(*shape[:-1], self.in_features // self.block, self.block) @ self.h
        return F.linear(xr.view(shape), self.weight_deq(), self.bias)


class TernaryEmbedding(torch.nn.Module):
    """e = (C[ids] * s[ids]) R^T, i.e. stored rows live in the rotated basis."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.num, self.dim = codes.shape
        self.group = self.dim // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def forward(self, ids):
        c = self.codes[ids].view(*ids.shape, -1, self.group).to(torch.bfloat16)
        s = (self.s0[ids] * torch.exp(self.log_mult[ids])).to(torch.bfloat16)
        rows = (c * s[..., None]).view(*ids.shape, self.dim // self.block, self.block)
        return (rows @ self.h).view(*ids.shape, self.dim) * self.signs


def shared_hadamard(block: int, device) -> Tensor:
    """The BF16 Hadamard block shared by every ternary module."""
    return hadamard_matrix(block, device).to(torch.bfloat16)


@torch.no_grad()
def dense_weight(codes: Tensor, scales: Tensor, signs: Tensor, block: int, *, device, dtype) -> Tensor:
    """(C * s) R^T in ``dtype``: the experiments' ``apply_state`` arithmetic."""
    codes = codes.to(device).float()
    scales = scales.to(device).float()
    out_f, in_f = codes.shape
    rot = rotation_matrix(signs, block, codes.device)
    deq = (codes.view(out_f, -1, in_f // scales.shape[1]) * scales[..., None]).view(out_f, in_f)
    return (deq @ rot.T).to(dtype)
