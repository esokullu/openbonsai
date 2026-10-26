"""Experimental module-local Hadamard rotations around CAT-Q ternarization.

No transform is commuted through a nonlinearity, gate, recurrent state or norm.
The original BF16 buffer is retained once; rotated FP32 weights are recomputed
on demand. This reference path needs a CUDA memory preflight before 9B use.
It is an independent calibration experiment, not Prism's training recipe.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .adapter import _PROJECTIONS, _RECURRENT_GATES, _text_module
from .hadamard import rotate, inverse_rotate
from .mixed_formats import ternary_candidate, with_rotation
from .quantizer import TernaryLinear


def fixed_signs(width, *, seed, name=''):
    """Explicit reproducible signs from SHA256, independent of Torch RNG version."""
    if type(width) is not int or width <= 0 or type(seed) is not int or not isinstance(name, str):
        raise ValueError('Invalid sign generation parameters')
    bits = bytearray()
    for counter in range((width + 255) // 256):
        bits.extend(hashlib.sha256(f'OpenBonsai-DH-v1:{seed}:{name}:{counter}'.encode()).digest())
    return torch.tensor([1 if (bits[i // 8] >> (i % 8)) & 1 else -1 for i in range(width)], dtype=torch.int8)


class RotatedTernaryLinear(TernaryLinear):
    """x -> xR -> ternary(W R)^T; teacher bypass uses original W exactly.

    R=D H acts on each input-width block. LoRA is learned in the rotated
    coordinate system before ternarization. ``rotation_only`` is a diagnostic
    floating-point transform path, separate from the exact original bypass.
    """
    def __init__(self, linear, group_size=128, *, signs, block_size=4096, **kwargs):
        # Validate dimensions before allocating the quantizer state.
        rotate(torch.zeros(1, linear.in_features), signs, block_size)
        super().__init__(linear, group_size, **kwargs)
        self.register_buffer('rotation_signs', signs.detach().to(device=self.weight.device, dtype=torch.int8).clone())
        self.block_size = block_size
        self.rotation_only = False
        with torch.no_grad():
            mean, scale = self._statistics(self._group(rotate(self.weight.float(), self.rotation_signs, block_size)))
            self.mu0.copy_(mean)
            self.alpha0.copy_(scale)

    def _effective_grouped(self, lora_A, lora_B):
        weight = rotate(self.weight.float(), self.rotation_signs, self.block_size)
        if self.lora_rank:
            weight = weight + (lora_B @ lora_A) / self.lora_rank
        return self._group(weight)

    def forward(self, x):
        if not self.quant_enabled or self.progress == 0:
            return F.linear(x, self.weight, self.bias)
        if self.rotation_only:
            # FP32 transforms/linear isolate mathematical placement from BF16
            # activation/storage rounding; output still uses native dtype.
            work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            return F.linear(rotate(x.to(work_dtype), self.rotation_signs, self.block_size),
                            rotate(self.weight.to(work_dtype), self.rotation_signs, self.block_size),
                            None if self.bias is None else self.bias.to(work_dtype)).to(x.dtype)
        return F.linear(rotate(x, self.rotation_signs, self.block_size), self.quantized_weight(), self.bias)

    def mixed_candidate(self, *, packing='auto', provenance=None):
        candidate = ternary_candidate(super().packed_state(packing='2bit'), packing=packing,
                                      provenance=provenance)
        return with_rotation(candidate, self.rotation_signs, self.block_size)

    def packed_state(self, **kwargs):
        raise ValueError('Rotated weights require mixed_candidate/save_mixed; legacy export would drop the rotation')


class RotatedTernaryEmbedding(RotatedTernaryLinear):
    """Experimental endpoint: quantize E R, gather, then apply inverse R.

    The decoder-window trainer does not train this endpoint. Explicit endpoint
    calibration is required. Dense quantizer reconstruction is a small-model
    reference; the exported MixedEmbedding gathers only requested rows.
    """
    def __init__(self, embedding, group_size=128, **kwargs):
        if not isinstance(embedding, nn.Embedding) or embedding.max_norm is not None:
            raise ValueError('Only an original plain nn.Embedding is supported')
        # No randomly initialized duplicate weight allocation.
        linear = nn.Linear(embedding.embedding_dim, embedding.num_embeddings, bias=False, device='meta')
        linear.weight = nn.Parameter(embedding.weight.detach(), requires_grad=False)
        super().__init__(linear, group_size, **kwargs)
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.calibration_scope = 'endpoint'

    def forward(self, ids):
        if not self.quant_enabled or self.progress == 0:
            return F.embedding(ids, self.weight, self.padding_idx)
        if self.rotation_only:
            dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
            rows = F.embedding(ids, rotate(self.weight.to(dtype), self.rotation_signs, self.block_size), self.padding_idx)
            return inverse_rotate(rows, self.rotation_signs, self.block_size).to(self.weight.dtype)
        rows = F.embedding(ids, self.quantized_weight(), self.padding_idx)
        return inverse_rotate(rows, self.rotation_signs, self.block_size)

    def mixed_candidate(self, *, packing='auto', provenance=None):
        candidate = ternary_candidate(TernaryLinear.packed_state(self, packing='2bit'), module_kind='embedding',
                                      packing=packing, provenance=provenance)
        return with_rotation(candidate, self.rotation_signs, self.block_size)


def rotation_targets(model, *, endpoints=False, aggressive_gates=False):
    """Explicit native text coverage; vision/convs/norms/dynamics stay unchanged."""
    prefix, text_model = _text_module(model)
    allowlist = _PROJECTIONS | (_RECURRENT_GATES if aggressive_gates else set())
    names = [f'{prefix}.layers.{i}.{relative}' for i, layer in enumerate(text_model.layers)
             for relative, module in layer.named_modules() if relative in allowlist]
    if endpoints:
        names += [f'{prefix}.embed_tokens', 'lm_head']
    return names


def replace_rotated_projections(model, *, group_size=128, block_size=4096, seed=5090,
                                 lora_rank=64, factor_mapping='sigmoid', endpoints=False,
                                 aggressive_gates=False):
    """Install reference quantizers with exact original teacher bypass.

    ``endpoints=True`` is for an explicit separate endpoint-calibration stage or
    an invariance test, not a request to silently treat them as decoder windows.
    All input widths must be divisible by the specified power-of-two block.
    """
    names = rotation_targets(model, endpoints=endpoints, aggressive_gates=aggressive_gates)
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            raise ValueError(f'Rotation needs original native weights, got {type(module).__name__}: {name}')
        if module.weight.shape[1] % block_size:
            raise ValueError(f'Rotation block does not divide input width: {name}')
    for name in names:
        module = model.get_submodule(name)
        cls = RotatedTernaryEmbedding if isinstance(module, nn.Embedding) else RotatedTernaryLinear
        replacement = cls(module, group_size, signs=fixed_signs(module.weight.shape[1], seed=seed, name=name),
                          block_size=block_size, lora_rank=lora_rank, factor_mapping=factor_mapping)
        if name == 'lm_head':
            replacement.calibration_scope = 'endpoint'
        model.set_submodule(name, replacement)
    return names
