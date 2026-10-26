"""Selected-row endpoint quantization around an immutable BF16 embedding.

This is a calibration/reference operator, not a low-bit inference kernel.
The original table shares frozen storage with its source. Only selected rows
are promoted, rotated and adapted; optimizer state remains dense compact state.
Complete-vocabulary fitting and export eligibility are separate trainer duties.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .endpoint_quantizer import BoundedTernaryHead
from .hadamard import inverse_rotate, rotate


def _quantizer_property(name, *, writable=False):
    def get(self):
        return getattr(self.quantizer, name)
    def set_value(self, value):
        setattr(self.quantizer, name, value)
    return property(get, set_value if writable else None)


class BoundedTernaryEmbedding(nn.Module):
    """Quantize E R in bounded row selections, gather, then apply inverse R.

    ``forward`` preserves token multiplicity and native padding-gradient
    semantics. ``reconstruct_rows`` is a table-compression objective: it fits
    every explicitly requested row, including a padding row. Neither method
    claims that unrequested vocabulary rows have been calibrated.
    """

    module_kind = 'embedding'
    weight = _quantizer_property('weight')
    bias = _quantizer_property('bias')
    in_features = _quantizer_property('in_features')
    out_features = _quantizer_property('out_features')
    group_size = _quantizer_property('group_size')
    row_chunk_size = _quantizer_property('row_chunk_size')
    rotation_signs = _quantizer_property('rotation_signs')
    block_size = _quantizer_property('block_size')
    source_weight_sha256 = _quantizer_property('source_weight_sha256')
    source_bias_sha256 = _quantizer_property('source_bias_sha256')
    lora_rank = _quantizer_property('lora_rank')
    factor_mapping = _quantizer_property('factor_mapping')
    hard_phase = _quantizer_property('hard_phase')
    soft_fraction = _quantizer_property('soft_fraction')
    max_sharpness = _quantizer_property('max_sharpness')
    progress = _quantizer_property('progress', writable=True)
    quant_enabled = _quantizer_property('quant_enabled', writable=True)
    rotation_only = _quantizer_property('rotation_only', writable=True)

    def __init__(self, source: nn.Embedding, group_size=128, *, row_chunk_size=64,
                 signs=None, block_size=4096, lora_rank=64, factor_mapping='sigmoid',
                 hard_phase='frozen_jacobian', soft_fraction=.8, max_sharpness=30.):
        super().__init__()
        if not isinstance(source, nn.Embedding):
            raise TypeError('BoundedTernaryEmbedding requires nn.Embedding')
        if source.max_norm is not None or source.scale_grad_by_freq or source.sparse:
            raise ValueError('Only plain embeddings without max_norm, scale_grad_by_freq or sparse gradients are supported')
        # A meta shell avoids any randomly initialized duplicate source table.
        linear = nn.Linear(source.embedding_dim, source.num_embeddings, bias=False, device='meta')
        linear.weight = nn.Parameter(source.weight.detach(), requires_grad=False)
        self.quantizer = BoundedTernaryHead(
            linear, group_size, row_chunk_size=row_chunk_size, signs=signs,
            block_size=block_size, lora_rank=lora_rank, factor_mapping=factor_mapping,
            hard_phase=hard_phase, soft_fraction=soft_fraction, max_sharpness=max_sharpness)
        self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim
        self.padding_idx = source.padding_idx
        self.calibration_scope = 'endpoint'

    def factor_parameters(self):
        return self.quantizer.factor_parameters()

    def lora_parameters(self):
        return self.quantizer.lora_parameters()

    def trainable_parameters(self):
        return self.quantizer.trainable_parameters()

    def begin_step(self, progress):
        self.quantizer.begin_step(progress)

    def clear_soft_snapshot(self):
        self.quantizer.clear_soft_snapshot()

    def validate_parameters(self):
        self.quantizer.validate_parameters()

    def assert_source_unchanged(self, *, verify_hash=False):
        self.quantizer.assert_source_unchanged(verify_hash=verify_hash)

    def _check_ids(self, ids, *, rows=False):
        self.assert_source_unchanged()
        if not isinstance(ids, Tensor) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('embedding IDs must be int32 or int64')
        if ids.device != self.weight.device:
            raise ValueError('embedding IDs must have the source device')
        if bool(((ids < 0) | (ids >= self.num_embeddings)).any()):
            raise IndexError('embedding ID outside vocabulary')
        if rows and (ids.ndim != 1 or ids.numel() > self.row_chunk_size
                     or torch.unique(ids).numel() != ids.numel()):
            raise ValueError('row IDs must be one-dimensional, unique and no larger than row_chunk_size')
        return ids.to(torch.int64)

    def teacher_rows(self, ids):
        """Return a bounded unique selection of unchanged original BF16 rows."""
        ids = self._check_ids(ids, rows=True)
        return F.embedding(ids, self.weight, self.padding_idx)

    def _empty(self, shape):
        result = self.weight.new_empty((*shape, self.embedding_dim))
        if (torch.is_grad_enabled() and self.quant_enabled and self.progress > 0
                and not self.rotation_only):
            # Empty token sets contribute zero gradients without reading the
            # whole parameter arrays or silently breaking a caller's graph.
            anchor = sum(p.reshape(-1)[:0].sum() for p in self.trainable_parameters())
            result = result + anchor.to(result.dtype)
        return result

    def _float_rows(self, ids):
        original = F.embedding(ids, self.weight, self.padding_idx).float()
        if self.rotation_signs is not None:
            original = inverse_rotate(rotate(original, self.rotation_signs, self.block_size),
                                      self.rotation_signs, self.block_size)
        return original.to(self.weight.dtype)

    def reconstruct_rows(self, ids):
        """Differentiable table-fit rows in original coordinates, including pad.

        IDs must be unique and fit one row chunk. Table fitting deliberately
        does not suppress padding-row gradients; native token forward does.
        """
        ids = self._check_ids(ids, rows=True)
        if not self.quant_enabled or self.progress == 0:
            return F.embedding(ids, self.weight, self.padding_idx)
        if ids.numel() == 0:
            return self._empty(ids.shape)
        if self.rotation_only:
            return self._float_rows(ids)
        rows = self.quantizer._quantized_rows(0, ids.numel(), row_ids=ids)
        return rows if self.rotation_signs is None else inverse_rotate(rows, self.rotation_signs, self.block_size)

    def hard_codes_rows(self, start, end):
        """Current latent codes/scales; independent of saved gradient snapshots."""
        return self.quantizer.hard_codes_rows(start, end)

    def forward(self, ids):
        ids = self._check_ids(ids)
        if not self.quant_enabled or self.progress == 0:
            return F.embedding(ids, self.weight, self.padding_idx)
        if ids.numel() == 0:
            return self._empty(ids.shape)
        unique, reverse = torch.unique(ids.reshape(-1), sorted=True, return_inverse=True)
        if self.rotation_only:
            rows = torch.cat([self._float_rows(chunk) for chunk in unique.split(self.row_chunk_size)])
            return F.embedding(reverse, rows).reshape(*ids.shape, self.embedding_dim)
        rows = torch.cat([self.quantizer._quantized_rows(0, chunk.numel(), row_ids=chunk)
                          for chunk in unique.split(self.row_chunk_size)])
        local_padding = None
        if self.padding_idx is not None:
            location = (unique == self.padding_idx).nonzero().flatten()
            if location.numel():
                local_padding = int(location[0])
        # F.embedding's duplicate accumulation and padding mask are part of the
        # BF16 training contract. Inverse rotation follows occurrence restore.
        result = F.embedding(reverse, rows, padding_idx=local_padding).reshape(*ids.shape, self.embedding_dim)
        return result if self.rotation_signs is None else inverse_rotate(result, self.rotation_signs, self.block_size)
