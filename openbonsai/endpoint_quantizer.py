"""Bounded-row CAT-Q head quantizer around immutable original BF16 weights.

This is a correctness/calibration operator, not an accelerated inference head.
The source weight/bias are shared frozen buffers: their owning nn.Linear must
not be modified. Statistics, factors and snapshots are compact group/LoRA
state; FP32 weights, rotations, BA products and codes exist only for at most
row_chunk_size complete rows. No implicit full-head forward is provided.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hadamard import rotate, validate_rotation
from .quantizer import hard_ternary, soft_ternary


class BoundedTernaryHead(nn.Module):
    def __init__(self, source: nn.Linear, group_size=128, *, row_chunk_size=64,
                 signs=None, block_size=4096, lora_rank=64, factor_mapping="sigmoid",
                 hard_phase="frozen_jacobian", soft_fraction=.8, max_sharpness=30.):
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError("BoundedTernaryHead requires nn.Linear")
        if source.weight.dtype != torch.bfloat16 or source.weight.device.type == "meta":
            raise ValueError("source must contain original materialized BF16 weights")
        if source.bias is not None and source.bias.dtype != torch.bfloat16:
            raise ValueError("source bias must be BF16")
        for name, value in (("group_size", group_size), ("row_chunk_size", row_chunk_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(lora_rank) is not int or lora_rank < 0:
            raise ValueError("lora_rank must be a nonnegative integer")
        if source.in_features <= 0 or source.out_features <= 0 or source.in_features % group_size:
            raise ValueError("positive input width must be divisible by group_size")
        if not math.isfinite(soft_fraction) or not 0 < soft_fraction < 1:
            raise ValueError("soft_fraction must be finite and in (0,1)")
        if not math.isfinite(max_sharpness) or max_sharpness <= 0:
            raise ValueError("max_sharpness must be finite and positive")
        if factor_mapping not in ("exp", "sigmoid"):
            raise ValueError("factor_mapping must be exp or sigmoid")
        if hard_phase not in ("frozen_jacobian", "soft_surrogate"):
            raise ValueError("unsupported hard_phase")
        if signs is not None:
            validate_rotation(signs, block_size, source.in_features)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.group_size, self.row_chunk_size = group_size, row_chunk_size
        self.groups_per_row = self.in_features // group_size
        self.numel = self.in_features * self.out_features
        self.num_groups = self.out_features * self.groups_per_row
        self.padded_numel = self.numel
        self.lora_rank, self.factor_mapping, self.hard_phase = lora_rank, factor_mapping, hard_phase
        self.soft_fraction, self.max_sharpness = float(soft_fraction), float(max_sharpness)
        self.block_size = block_size
        self.progress, self.quant_enabled, self.rotation_only = 0., True, False
        self.calibration_scope = "endpoint"
        self.register_buffer("weight", source.weight.detach())
        self.register_buffer("bias", None if source.bias is None else source.bias.detach())
        self.register_buffer("rotation_signs", None if signs is None else signs.detach().to(
            device=source.weight.device, dtype=torch.int8).clone())
        self.register_buffer("_device_probe", torch.empty(0, device=source.weight.device), persistent=False)
        if lora_rank:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, self.in_features,
                                                  device=self.weight.device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_rank,
                                                  device=self.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.register_buffer("mu0", torch.empty(self.num_groups, 1, device=self.weight.device, dtype=torch.float32))
        self.register_buffer("alpha0", torch.empty_like(self.mu0))
        self.raw_mu = nn.Parameter(torch.zeros_like(self.mu0))
        self.raw_scale = nn.Parameter(torch.zeros_like(self.mu0))
        self.raw_threshold = nn.Parameter(torch.zeros_like(self.mu0))
        for name in self._snapshot_names():
            self.register_buffer(name, None, persistent=False)
        self._hard_snapshot_sharpness = None
        self._step_number = 0
        self._snapshot_step = None
        # Validate/hash and initialize only complete row chunks. The original
        # storage is never cloned or promoted as an entire matrix.
        digest = hashlib.sha256()
        with torch.no_grad():
            for start in range(0, self.out_features, self.row_chunk_size):
                end = min(start + self.row_chunk_size, self.out_features)
                original = self.weight[start:end]
                if not bool(torch.isfinite(original).all()):
                    raise ValueError("source weights must be finite")
                digest.update(original.cpu().contiguous().view(torch.uint8).numpy().tobytes())
                grouped = self._base_rows(start, end).reshape(-1, self.group_size)
                mean, alpha = self._statistics(grouped)
                sl = self._group_slice(start, end)
                self.mu0[sl].copy_(mean)
                self.alpha0[sl].copy_(alpha)
        self.source_weight_sha256 = digest.hexdigest()
        if self.bias is not None and not bool(torch.isfinite(self.bias).all()):
            raise ValueError("source bias must be finite")
        self.source_bias_sha256 = self._bias_digest()
        self._remember_source_versions()

    @staticmethod
    def _snapshot_names():
        return ("_hard_snapshot_mu", "_hard_snapshot_scale", "_hard_snapshot_threshold",
                "_hard_snapshot_lora_A", "_hard_snapshot_lora_B",
                "_hard_snapshot_mu0", "_hard_snapshot_alpha0")

    def factor_parameters(self):
        return (self.raw_mu, self.raw_scale, self.raw_threshold)

    def lora_parameters(self):
        return (self.lora_A, self.lora_B) if self.lora_rank else ()

    def trainable_parameters(self):
        return self.factor_parameters() + self.lora_parameters()

    def _remember_source_versions(self):
        self._source_weight_version = self.weight._version
        self._source_bias_version = None if self.bias is None else self.bias._version

    def _bias_digest(self):
        if self.bias is None:
            return None
        return hashlib.sha256(self.bias.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def assert_source_unchanged(self, *, verify_hash=False):
        if (self.weight._version != self._source_weight_version
                or (None if self.bias is None else self.bias._version) != self._source_bias_version):
            raise RuntimeError("original source was modified after head construction")
        if verify_hash:
            digest = hashlib.sha256()
            for start in range(0, self.out_features, self.row_chunk_size):
                original = self.weight[start:start + self.row_chunk_size]
                digest.update(original.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            if digest.hexdigest() != self.source_weight_sha256 or self._bias_digest() != self.source_bias_sha256:
                raise RuntimeError("original source hash changed after head construction")

    def validate_parameters(self):
        self.assert_source_unchanged()
        for parameter in self.trainable_parameters():
            # The state is compact, but keep finite-check scratch bounded too.
            for chunk in parameter.detach().reshape(-1).split(1048576):
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError("nonfinite endpoint quantizer parameter")

    def _apply(self, fn, recurse=True):
        self.assert_source_unchanged()
        parameters = {name: (p.detach(), None if p.grad is None else p.grad.detach())
                      for name, p in self.named_parameters(recurse=False)}
        buffers = {name: self._buffers[name] for name in ("mu0", "alpha0", *self._snapshot_names())
                   if self._buffers[name] is not None}
        # Do not let module.float()/half() allocate a full converted source.
        original_weight, original_bias = self.weight, self.bias
        self._buffers["weight"], self._buffers["bias"] = None, None
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._buffers["weight"], self._buffers["bias"] = original_weight, original_bias
        device = self._device_probe.device
        self.weight = original_weight.to(device=device, dtype=original_weight.dtype)
        self.bias = None if original_bias is None else original_bias.to(device=device, dtype=original_bias.dtype)
        for name, (value, grad) in parameters.items():
            parameter = getattr(self, name)
            parameter.data = value.to(device=parameter.device, dtype=torch.float32)
            if grad is not None:
                parameter.grad = grad.to(device=parameter.device, dtype=torch.float32)
        for name, value in buffers.items():
            setattr(self, name, value.to(device=device, dtype=torch.float32))
        self._remember_source_versions()
        return result

    def _check_rows(self, start, end):
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= self.out_features
                or end - start > self.row_chunk_size):
            raise ValueError("row range must be nonempty, in bounds and no larger than row_chunk_size")
        self.assert_source_unchanged()

    def _group_slice(self, start, end):
        return slice(start * self.groups_per_row, end * self.groups_per_row)

    def _base_rows(self, start, end):
        value = self.weight[start:end].float()
        return value if self.rotation_signs is None else rotate(value, self.rotation_signs, self.block_size)

    def _statistics(self, grouped):
        mean = grouped.sum(dim=1, keepdim=True) / self.group_size
        alpha = (grouped - mean).abs().mean(dim=1, keepdim=True)
        return mean, alpha + 1e-6 if self.factor_mapping == "sigmoid" else alpha.clamp_min(1e-8)

    def _parameter_rows(self, start, end, *, snapshot=False):
        sl = self._group_slice(start, end)
        current = (*self.factor_parameters(), self.lora_A, self.lora_B)
        values = []
        for index, parameter in enumerate(current):
            if parameter is None:
                values.append(None)
                continue
            view = parameter[sl] if index < 3 else (parameter if index == 3 else parameter[start:end])
            if snapshot:
                saved = getattr(self, self._snapshot_names()[index])
                saved = saved[sl] if index < 3 else (saved if index == 3 else saved[start:end])
                view = saved + (view - view.detach())
            values.append(view)
        return tuple(values)

    def _components(self, start, end, parameters, *, snapshot=False):
        raw_mu, raw_scale, raw_threshold, lora_A, lora_B = parameters
        weight = self._base_rows(start, end)
        if self.lora_rank:
            weight = weight + (lora_B @ lora_A) / self.lora_rank
        grouped = weight.reshape(-1, self.group_size)
        if self.lora_rank:
            # Differentiate through statistics of the adapted rotated weight.
            mu0, alpha0 = self._statistics(grouped)
        else:
            sl = self._group_slice(start, end)
            mu0 = (self._hard_snapshot_mu0 if snapshot else self.mu0)[sl]
            alpha0 = (self._hard_snapshot_alpha0 if snapshot else self.alpha0)[sl]
        if self.factor_mapping == "sigmoid":
            mu = mu0 + (2 * torch.sigmoid(raw_mu) - 1) * alpha0
            alpha = (2 * torch.sigmoid(raw_scale)) * alpha0
            threshold = .5 * (2 * torch.sigmoid(raw_threshold))
        else:
            mu = mu0 + torch.tanh(raw_mu) * alpha0
            alpha = torch.exp(raw_scale) * alpha0
            threshold = .5 * torch.exp(raw_threshold)
        return grouped, mu, alpha, threshold

    def _soft_rows(self, start, end, sharpness, *, snapshot=False):
        grouped, mu, alpha, threshold = self._components(
            start, end, self._parameter_rows(start, end, snapshot=snapshot), snapshot=snapshot)
        return alpha * soft_ternary((grouped - mu) / alpha, threshold, sharpness)

    def clear_soft_snapshot(self):
        for name in self._snapshot_names():
            setattr(self, name, None)
        self._hard_snapshot_sharpness, self._snapshot_step = None, None

    def begin_step(self, progress):
        """Begin one logical optimizer step, before any no-grad/grad row pass.

        One caller invocation defines one step. During a soft step, capture all
        compact state here exactly once; repeated row forwards never replace it.
        The retained point precedes that soft step's optimizer update.
        """
        progress = float(progress)
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be finite and in [0,1]")
        self.assert_source_unchanged()
        if progress == 0 or progress < self.progress:
            self.clear_soft_snapshot()
        self.progress = progress
        self._step_number += 1
        if (self.quant_enabled and 0 < progress <= self.soft_fraction
                and self.hard_phase == "frozen_jacobian"):
            with torch.no_grad():
                values = (*self.factor_parameters(), self.lora_A, self.lora_B, self.mu0, self.alpha0)
                for name, value in zip(self._snapshot_names(), values):
                    setattr(self, name, None if value is None else value.detach().clone())
            self._hard_snapshot_sharpness = self.max_sharpness * progress / self.soft_fraction
            self._snapshot_step = self._step_number

    def _check_hidden(self, hidden):
        if (not isinstance(hidden, Tensor) or hidden.ndim == 0 or hidden.shape[-1] != self.in_features
                or hidden.dtype != self.weight.dtype or hidden.device != self.weight.device):
            raise ValueError("hidden states must have source dtype/device and the correct last dimension")

    def teacher_rows(self, hidden, start, end):
        """Unchanged original BF16 operator; no LoRA or rotation leakage."""
        self._check_rows(start, end)
        self._check_hidden(hidden)
        return F.linear(hidden, self.weight[start:end], None if self.bias is None else self.bias[start:end])

    @torch.no_grad()
    def hard_codes_rows(self, start, end):
        """Return bounded int8 latent codes and FP32 per-row group scales."""
        self._check_rows(start, end)
        parameters = self._parameter_rows(start, end)
        if any(value is not None and not bool(torch.isfinite(value).all()) for value in parameters):
            raise ValueError("nonfinite endpoint quantizer parameter")
        grouped, mu, alpha, threshold = self._components(start, end, parameters)
        if (not all(bool(torch.isfinite(v).all()) for v in (grouped, mu, alpha, threshold))
                or bool((alpha <= 0).any()) or bool((threshold <= 0).any())):
            raise ValueError("cannot export invalid mapped endpoint factors")
        codes = hard_ternary((grouped - mu) / alpha, threshold).to(torch.int8)
        return codes.reshape(end - start, self.in_features), alpha.reshape(end - start, self.groups_per_row)

    def forward_rows(self, hidden, start, end):
        self._check_rows(start, end)
        self._check_hidden(hidden)
        if not self.quant_enabled:
            return self.teacher_rows(hidden, start, end)
        if self.progress == 0:
            # Match the unrotated dense quantizer's zero-gradient graph. The
            # rotated dense implementation uses an exact teacher bypass.
            if self.rotation_signs is not None:
                return self.teacher_rows(hidden, start, end)
            zero = sum(p.sum() * 0 for p in self._parameter_rows(start, end) if p is not None)
            return F.linear(hidden, self.weight[start:end] + zero.to(self.weight.dtype),
                            None if self.bias is None else self.bias[start:end])
        bias = None if self.bias is None else self.bias[start:end]
        if self.rotation_only:
            work = hidden.float()
            weight = self.weight[start:end].float()
            if self.rotation_signs is not None:
                work = rotate(work, self.rotation_signs, self.block_size)
                weight = rotate(weight, self.rotation_signs, self.block_size)
            return F.linear(work, weight, None if bias is None else bias.float()).to(hidden.dtype)
        if self.progress <= self.soft_fraction:
            grouped = self._soft_rows(start, end, self.max_sharpness * self.progress / self.soft_fraction)
        else:
            codes, scales = self.hard_codes_rows(start, end)
            hard = (codes.reshape(-1, self.group_size).float() * scales.reshape(-1, 1))
            if torch.is_grad_enabled() and any(p.requires_grad for p in self.trainable_parameters()):
                if self.hard_phase == "frozen_jacobian":
                    if self._hard_snapshot_sharpness is None:
                        raise RuntimeError("hard-stage gradients require a preceding soft begin_step")
                    surrogate = self._soft_rows(start, end, self._hard_snapshot_sharpness, snapshot=True)
                else:
                    surrogate = self._soft_rows(start, end, self.max_sharpness)
                grouped = hard + (surrogate - surrogate.detach())
            else:
                grouped = hard
        weight = grouped.reshape(end - start, self.in_features).to(self.weight.dtype)
        work = hidden if self.rotation_signs is None else rotate(hidden, self.rotation_signs, self.block_size)
        return F.linear(work, weight, bias)

    def forward(self, *args, **kwargs):
        raise RuntimeError("BoundedTernaryHead has no full-head forward; use forward_rows")
