"""Independent CAT-Q-style grouped ternary quantizer used by the prototype.

Based on arXiv:2606.26650, inherited by ScaleQ-1.58 (arXiv:2608.01078).
This is a research reconstruction, not the authors' unreleased training code.
No LoRA is used. The original weight/bias are frozen; three factors per group
modulate centering, scale, and threshold. There is no mean addback.

The paper's 'last soft iteration' hard-stage backward rule is underspecified.
Our explicit default freezes the terminal-soft Jacobian at the transition;
an alternative recomputes the soft surrogate at the current factors. Neither
choice is claimed to reproduce an unpublished implementation exactly.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .packing import pack_grouped_weight, unpack_ternary_codes, validate_packed_state


def soft_ternary(z: Tensor, threshold: Tensor | float, sharpness: float) -> Tensor:
    """Smooth ternary map with its analytic k=0 identity limit."""
    if not math.isfinite(sharpness) or sharpness < 0:
        raise ValueError("sharpness must be finite and nonnegative")
    if sharpness == 0:
        return z
    return (torch.tanh(sharpness * (z - threshold)) + torch.tanh(sharpness * (z + threshold))) / (2 * math.tanh(sharpness))


def hard_ternary(z: Tensor, threshold: Tensor | float) -> Tensor:
    """Exact ternary map; values exactly at either threshold map to zero."""
    return (z > threshold).to(z.dtype) - (z < -threshold).to(z.dtype)


class TernaryLinear(nn.Module):
    """Trainable grouped factors around a frozen ``nn.Linear``.

    Groups flatten the row-major weight, including a final partial group.
    Group statistics ignore padding. ``quant_enabled=False`` is the original
    dense teacher. At progress 0, the dense endpoint is also used; at 0<p<=.8
    the smooth map is used, and at p>.8 the exact ternary map is used.
    This explicit dense endpoint avoids calling a centered weight 'original'.
    """

    def __init__(
        self, linear: nn.Linear, group_size: int = 128, *, soft_fraction: float = 0.8,
        max_sharpness: float = 30.0, hard_phase: str = "frozen_jacobian",
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("TernaryLinear requires an nn.Linear")
        if linear.in_features <= 0 or linear.out_features <= 0:
            raise ValueError("source linear dimensions must be positive")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if not 0 < soft_fraction < 1 or not math.isfinite(max_sharpness) or max_sharpness <= 0:
            raise ValueError("soft_fraction must be in (0,1) and max_sharpness positive")
        if hard_phase not in ("frozen_jacobian", "soft_surrogate"):
            raise ValueError("hard_phase must be frozen_jacobian or soft_surrogate")
        if not linear.weight.is_floating_point() or not bool(torch.isfinite(linear.weight).all()):
            raise ValueError("source weights must be finite floating point values")
        if linear.bias is not None and not bool(torch.isfinite(linear.bias).all()):
            raise ValueError("source bias must be finite")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.group_size = group_size
        self.numel = linear.weight.numel()
        self.num_groups = math.ceil(self.numel / group_size)
        self.padded_numel = self.num_groups * group_size
        self.soft_fraction, self.max_sharpness, self.hard_phase = soft_fraction, max_sharpness, hard_phase
        self.quant_enabled = True
        self.progress = 0.0
        self.register_buffer("weight", linear.weight.detach().clone())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        grouped = self._grouped_weight()
        valid = torch.arange(self.padded_numel, device=grouped.device).reshape_as(grouped) < self.numel
        counts = valid.sum(dim=1, keepdim=True).to(grouped.dtype)
        mean = grouped.sum(dim=1, keepdim=True) / counts
        alpha = ((grouped - mean).abs() * valid).sum(dim=1, keepdim=True) / counts
        # Degenerate constant groups need a finite denominator; this is a
        # numerical guard, not a claim of lossless representation of constants.
        alpha = alpha.clamp_min(1e-8)
        self.register_buffer("mu0", mean)
        self.register_buffer("alpha0", alpha)
        self.raw_mu = nn.Parameter(torch.zeros_like(mean))
        self.raw_scale = nn.Parameter(torch.zeros_like(mean))
        self.raw_threshold = nn.Parameter(torch.zeros_like(mean))
        for name in ("_hard_snapshot_mu", "_hard_snapshot_scale", "_hard_snapshot_threshold"):
            self.register_buffer(name, None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128, **kwargs: Any) -> "TernaryLinear":
        return cls(linear, group_size, **kwargs)

    def _grouped_weight(self) -> Tensor:
        flat = self.weight.float().flatten()
        return F.pad(flat, (0, self.padded_numel - self.numel)).reshape(self.num_groups, self.group_size)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (self.raw_mu, self.raw_scale, self.raw_threshold)

    def reset_annealing(self) -> None:
        """Restart the schedule while preserving learned factors."""
        self.set_progress(0.0)

    def set_progress(self, progress: float) -> None:
        progress = float(progress)
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be finite and in [0,1]")
        if progress <= self.soft_fraction:
            self._hard_snapshot_mu = None
            self._hard_snapshot_scale = None
            self._hard_snapshot_threshold = None
        elif self._hard_snapshot_mu is None and self.hard_phase == "frozen_jacobian":
            self._hard_snapshot_mu = self.raw_mu.detach().clone()
            self._hard_snapshot_scale = self.raw_scale.detach().clone()
            self._hard_snapshot_threshold = self.raw_threshold.detach().clone()
        self.progress = progress

    def _factors(self, raw_mu: Tensor, raw_scale: Tensor, raw_threshold: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu = self.mu0 + torch.tanh(raw_mu) * self.alpha0
        alpha = torch.exp(raw_scale) * self.alpha0
        threshold = 0.5 * torch.exp(raw_threshold)
        return mu, alpha, threshold

    def _soft_grouped(self, sharpness: float) -> Tensor:
        mu, alpha, threshold = self._factors(self.raw_mu, self.raw_scale, self.raw_threshold)
        z = (self._grouped_weight() - mu) / alpha
        return alpha * soft_ternary(z, threshold, sharpness)

    def _hard_codes_and_scales(self) -> tuple[Tensor, Tensor]:
        mu, alpha, threshold = self._factors(self.raw_mu, self.raw_scale, self.raw_threshold)
        z = (self._grouped_weight() - mu) / alpha
        codes = hard_ternary(z, threshold)
        if self.padded_numel > self.numel:
            codes = codes.flatten().clone()
            codes[self.numel:] = 0
            codes = codes.reshape(self.num_groups, self.group_size)
        return codes, alpha

    def hard_weight(self) -> Tensor:
        """Exact hard ternary reconstruction, without mean addback."""
        codes, alpha = self._hard_codes_and_scales()
        return (codes * alpha).flatten()[:self.numel].reshape_as(self.weight).to(self.weight.dtype)

    def _frozen_jacobian_surrogate(self) -> Tensor:
        if self._hard_snapshot_mu is None:
            # Also handle callers that set .progress directly; set_progress is
            # the supported public entry point for deliberate schedule changes.
            self.set_progress(self.progress)
        rm, rs, rt = self._hard_snapshot_mu, self._hard_snapshot_scale, self._hard_snapshot_threshold
        with torch.no_grad():
            mu, alpha, threshold = self._factors(rm, rs, rt)
            z = (self._grouped_weight() - mu) / alpha
            k = self.max_sharpness
            minus, plus = torch.tanh(k * (z - threshold)), torch.tanh(k * (z + threshold))
            denominator = 2 * math.tanh(k)
            f = (minus + plus) / denominator
            fz = k * (2 - minus.square() - plus.square()) / denominator
            ft = k * (minus.square() - plus.square()) / denominator
            j_mu = -fz * self.alpha0 * (1 - torch.tanh(rm).square())
            j_scale = alpha * (f - z * fz)
            j_threshold = alpha * ft * threshold
        return j_mu * (self.raw_mu - rm) + j_scale * (self.raw_scale - rs) + j_threshold * (self.raw_threshold - rt)

    def quantized_weight(self) -> Tensor:
        if self.progress == 0:
            # Keep a zero-gradient graph so a progress-zero training step can
            # run backward; the source weight itself remains a frozen buffer.
            zero = sum(p.sum() * 0 for p in self.trainable_parameters())
            return self.weight + zero.to(self.weight.dtype)
        if self.progress <= self.soft_fraction:
            grouped = self._soft_grouped(self.max_sharpness * self.progress / self.soft_fraction)
        else:
            if not torch.is_grad_enabled() or not any(p.requires_grad for p in self.trainable_parameters()):
                return self.hard_weight()
            codes, alpha = self._hard_codes_and_scales()
            hard = (codes * alpha).detach()
            surrogate = (self._frozen_jacobian_surrogate() if self.hard_phase == "frozen_jacobian"
                         else self._soft_grouped(self.max_sharpness))
            grouped = hard + (surrogate - surrogate.detach())
        return grouped.flatten()[:self.numel].reshape_as(self.weight).to(self.weight.dtype)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.quantized_weight() if self.quant_enabled else self.weight, self.bias)

    @torch.no_grad()
    def packed_state(self) -> dict[str, Any]:
        """Export only hard codes/scales/bias, without dense training buffers."""
        if not all(bool(torch.isfinite(p).all()) for p in self.trainable_parameters()):
            raise ValueError("cannot pack nonfinite quantizer factors")
        mu, alpha, threshold = self._factors(self.raw_mu, self.raw_scale, self.raw_threshold)
        if (not all(bool(torch.isfinite(t).all()) for t in (mu, alpha, threshold))
                or bool((alpha <= 0).any()) or bool((threshold <= 0).any())):
            raise ValueError("cannot pack invalid mapped quantizer factors")
        codes, alpha = self._hard_codes_and_scales()
        return pack_grouped_weight(codes, alpha, shape=list(self.weight.shape), group_size=self.group_size, bias=self.bias)

    @classmethod
    def from_packed_state(
        cls, state: dict[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> "PackedTernaryLinear":
        """Restore inference-only weights; packed data cannot resume training."""
        return PackedTernaryLinear(state, device=device, dtype=dtype)


class PackedTernaryLinear(nn.Module):
    """Portable packed storage with dense dequantization on every forward.

    This supplies correctness/reference inference, not an optimized low-bit
    kernel. Runtime peak memory includes the reconstructed dense matrix.
    """

    def __init__(
        self, state: dict[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        validate_packed_state(state)
        self.out_features, self.in_features = state["shape"]
        self.group_size, self.numel = state["group_size"], state["numel"]
        self.num_groups, self.padded_numel = state["num_groups"], state["padded_numel"]
        self.register_buffer("codes", state["codes"].detach().clone().to(device=device))
        self.register_buffer("scales", state["scales"].detach().clone().to(device=device))
        self.register_buffer("bias", None if state["bias"] is None else state["bias"].detach().clone().to(device=device))
        self.register_buffer("_compute_reference", torch.empty(0, device=device, dtype=dtype or torch.float32), persistent=False)

    def _apply(self, fn, recurse: bool = True):
        # .to(dtype=...) changes computation dtype, while storage remains FP16.
        # Preserve original values before an intermediate BF16 cast can round
        # them; device movement still follows the normal module mechanism.
        old_scales, old_bias = self.scales, self.bias
        result = super()._apply(fn, recurse=recurse)
        self.scales = old_scales.to(device=self.scales.device, dtype=torch.float16)
        if old_bias is not None:
            self.bias = old_bias.to(device=self.bias.device, dtype=torch.float16)
        return result

    @property
    def weight(self) -> Tensor:
        codes = unpack_ternary_codes(self.codes, self.padded_numel).to(self._compute_reference.dtype)
        grouped = codes.reshape(self.num_groups, self.group_size) * self.scales.to(self._compute_reference.dtype)[:, None]
        return grouped.flatten()[:self.numel].reshape(self.out_features, self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def packed_state(self) -> dict[str, Any]:
        codes = unpack_ternary_codes(self.codes, self.padded_numel)
        return pack_grouped_weight(codes, self.scales, shape=[self.out_features, self.in_features], group_size=self.group_size, bias=self.bias)


def restore_ternary_linear(
    state: dict[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> PackedTernaryLinear:
    return PackedTernaryLinear(state, device=device, dtype=dtype)
