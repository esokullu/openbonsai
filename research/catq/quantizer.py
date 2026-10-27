"""Independent CAT-Q-style grouped ternary quantizer used by the prototype.

Based on arXiv:2606.26650, inherited by ScaleQ-1.58 (arXiv:2608.01078).
This is a research reconstruction, not the authors' unreleased training code.
The original weight/bias are frozen. An optional low-rank correction is merged
before ternarization; three factors per group modulate centering, scale, and
threshold. There is no mean addback. Rank 64 is the default; rank 0 disables LoRA.

The paper's 'last soft iteration' hard-stage backward rule is underspecified.
Our explicit default freezes the Jacobian at the actual last soft forward point;
an alternative recomputes the soft surrogate at the current factors. Neither
choice is claimed to reproduce an unpublished implementation exactly.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .packing import (
    BASE3_FORMAT, BITCOS_FORMAT, FORMAT as TERNARY_FORMAT, pack_grouped_weight,
    unpack_base3_codes, unpack_bitcos_codes, unpack_ternary_codes, validate_packed_state,
)


# Bound differentiable FP32 scratch independently of full projection size.
_GROUP_CHUNK_ELEMENTS = 8 << 20


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
    """Trainable grouped factors and low-rank update around a frozen linear.

    Groups flatten the row-major weight, including a final partial group.
    Group statistics ignore padding. ``quant_enabled=False`` is the original
    dense teacher. At progress 0, the dense endpoint is also used; at 0<p<=.8
    the smooth map is used, and at p>.8 the exact ternary map is used.
    Both dense paths ignore the low-rank update. For rank > 0, statistics are
    computed on W + B @ A / rank, matching the released inference formulation.
    Kaiming-uniform A / zero B initialization and differentiation through these
    statistics are independent training choices, not a released training recipe.
    ``factor_mapping="exp"`` preserves the original exponential/tanh mapping
    and scale floor. ``"sigmoid"`` uses the released CAT-Q bounded mappings and
    adds 1e-6 to the centered mean absolute scale. FP32 sigmoid can saturate at
    its endpoints; export rejects a zero mapped scale or threshold.
    """

    def __init__(
        self, linear: nn.Linear, group_size: int = 128, *, soft_fraction: float = 0.8,
        max_sharpness: float = 30.0, hard_phase: str = "frozen_jacobian",
        lora_rank: int = 64, factor_mapping: str = "exp", copy_weight: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("TernaryLinear requires an nn.Linear")
        if linear.in_features <= 0 or linear.out_features <= 0:
            raise ValueError("source linear dimensions must be positive")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 0:
            raise ValueError("lora_rank must be a nonnegative integer")
        if not 0 < soft_fraction < 1 or not math.isfinite(max_sharpness) or max_sharpness <= 0:
            raise ValueError("soft_fraction must be in (0,1) and max_sharpness positive")
        if hard_phase not in ("frozen_jacobian", "soft_surrogate"):
            raise ValueError("hard_phase must be frozen_jacobian or soft_surrogate")
        if factor_mapping not in ("exp", "sigmoid"):
            raise ValueError("factor_mapping must be exp or sigmoid")
        if not isinstance(copy_weight, bool):
            raise TypeError("copy_weight must be a bool")
        if not linear.weight.is_floating_point():
            raise ValueError("source weights must be finite floating point values")
        if linear.bias is not None and not bool(torch.isfinite(linear.bias).all()):
            raise ValueError("source bias must be finite")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.group_size = group_size
        self.lora_rank = lora_rank
        self.factor_mapping = factor_mapping
        self.numel = linear.weight.numel()
        self.num_groups = math.ceil(self.numel / group_size)
        self.padded_numel = self.num_groups * group_size
        self.soft_fraction, self.max_sharpness, self.hard_phase = soft_fraction, max_sharpness, hard_phase
        self.quant_enabled = True
        self.progress = 0.0
        self.register_buffer("weight", linear.weight.detach().clone() if copy_weight else linear.weight.detach())
        self.register_buffer("bias", None if linear.bias is None else
                             (linear.bias.detach().clone() if copy_weight else linear.bias.detach()))
        if lora_rank:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, self.in_features, device=linear.weight.device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_rank, device=linear.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        mean, alpha = self._initial_statistics()
        self.register_buffer("mu0", mean)
        self.register_buffer("alpha0", alpha)
        self.raw_mu = nn.Parameter(torch.zeros_like(mean))
        self.raw_scale = nn.Parameter(torch.zeros_like(mean))
        self.raw_threshold = nn.Parameter(torch.zeros_like(mean))
        for name in self._snapshot_names():
            self.register_buffer(name, None, persistent=False)
        self._hard_snapshot_sharpness: float | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128, **kwargs: Any) -> "TernaryLinear":
        return cls(linear, group_size, **kwargs)

    @staticmethod
    def _snapshot_names() -> tuple[str, ...]:
        return ("_hard_snapshot_mu", "_hard_snapshot_scale", "_hard_snapshot_threshold",
                "_hard_snapshot_lora_A", "_hard_snapshot_lora_B")

    def _apply(self, fn, recurse: bool = True):
        # Module-wide .to(dtype=...) must not silently round optimizer variables,
        # group statistics, or a saved derivative point through BF16/FP16.
        parameters = {name: (parameter.detach(), None if parameter.grad is None else parameter.grad.detach())
                      for name, parameter in self.named_parameters(recurse=False)}
        buffers = {name: self._buffers[name] for name in ("mu0", "alpha0", *self._snapshot_names())
                   if self._buffers[name] is not None}
        result = super()._apply(fn, recurse=recurse)
        for name, (value, grad) in parameters.items():
            parameter = getattr(self, name)
            parameter.data = value.to(device=parameter.device, dtype=torch.float32)
            if grad is not None:
                parameter.grad = grad.to(device=parameter.device, dtype=torch.float32)
        for name, value in buffers.items():
            setattr(self, name, value.to(device=getattr(self, name).device, dtype=torch.float32))
        return result

    def _group(self, weight: Tensor) -> Tensor:
        flat = weight.flatten()
        if self.padded_numel != self.numel:
            flat = F.pad(flat, (0, self.padded_numel - self.numel))
        return flat.reshape(self.num_groups, self.group_size)

    def _group_ranges(self):
        groups_per_chunk = max(1, _GROUP_CHUNK_ELEMENTS // self.group_size)
        for start in range(0, self.num_groups, groups_per_chunk):
            yield start, min(start + groups_per_chunk, self.num_groups)

    def _effective_chunk(self, start: int, end: int, lora_A, lora_B) -> Tensor:
        """Differentiable effective weights for complete flattened groups.

        A group can cross a row. Compute only its enclosing rows of B@A,
        then trim them; this bounds scratch by a chunk plus two source rows.
        The final group is zero padded only after adding the correction.
        """
        first, last = start * self.group_size, min(end * self.group_size, self.numel)
        first_row = first // self.in_features
        last_row = (last + self.in_features - 1) // self.in_features
        offset = first - first_row * self.in_features
        # Slice rows before flattening so even a noncontiguous source cannot
        # cause reshape() to copy the entire frozen matrix for each chunk.
        values = self.weight[first_row:last_row].reshape(-1)[offset:offset + last - first].float()
        if lora_A is not None:
            correction = (lora_B[first_row:last_row] @ lora_A) / self.lora_rank
            values = values + correction.flatten()[offset:offset + last - first]
        padding = (end - start) * self.group_size - values.numel()
        if padding:
            values = F.pad(values, (0, padding))
        return values.reshape(end - start, self.group_size)

    def _chunk_statistics(self, grouped: Tensor, end: int) -> tuple[Tensor, Tensor]:
        counts = grouped.new_full((grouped.shape[0], 1), self.group_size)
        last_count = self.numel - (self.num_groups - 1) * self.group_size if end == self.num_groups else self.group_size
        counts[-1] = last_count
        mean = grouped.sum(dim=1, keepdim=True) / counts
        absolute = (grouped - mean).abs()
        alpha = absolute.mean(dim=1, keepdim=True)
        if last_count != self.group_size:
            alpha = torch.cat((alpha[:-1], absolute[-1, :last_count].mean().reshape(1, 1)))
        return mean, alpha + 1e-6 if self.factor_mapping == "sigmoid" else alpha.clamp_min(1e-8)

    @torch.no_grad()
    def _initial_statistics(self) -> tuple[Tensor, Tensor]:
        mean = torch.empty((self.num_groups, 1), device=self.weight.device, dtype=torch.float32)
        alpha = torch.empty_like(mean)
        for start, end in self._group_ranges():
            grouped = self._effective_chunk(start, end, None, None)
            if not bool(torch.isfinite(grouped).all()):
                raise ValueError("source weights must be finite floating point values")
            mean[start:end], alpha[start:end] = self._chunk_statistics(grouped, end)
        return mean, alpha

    def _mapped_chunk(self, start, end, parameters):
        raw_mu, raw_scale, raw_threshold, lora_A, lora_B = parameters
        grouped = self._effective_chunk(start, end, lora_A, lora_B)
        statistics = (self._chunk_statistics(grouped, end) if self.lora_rank else
                      (self.mu0[start:end], self.alpha0[start:end]))
        mu, alpha, threshold = self._factors(raw_mu[start:end], raw_scale[start:end],
                                            raw_threshold[start:end], statistics=statistics)
        return grouped, mu, alpha, threshold

    def _hard_chunk(self, start, end, parameters):
        grouped, mu, alpha, threshold = self._mapped_chunk(start, end, parameters)
        codes = hard_ternary((grouped - mu) / alpha, threshold)
        if end == self.num_groups and self.padded_numel > self.numel:
            codes = codes.clone()
            codes[-1, self.numel % self.group_size:] = 0
        return codes, alpha

    def _reconstruct_chunk(self, start, end, sharpness, hard, *parameters):
        if sharpness is None:
            codes, alpha = self._hard_chunk(start, end, parameters)
            values = codes * alpha
        else:
            surrogate_parameters = parameters[5:] if hard else parameters
            grouped, mu, alpha, threshold = self._mapped_chunk(start, end, surrogate_parameters)
            surrogate = alpha * soft_ternary((grouped - mu) / alpha, threshold, sharpness)
            if hard:
                with torch.no_grad():
                    codes, current_alpha = self._hard_chunk(start, end, parameters[:5])
                    values = codes * current_alpha
                values = values + (surrogate - surrogate.detach())
            else:
                values = surrogate
        count = min((end - start) * self.group_size, self.numel - start * self.group_size)
        return values.flatten()[:count].to(self.weight.dtype)

    def _bounded_weight(self, sharpness, hard, parameters):
        differentiable = torch.is_grad_enabled() and any(p is not None and p.requires_grad for p in parameters)
        chunks = []
        output = None if differentiable else torch.empty_like(self.weight).reshape(-1)
        for start, end in self._group_ranges():
            # Bind loop indices now: checkpoint calls this same pure function
            # again during backward, after the Python loop has finished.
            def reconstruct(*values, start=start, end=end):
                return self._reconstruct_chunk(start, end, sharpness, hard, *values)
            if differentiable:
                chunk = checkpoint(reconstruct, *parameters, use_reentrant=False, preserve_rng_state=False)
                chunks.append(chunk)
            else:
                chunk = reconstruct(*parameters)
                first = start * self.group_size
                output[first:first + chunk.numel()] = chunk
        # Only source-dtype outputs are concatenated. No full FP32 effective
        # matrix, statistics graph, tanh activations, or Jacobian is retained.
        if differentiable:
            output = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return output.reshape_as(self.weight)

    def _grouped_weight(self) -> Tensor:
        return self._effective_grouped(self.lora_A, self.lora_B)

    def _effective_grouped(self, lora_A: Tensor | None, lora_B: Tensor | None) -> Tensor:
        weight = self.weight.float()
        if self.lora_rank:
            weight = weight + (lora_B @ lora_A) / self.lora_rank
        return self._group(weight)

    def _statistics(self, grouped: Tensor) -> tuple[Tensor, Tensor]:
        # Only the last group can contain padding. Avoid a dense validity mask.
        counts = grouped.new_full((self.num_groups, 1), self.group_size)
        last_count = self.numel - (self.num_groups - 1) * self.group_size
        counts[-1] = last_count
        mean = grouped.sum(dim=1, keepdim=True) / counts
        absolute = (grouped - mean).abs()
        alpha = absolute.mean(dim=1, keepdim=True)
        if last_count != self.group_size:
            alpha = torch.cat((alpha[:-1], absolute[-1, :last_count].mean().reshape(1, 1)))
        return mean, alpha + 1e-6 if self.factor_mapping == "sigmoid" else alpha.clamp_min(1e-8)

    def factor_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.raw_mu, self.raw_scale, self.raw_threshold)

    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.lora_A, self.lora_B) if self.lora_rank else ()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.factor_parameters() + self.lora_parameters()

    def clear_soft_snapshot(self) -> None:
        """Release the derivative point after a window or before a new schedule."""
        for name in self._snapshot_names():
            setattr(self, name, None)
        self._hard_snapshot_sharpness = None

    def reset_annealing(self) -> None:
        """Restart the schedule while preserving learned factors."""
        self.set_progress(0.0)

    def set_progress(self, progress: float) -> None:
        progress = float(progress)
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be finite and in [0,1]")
        if progress == 0 or progress < self.progress:
            self.clear_soft_snapshot()
        self.progress = progress

    def _factors(
        self, raw_mu: Tensor, raw_scale: Tensor, raw_threshold: Tensor,
        *, statistics: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu0, alpha0 = (self.mu0, self.alpha0) if statistics is None else statistics
        if self.factor_mapping == "sigmoid":
            mu = mu0 + (2 * torch.sigmoid(raw_mu) - 1) * alpha0
            alpha = (2 * torch.sigmoid(raw_scale)) * alpha0
            threshold = 0.5 * (2 * torch.sigmoid(raw_threshold))
        else:
            mu = mu0 + torch.tanh(raw_mu) * alpha0
            alpha = torch.exp(raw_scale) * alpha0
            threshold = 0.5 * torch.exp(raw_threshold)
        return mu, alpha, threshold

    def _soft_grouped(self, sharpness: float) -> Tensor:
        return self._soft_at(sharpness, *self.factor_parameters(), self.lora_A, self.lora_B)

    def _soft_at(self, sharpness, raw_mu, raw_scale, raw_threshold, lora_A, lora_B) -> Tensor:
        grouped = self._effective_grouped(lora_A, lora_B)
        statistics = self._statistics(grouped) if self.lora_rank else None
        mu, alpha, threshold = self._factors(raw_mu, raw_scale, raw_threshold, statistics=statistics)
        z = (grouped - mu) / alpha
        return alpha * soft_ternary(z, threshold, sharpness)

    def _hard_codes_and_scales(self, grouped: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if grouped is None:
            grouped = self._grouped_weight()
        statistics = self._statistics(grouped) if self.lora_rank else None
        mu, alpha, threshold = self._factors(*self.factor_parameters(), statistics=statistics)
        z = (grouped - mu) / alpha
        codes = hard_ternary(z, threshold)
        if self.padded_numel > self.numel:
            codes = codes.flatten().clone()
            codes[self.numel:] = 0
            codes = codes.reshape(self.num_groups, self.group_size)
        return codes, alpha

    def hard_weight(self) -> Tensor:
        """Exact hard ternary reconstruction, without mean addback."""
        return self._bounded_weight(None, True, (*self.factor_parameters(), self.lora_A, self.lora_B))

    def _capture_soft_snapshot(self, sharpness: float) -> None:
        # Record the forward point, before its optimizer update. Dense weights
        # are frozen, so only compact factors/A/B and the actual k are needed.
        with torch.no_grad():
            values = (*self.factor_parameters(), self.lora_A, self.lora_B)
            for name, value in zip(self._snapshot_names(), values):
                setattr(self, name, None if value is None else value.detach().clone())
        self._hard_snapshot_sharpness = sharpness

    def _frozen_jacobian_surrogate(self) -> Tensor:
        return self._soft_at(self._hard_snapshot_sharpness, *self._frozen_jacobian_parameters())

    def _frozen_jacobian_parameters(self):
        if self._hard_snapshot_sharpness is None:
            raise RuntimeError(
                "Hard-stage training requires a preceding soft forward to capture its derivative; "
                "use hard_weight() or torch.no_grad() for inference"
            )
        # Values equal the saved point; derivatives flow to today's parameters.
        # Autograd then computes the full saved Jacobian, including the low-rank
        # product and effective-weight statistics, without storing dense Jacobians.
        current = (*self.factor_parameters(), self.lora_A, self.lora_B)
        proxies = tuple(None if value is None else getattr(self, name) + (value - value.detach())
                        for name, value in zip(self._snapshot_names(), current))
        return proxies

    def quantized_weight(self) -> Tensor:
        if self.progress == 0:
            # Keep a zero-gradient graph so a progress-zero training step can
            # run backward; the source weight itself remains a frozen buffer.
            zero = sum(p.sum() * 0 for p in self.trainable_parameters())
            return self.weight + zero.to(self.weight.dtype)
        if self.progress <= self.soft_fraction:
            sharpness = self.max_sharpness * self.progress / self.soft_fraction
            if (self.hard_phase == "frozen_jacobian" and torch.is_grad_enabled()
                    and any(p.requires_grad for p in self.trainable_parameters())):
                self._capture_soft_snapshot(sharpness)
            return self._bounded_weight(sharpness, False, (*self.factor_parameters(), self.lora_A, self.lora_B))
        else:
            if not torch.is_grad_enabled() or not any(p.requires_grad for p in self.trainable_parameters()):
                return self.hard_weight()
            current = (*self.factor_parameters(), self.lora_A, self.lora_B)
            surrogate = self._frozen_jacobian_parameters() if self.hard_phase == "frozen_jacobian" else current
            sharpness = self._hard_snapshot_sharpness if self.hard_phase == "frozen_jacobian" else self.max_sharpness
            return self._bounded_weight(sharpness, True, (*current, *surrogate))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.quantized_weight() if self.quant_enabled else self.weight, self.bias)

    @torch.no_grad()
    def packed_state(self, *, packing: str = "2bit") -> dict[str, Any]:
        """Export only hard codes/scales/bias, without dense training buffers."""
        if not all(bool(torch.isfinite(p).all()) for p in self.trainable_parameters()):
            raise ValueError("cannot pack nonfinite quantizer factors")
        parameters = (*self.factor_parameters(), self.lora_A, self.lora_B)
        codes = torch.empty(self.padded_numel, dtype=torch.int8, device="cpu")
        alpha = torch.empty(self.num_groups, dtype=torch.float32, device="cpu")
        for start, end in self._group_ranges():
            grouped, mu, chunk_alpha, threshold = self._mapped_chunk(start, end, parameters)
            if (not all(bool(torch.isfinite(t).all()) for t in (mu, chunk_alpha, threshold))
                    or bool((chunk_alpha <= 0).any()) or bool((threshold <= 0).any())):
                raise ValueError("cannot pack invalid mapped quantizer factors")
            chunk_codes = hard_ternary((grouped - mu) / chunk_alpha, threshold)
            codes[start * self.group_size:end * self.group_size] = chunk_codes.flatten().to(device="cpu", dtype=torch.int8)
            alpha[start:end] = chunk_alpha.flatten().cpu()
        codes[self.numel:] = 0
        return pack_grouped_weight(codes, alpha, shape=list(self.weight.shape),
                                   group_size=self.group_size, bias=self.bias, packing=packing)

    @torch.no_grad()
    def audit_weight_statistics(self) -> dict[str, Any]:
        """Boundary-only reductions without materializing dense FP32 weights."""
        parameters = (*self.factor_parameters(), self.lora_A, self.lora_B)
        energy = torch.zeros(3, device=self.weight.device, dtype=torch.float64)
        counts = torch.zeros(3, device=self.weight.device, dtype=torch.int64)
        for start, end in self._group_ranges():
            grouped, mu, alpha, threshold = self._mapped_chunk(start, end, parameters)
            count = min((end - start) * self.group_size, self.numel - start * self.group_size)
            effective = grouped.flatten()[:count]
            codes = hard_ternary((grouped - mu) / alpha, threshold)
            hard = (codes * alpha).flatten()[:count]
            original = self._effective_chunk(start, end, None, None).flatten()[:count]
            for index, values in enumerate((original, effective, hard)):
                energy[index] += values.square().sum(dtype=torch.float64)
            valid_codes = codes.flatten()[:count]
            counts += torch.stack(tuple((valid_codes == code).sum() for code in (-1, 0, 1)))
        rms = (energy / self.numel).sqrt().tolist()
        return dict(zip(("original_rms", "effective_rms", "hard_rms"), rms),
                    numel=self.numel, code_counts=dict(zip(("negative", "zero", "positive"), counts.tolist())))

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
        self.packing_format = state["format"]
        for key in ("codes", "presence", "signs"):
            value = state.get(key)
            self.register_buffer(key, None if value is None else value.detach().clone().to(device=device))
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
        return self._dequantized_weight(self._compute_reference.dtype)

    @staticmethod
    def _bit_range(stream: Tensor, first: int, last: int) -> Tensor:
        selected = stream[first // 8:(last + 7) // 8]
        bits = torch.stack(tuple((selected >> shift) & 1 for shift in range(8)), dim=1).flatten()
        return bits[first % 8:first % 8 + last - first]

    def _code_chunks(self):
        groups_per_chunk = max(1, _GROUP_CHUNK_ELEMENTS // self.group_size)
        sign_offset = 0
        for start in range(0, self.num_groups, groups_per_chunk):
            end = min(start + groups_per_chunk, self.num_groups)
            first, last = start * self.group_size, end * self.group_size
            if self.packing_format == TERNARY_FORMAT:
                selected = self.codes[first // 4:(last + 3) // 4]
                values = unpack_ternary_codes(selected, selected.numel() * 4)
                codes = values[first % 4:first % 4 + last - first]
            elif self.packing_format == BASE3_FORMAT:
                selected = self.codes[first // 5:(last + 4) // 5]
                values = unpack_base3_codes(selected, selected.numel() * 5)
                codes = values[first % 5:first % 5 + last - first]
            elif self.packing_format == BITCOS_FORMAT:
                presence = self._bit_range(self.presence, first, last)
                count = int(presence.sum().item())
                signs = self._bit_range(self.signs, sign_offset, sign_offset + count)
                codes = torch.zeros(last - first, dtype=torch.int8, device=presence.device)
                codes[presence.bool()] = 1 - 2 * signs.to(torch.int8)
                sign_offset += count
            else:
                raise ValueError(f"Unsupported packed format: {self.packing_format!r}")
            yield start, end, codes.reshape(end - start, self.group_size)

    def _dequantized_weight(self, dtype: torch.dtype) -> Tensor:
        output = torch.empty(self.numel, dtype=dtype, device=self.scales.device)
        for start, end, codes in self._code_chunks():
            compute_dtype = self._compute_reference.dtype
            values = (codes.to(compute_dtype) * self.scales[start:end].to(compute_dtype)[:, None]).flatten()
            first, last = start * self.group_size, min(end * self.group_size, self.numel)
            output[first:last] = values[:last - first]
        return output.reshape(self.out_features, self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        weight = self._dequantized_weight(x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def _unpack_codes(self) -> Tensor:
        if self.packing_format == TERNARY_FORMAT:
            return unpack_ternary_codes(self.codes, self.padded_numel)
        if self.packing_format == BITCOS_FORMAT:
            return unpack_bitcos_codes(self.presence, self.signs, self.padded_numel)
        if self.packing_format == BASE3_FORMAT:
            return unpack_base3_codes(self.codes, self.padded_numel)
        raise ValueError(f"Unsupported packed format: {self.packing_format!r}")

    def packed_state(self, *, packing: str | None = None) -> dict[str, Any]:
        """Keep the stored codec unless explicitly asked to repack it."""
        if packing is None:
            packing = {BITCOS_FORMAT: "bitcos", BASE3_FORMAT: "base3", TERNARY_FORMAT: "2bit"}[self.packing_format]
        return pack_grouped_weight(self._unpack_codes(), self.scales,
                                   shape=[self.out_features, self.in_features],
                                   group_size=self.group_size, bias=self.bias, packing=packing)


def restore_ternary_linear(
    state: dict[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> PackedTernaryLinear:
    return PackedTernaryLinear(state, device=device, dtype=dtype)
