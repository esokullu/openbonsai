"""Opt-in, read-only summaries of stored standard AdamW moments.

No training hook is installed. Call between completed optimizer updates while
nothing else mutates the optimizer. Only bounded contiguous moment slices move
to CPU FP64; parameters and gradients are not read, copied, or changed.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch


FORMAT = "openbonsai.adamw-coordinate-diagnostics.v1"


def _scalar(value, label):
    if torch.is_tensor(value):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool or value.device.type == "meta":
            raise ValueError(f"{label} must be a materialized real scalar")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real scalar")
    return float(value)


def _group_settings(group, index):
    if group.get("amsgrad", False) is not False:
        raise ValueError("AMSGrad diagnostics are unsupported")
    if group.get("differentiable", False) is not False:
        raise ValueError("Differentiable AdamW diagnostics are unsupported")
    if group.get("decoupled_weight_decay", True) is not True:
        raise ValueError("Expected decoupled AdamW weight decay")
    betas = group.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("AdamW betas must contain two real scalars")
    betas = [_scalar(value, "beta") for value in betas]
    if not all(0 <= value < 1 for value in betas):
        raise ValueError("AdamW betas must lie in [0,1)")
    eps, lr, decay = (_scalar(group.get(key), key) for key in ("eps", "lr", "weight_decay"))
    if eps <= 0:
        raise ValueError("Diagnostics require positive epsilon; r/epsilon is undefined at zero")
    if lr < 0 or decay < 0:
        raise ValueError("Learning rate and weight decay must be nonnegative")
    flags = {}
    for key in ("maximize", "capturable", "foreach", "fused"):
        value = group.get(key, None if key in ("foreach", "fused") else False)
        if type(value) is not bool and not (value is None and key in ("foreach", "fused")):
            raise ValueError(f"Unsupported AdamW {key} setting")
        flags[key] = value
    return {"index": index, "betas": betas, "eps": eps, "lr": lr,
            "weight_decay": decay, "amsgrad": False, "differentiable": False, **flags}


def _bias_correction(beta, step):
    return 1. if beta == 0 else -math.expm1(step * math.log(beta))


class _Sum:
    """Constant-space compensated accumulation of bounded-chunk reductions."""
    def __init__(self):
        self.total = self.correction = 0.

    def add(self, value):
        if not math.isfinite(value):
            raise ValueError("Diagnostic arithmetic exceeded finite FP64 range")
        result = self.total + value
        self.correction += ((self.total - result) + value if abs(self.total) >= abs(value)
                            else (value - result) + self.total)
        self.total = result
        if not math.isfinite(self.total) or not math.isfinite(self.correction):
            raise ValueError("Diagnostic accumulation exceeded finite FP64 range")

    @property
    def value(self):
        result = self.total + self.correction
        if not math.isfinite(result):
            raise ValueError("Diagnostic accumulation exceeded finite FP64 range")
        return result


_SUM_KEYS = ("ratio_sq", "mhat_sq", "undamped_sq", "damped_sq", "damped_positive_r_sq",
             "damping_sum", "damping_sq")
_COUNT_KEYS = ("observed", "positive_r", "zero_m_zero_v", "nonzero_m_zero_v", "zero_m_positive_v",
               "nonzero_undamped", "below_1", "below_9", "below_99")


class _Accumulator:
    def __init__(self):
        self.initialized_parameters = self.uninitialized_parameters = self.total_coordinates = 0
        self.uninitialized_coordinates = 0
        self.counts = {key: 0 for key in _COUNT_KEYS}
        self.sums = {key: _Sum() for key in _SUM_KEYS}
        self.ratio_min = self.ratio_max = self.damping_min = self.damping_max = None

    def parameter(self, count, initialized):
        self.total_coordinates += count
        if initialized:
            self.initialized_parameters += 1
        else:
            self.uninitialized_parameters += 1
            self.uninitialized_coordinates += count

    def add(self, chunk):
        for key in _COUNT_KEYS:
            self.counts[key] += chunk[key]
        for key in _SUM_KEYS:
            self.sums[key].add(chunk[key])
        for name in ("ratio", "damping"):
            lo, hi = chunk[name + "_min"], chunk[name + "_max"]
            previous_lo, previous_hi = getattr(self, name + "_min"), getattr(self, name + "_max")
            setattr(self, name + "_min", lo if previous_lo is None else min(previous_lo, lo))
            setattr(self, name + "_max", hi if previous_hi is None else max(previous_hi, hi))

    def result(self):
        count, positive = self.counts["observed"], self.counts["positive_r"]
        def rms(key, n):
            return math.sqrt(self.sums[key].value / n) if n else None
        energy = self.sums["undamped_sq"].value
        attenuation = math.sqrt(self.sums["damped_positive_r_sq"].value / energy) if energy > 0 else None
        return {
            "initialized_parameters": self.initialized_parameters,
            "uninitialized_parameters": self.uninitialized_parameters,
            "coordinates": {"total": self.total_coordinates, "observed": count,
                "uninitialized": self.uninitialized_coordinates, "positive_r": positive,
                "stored_zero_m_and_v": self.counts["zero_m_zero_v"],
                "stored_nonzero_m_zero_v": self.counts["nonzero_m_zero_v"],
                "stored_zero_m_positive_v": self.counts["zero_m_positive_v"]},
            "r_over_epsilon": {"count": count, "min": self.ratio_min, "max": self.ratio_max,
                "rms": rms("ratio_sq", count),
                "fractions_below": {str(limit): self.counts[f"below_{limit}"] / count if count else None
                                    for limit in (1, 9, 99)}},
            "mhat_rms": rms("mhat_sq", count),
            "undamped_direction": {"domain": "r > 0", "count": positive,
                "nonzero_count": self.counts["nonzero_undamped"], "rms": rms("undamped_sq", positive)},
            "epsilon_direction": {"domain": "all observed coordinates", "count": count,
                                  "rms": rms("damped_sq", count)},
            "damping": {"definition": "r / (r + epsilon)", "count": count,
                "min": self.damping_min, "max": self.damping_max,
                "mean": self.sums["damping_sum"].value / count if count else None,
                "rms": rms("damping_sq", count)},
            "direction_energy_attenuation": {
                "definition": "sqrt(sum_{r>0} (mhat/(r+epsilon))^2 / sum_{r>0} (mhat/r)^2)",
                "domain_count": positive, "nonzero_undamped_count": self.counts["nonzero_undamped"],
                "ratio": attenuation,
                "undefined_comparator_coordinates": self.counts["nonzero_m_zero_v"],
                "zero_undamped_energy": energy == 0,
                "includes_learning_rate_or_weight_decay": False},
        }


def _moment_chunk(moment, start, end):
    # A contiguous view is sliced before conversion: never copy a whole moment
    # just to obtain a bounded slice, including on CUDA.
    return moment.detach().view(-1)[start:end].to(device="cpu", dtype=torch.float64)


def _chunk_summary(m, v, correction1, correction2, eps):
    if not bool(torch.isfinite(m).all()) or not bool(torch.isfinite(v).all()) or bool((v < 0).any()):
        raise ValueError("AdamW moments must be finite with nonnegative second moments")
    mhat, r = m / correction1, (v / correction2).sqrt()
    positive = r > 0
    ratio, damping = r / eps, r / (r + eps)
    damped = mhat / (r + eps)
    undamped = mhat[positive] / r[positive]
    damped_positive = damped[positive]
    values = (mhat, r, ratio, damping, damped, undamped)
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Diagnostic arithmetic exceeded finite FP64 range")
    return {
        "observed": m.numel(), "positive_r": int(positive.sum()),
        "zero_m_zero_v": int(((m == 0) & (v == 0)).sum()),
        "nonzero_m_zero_v": int(((m != 0) & (v == 0)).sum()),
        "zero_m_positive_v": int(((m == 0) & (v > 0)).sum()),
        "nonzero_undamped": int((undamped != 0).sum()),
        **{f"below_{limit}": int((ratio < limit).sum()) for limit in (1, 9, 99)},
        "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
        "damping_min": float(damping.min()), "damping_max": float(damping.max()),
        "ratio_sq": float(ratio.square().sum()), "mhat_sq": float(mhat.square().sum()),
        "undamped_sq": float(undamped.square().sum()), "damped_sq": float(damped.square().sum()),
        "damped_positive_r_sq": float(damped_positive.square().sum()),
        "damping_sum": float(damping.sum()), "damping_sq": float(damping.square().sum()),
    }


@torch.no_grad()
def summarize_adamw_coordinates(optimizer, named_parameters, *, chunk_elements=65536):
    """Return JSON-safe stored-moment observations, without changing the optimizer.

    ``named_parameters`` is a mapping or iterable of unique ``(name, tensor)``
    pairs belonging to this optimizer; an explicit subset is allowed. Families
    are the last dotted name component. Uninitialized coordinates are counted
    but excluded from numerical summaries. Epsilon must be positive.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("Only standard torch.optim.AdamW is supported")
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ValueError("chunk_elements must be a positive integer")
    groups, membership = [], {}
    for index, group in enumerate(optimizer.param_groups):
        settings = _group_settings(group, index)
        group_parameters = group.get("params")
        if not isinstance(group_parameters, (list, tuple)):
            raise ValueError("Optimizer group params must be a sequence")
        for parameter in group_parameters:
            if not isinstance(parameter, torch.Tensor):
                raise ValueError("Optimizer parameter must be a tensor")
            if id(parameter) in membership:
                raise ValueError("Duplicate optimizer parameter membership is ambiguous")
            if (parameter.dtype != torch.float32 or parameter.layout != torch.strided
                    or parameter.device.type == "meta"):
                raise ValueError("Only materialized dense real FP32 optimizer parameters are supported")
            membership[id(parameter)] = (parameter, index)
        groups.append({**settings, "parameters": len(group_parameters),
                       "elements": sum(p.numel() for p in group_parameters)})
    if any(id(parameter) not in membership for parameter in optimizer.state):
        raise ValueError("Optimizer state contains an unregistered parameter")
    pairs = list(named_parameters.items() if isinstance(named_parameters, Mapping) else named_parameters)
    if not pairs:
        raise ValueError("At least one named optimizer parameter is required")
    names, ids, selected = set(), set(), []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Expected (name, parameter) pairs")
        name, parameter = pair
        if not isinstance(name, str) or not name or not all(name.split('.')) or name in names:
            raise ValueError("Parameter names must be unique nonempty dotted names")
        if id(parameter) in ids or id(parameter) not in membership:
            raise ValueError("Named parameters must have unique optimizer membership")
        names.add(name); ids.add(id(parameter))
        selected.append((name, parameter, membership[id(parameter)][1]))
    parameters, families, family_groups, overall = {}, {}, {}, _Accumulator()
    for name, parameter, group_index in selected:
        settings = groups[group_index]
        family = name.rsplit('.', 1)[-1]
        family_summary = families.setdefault(family, _Accumulator())
        family_groups.setdefault(family, set()).add(group_index)
        accumulator = _Accumulator()
        state = optimizer.state.get(parameter)
        if state is not None and not isinstance(state, dict):
            raise ValueError("AdamW parameter state must be a dictionary")
        initialized = bool(state)
        for summary in (accumulator, family_summary, overall):
            summary.parameter(parameter.numel(), initialized)
        step = correction1 = correction2 = None
        if initialized:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Unsupported or partial AdamW moment state schema")
            raw_step = _scalar(state["step"], "stored step")
            if raw_step <= 0 or not raw_step.is_integer():
                raise ValueError("Initialized AdamW state requires a positive integral stored step")
            step = int(raw_step)
            correction1, correction2 = (_bias_correction(beta, step) for beta in settings["betas"])
            m, v = state["exp_avg"], state["exp_avg_sq"]
            for moment in (m, v):
                if (not isinstance(moment, torch.Tensor) or moment.dtype != torch.float32
                        or moment.layout != torch.strided or not moment.is_contiguous()
                        or moment.device != parameter.device or moment.shape != parameter.shape
                        or moment.requires_grad):
                    raise ValueError("Moments must be matching contiguous real FP32 tensors without gradients")
            for start in range(0, parameter.numel(), chunk_elements):
                end = min(start + chunk_elements, parameter.numel())
                chunk = _chunk_summary(_moment_chunk(m, start, end), _moment_chunk(v, start, end),
                                       correction1, correction2, settings["eps"])
                for summary in (accumulator, family_summary, overall):
                    summary.add(chunk)
        parameters[name] = {"family": family, "group_index": group_index, "shape": list(parameter.shape),
            "state_status": "initialized" if initialized else "uninitialized", "stored_step": step,
            "bias_correction1": correction1, "bias_correction2": correction2, **accumulator.result()}
    return {"format": FORMAT, "optimizer": "torch.optim.AdamW", "chunk_elements": chunk_elements,
        "arithmetic": "CPU FP64 estimates from bounded stored FP32 moment chunks",
        "step_basis": "each parameter's stored completed-update step; no upcoming-step assumption",
        "aggregation_scope": "named initialized coordinates only; uninitialized coordinates excluded",
        "zero_moment_interpretation": "stored zeros do not establish zero gradient or moment history",
        "direction_interpretation": "normalized adaptive directions only; no LR, decay, native rounding or quality attribution",
        "coverage": {"optimizer_parameters": len(membership),
            "optimizer_elements": sum(p.numel() for p, _ in membership.values()),
            "named_parameters": len(selected), "named_elements": sum(p.numel() for _, p, _ in selected),
            "complete_optimizer_coverage": len(selected) == len(membership)},
        "groups": groups, "parameters": parameters,
        "families": {name: {"group_indices": sorted(family_groups[name]), **summary.result()}
                     for name, summary in families.items()}, "overall": overall.result()}
