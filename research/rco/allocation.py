"""RCO allocation over real serialized, tensor-specific weight candidates.

The manifold primitives and optimization order adapt IST-DASLab/RCO
``src/manifold.py`` and ``src/search/quant.py`` at
9a1e09c07d468109cbe60a1b87d5036034a79d10 (Apache-2.0; see
``_vendor/rco/LICENSE`` and ``_vendor/rco/NOTICE``). This is an adaptation,
not an assertion of paper reproduction or a Qwen3.5 quality result.

Unlike a sum of independent weight errors, the objective supplied to
``optimize_rco`` evaluates the assembled model. Forward assignments obey
the actual integer byte budget; their backward estimator is Gumbel-softmax.
The expectation constraint is maintained by tangent projection, binary
retraction, and transport of Adam's first moment. Large discrete problems
use a conservative DP grid: feasibility is exact, optimality need not be.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CandidateTable:
    names: tuple[str, ...]
    candidates: tuple[tuple[Any, ...], ...]
    costs_bytes: Tensor
    valid: Tensor
    fixed_bytes: int = 0

    @property
    def minimum_bytes(self) -> int:
        return self.fixed_bytes + sum(int(self.costs_bytes[i, self.valid[i]].min())
                                      for i in range(len(self.names)))

    @property
    def maximum_bytes(self) -> int:
        return self.fixed_bytes + sum(int(self.costs_bytes[i, self.valid[i]].max())
                                      for i in range(len(self.names)))

    def selected_candidates(self, indices: Sequence[int]) -> dict[str, Any]:
        if len(indices) != len(self.names):
            raise ValueError("One candidate index is required per tensor")
        result = {}
        for i, (name, index) in enumerate(zip(self.names, indices, strict=True)):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.candidates[i]):
                raise ValueError(f"Invalid candidate index for {name}")
            result[name] = self.candidates[i][index]
        return result


def build_candidate_table(candidates: Mapping[str, Sequence[Any]], *, fixed_bytes: int = 0) -> CandidateTable:
    """Charge each encoded candidate's full bytes, not a nominal bits/weight.

    ``fixed_bytes`` must include all other exported data, including the root
    manifest. If that manifest depends on the assignment, its variable bytes
    must already be accounted for by the caller. The final artifact must be
    measured again; this function cannot infer an arbitrary exporter layout.
    """
    from .mixed_formats import candidate_nbytes

    if isinstance(fixed_bytes, bool) or not isinstance(fixed_bytes, int) or fixed_bytes < 0:
        raise ValueError("fixed_bytes must be a nonnegative integer")
    if not candidates or any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("At least one named tensor is required")
    names = tuple(sorted(candidates))
    rows = tuple(tuple(candidates[name]) for name in names)
    if any(not row for row in rows):
        raise ValueError("Every tensor needs at least one candidate")
    width = max(map(len, rows))
    costs = torch.zeros((len(rows), width), dtype=torch.int64)
    valid = torch.zeros_like(costs, dtype=torch.bool)
    for i, row in enumerate(rows):
        shape, kind = row[0].shape, row[0].module_kind
        if any(candidate.shape != shape or candidate.module_kind != kind for candidate in row):
            raise ValueError(f"Candidate shape/module kind mismatch for {names[i]}")
        for j, candidate in enumerate(row):
            costs[i, j] = candidate_nbytes(candidate)
            valid[i, j] = True
    return CandidateTable(names, rows, costs, valid, fixed_bytes)


@dataclass(frozen=True)
class DiscreteAllocation:
    indices: tuple[int, ...]
    total_bytes: int
    score: float
    exact_optimization: bool
    byte_unit: int


class ByteBudgetDecoder:
    """Multiple-choice knapsack with exact bytes and a bounded DP workspace.

    Subtract each row's minimum cost, then divide increments by their GCD.
    If that exact grid exceeds ``max_bins``, round each increment UP and the
    remaining budget DOWN. This always retains a minimum-cost solution,
    but can exclude other feasible solutions. We maximize summed scores
    across *all* reachable budgets, not the largest reachable budget.
    """

    def __init__(self, costs_bytes: Tensor, budget_bytes: int, *, valid: Tensor | None = None,
                 fixed_bytes: int = 0, max_bins: int = 4096):
        if costs_bytes.ndim != 2 or min(costs_bytes.shape) < 1 or costs_bytes.dtype != torch.int64:
            raise ValueError("costs_bytes must be a nonempty int64 matrix")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0
               for x in (budget_bytes, fixed_bytes)):
            raise ValueError("Byte budgets must be nonnegative integers")
        if isinstance(max_bins, bool) or not isinstance(max_bins, int) or max_bins < 1:
            raise ValueError("max_bins must be a positive integer")
        self.costs = costs_bytes.detach().cpu().clone()
        self.valid = torch.ones_like(self.costs, dtype=torch.bool) if valid is None else valid.detach().cpu().clone()
        if self.valid.shape != self.costs.shape or self.valid.dtype != torch.bool or not self.valid.any(dim=1).all():
            raise ValueError("valid must be boolean, correctly shaped, and nonempty per row")
        if (self.costs[self.valid] < 0).any():
            raise ValueError("Candidate costs cannot be negative")
        self.fixed_bytes, self.budget_bytes = fixed_bytes, budget_bytes
        minimum = torch.tensor([int(self.costs[i, self.valid[i]].min()) for i in range(len(self.costs))])
        self.minimum_bytes = fixed_bytes + sum(map(int, minimum))
        if self.minimum_bytes > budget_bytes:
            raise ValueError(f"Infeasible byte budget: minimum {self.minimum_bytes}, requested {budget_bytes}")
        remaining = budget_bytes - self.minimum_bytes
        increments = self.costs - minimum[:, None]
        deltas = [int(x) for x in increments[self.valid].tolist() if x > 0]
        gcd = reduce(math.gcd, deltas, 0) or 1
        # Capacity need not exceed the most expensive assignment.
        useful = min(remaining, sum(int(increments[i, self.valid[i]].max()) for i in range(len(self.costs))))
        self.exact_optimization = useful // gcd <= max_bins
        self.byte_unit = gcd if self.exact_optimization else max(1, math.ceil(useful / max_bins))
        self.capacity = useful // self.byte_unit
        self.bin_costs = torch.div(increments + self.byte_unit - 1, self.byte_unit, rounding_mode="floor")

    def decode(self, scores: Tensor) -> DiscreteAllocation:
        if scores.shape != self.costs.shape or not torch.isfinite(scores.detach().cpu()[self.valid]).all():
            raise ValueError("Scores must match the table and be finite for every valid candidate")
        scores = scores.detach().to(device="cpu", dtype=torch.float64)
        previous = torch.full((self.capacity + 1,), -torch.inf, dtype=torch.float64)
        previous[0] = 0
        back = torch.full((len(self.costs), self.capacity + 1), -1, dtype=torch.int64)
        for i in range(len(self.costs)):
            current = torch.full_like(previous, -torch.inf)
            for j in range(self.costs.shape[1]):
                if not self.valid[i, j]:
                    continue
                amount = int(self.bin_costs[i, j])
                if amount > self.capacity:
                    continue
                values = previous[:self.capacity + 1 - amount] + scores[i, j]
                improve = values > current[amount:]
                current[amount:] = torch.where(improve, values, current[amount:])
                back[i, amount:] = torch.where(improve, j, back[i, amount:])
            previous = current
        position = int(previous.argmax())
        score = float(previous[position])
        if not math.isfinite(score):
            raise RuntimeError("No reachable allocation despite a feasible minimum")
        indices = [0] * len(self.costs)
        for i in range(len(self.costs) - 1, -1, -1):
            j = int(back[i, position])
            if j < 0:
                raise RuntimeError("Invalid DP backtracking state")
            indices[i] = j
            position -= int(self.bin_costs[i, j])
        actual = self.fixed_bytes + sum(int(self.costs[i, j]) for i, j in enumerate(indices))
        if actual > self.budget_bytes:
            raise RuntimeError("Decoded allocation exceeds its actual byte budget")
        return DiscreteAllocation(tuple(indices), actual, score, self.exact_optimization, self.byte_unit)


def _probabilities(alpha: Tensor, valid: Tensor) -> Tensor:
    return alpha.masked_fill(~valid, -torch.inf).softmax(dim=-1)


def expected_cost(alpha: Tensor, costs: Tensor, valid: Tensor) -> Tensor:
    return (_probabilities(alpha, valid) * costs).sum()


def budget_normal(alpha: Tensor, costs: Tensor, valid: Tensor) -> Tensor:
    p = _probabilities(alpha, valid)
    return p * (costs - (p * costs).sum(dim=-1, keepdim=True))


@torch.no_grad()
def project_gradient(alpha: Tensor, costs: Tensor, valid: Tensor) -> tuple[float, float]:
    if alpha.grad is None:
        raise ValueError("The assembled objective did not produce allocation gradients")
    normal = budget_normal(alpha, costs, valid)
    raw_norm = float(alpha.grad.norm())
    norm_sq = normal.square().sum()
    if norm_sq > 0:
        alpha.grad.sub_((alpha.grad * normal).sum() / norm_sq * normal)
    alpha.grad.masked_fill_(~valid, 0)
    return raw_norm, float(alpha.grad.norm())


@torch.no_grad()
def retract(alpha: Tensor, costs: Tensor, valid: Tensor, target: float, *, tolerance: float = 1e-9,
            max_iterations: int = 100) -> float:
    """Monotonic cost-direction shift with binary search, not a penalty term."""
    minimum = float(costs.masked_fill(~valid, torch.inf).min(dim=-1).values.sum())
    maximum = float(costs.masked_fill(~valid, -torch.inf).max(dim=-1).values.sum())
    if not math.isfinite(target) or target < minimum - tolerance or target > maximum + tolerance:
        raise ValueError("Retraction target lies outside attainable expected costs")
    current = float(expected_cost(alpha, costs, valid))
    if abs(current - target) <= tolerance:
        return current
    lo, hi = -1.0, 1.0
    def value(shift):
        return float(expected_cost(alpha + shift * costs, costs, valid))
    for _ in range(100):
        if value(lo) <= target + tolerance and value(hi) >= target - tolerance:
            break
        if value(lo) > target:
            lo *= 2
        if value(hi) < target:
            hi *= 2
    else:
        raise RuntimeError("Could not bracket the expected-budget manifold")
    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        achieved = value(mid)
        if abs(achieved - target) <= tolerance:
            lo = hi = mid
            break
        if achieved > target:
            hi = mid
        else:
            lo = mid
    alpha.add_(((lo + hi) / 2) * costs)
    # A rowwise constant does not change probabilities; removing it limits drift.
    alpha.sub_(alpha.masked_fill(~valid, -torch.inf).max(dim=-1, keepdim=True).values)
    alpha.masked_fill_(~valid, 0)
    achieved = float(expected_cost(alpha, costs, valid))
    if abs(achieved - target) > max(tolerance * 4, 1e-12):
        raise RuntimeError("Retraction did not reach the requested tolerance")
    return achieved


@torch.no_grad()
def transport_momentum(optimizer: torch.optim.Optimizer, alpha: Tensor, costs: Tensor, valid: Tensor) -> None:
    """Transport only allocation Adam momentum; never touch model parameters."""
    moment = optimizer.state.get(alpha, {}).get("exp_avg")
    if moment is None:
        return
    normal = budget_normal(alpha, costs, valid)
    norm_sq = normal.square().sum()
    if norm_sq > 0:
        moment.sub_((moment * normal).sum() / norm_sq * normal)
    moment.masked_fill_(~valid, 0)


@dataclass(frozen=True)
class RCOResult:
    allocation: DiscreteAllocation
    names: tuple[str, ...]
    probabilities: Tensor
    history: tuple[dict[str, float | int | bool], ...]
    method: str = "rco-tensor-byte-adaptation"


def optimize_rco(table: CandidateTable, budget_bytes: int, objective: Callable[[Tensor], Tensor], *,
                 steps: int = 100, learning_rate: float = 0.05, tau_start: float = 1.0,
                 tau_end: float = 0.05, seed: int = 0, max_bins: int = 4096,
                 device: str | torch.device = "cpu", callback: Callable[[dict], None] | None = None) -> RCOResult:
    """Optimize assembled-model loss; all calibration selection stays in objective.

    The callback receives a fresh step record. Training data must be separate
    from evaluation data. Returned allocation decodes final logits, not a
    supposedly 'best' loss measured on potentially different minibatches.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not all(math.isfinite(x) and x > 0 for x in (learning_rate, tau_start, tau_end)) or tau_end > tau_start:
        raise ValueError("Learning rate and temperatures must be positive; temperature must not increase")
    decoder = ByteBudgetDecoder(table.costs_bytes, budget_bytes, valid=table.valid,
                                fixed_bytes=table.fixed_bytes, max_bins=max_bins)
    # Common positive scaling leaves the manifold/tangent plane unchanged and
    # avoids poorly conditioned calculations with multi-gigabyte integer costs.
    scale = max(1, table.maximum_bytes - table.fixed_bytes)
    costs = table.costs_bytes.to(device=device, dtype=torch.float64) / scale
    valid = table.valid.to(device)
    alpha = nn.Parameter(torch.zeros_like(costs))
    target = min(budget_bytes - table.fixed_bytes, table.maximum_bytes - table.fixed_bytes) / scale
    retract(alpha, costs, valid, target)
    optimizer = torch.optim.Adam([alpha], lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    for step in range(steps):
        temperature = tau_start * (tau_end / tau_start) ** (step / max(steps - 1, 1))
        optimizer.zero_grad(set_to_none=True)
        uniform = torch.rand(alpha.shape, generator=generator, device=device, dtype=alpha.dtype)
        uniform = uniform.clamp(torch.finfo(alpha.dtype).eps, 1 - torch.finfo(alpha.dtype).eps)
        noise = -torch.log(-torch.log(uniform))
        noisy_logits = (alpha + noise) / temperature
        soft = _probabilities(noisy_logits, valid)
        discrete = decoder.decode(noisy_logits)
        indices = torch.tensor(discrete.indices, device=device)
        hard = F.one_hot(indices, alpha.shape[1]).to(alpha.dtype)
        coefficients = hard - soft.detach() + soft
        loss = objective(coefficients)
        if not isinstance(loss, Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("The assembled objective must return a finite scalar Tensor")
        if not loss.requires_grad:
            raise ValueError("The assembled objective is disconnected from allocation coefficients")
        loss.backward()
        if alpha.grad is None or not torch.isfinite(alpha.grad).all():
            raise ValueError("Missing or nonfinite allocation gradients")
        raw_norm, projected_norm = project_gradient(alpha, costs, valid)
        optimizer.step()
        expected = retract(alpha, costs, valid, target)
        transport_momentum(optimizer, alpha, costs, valid)
        record = {"step": step, "loss": float(loss.detach()), "temperature": temperature,
                  "actual_bytes": discrete.total_bytes,
                  "expected_bytes": expected * scale + table.fixed_bytes,
                  "raw_gradient_norm": raw_norm, "projected_gradient_norm": projected_norm,
                  "exact_discrete_optimization": discrete.exact_optimization}
        history.append(record)
        if callback is not None:
            callback(dict(record))
    return RCOResult(decoder.decode(alpha), table.names, _probabilities(alpha, valid).detach().cpu(), tuple(history))


class _CandidateModule(nn.Module):
    def __init__(self, original: nn.Module, candidates: tuple[Any, ...]):
        super().__init__()
        object.__setattr__(self, "_original", original)
        self.candidates = candidates
        self.coefficients: Tensor | None = None

    @property
    def weight(self):
        # Native callers sometimes inspect dtype/device. Direct functional use
        # of this property is outside the supported module-call adaptation.
        return self._original.weight

    @property
    def bias(self):
        return getattr(self._original, "bias", None)

    def forward(self, inputs: Tensor) -> Tensor:
        from .mixed_formats import candidate_weight
        if self.coefficients is None:
            raise RuntimeError("Candidate module called outside an assembled objective")
        output = None
        for index, candidate in enumerate(self.candidates):
            def contribution(x, coefficient, chosen=candidate):
                weight = candidate_weight(chosen, device=self.weight.device, dtype=self.weight.dtype)
                if isinstance(self._original, nn.Linear):
                    # Bias stays in original module and is added exactly once.
                    result = F.linear(x, weight, None)
                else:
                    result = F.embedding(x, weight, padding_idx=self._original.padding_idx)
                return result * coefficient.to(result.dtype)
            # Recompute candidate decoding in backward instead of retaining K
            # decoded weights/outputs per module. This costs compute and is a
            # reference implementation, not an optimized ternary kernel.
            contribution_value = checkpoint(contribution, inputs, self.coefficients[index],
                                            use_reentrant=False, preserve_rng_state=False)
            output = contribution_value if output is None else output + contribution_value
        if self.bias is not None:
            output = output + self.bias
        return output


class AssembledCandidateModel:
    """Temporarily replace ordinary Linear/Embedding calls for true model loss.

    Original weights stay frozen and are restored on exit, including after
    exceptions. Candidate payloads should reside on CPU. Decoding is sequential
    and checkpointed, but the original model, activations, and one decoded
    tensor must still fit the chosen device. Large output logits can dominate
    memory. Direct external use of a selected module's ``.weight`` is unsupported.
    """

    def __init__(self, model: nn.Module, table: CandidateTable):
        self.model, self.table = model, table
        self._replacements: list[tuple[nn.Module, str, nn.Module, _CandidateModule]] = []
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Assembled candidate context is already active")
        replacements = []
        for name, candidates in zip(self.table.names, self.table.candidates, strict=True):
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            original = parent.get_submodule(child_name)
            if not isinstance(original, (nn.Linear, nn.Embedding)):
                raise ValueError(f"Only Linear and Embedding module calls are supported: {name}")
            kind = "linear" if isinstance(original, nn.Linear) else "embedding"
            if any(candidate.shape != tuple(original.weight.shape) or candidate.module_kind != kind
                   for candidate in candidates):
                raise ValueError(f"Candidate does not match original module {name}")
            if any("rotation" in candidate.metadata for candidate in candidates):
                raise ValueError("RCO assembled search does not yet support rotated candidates; use the rotation baseline")
            original_bias = getattr(original, "bias", None)
            for candidate in candidates:
                candidate_bias = candidate.tensors.get("bias")
                if (original_bias is None) != (candidate_bias is None) or (
                        original_bias is not None and not torch.equal(
                            original_bias.detach().cpu(), candidate_bias.to(dtype=original_bias.dtype, device="cpu"))):
                    raise ValueError(f"Candidate bias must preserve the original module bias: {name}")
            if isinstance(original, nn.Embedding) and original.max_norm is not None:
                raise ValueError("Embedding max_norm mutates weights and is unsupported")
            replacements.append((parent, child_name, original, _CandidateModule(original, candidates)))
        self._parameter_flags = [(p, p.requires_grad) for p in self.model.parameters()]
        self._training_flags = [(module, module.training) for module in self.model.modules()]
        for parameter, _ in self._parameter_flags:
            parameter.requires_grad_(False)
        self.model.eval()
        self._replacements = replacements
        for parent, name, _, replacement in replacements:
            parent._modules[name] = replacement
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for parent, name, original, replacement in self._replacements:
            replacement.coefficients = None
            parent._modules[name] = original
        for parameter, flag in self._parameter_flags:
            parameter.requires_grad_(flag)
        for module, flag in self._training_flags:
            module.training = flag
        self._active = False
        return False

    def __call__(self, coefficients: Tensor, *args, **kwargs):
        if not self._active:
            raise RuntimeError("Use AssembledCandidateModel as a context manager")
        if coefficients.shape != self.table.costs_bytes.shape:
            raise ValueError("Coefficient shape differs from candidate table")
        for i, (_, _, _, replacement) in enumerate(self._replacements):
            replacement.coefficients = coefficients[i, :len(replacement.candidates)]
        return self.model(*args, **kwargs)


def causal_lm_loss(logits: Tensor, input_ids: Tensor, attention_mask: Tensor | None = None, *,
                   labels: Tensor | None = None, token_chunk_size: int = 128) -> Tensor:
    """Masked assembled next-token CE with bounded FP32 loss working memory."""
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2] or input_ids.shape[1] < 2:
        raise ValueError("Expected [batch, sequence, vocabulary] logits and matching tokens")
    if not isinstance(token_chunk_size, int) or token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    targets = (input_ids if labels is None else labels).to(logits.device)
    if targets.shape != input_ids.shape:
        raise ValueError("labels must match input_ids")
    targets = targets[:, 1:].clone()
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        mask = attention_mask.to(logits.device).bool()
        targets.masked_fill_(~(mask[:, 1:] & mask[:, :-1]), -100)
    denominator = (targets != -100).sum()
    if not denominator:
        raise ValueError("No valid next-token targets")
    numerator = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.shape[1], token_chunk_size):
        stop = min(start + token_chunk_size, targets.shape[1])
        numerator = numerator + F.cross_entropy(logits[:, start:stop].float().reshape(-1, logits.shape[-1]),
                                                targets[:, start:stop].reshape(-1), ignore_index=-100,
                                                reduction="sum")
    return numerator / denominator
