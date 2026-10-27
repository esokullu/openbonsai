"""Original-BF16 candidate fitting with pinned RCO/GPTQ kernels.

GPTQ uses real input covariance for Linear modules. Embeddings use an explicitly
labelled grouped RTN baseline. Both persist the exact FP32 scales used to produce
codes; neither requantizes a ternary checkpoint. These are candidate builders,
not an allocation algorithm or accelerated inference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from ._vendor.rco.quant.fast_obq import FastOBQ
from ._vendor.rco.quant.quant_utils import Quantizer
from .mixed_formats import MixedWeightCandidate, affine_candidate

RCO_REVISION = "9a1e09c07d468109cbe60a1b87d5036034a79d10"


@dataclass
class InputHessian:
    matrix: Tensor
    token_count: int
    source: str = "original-model-inputs"


@contextmanager
def _fp32_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def source_weight_sha256(weight: Tensor, *, row_chunk: int = 256) -> str:
    """SHA256 of contiguous row-major original BF16 bytes, without a header."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("source weight must be an original BF16 matrix")
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("row_chunk must be a positive integer")
    digest = hashlib.sha256()
    for start in range(0, weight.shape[0], row_chunk):
        chunk = weight[start:start + row_chunk].detach().cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate(weight, bits, group_size, row_chunk):
    if (not isinstance(weight, Tensor) or weight.ndim != 2
            or weight.dtype != torch.bfloat16 or min(weight.shape) <= 0):
        raise ValueError("candidate fitting requires an original BF16 matrix")
    if type(group_size) is not int or group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("input width must be divisible by positive group_size")
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("row_chunk must be a positive integer")
    bits = tuple(bits)
    if not bits or len(set(bits)) != len(bits) or any(type(b) is not int or b not in (4, 6) for b in bits):
        raise ValueError("bits must contain distinct supported values 4 and/or 6")
    for start in range(0, weight.shape[0], row_chunk):
        if not bool(torch.isfinite(weight[start:start + row_chunk]).all()):
            raise ValueError("original weights must be finite")
    return bits


def _selected_inputs(inputs: Tensor, attention_mask: Tensor | None, remaining: int) -> Tensor:
    if inputs.ndim < 2:
        raise ValueError("linear inputs must have a feature dimension and at least one token")
    values = inputs.detach().reshape(-1, inputs.shape[-1])
    if attention_mask is not None:
        mask = attention_mask.detach().reshape(-1)
        if mask.numel() != values.shape[0]:
            raise ValueError("attention mask does not match captured input token positions")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("attention mask must contain zero/one values")
        # Select at most the token budget before copying or converting activations.
        indices = torch.nonzero(mask, as_tuple=False).flatten()[:remaining]
        return values.index_select(0, indices.to(values.device))
    return values[:remaining]


def hessian_from_activations(inputs: Tensor, *, attention_mask: Tensor | None = None,
                             max_tokens: int = 2048, accumulation_device=None) -> InputHessian:
    """Return 2 X.T X / valid_tokens, using a bounded deterministic prefix."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    values = _selected_inputs(inputs, attention_mask, max_tokens)
    if values.shape[0] == 0:
        raise ValueError("no valid activation tokens")
    device = inputs.device if accumulation_device is None else accumulation_device
    matrix = torch.zeros((inputs.shape[-1], inputs.shape[-1]), device=device, dtype=torch.float32)
    with _fp32_matmul():
        for chunk in values.split(256):
            x = chunk.to(device=device, dtype=torch.float32)
            if not bool(torch.isfinite(x).all()):
                raise ValueError("activation inputs must be finite")
            matrix.addmm_(x.T, x)
    matrix.mul_(2.0 / values.shape[0])
    return InputHessian(matrix, values.shape[0], "provided-original-activations")


def collect_input_hessians(model: nn.Module, batches: Iterable,
                          module_names: Iterable[str], *, max_tokens: int = 2048,
                          max_modules: int = 2, accumulation_device=None,
                          forward_kwargs: Mapping[str, Any] | None = None) -> dict[str, InputHessian]:
    """Capture one/few original Linear input Hessians without changing weights.

    Batches are tensors of input IDs or dictionaries already on the model's
    input device. Dictionary attention_mask excludes padding. Selects the first
    max_tokens valid tokens per module; does not collect all activations. The
    caller must supply an unchanged BF16 model, not quantized wrappers.
    """
    names = list(module_names)
    if (not names or len(set(names)) != len(names) or type(max_modules) is not int
            or max_modules <= 0 or len(names) > max_modules):
        raise ValueError("choose distinct module_names within max_modules")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    from .quantizer import TernaryLinear, PackedTernaryLinear
    from .mixed_formats import MixedLinear, MixedEmbedding
    if any(isinstance(module, (TernaryLinear, PackedTernaryLinear, MixedLinear, MixedEmbedding))
           for module in model.modules()):
        raise ValueError("input Hessians require the original model, not a quantized student")
    modules = {name: model.get_submodule(name) for name in names}
    for name, module in modules.items():
        if type(module) is not nn.Linear or module.weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be an unchanged BF16 nn.Linear")
        if hasattr(module, "parametrizations"):
            raise ValueError(f"{name} must not have weight parametrizations")
    counts = {name: 0 for name in names}
    matrices: dict[str, Tensor] = {}
    current_mask = None
    hooks = []
    training_modes = [(module, module.training) for module in model.modules()]

    def make_hook(name):
        def hook(module, args):
            if counts[name] >= max_tokens:
                return
            if not args or not isinstance(args[0], Tensor):
                raise ValueError(f"{name} has no positional tensor input")
            values = _selected_inputs(args[0], current_mask, max_tokens - counts[name])
            if not values.shape[0]:
                return
            device = values.device if accumulation_device is None else accumulation_device
            if name not in matrices:
                width = module.in_features
                matrices[name] = torch.zeros((width, width), device=device, dtype=torch.float32)
            for part in values.split(256):
                x = part.to(device=device, dtype=torch.float32)
                if not bool(torch.isfinite(x).all()):
                    raise ValueError(f"{name} input is nonfinite")
                matrices[name].addmm_(x.T, x)
            counts[name] += values.shape[0]
        return hook

    try:
        for name, module in modules.items():
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
        model.eval()
        with torch.no_grad(), _fp32_matmul():
            for batch in batches:
                kwargs = dict(forward_kwargs or {})
                kwargs.update(batch if isinstance(batch, Mapping) else {"input_ids": batch})
                current_mask = kwargs.get("attention_mask")
                model(**kwargs)
                if all(count >= max_tokens for count in counts.values()):
                    break
    finally:
        for hook in hooks:
            hook.remove()
        for module, mode in training_modes:
            module.training = mode
    missing = [name for name, count in counts.items() if not count]
    if missing:
        raise ValueError(f"no valid original-model inputs captured for {missing}")
    return {name: InputHessian(matrices[name].mul_(2.0 / counts[name]), counts[name])
            for name in names}


def _provenance(weight, provenance, **details):
    result = dict(provenance or {})
    digest = source_weight_sha256(weight)
    if "source_weight_sha256" in result and result["source_weight_sha256"] != digest:
        raise ValueError("source_weight_sha256 does not match original BF16 bytes")
    result.update(source_weight_sha256=digest, source_dtype="bfloat16",
                  rco_revision=RCO_REVISION, **details)
    return result


class _CachedFactorOBQ(FastOBQ):
    """Share Cholesky across row chunks with the same zero-column pattern.

    The upstream algorithm clones the factor per bitwidth and does not mutate
    the cached factor. Cache only the common no-zero-columns case, bounding
    memory even for pathological row chunks with many distinct zero patterns.
    """
    def __init__(self, *args, factor_cache, **kwargs):
        super().__init__(*args, **kwargs)
        self._factor_cache = factor_cache

    def _prepare(self):
        no_zero_columns = not bool(self.W.eq(0).all(dim=0).any())
        if no_zero_columns and "factor" in self._factor_cache:
            return self.W, self._factor_cache["factor"]
        weight, factor = super()._prepare()
        if no_zero_columns:
            self._factor_cache["factor"] = factor
        return weight, factor


def _buffers(weight, bits, group_size):
    rows, width = weight.shape
    return {bit: (torch.empty((rows, width), dtype=torch.uint8),
                  torch.empty((rows, width // group_size), dtype=torch.float32),
                  torch.empty((rows, width // group_size), dtype=torch.int64)) for bit in bits}


def _store_rows(buffers, start, qcodes, scales, zeros, original, diagnostics):
    for bit, (codes_out, scales_out, zeros_out) in buffers.items():
        q, s, z = qcodes[bit], scales[bit], zeros[bit]
        if (not bool(torch.isfinite(s).all()) or not bool(torch.isfinite(z).all())
                or not bool((s > 0).all()) or not torch.equal(z, z.round())
                or bool((z.abs() >= 2**63).any())):
            raise ValueError("GPTQ produced unrepresentable scales or integer zero points")
        end = start + q.shape[0]
        codes_out[start:end].copy_(q.cpu())
        scales_out[start:end].copy_(s.cpu())
        zeros_out[start:end].copy_(z.to(device="cpu", dtype=torch.int64))
        grouped = q.float().reshape(*s.shape, -1)
        restored = ((grouped - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape_as(q)
        errors = restored - original
        diagnostics[bit]["squared_error_sum"] += errors.square().sum(dtype=torch.float64).item()
        diagnostics[bit]["max_abs_error"] = max(diagnostics[bit]["max_abs_error"], errors.abs().max().item())
        diagnostics[bit]["endpoint_codes"] += int(((q == 0) | (q == 2**bit - 1)).sum())


def _finish(weight, buffers, diagnostics, group_size, bias, module_kind, source, provenance):
    result = {}
    for bit, (q, scales, zeros) in buffers.items():
        d = diagnostics[bit]
        info = dict(provenance, weight_reconstruction_mse=d["squared_error_sum"] / weight.numel(),
                    max_abs_weight_error=d["max_abs_error"],
                    code_endpoint_fraction=d["endpoint_codes"] / weight.numel(),
                    endpoint_fraction_is_not_clipping_rate=True, scale_rounding_applied=False)
        result[bit] = affine_candidate(q, scales, zeros, bits=bit, group_size=group_size,
                                      bias=bias, module_kind=module_kind, source=source, provenance=info)
    return result


@torch.no_grad()
def fit_gptq_candidates(weight: Tensor, hessian: Tensor | InputHessian | None = None, *,
                        activations: Tensor | None = None, group_size: int = 128,
                        bits=(4, 6), row_chunk: int = 256, bias: Tensor | None = None,
                        provenance: Mapping[str, Any] | None = None,
                        rel_damp: float = 0.01, device=None) -> dict[int, MixedWeightCandidate]:
    """Fit 4/6-bit candidates from unchanged BF16 with actual GPTQ error feedback.

    Input H is 2 X.T X / token_count, captured from original-model inputs.
    Group and block size match. Output columns retain original order (no
    act_order). Original weight and H are never mutated. Row chunks bound FP32
    copies, and shared Cholesky prevents a head-sized repeated factorization.
    """
    bits = _validate(weight, bits, group_size, row_chunk)
    if (hessian is None) == (activations is None):
        raise ValueError("provide exactly one of hessian or original activations")
    if not math.isfinite(rel_damp) or rel_damp <= 0:
        raise ValueError("rel_damp must be finite and positive")
    if activations is not None:
        hessian = hessian_from_activations(activations)
    stats = hessian if isinstance(hessian, InputHessian) else None
    if stats is not None and (type(stats.token_count) is not int or stats.token_count <= 0):
        raise ValueError("InputHessian must record a positive valid token count")
    h = stats.matrix if stats is not None else hessian
    width = weight.shape[1]
    if (not isinstance(h, Tensor) or h.shape != (width, width)
            or not h.is_floating_point() or not bool(torch.isfinite(h).all())
            or not torch.allclose(h, h.T, rtol=1e-5, atol=1e-6)
            or bool((h.diag() < 0).any()) or not bool((h.diag() > 0).any())):
        raise ValueError("hessian must be finite, symmetric, nonnegative-diagonal and informative")
    device = weight.device if device is None else torch.device(device)
    h = h.detach().to(device=device, dtype=torch.float32)
    info = _provenance(weight, provenance, quantizer="rco-gptq-fastobq", group_size=group_size,
                       row_chunk=row_chunk, act_order=False, symmetric=False, rel_damp=rel_damp,
                       hessian_source=stats.source if stats else "provided-original-input-hessian",
                       hessian_tokens=stats.token_count if stats else None,
                       dead_input_columns=int((h.diag() == 0).sum()))
    buffers = _buffers(weight, bits, group_size)
    diagnostics = {bit: dict(squared_error_sum=0.0, max_abs_error=0.0, endpoint_codes=0) for bit in bits}
    factor_cache = {}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for start in range(0, weight.shape[0], row_chunk):
            original = weight[start:start + row_chunk].detach().to(device=device, dtype=torch.float32)
            linear = nn.Linear(width, original.shape[0], bias=False, device="meta", dtype=torch.float32)
            linear.weight = nn.Parameter(original.clone(), requires_grad=False)
            handle = _CachedFactorOBQ(linear, list(bits), group_size=group_size,
                                      block_size=group_size, sym=False, rel_damp=rel_damp,
                                      act_order=False, factor_cache=factor_cache)
            handle.H = h.clone()
            qcodes, scales, zeros, perm = handle.quantize(list(bits))
            if perm is not None:
                raise RuntimeError("unexpected GPTQ column permutation")
            _store_rows(buffers, start, qcodes, scales, zeros, original, diagnostics)
            del handle, linear, original, qcodes, scales, zeros
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return _finish(weight, buffers, diagnostics, group_size, bias, "linear", "gptq", info)


@torch.no_grad()
def fit_embedding_rtn_candidates(weight: Tensor, *, group_size: int = 128,
                                 bits=(4, 6), row_chunk: int = 256,
                                 provenance: Mapping[str, Any] | None = None,
                                 device=None) -> dict[int, MixedWeightCandidate]:
    """Grouped RTN baseline from original BF16 embedding, not calibrated GPTQ."""
    bits = _validate(weight, bits, group_size, row_chunk)
    device = weight.device if device is None else torch.device(device)
    info = _provenance(weight, provenance, quantizer="embedding-grouped-affine-rtn",
                       calibration="none-identity-weight-error-baseline", group_size=group_size,
                       row_chunk=row_chunk, symmetric=False)
    buffers = _buffers(weight, bits, group_size)
    diagnostics = {bit: dict(squared_error_sum=0.0, max_abs_error=0.0, endpoint_codes=0) for bit in bits}
    for start in range(0, weight.shape[0], row_chunk):
        original = weight[start:start + row_chunk].detach().to(device=device, dtype=torch.float32)
        grouped = original.reshape(-1, group_size)
        qcodes, scales, zeros = {}, {}, {}
        for bit in bits:
            quantizer = Quantizer()
            quantizer.configure(bits=bit, perchannel=True, sym=False)
            quantizer.find_params(grouped, weight=True)
            qcodes[bit] = quantizer.quantize(grouped).to(torch.uint8).reshape_as(original)
            scales[bit] = quantizer.scale.reshape(original.shape[0], -1)
            zeros[bit] = quantizer.zero.reshape(original.shape[0], -1)
        _store_rows(buffers, start, qcodes, scales, zeros, original, diagnostics)
    return _finish(weight, buffers, diagnostics, group_size, None, "embedding", "embedding-rtn", info)


def eligible_mixed_modules(model, *, include_embedding=True, include_lm_head=True,
                           aggressive_gates=False) -> dict[str, str]:
    """Native dense Qwen3.5 text-only allowlist, mapping module paths to kinds."""
    from .adapter import _PROJECTIONS, _RECURRENT_GATES, _text_module
    prefix, text = _text_module(model)
    result = {}
    if include_embedding:
        name = f"{prefix}.embed_tokens"
        if not isinstance(model.get_submodule(name), nn.Embedding):
            raise ValueError("native input embedding is unavailable")
        result[name] = "embedding"
    allowed = _PROJECTIONS | (_RECURRENT_GATES if aggressive_gates else set())
    for index, layer in enumerate(text.layers):
        for relative, module in layer.named_modules():
            if relative in allowed:
                if type(module) is not nn.Linear:
                    raise ValueError("mixed candidates require original unwrapped Linear modules")
                result[f"{prefix}.layers.{index}.{relative}"] = "linear"
    if include_lm_head:
        head = model.get_output_embeddings()
        if type(head) is not nn.Linear:
            raise ValueError("native output Linear head is unavailable")
        name = next((name for name, module in model.named_modules() if module is head), None)
        if name is None:
            raise ValueError("output head path is unavailable")
        result[name] = "linear"
    return result
