"""Read-only, one-sample activation diagnostics for native dense Qwen3.5.

This first diagnostic retains the original plain text embedding and default
RoPE. It compares hard decoder quantization without changing its schedule,
derivative snapshots, parameters or original sources. No vocabulary logits,
second model, accumulated activation cache or quality metric is produced.
"""
from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor, nn

from .adapter import _text_module, get_text_layers, set_quant_enabled
from .mixed_formats import MixedEmbedding, MixedLinear
from .quantizer import PackedTernaryLinear, TernaryLinear
from .rotations import RotatedTernaryLinear
from .trainer import _Captured, _device, _hidden, _run_until_capture, cache_window, replay_window


def _validate_model(model, start, end):
    layers = get_text_layers(model)  # Fail closed on non-native/model families.
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(layers):
        raise ValueError("Invalid activation diagnostic window")
    embedding = model.get_input_embeddings()
    if type(embedding) is not nn.Embedding or embedding.max_norm is not None:
        raise ValueError("Activation diagnostic requires a retained original plain nn.Embedding")
    text_model = _text_module(model)[1]
    if getattr(getattr(text_model, "rotary_emb", None), "rope_type", None) != "default":
        raise ValueError("Activation diagnostic currently supports only default RoPE")
    modules = []
    for module in model.modules():
        if isinstance(module, (PackedTernaryLinear, MixedLinear, MixedEmbedding)):
            raise ValueError("Activation diagnostic requires original sources, not packed/mixed modules")
        if hasattr(module, "quant_enabled"):
            if type(module) not in (TernaryLinear, RotatedTernaryLinear):
                raise ValueError("Unsupported quantizer in activation diagnostic")
            if (not isinstance(module.progress, (int, float)) or not isfinite(module.progress)
                    or not module.soft_fraction < module.progress <= 1):
                raise ValueError("Activation diagnostic requires hard-phase quantizers; schedule is not changed")
            if getattr(module, "rotation_only", False):
                raise ValueError("rotation_only is not hard quantization")
            modules.append(module)
    if not modules:
        raise ValueError("Activation diagnostic requires trainable original-source quantizers")
    return layers, modules


def _validate_sample(sample, model):
    if (not isinstance(sample, dict) or not {"input_ids", "attention_mask"} <= sample.keys()
            or set(sample) - {"input_ids", "attention_mask", "position_ids"}):
        raise ValueError("Diagnostic samples require input_ids/attention_mask and optional position_ids only; "
                         "cache state, media and unknown kwargs are unsupported")
    ids, mask = sample["input_ids"], sample["attention_mask"]
    if (not isinstance(ids, Tensor) or ids.dtype not in (torch.int32, torch.int64)
            or ids.ndim != 2 or not all(ids.shape)):
        raise ValueError("input_ids must be a nonempty two-dimensional integer tensor")
    if bool(((ids < 0) | (ids >= model.get_input_embeddings().num_embeddings)).any()):
        raise ValueError("input_ids outside the original embedding vocabulary")
    if (not isinstance(mask, Tensor) or mask.shape != ids.shape
            or mask.is_complex() or not bool(((mask == 0) | (mask == 1)).all())
            or not bool(mask.any(dim=-1).all())):
        raise ValueError("attention_mask must be binary, match input_ids, and contain valid tokens in every row")
    if "position_ids" in sample:
        positions = sample["position_ids"]
        if (not isinstance(positions, Tensor) or positions.dtype not in (torch.int32, torch.int64)
                or positions.shape not in (ids.shape, (3, *ids.shape))
                or bool((positions < 0).any())):
            raise ValueError("position_ids must be nonnegative integer native text/three-axis positions")


def _cpu_clone(value):
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_clone(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_clone(part) for part in value)
    return value


def _check_empty_state(kwargs):
    if any(kwargs.get(key) is not None for key in ("past_key_value", "past_key_values")):
        raise ValueError("Activation diagnostic requires empty KV/recurrent state")


def _original_path(model, sample, layers, start, end):
    captured, handles = {}, []

    def check_input(index):
        def hook(_module, args, kwargs):
            _check_empty_state(kwargs)
            if len(args) > 1:
                raise TypeError("Only hidden_states may be positional in a supported decoder")
            if index == start:
                captured["x0"] = _cpu_clone(args[0] if args else kwargs["hidden_states"])
        return hook

    def target(_module, _args, output):
        captured["t0"] = _cpu_clone(_hidden(output))
        raise _Captured

    try:
        for index in range(end):
            handles.append(layers[index].register_forward_pre_hook(check_input(index), with_kwargs=True))
        handles.append(layers[end - 1].register_forward_hook(target))
        _run_until_capture(model, sample)
    finally:
        for handle in handles:
            handle.remove()
    if captured.keys() != {"x0", "t0"}:
        raise RuntimeError("Incomplete original activation capture")
    return captured


@torch.no_grad()
def _collect_one(model, sample, start, end):
    layers, quantizers = _validate_model(model, start, end)
    _validate_sample(sample, model)  # No mode/quantization mutation before validation.
    enabled = [(module, module.quant_enabled) for module in quantizers]
    training = [(module, module.training) for module in model.modules()]
    # Native conditional-generation wrappers can cache multimodal position
    # deltas. Preserve the exact incoming object, even though media is excluded.
    deltas = [(module, module.rope_deltas) for module in model.modules() if hasattr(module, "rope_deltas")]
    try:
        for module, _ in training:
            module.training = False
        conditional = _cpu_clone(cache_window(model, [sample], start, end, teacher_prefix="quantized")[0])
        for kwargs in conditional["kwargs"].values():
            _check_empty_state(kwargs)
        set_quant_enabled(model, False)
        original = _original_path(model, sample, layers, start, end)
        set_quant_enabled(model, True)
        student = _cpu_clone(replay_window(layers, conditional, start, end, _device(model)))
        record = {"xq": conditional["input"], "x0": original["x0"],
                  "tq": conditional["target"], "t0": original["t0"],
                  "y": student, "mask": conditional["mask"]}
        for name in ("xq", "x0", "tq", "t0", "y"):
            value = record[name]
            if value.shape != record["t0"].shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Activation records must have matching finite hidden-state shapes")
        return record
    finally:
        for module, state in enabled:
            module.quant_enabled = state
        for module, state in training:
            module.training = state
        for module, state in deltas:
            module.rope_deltas = state


def iter_activation_records(model, samples, start: int, end: int):
    """Yield owned CPU ``xq,x0,tq,t0,y,mask`` tensors, one sample at a time.

    ``xq`` is the hard-quantized prefix input and ``x0`` the original prefix
    input to [start,end). ``tq=F_window(xq)``, ``t0=F_window(x0)``, and
    ``y=Q_window(xq)``. Both native passes stop before the vocabulary head;
    replay uses the conditional pass's per-block masks/rotary arguments.

    Callers must keep source/factor/sign state fixed for the iterator's lifetime.
    Measurements enable all supported hard quantizers on the student prefix
    and window, regardless of incoming enabled flags, then restore those flags.
    Each record runs under no_grad in temporary evaluation mode;
    exact incoming flags/modes are restored before yield, including on failure.
    Progress, snapshots, gradients and weight/factor/sign tensors are untouched.
    Closing or abandoning the iterator never leaves temporary hooks or modes.
    The collector retains no earlier record and never consumes all samples.
    Hidden states must be finite at every position, including padding; this
    collector rejects nonfinite padding rather than hiding it with a mask.
    """
    _validate_model(model, start, end)

    def records():
        for sample in samples:
            yield _collect_one(model, sample, start, end)

    return records()
