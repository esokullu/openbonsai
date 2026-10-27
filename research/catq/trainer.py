"""OpenBonsai's independent ScaleQ-style reconstruction for native Qwen3.5.

This is a research adaptation, not the authors' implementation. In particular,
the hybrid attention projection policy and schedule choices are documented in
README.md. No language-model cross-entropy or external teacher is used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
import random
from pathlib import Path
import tempfile
import time
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .adapter import get_text_layers, iter_ternary_modules, set_quant_enabled


@dataclass
class TrainingConfig:
    epochs: int = 60
    batch_size: int = 3
    window_size: int = 4
    stride: int = 2
    learning_rate: float = 0.001
    lora_learning_rate: float = 0.001
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 2
    gradient_checkpointing: bool = False
    reconstruction_loss: str = "mse"
    huber_delta: float = 1.0
    teacher_prefix: str = "quantized"

    def validate(self):
        for name in ("epochs", "batch_size", "window_size", "stride"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stride > self.window_size:
            raise ValueError("stride must not exceed window_size (would skip layers)")
        if (not all(isfinite(value) for value in (
                self.learning_rate, self.lora_learning_rate, self.weight_decay, self.grad_clip))
                or self.learning_rate <= 0 or self.lora_learning_rate <= 0
                or self.weight_decay < 0 or self.grad_clip <= 0):
            raise ValueError("learning rates/grad_clip must be finite and positive; weight_decay finite and nonnegative")
        _validate_reconstruction_loss(self.reconstruction_loss, self.huber_delta)
        _validate_teacher_prefix(self.teacher_prefix)


def _validate_teacher_prefix(teacher_prefix):
    if type(teacher_prefix) is not str or teacher_prefix not in ("quantized", "original"):
        raise ValueError("teacher_prefix must be quantized or original")


def _validate_reconstruction_loss(reconstruction_loss, huber_delta):
    if reconstruction_loss not in ("mse", "huber"):
        raise ValueError("reconstruction_loss must be mse or huber")
    if (isinstance(huber_delta, bool) or not isinstance(huber_delta, (int, float))
            or not isfinite(huber_delta) or huber_delta <= 0):
        raise ValueError("huber_delta must be finite and positive")


class _Captured(Exception):
    """Internal early exit: avoid computing the vocabulary head during calibration."""


class DiskRecords:
    """CPU cache with one sample resident at a time when read by the trainer."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = []

    def append(self, record):
        path = self.directory / f"sample-{len(self.paths):06d}.pt"
        torch.save(record, path)
        self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location="cpu", weights_only=True)


def _tree_to(value: Any, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_tree_to(v, device) for v in value)
    if isinstance(value, list):
        return [_tree_to(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _tree_to(v, device) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported cached forward value {type(value).__name__}; cache objects are not allowed")


def _device(model):
    return next(model.parameters()).device


def _phase_time(device):
    """Complete queued GPU work only at reporting phase boundaries."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported block output {type(output).__name__}")


def _module_values(model):
    # Adapter returns (qualified_name, module) pairs.
    return [module for _, module in iter_ternary_modules(model)]


def _original_teacher_extra_modules(model, layers, end):
    """Validate original-source availability before changing any model state.

    The decoder adapter intentionally excludes bounded endpoint prototypes.
    Handle the known embedding's exact bypass locally; a packed/mixed artifact
    does not establish an original source and cannot supply this teacher.
    """
    from .embedding_quantizer import BoundedTernaryEmbedding
    from .mixed_formats import MixedLinear
    from .quantizer import PackedTernaryLinear
    from .rotations import RotatedTernaryEmbedding

    embedding = model.get_input_embeddings()
    if type(embedding) not in (torch.nn.Embedding, RotatedTernaryEmbedding, BoundedTernaryEmbedding):
        raise ValueError("original teacher prefix requires a supported original-source embedding; "
                         "packed/mixed or unknown replaced embeddings cannot supply original targets")
    if (any(isinstance(module, PackedTernaryLinear) for module in model.modules())
            or any(isinstance(module, MixedLinear)
                   for layer in layers[:end] for module in layer.modules())):
        raise ValueError("original teacher prefix cannot use packed/mixed decoder weights without an original source")
    if isinstance(embedding, BoundedTernaryEmbedding):
        embedding.assert_source_unchanged()
        return [embedding]
    return []


def build_windows(num_layers: int, width: int, stride: int):
    if num_layers < 1 or width < 1 or stride < 1 or stride > width:
        raise ValueError("Invalid layer count/window size/stride")
    return [(start, min(start + width, num_layers)) for start in range(0, num_layers, stride)]


def _run_until_capture(model, sample):
    try:
        model(**_tree_to(sample, _device(model)), use_cache=False)
    except _Captured:
        return
    raise RuntimeError("Capture hook did not fire; model layout is incompatible")


@torch.no_grad()
def cache_window(model, samples, start: int, end: int, cache_directory=None, *, teacher_prefix="quantized"):
    """Cache quantized-prefix student inputs and an explicit teacher target.

    Masks/rotary kwargs are captured per block. Qwen3.5 full-attention and
    GatedDeltaNet blocks must not share a blindly copied attention mask.
    The default retains the local conditional objective: original window
    weights receive the same quantized-prefix input as the student. The
    original option instead obtains targets from a second native forward with
    every quantizer disabled. Paper notation alone does not establish which
    prefix produced X; this switch records that independent implementation
    choice. Passes stop at the window boundary without vocabulary logits.
    Student kwargs always come from the first pass. Caches live on CPU/disk;
    no second model or extra retained hidden-state stream is required.
    A bounded trainable embedding retains its caller-selected student state
    and uses its original-source bypass for the teacher. Packed/mixed sources
    on the teacher path are rejected before mutation in original mode.
    """
    _validate_teacher_prefix(teacher_prefix)
    layers = get_text_layers(model)
    if not 0 <= start < end <= len(layers):
        raise ValueError("Invalid capture window")
    extra_modules = (_original_teacher_extra_modules(model, layers, end)
                     if teacher_prefix == "original" else [])
    extra_states = [(module, module.quant_enabled) for module in extra_modules]
    prior_states = [(module, module.quant_enabled) for module in _module_values(model)]
    prior_states.extend(extra_states)
    records = [] if cache_directory is None else DiskRecords(cache_directory)
    try:
        for sample in samples:
            record = {"kwargs": {}, "mask": sample["attention_mask"].detach().cpu()}
            hooks = []
            set_quant_enabled(model, True)
            # The legacy decoder adapter leaves bounded endpoints alone. Keep
            # precisely that student path, including across multiple samples.
            for module, enabled in extra_states:
                module.quant_enabled = enabled
            for layer in layers[start:end]:
                for module in layer.modules():
                    if module in dict(prior_states):
                        module.quant_enabled = False

            def capture_kwargs(index):
                def hook(_module, args, kwargs):
                    if len(args) > 1:
                        raise TypeError("Only hidden_states may be positional in a supported decoder")
                    if index == start:
                        record["input"] = (args[0] if args else kwargs["hidden_states"]).detach().cpu()
                    kwargs = dict(kwargs)
                    if "hidden_states" in kwargs:
                        kwargs.pop("hidden_states")
                    for key in ("past_key_value", "past_key_values"):
                        if kwargs.get(key) is not None:
                            raise ValueError("Reconstruction must run with empty KV/recurrent state")
                    record["kwargs"][index] = _tree_to(kwargs, "cpu")
                return hook

            def capture_target(_module, _args, output):
                record["target"] = _hidden(output).detach().cpu()
                raise _Captured

            def stop_window(_module, _args, _output):
                raise _Captured

            try:
                for index in range(start, end):
                    hooks.append(layers[index].register_forward_pre_hook(capture_kwargs(index), with_kwargs=True))
                hooks.append(layers[end - 1].register_forward_hook(
                    capture_target if teacher_prefix == "quantized" else stop_window))
                _run_until_capture(model, sample)
            finally:
                for hook in hooks:
                    hook.remove()

            if teacher_prefix == "original":
                # Keep the student's quantized-prefix input and native kwargs.
                # Only the target comes from this independent original path.
                set_quant_enabled(model, False)
                for module in extra_modules:
                    module.quant_enabled = False
                target_hook = layers[end - 1].register_forward_hook(capture_target)
                try:
                    _run_until_capture(model, sample)
                finally:
                    target_hook.remove()

            if len(record["kwargs"]) != end - start or "target" not in record or "input" not in record:
                raise RuntimeError("Incomplete window capture")
            records.append(record)
    finally:
        for module, enabled in prior_states:
            module.quant_enabled = enabled
    return records


def replay_window(layers, record, start, end, device, gradient_checkpointing=False):
    hidden = record["input"].to(device)
    for index in range(start, end):
        kwargs = _tree_to(record["kwargs"][index], device)
        block = layers[index]

        def forward(value, block=block, kwargs=kwargs):
            return _hidden(block(value, **kwargs))

        if gradient_checkpointing and torch.is_grad_enabled():
            hidden = checkpoint(forward, hidden, use_reentrant=False)
        else:
            hidden = forward(hidden)
    return hidden


def _loss_numerator(output, record, *, reconstruction_loss="mse", huber_delta=1.0):
    """Sum FP32 element losses over valid tokens; the caller supplies the denominator."""
    _validate_reconstruction_loss(reconstruction_loss, huber_delta)
    target = record["target"].to(output.device)
    if output.shape != target.shape:
        raise ValueError("Replayed output and teacher target shapes differ")
    mask = record["mask"].to(output.device).unsqueeze(-1)
    if reconstruction_loss == "huber":
        element_loss = F.huber_loss(output.float(), target.float(), reduction="none", delta=huber_delta)
    else:
        element_loss = (output.float() - target.float()).square()
    return (element_loss * mask).sum()


def _elements(record):
    return int(record["mask"].sum()) * record["target"].shape[-1]


@torch.no_grad()
def window_mse(layers, records, start, end, device):
    """Measure actual MSE regardless of the objective selected for training."""
    denominator = sum(_elements(record) for record in records)
    if denominator == 0:
        raise ValueError("Calibration batch has no nonpadding tokens")
    return sum(float(_loss_numerator(replay_window(layers, r, start, end, device), r)) for r in records) / denominator


def train(model, samples, config: TrainingConfig, progress_callback=None, cache_directory=None):
    """Optimize group factors and low-rank corrections; return an audit log.

    Batch size is implemented by gradient accumulation over samples to avoid
    incorrectly batching multi-axis rotary tensors and hybrid attention masks.
    Earlier frozen windows remain hard-quantized; overlapping windows warm-start
    factors and restart their soft-to-hard schedule. Original weights never train.
    Phase times include cache I/O and progress callbacks; their aggregate
    excludes initial model/configuration setup.
    """
    config.validate()
    if not samples:
        raise ValueError("No calibration samples")
    layers = get_text_layers(model)
    if config.teacher_prefix == "original":
        _original_teacher_extra_modules(model, layers, len(layers))
    all_modules = _module_values(model)
    decoder_modules = {id(module) for layer in layers for module in layer.modules()}
    if any(id(module) not in decoder_modules for module in all_modules):
        raise ValueError("Embedding/head quantizers require explicit endpoint calibration; decoder windows do not train them")
    if any(getattr(module, 'rotation_only', False) for module in all_modules):
        raise ValueError("rotation_only is a floating-point diagnostic, not a calibration mode")
    if not all_modules:
        raise ValueError("No ternary projections; call replace_projections first")
    total_steps = config.epochs * ceil(len(samples) / config.batch_size)
    if any(module.hard_phase == "frozen_jacobian" and 1 / total_steps > module.soft_fraction
           for module in all_modules):
        raise ValueError("The schedule needs at least one soft update before the hard phase; "
                         "increase epochs or use a smaller batch size")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for module in all_modules:
        module.clear_soft_snapshot()
        module.set_progress(1.0)
    set_quant_enabled(model, True)
    loss_metadata = {"reconstruction_loss": config.reconstruction_loss,
                     "huber_delta": config.huber_delta if config.reconstruction_loss == "huber" else None}
    report = {"training_config": asdict(config), "windows": [], "samples": len(samples),
              "reconstruction_objective": {
                  "name": config.reconstruction_loss, "huber_delta": loss_metadata["huber_delta"],
                  "reduction": "mean over nonpadding hidden-state elements",
                  "teacher_prefix": config.teacher_prefix,
                  "student_input": "frozen hard-quantized prefix output",
                  "teacher_target": (
                      "original-weight window on the same quantized-prefix input"
                      if config.teacher_prefix == "quantized" else
                      "original-weight window on original-weight prefix input"),
              },
              "nonpadding_tokens": sum(int(x["attention_mask"].sum()) for x in samples),
              "quantization": {
                  "lora_ranks": sorted({module.lora_rank for module in all_modules}),
                  "factor_mappings": sorted({module.factor_mapping for module in all_modules}),
                  "lora_scaling": "1/rank; correction merged before ternarization",
                  "hard_phases": sorted({module.hard_phase for module in all_modules}),
                  "frozen_jacobian_reference": "last actual soft forward, before its optimizer update",
                  "factor_parameters": sum(p.numel() for m in all_modules for p in m.factor_parameters()),
                  "lora_parameters": sum(p.numel() for m in all_modules for p in m.lora_parameters()),
              },
              "quality_claim": "software execution only; no full-model benchmark reproduction"}
    device = _device(model)
    if cache_directory is not None:
        Path(cache_directory).mkdir(parents=True, exist_ok=True)
    for window_id, (start, end) in enumerate(build_windows(len(layers), config.window_size, config.stride)):
        window_started = _phase_time(device)
        active = [module for layer in layers[start:end] for module in layer.modules()
                  if module in all_modules]
        if not active:
            raise RuntimeError(f"No trainable projections in window {start}:{end}")
        # A per-window temporary directory removes caches after a completed
        # window. On an exception TemporaryDirectory still owns its cleanup.
        cache_owner = (tempfile.TemporaryDirectory(prefix=f"window-{window_id}-", dir=cache_directory)
                       if cache_directory is not None else None)
        records = cache_window(model, samples, start, end, cache_owner.name if cache_owner else None,
                               teacher_prefix=config.teacher_prefix)
        cache_finished = _phase_time(device)
        before = window_mse(layers, records, start, end, device)
        evaluation_before_finished = _phase_time(device)
        parameters = []
        for module in active:
            module.set_progress(0.0)
            for parameter in module.trainable_parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
        parameter_groups = [{"params": [p for m in active for p in m.factor_parameters()],
                             "lr": config.learning_rate, "initial_lr": config.learning_rate}]
        lora_parameters = [p for m in active for p in m.lora_parameters()]
        if lora_parameters:
            parameter_groups.append({"params": lora_parameters, "lr": config.lora_learning_rate,
                                     "initial_lr": config.lora_learning_rate})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
        total_steps = config.epochs * ceil(len(records) / config.batch_size)
        losses = []
        step = 0
        rng = random.Random(config.seed + window_id)
        for epoch in range(config.epochs):
            order = list(range(len(records)))
            rng.shuffle(order)
            for offset in range(0, len(order), config.batch_size):
                batch = [records[i] for i in order[offset:offset + config.batch_size]]
                for module in active:
                    module.set_progress((step + 1) / total_steps)
                # First update uses the stated LR; final positive update uses LR/steps.
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * (1.0 - step / total_steps)
                optimizer.zero_grad(set_to_none=True)
                denominator = sum(_elements(r) for r in batch)
                if denominator <= 0:
                    raise ValueError("All-padding batch")
                value = 0.0
                for record in batch:
                    output = replay_window(layers, record, start, end, device, config.gradient_checkpointing)
                    loss = _loss_numerator(output, record, reconstruction_loss=config.reconstruction_loss,
                                           huber_delta=config.huber_delta) / denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite reconstruction loss in window {start}:{end}")
                    loss.backward()
                    value += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                losses.append(value)
                step += 1
                if progress_callback:
                    progress_callback({"window": window_id, "layers": [start, end], "epoch": epoch + 1,
                                       "step": step, "total_steps": total_steps, "loss": value,
                                       "grad_norm": float(norm), **loss_metadata})
        optimization_finished = _phase_time(device)
        for module in active:
            for parameter in module.trainable_parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            module.clear_soft_snapshot()
            module.set_progress(1.0)
        after = window_mse(layers, records, start, end, device)
        evaluation_after_finished = _phase_time(device)
        del records, optimizer
        if cache_owner is not None:
            cache_owner.cleanup()
        window_finished = time.perf_counter()
        timing = {"cache": cache_finished - window_started,
                  "evaluation_before": evaluation_before_finished - cache_finished,
                  "optimization": optimization_finished - evaluation_before_finished,
                  "evaluation_after": evaluation_after_finished - optimization_finished,
                  "cleanup": window_finished - evaluation_after_finished,
                  "total": window_finished - window_started}
        report["windows"].append({"start": start, "end": end, "hard_mse_before": before,
                                  "hard_mse_after": after, "steps": step, "losses": losses,
                                  "timing_seconds": timing, **loss_metadata})
    report["timing_seconds"] = {name: sum(window["timing_seconds"][name] for window in report["windows"])
                                for name in report["windows"][0]["timing_seconds"]}
    return report
