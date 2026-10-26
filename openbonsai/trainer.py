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
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from .adapter import get_text_layers, iter_ternary_modules, set_quant_enabled


@dataclass
class TrainingConfig:
    epochs: int = 60
    batch_size: int = 3
    window_size: int = 4
    stride: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 2
    gradient_checkpointing: bool = False

    def validate(self):
        for name in ("epochs", "batch_size", "window_size", "stride"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stride > self.window_size:
            raise ValueError("stride must not exceed window_size (would skip layers)")
        if (not all(isfinite(value) for value in (self.learning_rate, self.weight_decay, self.grad_clip))
                or self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0):
            raise ValueError("learning_rate/grad_clip must be finite and positive; weight_decay finite and nonnegative")


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


def _hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported block output {type(output).__name__}")


def _module_values(model):
    # Adapter returns (qualified_name, module) pairs.
    return [module for _, module in iter_ternary_modules(model)]


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
def cache_window(model, samples, start: int, end: int, cache_directory=None):
    """Cache one shared window input and its original-weight window target.

    Masks/rotary kwargs are captured per block. Qwen3.5 full-attention and
    GatedDeltaNet blocks must not share a blindly copied attention mask.
    The prefix is already quantized; teacher and student windows receive the
    same X (CAT-Q Eq. 7). All tensors are cached on CPU or optional disk.
    """
    layers = get_text_layers(model)
    if not 0 <= start < end <= len(layers):
        raise ValueError("Invalid capture window")
    prior_states = [(module, module.quant_enabled) for module in _module_values(model)]
    records = [] if cache_directory is None else DiskRecords(cache_directory)
    try:
        for sample in samples:
            record = {"kwargs": {}, "mask": sample["attention_mask"].detach().cpu()}
            hooks = []
            set_quant_enabled(model, True)
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

            try:
                for index in range(start, end):
                    hooks.append(layers[index].register_forward_pre_hook(capture_kwargs(index), with_kwargs=True))
                hooks.append(layers[end - 1].register_forward_hook(capture_target))
                _run_until_capture(model, sample)
            finally:
                for hook in hooks:
                    hook.remove()

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


def _loss_numerator(output, record):
    target = record["target"].to(output.device)
    if output.shape != target.shape:
        raise ValueError("Replayed output and teacher target shapes differ")
    mask = record["mask"].to(output.device).unsqueeze(-1)
    return ((output.float() - target.float()).square() * mask).sum()


def _elements(record):
    return int(record["mask"].sum()) * record["target"].shape[-1]


@torch.no_grad()
def window_mse(layers, records, start, end, device):
    denominator = sum(_elements(record) for record in records)
    if denominator == 0:
        raise ValueError("Calibration batch has no nonpadding tokens")
    return sum(float(_loss_numerator(replay_window(layers, r, start, end, device), r)) for r in records) / denominator


def train(model, samples, config: TrainingConfig, progress_callback=None, cache_directory=None):
    """Optimize modulation factors only; return a machine-readable audit log.

    Batch size is implemented by gradient accumulation over samples to avoid
    incorrectly batching multi-axis rotary tensors and hybrid attention masks.
    Earlier frozen windows remain hard-quantized; overlapping windows warm-start
    factors and restart their soft-to-hard schedule. Original weights never train.
    """
    config.validate()
    if not samples:
        raise ValueError("No calibration samples")
    model.eval()
    layers = get_text_layers(model)
    all_modules = _module_values(model)
    if not all_modules:
        raise ValueError("No ternary projections; call replace_projections first")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for module in all_modules:
        module.set_progress(1.0)
    set_quant_enabled(model, True)
    report = {"training_config": asdict(config), "windows": [], "samples": len(samples),
              "nonpadding_tokens": sum(int(x["attention_mask"].sum()) for x in samples),
              "quality_claim": "software execution only; no full-model benchmark reproduction"}
    device = _device(model)
    if cache_directory is not None:
        Path(cache_directory).mkdir(parents=True, exist_ok=True)
    for window_id, (start, end) in enumerate(build_windows(len(layers), config.window_size, config.stride)):
        active = [module for layer in layers[start:end] for module in layer.modules()
                  if module in all_modules]
        if not active:
            raise RuntimeError(f"No trainable projections in window {start}:{end}")
        # A per-window temporary directory removes caches after a completed
        # window. On an exception TemporaryDirectory still owns its cleanup.
        cache_owner = (tempfile.TemporaryDirectory(prefix=f"window-{window_id}-", dir=cache_directory)
                       if cache_directory is not None else None)
        records = cache_window(model, samples, start, end, cache_owner.name if cache_owner else None)
        before = window_mse(layers, records, start, end, device)
        parameters = []
        for module in active:
            module.set_progress(0.0)
            for parameter in module.trainable_parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
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
                    group["lr"] = config.learning_rate * (1.0 - step / total_steps)
                optimizer.zero_grad(set_to_none=True)
                denominator = sum(_elements(r) for r in batch)
                if denominator <= 0:
                    raise ValueError("All-padding batch")
                value = 0.0
                for record in batch:
                    output = replay_window(layers, record, start, end, device, config.gradient_checkpointing)
                    loss = _loss_numerator(output, record) / denominator
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
                                       "grad_norm": float(norm)})
        for module in active:
            module.set_progress(1.0)
            for parameter in module.trainable_parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        after = window_mse(layers, records, start, end, device)
        report["windows"].append({"start": start, "end": end, "hard_mse_before": before,
                                  "hard_mse_after": after, "steps": step, "losses": losses})
        del records, optimizer
        if cache_owner is not None:
            cache_owner.cleanup()
    return report
