"""Extract a native language-only export/reference view of original Qwen3.5.

The returned modules and tensor objects are independent, but frozen tensor
storage is shared. This is not an independently mutable fine-tuning clone.
Extract before calibration; use existing mixed export for the resulting model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib

import torch
import transformers
from accelerate import init_empty_weights
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from .adapter import model_from_config
from .export import _all_tensors, _json_bytes, _tensor_description


def text_module_name(name: str) -> str:
    """Map an original language module/tensor path; reject vision/unknown paths."""
    if not isinstance(name, str) or not name or not all(name.split(".")):
        raise ValueError("Expected a nonempty language/head path without empty components")
    if name == "model.language_model":
        return "model"
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name == "lm_head" or name.startswith("lm_head."):
        return name
    raise ValueError(f"Not an original Qwen3.5 language/head path: {name!r}")


@dataclass
class TextExtraction:
    """Frozen native text view and provenance suitable for mixed export.

    Neither the source nor this view may receive in-place weight updates while
    storage is shared. Replacing view modules for calibration is supported;
    modifying their shared original weight masters is not.
    """
    model: torch.nn.Module
    metadata: dict
    tensor_map: dict[str, str]


def _config_hash(config):
    return hashlib.sha256(_json_bytes(config.to_dict())).hexdigest()


def _check_source(source):
    if (type(source) is not Qwen3_5ForConditionalGeneration
            or type(source.model) is not Qwen3_5Model
            or type(source.config) is not Qwen3_5Config
            or type(source.config.text_config) is not Qwen3_5TextConfig):
        raise ValueError("Expected an original native Qwen3.5 conditional-generation model")
    configs = (source.config, source.config.text_config, source.model.language_model.config)
    if any(getattr(c, "quantization_config", None) is not None for c in configs):
        raise ValueError("Text extraction requires original unquantized weights")
    if any(getattr(c, "tie_word_embeddings", False) for c in configs):
        raise ValueError("Tied embedding/head configurations are unsupported")
    text_config = source.config.text_config
    if _json_bytes(text_config.to_dict()) != _json_bytes(source.model.language_model.config.to_dict()):
        raise ValueError("Loaded text configuration disagrees with source configuration")
    if (text_config._attn_implementation != "eager"
            or source.model.language_model.config._attn_implementation != "eager"
            or text_config.use_cache):
        raise ValueError("Initial export scope requires eager attention and text config use_cache=False")
    if text_config.rope_parameters.get("rope_type") != "default":
        raise ValueError("Initial text extraction supports default RoPE only")
    if source.model.rope_deltas is not None:
        raise ValueError("Extract a fresh source before generation; rope_deltas must be None")
    # Do not erase request state or hide a custom execution path during export.
    for name, module in source.named_modules():
        if module.training:
            raise ValueError(f"Text extraction requires eval mode: {name}")
        if ("forward" in vars(module) or hasattr(module, "_hf_hook")
                or module._forward_hooks or module._forward_pre_hooks
                or module._backward_hooks or module._backward_pre_hooks):
            raise ValueError(f"Custom forward/hooks/offload are unsupported: {name}")
    if any(p.requires_grad for p in source.parameters()):
        raise ValueError("Text extraction requires frozen original parameters")
    if set(source._modules) != {"model", "lm_head"} or set(source.model._modules) != {"visual", "language_model"}:
        raise ValueError("Unexpected source module roots")


def extract_text_model(source, *, source_id: str, source_revision: str) -> TextExtraction:
    """Make a memory-efficient native text view, excluding only the vision tower.

    ``source_id`` and ``source_revision`` identify the original BF16 checkpoint
    (including fine-tunes); a local content digest may serve as its revision.
    These are caller-supplied provenance, not an independently verified download.
    FP32 native fixtures are also supported; individual tensor dtypes are never
    cast. Mixed export still requires original BF16 candidate weights.

    Certified inputs are fresh text token requests with the same masks/positions,
    eager/default RoPE and explicit generation length limits. No image/video,
    multimodal continuation, or transferred request cache is supported. The native
    CausalLM accepts **kwargs: callers must enforce this text-only input contract.
    Fresh native dynamic-cache generation is tested separately from no-KV forward.
    """
    if not all(isinstance(v, str) and v.strip() for v in (source_id, source_revision)):
        raise ValueError("Nonempty original source_id and source_revision are required")
    _check_source(source)
    original = dict((name, (tensor, kind, persistent))
                    for name, tensor, kind, persistent in _all_tensors(source))
    mapping = {}
    storage_owners = {}
    for name, (tensor, kind, _) in original.items():
        if tensor.is_meta or tensor.layout != torch.strided:
            raise ValueError(f"Expected materialized strided source tensor: {name}")
        if tensor.numel():
            key = (tensor.device, tensor.untyped_storage().data_ptr())
            storage_owners.setdefault(key, []).append((name, kind))
        if not name.startswith("model.visual."):
            mapping[name] = text_module_name(name)
    for owners in storage_owners.values():
        # Native rotary buffer aliases are harmless; parameter aliasing would
        # change the independent module replacement/export contract.
        if len(owners) > 1 and any(kind == "parameter" for _, kind in owners):
            raise ValueError(f"Tied or aliased parameter storage is unsupported: {owners}")

    config = copy.deepcopy(source.config.text_config)
    config.architectures = ["Qwen3_5ForCausalLM"]
    with torch.random.fork_rng(devices=[]), no_init_weights(), init_empty_weights(include_buffers=False):
        model = model_from_config(config)
    expected_modules = dict(model.named_modules())
    source_modules = {text_module_name(n): m for n, m in source.named_modules()
                      if n == "model.language_model" or n.startswith("model.language_model.")
                      or n == "lm_head" or n.startswith("lm_head.")}
    if set(source_modules) != set(expected_modules) - {""}:
        raise ValueError("Original language module topology differs from native text model")
    for name, module in source_modules.items():
        target = expected_modules[name]
        if type(module) is not type(target):
            raise ValueError(f"Expected original native module at {name}")
        if isinstance(module, torch.nn.Embedding):
            attrs = ("padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse")
            if any(getattr(module, attr) != getattr(target, attr) for attr in attrs):
                raise ValueError(f"Embedding semantics differ from native configuration: {name}")

    expected = {n: (t, k, p) for n, t, k, p in _all_tensors(model)}
    if set(mapping.values()) != set(expected):
        raise ValueError("Original language tensor names do not cover native text state exactly")
    for name, dest in mapping.items():
        tensor, kind, persistent = original[name]
        target, target_kind, target_persistent = expected[dest]
        if (tensor.shape != target.shape or (kind, persistent) != (target_kind, target_persistent)):
            raise ValueError(f"Original tensor shape/kind/persistence mismatch: {name}")

    # Detached wrappers are essential: assign=True with source Parameter objects
    # could reuse the objects themselves and mutate source requires_grad flags.
    state = {mapping[n]: t.detach() for n, (t, _, p) in original.items() if n in mapping and p}
    model.load_state_dict(state, strict=True, assign=True)
    for name, dest in mapping.items():
        tensor, kind, persistent = original[name]
        if kind == "buffer" and not persistent:
            parent, _, attr = dest.rpartition(".")
            model.get_submodule(parent)._buffers[attr] = tensor.detach()
    model.generation_config = copy.deepcopy(source.generation_config)
    model.eval().requires_grad_(False)

    transferred = {n: (t, k, p) for n, t, k, p in _all_tensors(model)}
    descriptions = {}
    for name, dest in mapping.items():
        tensor, kind, persistent = original[name]
        result, new_kind, new_persistent = transferred[dest]
        if (tensor is result or result.dtype != tensor.dtype or result.device != tensor.device
                or result.stride() != tensor.stride() or result.storage_offset() != tensor.storage_offset()
                or result.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
                or (kind, persistent) != (new_kind, new_persistent)):
            raise RuntimeError(f"Text extraction did not preserve detached source storage: {name}")
        descriptions[name] = {"target": dest, **_tensor_description(tensor),
                              "kind": kind, "persistent": persistent}
    metadata = {
        "source_checkpoint": source_id, "source_revision": source_revision,
        "text_extraction": {
            "format": "openbonsai.native-text-extraction.v1", "scope": "language-only",
            "source_config_sha256": _config_hash(source.config),
            "text_config_sha256": _config_hash(model.config),
            "generation_config_sha256": _config_hash(source.generation_config),
            "attention_backend": "eager", "torch": torch.__version__,
            "transformers": transformers.__version__, "shared_frozen_storage": True,
            "tensors": descriptions,
            "language_tensor_bytes": sum(d["tensor_bytes"] for d in descriptions.values()),
            "excluded_vision_tensor_bytes": sum(t.numel() * t.element_size()
                for n, (t, _, _) in original.items() if n.startswith("model.visual.")),
            "quality_claim": False,
        },
    }
    return TextExtraction(model, metadata, mapping)
