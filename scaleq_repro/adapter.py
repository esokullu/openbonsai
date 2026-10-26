"""Native Transformers adapters for the dense Qwen3.5 architecture only.

The 9B model mixes recurrent and full attention. Keeping its native decoder
also preserves its gates, normalization, RoPE, and per-block mask semantics.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import (
    AutoConfig,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
)

from .quantizer import PackedTernaryLinear, TernaryLinear


_PROJECTIONS = {
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
}
_RECURRENT_GATES = {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}


def _text_module(model):
    if isinstance(model, Qwen3_5ForCausalLM) and model.config.model_type == "qwen3_5_text":
        return "model", model.model
    if isinstance(model, Qwen3_5ForConditionalGeneration) and model.config.model_type == "qwen3_5":
        return "model.language_model", model.model.language_model
    raise ValueError("Only native dense Qwen3.5 text or conditional-generation models are supported")


def get_text_layers(model):
    """Return the native text decoder ModuleList, never vision blocks."""
    return _text_module(model)[1].layers


def iter_ternary_modules(model):
    """Yield ``(fully_qualified_name, module)`` for trainable or packed modules."""
    for name, module in model.named_modules():
        if isinstance(module, (TernaryLinear, PackedTernaryLinear)):
            yield name, module


def replace_projections(model, group_size=128, aggressive_gates=False):
    """Replace large text projections; protect tiny recurrence gates by default.

    This projection policy is an experimental Qwen3.5 adaptation, not a claim
    that the original ScaleQ paper evaluated this model or this policy.
    """
    prefix, text_model = _text_module(model)
    allowlist = _PROJECTIONS | (_RECURRENT_GATES if aggressive_gates else set())
    candidates = []
    for index, layer in enumerate(text_model.layers):
        if layer.block_type not in {"linear_attention", "full_attention"}:
            raise ValueError(f"Unsupported decoder type at layer {index}: {layer.block_type}")
        for relative_name, module in layer.named_modules():
            if relative_name not in allowlist:
                continue
            if isinstance(module, (TernaryLinear, PackedTernaryLinear)):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at layer {index}.{relative_name}")
            candidates.append(f"{prefix}.layers.{index}.{relative_name}")
    # Keep only paths: holding the old Linear objects in a candidate list would
    # retain every original GPU weight while the quantizers clone them all.
    for name in candidates:
        source = model.get_submodule(name)
        replacement = TernaryLinear.from_linear(source, group_size=group_size)
        model.set_submodule(name, replacement)
        del source, replacement
    return candidates


def set_quant_enabled(model, enabled):
    modules = list(iter_ternary_modules(model))
    if not enabled and any(isinstance(module, PackedTernaryLinear) for _, module in modules):
        raise ValueError("A packed model has no original floating weights; quantization cannot be disabled")
    for _, module in modules:
        if isinstance(module, TernaryLinear):
            module.quant_enabled = bool(enabled)


def set_progress(model, progress, layer_indices=None):
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("progress must be finite and between 0 and 1")
    layers = get_text_layers(model)
    indices = list(range(len(layers))) if layer_indices is None else list(layer_indices)
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(layers) for i in indices):
        raise ValueError("layer_indices contains an invalid decoder index")
    modules = [module for i in indices for module in layers[i].modules()
               if isinstance(module, (TernaryLinear, PackedTernaryLinear))]
    if progress != 1 and any(isinstance(module, PackedTernaryLinear) for module in modules):
        raise ValueError("Packed models are fixed at the hard ternary endpoint")
    for module in modules:
        if isinstance(module, TernaryLinear):
            module.set_progress(float(progress))


def model_from_config(config):
    """Construct an explicitly supported native model without loading weights."""
    if config.model_type == "qwen3_5_text" and isinstance(config, Qwen3_5TextConfig):
        config._attn_implementation = "eager"
        config.use_cache = False
        return Qwen3_5ForCausalLM(config)
    if config.model_type == "qwen3_5" and isinstance(config, Qwen3_5Config):
        if config.text_config.model_type != "qwen3_5_text" or config.vision_config.model_type != "qwen3_5_vision":
            raise ValueError("Unsupported Qwen3.5 text or vision subconfiguration")
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        config.text_config.use_cache = False
        return Qwen3_5ForConditionalGeneration(config)
    raise ValueError(f"Unsupported model_type: {getattr(config, 'model_type', None)!r}")


def config_from_dict(data):
    model_type = data.get("model_type")
    if model_type == "qwen3_5_text":
        return Qwen3_5TextConfig.from_dict(data)
    if model_type == "qwen3_5":
        return Qwen3_5Config.from_dict(data)
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def load_base_model(model_id, device="cpu", dtype="auto", *, revision="main", local_files_only=False):
    """Load only the two supported native model families; no remote Python code."""
    config = AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only, trust_remote_code=False,
    )
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("Reconstruction requires original unquantized target weights")
    if config.model_type == "qwen3_5_text":
        constructor = Qwen3_5ForCausalLM
    elif config.model_type == "qwen3_5":
        if config.text_config.model_type != "qwen3_5_text" or config.vision_config.model_type != "qwen3_5_vision":
            raise ValueError("Unsupported Qwen3.5 subconfiguration")
        constructor = Qwen3_5ForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type!r}; expected dense Qwen3.5")
    if isinstance(dtype, str) and dtype != "auto":
        choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in choices:
            raise ValueError("dtype must be auto, float32, float16, bfloat16, or a torch.dtype")
        dtype = choices[dtype]
    model = constructor.from_pretrained(
        model_id, config=config, dtype=dtype, attn_implementation="eager", trust_remote_code=False,
        revision=getattr(config, "_commit_hash", None) or revision, local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    _text_module(model)[1].config.use_cache = False
    model.requires_grad_(False)
    return model


def tiny_model(seed=0, multimodal=False):
    """Random native smoke fixture. Its outputs say nothing about 9B quality."""
    text = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True},
        use_cache=False, eos_token_id=2, pad_token_id=0,
    )
    config = text
    if multimodal:
        config = Qwen3_5Config(
            text_config=text,
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 128, "patch_size": 2,
                           "spatial_merge_size": 2, "temporal_patch_size": 2,
                           "num_position_embeddings": 16},
            image_token_id=252, video_token_id=253, vision_start_token_id=254,
            vision_end_token_id=255,
        )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = model_from_config(config)
    model.eval()
    model.requires_grad_(False)
    return model
