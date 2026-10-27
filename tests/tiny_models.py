"""Tiny random Qwen3.5 fixtures for CPU tests; outputs say nothing about 9B quality.

Copied from the legacy adapter's ``tiny_model`` (now research/catq/adapter.py)
so tests/ never imports research code. The package step moves this into
per-adapter conftest fixtures.
"""
import torch
from transformers import (Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
                          Qwen3_5TextConfig)


def tiny_qwen3_5(seed=0, multimodal=False):
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
    text._attn_implementation = "eager"
    config, constructor = text, Qwen3_5ForCausalLM
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
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        config.text_config.use_cache = False
        constructor = Qwen3_5ForConditionalGeneration
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = constructor(config)
    model.eval()
    model.requires_grad_(False)
    return model
