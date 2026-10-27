"""Tiny random Qwen3.5 fixtures for CPU tests; outputs say nothing about 9B quality.

The text model has three Gated DeltaNet layers and one full-attention layer.
Every rotated width (hidden 128, MLP 256, value/head dims 128) is a multiple
of 128, so the derived Hadamard block is 128.
"""
import torch
from transformers import (Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
                          Qwen3_5TextConfig)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


def tiny_text_config(vocab_size=256, eos_token_id=2, pad_token_id=0, attn="eager", **overrides):
    params = dict(
        vocab_size=vocab_size, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True},
        use_cache=False, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )
    text = Qwen3_5TextConfig(**{**params, **overrides})
    text._attn_implementation = attn
    return text


def tiny_qwen3_5(seed=0, multimodal=False, *, vocab_size=256, eos_token_id=2, pad_token_id=0, attn="eager",
                 random_norms=False):
    text = tiny_text_config(vocab_size, eos_token_id, pad_token_id, attn)
    config, constructor = text, Qwen3_5ForCausalLM
    if multimodal:
        config = Qwen3_5Config(
            text_config=text,
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 128, "patch_size": 2,
                           "spatial_merge_size": 2, "temporal_patch_size": 2,
                           "num_position_embeddings": 16},
            image_token_id=vocab_size - 4, video_token_id=vocab_size - 3, vision_start_token_id=vocab_size - 2,
            vision_end_token_id=vocab_size - 1,
        )
        config._attn_implementation = attn
        config.text_config._attn_implementation = attn
        config.vision_config._attn_implementation = attn
        config.text_config.use_cache = False
        constructor = Qwen3_5ForConditionalGeneration
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = constructor(config)
        if random_norms:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name.endswith("layernorm.weight") or name.endswith("model.norm.weight") \
                            or name.endswith("language_model.norm.weight"):
                        p.normal_(0.0, 0.3)
    model.eval()
    model.requires_grad_(False)
    return model


def byte_tokenizer():
    """Byte-level BPE with no merges (ids 0..255 = bytes) plus Qwen-style special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from openbonsai.eval.backends import _byte_decoder
    vocab = {char: i for i, char in enumerate(_byte_decoder())}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=[t for t in SPECIAL if t not in ("<|im_end|>", "<|endoftext|>")])
    tok.chat_template = CHAT_TEMPLATE
    return tok


def save_tiny_qwen3_5(path, *, seed=0, multimodal=False, vocab_size=320):
    """Write a loadable tiny model directory (config, weights, tokenizer, chat template)."""
    tok = byte_tokenizer()
    eos, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    model = tiny_qwen3_5(seed, multimodal, vocab_size=vocab_size, eos_token_id=eos, pad_token_id=pad,
                         attn="sdpa", random_norms=True)
    model.generation_config.eos_token_id = eos
    model.generation_config.pad_token_id = pad
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path
