"""Shared-source native extraction tests; random fixtures are not 9B evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
from transformers import Qwen3_5ForCausalLM

from openbonsai.adapter import tiny_model
from openbonsai.candidates import source_weight_sha256
from openbonsai.mixed_export import prepare_mixed_export, save_mixed, load_mixed
from openbonsai.mixed_formats import MixedEmbedding, MixedLinear, ternary_candidate
from openbonsai.packing import pack_grouped_weight
from openbonsai.rotations import fixed_signs
from openbonsai.text_export import extract_text_model, text_module_name


SOURCE_ID = "random-native-extraction-fixture"
REVISION = "f" * 40


def native_source():
    torch.set_num_threads(2)
    model = tiny_model(seed=173, multimodal=True)
    model.resize_token_embeddings(260, mean_resizing=False)
    model.to(torch.bfloat16).eval().requires_grad_(False)
    # Retained native dynamics must not silently follow the large matrices' dtype.
    for layer in model.model.language_model.layers[:3]:
        layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
        layer.linear_attn.dt_bias.data = layer.linear_attn.dt_bias.data.float()
    with torch.no_grad():
        model.model.language_model.embed_tokens.weight[0].fill_(0.125)
    model.generation_config.eos_token_id = [2, 3]
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 1
    model.generation_config.max_new_tokens = 7
    model.generation_config.do_sample = False
    model.generation_config.use_cache = False
    return model


def tensors(model):
    return {**dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False))}


def extract(source):
    return extract_text_model(source, source_id=SOURCE_ID, source_revision=REVISION)


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name,expected", [
    ("model.language_model", "model"),
    ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
    ("model.language_model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.q_proj"),
    ("lm_head", "lm_head"), ("lm_head.weight", "lm_head.weight"),
])
def test_exact_module_prefix_mapping(name, expected):
    assert text_module_name(name) == expected


@pytest.mark.parametrize("name", ["model", "model.visual", "model.language_models.x",
                                   "model.language_model.", "model.language_model..layers",
                                   "lm_head.", "lm_head_extra.weight", ""])
def test_mapping_rejects_other_or_malformed_names(name):
    with pytest.raises(ValueError):
        text_module_name(name)


def test_extraction_preserves_source_and_all_tensor_layouts_without_dense_copy():
    source = native_source()
    before = {name: value.detach().clone() for name, value in tensors(source).items()}
    source_config = copy.deepcopy(source.config.to_dict())
    source_generation = copy.deepcopy(source.generation_config.to_dict())
    rng_before = torch.random.get_rng_state().clone()
    result = extract(source)
    assert_exact(torch.random.get_rng_state(), rng_before)
    text = result.model
    assert type(text) is Qwen3_5ForCausalLM
    assert text.model is not source.model.language_model
    assert text.lm_head is not source.lm_head
    assert not any(module.training for module in text.modules())
    assert not any(p.requires_grad for p in text.parameters())
    assert [layer.block_type for layer in text.model.layers] == ["linear_attention"] * 3 + ["full_attention"]
    source_tensors, target_tensors = tensors(source), tensors(text)
    expected_map = {name: text_module_name(name) for name in source_tensors
                    if name.startswith("model.language_model.") or name.startswith("lm_head.")}
    assert result.tensor_map == expected_map
    assert set(target_tensors) == set(expected_map.values())
    assert any("original_inv_freq" in name for name in expected_map)
    for old, new in expected_map.items():
        a, b = source_tensors[old], target_tensors[new]
        assert a is not b
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        assert a.storage_offset() == b.storage_offset() and a.stride() == b.stride()
        assert a.dtype == b.dtype
        assert_exact(b, before[old])
    for name, value in source_tensors.items():
        assert_exact(value, before[name])
    for name, module in text.named_modules():
        if name and (name == "model" or name.startswith("model.") or name.startswith("lm_head")):
            original_name = "model.language_model" + name[len("model"):] if name.startswith("model") else name
            assert module is not source.get_submodule(original_name)
    assert source.config.to_dict() == source_config
    assert source.generation_config.to_dict() == source_generation
    assert source.model.rope_deltas is None
    assert text.generation_config.to_dict() == source_generation
    expected_config = copy.deepcopy(source.config.text_config.to_dict())
    actual_config = copy.deepcopy(text.config.to_dict())
    expected_config.pop("architectures", None)
    actual_config.pop("architectures", None)
    assert actual_config == expected_config
    assert text.config is not source.config.text_config
    assert text.generation_config is not source.generation_config
    assert result.metadata["source_checkpoint"] == SOURCE_ID
    assert result.metadata["source_revision"] == REVISION
    json.dumps(result.metadata, allow_nan=False)
    text.generation_config.eos_token_id.append(4)
    assert source.generation_config.eos_token_id == [2, 3]


@pytest.mark.parametrize("positions", ["implicit", "explicit2", "explicit4"])
def test_shared_source_native_forward_with_padding_positions_and_full_vocabulary(positions):
    source = native_source()
    text = extract(source).model
    batches = [
        (torch.tensor([[4, 259, 6, 7, 8]]), torch.ones(1, 5, dtype=torch.long)),
        (torch.tensor([[0, 0, 4, 259, 6], [7, 8, 9, 0, 0]]),
         torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])),
    ]
    for ids, mask in batches:
        kwargs = dict(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        if positions != "implicit":
            pos = torch.arange(ids.shape[1]).expand_as(ids) + torch.arange(ids.shape[0])[:, None] * 11 + 7
            kwargs["position_ids"] = pos if positions == "explicit2" else pos.unsqueeze(0).expand(4, -1, -1)
        with torch.no_grad():
            expected, actual = source(**kwargs), text(**kwargs)
        assert expected.logits.shape[-1] == actual.logits.shape[-1] == 260
        assert_exact(actual.logits, expected.logits)
        assert_exact(actual.hidden_states[-1], expected.hidden_states[-1])
        assert torch.isfinite(actual.logits[mask.bool()]).all()


def generate(model, ids, mask, *, cache, eos, maximum=3):
    config = copy.deepcopy(model.generation_config)
    config.eos_token_id = eos
    config.max_new_tokens = maximum
    config.use_cache = cache
    with torch.no_grad():
        return model.generate(input_ids=ids, attention_mask=mask, generation_config=config,
                              return_dict_in_generate=True, output_logits=True)


@pytest.mark.parametrize("cache", [False, True])
def test_native_generation_eos_length_and_independent_calls(cache):
    source = native_source()
    text = extract(source).model
    ids = torch.tensor([[0, 0, 4, 259, 6], [7, 8, 9, 10, 11]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    # No EOS gives a deterministic cap and exercises all incremental steps.
    expected = generate(source, ids, mask, cache=cache, eos=None)
    actual = generate(text, ids, mask, cache=cache, eos=None)
    assert expected.sequences.shape[1] == ids.shape[1] + 3
    assert_exact(actual.sequences, expected.sequences)
    assert len(expected.logits) == len(actual.logits) == 3
    for a, b in zip(actual.logits, expected.logits):
        assert_exact(a, b)
    other = torch.tensor([[12, 13]])
    generate(source, other, torch.ones_like(other), cache=cache, eos=None)
    generate(text, other, torch.ones_like(other), cache=cache, eos=None)
    source.model.rope_deltas = None
    repeated = generate(source, ids, mask, cache=cache, eos=None)
    replayed = generate(text, ids, mask, cache=cache, eos=None)
    assert_exact(repeated.sequences, expected.sequences)
    assert_exact(replayed.sequences, expected.sequences)
    for a, b in zip(replayed.logits, repeated.logits):
        assert_exact(a, b)
    first = int(expected.sequences[1, ids.shape[1]])
    eos = [(first + 1) % 260, first]
    one, one_mask = ids[1:], mask[1:]
    stopped_source = generate(source, one, one_mask, cache=cache, eos=eos)
    stopped_text = generate(text, one, one_mask, cache=cache, eos=eos)
    assert stopped_source.sequences.shape[1] == one.shape[1] + 1
    assert int(stopped_text.sequences[0, -1]) == first == eos[1]
    assert_exact(stopped_text.sequences, stopped_source.sequences)


@pytest.mark.parametrize("invalid", ["training", "child-training", "trainable", "meta",
    "root-tie", "text-tie", "actual-alias", "zero-rope-state", "nonzero-rope-state",
    "nondefault-rope", "sdpa", "text-source", "missing-tensor", "wrong-shape"])
def test_rejects_sources_outside_frozen_native_extraction_contract(invalid):
    source = native_source()
    if invalid == "training":
        source.train()
    elif invalid == "child-training":
        source.model.language_model.layers[0].train()
    elif invalid == "trainable":
        source.lm_head.weight.requires_grad_(True)
    elif invalid == "meta":
        source.lm_head.weight = nn.Parameter(torch.empty_like(source.lm_head.weight, device="meta"), requires_grad=False)
    elif invalid == "root-tie":
        source.config.tie_word_embeddings = True
    elif invalid == "text-tie":
        source.config.text_config.tie_word_embeddings = True
    elif invalid == "actual-alias":
        source.lm_head.weight = source.model.language_model.embed_tokens.weight
    elif invalid in ("zero-rope-state", "nonzero-rope-state"):
        source.model.rope_deltas = torch.full((1, 1), int(invalid == "nonzero-rope-state"))
    elif invalid == "nondefault-rope":
        source.config.text_config.rope_parameters["rope_type"] = "linear"
        source.config.text_config.rope_parameters["factor"] = 2.0
    elif invalid == "sdpa":
        source.config.text_config._attn_implementation = "sdpa"
    elif invalid == "text-source":
        source = tiny_model().eval().requires_grad_(False)
    elif invalid == "missing-tensor":
        del source.model.language_model.layers[0].linear_attn._parameters["dt_bias"]
    elif invalid == "wrong-shape":
        source.lm_head.weight = nn.Parameter(source.lm_head.weight[:-1], requires_grad=False)
    with pytest.raises((ValueError, TypeError)):
        extract(source)


def test_mapped_rotated_candidates_export_and_fresh_process_reload(tmp_path):
    source = native_source()
    extraction = extract(source)
    text = extraction.model
    original_names = ["model.language_model.embed_tokens", "lm_head",
                      "model.language_model.layers.0.mlp.down_proj",
                      "model.language_model.layers.3.self_attn.o_proj"]
    candidates = {}
    for index, old in enumerate(original_names):
        weight = source.get_submodule(old).weight
        rows, width = weight.shape
        codes = ((torch.arange(weight.numel()).reshape(rows, width) + index) % 3 - 1).to(torch.int8)
        state = pack_grouped_weight(codes, torch.full((weight.numel() // 128,), 0.015625),
                                    shape=weight.shape, group_size=128, packing="base3")
        candidate = ternary_candidate(state, packing="base3", module_kind="embedding" if index == 0 else "linear",
            provenance={"source_weight_sha256": source_weight_sha256(weight)},
            rotation_signs=fixed_signs(width, seed=91 + index), rotation_block_size=64)
        candidates[text_module_name(old)] = candidate
    output = tmp_path / "language"
    with prepare_mixed_export(text, candidates, metadata=extraction.metadata, row_chunk_size=17) as plan:
        manifest = save_mixed(plan, candidates, output, budget_bytes=10**8)
    assert int(manifest["storage"]["vision_file_bytes"]) == 0
    assert int(manifest["storage"]["artifact_bytes"]) == sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert not any("visual" in n or "language_model" in n for n in manifest["retained_tensors"])
    assert "model.rotary_emb.inv_freq" in manifest["retained_tensors"]
    assert "model.rotary_emb.original_inv_freq" in manifest["retained_tensors"]
    assert json.loads((output / "config.json").read_text())["model_type"] == "qwen3_5_text"
    for old in original_names:
        name, candidate = text_module_name(old), candidates[text_module_name(old)]
        assert manifest["modules"][name]["source_weight_sha256"] == source_weight_sha256(source.get_submodule(old).weight)
        cls = MixedEmbedding if candidate.module_kind == "embedding" else MixedLinear
        options = {"padding_idx": 0} if candidate.module_kind == "embedding" else {}
        source.set_submodule(old, cls(candidate, row_chunk_size=17, **options))
        text.set_submodule(name, cls(candidate, row_chunk_size=17, **options))
    ids = torch.tensor([[4, 259, 6, 7, 8]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        expected = source(input_ids=ids, attention_mask=mask, use_cache=False).logits
        assert_exact(text(input_ids=ids, attention_mask=mask, use_cache=False).logits, expected)
    expected_generation = generate(text, ids, mask, cache=False, eos=None).sequences
    payload = tmp_path / "fresh.pt"
    script = """
import copy, sys, torch
from openbonsai.mixed_export import load_mixed
torch.set_num_threads(2)
m = load_mixed(sys.argv[1]); ids = torch.tensor([[4,259,6,7,8]]); mask = torch.ones_like(ids)
g = copy.deepcopy(m.generation_config); g.eos_token_id = None; g.max_new_tokens = 3; g.use_cache = False
with torch.no_grad():
    logits = m(input_ids=ids,attention_mask=mask,use_cache=False).logits
    generated = m.generate(input_ids=ids,attention_mask=mask,generation_config=g)
torch.save({'logits':logits,'generated':generated,'eos':m.generation_config.eos_token_id,
            'tensors':{**dict(m.named_parameters()),**dict(m.named_buffers())}},sys.argv[2])
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                       CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-c", script, str(output), str(payload)],
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    restored = torch.load(payload, weights_only=True)
    assert_exact(restored["logits"], expected)
    assert_exact(restored["generated"], expected_generation)
    assert restored["eos"] == [2, 3]
    retained = manifest["retained_tensors"]
    for name in retained:
        assert_exact(restored["tensors"][name], tensors(text)[name])
        assert restored["tensors"][name].dtype == tensors(text)[name].dtype
    loaded = load_mixed(output)
    assert type(loaded) is Qwen3_5ForCausalLM
    assert not hasattr(loaded.model, "visual")
