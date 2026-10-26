import json
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file

from scaleq_repro import adapter
from scaleq_repro.adapter import (
    get_text_layers, iter_ternary_modules, load_base_model, replace_projections,
    set_progress, set_quant_enabled, tiny_model,
)
from scaleq_repro.export import load_compressed, save_compressed
from scaleq_repro.quantizer import PackedTernaryLinear, TernaryLinear


def logits(model):
    with torch.no_grad():
        return model(input_ids=torch.tensor([[3, 4, 5, 6, 7]]), use_cache=False).logits


def test_native_hybrid_projection_policy_and_dense_teacher():
    model = tiny_model(seed=13)
    expected = logits(model)
    assert [block.block_type for block in get_text_layers(model)] == ["linear_attention"] * 3 + ["full_attention"]
    names = replace_projections(model)
    assert len(names) == 25
    assert len(list(iter_ternary_modules(model))) == 25
    assert all(name.startswith("model.layers.") for name in names)
    assert isinstance(model.model.layers[0].linear_attn.in_proj_a, nn.Linear)
    assert isinstance(model.model.layers[0].linear_attn.in_proj_b, nn.Linear)
    assert isinstance(model.model.layers[0].linear_attn.conv1d, nn.Conv1d)
    assert isinstance(model.lm_head, nn.Linear)
    set_progress(model, 1)
    set_quant_enabled(model, False)
    torch.testing.assert_close(logits(model), expected, rtol=0, atol=0)
    assert replace_projections(model) == []
    assert len(replace_projections(model, aggressive_gates=True)) == 6


def test_layer_progress_changes_only_selected_blocks_and_validates_indices():
    model = tiny_model()
    replace_projections(model)
    set_progress(model, 0.3, layer_indices=[1])
    for index, block in enumerate(get_text_layers(model)):
        values = [module.progress for module in block.modules() if isinstance(module, TernaryLinear)]
        assert values and all(value == (0.3 if index == 1 else 0) for value in values)
    for indices in ([-1], [4], [True], [0.5]):
        with pytest.raises(ValueError, match="decoder index"):
            set_progress(model, 0.5, layer_indices=indices)
    for progress in (float("nan"), -0.1, 1.1):
        with pytest.raises(ValueError, match="progress"):
            set_progress(model, progress)
    with pytest.raises(ValueError, match="Only native"):
        replace_projections(nn.Linear(4, 4))


def test_projection_conversion_releases_each_original_before_the_next(monkeypatch):
    model = tiny_model()
    original_from_linear = TernaryLinear.from_linear
    originals = []

    def checking_from_linear(source, **kwargs):
        # A weak reference proves collection rather than estimating CPU/GPU
        # allocator behavior, which may retain released blocks for later reuse.
        assert all(reference() is None for reference in originals)
        originals.append(weakref.ref(source))
        return original_from_linear(source, **kwargs)

    monkeypatch.setattr(TernaryLinear, "from_linear", checking_from_linear)
    assert len(replace_projections(model)) == 25
    assert all(reference() is None for reference in originals)


@pytest.mark.parametrize("multimodal", [False, True])
def test_standalone_export_preserves_floats_and_roundtrip_logits(tmp_path, multimodal):
    model = tiny_model(seed=7, multimodal=multimodal)
    names = replace_projections(model)
    set_progress(model, 1)
    expected = logits(model)
    artifact = tmp_path / "packed"
    manifest = save_compressed(model, artifact, metadata={"test_fixture": True})
    loaded = load_compressed(artifact)
    actual = logits(loaded)
    # Export deliberately stores scales in fp16. The test bounds only that
    # round-trip error, not ternarization quality against the original model.
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=5e-4)
    assert len(list(iter_ternary_modules(loaded))) == 25
    assert all(isinstance(module, PackedTernaryLinear) for _, module in iter_ternary_modules(loaded))
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert manifest["optimized_kernels"] is False and manifest["gguf"] is False
    assert manifest["metadata"] == {"test_fixture": True}
    assert manifest["storage"]["artifact_bytes"] == sum(path.stat().st_size for path in artifact.iterdir())
    assert manifest["storage"]["manifest_bytes"] == (artifact / "manifest.json").stat().st_size
    assert json.loads((artifact / "manifest.json").read_text()) == manifest

    original_tensors = dict(model.named_parameters(remove_duplicate=False))
    original_tensors.update(model.named_buffers(remove_duplicate=False))
    loaded_tensors = dict(loaded.named_parameters(remove_duplicate=False))
    loaded_tensors.update(loaded.named_buffers(remove_duplicate=False))
    expected_retained = {name for name in original_tensors
                         if not any(name.startswith(prefix + ".") for prefix in names)}
    assert set(manifest["retained_tensors"]) == expected_retained
    assert any(not item["persistent"] for item in manifest["retained_tensors"].values())
    for name in expected_retained:
        torch.testing.assert_close(loaded_tensors[name], original_tensors[name], rtol=0, atol=0)
        assert loaded_tensors[name].dtype == original_tensors[name].dtype
    if multimodal:
        assert any(name.startswith("model.visual.") for name in expected_retained)
        assert all(name.startswith("model.language_model.layers.") for name in names)
    assert not any("raw_scale" in name or "mu0" in name for name in manifest["retained_tensors"])

    for info in manifest["packed_modules"].values():
        payload = load_file(str(artifact / info["file"]))
        state = info["state"]
        assert payload["codes"].numel() == (state["padded_numel"] + 3) // 4
        assert payload["scales"].numel() == state["num_groups"]
        assert payload["codes"].dtype == torch.uint8
        assert payload["scales"].dtype == torch.float16
        assert "weight" not in payload

    with pytest.raises(ValueError, match="cannot be disabled"):
        set_quant_enabled(loaded, False)
    with pytest.raises(ValueError, match="hard ternary endpoint"):
        set_progress(loaded, 0.2)
    set_progress(loaded, 1)

    # A packed artifact remains standalone when re-exported, with exact logits.
    again = tmp_path / "again"
    save_compressed(loaded, again)
    torch.testing.assert_close(logits(load_compressed(again)), actual, rtol=0, atol=0)


def test_export_refuses_overwrite_and_detects_corruption(tmp_path):
    model = tiny_model()
    replace_projections(model)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "user-work.txt"
    sentinel.write_text("keep this")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_compressed(model, destination)
    assert sentinel.read_text() == "keep this"
    artifact = tmp_path / "packed"
    manifest = save_compressed(model, artifact)
    filename = next(iter(manifest["packed_modules"].values()))["file"]
    data = bytearray((artifact / filename).read_bytes())
    data[-1] ^= 1
    (artifact / filename).write_bytes(data)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_compressed(artifact)


def test_native_loader_pins_resolved_revision_without_remote_code(monkeypatch):
    model = tiny_model()
    config = model.config
    config._commit_hash = "abc123resolved"
    calls = {}

    def read_config(model_id, **kwargs):
        calls["config"] = (model_id, kwargs)
        return config

    def read_model(model_id, **kwargs):
        calls["model"] = (model_id, kwargs)
        return model

    monkeypatch.setattr(adapter.AutoConfig, "from_pretrained", read_config)
    monkeypatch.setattr(adapter.Qwen3_5ForCausalLM, "from_pretrained", read_model)
    loaded = load_base_model("fixture/model", dtype="float32", revision="reviewed", local_files_only=True)
    assert loaded is model
    assert calls["config"][1]["revision"] == "reviewed"
    assert calls["model"][1]["revision"] == "abc123resolved"
    assert calls["model"][1]["dtype"] == torch.float32
    assert all(kwargs["trust_remote_code"] is False and kwargs["local_files_only"] is True
               for _, kwargs in calls.values())
    config.quantization_config = {"quant_method": "compressed-tensors"}
    with pytest.raises(ValueError, match="original unquantized"):
        load_base_model("already-compressed/model", local_files_only=True)
    monkeypatch.setattr(adapter.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_moe"))
    with pytest.raises(ValueError, match="expected dense Qwen3.5"):
        load_base_model("unsupported/model", local_files_only=True)
