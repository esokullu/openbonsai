"""Native hybrid integration, not a benchmark of pretrained-model quality."""
import math

import pytest
import torch

from openbonsai.adapter import tiny_model, replace_projections, get_text_layers, iter_ternary_modules, set_progress
from openbonsai.trainer import TrainingConfig, build_windows, cache_window, replay_window, train


@pytest.fixture
def model():
    torch.set_num_threads(2)
    value = tiny_model(seed=3)
    replace_projections(value)
    set_progress(value, 1)
    return value


def samples():
    return [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 0, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])}]


def test_hybrid_masks_and_shared_window_input(model):
    layers = get_text_layers(model)
    records = cache_window(model, samples(), 1, 4)
    record = records[0]
    assert record["kwargs"][1]["attention_mask"].ndim == 2
    assert record["kwargs"][3]["attention_mask"].ndim == 4
    for layer in layers[1:4]:
        for module in layer.modules():
            if hasattr(module, "quant_enabled"):
                module.quant_enabled = False
    with torch.no_grad():
        replay = replay_window(layers, record, 1, 4, "cpu")
    torch.testing.assert_close(replay, record["target"], atol=0, rtol=0)


def test_disk_cache_matches_memory_and_removes_hooks(model, tmp_path):
    before = sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())
    memory = cache_window(model, samples(), 0, 4)
    disk = cache_window(model, samples(), 0, 4, tmp_path)
    assert len(disk) == 1
    torch.testing.assert_close(memory[0]["target"], disk[0]["target"], atol=0, rtol=0)
    assert all(module.quant_enabled for _, module in iter_ternary_modules(model))
    assert before == sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_train_factors_only_with_overlap(model, tmp_path, checkpointing):
    base = {name: tensor.clone() for name, tensor in model.named_buffers()
            if not name.rsplit(".", 1)[-1].startswith("_hard_snapshot")}
    factors = {name: p.clone() for name, p in model.named_parameters() if ".raw_" in name}
    report = train(model, samples(), TrainingConfig(epochs=3, batch_size=1, window_size=3,
                 stride=2, gradient_checkpointing=checkpointing), cache_directory=tmp_path)
    assert [(w["start"], w["end"]) for w in report["windows"]] == [(0, 3), (2, 4)]
    assert all(math.isfinite(loss) for w in report["windows"] for loss in w["losses"])
    assert all(torch.equal(dict(model.named_buffers())[name], value) for name, value in base.items())
    assert any(not torch.equal(dict(model.named_parameters())[name], value) for name, value in factors.items())
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert not list(tmp_path.iterdir())
    assert all(m.progress == 1 for _, m in iter_ternary_modules(model))


def test_window_validation():
    assert build_windows(5, 4, 2) == [(0, 4), (2, 5), (4, 5)]
    with pytest.raises(ValueError):
        TrainingConfig(stride=5, window_size=4).validate()
    with pytest.raises(ValueError):
        build_windows(4, 2, 3)
    for field in ("learning_rate", "weight_decay", "grad_clip"):
        for value in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                TrainingConfig(**{field: value}).validate()
