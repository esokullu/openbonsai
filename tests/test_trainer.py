"""Native hybrid integration, not a benchmark of pretrained-model quality."""
import math

import pytest
import torch

from openbonsai.adapter import tiny_model, replace_projections, get_text_layers, iter_ternary_modules, set_progress
from openbonsai.trainer import (
    TrainingConfig, _elements, _loss_numerator, build_windows, cache_window,
    replay_window, train, window_mse,
)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("delta,expected,gradient", [
    (0.5, 2.53125, [0.25, -0.5, 0.5, -0.5]),
    (1.5, 5.5625, [0.25, -0.75, 1.5, -1.5]),
])
def test_masked_huber_value_gradient_and_valid_element_denominator(dtype, delta, expected, gradient):
    # Hand-calculated values exercise the quadratic and linear branches.
    # Large padded errors contribute neither loss nor gradient.
    output = torch.tensor([[[0.25, -0.75], [2.0, -3.0], [300.0, -300.0]]],
                          dtype=dtype, requires_grad=True)
    record = {"target": torch.zeros_like(output), "mask": torch.tensor([[1, 1, 0]])}
    numerator = _loss_numerator(output, record, reconstruction_loss="huber", huber_delta=delta)
    assert numerator.dtype == torch.float32
    assert numerator.item() == expected
    denominator = _elements(record)
    assert denominator == 4
    loss = numerator / denominator
    assert loss.item() == expected / 4
    loss.backward()
    expected_gradient = torch.tensor([[gradient[:2], gradient[2:], [0.0, 0.0]]], dtype=dtype) / 4
    torch.testing.assert_close(output.grad, expected_gradient, rtol=0, atol=0)


def test_default_reconstruction_loss_remains_masked_fp32_mse():
    output = torch.tensor([[[0.25, -0.75], [2.0, -3.0], [300.0, -300.0]]],
                          dtype=torch.float16, requires_grad=True)
    record = {"target": torch.zeros_like(output), "mask": torch.tensor([[1, 1, 0]])}
    loss = _loss_numerator(output, record) / _elements(record)
    assert loss.dtype == torch.float32
    assert loss.item() == 13.625 / 4
    loss.backward()
    torch.testing.assert_close(output.grad, torch.tensor(
        [[[0.125, -0.375], [1.0, -1.5], [0.0, 0.0]]], dtype=torch.float16), rtol=0, atol=0)


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


def test_cached_teacher_ignores_nonzero_window_corrections(model):
    layers = get_text_layers(model)
    before = cache_window(model, samples(), 1, 4)[0]
    with torch.no_grad():
        student_before = replay_window(layers, before, 1, 4, "cpu")
        for layer in layers[1:4]:
            for module in layer.modules():
                if hasattr(module, "lora_parameters"):
                    module.lora_A.normal_(std=0.2)
                    module.lora_B.normal_(std=0.2)
                    module.raw_scale.add_(0.1)
    after = cache_window(model, samples(), 1, 4)[0]
    # The quantized prefix is unchanged and the teacher must still use only
    # original window weights, even though the student's correction is large.
    torch.testing.assert_close(after["input"], before["input"], rtol=0, atol=0)
    torch.testing.assert_close(after["target"], before["target"], rtol=0, atol=0)
    with torch.no_grad():
        student_after = replay_window(layers, after, 1, 4, "cpu")
    assert not torch.equal(student_after, student_before)
    assert all(module.quant_enabled for _, module in iter_ternary_modules(model))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_train_factors_and_corrections_with_overlap(model, tmp_path, checkpointing):
    base = {name: tensor.clone() for name, tensor in model.named_buffers()
            if not name.rsplit(".", 1)[-1].startswith("_hard_snapshot")}
    factors = {name: p.clone() for name, p in model.named_parameters() if ".raw_" in name}
    corrections = {name: p.clone() for name, p in model.named_parameters() if ".lora_" in name}
    report = train(model, samples(), TrainingConfig(epochs=3, batch_size=1, window_size=3,
                 stride=2, gradient_checkpointing=checkpointing), cache_directory=tmp_path)
    assert [(w["start"], w["end"]) for w in report["windows"]] == [(0, 3), (2, 4)]
    assert all(math.isfinite(loss) for w in report["windows"] for loss in w["losses"])
    for window in report["windows"]:
        timing = window["timing_seconds"]
        assert all(math.isfinite(value) and value >= 0 for value in timing.values())
        assert math.isclose(timing["total"], sum(value for name, value in timing.items() if name != "total"))
    assert report["timing_seconds"]["total"] == sum(w["timing_seconds"]["total"] for w in report["windows"])
    assert all(torch.equal(dict(model.named_buffers())[name], value) for name, value in base.items())
    assert any(not torch.equal(dict(model.named_parameters())[name], value) for name, value in factors.items())
    for suffix in ("lora_A", "lora_B"):
        assert any(not torch.equal(dict(model.named_parameters())[name], value)
                   for name, value in corrections.items() if name.endswith(suffix))
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert not list(tmp_path.iterdir())
    assert all(m.progress == 1 for _, m in iter_ternary_modules(model))


def test_native_reconstruction_materially_reduces_hard_error():
    """A corrective optimization regression, not a pretrained quality claim."""
    torch.set_num_threads(2)
    results = {}
    for rank in (0, 4):
        model = tiny_model(seed=3)
        torch.manual_seed(3)
        replace_projections(model, lora_rank=rank)
        set_progress(model, 1)
        layers = get_text_layers(model)
        records = cache_window(model, samples(), 0, 4)
        source_weights = {name: module.weight.clone() for name, module in iter_ternary_modules(model)}
        initial_corrections = {name: p.clone() for name, p in model.named_parameters() if ".lora_" in name}
        before = window_mse(layers, records, 0, 4, "cpu")
        report = train(model, samples(), TrainingConfig(
            epochs=32, batch_size=1, window_size=4, stride=4,
            learning_rate=0.01, lora_learning_rate=0.02,
        ))
        after = window_mse(layers, records, 0, 4, "cpu")
        assert after < before * 0.25
        assert report["windows"][0]["steps"] == 32
        assert report["windows"][0]["hard_mse_before"] == pytest.approx(before)
        assert report["windows"][0]["hard_mse_after"] == pytest.approx(after)
        assert report["quantization"]["lora_ranks"] == [rank]
        for name, module in iter_ternary_modules(model):
            torch.testing.assert_close(module.weight, source_weights[name], rtol=0, atol=0)
        if rank:
            for suffix in ("lora_A", "lora_B"):
                assert any(not torch.equal(dict(model.named_parameters())[name], value)
                           for name, value in initial_corrections.items() if name.endswith(suffix))
        else:
            assert report["quantization"]["lora_parameters"] == 0
        results[rank] = after
    # This controlled fitting task must exercise the extra corrective capacity.
    # It says nothing about downstream language-model scores or generalization.
    assert results[4] < results[0] * 0.25


def test_checkpointed_sigmoid_huber_training_keeps_hard_metrics_as_mse():
    torch.set_num_threads(2)
    model = tiny_model(seed=73)
    torch.manual_seed(74)
    replace_projections(model, lora_rank=4, factor_mapping="sigmoid")
    layers = get_text_layers(model)
    set_progress(model, 1)
    record = cache_window(model, samples(), 0, 4)[0]
    mask = record["mask"].unsqueeze(-1)
    denominator = int(mask.sum()) * record["target"].shape[-1]
    with torch.no_grad():
        initial_error = replay_window(layers, record, 0, 4, "cpu").float() - record["target"].float()
        expected_before = float((initial_error.square() * mask).sum()) / denominator
        set_progress(model, 0.25)
        first_error = replay_window(layers, record, 0, 4, "cpu").float() - record["target"].float()
        # Independent Huber oracle: a clipped quadratic plus a linear tail.
        delta = 0.001
        absolute = first_error.abs()
        clipped = absolute.clamp(max=delta)
        first_huber = float(((0.5 * clipped.square() + delta * (absolute - clipped)) * mask).sum()) / denominator
        first_mse = float((first_error.square() * mask).sum()) / denominator
    events = []
    report = train(model, samples(), TrainingConfig(
        epochs=4, batch_size=1, window_size=4, stride=4,
        gradient_checkpointing=True, reconstruction_loss="huber", huber_delta=delta,
    ), progress_callback=events.append)
    window = report["windows"][0]
    assert window["losses"][0] == pytest.approx(first_huber, rel=1e-5, abs=1e-12)
    assert first_huber < first_mse * 0.25
    assert len(events) == 4
    assert window["losses"] == [event["loss"] for event in events]
    assert all(math.isfinite(event["loss"]) for event in events)
    for item in [window, *events]:
        assert item["reconstruction_loss"] == "huber" and item["huber_delta"] == delta
    assert report["reconstruction_objective"] == {
        "name": "huber", "huber_delta": delta,
        "reduction": "mean over nonpadding hidden-state elements",
    }
    assert report["quantization"]["factor_mappings"] == ["sigmoid"]
    with torch.no_grad():
        final_error = replay_window(layers, record, 0, 4, "cpu").float() - record["target"].float()
        expected_after = float((final_error.square() * mask).sum()) / denominator
    assert window["hard_mse_before"] == pytest.approx(expected_before)
    assert window["hard_mse_after"] == pytest.approx(expected_after)
    assert window_mse(layers, [record], 0, 4, "cpu") == pytest.approx(expected_after)
    assert any(torch.count_nonzero(module.lora_B) for _, module in iter_ternary_modules(model))


@pytest.mark.parametrize("rank", [0, 4])
def test_optimizer_preserves_distinct_factor_and_lora_learning_rates(monkeypatch, rank):
    model = tiny_model(seed=6)
    replace_projections(model, lora_rank=rank)
    modules = [module for _, module in iter_ternary_modules(model)]
    factor_ids = {id(p) for module in modules for p in module.factor_parameters()}
    lora_ids = {id(p) for module in modules for p in module.lora_parameters()}
    original_step = torch.optim.AdamW.step
    observed = []

    def step(optimizer, *args, **kwargs):
        observed.append([(group["lr"], {id(p) for p in group["params"]})
                         for group in optimizer.param_groups])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    train(model, samples(), TrainingConfig(epochs=3, batch_size=1, window_size=4, stride=4,
                                          learning_rate=0.001, lora_learning_rate=0.002))
    assert len(observed) == 3
    assert factor_ids.isdisjoint(lora_ids)
    for index, groups in enumerate(observed):
        decay = 1 - index / 3
        assert len(groups) == (2 if rank else 1)
        assert groups[0][0] == pytest.approx(0.001 * decay)
        assert groups[0][1] == factor_ids
        if rank:
            assert groups[1][0] == pytest.approx(0.002 * decay)
            assert groups[1][1] == lora_ids


def test_one_step_schedule_rejected_before_model_mutation(model, tmp_path):
    model.train()
    set_progress(model, 0.3)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="soft update"):
        train(model, samples(), TrainingConfig(epochs=1, batch_size=1), cache_directory=tmp_path / "cache")
    assert model.training
    assert all(module.progress == 0.3 for _, module in iter_ternary_modules(model))
    assert flags == {name: p.requires_grad for name, p in model.named_parameters()}
    assert not (tmp_path / "cache").exists()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("settings", [
    {"reconstruction_loss": "smooth_l1"}, {"reconstruction_loss": None},
    {"huber_delta": 0}, {"huber_delta": -1}, {"huber_delta": True},
    {"huber_delta": float("nan")}, {"huber_delta": float("inf")},
])
def test_invalid_reconstruction_objective_rejected_before_mutation(model, tmp_path, settings):
    model.train()
    set_progress(model, 0.3)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="reconstruction_loss|huber_delta"):
        train(model, samples(), TrainingConfig(**settings), cache_directory=tmp_path / "cache")
    assert model.training
    assert all(module.progress == 0.3 for _, module in iter_ternary_modules(model))
    assert flags == {name: p.requires_grad for name, p in model.named_parameters()}
    assert not (tmp_path / "cache").exists()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_window_validation():
    assert build_windows(5, 4, 2) == [(0, 4), (2, 5), (4, 5)]
    with pytest.raises(ValueError):
        TrainingConfig(stride=5, window_size=4).validate()
    with pytest.raises(ValueError):
        build_windows(4, 2, 3)
    for field in ("learning_rate", "lora_learning_rate", "weight_decay", "grad_clip"):
        for value in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                TrainingConfig(**{field: value}).validate()
    for value in (0, -0.001):
        with pytest.raises(ValueError, match="positive"):
            TrainingConfig(lora_learning_rate=value).validate()
