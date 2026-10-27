"""Native path oracles and read-only streaming contracts, not quality evidence."""
from copy import deepcopy
import gc
import weakref

import pytest
import torch

from openbonsai.activation_capture import iter_activation_records
from openbonsai.adapter import get_text_layers, iter_ternary_modules, replace_projections, set_progress, tiny_model
from openbonsai.rotations import replace_rotated_projections


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def sample():
    return {"input_ids": torch.tensor([[4, 7, 10, 15, 20, 25, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])}


def models(dtype=torch.float32, rotated=False, multimodal=False):
    original = tiny_model(seed=811, multimodal=multimodal).to(dtype)
    model = deepcopy(original)
    with torch.random.fork_rng():
        torch.manual_seed(812)
        if rotated:
            replace_rotated_projections(model, block_size=32, lora_rank=4)
        else:
            replace_projections(model, lora_rank=4, factor_mapping="sigmoid")
        with torch.no_grad():
            for _, module in iter_ternary_modules(model):
                module.lora_B.normal_(std=.08)
                module.raw_scale.add_(.3)
    # Establish real saved derivative snapshots and nonempty gradient buffers.
    set_progress(model, .4)
    model(**sample(), use_cache=False).logits.float().square().mean().backward()
    set_progress(model, 1.)
    for index, (_, module) in enumerate(iter_ternary_modules(model)):
        module.quant_enabled = bool(index % 2)
    model.train()
    get_text_layers(model)[1].eval()  # Preserve heterogeneous incoming modes.
    if multimodal:
        model.model.rope_deltas = torch.tensor([[123]])
    return original, model


def state(model):
    return {
        "tensors": {name: (value, value.detach().clone())
                    for name, value in list(model.named_parameters()) + list(model.named_buffers())},
        "grads": {name: (p.grad, None if p.grad is None else p.grad.detach().clone(), p.requires_grad)
                  for name, p in model.named_parameters()},
        "modes": [module.training for module in model.modules()],
        "quantizers": [(m.quant_enabled, m.progress, getattr(m, "rotation_only", False),
                        m._hard_snapshot_sharpness) for _, m in iter_ternary_modules(model)],
        "hooks": [(dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in model.modules()],
        "deltas": [(m, m.rope_deltas) for m in model.modules() if hasattr(m, "rope_deltas")],
    }


def assert_state(model, before):
    current = dict(list(model.named_parameters()) + list(model.named_buffers()))
    assert current.keys() == before["tensors"].keys()
    for name, (identity, value) in before["tensors"].items():
        assert current[name] is identity
        assert torch.equal(current[name], value), name
    for name, p in model.named_parameters():
        identity, value, flag = before["grads"][name]
        assert p.grad is identity and p.requires_grad == flag
        if value is not None:
            assert torch.equal(p.grad, value)
    assert [m.training for m in model.modules()] == before["modes"]
    assert [(m.quant_enabled, m.progress, getattr(m, "rotation_only", False), m._hard_snapshot_sharpness)
            for _, m in iter_ternary_modules(model)] == before["quantizers"]
    assert [(dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in model.modules()] == before["hooks"]
    for module, value in before["deltas"]:
        assert module.rope_deltas is value


@torch.no_grad()
def native_oracle(model, row, start, end, *, path):
    """Independent native forward; no production capture or replay helpers."""
    layers = get_text_layers(model)
    flags = [(m, m.quant_enabled) for _, m in iter_ternary_modules(model)]
    modes = [(m, m.training) for m in model.modules()]
    result, hooks = {"kwargs": {}}, []

    class Stop(Exception):
        pass

    def capture(index):
        def hook(_module, args, kwargs):
            if index == start:
                result["input"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
            result["kwargs"][index] = kwargs
        return hook

    def output(_module, _args, value):
        result["target"] = (value if torch.is_tensor(value) else value[0]).detach().clone()
        raise Stop

    try:
        model.eval()
        for module, _ in flags:
            module.quant_enabled = path != "original"
        if path == "conditional":
            for layer in layers[start:end]:
                for module in layer.modules():
                    if any(module is quantizer for quantizer, _ in flags):
                        module.quant_enabled = False
        for index in range(start, end):
            hooks.append(layers[index].register_forward_pre_hook(capture(index), with_kwargs=True))
        hooks.append(layers[end - 1].register_forward_hook(output))
        with pytest.raises(Stop):
            model(**row, use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
        for module, enabled in flags:
            module.quant_enabled = enabled
        for module, training in modes:
            module.training = training
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotated,multimodal", [(False, False), (True, True)])
def test_native_original_conditional_and_hard_student_oracles(dtype, rotated, multimodal, monkeypatch):
    original, model = models(dtype, rotated, multimodal)

    def no_head(*_args, **_kwargs):
        raise AssertionError("Vocabulary logits are forbidden in this collector")

    monkeypatch.setattr(model.lm_head, "forward", no_head)
    monkeypatch.setattr(original.lm_head, "forward", no_head)
    before = state(model)
    row = sample()
    iterator = iter_activation_records(model, [row], 1, 4)
    result = next(iterator)
    assert_state(model, before)  # Restoration is before yield, not on exhaustion.
    clean = native_oracle(original, row, 1, 4, path="original")
    conditional = native_oracle(model, row, 1, 4, path="conditional")
    quantized = native_oracle(model, row, 1, 4, path="hard")
    for key, expected in (("x0", clean["input"]), ("t0", clean["target"]),
                          ("xq", conditional["input"]), ("tq", conditional["target"]),
                          ("y", quantized["target"])):
        assert torch.equal(result[key], expected), key
        assert result[key].dtype == dtype
    assert result.keys() == {"xq", "x0", "tq", "t0", "y", "mask"}
    assert all(t.device.type == "cpu" and not t.requires_grad for t in result.values())
    assert not torch.equal(result["xq"], result["x0"])
    assert not torch.equal(result["tq"], result["t0"])
    assert not torch.equal(result["y"], result["tq"])
    assert conditional["kwargs"][1]["attention_mask"].ndim == 2
    assert conditional["kwargs"][3]["attention_mask"].ndim == 4
    for kwargs in conditional["kwargs"].values():
        assert kwargs.get("past_key_value") is None and kwargs.get("past_key_values") is None
        assert kwargs.get("position_embeddings") is not None
    assert torch.equal(result["mask"], row["attention_mask"])
    assert result["mask"].data_ptr() != row["attention_mask"].data_ptr()
    iterator.close()
    assert_state(model, before)


@pytest.mark.parametrize("end", [1, 4])
def test_window_zero_has_identical_original_and_quantized_inputs_and_targets(end):
    _, model = models(torch.bfloat16, rotated=True)
    result = next(iter_activation_records(model, [sample()], 0, end))
    assert torch.equal(result["xq"], result["x0"])
    assert torch.equal(result["tq"], result["t0"])
    assert not torch.equal(result["y"], result["t0"])
    assert result["xq"].data_ptr() != result["x0"].data_ptr()


def test_stream_consumes_one_sample_and_retains_no_previous_record():
    _, model = models()
    consumed = []

    def rows():
        for index in range(3):
            consumed.append(index)
            yield sample()

    iterator = iter_activation_records(model, rows(), 1, 4)
    assert consumed == []
    first = next(iterator)
    assert consumed == [0]
    ref = weakref.ref(first["xq"])
    del first
    gc.collect()
    assert ref() is None
    before = state(model)
    second = next(iterator)
    assert consumed == [0, 1] and second["xq"].shape == (1, 8, 128)
    assert_state(model, before)
    iterator.close()
    assert consumed == [0, 1]
    assert_state(model, before)


@pytest.mark.parametrize("failing_pass", [1, 2, 3])
def test_failure_in_each_pass_restores_modes_flags_hooks_gradients_and_snapshots(monkeypatch, failing_pass):
    _, model = models(torch.bfloat16, rotated=True, multimodal=True)
    layer = get_text_layers(model)[2]
    native = layer.forward
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failing_pass:
            raise RuntimeError("controlled activation path failure")
        return native(*args, **kwargs)

    monkeypatch.setattr(layer, "forward", fail)
    existing = layer.register_forward_pre_hook(lambda *_args: None)
    before = state(model)
    try:
        iterator = iter_activation_records(model, [sample()], 1, 4)
        with pytest.raises(RuntimeError, match="controlled activation path failure"):
            next(iterator)
        assert calls == failing_pass
        assert_state(model, before)
        iterator.close()
        assert_state(model, before)
    finally:
        existing.remove()


@pytest.mark.parametrize("problem", ["soft", "rotation_only", "mixed", "packed", "embedding", "dynamic_rope"])
def test_unsupported_model_states_fail_before_mutation(problem):
    from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
    from openbonsai.mixed_formats import MixedLinear, ternary_candidate
    from openbonsai.quantizer import PackedTernaryLinear

    _, model = models(torch.bfloat16, rotated=True)
    name, module = next(iter_ternary_modules(model))
    if problem == "soft":
        module.progress = module.soft_fraction
    elif problem == "rotation_only":
        module.rotation_only = True
    elif problem in ("mixed", "packed"):
        from openbonsai.quantizer import TernaryLinear
        packed = TernaryLinear.packed_state(module)
        replacement = MixedLinear(ternary_candidate(packed)) if problem == "mixed" else PackedTernaryLinear(packed)
        model.set_submodule(name, replacement)
    elif problem == "embedding":
        model.set_input_embeddings(BoundedTernaryEmbedding(model.get_input_embeddings(), lora_rank=0))
    else:
        model.model.rotary_emb.rope_type = "dynamic"
    modes = [m.training for m in model.modules()]
    tensors = {name: value.clone() for name, value in model.state_dict().items()}
    hooks = [(dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in model.modules()]
    with pytest.raises(ValueError, match="hard-phase|rotation_only|original sources|nn.Embedding|RoPE"):
        iter_activation_records(model, [sample()], 1, 4)
    assert [m.training for m in model.modules()] == modes
    assert [(dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in model.modules()] == hooks
    for name, value in model.state_dict().items():
        assert torch.equal(value, tensors[name])


@pytest.mark.parametrize("problem", ["cache", "unknown", "empty", "bad_mask", "all_padding", "one_empty_row", "bad_ids", "position_ids"])
def test_unsupported_samples_fail_before_model_mutation(problem):
    _, model = models()
    row = sample()
    if problem == "cache":
        row["past_key_values"] = object()
    elif problem == "unknown":
        row["pixel_values"] = torch.ones(1)
    elif problem == "empty":
        row["input_ids"] = row["input_ids"][:, :0]
        row["attention_mask"] = row["attention_mask"][:, :0]
    elif problem == "bad_mask":
        row["attention_mask"][0, 0] = 2
    elif problem == "all_padding":
        row["attention_mask"].zero_()
    elif problem == "one_empty_row":
        row = {key: value.repeat(2, 1) for key, value in row.items()}
        row["attention_mask"][1].zero_()
    elif problem == "bad_ids":
        row["input_ids"][0, 0] = 256
    else:
        row["position_ids"] = torch.tensor([[-1] * 8])
    before = state(model)
    with pytest.raises(ValueError):
        next(iter_activation_records(model, [row], 1, 4))
    assert_state(model, before)


def test_explicit_native_positions_and_reverse_order_are_record_independent():
    original, model = models(torch.bfloat16, rotated=True, multimodal=True)
    first, second = sample(), sample()
    first["position_ids"] = torch.arange(8).reshape(1, -1)
    second["input_ids"] = second["input_ids"] + 1
    second["position_ids"] = torch.arange(8).reshape(1, -1).expand(3, 1, -1) + 2
    forward = list(iter_activation_records(model, [first, second], 1, 4))
    reverse = list(iter_activation_records(model, [second, first], 1, 4))
    for left, right in zip(forward, reversed(reverse)):
        assert all(torch.equal(left[key], right[key]) for key in left)
    before = state(model)
    for record, row in zip(forward, (first, second)):
        clean = native_oracle(original, row, 1, 4, path="original")
        hard = native_oracle(model, row, 1, 4, path="hard")
        assert torch.equal(record["x0"], clean["input"])
        assert torch.equal(record["t0"], clean["target"])
        assert torch.equal(record["y"], hard["target"])
    assert_state(model, before)


def test_nonfinite_padding_is_explicitly_rejected_and_state_restored(monkeypatch):
    _, model = models()
    import openbonsai.activation_capture as capture
    original_replay = capture.replay_window

    def nonfinite_replay(*args, **kwargs):
        value = original_replay(*args, **kwargs).clone()
        value[0, -1, 0] = float("nan")  # A masked padding position is still rejected.
        return value

    monkeypatch.setattr(capture, "replay_window", nonfinite_replay)
    before = state(model)
    with pytest.raises(ValueError, match="finite hidden-state"):
        next(iter_activation_records(model, [sample()], 1, 4))
    assert_state(model, before)
