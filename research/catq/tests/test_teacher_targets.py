"""Native teacher-prefix semantics; random fixtures are not model-quality evidence."""
from copy import deepcopy
import math

import pytest
import torch

from openbonsai.adapter import (
    get_text_layers, iter_ternary_modules, replace_projections, set_progress, tiny_model,
)
from openbonsai.rotations import replace_rotated_projections
from openbonsai.trainer import TrainingConfig, cache_window, train


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _samples():
    return [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 0, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])}]


def _models(*, dtype=torch.float32, rotated=False, multimodal=False):
    original = tiny_model(seed=121, multimodal=multimodal).to(dtype=dtype)
    quantized = deepcopy(original)
    with torch.random.fork_rng():
        torch.manual_seed(122)
        if rotated:
            replace_rotated_projections(quantized, block_size=32, lora_rank=4)
        else:
            replace_projections(quantized, lora_rank=4, factor_mapping="sigmoid")
        # Nonzero corrections make an accidentally learned teacher observable.
        with torch.no_grad():
            for _, module in iter_ternary_modules(quantized):
                module.lora_B.normal_(std=0.1)
                module.raw_scale.add_(0.2)
    set_progress(quantized, 1)
    return original, quantized


def _equal_tree(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.device == right.device
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal_tree(a, b)
    else:
        assert left == right


def _assert_detached_cpu(value):
    if torch.is_tensor(value):
        assert value.device.type == "cpu" and not value.requires_grad
    elif isinstance(value, dict):
        for part in value.values():
            _assert_detached_cpu(part)
    elif isinstance(value, (list, tuple)):
        for part in value:
            _assert_detached_cpu(part)


def _hook_counts(model):
    return [(len(module._forward_pre_hooks), len(module._forward_hooks))
            for module in model.modules()]


@torch.no_grad()
def _independent_original_forward(original, sample, start, end):
    """Use an untouched native model, without any production cache helper."""
    result = {}
    layers = get_text_layers(original)

    class StopOriginal(Exception):
        pass

    def input_hook(_module, args, kwargs):
        result["input"] = (args[0] if args else kwargs["hidden_states"]).clone()

    def output_hook(_module, _args, output):
        result["target"] = (output if torch.is_tensor(output) else output[0]).clone()
        raise StopOriginal

    first = layers[start].register_forward_pre_hook(input_hook, with_kwargs=True)
    last = layers[end - 1].register_forward_hook(output_hook)
    try:
        with pytest.raises(StopOriginal):
            original(**sample, use_cache=False)
    finally:
        first.remove()
        last.remove()
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotated,multimodal", [(False, False), (True, True)])
def test_original_targets_match_independent_native_teacher_and_keep_student_inputs(
        dtype, rotated, multimodal, monkeypatch):
    original, quantized = _models(dtype=dtype, rotated=rotated, multimodal=multimodal)
    sources = {name: module.weight.clone() for name, module in iter_ternary_modules(quantized)}

    def forbidden_logits(*_args, **_kwargs):
        raise AssertionError("Teacher caching must stop before the vocabulary head")

    monkeypatch.setattr(original.lm_head, "forward", forbidden_logits)
    monkeypatch.setattr(quantized.lm_head, "forward", forbidden_logits)
    local = cache_window(quantized, _samples(), 1, 4, teacher_prefix="quantized")[0]
    clean = cache_window(quantized, _samples(), 1, 4, teacher_prefix="original")[0]
    oracle = _independent_original_forward(original, _samples()[0], 1, 4)
    for key in ("input", "kwargs", "mask"):
        _equal_tree(local[key], clean[key])
    _equal_tree(clean["target"], oracle["target"])
    assert not torch.equal(clean["input"], oracle["input"])
    assert not torch.equal(clean["target"], local["target"])
    assert clean["kwargs"][1]["attention_mask"].ndim == 2
    assert clean["kwargs"][3]["attention_mask"].ndim == 4
    for kwargs in clean["kwargs"].values():
        assert kwargs.get("past_key_value") is None
        assert kwargs.get("past_key_values") is None
        assert kwargs["position_embeddings"] is not None
    # Alter even the prefix: the new teacher remains the original model, while
    # the student's cached input changes. This also checks LoRA/rotation bypass.
    with torch.no_grad():
        for _, module in iter_ternary_modules(quantized):
            module.raw_scale.add_(0.4)
            module.lora_B.add_(0.2)
    changed = cache_window(quantized, _samples(), 1, 4, teacher_prefix="original")[0]
    _equal_tree(changed["target"], oracle["target"])
    assert not torch.equal(changed["input"], clean["input"])
    for name, module in iter_ternary_modules(quantized):
        _equal_tree(module.weight, sources[name])


@pytest.mark.parametrize("end", [1, 4])
def test_first_window_modes_coincide_and_default_is_local(end):
    _, model = _models()
    default = cache_window(model, _samples(), 0, end)
    local = cache_window(model, _samples(), 0, end, teacher_prefix="quantized")
    original = cache_window(model, _samples(), 0, end, teacher_prefix="original")
    _equal_tree(default, local)
    _equal_tree(original, local)
    assert TrainingConfig().teacher_prefix == "quantized"


@pytest.mark.parametrize("mode", ["original", "quantized"])
def test_disk_cache_is_identical_bounded_records_and_restores_nonuniform_flags(tmp_path, mode):
    _, model = _models(rotated=True)
    modules = [module for _, module in iter_ternary_modules(model)]
    for index, module in enumerate(modules):
        module.quant_enabled = bool(index % 2)
    before_flags = [module.quant_enabled for module in modules]
    before_hooks = _hook_counts(model)
    samples = _samples() + [{"input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
                             "attention_mask": torch.ones((1, 5), dtype=torch.long)}]
    memory = cache_window(model, samples, 1, 4, teacher_prefix=mode)
    disk = cache_window(model, samples, 1, 4, tmp_path, teacher_prefix=mode)
    assert len(memory) == len(disk) == 2
    for index in range(2):
        _equal_tree(memory[index], disk[index])
        assert memory[index].keys() == {"input", "target", "kwargs", "mask"}
        _assert_detached_cpu(disk[index])
    assert len(list(tmp_path.iterdir())) == 2
    assert [module.quant_enabled for module in modules] == before_flags
    assert _hook_counts(model) == before_hooks


@pytest.mark.parametrize("failing_pass", [1, 2])
def test_capture_failure_restores_quantizer_flags_and_existing_hooks(monkeypatch, failing_pass):
    _, model = _models()
    modules = [module for _, module in iter_ternary_modules(model)]
    for index, module in enumerate(modules):
        module.quant_enabled = bool(index % 3)
    before_flags = [module.quant_enabled for module in modules]
    call = 0

    def count_forward(_module, _args):
        nonlocal call
        call += 1

    existing_hook = model.register_forward_pre_hook(count_forward)
    before_hooks = _hook_counts(model)
    layer = get_text_layers(model)[2]
    native_forward = layer.forward

    def fail(*args, **kwargs):
        if call == failing_pass:
            raise RuntimeError("controlled native decoder failure")
        return native_forward(*args, **kwargs)

    monkeypatch.setattr(layer, "forward", fail)
    try:
        with pytest.raises(RuntimeError, match="controlled native decoder failure"):
            cache_window(model, _samples(), 1, 4, teacher_prefix="original")
        assert call == failing_pass
        assert [module.quant_enabled for module in modules] == before_flags
        assert _hook_counts(model) == before_hooks
    finally:
        existing_hook.remove()


@pytest.mark.parametrize("invalid", [None, True, 0, "", "fp16", "Original", [], {}])
def test_invalid_mode_rejected_before_any_model_or_disk_access(invalid, tmp_path):
    with pytest.raises(ValueError, match="teacher_prefix"):
        TrainingConfig(teacher_prefix=invalid).validate()
    # An invalid object would fail model introspection if validation were late.
    with pytest.raises(ValueError, match="teacher_prefix"):
        cache_window(object(), _samples(), 0, 1, tmp_path / "cache", teacher_prefix=invalid)
    with pytest.raises(ValueError, match="teacher_prefix"):
        train(object(), _samples(), TrainingConfig(teacher_prefix=invalid), cache_directory=tmp_path / "train")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_original_target_training_with_overlap_preserves_sources_and_reports_definition(tmp_path, checkpointing):
    _, model = _models(rotated=True)
    sources = {name: module.weight.clone() for name, module in iter_ternary_modules(model)}
    initial = {name: value.clone() for name, value in model.named_parameters() if ".raw_" in name}
    report = train(model, _samples(), TrainingConfig(
        epochs=3, batch_size=1, window_size=3, stride=2,
        teacher_prefix="original", gradient_checkpointing=checkpointing), cache_directory=tmp_path)
    assert report["training_config"]["teacher_prefix"] == "original"
    assert report["reconstruction_objective"]["teacher_prefix"] == "original"
    assert report["reconstruction_objective"]["teacher_target"] == (
        "original-weight window on original-weight prefix input")
    assert report["reconstruction_objective"]["student_input"] == "frozen hard-quantized prefix output"
    assert [(w["start"], w["end"]) for w in report["windows"]] == [(0, 3), (2, 4)]
    assert all(math.isfinite(loss) for window in report["windows"] for loss in window["losses"])
    assert any(not torch.equal(dict(model.named_parameters())[name], value) for name, value in initial.items())
    for name, module in iter_ternary_modules(model):
        _equal_tree(module.weight, sources[name])
        assert module.quant_enabled and module.progress == 1
        assert module._hard_snapshot_scale is None
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not list(tmp_path.iterdir())


def test_default_and_explicit_local_training_have_identical_updates_and_losses():
    _, default_model = _models()
    explicit_model = deepcopy(default_model)
    settings = dict(epochs=3, batch_size=1, window_size=3, stride=2)
    default = train(default_model, _samples(), TrainingConfig(**settings))
    explicit = train(explicit_model, _samples(), TrainingConfig(**settings, teacher_prefix="quantized"))
    _equal_tree(default_model.state_dict(), explicit_model.state_dict())
    for before, after in zip(default["windows"], explicit["windows"]):
        for field in ("losses", "hard_mse_before", "hard_mse_after", "steps"):
            assert before[field] == after[field]
    assert default["reconstruction_objective"] == explicit["reconstruction_objective"]
    assert default["reconstruction_objective"]["teacher_target"] == (
        "original-weight window on the same quantized-prefix input")


@pytest.mark.parametrize("enabled", [False, True])
def test_bounded_embedding_exact_teacher_bypass_keeps_legacy_student_state(enabled):
    from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
    from openbonsai.rotations import fixed_signs

    original, model = _models(dtype=torch.bfloat16, rotated=True, multimodal=True)
    embedding = BoundedTernaryEmbedding(
        model.get_input_embeddings(), row_chunk_size=3, lora_rank=2,
        signs=fixed_signs(128, seed=125), block_size=32)
    embedding.progress, embedding.quant_enabled = 1., enabled
    with torch.no_grad():
        embedding.quantizer.lora_B.fill_(0.2)
        embedding.quantizer.raw_scale.add_(0.3)
    model.set_input_embeddings(embedding)
    sources = deepcopy(embedding.state_dict())
    samples = _samples() * 2
    local = cache_window(model, samples, 1, 4, teacher_prefix="quantized")
    flags = []
    hook = embedding.register_forward_pre_hook(lambda _module, _args: flags.append(embedding.quant_enabled))
    try:
        clean = cache_window(model, samples, 1, 4, teacher_prefix="original")
    finally:
        hook.remove()
    assert flags == [enabled, False, enabled, False]
    oracle = _independent_original_forward(original, samples[0], 1, 4)
    for record, legacy in zip(clean, local):
        _equal_tree(record["target"], oracle["target"])
        for key in ("input", "kwargs", "mask"):
            _equal_tree(record[key], legacy[key])
    assert embedding.quant_enabled is enabled
    _equal_tree(embedding.state_dict(), sources)
    embedding.assert_source_unchanged(verify_hash=True)


def test_bounded_embedding_teacher_failure_restores_its_state(monkeypatch):
    from openbonsai.embedding_quantizer import BoundedTernaryEmbedding

    _, model = _models(dtype=torch.bfloat16)
    embedding = BoundedTernaryEmbedding(model.get_input_embeddings(), row_chunk_size=2, lora_rank=0)
    embedding.progress = 1.
    model.set_input_embeddings(embedding)
    before_hooks = _hook_counts(model)
    original_forward = embedding.forward

    def fail_teacher(ids):
        if not embedding.quant_enabled:
            raise RuntimeError("controlled original embedding failure")
        return original_forward(ids)

    monkeypatch.setattr(embedding, "forward", fail_teacher)
    with pytest.raises(RuntimeError, match="controlled original embedding failure"):
        cache_window(model, _samples(), 1, 4, teacher_prefix="original")
    assert embedding.quant_enabled
    assert _hook_counts(model) == before_hooks
    assert all(module.quant_enabled for _, module in iter_ternary_modules(model))


@pytest.mark.parametrize("replacement", ["mixed_embedding", "unknown_embedding", "mixed_decoder", "packed_decoder"])
def test_original_mode_rejects_unavailable_original_sources_before_mutation(tmp_path, replacement):
    from openbonsai.mixed_formats import MixedEmbedding, MixedLinear, bf16_candidate, ternary_candidate
    from openbonsai.quantizer import PackedTernaryLinear

    _, model = _models(dtype=torch.bfloat16)
    if replacement == "mixed_embedding":
        source = model.get_input_embeddings()
        # Even a BF16 candidate lacks the trainable operator's original-source
        # ownership contract; packed artifact provenance is not a bypass.
        model.set_input_embeddings(MixedEmbedding(bf16_candidate(source.weight, module_kind="embedding")))
    elif replacement == "unknown_embedding":
        class UnknownEmbedding(torch.nn.Module):
            def forward(self, ids):
                raise AssertionError("Unknown embedding must not be executed")
        model.set_input_embeddings(UnknownEmbedding())
    else:
        name, module = next(iter_ternary_modules(model))
        packed = module.packed_state()
        new = (MixedLinear(ternary_candidate(packed)) if replacement == "mixed_decoder"
               else PackedTernaryLinear(packed))
        model.set_submodule(name, new)
    model.train()
    modules = [module for _, module in iter_ternary_modules(model) if hasattr(module, "quant_enabled")]
    for module in modules:
        module.quant_enabled = False
    state = deepcopy(model.state_dict())
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    before_hooks = _hook_counts(model)
    for operation in ("cache", "train"):
        with pytest.raises(ValueError, match="original teacher prefix"):
            if operation == "cache":
                cache_window(model, _samples(), 1, 4, tmp_path / "cache", teacher_prefix="original")
            else:
                train(model, _samples(), TrainingConfig(epochs=3, teacher_prefix="original"),
                      cache_directory=tmp_path / "train")
        assert model.training and not any(module.quant_enabled for module in modules)
        assert parameter_flags == [parameter.requires_grad for parameter in model.parameters()]
        assert _hook_counts(model) == before_hooks
        _equal_tree(model.state_dict(), state)
        assert not list(tmp_path.iterdir())
