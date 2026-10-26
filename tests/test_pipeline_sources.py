"""Local receipts, byte-only tokenizer packaging and tiny native source loading."""
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from openbonsai.adapter import tiny_model
from openbonsai.candidates import source_weight_sha256
from openbonsai.embedding_calibration import EmbeddingTrainingConfig
from openbonsai.head_calibration import HeadTrainingConfig
from openbonsai.mixed_formats import ternary_candidate
from openbonsai.packing import pack_grouped_weight
from openbonsai.pipeline_sources import (candidate_provenance, checkpoint_tensors, code_identity,
    json_sha256, load_candidate_stage, load_context, load_original, load_original_source,
    package_tokenizer, publish_candidates, seal_job, tokenizer_identity, validate_stage_output, verify_context)
from openbonsai.eval_backends import _identity, local_directory_identity
from openbonsai.rotations import fixed_signs
from openbonsai.trainer import TrainingConfig


def recipes():
    common = dict(group_size=128, block_size=64, rotation_seed=11, lora_rank=2,
                  factor_mapping="sigmoid", hard_phase="frozen_jacobian", soft_fraction=.8,
                  max_sharpness=30.)
    endpoint = dict(common, row_chunk_size=32, initialization_seed=31)
    return dict(decoder={"quantization": dict(common, aggressive_gates=False),
                         "training": asdict(TrainingConfig(epochs=2, batch_size=1, window_size=1, stride=1)),
                         "packing": "auto"},
                head={"quantization": dict(endpoint), "training": asdict(HeadTrainingConfig()),
                      "packing": "auto", "cache": {"max_positions": None, "shard_positions": 4}},
                embedding={"quantization": dict(endpoint), "training": asdict(EmbeddingTrainingConfig()), "packing": "auto"},
                assembly={"coverage": "all", "output_scope": "language-only", "budget_bytes": 100000000,
                          "row_chunk_size": 32, "cache_decoded_weights": False})


def make_source(root, *, multimodal=False):
    torch.set_num_threads(2)
    model = tiny_model(seed=993, multimodal=multimodal).to(torch.bfloat16).eval().requires_grad_(False)
    layers = model.model.language_model.layers if multimodal else model.model.layers
    # Values intentionally not representable in BF16; source loader must restore them.
    layers[0].linear_attn.A_log.data = torch.full_like(layers[0].linear_attn.A_log, .1234567, dtype=torch.float32)
    model.save_pretrained(root)
    return model


def make_tokenizer(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_text('{"version":"1.0","model":{}}')
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')
    (root / "chat_template.jinja").write_bytes(b"{{ messages }}\r\nsecond\r\n")
    return root


def producer(path, source, tokenizer, trace, **updates):
    body = {"kind": "caller-declared-token-records", "description": "Synthetic exact-ID source fixture, no quality evidence",
            "source_content_sha256": local_directory_identity(source)["sha256"],
            "tokenizer_content_sha256": tokenizer_identity(tokenizer)["sha256"],
            "source_jsonl_sha256": _identity(trace)["sha256"], **updates}
    path.write_text(json.dumps(body))
    return {"path": str(path), "sha256": _identity(path)["sha256"]}


def make_job(root, *, multimodal=False, traces=None, extra_recipes=None):
    source = root / "source"
    make_source(source, multimodal=multimodal)
    tokenizer = make_tokenizer(root / "tokenizer-input")
    trace = root / "traces.jsonl"
    trace.write_text(json.dumps({"input_ids": [4, 5, 6, 7], "attention_mask": [1, 1, 1, 0]}) + "\n")
    receipt = producer(root / "producer.json", source, tokenizer, trace)
    spec = {"path": str(trace), "sequence_length": 6, "max_samples": None, "producer": receipt}
    configs = recipes()
    if extra_recipes:
        configs.update(extra_recipes)
    return seal_job(model_directory=source, tokenizer_directory=tokenizer,
        traces_by_role=traces or {"calibration": spec}, output=root / "job", **configs)


@pytest.mark.parametrize("multimodal", [False, True])
def test_native_original_load_preserves_fp32_exceptions_and_source_files(tmp_path, multimodal):
    source = tmp_path / "source"
    original = make_source(source, multimodal=multimodal)
    info = {"model_directory": local_directory_identity(source), "checkpoint_tensors": checkpoint_tensors(source)}
    loaded = load_original_source(info)
    for name, tensor in original.state_dict().items():
        result = loaded.state_dict()[name]
        assert result.dtype == tensor.dtype
        torch.testing.assert_close(result, tensor, rtol=0, atol=0)
    assert not any(p.requires_grad for p in loaded.parameters())
    assert all(not m.training for m in loaded.modules())
    assert loaded._openbonsai_source_receipt["restored_saved_dtype_tensors"]
    assert local_directory_identity(source) == info["model_directory"]


def test_unused_mtp_is_explicit_and_not_compression(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    weights = load_file(source / "model.safetensors")
    weights["mtp.test.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
    save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
    loaded = load_original_source({"model_directory": local_directory_identity(source), "checkpoint_tensors": checkpoint_tensors(source)})
    assert set(loaded._openbonsai_source_receipt["unused_mtp_tensors"]) == {"mtp.test.weight"}
    assert loaded._openbonsai_source_receipt["unused_mtp_is_not_compression"] is True


def test_tokenizer_package_copies_exact_assets_not_weights(tmp_path):
    source = make_tokenizer(tmp_path / "source")
    (source / "model.safetensors").write_bytes(b"do not copy checkpoint")
    (source / "config.json").write_text("{}")
    (source / "preprocessor_config.json").write_text("{}")
    (source / "chat_templates").mkdir()
    (source / "chat_templates" / "named.jinja").write_bytes(b"named\r\n")
    result = package_tokenizer(source, tmp_path / "bundle")
    assert set(result["files"]) == {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_templates/named.jinja"}
    for name in result["files"]:
        assert (tmp_path / "bundle" / name).read_bytes() == (source / name).read_bytes()
    with pytest.raises(FileExistsError):
        package_tokenizer(source, tmp_path / "bundle")


@pytest.mark.parametrize("field", ["auto_map", "fast_tokenizer_files"])
def test_unsupported_tokenizer_assets_fail_closed(tmp_path, field):
    source = make_tokenizer(tmp_path / "source")
    (source / "tokenizer_config.json").write_text(json.dumps({field: ["custom"]}))
    with pytest.raises(ValueError, match="Custom or versioned"):
        package_tokenizer(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_tokenizer_symlink_rejected(tmp_path):
    source = make_tokenizer(tmp_path / "source")
    (source / "chat_template.jinja").unlink()
    (source / "chat_template.jinja").symlink_to(source / "tokenizer.json")
    with pytest.raises(ValueError, match="regular file"):
        tokenizer_identity(source)


def test_job_round_trip_source_and_records_receipts(tmp_path):
    context = make_job(tmp_path)
    restored = load_context(context.job_path)
    assert restored.job == context.job
    assert context.calibration_receipt["role"] == "calibration"
    assert context.calibration_records[0]["input_ids"].tolist() == [[4, 5, 6, 7, 0, 0]]
    assert context.calibration_records[0]["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0]]
    assert context.calibration_receipt["padding_tokens"] == 2
    assert context.calibration_receipt["producer"]["kind"] == "caller-declared-token-records"
    assert "not independent generation proof" in context.calibration_receipt["provenance_status"]
    assert context.source_provenance["source_checkpoint"] == "sha256:" + context.source_provenance["source_content_sha256"]
    assert load_original(context).config.model_type == "qwen3_5_text"


@pytest.mark.parametrize("mutation", ["model", "tokenizer", "extra-tokenizer-file", "data", "producer", "memory", "job"])
def test_context_refuses_every_changed_input(tmp_path, mutation):
    context = make_job(tmp_path)
    if mutation == "memory":
        context.calibration_records[0]["input_ids"][0, 0] = 9
    elif mutation == "model":
        (tmp_path / "source" / "config.json").write_text("{}")
    elif mutation == "tokenizer":
        (context.tokenizer_directory / "chat_template.jinja").write_text("changed")
    elif mutation == "extra-tokenizer-file":
        (context.tokenizer_directory / "model.safetensors").write_bytes(b"injected")
    elif mutation in {"data", "producer"}:
        key = "file" if mutation == "data" else "producer_file"
        Path(context.calibration_receipt[key]["path"]).write_text("changed")
    else:
        context.job["assembly"]["budget_bytes"] += 1
    with pytest.raises((ValueError, RuntimeError)):
        verify_context(context)


def test_source_matrix_fp32_is_not_silently_cast(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    weights = load_file(source / "model.safetensors")
    weights["lm_head.weight"] = weights["lm_head.weight"].float()
    save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="already be BF16"):
        checkpoint_tensors(source)


def test_partial_recipe_is_rejected_before_job_creation(tmp_path):
    settings = recipes()
    del settings["head"]["training"]["learning_rate"]
    with pytest.raises(ValueError, match="All head training"):
        make_job(tmp_path, extra_recipes=settings)
    assert not (tmp_path / "job").exists()


def candidate(context, stage="head", name="lm_head"):
    original = torch.zeros(2, 128, dtype=torch.bfloat16)
    state = pack_grouped_weight(torch.zeros_like(original, dtype=torch.int8), torch.ones(2), shape=original.shape, group_size=128)
    return ternary_candidate(state, packing="auto", provenance=candidate_provenance(context, stage, name,
        source_weight_sha256=source_weight_sha256(original)), rotation_signs=fixed_signs(128, seed=11, name=name), rotation_block_size=64)


def test_completed_candidates_lazy_round_trip_and_rotation_identity(tmp_path, monkeypatch):
    context = make_job(tmp_path)
    expected = candidate(context)
    class Lazy(Mapping):
        def __iter__(self): return iter(["lm_head"])
        def __len__(self): return 1
        def __getitem__(self, key):
            assert key == "lm_head"
            return expected
    receipt = publish_candidates(context, "head", Lazy(), tmp_path / "stage", report={"quality_claim": False})
    import openbonsai.mixed_formats as formats
    actual_load = formats.load_candidate
    calls = []
    monkeypatch.setattr(formats, "load_candidate", lambda path: (calls.append(path), actual_load(path))[1])
    lazy, reloaded = load_candidate_stage(context, tmp_path / "stage", expected_stage="head")
    assert not calls
    actual = lazy["lm_head"]
    assert len(calls) == 1 and receipt == reloaded
    assert actual.metadata == expected.metadata
    for name in actual.tensors:
        torch.testing.assert_close(actual.tensors[name], expected.tensors[name], rtol=0, atol=0)
    assert receipt["candidates"]["lm_head"]["rotation_signs_sha256"] == hashlib.sha256(expected.tensors["rotation_signs"].numpy().tobytes()).hexdigest()
    with pytest.raises(FileExistsError):
        publish_candidates(context, "head", Lazy(), tmp_path / "stage", report={})


def test_partial_candidate_stage_never_loads(tmp_path):
    context = make_job(tmp_path)
    item = candidate(context)
    item.metadata["provenance"]["pipeline_job_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="before packing"):
        publish_candidates(context, "head", {"lm_head": item}, tmp_path / "partial", report={})
    assert not (tmp_path / "partial" / "stage.json").exists()
    with pytest.raises(FileNotFoundError):
        load_candidate_stage(context, tmp_path / "partial")


def test_candidate_changed_after_index_load_is_rejected(tmp_path):
    context = make_job(tmp_path)
    publish_candidates(context, "head", {"lm_head": candidate(context)}, tmp_path / "stage", report={})
    lazy, _ = load_candidate_stage(context, tmp_path / "stage")
    (tmp_path / "stage" / "candidate-00000.obc").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        lazy["lm_head"]


def test_code_identity_binds_pipeline_source():
    assert "pipeline_sources.py" in code_identity()["files"]


@pytest.mark.parametrize("field", ["source_content_sha256", "tokenizer_content_sha256", "source_jsonl_sha256"])
def test_producer_must_bind_all_exact_inputs(tmp_path, field):
    context = make_job(tmp_path)
    # Construct a separately authenticated but inconsistent producer receipt.
    path = tmp_path / "wrong-producer.json"
    value = dict(context.calibration_receipt["producer"], **{field: "f" * 64})
    path.write_text(json.dumps(value))
    spec = {"path": context.calibration_receipt["file"]["path"], "sequence_length": 6,
            "max_samples": None, "producer": {"path": str(path), "sha256": _identity(path)["sha256"]}}
    with pytest.raises(ValueError, match="must bind"):
        seal_job(model_directory=tmp_path / "source", tokenizer_directory=context.tokenizer_directory,
                 traces_by_role={"calibration": spec}, output=tmp_path / "wrong-job", **recipes())
    assert not (tmp_path / "wrong-job").exists()


def test_roles_cannot_reuse_exact_input_records(tmp_path):
    context = make_job(tmp_path)
    spec = {"path": context.calibration_receipt["file"]["path"], "sequence_length": 6,
            "max_samples": None, "producer": {key: context.calibration_receipt["producer_file"][key] for key in ("path", "sha256")}}
    with pytest.raises(ValueError, match="overlap"):
        seal_job(model_directory=tmp_path / "source", tokenizer_directory=context.tokenizer_directory,
                 traces_by_role={"calibration": spec, "evaluation": spec}, output=tmp_path / "overlap", **recipes())


def test_real_fast_tokenizer_and_crlf_template_survive_exact_package(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    original = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    original.save_pretrained(tmp_path / "tokens")
    template = b"{% for message in messages %}{{ message['content'] }}\r\n{% endfor %}"
    (tmp_path / "tokens" / "chat_template.jinja").write_bytes(template)
    package_tokenizer(tmp_path / "tokens", tmp_path / "bundle")
    first = AutoTokenizer.from_pretrained(tmp_path / "tokens", local_files_only=True, trust_remote_code=False)
    second = AutoTokenizer.from_pretrained(tmp_path / "bundle", local_files_only=True, trust_remote_code=False)
    messages = [{"role": "user", "content": "hello world"}]
    assert first.chat_template == second.chat_template
    assert first.apply_chat_template(messages) == second.apply_chat_template(messages)
    assert (tmp_path / "bundle" / "chat_template.jinja").read_bytes() == template


def test_original_source_directory_is_never_output(tmp_path):
    tokens = make_tokenizer(tmp_path / "tokens")
    with pytest.raises(ValueError, match="original source"):
        package_tokenizer(tokens, tokens / "nested")
    assert not (tokens / "nested").exists()


@pytest.mark.parametrize("tree", ["model", "tokenizer", "job", "code"])
def test_stage_output_rejected_before_touching_immutable_inputs(tmp_path, tree):
    context = make_job(tmp_path)
    import openbonsai.pipeline_sources as module
    root = {"model": tmp_path / "source", "tokenizer": context.tokenizer_directory,
            "job": context.job_path.parent, "code": Path(module.__file__).parent}[tree]
    output = root / "never-created-pipeline-test-output"
    with pytest.raises(ValueError, match="outside sealed"):
        validate_stage_output(context, output)
    with pytest.raises(ValueError, match="outside sealed"):
        publish_candidates(context, "head", {"lm_head": candidate(context)}, output, report={})
    assert not output.exists()
    verify_context(context)
