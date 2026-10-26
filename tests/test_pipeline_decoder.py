"""Native-fixture stage composition, not pretrained-model quality evidence."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from openbonsai.adapter import tiny_model
from openbonsai.candidates import source_weight_sha256
from openbonsai.endpoint_export import export_embedding_candidate, export_head_candidate
from openbonsai.endpoint_quantizer import BoundedTernaryHead
from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
from openbonsai.embedding_calibration import EmbeddingTrainingConfig, fit_embedding_table
from openbonsai.head_calibration import HeadTrainingConfig, cache_original_head_inputs, fit_head_from_cache
from openbonsai.mixed_export import load_mixed
from openbonsai.mixed_formats import MixedEmbedding, MixedLinear, load_candidate
from openbonsai.pipeline_decoder import fit_decoder_stage, assemble_stage, _embedding_export_certificate
from openbonsai.rotations import fixed_signs, rotation_targets
from openbonsai.text_export import text_module_name
from openbonsai.trainer import TrainingConfig
import openbonsai.pipeline_sources as sources


def context(tmp_path, monkeypatch, *, multimodal=True, coverage="all", output_scope="language-only"):
    quantization = dict(group_size=128, block_size=64, rotation_seed=17, lora_rank=2,
                        factor_mapping="sigmoid", hard_phase="frozen_jacobian", soft_fraction=.8,
                        max_sharpness=30., aggressive_gates=False)
    training = asdict(TrainingConfig(epochs=2, batch_size=1, window_size=4, stride=4,
                         teacher_prefix="original", seed=12))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"fixture":true}')
    job = {"decoder": {"quantization": quantization, "training": training, "packing": "auto"},
           "head": {"fixture": "actual bounded KL fit"}, "embedding": {"fixture": "actual full table fit"},
           "assembly": {"coverage": coverage, "output_scope": output_scope,
                        "budget_bytes": 10**8, "row_chunk_size": 17, "cache_decoded_weights": False},
           "code": {"sha256": "c" * 64},
           "source": {"model_directory": {"path": str(tmp_path / "sealed-model")},
                      "tokenizer_directory": {"path": str(tokenizer)}}}
    provenance = {"source_checkpoint": "sha256:" + "a" * 64,
                  "source_revision": "a" * 64, "source_content_sha256": "a" * 64}
    ctx = SimpleNamespace(job=job, job_path=tmp_path / "job-seal" / "job.json",
        job_sha256="b" * 64, source_provenance=provenance,
        tokenizer_directory=tokenizer, calibration_receipt={"sha256": "d" * 64},
        calibration_records=[{"input_ids": torch.tensor([[4, 7, 9, 11]]),
                              "attention_mask": torch.ones(1, 4, dtype=torch.long)}])
    loaded = []
    def original(_ctx, device="cpu"):
        assert _ctx is ctx
        model = tiny_model(seed=8, multimodal=multimodal).to(device=device, dtype=torch.bfloat16)
        loaded.append(model)
        return model
    monkeypatch.setattr(sources, "load_original", original)
    monkeypatch.setattr(sources, "verify_context", lambda value: None)
    return ctx, original, loaded


def endpoint_stages(ctx, original, tmp_path):
    model = original(ctx)
    prefix = "model.language_model" if model.config.model_type == "qwen3_5" else "model"
    head = BoundedTernaryHead(model.lm_head, group_size=128, row_chunk_size=31,
        signs=fixed_signs(128, seed=17, name="lm_head"), block_size=64, lora_rank=2)
    cache = cache_original_head_inputs(model, ctx.calibration_records, tmp_path / "head-cache",
                                      ctx.source_provenance, shard_positions=2)
    head_report = fit_head_from_cache(head, cache,
        HeadTrainingConfig(epochs=2, token_microbatch=2, vocabulary_chunk=31),
        source_provenance=ctx.source_provenance).report
    head_candidate = export_head_candidate(head, provenance=sources.candidate_provenance(
        ctx, "head", "lm_head", source_weight_sha256=head.source_weight_sha256))
    sources.publish_candidates(ctx, "head", {"lm_head": head_candidate}, tmp_path / "head", report=head_report)
    name = prefix + ".embed_tokens"
    embedding = BoundedTernaryEmbedding(model.get_input_embeddings(), group_size=128, row_chunk_size=31,
        signs=fixed_signs(128, seed=17, name=name), block_size=64, lora_rank=2)
    embedding_report = fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2),
                                          source_provenance=ctx.source_provenance).report
    embedding_candidate = export_embedding_candidate(embedding, provenance=sources.candidate_provenance(
        ctx, "embedding", name, source_weight_sha256=embedding.source_weight_sha256))
    sources.publish_candidates(ctx, "embedding", {name: embedding_candidate}, tmp_path / "embedding",
                               report=embedding_report)
    return [tmp_path / "head", tmp_path / "embedding"]


@pytest.mark.parametrize("multimodal", [True, False])
def test_actual_decoder_stage_preserves_original_sources_rng_signs_and_hard_state(tmp_path, monkeypatch, multimodal):
    ctx, original, loaded = context(tmp_path, monkeypatch, multimodal=multimodal, coverage="decoder")
    before = original(ctx)
    original_refs = {name: parameter for name, parameter in before.named_parameters()}
    values = {name: parameter.detach().clone() for name, parameter in original_refs.items()}
    monkeypatch.setattr(sources, "load_original", lambda *_a, **_k: before)
    rng = torch.random.get_rng_state().clone()
    events, operations = [], []
    def operation(label, fn):
        operations.append(label)
        return fn()
    receipt = fit_decoder_stage(ctx, tmp_path / "decoder", progress_callback=events.append, operation=operation)
    assert receipt["status"] == "complete"
    assert set(receipt["candidates"]) == set(rotation_targets(before))
    assert len(receipt["candidates"]) == 25
    assert receipt["report"]["reconstruction_objective"]["teacher_prefix"] == "original"
    assert len(events) == 2 and "decoder.train" in operations
    assert operations.index("decoder.construct") < operations.index("decoder.train")
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, parameter in original_refs.items():
        assert torch.equal(parameter, values[name])
    for name, desc in receipt["candidates"].items():
        module = before.get_submodule(name)
        assert module.weight.data_ptr() == original_refs[name + ".weight"].data_ptr()
        assert module.progress == 1 and module.quant_enabled
        assert all(not p.requires_grad and p.grad is None for p in module.trainable_parameters())
        candidate = load_candidate(tmp_path / "decoder" / desc["file"])
        assert candidate.metadata["provenance"]["original_module_name"] == name
        assert torch.equal(candidate.tensors["rotation_signs"], fixed_signs(module.in_features, seed=17, name=name))
        assert desc["source_weight_sha256"] == source_weight_sha256(original_refs[name + ".weight"])


@pytest.mark.parametrize("multimodal,output_scope", [(True, "language-only"), (True, "retain-vision"), (False, "language-only")])
def test_all_stage_assembly_exact_payload_native_reload_and_scopes(tmp_path, monkeypatch, multimodal, output_scope):
    ctx, original, loaded = context(tmp_path, monkeypatch, multimodal=multimodal, output_scope=output_scope)
    fit_decoder_stage(ctx, tmp_path / "decoder")
    endpoints = endpoint_stages(ctx, original, tmp_path)
    stages = [tmp_path / "decoder", *endpoints]
    import weakref
    import openbonsai.mixed_formats as formats
    real_load = formats.load_candidate
    live, peaks = [], []
    def tracked_load(path):
        candidate = real_load(path)
        live.append(weakref.ref(candidate))
        peaks.append(sum(item() is not None for item in live))
        return candidate
    monkeypatch.setattr(formats, "load_candidate", tracked_load)
    report = assemble_stage(ctx, stages, tmp_path / "assembled")
    assert max(peaks) <= 2  # serializer may transiently retain adjacent candidates
    assert report["module_count"] == 27 and report["candidate_payloads_unchanged"]
    artifact = tmp_path / "assembled" / "artifact"
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert int(manifest["storage"]["artifact_bytes"]) == sum(p.stat().st_size for p in artifact.rglob("*") if p.is_file())
    reference = original(ctx)
    original_config = reference.config.to_dict()
    for stage in stages:
        candidates, receipt = sources.load_candidate_stage(ctx, stage)
        for name in candidates:
            candidate = candidates[name]
            native = reference.get_submodule(name)
            replacement = (MixedEmbedding(candidate, padding_idx=native.padding_idx, row_chunk_size=17)
                           if candidate.module_kind == "embedding" else MixedLinear(candidate, row_chunk_size=17))
            reference.set_submodule(name, replacement)
            dest = report["original_to_artifact_module_names"][name]
            info = manifest["modules"][dest]
            assert (artifact / info["file"]).read_bytes() == (stage / receipt["candidates"][name]["file"]).read_bytes()
            if multimodal and output_scope == "language-only":
                assert dest == text_module_name(name)
    restored = load_mixed(artifact)
    sample = ctx.calibration_records[0]
    with torch.no_grad():
        expected = reference(**sample, use_cache=False).logits
        actual = restored(**sample, use_cache=False).logits
    assert torch.equal(expected, actual)
    assert restored.generation_config.eos_token_id == reference.generation_config.eos_token_id
    assert (restored.config.model_type == "qwen3_5") == (output_scope == "retain-vision")
    assert (int(manifest["storage"]["vision_file_bytes"]) > 0) == (output_scope == "retain-vision")
    # Fresh assembly never mutated its original source modules/configuration.
    assembly_source = loaded[-2]  # the later reference load is the only subsequent source load
    assert all(type(assembly_source.get_submodule(name)) is torch.nn.Linear
               for name in rotation_targets(assembly_source))
    assert assembly_source.config.to_dict() == original_config
    assert not (artifact / "stage.json").exists()


def test_decoder_only_is_explicit_and_preserves_bf16_endpoints(tmp_path, monkeypatch):
    ctx, original, _ = context(tmp_path, monkeypatch, coverage="decoder")
    fit_decoder_stage(ctx, tmp_path / "decoder")
    report = assemble_stage(ctx, [tmp_path / "decoder"], tmp_path / "assembled")
    assert report["module_count"] == 25
    model = load_mixed(tmp_path / "assembled" / "artifact")
    assert type(model.get_input_embeddings()) is torch.nn.Embedding
    assert type(model.lm_head) is torch.nn.Linear
    source = original(ctx)
    assert torch.equal(model.lm_head.weight, source.lm_head.weight)
    assert torch.equal(model.get_input_embeddings().weight, source.get_input_embeddings().weight)
    ctx.job["assembly"]["coverage"] = "all"
    with pytest.raises(ValueError, match="Incomplete"):
        assemble_stage(ctx, [tmp_path / "decoder"], tmp_path / "missing-endpoints")


@pytest.mark.parametrize("change", ["teacher", "quant", "packing", "block", "bool", "seed"])
def test_incomplete_or_invalid_recipe_fails_before_load(tmp_path, monkeypatch, change):
    ctx, _, _ = context(tmp_path, monkeypatch)
    if change == "teacher":
        del ctx.job["decoder"]["training"]["teacher_prefix"]
    elif change == "quant":
        del ctx.job["decoder"]["quantization"]["soft_fraction"]
    elif change == "packing":
        ctx.job["decoder"]["packing"] = "fake"
    elif change == "block":
        ctx.job["decoder"]["quantization"]["block_size"] = 3
    elif change == "bool":
        ctx.job["decoder"]["quantization"]["aggressive_gates"] = 1
    else:
        ctx.job["decoder"]["training"]["seed"] = True
    monkeypatch.setattr(sources, "load_original", lambda *_a, **_k: pytest.fail("must fail before load"))
    with pytest.raises(ValueError):
        fit_decoder_stage(ctx, tmp_path / "decoder")


def test_existing_output_does_not_train(tmp_path, monkeypatch):
    ctx, _, _ = context(tmp_path, monkeypatch)
    (tmp_path / "decoder").mkdir()
    monkeypatch.setattr(sources, "load_original", lambda *_a, **_k: pytest.fail("must fail before load"))
    with pytest.raises(FileExistsError):
        fit_decoder_stage(ctx, tmp_path / "decoder")


@pytest.mark.parametrize("damage", ["missing", "digest", "coverage", "source", "sign-config"])
def test_exported_embedding_certificate_is_required_and_consistent(tmp_path, monkeypatch, damage):
    import copy
    from openbonsai.export import _json_bytes
    ctx, original, _ = context(tmp_path, monkeypatch)
    endpoint_stages(ctx, original, tmp_path)
    candidates, _ = sources.load_candidate_stage(ctx, tmp_path / "embedding")
    name = next(iter(candidates))
    candidate = candidates[name]
    certificate = candidate.metadata["provenance"]["calibration_certificate"]
    if damage == "missing":
        del candidate.metadata["provenance"]["calibration_certificate"]
    elif damage == "digest":
        certificate["certificate_sha256"] = "0" * 64
    else:
        if damage == "coverage":
            certificate["coverage"]["hard_rows"] -= 1
        elif damage == "source":
            certificate["source_provenance"]["source_revision"] = "0" * 64
        else:
            certificate["configuration"]["block_size"] = 128
        body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
        certificate["certificate_sha256"] = hashlib.sha256(_json_bytes(body)).hexdigest()
    source = original(ctx).get_input_embeddings()
    with pytest.raises(ValueError, match="certificate|inconsistent"):
        _embedding_export_certificate(candidate, source, ctx.source_provenance)


def test_duplicate_stage_corruption_budget_and_scope_rejected(tmp_path, monkeypatch):
    ctx, original, _ = context(tmp_path, monkeypatch, multimodal=False, coverage="decoder")
    fit_decoder_stage(ctx, tmp_path / "decoder")
    with pytest.raises(ValueError, match="duplicate"):
        assemble_stage(ctx, [tmp_path / "decoder"] * 2, tmp_path / "duplicate")
    ctx.job["assembly"]["output_scope"] = "retain-vision"
    with pytest.raises(ValueError, match="retain-vision"):
        assemble_stage(ctx, [tmp_path / "decoder"], tmp_path / "fake-vision")
    ctx.job["assembly"].update(output_scope="language-only", budget_bytes=1)
    with pytest.raises(ValueError, match="budget"):
        assemble_stage(ctx, [tmp_path / "decoder"], tmp_path / "over-budget")
    assert not (tmp_path / "over-budget" / "artifact").exists()
    ctx.job["assembly"]["budget_bytes"] = 10**8
    receipt = json.loads((tmp_path / "decoder" / "stage.json").read_text())
    file = tmp_path / "decoder" / next(iter(receipt["candidates"].values()))["file"]
    content = file.read_bytes()
    file.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA|sha|hash"):
        assemble_stage(ctx, [tmp_path / "decoder"], tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()
