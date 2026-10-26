"""Explicit coverage derivatives, not new fitting or pretrained quality evidence."""
import copy
import hashlib
import json

import pytest
import torch

from openbonsai import pipeline
from openbonsai.export import _all_tensors, _under_module
from openbonsai.mixed_export import load_mixed
from openbonsai.mixed_formats import MixedEmbedding, MixedLinear
from openbonsai.pipeline_decoder import (assemble_stage, fit_decoder_stage,
                                        validate_assembly_control)
from openbonsai.text_export import text_module_name
import openbonsai.pipeline_sources as sources
from test_pipeline_decoder import context, endpoint_stages


ROLES = ("decoder", "embedding", "head")
CASES = [("D-only", ["decoder"]), ("E-only", ["embedding"]), ("H-only", ["head"]),
         ("D+E", ["decoder", "embedding"]), ("D+H", ["decoder", "head"])]


def control(label="E-only", roles=None, scope="language-only", budget=10**8):
    roles = ["embedding"] if roles is None else roles
    return {"quantized_components": roles,
            "retained_original_components": [role for role in ROLES if role not in roles],
            "output_scope": scope, "evaluation_label": label, "budget_bytes": budget}


@pytest.mark.parametrize("multimodal,scope", [(True, "language-only"), (True, "retain-vision"),
                                             (False, "language-only")])
def test_same_candidates_all_controls_preserve_exact_original_and_reload(tmp_path, monkeypatch,
                                                                        multimodal, scope):
    ctx, original, _ = context(tmp_path, monkeypatch, multimodal=multimodal, output_scope=scope)
    # A non-BF16-representable retained native exception makes a blanket cast visible.
    def preserved_source(*args, **kwargs):
        model = original(*args, **kwargs)
        for name, parameter in model.named_parameters():
            if name.endswith("A_log"):
                parameter.data = parameter.data.float().fill_(.1234567)
        return model
    monkeypatch.setattr(sources, "load_original", preserved_source)
    fit_decoder_stage(ctx, tmp_path / "decoder")
    endpoint_stages(ctx, preserved_source, tmp_path)
    dirs = {role: tmp_path / role for role in ROLES}
    producer_before = {str(p): p.read_bytes() for directory in dirs.values()
                       for p in directory.rglob("*") if p.is_file()}
    ctx.job["assembly"]["budget_bytes"] = 1  # Controls must not pretend this final target was met.
    job_before = copy.deepcopy(ctx.job)
    candidates_seen = {}
    for label, selected in CASES:
        request = control(label, selected, scope)
        request_before = copy.deepcopy(request)
        dest = tmp_path / label
        report = assemble_stage(ctx, [dirs[role] for role in selected], dest, assembly_control=request)
        assert ctx.job == job_before and request == request_before
        assert report["coverage"] == report["pipeline_assembly"]["coverage"] == "control"
        assert report["delivery_target_complete"] is False and report["quality_claim"] is False
        assert report["sealed_pipeline_assembly"]["budget_bytes"] == 1
        assert report["pipeline_assembly"]["budget_bytes"] == request["budget_bytes"]
        assert report["pipeline_assembly"]["row_chunk_size"] == 17
        assert report["pipeline_assembly"]["cache_decoded_weights"] is False
        assert report["pipeline_job_sha256"] == ctx.job_sha256
        assert set(report["candidate_stage_receipts_sha256"]) == set(selected)
        artifact = dest / "artifact"
        manifest = json.loads((artifact / "manifest.json").read_bytes())
        assert manifest["metadata"]["assembly_control"] == request
        assert int(manifest["storage"]["artifact_bytes"]) == sum(p.stat().st_size for p in artifact.rglob("*") if p.is_file())
        assert 1 < int(manifest["storage"]["artifact_bytes"]) <= request["budget_bytes"]
        assert not (artifact / "stage.json").exists()
        expected = preserved_source(ctx)
        original_tensors = {name: (value.clone(), kind, persistent)
                            for name, value, kind, persistent in _all_tensors(expected)}
        selected_names = set()
        for role in selected:
            candidates, receipt = sources.load_candidate_stage(ctx, dirs[role])
            for name in candidates:
                selected_names.add(name)
                candidate = candidates[name]
                native = expected.get_submodule(name)
                target = report["original_to_artifact_module_names"][name]
                payload = (artifact / manifest["modules"][target]["file"]).read_bytes()
                assert payload == (dirs[role] / receipt["candidates"][name]["file"]).read_bytes()
                if name in candidates_seen:
                    assert payload == candidates_seen[name]
                candidates_seen[name] = payload
                replacement = (MixedEmbedding(candidate, padding_idx=native.padding_idx, row_chunk_size=17)
                               if role == "embedding" else MixedLinear(candidate, row_chunk_size=17))
                expected.set_submodule(name, replacement)
        assert len(manifest["modules"]) == sum(25 if role == "decoder" else 1 for role in selected)
        restored = load_mixed(artifact, cache_decoded_weights=False)
        restored_tensors = {name: (value, kind, persistent) for name, value, kind, persistent in _all_tensors(restored)}
        fp32_exceptions = 0
        for name, (source_value, kind, persistent) in original_tensors.items():
            if _under_module(name, selected_names) or (scope == "language-only" and name.startswith("model.visual.")):
                continue
            target = text_module_name(name) if multimodal and scope == "language-only" else name
            actual, saved_kind, saved_persistent = restored_tensors[target]
            assert actual.dtype == source_value.dtype and torch.equal(actual, source_value)
            assert (saved_kind, saved_persistent) == (kind, persistent)
            fp32_exceptions += int(name.endswith("A_log") and actual.dtype == torch.float32)
        assert fp32_exceptions > 0
        for role, accounting in report["retained_original_component_accounting"].items():
            rows = [desc for name, desc in manifest["retained_tensors"].items()
                    if _under_module(name, accounting["artifact_module_names"])]
            assert accounting["tensor_count"] == len(rows)
            assert accounting["tensor_bytes"] == sum(row["tensor_bytes"] for row in rows)
            assert accounting["bytes_by_dtype"] == {"bfloat16": accounting["tensor_bytes"]}
        sample = {"input_ids": torch.tensor([[4, 7, 4, 11, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}
        with torch.no_grad():
            assert torch.equal(expected(**sample, use_cache=False).logits,
                               restored(**sample, use_cache=False).logits)
        assert restored.generation_config.eos_token_id == expected.generation_config.eos_token_id
        assert (int(manifest["storage"]["vision_file_bytes"]) > 0) == (scope == "retain-vision")
    assert producer_before == {str(p): p.read_bytes() for directory in dirs.values()
                              for p in directory.rglob("*") if p.is_file()}
    # Full delivery has no override and still uses the unchanged final budget.
    with pytest.raises(ValueError, match="budget"):
        assemble_stage(ctx, list(dirs.values()), tmp_path / "all-over-budget")
    assert not (tmp_path / "all-over-budget" / "artifact").exists()


@pytest.mark.parametrize("damage", ["empty", "full", "unknown", "duplicate", "order", "label", "complement",
                                   "null", "scope", "budget", "bool-budget", "cache-override", "dtype-override"])
def test_invalid_controls_fail_before_load_and_output(tmp_path, monkeypatch, damage):
    ctx, _, _ = context(tmp_path, monkeypatch)
    request = control()
    if damage == "empty": request["quantized_components"] = []
    elif damage == "full": request["quantized_components"] = list(ROLES)
    elif damage == "unknown": request["quantized_components"] = ["vision"]
    elif damage == "duplicate": request["quantized_components"] = ["embedding", "embedding"]
    elif damage == "order": request = control("D+E", ["embedding", "decoder"])
    elif damage == "label": request["evaluation_label"] = "H-only"
    elif damage == "complement": request["retained_original_components"] = ["head"]
    elif damage == "null": request = None
    elif damage == "scope": request["output_scope"] = "retain-vision"
    elif damage == "budget": request["budget_bytes"] = 0
    elif damage == "bool-budget": request["budget_bytes"] = True
    elif damage == "cache-override": request["cache_decoded_weights"] = True
    else: request["dtype"] = "bfloat16"
    monkeypatch.setattr(sources, "load_original", lambda *_a, **_k: pytest.fail("must fail before source load"))
    if damage == "null":
        with pytest.raises(ValueError):
            pipeline.validate_request({"format": pipeline.REQUEST_FORMAT, "stage": "assemble", "parameters": {
                "job": {}, "candidate_stages": [{}], "assembly_control": None}})
    else:
        with pytest.raises(ValueError):
            assemble_stage(ctx, [tmp_path / "unused"], tmp_path / "output", assembly_control=request)
    assert not (tmp_path / "output").exists()


def test_decoder_seal_cannot_expand_or_relax_coverage(tmp_path, monkeypatch):
    ctx, _, _ = context(tmp_path, monkeypatch, coverage="decoder")
    monkeypatch.setattr(sources, "load_original", lambda *_a, **_k: pytest.fail("must fail before source load"))
    for request in [control(), control("D-only", ["decoder"])]:
        with pytest.raises(ValueError, match="proper nonempty subset"):
            assemble_stage(ctx, [tmp_path / "unused"], tmp_path / "output", assembly_control=request)


def test_control_rejects_missing_extra_duplicate_corrupt_and_foreign_stages(tmp_path, monkeypatch):
    ctx, original, _ = context(tmp_path, monkeypatch)
    endpoint_stages(ctx, original, tmp_path)
    request = control()
    for label, stages in [("missing", []), ("extra", [tmp_path / "embedding", tmp_path / "head"]),
                          ("wrong", [tmp_path / "head"]),
                          ("duplicate", [tmp_path / "embedding"] * 2)]:
        with pytest.raises(ValueError):
            assemble_stage(ctx, stages, tmp_path / label, assembly_control=request)
        assert not (tmp_path / label).exists()
    with pytest.raises(ValueError, match="budget"):
        assemble_stage(ctx, [tmp_path / "embedding"], tmp_path / "budget", assembly_control=control(budget=1))
    assert not (tmp_path / "budget" / "artifact").exists()
    receipt_path = tmp_path / "embedding" / "stage.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    receipt["job_sha256"] = "f" * 64
    receipt = sources._seal_digest(receipt)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="job|stage|Stage"):
        assemble_stage(ctx, [tmp_path / "embedding"], tmp_path / "foreign", assembly_control=request)
    receipt_path.write_bytes(receipt_bytes)
    receipt = json.loads(receipt_bytes)
    payload = tmp_path / "embedding" / next(iter(receipt["candidates"].values()))["file"]
    data = payload.read_bytes()
    payload.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA|sha|hash"):
        assemble_stage(ctx, [tmp_path / "embedding"], tmp_path / "corrupt", assembly_control=request)
    assert not (tmp_path / "corrupt").exists()
