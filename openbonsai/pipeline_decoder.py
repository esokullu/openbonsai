"""Explicit rotated decoder stages and source-bound native mixed assembly.

This joins existing research operators; it defines no new calibration recipe.
Candidate payloads retain original module names and signs until assembly maps
only their manifest keys. Every stage owns one original model, with no deepcopy.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import copy
import hashlib
import json
import math

import torch
from torch import nn

from .adapter import _text_module
from .candidates import source_weight_sha256
from .mixed_export import prepare_mixed_export, save_mixed
from .rotations import RotatedTernaryLinear, fixed_signs, rotation_targets
from .text_export import extract_text_model, text_module_name
from .trainer import TrainingConfig, train


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sign_sha(signs):
    return hashlib.sha256(signs.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _call(operation, label, callback):
    return callback() if operation is None else operation(label, callback)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} requires exactly {sorted(expected)}")


_COMPONENTS = ("decoder", "embedding", "head")
_CONTROL_LABELS = {
    ("decoder",): "D-only", ("embedding",): "E-only", ("head",): "H-only",
    ("decoder", "embedding"): "D+E", ("decoder", "head"): "D+H",
}


def validate_assembly_control(control, *, sealed_assembly=None):
    """Validate an explicit derivative without editing the sealed job.

    Full coverage uses ordinary assembly and its sealed budget. Controls have
    their own complete-artifact budget because omitted matrices stay original.
    """
    _keys(control, {"quantized_components", "retained_original_components", "output_scope",
                    "evaluation_label", "budget_bytes"}, "assembly_control")
    for key in ("quantized_components", "retained_original_components"):
        roles = control[key]
        if (not isinstance(roles, list) or not roles
                or any(not isinstance(role, str) or role not in _COMPONENTS for role in roles)
                or roles != [role for role in _COMPONENTS if role in roles]):
            raise ValueError(f"{key} requires unique canonical component names")
    selected = tuple(control["quantized_components"])
    if selected not in _CONTROL_LABELS or control["evaluation_label"] != _CONTROL_LABELS[selected]:
        raise ValueError("Unsupported control components or mismatched evaluation label")
    if control["retained_original_components"] != [role for role in _COMPONENTS if role not in selected]:
        raise ValueError("Control must declare the exact retained-original complement")
    if (not isinstance(control["output_scope"], str)
            or control["output_scope"] not in {"language-only", "retain-vision"}):
        raise ValueError("Invalid control output scope")
    if type(control["budget_bytes"]) is not int or control["budget_bytes"] <= 0:
        raise ValueError("Control budget_bytes must be a positive integer")
    if sealed_assembly is not None:
        sealed = set(_COMPONENTS if sealed_assembly["coverage"] == "all" else ("decoder",))
        if not set(selected) < sealed:
            raise ValueError("Control must be a proper nonempty subset of sealed coverage")
        if control["output_scope"] != sealed_assembly["output_scope"]:
            raise ValueError("Control output scope must match the sealed assembly")
    return copy.deepcopy(control)


def _recipe(context):
    value = copy.deepcopy(context.job["decoder"])
    _keys(value, {"quantization", "training", "packing"}, "decoder")
    quant = value["quantization"]
    _keys(quant, {"group_size", "block_size", "rotation_seed", "lora_rank", "factor_mapping",
                  "hard_phase", "soft_fraction", "max_sharpness", "aggressive_gates"}, "decoder.quantization")
    for key in ("group_size", "block_size"):
        if type(quant[key]) is not int or quant[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if quant["block_size"] & (quant["block_size"] - 1):
        raise ValueError("block_size must be a power of two")
    if type(quant["rotation_seed"]) is not int or type(quant["lora_rank"]) is not int or quant["lora_rank"] < 0:
        raise ValueError("rotation_seed must be an integer and lora_rank nonnegative")
    if type(quant["aggressive_gates"]) is not bool:
        raise ValueError("aggressive_gates must be a bool")
    if quant["factor_mapping"] not in {"exp", "sigmoid"} or quant["hard_phase"] not in {"frozen_jacobian", "soft_surrogate"}:
        raise ValueError("Unsupported factor mapping or hard phase")
    for key in ("soft_fraction", "max_sharpness"):
        if type(quant[key]) not in (int, float) or not math.isfinite(quant[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < quant["soft_fraction"] < 1 or quant["max_sharpness"] <= 0:
        raise ValueError("soft_fraction must be in (0,1), max_sharpness positive")
    _keys(value["training"], {field.name for field in fields(TrainingConfig)}, "decoder.training")
    training = TrainingConfig(**value["training"])
    training.validate()
    if (type(training.seed) is not int or not -(2**63) <= training.seed < 2**64
            or type(training.gradient_checkpointing) is not bool):
        raise ValueError("training seed/checkpointing types are invalid")
    if any(type(value["training"][key]) not in (int, float) for key in
           ("learning_rate", "lora_learning_rate", "weight_decay", "grad_clip", "huber_delta")):
        raise ValueError("Training numeric settings must be real numbers, not bools")
    if value["packing"] not in {"auto", "2bit", "bitcos", "base3"}:
        raise ValueError("Unsupported packing")
    return value, quant, training


def _target_sources(model, quant):
    names = rotation_targets(model, endpoints=False, aggressive_gates=quant["aggressive_gates"])
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected unique nonempty decoder coverage")
    sources = {}
    for name in names:
        module = model.get_submodule(name)
        if type(module) is not nn.Linear or module.weight.dtype != torch.bfloat16:
            raise ValueError(f"Expected original native BF16 Linear: {name}")
        if module.in_features % quant["block_size"]:
            raise ValueError(f"Rotation block does not divide input width: {name}")
        sources[name] = source_weight_sha256(module.weight)
    return names, sources


class _DecoderCandidates(Mapping):
    """Generate only one projection's packed payload at a time."""
    def __init__(self, context, model, names, sources, report_sha256, packing):
        self.context = context
        self.model, self.names, self.sources = model, tuple(names), sources
        self.report_sha256, self.packing = report_sha256, packing

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        if name not in self.sources:
            raise KeyError(name)
        module = self.model.get_submodule(name)
        if source_weight_sha256(module.weight) != self.sources[name]:
            raise ValueError(f"Original decoder weight changed during calibration: {name}")
        from .pipeline_sources import candidate_provenance
        provenance = candidate_provenance(self.context, "decoder", name,
            source_weight_sha256=self.sources[name], training_report_sha256=self.report_sha256)
        return module.mixed_candidate(packing=self.packing, provenance=provenance)


def fit_decoder_stage(context, output, device="cpu", *, progress_callback=None, operation=None):
    """Calibrate and publish rotated decoder candidates from one sealed source.

    The caller's stage supervisor owns process/deadline/resource limits. Existing
    train() owns its per-window optimizers and disk caches; no resumable training
    checkpoint or endpoint fitting is implied by this completed stage.
    """
    from .pipeline_sources import load_original, publish_candidates, verify_context, validate_stage_output
    recipe, quant, training = _recipe(context)
    output = validate_stage_output(context, output)
    verify_context(context)
    model = _call(operation, "decoder.load_original", lambda: load_original(context, device=device))
    names, sources = _call(operation, "decoder.source_hashes", lambda: _target_sources(model, quant))
    # Resolve every target/shape before mutating modules. Original BF16 storage
    # is shared by each replacement; source values are never cloned or trained.
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(training.seed)
        def construct():
            for name in names:
                original = model.get_submodule(name)
                replacement = RotatedTernaryLinear(
                    original, quant["group_size"], signs=fixed_signs(original.in_features,
                        seed=quant["rotation_seed"], name=name), block_size=quant["block_size"],
                    lora_rank=quant["lora_rank"], factor_mapping=quant["factor_mapping"],
                    hard_phase=quant["hard_phase"], soft_fraction=quant["soft_fraction"],
                    max_sharpness=quant["max_sharpness"], copy_weight=False)
                model.set_submodule(name, replacement)
        _call(operation, "decoder.construct", construct)
        # Context records are CPU inputs bound to this source and tokenizer.
        # They are replayed by the existing trainer without changing IDs/masks.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="openbonsai-decoder-cache-") as cache:
            report = _call(operation, "decoder.train", lambda: train(
                model, context.calibration_records, training, progress_callback=progress_callback,
                cache_directory=cache))
    signs = {name: {"sha256": _sign_sha(model.get_submodule(name).rotation_signs),
                    "block_size": quant["block_size"], "convention": "row-DH"} for name in names}
    report.update(stage="decoder", original_module_names=names, source_weight_sha256=sources,
                  rotations=signs, recipe=recipe, calibration=copy.deepcopy(context.calibration_receipt),
                  source_provenance=copy.deepcopy(context.source_provenance),
                  native_source_receipt=copy.deepcopy(getattr(model, "_openbonsai_source_receipt", {})),
                  export_precision="hard ternary codes; stored scales and bias are rounded to FP16 before lossless packing",
                  ownership="one original model; frozen BF16 source storage shared by replacements; one candidate at a time",
                  deployment_runtime="reference float reconstruction; no low-bit kernel")
    candidates = _DecoderCandidates(context, model, names, sources, _sha(report), recipe["packing"])
    verify_context(context)
    return _call(operation, "decoder.publish", lambda: publish_candidates(
        context, "decoder", candidates, output, report=report))


def _embedding_export_certificate(candidate, source, provenance):
    """Check the exported coverage receipt, without claiming live-state resume.

    The producer checked live factors/signs/source before and after packing.
    Here stage-file hashes authenticate that already certified candidate; this
    check rejects an accidentally omitted or inconsistent coverage receipt.
    """
    from .embedding_calibration import CERTIFICATE_FORMAT
    from .export import _json_bytes
    certificate = candidate.metadata["provenance"].get("calibration_certificate")
    if not isinstance(certificate, dict) or certificate.get("format") != CERTIFICATE_FORMAT:
        raise ValueError("Embedding candidate lacks complete calibration certificate")
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    if hashlib.sha256(_json_bytes(body)).hexdigest() != certificate.get("certificate_sha256"):
        raise ValueError("Embedding candidate calibration certificate was modified")
    coverage = certificate.get("coverage", {})
    config = certificate.get("configuration", {})
    rows, width = candidate.shape
    soft, hard, total = (certificate.get(key) for key in ("soft_sweeps", "hard_sweeps", "completed_sweeps"))
    if (any(type(count) is not int or count <= 0 for count in (soft, hard, total))
            or total != soft + hard
            or certificate.get("source_shape") != [rows, width]
            or certificate.get("source_provenance") != provenance
            or certificate.get("source_weight_sha256") != candidate.metadata["provenance"]["source_weight_sha256"]
            or any(coverage.get(key) != rows for key in
                   ("vocabulary_rows", "soft_rows", "hard_rows", "last_soft_rows", "final_hard_rows"))
            or any(coverage.get(f"{phase}_visits_{extreme}") != count
                   for phase, count in (("soft", soft), ("hard", hard)) for extreme in ("min", "max"))
            or config.get("padding_idx") != source.padding_idx
            or config.get("group_size") != candidate.metadata["group_size"]
            or config.get("block_size") != candidate.metadata["rotation"]["block_size"]
            or config.get("rotated") is not True or config.get("rotation_only") is not False
            or config.get("progress") != 1. or config.get("quant_enabled") is not True):
        raise ValueError("Embedding candidate calibration coverage/source/configuration is inconsistent")


class _AssemblyCandidates(Mapping):
    def __init__(self, by_original, name_map, native, provenance):
        self.by_original, self.name_map = by_original, name_map
        self.inverse = {target: original for original, target in name_map.items()}
        self.native, self.provenance = native, provenance
        if len(self.inverse) != len(name_map):
            raise ValueError("Candidate name mapping is not one-to-one")

    def __iter__(self):
        return iter(sorted(self.inverse))

    def __len__(self):
        return len(self.inverse)

    def __getitem__(self, target):
        original = self.inverse[target]
        candidate = self.by_original[original][original]
        source = self.native.get_submodule(original)
        details = candidate.metadata["provenance"]
        if details.get("original_module_name") != original:
            raise ValueError(f"Candidate original module identity mismatch: {original}")
        if any(details.get(key) != value for key, value in self.provenance.items()):
            raise ValueError(f"Candidate source provenance mismatch: {original}")
        if (not candidate.encoding.startswith("ternary-") or "rotation" not in candidate.metadata
                or "rotation_signs" not in candidate.tensors):
            raise ValueError(f"Expected saved rotated ternary candidate: {original}")
        expected_kind = "embedding" if isinstance(source, nn.Embedding) else "linear"
        if candidate.module_kind != expected_kind or list(candidate.shape) != list(source.weight.shape):
            raise ValueError(f"Candidate shape/kind mismatch: {original}")
        if candidate.module_kind == "embedding":
            _embedding_export_certificate(candidate, source, self.provenance)
        # prepare_mixed_export/save_mixed additionally validate source-weight
        # and bias identity. No regeneration, repacking, or metadata edits here.
        return candidate


def assemble_stage(context, candidate_stages, output, *, operation=None, assembly_control=None):
    """Assemble exact sealed coverage, or an explicitly requested subset control.

    ``output/artifact`` is the accounted deployable package; the returned stage
    report belongs outside it. Input .obc files are decoded one at a time and
    encoded identically by save_mixed, preserving their bytes and actual signs.
    The serializer may transiently retain adjacent candidates during iteration;
    it never materializes the complete candidate assignment.
    """
    from .pipeline_sources import load_original, load_candidate_stage, verify_context, validate_stage_output
    recipe, quant, _ = _recipe(context)
    config = copy.deepcopy(context.job["assembly"])
    _keys(config, {"coverage", "output_scope", "budget_bytes", "row_chunk_size", "cache_decoded_weights"}, "assembly")
    if config["coverage"] not in {"all", "decoder"} or config["output_scope"] not in {"language-only", "retain-vision"}:
        raise ValueError("Invalid assembly coverage/output scope")
    for key in ("budget_bytes", "row_chunk_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if type(config["cache_decoded_weights"]) is not bool:
        raise ValueError("cache_decoded_weights must be a bool")
    sealed_config = copy.deepcopy(config)
    control = (None if assembly_control is None else
               validate_assembly_control(assembly_control, sealed_assembly=sealed_config))
    selected_roles = (list(_COMPONENTS if config["coverage"] == "all" else ("decoder",))
                      if control is None else control["quantized_components"])
    retained_roles = [role for role in _COMPONENTS if role not in selected_roles]
    if control is not None:
        config.update(coverage="control", budget_bytes=control["budget_bytes"],
                      quantized_components=selected_roles,
                      retained_original_components=retained_roles,
                      evaluation_label=control["evaluation_label"])
    if not isinstance(candidate_stages, (list, tuple)) or not candidate_stages:
        raise ValueError("candidate_stages must be a nonempty list of completed directories")
    output = validate_stage_output(context, output)
    verify_context(context)
    model = _call(operation, "assemble.load_original", lambda: load_original(context, device="cpu"))
    decoder_names, _ = _target_sources(model, quant)
    prefix, _ = _text_module(model)
    all_stage_names = {"decoder": set(decoder_names), "head": {"lm_head"},
                       "embedding": {prefix + ".embed_tokens"}}
    stage_names = {role: all_stage_names[role] for role in selected_roles}
    expected = set().union(*stage_names.values())
    by_original, descriptions, receipts = {}, {}, {}
    for path in candidate_stages:
        candidates, receipt = load_candidate_stage(context, path)
        stage = receipt["stage"]
        if stage not in stage_names or stage in receipts:
            raise ValueError("Unexpected or duplicate candidate stage")
        if set(candidates) != stage_names[stage]:
            raise ValueError(f"Stage {stage} does not cover its exact original modules")
        receipts[stage] = receipt
        for name in candidates:
            if name in by_original:
                raise ValueError(f"Duplicate candidate: {name}")
            by_original[name] = candidates
            descriptions[name] = receipt["candidates"][name]
    if set(by_original) != expected or set(receipts) != set(stage_names):
        raise ValueError("Incomplete candidate coverage; endpoint BF16 fallback is not implicit")
    multimodal = prefix == "model.language_model"
    metadata = {**context.source_provenance, "pipeline_job_sha256": context.job_sha256,
                "native_source_receipt": copy.deepcopy(getattr(model, "_openbonsai_source_receipt", {})),
                "pipeline_assembly": config, "decoder_recipe_sha256": _sha(recipe),
                "candidate_stage_receipts_sha256": {stage: _sha(value) for stage, value in receipts.items()}}
    if control is not None:
        metadata.update(artifact_purpose="isolated-coverage-control", delivery_target_complete=False,
                        sealed_pipeline_assembly=sealed_config, assembly_control=control,
                        assembly_control_sha256=_sha(control),
                        quantized_component_modules={role: sorted(stage_names[role]) for role in selected_roles},
                        retained_original_component_modules={role: sorted(all_stage_names[role])
                                                             for role in retained_roles})
    if config["output_scope"] == "retain-vision":
        if not multimodal:
            raise ValueError("retain-vision requires an original conditional model with a vision tower")
        export_model, mapping = model, {name: name for name in expected}
    elif multimodal:
        extraction = extract_text_model(model,
            source_id=context.source_provenance["source_checkpoint"],
            source_revision=context.source_provenance["source_revision"])
        export_model = extraction.model
        metadata.update(extraction.metadata)
        mapping = {name: text_module_name(name) for name in expected}
    else:
        export_model, mapping = model, {name: name for name in expected}
        metadata["text_extraction"] = {"scope": "language-only", "source_already_causal": True}
    if control is not None:
        from .export import _all_tensors, _under_module
        retained_names = {role: [text_module_name(name) if multimodal and config["output_scope"] == "language-only"
                                 else name for name in sorted(all_stage_names[role])]
                          for role in retained_roles}
        accounting = {role: {"artifact_module_names": names, "tensor_count": 0,
                             "tensor_bytes": 0, "bytes_by_dtype": {}}
                      for role, names in retained_names.items()}
        for name, tensor, _kind, _persistent in _all_tensors(export_model):
            for role, names in retained_names.items():
                if _under_module(name, names):
                    row = accounting[role]
                    size = tensor.numel() * tensor.element_size()
                    dtype = str(tensor.dtype).removeprefix("torch.")
                    row["tensor_count"] += 1
                    row["tensor_bytes"] += size
                    row["bytes_by_dtype"][dtype] = row["bytes_by_dtype"].get(dtype, 0) + size
        metadata["retained_original_component_accounting"] = accounting
        metadata["vision_scope"] = "retained-original" if config["output_scope"] == "retain-vision" else "excluded"
    # Freeze native source/export view before any candidate module replacement.
    # No replacement is needed during assembly at all.
    assignment = _AssemblyCandidates(by_original, mapping, model, context.source_provenance)
    with prepare_mixed_export(export_model, sorted(mapping.values()), metadata=metadata,
                              tokenizer_dir=context.tokenizer_directory,
                              row_chunk_size=config["row_chunk_size"],
                              cache_decoded_weights=config["cache_decoded_weights"]) as plan:
        output.mkdir(parents=True)
        manifest = _call(operation, "assemble.save_mixed", lambda: save_mixed(
            plan, assignment, output / "artifact", budget_bytes=config["budget_bytes"]))
    for original, target in mapping.items():
        saved = manifest["modules"][target]
        declared = descriptions[original]
        if saved["sha256"] != declared["sha256"] or int(saved["bytes"]) != declared["bytes"]:
            raise RuntimeError(f"Assembly changed stored candidate bytes: {original}")
    verify_context(context)
    report = {"stage": "assemble", "artifact": "artifact", "coverage": config["coverage"],
            "output_scope": config["output_scope"], "module_count": len(mapping),
            "original_to_artifact_module_names": mapping,
            "source_provenance": copy.deepcopy(context.source_provenance),
            "native_source_receipt": metadata["native_source_receipt"],
            "candidate_payloads_unchanged": True, "storage": manifest["storage"],
            "artifact_manifest_sha256": hashlib.sha256((output / "artifact" / "manifest.json").read_bytes()).hexdigest(),
            "quality_claim": False,
            "runtime": "reference floating-point reconstruction; decoded caches are not artifact bytes"}
    if control is not None:
        report.update({key: copy.deepcopy(metadata[key]) for key in (
            "artifact_purpose", "delivery_target_complete", "sealed_pipeline_assembly", "pipeline_assembly",
            "assembly_control", "assembly_control_sha256", "quantized_component_modules",
            "retained_original_component_modules", "retained_original_component_accounting", "vision_scope",
            "pipeline_job_sha256", "candidate_stage_receipts_sha256")})
    return report
