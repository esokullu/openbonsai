"""Full-vocabulary, bounded table reconstruction for ternary embeddings.

One optimizer update is one complete vocabulary sweep, irrespective of the
row chunk size. This fits known BF16 weights; its reconstruction error is a
training metric, not held-out language quality. No evaluation text is accepted.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from .export import _json_bytes

CERTIFICATE_FORMAT = "openbonsai.complete-embedding-calibration.v1"
_CERTIFICATE_ATTRIBUTE = "_embedding_calibration_certificate"
_HASH_CHUNK = 1048576


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _provenance(value):
    if not isinstance(value, Mapping):
        raise ValueError("source_provenance must identify the original embedding checkpoint")
    value = copy.deepcopy(dict(value))
    _json_bytes(value)
    if not isinstance(value.get("source_checkpoint"), str) or not value["source_checkpoint"]:
        raise ValueError("source_checkpoint is required")
    revision = value.get("source_revision", value.get("resolved_revision"))
    if ("source_revision" in value and "resolved_revision" in value
            and value["source_revision"] != value["resolved_revision"]):
        raise ValueError("Conflicting source revisions")
    content = value.get("source_content_sha256")
    if content is not None and not _digest_string(content):
        raise ValueError("source_content_sha256 must be an original checkpoint SHA256")
    if revision is not None:
        if not isinstance(revision, str) or not revision or revision in {"main", "master", "HEAD", "latest"}:
            raise ValueError("source_revision must identify an immutable checkpoint")
        value["source_revision"] = revision
        value.pop("resolved_revision", None)
    if revision is None and content is None:
        raise ValueError("Original source_revision or source_content_sha256 is required")
    return value


def _tensor_chunks(value, chunk_elements=_HASH_CHUNK):
    """Slice before copying/casting, including noncontiguous compact state."""
    _positive_int(chunk_elements, "chunk_elements")
    value = value.detach() if value.ndim else value.detach().reshape(1)
    rows = max(1, chunk_elements // max(1, math.prod(value.shape[1:])))
    for start in range(0, value.shape[0], rows):
        yield value[start:start + rows]


def _hash_tensor(digest, name, value, chunk_elements):
    digest.update(_json_bytes({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}))
    for chunk in _tensor_chunks(value, chunk_elements):
        digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())


def _configuration(embedding):
    quantizer = embedding.quantizer
    return {"module_kind": "embedding", "num_embeddings": embedding.num_embeddings,
            "embedding_dim": embedding.embedding_dim, "padding_idx": embedding.padding_idx,
            "source_dtype": str(embedding.weight.dtype),
            "group_size": quantizer.group_size, "factor_mapping": quantizer.factor_mapping,
            "lora_rank": quantizer.lora_rank, "hard_phase": quantizer.hard_phase,
            "soft_fraction": quantizer.soft_fraction, "max_sharpness": quantizer.max_sharpness,
            "row_chunk_size": embedding.row_chunk_size, "block_size": embedding.block_size,
            "rotated": embedding.rotation_signs is not None,
            "rotation_convention": "row-DH; original-coordinate inverse HD",
            "quant_enabled": embedding.quant_enabled, "progress": embedding.progress,
            "rotation_only": getattr(embedding, "rotation_only", False)}


def calibration_state_sha256(embedding, *, chunk_elements=_HASH_CHUNK, include_snapshots=False):
    """Bind current inference math without copying the original vocabulary table.

    Original bytes are checked separately by assert_source_unchanged. The default
    binds factors, A/B, initialization statistics, signs and configuration. Frozen
    Jacobian snapshots affect training only; callback guards include those too.
    """
    _positive_int(chunk_elements, "chunk_elements")
    digest = hashlib.sha256(_json_bytes(_configuration(embedding)))
    values = dict(embedding.named_parameters())
    values.update(embedding.named_buffers())
    for name, value in sorted(values.items()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"weight", "bias", "_device_probe"}:
            continue
        if leaf.startswith("_hard_snapshot_") and not include_snapshots:
            continue
        _hash_tensor(digest, name, value, chunk_elements)
    if include_snapshots:
        quantizer = embedding.quantizer
        digest.update(_json_bytes({key: getattr(quantizer, key, None) for key in (
            "_hard_snapshot_sharpness", "_snapshot_step", "_step_number")}))
    return digest.hexdigest()


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _check_operator(embedding):
    from .embedding_quantizer import BoundedTernaryEmbedding
    if not isinstance(embedding, BoundedTernaryEmbedding):
        raise TypeError("Expected BoundedTernaryEmbedding")
    if not embedding.quant_enabled or getattr(embedding, "rotation_only", False):
        raise ValueError("Embedding calibration requires quantization enabled and rotation_only=False")
    if embedding.weight.dtype != torch.bfloat16 or embedding.weight.requires_grad:
        raise ValueError("Original embedding must be frozen BF16")
    parameters = tuple(embedding.trainable_parameters())
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Embedding trainables must be FP32")
    device = embedding.weight.device
    if device.type == "meta" or any(p.device != device for p in parameters):
        raise ValueError("Embedding source and trainables must share one materialized device")
    embedding.validate_parameters()
    return parameters, device


@dataclass
class EmbeddingTrainingConfig:
    steps: int = 8
    factor_lr: float = .005
    lora_lr: float = .001
    weight_decay: float = 0.
    grad_clip: float = 1.
    loss: str = "mse"
    huber_delta: float = 1.

    def validate(self):
        _positive_int(self.steps, "steps")
        for name in ("factor_lr", "lora_lr", "weight_decay", "grad_clip", "huber_delta"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (name != "weight_decay" and value == 0)):
                raise ValueError("Learning rates, clipping and Huber delta must be positive; weight decay nonnegative")
        if self.loss not in {"mse", "huber"}:
            raise ValueError("loss must be 'mse' or 'huber'")


@dataclass
class EmbeddingFitResult:
    report: dict


def _row_spans(embedding):
    for start in range(0, embedding.num_embeddings, embedding.row_chunk_size):
        yield start, min(start + embedding.row_chunk_size, embedding.num_embeddings)


def _rows(embedding, start, end, device):
    ids = torch.arange(start, end, dtype=torch.long, device=device)
    student = embedding.reconstruct_rows(ids)
    with torch.no_grad():
        teacher = embedding.teacher_rows(ids)
    expected = (end - start, embedding.embedding_dim)
    if any(value.shape != expected or value.dtype != torch.bfloat16 or value.device != device
           for value in (student, teacher)):
        raise RuntimeError("Embedding row operator must return bounded original-coordinate BF16 rows")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise FloatingPointError("Nonfinite embedding reconstruction or target")
    return student, teacher.detach()


def _loss_sum(student, teacher, loss, delta):
    student, teacher = student.float(), teacher.float()
    return ((student - teacher).square().sum() if loss == "mse" else
            F.huber_loss(student, teacher, reduction="sum", delta=delta))


@torch.no_grad()
def evaluate_embedding_table(embedding, *, loss="mse", huber_delta=1.):
    """All-row current-hard reconstruction only; no downstream quality claim."""
    EmbeddingTrainingConfig(loss=loss, huber_delta=huber_delta).validate()
    _, device = _check_operator(embedding)
    original_progress = embedding.progress
    total = squares = source_squares = max_row_mse = max_error = 0.
    padding_mse = None
    count = 0
    try:
        # No begin_step: evaluation must not capture/overwrite a soft snapshot.
        embedding.progress = 1.
        for start, end in _row_spans(embedding):
            student, teacher = _rows(embedding, start, end, device)
            difference = student.float() - teacher.float()
            row_squares = difference.square().sum(dim=1)
            squares += float(row_squares.sum())
            source_squares += float(teacher.float().square().sum())
            total += float(_loss_sum(student, teacher, loss, huber_delta))
            max_row_mse = max(max_row_mse, float(row_squares.max()) / embedding.embedding_dim)
            max_error = max(max_error, float(difference.abs().max()))
            if embedding.padding_idx is not None and start <= embedding.padding_idx < end:
                padding_mse = float(row_squares[embedding.padding_idx - start]) / embedding.embedding_dim
            count += end - start
    finally:
        embedding.progress = original_progress
    if count != embedding.num_embeddings:
        raise RuntimeError("Hard evaluation did not cover every vocabulary row")
    denominator = embedding.num_embeddings * embedding.embedding_dim
    if not all(math.isfinite(x) for x in (total, squares, source_squares, max_row_mse, max_error)):
        raise FloatingPointError("Nonfinite full-table reconstruction metric")
    return {"role": "training-reconstruction", "rows": count, "elements": denominator,
            "loss": total / denominator, "loss_name": loss, "mse": squares / denominator,
            "rmse": math.sqrt(squares / denominator), "source_rms": math.sqrt(source_squares / denominator),
            "relative_rmse": math.sqrt(squares / source_squares) if source_squares else None,
            "max_row_mse": max_row_mse, "max_absolute_error": max_error,
            "padding_idx": embedding.padding_idx, "padding_row_mse": padding_mse,
            "precision": "BF16 original-coordinate output and target; FP32 elementwise loss/reduction",
            "quality_claim": False}


def _check_gradients(parameters):
    for parameter in parameters:
        if parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise RuntimeError("Every compact trainable must receive an FP32 gradient")
        for chunk in _tensor_chunks(parameter.grad):
            if not bool(torch.isfinite(chunk).all()):
                raise FloatingPointError("Nonfinite embedding calibration gradient")


def _check_optimizer(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and value.is_floating_point():
                if value.dtype != torch.float32:
                    raise RuntimeError("Embedding optimizer state must be FP32")
                for chunk in _tensor_chunks(value):
                    if not bool(torch.isfinite(chunk).all()):
                        raise FloatingPointError("Nonfinite embedding optimizer state")


def _coverage(soft, hard, last_soft_step, hard_step):
    digest = hashlib.sha256()
    _hash_tensor(digest, "soft_visits", soft, _HASH_CHUNK)
    _hash_tensor(digest, "hard_visits", hard, _HASH_CHUNK)
    return {"vocabulary_rows": soft.numel(), "soft_rows": int((soft > 0).sum()),
            "hard_rows": int((hard > 0).sum()), "soft_visits_min": int(soft.min()),
            "soft_visits_max": int(soft.max()), "hard_visits_min": int(hard.min()),
            "hard_visits_max": int(hard.max()), "last_soft_step": last_soft_step,
            "final_hard_step": hard_step, "coverage_sha256": digest.hexdigest(),
            "last_soft_rows": soft.numel(), "final_hard_rows": hard.numel()}


def _factor_summary(embedding):
    """Summarize actual mapping coefficients with bounded compact-state scratch."""
    result = {}
    quantizer = embedding.quantizer
    for name in ("mu", "scale", "threshold"):
        raw = getattr(quantizer, "raw_" + name)
        total = squares = 0.
        minimum, maximum = math.inf, -math.inf
        for part in _tensor_chunks(raw):
            if quantizer.factor_mapping == "sigmoid":
                mapped = 2 * part.sigmoid()
                mapped = mapped - 1 if name == "mu" else (.5 * mapped if name == "threshold" else mapped)
            else:
                mapped = part.tanh() if name == "mu" else part.exp()
                if name == "threshold":
                    mapped = .5 * mapped
            if not bool(torch.isfinite(mapped).all()):
                raise FloatingPointError("Nonfinite mapped embedding factors")
            total += float(mapped.sum())
            squares += float(mapped.square().sum())
            minimum, maximum = min(minimum, float(mapped.min())), max(maximum, float(mapped.max()))
        result[name] = {"mean": total / raw.numel(), "rms": math.sqrt(squares / raw.numel()),
                        "min": minimum, "max": maximum}
    result["units"] = "mu offset / adapted alpha0; scale / adapted alpha0; threshold in normalized weight coordinates"
    return result


def require_complete_calibration(embedding):
    """Return an independent compact certificate or reject incomplete/stale state.

    This is an integrity/coverage gate, not a security boundary against callers
    deliberately forging Python state. Exporters must call it before and after
    packing and include the certificate in candidate provenance before costing.
    """
    _check_operator(embedding)
    certificate = getattr(embedding, _CERTIFICATE_ATTRIBUTE, None)
    if not isinstance(certificate, dict) or certificate.get("format") != CERTIFICATE_FORMAT:
        raise RuntimeError("Complete embedding calibration certificate is missing")
    body = {k: v for k, v in certificate.items() if k != "certificate_sha256"}
    if hashlib.sha256(_json_bytes(body)).hexdigest() != certificate.get("certificate_sha256"):
        raise RuntimeError("Embedding calibration certificate was modified")
    coverage = certificate["coverage"]
    rows = embedding.num_embeddings
    if (embedding.progress != 1. or certificate["source_shape"] != [rows, embedding.embedding_dim]
            or certificate["completed_sweeps"] != certificate["soft_sweeps"] + certificate["hard_sweeps"]
            or min(certificate["soft_sweeps"], certificate["hard_sweeps"]) < 1
            or any(coverage[key] != rows for key in ("vocabulary_rows", "soft_rows", "hard_rows", "last_soft_rows", "final_hard_rows"))
            or any(coverage[f"{phase}_visits_{extreme}"] != certificate[f"{phase}_sweeps"]
                   for phase in ("soft", "hard") for extreme in ("min", "max"))):
        raise RuntimeError("Embedding calibration coverage is incomplete")
    embedding.assert_source_unchanged(verify_hash=True)
    if (embedding.source_weight_sha256 != certificate["source_weight_sha256"]
            or calibration_state_sha256(embedding) != certificate["state_sha256"]):
        raise RuntimeError("Embedding calibration certificate is stale")
    return copy.deepcopy(certificate)


def fit_embedding_table(embedding, config: EmbeddingTrainingConfig, *, source_provenance,
                        progress_callback=None) -> EmbeddingFitResult:
    """Fit every original row once per AdamW update, then certify complete state.

    Padding rows participate in this weight-reconstruction objective even though
    native token lookup suppresses padding gradients. Constant per-group learning
    rates and one global clip/step are independent of row chunking. A failed or
    interrupted fit invalidates any prior certificate and leaves changed trainable
    state uncertified; the original BF16 source must remain unchanged.
    """
    setattr(embedding, _CERTIFICATE_ATTRIBUTE, None)
    # Callbacks may close over the caller's object. Their edits must not change
    # a running objective/schedule while the report retains the old settings.
    config = copy.deepcopy(config)
    config.validate()
    parameters, device = _check_operator(embedding)
    provenance = _provenance(source_provenance)
    if (provenance.get("source_weight_sha256", embedding.source_weight_sha256)
            != embedding.source_weight_sha256):
        raise ValueError("Declared embedding source weight hash does not match")
    embedding.assert_source_unchanged(verify_hash=True)
    soft_steps = sum(step / config.steps <= embedding.soft_fraction for step in range(1, config.steps + 1))
    if not 0 < soft_steps < config.steps:
        raise ValueError("Embedding calibration needs at least one soft and one hard update")
    groups = [{"params": list(embedding.factor_parameters()), "lr": config.factor_lr}]
    if embedding.lora_parameters():
        groups.append({"params": list(embedding.lora_parameters()), "lr": config.lora_lr})
    if {id(p) for group in groups for p in group["params"]} != {id(p) for p in parameters}:
        raise RuntimeError("Optimizer groups must cover exactly the embedding trainables")
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay, foreach=False)
    flags = [(p, p.requires_grad) for p in parameters]
    training_flags = [(module, module.training) for module in embedding.modules()]
    soft_visits = torch.zeros(embedding.num_embeddings, dtype=torch.int32, device="cpu")
    hard_visits = torch.zeros_like(soft_visits)
    denominator = embedding.num_embeddings * embedding.embedding_dim
    started = _clock(device)
    report = {"stage": "embedding-only-complete-table", "role": "calibration", "config": asdict(config),
              "source_provenance": provenance, "source_weight_sha256": embedding.source_weight_sha256,
              "source_identity_scope": "checkpoint identity is caller-declared; original BF16 embedding bytes are verified",
              "objective": {"name": config.loss, "coordinates": "original embedding output",
                            "denominator": denominator, "reduction": "sum of all row elements / vocabulary*width",
                            "logical_update": "one disjoint complete vocabulary sweep; one clip and one AdamW step",
                            "learning_rate_schedule": "constant", "padding_row_is_fitted": True},
              "scope": "table reconstruction of original BF16 weights; no text, answers, decoder or head fitted",
              "quality_claim": False, "torch": torch.__version__, "device": str(device),
              "configuration": _configuration(embedding), "steps": [],
              "initial_state_sha256": calibration_state_sha256(embedding),
              "effective_row_chunk": embedding.row_chunk_size,
              "precision": {"source": "bfloat16", "student_output": "bfloat16", "loss": "float32",
                            "parameters": "float32", "gradients": "float32", "optimizer_state": "float32"}}
    try:
        embedding.eval()
        for parameter in parameters:
            parameter.requires_grad_(True)
        embedding.clear_soft_snapshot()
        report["initial_hard_reconstruction"] = evaluate_embedding_table(
            embedding, loss=config.loss, huber_delta=config.huber_delta)
        initialized = _clock(device)
        last_soft_step = 0
        for step in range(1, config.steps + 1):
            step_started = _clock(device)
            progress = step / config.steps
            phase = "soft" if progress <= embedding.soft_fraction else "hard"
            optimizer.zero_grad(set_to_none=True)
            embedding.begin_step(progress)
            visited = torch.zeros_like(soft_visits)
            total_loss = 0.
            for start, end in _row_spans(embedding):
                if start < 0 or end > embedding.num_embeddings or start >= end or bool(visited[start:end].any()):
                    raise RuntimeError("Vocabulary sweep contains invalid or duplicate rows")
                student, teacher = _rows(embedding, start, end, device)
                loss = _loss_sum(student, teacher, config.loss, config.huber_delta) / denominator
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite embedding calibration loss")
                loss.backward()
                total_loss += float(loss.detach())
                visited[start:end] += 1
                del student, teacher, loss
            if not bool((visited == 1).all()):
                raise RuntimeError("Vocabulary sweep did not cover every row exactly once")
            _check_gradients(parameters)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            embedding.validate_parameters()
            _check_optimizer(optimizer)
            if phase == "soft":
                soft_visits += visited
                last_soft_step = step
            else:
                hard_visits += visited
            event = {"step": step, "total_steps": config.steps, "phase": phase, "progress": progress,
                     "rows": int(visited.sum()), "elements": denominator, "loss": total_loss,
                     "grad_norm_before_clip": float(norm), "factor_lr": config.factor_lr, "lora_lr": config.lora_lr,
                     "seconds": _clock(device) - step_started}
            report["steps"].append(event)
            if progress_callback is not None:
                before = calibration_state_sha256(embedding, include_snapshots=True)
                progress_callback(copy.deepcopy(event))
                if calibration_state_sha256(embedding, include_snapshots=True) != before:
                    raise RuntimeError("Progress callback changed embedding calibration state")
                # Version counters do not catch .data writes. A callback must
                # not alter targets between sweeps and restore them at the end.
                embedding.assert_source_unchanged(verify_hash=True)
        optimized = _clock(device)
        embedding.assert_source_unchanged(verify_hash=True)
        embedding.clear_soft_snapshot()
        report["final_hard_reconstruction"] = evaluate_embedding_table(
            embedding, loss=config.loss, huber_delta=config.huber_delta)
        report["mapped_factor_summary"] = _factor_summary(embedding)
        report["coverage"] = _coverage(soft_visits, hard_visits, last_soft_step, config.steps)
        final_hash = calibration_state_sha256(embedding)
        report["final_state_sha256"] = final_hash
        certificate = {"format": CERTIFICATE_FORMAT, "stage": report["stage"],
                       "source_provenance": copy.deepcopy(provenance),
                       "source_weight_sha256": embedding.source_weight_sha256,
                       "source_shape": [embedding.num_embeddings, embedding.embedding_dim],
                       "completed_sweeps": config.steps, "soft_sweeps": soft_steps,
                       "hard_sweeps": config.steps - soft_steps, "coverage": copy.deepcopy(report["coverage"]),
                       "state_sha256": final_hash, "configuration": _configuration(embedding),
                       "training_config": asdict(config),
                       "objective": copy.deepcopy(report["objective"]), "quality_claim": False}
        certificate["certificate_sha256"] = hashlib.sha256(_json_bytes(certificate)).hexdigest()
        setattr(embedding, _CERTIFICATE_ATTRIBUTE, copy.deepcopy(certificate))
        report["certificate"] = require_complete_calibration(embedding)
        finished = _clock(device)
        report["timing_seconds"] = {"initial_audit_and_evaluation": initialized - started,
                                   "optimization_and_callbacks": optimized - initialized,
                                   "final_evaluation_and_certification": finished - optimized,
                                   "total": finished - started}
        report["timing_scope"] = "synchronized fitting/evaluation/audit; excludes construction and initial source verification"
        report["step_timing_scope"] = "full sweep, backward, clip, optimizer and finite-state audit; excludes callback"
        return EmbeddingFitResult(report)
    except BaseException:
        setattr(embedding, _CERTIFICATE_ATTRIBUTE, None)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, flag in training_flags:
            module.training = flag
