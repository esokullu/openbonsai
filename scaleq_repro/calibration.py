"""Validated token-ID calibration data; no dataset or tokenizer downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _token_ids(value: Any, location: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: input_ids must be a nonempty list")
    if any(type(token) is not int or token < 0 or token > 2**63 - 1 for token in value):
        raise ValueError(f"{location}: input_ids must contain nonnegative int64 integers")
    return value


def load_calibration(
    path: str | Path,
    model_id: str,
    sequence_length: int,
    pad_token_id: int = 0,
    max_samples: int | None = None,
    expected_revision: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read JSONL into CPU long tensors of shape ``[1, sequence_length]``.

    Each record contains ``input_ids`` and optionally ``attention_mask`` and
    ``model_id`` (the alias ``model`` is also checked). Existing padding masks
    are preserved. Right truncation and right padding are explicitly counted
    in the returned JSON-serializable manifest. No reasoning tokens are removed.
    Missing provenance is allowed for user-created token files and counted.
    If ``expected_revision`` is given, every present ``resolved_revision`` or
    ``revision`` must equal that pinned revision exactly; branch aliases are
    not resolved implicitly. Observed revisions are recorded in the manifest.
    """
    _positive_integer(sequence_length, "sequence_length")
    if max_samples is not None:
        _positive_integer(max_samples, "max_samples")
    if type(pad_token_id) is not int or not 0 <= pad_token_id <= 2**63 - 1:
        raise ValueError("pad_token_id must be a nonnegative int64 integer")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    if expected_revision is not None and (
        not isinstance(expected_revision, str) or not expected_revision.strip()
    ):
        raise ValueError("expected_revision must be a nonempty string or None")

    # Kept lazy so inspecting the data helpers does not load the ML runtime.
    import torch

    source = Path(path)
    samples: list[dict[str, Any]] = []
    revisions: set[str] = set()
    manifest: dict[str, Any] = {
        "source": str(source.resolve()),
        "model_id": model_id,
        "expected_revision": expected_revision,
        "sequence_length": sequence_length,
        "pad_token_id": pad_token_id,
        "truncation_side": "right",
        "padding_side": "right",
        "max_samples": max_samples,
        "num_samples": 0,
        "source_tokens": 0,
        "retained_tokens": 0,
        "active_tokens": 0,
        "padding_tokens": 0,
        "truncated_tokens": 0,
        "truncated_records": 0,
        "records_without_model_id": 0,
        "records_without_revision": 0,
        "prepared_records_hitting_generation_limit": 0,
    }
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            location = f"{source}:{line_number}"
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"{location}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected a JSON object")
            provenance_keys = [key for key in ("model_id", "model") if key in record]
            for key in provenance_keys:
                if record[key] != model_id:
                    raise ValueError(
                        f"{location}: {key} {record[key]!r} does not match target {model_id!r}"
                    )
            revision_keys = [key for key in ("resolved_revision", "revision") if key in record]
            for key in revision_keys:
                revision = record[key]
                if not isinstance(revision, str) or not revision.strip():
                    raise ValueError(f"{location}: {key} must be a nonempty revision string")
                if expected_revision is not None and revision != expected_revision:
                    raise ValueError(
                        f"{location}: {key} {revision!r} does not match expected revision {expected_revision!r}"
                    )
                revisions.add(revision)
            ids = _token_ids(record.get("input_ids"), location)
            mask = record.get("attention_mask", [1] * len(ids))
            if (
                not isinstance(mask, list)
                or len(mask) != len(ids)
                or any(type(value) is not int or value not in (0, 1) for value in mask)
            ):
                raise ValueError(f"{location}: attention_mask must be a matching list of 0/1 integers")
            kept = min(len(ids), sequence_length)
            removed = len(ids) - kept
            missing = sequence_length - kept
            kept_mask = mask[:kept]
            if not any(kept_mask):
                raise ValueError(f"{location}: no active tokens remain after truncation")
            samples.append({
                "input_ids": torch.tensor([ids[:kept] + [pad_token_id] * missing], dtype=torch.long),
                "attention_mask": torch.tensor([kept_mask + [0] * missing], dtype=torch.long),
            })
            manifest["source_tokens"] += len(ids)
            manifest["retained_tokens"] += kept
            manifest["active_tokens"] += sum(kept_mask)
            manifest["padding_tokens"] += missing
            manifest["truncated_tokens"] += removed
            manifest["truncated_records"] += int(removed > 0)
            manifest["records_without_model_id"] += int(not provenance_keys)
            manifest["records_without_revision"] += int(not revision_keys)
            manifest["prepared_records_hitting_generation_limit"] += int(
                record.get("hit_generation_limit") is True
            )
    if not samples:
        raise ValueError(f"{source}: no calibration records found")
    manifest["num_samples"] = len(samples)
    manifest["revisions"] = sorted(revisions)
    # With max_samples, this identifies exactly the consumed input prefix.
    manifest["consumed_jsonl_sha256"] = digest.hexdigest()
    return samples, manifest
