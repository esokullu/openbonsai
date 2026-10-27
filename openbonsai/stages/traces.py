"""AYOT traces: the original model answers prompts with its own chat template.

The traces become calibration and distillation data, so the quantized model
is fitted on the distribution it actually produces. Prompts come from a JSONL
file (``prompt``, ``question`` or ``messages`` per line) or from the first user
turn of UltraChat rows below the calibration rows. Generation is greedy and
batched with left padding; every batch is written atomically as one part
file, so a rerun resumes after the last finished batch.

Output: ``traces/traces.jsonl`` with ``input_ids`` (prompt + generated tokens,
EOS included), ``attention_mask`` and counts per record.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from ..formats.atomic import atomic_write_bytes, atomic_write_json


def read_prompts(path, max_samples: int | None = None) -> list[list[dict]]:
    """JSONL prompts as chat message lists."""
    prompts: list[list[dict]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if max_samples is not None and len(prompts) >= max_samples:
            break
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{location}: expected a JSON object")
        if "messages" in record:
            msgs = record["messages"]
            if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
                raise ValueError(f"{location}: messages must be a nonempty list of objects")
            prompts.append(msgs)
            continue
        if "prompt" in record and "question" in record and record["prompt"] != record["question"]:
            raise ValueError(f"{location}: conflicting prompt and question fields")
        prompt = record.get("prompt", record.get("question"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{location}: prompt or question must be a nonempty string")
        prompts.append([{"role": "user", "content": prompt}])
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def ultrachat_prompts(path, *, rows=(0, 6000), n=128, seed=5090) -> list[list[dict]]:
    """First user message of UltraChat rows [rows[0], rows[1]), deduplicated, seeded sample."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["messages"]).column("messages").to_pylist()[rows[0]:rows[1]]
    seen, pool = set(), []
    for msgs in table:
        first = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        key = " ".join(str(first).split()) if first else None
        if key and key not in seen:
            seen.add(key)
            pool.append([{"role": "user", "content": first}])
    random.Random(seed).shuffle(pool)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} distinct UltraChat prompts in rows {rows}")
    return pool[:n]


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[list[dict]], *, chat_kwargs: dict, eos_ids: list[int],
                   max_new_tokens: int, device, pad_id: int | None = None) -> list[dict[str, Any]]:
    """Greedy, left-padded batch generation; returns one record per prompt."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    encoded = []
    for msgs in prompts:
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
                                            **chat_kwargs)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = [int(t) for t in (ids[0] if ids and isinstance(ids[0], list) else ids)]
        if not ids:
            raise ValueError("chat template produced no tokens")
        encoded.append(ids)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    width = max(len(ids) for ids in encoded)
    input_ids = torch.tensor([[pad_id] * (width - len(ids)) + ids for ids in encoded], device=device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded], device=device)
    out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                         do_sample=False, num_beams=1, eos_token_id=list(eos_ids), pad_token_id=pad_id,
                         use_cache=True)
    eos = set(eos_ids)
    records = []
    for row, (msgs, ids) in enumerate(zip(prompts, encoded)):
        gen = out[row, width:].tolist()
        stop = next((k for k, t in enumerate(gen) if t in eos), None)
        gen = gen[:stop + 1] if stop is not None else gen
        if not gen:
            raise ValueError("model generated no tokens")
        records.append({"input_ids": ids + gen, "attention_mask": [1] * (len(ids) + len(gen)),
                        "prompt_token_count": len(ids), "generated_token_count": len(gen),
                        "hit_generation_limit": stop is None,
                        "prompt": next((m.get("content") for m in msgs if m.get("role") == "user"), None)})
    return records


def load_prompts(ctx) -> list[list[dict]]:
    spec = ctx.cfg["traces"]["prompts"] or {}
    kind = spec.get("kind", "jsonl")
    n = spec.get("n")
    if kind == "jsonl":
        if not spec.get("path"):
            raise ValueError("traces.prompts.path is required for kind jsonl")
        return read_prompts(spec["path"], n)
    if kind == "ultrachat":
        from .calibrate import resolve_file
        path = resolve_file({"kind": "ultrachat", **spec},
                            local_files_only=bool(ctx.cfg["model"].get("local_files_only")))
        return ultrachat_prompts(path, rows=tuple(spec.get("rows", (0, 6000))), n=int(n or 128),
                                 seed=int(spec.get("seed", 5090)))
    raise ValueError(f"unknown traces.prompts.kind {kind!r}")


def run(ctx) -> dict:
    c = ctx.cfg["traces"]
    log = ctx.log("traces")
    out = ctx.run.stage_dir("traces")
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    prompts = load_prompts(ctx)
    batch = int(c.get("batch") or 8)
    adapter = ctx.adapter()
    model = tokenizer = None
    for k, b0 in enumerate(range(0, len(prompts), batch)):
        part = parts / f"{k:05d}.jsonl"
        if part.exists():
            continue
        if model is None:
            model, tokenizer = ctx.load_model()
            torch.manual_seed(0)
        recs = generate_batch(model, tokenizer, prompts[b0:b0 + batch],
                              chat_kwargs=adapter.chat_kwargs(thinking=bool(c.get("thinking"))),
                              eos_ids=adapter.eos_ids(tokenizer, model), device=ctx.device,
                              max_new_tokens=int(c["max_new_tokens"]))
        atomic_write_bytes(part, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8"))
        log(f"batch {k}: {sum(r['generated_token_count'] for r in recs)} generated tokens")
    files = sorted(parts.glob("*.jsonl"))
    text = "".join(f.read_text(encoding="utf-8") for f in files)
    atomic_write_bytes(out / "traces.jsonl", text.encode("utf-8"))
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    summary = {"records": len(rows), "tokens": sum(len(r["input_ids"]) for r in rows),
               "generated_tokens": sum(r["generated_token_count"] for r in rows),
               "hit_generation_limit": sum(bool(r["hit_generation_limit"]) for r in rows),
               "thinking": bool(c.get("thinking")), "max_new_tokens": int(c["max_new_tokens"])}
    atomic_write_json(out / "summary.json", summary)
    log(f"{summary['records']} traces, {summary['tokens']} tokens")
    return summary
