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


def encode_chat(tokenizer, msgs: list[dict], chat_kwargs: dict) -> list[int]:
    """Token ids of ``msgs`` rendered by the model's own chat template with the generation prompt
    (and the fixed date of ``adapters.base.template_vars``)."""
    from ..adapters.base import template_vars
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False,
                                        **template_vars(chat_kwargs))
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [int(t) for t in (ids[0] if ids and isinstance(ids[0], list) else ids)]
    if not ids:
        raise ValueError("chat template produced no tokens")
    return ids


# What generate_encoded does, for result files and cache keys (bump when the decoding changes).
DECODING = {"strategy": "greedy", "num_beams": 1, "kv_cache": True,
            "model_generation_config": "ignored: transformers' library defaults (no repetition penalty, "
                                       "n-gram blocking, suppressed or bad tokens, minimum length)"}


def greedy_config(*, max_new_tokens: int, eos_ids, pad_id: int):
    """The complete GenerationConfig of a greedy generation."""
    from transformers import GenerationConfig
    return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=int(max_new_tokens),
                            eos_token_id=[int(e) for e in eos_ids], pad_token_id=int(pad_id), use_cache=True)


@torch.no_grad()
def generate_encoded(model, encoded: list[list[int]], *, eos_ids: list[int], max_new_tokens: int, device,
                     pad_id: int) -> list[tuple[list[int], bool]]:
    """Greedy, left-padded batch generation with the KV cache: plain argmax decoding.

    ``generate`` fills every option a call leaves unset from ``model.generation_config``, i.e. a
    checkpoint's ``generation_config.json`` (``repetition_penalty``, ``no_repeat_ngram_size``,
    ``bad_words_ids``, ``suppress_tokens``, ``min_length`` ...), which a model built from a GGUF
    does not have. So the call runs with the library defaults in its place: the same decoding for
    every model, however it was built.

    Returns per prompt (generated ids through the first EOS, hit ``max_new_tokens``)."""
    from transformers import GenerationConfig
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    width = max(len(ids) for ids in encoded)
    input_ids = torch.tensor([[pad_id] * (width - len(ids)) + ids for ids in encoded], device=device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded], device=device)
    config = greedy_config(max_new_tokens=max_new_tokens, eos_ids=eos_ids, pad_id=pad_id)
    saved = model.generation_config
    model.generation_config = GenerationConfig()
    try:
        out = model.generate(input_ids=input_ids, attention_mask=mask, generation_config=config)
    finally:
        model.generation_config = saved
    eos = set(eos_ids)
    results = []
    for row in range(len(encoded)):
        gen = out[row, width:].tolist()
        stop = next((k for k, t in enumerate(gen) if t in eos), None)
        gen = gen[:stop + 1] if stop is not None else gen
        if not gen:
            raise ValueError("model generated no tokens")
        results.append((gen, stop is None))
    return results


def checked_chat_kwargs(adapter, thinking: bool, where: str) -> dict:
    """``adapter.chat_kwargs``; a thinking mode the chat template has no switch for is a ConfigError."""
    from ..config import ConfigError
    try:
        return adapter.chat_kwargs(thinking=bool(thinking))
    except ValueError as error:
        raise ConfigError(f"{where}: {error}") from None


def pad_token(tokenizer, eos_ids) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else int(eos_ids[0])


def generate_batch(model, tokenizer, prompts: list[list[dict]], *, chat_kwargs: dict, eos_ids: list[int],
                   max_new_tokens: int, device, pad_id: int | None = None) -> list[dict[str, Any]]:
    """Greedy, left-padded batch generation; returns one record per prompt."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    encoded = [encode_chat(tokenizer, msgs, chat_kwargs) for msgs in prompts]
    if pad_id is None:
        pad_id = pad_token(tokenizer, eos_ids)
    generated = generate_encoded(model, encoded, eos_ids=eos_ids, max_new_tokens=max_new_tokens, device=device,
                                 pad_id=pad_id)
    records = []
    for msgs, ids, (gen, hit) in zip(prompts, encoded, generated):
        records.append({"input_ids": ids + gen, "attention_mask": [1] * (len(ids) + len(gen)),
                        "prompt_token_count": len(ids), "generated_token_count": len(gen),
                        "hit_generation_limit": hit,
                        "prompt": next((m.get("content") for m in msgs if m.get("role") == "user"), None)})
    return records


def load_prompts(ctx) -> list[list[dict]]:
    """The trace prompts; refuses any that contain a reasoning suite's test problem (eval.heldout)."""
    from ..eval import heldout
    spec = ctx.cfg["traces"]["prompts"] or {}
    kind = spec.get("kind", "jsonl")
    n = spec.get("n")
    if kind == "jsonl":
        if not spec.get("path"):
            raise ValueError("traces.prompts.path is required for kind jsonl")
        heldout.check_source(spec, "traces.prompts", path=spec["path"])
        prompts = read_prompts(spec["path"], n)
    elif kind == "ultrachat":
        from .calibrate import resolve_file
        heldout.check_source({"kind": "ultrachat", **spec}, "traces.prompts")
        path = resolve_file({"kind": "ultrachat", **spec},
                            local_files_only=bool(ctx.cfg["model"].get("local_files_only")))
        prompts = ultrachat_prompts(path, rows=tuple(spec.get("rows", (0, 6000))), n=int(n or 128),
                                    seed=int(spec.get("seed", 5090)))
    else:
        raise ValueError(f"unknown traces.prompts.kind {kind!r}")
    heldout.check_texts(["\n".join(str(m.get("content", "")) for m in msgs) for msgs in prompts], "traces.prompts")
    return prompts


def run(ctx) -> dict:
    c = ctx.cfg["traces"]
    log = ctx.log("traces")
    out = ctx.run.stage_dir("traces")
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    prompts = load_prompts(ctx)
    batch = int(c.get("batch") or 8)
    adapter = ctx.adapter()
    chat_kwargs = checked_chat_kwargs(adapter, bool(c.get("thinking")), "traces.thinking")   # before loading
    model = tokenizer = None
    for k, b0 in enumerate(range(0, len(prompts), batch)):
        part = parts / f"{k:05d}.jsonl"
        if part.exists():
            continue
        if model is None:
            model, tokenizer = ctx.load_model()
            torch.manual_seed(0)
        recs = generate_batch(model, tokenizer, prompts[b0:b0 + batch], chat_kwargs=chat_kwargs,
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
    needed = trace_tokens_needed(ctx.cfg)
    if summary["tokens"] < needed:
        log(f"WARNING: calibrate needs about {needed} trace tokens but the traces hold {summary['tokens']}; "
            "raise traces.prompts.n (or max_new_tokens) before calibrating")
    return summary


def trace_tokens_needed(cfg: dict) -> int:
    """Tokens the calibrate sources read from traces.jsonl (block count x seq_len)."""
    c = cfg["calibrate"]
    return sum(int(s.get("n") or 0) * int(c["seq_len"]) for s in c["sources"]
               if s.get("kind") == "traces" and not s.get("path"))
