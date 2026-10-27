"""Score a run's model (BF16, or after quantize/endpoints/distill) on the suites.

Sources (``--from``):

- ``bf16``: the original model;
- ``quantize``, ``endpoints``: dequantized weights (C * s) R^T written into the
  model, as the decoder-only and endpoint results were scored;
- ``distill``: TernaryLinear/TernaryEmbedding modules with the distilled fp16
  scales (the runtime arithmetic, as e2e_kd.py scored it);
- a path: a state ``.safetensors`` or a legacy research ``.pt`` (decoder
  ``ternary_state.pt`` and/or ``endpoint_state.pt``; comma-separated). Legacy
  per-module signs are applied exactly as ``apply_state`` did.

Results go to ``eval/<source>.<suite>.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..eval import suites as S
from ..formats.atomic import atomic_write_json
from ..formats.state import load_legacy_pt, load_state
from . import calibrate
from .common import apply_dense, install_ternary, legacy_records

STAGE_SOURCES = ("bf16", "quantize", "endpoints", "distill")


def resolve_source(ctx, source: str | None) -> str:
    if source in (None, "auto"):
        for stage in ("distill", "endpoints", "quantize"):
            if ctx.run.done_info(stage) is not None:
                return stage
        return "bf16"
    return source


def install(ctx, model, source: str, mode: str = "auto") -> str:
    """Put the source's weights into ``model``; returns a label for file names."""
    adapter = ctx.adapter()
    if source == "bf16":
        return "bf16"
    if source in STAGE_SOURCES:
        state = ctx.state_through(source)
        ternary = mode == "ternary" or (mode == "auto" and source == "distill")
        if ternary:
            install_ternary(adapter, model, state.records(), ctx.device)
        else:
            apply_dense(adapter, model, state.records())
        return source
    for part in source.split(","):
        path = Path(part)
        if path.suffix == ".safetensors":
            records = load_state(path).records()
        else:
            records = legacy_records(load_legacy_pt(path))
        if mode == "ternary":
            install_ternary(adapter, model, records, ctx.device)
        else:
            apply_dense(adapter, model, records)
    return "+".join(Path(p).stem for p in source.split(","))


def compare_blocks(ctx, tok) -> torch.Tensor:
    c = ctx.cfg["eval"]["compare"]
    adapter = ctx.adapter()
    sources = c.get("sources") or [{"kind": "wikitext2", "n": int(c["n"]),
                                    "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet"}]
    blocks, _ = calibrate.build(tok, sources, seq_len=int(c["seq_len"]), seed=int(c["seed"]),
                                chat_kwargs=adapter.chat_kwargs(thinking=False), bos=adapter.bos_id(tok),
                                traces_path=ctx.run.stage_dir("traces", create=False) / "traces.jsonl",
                                local_files_only=bool(ctx.cfg["model"].get("local_files_only")))
    return blocks


def _reference(ctx, blocks) -> dict:
    """BF16 reference statistics, cached per block set."""
    from ..config import canonical_hash
    out = ctx.run.stage_dir("eval")
    key = canonical_hash({"blocks": blocks.tolist(), "model": ctx.cfg["model"].get("path"),
                          "topk": int(ctx.cfg["eval"]["compare"]["topk"])})[:16]
    path = out / f"compare-reference-{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    model, _ = ctx.load_model()
    ref = S.compare_reference(model, blocks, topk=int(ctx.cfg["eval"]["compare"]["topk"]))
    del model
    from ..formats.atomic import durable_save
    durable_save(ref, path)
    return ref


def run(ctx, source: str | None = None, suites=None, install_mode: str = "auto") -> dict:
    e = ctx.cfg["eval"]
    log = ctx.log("eval")
    source = resolve_source(ctx, source or e.get("from"))
    suites = list(suites or e.get("suites") or ["dev"])
    unknown = set(suites) - set(S.SUITES)
    if unknown:
        raise ValueError(f"unknown suites {sorted(unknown)}; known: {S.SUITES}")
    out = ctx.run.stage_dir("eval")
    results = {}
    reference = blocks = None
    if "compare" in suites:
        blocks = compare_blocks(ctx, ctx.load_tokenizer())
        reference = _reference(ctx, blocks)
    model, tok = ctx.load_model()
    label = install(ctx, model, source, install_mode)
    model.eval()
    for suite in suites:
        try:
            if suite == "dev":
                r = S.dev(model, tok, generations=bool(e.get("generations", True)))
            elif suite == "extended":
                r = S.extended(model, tok)
            else:
                r = S.compare(model, blocks, reference)
        except S.SuiteNotApplicable as error:
            log(f"{suite}: skipped ({error})")
            results[suite] = {"suite": suite, "skipped": str(error)}
            continue
        r = {"source": source, **r}
        atomic_write_json(out / f"{label}.{suite}.json", r)
        log(json.dumps({k: v for k, v in r.items() if k != "generations"}))
        results[suite] = r
    return results
