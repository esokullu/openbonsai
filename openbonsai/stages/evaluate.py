"""Score a run's model (BF16, or after quantize/endpoints/distill) on the suites.

Sources (``--from``):

- ``bf16``: the original model;
- ``quantize``, ``endpoints``: dequantized weights (C * s) R^T written into the
  model, as the decoder-only and endpoint results were scored;
- ``distill``: TernaryLinear/TernaryEmbedding modules with the distilled fp16
  scales (the runtime arithmetic, as e2e_kd.py scored it), plus the distilled
  high-precision tensors when ``distill.train.small``;
- a path: a state ``.safetensors`` or a legacy research ``.pt`` (decoder
  ``ternary_state.pt`` and/or ``endpoint_state.pt``; comma-separated). Legacy
  per-module signs are applied exactly as ``apply_state`` did;
- a Prism PTQ1_0 ``.gguf`` (ours or the official Ternary Bonsai): the whole
  model is built from the file (``adapter.from_gguf``: ternary modules plus its
  F32 tensors); only the config and tokenizer come from ``model.path``. The
  ``compare`` reference still comes from the BF16 model.

``compare`` caches the BF16 reference in ``eval/compare-reference-<key>.pt``
(NLL, top-1, top-k) and, with ``eval.compare.full`` (the default), the
reference's raw logits per block in ``compare-reference-<key>.logits/`` for
the full-vocabulary ``ear_full`` and ``kl_full`` (n x (seq_len - 1) x vocab in
the model's dtype: about 2 GB for Qwen3.5-9B at the defaults). Later
evaluations of other sources reuse both without loading the BF16 model.
``teacher`` scores against the first ``eval.teacher.rows`` rows of this run's
distillation teacher shards (top-k metrics only; no BF16 model).

A stage source must be enabled, complete and current (same config, inputs and
upstream outputs as when it ran); ``auto`` picks the last such stage. So a
rerun quantize never meets distilled scales trained on the old codes.

Results go to ``eval/<source>.<suite>.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import ConfigError
from ..eval import suites as S
from ..formats.atomic import atomic_write_json
from ..formats.state import load_legacy_pt, load_state
from . import calibrate
from .common import apply_dense, apply_small, free, install_ternary, legacy_records

STAGE_SOURCES = ("bf16", "quantize", "endpoints", "distill")


def is_gguf(source) -> bool:
    return isinstance(source, str) and source.lower().endswith(".gguf") and "," not in source


def resolve_source(ctx, source: str | None) -> str:
    if source in (None, "auto"):
        for stage in ("distill", "endpoints", "quantize"):
            if ctx.stale_reason(stage) is None:
                return stage
        return "bf16"
    if source in STAGE_SOURCES[1:]:
        ctx.require_current(source)
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
        if source == "distill":
            apply_small(adapter, model, ctx.distill_small())
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


def _reference(ctx, blocks, model) -> tuple[dict, object]:
    """BF16 reference statistics of the unmodified model, cached per model and block set.

    ``model``: the model, or a callable returning it (called only when the cache misses).
    Returns (statistics, ``ref_logits(b)`` loader of the cached raw logits, or None
    when ``eval.compare.full`` is off)."""
    from ..config import canonical_hash
    from ..formats.atomic import durable_save
    out = ctx.run.stage_dir("eval")
    m = ctx.cfg["model"]
    c = ctx.cfg["eval"]["compare"]
    key = canonical_hash({"blocks": blocks.tolist(), "model": [m.get(k) for k in ("path", "revision", "dtype", "attn")],
                          "model_sha256": ctx.model_digest(), "topk": int(c["topk"])})[:16]
    path = out / f"compare-reference-{key}.pt"
    full = bool(c.get("full", True))
    logits_dir = out / f"compare-reference-{key}.logits"
    files = [logits_dir / f"block_{b:04d}.pt" for b in range(len(blocks))]
    loader = (lambda b: torch.load(files[b], weights_only=True)) if full else None
    if path.exists() and (not full or all(f.exists() for f in files)):
        return torch.load(path, weights_only=True), loader
    if callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
    model.eval()
    keep = None
    if full:
        vocab = S.vocab_size(model)
        ctx.log("eval")(f"compare: keeping the reference logits for ear_full/kl_full in {logits_dir.name}: "
                        f"{len(blocks)} x {blocks.shape[1] - 1} x {vocab} in {ctx.dtype} "
                        f"(~{len(blocks) * (blocks.shape[1] - 1) * vocab * ctx.dtype.itemsize / 2**30:.2f} GiB)")

        def keep(b, logits):
            durable_save(logits, files[b])
    ref = S.compare_reference(model, blocks, topk=int(c["topk"]), keep_logits=keep)
    durable_save(ref, path)
    return ref, loader


def teacher_rows(ctx):
    """(ids, idx, logp) of the first ``eval.teacher.rows`` rows of the run's distillation teacher shards."""
    from .distill import read_teacher_rows
    folder = ctx.run.stage_dir("distill", create=False) / "teacher"
    try:
        return read_teacher_rows(folder, int(ctx.cfg["eval"]["teacher"]["rows"]))
    except FileNotFoundError as error:
        raise S.SuiteNotApplicable(f"no distillation teacher shards in this run ({error})") from None


def load_gguf(ctx, path, *, small_fp32: bool = False):
    """The native model built from a PTQ1_0 GGUF on the run's device and dtype."""
    m = ctx.cfg["model"]
    return ctx.adapter().from_gguf(path, model_dir=ctx.model_dir(), dtype=ctx.dtype, device=ctx.device,
                                   attn=m.get("attn") or "sdpa", small_fp32=small_fp32, log=ctx.log("eval"))


def run(ctx, source: str | None = None, suites=None, install_mode: str = "auto",
        gguf_small_fp32: bool = False) -> dict:
    e = ctx.cfg["eval"]
    log = ctx.log("eval")
    source = resolve_source(ctx, source or e.get("from"))
    suites = list(suites or e.get("suites") or ["dev"])
    unknown = set(suites) - set(S.SUITES)
    if unknown:
        raise ValueError(f"unknown suites {sorted(unknown)}; known: {S.SUITES}")
    out = ctx.run.stage_dir("eval")
    results = {}
    reference = blocks = ref_logits = None
    if is_gguf(source):
        if install_mode == "dense":
            raise ConfigError("a GGUF source is installed as ternary modules; --install dense does not apply")
        tok = ctx.load_tokenizer()
        if "compare" in suites:
            blocks = compare_blocks(ctx, tok)
            reference, ref_logits = _reference(ctx, blocks, lambda: ctx.load_model()[0])
            free(ctx.device)
        model = load_gguf(ctx, source, small_fp32=gguf_small_fp32)
        label = Path(source).stem + ("-fp32small" if gguf_small_fp32 else "")
    else:
        model, tok = ctx.load_model()
        if "compare" in suites:
            blocks = compare_blocks(ctx, tok)
            model.eval()
            reference, ref_logits = _reference(ctx, blocks, model)   # before any ternary weights go in
        label = install(ctx, model, source, install_mode)
    model.eval()
    for suite in suites:
        try:
            if suite == "dev":
                r = S.dev(model, tok, generations=bool(e.get("generations", True)))
            elif suite == "extended":
                r = S.extended(model, tok)
            elif suite == "teacher":
                r = S.teacher(model, *teacher_rows(ctx))
            else:
                r = S.compare(model, blocks, reference, ref_logits=ref_logits)
        except S.SuiteNotApplicable as error:
            log(f"{suite}: skipped ({error})")
            results[suite] = {"suite": suite, "skipped": str(error)}
            continue
        r = {"source": source, **r}
        atomic_write_json(out / f"{label}.{suite}.json", r)
        log(json.dumps({k: v for k, v in r.items() if k != "generations"}))
        results[suite] = r
    return results
