"""``openbonsai`` command line.

    openbonsai run --model Qwen/Qwen3.5-9B --config examples/qwen3_5-9b.yaml --out runs/my-model
        = traces -> calibrate -> quantize -> endpoints -> distill -> export -> eval

    openbonsai <stage> --run runs/my-model [--set section.key=value ...] [--force]
        stages: traces calibrate quantize endpoints distill export eval

    openbonsai export --run runs/my-model [--out X.gguf] [--name NAME] [--verify | --no-verify]
    openbonsai eval --run runs/my-model --from bf16|quantize|endpoints|distill|PATH --suite dev,extended,compare
        PATH: state.safetensors, legacy .pt[,.pt], or a PTQ1_0 .gguf (ours or the official
        Ternary Bonsai; --run defaults to <gguf>.eval and --model gives the config and tokenizer)
    openbonsai inspect PATH            summary of a state file, GGUF or run directory

A stage whose output exists with the same config hash is skipped; ``--force``
reruns it from scratch, discarding its resumable work (with ``run``, also every
later stage; distill then keeps its teacher shards unless distill itself is
named, since they depend only on the BF16 model and data, which their own
input hash covers). Interrupted stages resume: quantize per layer,
traces per batch, distill per teacher shard and checkpoint. A stage refuses to
run on a missing, disabled or stale upstream output; a later stage whose
inputs changed (another config, or a rerun with a different result) reruns.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import STAGES, ConfigError, load_config, stage_enabled
from .rundir import RunDir, StageNotReady

# Resumable units that are valid only for the inputs they were made from.
_RESUMABLE = {"traces": ("parts",), "quantize": ("layers",)}
# Everything --force discards before rerunning a stage.
_FORCE_RESET = {**_RESUMABLE, "distill": ("checkpoint.pt", "train.jsonl", "TRAIN_INPUTS.json")}
_TEACHER = ("teacher", "TEACHER_INPUTS.json")


def _stage_module(stage: str):
    from .stages import calibrate, distill, endpoints, evaluate, export, quantize, traces
    return {"traces": traces, "calibrate": calibrate, "quantize": quantize, "endpoints": endpoints,
            "distill": distill, "export": export, "eval": evaluate}[stage]


def _overrides(args) -> list[tuple[tuple[str, ...], object]]:
    """Stage options given as flags (persisted in run.yaml like --set, so they enter the stage hash)."""
    out = []
    if getattr(args, "stage", None) == "export":
        if args.out is not None:
            out.append((("export", "out"), str(Path(args.out))))
        if args.name is not None:
            out.append((("export", "name"), args.name))
        if args.verify is not None:
            out.append((("export", "verify"), bool(args.verify)))
    return out


def resolve_config(args, run: RunDir) -> dict:
    saved = run.load_config()
    if args.config is not None:
        cfg = load_config(args.config, args.set or (), model=args.model)
    elif saved is not None:
        from .config import apply_set, check_keys, deep_merge, DEFAULTS
        cfg = check_keys(deep_merge(DEFAULTS, saved))
        for assignment in args.set or ():
            apply_set(cfg, assignment)
        if args.model is not None:
            cfg["model"]["path"] = args.model
    else:
        cfg = load_config(None, args.set or (), model=args.model)
    for path, value in _overrides(args):
        node = cfg
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
    if not cfg["model"].get("path"):
        raise ConfigError("no model: pass --model or set model.path in the config")
    run.root.mkdir(parents=True, exist_ok=True)
    run.save_config(cfg)
    return cfg


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _prepare(ctx, stage: str, config_hash: str, force) -> None:
    """Drop resumable units made under different inputs (all of them on --force); record the inputs.

    ``force="downstream"``: forced only because an earlier stage was; distill keeps its teacher."""
    from .formats.atomic import atomic_write_json
    d = ctx.run.stage_dir(stage)
    if force:
        teacher = _TEACHER if stage == "distill" and force != "downstream" else ()
        for name in _FORCE_RESET.get(stage, ()) + teacher:
            _remove(d / name)
    if stage not in _RESUMABLE:
        return
    marker = d / "INPUTS.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("config_hash") != config_hash:
        for sub in _RESUMABLE[stage]:
            _remove(d / sub)
    atomic_write_json(marker, {"config_hash": config_hash})


def execute(ctx, stage: str, *, force=False, **kwargs) -> str:
    """Run one stage unless it is complete with the same config hash. Returns a status word.

    ``force``: False, True (rerun from scratch) or "downstream" (see ``_prepare``)."""
    from .digest import file_sha256
    from .stages.common import free
    log = ctx.log(stage)
    if not stage_enabled(ctx.cfg, stage):
        ctx.run.clear_done(stage)   # a disabled stage's old output must not feed later stages
        log("disabled in config; skipped")
        return "disabled"
    ctx.require_inputs(stage)
    config_hash = ctx.stage_hash(stage)
    if stage != "eval" and not force and ctx.run.is_done(stage, config_hash):
        log(f"up to date ({config_hash[:12]}); skipped")
        return "skipped"
    _prepare(ctx, stage, config_hash, force)
    ctx.run.clear_done(stage)
    try:
        result = _stage_module(stage).run(ctx, **kwargs)
    finally:
        free(ctx.device)
    if stage != "eval":
        output = ctx.run.output_path(stage)
        if output is None and isinstance(result, dict) and result.get("output"):
            output = Path(result["output"])
        ctx.run.mark_done(stage, config_hash, result=result,
                          output_sha256=file_sha256(output) if output is not None and output.exists() else None)
    return "ran"


def preflight(ctx) -> None:
    """Fail before any work on settings a later stage would refuse."""
    e = ctx.cfg["endpoints"]
    if stage_enabled(ctx.cfg, "endpoints") and "ternary" in (e.get("head"), e.get("embedding")) \
            and ctx.adapter().tied:
        raise ConfigError("this model ties its embedding and head; ternary endpoints are not supported yet. "
                          "Set endpoints.head and endpoints.embedding to bf16 (or endpoints.enabled=false)")
    if stage_enabled(ctx.cfg, "export"):
        ternary_ends = stage_enabled(ctx.cfg, "endpoints") and e.get("head") == "ternary" \
            and e.get("embedding") == "ternary"
        if ctx.adapter().gguf_arch is None:
            raise ConfigError(f"the {ctx.adapter().name} adapter has no PTQ1_0 GGUF layout; set export.enabled=false")
        if not ternary_ends:
            raise ConfigError("the PTQ1_0 GGUF stores the LM head and the token embedding as ternary: enable the "
                              "endpoints stage with head and embedding ternary, or set export.enabled=false")


def cmd_run(args) -> int:
    from .stages.common import StageContext
    run = RunDir(args.out)
    ctx = StageContext(resolve_config(args, run), run)
    preflight(ctx)
    stop = STAGES.index(args.until) if args.until else len(STAGES) - 1
    forced = set(args.force or ())
    statuses = {}
    for i, stage in enumerate(STAGES[:stop + 1]):
        if stage in (args.skip or ()):
            statuses[stage] = "skipped by --skip"
            continue
        force = True if stage in forced else ("downstream" if any(STAGES.index(f) < i for f in forced) else False)
        statuses[stage] = execute(ctx, stage, force=force)
    print(json.dumps({"run": str(run.root), "stages": statuses}, indent=1))
    return 0


def cmd_stage(args) -> int:
    from .stages.common import StageContext
    from .stages.evaluate import is_gguf
    if args.run is None:
        if args.stage == "eval" and is_gguf(args.source):
            args.run = str(args.source) + ".eval"
        else:
            raise ConfigError("--run is required (it may be omitted only for eval --from model.gguf)")
    run = RunDir(args.run)
    ctx = StageContext(resolve_config(args, run), run)
    kwargs = {}
    if args.stage == "eval":
        kwargs = {"source": args.source, "suites": args.suite.split(",") if args.suite else None,
                  "install_mode": args.install, "gguf_small_fp32": args.gguf_small_fp32}
    status = execute(ctx, args.stage, force=args.force, **kwargs)
    print(json.dumps({"run": str(run.root), "stage": args.stage, "status": status}))
    return 0


def cmd_inspect(args) -> int:
    from .formats.state import load_state
    path = Path(args.path)
    if path.suffix.lower() == ".gguf":
        from .formats.gguf import GGUFReader
        with GGUFReader(path) as r:
            types = {}
            for t in r.tensors:
                types[t.gtype] = types.get(t.gtype, 0) + 1
            kv = {k: (v if not isinstance(v, (list, str)) or len(v) <= 64 else f"<{type(v).__name__} of {len(v)}>")
                  for k, _, v in r.kv}
            print(json.dumps({"path": str(path), "bytes": r.size, "tensors": len(r.tensors),
                              "tensor_types": {str(k): v for k, v in sorted(types.items())}, "kv": kv}, indent=1))
        return 0
    if path.is_dir():
        run = RunDir(path)
        info = {"run": str(path), "stages": {s: run.done_info(s) for s in STAGES if run.done_info(s)}}
        print(json.dumps(info, indent=1, default=str))
        return 0
    state = load_state(path)
    cols = {}
    for t in state.tensors.values():
        cols[t.shape[1]] = cols.get(t.shape[1], 0) + 1
    weights = sum(t.codes.numel() for t in state.tensors.values())
    print(json.dumps({"path": str(path), "tensors": len(state.tensors), "weights": weights,
                      "block": state.block, "group": state.group, "sign_widths": sorted(state.signs),
                      "tensors_by_input_width": cols,
                      "ptq1_0_bytes": weights // 128 * 28, "meta": state.meta}, indent=1))
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="HF model id or local directory (sets model.path)")
    p.add_argument("--config", help="YAML config (defaults to the run's saved run.yaml)")
    p.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override one config value")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="openbonsai", description="Ternary (PTQ1_0) compression of dense LLMs.")
    ap.add_argument("--version", action="version", version=f"openbonsai {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every stage into one run directory")
    r.add_argument("--out", required=True, help="run directory")
    _common(r)
    r.add_argument("--until", choices=STAGES, help="stop after this stage")
    r.add_argument("--force", action="append", choices=STAGES, help="rerun this stage and every later one")
    r.add_argument("--skip", action="append", choices=STAGES, help="do not run this stage")
    r.set_defaults(func=cmd_run)
    for stage in STAGES:
        s = sub.add_parser(stage, help=f"run the {stage} stage")
        s.add_argument("--run", required=stage != "eval", default=None, help="run directory")
        _common(s)
        s.add_argument("--force", action="store_true", help="rerun even if up to date")
        if stage == "eval":
            s.add_argument("--from", dest="source", default=None,
                           help="bf16 | quantize | endpoints | distill | state.safetensors | legacy.pt[,more.pt] "
                                "| model.gguf")
            s.add_argument("--suite", default=None, help="comma-separated: dev, extended, compare")
            s.add_argument("--install", choices=("auto", "dense", "ternary"), default="auto",
                           help="dense weights or ternary modules (auto: ternary for distill and GGUF)")
            s.add_argument("--gguf-small-fp32", action="store_true",
                           help="GGUF source: keep 1-D F32 tensors (norms, A_log, dt_bias) in FP32 as Prism does")
        if stage == "export":
            s.add_argument("--out", default=None, help="output .gguf (default: <run>/export/<name>.gguf)")
            s.add_argument("--name", default=None, help="general.name (default: OpenBonsai-<model>-PTQ1_0)")
            s.add_argument("--verify", dest="verify", action="store_true", default=None,
                           help="verify the written file (the default)")
            s.add_argument("--no-verify", dest="verify", action="store_false", help="skip verification")
        s.set_defaults(func=cmd_stage, stage=stage)
    i = sub.add_parser("inspect", help="summarize a state file, GGUF or run directory")
    i.add_argument("path")
    i.set_defaults(func=cmd_inspect)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .formats.gguf import GGUFError
    from .stages.export import ExportError
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, NotImplementedError, StageNotReady, ExportError, GGUFError) as error:
        print(f"openbonsai: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
