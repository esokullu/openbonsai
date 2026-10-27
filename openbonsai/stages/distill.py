"""End-to-end distillation of the ternary group scales (codes frozen).

Port of research/experiments/e2e_kd.py with the same arithmetic. The codes
from GPTQ stay fixed; every (row, group) scale gets a trainable log
multiplier, trained through the whole model against the original BF16
model's top-k next-token distribution (cross-entropy on the top-k plus a
tail bucket for the remaining mass). This targets cross-layer error that
layer-local reconstruction cannot see. Norm training is not wired yet.

Two phases, both resumable:

- teacher: ``distill/teacher/input_ids.npy`` then ``topk_NNNNNN.npz`` shards
  (int32 ids, fp16 log-probs), each written atomically; finished shards are
  skipped on rerun.
- train: AdamW on the log multipliers with warmup + cosine; a checkpoint
  (parameters, optimizer, scheduler, position) every ``ckpt_every`` steps is
  resumed exactly. The result is ``distill/scales.safetensors`` (fp16).
"""
from __future__ import annotations

import io
import json
import math
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ..core.losses import kd_chunk_loss
from ..core.ternary import TernaryEmbedding, TernaryLinear
from ..formats.atomic import atomic_write_bytes, durable_save
from ..formats.state import save_scales
from . import calibrate
from .common import install_ternary


def teacher_sources(cfg: dict) -> list[dict]:
    """The experiments' mix: a quarter WikiText (<= 1000), 1/32 traces (<= 24), the rest chat."""
    t = cfg["distill"]["teacher"]
    if t.get("sources"):
        return t["sources"]
    n = int(t["n_seq"])
    n_wiki, n_ayot = min(n // 4, 1000), min(n // 32, 24)
    return [{"kind": "wikitext2", "n": n_wiki}, {"kind": "ultrachat", "n": n - n_wiki - n_ayot, "min_row": 6000},
            {"kind": "traces", "n": n_ayot}]


def _npy_bytes(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


@torch.no_grad()
def teacher_topk(model, data, out_dir, *, topk=64, shard=64, device, log=print):
    """Teacher top-k in atomic shards of ``shard`` rows; rerunning resumes after the last shard."""
    n, t = data.shape
    t0 = time.time()
    for s0 in range(0, n, shard):
        path = out_dir / f"topk_{s0:06d}.npz"
        if path.exists():
            continue
        rows = range(s0, min(s0 + shard, n))
        idx = np.empty((len(rows), t, topk), np.int32)
        lp = np.empty((len(rows), t, topk), np.float16)
        for k, i in enumerate(rows):
            logp = model(input_ids=data[i:i + 1].to(device), use_cache=False).logits[0].float().log_softmax(-1)
            v, j = logp.topk(topk, -1)
            idx[k] = j.int().cpu().numpy()
            lp[k] = v.half().cpu().numpy()
        atomic_write_bytes(path, _npz_bytes(idx=idx, logp=lp))
        log(f"teacher {s0 + len(rows)}/{n} {time.time() - t0:.0f}s "
            f"mass={np.exp(lp.astype(np.float32)).sum(-1).mean():.3f}")


def load_teacher(data_dir):
    ids = np.load(data_dir / "input_ids.npy")
    shards = sorted(data_dir.glob("topk_*.npz"))
    starts = [int(f.stem.split("_")[1]) for f in shards]
    idx = [np.load(f)["idx"] for f in shards]
    lp = [np.load(f)["logp"] for f in shards]
    expected = 0
    for s, a in zip(starts, idx):
        if s != expected:
            raise ValueError(f"teacher shards are not contiguous at row {expected}")
        expected += len(a)
    if expected != len(ids):
        raise ValueError(f"teacher incomplete: {expected}/{len(ids)} rows")
    return ids, np.concatenate(idx), np.concatenate(lp)


def run_teacher(ctx) -> dict:
    d = ctx.cfg["distill"]
    t = d["teacher"]
    log = ctx.log("distill")
    out = ctx.run.stage_dir("distill") / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "input_ids.npy"
    adapter = ctx.adapter()
    model, tok = ctx.load_model()
    if ids_path.exists():
        data = torch.from_numpy(np.load(ids_path).astype(np.int64))
    else:
        data, report = calibrate.build(tok, teacher_sources(ctx.cfg), seq_len=int(t["seq_len"]), seed=int(t["seed"]),
                                       chat_kwargs=adapter.chat_kwargs(thinking=False), bos=adapter.bos_id(tok),
                                       traces_path=ctx.run.stage_dir("traces", create=False) / "traces.jsonl",
                                       local_files_only=bool(ctx.cfg["model"].get("local_files_only")))
        atomic_write_bytes(out / "sources.json", json.dumps(report, indent=1).encode("utf-8"))
        atomic_write_bytes(ids_path, _npy_bytes(data.numpy().astype(np.int32)))
    teacher_topk(model, data, out, topk=int(t["topk"]), shard=int(t["shard"]), device=ctx.device, log=log)
    return {"rows": int(data.shape[0]), "seq_len": int(data.shape[1])}


def train(model, adapter, ids, tidx_all, tlogp_all, out_dir, *, lr=2e-3, epochs=1, accum=8, warmup=10,
          chunk=512, log_every=5, ckpt_every=16, max_steps=None, device, log=print):
    """Train every TernaryLinear/TernaryEmbedding log multiplier; returns the per-step loss history."""
    torch.manual_seed(0)
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params = [m.log_mult for m in model.modules() if isinstance(m, (TernaryLinear, TernaryEmbedding))]
    for p in scale_params:
        p.requires_grad_(True)
    log(f"trainable: {sum(p.numel() for p in scale_params)} scale params")
    lm = adapter.text_model(model)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    head = adapter.head(model)
    n = ids.shape[0]
    opt = torch.optim.AdamW([{"params": scale_params, "lr": lr}], betas=(0.9, 0.95), weight_decay=0.0, eps=1e-12)
    total_steps = max(1, epochs * n // accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    order = []
    for ep in range(epochs):
        order += torch.randperm(n, generator=torch.Generator().manual_seed(ep)).tolist()
    ckpt = out_dir / "checkpoint.pt"
    history_path = out_dir / "train.jsonl"
    step, start_it = 0, 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        for p, v in zip(scale_params, ck["params"]):
            p.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_it = ck["step"], ck["it"]
        log(json.dumps({"resumed_at_step": step}))
    model.train()
    t0, run_loss, run_tok, step_loss, step_tok = time.time(), 0.0, 0, 0.0, 0
    history = []
    for it, i in enumerate(order):
        if it < start_it:
            continue
        if max_steps is not None and step >= max_steps:
            break
        x = torch.from_numpy(ids[i:i + 1].astype(np.int64)).to(device)
        tidx = torch.from_numpy(tidx_all[i].astype(np.int64)).to(device)
        tlogp = torch.from_numpy(np.asarray(tlogp_all[i])).to(device)
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        loss = 0.0
        tlen = h.shape[0] - 1
        for c0 in range(0, tlen, chunk):
            c1 = min(c0 + chunk, tlen)
            loss = loss + checkpoint(kd_chunk_loss, h[c0:c1], head, tidx[c0:c1], tlogp[c0:c1], use_reentrant=False)
        (loss / (tlen * accum)).backward()
        run_loss += loss.item()
        run_tok += tlen
        step_loss += loss.item()
        step_tok += tlen
        if (it + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(scale_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "kd_ce": step_loss / step_tok, "lr": sched.get_last_lr()[0]}
            history.append(record)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            step_loss, step_tok = 0.0, 0
            if step % log_every == 0:
                el = time.time() - t0
                lm_abs = torch.cat([p.detach().flatten() for p in scale_params]).abs()
                log(json.dumps({"step": step, "kd_ce": run_loss / run_tok, "tok_s": run_tok / el, "elapsed": el,
                                "lr": sched.get_last_lr()[0], "mult_absmean": lm_abs.mean().item(),
                                "mult_absmax": lm_abs.max().item()}))
                run_loss, run_tok, t0 = 0.0, 0, time.time()
            if step % ckpt_every == 0:
                durable_save({"params": [p.detach() for p in scale_params], "opt": opt.state_dict(),
                              "sched": sched.state_dict(), "step": step, "it": it + 1}, ckpt)
    model.eval()
    return history


def trained_scales(model, adapter) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in adapter.layers(model).named_modules():
        if isinstance(m, TernaryLinear):
            out[f"layers.{name}"] = m.scales().detach().half().cpu()
    for key, m in (("lm_head", adapter.head(model)), ("embed_tokens", adapter.embedding(model))):
        if isinstance(m, (TernaryLinear, TernaryEmbedding)):
            out[key] = m.scales().detach().half().cpu()
    return out


def run_train(ctx) -> dict:
    d = ctx.cfg["distill"]
    if list(d.get("train") or ["scales"]) != ["scales"]:
        raise NotImplementedError(f"distill.train {d.get('train')}: only [scales] is implemented")
    log = ctx.log("distill")
    out = ctx.run.stage_dir("distill")
    adapter = ctx.adapter()
    model, _ = ctx.load_model()
    state = ctx.quantize_state()
    if d.get("endpoints", True):
        ep = ctx.endpoints_state()
        if ep is not None and ep.tensors:
            state = state.merged(ep)
    install_ternary(adapter, model, state.records(), ctx.device)
    ids, tidx, tlogp = load_teacher(out / "teacher")
    history = train(model, adapter, ids, tidx, tlogp, out, lr=float(d["lr"]), epochs=int(d["epochs"]),
                    accum=int(d["accum"]), warmup=int(d["warmup"]), chunk=int(d["chunk"]),
                    log_every=int(d["log_every"]), ckpt_every=int(d["ckpt_every"]),
                    max_steps=None if d.get("max_steps") is None else int(d["max_steps"]),
                    device=ctx.device, log=log)
    scales = trained_scales(model, adapter)
    save_scales(out / "scales.safetensors", scales, {"stage": "distill", "steps": len(history)})
    return {"scales": len(scales), "steps_this_run": len(history)}


def _reset_if_changed(marker, value: str, paths) -> None:
    """Drop resumable outputs made under different inputs, then record the inputs."""
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("hash") != value:
        for p in paths:
            if p.is_dir():
                for f in p.glob("*"):
                    f.unlink()
            elif p.exists():
                p.unlink()
    atomic_write_bytes(marker, json.dumps({"hash": value}).encode("utf-8"))


def run(ctx) -> dict:
    from ..config import canonical_hash, stage_inputs
    out = ctx.run.stage_dir("distill")
    teacher_dir = out / "teacher"
    teacher_dir.mkdir(exist_ok=True)
    uses_traces = any(s.get("kind") == "traces" and not s.get("path") for s in teacher_sources(ctx.cfg))
    traces_hash = (ctx.run.done_info("traces") or {}).get("config_hash") if uses_traces else None
    teacher_hash = canonical_hash({"model": stage_inputs(ctx.cfg, "distill")["model"],
                                   "teacher": ctx.cfg["distill"]["teacher"], "traces": traces_hash})
    _reset_if_changed(out / "TEACHER_INPUTS.json", teacher_hash, [teacher_dir])
    _reset_if_changed(out / "TRAIN_INPUTS.json", ctx.stage_hash("distill"),
                      [out / "checkpoint.pt", out / "train.jsonl", out / "scales.safetensors"])
    teacher = run_teacher(ctx)
    return {"teacher": teacher, "train": run_train(ctx)}
