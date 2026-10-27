"""End-to-end distillation of ternary group scales and norms (codes frozen).

Stage after gptq_ternary.py, in the spirit of EfficientQAT's E2E-QP: the ternary codes
from the layer-wise pass stay fixed; per-(row, group) scales (as log multipliers) and
all RMSNorm weights are trained through the whole model against the BF16 teacher's
top-k next-token distribution. This targets the cross-layer error accumulation that
layer-local reconstruction cannot see.

  python e2e_kd.py teacher --state S --data-out D   # precompute teacher top-k
  python e2e_kd.py train --state S --data D --out O # train scales/norms
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import calib_data
from gptq_ternary import hadamard
from quick_eval import FROZEN, evaluate, extended_eval, load_model

TOPK = 64


class TernaryLinear(torch.nn.Module):
    """y = (C * s) (R x) with R = blockdiag(H) diag(signs); s = s0 * exp(u)."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.out_features, self.in_features = codes.shape
        self.group = self.in_features // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h  # shared [block, block] BF16 Hadamard, not a parameter
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def weight_deq(self):
        s = self.scales().to(torch.bfloat16)
        c = self.codes.view(self.out_features, -1, self.group).to(torch.bfloat16)
        return (c * s[..., None]).view(self.out_features, self.in_features)

    def forward(self, x):
        shape = x.shape
        xr = (x * self.signs).view(*shape[:-1], self.in_features // self.block, self.block) @ self.h
        return F.linear(xr.view(shape), self.weight_deq())


class TernaryEmbedding(torch.nn.Module):
    """e = (C[ids] * s[ids]) R^T, i.e. stored rows live in the rotated basis."""

    def __init__(self, codes, scales, signs, block, h):
        super().__init__()
        self.num, self.dim = codes.shape
        self.group = self.dim // scales.shape[1]
        self.block = block
        self.register_buffer("codes", codes.to(torch.int8), persistent=False)
        self.register_buffer("s0", scales.float(), persistent=False)
        self.register_buffer("signs", signs.to(torch.bfloat16), persistent=False)
        self.h = h
        self.log_mult = torch.nn.Parameter(torch.zeros_like(self.s0))

    def scales(self):
        return self.s0 * torch.exp(self.log_mult)

    def forward(self, ids):
        c = self.codes[ids].view(*ids.shape, -1, self.group).to(torch.bfloat16)
        s = (self.s0[ids] * torch.exp(self.log_mult[ids])).to(torch.bfloat16)
        rows = (c * s[..., None]).view(*ids.shape, self.dim // self.block, self.block)
        return (rows @ self.h).view(*ids.shape, self.dim) * self.signs


def install_endpoints(model, ep):
    dev = next(model.parameters()).device
    lm = model.model.language_model
    h = hadamard(4096, dev).to(torch.bfloat16)
    if "lm_head" in ep:
        st = ep["lm_head"]
        model.lm_head = TernaryLinear(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], 4096).to(dev), st["block"], h)
    if "embed_tokens" in ep:
        st = ep["embed_tokens"]
        lm.embed_tokens = TernaryEmbedding(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], 4096).to(dev), st["block"], h)
    torch.cuda.empty_cache()


def signs_for(seed, in_features):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float()


def install_ternary(model, state):
    layers = model.model.language_model.layers
    dev = next(model.parameters()).device
    hs = {}
    for name, st in state["modules"].items():
        _, li, sub = name.split(".", 2)
        parent_name, attr = sub.rsplit(".", 1)
        parent = layers[int(li)].get_submodule(parent_name)
        lin = getattr(parent, attr)
        block = st["block"]
        if block not in hs:
            hs[block] = hadamard(block, dev).to(torch.bfloat16)
        tl = TernaryLinear(st["codes"].to(dev), st["scales"].to(dev), signs_for(st["seed"], lin.in_features).to(dev), block, hs[block])
        setattr(parent, attr, tl)
        del lin
    torch.cuda.empty_cache()


def norm_params(model):
    for n, p in model.model.language_model.named_parameters():
        if n.endswith("norm.weight") or ".norm.weight" in n:
            yield n, p


def build_data(tok, n_seq, seq_len, seed):
    n_wiki = min(n_seq // 4, 1000)
    n_ayot = min(n_seq // 32, 24)
    n_chat = n_seq - n_wiki - n_ayot
    return calib_data.build(tok, seq_len, n_wiki, n_chat, n_ayot, seed=seed)


def durable_save(obj, path, saver=torch.save):
    """Write to a temp file, fsync, then atomically rename (survives an OS crash)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        saver(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def cmd_teacher(args):
    """Teacher top-k in fsync'd shards of --shard rows; rerunning resumes after the last shard."""
    model, tok = load_model()
    out = Path(args.data_out)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "input_ids.npy"
    if ids_path.exists():
        data = torch.from_numpy(np.load(ids_path).astype(np.int64))
    else:
        data = build_data(tok, args.n_seq, args.seq_len, seed=args.seed)
        durable_save(data.numpy().astype(np.int32), ids_path, saver=lambda o, f: np.save(f, o))
    n, t = data.shape
    t0 = time.time()
    for s0 in range(0, n, args.shard):
        shard = out / f"topk_{s0:06d}.npz"
        if shard.exists():
            continue
        rows = range(s0, min(s0 + args.shard, n))
        idx = np.empty((len(rows), t, TOPK), np.int32)
        lp = np.empty((len(rows), t, TOPK), np.float16)
        for k, i in enumerate(rows):
            logp = model(input_ids=data[i:i + 1].cuda(), use_cache=False).logits[0].float().log_softmax(-1)
            v, j = logp.topk(TOPK, -1)
            idx[k] = j.int().cpu().numpy()
            lp[k] = v.half().cpu().numpy()
        durable_save({"idx": idx, "logp": lp}, shard, saver=lambda o, f: np.savez(f, **o))
        print(f"teacher {s0 + len(rows)}/{n} {time.time() - t0:.0f}s mass={np.exp(lp.astype(np.float32)).sum(-1).mean():.3f}", flush=True)
    print(f"teacher done {n * t} tokens in {time.time() - t0:.0f}s", flush=True)


def load_teacher(data_dir):
    ids = np.load(data_dir / "input_ids.npy")
    shards = sorted(data_dir.glob("topk_*.npz"))
    idx = np.concatenate([np.load(f)["idx"] for f in shards])
    lp = np.concatenate([np.load(f)["logp"] for f in shards])
    assert len(idx) == len(ids), f"teacher incomplete: {len(idx)}/{len(ids)} rows"
    return ids, idx, lp


def kd_chunk_loss(h, head, tidx, tlogp):
    """Cross-entropy to the teacher's top-k plus one tail bucket for its remaining mass,
    so the optimum is q = p on the top-k (not a sharpened, tail-free distribution)."""
    logq = head(h).float().log_softmax(-1)
    lqk = logq.gather(-1, tidx)
    p = tlogp.float().exp()
    tail_p = (1 - p.sum(-1)).clamp_min(0)
    log_tail_q = torch.log(-torch.expm1(lqk.logsumexp(-1).clamp_max(-1e-6)))
    return -(p * lqk).sum() - (tail_p * log_tail_q).sum()


def cmd_train(args):
    torch.manual_seed(0)
    model, tok = load_model()
    state = torch.load(args.state, map_location="cpu")
    install_ternary(model, state)
    del state
    if args.endpoints:
        install_endpoints(model, torch.load(args.endpoints, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    scale_params = [m.log_mult for m in model.modules() if isinstance(m, (TernaryLinear, TernaryEmbedding))]
    norms = [p for _, p in norm_params(model)]
    for p in scale_params:
        p.requires_grad_(True)
    if args.train_norms:
        raise NotImplementedError("norm training needs an FP32 master wrapper; not wired yet")
    print(f"trainable: {sum(p.numel() for p in scale_params)} scale params, {sum(p.numel() for p in norms) if args.train_norms else 0} norm params", flush=True)
    model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def dev_eval(gen=False):
        model.eval()
        r = evaluate(model, tok, frozen, gen=gen)
        r.update(extended_eval(model))
        model.train()
        return r

    r0 = dev_eval()
    log(json.dumps({"step": 0, "dev_nll": r0["nll"], "arc": r0["arc_correct"], "arc_preds": r0["arc_preds"], "wiki16k": r0["wiki16k_nll"], "arc100": r0["arc100"], "chat_nll": r0["chat_nll"]}))

    data_dir = Path(args.data)
    ids, tidx_all, tlogp_all = load_teacher(data_dir)
    n = ids.shape[0]
    groups = [{"params": scale_params, "lr": args.lr}]
    if args.train_norms:
        groups.append({"params": norms, "lr": args.lr_norm})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-12)
    total_steps = args.epochs * n // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    head = model.lm_head
    lm = model.model.language_model
    model.train()
    step, t0, run_loss, run_tok = 0, time.time(), 0.0, 0
    order = []
    for ep in range(args.epochs):
        order += torch.randperm(n, generator=torch.Generator().manual_seed(ep)).tolist()
    ckpt = out / "checkpoint.pt"
    start_it = 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cuda")
        for p, v in zip(scale_params, ck["params"]):
            p.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_it = ck["step"], ck["it"]
        log(json.dumps({"resumed_at_step": step}))
    for it, i in enumerate(order):
        if it < start_it:
            continue
        x = torch.from_numpy(ids[i:i + 1].astype(np.int64)).cuda()
        tidx = torch.from_numpy(tidx_all[i].astype(np.int64)).cuda()
        tlogp = torch.from_numpy(np.asarray(tlogp_all[i])).cuda()
        h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        loss = 0.0
        tlen = h.shape[0] - 1
        for c0 in range(0, tlen, args.chunk):
            c1 = min(c0 + args.chunk, tlen)
            loss = loss + checkpoint(kd_chunk_loss, h[c0:c1], head, tidx[c0:c1], tlogp[c0:c1], use_reentrant=False)
        (loss / (tlen * args.accum)).backward()
        run_loss += loss.item()
        run_tok += tlen
        if (it + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(scale_params + (norms if args.train_norms else []), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                lm_abs = torch.cat([p.detach().flatten() for p in scale_params]).abs()
                log(json.dumps({"step": step, "kd_ce": run_loss / run_tok, "tok_s": run_tok / el, "elapsed": el,
                                "lr": sched.get_last_lr()[0], "mult_absmean": lm_abs.mean().item(), "mult_absmax": lm_abs.max().item()}))
                run_loss, run_tok, t0 = 0.0, 0, time.time()
            if step % args.ckpt_every == 0:
                durable_save({"params": [p.detach() for p in scale_params], "opt": opt.state_dict(),
                              "sched": sched.state_dict(), "step": step, "it": it + 1}, ckpt)
            if step % args.eval_every == 0 or step == total_steps:
                r = dev_eval()
                log(json.dumps({"step": step, "dev_nll": r["nll"], "arc": r["arc_correct"], "arc_preds": r["arc_preds"], "wiki16k": r["wiki16k_nll"], "arc100": r["arc100"], "chat_nll": r["chat_nll"]}))
                t0 = time.time()
    r = dev_eval(gen=True)
    log(json.dumps({"final": True, "dev_nll": r["nll"], "ppl": r["ppl"], "arc": r["arc_correct"], "arc_preds": r["arc_preds"], "wiki16k": r["wiki16k_nll"], "arc100": r["arc100"], "chat_nll": r["chat_nll"]}))
    for g in r["generations"]:
        log("GEN: " + g[:400].replace("\n", " | "))
    mods = {}
    for name, m in model.model.language_model.layers.named_modules():
        if isinstance(m, TernaryLinear):
            mods[f"layers.{name}"] = m.scales().detach().half().cpu()
    for name, m in [("lm_head", model.lm_head), ("embed_tokens", model.model.language_model.embed_tokens)]:
        if isinstance(m, (TernaryLinear, TernaryEmbedding)):
            mods[name] = m.scales().detach().half().cpu()
    durable_save({"scales": mods, "norms": {n: p.detach().cpu() for n, p in norm_params(model)}, "args": vars(args)}, out / "e2e_state.pt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--data-out", required=True)
    t.add_argument("--n-seq", type=int, default=2048)
    t.add_argument("--seq-len", type=int, default=2048)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--shard", type=int, default=64)
    tr = sub.add_parser("train")
    tr.add_argument("--state", required=True)
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--endpoints", default="", help="endpoint_state.pt from endpoints.py")
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--accum", type=int, default=8)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--lr-norm", type=float, default=1e-4)
    tr.add_argument("--train-norms", action="store_true")
    tr.add_argument("--warmup", type=int, default=10)
    tr.add_argument("--chunk", type=int, default=512)
    tr.add_argument("--log-every", type=int, default=5)
    tr.add_argument("--eval-every", type=int, default=50)
    tr.add_argument("--ckpt-every", type=int, default=16)
    args = ap.parse_args()
    {"teacher": cmd_teacher, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
