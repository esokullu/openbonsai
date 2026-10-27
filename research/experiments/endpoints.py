"""Ternary token embedding and LM head in the same rotated group-128 format.

LM head: GPTQ with the Hessian of its actual inputs (final-norm hidden states of the
already-ternary decoder) on calibration data. Rows are independent, so the head is
processed in row chunks.

Embedding: no input Hessian exists (inputs are one-hot), but an embedding row only
reaches the network through layer 0's zero-centered RMSNorm, whose output is
proportional to (1 + w) * e. GPTQ with H = R^T diag((1 + w)^2) R therefore minimizes
the error the first layer actually sees, up to the per-token RMS factor.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import calib_data
from e2e_kd import durable_save
from gptq_ternary import apply_state, gptq_ternary, rotation_matrix, signs_from_seed
from quick_eval import FROZEN, evaluate, extended_eval, load_model


@torch.no_grad()
def head_hessian(model, calib, batch):
    lm = model.model.language_model
    dev = next(model.parameters()).device
    h, n = None, 0
    for b in range(0, len(calib), batch):
        x = lm(input_ids=calib[b:b + batch].to(dev), use_cache=False).last_hidden_state
        x = x.reshape(-1, x.shape[-1]).float()
        h = x.T @ x if h is None else h + x.T @ x
        n += x.shape[0]
    return h / n


@torch.no_grad()
def quantize_rows(weight, h, seed, block, group, damp, chunk):
    """GPTQ-ternary a [rows, in] matrix chunk by chunk on GPU; returns codes, scales, deq (original basis)."""
    rows, in_f = weight.shape
    dev = h.device
    rot = rotation_matrix(in_f, math.gcd(in_f, block), seed, dev)
    h_rot = rot.T @ h @ rot
    codes = torch.empty(rows, in_f, dtype=torch.int8)
    scales = torch.empty(rows, in_f // group, dtype=torch.float16)
    deq = torch.empty(rows, in_f, dtype=weight.dtype)
    for r0 in range(0, rows, chunk):
        w = weight[r0:r0 + chunk].to(dev).float() @ rot
        c, s = gptq_ternary(w, h_rot, group=group, damp=damp)
        codes[r0:r0 + chunk] = c.cpu()
        scales[r0:r0 + chunk] = s.half().cpu()
        d = (c.float().view(c.shape[0], -1, group) * s.half().float()[..., None]).view_as(w)
        deq[r0:r0 + chunk] = (d @ rot.T).to(weight.dtype).cpu()
    return codes, scales, deq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="decoder ternary_state.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-wiki", type=int, default=64)
    ap.add_argument("--n-chat", type=int, default=48)
    ap.add_argument("--n-ayot", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    model.requires_grad_(False)  # weights are overwritten in place below
    state = torch.load(args.state, map_location="cpu")
    apply_state(model, state)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    shared = all(st["seed"] == st["codes"].shape[1] for st in state["modules"].values())
    seed = 4096 if shared else 7001

    def report(tag):
        r = evaluate(model, tok, frozen, gen=tag == "both")
        r.update(extended_eval(model))
        print(json.dumps({"variant": tag, **{k: v for k, v in r.items() if k != "generations"}}), flush=True)
        for g in r.get("generations", []):
            print("GEN: " + g[:300].replace("\n", " | "), flush=True)
        return r

    report("decoder_only")
    calib = calib_data.build(tok, 2048, args.n_wiki, args.n_chat, args.n_ayot)
    t0 = time.time()
    h = head_hessian(model, calib, args.batch)
    head = model.lm_head
    head_orig = head.weight.detach().cpu().clone()
    hc, hs, hd = quantize_rows(head_orig, h, seed, args.block, args.group, args.damp, args.chunk)
    head.weight.copy_(hd.to(head.weight.device))
    print(f"head quantized in {time.time() - t0:.0f}s", flush=True)
    report("plus_head")

    emb = model.model.language_model.embed_tokens
    g = (1.0 + model.model.language_model.layers[0].input_layernorm.weight.float())
    t0 = time.time()
    emb_orig = emb.weight.detach().cpu().clone()
    ec, es, ed = quantize_rows(emb_orig, torch.diag(g * g), seed + 1 if not shared else 4096, args.block, args.group, args.damp, args.chunk)
    emb.weight.copy_(ed.to(emb.weight.device))
    print(f"embedding quantized in {time.time() - t0:.0f}s", flush=True)
    report("both")
    head.weight.copy_(head_orig.to(head.weight.device))
    report("embedding_only")
    eseed = seed + 1 if not shared else 4096
    durable_save({"lm_head": {"codes": hc, "scales": hs, "seed": seed, "signs": signs_from_seed(seed, 4096).to(torch.int8), "block": args.block},
                "embed_tokens": {"codes": ec, "scales": es, "seed": eseed, "signs": signs_from_seed(eseed, 4096).to(torch.int8), "block": args.block}},
               out / "endpoint_state.pt")


if __name__ == "__main__":
    main()
