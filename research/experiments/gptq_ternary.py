"""Sequential GPTQ-style ternary quantization of Qwen3.5-9B decoder linears.

Format follows Prism's Ternary Bonsai layout: per-module sign + Hadamard rotation on
the input axis (block 4096), ternary codes {-1,0,1} in groups of 128 along the
rotated input axis, one scale per (row, group). GatedDeltaNet in_proj_a/in_proj_b,
conv1d, A_log, dt_bias and norms stay in high precision.

Layers are processed in order. Each layer's inputs come from the already-quantized
prefix and are pushed forward in memory (no prefix recomputation). Within a layer,
sublayers are quantized in dependency order so later Hessians see quantized inputs.
The model ends up holding dequantized weights W_q = (C*s) R^T in BF16, which is
numerically what the packed runtime computes as (C*s)(R x).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

import calib_data
from quick_eval import FROZEN, evaluate, extended_eval, load_model

KEEP_HIGH_PRECISION = ("in_proj_a", "in_proj_b")
ORDER = {
    "linear_attention": [["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], ["linear_attn.out_proj"],
                         ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
    "full_attention": [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"],
                       ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
}


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def signs_from_seed(seed, in_features):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float()


def rotation_matrix(in_features, block, seed, device):
    """Dense R = diag(signs) @ blockdiag(H_block); W_rot = W R, x_rot = x R."""
    signs = signs_from_seed(seed, in_features).to(device)
    h = hadamard(block, device)
    r = torch.block_diag(*([h] * (in_features // block)))
    return signs[:, None] * r


def fit_group(w, ratios):
    """Per-row ternary (alpha, delta) for one [rows, group] block, least squares over a threshold grid."""
    a = w.abs()
    m = a.mean(1, keepdim=True)
    best_err, best_alpha, best_delta = None, None, None
    for r in ratios:
        delta = r * m
        mask = a > delta
        alpha = (a * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        err = (a - alpha * mask).pow(2).sum(1, keepdim=True)
        if best_err is None:
            best_err, best_alpha, best_delta = err, alpha, delta
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_alpha = torch.where(better, alpha, best_alpha)
            best_delta = torch.where(better, delta, best_delta)
    return best_alpha, best_delta


@torch.no_grad()
def gptq_ternary(w, h, group=128, damp=0.01, ratios=None, refit=True):
    """GPTQ error feedback with ternary group codes. w: [out, in] FP32 (rotated), h: [in, in]."""
    ratios = ratios or [0.3 + 0.05 * i for i in range(19)]
    w = w.clone()
    out_f, in_f = w.shape
    dead = torch.diag(h) == 0
    h = h.clone()
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(in_f, device=h.device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = torch.zeros_like(w, dtype=torch.int8)
    scales = torch.zeros(out_f, in_f // group, device=w.device)
    for i1 in range(0, in_f, group):
        i2 = i1 + group
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        alpha, delta = fit_group(w1, ratios)
        for j in range(group):
            col = w1[:, j]
            c = torch.sign(col) * (col.abs() > delta[:, 0])
            q = c * alpha[:, 0]
            codes[:, i1 + j] = c.to(torch.int8)
            e = (col - q) / hinv1[j, j]
            w1[:, j:] -= e[:, None] * hinv1[j, j:][None, :]
            err1[:, j] = e
        scales[:, i1 // group] = alpha[:, 0]
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes, scales


@torch.no_grad()
def refit_scales(w_rot, codes, h, group):
    """Closed-form per-(row, group) scale minimizing the Hessian-weighted error, codes fixed.

    Uses only the block-diagonal of H per group (exact within a group, ignores
    cross-group coupling), which keeps it cheap and stable.
    """
    out_f, in_f = w_rot.shape
    scales = torch.empty(out_f, in_f // group, device=w_rot.device)
    for g0 in range(0, in_f, group):
        hb = h[g0:g0 + group, g0:g0 + group]
        c = codes[:, g0:g0 + group].float()
        wv = w_rot[:, g0:g0 + group]
        num = ((c @ hb) * wv).sum(1)
        den = ((c @ hb) * c).sum(1).clamp_min(1e-12)
        scales[:, g0 // group] = num / den
    return scales


@torch.no_grad()
def apply_state(model, state, only=None):
    """Write dequantized ternary weights (C*s) R^T from a saved state into the model."""
    layers = model.model.language_model.layers
    for name, st in state["modules"].items():
        if only is not None and not any(o in name for o in only):
            continue
        _, li, sub = name.split(".", 2)
        lin = layers[int(li)].get_submodule(sub)
        codes = st["codes"].to(lin.weight.device).float()
        scales = st["scales"].to(lin.weight.device).float()
        out_f, in_f = codes.shape
        rot = rotation_matrix(in_f, st["block"], st["seed"], codes.device)
        deq = (codes.view(out_f, -1, in_f // scales.shape[1]) * scales[..., None]).view(out_f, in_f)
        lin.weight.copy_((deq @ rot.T).to(lin.weight.dtype))


def capture_layer0_inputs(model, calib, batch):
    layers = model.model.language_model.layers
    store = {"h": [], "kwargs": None}

    class Stop(Exception):
        pass

    def hook(mod, args, kwargs):
        store["h"].append(args[0].detach() if args else kwargs["hidden_states"].detach())
        if store["kwargs"] is None:
            store["kwargs"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise Stop

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    dev = next(model.parameters()).device
    for b in range(0, len(calib), batch):
        try:
            model(input_ids=calib[b:b + batch].to(dev), use_cache=False)
        except Stop:
            pass
    handle.remove()
    return store["h"], store["kwargs"]


def layer_forward(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def quantize_model(model, calib, args, log):
    layers = model.model.language_model.layers
    types = model.config.text_config.layer_types
    assert len(calib) % args.batch == 0, "RoPE tensors are captured for one batch size"
    inps, kwargs = capture_layer0_inputs(model, calib, args.batch)
    log(f"captured {len(inps)} batches; kwargs={list(kwargs)}")
    assert model.config.text_config._attn_implementation == "sdpa", "None mask is only causal under SDPA"
    assert kwargs.get("attention_mask") is None, "calibration blocks must be unpadded"
    state = {}
    for li, layer in enumerate(layers):
        t0 = time.time()
        mods = dict(layer.named_modules())
        for group_names in ORDER[types[li]]:
            hs = {n: None for n in group_names}
            counts = {n: 0 for n in group_names}

            def make_hook(n):
                def hook(mod, args_):
                    x = args_[0].reshape(-1, args_[0].shape[-1]).float()
                    hh = x.T @ x
                    hs[n] = hh if hs[n] is None else hs[n] + hh
                    counts[n] += x.shape[0]
                return hook

            handles = [mods[n].register_forward_pre_hook(make_hook(n)) for n in group_names]
            for x in inps:
                layer_forward(layer, x, kwargs)
            for hd in handles:
                hd.remove()
            for n in group_names:
                lin = mods[n]
                w = lin.weight.float()
                in_f = w.shape[1]
                # Shared signs (one vector per input width, as in Prism's GGUF) or per-module signs.
                seed = in_f if args.shared_signs else 5090 + 1000 * li + sum(map(ord, n))
                rot = rotation_matrix(in_f, math.gcd(in_f, args.block), seed, w.device)
                h = rot.T @ (hs[n] / counts[n]) @ rot
                w_rot = w @ rot
                codes, scales = gptq_ternary(w_rot, h, group=args.group, damp=args.damp)
                if args.refit:
                    scales = refit_scales(w_rot, codes, h, args.group)
                scales = scales.half().float()  # what gets stored and exported
                deq = (codes.float().view(w.shape[0], -1, args.group) * scales[..., None]).view_as(w)
                err = deq - w_rot
                rel = ((err.pow(2).sum() / w_rot.pow(2).sum()).item(), ((err @ h) * err).sum().item() / ((w_rot @ h) * w_rot).sum().item())
                lin.weight.copy_((deq @ rot.T).to(lin.weight.dtype))
                state[f"layers.{li}.{n}"] = {"codes": codes.cpu(), "scales": scales.half().cpu(), "seed": seed,
                                             "signs": signs_from_seed(seed, in_f).to(torch.int8),
                                             "block": math.gcd(in_f, args.block), "rel_mse": rel[0], "h_rel_err": rel[1]}
                del hs[n], rot, h, w_rot, codes, scales, deq
        inps = [layer_forward(layer, x, kwargs) for x in inps]
        torch.cuda.empty_cache()
        rels = {k.split(".", 2)[2]: (round(v["rel_mse"], 3), round(v["h_rel_err"], 4)) for k, v in state.items() if k.startswith(f"layers.{li}.")}
        log(f"layer {li} ({types[li]}) {time.time() - t0:.1f}s (weight rel-mse, H-weighted rel err) {rels}")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wiki", type=int, default=64)
    ap.add_argument("--n-chat", type=int, default=48)
    ap.add_argument("--n-ayot", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--shared-signs", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(json.dumps(vars(args)))
    model, tok = load_model()
    calib = calib_data.build(tok, args.seq_len, args.n_wiki, args.n_chat, args.n_ayot)
    log(f"calibration {tuple(calib.shape)} = {calib.numel()} tokens")
    t0 = time.time()
    state = quantize_model(model, calib, args, log)
    log(f"quantized in {time.time() - t0:.0f}s")
    tmp = out / "ternary_state.pt.tmp"
    with open(tmp, "wb") as f:  # fsync + atomic rename: survives an OS crash
        torch.save({"args": vars(args), "modules": state}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out / "ternary_state.pt")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    res = evaluate(model, tok, frozen, gen=True)
    res.update(extended_eval(model))
    log(json.dumps({k: v for k, v in res.items() if k != "generations"}))
    for g in res["generations"]:
        log("GEN: " + g[:400].replace("\n", " | "))
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
