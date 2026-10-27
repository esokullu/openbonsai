"""Build a SYNTHETIC shared-sign ternary state from the real Qwen3.5-9B weights, on CPU.

Test fixture for export_gguf.py while the real GPTQ shared-sign state is produced
elsewhere. Every decoder linear, the LM head and the token embedding are rotated with
OpenBonsai's convention (W_rot = W diag(s) blockdiag(H4096), s = one shared vector
per input width, seeded with the width as gptq_ternary.py --shared-signs does) and
round-to-nearest ternarized per (row, 128-group): threshold 0.7*mean|w|, scale =
mean |w| above the threshold.

Outputs, in the exact layouts of gptq_ternary.py / endpoints.py / e2e_kd.py:
  ternary_state.pt   {"args", "modules": {"layers.{li}.{sub}": {codes, scales, seed, signs, block, ...}}}
  endpoint_state.pt  {"lm_head": {...}, "embed_tokens": {...}}  (embedding rows are E_rot = (E*s) H)
  e2e_state.pt       {"scales": {key: fp16}, "norms": {}, "args"}  (optional, --e2e)

--edge-cases injects a few sign-flipped groups (codes and scale both negated, same
function), groups with a -0.0 scale, and all-zero groups, so the exporter's scale
normalization and zero handling are exercised.

CPU only: run with CUDA_VISIBLE_DEVICES="".
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch is imported

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

SNAPSHOT = "D:/openbonsai/runs/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PREFIX = "model.language_model."
LINEAR_SUBS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
FULL_SUBS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
BLOCK = 4096
GROUP = 128


def hadamard(n):
    """Normalized Sylvester Hadamard (same construction as gptq_ternary.hadamard)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def signs_from_seed(seed, in_features):
    """Identical to gptq_ternary.signs_from_seed / e2e_kd.signs_for."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float()


class Snapshot:
    def __init__(self, root):
        self.root = Path(root)
        self.map = json.loads((self.root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        self.handles = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.root / f), framework="pt")
        return self.handles[f].get_tensor(name)


@torch.no_grad()
def rtn_ternary_rotated(w_bf16, signs, h, chunk_rows, ratio=0.7):
    """Rotate rows ((w*s) blockdiag(H)) in FP32 and RTN-ternarize per (row, 128-group)."""
    rows, in_f = w_bf16.shape
    nb = in_f // BLOCK
    codes = torch.empty(rows, in_f, dtype=torch.int8)
    scales = torch.empty(rows, in_f // GROUP, dtype=torch.float16)
    err_num, err_den = 0.0, 0.0
    for r0 in range(0, rows, chunk_rows):
        w = w_bf16[r0:r0 + chunk_rows].float() * signs
        n = w.shape[0]
        wr = (w.view(n, nb, BLOCK) @ h).view(n, in_f)
        g = wr.view(n, in_f // GROUP, GROUP)
        a = g.abs()
        mask = a > ratio * a.mean(-1, keepdim=True)
        s = (a * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        c = (torch.sign(g) * mask).to(torch.int8)
        s16 = s.half()
        codes[r0:r0 + n] = c.view(n, in_f)
        scales[r0:r0 + n] = s16
        deq = c.float() * s16.float()[..., None]
        err_num += (deq - g).pow(2).sum().item()
        err_den += g.pow(2).sum().item()
    return codes, scales, err_num / max(err_den, 1e-30)


def inject_edge_cases(codes, scales, gen, n_flip=64, n_negzero=4, n_zero=4):
    """Negate codes+scale on some groups (same function), set -0.0 scales, zero some groups."""
    rows, n_groups = scales.shape
    total = rows * n_groups
    pick = torch.randperm(total, generator=gen)[:n_flip + n_negzero + n_zero]
    cv = codes.view(rows, n_groups, GROUP)
    for k, gi in enumerate(pick.tolist()):
        r, g = divmod(gi, n_groups)
        if k < n_flip:
            cv[r, g] = -cv[r, g]
            scales[r, g] = -scales[r, g]
        elif k < n_flip + n_negzero:
            scales[r, g] = torch.tensor(-0.0, dtype=torch.float16)
        else:
            cv[r, g] = 0
            scales[r, g] = 0.0
    return {"flipped": n_flip, "neg_zero": n_negzero, "zeroed": n_zero}


def save(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--layers", type=int, default=32, help="number of decoder layers (all 32 for an exportable state)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-rows", type=int, default=8192)
    ap.add_argument("--edge-cases", action="store_true", help="inject sign-flipped / -0.0 / all-zero groups")
    ap.add_argument("--e2e", action="store_true", help="also write an e2e_state.pt with perturbed fp16 scales")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.device_count() == 0, "CPU only: CUDA devices must be hidden"
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(args.snapshot) / "config.json").read_text(encoding="utf-8"))["text_config"]
    types = cfg["layer_types"]
    snap = Snapshot(args.snapshot)
    h = hadamard(BLOCK)
    shared = {w: signs_from_seed(w, w) for w in (4096, 12288)}
    gen = torch.Generator().manual_seed(args.seed)
    t_all = time.time()
    modules, edge_log = {}, {}
    for li in range(args.layers):
        t0 = time.time()
        subs = LINEAR_SUBS if types[li] == "linear_attention" else FULL_SUBS
        for sub in subs:
            w = snap.get(f"{PREFIX}layers.{li}.{sub}.weight")
            in_f = w.shape[1]
            codes, scales, rel = rtn_ternary_rotated(w, shared[in_f], h, args.chunk_rows)
            key = f"layers.{li}.{sub}"
            if args.edge_cases:
                edge_log[key] = inject_edge_cases(codes, scales, gen)
            modules[key] = {"codes": codes, "scales": scales, "seed": in_f,
                            "signs": shared[in_f].to(torch.int8), "block": math.gcd(in_f, BLOCK),
                            "rel_mse": rel, "h_rel_err": float("nan")}
            del w
        print(f"layer {li} ({types[li]}) {time.time() - t0:.1f}s "
              + " ".join(f"{s.split('.')[-1]}={modules[f'layers.{li}.{s}']['rel_mse']:.3f}" for s in subs), flush=True)
    state_args = {"synthetic": True, "method": "rtn", "threshold_ratio": 0.7, "shared_signs": True,
                  "block": BLOCK, "group": GROUP, "layers": args.layers, "edge_cases": args.edge_cases, "seed": args.seed}
    t0 = time.time()
    save({"args": state_args, "modules": modules, "edge_cases": edge_log}, out / "ternary_state.pt")
    print(f"saved ternary_state.pt ({len(modules)} modules) in {time.time() - t0:.0f}s", flush=True)

    endpoints = {}
    for key, hf in (("lm_head", "lm_head.weight"), ("embed_tokens", f"{PREFIX}embed_tokens.weight")):
        t0 = time.time()
        w = snap.get(hf)
        codes, scales, rel = rtn_ternary_rotated(w, shared[4096], h, args.chunk_rows)
        if args.edge_cases:
            edge_log[key] = inject_edge_cases(codes, scales, gen)
        endpoints[key] = {"codes": codes, "scales": scales, "seed": 4096, "signs": shared[4096].to(torch.int8),
                          "block": BLOCK, "rel_mse": rel}
        del w
        print(f"{key} {tuple(codes.shape)} rel_mse={rel:.3f} {time.time() - t0:.1f}s", flush=True)
    save(endpoints, out / "endpoint_state.pt")

    if args.e2e:
        sc = {}
        for key, st in list(modules.items()) + list(endpoints.items()):
            mult = torch.exp(0.02 * torch.randn(st["scales"].shape, generator=gen))
            sc[key] = (st["scales"].float() * mult).half()
        save({"scales": sc, "norms": {}, "args": {"synthetic": True, "note": "scales * exp(0.02 N(0,1))"}},
             out / "e2e_state.pt")
        print(f"saved e2e_state.pt ({len(sc)} scale tensors)", flush=True)
    (out / "synthetic_info.json").write_text(json.dumps({"args": state_args, "edge_cases": edge_log,
                                                         "rel_mse": {k: v["rel_mse"] for k, v in {**modules, **endpoints}.items()},
                                                         "seconds": time.time() - t_all}, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
