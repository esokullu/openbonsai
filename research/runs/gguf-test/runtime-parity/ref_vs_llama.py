"""End-to-end check: Prism llama.cpp logits (llama-perplexity --kl-divergence-base dump) vs a PyTorch fp32 CPU
reference of the SAME synthetic ternary model, computed from the OpenBonsai state (not from the GGUF):
  ternary linears  y = (C*s) @ H(sig * x)   (codes/scales from ternary_state/endpoint_state, e2e scales if given)
  embedding        e = sig * (H z),  z = C[id]*s[id]
  everything else  original snapshot tensors (HF layout, HF modules: GatedDeltaNet, gated attention, mRoPE)
--mode orig runs the original BF16 weights (fp32 compute) on the same tokens as a quality baseline.
Streams one decoder layer at a time. CPU only.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

SNAP = "D:/openbonsai/runs/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HF = "model.language_model."
TERN_SUBS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj",
             "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
B = 4096


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).float()


H = hadamard(B)


def rot_act(x, sig):
    """H(sig*x) per 4096 block along the last dim (H symmetric)."""
    shp = x.shape
    return ((x * sig).reshape(*shp[:-1], shp[-1] // B, B) @ H).reshape(shp)


class FQLinear(torch.nn.Module):
    def __init__(self, codes, scales, sig):
        super().__init__()
        o, i = codes.shape
        self.w = (codes.float().view(o, -1, 128) * scales.float()[..., None]).view(o, i)
        self.sig = sig.float()

    def forward(self, x):
        return torch.nn.functional.linear(rot_act(x, self.sig), self.w)


class Snap:
    def __init__(self):
        self.map = json.loads(Path(f"{SNAP}/model.safetensors.index.json").read_text())["weight_map"]
        self.h = {}

    def get(self, n):
        f = self.map[n]
        if f not in self.h:
            self.h[f] = safe_open(f"{SNAP}/{f}", framework="pt")
        return self.h[f].get_tensor(n)


def read_dump(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == b"_logits_"
    n_ctx, n_vocab, n_chunk = np.frombuffer(raw, np.int32, 3, 8)
    off = 20
    toks = np.frombuffer(raw, np.int32, n_chunk * n_ctx, off).reshape(n_chunk, n_ctx)
    off += 4 * n_chunk * n_ctx
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    n_sc = n_ctx - 1 - first
    rec = np.frombuffer(raw, np.uint16, n_chunk * n_sc * nv, off).reshape(n_chunk, n_sc, nv)
    assert off + rec.nbytes == len(raw), (off + rec.nbytes, len(raw))
    hdr = rec[..., :4].copy().view(np.float32)  # [..., 2]: scale, min_log_prob
    q = rec[..., 4:4 + n_vocab]
    return int(n_ctx), int(n_vocab), int(n_chunk), toks, hdr, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--mode", choices=["fq", "orig"], default="fq")
    ap.add_argument("--state-dir", default="D:/openbonsai/runs/gguf-test/synthetic")
    ap.add_argument("--e2e", default="", help="e2e_state.pt (scales override), must match the GGUF")

    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    assert torch.cuda.device_count() == 0
    torch.set_num_threads(args.threads)
    t_all = time.time()
    n_ctx, n_vocab, n_chunk, toks, hdr, q = read_dump(args.dump)
    first = n_ctx // 2
    print(f"dump: n_ctx {n_ctx}, n_vocab {n_vocab}, n_chunk {n_chunk}", flush=True)
    cfg = AutoConfig.from_pretrained(SNAP).text_config
    cfg._attn_implementation = "sdpa"
    snap = Snap()
    if args.mode == "fq":
        st = torch.load(f"{args.state_dir}/ternary_state.pt", map_location="cpu", mmap=True, weights_only=True)["modules"]
        ep = torch.load(f"{args.state_dir}/endpoint_state.pt", map_location="cpu", mmap=True, weights_only=True)
        e2e = torch.load(args.e2e, map_location="cpu", mmap=True, weights_only=True)["scales"] if args.e2e else {}

        def entry(k):
            return ep[k] if k in ("lm_head", "embed_tokens") else st[k]

        def scales(k):
            return e2e[k] if k in e2e else entry(k)["scales"]

        def signs(k):
            return entry(k)["signs"].float()

    ids = torch.from_numpy(toks.astype(np.int64))
    with torch.no_grad():
        # embedding
        if args.mode == "fq":
            c = entry("embed_tokens")["codes"][ids.reshape(-1)].float()
            s = scales("embed_tokens")[ids.reshape(-1)].float()
            z = (c.view(c.shape[0], -1, 128) * s[..., None]).view(c.shape[0], -1)
            emb = rot_act(z, torch.ones(z.shape[-1])) * signs("embed_tokens")  # s * (H z)
        else:
            emb = snap.get(HF + "embed_tokens.weight")[ids.reshape(-1)].float()
        hs = emb.view(n_chunk, n_ctx, -1)
        pos = torch.arange(n_ctx).view(1, 1, -1).expand(4, n_chunk, -1)
        text_pos, mpos = pos[0], pos[1:]
        mk = {"config": cfg, "inputs_embeds": hs, "attention_mask": None, "past_key_values": None, "position_ids": text_pos}
        masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
        pe = M.Qwen3_5TextRotaryEmbedding(config=cfg)(hs, mpos)
        for li in range(cfg.num_hidden_layers):
            t0 = time.time()
            with torch.device("meta"):
                layer = M.Qwen3_5DecoderLayer(cfg, li)
            layer = layer.to_empty(device="cpu")
            assert not list(layer.buffers()), "unexpected buffers"
            sd = {n: snap.get(f"{HF}layers.{li}.{n}").float() for n, _ in layer.named_parameters()}
            layer.load_state_dict(sd, strict=True)
            if args.mode == "fq":
                for sub in TERN_SUBS:
                    key = f"layers.{li}.{sub}"
                    if key not in st:
                        continue
                    parent, attr = sub.rsplit(".", 1)
                    setattr(layer.get_submodule(parent), attr, FQLinear(st[key]["codes"], scales(key), signs(key)))
            layer.eval()
            hs = layer(hs, position_embeddings=pe, attention_mask=masks[cfg.layer_types[li]], position_ids=text_pos,
                       past_key_values=None, use_cache=False)
            del layer, sd
            print(f"layer {li} {cfg.layer_types[li]} {time.time() - t0:.1f}s |h| {hs.norm(dim=-1).mean():.2f}", flush=True)
        norm = M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.weight.data = snap.get(HF + "norm.weight").float()
        hs = norm(hs)[:, first:n_ctx - 1]  # scored positions
        x = hs.reshape(-1, hs.shape[-1])
        if args.mode == "fq":
            x = rot_act(x, signs("lm_head"))
            hc, hsc = entry("lm_head")["codes"], scales("lm_head")
        else:
            hw = snap.get("lm_head.weight")
        logits = torch.empty(x.shape[0], n_vocab)
        for r0 in range(0, n_vocab, 16384):
            r1 = min(n_vocab, r0 + 16384)
            if args.mode == "fq":
                w = (hc[r0:r1].float().view(r1 - r0, -1, 128) * hsc[r0:r1].float()[..., None]).view(r1 - r0, -1)
            else:
                w = hw[r0:r1].float()
            logits[:, r0:r1] = x @ w.T
    lp_ref = torch.log_softmax(logits.double(), -1).view(n_chunk, n_ctx - 1 - first, n_vocab)
    tgt = torch.from_numpy(toks[:, first + 1:].astype(np.int64))
    nll_ref = -lp_ref.gather(-1, tgt[..., None])[..., 0]
    # llama log-probs from the dump: lp = min_log_prob + scale*q (q = 0: at or below the 16-nat window)
    scale = torch.from_numpy(hdr[..., 0].astype(np.float64))[..., None]
    minlp = torch.from_numpy(hdr[..., 1].astype(np.float64))[..., None]
    lp_l = minlp + scale * torch.from_numpy(q.astype(np.float64))
    nll_l = -lp_l.gather(-1, tgt[..., None])[..., 0]
    p_l = lp_l.exp()
    p_l = p_l / p_l.sum(-1, keepdim=True)
    kl = (p_l * (p_l.clamp_min(1e-300).log() - lp_ref)).sum(-1)  # KL(llama || ref)
    top1_l = lp_l.argmax(-1)
    top1_r = lp_ref.argmax(-1)
    agree = (top1_l == top1_r).double().mean().item()
    corr = float(np.corrcoef(nll_l.reshape(-1).numpy(), nll_ref.reshape(-1).numpy())[0, 1])
    res = {"mode": args.mode, "e2e": args.e2e, "dump": args.dump, "n_ctx": n_ctx, "n_chunk": n_chunk,
           "scored_tokens": int(nll_ref.numel()),
           "ppl_ref": math.exp(nll_ref.mean().item()), "ppl_llama_from_dump": math.exp(nll_l.mean().item()),
           "ppl_ref_per_chunk_cumulative": [math.exp(nll_ref[:k + 1].mean().item()) for k in range(n_chunk)],
           "mean_kl_llama_vs_ref": kl.mean().item(), "max_kl": kl.max().item(),
           "top1_agreement": agree, "nll_corr": corr,
           "mean_abs_nll_diff": (nll_l - nll_ref).abs().mean().item(), "seconds": time.time() - t_all}
    print(json.dumps(res, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
