"""Prism's runtime against PyTorch on the same real weights: the exported Qwen3-8B PTQ1_0 GGUF.

Prism's llama-perplexity (CPU build) writes its full-vocabulary log-probs (``--kl-divergence-base``) for
WikiText-2 test chunks; the FP32 torch model with the distilled state's dequantized weights scores the
same token chunks on the CPU (as tests/test_dense.py does for tiny models). Reported over the scored
positions (``n_ctx // 2 .. n_ctx - 2`` of each chunk): KL(torch || Prism) over the whole vocabulary,
top-1 agreement, |log-prob difference| where Prism's uint16 log-probs are in range, and each side's
perplexity of the actual next tokens.

Prism stores every log-prob more than 16 nats below the top one as that floor, so its decoded log-probs
sum to slightly more than one and the plain KL can come out a little below zero. Two estimates bracket
the KL instead: ``kl_lumped_*`` merges the floored tokens of each position into one bucket (Prism's mass
there is one minus its stored mass), a lower bound up to the uint16 rounding, since merging categories
cannot raise a KL; ``kl_renorm_*`` renormalizes Prism's decoded log-probs, which gives the floored tokens
more than their mass and so runs high. ``kl_prism_torch_renorm_*`` is KL(Prism || torch) with Prism
renormalized, the estimator of the 9B check (research/runs/gguf-test/runtime-parity/ref_vs_llama.py).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo> python runtime_parity.py --run <run dir> \
        --bin <prism build>/bin --text wiki.test.raw --out runtime_parity.json [--reuse]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from openbonsai.config import DEFAULTS, check_keys, deep_merge
from openbonsai.rundir import RunDir
from openbonsai.stages.common import StageContext, apply_dense

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="read Prism's log-probs from an earlier run's file")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("compare_logits", ROOT / "tools" / "prism" / "compare_logits.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    run = RunDir(args.run)
    ctx = StageContext(check_keys(deep_merge(DEFAULTS, run.load_config())), run)
    gguf = Path(json.loads((run.root / "export" / "DONE.json").read_text(encoding="utf-8"))["result"]["output"])
    t0 = time.time()
    logits = Path(args.out).with_suffix(".prism-logits")
    if args.reuse and logits.is_file():
        n_ctx, tokens, lp, valid, first = tool.read_logits(logits)
        old = Path(args.out)
        prism_s = json.loads(old.read_text(encoding="utf-8")).get("prism_seconds") if old.is_file() else None
    else:
        n_ctx, tokens, lp, valid, first = tool.prism_logprobs(args.bin, gguf, args.text, n_ctx=args.n_ctx,
                                                              chunks=args.chunks, threads=args.threads,
                                                              out=logits)
        prism_s = time.time() - t0
    adapter = ctx.adapter()
    m = ctx.cfg["model"]
    ref, tok = adapter.load(ctx.model_path(), revision=m.get("revision"), dtype=torch.float32, device="cpu",
                            local_files_only=bool(m.get("local_files_only")))
    apply_dense(adapter, ref, ctx.state_through("distill").records())
    kl, kl_norm, kl_lump, kl_rev, agree, absd, nll_t, nll_p, n = [], [], [], [], 0, [], [], [], 0
    t0 = time.time()
    for c in range(len(tokens)):
        ids = torch.from_numpy(tokens[c].astype(np.int64))
        if adapter.add_bos:
            ids[0] = tok.bos_token_id
        with torch.no_grad():
            t = ref(input_ids=ids[None], use_cache=False).logits[0].double().log_softmax(-1).numpy()[first:n_ctx - 1]
        p = lp[c][:, :t.shape[1]]
        pt = np.exp(t)
        kl.append((pt * (t - p)).sum(-1))
        top = p.max(-1, keepdims=True)
        pn = p - top - np.log(np.exp(p - top).sum(-1, keepdims=True))
        kl_norm.append((pt * (t - pn)).sum(-1))
        kl_rev.append((np.exp(pn) * (pn - t)).sum(-1))
        kept = valid[c][:, :t.shape[1]]
        r_t = np.where(kept, 0.0, pt).sum(-1)
        r_p = np.clip(1.0 - np.where(kept, np.exp(p), 0.0).sum(-1), 1e-30, None)
        kl_lump.append(np.where(kept, pt * (t - p), 0.0).sum(-1) + r_t * np.log(np.clip(r_t, 1e-300, None) / r_p))
        agree += int((t.argmax(-1) == p.argmax(-1)).sum())
        absd.append(np.abs(t - p)[valid[c][:, :t.shape[1]]])
        nxt = tokens[c][first + 1:n_ctx]
        nll_t.append(-t[np.arange(len(nxt)), nxt])
        nll_p.append(-p[np.arange(len(nxt)), nxt])
        n += len(nxt)
    kl, kl_norm, kl_lump, kl_rev, absd, nll_t, nll_p = (np.concatenate(x) for x in (kl, kl_norm, kl_lump, kl_rev,
                                                                                     absd, nll_t, nll_p))
    out = {"gguf": str(gguf), "text": str(args.text), "n_ctx": n_ctx, "chunks": int(len(tokens)), "positions": n,
           "kl_torch_prism_mean": float(kl.mean()), "kl_torch_prism_max": float(kl.max()),
           "kl_lumped_mean": float(kl_lump.mean()), "kl_lumped_max": float(kl_lump.max()),
           "kl_renorm_mean": float(kl_norm.mean()), "kl_renorm_max": float(kl_norm.max()),
           "kl_prism_torch_renorm_mean": float(kl_rev.mean()), "kl_prism_torch_renorm_max": float(kl_rev.max()),
           "top1_agreement": agree / n, "abs_logprob_diff_mean": float(absd.mean()),
           "abs_logprob_diff_max": float(absd.max()), "ppl_torch": float(np.exp(nll_t.mean())),
           "ppl_prism": float(np.exp(nll_p.mean())), "prism_seconds": prism_s and round(prism_s, 1),
           "torch_seconds": round(time.time() - t0, 1),
           "torch": "FP32 on the CPU, distill state's dequantized weights (apply_dense)",
           "prism": "llama-perplexity --kl-divergence-base, uint16 log-probs (16-nat range)"}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
