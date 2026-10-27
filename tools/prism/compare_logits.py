"""Score a PTQ1_0 GGUF with Prism's llama-perplexity and read back its full log-probabilities.

``llama-perplexity --kl-divergence-base FILE`` writes, per evaluated position,
the log-softmax over the whole vocabulary quantized to uint16 (16-nat range,
resolution about 2.4e-4). ``prism_logprobs`` runs it and returns the token
chunks it evaluated (BOS put in front of each chunk when the GGUF adds BOS) and
the log-probs of positions ``n_ctx // 2 .. n_ctx - 2``, so a caller can compare
them with the torch model the GGUF was exported from (tests/test_dense.py).

    python tools/prism/compare_logits.py --bin <prism build>/bin --gguf model.gguf --text text.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def find_tool(bin_dir, name: str = "llama-perplexity") -> Path:
    for candidate in (Path(bin_dir) / name, Path(bin_dir) / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {name} in {bin_dir}")


def read_logits(path):
    """(n_ctx, tokens [chunks, n_ctx], logprobs [chunks, n_ctx - 1 - n_ctx // 2, vocab], valid mask, first)."""
    b = Path(path).read_bytes()
    if b[:8] != b"_logits_":
        raise ValueError(f"{path}: not a llama-perplexity logits file")
    n_ctx, n_vocab, n_chunk = (int(x) for x in np.frombuffer(b, "<i4", 3, 8))
    off = 20
    tokens = np.frombuffer(b, "<i4", n_chunk * n_ctx, off).reshape(n_chunk, n_ctx)
    off += 4 * n_chunk * n_ctx
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    per = n_ctx - 1 - first
    recs = np.frombuffer(b, "<u2", n_chunk * per * nv, off).reshape(n_chunk, per, nv)
    head = recs[..., :4].copy().view("<f4").astype(np.float64)   # scale, min log-prob
    q = recs[..., 4:4 + n_vocab].astype(np.float64)
    logprobs = head[..., 1:2] + head[..., 0:1] * q
    return n_ctx, tokens.copy(), logprobs, q > 0, first


def prism_logprobs(bin_dir, gguf, text, *, n_ctx: int = 64, chunks: int = 4, threads: int = 4, out=None):
    out = Path(out) if out is not None else Path(gguf).with_suffix(".prism-logits")
    cmd = [str(find_tool(bin_dir)), "-m", str(gguf), "-f", str(text), "-c", str(n_ctx), "-b", str(n_ctx),
           "--chunks", str(chunks), "--kl-divergence-base", str(out), "-t", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"llama-perplexity failed ({r.returncode}): {r.stderr[-2000:]}")
    return read_logits(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", required=True, help="Prism build bin directory (llama-perplexity)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--text", required=True, help="plain text of at least 2 x ctx tokens")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--chunks", type=int, default=4)
    args = ap.parse_args(argv)
    n_ctx, tokens, lp, valid, first = prism_logprobs(args.bin, args.gguf, args.text, n_ctx=args.ctx,
                                                     chunks=args.chunks)
    nll = -np.take_along_axis(lp, tokens[:, first + 1:, None], -1)[..., 0]
    print(json.dumps({"chunks": int(len(tokens)), "n_ctx": n_ctx, "positions": int(nll.size),
                      "nll": float(nll.mean())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
