"""Build fixed-length calibration token blocks from on-disk sources.

Mix (by block count): WikiText-2 *train* contiguous chunks, UltraChat conversations
rendered with the model's chat template, and the original model's self-generated
(AYOT) traces. The development suite uses WikiText-2 *test* slices, ARC-Easy and
hand-written prompts, so none of these sources overlap with it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch

RUNS = Path("D:/openbonsai/runs")
WIKI_TRAIN = RUNS / "hf-cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet"
ULTRACHAT = RUNS / "2026-10-26-5090-ultrachat-attempt2/source/data/train_sft-00000-of-00003-a3ecf92756993583.parquet"
AYOT = RUNS / "2026-10-26-5090-ultrachat-attempt2/data/ayot128.jsonl"


def _pack(stream, seq_len, n_blocks):
    blocks, buf = [], []
    for ids in stream:
        buf.extend(ids)
        while len(buf) >= seq_len and len(blocks) < n_blocks:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(blocks) >= n_blocks:
            break
    return blocks


def wiki_blocks(tok, seq_len, n, seed):
    text = "".join(pq.read_table(WIKI_TRAIN).column("text").to_pylist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    off = rng.randrange(seq_len)  # non-overlapping windows at a seed-dependent phase
    starts = sorted(rng.sample(range(off, len(ids) - seq_len, seq_len), n))
    return [ids[s:s + seq_len] for s in starts]


def ultrachat_blocks(tok, seq_len, n, seed, skip_rows=6000):
    # Rows < skip_rows were the pool for the earlier AYOT prompt selection.
    rows = pq.read_table(ULTRACHAT, columns=["messages"]).column("messages").to_pylist()[skip_rows:]
    random.Random(seed).shuffle(rows)

    def gen():
        for msgs in rows:
            text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            yield tok(text, add_special_tokens=False)["input_ids"]

    return _pack(gen(), seq_len, n)


def ayot_blocks(seq_len, n):
    def gen():
        for line in AYOT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            yield [t for t, m in zip(r["input_ids"], r["attention_mask"]) if m]

    blocks = _pack(gen(), seq_len, n)
    assert len(blocks) == n, f"only {len(blocks)} AYOT blocks of {seq_len} available"
    return blocks


def build(tok, seq_len=2048, n_wiki=64, n_chat=48, n_ayot=16, seed=0):
    blocks = wiki_blocks(tok, seq_len, n_wiki, seed) + ultrachat_blocks(tok, seq_len, n_chat, seed) + ayot_blocks(seq_len, n_ayot)
    out = torch.tensor(blocks, dtype=torch.long)
    perm = torch.randperm(len(out), generator=torch.Generator().manual_seed(seed))
    return out[perm]
