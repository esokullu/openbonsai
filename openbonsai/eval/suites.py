"""Evaluation suites scored in-process with torch.

- ``dev``: the frozen development manifest (assets/bonsai-dev/frozen-inputs.json):
  4 x 513-token WikiText-2 test NLL windows (2048 scored targets), 20 ARC-Easy
  letter choices, 4 greedy generations. FP32 logits, FP64 log-softmax.
  Reference points: BF16 Qwen3.5-9B 2.42, official Ternary Bonsai 3.192 (16/20).
- ``extended``: 16,384 WikiText-2 test tokens, 100 ARC-Easy letters and NLL on
  16 self-generated chat traces disjoint from calibration
  (assets/bonsai-dev/legacy-eval-inputs.json, chat-traces16.jsonl).
- ``compare``: the model against its own BF16 original on held-out blocks:
  NLL of both, top-1 agreement and KL(p || q) on the reference top-k plus a
  tail bucket. Works for any model and tokenizer (use it for your fine-tune).

``dev`` and ``extended`` are tokenized for Qwen/Qwen3.5-9B and refuse other
tokenizers. Their arithmetic is research/experiments/quick_eval.py's
``evaluate`` and ``extended_eval``; every input file is SHA256-pinned.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from ..core.losses import topk_kl
from .manifest import load_manifest

ASSETS = Path(__file__).resolve().parent / "assets" / "bonsai-dev"
DEV_MANIFEST = ASSETS / "frozen-inputs.json"
DEV_SHA256 = "9936a5382e9cc9921ac21ccae0ceb84fab8a9a7681d46fd0f3efbbc05e6b70be"
LEGACY_INPUTS = ASSETS / "legacy-eval-inputs.json"
LEGACY_SHA256 = "2952cf072dff2e9b00b09a59a5f2345a02b3e509725854af07ea8478f8a7a64c"
CHAT_TRACES = ASSETS / "chat-traces16.jsonl"
CHAT_SHA256 = "443aaba5e5b41c641c27cdcd45f34fd5a2e770fa466321c9bbab84436ff40aaf"
SUITES = ("dev", "extended", "compare")


class SuiteNotApplicable(ValueError):
    """The suite's frozen token IDs do not belong to this model's tokenizer."""


def _pinned_bytes(path: Path, sha: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"{path.name}: SHA256 differs from the pinned asset")
    return data


def load_dev_manifest() -> dict:
    return load_manifest(DEV_MANIFEST, expected_file_sha256=DEV_SHA256)


def load_extended_inputs() -> tuple[dict, list[dict]]:
    legacy = json.loads(_pinned_bytes(LEGACY_INPUTS, LEGACY_SHA256).decode("utf-8"))
    chat = [json.loads(line) for line in _pinned_bytes(CHAT_TRACES, CHAT_SHA256).decode("utf-8").splitlines()
            if line.strip()]
    return legacy, chat


def _vocab(model) -> int:
    return model.get_output_embeddings().out_features if hasattr(model.get_output_embeddings(), "out_features") \
        else model.config.get_text_config().vocab_size


def check_tokenizer(tokenizer, manifest: dict, model=None) -> None:
    """Refuse a model/tokenizer that cannot reproduce the manifest's frozen IDs."""
    max_id = max(max(t["input_ids"]) for t in manifest["texts"])
    if model is not None and max_id >= _vocab(model):
        raise SuiteNotApplicable(f"manifest token id {max_id} exceeds the model vocabulary; "
                                 f"it is tokenized for {manifest['tokenizer_identity'].get('id')}")
    for text in manifest["texts"][:4]:
        rendered = bytes.fromhex(text["rendered_utf8_hex"]).decode("utf-8")
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if list(ids) != text["input_ids"]:
            raise SuiteNotApplicable("this tokenizer does not reproduce the frozen token IDs; "
                                     f"the suite is tokenized for {manifest['tokenizer_identity'].get('id')}")


@torch.no_grad()
def dev(model, tokenizer, manifest: dict | None = None, *, generations: bool = True, check: bool = True) -> dict:
    manifest = load_dev_manifest() if manifest is None else manifest
    if check:
        check_tokenizer(tokenizer, manifest, model)
    dev_ = next(model.parameters()).device
    nll_sum, nll_n = 0.0, 0
    for rec in manifest["nll"]:
        ids = torch.tensor([rec["input_ids"]], device=dev_)
        pos = torch.tensor(rec["score_positions"], device=dev_)
        logits = model(input_ids=ids, use_cache=False).logits[0, pos - 1].float().double()
        tgt = ids[0, pos]
        lp = logits.log_softmax(-1).gather(-1, tgt[:, None])[:, 0]
        nll_sum += float(-lp.sum())
        nll_n += tgt.numel()
    correct, preds = 0, []
    for q in manifest["choices"]:
        scores = {}
        for opt in q["options"]:
            ids = torch.tensor([opt["input_ids"]], device=dev_)
            logits = model(input_ids=ids, use_cache=False).logits[0].float().double()
            s = 0.0
            for p in opt["score_positions"]:
                s += float(logits[p - 1].log_softmax(-1)[opt["input_ids"][p]])
            scores[opt["id"]] = s
        pred = max(scores, key=scores.get)
        preds.append(pred)
        correct += pred == q.get("answer_id")
    gens = []
    if generations:
        for g in manifest["generations"]:
            ids = torch.tensor([g["input_ids"]], device=dev_)
            out = model.generate(input_ids=ids, max_new_tokens=g["max_new_tokens"], do_sample=False,
                                 eos_token_id=g["eos_ids"], pad_token_id=g["eos_ids"][0], use_cache=True)
            gens.append(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False))
    nll = nll_sum / nll_n
    return {"suite": "dev", "nll": nll, "ppl": math.exp(nll), "scored_tokens": nll_n,
            "arc_correct": correct, "arc_total": len(manifest["choices"]), "arc_preds": "".join(preds),
            "manifest_frozen_sha256": manifest["frozen_sha256"], "generations": gens}


@torch.no_grad()
def extended(model, tokenizer=None, *, check: bool = True) -> dict:
    legacy, chat = load_extended_inputs()
    if check:
        if max(max(r["input_ids"]) for r in legacy["nll"]) >= _vocab(model):
            raise SuiteNotApplicable("extended suite is tokenized for Qwen/Qwen3.5-9B")
        if tokenizer is not None:   # same tokenizer as the dev manifest
            check_tokenizer(tokenizer, load_dev_manifest(), model)
    dev_ = next(model.parameters()).device

    def nll_of(ids, mask):
        ids_t = torch.tensor([ids], device=dev_)
        logits = model(input_ids=ids_t, use_cache=False).logits[0, :-1].float().double()
        lp = logits.log_softmax(-1).gather(-1, ids_t[0, 1:, None])[:, 0]
        m = torch.tensor(mask[1:], device=dev_, dtype=torch.bool) & torch.tensor(mask[:-1], device=dev_, dtype=torch.bool)
        return float(-lp[m].sum()), int(m.sum())

    s, n = 0.0, 0
    for rec in legacy["nll"]:
        a, b = nll_of(rec["input_ids"], rec["attention_mask"])
        s, n = s + a, n + b
    wiki = s / n
    correct = 0
    for q in legacy["questions"]:
        logits = model(input_ids=torch.tensor([q["input_ids"]], device=dev_), use_cache=False
                       ).logits[0, -1].float().double().log_softmax(-1)
        scores = {c["label"]: float(logits[c["answer_ids"][0]]) for c in q["choices"]}
        correct += max(scores, key=scores.get) == q["answer"]
    s, n = 0.0, 0
    for r in chat:
        a, b = nll_of(r["input_ids"], r["attention_mask"])
        s, n = s + a, n + b
    return {"suite": "extended", "wiki16k_nll": wiki, "arc100": correct, "chat_nll": s / n}


@torch.no_grad()
def compare_reference(model, blocks: torch.Tensor, *, topk: int = 64) -> dict:
    """BF16 reference statistics per scored position (targets 1..T-1 of each block)."""
    dev_ = next(model.parameters()).device
    nll, top1, idx, logp = [], [], [], []
    for row in blocks:
        ids = row[None].to(dev_)
        lp = model(input_ids=ids, use_cache=False).logits[0, :-1].float().double().log_softmax(-1)
        nll.append(-lp.gather(-1, ids[0, 1:, None])[:, 0].cpu())
        top1.append(lp.argmax(-1).cpu())
        v, j = lp.topk(topk, -1)
        idx.append(j.cpu())
        logp.append(v.cpu())
    return {"nll": torch.stack(nll), "top1": torch.stack(top1), "idx": torch.stack(idx), "logp": torch.stack(logp)}


@torch.no_grad()
def compare(model, blocks: torch.Tensor, reference: dict) -> dict:
    dev_ = next(model.parameters()).device
    nll, agree, kl = [], [], []
    for k, row in enumerate(blocks):
        ids = row[None].to(dev_)
        lp = model(input_ids=ids, use_cache=False).logits[0, :-1].float().double().log_softmax(-1)
        nll.append(-lp.gather(-1, ids[0, 1:, None])[:, 0].cpu())
        agree.append((lp.argmax(-1).cpu() == reference["top1"][k]).double())
        kl.append(topk_kl(reference["idx"][k].to(dev_), reference["logp"][k].to(dev_), lp).cpu())
    nll, agree, kl = torch.cat(nll), torch.cat(agree), torch.cat(kl)
    ref_nll = reference["nll"].reshape(-1)
    return {"suite": "compare", "tokens": int(nll.numel()), "nll": float(nll.mean()), "ref_nll": float(ref_nll.mean()),
            "delta_nll": float(nll.mean() - ref_nll.mean()), "top1_agreement": float(agree.mean()),
            "kl_topk": float(kl.mean())}
