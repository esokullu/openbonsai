"""Fast sanity evaluation of BF16 and uncalibrated (fake-quant) ternary variants.

Scores the frozen development suite (4x513 WikiText NLL, 20 ARC letters, 4 greedy
generations) on a native Qwen3.5-9B. Ternary variants are simulated in place by
replacing decoder linear weights with dequantized ternary values, so this measures
quantization error only, not packing or runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]  # repo root; this file lives in research/experiments/
SNAPSHOT = Path("D:/openbonsai/runs/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
FROZEN = ROOT / "openbonsai/eval/assets/bonsai-dev/frozen-inputs.json"


def load_model(device="cuda"):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(SNAPSHOT, dtype=torch.bfloat16, device_map=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    return model, tok


def decoder_linears(model):
    layers = model.model.language_model.layers
    out = {}
    for i, layer in enumerate(layers):
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                out[f"layers.{i}.{name}"] = mod
    return out


@torch.no_grad()
def evaluate(model, tok, frozen, gen=True):
    dev = next(model.parameters()).device
    nll_sum, nll_n = 0.0, 0
    for rec in frozen["nll"]:
        ids = torch.tensor([rec["input_ids"]], device=dev)
        pos = torch.tensor(rec["score_positions"], device=dev)
        logits = model(input_ids=ids).logits[0, pos - 1].float().double()
        tgt = ids[0, pos]
        lp = logits.log_softmax(-1).gather(-1, tgt[:, None])[:, 0]
        nll_sum += float(-lp.sum())
        nll_n += tgt.numel()
    correct = 0
    preds = []
    for q in frozen["choices"]:
        scores = {}
        for opt in q["options"]:
            ids = torch.tensor([opt["input_ids"]], device=dev)
            logits = model(input_ids=ids).logits[0].float().double()
            s = 0.0
            for p in opt["score_positions"]:
                s += float(logits[p - 1].log_softmax(-1)[opt["input_ids"][p]])
            scores[opt["id"]] = s
        pred = max(scores, key=scores.get)
        preds.append(pred)
        correct += pred == q["answer_id"]
    gens = []
    if gen:
        for g in frozen["generations"]:
            ids = torch.tensor([g["input_ids"]], device=dev)
            out = model.generate(input_ids=ids, max_new_tokens=g["max_new_tokens"], do_sample=False,
                                 eos_token_id=g["eos_ids"], pad_token_id=g["eos_ids"][0])
            gens.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False))
    nll = nll_sum / nll_n
    return {"nll": nll, "ppl": math.exp(nll), "arc_correct": correct, "arc_preds": "".join(preds), "generations": gens}


LEGACY = Path("D:/openbonsai/runs/2026-10-26-5090-pilot/eval-inputs.json")
CHAT = Path("D:/openbonsai/runs/2026-10-26-5090-activation-diagnostic-v1/data/search-consumed16.jsonl")


@torch.no_grad()
def extended_eval(model):
    """Larger held-out checks: 16,384 WikiText-2 test tokens, 100 ARC-Easy letters,
    and NLL on 16 self-generated chat traces disjoint from all calibration data."""
    dev = next(model.parameters()).device
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))

    def nll_of(ids, mask):
        ids_t = torch.tensor([ids], device=dev)
        logits = model(input_ids=ids_t).logits[0, :-1].float().double()
        lp = logits.log_softmax(-1).gather(-1, ids_t[0, 1:, None])[:, 0]
        m = torch.tensor(mask[1:], device=dev, dtype=torch.bool) & torch.tensor(mask[:-1], device=dev, dtype=torch.bool)
        return float(-lp[m].sum()), int(m.sum())

    s, n = 0.0, 0
    for rec in legacy["nll"]:
        a, b = nll_of(rec["input_ids"], rec["attention_mask"])
        s, n = s + a, n + b
    wiki = s / n
    correct = 0
    for q in legacy["questions"]:
        logits = model(input_ids=torch.tensor([q["input_ids"]], device=dev)).logits[0, -1].float().double().log_softmax(-1)
        scores = {c["label"]: float(logits[c["answer_ids"][0]]) for c in q["choices"]}
        correct += max(scores, key=scores.get) == q["answer"]
    s, n = 0.0, 0
    for line in CHAT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        a, b = nll_of(r["input_ids"], r["attention_mask"])
        s, n = s + a, n + b
    return {"wiki16k_nll": wiki, "arc100": correct, "chat_nll": s / n}


def hadamard(n, device):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def block_rotation(in_features, block, seed, device):
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).float().to(device)
    return signs, hadamard(block, device)


def rotate_in(w, signs, h, block):
    # W R with R = diag(signs) blockdiag(H)
    o, i = w.shape
    return ((w * signs).view(o, i // block, block) @ h).view(o, i)


def unrotate_in(w, signs, h, block):
    o, i = w.shape
    return ((w.view(o, i // block, block) @ h.T).view(o, i)) * signs


def ternary(w, group, method):
    o, i = w.shape
    g = w.view(o, i // group, group)
    a = g.abs()
    if method == "twn":
        delta = 0.7 * a.mean(-1, keepdim=True)
        mask = a > delta
        alpha = (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        q = torch.sign(g) * mask * alpha
    elif method == "absmean":
        s = a.mean(-1, keepdim=True).clamp_min(1e-8)
        q = (g / s).round().clamp(-1, 1) * s
    elif method == "lsq":
        # per-group threshold search minimizing L2 with optimal scale
        best = None
        for r in torch.linspace(0.3, 1.2, 10):
            delta = r * a.mean(-1, keepdim=True)
            mask = a > delta
            alpha = (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
            q = torch.sign(g) * mask * alpha
            err = (q - g).pow(2).sum(-1, keepdim=True)
            if best is None:
                best, best_err = q, err
            else:
                better = err < best_err
                best = torch.where(better, q, best)
                best_err = torch.minimum(err, best_err)
        q = best
    else:
        raise ValueError(method)
    return q.view(o, i)


def kind_of(name):
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name:
        return "attn"
    if ".linear_attn." in name:
        return "linattn"
    return "other"


@torch.no_grad()
def apply_variant(linears, originals, variant):
    method = variant.get("method")
    group = variant.get("group", 128)
    rotate = variant.get("rotate", False)
    kinds = set(variant.get("kinds", ["mlp", "attn", "linattn"]))
    layers = variant.get("layers")
    block = variant.get("block", 4096)
    exclude = variant.get("exclude", [])
    only = variant.get("only")
    for idx, (name, mod) in enumerate(linears.items()):
        w0 = originals[name].to(mod.weight.device, non_blocking=True)
        layer_idx = int(name.split(".")[1])
        skip = any(e in name for e in exclude) or (only is not None and not any(o in name for o in only))
        if method is None or skip or kind_of(name) not in kinds or (layers is not None and layer_idx not in layers):
            mod.weight.copy_(w0)
            continue
        w = w0.float()
        if rotate:
            b = math.gcd(w.shape[1], block)
            signs, h = block_rotation(w.shape[1], b, 5090 + idx, w.device)
            q = unrotate_in(ternary(rotate_in(w, signs, h, b), group, method), signs, h, b)
        else:
            q = ternary(w, group, method)
        only_rows = variant.get("only_rows")
        if only_rows is not None:
            sel = torch.zeros(w.shape[0], dtype=torch.bool, device=w.device)
            sel[only_rows] = True
            q[~sel] = w[~sel]
        keep = variant.get("keep_rows")
        if keep:
            rn = w.norm(dim=1)
            rows = rn > keep * rn.median()
            q[rows] = w[rows]
            variant.setdefault("_kept", 0)
            variant["_kept"] += int(rows.sum())
        mod.weight.copy_(q.to(mod.weight.dtype))


VARIANTS = {
    "bf16": {},
    "twn": {"method": "twn"},
    "absmean": {"method": "absmean"},
    "lsq": {"method": "lsq"},
    "had_twn": {"method": "twn", "rotate": True},
    "had_lsq": {"method": "lsq", "rotate": True},
    "had_lsq_mlp": {"method": "lsq", "rotate": True, "kinds": ["mlp"]},
    "had_lsq_attn": {"method": "lsq", "rotate": True, "kinds": ["attn"]},
    "had_lsq_linattn": {"method": "lsq", "rotate": True, "kinds": ["linattn"]},
    "had_lsq_first8": {"method": "lsq", "rotate": True, "layers": list(range(8))},
    "had_lsq_last8": {"method": "lsq", "rotate": True, "layers": list(range(24, 32))},
    "had_lsq_noab": {"method": "lsq", "rotate": True, "exclude": ["in_proj_a", "in_proj_b"]},
    "lsq_noab": {"method": "lsq", "exclude": ["in_proj_a", "in_proj_b"]},
    "had_lsq_ab": {"method": "lsq", "rotate": True, "only": ["in_proj_a", "in_proj_b"]},
    "had_lsq_qkv": {"method": "lsq", "rotate": True, "only": ["in_proj_qkv"]},
    "had_lsq_z": {"method": "lsq", "rotate": True, "only": ["in_proj_z"]},
    "had_lsq_out": {"method": "lsq", "rotate": True, "only": ["linear_attn.out_proj"]},
    "had_lsq_down": {"method": "lsq", "rotate": True, "only": ["down_proj"]},
    "had_lsq_gateup": {"method": "lsq", "rotate": True, "only": ["gate_proj", "up_proj"]},
    "had_lsq_noab_g64": {"method": "lsq", "rotate": True, "group": 64, "exclude": ["in_proj_a", "in_proj_b"]},
    "had_lsq_noab_l0": {"method": "lsq", "rotate": True, "exclude": ["in_proj_a", "in_proj_b"], "layers": [0]},
    "had_lsq_noab_l1_31": {"method": "lsq", "rotate": True, "exclude": ["in_proj_a", "in_proj_b"], "layers": list(range(1, 32))},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bf16,twn,had_twn,had_lsq")
    ap.add_argument("--out", default=str(ROOT / "experiments/out/quick_eval.jsonl"))
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--state", default="", help="comma-separated ternary_state.pt paths to evaluate")
    args = ap.parse_args()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    t = time.time()
    model, tok = load_model()
    print(f"loaded in {time.time()-t:.1f}s", flush=True)
    linears = decoder_linears(model)
    print(f"{len(linears)} decoder linears; kinds:", {k: sum(kind_of(n) == k for n in linears) for k in ["mlp", "attn", "linattn", "other"]}, flush=True)
    print([n for n in linears if n.startswith("layers.0.") or n.startswith("layers.3.")], flush=True)
    originals = {n: m.weight.detach().to("cpu", copy=True).pin_memory() for n, m in linears.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.state:
        from gptq_ternary import apply_state
        for path in args.state.split(","):
            t = time.time()
            apply_state(model, torch.load(path, map_location="cpu"))
            res = evaluate(model, tok, frozen, gen=not args.no_gen)
            res.update(extended_eval(model))
            res = {"variant": path, **res, "seconds": time.time() - t}
            print(json.dumps({k: v for k, v in res.items() if k != "generations"}), flush=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
        return
    for name in args.variants.split(","):
        t = time.time()
        if name.startswith("only="):
            spec = {"method": "lsq", "rotate": True, "only": name[5:].split("+")}
        elif name.startswith("keep="):
            spec = {"method": "lsq", "rotate": True, "exclude": ["in_proj_a", "in_proj_b"], "keep_rows": float(name[5:])}
        elif name.startswith("l0keep="):
            spec = {"method": "lsq", "rotate": True, "only": ["layers.0.linear_attn.out_proj"], "keep_rows": float(name[7:])}
        elif name.startswith("l0row="):
            spec = {"method": "lsq", "rotate": True, "only": ["layers.0.linear_attn.out_proj"], "only_rows": [int(x) for x in name[6:].split("+")]}
        elif name.startswith("keepnorot="):
            spec = {"method": "lsq", "exclude": ["in_proj_a", "in_proj_b"], "keep_rows": float(name[10:])}
        elif name.startswith("excl="):
            spec = {"method": "lsq", "rotate": True, "exclude": name[5:].split("+")}
        else:
            spec = VARIANTS[name]
        apply_variant(linears, originals, spec)
        torch.cuda.synchronize()
        res = evaluate(model, tok, frozen, gen=not args.no_gen)
        if args.extended:
            res.update(extended_eval(model))
        res = {"variant": name, **res, "seconds": time.time() - t, "kept_rows": spec.get("_kept")}
        print(json.dumps({k: v for k, v in res.items() if k != "generations"}), flush=True)
        for g in res["generations"]:
            print("   GEN:", g[:300].replace("\n", " | "), flush=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
