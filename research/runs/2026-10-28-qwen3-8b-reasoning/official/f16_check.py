"""Checks on Prism ML's Ternary Bonsai 8B, unpacked (F16 safetensors), before eval scores it in F16.

1. Every linear layer, the head and the embedding hold ternary weights per group of 128 inputs: one value
   s or -s or 0 per group, so the F16 file stores the codes times an FP16 scale exactly. Counts the scales
   that BF16 cannot hold (what loading the checkpoint as BF16 rounds) and their largest relative error.
2. The rope settings transformers builds from the checkpoint's config (YaRN: factor 4 over 16,384).
3. A forward pass in F16 on 8 x 512 WikiText-2 test tokens: every decoder layer's output and every input to
   o_proj and down_proj must be finite (F16 overflows at 65,504; the largest magnitudes are recorded), with
   the NLL. Then the checkpoint is loaded again as BF16 (what ``--from bf16`` at the default dtype does) for
   the NLL difference and top-1 agreement on the same tokens.

    python -I f16_check.py <model snapshot dir> <wikitext-2 test parquet> <out.json>

Exit status 3 when anything is non-finite in F16 (the launcher then stops before eval).
"""
import gc
import json
import math
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP = 128
N_BLOCKS, SEQ = 8, 512
model_dir, parquet, out_path = sys.argv[1:4]
device = torch.device("cuda")
result = {"model_dir": model_dir, "torch": torch.__version__}


def load(dtype):
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map="cuda", attn_implementation="sdpa",
                                             local_files_only=True, trust_remote_code=False)
    m.eval().requires_grad_(False)
    m.config.use_cache = False
    print(f"loaded {dtype} in {time.time() - t:.0f} s", flush=True)
    return m


def ternary_check(model):
    rows = {}
    total = {"groups": 0, "non_ternary_groups": 0, "weights": 0, "zeros": 0, "scales": 0, "scales_not_bf16": 0,
             "max_rel_bf16_error": 0.0}
    for name, mod in model.named_modules():
        if not isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        w = mod.weight
        out_f, in_f = w.shape
        bad = zeros = nscale = not_bf16 = 0
        rel = 0.0
        for r0 in range(0, out_f, 16384):
            g = w[r0:r0 + 16384].view(-1, in_f // GROUP, GROUP)
            a = g.abs()
            s = a.amax(-1, keepdim=True)
            bad += int((~((a == s) | (g == 0)).all(-1)).sum())
            zeros += int((g == 0).sum())
            s = s[s > 0]
            nscale += s.numel()
            rounded = s.to(torch.bfloat16).to(s.dtype)
            not_bf16 += int((rounded != s).sum())
            if s.numel():
                rel = max(rel, float(((rounded.float() - s.float()).abs() / s.float()).max()))
            del g, a, s, rounded
        groups = out_f * in_f // GROUP
        rows[name] = {"shape": [out_f, in_f], "dtype": str(w.dtype), "non_ternary_groups": bad,
                      "zero_fraction": zeros / w.numel()}
        total["groups"] += groups
        total["non_ternary_groups"] += bad
        total["weights"] += w.numel()
        total["zeros"] += zeros
        total["scales"] += nscale
        total["scales_not_bf16"] += not_bf16
        total["max_rel_bf16_error"] = max(total["max_rel_bf16_error"], rel)
    total["zero_fraction"] = total["zeros"] / total["weights"]
    total["modules"] = len(rows)
    return total, rows


def rope_info(model):
    rot = model.model.rotary_emb
    cfg = model.config
    inv = rot.inv_freq.float().cpu()
    dim = inv.numel() * 2
    theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta") or getattr(cfg, "rope_theta", None)
    plain = 1.0 / (float(theta) ** (torch.arange(0, dim, 2).float() / dim))
    return {"rope_parameters": getattr(cfg, "rope_parameters", None), "rope_scaling": getattr(cfg, "rope_scaling", None),
            "max_position_embeddings": cfg.max_position_embeddings, "rope_type": getattr(rot, "rope_type", None),
            "attention_scaling": float(getattr(rot, "attention_scaling", 1.0)),
            "inv_freq_changed": int((~torch.isclose(inv, plain, rtol=1e-4)).sum()), "inv_freq_n": inv.numel()}


def blocks(tok):
    text = "".join(pq.read_table(parquet).column("text").to_pylist()[:1000])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= N_BLOCKS * SEQ, len(ids)
    return torch.tensor(ids[:N_BLOCKS * SEQ]).view(N_BLOCKS, SEQ)


def forward(model, x, hooks: bool):
    peaks = {}

    def record(key, t):
        t = t[0] if isinstance(t, tuple) else t
        p = peaks.setdefault(key, [0.0, True])
        p[0] = max(p[0], float(t.detach().abs().float().amax()))
        p[1] = p[1] and bool(torch.isfinite(t).all())

    handles = []
    if hooks:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(lambda m, a, o, i=i: record(f"layer{i}.out", o)))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, i=i: record(f"layer{i}.o_proj.in", a)))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda m, a, i=i: record(f"layer{i}.down_proj.in", a)))
    nll, top1, target_lp, finite_logits = [], [], [], True
    with torch.no_grad():
        for b in x:
            ids = b.unsqueeze(0).to(device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            finite_logits = finite_logits and bool(torch.isfinite(logits).all())
            lp = torch.log_softmax(logits, -1)
            t = lp.gather(-1, ids[0, 1:, None])[:, 0]
            target_lp.append(t.cpu())
            nll.append(-t.mean().item())
            top1.append(logits.argmax(-1).cpu())
            del logits, lp
    for h in handles:
        h.remove()
    return {"nll": sum(nll) / len(nll), "finite_logits": finite_logits}, torch.cat(top1), torch.cat(target_lp), peaks


tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
x = blocks(tok)
result["tokens"] = {"blocks": N_BLOCKS, "seq_len": SEQ, "scored": N_BLOCKS * (SEQ - 1),
                    "source": "the first rows of the WikiText-2 test parquet joined, consecutive blocks"}

model = load(torch.float16)
result["vocab_rows"] = model.get_input_embeddings().weight.shape[0]
result["tokenizer_len"] = len(tok)
result["tied"] = bool(model.config.tie_word_embeddings)
try:
    result["rope"] = rope_info(model)
except Exception as error:      # a config schema detail must not stop the overflow check
    result["rope"] = {"error": repr(error)}
print(f"rope: {json.dumps(result['rope'], default=str)}", flush=True)
t = time.time()
result["ternary"], result["ternary_per_module"] = ternary_check(model)
print(f"ternary check {time.time() - t:.0f} s: {json.dumps(result['ternary'])}", flush=True)
f16, top1_f16, lp_f16, peaks = forward(model, x, hooks=True)
result["f16"] = f16
result["f16_all_finite"] = all(p[1] for p in peaks.values()) and f16["finite_logits"]
for kind in ("out", "o_proj.in", "down_proj.in"):
    vals = {k: v[0] for k, v in peaks.items() if k.endswith("." + kind)}
    worst = max(vals, key=vals.get)
    result["f16"][f"max_abs_{kind}"] = {"value": vals[worst], "where": worst}
result["f16"]["peaks"] = {k: v[0] for k, v in peaks.items()}
print(json.dumps({k: v for k, v in result["f16"].items() if k != "peaks"}), flush=True)
del model
gc.collect()
torch.cuda.empty_cache()

model = load(torch.bfloat16)
bf16, top1_bf16, lp_bf16, _ = forward(model, x, hooks=False)
result["bf16"] = {**bf16, "nll_minus_f16": bf16["nll"] - f16["nll"],
                  "top1_agreement_with_f16": float((top1_bf16 == top1_f16).float().mean()),
                  "mean_abs_target_logprob_diff": float((lp_bf16 - lp_f16).abs().mean())}
result["f16"]["perplexity"] = math.exp(f16["nll"])
result["bf16"]["perplexity"] = math.exp(bf16["nll"])
print(json.dumps(result["bf16"]), flush=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1, default=str)
sys.exit(0 if result["f16_all_finite"] else 3)
