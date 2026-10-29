"""Apply the pre-registered gate (PREREGISTRATION.md) to the confirming-eval results. Read-only apart from gate.json.

Usage: python -I -X utf8 gate.py   (from any directory; missing result files are reported, not guessed)
"""
import json, math
from pathlib import Path

C = Path("D:/openbonsai/runs/code-gap/confirm")
STORED = {"P": Path("D:/openbonsai/runs/2026-10-27-reasoning-mix-v1/eval"),
          "A1": Path("D:/openbonsai/runs/2026-10-28-code-gap-1epoch/eval")}
he_ids = [p["id"] for p in json.load(open(C / "manifests/humaneval-164.json", encoding="utf-8"))["problems"]]
gs_ids = [p["id"] for p in json.load(open(C / "manifests/gsm8k-1319.json", encoding="utf-8"))["problems"]]
flags = json.load(open(C / "screen/FROZEN_FLAGS.json", encoding="utf-8"))
flagged = set(flags["flagged"])                      # same or related, any scope
a1_only = set(flags["flagged_A1_only"])              # evidence only in traces P never saw


def load(model, name):
    f = C / f"results/{model}.{name}.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


def ok(rec):
    return bool(rec.get("passed", rec.get("correct")))


def by_id(res):
    return {r["id"]: r for r in res["problems"]}


def sign_p(b, c):
    """One-sided exact sign test: P(X >= b), X ~ Bin(b + c, 1/2)."""
    n = b + c
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n if n else 1.0


def paired(ids, a, p):
    b = sum(1 for i in ids if ok(a[i]) and not ok(p[i]))
    c = sum(1 for i in ids if ok(p[i]) and not ok(a[i]))
    n = len(ids)
    d = (b - c) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0)) / n
    return {"n": n, "A1": sum(ok(a[i]) for i in ids), "P": sum(ok(p[i]) for i in ids), "only_A1": b, "only_P": c,
            "sign_p_one_sided": round(sign_p(b, c), 6), "diff": round(d, 4),
            "ci95": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)]}


def prefix_check(model, name, suite, mode, ids, rebatched):
    """Frozen-prefix reproduction vs the stored record of the original run dir."""
    res = load(model, name)
    stored = json.load(open(STORED[model] / f"distill.{suite}.think-{mode}.json", encoding="utf-8"))
    new, old = by_id(res), by_id(stored)
    diff = [i for i in ids if ok(new[i]) != ok(old[i])]
    return {"model": model, "result": name, "n": len(ids), "stored_score": stored["score"],
            "new_on_prefix": sum(ok(new[i]) for i in ids), "differs": diff,
            "differs_outside_rebatched": [i for i in diff if i not in rebatched],
            "cached_batches": f"{res['cached_batches']}/{res['batches']}"}


out = {"checks": [], "gate": {}}
R = {(m, n): load(m, n) for m in ("P", "A1") for n in ("humaneval164.think-off", "humaneval75.think-on", "gsm8k1319.think-off")}
out["missing"] = [f"{m}.{n}" for (m, n), v in R.items() if v is None]

if R["P", "humaneval164.think-off"] and R["A1", "humaneval164.think-off"]:
    p, a = by_id(R["P", "humaneval164.think-off"]), by_id(R["A1", "humaneval164.think-off"])
    fresh = he_ids[50:164]
    g = paired(fresh, a, p)
    unfl = [i for i in fresh if i not in flagged]
    gu = paired(unfl, a, p)
    no_a1only = paired([i for i in fresh if i not in a1_only], a, p)
    g_pass = g["sign_p_one_sided"] < 0.05 and gu["A1"] > gu["P"]
    out["gate"]["a"] = {"fresh_51_164": g, "unflagged": gu, "excluding_A1_only_flags": no_a1only,
                        "pass": g_pass, "nominally_ahead": g["only_A1"] > g["only_P"]}
    out["pooled_humaneval164_includes_selection_set"] = paired(he_ids[:164], a, p)
    for m in ("P", "A1"):
        out["checks"].append(prefix_check(m, "humaneval164.think-off", "humaneval", "off", he_ids[:50], set(he_ids[48:50])))

if R["P", "gsm8k1319.think-off"] and R["A1", "gsm8k1319.think-off"]:
    p, a = by_id(R["P", "gsm8k1319.think-off"]), by_id(R["A1", "gsm8k1319.think-off"])
    g = paired(gs_ids[100:1319], a, p)
    g["hit_cap"] = {"A1": sum(a[i]["hit_cap"] for i in gs_ids[100:]), "P": sum(p[i]["hit_cap"] for i in gs_ids[100:])}
    out["gate"]["b"] = {"fresh_101_1319": g, "pass": g["diff"] >= -0.03}
    for m in ("P", "A1"):
        out["checks"].append(prefix_check(m, "gsm8k1319.think-off", "gsm8k", "off", gs_ids[:100], set(gs_ids[96:100])))

if R["P", "humaneval75.think-on"] and R["A1", "humaneval75.think-on"]:
    p, a = by_id(R["P", "humaneval75.think-on"]), by_id(R["A1", "humaneval75.think-on"])
    ids = he_ids[25:75]
    g = paired(ids, a, p)
    for m, d in (("A1", a), ("P", p)):
        g[f"hit_cap_{m}"] = sum(d[i]["hit_cap"] for i in ids)
        g[f"unfinished_{m}"] = sum(not d[i].get("finished_thinking", True) for i in ids)
    out["gate"]["c"] = {"fresh_on_26_75": g, "pass": g["A1"] >= g["P"] - 2}
    for m in ("P", "A1"):
        out["checks"].append(prefix_check(m, "humaneval75.think-on", "humaneval", "on", he_ids[:25], {he_ids[24]}))

gt = out["gate"]
if all(k in gt for k in "abc"):
    a_, b_, c_ = gt["a"]["pass"], gt["b"]["pass"], gt["c"]["pass"]
    if a_ and b_ and c_:
        out["outcome"] = "all pass: recommend replacing the HF GGUF with A1 (tag 40964de first; user's decision)"
    elif a_:
        out["outcome"] = "(a) passes, (b) or (c) fails: bring the trade-off to the user"
    elif gt["a"]["nominally_ahead"]:
        out["outcome"] = "(a) only nominally ahead: keep P, offer A1 as a second file"
    else:
        out["outcome"] = "A1 not ahead on fresh HumanEval: keep P"
json.dump(out, open(C / "gate.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
