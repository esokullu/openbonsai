"""Per-problem comparison of the HumanEval-gap runs against the published reasoning-mix run.

    python research/runs/2026-10-28-humaneval-gap/compare.py [base]

For each reasoning suite: the score, answers that hit the token cap, mean generated tokens, how many
finished answers are correct, and a two-sided sign test over the problems only one of the two runs
solved (base: the published run, or the run named on the command line).
"""
import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
SUITES = ("humaneval.think-off", "gsm8k.think-off", "humaneval.think-on", "gsm8k.think-on")
ROWS = [  # name, directory, file prefix
    ("published", RUNS / "2026-10-27-reasoning-mix-v1" / "eval", "distill"),
    ("published-2epoch", HERE / "reasoning-mix-2epoch" / "eval", "distill"),
    ("code-gap-2epoch", HERE / "code-gap-2epoch" / "eval", "distill"),
    ("code-gap-1epoch", HERE / "code-gap-1epoch" / "eval", "distill"),
    ("bf16", RUNS / "2026-10-27-reasoning-v1", "bf16"),
]


def load(d: Path, prefix: str, suite: str):
    p = d / f"{prefix}.{suite}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def solved(rec) -> bool:
    return bool(rec["passed"] if rec.get("passed") is not None else rec.get("correct"))


def sign_test(a, b):
    """(solved only by a, solved only by b, two-sided p)."""
    sa = {r["id"]: solved(r) for r in a["problems"]}
    sb = {r["id"]: solved(r) for r in b["problems"]}
    x = sum(sa[i] and not sb[i] for i in sa if i in sb)
    y = sum(sb[i] and not sa[i] for i in sa if i in sb)
    n, k = x + y, min(x, y)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def main(base_name: str = "published"):
    base = next(r for r in ROWS if r[0] == base_name)
    for suite in SUITES:
        print(suite)
        b = load(base[1], base[2], suite)
        for name, d, prefix in ROWS:
            j = load(d, prefix, suite)
            if j is None:
                print(f"  {name:17s} missing")
                continue
            finished = [r for r in j["problems"] if not r.get("hit_cap")]
            line = (f"  {name:17s} {j['score']:>7s}  cap {j['hit_cap']:>2}  mean tokens {j['mean_generated_tokens']:6.0f}"
                    f"  finished correct {sum(map(solved, finished))}/{len(finished)}")
            if b is not None and name != base_name:
                x, y, p = sign_test(b, j)
                line += f"  only {base_name} {x}, only this {y}, p = {p:.3f}"
            print(line)


if __name__ == "__main__":
    main(*sys.argv[1:])
