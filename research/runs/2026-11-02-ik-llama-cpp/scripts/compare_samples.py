"""Compare ik_llama.cpp's greedy server replies with Prism's llama-cli single-turn replies.

Usage: python -I -X utf8 compare_samples.py RESULTS_DIR
"""
import json
import os
import re
import sys

res = sys.argv[1]
ik = {r["name"]: r for r in json.load(open(os.path.join(res, "samples_ik_rtr.json"), encoding="utf-8"))}
for name in ("arith", "word_problem", "code"):
    raw = open(os.path.join(res, f"prism_cli_{name}.out"), encoding="utf-8", errors="replace").read()
    # the reply sits between the echoed "> question" line and the "[ Prompt: ... ]" timing line
    m = re.search(r"\n> [^\n]*\n(.*?)\n\n\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]", raw, re.S)
    prism = m.group(1) if m else None
    a = (ik[name]["content"] or "").strip()
    b = (prism or "").strip()
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    print(f"== {name}: ik {len(a)} chars ({ik[name]['tokens_predicted']} tokens, eos={ik[name]['stopped_eos']}), "
          f"prism {len(b)} chars, prism speed prompt/gen {m.group(2) if m else '?'}/{m.group(3) if m else '?'} t/s")
    print("identical" if first is None else f"first difference at char {first}: ik {a[first:first+40]!r} | prism {b[first:first+40]!r}")
    if first is not None or name == "arith":
        print("--- prism reply:\n" + b)
