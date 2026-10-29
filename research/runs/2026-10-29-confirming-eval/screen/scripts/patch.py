import re
src = "D:/openbonsai/runs/code-gap/publish-review/leakage/scripts/"
LOAD = '''MAN = "D:/openbonsai/runs/code-gap/confirm/manifests/"
A1_TRACES = "D:/openbonsai/runs/2026-10-28-code-gap-1epoch/traces/traces.jsonl"
P_TRACES = "D:/openbonsai/runs/2026-10-27-reasoning-mix-v1/traces/traces.jsonl"
'''
UNION = '''def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _up(r):
    return r["prompt"] if isinstance(r["prompt"], str) else json.dumps(r["prompt"])


traces, _seen = [], {}
for tag, path in (("A1", A1_TRACES), ("P", P_TRACES)):
    for j, r in enumerate(_load(path)):
        key = (_up(r), bool(r["thinking"]), r["completion"])
        if key in _seen:
            traces[_seen[key]]["files"].append(f"{tag}:{j}")
            continue
        _seen[key] = len(traces)
        traces.append({"i": len(traces), "source": r["source"], "kind": r.get("kind"), "row": r["row"],
                       "thinking": r["thinking"], "prompt": _up(r), "completion": r["completion"], "files": [f"{tag}:{j}"]})
for t in traces:
    t["in_A1"] = any(x.startswith("A1:") for x in t["files"])
    t["in_P"] = any(x.startswith("P:") for x in t["files"])
print("union traces", len(traces), "A1", sum(t["in_A1"] for t in traces), "P", sum(t["in_P"] for t in traces),
      "both", sum(t["in_A1"] and t["in_P"] for t in traces), flush=True)
'''
# overlap
s = open(src + "overlap.py", encoding="utf-8").read()
s = re.sub(r'SNAP = .*?\nP_TRACES = .*?\n', LOAD, s, count=1, flags=re.S)
a = s.index("he = json.load(open(SNAP")
b = s.index("# ---------------------------------------------------------------- TF-IDF")
s = s[:a] + ('he = json.load(open(MAN + "humaneval-164.json", encoding="utf-8"))["problems"][50:]\n'
             'gs = json.load(open(MAN + "gsm8k-1319.json", encoding="utf-8"))["problems"][100:]\n'
             'assert len(he) == 114 and len(gs) == 1219\n\n' + UNION + "\n") + s[b:]
s = s.replace('"in_P": t["in_P"], **{a:', '"in_P": t["in_P"], "in_A1": t["in_A1"], "files": t["files"], **{a:')
s = s.replace('top.append({"trace": i, "source": t["source"], "row": t["row"], "in_P": t["in_P"], **m,',
              'top.append({"trace": i, "source": t["source"], "row": t["row"], "in_P": t["in_P"], "in_A1": t["in_A1"], "files": t["files"], **m,')
s = s.replace('"traces": {"file": A1_TRACES, "n": len(traces), "in_P": sum(t["in_P"] for t in traces)}',
              '"traces": {"files": [A1_TRACES, P_TRACES], "n": len(traces), "in_P": sum(t["in_P"] for t in traces), "in_A1": sum(t["in_A1"] for t in traces)}')
s = s.replace('"""Independent test-set overlap check: frozen HumanEval-50 / GSM8K-100',
              '"""FRESH-problem version (HumanEval selection positions 51-164, GSM8K 101-1319; union of A1 and P trace files,\neach trace tagged in_A1/in_P). Adapted from publish-review/leakage/scripts/overlap.py.\n\nOriginal: Independent test-set overlap check: frozen HumanEval-50 / GSM8K-100')
open("overlap_fresh.py", "w", encoding="utf-8").write(s)
# func check
s = open(src + "func_check.py", encoding="utf-8").read()
s = re.sub(r'SNAP = .*?\nP_TRACES = .*?\n', LOAD, s, count=1, flags=re.S)
a = s.index("he = json.load(open(SNAP")
b = s.index("df = Counter()")
s = s[:a] + ('he = json.load(open(MAN + "humaneval-164.json", encoding="utf-8"))["problems"][50:]\nassert len(he) == 114\n\n'
             + UNION + "\n\n") + s[b:]
s = s.replace('"in_P": t["in_P"],\n', '"in_P": t["in_P"], "in_A1": t["in_A1"], "files": t["files"],\n')
s = s.replace("({sum(1 for x in rec['passes'] if not x['in_P'])} new)", "(A1 {sum(1 for x in rec['passes'] if x['in_A1'])}, P {sum(1 for x in rec['passes'] if x['in_P'])})")
s = s.replace('"""Functional \'same task\' check: does any function written in a trace completion pass a frozen',
              '"""FRESH-problem version (HumanEval positions 51-164; union of A1 and P traces). Adapted from\npublish-review/leakage/scripts/func_check.py.\n\nOriginal: Functional \'same task\' check: does any function written in a trace completion pass a frozen')
open("func_check_fresh.py", "w", encoding="utf-8").write(s)
