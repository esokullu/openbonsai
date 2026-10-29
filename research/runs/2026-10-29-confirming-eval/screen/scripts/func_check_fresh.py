"""FRESH-problem version (HumanEval positions 51-164; union of A1 and P traces). Adapted from
publish-review/leakage/scripts/func_check.py.

Original: Functional 'same task' check: does any function written in a trace completion pass a frozen
HumanEval problem's own unit test when used as the candidate?

Candidates per problem: the TOPK traces by TF-IDF cosine (docstring prose vs trace prompt) plus every
trace that defines 'def <entry_point>('. From each candidate completion (thinking dropped), every
```python block is parsed; its top-level imports and function/class definitions are kept (no other
top-level statements run), and each top-level function whose positional arity fits the entry point
is tried as the candidate under the HumanEval test, after the problem prompt (so prompt helpers such
as encode_shift exist). Programs run one at a time through openbonsai.eval.scoring.run_program
(separate process, temp dir, timeout, memory limit). Read-only apart from argv[1]/func_check.json.
"""
import ast, json, math, os, re, sys, time
from collections import Counter

sys.path.insert(0, "D:/openbonsai/work/code-gap-snapshot")
from openbonsai.eval.scoring import run_program  # noqa: E402

OUT = sys.argv[1]
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MAN = "D:/openbonsai/runs/code-gap/confirm/manifests/"
A1_TRACES = "D:/openbonsai/runs/2026-10-28-code-gap-1epoch/traces/traces.jsonl"
P_TRACES = "D:/openbonsai/runs/2026-10-27-reasoning-mix-v1/traces/traces.jsonl"
WORD = re.compile(r"\w+")
STOP = set("""a an the of to in and or is are be for on that this with as by it its at from if
then else return returns returned given list string number numbers int str function write
should which will you your we e g i ie example examples for true false none""".split())


def words(t):
    return WORD.findall(str(t).lower())


he = json.load(open(MAN + "humaneval-164.json", encoding="utf-8"))["problems"][50:]
assert len(he) == 114

def _load(path):
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


df = Counter()
tfs = []
for t in traces:
    c = Counter(w for w in words(t["prompt"]) if w not in STOP and len(w) > 1)
    tfs.append(c)
    df.update(c.keys())
N = len(traces)


def vec(c):
    v = {w: (1 + math.log(n)) * math.log((N + 1) / (df.get(w, 0) + 1)) for w, n in c.items()}
    s = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / s for w, x in v.items()}


tv = [vec(c) for c in tfs]


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def prose(p):
    ep = p["entry_point"]
    text = p["prompt"][max(p["prompt"].find(f"def {ep}("), 0):]
    m = re.search(r'("""|\'\'\')(.*?)(\1|$)', text, re.S)
    doc = m.group(2) if m else text
    return re.split(r">>>|\bfor example\b|\bexamples?\b\s*:?|\bexample\b", doc, maxsplit=1, flags=re.I)[0]


def arity(fn):
    a = fn.args
    pos = len(a.posonlyargs) + len(a.args)
    required = pos - len(a.defaults)
    return required, (10 ** 6 if a.vararg else pos)


def blocks(completion):
    text = completion.split("</think>")[-1]
    text = text.replace("<|im_end|>", "")
    found = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    return found or [text]


def ep_arity(p):
    tree = ast.parse(p["prompt"])
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == p["entry_point"])
    req, tot = arity(fn)
    return tot


results = []
runs = 0
t0 = time.time()
for p in he:
    ep, n_ep = p["entry_point"], ep_arity(p)
    pv = vec(Counter(w for w in words(prose(p)) if w not in STOP and len(w) > 1))
    sims = sorted(((cos(pv, tv[t["i"]]), t["i"]) for t in traces), reverse=True)
    cand = {i: s for s, i in sims[:TOPK]}
    dre = re.compile(r"def\s+" + re.escape(ep) + r"\s*\(")
    for t in traces:
        if dre.search(t["completion"]) or dre.search(t["prompt"]):
            cand.setdefault(t["i"], cos(pv, tv[t["i"]]))
    rec = {"id": p["id"], "entry_point": ep, "candidates": len(cand), "tried": 0, "passes": []}
    for i, sim in sorted(cand.items(), key=lambda kv: -kv[1]):
        t = traces[i]
        for b in blocks(t["completion"]):
            try:
                tree = ast.parse(b)
            except SyntaxError:
                continue
            keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                                            ast.AsyncFunctionDef, ast.ClassDef))]
            fns = [n for n in keep if isinstance(n, ast.FunctionDef)]
            code = "\n\n".join(ast.unparse(n) for n in keep)
            for fn in fns:
                req, tot = arity(fn)
                if not (req <= n_ep <= tot):
                    continue
                prog = (p["prompt"] + "\n    pass\n\n" + code + f"\n\n__candidate = {fn.name}\n\n" + p["test"]
                        + "\n\ncheck(__candidate)\n")
                r = run_program(prog, timeout=5.0)
                runs += 1
                rec["tried"] += 1
                if r["status"] == "passed":
                    rec["passes"].append({"trace": i, "source": t["source"], "row": t["row"], "in_P": t["in_P"], "in_A1": t["in_A1"], "files": t["files"],
                                          "thinking": t["thinking"], "function": fn.name, "tfidf": round(sim, 4),
                                          "prompt_head": t["prompt"][:300]})
    results.append(rec)
    print(f"{p['id']:14s} {ep:28s} cand={rec['candidates']:3d} tried={rec['tried']:3d} "
          f"passes={len(rec['passes'])} (A1 {sum(1 for x in rec['passes'] if x['in_A1'])}, P {sum(1 for x in rec['passes'] if x['in_P'])}) "
          f"[{runs} runs, {time.time() - t0:.0f}s]", flush=True)

json.dump({"topk": TOPK, "results": results}, open(os.path.join(OUT, f"func_check.top{TOPK}.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("total runs", runs, "seconds", round(time.time() - t0))
