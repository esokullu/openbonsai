"""FRESH-problem version (HumanEval selection positions 51-164, GSM8K 101-1319; union of A1 and P trace files,
each trace tagged in_A1/in_P). Adapted from publish-review/leakage/scripts/overlap.py.

Original: Independent test-set overlap check: frozen HumanEval-50 / GSM8K-100 vs the A1 (= C2) trace file,
with P's trace file as the reference for which traces are new in the code-gap data.

Read-only. Single process, pure Python (no torch, no model). Writes JSON/TSV into the output folder
given as argv[1].

Measures per (problem, trace):
  HumanEval
    def_hit      'def <entry_point>(' appears in the trace prompt or completion
    name_prompt  entry_point appears as a word in the trace prompt
    c8_all       share of the problem's word 8-grams (from 'def ep(' on, as the guard) found in prompt+completion
    c8_prompt    same, prompt only;  j8_prompt  8-gram Jaccard problem vs prompt
    g13          number of shared word 13-grams (the guard's near-duplicate rule) in prompt+completion
    tfidf        TF-IDF cosine, docstring prose vs trace prompt (paraphrase detector)
    lcw          longest common contiguous word run, problem vs prompt+completion (top candidates only)
    ex_hits      doctest/example calls 'ep(args)' (whitespace removed) found verbatim in the trace
    sol10        share of canonical-solution code-token 10-grams found in the completion
  GSM8K
    g13          shared word 13-grams;  c8m  share of number-masked 8-grams (template twins) found;
    tfidf        TF-IDF cosine question vs trace prompt;  lcw on top candidates
"""
import difflib, json, math, os, re, sys
from collections import Counter, defaultdict

OUT = sys.argv[1]
MAN = "D:/openbonsai/runs/code-gap/confirm/manifests/"
A1_TRACES = "D:/openbonsai/runs/2026-10-28-code-gap-1epoch/traces/traces.jsonl"
P_TRACES = "D:/openbonsai/runs/2026-10-27-reasoning-mix-v1/traces/traces.jsonl"

WORD = re.compile(r"\w+")
CODE = re.compile(r"\w+|[^\w\s]")
STOP = set("""a an the of to in and or is are be for on that this with as by it its at from if
then else return returns returned given list string number numbers int str function write
should which will you your we e g i ie example examples for true false none""".split())


def words(t):
    return WORD.findall(str(t).lower())


def grams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def mask(ws):
    return ["<n>" if w.isdigit() else w for w in ws]


def nospace(t):
    return "".join(str(t).split())


def he_parts(p):
    ep = p["entry_point"]
    k = p["prompt"].find(f"def {ep}(")
    text = p["prompt"][max(k, 0):]
    m = re.search(r'("""|\'\'\')(.*?)(\1|$)', text, re.S)
    doc = m.group(2) if m else text
    prose = re.split(r">>>|\bfor example\b|\bexamples?\b\s*:?|\bexample\b", doc, maxsplit=1, flags=re.I)[0]
    calls = []
    for mm in re.finditer(re.escape(ep) + r"\s*\(", doc):
        depth, j = 0, mm.end() - 1
        while j < len(doc):
            if doc[j] == "(":
                depth += 1
            elif doc[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        call = nospace(doc[mm.start():j + 1])
        if len(call) > len(ep) + 2:
            calls.append(call)
    return text, doc, prose, sorted(set(calls))


def user_prompt(rec):
    return rec["prompt"] if isinstance(rec["prompt"], str) else json.dumps(rec["prompt"])


# ---------------------------------------------------------------- data
he = json.load(open(MAN + "humaneval-164.json", encoding="utf-8"))["problems"][50:]
gs = json.load(open(MAN + "gsm8k-1319.json", encoding="utf-8"))["problems"][100:]
assert len(he) == 114 and len(gs) == 1219

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

# ---------------------------------------------------------------- TF-IDF on trace prompts
df = Counter()
tp_tf = []
for t in traces:
    c = Counter(w for w in words(t["prompt"]) if w not in STOP and len(w) > 1)
    tp_tf.append(c)
    df.update(c.keys())
N = len(traces)


def vec(counter):
    v = {w: (1 + math.log(c)) * math.log((N + 1) / (df.get(w, 0) + 1)) for w, c in counter.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


tp_vec = [vec(c) for c in tp_tf]


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


# ---------------------------------------------------------------- HumanEval
he_info = []
idx8, idx13 = defaultdict(set), defaultdict(set)
for k, p in enumerate(he):
    text, doc, prose, calls = he_parts(p)
    ws = words(text)
    g8, g13 = set(grams(ws, 8)), set(grams(ws, 13))
    for g in g8:
        idx8[g].add(k)
    for g in g13:
        idx13[g].add(k)
    sol = CODE.findall(p["canonical_solution"])
    sol10 = set(grams(sol, 10))
    he_info.append({"id": p["id"], "ep": p["entry_point"], "words": ws, "n8": len(g8), "n13": len(g13),
                    "prose": prose.strip(), "calls": calls, "sol10": sol10,
                    "vec": vec(Counter(w for w in words(prose) if w not in STOP and len(w) > 1)),
                    "defre": re.compile(r"def\s+" + re.escape(p["entry_point"]) + r"\s*\("),
                    "namere": re.compile(r"\b" + re.escape(p["entry_point"]) + r"\b")})

he_pairs = defaultdict(dict)       # (k, trace i) -> metrics, only for nonzero signals
for t in traces:
    pw, cw = words(t["prompt"]), words(t["completion"])
    allw = pw + ["\u2016"] + cw          # separator so no gram spans prompt|completion
    g8p = set(grams(pw, 8))
    g8a = set(grams(allw, 8))
    g13a = set(grams(allw, 13))
    hit8a, hit8p, hit13 = Counter(), Counter(), Counter()
    for g in g8a:
        for k in idx8.get(g, ()):
            hit8a[k] += 1
    for g in g8p:
        for k in idx8.get(g, ()):
            hit8p[k] += 1
    for g in g13a:
        for k in idx13.get(g, ()):
            hit13[k] += 1
    ns_all = nospace(t["prompt"]) + "\u2016" + nospace(t["completion"])
    ctoks = CODE.findall(t["completion"])
    c10 = None
    for k, h in enumerate(he_info):
        m = {}
        if hit8a[k]:
            m["c8_all"] = hit8a[k] / max(h["n8"], 1)
        if hit8p[k]:
            m["c8_prompt"] = hit8p[k] / max(h["n8"], 1)
            m["j8_prompt"] = hit8p[k] / (h["n8"] + len(g8p) - hit8p[k])
        if hit13[k]:
            m["g13"] = hit13[k]
        if h["defre"].search(t["prompt"]) or h["defre"].search(t["completion"]):
            m["def_hit"] = True
        if h["namere"].search(t["prompt"]):
            m["name_prompt"] = True
        ex = [c for c in h["calls"] if c in ns_all]
        if ex:
            m["ex_hits"] = len(ex)
            m["ex_total"] = len(h["calls"])
        cs = cos(h["vec"], tp_vec[t["i"]])
        if cs >= 0.15:
            m["tfidf"] = round(cs, 4)
        if h["sol10"] and (m.get("def_hit") or cs >= 0.25 or hit8a[k]):
            if c10 is None:
                c10 = set(grams(ctoks, 10))
            s = len(h["sol10"] & c10) / len(h["sol10"])
            if s:
                m["sol10"] = round(s, 3)
        if m:
            he_pairs[k][t["i"]] = m


def he_score(m):
    return (max(m.get("c8_all", 0), m.get("tfidf", 0) * 0.5) + (0.3 if m.get("def_hit") else 0)
            + 0.3 * (m.get("ex_hits", 0) / max(m.get("ex_total", 1), 1)))


he_rows = []
for k, h in enumerate(he_info):
    pairs = he_pairs.get(k, {})
    ranked = sorted(pairs.items(), key=lambda kv: -he_score(kv[1]))[:5]
    top = []
    for i, m in ranked:
        t = traces[i]
        sm = difflib.SequenceMatcher(None, h["words"], words(t["prompt"]) + ["\u2016"] + words(t["completion"]),
                                     autojunk=False)
        lm = sm.find_longest_match(0, len(h["words"]), 0, len(sm.b))
        top.append({"trace": i, "source": t["source"], "row": t["row"], "thinking": t["thinking"],
                    "in_P": t["in_P"], "in_A1": t["in_A1"], "files": t["files"], **{a: (round(b, 4) if isinstance(b, float) else b) for a, b in m.items()},
                    "lcw": lm.size, "lcw_text": " ".join(h["words"][lm.a:lm.a + lm.size]),
                    "score": round(he_score(m), 4), "prompt_head": t["prompt"][:400]})
    agg = {
        "max_c8_all": max((m.get("c8_all", 0) for m in pairs.values()), default=0),
        "max_c8_all_new": max((m.get("c8_all", 0) for i, m in pairs.items() if not traces[i]["in_P"]), default=0),
        "max_tfidf": max((m.get("tfidf", 0) for m in pairs.values()), default=0),
        "max_tfidf_new": max((m.get("tfidf", 0) for i, m in pairs.items() if not traces[i]["in_P"]), default=0),
        "n_g13": sum(1 for m in pairs.values() if m.get("g13")),
        "n_def": sum(1 for m in pairs.values() if m.get("def_hit")),
        "n_def_new": sum(1 for i, m in pairs.items() if m.get("def_hit") and not traces[i]["in_P"]),
        "n_ex": sum(1 for m in pairs.values() if m.get("ex_hits")),
        "max_sol10": max((m.get("sol10", 0) for m in pairs.values()), default=0),
        "max_lcw_top5": max((x["lcw"] for x in top), default=0),
    }
    he_rows.append({"id": h["id"], "entry_point": h["ep"], "n_examples": len(h["calls"]),
                    "prose": h["prose"][:300], **agg, "top": top})

# ---------------------------------------------------------------- GSM8K
gidx13, gidx8m = defaultdict(set), defaultdict(set)
g_info = []
for k, p in enumerate(gs):
    ws = words(p["question"])
    g13, g8m = set(grams(ws, 13)), set(grams(mask(ws), 8))
    for g in g13:
        gidx13[g].add(k)
    for g in g8m:
        gidx8m[g].add(k)
    g_info.append({"id": p["id"], "words": ws, "n13": len(g13), "n8m": len(g8m),
                   "vec": vec(Counter(w for w in ws if w not in STOP and len(w) > 1))})

g_pairs = defaultdict(dict)
for t in traces:
    allw = words(t["prompt"]) + ["\u2016"] + words(t["completion"])
    h13, h8 = Counter(), Counter()
    for g in set(grams(allw, 13)):
        for k in gidx13.get(g, ()):
            h13[k] += 1
    for g in set(grams(mask(allw), 8)):
        for k in gidx8m.get(g, ()):
            h8[k] += 1
    for k, g in enumerate(g_info):
        m = {}
        if h13[k]:
            m["g13"] = h13[k]
        if h8[k]:
            m["c8m"] = round(h8[k] / max(g["n8m"], 1), 4)
        cs = cos(g["vec"], tp_vec[t["i"]])
        if cs >= 0.2:
            m["tfidf"] = round(cs, 4)
        if m:
            g_pairs[k][t["i"]] = m

g_rows = []
for k, g in enumerate(g_info):
    pairs = g_pairs.get(k, {})
    ranked = sorted(pairs.items(), key=lambda kv: -max(kv[1].get("c8m", 0), kv[1].get("tfidf", 0) * 0.5))[:3]
    top = []
    for i, m in ranked:
        t = traces[i]
        sm = difflib.SequenceMatcher(None, g["words"], words(t["prompt"]), autojunk=False)
        lm = sm.find_longest_match(0, len(g["words"]), 0, len(sm.b))
        top.append({"trace": i, "source": t["source"], "row": t["row"], "in_P": t["in_P"], "in_A1": t["in_A1"], "files": t["files"], **m,
                    "lcw_prompt": lm.size, "lcw_text": " ".join(g["words"][lm.a:lm.a + lm.size]),
                    "prompt_head": t["prompt"][:300]})
    g_rows.append({"id": g["id"], "question": gs[k]["question"][:300],
                   "n_g13": sum(1 for m in pairs.values() if m.get("g13")),
                   "max_c8m": max((m.get("c8m", 0) for m in pairs.values()), default=0),
                   "max_c8m_new": max((m.get("c8m", 0) for i, m in pairs.items() if not traces[i]["in_P"]), default=0),
                   "max_tfidf": max((m.get("tfidf", 0) for m in pairs.values()), default=0),
                   "max_lcw_top3": max((x["lcw_prompt"] for x in top), default=0), "top": top})

os.makedirs(OUT, exist_ok=True)
json.dump({"humaneval": he_rows, "gsm8k": g_rows,
           "traces": {"files": [A1_TRACES, P_TRACES], "n": len(traces), "in_P": sum(t["in_P"] for t in traces), "in_A1": sum(t["in_A1"] for t in traces)}},
          open(os.path.join(OUT, "overlap.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", os.path.join(OUT, "overlap.json"))
