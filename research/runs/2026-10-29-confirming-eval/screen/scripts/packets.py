"""Per-problem review packets for the fresh-HumanEval same-task judging, plus a compact trace index."""
import json, os
S = "D:/openbonsai/runs/code-gap/confirm/screen/"
A1_TRACES = "D:/openbonsai/runs/2026-10-28-code-gap-1epoch/traces/traces.jsonl"
P_TRACES = "D:/openbonsai/runs/2026-10-27-reasoning-mix-v1/traces/traces.jsonl"
he = json.load(open("D:/openbonsai/runs/code-gap/confirm/manifests/humaneval-164.json", encoding="utf-8"))["problems"]
pos = {p["id"]: k + 1 for k, p in enumerate(he)}
up = lambda r: r["prompt"] if isinstance(r["prompt"], str) else json.dumps(r["prompt"])
pkeys = set()
for line in open(P_TRACES, encoding="utf-8"):
    r = json.loads(line); pkeys.add((up(r), bool(r["thinking"]), r["completion"]))
tr = []
for i, line in enumerate(open(A1_TRACES, encoding="utf-8")):
    r = json.loads(line)
    tr.append({"i": i, "source": r["source"], "row": r["row"], "thinking": r["thinking"],
               "in_P": (up(r), bool(r["thinking"]), r["completion"]) in pkeys, "prompt": up(r),
               "answer": r["completion"].split("</think>")[-1][:2500]})
with open(S + "traces_index.jsonl", "w", encoding="utf-8") as f:
    for t in tr:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
ov = {r["id"]: r for r in json.load(open(S + "overlap.json", encoding="utf-8"))["humaneval"]}
fc = {r["id"]: r for r in json.load(open(S + "func_check.top40.json", encoding="utf-8"))["results"]}
summary = []
for p in he[50:]:
    o, f = ov[p["id"]], fc[p["id"]]
    passes = sorted({x["trace"] for x in f["passes"]})
    lex = [x["trace"] for x in o["top"]]
    sig = {"func_pass": len(passes), "func_pass_A1_only": sum(1 for i in passes if not tr[i]["in_P"]),
           "func_pass_in_P": sum(1 for i in passes if tr[i]["in_P"]), "g13": o["n_g13"], "max_c8_all": o["max_c8_all"],
           "n_def": o["n_def"], "max_tfidf": o["max_tfidf"], "max_sol10": o["max_sol10"], "n_ex": o["n_ex"]}
    pk = {"id": p["id"], "position": pos[p["id"]], "entry_point": p["entry_point"], "prompt": p["prompt"],
          "canonical_solution": p["canonical_solution"], "test": p["test"][:2500], "signals": sig,
          "functional_passes": [{**{k: tr[i][k] for k in ("i", "source", "row", "thinking", "in_P", "prompt")},
                                 "answer": tr[i]["answer"][:1500],
                                 "function": next(x["function"] for x in f["passes"] if x["trace"] == i)} for i in passes[:8]],
          "functional_passes_total": len(passes),
          "lexical_top5": [{**{k: tr[i][k] for k in ("i", "source", "row", "thinking", "in_P", "prompt")},
                            "answer": tr[i]["answer"][:1200],
                            "metrics": {k: v for k, v in x.items() if k not in ("prompt_head", "lcw_text", "files")}}
                           for i, x in zip(lex, o["top"])]}
    json.dump(pk, open(S + f"packets/{p['id'].replace('/', '_')}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    summary.append({"id": p["id"], "position": pos[p["id"]], "entry_point": p["entry_point"], **sig})
json.dump(summary, open(S + "signals.json", "w", encoding="utf-8"), indent=1)
print("packets", len(summary), "with func pass", sum(1 for s in summary if s["func_pass"]),
      "A1-only func pass", sum(1 for s in summary if s["func_pass"] and not s["func_pass_in_P"]))
