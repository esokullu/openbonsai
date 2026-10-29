"""Scores, caps and loops for Prism's Ternary Bonsai 8B (unpacked checkpoint) next to this run's two models.

A capped answer loops when the first 100 characters of its last 300 already appear earlier in it (the rule
behind the report's capped / looping table: it gives 1, 1, 7 and 12 for this run's thinking-on files).
Also counts completions that contain <|endoftext|>, which stops Qwen3-8B's generation but not Prism's.

    python -I unpacked_summary.py <official/eval> <run eval/> [<out.json>]

Prism's files: ternary-bonsai-8b-unpacked.<suite>.think-off.json and, with Qwen3-8B's chat template,
ternary-bonsai-8b-unpacked-qwen3-template.<suite>.think-on.json; this run's: bf16.* and distill.*.
"""
import json
import sys
from pathlib import Path


def loops(text: str) -> bool:
    return len(text) > 300 and text[-300:][:100] in text[:-300]


def summary(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    probs = d["problems"]
    capped = [p for p in probs if p["hit_cap"]]
    passed = {p["id"] for p in probs if p.get("passed", p.get("correct"))}
    return {"score": d["score"], "hit_cap": d["hit_cap"], "looping": sum(loops(p["completion"]) for p in capped),
            # Prism's generation config stops at <|im_end|> only; Qwen3-8B's also at <|endoftext|>
            "endoftext_in_completion": sum("<|endoftext|>" in p["completion"] for p in probs),
            "unfinished_thinking": d.get("unfinished_thinking") if d.get("thinking") == "on" else None,
            "mean_generated_tokens": d.get("mean_generated_tokens"), "generation_seconds": d.get("generation_seconds"),
            "tokens_per_second": d.get("tokens_per_second"), "batch": d.get("batch"),
            "max_new_tokens": d.get("max_new_tokens"), "chat_kwargs": d.get("chat_kwargs"),
            "_passed": passed, "_ids": [p["id"] for p in probs]}


def main():
    official, ours = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    for suite in ("gsm8k", "humaneval"):
        for mode in ("off", "on"):
            key = f"{suite}.think-{mode}"
            row = {}
            prism = official / f"ternary-bonsai-8b-unpacked.{key}.json"
            if not prism.is_file():     # thinking on: Prism's weights with Qwen3-8B's chat template
                prism = official / f"ternary-bonsai-8b-unpacked-qwen3-template.{key}.json"
            for name, path in (("prism", prism), ("ours", ours / f"distill.{key}.json"),
                               ("bf16", ours / f"bf16.{key}.json")):
                if path.is_file():
                    row[name] = summary(path)
            if "prism" in row and "ours" in row:
                assert row["prism"]["_ids"] == row["ours"]["_ids"], f"{key}: different problems"
                p, o = row["prism"]["_passed"], row["ours"]["_passed"]
                row["overlap"] = {"both": len(p & o), "prism_only": len(p - o), "ours_only": len(o - p),
                                  "neither": len(row["prism"]["_ids"]) - len(p | o)}
            for r in row.values():
                r.pop("_passed", None)
                r.pop("_ids", None)
            out[key] = row
    text = json.dumps(out, indent=1)
    print(text)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
