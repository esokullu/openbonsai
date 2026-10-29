# Closing the HumanEval gap of the reasoning recipe (28 October)

The published model (`examples/qwen3_5-9b-reasoning.yaml`, `../2026-10-27-reasoning-mix-v1/`) scores 19/50 on HumanEval with thinking off, against 44/50 for BF16 Qwen3.5-9B and 45/50 for Prism's Ternary Bonsai 2 9B. Its answers are well formed; they fail on logic and syntax. This folder holds one run with more code data and two ablations of it, all on one RTX 5090.

This report was written on 28 October, when the reasoning-mix model was the one on Hugging Face, and updated on 29 October with the comparison's result under Caveats and the Recommendation; "published" in this report means the reasoning-mix model (GGUF SHA-256 `4f2f3d10...`). On 29 October 2026 the `code-gap-1epoch` model (GGUF SHA-256 `818d8783...`) replaced it there, by the owner's decision, taken before the comparison under Caveats had a result; the previous file stays there at revision `40964de`, tag `reasoning-mix-v1`.

## Runs

| Run | Data | Epochs | HumanEval off / on | GSM8K off / on | Dev NLL | ARC /20 | Wiki 16K | ARC /100 | Chat NLL | KL vs BF16 | EAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Published (`../2026-10-27-reasoning-mix-v1/`) | reasoning mix | 1 | 19/50 / 10/25 | 69/100 / 25/50 | 2.960 | 17 | 2.698 | 84 | 1.439 | 0.554 | 0.706 |
| `reasoning-mix-2epoch/` | reasoning mix | 2 | 21/50 / 10/25 | 71/100 / 20/50 | 2.945 | 16 | 2.670 | 85 | 1.403 | 0.532 | 0.717 |
| `code-gap-2epoch/` | more code | 2 | 29/50 / 15/25 | 74/100 / 18/50 | 2.998 | 18 | 2.735 | 90 | 1.422 | 0.581 | 0.702 |
| **`code-gap-1epoch/`** (`examples/qwen3_5-9b-reasoning-code.yaml`; on Hugging Face since 29 October) | more code | 1 | **34/50 / 14/25** | 65/100 / **32/50** | 3.012 | 18 | 2.763 | 89 | 1.459 | 0.607 | 0.693 |
| BF16 (`../2026-10-27-reasoning-v1/`) | | | 44/50 / 22/25 | 77/100 / 36/50 | 2.423 | 19 | 2.218 | 97 | 0.554 | | |
| Ternary Bonsai 2 9B (Prism ML) | | | 45/50 / 24/25 | 77/100 / 46/50 | 3.191 | 17 | | | | | |

"More code" against the reasoning mix:

- **traces**: two more OpenCodeInstruct sources, `opencodeinstruct_2` (280 prompts, thinking on, 1,536-token cap, seed 7) and `opencodeinstruct_off` (600 prompts, thinking off, 768-token cap, seed 8). That makes 2,400 traces and 2.10M generated tokens; code goes from 32% to 50% of them. The first seven sources and their 1,520 traces are unchanged; they were served from a copy of the published run's batch cache.
- **calibrate**: traces 128 to 192 of the 256 blocks (WikiText-2 64 to 32, UltraChat 64 to 32).
- **distill**: 1,856 teacher sequences instead of 1,536, with traces 640 to 1,088, UltraChat 640 to 512 and WikiText-2 256.

Everything else is the published recipe. The 2-epoch code-gap run was the first try. The ablations then changed one thing each. `code-gap-1epoch` keeps the code-gap stages and teacher and trains one epoch. `reasoning-mix-2epoch` keeps the published stages and teacher and trains two. Each was copied from its source run up to the teacher and resumed with `--set distill.epochs=N` (`ablations.sh`), so only distillation training, export and eval ran. Every stage before distillation, and the teacher, is bit-identical to its source's (same stage hashes).

## Findings

**The data closes more than half of the gap; the second epoch does not help.** With thinking off, both code-gap runs beat the published model on HumanEval:

- With 1 epoch, 18 problems are solved only by the new run and 3 only by the published one (two-sided sign test p = 0.0015).
- With 2 epochs, the split is 12 to 2 (p = 0.013).
- Of the 44 problems BF16 solves, the published model misses 25 and the 1-epoch run 12.
- The 1-epoch run's failures are mostly wrong answers that run: 10 fail an assertion and 3 raise an error inside the tests, against 18 and 6 for the published model (11 against 23 among answers not cut at the cap). Two syntax errors remain, both cut at the 512-token cap, against 6 before.
- A second epoch on the published data does not do it: 21/50, with 5 problems gained and 3 lost (p = 0.73).

**The second epoch makes thinking run longer, on both data sets.** The code-gap data with two epochs scores 18/50 on GSM8K with thinking on, against 32/50 with one (2 against 16 problems, p = 0.0013). The published data with two epochs scores 20/50, against 25/50 with one (3 against 8, p = 0.23).

| Thinking-on GSM8K | Mean generated tokens | Cut at 4,096 | Finished and correct |
|---|---|---|---|
| Published data, 1 epoch | 2,652 | 23/50 | 25/27 |
| Published data, 2 epochs | 3,043 | 28/50 | 20/22 |
| Code-gap data, 2 epochs | 3,097 | 32/50 | 18/18 |
| Code-gap data, 1 epoch | 2,346 | 18/50 | 32/32 |
| BF16 | 2,266 | 14/50 | 36/36 |

- Nearly every finished answer is correct in every run, so the losses are answers cut off mid-thought, not wrong reasoning.
- The code-gap data with one epoch thinks the least of the four ternary runs, close to BF16.
- With thinking off, the second epoch is nominally ahead on GSM8K with the code-gap data, 74/100 against 65/100 (15 against 6 problems, p = 0.078), and behind on HumanEval, 29/50 against 34/50 (1 against 6, p = 0.125). With the published data it moves neither (71/100 and 21/50, against 69/100 and 19/50).

**GSM8K with thinking off is within noise.** The 1-epoch run scores 65/100 against the published 69/100: 10 problems solved only by the new run and 14 only by the published one (p = 0.54).

**The generic measures move the other way.** The second epoch improves dev NLL, WikiText 16K, chat NLL, KL and EAR with either data set, and the published data with two epochs is the best of the four on all of them (KL 0.532). The code-gap data with one epoch is the worst of the four on all of them (KL 0.607), and the best on HumanEval and thinking-on GSM8K. ARC (18/20 and 89/100, against 17 and 84) is the exception. On this recipe, KL to BF16 does not predict the reasoning scores.

## Recommendation

`examples/qwen3_5-9b-reasoning-code.yaml` holds the 1-epoch code-gap settings. On the frozen suites it was picked on, it improves HumanEval with thinking off by 15 problems (p = 0.0015) at the same model size; GSM8K is 7 higher with thinking on (p = 0.09) and 4 lower with thinking off (p = 0.54), both within noise. On HumanEval it is still 10 problems short of BF16 and 11 short of Prism's model. It also costs about 1 h more GPU time: the two extra trace sources take 3,744 s, and the larger teacher takes 7 min instead of 6. The published recipe, `examples/qwen3_5-9b-reasoning.yaml`, is unchanged. The owner chose to publish the 1-epoch model, and it replaced the published model on Hugging Face on 29 October 2026. That decision was the owner's, taken before the comparison on new problems under Caveats had a result, not under that comparison's rule; the result, under Caveats, gives the rule's outcome to replace the published model.

## Caveats

- Single runs, greedy decoding, small suites: 50 and 25 HumanEval problems, 100 and 50 GSM8K. The sign tests treat each problem as one paired trial and do not account for having looked at several suites.
- The two ablations ran while two virtual machines took most of the CPU, so their thinking-on evals decoded at 27-51 tokens/s against 89-132 for the published run. This changes time, not results (greedy, same batches).
- The code-gap traces were screened for HumanEval and GSM8K test problems like the published ones (`docs/evaluation.md`). The new seeds drew none of the 3 OpenCodeInstruct rows (first shard) that share a 13-word run with a test problem (`code-gap-2epoch/traces/summary.json`, `near_duplicates_skipped`).
- That guard matches text, so it does not catch a task restated in other words. A later screen of the traces (candidate traces picked by TF-IDF and entry-point name, with their functions run against the problem's tests; it can miss restatements with little word overlap) found reworded versions of at least six of the 50 HumanEval tasks. 23 and 61 are in the distillation data of both the published and the code-gap runs; 13, 20, 35 and 90 only in the code-gap runs' distillation data (the published run also generated traces for 13 and 90, but neither reached its calibration or distillation blocks). Two more are related tasks with a different specification (111, 156). Without all eight, the 1-epoch run still beats the published model 28/42 to 15/42 (16 problems against 3, p = 0.0044). No trace shares a 13-word run with the 50 HumanEval or 100 GSM8K problems.
- `code-gap-1epoch` was picked from these four runs on the same suites it is scored on, so its 34/50 probably overstates the gain on new problems. A comparison with the published model covers the problems after the frozen ones in the same seeded order: HumanEval positions 51-164 and GSM8K positions 101-1,319 with thinking off, and HumanEval positions 26-75 with thinking on. Its rule was fixed before any score was read: (1) on HumanEval positions 51-164 the 1-epoch config beats the published model in a one-sided exact sign test at p < 0.05, and (2) it also leads on the problems among those 114 that a same-task screen, run before scoring, does not flag; (3) on GSM8K positions 101-1,319 it is at most 3 percentage points below; (4) with thinking on, on HumanEval positions 26-75 (26-50 were scored with thinking off during the selection), it passes at least the published model's count minus 2. If all four hold, the recommendation is to replace the published model; if (1) and (2) hold but (3) or (4) fails, the trade-off goes to the owner; otherwise the published model stays.
- The comparison ran on 28-29 October; its pre-registration, logs, per-problem results and the rule applied are in `../2026-10-29-confirming-eval/`. The rule and the same-task screen were frozen at 21:22 on 28 October; the first eval started at 01:40 on 29 October. The `PREREGISTRATION.md` there states the same rule as three conditions: (1) and (2) are its (a), (3) its (b) and (4) its (c). All four conditions hold (sign tests one-sided, as the rule states; differences in percentage points with 95% CIs):
  - (1) HumanEval positions 51-164, thinking off: the 1-epoch run passes 62/114 and the published model 45/114; 24 problems are solved only by the 1-epoch run and 7 only by the published one (p = 0.0017), a difference of +14.9 points (+5.7 to +24.1).
  - (2) The screen flagged 34 of the 114 as the same task as, or related to, one in the traces (11 same, 23 related; 17 with evidence only in traces the published run never had). On the 80 unflagged problems the 1-epoch run passes 38 and the published model 26 (16 problems against 4, p = 0.0059), +15.0 points (+4.5 to +25.5).
  - (3) GSM8K positions 101-1,319, thinking off: 881/1,219 against 871/1,219 (140 problems against 130), +0.8 points (−1.8 to +3.5). No measurable difference.
  - (4) HumanEval positions 26-75, thinking on: 27/50 each (13 problems against 13). The 1-epoch run ran out of its 4,096 tokens on 21 of the 50 and the published model on 8; no answer cut off passed, and all 13 problems only the published model solved are ones the 1-epoch run was cut off on. With 50 problems the interval is ±20 points, so this shows only that the guard held.

  Every frozen prefix (HumanEval 1-50 with thinking off and 1-25 with thinking on, GSM8K 1-100) reproduced its stored score with no per-problem difference. The rule's outcome is to replace the published model. The owner had already decided that replacement on 29 October, before the result, so the decision was not taken under the rule; the rule's outcome agrees with it. On new problems the HumanEval gain is about half the 30 points on the 50 it was picked on, which fits the selection caveat above. Those problems played no part in choosing the 1-epoch run, so the selection caveat does not apply to them, but restated tasks that the screen missed cannot be ruled out. Pooled over all 164 HumanEval problems, which include the 50 used for selection and are not part of the rule, the 1-epoch run passes 96 and the published model 64 (42 problems against 10).

## Files

- `code-gap-2epoch/`: the first try (`config.yaml` is its config with notes, `run.yaml` the resolved one), with logs of every stage, the traces summary, the distillation log (`distill/train.jsonl`), the export report and every eval result.
- `code-gap-1epoch/` and `reasoning-mix-2epoch/`: the ablations. Their trace, calibrate, quantize and endpoint logs say "up to date; skipped", because those stages were copied from the source runs. The teacher was reused for the same reason.
- `compare.py`: the per-problem comparison (`python research/runs/2026-10-28-humaneval-gap/compare.py [base]`).
- `ablations.sh`: the launcher for the two ablations.
- Per-answer generations (`eval/generations/`), the GGUF files, the scales and the teachers are not copied.
