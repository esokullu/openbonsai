# Confirming eval: the code-gap model against the reasoning mix on fresh problems (28-29 October)

This folder compares two models on test problems outside the frozen subsets they were scored on before:

- **P**: the reasoning mix, `../2026-10-27-reasoning-mix-v1/` (`examples/qwen3_5-9b-reasoning.yaml`), the model on Hugging Face until 29 October 2026. GGUF SHA-256 `4f2f3d10...`; it stays there at revision [`40964de`](https://huggingface.co/webbrain-one/openbonsai-qwen-3.5-9b/tree/40964de45342e7c1cdd6605cdb00533c9adb567d), tag `reasoning-mix-v1`.
- **A1**: the 1-epoch code-gap run, `../2026-10-28-humaneval-gap/code-gap-1epoch/` (`examples/qwen3_5-9b-reasoning-code.yaml`). GGUF SHA-256 `818d8783bc39d92b2777a0333d38f7a25021a7b906f47344b83b7109eba6f7a0`.

A1 was picked from four runs on the same frozen 50 HumanEval and 100 GSM8K problems it is reported on (`../2026-10-28-humaneval-gap/`), so its 34/50 there probably overstates its gain. This eval scores both models on the problems that come after the frozen ones in the same seeded order, under a rule written and hashed before any score existed. None of the thinking-off problems played any part in choosing A1. Half of the thinking-on problems, HumanEval positions 26-50, were among the 50 it was chosen on, though scored then only with thinking off.

The owner decided on 29 October 2026 to replace P with A1 on Hugging Face before this eval had a result. That decision was not taken under the rule. The result is now in, and the rule's outcome is the same: replace P with A1.

## The rule

From `PREREGISTRATION.md`, frozen on 28 October 2026 at 21:22. `../2026-10-28-humaneval-gap/README.md` gave the same rule under Caveats before the eval, numbered (1)-(4): (a) is its (1) and (2), (b) its (3) and (c) its (4), and its "otherwise the published model stays" covers the last two outcomes below. All three conditions are on fresh problems only:

- **(a) HumanEval, thinking off, positions 51-164.** Let b be the number of problems A1 passes and P fails, and c the reverse. (a) passes if the one-sided exact sign test P(X ≥ b | X ~ Bin(b+c, 1/2)) is below 0.05, and A1 passes strictly more problems than P on the unflagged fresh problems (the same-task flags below).
- **(b) GSM8K, thinking off, positions 101-1,319.** Passes if (A1 correct − P correct) / 1,219 ≥ −0.03, that is, A1 at most 3 percentage points below P. The paired 95% CI is also reported.
- **(c) HumanEval, thinking on, positions 26-75.** Passes if A1 passes at least P's count minus 2. Cap hits and unfinished-thinking counts are also reported.

Outcomes:

- All three pass: recommend replacing the GGUF on Hugging Face in place with A1 (tag `40964de` first; GGUF and card in one commit). The upload remains the owner's decision.
- (a) passes but (b) or (c) fails: bring the trade-off to the owner (A1 with an explicit caveat, or A1 as a second file).
- (a) is nominally ahead (b > c) but not significant, or it fails only on the unflagged condition: keep P, and offer A1 as a second file.
- A1 is not ahead (b ≤ c): keep P.

Also reported, not part of the rule: the frozen-prefix reproduction check, the HumanEval split without the flags whose evidence is only in A1's traces, and pooled HumanEval-164, labelled as including the selection set. A held-out refusal (a problem found in stored calibration or teacher tokens) would have stopped the run; none occurred.

## Problem sets and harness

- **Problems.** The framework's own seeded selector (`sha256-rank/1`, seed 2026) ordered the full HumanEval test split (164 problems) and GSM8K test split (1,319): `manifests/humaneval-164.json` (content SHA-256 `53841a82...`) and `manifests/gsm8k-1319.json` (`903c60e0...`). `build_manifests.py` built them and checked that their first 50 and 100 rows are the frozen subsets. "Position k" is the k-th problem in that order.
- **Fresh sets.** HumanEval positions 51-164 with thinking off (114 problems; none played any part in choosing A1). HumanEval positions 26-75 with thinking on (50; positions 26-50 were scored with thinking off during the selection, never with thinking on). They include four of the eight tasks `../2026-10-28-humaneval-gap/` found restated in, or related to, the traces: HumanEval/23, 20, 13 and 111, at positions 31, 32, 34 and 43. A1 passes two of them (20 and 111), P three (13, 23 and 111). GSM8K positions 101-1,319 with thinking off (1,219).
- **Harness.** Both models were scored from their distillation-stage state, in copies of their run directories, with `openbonsai eval --run <copy> --from distill` (`confirm_eval.sh`). The framework was at 7e82be5; its eval code is identical to 3d20957, which scored A1. P was scored at 43b88b6, which differs only in refine-stage plumbing. PyTorch on one RTX 5090, greedy decoding, with the same caps and batches as the stored records: 512 tokens and batch 16 with thinking off, 4,096 tokens and batch 8 with thinking on. Cached generation batches were reused where the batch was identical.
- **Same-task screen.** Before any score, the 114 fresh HumanEval problems were screened against the union of both trace files (A1's 2,400 traces include all 1,520 of P's): word 8- and 13-grams, TF-IDF, entry-point names, doctest calls and canonical-solution 10-grams (`overlap_fresh.py`); every function in the top-40 TF-IDF traces and in every trace that defines the entry point, run against the problem's own tests (2,632 programs, `func_check_fresh.py`); two independent judges on each of the 64 problems with any signal, and two sweepers searching all 114 for reworded tasks. A problem is flagged if any judge or sweeper labels it same or related, or any trace function passes its tests. **34 of the 114 are flagged** (11 same, 23 related; 17 with evidence only in traces P never saw), and 80 are not. The table is in `PREREGISTRATION.md`, the lists in `screen/FROZEN_FLAGS.json`. Flags count presence in a trace file, not whether the trace reached the calibration or distillation blocks. GSM8K positions 101-1,319 have no flags: no word 13-gram is shared with any trace, and the highest number-masked 8-gram coverage is 0.32.

## Timeline

Local time (UTC+3). "Log" is a line in `logs/confirm_eval.log`; "file time" is a modification time on the machine the eval ran on, which git does not keep, so it cannot be checked from this repository.

| When | What | Source |
|---|---|---|
| 28 Oct, 20:59 | Manifests built | file time |
| 28 Oct, 21:01 | Launcher started; it waits for `PREREGISTRATION.sha256` | log |
| 28 Oct, 21:22 | `screen/FROZEN_FLAGS.json` written (21:22:07), then `PREREGISTRATION.md` and `PREREGISTRATION.sha256` (21:22:19) | file time |
| 28 Oct, 21:22:40 | The launcher found `PREREGISTRATION.sha256` and logged its six hashes, the same as in the file here | log |
| 28 Oct, 22:51 | The rule, as (1)-(4), reached `main` in `../2026-10-28-humaneval-gap/README.md` (commit a491c15, merged in pull request #6) | git |
| 29 Oct, 01:40 | GPU lock acquired; the first eval (P, HumanEval with thinking off) started | log |
| 29 Oct, 01:47 | First eval done | log |
| 29 Oct, 05:55 | Last eval done (A1, GSM8K) | log |
| 29 Oct, 05:56 | `gate.json` written | file time |

`PREREGISTRATION.sha256` holds the SHA-256 of the pre-registration and of the five files it fixes: `screen/FROZEN_FLAGS.json`, `gate.py`, `confirm_eval.sh` and both manifests. `sha256sum -c PREREGISTRATION.sha256`, run in this folder, checks all six. From this repository alone, the evidence that the rule and the flags were fixed before any score is the log line at 21:22:40, which carries these hashes, and the first eval's start at 01:40, not the file times.

## Reproduction check

Each run also scored the frozen prefix of its set. Every prefix reproduced its stored score exactly, with no per-problem differences (`gate.json`, `checks`):

| Model | Set | Prefix | Stored | Now | Problems that differ | Batches from the cache |
|---|---|---|---|---|---|---|
| P | HumanEval, thinking off | 1-50 | 19/50 | 19/50 | none | 3 of 11 |
| A1 | HumanEval, thinking off | 1-50 | 34/50 | 34/50 | none | 3 of 11 |
| P | HumanEval, thinking on | 1-25 | 10/25 | 10/25 | none | 3 of 10 |
| A1 | HumanEval, thinking on | 1-25 | 14/25 | 14/25 | none | 3 of 10 |
| P | GSM8K, thinking off | 1-100 | 69/100 | 69/100 | none | 6 of 83 |
| A1 | GSM8K, thinking off | 1-100 | 65/100 | 65/100 | none | 6 of 83 |

The cached batches cover positions 1-48, 1-24 and 1-96, where a match is automatic. The other prefix rows (HumanEval 49-50 with thinking off, 25 with thinking on, GSM8K 97-100) were generated again in a different batch. HumanEval 49-50 came out identical; HumanEval 25 with thinking on (both models) and three of GSM8K 97-100 (each model) came out with different text and the same result. The stored records are `distill.*.json` in `../2026-10-27-reasoning-mix-v1/eval/` and `../2026-10-28-humaneval-gap/code-gap-1epoch/eval/`.

## Results

Each row pairs the two models on the same problems. The difference is (A1 − P) / n, in percentage points; its 95% interval is the normal approximation `gate.py` uses, ±1.96 × √((b + c) − (b − c)² / n) / n. The sign tests are one-sided, as in the rule (`../2026-10-28-humaneval-gap/` uses two-sided ones).

| Condition | n | A1 | P | Only A1 | Only P | One-sided sign p | Difference (95% CI) |
|---|---|---|---|---|---|---|---|
| (a) HumanEval off, 51-164 | 114 | 62 | 45 | 24 | 7 | 0.0017 | +14.9 (+5.7 to +24.1) |
| (a) HumanEval off, 51-164, unflagged | 80 | 38 | 26 | 16 | 4 | 0.0059 | +15.0 (+4.5 to +25.5) |
| HumanEval off, 51-164, without the 17 A1-only flags (not in the rule) | 97 | 52 | 35 | 21 | 4 | 0.00046 | +17.5 (+8.0 to +27.0) |
| (b) GSM8K off, 101-1,319 | 1,219 | 881 | 871 | 140 | 130 | 0.29 | +0.8 (−1.8 to +3.5) |
| (c) HumanEval on, 26-75 | 50 | 27 | 27 | 13 | 13 | 0.58 | 0.0 (−20.0 to +20.0) |
| HumanEval off, 1-164 pooled (includes the 50 selection problems; not in the rule) | 164 | 96 | 64 | 42 | 10 | 0.0000045 | +19.5 (+11.4 to +27.6) |

As rates: HumanEval 51-164 with thinking off, 54.4% against 39.5%; its unflagged problems, 47.5% against 32.5%; GSM8K 101-1,319, 72.3% against 71.5%; HumanEval 26-75 with thinking on, 54% each. On the 34 flagged problems (computed from `results/`; not in `gate.json`), A1 passes 24 and P 19 (8 problems against 3), about the same gap as on the unflagged ones.

### Cap hits

| Set | Cap | A1 cut off | P cut off |
|---|---|---|---|
| HumanEval on, 26-75 | 4,096 | 21/50 (20 still thinking) | 8/50 (8 still thinking) |
| HumanEval off, 51-164 | 512 | 2/114 | 7/114 |
| GSM8K off, 101-1,319 | 512 | 250/1,219 | 204/1,219 |

With thinking on, the two scores are equal but made differently. No answer cut off at the cap passed, for either model. A1 passed 27 of the 29 answers it finished, P 27 of 42. All 13 problems that only P passed are ones where A1 ran out of tokens. A1 generated 2,374 tokens per problem on average, P 1,344. The frozen 25 HumanEval problems with thinking on showed the same pattern (A1 cut off on 9, P on 5). On GSM8K with thinking on, which this eval did not cover, A1 was cut off less often than P on the frozen 50 (18 against 23; `../2026-10-28-humaneval-gap/`).

The thinking-on counts are from `gate.json` (`c`); the per-answer split, the thinking-off HumanEval cap hits and the mean tokens are computed from `results/`. The GSM8K cap hits are from `gate.json` (`b`).

## Outcome

All three conditions hold (`gate.json`, `outcome`):

- (a): p = 0.0017, below 0.05, and on the 80 unflagged problems A1 passes 38 against P's 26.
- (b): +0.8 percentage points, above −3.
- (c): 27 against 27, at least 27 − 2.

The rule's outcome is to recommend replacing the GGUF on Hugging Face with A1, tagging `40964de` first. The owner had already decided on 29 October, before this result, to make that replacement; the decision was not taken under the rule, and the rule's outcome agrees with it. P stays at revision `40964de`, tag `reasoning-mix-v1`.

## What this shows and what it does not

- **A1's HumanEval gain holds on new problems, at about half the size.** On 114 problems that played no part in choosing A1, A1 passes 17 more than P (+14.9 percentage points, one-sided p = 0.0017). On the frozen 50 it was picked on, the difference was 30 points (34/50 against 19/50). That fits the selection effect the humaneval-gap report expected, though the two sets also differ in their problems. The fresh result reduces the selection concern; it does not make the frozen 34/50 an unbiased number.
- **Reworded test tasks are still possible.** The screen flagged 34 of the 114 as the same task as, or related to, trace content, which is why the rule also requires A1 to lead on the 80 unflagged problems. It does, by about the same margin (+15.0 points, p = 0.0059). The screen can miss restatements that share few words, so the unflagged problems may still include some.
- **GSM8K shows no measurable difference** with thinking off: +0.8 points, with an interval from −1.8 to +3.5. The rule only asked that A1 be no more than 3 points below.
- **The thinking-on guard is small.** With 50 problems, the interval on the difference is ±20 points, so equal scores do not show equal ability with thinking on; they show that A1 met the guard at this cap. A1 runs out of its 4,096 tokens far more often (21 against 8), and every problem it lost to P was one of those. A larger cap was not tried, and GSM8K with thinking on was not part of this eval.
- One run per model, greedy decoding, in the PyTorch harness on the distillation-stage state, not in Prism's runtime. BF16 and Prism's Ternary Bonsai 2 9B were not scored on these problems, so the gap to them on fresh problems is not measured here.

## Files

- `PREREGISTRATION.md`: the rule, the problem sets, the harness and the frozen flag table, written before any score. `PREREGISTRATION.sha256`: the hashes of it and of the five files it fixes.
- `build_manifests.py`, `manifests/`: the ordered problem lists.
- `screen/`: the same-task screen. `FROZEN_FLAGS.json` holds the flag lists the rule uses; `signals.json` the per-problem signals; `overlap.log`, `func_check.log` and `func_check.top40.json` the lexical and functional checks; `judging_workflow_output.json` the judges' and sweepers' output; `labels_merged.json` the merged labels. `scripts/`: `overlap_fresh.py` and `func_check_fresh.py`, the fresh-problem versions of the earlier frozen-set screen, which `patch.py` generated; `packets.py`, which wrote the judges' review packets and `signals.json`. The review packets, the trace index and `overlap.json` are not copied.
- `confirm_eval.sh`: the launcher. It waits for the hash file, then for the GPU lock, and runs the six evals in order.
- `logs/`: `confirm_eval.log` (the launcher's log) and `eval.<model>.<set>.stdout` (each eval's output).
- `results/`: the six per-problem records, `<model>.<set>.json`, with every completion.
- `gate.py`, `gate.json`: the rule applied to `results/`, with the reproduction checks.

The scripts are kept as run, not as portable tools. They carry absolute `D:/openbonsai/...` paths from the machine they ran on: the working folder `runs/code-gap/confirm/`, of which this folder is a copy, the original run directories and the framework snapshot. The copied run directories, the model states and the trace files are not here.
