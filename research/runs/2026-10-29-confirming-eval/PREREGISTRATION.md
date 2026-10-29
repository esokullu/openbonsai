# Confirming eval, P vs A1: pre-registration (written 2026-10-28, before any confirming-eval score exists)

Models (scored from the distill stage, in copies of the run dirs; the originals are untouched):

- **P**: the published model, `runs/2026-10-27-reasoning-mix-v1` → copy `runs/code-gap/confirm/runs/P`. GGUF on HF: sha256 4f2f3d10..., HF revision 40964de.
- **A1**: code-gap data, 1 epoch, `runs/2026-10-28-code-gap-1epoch` → copy `runs/code-gap/confirm/runs/A1`. GGUF sha256 818d8783bc39d92b2777a0333d38f7a25021a7b906f47344b83b7109eba6f7a0.

Harness: `D:/openbonsai/work/code-gap-snapshot` at 7e82be5 (origin/main). Its eval code (`openbonsai/eval`, `stages/evaluate.py`, `heldout.py`) is identical to 3d20957, which scored A1. P was scored at 43b88b6, which differs only in refine-stage plumbing (cli.py, evaluate.py docstrings and source lists). Python `D:/openbonsai/.venv-fla`, RTX 5090, under gpu.lock. Greedy decoding with the same caps and batches as the stored records: thinking off 512 tokens, batch 16; thinking on 4,096 tokens, batch 8. Cached generation batches are reused where the batch is identical.

Launcher: `confirm_eval.sh` (it starts only after `PREREGISTRATION.sha256` exists). In order:

1. P, then A1: `openbonsai eval --run <copy> --from distill --suite humaneval --set 'eval.reasoning.thinking=[off]' --set eval.reasoning.humaneval.n=164`
2. P, then A1: `... --suite humaneval --set 'eval.reasoning.thinking=[on]' --set eval.reasoning.humaneval.n_thinking=75`
3. P, then A1: `... --suite gsm8k --set 'eval.reasoning.thinking=[off]' --set eval.reasoning.gsm8k.n=1319`

Results are copied to `results/<model>.<name>.json`.

## Problem sets

The manifests were built offline with the framework's own selector (`sha256-rank/1`, seed as frozen). Their first 50 / 100 rows equal the frozen subsets (checked):

- `manifests/humaneval-164.json`, content sha256 53841a826e8d46af86cabf2d8f60e815da9ca92732ad5fddfe2a0541adc7749d
- `manifests/gsm8k-1319.json`, content sha256 903c60e088aee1a6c83b42dbd638cb3e56a86c20809b3f143fbcbdd99ad574d1

"Position k" means the k-th problem in selection order (1-based).

- **Fresh HumanEval, thinking off**: positions 51-164 (114 problems). None of them played any part in choosing A1.
- **Fresh HumanEval, thinking on**: positions 26-75 (50 problems). Positions 26-50 were seen thinking off during selection but never thinking on.
- **Fresh GSM8K, thinking off**: positions 101-1319 (1,219 problems).

## Gate (all on fresh problems only)

- **(a) HumanEval thinking off, positions 51-164.** Let b = problems A1 passes and P fails, and c = the reverse. Pass if:
  - the one-sided exact sign test P(X >= b | X ~ Bin(b+c, 1/2)) < 0.05, and
  - A1 passes strictly more than P on the unflagged fresh subset (the flag list is frozen below).
- **(b) GSM8K thinking off, positions 101-1319.** Pass if (A1 correct - P correct) / 1219 >= -0.03. Also report the paired 95% CI.
- **(c) HumanEval thinking on, positions 26-75.** Pass if A1 passes >= P passes - 2. Also report the cap hits and unfinished-thinking counts.

Outcomes:

- All three pass: recommend replacing the HF GGUF in place with A1 (tag 40964de first; GGUF and card in one commit). The upload remains the user's decision.
- (a) passes but (b) or (c) fails: bring the trade-off to the user (A1 with an explicit caveat, or A1 as a second file).
- (a) is nominally ahead (b > c) but not significant, or it fails only on the unflagged-subset condition: keep P, and offer A1 as a second file as the user's choice.
- A1 is not ahead (b <= c): keep P.

## Also reported (not part of the gate)

- **Frozen-prefix reproduction check.** Positions 1-50 (HumanEval off), 1-25 (on) and 1-100 (GSM8K) should match the stored records. The exceptions are re-batched rows: HumanEval off 49-50, HumanEval on 25, GSM8K 97-100. A cache hit makes a match trivial. Any other mismatch is investigated before the fresh numbers are used.
- **Flag-scope split.** Results excluding problems whose same-task traces are in A1's trace file but not in P's. This addresses the bias toward A1 from data only A1 saw.
- **Pooled HumanEval-164.** Labelled as including the selection set.
- **Held-out refusal.** If the eval-time held-out check refuses to score a problem (it is in stored calibration or teacher tokens), the run stops. That is reported to the user, not worked around.

## Frozen same-task flags (fixed before any confirming-eval score)

The screen covered fresh HumanEval positions 51-164 against the union of A1's and P's trace files. A1's file has 2,400 traces, and all 1,520 of P's traces are among them. Scripts and outputs are in `screen/`:

- `overlap_fresh.py`: word 8- and 13-grams, TF-IDF, entry-point names, doctest calls and canonical-solution 10-grams.
- `func_check_fresh.py`: every function in the top-40 TF-IDF traces and every `def <entry_point>` trace, run against the problem's own tests (2,632 programs).
- Judging workflow `wf_aaec25c0-db5`: two independent judges on each of the 64 problems with any signal, plus two sweepers searching all 114 for reworded tasks. Results are in `labels_merged.json`.

Rule: flagged = any judge or sweeper labels same/related, or any trace function passes the problem's tests (union, conservative); scope A1_only = no evidence trace is in P's trace file.

**Flagged: 34 of 114** (11 same, 23 related; 17 with evidence only in traces P never saw). **Unflagged: 80.**

| Position | Problem | Label | Scope | Judges | Sweep | Functional passes |
|---|---|---|---|---|---|---|
| 52 | HumanEval/59 `largest_prime_factor` | same | both | same, same | same | 0 |
| 57 | HumanEval/31 `is_prime` | same | both | same, same | same | 10 |
| 58 | HumanEval/34 `unique` | same | A1_only | same, same | same | 2 |
| 66 | HumanEval/51 `remove_vowels` | same | A1_only | same, same | same | 2 |
| 67 | HumanEval/30 `get_positive` | related | both | related, related | related | 0 |
| 69 | HumanEval/22 `filter_integers` | related | both | related, related | - | 0 |
| 72 | HumanEval/7 `filter_by_substring` | same | A1_only | same, same | same | 1 |
| 82 | HumanEval/16 `count_distinct_characters` | related | A1_only | - | related | 0 |
| 84 | HumanEval/119 `match_parens` | related | both | related, unrelated | related | 0 |
| 87 | HumanEval/29 `filter_by_prefix` | related | A1_only | unrelated, unrelated | related | 0 |
| 88 | HumanEval/126 `is_sorted` | related | A1_only | related, related | related | 0 |
| 89 | HumanEval/151 `double_the_difference` | related | A1_only | - | related | 0 |
| 90 | HumanEval/68 `pluck` | related | both | related, related | related | 0 |
| 97 | HumanEval/40 `triples_sum_to_zero` | related | both | - | related | 0 |
| 98 | HumanEval/47 `median` | same | both | same, same | same | 4 |
| 99 | HumanEval/48 `is_palindrome` | same | A1_only | same, same | same | 7 |
| 101 | HumanEval/18 `how_many_times` | related | A1_only | - | related | 0 |
| 108 | HumanEval/10 `make_palindrome` | related | A1_only | - | related | 0 |
| 109 | HumanEval/54 `same_chars` | related | A1_only | related, related | related | 0 |
| 113 | HumanEval/152 `compare` | related | both | unrelated, related | related | 0 |
| 118 | HumanEval/56 `correct_bracketing` | related | both | related, related | related | 0 |
| 120 | HumanEval/26 `remove_duplicates` | related | both | related, related | related | 0 |
| 122 | HumanEval/120 `maximum` | related | A1_only | related, related | related | 0 |
| 128 | HumanEval/124 `valid_date` | related | A1_only | related, related | related | 0 |
| 130 | HumanEval/58 `common` | same | both | same, same | same | 3 |
| 131 | HumanEval/94 `skjkasdkd` | related | A1_only | related, related | related | 0 |
| 136 | HumanEval/82 `prime_length` | related | both | related, unrelated | related | 0 |
| 138 | HumanEval/53 `add` | same | both | same, same | same | 1 |
| 141 | HumanEval/64 `vowels_count` | related | both | related, related | related | 0 |
| 151 | HumanEval/85 `add` | related | A1_only | unrelated, unrelated | related | 0 |
| 153 | HumanEval/25 `factorize` | same | A1_only | same, same | same | 2 |
| 158 | HumanEval/133 `sum_squares` | related | A1_only | - | related | 0 |
| 159 | HumanEval/43 `pairs_sum_to_zero` | related | both | - | related | 0 |
| 161 | HumanEval/55 `fib` | same | both | same, same | same | 10 |

The screen judged traces in the trace files. Whether a trace reached the calibration or teacher blocks was not checked: flags count trace-file presence.

GSM8K positions 101-1319: no flags. No word 13-gram is shared with any trace, and the highest number-masked 8-gram coverage is 0.32. TF-IDF-only topical matches are not flagged, the same rule as for the frozen set.

Gate (a)'s unflagged condition uses the 80 unflagged problems. The flag-scope split (reported, not gated) drops the 17 A1-only problems.
