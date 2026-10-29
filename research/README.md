# OpenBonsai research archive

This folder holds the history and evidence behind the OpenBonsai framework:
- the experiment scripts that produced the current method;
- the older CAT-Q and staged-pipeline code;
- findings documents, validation receipts and run reports.

Nothing here is part of the package. `openbonsai/` never imports from `research/`, and pytest does not collect it (`testpaths = ["tests"]`, and `norecursedirs` includes `research`). The method the package ships, with its evidence, is in [../docs/method.md](../docs/method.md). The framework plan is [../FRAMEWORK_PLAN.md](../FRAMEWORK_PLAN.md).

## The research line in brief

1. **26 October 2026: CAT-Q staged pipeline.** We started from the ScaleQ-1.58 paper ([papers/](papers/README.md)): AYOT traces plus CAT-Q calibration, with Hadamard rotations added to match Prism's layout. On one RTX 5090 at small budgets (16 to 128 samples), every compressed 9B model failed. Dev NLL stayed between 8.3 and 13.3 (BF16: 2.42), ARC answers collapsed to one letter, and the generations were unusable. Rotations, teacher-target controls, longer first-layer training and optimizer settings each improved a local reconstruction metric without restoring the model. The reports are in `runs/2026-10-26-*` and `runs/2026-10-27-5090-*`, and the findings in `docs/`.
2. **27 October 2026: GPTQ.** A fake-quant sweep (`experiments/quick_eval.py`, logs in `experiments/out/`) located the damage. Sequential GPTQ ternary rounding (`experiments/gptq_ternary.py`) then reached dev NLL 3.30 in about 10 minutes, with BF16 endpoints: for the first time close to the official Ternary Bonsai's 3.19, though only with the head and embedding in BF16 (with both ternary it was 3.48). Scale refitting after GPTQ was tried and dropped (5.15).
3. **Endpoints, distillation, export.** The ternary head and embedding (`experiments/endpoints.py`), scale distillation (`experiments/e2e_kd.py`) and the PTQ1_0 GGUF export (`experiments/export_gguf.py`) completed the method: dev NLL 2.892 with every weight ternary, in a 2 GB file that Prism's llama.cpp runs.
4. **The framework.** `openbonsai/` ports those scripts with the same arithmetic. CPU tests check the ports bit for bit against embedded copies of the script functions, and given the measured state, the package's exporter writes a 9B GGUF byte-identical to the research export (`validation/p2-export-parity.md`). The package's own `openbonsai run` of the 9B example (traces off, the measured run's calibration data) reproduced it on the GPU: its quantize stage is bit-identical to the research GPTQ state (0 code or scale differences in 6.9B weights; dev NLL 3.3183, ARC 15/20), and the full run through endpoints, distillation, verified export and eval gave dev NLL 2.887, ARC 19/20, Wiki 16K 2.630, ARC 89/100, chat NLL 1.322 (`research/runs/2026-10-27-framework-acceptance/`). The CAT-Q code stays here until it beats GPTQ plus distillation on the same data.

## Rerunning history

The code here is kept to be read, not run in place. Modules moved out of `openbonsai/` keep their relative imports (`from .quantizer import ...`), which no longer resolve from their new folders.

To run the old code, check out a tag or commit:

| What | Where it last ran in place |
|---|---|
| Staged pipeline, CAT-Q, mixed formats, endpoint calibration, RCO, diagnostics, smoke and preflight modules, and all their tests | tag `legacy-staged-pipeline` (2fadacf) |
| `experiments/` scripts (GPTQ ternary, endpoints, KD, quick eval) | commit 5373d45, at `experiments/` |
| `experiments/export_gguf.py`, `make_synthetic_state.py`, `ref/` and the endpoints `requires_grad_(False)` fix | branch `research/scaleq-recovery` (c2dc532, 5eba901), at `experiments/`; applied here under `research/experiments/` with the same content |

For the tag, use a separate worktree, so your current checkout and its runs stay untouched:

```
git worktree add ../openbonsai-legacy legacy-staged-pipeline
cd ../openbonsai-legacy
python -m pytest
```

The experiment scripts still contain machine-specific `D:/` paths for their data and outputs; change them before running.

Paths inside `docs/`, `validation/` and `runs/` refer to the layout at those commits (for example `docs/...`, `openbonsai/pipeline.py`, `tools/...`).

### Following file history across the move

Most moves were pure renames, so `git log --follow <new path>` reaches the old history. Two renames in commit c331407 are misleading. The renamed files were then rewritten (323884c) as ports of different code, so their followed history belongs to unrelated predecessors:

| New path | `git log --follow` shows | Real ancestor |
|---|---|---|
| `openbonsai/stages/calibrate.py` | the old `openbonsai/calibration.py` (a token-ID loader) | `research/experiments/calib_data.py` |
| `tests/test_traces.py` | the old `tests/test_calibration.py` | only that file's `openbonsai/prepare.py` tests (prompt reading, generation) are ancestors; its token-ID loader tests went with `calibration.py` |

For those, read the ancestor's history directly (`git log -- research/experiments/calib_data.py experiments/calib_data.py`).

## Layout

| Path | Contents | Moved from |
|---|---|---|
| `experiments/` | The scripts behind the current method: `gptq_ternary.py` (sequential GPTQ ternary rounding), `endpoints.py` (ternary embedding and head), `e2e_kd.py` (scale distillation from top-64 teacher outputs), `export_gguf.py` (PTQ1_0 GGUF), `quick_eval.py` (dev suite, fake-quant sweep), `calib_data.py`. `out/` holds the fake-quant sweep logs. Reference for the package ports; still contains machine-specific `D:/` paths. | `experiments/` |
| `catq/` | CAT-Q ternary quantizer, window trainer, rotated ternary modules and the legacy Qwen3.5 adapter, with tests. Moves into the package only if it beats GPTQ + distillation on the same data. | `openbonsai/` |
| `endpoints/` | CAT-Q embedding and head quantizers, calibration, export and head evaluation, with tests. Superseded by `experiments/endpoints.py`. | `openbonsai/` |
| `rco/` | RCO mixed-precision allocation and candidates, plus vendored RCO code under `_vendor/rco/` with its own LICENSE and NOTICE, with tests. | `openbonsai/` |
| `diagnostics/` | Activation capture and metrics, AdamW optimizer diagnostics, with tests. | `openbonsai/` |
| `legacy/` | The v0 artifact format (`export.py`, `text_export.py`), mixed formats (`mixed_formats.py`, `mixed_export.py`) and the v0 CLI, with tests, including the base-3 codec tests. | `openbonsai/` |
| `tools/` | Official Bonsai GGUF inventory, the PTQ1_0 code/scale audit with its independent decoder (`audit_bonsai_ternary.py`, `bonsai_ternary_decoder.cpp`), random-fixture preparation and probes. | `tools/` |
| `evaluation/bonsai-development/` | A pointer: the frozen development manifest moved into the package as `openbonsai/eval/assets/bonsai-dev/frozen-inputs.json` (same bytes); `experiments/quick_eval.py` reads it there. | (already here) |
| `docs/` | 21 findings documents, plus `research-log.md`, the former root README (index below). | `docs/`, `README.md` |
| `validation/` | 27 validation receipts (JSON) and `p2-export-parity.md`, the record of the package export's byte parity. | repo root |
| `papers/` | The ScaleQ-1.58 paper (arXiv:2608.01078) and its provenance. | `papers/` |
| `runs/<run>/` | Small reports copied from the local `runs/` directory (index below). No checkpoints, tensors, caches or receipts. | local `runs/` |

The optional Prism CPU manifest runner moved to `tools/prism/` (outside `research/`), and its tests to `tests/prism/`. Its admission forecast, deadline and process-group supervision were dropped; the tag `legacy-staged-pipeline` keeps them.

The package ports of `experiments/` (core/gptq, stages/quantize, endpoints, distill, calibrate, eval/suites) keep the scripts' arithmetic. `tests/test_gptq_parity.py` and `tests/test_calibrate.py` check them bit for bit against embedded copies of the script functions, without importing `research/`.

## Run reports

`stdout.log` is a run's `runs/<name>.stdout`. Numbers are on the frozen dev suite (2,048 WikiText-2 test tokens, 20 ARC-Easy questions) unless stated.

**The current method (27 October):**

| Run | What it holds |
|---|---|
| `runs/2026-10-27-gptq-ternary-v1/` | GPTQ, 128 calibration blocks, scale refit after GPTQ: dev NLL 5.149, ARC 8 (`eval.json`, `log.txt`) |
| `runs/2026-10-27-gptq-ternary-norefit/` | Same without the refit: 3.319, ARC 9 |
| `runs/2026-10-27-gptq-ternary-256/` | 256 blocks with refit: started, not evaluated (`log.txt` holds the config) |
| `runs/2026-10-27-gptq-ternary-256-norefit/` | 256 blocks, no refit, per-module signs: 3.2954, ARC 13 |
| `runs/2026-10-27-gptq-shared-256/` | Same with one sign vector per width (Prism's layout): 3.3183, ARC 15; extended suite 3.103, ARC 78/100, chat 1.964 |
| `runs/2026-10-27-endpoints-256/` | Endpoints on the per-module-sign decoder: one line of log, no result |
| `runs/2026-10-27-endpoints-shared-256/` | Ternary endpoints on the shared-sign decoder: head only 3.3831 (ARC 16), embedding only 3.4161 (11), both 3.4793 (13) (`stdout.log`, progress bars collapsed to their last state) |
| `runs/gguf-test/` | The PTQ1_0 GGUF exporter on a synthetic 9B state with edge cases: export logs, verification reports, a negative control, and Prism's load log |
| `runs/gguf-test/runtime-parity/` | Prism's logits against a PyTorch FP32 reference of the same synthetic state (`ref_vs_llama.py`): mean KL 0.0028 nats, top-1 95.1% over 508 tokens (4 x 256, WikiText-2 test) with the distilled scales of the file (`ref_fq_e2e`); 0.068 / 68.1% against the reference without them (`ref_fq_plain_vs_e2efile`); BF16 weights as a baseline (`ref_orig`). The synthetic state has perplexity about 13,700; the check was not run on the distilled 9B (it was on the distilled Qwen3-8B, below). The 252 MB logit dump is not copied |
| `runs/2026-10-27-bf16-extended/` | BF16 Qwen3.5-9B: dev 2.423 (ARC 19), extended suite 2.218, ARC 97/100, chat 0.554 |
| `runs/2026-10-27-e2e-kd-shared-v1/` | Scale distillation of the all-ternary shared-sign state, 4.2M tokens (`log.txt`): 2.892, ARC 19; extended 2.626, ARC 87/100, chat 1.318 |
| `runs/2026-10-27-e2e-kd-shared-small-v1/` | The same with the small high-precision tensors trained too: 2.884, ARC 19; extended 2.616, ARC 87/100, chat 1.308 |
| `runs/2026-10-27-gguf-shared-v1/` | The distilled state exported as a PTQ1_0 GGUF (`export.log`, `*.export.json`) and Prism's CPU perplexity (8 x 512) on a copy of WikiText-2 test with the blank lines removed and CRLF line endings: 11.93, against 16.51 for the official model (`prism_ppl_*.log`). On upstream llama.cpp's `wiki.test.raw` they are 12.88 and 17.21 (`runs/2026-10-28-prism-ppl-upstream/`). The GGUF itself is not copied |
| `runs/2026-10-27-compare-ear-v1/` | The package's `compare` suite (4,088 WikiText-2 test tokens, full vocabulary, against BF16 Qwen3.5-9B) on two GGUFs: `gguf-shared-v1` KL 0.456, EAR 0.737; the official Ternary Bonsai 2 9B KL 0.693, EAR 0.676 (`eval/*.compare.json`, `logs/eval.log`). The BF16 reference logits are not copied |
| `runs/2026-10-28-prism-ppl-upstream/` | Prism's CPU perplexity (c9c34bc, 8 x 512) on upstream llama.cpp's `wiki.test.raw`: official Ternary Bonsai 2 9B 17.21, `gguf-shared-v1` 12.88, the reasoning mix (the previous Hugging Face model) 13.61 (its `runs/2026-10-27-reasoning-mix-v1/export/prism_ppl.txt` gives 12.59 on the old copy), and the two code-gap runs of `runs/2026-10-28-humaneval-gap/` 13.93 (2 epochs) and 14.15 (1 epoch, the current Hugging Face model). The official file on the old copy gives 16.51 (`official-old-copy.log`) |
| `runs/2026-10-27-accept-256/eval/` | The package's own `openbonsai eval` of `2026-10-27-gptq-ternary-256-norefit`'s state on the GPU (acceptance run): dev 3.293, ARC 14; extended 3.082, ARC 67/100, chat 1.948 |
| `runs/2026-10-28-qwen3-8b-reasoning/` | Qwen3-8B through the dense adapter with the 9B reasoning recipe (`report.md`): GSM8K 70/100 and HumanEval 22/50 with thinking off (BF16 88 and 41), `compare` ΔNLL +0.317; Prism perplexity 14.94 and runtime parity on the real weights (98.8% top-1); the 9B perplexities re-measured on upstream `wiki.test.raw` (`prism/`, the same numbers as `runs/2026-10-28-prism-ppl-upstream/`) |
| `runs/2026-10-29-confirming-eval/` | The pre-registered comparison of the 1-epoch code-gap model of `runs/2026-10-28-humaneval-gap/` (the current Hugging Face model) with the reasoning mix (the previous one) on test problems outside the frozen subsets (`README.md`; rule and same-task screen frozen before any score): HumanEval positions 51-164 with thinking off 62/114 against 45/114 (24 problems against 7, one-sided sign test p = 0.0017), 38/80 against 26/80 on the problems the screen does not flag; GSM8K positions 101-1,319 881 against 871 of 1,219; HumanEval positions 26-75 with thinking on 27/50 each, with 21 against 8 answers cut at 4,096 tokens. The rule's outcome: replace. The scripts are kept as run, with `D:/` paths |

**Starting from Prism's weights (separate experiment, not a from-scratch result):**

| Run | What it holds |
|---|---|
| `runs/2026-10-27-bonsai-start-convert/` | The official Ternary Bonsai 2 9B converted to PyTorch and scored: 3.191, ARC 17 |
| `runs/2026-10-27-bonsai-start-scales/` | Prism's codes kept, group scales distilled on 2.1M tokens: 2.706, ARC 18 |
| `runs/2026-10-27-bonsai-start-lora/` | Same budget, with low-rank adapters instead of scales: 2.713, ARC 17 |
| `runs/2026-10-27-bonsai-start-heldout/` | The base, scale and LoRA variants against the teacher on held-out data |

**The CAT-Q line (26–27 October), all failed on quality:**

| Run | What it holds |
|---|---|
| `runs/2026-10-26-5090-pilot/` | First small pilot: exact serialization, quality lost (NLL +10.5) |
| `runs/2026-10-26-5090-diagnosis/` | Diagnosis of the pilot: damage spread over all families; a fixed-code least-squares scale recovery lowers NLL by 2 nats but leaves ARC at 4/20 and the answers unusable |
| `runs/2026-10-26-5090-corrected/` | Corrected 16×1 pipeline: 11.19, ARC 1 |
| `runs/2026-10-26-5090-rotations/` | Rotated vs unrotated 16×8 pair: 9.26 vs 14.44 |
| `runs/2026-10-26-5090-ultrachat/`, `-ultrachat-attempt2/` | UltraChat calibration, rotated 16×8 vs 128×1: 8.50 and 9.12, all ARC answers "A" |
| `runs/2026-10-26-5090-diversity-gate/` | Diversity control, stopped at the source-data gate |
| `runs/2026-10-26-5090-activation-diagnostic-v1/`, `-v2/` | Prefix drift versus conditional reconstruction: v1 stopped, v2 measurement only |
| `runs/2026-10-26-5090-teacher-target-repair-v1/`, `-v2/` | Original-prefix teacher targets: v1 stopped before fitting; v2 a local improvement, quality still failed |
| `runs/2026-10-26-5090-original-prefix-sequential-v1/`, `-v2/` | Sequential original-prefix run: v1 stopped at the resource gate; v2 completed, 11.85, all ARC answers "A" |
| `runs/2026-10-27-5090-v2-raw-factor-evidence-review/` | Evidence review of the v2 run |
| `runs/2026-10-27-5090-window0-budget-control-v1/` | 512 vs 128 first-layer updates: local MSE −46% |
| `runs/2026-10-27-5090-window0-epsilon-control-v1/`, `-v2/` | AdamW epsilon 1e-9 vs 1e-8: v1 stopped before launch; v2 local MSE −24.5% |
| `runs/2026-10-27-5090-head-rotation-preflight-v1/`, `-pair-v1/` | Head rotation preflight (failed) and test logs |
| `runs/2026-10-27-5090-cuda-workspace-diagnostic-v1/` | A synthetic CUDA workspace diagnostic |

## Findings documents

| Document | Subject |
|---|---|
| [docs/research-log.md](docs/research-log.md) | The former root README: the research line as it was written at the time |
| [docs/research-direction.md](docs/research-direction.md) | The rotation research plan and its decision gates |
| [docs/method.md](docs/method.md) | The CAT-Q-era compression method and its reference boundaries (superseded by [../docs/method.md](../docs/method.md)) |
| [docs/staged-pipeline.md](docs/staged-pipeline.md) | The staged local BF16 compression pipeline (tag `legacy-staged-pipeline`) |
| [docs/rotations-audit.md](docs/rotations-audit.md) | Prism's rotation contract and our prototype |
| [docs/5090-runtime.md](docs/5090-runtime.md) | The first rotated 5090 result and runtime integration |
| [docs/activation-diagnostic.md](docs/activation-diagnostic.md) | Prefix drift versus conditional reconstruction |
| [docs/teacher-target-control.md](docs/teacher-target-control.md) | Teacher target provenance control |
| [docs/sequential-original-target-result.md](docs/sequential-original-target-result.md) | Sequential original targets: alignment improved, quality did not |
| [docs/window0-calibration-result.md](docs/window0-calibration-result.md) | Longer first-layer calibration: a local improvement |
| [docs/window0-epsilon-result.md](docs/window0-epsilon-result.md) | Smaller AdamW epsilon: a second local improvement |
| [docs/optimizer-diagnostics.md](docs/optimizer-diagnostics.md) | Observing stored AdamW moments |
| [docs/endpoint-calibration.md](docs/endpoint-calibration.md), [docs/embedding-calibration.md](docs/embedding-calibration.md) | CAT-Q head and embedding calibration |
| [docs/bonsai-reference-preflight.md](docs/bonsai-reference-preflight.md) | Inventory of the official Bonsai checkpoint |
| [docs/bonsai-ternary-statistics.md](docs/bonsai-ternary-statistics.md) | The official Bonsai codes: 32.76% zeros, actual and hypothetical packing sizes |
| [docs/official-bonsai-development-result.md](docs/official-bonsai-development-result.md) | The official Bonsai on the frozen dev suite in Prism's CPU runtime: 3.192, ARC 16/20 |
| [docs/matched-evaluation.md](docs/matched-evaluation.md) | Matched evaluation across transformers and Prism |
| [docs/storage-budget.md](docs/storage-budget.md) | The broad-coverage storage budget |
| [docs/base3-packing.md](docs/base3-packing.md) | Lossless base-3 packing, and when BITCOS would win |
| [docs/text-only-export.md](docs/text-only-export.md) | Native text-only extraction |
| [docs/rco-audit.md](docs/rco-audit.md) | Source audit of RCO mixed-precision allocation |

## Removed from the tree

These exist only at the tag `legacy-staged-pipeline`:

- Staged-pipeline orchestration: `openbonsai/pipeline.py`, `pipeline_sources.py`, `pipeline_decoder.py`, `pipeline_endpoints.py`, `pipeline_prepare.py`, `stage_runtime.py`, `evaluation_supervisor.py`.
- Smoke modules: `openbonsai/rotation_smoke.py`, `head_smoke.py`, `embedding_smoke.py`.
- Supervised and preflight tools: `tools/evaluate_transformers_manifest.py` with `configs/evaluation/bonsai-development-native.json`, `tools/preflight_bonsai_reference.py`, `tools/smoke_prism_eval.py`.
- Their tests: 11 `test_pipeline_*.py`, `test_stage_runtime.py`, `test_evaluation_supervisor.py`, `test_transformers_manifest_runner.py`, `test_transformers_runner_controls.py`, `test_native_dtype_policy.py`, `test_bonsai_reference_preflight.py`.
