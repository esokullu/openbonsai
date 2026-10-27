# OpenBonsai research archive

This folder holds the history and evidence behind the OpenBonsai framework: the
experiment scripts that produced the current method, the older CAT-Q and
staged-pipeline code, findings documents, validation receipts and run reports.

Nothing here is part of the package. `openbonsai/` never imports from
`research/`, and pytest does not collect it (`testpaths = ["tests"]`,
`norecursedirs` includes `research`). The framework plan is
[../FRAMEWORK_PLAN.md](../FRAMEWORK_PLAN.md).

## Rerunning history

The code here is kept to be read, not run in place. Modules moved out of
`openbonsai/` keep their relative imports (`from .quantizer import ...`), which
no longer resolve from their new folders.

To run the old code, check out a tag or commit:

| What | Where it last ran in place |
|---|---|
| Staged pipeline, CAT-Q, mixed formats, endpoint calibration, RCO, diagnostics, smoke and preflight modules, and all their tests | tag `legacy-staged-pipeline` (2fadacf): `git checkout legacy-staged-pipeline && python -m pytest` |
| `experiments/` scripts (GPTQ ternary, endpoints, KD, quick eval) | commit 5373d45, at `experiments/` |
| `experiments/export_gguf.py`, `make_synthetic_state.py`, `ref/` and the endpoints `requires_grad_(False)` fix | branch `research/scaleq-recovery` (c2dc532, 5eba901), at `experiments/`; applied here under `research/experiments/` with the same content |

Paths inside `docs/`, `validation/` and `runs/` refer to the layout at those
commits (for example `docs/...`, `openbonsai/pipeline.py`, `tools/...`).

### Following file history across the move

Most moves were pure renames, so `git log --follow <new path>` reaches the old
history. Two renames in commit c331407 are misleading: the renamed files were
then rewritten (323884c) as ports of different code, so their followed history
belongs to unrelated predecessors:

| New path | `git log --follow` shows | Real ancestor |
|---|---|---|
| `openbonsai/stages/calibrate.py` | the old `openbonsai/calibration.py` (a token-ID loader) | `research/experiments/calib_data.py` |
| `tests/test_traces.py` | the old `tests/test_calibration.py` | only that file's `openbonsai/prepare.py` tests (prompt reading, generation) are ancestors; its token-ID loader tests went with `calibration.py` |

For those, read the ancestor's history directly
(`git log -- research/experiments/calib_data.py experiments/calib_data.py`).

## Layout

| Path | Contents | Moved from |
|---|---|---|
| `experiments/` | The scripts behind the current method: `gptq_ternary.py` (sequential GPTQ ternary rounding), `endpoints.py` (ternary embedding and head), `e2e_kd.py` (scale distillation from top-64 teacher outputs), `quick_eval.py` (dev suite), `calib_data.py`; `out/` holds the fake-quant sweep logs. Reference for the package ports; still contains machine-specific `D:/` paths. | `experiments/` |
| `catq/` | CAT-Q ternary quantizer, window trainer, rotated ternary modules and the legacy Qwen3.5 adapter, with tests. Moves into the package only if it beats GPTQ + distillation on the same data. | `openbonsai/` |
| `endpoints/` | CAT-Q embedding and head quantizers, calibration, export and head evaluation, with tests. Superseded by `experiments/endpoints.py`. | `openbonsai/` |
| `rco/` | RCO mixed-precision allocation and candidates, plus vendored RCO code under `_vendor/rco/` with its own LICENSE and NOTICE, with tests. | `openbonsai/` |
| `diagnostics/` | Activation capture and metrics, AdamW optimizer diagnostics, with tests. | `openbonsai/` |
| `legacy/` | The v0 artifact format (`export.py`, `text_export.py`), mixed formats (`mixed_formats.py`, `mixed_export.py`) and the v0 CLI, with tests, including the base-3 codec tests. | `openbonsai/` |
| `tools/` | Official Bonsai GGUF inventory, the PTQ1_0 code/scale audit with its independent decoder (`audit_bonsai_ternary.py`, `bonsai_ternary_decoder.cpp`), random-fixture preparation and probes. | `tools/` |
| `evaluation/bonsai-development/` | A pointer: the frozen development manifest moved into the package as `openbonsai/eval/assets/bonsai-dev/frozen-inputs.json` (same bytes); `experiments/quick_eval.py` reads it there. | (already here) |
| `docs/` | 21 findings documents, plus `research-log.md`, the former root README. | `docs/`, `README.md` |
| `validation/` | 27 validation receipts. | repo root |
| `papers/` | The ScaleQ-1.58 paper (arXiv:2608.01078) and its provenance. | `papers/` |
| `runs/<run>/` | Small reports copied from the local `runs/` directory: `REPORT.md`/`report.md`, `results.json`, `experiment-plan.json`, `summary.md`, `analysis/*.md`, `log.txt`, `eval.json`, and for the 2026-10-27 GPTQ/endpoint/Bonsai-start runs their logs (`stdout.log` is the run's `runs/<name>.stdout`). No checkpoints, tensors, caches or receipts. `2026-10-27-e2e-kd-shared-v1` was still training when these were copied and is not included yet. | local `runs/` |

The optional Prism CPU manifest runner moved to `tools/prism/` (outside
`research/`), and its tests to `tests/prism/`. Its admission forecast,
deadline and process-group supervision were dropped (tag
`legacy-staged-pipeline` keeps them).

The package ports of `experiments/` (core/gptq, stages/quantize, endpoints,
distill, calibrate, eval/suites) keep the scripts' arithmetic; tests/
test_gptq_parity.py and tests/test_calibrate.py check them bitwise against
embedded copies of the script functions, without importing `research/`.

## Removed from the tree

These exist only at tag `legacy-staged-pipeline`:

- Staged-pipeline orchestration: `openbonsai/pipeline.py`, `pipeline_sources.py`, `pipeline_decoder.py`, `pipeline_endpoints.py`, `pipeline_prepare.py`, `stage_runtime.py`, `evaluation_supervisor.py`.
- Smoke modules: `openbonsai/rotation_smoke.py`, `head_smoke.py`, `embedding_smoke.py`.
- Supervised and preflight tools: `tools/evaluate_transformers_manifest.py` with `configs/evaluation/bonsai-development-native.json`, `tools/preflight_bonsai_reference.py`, `tools/smoke_prism_eval.py`.
- Their tests: 11 `test_pipeline_*.py`, `test_stage_runtime.py`, `test_evaluation_supervisor.py`, `test_transformers_manifest_runner.py`, `test_transformers_runner_controls.py`, `test_native_dtype_policy.py`, `test_bonsai_reference_preflight.py`.

## Where the main results are

| Result | Source |
|---|---|
| GPTQ ternary decoder, 256 calibration samples, no scale refit: dev NLL 3.2954, ARC 13/20 (BF16: 2.424, ARC 19/20) | `runs/2026-10-27-gptq-ternary-256-norefit/` |
| The same with one shared sign vector per width (Prism's layout): dev NLL 3.3183, ARC 15/20 | `runs/2026-10-27-gptq-shared-256/` |
| Ternary endpoints on the shared-sign decoder: head only 3.3831 (ARC 16/20), embedding only 3.4161 (11/20), both 3.4793 (13/20) | `runs/2026-10-27-endpoints-shared-256/stdout.log` |
| 128 samples: with scale refit NLL 5.149 (ARC 8/20), without 3.319 (ARC 9/20) | `runs/2026-10-27-gptq-ternary-v1/`, `runs/2026-10-27-gptq-ternary-norefit/` |
| Official Bonsai baseline: dev NLL 3.192, ARC 16/20 | `docs/official-bonsai-development-result.md` |
| Fake-quant sweep (which layer types are most sensitive) | `experiments/out/` |
| Earlier CAT-Q runs on the RTX 5090 | `docs/` and `runs/2026-10-26-*`, `runs/2026-10-27-5090-*` |
