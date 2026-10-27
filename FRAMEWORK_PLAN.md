# OpenBonsai framework plan (for approval)

> **Status, 27 October 2026.** P1 to P3 are on `framework/v1`; P4 (README, docs, web copy) is the docs commit after them. P5 (CAT-Q) has not started. This page is kept as the plan that was approved; the measured numbers below are the ones known when it was written. Current results: [README.md](README.md#results) and [docs/method.md](docs/method.md).

Goal: the repo becomes a reusable, Bonsai-format ternary compression framework.
Qwen3.5-9B is supported today. Other dense models (Nanbeige, other Qwen sizes, Llama-style) get small per-architecture adapters.
Research history lives in `research/`, in the same repo.

## Method the package ships

1. **Rotation, Prism layout.** One sign vector per input width, Hadamard block 4096, group 128, fp16 scale per (row, group).
2. **Sequential GPTQ ternary rounding.** Layer by layer, on calibration data that includes the model's own traces (AYOT).
3. **Ternary embedding and head.**
   - Head: GPTQ with the final-hidden-state Hessian.
   - Embedding: GPTQ weighted by the first norm's gain.
4. **End-to-end distillation of the group scales** from the BF16 model's top-64 outputs. Codes stay frozen.
5. **Export to Prism's PTQ1_0 GGUF**, about 2 GB, runnable in Prism's llama.cpp.
6. **ScaleQ/CAT-Q block refinement.** Stays in `research/` until it beats GPTQ + distillation on the same data; then it moves into the package.

Measured so far, dev NLL (BF16 2.42; official Bonsai 3.19):

| Setup | Dev NLL |
|---|---|
| Plain ternary rounding | 8.9 |
| GPTQ with scale refit | 5.15 |
| GPTQ without refit | 3.30 (ARC 13/20) |

Endpoint and distillation numbers come from the shared-sign run now queued on the GPU.

## Target tree

```
README.md  LICENSE  NOTICE  pyproject.toml
openbonsai/
  cli.py  config.py  rundir.py
  adapters/  base.py  registry.py  qwen3_5.py  dense.py      # dense.py: Llama/Qwen3/Nanbeige family table
  core/      hadamard.py  ternary.py  gptq.py  losses.py
  stages/    traces.py  calibrate.py  quantize.py  endpoints.py  distill.py  export.py  evaluate.py
  formats/   state.py  ptq1_0.py  gguf.py                    # PTQ1_0 codec + GGUF reader/writer (no gguf-py dependency)
  eval/      manifest.py  suites.py  assets/bonsai-dev/      # frozen manifest + FP64 scorer, extended suite
tests/       tiny random models per adapter, codec, GGUF round trip, resume
docs/        method.md  adding-a-model.md  formats.md  evaluation.md  cli.md
examples/    qwen3_5-9b.yaml  tiny-*.yaml  prompts.jsonl
tools/prism/ optional Prism CPU manifest runner (official-Bonsai comparisons)
web/         site; copy updated to the 4-stage method, current video relabeled "Background: how ScaleQ works"
research/    README.md (links tag legacy-staged-pipeline), experiments/, runs/<run>/ (small reports),
             catq/ (CAT-Q refine, LoRA), docs/ (21 findings docs), validation/, papers/, tools/
```

`openbonsai/` shrinks from 12.2k lines to about 4k.

## CLI

Each stage writes into one run directory. Outputs are written atomically, so a crash loses at most the unit in progress. Resume points: per layer in quantize, per shard in the teacher pass, per checkpoint in distillation.

```
openbonsai run --model <HF id or local fine-tune> --config examples/qwen3_5-9b.yaml --out runs/my-model
  = traces -> calibrate -> quantize -> endpoints -> distill -> export -> eval
```

Using it on your own fine-tune:
- your model is the teacher and generates its own traces with its own chat template;
- optional JSONL of your own data is mixed into calibration;
- evaluation compares the result against your BF16 model (KL, top-1 agreement, NLL);
- your tokenizer and chat template go into the GGUF.

## ModelAdapter (what a new architecture provides)

- `load()`, `text_model()`, `head()`.
- Per-layer plan:
  - sequential groups of linears that share an input (for example qkv+z, then out, then gate+up, then down);
  - linears kept high precision (Qwen3.5: GatedDeltaNet `in_proj_a/b`; also conv1d, A_log, dt_bias and norms).
- Norm convention: `(1+w)` vs `w`. This sets the embedding Hessian rule.
- Tied or untied embedding/head. Tied tables stay Q8/F16 by default; ternary is opt-in.
- Chat template, thinking flag, EOS ids.
- GGUF arch name and tensor map, used by the export.

Geometry (block size, sign widths) is derived from the model by the core, not the adapter. A new dense model means one table row; a new architecture means one file of about 150 lines plus a tiny test config.

## What moves where

| Destination | Contents |
|---|---|
| **Package (ported or moved)** | <ul><li>`hadamard.py` → core</li><li>PTQ1_0 codec → formats</li><li>`experiments/gptq_ternary.py` → core/gptq + stages/quantize</li><li>`endpoints.py`, `e2e_kd.py` → stages (tail-bucket KD loss)</li><li>`calib_data.py` → stages/calibrate (pinned HF datasets, no `D:/` paths)</li><li>`quick_eval.py`, `evaluation.py`, `eval_backends.py` → eval/</li><li>`prepare.py` → stages/traces (lean, batched)</li><li>`export_gguf.py` → formats/gguf + stages/export</li><li>Sibling branch's PTQ1_0 decoder → formats (for `eval --from model.gguf`; authorship kept by cherry-pick)</li></ul> |
| **`research/`** | <ul><li>CAT-Q quantizer/trainer/rotations</li><li>Endpoint/embedding calibration</li><li>RCO (`_vendor`, `allocation`, `candidates`)</li><li>Diagnostics</li><li>`docs/*.md` findings</li><li>`validation-*.json`</li><li>`papers/`</li><li>Experiment scripts snapshot</li><li>Run reports copied from `D:/openbonsai/runs` (REPORT.md, results.json, experiment-plan.json, analysis/*.md; under ~1 MB each)</li></ul> |
| **Dropped** (tag `legacy-staged-pipeline` at 2fadacf keeps them runnable) | <ul><li>`pipeline*.py`, `stage_runtime`, `evaluation_supervisor`</li><li>Supervisors, parent-PID and worker scans, admission forecasts, seal/freeze/provenance tools</li><li>Preflight and smoke modules, and their tests</li></ul> |

## Phases

Each phase lands as separate commits on a new branch `framework/v1`, in a fresh worktree. The running GPU pipeline is not touched.

| Phase | Scope | Acceptance |
|---|---|---|
| P1 | <ul><li>Tag</li><li>Restructure (`git mv` rename-only commits first)</li><li>Qwen3.5 adapter</li><li>Quantize, endpoints, distill and eval stages</li><li>CLI</li><li>CPU tests</li></ul> | <ul><li>Bitwise parity with `experiments/` on a tiny Qwen3.5</li><li>Re-evaluating the existing 256-sample state gives NLL 3.2954 and ARC 13/20</li></ul> |
| P2 | <ul><li>PTQ1_0 GGUF export inside the package</li></ul> | <ul><li>Exact codes and scales after export → decode (our decoder and the sibling's)</li><li>Torch NLL unchanged</li><li>Loads in Prism's llama.cpp, if allowed</li></ul> |
| P3 | <ul><li>Dense adapter (Llama, Qwen3, Nanbeige)</li><li>Tied endpoints</li></ul> | <ul><li>Tiny random Llama, Qwen3 (tied) and Qwen3.5 run end to end on CPU in under 60 s each</li></ul> |
| P4 | <ul><li>README, docs, web copy</li></ul> | <ul><li>Quickstart runs verbatim on a tiny config</li><li>Every documented command exists</li></ul> |
| P5 | <ul><li>CAT-Q refine port</li></ul> | <ul><li>Only if it beats GPTQ + KD on GPU; otherwise it stays in research</li></ul> |

## Open items (answers requested separately)

- **License.**
- **The main checkout's uncommitted edits.**
- **Official Bonsai weights:** offered as an optional starting point, or used as a baseline only.
- **Prism's llama.cpp fork:** clone and build it to validate runtime loading.
- **Not changed without asking:**
  - remote `research/*` branches (kept);
  - pushes to GitHub (each one asked);
  - the `web/` mp4 (kept as is).
