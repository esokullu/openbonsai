# Refine: block-wise training of codes and scales (design)

**Status: design, not implemented.** This page specifies an optional stage `refine` that runs after quantize and endpoints and before distill. Nothing here is measured. The numbers are arithmetic from shapes, or estimates from the throughput of earlier runs, and each one says which.

## Why

Today only the scales are trained. Quantize fixes every code by GPTQ, and distill trains the per-(row, group) scales end to end with the codes frozen ([method.md](method.md#4-scale-distillation)). That gives dev NLL 2.892 (BF16 2.423, official Bonsai 3.191), but reasoning collapsed. With thinking off, GSM8K is 30/100 (Bonsai 77, BF16 77) and HumanEval 6/50 (Bonsai 45, BF16 44), in `research/runs/2026-10-27-reasoning-v1/`. The reasoning-trace data fix is running now. This stage is the second lever: it lets the codes change.

Three observations point there:
- Bonsai's codes are not naive rounding. A third-party analysis of Bonsai 1 reports about 28% sign flips against the base weights (not verified by us). In the 9B checkpoint, 96.65% of groups hold exactly 42 zeros out of 128, and none holds more ([audit](../research/docs/bonsai-ternary-statistics.md)). Both are consistent with trained latent weights, and neither proves a recipe.
- EfficientQAT has two phases. Block-AP trains the weights and the quantization parameters block by block, with a straight-through estimator (STE) and an MSE loss to the full-precision block output. E2E-QP then trains only the quantization parameters, end to end. Our distill stage already is E2E-QP, and refine is the missing Block-AP. EfficientQAT reports 2-bit Llama-2-7B in about 4.8 hours on one A100. GSQ (IST-DASLab, arXiv 2604.18556) learns codes and scales jointly on Bonsai's grid; its reported cost (about 10 hours on 8×H200 for 8B) is unverified.
- GPTQ is greedy. It picks each code once, column by column, and never revisits it.

Budget: everything fits one RTX 5090 (32 GB, Windows). When VRAM overflows, the excess spills to system RAM and runs very slowly instead of failing, so the planned peak stays well under 28 GB. No rented GPUs.

## Where it sits

The stage order becomes traces → calibrate → quantize → endpoints → **refine** → distill → export → eval. Refine is off by default (`refine.enabled: false`).

- **Input.** The quantize state (codes, fp16 scales and shared signs per width), the calibration tokens, and the ternary embedding from endpoints when endpoints made one. The ternary embedding goes into the student's path, so layer 0 sees the real input error.
- **Output.** `refine/state.safetensors` in the same `TernaryState` format, with the same keys, signs, block and group as quantize's state. Layers outside `refine.layers` are copied from quantize unchanged. Distill, export and eval read it through one accessor, `ctx.decoder_state()`: refine's state when refine is enabled, otherwise quantize's. GGUF export, the codec and the evaluators need no change. Distill then trains log multipliers starting from refine's scales.
- **The head.** Endpoints rounded the head with the Hessian of the *unrefined* decoder's final hidden states. Refine leaves the head alone, and distill retrains its scales. Re-running the head's GPTQ after refine takes about one minute. It is a follow-up, if the pilot shows the head Hessian has gone stale.

## Parametrization

Each rounded linear of the layer in training becomes a `LatentTernaryLinear`. This is a subclass of `TernaryLinear`, so the forward pass is the runtime's arithmetic, y = (C ⊙ s)(R x), with the activations rotated in BF16. It holds:

- `V` (FP32, [out, in], rotated basis): the latent weight, in units of its group's input scale. The latent weight in weight units is W_lat = S0 ⊙ V. Expressing it in scale units makes one learning rate move every group's thresholds at the same rate, even though group scales span about 3× (p01 0.0086 to p99 0.0235 in Bonsai).
- `u` (FP32, [out, in/128]): a log multiplier, as in distill. The scale is s = S0 · exp(u), starting at u = 0.
- The codes C, by one of two assignment rules:
  - **`nearest` (default).** C = clamp(round(V · e^(−u)), −1, 1). The threshold sits at s/2 in weight units. That is the least-squares assignment for a given scale: a weight is better approximated by ±s than by 0 exactly when |w| > s/2. So learning codes and scales together never fights the threshold rule. GPTQ's own thresholds δ = r · mean|w| do not matter after step 0.
  - **`quantile` (ablation, Bonsai-like).** In each (row, group), the k smallest |V| become 0 and the rest become sign(V). Step 0 is exact only if k starts at that group's GPTQ zero count, so k is annealed linearly to `zeros` (42) over the first `zero_anneal` steps. Ties in |V| are broken by the initial residual (see below).
- **STE.** The weight is computed as q = s ⊙ (C + (V − V.detach())), then cast to BF16 as `TernaryLinear.weight_deq` does. The added term is exactly zero in the forward pass, so the forward pass is bit for bit the hard ternary layer. In the backward pass, ∂L/∂V = s ⊙ ∂L/∂q (identity STE), and ∂L/∂u = Σ_group s ⊙ C ⊙ ∂L/∂q, which is distill's scale gradient. After each step V is clamped to [−1.5, 1.5], so no latent drifts far beyond the outer threshold and goes dead.
- **Initialization.** `center` (default): V = C0 exactly, and u = 0. Then round(V) = C0, s = S0 and q is quantize's weight, so step 0 reproduces the input state bit for bit. `original`: V = C0 + clip((W_rot − C0 · S0)/S0, ±0.25), which puts latents nearest the threshold where the original weights disagree with GPTQ. The codes still round to C0, because |residual| < 0.5. This option is an ablation. The scale refit taught us that pulling toward the original weights can undo GPTQ's error compensation ([what did not work](method.md#what-did-not-work)).
  - We considered starting from GPTQ's error-updated weights and dropped it for v1. Quantize does not keep them (28 GB at 9B in FP32). Recomputing them needs a GPTQ re-run on a prefix that refine has already changed, so step 0 would match the re-run, not the input state.
- **Code freezing.** After `freeze_codes_after` × (the layer's steps), C is fixed and V is freed. The remaining steps train only u: a local, hard-code phase that stops STE oscillation at the end (Nagel et al., 2022, freeze oscillating weights). Setting 0 gives scale-only refinement, and 1 never freezes.

## Inputs and targets

Two activation streams per calibration block, each BF16 [2048, 4096] (16 MiB per block):
- **Student input** X_q^i: the hidden states entering layer i through the refined layers 0..i−1, with the ternary embedding if there is one. As in GPTQ, later layers see, and can compensate for, the earlier layers' final codes.
- **Target** T^i, chosen by `refine.target`:

| `target` | Student | Target | Extra cost |
|---|---|---|---|
| `original` (default) | L_i^q(X_q^i) | L_i(X_fp^i), where X_fp^i passes through the original layers 0..i−1 | One forward of the original layer on the FP stream (the FP stream is advanced anyway) |
| `quantized` | L_i^q(X_q^i) | L_i(X_q^i), the original layer i on the student's input | None |
| `mix` | both | (1 − `mix`) · original loss + `mix` · quantized loss (OmniQuant's auxiliary loss) | As `original` |

**Default `original` (asymmetric, GPTAQ-style).** The reasons:
1. **Literature.** GPTAQ (formerly GPTQv2) derives the asymmetric objective, ‖W X_fp − Q X_q‖, and reports gains over GPTQ's symmetric one at low bit widths. EfficientQAT's Block-AP, the closest published recipe to this stage, keeps a full-precision stream for targets and a quantized stream for inputs. That is how we read their released code, following OmniQuant; we have not re-checked it offline for this note. SliderQuant's default base loss does the same ([source links](../research/docs/teacher-target-control.md#what-the-source-establishes)).
2. **Our one controlled comparison on pretrained 9B weights.** It was a single-variable warm-start repair of window [16, 17), with 128 updates in each arm and only the target changed. The original-prefix target gave 32.8% lower common-target MSE and dev NLL 8.28 against 9.12 for the quantized-prefix target ([teacher-target control](../research/docs/teacher-target-control.md#completed-pretrained-single-window-control)).
3. **Mechanism.** The quantized target asks layer i to copy itself on drifted inputs. GPTQ already minimizes exactly that, since its Hessian comes from the quantized prefix, so refine with that target could only improve GPTQ's greedy search within the same objective. Error made early is the expensive kind, and it propagates. With round-to-nearest, the first 8 layers rounded alone give dev NLL 7.70 against 3.05 for the last 8, and layer 0's `out_proj` alone costs +1.6 nats ([sensitivity](method.md#2-sequential-gptq-ternary-rounding)). Only the original target lets a later layer pay some of that back.

**Evidence against.** The sequential CAT-Q run with original-prefix targets aligned better (cosine at the last boundary 0.20 → 0.55) yet had worse NLL: 11.85 against 9.12 ([result](../research/docs/sequential-original-target-result.md)). Both arms of that recipe were unusable (soft ternarization from BF16 at 128 records × 1 epoch), so it does not decide between the two targets. It does repeat the standing lesson: local error does not predict model quality. So the pilot runs `quantized` as a control arm and decides on NLL and reasoning, not on MSE. The default is provisional until then. Two safeguards are built in:
- **Held-out blocks.** Each layer is measured on `val_blocks` calibration blocks not used to train it.
- **Rollback.** A layer whose held-out error against its target is not lower than at step 0 keeps its input codes and scales, and the report says so (`rollback: true`).

At layer 0 the two targets coincide, unless the embedding is ternary.

## Loss and optimizer

- **Loss: per-token relative error.** For each token t, L = mean over t of ‖y_t − T_t‖² / max(‖T_t‖², 1e-6 · mean over t of ‖T_t‖²). The outputs are cast to FP32 before the subtraction, and per-micro-batch sums are accumulated in FP32. Reported metrics are summed in FP64.
- **Why not plain MSE.** The old CAT-Q objective was absolute MSE or Huber. At window 0 the MSE was about 1e-4, and at depth 31 it was 6 to 18. Gradients that small at shallow depth were damped by Adam's epsilon: going from 1e-8 to 1e-9 alone cut the window-0 error by 27% ([epsilon result](../research/docs/window0-epsilon-result.md)). A per-token relative loss is about the size of the relative error at every depth. It also stops the few massive-activation tokens, whose norms are orders of magnitude above the rest, from dominating the sum. The floor is relative, so no absolute epsilon is introduced.
- **AdamW.** β = (0.9, 0.95), weight decay 0, `eps` 1e-10. Decay would pull V toward 0, and so toward code 0. Each layer starts a fresh optimizer, with 5% warmup and a cosine schedule.
- **Checking epsilon.** By our estimate, gradients on V are around 1e-8 (relative-loss gradient times activation times scale). That is two orders of magnitude above `eps`. The trainer logs, per layer, the fraction of V coordinates with √v̂ < 10 · eps, so this is checked rather than assumed. More than 1% is a defect.
- **`lr_codes` 5e-3 (units of scale).** Adam moves a coordinate by at most about lr per step. At N = 128 and E = 4, with one block per step, a layer gets 512 steps, and 410 of them come before codes freeze at 0.8. Under the cosine schedule (average factor 0.62 over those steps), a latent pushed the same way every step travels up to about 1.3. That is enough to cross one threshold (distance 0.5). A latent whose gradient is pure noise random-walks about 20 · lr ≈ 0.1 (β1 = 0.9), so noise rarely flips a code. As a cross-check, EfficientQAT's weight learning rate (2e-5 for 2-bit, as we recall it) is about 0.7e-3 to 1.4e-3 per step in scale units, for 2-bit steps of about 0.014 to 0.03. Block-AP takes about 8× more steps per block than our default, so 5e-3 over 512 steps allows a total travel of the same order.
- **`lr_scales` 5e-4** for u. Distill (1e-3, end to end) ended with a mean |u| of 0.013.

## Per-layer procedure

1. **Setup, once.** Load the model, capture the layer-0 inputs and each layer's kwargs (rotary tables and masks) at `refine.batch` with quantize's `capture_layer0_inputs` and `capture_layer_kwargs`, then move the decoder layers to the CPU. The GPU holds one layer at a time (two with `window: 2`).
   - **Calibration subset.** Training uses a seeded subset of `blocks` calibration blocks, stratified by source, plus `val_blocks` more. The calibration file is ordered by source, so a prefix would be all WikiText. The reasoning config's file is half traces, and the subset keeps that share.
2. **Target pass.** Move original layer i to the GPU and compute T^i under `no_grad`. In `original` mode, T^i overwrites X_fp^i in place and becomes X_fp^(i+1). The held-out blocks always get both targets, so every arm is also scored against the common original target, as in the teacher-target control.
3. **Student.** Copy the layer, install `LatentTernaryLinear` for its planned linears (V, u from the current state), and keep every other tensor frozen at its original value: norms, conv1d, `in_proj_a`/`in_proj_b`, `A_log` and `dt_bias`. Distill's `train.small` can train those later.
4. **Train.** Train for E epochs over the training blocks in a seeded order, one optimizer step per `batch` blocks. Freeze the codes at `freeze_codes_after`.
5. **Harden.** Codes are the final C. Scales are (S0 · e^u) in fp16, as distill stores them. Measure the held-out error and apply rollback.
6. **Save.** Write `refine/layers/NNNN.safetensors` atomically, a `TernaryState` with that layer's stats in its meta: errors before and after against both targets, code flips (sign ±1↔∓1 and zero↔nonzero, against the input codes), zero fraction (Bonsai: 32.76%), eps-damped fraction, steps, seconds and peak VRAM.
7. **Advance.** Compute X_q^(i+1) = L_i^q(X_q^i) with the hardened layer, in place, block by block. Free the layer.

**Window 2** trains layers (i, i+1) together against the target at i+1's output, with gradient checkpointing per layer. The stride is 2, so windows do not overlap.

**Resume.** A finished layer is never retrained, as in quantize. On restart, the streams are rebuilt by pushing the calibration blocks through the saved layers 0..k−1 and the original layers 0..k−1. That is 2k forward passes, under two minutes at k = 31 (estimate). A crash inside a layer loses only that layer, under a minute. No stream files are kept between runs.

## Memory and time on one RTX 5090

**Parameters (arithmetic from Qwen3.5-9B's shapes).**
- **Gated DeltaNet layers (24 of 32): 218,103,808 rounded weights each.** `in_proj_qkv` 8192×4096, `in_proj_z` and `out_proj` 4096×4096, MLP 3 × 12288×4096.
- **Full-attention layers (8): 209,715,200 each.** `q_proj` 8192×4096 (query plus gate), `k_proj`/`v_proj` 1024×4096, `o_proj` 4096×4096, plus the same MLP.
- **Check.** Together with the 248,320×4096 embedding and head, that is 8,946,450,432 weights, exactly the official checkpoint's count.
- **Training state: 16 bytes per weight.** V, its gradient and Adam's two moments, all FP32.

**VRAM, window 1, N = 128 with the streams on the GPU** (activations are an estimate):

| Item | GB |
|---|---:|
| V, its gradient, Adam m and v (FP32) | 3.49 |
| u, its gradient and Adam moments; C0 (int8, for flip stats); S0 | 0.26 |
| Original layer in BF16 (target pass) | 0.44 |
| BF16 weight saved for the backward pass, plus a transient gradient (largest linear) | 0.54 |
| Activations of one 2,048-token block, no checkpointing (estimate) | ≤ 1.5 |
| Streams X_q and T (2 × 2.1), plus 8 held-out blocks × 3 streams | 4.7 |
| CUDA context, allocator slack | ~1.5 |
| **Total** | **~12.5** |

- **Window 2** adds about 4.5 GB, for about 17 GB.
- **Streams.** In `original` mode only two streams are resident: T^i overwrites X_fp^i, and X_q^(i+1) overwrites X_q^i. That is 2 × N × 16 MiB in all. `cache: auto` keeps them on the GPU while the total stays under 20 GB, otherwise in pinned CPU memory, otherwise on disk (a store like `DiskRecords` in [the CAT-Q trainer](../research/catq/trainer.py)).
- **This machine** has 256 GiB of RAM: the BF16 model on the CPU (about 18 GB) plus streams up to N = 4096 (128 GiB) fit.
- **The GPU's peak** stays near 8 GB once the streams are off it.

**Time (estimate).** The estimate rests on two throughput figures from earlier runs:
- **Training: 45K tokens/s per layer.** Distill ran at 1,590 tokens/s for one 2,048-token sequence with forward, backward and recompute through 32 layers plus the head (`research/runs/2026-10-27-e2e-kd-shared-v1/log.txt`). That is about 58K tokens/s for one layer's share of the matmul weights. We take off 25% for the STE and an AdamW step over 218M FP32 weights every 2,048 tokens. Distill's figure already includes the PyTorch fallback for Gated DeltaNet, since `fla` is not installed.
- **Forward only: 150K tokens/s per layer.** GPTQ took 17 s per layer for five forward passes over 256 blocks plus its own arithmetic (`research/runs/2026-10-27-gptq-shared-256/log.txt`).

Per layer: (E · N · 2048) / 45K + 3 · (N · 2048) / 150K, plus about 3 s fixed.

| Calibration | Steps per layer | Per layer | 32 layers |
|---|---:|---:|---:|
| N = 128, E = 2 | 256 | ~20 s | ~11 min |
| N = 128, E = 4 (default) | 512 | ~32 s | ~17 min |
| N = 128, E = 8 | 1,024 | ~55 s | ~29 min |
| N = 512, E = 4 (streams in CPU RAM) | 2,048 | ~2 min | ~1 h |
| N = 4,096, E = 2 (Block-AP's budget as we recall it) | 8,192 | ~9 min | ~4.8 h |

Add about 3 minutes of setup (load and capture). Read every row as within a factor of 0.7 to 2, because the PyTorch Gated DeltaNet backward is the least known part. All rows fit the card. The default is deliberately small: grow N only if the pilot shows the gain grows with it.

## Configuration

Defaults as proposed. None of these keys exists yet.

```yaml
refine:
  enabled: false
  layers: null              # [start, stop) of decoder layers; null = all
  window: 1                 # 1, or 2 layers trained together (stride = window)
  blocks: 128               # training blocks: seeded subset of calibrate's, stratified by source
  val_blocks: 8             # held out: per-layer metrics and rollback
  seed: 0
  batch: 1                  # blocks per optimizer step (kwargs are captured at this batch size)
  epochs: 4
  target: original          # original | quantized | mix
  mix: 0.5                  # weight of the quantized-target term when target: mix
  assign: nearest           # nearest (threshold s/2) | quantile
  zeros: 42                 # zeros per 128-weight group, assign: quantile
  zero_anneal: 0.25         # fraction of steps to move from GPTQ's zero count to `zeros`
  init: center              # center (V = C0) | original (clipped residual of the rotated weights)
  lr_codes: 5.0e-3          # V, in units of the group's input scale
  lr_scales: 5.0e-4         # log multipliers
  warmup: 0.05
  freeze_codes_after: 0.8   # fraction of steps; 0 = scales only, 1 = never freeze
  eps: 1.0e-10
  checkpoint: auto          # gradient checkpointing per layer: on for window 2 or batch > 2
  cache: auto               # cuda | cpu | disk for the activation streams
  rollback: true
```

The stage's hash covers `rotation` and `refine`, and its upstream is quantize, calibrate, and endpoints when the embedding is ternary. Distill's upstream and its recorded `base` digests gain refine when refine is enabled, so a changed refine makes distill stale.

## Code changes

- **`openbonsai/core/ternary.py`:** `LatentTernaryLinear(TernaryLinear)`, with the assignment rules, the STE, `freeze()`, and `hard()`, which returns (int8 codes, fp16 scales).
- **`openbonsai/stages/refine.py` (new):** the stream store (GPU, pinned CPU or disk), the per-layer loop and the training loop, per-layer save and resume, and `report.json`. Capture and the layer forward come from `stages/quantize.py`.
- **`openbonsai/config.py`:** the `refine` defaults, `refine` added to `STAGES` after endpoints, `stage_inputs`, and `upstream`.
- **`openbonsai/stages/common.py`:** `decoder_state()`, `state_through("refine")`, and a refine digest in `distill_base()`. Distill switches from `quantize_state()` to `decoder_state()`.
- **Eval:** `refine` becomes a `--from` source.
- **Tests:** `tests/test_refine.py`.

## Acceptance (tiny models, CPU)

On the tiny Qwen3.5 and dense fixtures, with `CUDA_VISIBLE_DEVICES=-1`:
1. **Step 0 is the identity.** With `epochs: 0` (or both learning rates 0), `refine/state.safetensors` holds bit-for-bit the quantize state's codes and scales. Before training, each student layer's output equals `TernaryLinear`'s output bitwise. Under `assign: quantile` this holds at the first step, before annealing starts.
2. **Loss decreases.** The held-out relative error after training is below step 0 for the mean over layers, and no rolled-back layer gets worse.
3. **Codes change.** With a learning rate large enough for the fixture, some codes flip, and the flip stats in the layer meta are nonzero and match a direct comparison. With `freeze_codes_after: 0`, the codes equal the input codes and only the scales differ. Under `assign: quantile`, every group ends with exactly `zeros` zeros.
4. **Resumable per layer.** Stopped after layer k and rerun, refine produces a `state.safetensors` byte-identical to an uninterrupted run's, the same check `tests/test_quantize.py` makes for quantize.
5. **Downstream unchanged.** Distill trains on the refine state, and its scales record refine's digest. Export writes a GGUF that passes its own verify. Eval scores `--from refine`. A changed `refine` setting makes refine, distill, export and eval stale.

## 9B pilot

Run it after the reasoning-mix job finishes, in a new run directory. Copy that run's calibrate, quantize and endpoints outputs into it, and do not touch the running job's directories.

- **Scope.** `refine.layers: [0, 8]`. The first 8 layers are where rounding hurts most (7.70 against 3.05 for the last 8). Layers 8 to 31 keep their GPTQ codes. Each arm costs about 7 minutes of refine (estimate).
- **Arms.** A0: no refine (the reasoning-mix state as it is). A1: `target: original`, defaults otherwise. A2: `target: quantized`. Optional A3, only if A1 or A2 wins: `assign: quantile`, or E = 8.
- **Before distill (minutes per arm).** Dev NLL and Wiki 16K, the `compare` KL against BF16, and GSM8K limited to 50 problems with thinking off (`eval.reasoning.gsm8k.n`). Per-layer held-out error against the common original target, flips, zero fraction, eps-damped fraction, seconds and peak VRAM, from the layer metas.
- **Gate.** The best arm must beat A0 by at least 0.05 dev NLL before distill, with a lower KL. GSM8K before distill is likely near the floor and is only a tiebreaker: 50 problems have a standard error of about ±3.
- **If it passes.** Refine all 32 layers with the winning target (about 17 min), then distill (about 1 h) and the full reasoning suites: GSM8K 100 and HumanEval 50 with thinking off, against the reasoning-mix distilled result. Only that comparison decides whether refine stays in the pipeline.

## Risks and open questions

- **Overfitting the calibration set** under `original` targets: the per-layer held-out blocks and rollback catch it locally, and the 9B pilot globally.
- **STE oscillation near thresholds**, which wastes steps: freezing the codes for the last 20% bounds the damage, and the flip statistics show how much oscillation there is.
- **The head Hessian goes stale** after refine. Distill retrains the head's scales; re-running the head's GPTQ is cheap if needed.
- **The time estimates** rest on distill's and GPTQ's throughput, not on a refine run. The first pilot arm measures them, and this page is corrected afterwards.
