# Method

This page explains why each stage of the pipeline exists, with the measurement that justifies it, and what we tried that did not work. All numbers are for Qwen3.5-9B, measured with the research scripts in `research/experiments/` that the package ports (see the caveats in the [README](../README.md#results)).

The metrics:
- **Dev NLL**: mean next-token negative log-likelihood on 2,048 WikiText-2 test tokens. Lower is better.
- **ARC /20, ARC /100**: ARC-Easy questions answered correctly, by the likelihood of the answer letter.
- **Wiki 16K**: NLL on 16,384 WikiText-2 test tokens.
- **Chat**: NLL on 16 held-out answers the BF16 model generated itself.

Details of the suites: [evaluation.md](evaluation.md).

## The ladder

Each row adds one step to the row above it, unless it says otherwise.

| Step | Dev NLL | ARC /20 | Wiki 16K | ARC /100 | Chat | Source |
|---|---|---|---|---|---|---|
| BF16 Qwen3.5-9B | 2.423 | 19 | 2.218 | 97 | 0.554 | dev: `research/experiments/out/quick_eval.jsonl`; Wiki 16K and ARC /100: `research/runs/2026-10-26-5090-pilot/report.md` (2.2175, 97); all three extended numbers also in `research/runs/2026-10-27-bf16-extended/` |
| Plain ternary rounding (TWN), no rotation | 10.20 | 9 | | | | same |
| Same, rotated | 9.10 | 1 | | | | same |
| Rotated, least-squares scales (RTN) | 8.94 | 4 | | | | same |
| Rotated, sequential GPTQ, 128 calibration blocks, scale refit after GPTQ | 5.15 | 8 | | | | `research/runs/2026-10-27-gptq-ternary-v1/` |
| Same, no refit | 3.32 | 9 | | | | `research/runs/2026-10-27-gptq-ternary-norefit/` |
| Same, 256 blocks | 3.295 | 13 | | | | `research/runs/2026-10-27-gptq-ternary-256-norefit/` |
| Same, one sign vector per width (Prism's layout) | 3.318 | 15 | 3.103 | 78 | 1.964 | `research/runs/2026-10-27-gptq-shared-256/` |
| + ternary head | 3.383 | 16 | 3.167 | 74 | 2.051 | `research/runs/2026-10-27-endpoints-shared-256/` |
| Instead: ternary embedding only | 3.416 | 11 | 3.218 | 59 | 2.139 | same |
| + ternary head and embedding (all ternary, ~2 GB) | 3.480 | 13 | 3.281 | 62 | 2.225 | same, and step 0 of the next row |
| + scale distillation, 4.2M tokens | **2.892** | **19** | **2.626** | **87** | **1.318** | `research/runs/2026-10-27-e2e-kd-shared-v1/log.txt` |
| + small high-precision tensors trained too | 2.884 | 19 | 2.616 | 87 | 1.308 | `research/runs/2026-10-27-e2e-kd-shared-small-v1/log.txt` |
| Official Ternary Bonsai 2 9B, for reference | 3.191 | 17 | | | | `research/runs/2026-10-27-bonsai-start-convert/eval_bf16.json` (PyTorch, unmodified official weights); 3.192 and ARC 16 in Prism's CPU runtime |

All rows round in groups of 128 weights, and keep the embedding and head in BF16 until the endpoint rows. The three rounding rows use no calibration data: "TWN" is the Ternary Weight Networks rule (threshold 0.7 × mean |w|, scale = mean |w| above it), and "RTN" searches the threshold for the least squared error. They round every decoder linear layer, `in_proj_a`/`in_proj_b` of the Gated DeltaNet layers (Qwen3.5's linear-attention layers, three of every four) included. The GPTQ rows keep those two in high precision (section 2). The ternary-endpoint rows were scored with dense dequantized weights; the "all ternary" row repeats the first evaluation of the distillation run, which uses the runtime arithmetic (the dense evaluation gave 3.479, ARC 13, Wiki 16K 3.280, ARC /100 63, chat 2.222).

## Data

Three datasets feed the pipeline, all at pinned revisions. The numbers below are those of `examples/qwen3_5-9b.yaml`.

- **Traces (AYOT).** The BF16 model answers 160 UltraChat first-turn prompts, from rows 5,000 to 5,999, with its own chat template (rows below 5,000 hold the extended suite's chat-NLL prompts). Decoding is greedy, without thinking, up to 512 tokens. The ScaleQ paper calls this "attend to your own thoughts": the compressed model is fitted on the text the model actually produces, not only on text written by people.
- **Calibration (quantize and endpoints).** 256 blocks of 2,048 tokens: 128 from WikiText-2 *train*, 96 UltraChat conversations from row 6,000 on (rendered with the chat template), and 32 from the traces.
- **Distillation.** 2,048 sequences of 2,048 tokens (4.2M tokens): 512 WikiText-2 *train*, 24 traces and 1,512 UltraChat.

`examples/qwen3_5-9b-reasoning.yaml` (measured: GSM8K 69/100 and HumanEval 19/50 with thinking off, against 30/100 and 6/50 for the generic-calibration model; this is the Hugging Face model, `research/runs/2026-10-27-reasoning-mix-v1/`) moves the data toward reasoning, as the ScaleQ paper suggests: traces from GSM8K, MATH, MBPP and OpenCodeInstruct *training* prompts answered with thinking on, some GSM8K and MBPP ones without thinking, plus UltraChat without thinking (`traces.sources`), then 128 of 256 calibration blocks and 640 of 1,536 distillation sequences from those traces.

The reasoning suites' test problems are refused everywhere (see [evaluation.md](evaluation.md#reasoning-suites-gsm8k-humaneval)). WikiText-2 *train* in the data makes our WikiText *test* numbers partly in-domain. The measured runs took their traces from a fixed set of 128 prompts, disjoint from the chat-NLL prompts; the example config's sampled prompts are not (see [evaluation.md](evaluation.md#keeping-test-data-out)).

## 1. Rotation

**What.** Every rounded weight is stored in a rotated basis: the file holds W·S·H, where S = diag(s) is a ±1 sign vector and H is the normalized Hadamard matrix in blocks of 4,096 (block-diagonal). The runtime applies H(s⊙x) to the activations instead. Since (S·H)(S·H)ᵀ = I, the product is unchanged. The format is spelled out in [formats.md](formats.md#rotation).

**Why.** A ternary group stores one scale for 128 weights, so every weight becomes 0 or ±scale. Weight rows of language models have heavy tails. A few large weights set the scale, and the many small weights round to zero. A Hadamard rotation mixes the 4,096 inputs of a block, so each rotated weight is a sum of many original weights. The values in a group come out closer to Gaussian, with fewer outliers, which three levels fit much better.

**Evidence.** On its own, rotation helps plain rounding (TWN 10.20 → 9.10) but leaves the model unusable. In the earlier CAT-Q experiments, a matched pair of runs differing only in the rotation gave dev NLL 9.26 rotated and 14.44 unrotated. Rotation is necessary, but it is not enough. Its job is to make the next step work.

**Prism's constraints.** Prism's file holds one block size and one sign vector per input width, shared by every tensor of that width. Our first GPTQ runs drew separate signs for every module. Moving to one shared vector per width cost little: +0.023 dev NLL (3.295 → 3.318), while ARC went from 13 to 15. The block is derived from the model, as the largest power of two up to 4,096 that divides every rotated width. Prism's runtime has run real models only at block 4,096 (Qwen3.5-9B and Qwen3-8B), and tiny random models at block 128 in our tests; other blocks (2,048 for Llama 3.1 8B, for example) are untested on real weights.

## 2. Sequential GPTQ ternary rounding

**What.** The model is processed one decoder layer at a time, from the bottom. Inside a layer, linear layers that read the same input are grouped, and the groups are rounded in data-flow order. In Qwen3.5's Gated DeltaNet layers the order is: `in_proj_qkv` and `in_proj_z`, then `out_proj`, then `gate_proj` and `up_proj`, then `down_proj`. For each group:

1. Run the calibration blocks through the already-rounded layers below, and collect the Hessian H = E[x xᵀ] of the group's input. Rotate it to the weights' basis.
2. Walk the rotated input columns in groups of 128. At the start of each group, choose each row's threshold δ and scale α. The candidates are δ = r · mean|w| for r = 0.30, 0.35, …, 1.20; for each, α is the mean |w| above the threshold, and the pair with the least squared error wins (the same search as the RTN row, on the weights as GPTQ has updated them so far).
3. Round column by column: code = sign(w) if |w| > δ, else 0. Push the rounding error onto the columns not yet rounded, through the Cholesky factor of H⁻¹ (GPTQ's error feedback, damping 0.01 of the mean diagonal).

**Why.** Rounding to three levels loses a lot of information. A weight matrix after rounding differs from the original by a relative squared error of about 25% (median over the 200 rounded matrices of the 9B; range 22% to 38%). GPTQ cannot make the weights closer. Instead, it moves the error into directions the layer's real inputs rarely use. Weighted by the input Hessian, which is what the layer's outputs see, the median relative error is 3.6% (range 0.1% to 10%). Rounding the layers in order means each layer's Hessian comes from inputs that already carry the error of the layers below, so later layers compensate for earlier ones.

**Evidence.** This is the largest single step: 8.94 → 3.32 dev NLL, with 128 calibration blocks. Doubling the calibration to 256 blocks of 2,048 tokens gave 3.295 and ARC 13/20. GPTQ on the 9B takes about 10 minutes on one RTX 5090.

**What stays high precision.** In Qwen3.5, `in_proj_a` and `in_proj_b` (the per-head decay and update gates of Gated DeltaNet) stay in high precision. We did not keep them for sensitivity: rounding only those two, to nearest with a rotation, changes dev NLL by less than 0.02. We keep them because Prism's file stores them as F32 (`ssm_alpha`, `ssm_beta`), and they are about 0.1% of the decoder. Convolutions, `A_log`, `dt_bias` and the norms are not linear layers and are never rounded.

**Sensitivity.** The fake-quant sweep found the sensitive parts. Fake-quant means the rounded weights are dequantized back into an otherwise ordinary BF16 model and scored there. The sweep rounded to nearest with rotation, one part of the model at a time, everything else BF16 (`research/experiments/out/`):

| Rounded alone | Dev NLL (BF16 2.423) |
|---|---|
| Gated DeltaNet `out_proj`, all layers | 4.253 |
| `gate_proj` + `up_proj` | 3.082 |
| `down_proj` | 2.638 |
| `in_proj_qkv` | 2.511 |
| `in_proj_z` | 2.487 |
| `in_proj_a` + `in_proj_b` | 2.412 |
| All of layer 0 | 5.571 |
| Layer 0's `out_proj` only | 4.013 |
| Layer 1, 2, 4, 8, 16, 24 or 30's `out_proj` only | 2.417 to 2.445 |
| First 8 layers / last 8 layers | 7.699 / 3.046 |

Layer 0's Gated DeltaNet `out_proj` alone costs +1.6 nats when rounded to nearest; the same matrix in any other layer we tested costs about 0.02 at most. Prism's format stores `out_proj` as PTQ1_0, so it cannot simply be kept in high precision. GPTQ has to handle it, and it does well enough for the results above. How to run such a scan on another model: [adding-a-model.md](adding-a-model.md#choosing-the-high-precision-tensors-a-sensitivity-scan).

## 3. ScaleQ/CAT-Q refinement (research only)

**What.** ScaleQ-1.58 ([arXiv:2608.01078](https://arxiv.org/abs/2608.01078)) combines AYOT traces with CAT-Q. CAT-Q learns a center, a threshold and a scale for each group's ternary conversion. A soft rounding sharpens into hard codes during training, and the objective is to reconstruct a block of layers' outputs. Our implementation is in `research/catq/`.

**Why it is not in the pipeline.** Before GPTQ, we ran CAT-Q alone as the quantizer at small budgets: 16 to 128 calibration samples of up to 512 tokens, with BF16 embedding and head. None of those models was usable:

| Run | Dev NLL | ARC /20 |
|---|---|---|
| First pilot / the same pipeline corrected (16 samples) | 13.26 / 11.19 | 1 / 1 |
| Rotated, 16 samples × 8 epochs (unrotated: 14.44, ARC 1) | 9.26 | 8 |
| UltraChat, rotated 16×8 / 128×1 | 8.50 / 9.12 | 8 / 8 (all answers "A") |
| Single-window control, original-prefix teacher targets | 8.28 | degenerate |
| Sequential, original-prefix targets, 128×1 | 11.85 | 8 (all answers "A") |

None of the dev generations in these runs was a usable answer: they looped, stopped at once, or wrote nonanswers. Several changes improved the local reconstruction error but not the model. Original-prefix teacher targets cut the boundary error by 33%, and the sequential run tracked the original model more closely at the windows measured, yet it had the worst NLL. Local reconstruction error did not predict model quality.

These results say CAT-Q *alone at small budgets* failed here. They say nothing about CAT-Q at the paper's settings, or as a refinement of GPTQ's codes. That combination is the next experiment. CAT-Q moves into the package only if it beats GPTQ plus distillation on the same data. The reports are in `research/runs/2026-10-26-*` and `research/docs/`.

A different refinement of GPTQ's codes, block-wise training of latent weights and scales with a straight-through estimator (EfficientQAT's Block-AP, started from GPTQ's codes), is in the package as the optional `refine` stage, off by default. On 9B it lowered the error before distillation (dev NLL 3.69 → 3.40), but after distillation its GSM8K and HumanEval scores were within noise of the model without it, so it stays off: [refine.md](refine.md#9b-results).

## Endpoints: ternary head and embedding

**What.** The LM head is rounded by GPTQ, with the Hessian of its real inputs: the final hidden states of the already-ternary decoder, on all 256 calibration blocks. The token embedding has no useful input Hessian, because its input is a one-hot vector. But an embedding row reaches the network only through layer 0's input norm, whose output is proportional to gain ⊙ e. Here gain = 1 + w for Qwen3.5's zero-centered norm, and w for the usual one. So the embedding is rounded by GPTQ with H = diag(gain²), rotated, which minimizes the error the first layer actually sees. Rows are independent, so both run in chunks of rows. Together they take about a minute.

**Why.** At 248,320 × 4,096, Qwen3.5-9B's head and embedding are 2 GB each in BF16, as large as the whole ternary model. The 2 GB target is only reachable with both ternary.

**Evidence.** The head costs little: dev NLL 3.318 → 3.383. The embedding costs more, especially on ARC: alone, ARC goes from 15 to 11 and ARC /100 from 78 to 59. With both, the model scores 3.480, ARC 13, ARC /100 62. Distillation recovers most of it (next section). A model whose embedding and head share one table (Qwen3 0.6B to 4B, Llama 3.2 1B and 3B) keeps the table in high precision: one joint GPTQ for a table that is both input and output is not implemented.

## 4. Scale distillation

**What.** The codes stay frozen. Every (row, group) scale s₀ gets a trainable log multiplier u, so the scale is s₀ · exp(u), starting at s₀. For the 9B that is 70 million parameters, all scales of all 202 ternary matrices, the endpoints included. The model runs with the runtime's arithmetic: rotated activations times codes times scales. Training minimizes the cross-entropy to the BF16 model's top-64 next-token probabilities, plus one bucket for the probability mass outside the top 64. With the bucket, the optimum is the teacher's distribution, not a sharpened version of it.

The run: a teacher pass stores the top-64 log-probabilities of 4.2M tokens (about 8 minutes, 1.6 GB on disk). Then one epoch of AdamW at learning rate 1e-3 with 10 warmup steps and a cosine schedule, 8 sequences per step: 256 steps, about 45 minutes on one RTX 5090.

**Why.** GPTQ is local. Each layer minimizes its own output error, given the layers below, but nothing tells it how errors add up through 32 layers and the head. Training the scales end to end uses the model's real objective, and it can only move the scales, so the result stays exactly representable in PTQ1_0.

**Evidence.** All ternary, 3.480 → 2.892 dev NLL, ARC 13 → 19, ARC /100 62 → 87, chat NLL 2.225 → 1.318. Most of it comes early: after 64 of 256 steps the model was at 2.911 and ARC 19. Also training the high-precision tensors (norms, `conv1d`, `A_log`, `dt_bias`, `in_proj_a`/`in_proj_b`; `distill.train.small: true`) adds little (2.884), so it is off by default.

**Where it falls short.** Chat NLL is still 0.76 nats above BF16, against 0.47 on WikiText. On the dev prompts, the distilled model gets the chickens-and-rabbits word problem wrong, and its Turkish rewrite collapses into repetition. Our distillation data is English WikiText, UltraChat and the model's own English answers, with little math. Data with the skills you want to keep was the next lever: calibrating and distilling on the model's own math and code answers (`examples/qwen3_5-9b-reasoning.yaml`) raised GSM8K to 69/100 and HumanEval to 19/50 with thinking off ([README](../README.md#results)). Data from test sets is refused.

## Export

The final state goes into Prism's PTQ1_0 GGUF: 28 bytes per 128 weights, Prism's `prism.hadamard.*` metadata, F32 norms and Gated DeltaNet tensors, and the model's tokenizer and chat template. For the 9B config the tensor layout and metadata keys equal the official Ternary-Bonsai-2-9B file, and both files are 1,997,486,976 bytes. The export then re-reads the file and checks it: every block decoded against the state, every F32 tensor bit for bit, and a function check in FP64 through the runtime's own transform. Format details: [formats.md](formats.md).

In Prism's llama.cpp (CPU, upstream `wiki.test.raw`, 8 chunks of 512 tokens), the result has perplexity 12.88, against 17.21 for the official model; the same in-domain caveat applies. (The 11.93 and 16.51 given earlier were measured on a copy of the file that drops its blank lines; `research/runs/2026-10-28-qwen3-8b-reasoning/prism/`.) Prism's runtime also matches our PyTorch reference of the same weights with a mean KL of 0.0028 nats and 95% top-1 agreement over 508 scored tokens (4 chunks of 256, WikiText-2 test), but that check ran on the exporter's synthetic test state (round-to-nearest codes with edge cases, perplexity about 13,700, so not a usable model; `research/runs/gguf-test/runtime-parity/`), not on the distilled model. On real distilled weights it has run for Qwen3-8B: 98.8% top-1 agreement over 2,040 positions, and perplexity 14.92 in Prism against 14.93 in PyTorch. Its KL is not comparable with the synthetic state's: Prism stores log-probs only within 16 nats of the top one, and that floor dominates the estimator on a confident model (`research/runs/2026-10-28-qwen3-8b-reasoning/report.md`). Prism quantizes activations to 8 bits, so the two are not identical. The perplexity logs and the export report are in `research/runs/2026-10-27-gguf-shared-v1/`.

## What did not work

**Refitting the scales after GPTQ.** After rounding, refitting each group's scale by least squares against the original rotated weights sounds like a free improvement. It made the model much worse: dev NLL 5.15 against 3.32 without it, at 128 blocks. GPTQ chooses codes that are deliberately wrong in weight space, so that they cancel errors from earlier columns under the input Hessian. A refit in weight space pulls the scales back toward the original weights and undoes that compensation. The option was removed.

**CAT-Q alone at small budgets.** Dev NLL 8.3 to 13.3, with collapsed ARC answers and unusable generations (section 3). Better local reconstruction did not mean a better model.

**Better packing.** Earlier we built BITCOS (a presence bit per weight, plus a sign bit per nonzero) and base-3 codecs. With 33% zeros, which is what Prism's own codes have (32.76% over all 202 matrices), BITCOS costs about 2 − 0.33 ≈ 1.67 bits per weight before scales, more than PTQ1_0's 1.625. On the official tensors, BITCOS would need 2,010,072,449 bytes for codes and scales against PTQ1_0's 1,957,036,032. Ideal base-3 packing would save 28 MB, 1.4%. Size was never the problem: quality was, and the target runtime fixes the format anyway. Details: `research/docs/bonsai-ternary-statistics.md`, `research/docs/base3-packing.md`.

**Large high-precision exceptions.** Early deployments kept the embedding, head and vision tower in BF16 and came to 6.6 GB, three times the target, while still failing on quality. Keeping a large, sensitive family such as the Gated DeltaNet `out_proj` in high precision is not possible in Prism's format either. The pipeline keeps only what the format stores as F32.
