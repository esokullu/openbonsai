# Qwen3.5-9B quantization diagnosis and controlled recovery

This is a small diagnostic experiment on the original official BF16 checkpoint, without task fine-tuning. The tables below use **2,048 scored WikiText tokens, 20 ARC-Easy questions, and four generation prompts**. They must not be presented as the complete pilot's 16,384 tokens, 100 questions, and ten prompts.

The full pilot measured BF16 NLL 2.217540 / ARC 97/100, and packed NLL 12.699258 / ARC 23/100. Its packed outputs were nonblank in 0/10 cases. Full results remain in [the preserved pilot report](../2026-10-26-5090-pilot/report.md).

## Where quality changes

| Model state | NLL | ΔNLL vs BF16 | Perplexity | ARC-Easy | ΔARC vs BF16 | Unfinished | Nonblank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original BF16 | 2.423777 | +0.000000 | 11.2884 | 19/20 (95%) | +0 pp | 2/4 | 4/4 |
| Wrappers, quantization disabled | 2.423777 | +0.000000 | 11.2884 | 19/20 (95%) | +0 pp | 2/4 | 4/4 |
| Hard quantization, FP32 learned factors | 13.256495 | +10.832718 | 571,771.5683 | 1/20 (5%) | -90 pp | 4/4 | 0/4 |
| Export representation, FP16 scales | 13.257692 | +10.833916 | 572,456.6800 | 1/20 (5%) | -90 pp | 4/4 | 0/4 |
| Original pilot artifact, fresh reload | 13.257692 | +10.833916 | 572,456.6800 | 1/20 (5%) | -90 pp | 4/4 | 0/4 |

NLL is next-token loss, weighted by the number of scored nonpadding tokens; lower is better. ARC uses option-label conditional likelihood, not generated answers. Nonblank only means decoded text contains a non-whitespace character; it is not a correctness, intelligibility, or relevance score. Unfinished means the 128-token cap was reached without EOS.

Installing disabled wrappers gave exact fixed-prompt tensors: **True** across 200 projections. The hard FP32-factor stage changes NLL by +10.832718 nats versus disabled wrappers. Converting that stage to FP16 exported scales adds +0.001198 nats; fresh reload adds +0.000000 nats. Export-reference versus fresh-reload fixed-prompt tensor equality is **True**.

The original pilot did not save its learned FP32 factors. Those factors were reconstructed in a separate deterministic replay using the original calibration data and training settings; the original checkpoint and pilot files were preserved. Replay packed codes/scales/biases match all 200 original pilot projection payloads exactly: **True**. This verifies reproduction of the saved payload; it does not establish bitwise equality of the unavailable original FP32 factors.

All 512 original training events were compared with 512 replay events: losses exact **True**, gradient norms exact **True**, and window/step/soft-or-hard stage identities exact **True**. These are recorded per-step training comparisons, not just equality of summary losses.

## Projection and depth controls

These are forward-pass controls using factors fitted in the fully quantized training pipeline; they are not independently retrained alternative projection policies. Each row starts from original BF16 weights plus the replayed learned factors. `only_*` quantizes that family/projection; `retain_*` keeps that family floating point and quantizes the other selected projections. `first_N_layers` quantizes only the first N decoder layers. Generation was deliberately skipped for these 16 controls.

| Model state | NLL | ΔNLL vs BF16 | Perplexity | ARC-Easy | ΔARC vs BF16 | Unfinished | Nonblank |
|---|---:|---:|---:|---:|---:|---:|---:|
| only_mlp | 10.741537 | +8.317760 | 46,237.0454 | 8/20 (40%) | -55 pp | not run | not run |
| only_linear_attention | 10.241759 | +7.817982 | 28,050.4170 | 2/20 (10%) | -85 pp | not run | not run |
| only_full_attention | 4.390061 | +1.966284 | 80.6453 | 8/20 (40%) | -55 pp | not run | not run |
| retain_mlp | 13.149479 | +10.725703 | 513,743.5305 | 4/20 (20%) | -75 pp | not run | not run |
| retain_linear_attention | 12.876343 | +10.452567 | 390,953.2820 | 8/20 (40%) | -55 pp | not run | not run |
| retain_full_attention | 10.651272 | +8.227495 | 42,246.2795 | 1/20 (5%) | -90 pp | not run | not run |
| only_gate_proj | 6.270979 | +3.847202 | 528.9949 | 13/20 (65%) | -30 pp | not run | not run |
| only_up_proj | 3.346235 | +0.922458 | 28.3956 | 16/20 (80%) | -15 pp | not run | not run |
| only_down_proj | 4.003182 | +1.579405 | 54.7721 | 18/20 (90%) | -5 pp | not run | not run |
| only_in_proj_qkv | 2.820819 | +0.397042 | 16.7906 | 16/20 (80%) | -15 pp | not run | not run |
| only_in_proj_z | 2.954871 | +0.531095 | 19.1993 | 10/20 (50%) | -45 pp | not run | not run |
| only_out_proj | 7.324950 | +4.901173 | 1,517.6980 | 9/20 (45%) | -50 pp | not run | not run |
| first_4_layers | 6.381065 | +3.957288 | 590.5562 | 7/20 (35%) | -60 pp | not run | not run |
| first_8_layers | 8.830579 | +6.406803 | 6,840.2494 | 4/20 (20%) | -75 pp | not run | not run |
| first_16_layers | 11.363617 | +8.939840 | 86,130.2998 | 2/20 (10%) | -85 pp | not run | not run |
| first_24_layers | 11.255560 | +8.831783 | 77,308.5670 | 5/20 (25%) | -70 pp | not run | not run |

Layer drift below compares the fully hard-quantized model to BF16 on the first frozen NLL segment. It measures accumulated hidden-state differences, not an isolated intervention on each layer.

| Layer | BF16 RMS | Hard RMS | Relative RMSE | Cosine |
|---:|---:|---:|---:|---:|
| 0 | 0.109256 | 0.048527 | 0.612134 | 0.926427 |
| 1 | 0.138384 | 0.063703 | 0.609563 | 0.913164 |
| 2 | 0.134567 | 0.057917 | 0.686874 | 0.829046 |
| 3 | 0.208859 | 0.117640 | 0.563168 | 0.888202 |
| 4 | 0.230460 | 0.109858 | 0.644996 | 0.851228 |
| 5 | 0.270523 | 0.117059 | 0.675687 | 0.844618 |
| 6 | 0.322410 | 0.118533 | 0.741370 | 0.796601 |
| 7 | 0.349093 | 0.150991 | 0.708445 | 0.792356 |
| 8 | 0.360049 | 0.162861 | 0.732243 | 0.739055 |
| 9 | 0.395197 | 0.192582 | 0.707485 | 0.756306 |
| 10 | 0.449615 | 0.148893 | 0.816971 | 0.667812 |
| 11 | 0.472161 | 0.234864 | 0.727280 | 0.722381 |
| 12 | 0.472305 | 0.227011 | 0.756786 | 0.684940 |
| 13 | 0.506555 | 0.220090 | 0.782115 | 0.664196 |
| 14 | 0.551763 | 0.189959 | 0.894969 | 0.461227 |
| 15 | 0.574253 | 0.465881 | 0.977867 | 0.432644 |
| 16 | 0.569601 | 0.425748 | 1.012940 | 0.356311 |
| 17 | 0.618480 | 0.385694 | 1.046686 | 0.235199 |
| 18 | 0.711766 | 0.365029 | 1.101067 | 0.049397 |
| 19 | 0.817758 | 0.412876 | 1.089080 | 0.068147 |
| 20 | 0.931687 | 0.418391 | 1.035449 | 0.144191 |
| 21 | 1.058579 | 0.448298 | 0.988548 | 0.238624 |
| 22 | 1.243060 | 0.470124 | 0.997469 | 0.195777 |
| 23 | 1.391988 | 0.623252 | 0.948407 | 0.336116 |
| 24 | 1.571925 | 0.687124 | 0.934634 | 0.363197 |
| 25 | 1.717891 | 0.760634 | 0.912940 | 0.409457 |
| 26 | 1.833795 | 0.713545 | 0.937765 | 0.349511 |
| 27 | 2.083613 | 0.929807 | 0.917031 | 0.401331 |
| 28 | 2.392465 | 1.106300 | 0.899276 | 0.438069 |
| 29 | 2.628025 | 1.219839 | 0.905485 | 0.426086 |
| 30 | 2.821847 | 1.194261 | 0.964422 | 0.294162 |
| 31 | 3.671063 | 1.638226 | 1.033607 | 0.146542 |

The loss increase and whitespace-only outputs are already present in hard quantization with FP32 factors, before scale casting, export, reload, or BITCOS repacking. Those later storage steps cannot explain the origin of this observed collapse. The ablations characterize sensitivity within the hard-quantized model; they do not establish that a larger training budget will fix it.

## Fixed-code least-squares control

This is an explicit method modification: per-group scales are fitted analytically against the original frozen projection weights, while the learned ternary codes stay fixed. It uses no additional calibration examples, optimizer steps, held-out fitting, or hyperparameter sweep. It is not a reproduction claim for CAT-Q/ScaleQ quality results.

| Model state | NLL | ΔNLL vs BF16 | Perplexity | ARC-Easy | ΔARC vs BF16 | Unfinished | Nonblank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original failed pilot, reloaded | 13.257692 | +10.833916 | 572,456.6800 | 1/20 (5%) | -90 pp | 4/4 | 0/4 |
| Fixed-code LS, FP32 scales | 11.247218 | +8.823441 | 76,666.3335 | 4/20 (20%) | -75 pp | 4/4 | 4/4 |
| Fixed-code LS, exported and reloaded | 11.247840 | +8.824063 | 76,714.0056 | 4/20 (20%) | -75 pp | 4/4 | 4/4 |

Weight-energy-weighted relative RMSE changes from 0.526417 to 0.475248. All final ternary codes remain exact: **True**; fixed-code fallback was used for 108 projection(s). Original replay factor file unchanged: **True**. All 200 exported payloads exact: **True**.

FP32-to-FP16 scale casting changes fixed-prompt logits by at most 0.40234375; fresh reload versus the FP16 reference is bitwise exact: **True**, maximum difference 0. Serialization agreement establishes correct transport of this representation, not preserved BF16 quality.

The preregistered small-subset gate requires at least 2.0 nats NLL reduction and 50% ARC accuracy, plus four intelligible, relevant, nondegenerate answers. Measured packed recovery reduction is 2.009853 nats, ARC 20%, and nonblank outputs 4/4; the numerical gate **fails**. Nonblank counts cannot certify the qualitative gate. A passing small diagnostic result would still require the complete held-out comparison before any 128-sample run. No larger run is launched by this report builder, and increasing calibration size or epochs is not assumed to repair the failure.

Recorded recovery decision (`recovery-decision.json`):

```json
{
  "status": "completed",
  "reviewed_utc": "2026-10-26T03:27:30.809491+00:00",
  "decision": "stop_after_diagnosis_and_repack; do_not_launch_128_sample_run",
  "larger_run_launched": false,
  "full_recovery_confirmation_run": false,
  "reason": "The predeclared ARC and qualitative gates fail. The NLL improvement alone does not establish useful recovery.",
  "frozen_gate_file": "recovery-plan.json",
  "gate_sha256": "677f73ff5e54215f4c38d208b6709d80d9fcc896207bc8522282fedd75a7e733",
  "nll_improvement_nats": 2.009852771036094,
  "nll_gate_passed": true,
  "arc_correct": 4,
  "arc_count": 20,
  "arc_accuracy": 0.2,
  "arc_gate_passed": false,
  "qualitative_gate_passed": false,
  "all_gate_conditions_passed": false,
  "generation_reviews": [
    {
      "id": "manual-heldout-00",
      "category": "instruction",
      "generated_tokens": 128,
      "unfinished": true,
      "intelligible": false,
      "task_relevant": false,
      "nondegenerate": false,
      "assessment": "Repetitive unrelated Chinese function characters (primarily 的/了), followed to the 128-token cap; does not answer the prompt.",
      "text_excerpt": "的了的了 的了的的的的的的的的的的的的的的的的的"
    },
    {
      "id": "manual-heldout-02",
      "category": "reasoning",
      "generated_tokens": 128,
      "unfinished": true,
      "intelligible": false,
      "task_relevant": false,
      "nondegenerate": false,
      "assessment": "Repetitive unrelated Chinese function characters (primarily 的/了), followed to the 128-token cap; does not answer the prompt.",
      "text_excerpt": "的的 的 的 的的的的的的的的的的的的的的的的的"
    },
    {
      "id": "manual-heldout-04",
      "category": "coding",
      "generated_tokens": 128,
      "unfinished": true,
      "intelligible": false,
      "task_relevant": false,
      "nondegenerate": false,
      "assessment": "Repetitive unrelated Chinese function characters (primarily 的/了), followed to the 128-token cap; does not answer the prompt.",
      "text_excerpt": "了了了 了 了 了 的了的的的的的的的的的的的的"
    },
    {
      "id": "manual-heldout-07",
      "category": "turkish",
      "generated_tokens": 128,
      "unfinished": true,
      "intelligible": false,
      "task_relevant": false,
      "nondegenerate": false,
      "assessment": "Repetitive unrelated Chinese function characters (primarily 的/了), followed to the 128-token cap; does not answer the prompt.",
      "text_excerpt": "的了了的了了的了的了的的的的的的的的的的的的的的"
    }
  ],
  "limitations": [
    "This diagnostic subset has now informed diagnosis and is not a new independent confirmation set.",
    "Fixed-code least-squares scale recovery is an explicit method modification with zero additional calibration or optimizer steps.",
    "The original failed artifact and complete pilot results remain preserved; BITCOS parity carries those results over.",
    "No evidence establishes that more calibration samples or epochs alone will repair this failure."
  ]
}
```

Recovery artifact size, measured from saved safetensors headers and directory files:

| Scope | Original pilot bytes | LS recovery bytes |
|---|---:|---:|
| Compressed codes | 1,728,053,248 | 1,728,053,248 |
| Compressed scales | 108,003,328 | 108,003,328 |
| Retained floating-point tensors | 4,995,201,904 | 4,995,201,904 |
| Total tensor payload | 6,831,258,480 | 6,831,258,480 |
| Containers and metadata | 525,007 | 479,126 |
| Model artifact | 6,831,783,487 | 6,831,737,606 |
| Tokenizer | 19,998,761 | 19,998,761 |
| Total deployable directory | 6,851,782,248 | 6,851,736,367 |

The recovery keeps the same 6,831,258,480-byte tensor payload and tokenizer size. Its 45,881-byte smaller artifact is entirely a container/metadata difference, not additional weight compression. The LS control changes scale values, not their packed storage format.

The equivalent original scope contains 18,819,619,808 BF16 tensor bytes plus 15,360 other floating-point tensor bytes. The recovery tensor-payload ratio is 2.7549×, or 63.7014% reduction, using all equivalent original tensor bytes as the denominator. The loader does not use 15 original MTP tensors totaling 486,581,248 bytes; they are excluded from this equivalent-scope denominator. No Hugging Face cache duplication is counted.

## BITCOS storage and integration

BITCOS repacks the existing failed pilot's ternary codes losslessly. It does not retrain, change scales, or repair quality. Artifact and deployment totals below include the retained floating-point components; they are not projection-only compression claims.

| Scope | Original 2-bit artifact bytes | BITCOS artifact bytes |
|---|---:|---:|
| Packed codes and scales | 1,836,056,576 | 1,565,559,851 |
| Retained floating-point tensors | 4,995,201,904 | 4,995,201,904 |
| Model artifact, including metadata | 6,831,783,487 | 6,561,349,799 |
| Tokenizer assets | 19,998,761 | 19,998,761 |
| Total deployable directory | 6,851,782,248 | 6,581,348,560 |

The model artifact saves 270,433,688 bytes (3.9585% of the original packed model artifact); the complete directory saves 270,433,688 bytes (3.9469% of the original deployable directory). BITCOS stream bytes: {"presence": 864026624, "scales": 108003328, "signs": 593529899}.

Verified projections: 200; retained tensors: 564. Exact decoded codes/scales/biases: **True**; exact retained tensors: **True**; tokenizer assets exact: **True**; all manifest checksums verified: **True**. Configuration/training metadata equal: **True**. CUDA BF16 fixed-input logits equal: **True**, maximum difference 0. This fixed-input parity check is separate from a quality evaluation.

Retained tensor accounting:

| Retained component | Bytes |
|---|---:|
| output head | 2,034,237,440 |
| token embeddings | 2,034,237,440 |
| text normalization | 546,816 |
| recurrent dynamics | 3,072 |
| recurrent convolution | 1,572,864 |
| recurrent gates | 12,582,912 |
| rotary buffers | 256 |
| vision attention | 286,903,296 |
| vision mlp | 535,781,088 |
| vision normalization | 248,832 |
| vision merger | 80,238,080 |
| vision patch embedding | 3,541,248 |
| vision position embedding | 5,308,416 |
| vision rotary buffers | 144 |

The reference runtime reconstructs dense weights for computation. Storage reduction, inference memory, and speed are separate measurements; BITCOS does not provide an optimized low-bit kernel.

The preserved pilot repository commit is `891d260af96a7ce61ccbcb23474015d975bb18dc`. The isolated integration starts at upstream commit `997b9c3fa8788bf31b12b1cecb713655a7953361` in `D:\openbonsai\runs\2026-10-26-5090-diagnosis\bitcos-checkout`. The recorded source snapshot is `D:\openbonsai\runs\2026-10-26-5090-diagnosis\sources\bitcos-integrated-parity-fix` with 21 hashed source files in `bitcos-integrated-sources.json`; `bitcos-integration.patch` includes the local changes and new files. The original pilot preservation check compares 231 files: all unchanged **True**.

The latest completed preservation audit (`final-pilot-preservation.json`, 2026-10-26T03:20:30.022170+00:00) checks 231 pilot file hashes: all equal **True**. It also checks 35 archived source files; recorded source changes: 0. Original HEAD `891d260af96a7ce61ccbcb23474015d975bb18dc` matches its expected revision: **True**; BITCOS HEAD `997b9c3fa8788bf31b12b1cecb713655a7953361` matches: **True**. No tracked model artifacts: **True**; no tracked files above the recorded large-file threshold: **True**.

Preserved upstream features: BITCOS presence/sign codec and v2 artifact format; 2bit/bitcos/auto packing selection; CPU repack without model construction; checksum/path/metadata validation; checksummed supplementary tokenizer assets.

Ported local fixes: transfer frozen projection storage rather than clone; chunk FP32 statistics and hard-weight reconstruction; preserve FP32 trainable factors/statistics/Jacobian snapshots; release snapshots after finished training windows; explicit tokenizer return_dict=False and combined model/tokenizer EOS IDs; trainer phase timing; reuse unaliased CPU tensors when saving; construct meta parameters when loading while retaining initialized buffers.

Isolated CPU tests: 230 passed, 2 skipped, 0 failed (16.55 seconds). Upstream symlink tests require Windows symlink privilege: skip only WinError1314; two skipped tests reported explicitly. CPU tiny fixture fork_rng attempted to initialize all CUDA devices: tiny_model now forks CPU RNG only using devices=[]; validation confirms zero visible CUDA devices.

The first direct parity-helper invocation failed its import-path guard before model loading and supplied no parity evidence. The helper's import precedence and deterministic execution settings were corrected before the successful CUDA check. Core implementation unchanged during this correction: **True**; the original and corrected source snapshots and helper hashes are recorded in `bitcos-parity-import-followup.json`.

## Calibration and reproducibility

Replay uses the original 16 AYOT traces, thinking=False, sampling=False, generation cap 512, EOS IDs [248044, 248046]. The traces contain 557 prompt tokens and 4773 generated tokens; 3 hit the generation cap unfinished. Training consumes 5234 active tokens plus 2958 padding tokens. Right truncation removes 96 tokens from 3 traces. Consumed AYOT SHA256: `09df0a26800a20e46288b5a775075c91dc5c0e98bc0bf81ac7906f304a0d820f`. Configuration: `{"batch_size": 1, "epochs": 1, "grad_clip": 1.0, "gradient_checkpointing": true, "learning_rate": 0.001, "seed": 5090, "stride": 1, "weight_decay": 0.0, "window_size": 1}`. Original pilot conservative projection count: 200; aggressive recurrent gates: False.

Model: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a), revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Diagnostic inputs SHA256 `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`; parent inputs SHA256 `2952cf072dff2e9b00b09a59a5f2345a02b3e509725854af07ea8478f8a7a64c`. Frozen settings: `{"do_sample": false, "enable_thinking": false, "eos_token_ids": [248044, 248046], "max_new_tokens": 128, "pad_token_id": 248044, "seed": 5090, "use_cache": false}`. Every stage uses the same cached token IDs, tokenizer, chat template, greedy decoding, and no KV cache; original model and packed states are evaluated separately. WikiText source-ID metadata includes some separator-only empty rows despite its scope description; the frozen token stream and scored offsets were audited and preserved.

Datasets: [WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3) revision `b08601e04326c79dfdd32d625aee71d232d685c3`; [ARC-Easy validation](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) revision `210d026faf9955653af8916fad021475a3f00453`. Sample IDs and seed are recorded in `diagnostic-inputs.json`. The selected held-out inputs remain separate from calibration and LS weight fitting, but diagnosis and model selection now use their outcomes; they are not a fresh unbiased confirmation set for future method development.

## Measured operational costs

These instrumented phases cover model loading, factor reconstruction, recovery fitting, export, and reload. Evaluation calls are outside these phase timers and appear separately below. Times are measured, not extrapolated; phase totals do not include uninstrumented bookkeeping or process startup. CUDA columns are absolute PyTorch allocated/reserved peaks.

| Operation and phase | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| Replay: load | 23.498 | 17.538 | 17.588 |
| Replay: conversion | 0.739 | 18.713 | 18.738 |
| Replay: recover_fp32_factors | 162.476 | 23.655 | 24.322 |
| Replay: exported_fp16_reconstruction | 89.970 | 19.272 | 20.162 |
| Pilot packed reload: load | 18.416 | 6.373 | 6.396 |
| Ablations: load_and_restore_factors | 19.579 | 18.682 | 18.707 |
| LS recovery: load_and_restore | 17.611 | 18.682 | 18.707 |
| LS recovery: fixed_code_least_squares | 4.350 | 18.804 | 18.949 |
| LS recovery: export | 64.265 | 11.462 | 12.447 |
| LS recovery: fp16_export_reference | 62.715 | 11.311 | 12.447 |
| LS recovery: independent_packed_reload | 16.467 | 6.404 | 6.428 |
| BITCOS CPU exact-payload verification | 109.484 | not measured | not measured |
| BITCOS CUDA reload and one fixed-prompt forward | 66.078 | not measured | not measured |

The BITCOS repack command did not record a dedicated elapsed-time field; no repacking duration or speedup is inferred from file size. Its CPU verification and CUDA reload/parity times above measure different operations.

## Measured evaluation cost

These CUDA peaks and elapsed times cover evaluation of an already resident model, excluding loading, replay training, LS fitting, and export. Allocated/reserved values are absolute PyTorch peaks; wrappers retain original frozen weights and are diagnostic states, not deployment memory measurements.

| State | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| Original BF16 | 34.278 | 17.954 | 18.191 |
| Wrappers, quantization disabled | 23.893 | 18.985 | 19.197 |
| Hard quantization, FP32 learned factors | 122.673 | 19.016 | 19.264 |
| Export representation, FP16 scales | 108.525 | 6.821 | 7.262 |
| Original pilot artifact, fresh reload | 124.753 | 6.789 | 7.172 |
| only_mlp | 5.853 | 18.985 | 19.367 |
| only_linear_attention | 2.771 | 18.985 | 19.230 |
| only_full_attention | 2.772 | 18.985 | 19.355 |
| retain_mlp | 3.235 | 18.985 | 19.230 |
| retain_linear_attention | 4.836 | 18.985 | 19.367 |
| retain_full_attention | 5.492 | 18.985 | 19.232 |
| only_gate_proj | 2.943 | 18.985 | 19.367 |
| only_up_proj | 3.425 | 18.985 | 19.367 |
| only_down_proj | 3.309 | 18.985 | 19.367 |
| only_in_proj_qkv | 2.524 | 18.985 | 19.230 |
| only_in_proj_z | 2.469 | 18.985 | 19.260 |
| only_out_proj | 1.960 | 18.985 | 19.365 |
| first_4_layers | 3.516 | 18.985 | 19.232 |
| first_8_layers | 3.191 | 18.985 | 19.232 |
| first_16_layers | 3.535 | 18.985 | 19.232 |
| first_24_layers | 4.628 | 18.985 | 19.232 |
| Fixed-code LS, FP32 scales | 172.225 | 11.088 | 11.498 |
| Fixed-code LS, exported and reloaded | 106.401 | 6.789 | 7.203 |

Ablation rows include NLL and ARC only; their generation phase was skipped. Stage and recovery rows include all four generations.

## Recovery generation evidence

The four final packed-recovery answers below preserve the recorded strings and special tokens; HTML escaping only affects display. Raw JSON also retains every generated token ID.

### manual-heldout-00 (instruction)

Prompt: Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

Tokens: 128; unfinished: True; EOS: False.

<pre>的了的了 的了的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的</pre>

### manual-heldout-02 (reasoning)

Prompt: A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

Tokens: 128; unfinished: True; EOS: False.

<pre>的的 的 的 的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的</pre>

### manual-heldout-04 (coding)

Prompt: Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

Tokens: 128; unfinished: True; EOS: False.

<pre>了了了 了 了 了 的了的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的</pre>

### manual-heldout-07 (turkish)

Prompt: Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

Tokens: 128; unfinished: True; EOS: False.

<pre>的了了的了了的了的了的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的</pre>

## Evidence hashes

Generated solely from completed local files. This script launches no model, training, or larger experiment.

| Evidence | SHA256 |
|---|---|
| `ablation-metrics.json` | `8ab1c63607e3036954d880f6181e1317ee558bde882965a8ddbb7725af22ad5d` |
| `ablations\first_16_layers.json` | `46e1bcb83564f761ffc283d8998381857d014f437a9b1db591827e539988ddbf` |
| `ablations\first_24_layers.json` | `adab81a1fb352aac309ba0f739de1ad73765d61924b43e4ddf19fce75705fe89` |
| `ablations\first_4_layers.json` | `3eb3222caceba05c3d2cc73e9eaba7c124ac18051b2a0a7ee788bd9bc98b4dfd` |
| `ablations\first_8_layers.json` | `c2dbd253033c4b9ad0fba255b9ff994441671ce3a3cd7d02e28a68b7552abb99` |
| `ablations\only_down_proj.json` | `d2f9cb8a80562ae16d9f9040fcb6eefa7969789e3b7e5cab0ffc75e844057760` |
| `ablations\only_full_attention.json` | `d2e41ee7f5af6517fe77d890fffdda00b2045a57c998c7f4eab4bc8ac7e3fec8` |
| `ablations\only_gate_proj.json` | `c854372303eb201abb48e49cd6734e1524e29d01cfad5701a5ed01fc979869dd` |
| `ablations\only_in_proj_qkv.json` | `0f54217254d17e4d3c2afff03f060fae08025c7c6f0ce1453a32caef1623c37d` |
| `ablations\only_in_proj_z.json` | `a9b127f5a23a0559df38dcc420b2a07a32290bf537bd42bdd7154152341aeb95` |
| `ablations\only_linear_attention.json` | `7ac61aa5063238f949dc15e30a5919ea0e2e521b58a5db536616b63b86bcbc63` |
| `ablations\only_mlp.json` | `de71ed529186e9b007f3628dc11f027e30701f662488cb05d53c20c2286e0095` |
| `ablations\only_out_proj.json` | `5f9776c2633d8da3b596991316bd85bbac48eca50d30cb7284d2fa9ae4b24063` |
| `ablations\only_up_proj.json` | `23f38fd2edbabb60b563194799c24fa2b81bf133c2fc26268c1e7f791b6c18a8` |
| `ablations\retain_full_attention.json` | `e1c21f24072c77f203069d5346bd0f08e76f207ca737428795f1d85bf52bafb4` |
| `ablations\retain_linear_attention.json` | `86980a71783431dc3b4775046f58b505fd94a4d10ed2a502211d4285cbd13f35` |
| `ablations\retain_mlp.json` | `bd8376eadf05febac72db281867f76bbe1038019ba0bab4c222c8b3c35862e42` |
| `bitcos-integrated-sources.json` | `8ea8d39a55ed766a91384305ee5edcc572ced9476d841c670624277ea8a539bc` |
| `bitcos-integration.json` | `4e9620af8a226678ac0ecdd8be32f8f781e525ab582668d794622a7901906fe7` |
| `bitcos-logit-parity.json` | `46477485879c22d5bf70b484aa0304601b1f8c1f3d139dd70b4028fe767e9fec` |
| `bitcos-parity-import-followup.json` | `84878ad62cf320abc5201c1451da1e84f45eb7f9175514964e8931e7c31708a4` |
| `bitcos-pilot-preservation.json` | `c3f225088a3c80b072da8a55ebb7a84554d0a52e81d86933df28d34e18797492` |
| `bitcos-verification.json` | `d93c4a3299b3e224a6c8992529264805902c962b92a09b8ad478b13475dde562` |
| `diagnostic-inputs.json` | `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a` |
| `export-reload-logit-parity.json` | `83d30aab6a2033521449961f1bdf84676afaab0b24866a434a7508a6116f5879` |
| `exported-fp16.json` | `fec465917dcb6c60662d746b08dc2c8295262aaac48c3a7754e5164866bbe123` |
| `final-pilot-preservation.json` | `1f98e93783909e2d713d721aaf70370ee13b41083c41058c558923ab062912b2` |
| `hard-fp32.json` | `41a716fbf42262d5ce1fe6e8a31ce10aae49005a54bf9eccc48eeed782c31999` |
| `hard-layer-drift.json` | `18ef78bd0ca20c1c3314147b8629b1fbe0d4d909437a0ed3fe600777a02b2e22` |
| `original.json` | `a673bd24dd81eda919eaaa14437529c464f8794c3e3a335295a73bf76d4356ad` |
| `packed-reload-metrics.json` | `99ca0bf9fca97f92eea2cc27b275258dc10521491be103332cb147eb4058afdf` |
| `packed-reloaded.json` | `6879076df28f38535c66ce44873f913891b64f9975d23d3e52e422bc299696a2` |
| `pilot-preserved-validation.json` | `0ac27980facc97cb8e21fee3f2f3558cef84ffa7c6e58aa5f7975993c10beba0` |
| `recovery-decision.json` | `22f7dcfd198fbe1a7baa726e26a52279cc12e48266c20a4785737e40f67bcfbf` |
| `recovery-ls\export-check.json` | `262e9e41d843e0d54d643a5ac46e03b20ac9cd6c4299d2f10c37970200020d05` |
| `recovery-ls\fp32-evaluation.json` | `88031737a4bf568b21b515ab372e195ad299965d98bde2a7b2f849a435bec70e` |
| `recovery-ls\metrics.json` | `b90d79274ee264580468c8ebaa19c779f6c3350ce71343bb61b3d90616ef8432` |
| `recovery-ls\packed-evaluation.json` | `5f25e32781292ccffe8c50ecf540bfbb1f3d327b8c9e825ac232d397bd4453e3` |
| `recovery-ls\recovery-audit.json` | `6682b97a2605e071bafcca9f234b139db5ae954f3719fde85cca50b76ed1e51b` |
| `recovery-ls\sizes.json` | `43c0ecfbe0aa0bef6b3c69b0e10a95263b79c925fcc62c152a3278561dcf91e7` |
| `recovery-plan.json` | `677f73ff5e54215f4c38d208b6709d80d9fcc896207bc8522282fedd75a7e733` |
| `replay-event-parity.json` | `721de255d0d9cf15cefe61c9af83a2f73a410df1b19e91e3b6e53c65293d6a19` |
| `replay-metrics.json` | `aaf6ac835da2e35535452c01da09bf735d9d03871b07115d4221835617354c73` |
| `replay-packed-agreement.json` | `c05c83998053d227dfe7f9d40104b1b882849bcdf0d5ce6aa0c888af1d999003` |
| `replay-training.json` | `80780abcbe40974b9da69c3059bf0a12085d7ab9774eee151620384bab2b868e` |
| `wrapper-parity.json` | `2913dceb4dc812d78fa6007560bb7311af6f6cf2817aec966c612700d9dc6267` |
| `wrappers-disabled.json` | `0d6065615c29eda1b21fd5eee8bd977a60a18970094149ea68c5131284578920` |
| `D:\openbonsai\runs\2026-10-26-5090-pilot\baseline-evaluation.json` | `99419e600636ef6b064359da302c2d463be8f7a7faa639bca353719de7cfce6e` |
| `D:\openbonsai\runs\2026-10-26-5090-pilot\compressed-evaluation.json` | `3f61b57632f1bc89f0bdfbb7c96a3a8d09a5a255e2ea87088a8b1ad40e83ac6b` |
| `D:\openbonsai\runs\2026-10-26-5090-pilot\prepare-metrics.json` | `9dc2a9ad1c712cc431fbee8d10b7d7f1c1b3c8b2cd8d87e2d86dce0c285f54eb` |
| `D:\openbonsai\runs\2026-10-26-5090-pilot\sizes.json` | `f9b53e2e4263290300dca0c2663074d43162317accec63c5b8699fc253b797ec` |
| `D:\openbonsai\runs\2026-10-26-5090-pilot\training-report.json` | `35faaceb692dd24c68ce6c50354275d97897b8d0d60a48ac98c7bcff936d8257` |

<details>
<summary>All 231 preserved original pilot file hashes</summary>

These are the complete recorded hashes in the latest `final-pilot-preservation.json`, also present in `bitcos-pilot-preservation.json` and checked against the original `pilot-preserved-validation.json` hash map. The report builder validates audit completeness; it does not rehash the multi-gigabyte weights.

| Original pilot file | Preserved SHA256 |
|---|---|
| `all-tests.log` | `0e7b9aab912aa9862d907bc724385dbb68926a538c2305fa6359daafa65367a8` |
| `baseline-evaluation.json` | `99419e600636ef6b064359da302c2d463be8f7a7faa639bca353719de7cfce6e` |
| `compressed-evaluation.json` | `3f61b57632f1bc89f0bdfbb7c96a3a8d09a5a255e2ea87088a8b1ad40e83ac6b` |
| `fresh-reload.json` | `18265e248bc03979372992b98f1144a62be1dbfd79f1ce0358490c82a4b4e559` |
| `generations.md` | `f76e53cee46fcef4e25ddc1e268450f9156a2098ccedcf7cd3a84b09fda22bcb` |
| `packed/config.json` | `14620a214ae116717fa04a9c0ba10a9a4223adc833f2736c8b7a0d9b27f55b3a` |
| `packed/manifest.json` | `cc83d788f1873aa6f9178abd498bde236579b08b6aef632bce5c8a553216e4bf` |
| `packed/retained-00000.safetensors` | `106ec926c62c1aac2c8ab808833db89012a9132c53f767fa7609ec69e377d4ab` |
| `packed/retained-00001.safetensors` | `a01f5aff6f0c0956b006370df5a0002afb8b0ee8f9996f4e18bbeac98893e8b3` |
| `packed/retained-00002.safetensors` | `91189637bbbf2a0b722b9a6f6ef260798021cf824b31899036c51662d6895a55` |
| `packed/retained-00003.safetensors` | `6ab4f8db744f979f04bc885b25d3e8cc3c08cb92105d08e4405b2869cee44cd9` |
| `packed/retained-00004.safetensors` | `17c3c7146464d61a9274586a78dbba2096c45edb67314635657ee328229470a3` |
| `packed/retained-00005.safetensors` | `47e254e1b93698903a5b00e0c114cf04d5b2ada8684a17d4d0feffbfd92aa634` |
| `packed/retained-00006.safetensors` | `8a4ae166a7fae968445f7fce9f901c80f853e1ebf156b9be0ce93b649e49e540` |
| `packed/retained-00007.safetensors` | `5893c94629a7f319af04a9db7761887a25efd578d16a6d5b4b9d2ee67a755ec9` |
| `packed/retained-00008.safetensors` | `52eb8f660c4dd06b7287af97db229e0d843112a414bac6c14266684bb1033def` |
| `packed/retained-00009.safetensors` | `72643f8670fb88724e7e9b26205c1a448bc74017f28c68ec14271629e8243cef` |
| `packed/retained-00010.safetensors` | `37316306a237bcc841238733bf6a44065782249eadf2f3a7360175d03356fc61` |
| `packed/retained-00011.safetensors` | `59449ab62676858df6f3ba93f063a4ec8122fcb767571fbfe0511d53ad895f83` |
| `packed/retained-00012.safetensors` | `37966639e04401510ea0cb450d122b1d3b4ad134395ebf440af6f8930ed4c8e1` |
| `packed/retained-00013.safetensors` | `6c97914c2cab6e79b7dda466ef90ce6143634a797265b481f3d3813e5116b7ed` |
| `packed/retained-00014.safetensors` | `0f4713a04e1b8f7ea0ed403f1201a46d5ab3437c7c235733cd040ab11490c13a` |
| `packed/retained-00015.safetensors` | `fb5f0e0ca8f88c397322c75685a9dc9f46766d4fb3945159ed01760054b51aaf` |
| `packed/retained-00016.safetensors` | `14487aaa5c646379e90391e326ea4701323d43bc8006d885c1c34be338f0eaea` |
| `packed/retained-00017.safetensors` | `92b62afff99bdce719756458e07f8465b3ae4c36ec0131edd5a8ec95d50adbfe` |
| `packed/ternary-00000.safetensors` | `c9c12d94147bf27920490fa36139b7e529c40894796b2d7a147c66a96e94d25d` |
| `packed/ternary-00001.safetensors` | `64e3bff2f83c32dad5661e5735f5d6498641296903efdc0985a77f9becf5b85a` |
| `packed/ternary-00002.safetensors` | `58421c1111b7dc622f080f7236814f4d11ff451e1446c60cb6c5366feccde326` |
| `packed/ternary-00003.safetensors` | `ce9de4af5256d84e1fc1ab437476a85de2e43ce93829cf4bb734a1d284b9f6a0` |
| `packed/ternary-00004.safetensors` | `b225e12972016104bb5ef74666618c4fedd555f3dcfc0e12f7e61a56bbd7043d` |
| `packed/ternary-00005.safetensors` | `072d61d75976b0441c87766006a96fccd8d8b22765fba29397fc75aabc3d8c4e` |
| `packed/ternary-00006.safetensors` | `a98f282bdcf3368a87df91efe092286ce5ad8e0de2dba7651c5d90490569b5eb` |
| `packed/ternary-00007.safetensors` | `455211f98129277d4a1ad34e3380a066ae0ad160cff8a53de25368d7fc6d22ac` |
| `packed/ternary-00008.safetensors` | `19958ba8ed60518615fa3b12f035112e42c27a25c89facb99308df2e20bd2c84` |
| `packed/ternary-00009.safetensors` | `a43f604bf4eda2beb7785d38ed3ca641ce67ea02dc374639af66d5e570256119` |
| `packed/ternary-00010.safetensors` | `2dbd5832d7b3e16d9ce8ad43fe97112e0342ea9ba2f1b01099c8eb65dbcdbdac` |
| `packed/ternary-00011.safetensors` | `96900fd5e5098b574eb06ea9773d2d2edad84ff29afff38ab6c9272d1ef48b32` |
| `packed/ternary-00012.safetensors` | `507fbe8b408f437c15e8ac58011306bdfb43f648a1dec22677641d5b15807878` |
| `packed/ternary-00013.safetensors` | `43d9f28ca33695593f751215b25b091feca57e96b96482528b076b48d06c25f3` |
| `packed/ternary-00014.safetensors` | `7d3ee8e401e0a3ef197a2b5c93185a35d652ec731b345cb6e840e0531f1b2436` |
| `packed/ternary-00015.safetensors` | `bb25f20ca7eef5d3341b121b8a18e2c3184463dce4ed0fdbcff2b358656f8e2b` |
| `packed/ternary-00016.safetensors` | `a8bafa11e706d82d9d73ba79c6c7b5f33381664167054a9a16e8a9bf3a0360de` |
| `packed/ternary-00017.safetensors` | `36cacb62bb26dc4a7f38ce9232d55c89f7eecdb0dfcac11fdfbe142af58cfd0d` |
| `packed/ternary-00018.safetensors` | `b41b1d784832e744280f8232f7c3084ec2faabf7f631d681667000feec9781db` |
| `packed/ternary-00019.safetensors` | `affb71ca8ee73e163dd2da33a1f590107c3747d45eba912cd77fe86f5f1ee523` |
| `packed/ternary-00020.safetensors` | `bddfd63e57e86808a89849e9bbc7535797129763e315add2613dbf39d8dc98ef` |
| `packed/ternary-00021.safetensors` | `f826aa85655fd62aa3a8df232f29d8294d1d5d900c6d69d3deee0e01080049a6` |
| `packed/ternary-00022.safetensors` | `2edf1bac4ad8dbc7e9081dfefa3617c99417330877e6ac1ded53dadccb387c30` |
| `packed/ternary-00023.safetensors` | `71e92bad1fe0d426efef828e4b040904a0c2ba69e870dadcb56eb829e439e182` |
| `packed/ternary-00024.safetensors` | `16eacf4185e63287e881489ccbfe7e743ae25e3dec9dd6116a04b0134628bb98` |
| `packed/ternary-00025.safetensors` | `0f85cdd92808fab32694545b2911e9eda6f6ff86703a3693256833a70e760b91` |
| `packed/ternary-00026.safetensors` | `09880a6fe233a174b06c5a7dc79790b19d9aef6544048f654da8b67dcd637a39` |
| `packed/ternary-00027.safetensors` | `0766056f92cf88d915d2cd471ac2675ad230a3bc5444e771c298f39823816ee1` |
| `packed/ternary-00028.safetensors` | `a69e4911566b841bea2de659418b7169f28bdb7802d4292f4bb74b850249f210` |
| `packed/ternary-00029.safetensors` | `bc49479003fab14aa4c927d2df488b2f2855a25889e987f5e3952b19ecff4f7b` |
| `packed/ternary-00030.safetensors` | `4b133a1da41ac7423b5b02f3fd54cb3ad199195a48c659d166e0e0bbbfe5172e` |
| `packed/ternary-00031.safetensors` | `0ca35eefdcd219e7f9b3c2bef250663914406b664df11370ad1cf00bd3c54868` |
| `packed/ternary-00032.safetensors` | `7d425302083e7e3f69dcf53a6d00ad8764f952ca9fbbf1fdaa2267973c15d4fc` |
| `packed/ternary-00033.safetensors` | `a80e8fb0e1658cbcfec4780d9c22acee5ec603c7dc99885b69a6756247229e1e` |
| `packed/ternary-00034.safetensors` | `02769b65eab2522e46da64b9aad023a7c436ba8553505537f1da895b9029b13e` |
| `packed/ternary-00035.safetensors` | `ab9103a4176b4c6e80bedbd69e01734fd476b19a7aed621077d0c35ae5064617` |
| `packed/ternary-00036.safetensors` | `98fb044c9e67e55b88521509de5478d838fbff1c82003b4e6f5021f5b2869595` |
| `packed/ternary-00037.safetensors` | `d1df2d9d0f02007db61dde83171c3cb0b5d921821a6dd3a3ff271291b3d7a23a` |
| `packed/ternary-00038.safetensors` | `f5e81475391913d25e44849f3c059f7e10e3b17b8e6a7dccab9b96e363076323` |
| `packed/ternary-00039.safetensors` | `c1197bbdd9cddb86334ac97e3ca627276fecda7de846374b04b1d3b6c46bf23a` |
| `packed/ternary-00040.safetensors` | `a82c4d9ad450288130d5eed90014c7d5648be667826a02956d2fff0d34ebace3` |
| `packed/ternary-00041.safetensors` | `a050c0bdf8cd6e04d4f5ff4be4648aca83756cb4cdbaa7344ffa1e8cf28564f3` |
| `packed/ternary-00042.safetensors` | `711b7526b8069305ba424051a9612d386b51dfc0720d92cf77210a271cec00d8` |
| `packed/ternary-00043.safetensors` | `b92bd2969a96acd3ae982f0363843f17d83cce60f13964e044fbf8e295cb60c5` |
| `packed/ternary-00044.safetensors` | `3af42be14b9dcd07f1600eede2d7748e3956118398c9957761f57acb5bbe6a7a` |
| `packed/ternary-00045.safetensors` | `89bc840b7da81b3b4f469893a8db34314433ff02ea1a07f9d3b202dfe918f9e3` |
| `packed/ternary-00046.safetensors` | `b0b9f1c9f4049706da8f3c269520dddf8abd515e816c719b77b6a0a3fafebed2` |
| `packed/ternary-00047.safetensors` | `de88f3741fefa81b7eef1666ef8823b9ab50d687a63f34cfec151e3da7e05acf` |
| `packed/ternary-00048.safetensors` | `b78ab1a4f007c19f2490be4f569e6e19d007112a59397aaaac345e63fdde8f06` |
| `packed/ternary-00049.safetensors` | `288b8b0e7aa3f3511c686d79c297472567061a1e398baf0ab05bef286361aad8` |
| `packed/ternary-00050.safetensors` | `9c9d8aa9514d7a32c22caa3d374859110724a6c2fc802bfdec02a9a22d515a18` |
| `packed/ternary-00051.safetensors` | `b34395b6ba5adabfdf4fc7500640c75bbfa35b512bed26bfb8f2785add2a0a08` |
| `packed/ternary-00052.safetensors` | `310a79f2ec8956345d362584707d5772208cbdafbbf1faafcae48fa3d79cd98e` |
| `packed/ternary-00053.safetensors` | `babbafb2ca85fac35f55f11244439c38406b7c073a14832ccd4693822615fb80` |
| `packed/ternary-00054.safetensors` | `1636f1d333475a0fd8fa02040b72e52bf1c5686ffa40a021ad20b81a7c456097` |
| `packed/ternary-00055.safetensors` | `0e0754bfda710bb0177cb324dab3acb93ef00eb56d5d4702111dd5bea98a39f5` |
| `packed/ternary-00056.safetensors` | `bcf6101344882eca1de3a6e8e7de147852635c4d306e33a74e3a3c41ffbad973` |
| `packed/ternary-00057.safetensors` | `c1de0644336e8d8ca11760f18ccfd87dca7179dc0b85dbd0a99dab22dd3874ce` |
| `packed/ternary-00058.safetensors` | `02258205176204801f8abd035aab5dd99e93d391f93d64d8f2efb23357ced66d` |
| `packed/ternary-00059.safetensors` | `982b9b91bfaa9cfbd94e3aa85182d5e3843b883c5cc015f58145243208e89bcc` |
| `packed/ternary-00060.safetensors` | `45daf42823d8ac11c73dbfccff0943d51503f2f2787fc4e5c5db819d6acec493` |
| `packed/ternary-00061.safetensors` | `b1a569f95d213c0357524ca5f61f275e36f1a3d4723a3b6c50ea75e750c71380` |
| `packed/ternary-00062.safetensors` | `13780625bac6d8048a1562067821f3e4504fae5400f48d3e1835a48fb1b4bda9` |
| `packed/ternary-00063.safetensors` | `cc381882eeb16f21e542b24f9f769f978de575da9c4bb83d11f3d4dbe9960925` |
| `packed/ternary-00064.safetensors` | `434780e6eb0a0103623e94bcf2d12b9bdc58a3fdba9420e106969005ac39879e` |
| `packed/ternary-00065.safetensors` | `446d558d9b8dd89e9d4a4f9d71fd145e65204981bd1905644ba83508f29f32bd` |
| `packed/ternary-00066.safetensors` | `25e7928cd91eac2d3446461544d33d907995865580c2bb49ae2ef63dd0aef34b` |
| `packed/ternary-00067.safetensors` | `bccc81937d8c348c6bce24764496bdb02d1d102903115ded5343683ddf15c0b6` |
| `packed/ternary-00068.safetensors` | `6e187980bad57df98a480868f0874b1e3685bc63037b84a35818a6dd5e3b4c54` |
| `packed/ternary-00069.safetensors` | `ad30df82a73ebc0027d2b5ca9784595c3bfdbef9e1c4343f4336b4ce1cc7460b` |
| `packed/ternary-00070.safetensors` | `b6af985906dac9ea187110c031f83615720c37131c4d30f885a15d904917fc42` |
| `packed/ternary-00071.safetensors` | `cd0eaa25de32d8d1c560d037e1aabef09bd102afa6f99e2dd10d71950c2d0bb4` |
| `packed/ternary-00072.safetensors` | `c274821aabb27b8408c31294cf0872b3da8e4c403e9b549413cb4cff9159d975` |
| `packed/ternary-00073.safetensors` | `686b7a117411a659c096c902c88687d2d489679b2b4e433a278397603fea9587` |
| `packed/ternary-00074.safetensors` | `973131b4e0f06f77d1da62a4a8035271569c4206ed40a6061178801a43c9ad2a` |
| `packed/ternary-00075.safetensors` | `91fba8b754209a064177090a8ffb493f438946855fab4cda49c3a78186f41ff1` |
| `packed/ternary-00076.safetensors` | `60942f96c65b05c3a4333b88d1d6db043e80296f27be4cbb63c81fba8024a9ed` |
| `packed/ternary-00077.safetensors` | `4a7decba6e999defe5dd016369a36661cb06b5c8ff8961e013a824d5249f1133` |
| `packed/ternary-00078.safetensors` | `4c7dc3937406032d73fa062f52fab1f4dd27255c08a18001836d93500475876d` |
| `packed/ternary-00079.safetensors` | `86a3190204f2273c05fa9a87f2a5c094ad6f971a38691bcd99f378433a89e232` |
| `packed/ternary-00080.safetensors` | `f96b54f7f951c825763e1b90f851665a2b804f9cc3e756c7d55fcae8b176ea82` |
| `packed/ternary-00081.safetensors` | `2c5e0cce623831a9871dace18289d16314fcf74bb44dc773ce557158adbb5fee` |
| `packed/ternary-00082.safetensors` | `b5670e6bbea99ee0b54fed4218c1642a1d0ba19f17984bc58deaa410c58f1ca8` |
| `packed/ternary-00083.safetensors` | `da332d56246556f66c86b6ef740731138909960930f947e60294ad716dc89880` |
| `packed/ternary-00084.safetensors` | `292d34df145630091efb5f44a9a64e922b022aa7dfd1e4b442b84861feff8434` |
| `packed/ternary-00085.safetensors` | `ec87b2505f665a66868030f6591aff5b11472b3b00039b0bd9fa8f6f9a9c5e7b` |
| `packed/ternary-00086.safetensors` | `605e0235ad8f3f1af00c2682d3397baf49e4c4a440ddc34550310a5b15aba7d5` |
| `packed/ternary-00087.safetensors` | `808b834df57973497ba1d39d578688aba7737c93f99906d20e7f0f8f943c6858` |
| `packed/ternary-00088.safetensors` | `1b469f2670dcf11cab0e28901e1a378e55f01c75b8f73dc67d14c5749852ff5b` |
| `packed/ternary-00089.safetensors` | `ed3f7bb251948afd293dcdc410e1b58162a9b1412ff0d52fe6e30bde92541d22` |
| `packed/ternary-00090.safetensors` | `b022a2c0796b39d8bd8b9a676b0eb0a4415742858089abb57bf26e496ba00c13` |
| `packed/ternary-00091.safetensors` | `ebcdad2dc38bd1e4783e07e8804db7fa09cdf637996ead6220a4694b4fa18032` |
| `packed/ternary-00092.safetensors` | `d2c6dfa94a76638b2f62c9caf185070c97d8dbb99109662ad2e083615f2470a1` |
| `packed/ternary-00093.safetensors` | `1dc9377a9b5d26ba852dd96426cddad53cd8cedfd8405213e169af0d96ea171f` |
| `packed/ternary-00094.safetensors` | `20db795d200033ee33a15bd29ec91dab8ab2c70fd1cc2d13b877988da8f09fed` |
| `packed/ternary-00095.safetensors` | `b2b5a450c334aa7463267402943e0d9994ad025ff94f42384d89f03b3e53b30e` |
| `packed/ternary-00096.safetensors` | `cff7c65f52f79723dbd105fa7cd5cbfdc65d7fa0987ccd786691533e47795237` |
| `packed/ternary-00097.safetensors` | `c3bcefe2d3dbead6128a4cbc4b8e4ebf9a4363077be85b5d5e7bbcf842fd44d0` |
| `packed/ternary-00098.safetensors` | `1a71f110007e557d8ecaaa8f983fae6fa6a32f6f649d4cb9164ea770651c9d35` |
| `packed/ternary-00099.safetensors` | `fcf240db42c339ed1873333f8f533c8be31652b6a20442ad10eabe06938b7bfa` |
| `packed/ternary-00100.safetensors` | `d6f2355b3b9db81e1b1e474678c3a3e2b7a86c43cf4ef47c5c3d91ed78bf0f04` |
| `packed/ternary-00101.safetensors` | `303ab68d9f70ca8bf27067883328ecde0d2137d61135217838e841104715a322` |
| `packed/ternary-00102.safetensors` | `f2ce302355065a1bbfc50e60d4dff1ce42d687803c4a26cea532bfb62590eec6` |
| `packed/ternary-00103.safetensors` | `593deee6f8c24195671f48c61ed783720ba1fc990fdb839263ffe0e88758ce8a` |
| `packed/ternary-00104.safetensors` | `21eb2a5ad802e161e9d526dc5dfece92cf10f7bd0a0b34fe6b01a4820c474ceb` |
| `packed/ternary-00105.safetensors` | `51604a56591b1c75ed50c00c09701c66ab262d2633858b7db1622e4cd2a6aa03` |
| `packed/ternary-00106.safetensors` | `c7387b03ed00e5508243cfe3459e07bba317df218d1ddf5bb9801b45a6721919` |
| `packed/ternary-00107.safetensors` | `fb1fd8617af5db88e51924e45b419869682cbce0db2bf28fabe3b92a9980142b` |
| `packed/ternary-00108.safetensors` | `441cd91ccc5be574276edb2337873d13fbd26180425da72cdff8351e35589b0c` |
| `packed/ternary-00109.safetensors` | `f1b259b8bef1612c5ac95230cc12a1d519ffd566f9af67f4ce34f46e14c15950` |
| `packed/ternary-00110.safetensors` | `8d9ea7f35f31e87d00285901e808154da904cf792f617404eb879b83228b269b` |
| `packed/ternary-00111.safetensors` | `12cb870fd31562e0f9f1762b2219582c63a9064af3a097c68dc3e1beb0cfd379` |
| `packed/ternary-00112.safetensors` | `03aca2884dc30b3ba32ef91407599924d2e4acedbdad6a5d4b93339a8a173394` |
| `packed/ternary-00113.safetensors` | `0b4c9d12a227ff5a1b8b8cf3fa9b09fae67d43a8cd2ddd9544e691f5eaebd241` |
| `packed/ternary-00114.safetensors` | `7d3e37d6201e6ff71a5df782adf0190989c489bd321b55072cd7cb9a3c09bbe8` |
| `packed/ternary-00115.safetensors` | `2509eaab1faeee031319188a7c91f106912729a09318dc17dd78ec132d50d7e6` |
| `packed/ternary-00116.safetensors` | `0a4aea76f4652f0a7a290cf3dbcb4aef32a6d7714131b4ede1c28e569c1a986e` |
| `packed/ternary-00117.safetensors` | `4f035ecdb58b78256643d9007eee39d9649de8eefc6e1e51c66908d6c0143373` |
| `packed/ternary-00118.safetensors` | `5246ee0d3a31c2b0c54559a57c1f13752425aba14b5238d79f9add0c02de4027` |
| `packed/ternary-00119.safetensors` | `42f422f3b6169578ee6423af5e63de24f80cc522fd0ba0b7a663c2a8c1ffb25a` |
| `packed/ternary-00120.safetensors` | `f9234f51952ae245f3433b2505bc13c035e526b8d2fe10eb444b0f216c6f159c` |
| `packed/ternary-00121.safetensors` | `f9aeb1fa4ad604035fd6667d44e4728e4b277af5c2ddc6100abe155ad8fbfb1e` |
| `packed/ternary-00122.safetensors` | `9f41074b72180416f2fecdb605a140f7e0939fcd90354edfc43a300e32ee2dc5` |
| `packed/ternary-00123.safetensors` | `75cfd46ccd39ee37dc13674efefbacb1f0fadcf1f78326ce0a920a2fbd2ba654` |
| `packed/ternary-00124.safetensors` | `a2f698c39628418f21078dbc6680363c71c543983b5b237462de3992dda60e8b` |
| `packed/ternary-00125.safetensors` | `b22950981fad59205d0dc82c3b66be5759a6936e5ae461c729e2159dd8e1b08d` |
| `packed/ternary-00126.safetensors` | `34abbbaffb2cae3749c6b7ebd30b59311799f43c433dede04181819451dc41a6` |
| `packed/ternary-00127.safetensors` | `8e16d6ed370f7ce4d8da0102ceecfa4e94fcb0d0072d4ce14b84de8113f98970` |
| `packed/ternary-00128.safetensors` | `c1a892d92a76eb45a80f87daf4d1e359a01b3e9cc970a90d819ae6a3f65c870b` |
| `packed/ternary-00129.safetensors` | `9ff953bde9a9c4783010f80e69d90826c0ec809d62938a52cf99645dd6e9f418` |
| `packed/ternary-00130.safetensors` | `bd5fd822badf7e02f85614e3e8f5f6f970b1be1b844ad1f8d723df70d27fdc5f` |
| `packed/ternary-00131.safetensors` | `e0630b32870245a7c0d7369b7aac64ea596231131d338d928079c6b3f6fc6d84` |
| `packed/ternary-00132.safetensors` | `c7bf76c0cedc612e6ae3b18c1867f4eb11141549b4da8cb49604073bcb967c4a` |
| `packed/ternary-00133.safetensors` | `aa83c40d94168375617e3ec31ad550d5f31b53d9912f79e065565e22530a88ab` |
| `packed/ternary-00134.safetensors` | `6b2320a50c60f44bca10931bc78239966e88d3515f6fb2967c6d5e96fdd94428` |
| `packed/ternary-00135.safetensors` | `5d347305ed430801c32451ba303b45a3aa0a63a2214aea5785dd694a40d8a81f` |
| `packed/ternary-00136.safetensors` | `3713a2531b816c3f13214f7f0d710eb43f4ec74a35a9884ba0ef6b922564486e` |
| `packed/ternary-00137.safetensors` | `3d90e069c660dafc786a13747ff1d569b42e8551a09940b5e032d5e51f9b7fb2` |
| `packed/ternary-00138.safetensors` | `e24416cf07bd3679ae6c09063ddc3e419f7430b647a2a439c0a1a290452c6b23` |
| `packed/ternary-00139.safetensors` | `28a94a161ca1ecc4d0ad934945b3cd8ac08964207b7cfff5f316986a541b776b` |
| `packed/ternary-00140.safetensors` | `6225c0fc8eead11c917b8c10f104fe4ed501f76339b4c2ea214f49fbd72fc0e4` |
| `packed/ternary-00141.safetensors` | `e69b3884ffbfbee6381cb551eb86b3c4fd16b3151d3e4a22c9185685ecdb55dd` |
| `packed/ternary-00142.safetensors` | `7f65d5e97195af63e05d061a2582fea1a9fc5b67b4d584693a1b6f52a6f2b18a` |
| `packed/ternary-00143.safetensors` | `faefb02911cb247476220e8208ac7d0214f945aec7edd81a76bd96b87c7a3c3b` |
| `packed/ternary-00144.safetensors` | `ccce62bef2f3c69aaa7b0c5ec6a0140489a1637a0b2055a32fd4ff1f94045ac9` |
| `packed/ternary-00145.safetensors` | `3d49b300860d35cf23a28a5f78578c1d07bc00e9b0e685430da188eaed120a0c` |
| `packed/ternary-00146.safetensors` | `f907e38d34b666e09bb0b4661ed5c74c34171df19e611effa57c4c185396cde6` |
| `packed/ternary-00147.safetensors` | `f1597c5d88322298d25518e602c6aaa4c833df66ff6ec3e904eb7b6c1dc2c304` |
| `packed/ternary-00148.safetensors` | `d51e2d6beb3fe7ec7845ce0069469f9010c00b7a196ee028edadf9cbe420e5ae` |
| `packed/ternary-00149.safetensors` | `ad9c6cec138056dadee546e7b0b4b6902540cc080079a16e3f8111d95ffc7024` |
| `packed/ternary-00150.safetensors` | `d9e2b0a31daea661fde43d49ae3e55b9d7b7641ed8430d21016815836fd82b9c` |
| `packed/ternary-00151.safetensors` | `9d6f61f00cb93e32f3ec9052a3a482b26171e22ea60f71b96bb1ca656433eb62` |
| `packed/ternary-00152.safetensors` | `375d66249e9146eeed9a33e5364e41d52467531bf90c2bb59dc36b5e2dfd75c1` |
| `packed/ternary-00153.safetensors` | `39aff5699f9c599d31d248db69d4abffd0e2534e5d17a268f99d0244702e3869` |
| `packed/ternary-00154.safetensors` | `35bd512eab4844317a767139b1aadbfef705d9692a0f339184eee01c9c793ad4` |
| `packed/ternary-00155.safetensors` | `f3f59e9c41ed99a99ab4ce4e35146590109404b1e8a25d83e40bfd22b92b8e54` |
| `packed/ternary-00156.safetensors` | `e21d890bbf0a6f4d13a77fe101d8715e428b4aad17a59691cba19b3bf996b495` |
| `packed/ternary-00157.safetensors` | `56c1d24bb87d3930b875788b9ee0faf9e5c46848a7dd447051e0bc883cbba7c3` |
| `packed/ternary-00158.safetensors` | `6a3d7d44a77af87dd18ea3ae5547b8124e89d2bfdd4fac7bef3c28f1a86c563e` |
| `packed/ternary-00159.safetensors` | `8da926ab140357657cef2dcc667934bf4083ed7b831d5172baf70744b0cf8a1f` |
| `packed/ternary-00160.safetensors` | `909ff94a09e19a80c50320c9f436a0fa0ebc4cf264557bf24a8ae70cf09c1f2d` |
| `packed/ternary-00161.safetensors` | `8fe3095c6bbaf8388931449d405391b1cbd7805fae6418a49a5d5ac82a7ab28d` |
| `packed/ternary-00162.safetensors` | `5a5d26f2c949eff8719a4cd242edd35aca975c3de5aa93a8d032253b39bfaeb3` |
| `packed/ternary-00163.safetensors` | `cfac6dd8b757266de074929ff9736427dce226bfee7fec2a1f08d60a57e144b3` |
| `packed/ternary-00164.safetensors` | `369c3ece58a6ed38b3f0389c87060303ea416c775782d33300cdcca3f3c06032` |
| `packed/ternary-00165.safetensors` | `d1444d5be6a0a55df0b57b9a52d2b1327f82cf1cff5502f0c75904b4783660b0` |
| `packed/ternary-00166.safetensors` | `91bd41e9f83789b53b689f6d8fadf7a896f0673470b333f0fc5db0cf0a85373f` |
| `packed/ternary-00167.safetensors` | `da6be2a20d28d051f54143de9d6ef3f6c91a23ae6fa8b47ac85a7d178ddc884f` |
| `packed/ternary-00168.safetensors` | `113df5f9af05aedf6b190f2519215dacbfda79f0a16bb3b8779b8e93e32df51e` |
| `packed/ternary-00169.safetensors` | `c08c22e0626cb4f5ccc7329f74593c3fcaeb0db245587f660a7881445f5b8c8c` |
| `packed/ternary-00170.safetensors` | `67993c6d3e8c1320397b27267b2a92056dc79c5ca0d408d6d713376b6fc9f7a8` |
| `packed/ternary-00171.safetensors` | `e6ec5514d11a0dddaf40adf77df678218974bd1a4edbef1b27913234abe4c739` |
| `packed/ternary-00172.safetensors` | `45887a9e0e58d12e25fbe83635e2745cd674044f3ced9c452bc3039a01cdb34e` |
| `packed/ternary-00173.safetensors` | `9128365045419e624c4eca8e3c0998c2ed37827bc5ebd1c953bffd2ef2be11f2` |
| `packed/ternary-00174.safetensors` | `e27dbaf6fd6b89d0abe799bccc83365fec6071c6b053f1952a75dde0c40c446e` |
| `packed/ternary-00175.safetensors` | `a16aa60ccf4af34feeda3889663cb678fb3f33135c6a11e17d47b1fe600c82b7` |
| `packed/ternary-00176.safetensors` | `6b5e2a2fff29a481e972215d78f60db0696df07b41c817cb20410ede55610583` |
| `packed/ternary-00177.safetensors` | `c383496326375940626995f626f1ef7762c3b05c4a6ce88492a7edc7149c100f` |
| `packed/ternary-00178.safetensors` | `467c8350aafc56aea256a0371a92733d78d5cbd8593574eba33a5c1a90aea486` |
| `packed/ternary-00179.safetensors` | `7f2c462065f9130a604e1e0173aa6f5d62f8ac3194a97f6aa1d8224c872f1548` |
| `packed/ternary-00180.safetensors` | `72e99c7acae944dcb80b0b04fe271d9b04f64faabecade64ce18aa3cc679bc66` |
| `packed/ternary-00181.safetensors` | `36862f520eb140ca97fe26384d91f07ce87a4960367c4a9048985faa0796235a` |
| `packed/ternary-00182.safetensors` | `6140ca35f0b4b62f342acc086055395b40194360894eb5924acfb9b3e99149fc` |
| `packed/ternary-00183.safetensors` | `bfe85b6a10bb32053c66c26a28a00086d1cc1ef014ae1f4db6bed89203882e00` |
| `packed/ternary-00184.safetensors` | `e96c13b90736edc19630f30a1499653868b6f2937a518f8560295652bfd8446c` |
| `packed/ternary-00185.safetensors` | `d958a239394e3687e49f693d492b87ca75bfabf5bba12e13b66a58d9c57148e3` |
| `packed/ternary-00186.safetensors` | `25f7040650d6e8486dcf5e5c3bf61e2c27ffee78faef964b49ef5ef366f4d522` |
| `packed/ternary-00187.safetensors` | `07e38a168c20fa92952edb8f4dcf8075649f3c4ccbd8a98520d712a58cdedf17` |
| `packed/ternary-00188.safetensors` | `473319b0666b5699b4f5aa439b24b13edfa9735c35997acd0ddfb737d5fc7d32` |
| `packed/ternary-00189.safetensors` | `91f4cc313c7f366e12d15c940281106ee25800811adce5ce1703b46b9e959dd6` |
| `packed/ternary-00190.safetensors` | `184320a3aaaa74529b9500f1bb4d0e342ba34e7a9b7ea948562badd2c7abc7dc` |
| `packed/ternary-00191.safetensors` | `0a55d49d94be0d682de6f06f15c11118aca4df5eaa06e65049787246aa7d606a` |
| `packed/ternary-00192.safetensors` | `92ee7877174f40a383991173a674b7c0a773c97216df4408b454c1062c517800` |
| `packed/ternary-00193.safetensors` | `f6a76f0546817eac5d8254f696e6596295689e76ab4860b6bb69fadd208c3b08` |
| `packed/ternary-00194.safetensors` | `1744104b11f960faea2a70ffa765f0d5992477f8c7d328085485727ea992e3f5` |
| `packed/ternary-00195.safetensors` | `fa3d9a01326cb32782fcba1aeea623deae37a5598af1a78ba4defbb96496cf79` |
| `packed/ternary-00196.safetensors` | `badc458362c41fdd0035b2d60bb4f49ba7e9f41e3c1602b424e0b0449a4ce1a6` |
| `packed/ternary-00197.safetensors` | `e7474214e1ecef2af70cfe874516d887700913649e6e46d1e25f00cf4e34fc3b` |
| `packed/ternary-00198.safetensors` | `d48d771838d0ac8fa531db1c46681a6d07cbb704ec75c3a55faf02d2191c121a` |
| `packed/ternary-00199.safetensors` | `f4ca3f9c0d87b190cf80c94a17e821d71d88815dda04ab5e225666c987705bac` |
| `packed/tokenizer/chat_template.jinja` | `a92e1dd97cb1cb175c9b70c0828e146bea4371c2643319b661b777e89811972e` |
| `packed/tokenizer/tokenizer.json` | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` |
| `packed/tokenizer/tokenizer_config.json` | `551b5777495cac65d18528ca5337865f1f6aa61f6ef8c6972299ae16a0d1a7da` |
| `report.md` | `0a2f2521cd660c84328514e280d2aa9f47b5c7df6535d9b6ca100b65caa37dcf` |
| `sizes.json` | `f9b53e2e4263290300dca0c2663074d43162317accec63c5b8699fc253b797ec` |
| `training-report.json` | `35faaceb692dd24c68ce6c50354275d97897b8d0d60a48ac98c7bcff936d8257` |

</details>

