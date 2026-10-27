# Corrected Qwen3.5-9B compression experiment

The corrected 16×1 pipeline completed with finite training and exact fresh-reload parity. Its packed diagnostic NLL is 11.191338, 2.066354 nats below the original failed pilot, but ARC is 1/20. All four answers consist of 128 repetitions of “box” and hit the cap without EOS: **0/4 answers are task-relevant or usable. No useful quality recovery was demonstrated.**

Model [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a), revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; isolated source base `0cbf667fb5f65ef2e6a03ab269431dcb8f0f7db5`. This is a small research pilot on `NVIDIA GeForce RTX 5090`. Original pilot and diagnosis directories are read-only references.

## Measured quality

Diagnostic rows use identical frozen 2,048 scored next tokens, 20 ARC-Easy questions, and four held-out generation prompts. NLL is correctly shifted, padding-masked, and aggregated by scored tokens. ARC uses conditional option-label likelihood. Decoding is greedy, nonthinking, no KV cache, with the same 128-token cap and EOS IDs. Unfinished counts are cap hits without EOS.

| State | NLL | ΔNLL vs BF16 | Perplexity | ARC correct | ΔARC | Unfinished |
|---|---:|---:|---:|---:|---:|---:|
| Original BF16 | 2.423777 | +0.000000 | 11.288413 | 19/20 | +0.0 pp | 2/4 |
| Original failed pilot packed | 13.257692 | +10.833916 | 572,456.680041 | 1/20 | -90.0 pp | 4/4 |
| Corrected 16×1: Hard learned FP32 | 11.189648 | +8.765872 | 72,377.324246 | 1/20 | -90.0 pp | 4/4 |
| Corrected 16×1: In-memory exported FP16 scales | 11.191338 | +8.767561 | 72,499.732925 | 1/20 | -90.0 pp | 4/4 |
| Corrected 16×1: Fresh-process packed reload | 11.191338 | +8.767561 | 72,499.732925 | 1/20 | -90.0 pp | 4/4 |

The FP32 stage retains the learned correction and factors at FP32 while original weights and model computation use BF16. Export folds the low-rank correction into hard codes/scales and stores scales in FP16; A/B are not a separate deployable adapter. Fresh reload is a separate process. No quality claim follows from exact serialization alone.

### Complete pilot-sized comparison

These rows use **16,384 scored tokens / 100 ARC questions / ten generation prompts** and have different denominators from the diagnostic table.

| State | NLL | ΔNLL vs BF16 | Perplexity | ARC correct | ΔARC | Unfinished |
|---|---:|---:|---:|---:|---:|---:|
| Original BF16 | 2.217540 | +0.000000 | 9.184705 | 97/100 | +0.0 pp | 4/10 |
| Original failed pilot | 12.699258 | +10.481718 | 327,504.720532 | 23/100 | -74.0 pp | 10/10 |

No corrected full evaluation was launched under the recorded gate decision. Only the preserved original pilot has full-suite scores here.

## Recorded gate decision

```json
{
  "status": "completed",
  "stopped_at_completed_boundary": true,
  "bounded_screen_launched": false,
  "larger_run_launched": false,
  "full_evaluation_launched": false,
  "reason": "The latest delegated user instruction superseded automatic escalation and required stopping after the active corrected16x1 run. That run completed with finite updates and exact saved/reloaded parity, but did not recover useful quality. No unstarted training or search stage was launched.",
  "controlled16x1_diagnostics": {
    "hard_fp32": {
      "scope": "completed controlled16x1 diagnostic; not the unexecuted bounded16x8 screen",
      "nll": 11.189648328290787,
      "perplexity": 72377.32424576565,
      "nll_improvement_over_failed_diagnostic": 2.0680440152063966,
      "nll_improvement_at_least_2": true,
      "arc_correct": 1,
      "arc_total": 20,
      "arc_at_least_10": false,
      "generations_total": 4,
      "useful_generations": 0,
      "unfinished_generations": 4,
      "qualitative_assessment": "All four inspected answers are exactly 128 repetitions of box. They do not answer the instruction, reasoning, coding, or Turkish prompt.",
      "all_four_intelligible_relevant_nondegenerate": false,
      "passes_exploratory_quality_criteria": false,
      "elapsed_seconds": 215.45865749986842,
      "peak_allocated_bytes": 21043123712,
      "peak_reserved_bytes": 21340618752
    },
    "exported_fp16": {
      "scope": "completed controlled16x1 diagnostic; not the unexecuted bounded16x8 screen",
      "nll": 11.191338157048449,
      "perplexity": 72499.73292540392,
      "nll_improvement_over_failed_diagnostic": 2.0663541864487343,
      "nll_improvement_at_least_2": true,
      "arc_correct": 1,
      "arc_total": 20,
      "arc_at_least_10": false,
      "generations_total": 4,
      "useful_generations": 0,
      "unfinished_generations": 4,
      "qualitative_assessment": "All four inspected answers are exactly 128 repetitions of box. They do not answer the instruction, reasoning, coding, or Turkish prompt.",
      "all_four_intelligible_relevant_nondegenerate": false,
      "passes_exploratory_quality_criteria": false,
      "elapsed_seconds": 388.6006669998169,
      "peak_allocated_bytes": 7092318720,
      "peak_reserved_bytes": 7384072192
    },
    "fresh_packed": {
      "scope": "completed controlled16x1 diagnostic; not the unexecuted bounded16x8 screen",
      "nll": 11.191338157048449,
      "perplexity": 72499.73292540392,
      "nll_improvement_over_failed_diagnostic": 2.0663541864487343,
      "nll_improvement_at_least_2": true,
      "arc_correct": 1,
      "arc_total": 20,
      "arc_at_least_10": false,
      "generations_total": 4,
      "useful_generations": 0,
      "unfinished_generations": 4,
      "qualitative_assessment": "All four inspected answers are exactly 128 repetitions of box. They do not answer the instruction, reasoning, coding, or Turkish prompt.",
      "all_four_intelligible_relevant_nondegenerate": false,
      "passes_exploratory_quality_criteria": false,
      "elapsed_seconds": 419.0268451003358,
      "peak_allocated_bytes": 7043415552,
      "peak_reserved_bytes": 7344226304
    }
  },
  "bounded_screen_gate": {
    "status": "not_evaluated",
    "reason": "bounded16x8 was not launched under the superseding boundary instruction"
  },
  "full_gate": {
    "status": "not_evaluated",
    "reason": "No corrected full-suite evaluation was launched"
  },
  "next_direction": "Await a concrete bounded prompt for separately developed ScaleQ-inspired calibration plus actual RCO mixed-precision allocation and BITCOS. This combination is a research hypothesis, not a result of the completed run.",
  "preserved": [
    "original failed pilot",
    "original diagnosis and CPU BITCOS repack",
    "original calibration traces",
    "all previous evaluation outputs and local source fixes",
    "corrected16x1 packed model and learned FP32 research state"
  ]
}
```

The historical expansion plan specified a diagnostic gate for both learned FP32 and reloaded packed states: at least two nats improvement over the failed pilot, at least 10/20 ARC answers, and four intelligible, relevant, nondegenerate generations. Full confirmation, when authorized by that gate, uses at least two nats improvement, 50/100 ARC, and eight of ten meaningful outputs. These are resource-allocation thresholds, not evidence of lossless or near-BF16 compression. The later instruction to stop at the completed 16×1 boundary supersedes that automatic expansion plan. The bounded 16×8, full confirmation, and 128×3 experiments were not run; their gates have no measured result.

## Actual training and learning checks

| Run | Samples × epochs | Mapping / objective | Factor / A-B LR | Windows / updates | Soft / hard updates | Actual last soft k |
|---|---:|---|---|---:|---:|---|
| Corrected 16×1 | 16 × 1 | exp / mse | 0.001 / 0.001 | 32 / 512 | 384 / 128 | [28.125] |

Schedules progress continuously across all epochs within each window. The frozen hard-stage Jacobian uses the actual last soft forward, before its optimizer update, including its saved A/B and sharpness. All logged parameters, gradients, updates, and optimizer moments are finite; trainable parameters and Adam moments remain FP32. Unused snapshots are cleared after each window. Hard MSE is evaluated separately from the optimization objective; Huber objective values are not mislabeled as MSE.

### Corrected 16×1

First-step A gradient is zero in 32/32 windows, while B gradient is nonzero in 32/32. Later A gradients are nonzero in 480 steps. A zero initial A gradient is expected with zero-initialized B; subsequent gradients and actual updates demonstrate learning.

| Parameter family | Preclip gradient L2 min–max | Actual update L2 min–max | Nonzero update steps |
|---|---:|---:|---:|
| raw_mu | 7.94712e-05 – 0.149069 | 0.0301007 – 1.24262 | 512/512 |
| raw_scale | 6.61511e-05 – 0.0857093 | 0.0268189 – 1.23613 | 512/512 |
| raw_threshold | 3.06404e-05 – 0.0736969 | 0.0248698 – 1.21682 | 512/512 |
| lora_A | 0 – 0.0877016 | 0 – 1.08237 | 480/512 |
| lora_B | 6.70401e-06 – 0.106417 | 0.027378 – 1.44529 | 512/512 |

<details>
<summary>Window hard-MSE, correction, and representative weight statistics</summary>

Each row's RMS/zero fraction probes the first selected projection in that window. Correction size is the sum-of-squares norm across the window's selected corrections, computed through rank-sized Gram matrices. These local reconstruction measurements do not substitute for model quality.

| Layers | Hard MSE before → after | Final objective loss | Correction Frobenius | Representative effective/original RMS | Representative hard/original RMS | Final zero fraction |
|---|---:|---:|---:|---:|---:|---:|
| 0:1 | 0.0047237 → 0.00336985 | 0.00356474 | 0.0918194 | 1.000002 | 0.641905 | 0.317678 |
| 1:2 | 0.0047864 → 0.00424548 | 0.00324786 | 0.049626 | 1.000000 | 0.655381 | 0.313025 |
| 2:3 | 0.0232223 → 0.0211263 | 0.0229143 | 0.114115 | 1.000000 | 0.653793 | 0.313421 |
| 3:4 | 0.00292919 → 0.00212187 | 0.002854 | 0.0559241 | 1.000000 | 0.658688 | 0.311318 |
| 4:5 | 0.0073426 → 0.00661521 | 0.00678808 | 0.0555114 | 1.000001 | 0.655871 | 0.312686 |
| 5:6 | 0.00643541 → 0.00528831 | 0.00571532 | 0.100659 | 1.000001 | 0.654965 | 0.313104 |
| 6:7 | 0.0205565 → 0.0175601 | 0.0173199 | 0.174284 | 1.000003 | 0.651608 | 0.315432 |
| 7:8 | 0.00565823 → 0.00320157 | 0.00484519 | 0.150433 | 1.000000 | 0.654995 | 0.314193 |
| 8:9 | 0.024147 → 0.0134644 | 0.0136393 | 0.320008 | 1.000006 | 0.653136 | 0.314735 |
| 9:10 | 0.0131526 → 0.006047 | 0.00607968 | 0.293787 | 1.000002 | 0.654592 | 0.313847 |
| 10:11 | 0.0171748 → 0.00699039 | 0.00732653 | 0.343065 | 1.000002 | 0.653382 | 0.313989 |
| 11:12 | 0.213767 → 0.068392 | 0.0658983 | 0.461055 | 1.000003 | 0.652493 | 0.315886 |
| 12:13 | 0.136804 → 0.0465783 | 0.0474293 | 0.510126 | 1.000007 | 0.652942 | 0.314786 |
| 13:14 | 0.08654 → 0.0314178 | 0.0280459 | 0.484595 | 1.000005 | 0.654370 | 0.314045 |
| 14:15 | 0.0729553 → 0.0242521 | 0.0254711 | 0.47167 | 1.000005 | 0.649737 | 0.316054 |
| 15:16 | 0.114604 → 0.0318719 | 0.0325369 | 0.459805 | 1.000003 | 0.652971 | 0.314796 |
| 16:17 | 0.0678907 → 0.0213018 | 0.0215866 | 0.462423 | 1.000005 | 0.652434 | 0.314873 |
| 17:18 | 0.0792086 → 0.0230229 | 0.0237346 | 0.480625 | 1.000004 | 0.652854 | 0.314906 |
| 18:19 | 0.146693 → 0.0400866 | 0.0409605 | 0.524669 | 1.000006 | 0.648832 | 0.316763 |
| 19:20 | 0.143381 → 0.0430453 | 0.0436959 | 0.477613 | 1.000003 | 0.651518 | 0.316419 |
| 20:21 | 0.12274 → 0.032613 | 0.0339416 | 0.5267 | 1.000006 | 0.648551 | 0.317139 |
| 21:22 | 0.122955 → 0.0311586 | 0.0322619 | 0.520193 | 1.000004 | 0.648205 | 0.317170 |
| 22:23 | 0.247833 → 0.0819401 | 0.0845084 | 0.538227 | 1.000004 | 0.650340 | 0.316508 |
| 23:24 | 0.772266 → 0.337636 | 0.336757 | 0.572547 | 1.000002 | 0.655169 | 0.313827 |
| 24:25 | 0.143979 → 0.0363408 | 0.0369782 | 0.541872 | 1.000003 | 0.650408 | 0.316078 |
| 25:26 | 0.0862389 → 0.0189351 | 0.0191712 | 0.517636 | 1.000001 | 0.650502 | 0.315632 |
| 26:27 | 0.363065 → 0.172887 | 0.176846 | 0.558624 | 1.000004 | 0.654270 | 0.314229 |
| 27:28 | 0.641557 → 0.224101 | 0.232718 | 0.552651 | 1.000005 | 0.653233 | 0.314310 |
| 28:29 | 0.212999 → 0.0558564 | 0.0553658 | 0.562339 | 1.000004 | 0.652783 | 0.314751 |
| 29:30 | 0.200358 → 0.0568854 | 0.0596096 | 0.550858 | 1.000004 | 0.655449 | 0.313006 |
| 30:31 | 0.469606 → 0.135759 | 0.137813 | 0.584307 | 1.000006 | 0.647949 | 0.317133 |
| 31:32 | 2.22292 → 0.630722 | 0.638656 | 0.571052 | 1.000000 | 0.651928 | 0.314222 |

</details>

Local hard MSE improves in 32/32 trained windows. The following hidden-state drift uses only the first fixed NLL probe, not a general activation distribution. It compares the complete learned hard-FP32 model to the original BF16 model and therefore includes accumulated upstream differences.

| Layer | Original RMS | Learned hard RMS | Relative RMSE | Cosine |
|---:|---:|---:|---:|---:|
| 0 | 0.109256 | 0.050653 | 0.592417 | 0.932251 |
| 7 | 0.349093 | 0.165399 | 0.688094 | 0.792831 |
| 15 | 0.574253 | 0.479539 | 0.968908 | 0.454213 |
| 23 | 1.391988 | 0.779441 | 0.976136 | 0.322073 |
| 31 | 3.671063 | 2.282260 | 1.089220 | 0.160919 |

Improved local window reconstruction does not establish preserved whole-model language or reasoning quality.

## Serialization and deployable size

| Run | Saved projection payloads exact | Retained tensors checked | FP32→FP16 max logit difference | Fresh reload max difference | Research state bytes (excluded) |
|---|---:|---:|---:|---:|---:|
| Corrected 16×1 | 200/200 | 564 | 0.125 | 0 | 1,290,046,185 |

| Artifact | Code streams | Scales | Retained floating tensors | Model artifact | Tokenizer | Deployable total |
|---|---:|---:|---:|---:|---:|---:|
| Original failed pilot (2-bit) | 1,728,053,248 | 108,003,328 | 4,995,201,904 | 6,831,783,487 | 19,998,761 | 6,851,782,248 |
| Corrected 16×1 | 1,457,557,554 | 108,003,328 | 4,995,201,904 | 6,561,490,248 | 19,998,761 | 6,581,489,009 |

Retained floating-point components in the newly trained corrected artifact:

| Component | Bytes |
|---|---:|
| output head | 2,034,237,440 |
| token embeddings | 2,034,237,440 |
| text normalization | 546,816 |
| linear attention dynamics | 3,072 |
| linear attention convolutions | 1,572,864 |
| linear attention gates | 12,582,912 |
| text rotary buffers | 256 |
| vision encoder including vision rotary buffers | 912,021,104 |
| **Total retained tensors, including generated buffers** | **4,995,201,904** |

Equivalent original checkpoint scope: 18,819,619,808 BF16 tensor bytes plus 15,360 other floating-point bytes = 18,819,635,168 tensor bytes. The loader omits 15 original MTP tensors totaling 486,581,248 bytes; those are excluded from the equivalent-scope denominator. Retained embeddings, output head, vision, normalizations, recurrent components, and buffers remain included. No duplicate Hugging Face cache snapshots count toward model size.

The independent accounting audit distinguishes the original checkpoint bytes from the actual persistent tensors after BF16 loading. Generated nonpersistent buffers remain included in artifact/deployment bytes but are excluded from the strict persistent-tensor ratios below.

| Run | Original checkpoint persistent bytes | Loaded BF16 persistent bytes | Compressed persistent tensor bytes | Ratio vs loaded BF16 | Reduction vs loaded BF16 |
|---|---:|---:|---:|---:|---:|
| Corrected 16×1 | 18,819,635,168 | 18,819,627,488 | 6,560,762,386 | 2.868512× | 65.138723% |

Corrected 16×1: 48 retained source FP32 tensors occupy 15,360 source bytes and 7,680 BF16-loaded bytes (difference 7,680). Expected FP32→BF16 rounding verified: **True**. The other 512 persistent retained tensors are raw source-byte identical: **True**. These casts occur during official BF16 loading and also apply to the BF16 baseline. The artifact additionally contains 4 generated nonpersistent buffers totaling 400 bytes. Against raw equivalent checkpoint bytes, the strict persistent-tensor ratio is 2.868513× and reduction 65.138738%.

Tensor ratios compare equivalent persistent model scope. Model artifact bytes additionally include generated buffers, containers, configuration, and manifest. Deployable totals additionally include tokenizer files. Learned FP32 factors/A/B, optimizer audit logs, caches, and saved probe tensors are research artifacts and excluded. Tokenizer assets are checksummed supplementary files. BITCOS/auto is lossless code storage; ternarization itself remains lossy. The reference runtime reconstructs dense weights, so file-size reduction does not establish lower inference latency or optimized low-bit execution.

### Separate CPU repack of the original failed artifact

The earlier CPU-only BITCOS repack reduced the original failed model artifact from 6,831,783,487 to 6,561,349,799 bytes, saving 270,433,688 bytes (3.958464% of that original packed artifact). Its tokenizer stayed at 19,998,761 bytes. Decoded codes/scales/biases remained exact: **True**; retained tensors exact: **True**; tokenizer exact: **True**. The separate CUDA reference-logit check was exact: **True**, maximum difference 0. That repack did no training and preserved the old failed model's quality. It is a different artifact from the newly trained corrected 16×1 model above.

The [original diagnosis report](../2026-10-26-5090-diagnosis/report.md) contains the disabled-wrapper checks, projection-family and depth controls, and fixed-code least-squares recovery experiment.

## Runtime and CUDA memory

Measured real-run driver wall time is **1617.238147 seconds** (26.95 minutes). Its completed subprocess durations are `corrected16x1-train` 1108.664713 s; `corrected16x1-reload` 508.406013 s; `corrected16x1-size` 0.163400 s. This driver measurement includes subprocess startup and orchestration; preflights and earlier experiments are separate.

Allocated/reserved values are absolute PyTorch peaks, not free-VRAM readings. Trainer phase counters reset; the outer training peak is validated as at least the maximum of all trainer phase peaks. Outer training time includes the inner phases and audit overhead; these rows must not be added together.

| Run / operation | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|
| Corrected 16×1 / pipeline: load_original_bf16 | 18.076 | 17.538 | 17.588 |
| Corrected 16×1 / pipeline: conversion_rank64 | 1.027 | 19.271 | 19.334 |
| Corrected 16×1 / pipeline: training | 231.302 | 20.208 | 20.297 |
| Corrected 16×1 / pipeline: save_research_state | 2.464 | 19.213 | 19.254 |
| Corrected 16×1 / pipeline: export_bitcos_auto | 111.338 | 19.401 | 19.510 |
| Corrected 16×1 / pipeline: export_tensor_verification_and_fp16_reference | 130.664 | 19.401 | 19.443 |
| Corrected 16×1 / fresh reload: loading | 79.079 | 6.143 | 6.203 |
| Corrected 16×1 / fresh reload: exact_parity | 1.902 | 6.371 | 6.545 |
| Corrected 16×1 / inner training: cache | 121.419 | 19.520 | 20.297 |
| Corrected 16×1 / inner training: hard_evaluation_before | 15.292 | 19.517 | 20.297 |
| Corrected 16×1 / inner training: audit_boundary_before | 0.334 | 19.468 | 20.297 |
| Corrected 16×1 / inner training: optimization | 81.739 | 20.208 | 20.297 |
| Corrected 16×1 / inner training: freeze_window | 0.008 | 19.377 | 20.297 |
| Corrected 16×1 / inner training: hard_evaluation_after | 11.505 | 19.592 | 20.297 |
| Corrected 16×1 / inner training: audit_boundary_after | 0.339 | 19.546 | 20.297 |
| Corrected 16×1 / inner training: cleanup | 0.545 | 19.295 | 20.297 |

Evaluation is measured separately, on already resident models. FP32-wrapper diagnostics retain original frozen weights and correction state; their memory is not a deployable-checkpoint measurement.

| Run / stage | Seconds | Peak allocated / reserved GiB | Scored NLL tokens/s | Generated tokens/s |
|---|---:|---:|---:|---:|
| Corrected 16×1 / Hard learned FP32 | 215.459 | 19.598 / 19.875 | 1067.334 | 2.512 |
| Corrected 16×1 / In-memory exported FP16 scales | 388.601 | 6.605 / 6.877 | 723.873 | 1.374 |
| Corrected 16×1 / Fresh-process packed reload | 419.027 | 6.560 / 6.840 | 401.690 | 1.288 |

## CUDA preflight, implementation, and provenance

Integrated CPU verification: 370 passed, 2 skipped (Windows symlink privilege). Environment: PyTorch 2.14.1+cu130, CUDA build 13.0, GPU NVIDIA GeForce RTX 5090. The random native-width and native hybrid fixtures establish software execution only; their outputs do not measure pretrained quality.

| Native-width variant | Shape / sequence | A first/second/hard gradient L2 | B first gradient L2 | Peak allocated/reserved GiB |
|---|---|---|---:|---:|
| exp / mse | [12288, 4096] / 512 | [0.0, 3.052963393201935e-06, 3.1185370517050615e-06] | 1.63085e-05 | 0.656 / 0.693 |
| sigmoid / huber | [12288, 4096] / 512 | [0.0, 1.1549096825547167e-06, 1.166287461273896e-06] | 8.15361e-06 | 0.656 / 0.705 |

`preflight-exp-mse`: rank 64, 1 epoch(s), checkpointed BF16 native hybrid forward/backward/export, A/B changed in 25/25 projections; teacher unchanged True; separate-process packed parity True.

`preflight-sigmoid-huber`: rank 64, 8 epoch(s), checkpointed BF16 native hybrid forward/backward/export, A/B changed in 25/25 projections; teacher unchanged True; separate-process packed parity True.

Integrated changes: bounded group-aligned FP32 reconstruction and differentiable rank64 effective statistics; actual saved soft-forward Jacobian preserved through nested checkpointing; source storage transfer and meta-parameter reload with native buffers; FP32 factors,A/B,statistics,soft snapshots and optimizer-state checks; dual EOS stopping and tokenizer output compatibility; optional gradient/update/schedule/window/phase instrumentation; unchanged fixed evaluation helpers; tokenizer supplementary checksums.

Starting original HEAD `891d260af96a7ce61ccbcb23474015d975bb18dc`; new base `0cbf667fb5f65ef2e6a03ab269431dcb8f0f7db5`. 30 original local files were copied into starting-state evidence. Integration patch SHA256 `db6c408b43182704d5544a27f082cec1396e756df6fdeea1718f4b6db5e5ad1e`; source snapshot `sources/integrated-preflight`. Every real run records source hashes and checks that sources remain unchanged through training and evaluation. The source snapshot and model revision are separate provenance records.

### Exact Corrected 16×1 source hashes

Every current source file below was rehashed by this report builder and matches the hash recorded before the run. These hashes identify the executed Python source; they are distinct from model, dataset, and artifact hashes.

| Source file | SHA256 |
|---|---|
| `openbonsai\__init__.py` | `a6c56e7e9dede4ca074bb8dcae6c598cc76ee1f680f1e8274ce4bfb2eda7017e` |
| `openbonsai\__main__.py` | `102942c02cdbea1e98069b7486b9c6e21d2694d84d78fca0ff104d352566d68b` |
| `openbonsai\adapter.py` | `9309c3b2ab9bf7e7bfdb629ac083b527f765ba43c184412f96285df7f177497d` |
| `openbonsai\calibration.py` | `7b6d87d83157e9b61d7ac57bccd0865ee9b12709685d8694f5a4048f80d01808` |
| `openbonsai\cli.py` | `8d82b10f62401e805cd1b4c37011627e1e312c49b2d037b08085e2406e214a5c` |
| `openbonsai\export.py` | `2bc8b126e0970822fbba0b7685645549d96757737ea52d06bce95c8a2372cd1e` |
| `openbonsai\packing.py` | `4162082d5922fc3be76272e3496a060a1130d4ac1a4d08ede8f57bf95d3f7365` |
| `openbonsai\prepare.py` | `730c2a4815a285d121e9c87589395c5a8e8000b65fc14ee67cac045c83dfef45` |
| `openbonsai\quantizer.py` | `f3845c9e7f51550cdff4f502fc57c842ab9df79d99405977e0b01b8d54b59e22` |
| `openbonsai\trainer.py` | `07abd9459bf7b32db307d3438c2512f87e16e227785d6fba7ec9b7ce4a6003ce` |
| `scripts\capture_experiment.py` | `95f446502143fdc1747356c36db2a2608cee88a93f9e91c16f0754b79dc6b4e7` |
| `scripts\diagnostic_eval.py` | `776acb4861bf274a2087e0e1452a87e305f3dac21e270e2f9c6da7635c585d0e` |
| `scripts\evaluate.py` | `44c6e6a90f35b10546029c9c9179563601b7bfc4cee3a122cfb999c42b4cad23` |
| `scripts\model_size.py` | `9e9ef154dba4d14750c7cff24e0fe9f218b5ef9bcfad81905a6aacc055a7c500` |
| `scripts\pilot.py` | `c6faf0bfecfc62fb276b205e485a593ebdb917acfeb558e298d18aa26277afa3` |
| `scripts\research_runner.py` | `654840c106f1481331a85e84d964f0afc8ac687518e76d4d047aa804003e6c48` |

Preservation audit at 2026-10-26T03:49:57.189630+00:00: 970 hash checks; all pass **True**. Scope follows the prior manifests and explicitly excludes unlisted files and the old worktree, so this is not a claim about every byte in those directories.

| Preserved scope | Expected / checked files | All recorded hashes match |
|---|---:|---|
| pilot | 231 / 231 | True |
| diagnosis | 236 / 236 | True |
| diagnosis_artifact_payloads | 441 / 441 | True |
| original_local_sources | 30 / 30 | True |
| starting_local_snapshot | 30 / 30 | True |

Original repository HEAD unchanged: **True**; dirty source inventory matches the start: **True**. The audit's `pending_export` observation is historical: the export had not existed at that audit timestamp. It is superseded by the completed, **passed** `corrected16x1/artifact-accounting-audit.json`, which verifies the saved artifact, tokenizer, retained tensors, byte denominators, and reload evidence. There is no outstanding export-accounting work.

## Calibration, scope, and limitations

Corrected 16×1: 16 original-model AYOT traces; 5330 source tokens, 5234 active training tokens, 2958 padding tokens. 96 tokens removed from 3 records; 3 prepared traces hit their generation cap. Calibration hash `09df0a26800a20e46288b5a775075c91dc5c0e98bc0bf81ac7906f304a0d820f`.

Calibration and evaluation are disjoint; calibration uses original-model nonthinking traces and the recorded EOS union. The diagnostic/full fixed sets have now informed repeated method development and allocation decisions, so they are not fresh unbiased test sets for future claims. The sixteen-example calibration is short-context and text-only. Long-context, multimodal, task-finetuned, and deployment-kernel behavior remain untested.

Rank64 structural corrections are combined in the controlled 16×1 run. The proposed bounded 16×8 experiment was not run. It would have changed budget, factor mapping, objective, and factor learning rate together. Huber delta 1.0 is an explicit preflight/plan choice; the released training threshold is unavailable. Neither larger budgets nor lower local reconstruction loss are assumed to restore quality.

Frozen diagnostic inputs SHA256 `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`; full inputs `2952cf072dff2e9b00b09a59a5f2345a02b3e509725854af07ea8478f8a7a64c`. [WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3) and [ARC-Easy validation](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) revisions/sample IDs/seeds are preserved in those files. The known source-ID metadata caveat includes separator-only empty WikiText rows; the scored token stream and offsets were audited and remain unchanged.

## Evidence

`report-evidence.json` records the exact source-file hashes and per-run training summaries used to build this report. Raw evaluation JSONs preserve per-example losses, answer scores, generated IDs/text, timings, and memory. No model or training run is launched by this report builder.

