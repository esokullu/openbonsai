# 5090 rotation result and runtime integration

The completed Qwen3.5-9B comparison supports further rotation research, but
neither compressed model is useful yet. The loss is already present in hard
ternary execution before packing. This is an independent implementation; it
does not reproduce Prism's unpublished compression training recipe.

## Preserved matched experiment

| Condition | NLL | ARC | Useful generations | Complete artifact bytes |
|---|---:|---:|---:|---:|
| Original BF16 | 2.423777 | 19/20 | 2 complete, 2 relevant but truncated | Not measured in this comparison |
| Unrotated ternary | 14.437115 | 1/20 | 0/4 | 6,592,437,649 |
| Rotated ternary | 9.255346 | 8/20 | 0/4 | 6,596,776,129 |

These repeatedly inspected **development diagnostics** use 2,048 scored tokens,
20 ARC questions and four generation prompts. Both compressed conditions looped
to the 128-token cap on all four prompts. They are not a fresh final evaluation,
a broad benchmark reproduction, or evidence of Bonsai equivalence.

Both arms start from original BF16 revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, covering 200 decoder projections.
They use the same 16 records, eight epochs, sequence length 512, group 128,
rank 64, sigmoid factors, Huber delta 1, factor LR .005, LoRA LR .001,
gradient clip 1, weight decay 0, seed 5090 and window/stride 1. Each window has
128 updates. The rotated arm uses block size 4096 with fixed signs and matching
activation transforms. Endpoints, vision and small retained tensors have the
same treatment in both arms.

All 32 local hard-MSE windows improved in both conditions; gradient, optimizer,
snapshot, original-weight and original-logit checks passed. Final-layer relative
hidden RMSE remained 1.09071 unrotated and 1.07104 rotated. Local reconstruction
is conditioned on each student's quantized prefix, so an improved local loss
does not establish end-to-end fidelity.

Hard FP32-scale NLL was 14.439949 and 9.257334, close to packed values. All 200
payloads and 564 retained tensors passed checks; fresh-process packed logits
matched the packed reference exactly. Scale rounding and row-GEMM differences
were measured separately. Packing preserves learned values; BF16-to-ternary
conversion is not lossless.

The rotated artifact adds 4,338,480 bytes. Both totals include required tokenizer
and configuration files. Embedding/head BF16 tensors alone occupy
**4,068,474,880 bytes**, and vision tensors occupy **912,021,104 bytes**.
Subtracting vision does not create a standalone text artifact. Broad endpoint
compression is still required to approach Bonsai's language-model size.

Training took 731.303 seconds unrotated and 1,309.048 seconds rotated (1.79×),
with peak allocated memory of 20.207 and 20.221 GiB. The complete paired driver
took 104.55 minutes including evaluation/export. Packed generation measured
1.889 and 1.576 tokens/s with the reference runtime. Both conditions used a
separately accounted **13,824,425,984-byte decoded BF16 projection cache**;
this is additional memory, absent from the compressed files. No low-bit kernel
or acceleration claim follows from these results.

The source snapshot, configuration, datasets, reports and artifacts remain in
`D:\openbonsai\runs\2026-10-26-5090-rotations`. Its final manifest covers
2,465 files and 15,946,218,435 research bytes, SHA256
`d723275e9057ed27c8bb7e2b010c0e28ef5fa517ffb95294cac76830ccbb6c2f`.
Research-directory bytes are not deployable-model bytes. Preserve this sealed
run; the integration below does not retroactively change its provenance.

## Runtime code reconciled into the research branch

The completed experiment used memory/EOS fixes that were absent from the earlier
published research branch. This integration imports those changes from its
recorded source patches while preserving the newer streaming endpoint export:

- Decoder quantizers calculate FP32 statistics and low-rank corrections in
  bounded chunks, checkpointing their differentiable computations. Frozen
  last-soft snapshots remain separate from current hard forward parameters.
  Rotated chunks cover complete rows/groups before correction and statistics.
- Native replacement transfers ownership of frozen source storage instead of
  cloning the whole weight. Standalone quantizer construction still copies by
  default. Original projection weights are omitted from packed candidate
  payloads; retained BF16 endpoints and vision remain separately accounted.
- Candidate saving streams bounded rows; loading uses mmap plus owned tensor
  storage, avoiding a second full serialized blob and re-encoding. Native model
  parameters start on meta while constructor-created CPU buffers stay valid.
- Ternary codec choice counts the complete candidate, including rotation signs
  and BITCOS row indexes. Export still checks predicted versus actual bytes.
- `row_chunk_size` is recorded and can be overridden on load. Optional BF16
  linear caches preserve the same row GEMMs and are nonpersistent. They default
  to off for newly prepared artifacts and older artifacts without a cache flag.
  A saved explicit opt-in is restored by `load_mixed`; pass
  `cache_decoded_weights=False` to force cold loading. Warm caches require
  two additional bytes per weight, are invalidated on device/dtype or ordinary
  payload changes, and are excluded from file budgets.
- Calibration generation honors the union of model and tokenizer EOS IDs.
  Decoder training reports synchronized cache, evaluation, optimization and
  cleanup phase durations, excluding initial model/configuration setup.

This bounds FP32 intermediates, not total model memory. A full source-dtype
projection weight is still reconstructed during calibration. Endpoint
calibration, larger contexts and batches need their own CUDA preflight. The
embedding rotation prototype is not a memory-bounded embedding trainer.

[Validation](../validation-5090-runtime.json) records 699 CPU tests, paired
native decoder smoke checks, and output-head fit/export/fresh-process checks.
The head smoke has exactly unchanged KL from the previous branch, and zero
reload logit difference. The new runtime-options manifest adds 109 bytes per
tiny fixture artifact. This reconciled branch has not itself rerun the 9B
experiment; its CPU validation and the sealed remote measurements are distinct.

## Completed calibration-diversity comparison

The handwritten source above contained only 16 prompts. Its proposed same-source
128-sample expansion stopped before GPU work because a larger pool did not
exist. A separate, preregistered rotated pair then used nested 16- and 128-record
sets from UltraChat `train_sft` revision
`8049631c405ae6576f93f445c6b8166f76f5505a`, with original-BF16 continuations.
It retained the same model revision, 200 decoder projections, rotation and
128-update window recipe. Valid-token exposure was 59,856 versus 59,855 per
window: 16 records repeated eight times versus 128 records once. It tests that
breadth/repetition choice, not more optimization updates or the effect of
UltraChat versus handwritten prompts in isolation.

| Outcome | Rotated 16×8 | Rotated 128×1 |
|---|---:|---:|
| Hard FP32-scale NLL | 8.499409802891023 | 9.121630685214768 |
| Fresh packed NLL (reported to six decimals) | 8.498620 | 9.123808 |
| ARC correct | 8/20 | 8/20 |
| ARC predictions | A on all 20 | A on all 20 |
| Useful generations | 0/4 | 0/4 |
| Complete deployable directory bytes | 6,596,784,699 | 6,596,830,115 |
| Training seconds | 1,303.809 | 3,965.794 |

The 16×8 model produced two empty answers and two fragments; the 128×1 model
emitted immediate EOS on all four prompts. These differ from the older
experiment's looping failure. Eight of the 20 ARC answers are A, so 8/20 is not
evidence of useful question answering. The same repeatedly used development
protocol scores 2,048 NLL targets, 20 ARC items and four greedy non-thinking
generations capped at 128 tokens. Original BF16 NLL 2.4237768106 and ARC 19/20
were reused with authenticated source/probe/protocol bindings, not newly scored
on an untouched test set. Neither compressed arm recovered useful quality.

Both artifacts passed exact fresh-process parity against their **exported**
references: 200 candidate payloads and 564 retained tensors. This does not mean
FP32 calibration weights are unchanged by export. The 128×1 fixed-probe maximum
logit difference between the FP32 wrapper and packed reference was 0.109375;
FP16 scale conversion and the runtime's row-GEMM layout were measured separately.
The severe quality loss already exists before either export or packing.

The directory totals include tokenizer/configuration files and vision. Each
retains 4,068,474,880 bytes in BF16 embedding/head tensors. Vision **files** occupy
912,058,304 bytes, including headers; the older table's 912,021,104 figure counts
vision **tensor payloads**. Vision is already included in the totals and must
not be added again. Equivalent represented-scope tensor bytes are 6,576,199,974
and 6,576,245,389, versus 18,819,627,488 original BF16-loaded persistent tensor
bytes. The latter is not an original deployable-folder measurement.

Peak training allocated memory was 20.221 and 20.219 GiB. The 128×1 training
phase includes 2,604.822 seconds caching and 935.228 seconds optimization, so
additional distinct records increased more than the optimizer work. Evaluation
used a separate 13,824,425,984-byte decoded BF16 projection cache, excluded from
artifact storage. Shorter failed generations are not equal-work speed evidence.
These timings and sizes do not establish a useful approximately 2 GB model.

The sealed run is `D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2`.
Its final inventory contains 2,034 files / 16,192,227,790 research bytes,
SHA256 `d7ae0ccf66b38f223366beb105beba32d651e5c2c1be166c21624eac7885bf8a`.
This includes two artifacts and research evidence, not one deployment. The
frozen runtime archive hash is
`0a90d78f775d2c1e879c286646b4d545ec286231f97687e4bb99ea4f961f61ce`;
its source/model/data identities and earlier failed attempt remain preserved.
These results come from the reviewed remote receipts, not a rerun of this branch.

## Evidence guiding the selected next control

A subsequent [activation diagnostic](activation-diagnostic.md) restored the
final128 FP32 research state and measured fixed windows 0 and 16 on 16 additional
disjoint search records. All later-window records showed material disagreement
between original-prefix and conditional teacher outputs, while original/bypass/
hard restoration, state preservation and the window-0 negative control passed.
It supports one matched, warm-start window-16 target-provenance control; no
quality result from that control is available yet.

A separate initializer audit reproduces the neutral CAT-Q hard mapping. For
Gaussian weights, expected hard/source RMS is about .66274; shrinkage alone
does not establish a missing-normalization bug. The 128-update schedule has
102 soft and 26 hard updates, with only 4.25% of the cumulative decayed learning
rate in the hard phase. Neither observation proves a quality remedy. The next
control keeps the schedule and optimizer fixed while changing only its target.

Embedding/head calibration and a matched Prism inference evaluation remain
separate prerequisites. RCO is preserved for limited exceptions and has no
measured 9B quality result yet. No large allocation search is scheduled.
