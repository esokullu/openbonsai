# OpenBonsai RTX 5090 small-pilot result

Completed a local compression pilot of the untouched official [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) BF16 checkpoint at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. No task fine-tuning was applied. Calibration used 16 self-generated samples; compression used 1 epoch(s). This is a small execution and quality experiment, not a lossless-compression claim or paper-quality reproduction.

The artifact passed exact serialization parity, but its held-out NLL changed by +10.481718 and ARC-Easy accuracy changed by -74.0 percentage points. For this run, quality was not preserved. Matching evaluation inputs/settings and exact reload parity do not identify the cause of quality differences: calibration insufficiency, quantizer design, and implementation sensitivities have not been isolated. Local reconstruction-loss improvement alone is insufficient evidence of useful model quality.

| Measurement | Original BF16 | Reloaded packed | Packed minus baseline |
|---|---:|---:|---:|
| Equivalent tensor payload | 18,819,635,168 B (17.527 GiB) | 6,831,258,480 B (6.362 GiB) | -11,988,376,688 B |
| Entire deployable directory¹ | 19,329,392,091 B (18.002 GiB) | 6,851,782,248 B (6.381 GiB) | -12,477,609,843 B |
| NLL, 16,384 scored tokens | 2.217540 | 12.699258 | +10.481718 |
| Perplexity | 9.184705 | 327,504.720532 | +327,495.535827 |
| ARC-Easy correct / 100 | 97 (97.0%) | 23 (23.0%) | -74; -74.0 percentage points |
| Peak GPU allocated / reserved, load + evaluation | 17.954 / 18.191 GiB | 6.789 / 7.172 GiB | -11.165 / -11.020 GiB |
| Measured evaluation time, excluding load | 58.17 s (0.016 h) | 281.48 s (0.078 h) | +223.31 s |
| Full evaluation process measured time | 76.29 s (0.021 h) | 298.57 s (0.083 h) | +222.28 s |
| Unfinished qualitative outputs / 10 | 4 | 10 | +6 |

The packed model produced whitespace-only text in 10/10 qualitative examples. The side-by-side document preserves that whitespace, and the raw evaluation JSON retains every generated token ID.

The equivalent-payload reduction is **63.70% (2.755× smaller)**, using 18,819,635,168 original tensor bytes as denominator. ¹The full-directory reduction is 64.55% (2.821×), using 19,329,392,091 bytes from one official snapshot; this supplementary comparison includes source tensors unused by the loader and all files in that snapshot. The scope accounting below makes those differences explicit.

NLL and question comparisons have identical workloads. Generation uses identical prompts and limits but may produce different token counts; its wall time therefore measures each model's actual outputs. Storage, inference VRAM, and runtime are separate observations. The current packed runtime dequantizes projections for computation and has no optimized ternary kernel. Both models also use Transformers' reference PyTorch GatedDeltaNet/causal-convolution paths because the optional `flash-linear-attention` and `causal_conv1d` packages are absent. These timings describe this local reference runtime; they do not establish performance with optimized deployment kernels.

## Evaluation protocol

Frozen inputs: `eval-inputs.json`, SHA256 `2952cf072dff2e9b00b09a59a5f2345a02b3e509725854af07ea8478f8a7a64c`; seed 5090. The tokenizer and chat template are from the pinned source revision; template SHA256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Both models ran in separate processes with `torch.bfloat16`, nonthinking prompts, greedy argmax, EOS IDs [248044, 248046], 128 new-token cap, and `use_cache=False`.

- Language modeling: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3), configuration `wikitext-2-raw-v1`, split `test`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Exact row/sample IDs are frozen in the input file.
- Commonsense/science questions: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453), configuration `ARC-Easy`, split `validation`, revision `210d026faf9955653af8916fad021475a3f00453`. Exact row/sample IDs are frozen in the input file.

WikiText uses 32 segments of 513 tokens with a one-token boundary overlap, scoring 512 shifted labels per segment. Padding and segment-initial tokens are excluded; NLL is aggregated by scored-token count in float64 after float32 cross entropy. The 100 ARC-Easy items use conditional option-letter likelihood and deterministic argmax. This is a constrained multiple-choice score, not generated free-form answer accuracy; unfinished benchmark answers are not applicable. Ten separate manual instruction, reasoning, coding, and Turkish examples appear in [generations.md](generations.md), with full token IDs in both evaluation JSON files.

Provenance clarification: the frozen per-block WikiText `source_ids` also include empty separator rows, despite the `source_ids_scope` annotation saying they were omitted. This affects row attribution wording, not the frozen tokens or scored-token count. Token/character offsets define the actual scored spans; `evaluation-inputs-audit.json` records the independent verification and caveat.

## Storage accounting

| Scope | Exact bytes |
|---|---:|
| Entire original checkpoint tensor payload | 19,306,216,416 |
| Original tensors represented by artifact | 18,819,635,168 |
| BF16 bytes in that equivalent original scope | 18,819,619,808 |
| Other source dtypes in equivalent scope | 15,360 |
| Original checkpoint tensors unused by artifact/loader | 486,581,248 |
| Original serialized weight files, including headers | 19,306,310,880 |
| Other files in the official snapshot | 23,081,211 |
| Packed 2-bit codes | 1,728,053,248 |
| Packed FP16 group scales | 108,003,328 |
| Packed-module biases | 0 |
| Retained floating tensors and buffers | 4,995,201,904 |
| Artifact containers/config/manifest overhead | 525,007 |
| Model artifact, including configuration/manifest | 6,831,783,487 |
| Tokenizer directory | 19,998,761 |
| Deployable artifact + tokenizer | 6,851,782,248 |
| Additional files inside packed directory | 0 |
| Total packed directory | 6,851,782,248 |

15 original tensors are unused; exact names, shapes, dtypes, and bytes are listed in `sizes.json`. All persistent artifact tensors map to original checkpoint names/shapes. Packed codes plus scales use 2.125000 bits per targeted weight; 4 constructor-generated nonpersistent buffers are separately identified. Retained vision components remain present. The accounting reads one official snapshot and the packed directory; Hugging Face cache duplicates, scratch files, logs, and the Python environment are excluded.

## Compression execution

Actual configuration: CUDA BF16, 16 samples, sequence length 512, 1 epoch(s), batch 1, window 1, stride 1, group size [128], gradient checkpointing True, learning rate 0.001, seed 5090. Conservative projection policy; aggressive recurrent gates: False. Embeddings, vocabulary head, vision components, normalization, convolutions, and small recurrence gates are retained. Disk scratch is local to the run.

Preparation used nonthinking=True, sampling=False, new-token limit=512, EOS IDs=[248044, 248046], generation cache default=True. The 16 traces contain 557 prompt and 4,773 generated tokens; 3 were unfinished at the generation cap. Training retained 5,234 active tokens and 2,958 padding tokens; right truncation removed 96 tokens from 3 records. The original model generated the traces through the project's preparation pipeline. Generation caching can differ from evaluation, where both compared models explicitly disable it.

Optimization executed 512 updates across 32 windows: 384 soft and 128 hard updates. All recorded losses and gradient norms were finite; 512/512 updates had positive gradient norm (range 0.000122683–0.159048). Loss range: 0.0022127–1.35515. Per-window losses and timings are in `training-report.json`; individual updates are in `training-events.jsonl`.

Fresh-process reload matched the packed-reference logits exactly (maximum difference 0.0) and produced finite outputs. This verifies serialization of the exported FP16-scale representation. It does not establish preserved quality relative to the original BF16 model; the table above measures that separately.

| Phase | Elapsed | Peak allocated / reserved |
|---|---:|---:|
| Calibration: load | 23.80 s | 17.538 / 17.588 GiB |
| Calibration: all traces | 184.41 s | 17.643 / 17.684 GiB |
| Compression: load | 15.52 s | 17.538 / 17.588 GiB |
| Compression: conversion | 0.90 s | 18.682 / 18.707 GiB |
| Compression: training | 139.35 s | 23.655 / 24.322 GiB |
| Compression: export | 87.16 s | 19.390 / 19.912 GiB |
| Compression: packed_reference | 84.75 s | 19.272 / 20.162 GiB |
| Fresh reload: load | 16.90 s | 6.373 / 6.396 GiB |
| Fresh reload: serialization_check | 0.62 s | 6.593 / 6.922 GiB |
| Baseline: loading | 18.02 s | 17.538 / 17.588 GiB |
| Baseline: nll | 3.83 s | 17.954 / 18.191 GiB |
| Baseline: benchmark | 5.84 s | 17.625 / 17.656 GiB |
| Baseline: generation | 48.50 s | 17.626 / 17.658 GiB |
| Packed: loading | 17.02 s | 6.373 / 6.396 GiB |
| Packed: nll | 8.55 s | 6.789 / 7.172 GiB |
| Packed: benchmark | 19.48 s | 6.602 / 6.957 GiB |
| Packed: generation | 253.46 s | 6.604 / 6.959 GiB |

Peaks are absolute PyTorch allocated/reserved bytes, including persistent model state. The CUDA allocator cache is cleared and peak counters reset at each phase boundary. These counters exclude driver/display allocations and are not the same as whole-device `nvidia-smi` memory. Timing synchronizes CUDA at phase boundaries.

| Measured throughput | Original BF16 | Reloaded packed |
|---|---:|---:|
| Scored NLL tokens/s | 4280.72 | 1917.24 |
| ARC questions/s | 17.113 | 5.134 |
| Generated tokens/s | 16.844 | 5.050 |
| Actual generated tokens | 817 | 1280 |

## Environment, fixes, and verification

Repository commit: `891d260af96a7ce61ccbcb23474015d975bb18dc`. Initial local working tree: clean. Hardware: NVIDIA GeForce RTX 5090, compute capability [12, 0], driver 596.21, PyTorch 2.14.1+cu130 with CUDA 13.0. Initial free/total GPU memory: 30.26/31.84 GiB. System RAM free/total: 135.07/255.77 GiB. Initial D: free disk: 554.39 GiB. Python 3.12.11, Transformers 5.20.2, Accelerate 1.15.0, Safetensors 0.8.0; exact package versions are in `requirements-freeze.txt`.

Recorded full suite: **107 tests passed** (`all-tests.log`). CPU hybrid smoke roundtrip maximum logit difference: 0.0; tiny native multimodal/hybrid CUDA BF16 training/export/reload maximum difference: 0.0, finite outputs=True. CUDA BF16 matrix backward: passed.

Additional random-fixture preflight: 2 native hybrid layers, hidden size 4096, intermediate size 12288, reduced vocabulary 512; status=passed. Its training peak allocated/reserved memory was 5.952 / 6.609 GiB, and exported/reloaded logits had maximum difference 0.0. Pretrained weights loaded: False. This checks native dimensions and CUDA execution on random weights; it is not a real 9B quality result or a substitute for the full-model measurements above.

A separate random-fixture cached-generation preflight failed its initial strict BF16 full-sequence versus incremental comparison: maximum absolute logit difference 0.0654296875, 8 of 12288 logits outside the initial tolerance (atol=0.05, rtol=0.02). The follow-up promoted the same BF16-rounded weight values to FP32 and measured maximum difference 1.90138817e-05; within its FP32 tolerance: True. Cached BF16 generation produced 16 tokens with finite logits=True, reaching its cap=True. This was an unquantized, random two-layer fixture with no pretrained source weights; it prompted no core-model changes and does not establish BF16 cache parity. Both real-model evaluations explicitly disable caching. See `native-cached-generation-preflight.json` and `native-cached-generation-diagnostic.json`.

Code fixes used for the pilot:

- Projection replacement retains only module paths and transfers frozen source-weight storage into quantizers, avoiding duplicate full projection weights.
- Group-statistic initialization and inference-only hard reconstruction use bounded chunks; unnecessary hard-stage snapshots are released after training.
- Packed reload constructs native parameters on the meta device instead of first allocating a full FP32 source model; export avoids redundant CPU copies while preserving alias safety.
- Chat-template calls explicitly request list token IDs, matching Transformers 5's changed default return type.
- Calibration explicitly stops on the union of model/tokenizer EOS IDs. For this checkpoint, those are `248044` (`<|endoftext|>`) and `248046` (`<|im_end|>`); using the model configuration alone could continue past the assistant's closing token. The chosen IDs are recorded per trace and in the preparation manifest.
- Added frozen dataset preparation, token-weighted evaluation, byte accounting, CUDA instrumentation, phase timing, and regression coverage. Run-local download diagnostics preserve the official revision and verified source hashes.

Reproduction commands are in `reproduce.ps1`; script sources and raw JSON/log outputs remain beside the experiment. No cloud GPU was used, and no model artifact or code was pushed/uploaded.

## Next bounded experiment (recommendation only)

Use 64 representative, distinct calibration prompts, sequence length 512, two epochs, batch 1, window 1, stride 1, group 128, BF16, gradient checkpointing, and dedicated scratch. Keep the same held-out suite and conservative projection policy. Given the severe measured degradation, first inspect layerwise drift and sensitivity to hard quantization/FP16 export. The 64-sample, two-epoch recipe is only a bounded diagnostic that increases coverage; it is not a deployment recommendation or a claim that this recipe restores baseline quality.

The measured pilot spent 94.56 s building caches and checking reconstruction and 44.39 s optimizing. Extrapolating the former by 4.00× and the latter by 8.00×, with loading/conversion/export fixed at 103.58 s, gives approximately **837.36 s (0.233 h) for compression**, plus 761.45 s (0.212 h) for trace preparation. Combined estimate: 1,598.81 s (0.444 h), excluding download, fresh reload, serialization comparison, and unchanged held-out evaluation. This is an extrapolation, not a measured larger run: trace lengths, prompt distribution, disk I/O, thermal behavior, and the soft/hard work mix can change runtime. Do not extrapolate memory linearly from sample count when using disk caches; window geometry and sequence length remain the main activation-memory controls. No larger run was launched.

The pilot's 16 calibration traces and 100 benchmark questions are too small for broad quality conclusions. Calibration is short-context and text-only; long-context, multimodal, task-finetuned, and deployment-kernel behavior remain untested. The intended next workflow is BF16 task fine-tuning followed by fresh calibration/compression and comparison against that exact fine-tuned baseline.
