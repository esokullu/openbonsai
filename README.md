# OpenBonsai

[OpenBonsai](https://github.com/esokullu/openbonsai) is an independent compression research project for Qwen3.5 and its BF16 fine-tunes. Our current priority is **Hadamard rotations + ScaleQ-inspired ternary calibration + efficient ternary packing**. RCO mixed-precision allocation is an optional tool for identifying limited higher-precision exceptions. **This combination is a research hypothesis, not a reproduced Bonsai training pipeline or a proven quality-recovery method.**

The original [ScaleQ-1.58 paper is included as a PDF](papers/2608.01078.pdf), with [attribution and provenance](papers/README.md).

The target is `Qwen/Qwen3.5-9B`. Real-model pilots on an RTX 5090 have shown severe quality loss; a useful compressed 9B model has **not** been demonstrated. Rotations improved the first matched experiment, but both compressed models failed useful generation. A subsequent rotated 16-record × 8-pass versus 128-record × 1-pass comparison also failed: hard NLL was 8.499 and 9.122 (BF16: 2.424), with no useful answers from either arm. These repeatedly used development diagnostics are not a final quality evaluation. [Completed comparisons and runtime integration](docs/5090-runtime.md). Optional RCO has only software validation so far. Arbitrary dense architectures and MoE are unsupported.

## Current research direction

Prism's released inference implementation applies matching blockwise sign/Hadamard transforms to linear inputs and weights, and inverse transforms after embedding lookup. Its broad ternary coverage includes the embedding and output head. Our earlier artifact retained roughly **4.07 GB** in those two BF16 tensors alone. Keeping them uncompressed cannot approach the roughly 2 GB Bonsai language-model target. Vision must be reported separately. See the [pinned rotation audit](docs/rotations-audit.md) and [staged experiment plan](docs/research-direction.md).

The new prototype keeps transforms local to each operation, preserving native recurrent attention, gates and nonlinearities. It retains an exact original-weight teacher bypass, validates floating rotation equivalence separately, and exports explicit sign vectors with the packed weights. Deterministic signs and ScaleQ-style reconstruction are our choices; Prism's compression training recipe is not public in the audited sources. Rotations add metadata and computation and do not themselves compress weights.

Run the bounded **random-fixture** comparison offline:

```bash
python -m openbonsai.rotation_smoke --output runs/rotation-paired
```

Both conditions use the same original BF16 fixture, 25 decoder projections, calibration samples and training settings. The command reports hard versus exported NLL, exact packed reload, complete artifact bytes, training time and reference-forward time. Random-model NLL is a software diagnostic, not evidence of useful language quality. Embedding/head rotation, gradients and export are separately tested; the decoder-window trainer explicitly rejects endpoint quantizers because it does not train them.

A separate [output-head calibration prototype](docs/endpoint-calibration.md) uses cached original BF16 hidden states, bounded row quantization, a full-vocabulary teacher KL loss, and streaming ternary export. Its offline integration command is `python -m openbonsai.head_smoke --output runs/head-paired`. [Initial validation](validation-endpoint-head.json) records 571 passing CPU tests and exact fresh-process fixture reloads. Real 9B endpoint quality and CUDA memory remain unvalidated. It does not change the preserved decoder comparisons.

The [embedding prototype](docs/embedding-calibration.md) adds selected-row lookup and complete-vocabulary table calibration. Each optimizer step fits every original row; streaming export requires verified soft/hard coverage bound to the current source, parameters and rotation. Tied embedding/head replacement is rejected explicitly. Run `python -m openbonsai.embedding_smoke --output runs/embedding-paired` for a native random-fixture fit/export/fresh-process check. This is a separate endpoint baseline; uniform table reconstruction is not evidence of preserved language quality, and it was not integrated into the completed 5090 decoder comparisons.

The [staged compression command](docs/staged-pipeline.md), `openbonsai pipeline`, connects original-model trace preparation, rotated decoder fitting, bounded head/embedding calibration, and complete artifact assembly. Each stage runs separately with explicit settings and a deadline, and later stages require successful, hash-verified results. It binds a local BF16 checkpoint—including a merged fine-tune—to its tokenizer and calibration data. Full coverage requires both endpoint candidates; it cannot silently retain those large tensors in BF16. [CPU validation](validation-staged-pipeline.json) includes nine supervised stages, all 27 target matrices on a tiny native model, and exact fresh-process reload. Actual 9B endpoint quality and memory requirements remain unvalidated.

The [matched evaluation path](docs/matched-evaluation.md) freezes bytes, token IDs, score positions and generation rules for Transformers and a pinned Prism C-API worker. A real CPU comparison on a random hybrid Qwen fixture checks tokenization, full-vocabulary probabilities, recurrent-state resets and greedy decisions. A separate [official Bonsai CPU preflight](docs/bonsai-reference-preflight.md) verifies the 1,997,486,976-byte checkpoint, its tokenizer, state resets and short generation; its embedding and head occupy 444,989,440 bytes combined. These are evaluator checks, not a matched quality comparison. Windows/CUDA validation of the Prism reference remains outstanding; this is separate from OpenBonsai’s completed 5090 decoder experiments.

The official checkpoint has now [completed the frozen development suite](docs/official-bonsai-development-result.md) on CPU: **NLL 3.19215, perplexity 24.34065, ARC-Easy 16/20**, plus four fixed-cap generations. Its reasoning answer ended before the final answer, and its coding response contained a working function with unfinished formatting. This is a measured reference, not an OpenBonsai result. The [exact development inputs](research/evaluation/bonsai-development/README.md) are retained for the next comparison. Original BF16 and OpenBonsai still need rescoring with the same FP64 probability reduction; their historical NLL values above are not numerically matched to this row.

[Validation results](validation-rotations.json) record **444 passing CPU tests** and the paired fixture run. The rotated fixture was slightly larger and had higher random-model NLL; this is not evidence of a pretrained-model benefit. Its timings are single reference forwards, not a throughput benchmark.

The new `openbonsai.mixed_export` format supports rotated ternary, ordinary ternary, 4/6-bit affine and BF16 tensors. Its budget includes candidate headers, code streams, scales/zero points, row indices, rotation signs, retained tensors, config, generation settings and supplied tokenizer assets. Export verifies the predicted total against actual files. Automatic ternary packing compares **whole candidate file** costs for two-bit, BITCOS and base-3 storage. This format is separate from the legacy ternary format described below; use `load_mixed` for new artifacts. It reconstructs floating-point weights for execution and makes no inference acceleration claim.

The [native text-only extraction helper](docs/text-only-export.md) maps the original conditional model's language state into Qwen's native causal model, sharing frozen storage without duplicating the weights. It reuses mixed export and reports zero vision bytes. Small hybrid-model parity and fresh-process reload tests pass; actual 9B text-only conversion, a separate usable vision package, and broad ternary quality remain unverified.

The [lossless base-3 codec](docs/base3-packing.md) stores five ternary codes per byte and preserves existing learned values. It reduced a complete native random-model fixture from 261,555 bytes with two-bit storage to 225,547 bytes, with identical fresh-process logits. [Validation](validation-base3.json) records 1,381 passing CPU tests. This is a storage result, not quality recovery. A separate [official checkpoint statistics audit](docs/bonsai-ternary-statistics.md) explains why BITCOS is not always the smallest choice at the measured zero densities.

The 5090 experiment's bounded decoder training, loading and EOS fixes are now integrated into this research branch. [Validation](validation-5090-runtime.json) records 699 passing CPU tests and paired native-fixture checks. Full BF16 decoded-weight caching is explicitly optional, defaults to off, and adds runtime memory excluded from artifact bytes; the completed 9B comparison used 13.824 GB of this cache. Keep existing experiment snapshots unchanged when adopting this branch.

The read-only [activation diagnostic](docs/activation-diagnostic.md) restored the exact final 128-record 9B state and measured two fixed windows on 16 disjoint search records. At window 16, local conditional error was 0.064267 and total original-target error was 0.760929, both divided by original-target RMS. All 16 records met the predefined target-gap selector; original/bypass/hard probes and preservation checks passed. The resulting [single-window teacher-target control](docs/teacher-target-control.md#completed-pretrained-single-window-control) reduced common-target MSE and development NLL, but retained all-A ARC predictions and no useful generations. This supports further controlled investigation, not a claim of quality recovery. The offline fixture and the sealed remote 9B runs have separate validation scopes.

The preserved [RCO implementation and source audit](docs/rco-audit.md) use actual tangent projection, budget retraction, momentum transport and Gumbel straight-through optimization of assembled-model loss. Costs are tensor-specific serialized bytes. The final discrete decoder always checks integer-byte feasibility; large problems can use a conservatively rounded grid and explicitly report approximate optimization. GPTQ supplies optional 4/6-bit linear candidates from the original BF16 weights; embedding higher-precision candidates currently use a clearly labeled grouped RTN baseline. **Rotated candidates are currently rejected by the RCO assembled-search adapter**, pending a transform-aware integration and evidence that allocation is needed. No large allocation search is the next step.

## Motivation

Prism ML's Ternary Bonsai 2 family already includes Qwen3.5-based **2B, 4B, and 9B** models, with publicly available Apache 2.0 weights. [Prism's release announcement](https://prismml.com/news/bonsai-2-family). We want developers to be able to apply ternary compression to their own fine-tuned models. The official [9B release](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit) and [demo repository](https://github.com/PrismML-Eng/Bonsai-demo) reviewed on October 26, 2026 provide weights and inference tooling, but do not include a reproducible Bonsai 2 training/compression pipeline for that workflow.

**Our goal is to let everyone turn their own fine-tuned BF16 models into compact ternary models**, through an open, reproducible compression pipeline. Fine-tuning happens first, in BF16, using the developer's preferred training tools. OpenBonsai then learns ternary factors and a low-rank refinement around that customized checkpoint's frozen weights. The refinement is included before ternarization and is absorbed into the exported ternary weights. This workflow does not require fine-tuning the packed ternary model directly.

## Intended workflow

1. Fine-tune the BF16 model for your task. If using a mergeable adapter such as LoRA, merge it into a complete BF16 checkpoint before compression.
2. Generate AYOT calibration traces with **that fine-tuned checkpoint**, using prompts representative of the intended tasks. Preserve its tokenizer and chat template.
3. Run OpenBonsai using the same checkpoint and revision as both the weight source and the reconstruction reference. In this workflow, "original model" means the unquantized, fine-tuned checkpoint.
4. Compare the compressed model with the fine-tuned BF16 baseline on held-out task and general-quality evaluations before deployment.

For the rotation research path, use the [staged local workflow](docs/staged-pipeline.md), which authenticates the checkpoint's contents instead of relying on its name or path. The older `prepare`/`train` commands below remain available for the legacy unrotated decoder path: pass the same model and revision with `--dtype bfloat16` to both and preserve the files between commands. An adapter-only directory is not a complete model. Each new fine-tuned checkpoint needs its own traces and compression run. Compression of a real customized 9B checkpoint remains unvalidated.

The compact model retains its parameter count; the storage reduction comes from ternary weight representation. Quality preservation must be measured. BF16 task fine-tuning is external to this package, and packed checkpoints do not contain the original weights or optimizer state needed to resume compression training.

## What works

- Original-model generation preserves the exact question, reasoning, and answer token IDs; model ID and resolved revision are recorded.
- Frozen original weights provide the teacher; three learned factors per group and a rank-64 low-rank refinement adapt the student. `--lora-rank 0` disables the refinement for an ablation.
- Differentiable soft ternarization, followed by exact hard ternary forward values; no mean is added back to the reconstructed weights.
- Selectable CAT-Q sigmoid factor mappings and masked Huber reconstruction, alongside the original exponential mapping and MSE defaults.
- The hard-stage surrogate uses the parameters and sharpness from the actual last soft forward, with the current loss gradient. This is an explicit interpretation of the paper's backward rule.
- By default, sliding windows reconstruct original-weight window outputs from the **same input** as the quantized window, produced by the previously quantized prefix. An opt-in [original-prefix teacher control](docs/teacher-target-control.md) tests accumulated-error correction; it has not established a pretrained quality improvement. Padding is excluded from the loss.
- Native Qwen3.5 blocks retain their individual masks, rotary inputs, gates, and recurrent behavior. Gradient checkpointing is supported in the tested CPU reference path.
- Packed checkpoint export with lossless base-3, BITCOS or legacy two-bit storage, retained floating tensors, checksums, exact byte accounting, and offline reload.
- CPU-only repacking of existing artifacts, with unchanged ternary codes, scales, biases, and retained weights; no new calibration or training.

The default projection policy compresses the text MLP `gate/up/down`, full-attention `q/k/v/o`, and linear-attention `in_proj_qkv/in_proj_z/out_proj` projections. Embeddings, vocabulary head, vision tower, convolutions, norms, recurrent state parameters, and small `in_proj_a/in_proj_b` gates remain floating point. `--aggressive-gates` additionally compresses those two gates; its quality has not been evaluated. Preserving vision tensors does not establish preserved multimodal quality: the text stack changes, and calibration is text-only.

## Install and run the offline test

From this directory, with Python 3.12 and an appropriate PyTorch build for your hardware:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m openbonsai smoke --output runs/tiny-packed
```

Installing the package also provides the `openbonsai` CLI. Run `openbonsai --help` to see its commands; every `python -m openbonsai ...` invocation in this README can also be written as `openbonsai ...`.

The smoke command needs no network, model weights, or datasets. It trains a tiny hybrid model, exports it, and checks exact logits after reloading the same FP16-scale packed representation. A separate number measures rounding introduced when FP32 learned scales are exported to FP16. `runs/tiny-packed-report.json` records losses, bytes, and those checks. Existing nonempty output directories are refused.

The original `validation.json` records the initial **80 passing tests** and an end-to-end offline smoke run. The BITCOS update is recorded separately in [validation-bitcos.json](validation-bitcos.json): **221 passing tests**, an offline training/export smoke run, and exact full-model logits after repacking a pre-update artifact. On that tiny fixture, packed tensors shrink by **14.54%** and the complete model artifact by **3.54%**; these are software checks, not a size or quality benchmark for a real 9B model. The tested stack is Python 3.12.9, PyTorch 2.14.1, Transformers 5.20.2, Accelerate 1.15.0, Safetensors 0.8.0, and Pytest 9.1.1. Transformers is constrained to the tested minor version because this adapter calls native decoder blocks. The main suite runs on CPU; additional initial-prototype checks exercised MPS training and a 129-token multimodal-wrapper training run. Those initial checks do not validate CUDA, full-model mixed precision or optimized DeltaNet kernels. Later bounded 9B decoder CUDA measurements are recorded separately in the [5090 review](docs/5090-runtime.md).

The low-rank refinement and hard-gradient update is recorded in [validation-refinement.json](validation-refinement.json): **248 passing tests**, corrective optimization on a controlled tiny native model, and an offline rank-64 training/export smoke run with exact packed reload logits. These checks do not establish recovered pretrained-model quality or CUDA memory requirements.

The sigmoid/Huber options and tokenizer compatibility fix are recorded in [validation-catq-options.json](validation-catq-options.json): **291 passing tests**, independent formula/gradient checks, a real offline tokenizer regression, and exact packed reload logits for a sigmoid-plus-Huber smoke run. The default exponential mapping and MSE objective are preserved; the later 9B sigmoid/Huber experiments documented above did not recover useful quality.

## Run a small real-model pilot

These commands **load/download the original 9B weights** and require substantial memory. They are examples; they were not run as part of this delivery. Start with a small pilot before the paper-sized recipe.

Create `prompts.jsonl`, one object per line, containing `{"prompt":"..."}` or `{"question":"..."}`. The two prompts in `examples/prompts.jsonl` only demonstrate the format; they are not a sufficient calibration corpus.

```bash
python -m openbonsai prepare \
  --model Qwen/Qwen3.5-9B --revision main \
  --prompts prompts.jsonl --output runs/ayot.jsonl \
  --device cuda --dtype bfloat16 --enable-thinking \
  --max-samples 16 --max-new-tokens 4096

# Use the exact original-model revision that produced the calibration traces.
REVISION=$(python -c 'import json; print(json.load(open("runs/ayot.jsonl.manifest.json"))["resolved_revision"])')

python -m openbonsai train \
  --model Qwen/Qwen3.5-9B --revision "$REVISION" \
  --calibration runs/ayot.jsonl --output runs/qwen35-pilot \
  --device cuda --dtype bfloat16 \
  --max-samples 16 --sequence-length 512 --epochs 2 \
  --batch-size 1 --window-size 2 --stride 1 \
  --lora-rank 64 --lora-learning-rate 0.001 \
  --cache-dir runs/window-cache --gradient-checkpointing
```

This deliberately reduced pilot is an execution check, not a quality recipe. Both commands accept `--local-files-only` to require cached/local assets. Quantized source checkpoints are inappropriate: calibration must come from the original model. The loader rejects declared quantization metadata.

For a controlled comparison with the released CAT-Q configuration, add `--factor-mapping sigmoid --reconstruction-loss huber --huber-delta 1.0` to `train`. These are independent options; defaults remain `exp` and `mse`. Sigmoid mode also uses CAT-Q's `+1e-6` scale epsilon. Huber delta `1.0` is an explicit choice because the authors' training threshold is unavailable. Hard reconstruction diagnostics remain MSE in either mode. The flags also work with the offline `smoke` command. See [the equations and reduction details](docs/method.md#factor-mappings-and-reconstruction-loss).

The default rank-64 refinement adds `B @ A / 64` to each selected projection **before** ternarization; export contains no separate inference adapter. `--learning-rate` controls the group factors, while `--lora-learning-rate` controls the low-rank parameters; both default to `0.001`. Use `--lora-rank 0` to compare against factor-only calibration. A run needs at least one soft update before the hard stage; a one-update window is rejected.

Existing packed artifacts cannot adopt the refinement or the corrected hard-stage derivative through `repack`. Start a new compression run from the original BF16 checkpoint, using a new output directory. Valid calibration traces from that same checkpoint can be reused. These changes do not establish restored 9B model quality; compare with the BF16 baseline and preserve the prior artifact for a controlled comparison. [Method details and source boundaries](docs/method.md).

For the paper's default calibration budget, sample 1,024 mathematics prompts from MetaMathQA and 1,024 coding prompts from OpenCodeInstruct, generate traces with the original target model, then use 2,048 samples, sequence length 2,048, 60 epochs, and batch size 3. Dataset acquisition/sampling is left to the caller; it is not implemented or silently downloaded. Review dataset licenses and keep evaluation examples separate. The paper evaluates other model families; applying this recipe to Qwen3.5 is an experiment.

Generation is greedy by default (a documented implementation choice). Generation caps can cut off reasoning or answers: inspect `records_hitting_generation_limit` in the preparation manifest. Training right-truncates/pads the saved sequences and separately reports active, truncated, and padding tokens. **2,048 × 2,048 is a token-slot budget; it is not necessarily 4,194,304 active tokens.** Missing model/revision provenance is counted for manually supplied JSONL; present metadata is checked against the target. Avoid files without provenance for scientific comparisons.

## Memory and runtime

This trainer keeps the complete original model on one device, together with group factors, low-rank parameters, derivative snapshots, and training intermediates. It has no model sharding, optimizer sharding, CPU layer offload, or training resume. BF16 original 9B weights alone are roughly 18 GB before these costs. Decoder FP32 statistics and corrections use bounded chunks with checkpoint recomputation; a reconstructed source-dtype weight matrix is still materialized for each projection. Rotated chunks cover complete rows and groups. The sealed 16-sample, eight-epoch rank-64 experiment measured peak allocated VRAM of 20.207 GiB unrotated and 20.221 GiB rotated, using its recorded source snapshot and configuration. Those figures are not guarantees for larger batches, contexts, endpoint calibration or this reconciled branch. Run a fresh CUDA memory preflight before scaling. Packed file size is not a VRAM estimate.

Without `--cache-dir`, all window inputs/targets live in CPU RAM. At 2,048 samples × 2,048 tokens × hidden size 4,096, two BF16 hidden-state streams alone occupy **64 GiB per window**, plus masks and rotary inputs. Disk caching moves these records to scratch storage; it still needs more than 64 GiB of free space at that budget and repeatedly reads them over 60 epochs. FP32 doubles the hidden-state storage. Completed window caches are removed. The Python reference attention/DeltaNet path and full-precision reconstruction can be slow; runtime depends on the measured configuration.

## Storage and inference

New exports default to `--packing auto`. In the legacy `save_compressed`/CLI path, this selects the smallest **code payload** among two-bit, BITCOS and base-3 for each projection; ties prefer two-bit, then BITCOS, then base-3. Force a codec with `--packing 2bit`, `--packing bitcos` or `--packing base3`. The updated loader accepts legacy v1/v2 artifacts and v3 artifacts containing base-3 modules. Older loaders cannot read base-3. The newer mixed-candidate/endpoint exporters instead compare complete serialized candidate files, including metadata and BITCOS row indexes.

BITCOS stores one presence bit per weight and one sign bit per nonzero weight, following [Breaking the 1.58-bit Barrier for Ternary LLMs](https://arxiv.org/abs/2609.16338). With zero fraction `z`, the code streams cost approximately `2 - z` bits per weight, plus the same FP16 group scales. At 50% zeros and group size 128, the projection payload is approximately **1.625 bits per weight instead of 2.125**, about **23.5% smaller**. Actual savings depend on sparsity, group/byte padding, retained floating tensors, and file headers. `auto` compares code tensor bytes, so it does not guarantee a smaller complete artifact for tiny tensors with extra metadata.

Base-3 uses approximately **1.6 code bits per weight**, plus the same scales and rotation metadata. Its flattened stream crosses row and scale-group boundaries without adding byte padding at each boundary. Ignoring headers and row indexes, BITCOS becomes smaller at more than roughly 40% zeros. Neither codec changes sparsity or rounds stored scales. Base-3 is an independent OpenBonsai encoding, not Prism's PTQ1_0/GGUF format.

The implementation uses row-major, least-significant-bit-first presence and compacted sign streams (`1` means negative), with zero final-byte padding. It implements the paper's lossless storage principle, not its hardware-specific kernel layout. **Repacking an existing artifact changes no weight values and introduces no additional quantization or scale rounding.** It does not undo any quality loss from the earlier BF16-to-ternary conversion.

Repack an existing pilot on CPU, without loading the original model or retraining:

```bash
python -m openbonsai repack \
  --input runs/qwen35-pilot \
  --output runs/qwen35-pilot-repacked \
  --packing auto
```

The command verifies input checksums, processes one projection at a time, preserves retained tensors and tokenizer assets, and reports exact before/after model-artifact bytes. Use a new output directory; the source is preserved. `--packing 2bit` can convert back to the legacy format. Python callers can use `repack_compressed(source, destination, packing="auto")` from `openbonsai.export`, or choose the codec with `save_compressed(..., packing="auto")`.

That command handles legacy ternary artifacts. For the newer mixed format, `repack_ternary_candidate(candidate, packing="auto")` in `openbonsai.mixed_formats` re-encodes one already-loaded ternary candidate; it is not a complete mixed-artifact repack command. It preserves codes, stored scales/biases, rotation signs and provenance. It refuses BF16 candidates rather than quantizing them implicitly.

`manifest.json` accounts for packed tensors, retained tensors, safetensors headers, config, and the manifest itself. `train` additionally saves tokenizer assets in `tokenizer/`; those supplementary files are excluded from the reported model-artifact byte total. The artifact is neither GGUF nor an ordinary `from_pretrained` checkpoint. Load it through this package:

```python
import torch
from transformers import AutoTokenizer
from openbonsai.export import load_compressed

model = load_compressed("runs/qwen35-pilot", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("runs/qwen35-pilot/tokenizer")
inputs = tokenizer("A short test", return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**inputs, use_cache=False).logits
```

Reference inference dequantizes each projection to floating point for computation, including BITCOS projections. It provides compact persistent weight storage, but **no optimized ternary kernel, speed guarantee, or measured peak-memory reduction**. For maximum compatibility the loader disables caches; efficient incremental generation is future work. Packed artifacts omit original weights and optimizer state, so they cannot resume quantization training. The initial export rounds learned FP32 scales/biases to FP16; subsequent repacking preserves those stored FP16 values exactly.

## Explicit reconstruction choices

Some details are not sufficiently specified to claim an exact reproduction. These choices are recorded in code and training metadata:

| Detail | This implementation |
|---|---|
| Initialization | Low-rank `A` uses Kaiming uniform and `B` starts at zero; group statistics exclude partial-group padding |
| Low-rank refinement | Default rank 64, fixed scaling `1/r`; group statistics recomputed from `W + B @ A / r`; rank 0 retains factor-only statistics |
| Factor mappings | Default `exp`: `tanh` mean and exponential scale/threshold; optional `sigmoid`: released CAT-Q bounded mappings and scale epsilon |
| Schedule | First 80% soft, sharpness rising to 30; final 20% hard |
| Hard-stage backward | Full soft-map Jacobian at the actual last soft forward's pre-update factors, low-rank parameters, and sharpness; applied to the current upstream loss gradient |
| Progress zero | Exact original-weight bypass; training starts at positive progress |
| Reconstruction | MSE by default or configurable Huber over nonpadding elements; shared quantized-prefix input; hard diagnostics always MSE |
| Optimizer | AdamW; separate factor and low-rank learning rates, both 0.001 by default; linear decay, weight decay 0, gradient clipping 1 |
| Windows | Default width 4, stride 2, shrinking tail; overlapping factors and low-rank parameters warm-start and restart annealing |
| Architecture | Native dense Qwen3.5; conservative projection policy described above |
| Export | Ternarize the refined weights; no low-rank tensors or adapter computation remain at inference |

The hard-stage surrogate is explicitly an approximation to an underspecified rule. It freezes a local Jacobian, not the previous batch's loss gradients. A `soft_surrogate` alternative is available in the `TernaryLinear` Python API; it is not the CLI default. The public CAT-Q release confirms pre-quantization low-rank refinement, but provides inference/export code rather than its training implementation. Our initialization, backward interpretation, and Qwen3.5 adaptation remain independent choices. See [the method notes](docs/method.md) for pinned source links and remaining differences. The frozen source weights remain recoverable only during training. No generic ternarization can guarantee bitwise lossless recovery or unchanged behavior for every input.

Before using a compressed 9B model, evaluate the original and compressed checkpoints on the same held-out general, reasoning, coding, and intended-domain tasks with identical prompting/decoding. Check long-context and multimodal behavior separately if needed. The experimental matched evaluator supplies scoring and input contracts; a complete held-out comparison has not yet been run. Neither toy loss improvement nor serialization parity establishes quality preservation, superiority to GSQ-RCO/SLQ, or equivalence to Bonsai.

## Sources

- [ScaleQ-1.58, arXiv:2608.01078](https://arxiv.org/abs/2608.01078): AYOT and the CAT-Q composition; §2 and Appendix C.
- [CAT-Q, arXiv:2606.26650](https://arxiv.org/abs/2606.26650): grouped modulation, soft ternarization, sliding-window reconstruction.
- [BITCOS, arXiv:2609.16338](https://arxiv.org/abs/2609.16338): presence bitmap and compacted signs for lossless packing of already-ternary weights; §II-A. Our portable implementation does not include the paper's fused CPU/Intel GPU kernels.
- [Official BitTern repository](https://github.com/IntelChina-AI/BitTern/tree/5a8fcd4f7e0366554b732d300a37e6ea467c3c35/projects/cat-q), refreshed October 26, 2026 at commit `5a8fcd4f7e0366554b732d300a37e6ea467c3c35`: released low-rank refinement, inference/export behavior, and model-specific configurations; its training code was unavailable in that snapshot.
- [SliderQuant](https://github.com/deep-optimization/SliderQuant), inspected commit `bf2641ece550d5f6edb0028c2ae8b0ee658bc9bf`: sliding-window implementation context. This prototype does not vendor its code or reproduce its full training system.
- [Qwen3.5-9B configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json) and [native Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5): architecture adaptation.

OpenBonsai is an independent implementation, unaffiliated with the paper authors or the Bonsai model developers. Model, tokenizer, datasets, and dependencies retain their respective licenses.
