# Controlled rotation research, with optional precision exceptions

The goal is an open BF16-to-compressed-model pipeline for Qwen3.5-9B and its BF16
fine-tunes, with useful quality and broad ternary coverage approaching Bonsai's
language-model size. The current hypothesis is **Hadamard rotations +
ScaleQ-inspired calibration + efficient ternary packing**. RCO is preserved as
an optional way to choose limited higher-precision exceptions. Neither is a
proven recovery strategy. Do not replace the size goal with a large mixed model.

## Source facts and implementation choices

See [Prism's pinned placement audit](rotations-audit.md). Module-local row-vector
transforms use `R = signs * normalized_Hadamard`: `W_rot = W R`, `x_rot = x R`.
The linear output stays in original coordinates. Embeddings store `E R` and
inverse-transform gathered rows. The prototype never moves transforms through
RMSNorm, sigmoid gates, recurrent states, nonlinear activations or RoPE.

Prism documents block size 4096 and 202 ternary matrices, including embedding
and head. Our fixed SHA256-derived signs, calibration loss, initialization,
low-rank refinement and optimization schedule are independent choices. In
particular, the released inference implementation is not a compression recipe.

## Stages and decision boundaries

The first paired 9B decoder experiment has completed: rotations improved the
development metrics, but both compressed conditions failed useful generation.
The preserved results and imported runtime changes are recorded in
[the 5090 review](5090-runtime.md). The next controlled comparison is a fresh
rotated 16×8 versus 128×1 pair from one pinned prompt source, with matched updates
and valid-token exposure. It tests data diversity before changing the schedule
or starting endpoint/RCO work on the GPU. The stages below retain their original
gates; none of the software tests establishes successful quality recovery.

1. Finish the already running 5090 baseline unchanged, including generation,
   FP16 export and independent reload. Freeze its source/config/data/artifacts.
   Review results before starting any next experiment or editing its checkout.
2. Validate rotation algebra, native hybrid-model computation, soft/hard
   gradients, original teacher bypass and transform-aware export on tiny
   fixtures. Exact bypass and floating transform equivalence are separate tests.
3. Port rotation into the remote memory-bounded quantizer in a new checkout.
   Rotate complete row chunks *before* applying the low-rank correction and
   group statistics. Preserve group offsets, masks, snapshots, FP32 optimizer
   state and existing EOS fixes. Compare dense/chunked values and gradients
   before real-model use. Measure soft and hard backward memory, not only forward.
4. Run a paired decoder experiment from the same original BF16 revision. Keep
   tensor coverage, groups, data, epochs, loss, rank, learning rates, evaluation
   inputs and decoding identical. The first controlled proposal is 16 samples,
   length 512, batch one, eight epochs, window/stride 1, rank 64, sigmoid factors,
   Huber delta 1, factor LR .005 and low-rank LR .001; compare rotated and
   unrotated. This is a bounded diagnostic, not a converged ScaleQ recipe. Use
   the same original checkpoints for both, never requantize compressed weights.
5. Evaluate hard FP32, exported scales, and fresh reload separately. Use held-out
   NLL/perplexity, ARC and actual reasoning/code/instruction generations. Report
   diagnostic subsets separately from a full evaluation. Measure full artifact
   bytes, zero densities, transform/index overhead, retained bytes, peak memory,
   calibration time and inference overhead. Exact reload is a prerequisite, not
   a quality result. Compare to the existing baseline without claiming the
   changed eight-epoch recipe isolates rotation against its one-epoch run.
6. Before claiming near-Bonsai size, develop separate embedding/head calibration
   and an equal-coverage comparison. The decoder trainer does not optimize them
   and now rejects such a request explicitly. Small native endpoint gradient and
   export checks are not evidence of pretrained vocabulary coverage or quality.
   Retain vision as in the baseline and report its bytes separately; text-only
   evaluation does not establish preserved multimodal behavior.
7. Only then decide if limited higher-precision exceptions merit RCO. Predeclare
   one feasible complete artifact budget and compare to a fixed mixed baseline
   at approximately equal bytes. Do not launch a broad candidate grid or long
   training/search sweep before a bounded quality-versus-size gain is demonstrated.

Use disjoint calibration, experiment-selection/search-validation and evaluation
inputs, recording IDs and hashes. Established WikiText/ARC/generation evaluation
answers must not become fitting or allocation inputs. Current repeatedly used
diagnostics are development data; reserve a fresh confirmation set for a final
quality claim. After changing an allocation, inspect assembled-model validation
and activation drift before deciding on a final calibration pass. Recalibration
changes candidate values and potentially BITCOS sizes: export/recount/recheck
the same budget afterward.

## Reference APIs and storage

`replace_rotated_projections` installs decoder quantizers. `endpoints=True`
also supports embedding/head **for explicit endpoint work**. `rotation_only`
is a float-equivalence diagnostic and is rejected by the training loop. The
original BF16 master stays frozen. The imported decoder path recomputes rotated
FP32 weights in complete row/group chunks and uses checkpoint recomputation to
bound saved FP32 intermediates. It still reconstructs a complete source-dtype
weight for the projection. The reconciled branch has CPU parity coverage;
larger configurations require a new CUDA preflight. This decoder path does not
train embeddings. A [separate embedding operator and full-table trainer](embedding-calibration.md)
now provide bounded row computation and certified soft/hard vocabulary coverage;
pretrained endpoint quality and CUDA feasibility remain unvalidated.

`RotatedTernaryLinear.mixed_candidate()` includes the rotation. Legacy
`packed_state()`/`save_compressed()` reject it, preventing a silent missing
transform. `MixedLinear` reconstructs row chunks and transforms inputs;
`MixedEmbedding` decodes unique requested rows and inverse-transforms them.
No original dense master is stored in a ternary deployment candidate.

`prepare_mixed_export(original, names, metadata=..., tokenizer_dir=...)` freezes
all common files before search. `plan.fixed_bytes + sum(candidate_nbytes(...))`
is the exact final cost. Fixed-width byte fields keep manifest overhead constant
across assignments. `save_mixed(..., budget_bytes=...)` checks original-weight
fingerprints and predicted/actual bytes; `load_mixed` validates all state,
checksums and generation settings without network access. Supply tokenizer
assets for a deployable artifact; an artifact without them is only a model
payload. Preserve the BF16 fine-tune's exact tokenizer and chat template.

Optional `fit_gptq_candidates` fits 4/6-bit linear alternatives using original
activation covariance and pinned upstream GPTQ error feedback; it preserves
FP32 scales and signed integer zero points. Embedding alternatives are explicitly
`embedding_group_rtn`, not activation-calibrated GPTQ. `build_candidate_table`
charges candidate-specific full bytes, including zero-dependent BITCOS cost.
`optimize_rco` takes an assembled-model loss callback and reports exact integer
feasibility plus whether its bounded discrete DP was approximate. The present
assembled RCO adapter **rejects rotations**; it cannot silently consume latent
rotated weights as ordinary linear weights. This integration is deferred until
the rotation results justify it. [RCO audit and limitations](rco-audit.md).

All runtimes here reconstruct floating-point weights. Smaller files do not
establish lower latency or peak VRAM. The native fixture's one-forward timing
is a diagnostic, not a throughput benchmark or a 9B projection.
