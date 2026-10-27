# Separate embedding calibration

Qwen3.5-9B's embedding is a 248,320 × 4,096 BF16 matrix, occupying
2,034,237,440 bytes before file metadata. The equally large output head is a
separate endpoint. Both must be addressed to approach Bonsai's language-model
size; decoder-only ternarization cannot meet that target.

This prototype implements bounded embedding lookup, complete-vocabulary table
fitting, and coverage-checked streaming export. It has CPU software validation,
not pretrained embedding-quality or actual-shape CUDA evidence. It does not
change or join the live 5090 decoder comparison. The first real endpoint
experiment should keep decoder, head, vision and small tensors at the original
precision and evaluate embedding compression alone.

## Selected rows and the original computation

`BoundedTernaryEmbedding` wraps an original materialized BF16 `nn.Embedding`.
It shares frozen source storage and uses the same bounded row math as
`BoundedTernaryHead`. It supports ordinary embeddings, including native
`padding_idx`; `max_norm`, sparse gradients and frequency-scaled gradients are
explicitly unsupported. Groups must fit complete rows.

For a selected row chunk, the operator gathers BF16 source rows **before**
promoting them to FP32. In the rotated condition it computes `E_selected R`,
then adds the low-rank correction and recomputes differentiable group
statistics. Group factors, A/B, gradients and the saved soft derivative point
stay FP32. The original BF16 source is never optimized.

Token lookup deduplicates IDs, quantizes bounded unique chunks, restores token
occurrences with `F.embedding`, and then inverse-transforms the result. With
row-vector `R = D H`, the inverse applies Hadamard then signs. Restoring
occurrences before the inverse preserves the dense oracle's BF16 duplicate
gradient accumulation. Native padding suppresses lookup gradients for the
padding row; it does not zero a nonzero stored padding row.

`quant_enabled=False` or progress zero gives an exact original lookup with no
rotation or quantization. A separate `rotation_only` mode checks the floating
transform. No transform crosses gates, recurrent state, normalization, RoPE or
other nonlinear operations. Source `.weight` exposes the original BF16 table,
so unsupported consumers that bypass `forward` must not be treated as compressed.
The tested native Qwen path uses the embedding forward method.

`reconstruct_rows(ids)` is an explicit table-fitting operation. It accepts a
bounded unique row selection and fits the padding row too. `teacher_rows(ids)`
returns original rows; `hard_codes_rows(start, end)` supplies current latent
hard codes/scales to the exporter. Neither API implicitly reconstructs the
whole table.

## Full vocabulary coverage is separate from token coverage

A small prompt pool touches only part of the vocabulary. Successful lookup
calibration on those tokens does not establish that the remaining rows were
fitted. The initial objective therefore uses **all original embedding rows**,
with uniform original-coordinate MSE, or explicitly selected Huber loss.
There are no text inputs, token labels or benchmark answers in this stage.
Its reconstruction error is a training metric, not held-out language quality.

One logical optimizer update is one complete vocabulary sweep:

1. Call `begin_step(progress)` once and zero gradients once.
2. Reconstruct disjoint row chunks, compute summed FP32 loss divided by the
   same global `vocabulary × width` denominator, and backpropagate each chunk.
3. Verify every row was visited exactly once, clip once, and apply one AdamW
   update. Learning rates are constant in this initial table-fitting baseline.

Row chunk size does not change the mathematical gradient normalization or
number of optimizer steps. Floating accumulation can still differ across
devices. This complete-sweep schedule is a deliberate baseline, not a claim
to reproduce CAT-Q or ScaleQ's unreleased endpoint training.

At least one genuine soft and one hard update are required. A soft step saves
the complete compact pre-update state once; row calls never advance the
schedule. Hard forward uses current parameters, while its surrogate derivative
uses the actual last-soft point. Full-table soft and hard visits are recorded.
Progress callbacks cannot alter the source or fitting state. Failed or aborted
fits leave no export certificate.

The report includes objective/configuration, verified source-weight hash,
caller-declared checkpoint provenance, initial/final state hashes, visit counts,
padding error, mapped factors, FP32-state checks and synchronized timing. A
revision or whole-checkpoint content hash is required; the API does not itself
authenticate the downloaded checkpoint. Keep the supplied model/tokenizer and
fine-tuned BF16 checkpoint identity pinned through the entire pipeline.

## Certified export and exact bytes

`require_complete_calibration` verifies all-row soft/hard coverage and binds it
to the source bytes, current factors/A/B/statistics, rotation, dimensions,
padding and configuration. It rehashes compact state in bounded chunks, so
out-of-band changes that bypass tensor version counters also invalidate the
certificate. This integrity guard is not a security boundary against a caller
deliberately forging Python state, and it is not a certificate of model quality.

`export_embedding_candidate` requires that certificate before and after its
two packing passes. It sets embedding kind, source provenance, coverage,
scales, row indexes and rotation metadata **before** comparing complete
serialized BITCOS, two-bit and base-3 costs. It never changes a linear candidate's
kind after costing. Export checks current hard codes/FP32 scales across passes,
source/sign invariance, and FP16 scale representability. No original table or
separate low-rank adapter enters the packed candidate.

`MixedEmbedding` reconstructs requested packed rows and applies the inverse
rotation. `load_mixed` restores native `padding_idx`. `prepare_mixed_export`
rejects selected weights sharing storage with another model tensor, including
an unselected retained head, as well as declared endpoint tying. Even disjoint
views of shared storage are unsupported until an explicit shared-allocation
policy exists. This avoids silently breaking a tied embedding/head contract.

Use `prepare_mixed_export` on the unchanged original model, fit the separate
embedding operator, and pass its candidate to `save_mixed`. The budget is
`plan.fixed_bytes + candidate_nbytes(candidate)`; the exporter verifies it
against actual files. Include the exact tokenizer/configuration for deployment.
Lossless packing adds no change beyond the learned ternary values and the
existing declared FP16-scale export rounding, which must be measured separately.

## Validation and limits

```bash
python -m openbonsai.embedding_smoke --output runs/embedding-paired
```

This offline command fits the complete table of a random native hybrid Qwen
fixture with matched rotated/unrotated initialization and optimizer settings.
It reports table reconstruction, held-aside random-fixture logits, exact byte
budgets, separate scale-rounding differences and exact fresh-process packed
reload. It verifies original model hashes/bypass and both EOS tokens. Forced
generation is an execution check, not an answer-quality test. The fixture has
no real tokenizer and is not a deployable pretrained language model.

[Recorded validation](../validation-embedding.json) contains the CPU suite and
smoke results. The tiny table metrics do not predict 9B quality or establish
that rotation is better at embedding precision.

Per-row FP32 computation is bounded; total memory is not one row chunk. The
source BF16 table, dense compact factors/A/B/gradients/Adam/snapshots and compact
export payload all consume memory. Token forward can retain all unique-row
graphs until backward. Complete-table fitting backpropagates each chunk before
the next, avoiding that full-table graph. CUDA duplicate accumulation, dense
compact-gradient overhead, export host memory, runtime and real 9B quality still
need separate preflights and measurements.

Uniform table fitting may not preserve language behavior. If it fails an
embedding-only quality test, an activation-sensitive refinement through frozen
original blocks is a later controlled hypothesis. Its student path must retain
autograd; freezing backbone parameters is different from disabling gradients.
Finally, an embedding that works with the original decoder must be reevaluated
after combining it with a ternary decoder and calibrated head. No joint success
or Bonsai equivalence is established by these isolated software checks.
