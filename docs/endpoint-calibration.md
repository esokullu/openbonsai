# Separate output-head calibration

The head-only prototype addresses one necessary coverage gap: the original
Qwen3.5-9B output head is a 248320 x 4096 BF16 matrix, about 2.03 GB. The
embedding is another equally large matrix. Compressing the head alone cannot
meet the roughly 2 GB language-model target. The [embedding prototype](embedding-calibration.md)
is a separate stage, and useful pretrained endpoint quality has not been demonstrated.

The completed decoder comparisons remain preserved baselines. Endpoint work
uses separate experiments and resource admission; these prototypes do not
automatically launch a 9B run.

## Original source and hidden-state cache

`cache_original_head_inputs` captures the original native model's `lm_head`
input, after the final normalization, and stops before computing vocabulary
logits. It writes bounded disk shards of detached original BF16 hidden rows.
The selected positions have valid current and next tokens; shifting never
crosses record boundaries. EOS remains a valid target.

Use the exact original BF16 checkpoint, including any intended BF16 fine-tune.
Native recurrent/norm parameters retained in FP32 stay in their original
precision; the cache does not blanket-cast the teacher. The head and captured
hidden states must be BF16.
Provenance identifies its immutable revision or content manifest, configuration,
head weights, input records and selected positions. An unchanged head alone
does not prove that upstream hidden states came from the same decoder.
Source provenance is supplied by the caller and checked consistently; it does
not authenticate a downloaded checkpoint or detect benchmark leakage by itself.

Calibration, search-validation and final-evaluation caches have explicit roles.
Only calibration data can enter fitting. Supplied held-out caches are checked
for source consistency and record overlap before any parameter update. Keep
benchmark answers out of calibration and preserve a genuinely unused final
confirmation set. Repeatedly inspected diagnostics are development data.

After caching, release the original decoder before constructing the head's
optimizer. Keep only its original BF16 head, cached hidden states, learned
group factors and low-rank correction. Cache tensors must not retain a model
autograd graph. The first real experiment should keep all non-head tensors
at their original saved precision and compare original, unrotated ternary and
rotated ternary heads using the same inputs, initialization and optimizer budget.

## Bounded weight computation

`BoundedTernaryHead` is a dedicated matrix quantizer, separate from decoder
windows. It requires complete input groups per row and reconstructs at most
the configured number of output rows. Initialization, rotation, low-rank
correction, group statistics, soft/hard reconstruction and export all operate
on bounded row ranges. It does not provide an implicit full-matrix forward.

The original source remains BF16 and immutable. Factors, A/B, optimizer state
and the saved derivative point remain FP32. In the rotated condition,
`W_rot = W R` is computed before adding the low-rank correction, and hidden
states use the matching input transform. No rotation crosses normalization,
gates, recurrence or any other nonlinear operation.

`begin_step(progress)` captures one compact pre-update snapshot at each soft
optimizer step. The last such snapshot supplies the hard-phase Jacobian.
Row callbacks and no-gradient normalization passes never advance the schedule
or change that snapshot. Hard forward values come from current factors;
gradients use the saved soft point and its actual sharpness. There is one
gradient clip and optimizer update after the complete logical batch.

This bounds matrix scratch, not total memory. The original head, FP32 factors,
A/B, their gradients, Adam state, snapshot and final compact payload still
occupy memory. The real 9B shape needs CUDA soft/hard-backward and host export
measurements before selecting chunk sizes or claiming that it fits a 5090.

## Full-vocabulary teacher loss

The declared objective is `KL(original_head || ternary_head)` at temperature
one, averaged over valid positions. It preserves the teacher distribution;
it is not a token-label cross-entropy objective or a decoder Huber loss.

A first no-gradient pass computes one global teacher and student normalizer
across **all** vocabulary chunks. A second pass recomputes each chunk and
backpropagates `(student_probability - teacher_probability) / valid_positions`.
Shared low-rank parameters accumulate contributions from every vocabulary
and token chunk. Parameters remain unchanged between passes and chunks.

The reported loss is the actual KL, not a surrogate scalar that merely has
the same first derivative. Chunk-local softmax and independent per-chunk mean
losses are incorrect. No top-k target approximation is used. The first
reference uses FP64 normalization arithmetic; native head logits still use
the source BF16 compute contract. CUDA cost and rounding differences between
chunked and whole-head GEMMs require measurement.

## Export and reference runtime

`export_head_candidate` creates a normal mixed-format ternary candidate. It
counts and validates bounded code chunks, chooses among two-bit, BITCOS and base-3
using complete serialized candidate bytes, then packs only the selected
representation. Sign streams remain continuous across non-byte-aligned chunk
boundaries; row indexes, FP16 scales, rotation signs and metadata are counted.
Cross-pass checks reject changing codes/scales during export.

The candidate contains no original BF16 master or separate LoRA adapter. The
low-rank correction is absorbed before ternarization. Exported scales use the
existing FP16 contract; measure that rounding separately from exact codec
and reload parity. `save_mixed` streams candidate files and checks actual
artifact bytes against the predicted complete size. Supply tokenizer assets
for a self-contained deployment; the tiny fixture has no real tokenizer.

Packed execution still reconstructs floating-point row chunks. A real output
head has 3,880 chunks at the reference default of 64 rows. Measure larger
chunks or an explicitly reported dense latent-head cache before lengthy
generation evaluation. Smaller files do not establish faster inference or
smaller loaded VRAM. Keep vision bytes and modality scope explicit.

## Measuring the stored head on original inputs

`openbonsai.head_evaluation.evaluate_packed_head_from_cache` measures an exported
ternary `MixedLinear` against the original frozen BF16 `nn.Linear` on one
explicitly supplied original-input cache. It scores every valid position in
that cache, using the same full-vocabulary teacher KL and global FP64
normalization as head calibration. It does not fit parameters or discover other
data roles.

```python
from openbonsai.head_evaluation import evaluate_packed_head_from_cache

report = evaluate_packed_head_from_cache(
    packed_head, original_head, search_cache,
    source_provenance=source_provenance,
    token_microbatch=256,
    vocabulary_chunk=4096,
)
```

Prepare both heads on the same device with BF16 computation; `original_head`
must be the unchanged, frozen source endpoint. The packed head must have no
decoded-weight cache. Its `forward_rows` method decodes only the requested
output rows and applies the stored input rotation and bias. The effective row
range is bounded by both `vocabulary_chunk` and the packed runtime's
`row_chunk_size`; set and measure that runtime setting explicitly. The scorer
does not require full-vocabulary logits and never warms a complete decoded-weight
cache. Reconstructed-weight and logit scratch depend on the selected row and
token bounds. Payload validation has its own temporary allocations, and the
cache reader loads one stored shard at a time. The original head and compact
payloads remain resident; this is not a total-memory bound.

The report identifies the original source, cache, positions, stored candidate
bytes and complete serialized SHA256, rotation, precision and actual chunk
settings. Source hashes and supplied provenance claims must agree. A low-level
score does not authenticate an entire staged job: the caller must separately
verify its completed candidate/cache receipts. Streaming source and candidate
checks also consume time and belong in the experiment's resource budget.

Compare this stored-head KL with the final hard calibration/search KL using
the same cache, token batches and row ranges. The former uses exported FP16
scales and bias; the latter uses the learned pre-export scales. Their difference
can reveal export rounding effects and is separate from exact codec/reload
parity. Neither is a full-model NLL, ARC or generation result. The full-model
evaluator's separately declared warm-cache policy is unchanged.

[Focused validation](../validation-packed-head-scoring.json) covers bounded
row execution, source binding, dense-oracle agreement and serialized reload on
tiny CPU fixtures; it does not establish 9B quality or CUDA capacity.

## Offline integration check

```bash
python -m openbonsai.head_smoke --output runs/head-paired
```

This uses a random tiny native Qwen3.5 hybrid/multimodal fixture, independent
calibration and validation records, and matched unrotated/rotated head fits.
It verifies the original model remains unchanged, exports complete mixed
artifacts, and checks logits and multiple-EOS settings in a fresh Python
process. The report separates validation KL, serialized bytes and training
time. It is a software proof, not evidence of language quality, real-model
size, CUDA feasibility or Bonsai equivalence.

[Recorded validation](../validation-endpoint-head.json) contains 571 passing
CPU tests and both fresh-process fixture results. The rotated fixture has
slightly worse validation KL; no pretrained quality conclusion follows from
either random-model result.

Before combining a calibrated head with a compressed decoder, evaluate that
combination again. A head fitted on original hidden states may respond
differently to a compressed decoder's activation distribution. Head-only,
embedding-only and combined coverage experiments must remain distinguishable.
