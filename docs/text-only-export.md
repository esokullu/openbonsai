# Native text-only extraction

`extract_text_model` creates a native `Qwen3_5ForCausalLM` view of an original
Qwen3.5 conditional-generation checkpoint. It preserves its language weights,
small FP32 dynamics tensors, rotary buffers, text configuration and actual
generation configuration. It removes the vision tower from the export scope.
It does not quantize anything or improve model quality.

```python
from openbonsai.adapter import load_base_model
from openbonsai.text_export import extract_text_model, text_module_name

source = load_base_model(
    original_bf16_checkpoint, revision=pinned_revision, dtype="bfloat16",
)
extracted = extract_text_model(
    source, source_id=original_bf16_checkpoint, source_revision=pinned_revision,
)
text_model = extracted.model
```

Extract before running generation or fitting candidates. The source must be an
original, frozen, evaluation-mode native model with eager attention, default
RoPE and `use_cache=False` in its text configuration. Configured tying, parameter
storage aliases, quantizer/adaptor wrappers, custom forward hooks, meta weights
and existing multimodal `rope_deltas` are rejected. Start from the intended BF16
fine-tune; do not pass a previously compressed checkpoint.

The text model has separate modules and detached tensor objects, while sharing
frozen underlying storage with the source. This avoids a second full weight
allocation. **Do not update either model's shared weights in place.** Replacing
the text view's modules for calibration is supported; treating this view as an
independently mutable BF16 fine-tuning clone is not.

## Existing candidate and export path

The name mapping is `model.language_model.*` → `model.*`; `lm_head.*` stays the
same. Vision and unknown paths fail. Candidates fitted on the original full
wrapper can be remapped without changing their payload or original-weight
fingerprint:

```python
from openbonsai.mixed_export import prepare_mixed_export, save_mixed

text_assignment = {text_module_name(name): candidate
                   for name, candidate in original_assignment.items()}
with prepare_mixed_export(
    text_model, text_assignment, metadata=extracted.metadata,
    tokenizer_dir=frozen_tokenizer_assets,
) as plan:
    report = save_mixed(plan, text_assignment, output_dir,
                        budget_bytes=predeclared_budget_bytes)
```

Pass a directory containing **only tokenizer assets**, not the full source
checkpoint: the exporter recursively copies supplied files. Preserve the full
vocabulary, chat-template bytes and EOS policy even though vision is excluded.
The resulting manifest includes extraction provenance, tensor mappings and all
their serialized bytes. Source ID/revision are caller-supplied identities;
candidate fingerprints and artifact file checksums are verified by the existing
exporter and loader. The extractor itself does not hash or clone every weight.

Load the complete artifact with `load_mixed`. Its native text configuration and
retained-state inventory contain no vision tensors, and the manifest reports
zero vision bytes. This does not create a separate, usable vision package.

## Computation and evidence limits

Use fresh text token requests, identical masks/positions, and explicit generation
length limits within the copied context size. The test suite compares padded
forwards, explicit positions, fresh native cached and uncached greedy generation,
EOS behavior, mixed candidate export and fresh-process reload. It uses a random
four-block native hybrid fixture. No actual 9B conversion, CUDA preflight,
pretrained quality or 2 GB deployment is established by these checks.

Image/video inputs, multimodal continuation and transferred caches are outside
this contract. Native CausalLM accepts extra keyword arguments: callers must
reject media inputs rather than assuming that the class enforces this boundary.
Unparameterized generation defaults, arbitrary custom masks and alternative
cache implementations have not been certified.

This path makes language-only size accounting possible. Useful broad ternary
quality and the [projected storage budget](storage-budget.md) still require a
real, measured artifact. Execution continues to reconstruct floating-point
weights; no speedup is implied.
