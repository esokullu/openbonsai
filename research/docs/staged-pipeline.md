# Staged local BF16 compression

This workflow joins the existing rotation, decoder, endpoint and mixed-export
APIs around one immutable local Qwen3.5 checkpoint. A merged BF16 fine-tune is a
new source checkpoint: generate its calibration traces with that same source,
seal a new job, and fit new candidates. Candidates from the base model cannot
be relabeled as candidates for the fine-tune.

The workflow is experimental. Native tiny-model tests establish composition,
source preservation and reload behavior. They do not establish useful 9B
quality, 5090 memory fit, a good calibration recipe, or equivalence to Prism's
training method. The current real-model quality limitations remain documented
in [the research direction](research-direction.md) and
[the 5090 results](5090-runtime.md).

## One process and one output per stage

Install the process-monitor dependency with `python -m pip install -e '.[evaluation]'`.
The older `openbonsai train` command remains the unrotated decoder-only path;
these stages use `openbonsai pipeline`.

The command-line interface accepts a JSON request with exactly this envelope:

```json
{
  "format": "openbonsai.pipeline-request.v1",
  "stage": "decoder",
  "parameters": {}
}
```

Use absolute local paths. Each invocation owns a new output directory and one
externally supervised child process. Its deadline includes loading, hashing,
fitting, packing and publication; a timeout is not a request to shorten training
or retry. CUDA observations record the process's allocator peaks and device
free memory before and after each measured operation. They do not bound
unobserved transient allocations by other processes or enforce a host RAM cap.
The Windows supervisor requires the documented harmless native-process
validation before actual model work; Mac mocks do not establish Windows support
on a particular machine.

Run either the installed command or the module entry point. This example uses
environment variables for the caller-selected paths, exact request file hash
and time budget; the budget is a limit, not a runtime estimate:

```sh
openbonsai pipeline --run \
  --request "$REQUEST_FILE" --expected-request-sha256 "$REQUEST_SHA256" \
  --output "$NEW_OUTPUT_DIRECTORY" --device cpu --threads 2 \
  --max-seconds "$BUDGET_SECONDS" --minimum-free-cuda-bytes 0
```

`python -m openbonsai.pipeline` accepts the same flags. `--max-seconds` must be
positive and at most 14,400. CUDA uses `--device cuda:0` and requires an explicit
`--minimum-free-cuda-bytes` of at least 1,073,741,824. This lower bound is a guard,
not evidence that 1 GiB of margin suffices for a particular 9B operation. Seal
and assembly are CPU-only stages. Omitting `--run` does not launch work.

A downstream command references a completed upstream operation using:

```json
{"path": "/absolute/previous-output/result.json", "sha256": "<exact file SHA256>"}
```

The outer successful `result.json` is the handoff. Consumers check its payload
hash, successful process status and cleanup. An inner `stage.json`, candidate
file, or producer receipt alone does not satisfy the command-line stage contract.
The explicit historical-token input described below is an attributed data
import, not a claim that a preparation stage completed.
This distinction matters when a function publishes its files and a subsequent
deadline or memory check fails. Such files remain diagnostic evidence; the
failed operation is not an admitted input to the next command.

There is no optimizer resume. A failed attempt is preserved and a retry needs a
new output directory. Completed stages can be reused only with their exact job,
source, code, data and recipe bindings. Source, tokenizer, sealed job and Python
package directories cannot be stage output locations.

## Stage order and inputs

| Stage | Purpose and handoff |
|---|---|
| `prepare` | Load one original local native model and generate self-produced calibration traces from explicit prompts. Preserve prompt IDs and generated IDs without decoding and re-tokenizing them. Publish a producer receipt bound to the exact model, tokenizer, prompt file, generation configuration and output JSONL. |
| `seal` | Consume successful preparation results or explicitly attributed historical token records, package tokenizer assets, validate the declared roles and freeze all stage recipes. Publish `job.json`; original model weights remain at their existing path. |
| `decoder` | Fit the declared rotated native text projections using the existing window trainer. Preserve original module names, source hashes and actual rotation signs in each candidate. |
| `cache-head` | Capture original final-normalized hidden states before the large vocabulary projection. Run separately for each declared role, using the same original checkpoint and common source provenance. |
| `head` | Fit the bounded head with the existing full-vocabulary KL objective on the calibration cache, then measure final hard calibration KL. An optional search-validation cache is scored; an optional evaluation cache is bound and checked for separation, but is not scored by this fitting stage. |
| `embedding` | Fit the bounded embedding table using complete row sweeps against its original BF16 rows. Export requires the live all-row soft/hard coverage certificate. This stage does not train on the text traces. |
| `assemble` | Verify the completed candidate stages, original source and declared coverage. Write one mixed artifact with retained tensors, native configuration and tokenizer assets. Its exact artifact byte total excludes the outer operation receipts. |

The `parameters` object has these exact fields; unknown and missing fields are
rejected:

| Stage | Parameter fields |
|---|---|
| `prepare` | `model_directory`, `expected_model_sha256`, `tokenizer_directory`, `expected_tokenizer_sha256`, `prompts`, `expected_prompts_sha256`, `generation`, `role` |
| `seal` | `model_directory`, `tokenizer_directory`, `traces_by_role`, `decoder`, `head`, `embedding`, `assembly` |
| `decoder`, `embedding` | `job` |
| `cache-head` | `job`, `role` |
| `head` | `job`, `calibration_cache`, `search_cache`, `evaluation_cache`; use JSON `null` for absent optional caches |
| `assemble` | `job`, `candidate_stages` (a nonempty list of result references); optional `assembly_control` for an explicitly named subset control below |

`job` references a successful seal result. The preparation `generation` object
requires `seed`, `max_new_tokens`, `max_prompt_tokens`, `max_samples`,
`enable_thinking`, `do_sample`, `temperature`, and `top_p`. Tokenizer hashes use
the supported subset from `pipeline_sources.tokenizer_identity`, while model
hashes use `eval_backends.local_directory_identity`; they are different scopes
even when the files originally share one directory. Prompt and result references
use the SHA256 of the exact file bytes.

The preparation and seal operations are separate. Seal's usual role map has the form
`traces_by_role[role] = {prepared, sequence_length, max_samples}`, where
`prepared` is a successful preparation-result reference. The roles are
`calibration`, optional `search_validation`, and optional `evaluation`.
Head-cache inputs and assembly candidate-stage inputs also use successful
outer-result references. These checks prevent a pipeline command from silently
treating arbitrary token files or incomplete caches as completed preparation.
Historical token records have a separate, explicit input form below.

Stage recipe fields are explicit, with the backward-compatible head rotation
option described below. Decoder settings include both the
quantizer configuration and every `TrainingConfig` field, including teacher
prefix, objective, optimizer parameters and update schedule. Head and embedding
settings include every respective training and quantizer field. Endpoint LoRA
initialization has its own seed, separate from the rotation seed. Head cache
position/shard limits, packing choice, assembly coverage, output scope, byte
budget and runtime row chunk/cache policy are also part of the sealed job.
Changing any of them creates a different job; the workflow does not choose or
recommend a new 9B recipe automatically.

## Reusing historical token records

When the original IDs and masks already exist, `seal` can import them without
generating new traces. For each such role use exactly
`{historical, sequence_length, max_samples}` instead of the `prepared` form.
`historical` contains:

- `traces`: the absolute path and SHA256 of the original JSONL file.
- `producer`: the absolute path and SHA256 of a JSON attribution receipt.
- `expected_selection`: the predeclared `num_records`,
  `consumed_jsonl_sha256`, ordered `record_sha256s`, and ordered
  `raw_token_record_sha256s` for the specified selection.

The two forms cannot be combined. The importer verifies the pinned files and
requires the selected count, consumed prefix, raw token order and effective
padded/truncated token order to match. It copies the complete original JSONL
and producer receipt byte-for-byte into the sealed job; it does not decode,
retokenize, regenerate, reorder or overwrite the source records. The successful
supervised seal result remains the handoff for later stages. A failed selection
check does not produce an admissible completed seal.

The producer receipt uses the same content bindings as direct Python
`seal_job`: a nonempty `kind` and `description`, `source_content_sha256`,
`tokenizer_content_sha256`, and `source_jsonl_sha256`, with a matching `role`
when supplied. An attribution receipt should state that the records are
historical and retain the original producer identity, source revision,
generation settings, execution precision policy and supporting receipt hashes.
Do not describe imported records as newly generated by `prepare`.

These are caller-attributed provenance claims. Hashes authenticate the supplied
bytes; they do not independently prove the historical generation or verify
every supporting claim in arbitrary producer metadata. In particular, a
checkpoint directory hash does not establish whether its old producer cast
small FP32 tensors to BF16. Preserve that historical policy even when the new
cache and evaluation use `preserve-checkpoint`. New head caches must come from
the declared original computation, not from historical decoder-window caches.
Exact role-overlap checks still apply, and benchmark answers remain excluded
from fitting and search.

[CPU validation](../validation-historical-trace-import.json) includes 46 focused
checks and a real supervised seal followed by two original-head-cache stages.
Wrong record order fails the supervised handoff; a wrong input hash is rejected
before creating an output directory. The 1,795-test suite passed with one
Windows-only test skipped on macOS. These checks establish data handling and
command behavior on a tiny native fixture, not pretrained endpoint quality.

## Rotated and unrotated head controls

The head recipe accepts `head.quantization.rotation_mode` as `"hadamard"` or
`"none"`. Omitting it preserves the existing signed Hadamard behavior and leaves
the sealed recipe unchanged. The option applies only to the head; decoder and
embedding recipes continue to require their existing rotations.
Existing code-identity checks still apply: this option does not migrate or
retag jobs sealed against an earlier implementation.

For an unrotated head, add `"rotation_mode": "none"` to the otherwise complete
head quantization object. This supplies no rotation signs to the head operator;
it does not substitute all-positive signs, which would still apply a Hadamard
transform. Publication, loading and assembly check that the candidate's actual
rotation metadata and sign storage agree with its sealed head recipe.

Use two separate jobs for a matched comparison, changing only the head rotation
mode. Keep the original checkpoint and precision policy, token records, position
selection, initialization seed, training settings, packing and evaluation
settings identical. Each job requires its own bound cache/stage receipts; do not
relabel one job's cache or candidate as belonging to the other. Assemble each as
an `H-only` control so its decoder and embedding remain original, and evaluate O
through the original-model path. Head fitting continues to use complete
full-vocabulary KL and its own optimizer settings; decoder optimizer findings
do not automatically transfer to this objective.

After fitting, the head stage records `report.calibration` over every valid
position in its calibration cache and, when supplied, `report.search_validation`.
Both measure the final hard operator against the original teacher using native
logits and global FP64 probability normalization across the full vocabulary.
These measurements precede export and FP16 scale rounding; they are not packed
reload scores or substitutes for language evaluation. The calibration pass is
timed separately as `evaluate-head-calibration` and must be included in resource
budgets. It adds no optimizer updates and does not score the evaluation cache.
The [focused validation](../validation-head-final-calibration.json) checks both
rotation modes against a dense oracle and verifies state preservation and
failure before publication.

This option enables a controlled experiment. It does not establish that rotating
or ternarizing the pretrained 9B head preserves quality, nor that a head-only
artifact meets the full model's size target.

The [validation receipt](../validation-head-rotation-control.json) records
1,746 passing CPU tests and one Windows-only skip. The paired supervised native
fixture uses identical cached inputs, initialization and training budgets for
both head modes, and checks exact fresh-process logits, EOS settings, retained
FP32 tensors, candidate bytes and complete artifact sizes. These random-model
checks establish software behavior, not pretrained endpoint quality.

## Isolated endpoint controls from one job

An all-coverage job can also produce explicitly requested control artifacts
without changing its sealed recipes or refitting candidates. Add
`assembly_control` to an assemble request's `parameters`, for example:

```json
{
  "quantized_components": ["embedding"],
  "retained_original_components": ["decoder", "head"],
  "output_scope": "language-only",
  "evaluation_label": "E-only",
  "budget_bytes": 20000000000
}
```

This is the `assembly_control` object, not a complete command request. The
example budget is illustrative. `candidate_stages` must contain exactly the
successful embedding-stage result for this E-only control. The remaining
components come from the unchanged original checkpoint with its saved native
precision, including small FP32 exceptions. They are not newly quantized or
substituted with unrelated BF16 candidates.

The supported labels and quantized components are `D-only` (`decoder`),
`E-only` (`embedding`), `H-only` (`head`), `D+E` (`decoder`, `embedding`) and
`D+H` (`decoder`, `head`). Lists use canonical order `decoder`, `embedding`,
`head`; the retained list must be the exact complement. Each selected decoder
component requires its entire declared projection set. Missing, extra or
duplicate stages fail. Each endpoint candidate keeps the same source, job,
recipe, calibration certificate where required, actual signs, scales and
serialized bytes across controls.

Controls must be proper nonempty subsets of the sealed coverage. Their
`output_scope` must match the sealed job; row chunks and decoded-cache policy
also remain fixed. A decoder-only job has no supported proper subset at this
component granularity. Use the ordinary assemble request, without this object,
for the all-matrix final artifact; its full coverage and sealed budget remain
mandatory. O, the original baseline, uses the original evaluation path.

A control needs its own explicit complete-artifact budget because retained
original matrices can make it much larger than the final ternary target. The
artifact and stage receipt record effective `pipeline_assembly.coverage` as
`control`, preserve `sealed_pipeline_assembly`, and set
`delivery_target_complete=false`. They include the request digest, evaluation
label, selected and retained component names, retained dtype/byte accounting,
modality scope and actual whole-artifact storage. Passing the control budget
does not satisfy the sealed final budget. Outer run reports remain outside the
deployable artifact's byte count.

Per-component retained accounting covers the eligible candidate matrices in
each omitted role, including their biases. Always-retained normalization,
recurrence parameters/gates and other native tensors remain in the complete retained
tensor inventory; configuration and containers remain in fixed storage. These
component subtotals do not replace the complete artifact count.

These controls use current same-job stage receipts. They do not import
historical experiments, bypass a changed implementation seal, modify training
objectives or automatically launch evaluation. Compare E/H separately before
drawing conclusions about D+E/D+H or all coverage. Tiny native reconstruction
and reload checks establish software behavior, not pretrained quality or CUDA
feasibility.

## What the source and trace receipts prove

The source identity hashes the relative names, byte lengths and SHA256s of every
file in the materialized local checkpoint directory. Its common provenance is
`source_checkpoint = "sha256:<content digest>"`, with the same digest recorded
as `source_content_sha256` and `source_revision`. A local path, a model name,
`main`, or a missing Hugging Face commit hash is not an immutable identity.
There are no downloads or remote Python code execution in this workflow.

Supported sources are original native dense Qwen3.5 causal or conditional
checkpoints with merged safetensors weights. Candidate matrices must already be
BF16. Adapter-only, quantized, tied endpoint, unsupported custom topology, and
unsupported dtype sources are rejected. Loading uses eager native execution
with default RoPE and no inherited request cache. Bounded saved FP32 exceptions
are read back from their individual safetensors tensors if the native loader
would otherwise cast them; they are not rounded and relabeled as originals.
Unused checkpoint `mtp.*` tensors are recorded separately and are not counted
as compression savings.

Tokenizer packaging copies the supported files byte-for-byte: required
`tokenizer.json` and `tokenizer_config.json`; optional `vocab.json`, `merges.txt`,
`added_tokens.json`, `special_tokens_map.json`, `chat_template.jinja`, and direct
`chat_templates/*.jinja` files. It does not recursively copy the source
checkpoint directory, model weights, model/generation configuration or vision
processor assets. Custom tokenizer code and version-selected tokenizer files
are outside the initial supported contract. Symlinks are rejected. Raw template
newlines remain intact even when the native text loader normalizes them in
memory. The model's native configuration and generation settings are written
by the mixed exporter separately.

A trace receipt binds the raw JSONL and raw producer-receipt file hashes,
producer-declared source/tokenizer identities, role, ordered token/mask hashes,
sequence length, maximum consumed records, vocabulary and padding ID. It records
right-padding/right-truncation and counts the exact consumed prefix. Native
valid token IDs and binary masks are checked; head calibration requires valid
adjacent next-token positions. In-memory tensors are checked again against
their receipts before reuse. EOS tokens remain part of the data.

If the original text config omits `pad_token_id` or sets it to null, sealing
resolves the existing pad token from the authenticated local fast tokenizer.
This supports the original Qwen3.5 checkpoint without editing its config.
The token must already exist in the saved tokenizer vocabulary and fit the
model vocabulary. Explicit model and tokenizer declarations must agree;
invalid model IDs do not trigger fallback. There is no EOS or zero-ID default.
An explicit valid model ID retains the existing byte-only path when the
tokenizer does not declare padding.

Each new trace receipt records `padding_source` (`model_text_config` or
`tokenizer`), and loading checks that policy against the sealed source again.
Resolution uses a temporary tokenizer-only package without network access or
custom tokenizer code. It does not alter the runtime model config, generation
settings, embedding padding index, raw IDs, existing masks or EOS tokens.
Only missing right-tail positions receive the resolved pad ID and a zero mask.
The [padding validation receipt](../validation-staged-tokenizer-padding.json)
records the native CPU command proof and original-tokenizer check; it does not
claim pretrained CUDA or model-quality validation.

The supplied pipeline producer writes an attributable execution receipt. Hashes
authenticate its bytes and link the stages; they cannot prove that an arbitrary
third-party producer is honest. Direct Python `seal_job` also accepts a producer
receipt for caller-declared token records. That is a lower-level trust boundary,
not independent evidence that those tokens were generated by the claimed
checkpoint. The command-line workflow distinguishes its completed preparation
handoff from the explicit historical import above; neither makes arbitrary
external producer claims independently trustworthy.

Exact token-record overlap between declared roles is rejected. This is not a
semantic contamination detector: callers must keep benchmarks and final
confirmation data out of preparation prompts and calibration. The fitting
workflow does not create a new final benchmark or claim a quality improvement
from serialization success.

## Rotation identity, memory ownership and deployment

Candidate keys remain the original source module names throughout fitting and
stage publication. Rotation signs are generated from the declared seed and
original name, then stored as authoritative bytes with a digest. For a
conditional model's language-only export, assembly maps
`model.language_model.*` to `model.*`; `lm_head` is unchanged. It renames only
the manifest keys. Regenerating signs from the new text-only names would change
the model and is not allowed. Causal source names already use the text layout.

Each stage loads one source model. The native language-only extraction creates
separate module and tensor objects sharing frozen original storage; it does not
deep-copy a second full checkpoint. Neither view may mutate that shared source
storage. Endpoint fitting releases the original decoder before moving only the
selected endpoint to its compute device. Candidate publication and reading use
one candidate at a time rather than accumulating every decoded matrix.

The all-matrix scope combines the supported decoder projections with embedding
and output head. The random four-layer hybrid fixture has 27 such matrices; the
target 9B topology has 202 under the conservative gate policy. Tiny recurrence
gates, normalization/dynamics and other retained native tensors remain at
their recorded precision. A language-only artifact excludes vision through the
native extraction helper; a retain-vision artifact preserves that component.
See [text-only export](text-only-export.md) for its pure-text input contract.

Packing choices are `2bit`, `bitcos`, `base3` and `auto`. Auto compares complete
candidate bytes, including indexes and metadata. Repacking fixed ternary codes
is lossless. Original BF16-to-ternary fitting and export scale/bias rounding are
separate approximation steps and can change quality. Learned low-rank
corrections are folded into the final ternary decisions, not deployed as a full
additional LoRA weight sidecar. [The storage budget](storage-budget.md) explains
the conditional size projection and why it is not a quality result.

The deployed artifact uses the existing reference floating-point reconstruction
runtime. It is not an optimized ternary GPU kernel or GGUF. The optional decoded
BF16 cache is an explicit inference memory tradeoff and is excluded from disk
bytes. Use [the matched evaluator](matched-evaluation.md) on the original and
assembled model with the same frozen development protocol; endpoint losses and
an artifact reload do not substitute for assembled-model quality evaluation.
