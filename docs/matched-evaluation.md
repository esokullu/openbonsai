# Matched evaluation across Transformers and Prism

OpenBonsai needs a measured Bonsai comparison before claiming comparable quality.
The new evaluator freezes inputs once and uses the same token positions,
continuation boundaries, stopping rules and probability reduction for native
Qwen3.5, an OpenBonsai mixed artifact, and the pinned Prism runtime. It is an
experimental **text-only** evaluation path. It does not establish 9B quality,
multimodal preservation, CUDA feasibility or Bonsai equivalence.

## Contract

`openbonsai.evaluation` defines `openbonsai.frozen-evaluation.v1`. A frozen
manifest records original UTF-8 bytes, rendered UTF-8 bytes, raw token pieces,
input IDs, item IDs, source/tokenizer identity, role and a canonical SHA256.
Callers render templates and declare any preprocessing before freezing. The
scorer never inserts BOS, renders another template, truncates input or guesses
an answer boundary. Each backend verifies all frozen token IDs and byte pieces
before model evaluation.

- Target position `j` uses logits from input position `j-1`. The full vocabulary,
  including unused padded model rows, enters the normalizer. FP32 logits are
  centered and normalized with FP64 logsumexp; token losses are summed in FP64.
  This differs from the earlier FP32-cross-entropy diagnostic. Recompute every
  comparator under this protocol; do not put old and new metrics in one table.
- `sliding_windows` retains overlapping context and scores every document token
  after its first exactly once, including the tail. Target score positions are
  distinct from visible context. Documents, windows and options remain separate.
- A multiple-choice option supplies the complete frozen prefix-plus-continuation
  token sequence and its explicit boundary. Options share identical prefix IDs.
  Scores sum all continuation token log probabilities; they are not length
  normalized. The first declared option wins a score tie. Answer labels may be
  reported for scoring, but no fitting or allocation API consumes this manifest.
- Generation uses raw-logit greedy argmax, lowest token ID for ties, explicit EOS
  IDs and an exact cap. The terminating EOS is retained. No sampler, penalty,
  native stop string, context shift or extra token to finish UTF-8 is applied.
  Raw output bytes remain available even if the last character is incomplete.
- Inputs are unpadded, single-sequence records. Mask holes, implicit truncation,
  malformed positions, nonfinite probabilities and incompatible provenance are
  rejected. Exact input overlap can be checked across calibration, search and
  evaluation manifests, including raw text under different templates. This is
  not a semantic contamination audit; dataset provenance still matters.

Results retain per-token/per-item scores and full generation IDs/bytes, plus
length-weighted NLL, perplexity and choice accuracy where labels are supplied.
Current repeatedly consulted diagnostics are development data. A final quality
claim still requires a fresh held-out set and appropriately long generations.

## Runtime adapters

`TransformersBackend` accepts native dense Qwen3.5 causal or conditional models,
including their mixed-format modules. It executes the native decoder once per
score record with `use_cache=False`, then evaluates the vocabulary head in
bounded position chunks. The native head semantics are checked against full
model forwards in tests. This avoids allocating all sequence-by-vocabulary logits
at once. Generation recomputes the full prefix. Model dtypes and head chunking
are reported; FP32 logits do not mean all model operations use FP32.
Raw-piece verification currently supports ByteLevel fast tokenizers with ASCII
added tokens. Non-ASCII added tokens are rejected explicitly; fine-tunes that
introduce them need an extended, verified decoder before using this evaluator.

`PrismBackend` starts a persistent JSONL worker through an argument list. It
hashes local model and executable files, verifies the pinned clean runtime
source, validates responses and closes on protocol errors or timeouts without
an implicit retry. The transport uses bounded reader/writer threads for Windows
pipe compatibility; the current actual build has only been run on macOS CPU.
Logs are drained with a storage cap. Supply expected hashes when checking a
known published artifact. In-memory native identities are caller declarations;
`TransformersBackend.from_local` hashes the materialized artifact before loading.

The worker uses the C API at
[`c9c34bc9c4042c13a31598ebf43461cd0ad8f00d`](https://github.com/PrismML-Eng/llama.cpp/tree/c9c34bc9c4042c13a31598ebf43461cd0ad8f00d).
It clears **all** recurrent and attention memory before each independent record
and before each generation-prefix recomputation. Scoring consumes selected full
vocabulary logit rows in batches of at most 64. The reference configuration uses
F32 K/V cache, flash attention disabled and no MTP. CPU is the default; a CUDA
build and positive GPU-layer count are explicit choices requiring their own
5090 preflight. Faster cache reuse has not been validated.

The binary build receipt includes compiler/CMake settings, source and executable
hashes and same-code C++ numerical/input checks. Smaller model files do not
establish inference acceleration. Report runtime time and memory separately.

## Two measured compatibility differences

The pinned Qwen tokenizer applies **NFC normalization**. The converted GGUF's
native tokenizer does not: the decomposed string `e` plus combining acute maps
to different IDs from the single precomposed `é`. The smoke retains that mismatch
as a negative control. It then explicitly normalizes rendered bytes with the
reference tokenizer's normalizer, preserves original raw bytes, verifies that
HF IDs are unchanged, and checks exact C-API IDs and raw pieces for every frozen
input. This preprocessing is recorded in the manifest. The adapters do not
silently normalize text. A later dataset freezer must declare its policy and
pass the same complete-input checks, including code and special-token literals.

The recurrent Q/K normalization also differs. Installed Transformers uses
`x / sqrt(sum(x*x) + epsilon)`; the pinned Prism CPU implementation uses
`x / max(sqrt(sum(x*x)), epsilon)`. See the
[Qwen call site](https://github.com/PrismML-Eng/llama.cpp/blob/c9c34bc9c4042c13a31598ebf43461cd0ad8f00d/src/models/qwen35.cpp#L551)
and [CPU operation](https://github.com/PrismML-Eng/llama.cpp/blob/c9c34bc9c4042c13a31598ebf43461cd0ad8f00d/ggml/src/ggml-cpu/ops.cpp#L4177).
A diagnostic-only substitution of epsilon placement reduced the tiny fixture's
maximum target-log-probability difference from `1.963e-4` to `1.282e-7`; restoring
the function reproduced the original native scores exactly. **Neither official
baseline is patched for the comparison.** This establishes a runtime difference,
not the cause of OpenBonsai's large quality loss. CUDA behavior, BF16 rounding
and pretrained-model drift require separate measurements.

## Reproduce the CPU preflight

Use a clean checkout of the pinned Prism commit, CMake and a C++17 compiler.
Keep build products outside the Prism source directory. The commands below
read local tokenizer assets; they do not download pretrained model weights.

```bash
python tools/build_prism_eval.py \
  --source /path/to/pinned-prism-llama.cpp --build-dir /path/to/prism-eval-build

python tools/prepare_eval_fixture.py \
  --tokenizer-dir /path/to/pinned-qwen-tokenizer \
  --prism-source /path/to/pinned-prism-llama.cpp --output runs/eval-fixture

python tools/smoke_prism_eval.py \
  --fixture runs/eval-fixture \
  --worker /path/to/prism-eval-build/cmake/bin/prism-eval-worker \
  --prism-source /path/to/pinned-prism-llama.cpp --output runs/eval-smoke

# Optional, separately labeled explanation of the measured normalization drift:
python tools/probe_eval_normalization.py \
  --fixture runs/eval-fixture \
  --worker /path/to/prism-eval-build/cmake/bin/prism-eval-worker \
  --prism-source /path/to/pinned-prism-llama.cpp --output runs/eval-normalization
```

The preparation command creates a random F32 native hybrid Qwen with three
recurrent blocks and one full-attention block, using the original 248,320-wide
model vocabulary and locally supplied tokenizer. It converts that fixture with
`--outtype f32 --no-mtp`. It is not a useful pretrained or ternary model.

Recorded actual-backend validation scored 176 target tokens over overlapping
windows. Maximum target-log-probability drift was `1.304e-4` nats; mean NLL drift
was about `8.39e-6`. The native chunked-head adapter matched an independent full
forward exactly at the selected boundary positions. Both backends reproduced
record A exactly after scoring unrelated record B; their three greedy IDs
matched. The engineering threshold of `0.001` nats per target is scoped to this
fixture, not a universal model-quality tolerance. Forced/dynamic EOS checks
validate stopping only. [Validation record](../validation-evaluation.json).

## Run a frozen manifest on CPU

`tools/evaluate_prism_manifest.py` runs an existing frozen manifest through the
unchanged common evaluator. It requires local weights, a clean pinned Prism
checkout, a locally built worker and an explicit resource profile. It downloads
nothing and changes no model. This supervisor supports **macOS/Linux POSIX
process groups only**, even though the backend transport also supports Windows.

Install the optional process-monitor dependency with `pip install -e '.[evaluation]'`.
Then run from the repository root, replacing paths and setting the four hash
variables from authenticated file/build receipts:

```bash
python tools/evaluate_prism_manifest.py --run \
  --manifest /path/to/frozen-inputs.json \
  --expected-manifest-sha256 "$MANIFEST_FILE_SHA256" \
  --profile /path/to/resource-profile.json \
  --expected-profile-sha256 "$PROFILE_FILE_SHA256" \
  --gguf /path/to/model.gguf \
  --expected-gguf-sha256 "$GGUF_FILE_SHA256" \
  --worker /path/to/prism-eval-build/cmake/bin/prism-eval-worker \
  --expected-worker-sha256 "$WORKER_FILE_SHA256" \
  --prism-source /path/to/pinned-prism-llama.cpp \
  --max-seconds 10800 --output /path/to/new-evaluation-directory
```

These flags require SHA256 values of the **exact file bytes**. The profile's
`manifest_sha256` instead binds the manifest's **canonical `frozen_sha256`**;
these two manifest hashes are different identities. The executable hash comes
from the build being used, not a hardcoded binary for another platform. The
runtime source commit remains the pinned commit above. The output directory
must be new; this command does not resume or overwrite an earlier attempt.

The profile format is `openbonsai.prism-cpu-profile.v1`. It records context
length, vocabulary size, one to four CPU threads and three explicit timing
probes: an NLL record index, a question/option index, and a token slice from an
existing referenced text with one final target position. Probes must cover every
corresponding input length, decoded length, requested vocabulary-row count and
maximum rows in a 64-token worker batch. The generation probe must cover the
longest permitted full-prefix forward: prefix length plus cap minus one. An
unsuitable profile is rejected before loading the model; the runner does not
select substitute data or probes.

The bundled [Bonsai development profile](../configs/evaluation/bonsai-development-cpu.json)
is bound to one existing development manifest. It contains no questions,
answers, token sequences or weights and does not supply that manifest. It is not
a default profile for arbitrary datasets. Other manifests need their own
reviewed profile and expected file hashes. This bounded runner currently requires
NLL, choice and generation records in a `search_validation` or `evaluation`
manifest; it does not accept calibration-role inputs.

All frozen texts must pass exact native ID and raw-piece checks before resource
scoring. Work counts come from the manifest, preserving every item, option,
target position, EOS list and generation cap. Admission reserves every possible
generated token, multiplies measured work by 1.25, adds 300 seconds and includes
the common evaluator's repeated tokenization check. Resource probes are excluded
from quality aggregates. Shape coverage and margins are estimates, not runtime
guarantees: an external monotonic supervisor caps the whole attempt at the
explicit limit, which cannot exceed 10,800 seconds. Failed admission stops
without shortening the evaluation or retrying.

Outputs include source/file identities, the derived plan, native-tokenization
and admission receipts, individual calls, item scores and generation bytes,
sampled process RSS and final completion status. An incomplete run is not a
complete baseline. Neither this command nor its synthetic tests establishes
pretrained quality. The FP64 reduction caveat still applies: old FP32 diagnostic
scores are not numerically matched comparators. The official PTQ1_0 CPU path
also uses quantized activation dot products; FP32 logits and KV cache do not
mean an all-FP32 model or establish parity with a different kernel or CUDA run.

The portable wrapper passed 29 focused synthetic/process tests and the full
1,502-test suite. These establish runner behavior, not a completed pretrained
quality evaluation. [Validation record](../validation-prism-manifest-runner.json).

## Run an existing native model

`tools/evaluate_transformers_manifest.py` runs one materialized original BF16
checkpoint or one complete `openbonsai.mixed.qwen35.v1` artifact. It uses the
same frozen-manifest scorer as the Prism path. It does not fit, quantize, export,
rename modules or convert a multimodal model into a text-only artifact. Legacy
packed checkpoint formats require their own loader and are rejected here.

```bash
python tools/evaluate_transformers_manifest.py --run \
  --kind mixed --model /path/to/sealed-mixed-artifact \
  --expected-model-sha256 "$MODEL_DIRECTORY_SHA256" \
  --tokenizer /path/to/materialized-tokenizer \
  --expected-tokenizer-sha256 "$TOKENIZER_DIRECTORY_SHA256" \
  --manifest /path/to/frozen-inputs.json \
  --expected-manifest-sha256 "$MANIFEST_FILE_SHA256" \
  --profile /path/to/native-resource-profile.json \
  --expected-profile-sha256 "$PROFILE_FILE_SHA256" \
  --device cuda:0 --cache-policy warm \
  --max-seconds 10800 --output /path/to/new-native-evaluation
```

For original BF16, select `--kind original --cache-policy cold` and its original
checkpoint directory. CPU is also supported. The model and tokenizer hashes
are the `sha256` values returned by `openbonsai.eval_backends.local_directory_identity`:
they bind every file, relative path and byte count in a materialized directory.
They are not a Hub revision or a hash of just one shard. The runner rejects
symlinks and changed sources; authenticate these directory identities against
the intended checkpoint before dispatch. Inputs remain offline and read-only.

Original loading defaults to `--original-dtype-policy bf16-cast`, preserving
the historical baseline that casts retained small FP32 tensors to BF16. For a
[staged-pipeline](staged-pipeline.md) artifact, explicitly select
`--original-dtype-policy preserve-checkpoint`: its loader checks saved native
tensor names/shapes/dtypes and restores bounded FP32 exceptions from the sealed
safetensors source. The execution receipt records the selected policy and the
restoration/unused-MTP table. The original directory hash is identical under
both policies, so it cannot identify this execution choice by itself. The two
policies must not be conflated when attributing a quality difference to
compression. Mixed artifacts retain their stored dtypes unchanged and reject a
`preserve-checkpoint` override. Scoring, tokenization, EOS, generation caps and
resource admission use the same evaluator under either original policy; no
claim is made about the quality impact of the dtype difference.

The [native development profile](../configs/evaluation/bonsai-development-native.json)
binds the same development manifest as the CPU profile, with separate native
resource shapes. Native scoring forwards **all input IDs**, including trailing
unscored positions; Prism forwards through the last required prediction. Native
probe admission therefore counts full input length, total head rows and bounded
head chunks explicitly. Neither path changes the frozen target positions.
The full tokenization gate precedes scoring, and timing probes are excluded
from quality. Full generation caps, a 1.25 timing margin and 300 seconds are
reserved without assuming early EOS. Failure stops without shortening the suite.

The mixed loader starts with decoded caches disabled. `warm` explicitly builds
the caches module by module, preserves stored payloads and row chunks, and
requires bitwise-equal cold/warm probe logits. It reports extra nonpersistent
cache bytes separately from artifact bytes. `cold` leaves reconstruction on
the execution path. These are floating-point reference executions, not evidence
of a fast ternary inference kernel. Original and mixed runs retain their native
multimodal tensor names and saved rotation signs; they share text-only scoring
inputs, not a claim of identical model or kernel arithmetic.

Each CUDA operation records synchronized allocator peaks and before/after free
memory, with a conservative peak-reserved headroom check. The provided profile
requires 4 GiB free. This does not continuously observe allocations from other
processes. Runtime metadata records software, device, driver information, head
chunk size, payload dtypes and cache policy. Wall time includes hashing, loading,
verification and cleanup; it is not an isolated inference-speed measurement.

The native runner owns one child under an external monotonic deadline, using
a process group on POSIX or a kill-on-close Job Object on Windows. It does not
fall back to an uncontained Windows process. **Before a model run on Windows,
run `python -m pytest -q tests/test_evaluation_supervisor.py` on that host.**
The harmless child/descendant tests must pass, including the native Windows
supervisor-death test. Mock Windows calls on macOS are not Windows validation.
Then validate native model loading with the tiny CPU fixture tests before
launching any pretrained GPU evaluation. Keep current training processes and
their frozen code untouched until their completed results have been reviewed.

The native runner and supervisor passed 60 focused checks and the full 1,562-test
suite; one native Windows-only check was skipped on macOS. The tiny CPU checks
include real original/mixed disk loading, preserved rotation signs, exact cache
parity, and a complete subprocess evaluation with source and cleanup receipts.
They do not establish pretrained quality or CUDA feasibility.
[Validation record](../validation-native-manifest-runner.json).

## Remaining gates

The [official PTQ1_0 CPU preflight](bonsai-reference-preflight.md) now verifies
the actual GGUF hash, tokenizer, scoring, state reset and short generation.
Before a quality comparison, recompute every comparator under the common
protocol, measure a disjoint short BF16/OpenBonsai/Bonsai preflight and record
residual backend drift. A Windows/CUDA comparison additionally requires building
and testing that worker on the target hardware.
Keep the current 5090 compression experiment unchanged until its completed
results are reviewed.

Count actual deployable packages with identical modality scope. Prism's language
GGUF includes tokenizer metadata; OpenBonsai needs its tokenizer/configuration
assets counted too. Retained BF16 vision and optional compressed vision are
separate comparisons. Merely subtracting vision bytes does not create a working
text-only artifact. The random-fixture preflight above loaded no official weights;
the separately recorded official-checkpoint preflight does.
