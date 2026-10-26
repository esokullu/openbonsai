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

## Remaining gates

Before real comparison, verify the actual official Bonsai GGUF hash and tokenizer,
build and test the Windows/CUDA worker, measure a disjoint short BF16/OpenBonsai/
Bonsai preflight, and record residual backend drift. Keep the current 5090
compression experiment unchanged until its completed results are reviewed.

Count actual deployable packages with identical modality scope. Prism's language
GGUF includes tokenizer metadata; OpenBonsai needs its tokenizer/configuration
assets counted too. Retained BF16 vision and optional compressed vision are
separate comparisons. Merely subtracting vision bytes does not create a working
text-only artifact. No official Bonsai weights were loaded in this CPU preflight.
