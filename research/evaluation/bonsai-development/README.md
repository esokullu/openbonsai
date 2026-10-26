# Frozen Bonsai development comparison

[frozen-inputs.json](frozen-inputs.json) is the exact reviewed conversion input
supplied to the [completed official Bonsai CPU run](../../../docs/official-bonsai-development-result.md).
The run saved and evaluated a semantically identical JSON copy with different
serialization bytes; both hashes are recorded in the validation receipt. This
file is included for reproducible evaluation and transfer to the separate 5090
checkout. It contains no weights or tokenizer asset files.

This is **reused development data**, with role `search_validation`, not a fresh
held-out confirmation set. Its questions and answer keys are for scoring only;
do not use them for candidate fitting, teacher-trace preparation, or allocation
optimization. Historical `heldout` item names do not change that status.

| Identity | Value |
|---|---|
| Exact file bytes | 637466 |
| Exact file SHA256 | `9936a5382e9cc9921ac21ccae0ceb84fab8a9a7681d46fd0f3efbbc05e6b70be` |
| Canonical `frozen_sha256` | `8021d9cc8f54943a3b816af74cbc41d588173d385f51b73a681e979393ef5fdf` |
| Tokenizer | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Workload | 88 texts: four NLL inputs, 80 ARC option inputs, four generation prompts |
| Scoring | 2048 causal NLL targets; 20 ARC-Easy option-letter choices |
| Generation | Greedy; 128 tokens maximum per prompt; EOS `[248044, 248046]` |

The file is copied byte-for-byte from the reviewed conversion. Git text
conversion is disabled for this path so a Windows checkout preserves its hash.
Do not reformat it, render another template, retokenize answer boundaries, or
change caps for a matched comparison. Both backends must still pass the exact
native token-ID and raw-piece checks before scoring.

Public dataset provenance is recorded in the manifest:

- [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
  test split, four already selected token slices. Text fields are reconstructed
  from authenticated token pieces, not independently recovered source prose.
- [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453),
  20 already selected validation questions. The score is conditional likelihood
  of the A/B/C/D label, not the whole answer text.
- Four manually authored instruction, reasoning, coding and Turkish rewriting
  prompts already used in the original development diagnostics.

Dataset and tokenizer material retain their respective licenses. The original
transport, rendering helper hashes, tokenizer file hashes, selected source IDs,
and explicit normalization policy are recorded in the manifest and the
[validation receipt](../../../validation-official-bonsai-development.json).

Use the [native profile](../../../configs/evaluation/bonsai-development-native.json)
or [Prism CPU profile](../../../configs/evaluation/bonsai-development-cpu.json)
with the [matched evaluation runner](../../../docs/matched-evaluation.md).
Profile identity binds the canonical manifest hash; command-line expected
manifest hashes bind the exact file bytes. Models, authenticated tokenizer
assets and an appropriate runtime are still required. Packaging the inputs
does not authorize overlapping GPU work or establish native CUDA feasibility.

The official result uses centered FP64 normalization of FP32 logits. Earlier
OpenBonsai FP32 cross-entropy scalars cannot be converted into an exact matched
baseline; rescore the original and selected completed artifact. These common
inputs and probability rules do not make different backend kernels identical.
