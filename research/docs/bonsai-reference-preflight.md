# Official Bonsai checkpoint preflight

The pinned official **PTQ1_0 language checkpoint** now loads and runs through
OpenBonsai's matched evaluator on macOS CPU. This closes an engineering gate;
it is not a BF16/OpenBonsai/Bonsai quality comparison. No OpenBonsai calibration,
compression or 5090 experiment settings changed during this check.

## Actual storage inventory

The downloaded file matches the published SHA256
`3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`
at [repository revision `5b2abb0`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/tree/5b2abb02362efbed46bbe204014b02837778b2c2).
Reading the complete GGUF with the pinned Prism reader gives:

| Component | Tensor count | Actual bytes |
|---|---:|---:|
| PTQ1_0 ternary matrices, including scales | 202 | 1,957,036,032 |
| Retained F32 tensors | 225 | 29,411,328 |
| Container metadata, tokenizer, signs and alignment | — | 11,039,616 |
| **Complete language GGUF** | **427** | **1,997,486,976** |

The embedding and output head each occupy **222,494,720 bytes** within the
ternary payload: 444,989,440 bytes combined. Their dimensions remain
248,320 × 4,096. Our preserved decoder-only experiment retained those tensors
in BF16 at 4,068,474,880 bytes combined. This confirms the need to address
endpoint coverage; it does not demonstrate that our endpoint calibration can
preserve quality. None of the language-file numbers include vision or MTP.

The retained F32 payload includes recurrent input gates, convolution weights,
normalizations and state parameters. The GGUF stores one block-4096 transform,
two shared sign vectors of widths 4,096 and 12,288, 201 forward-transform
matrix names and one inverse-lookup embedding. The runtime reports loading all
202 transformed weights. The grouped recurrent-value ordering flag is true;
our native implementation retains its own ordering as explained in the
[rotation audit](rotations-audit.md).

## Measured engineering checks

The same pinned C-API worker used in the [matched evaluator](matched-evaluation.md)
ran with four CPU threads, context 256, F32 K/V cache, flash attention disabled,
and no MTP. No Transformers model was loaded: the original pinned Qwen tokenizer
provided the tokenization reference.

- All frozen input IDs and raw byte pieces matched after explicitly declared
  reference NFC normalization; original raw bytes remain in the manifest.
- The authored diagnostic and reset probes scored **33 targets total**. The
  97-token reset probe crosses the 64-token batch boundary. A–other–A scores and
  scores in a fresh worker were exactly equal.
- A short nonthinking instruction generated `tree` followed by token 248046.
  The explicit EOS and forced-first-token stopping checks passed. One simple
  authored choice also passed. These are deliberately trivial software probes.
- The whole preflight took **30.46 seconds**, including input hashing and two
  process starts. This is not generation throughput or a runtime benchmark;
  peak process memory was not measured.

The 14-target prose subcheck has mean token NLL 3.296663876957496 under the new FP64
reduction protocol. It is retained only for reproducibility. It uses different
text and probability reduction from the earlier 9B development experiment and
must not be compared with that experiment's NLL.

Two additional runtime details matter for future comparisons:

- This official runtime reports native EOG IDs 248044, 248046, 248063, 248064
  and 248065. The latter three are `<|fim_pad|>`, `<|repo_name|>` and
  `<|file_sep|>` in the reference tokenizer. The evaluator continues to use its
  explicitly frozen EOS pair 248044/248046; it does not inherit native defaults.
- PTQ1_0 CPU matmuls use Q8_0 activation operands; PQ2_0 uses Q8_K. See the
  [pinned CPU type traits](https://github.com/PrismML-Eng/llama.cpp/blob/c9c34bc9c4042c13a31598ebf43461cd0ad8f00d/ggml/src/ggml-cpu/ggml-cpu.c#L240).
  F32 logits and cache do not imply all-F32 arithmetic. Identical ternary weight
  values do not guarantee identical scores across packing kernels or CPU/CUDA.

## Reproduce

Use locally materialized files and a clean checkout of Prism commit
`c9c34bc9c4042c13a31598ebf43461cd0ad8f00d`. Build the worker as documented in
[matched evaluation](matched-evaluation.md). Record your executable SHA256 from
its build receipt and the tokenizer-directory identity from independently
verified tokenizer assets. Different legitimate builds need not have the same
executable hash. Both commands refuse to overwrite existing results.

```bash
mkdir -p runs
python tools/inspect_bonsai_reference.py \
  --gguf /path/to/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  --prism-source /path/to/pinned-prism \
  --output runs/bonsai-inventory.json

python tools/preflight_bonsai_reference.py \
  --gguf /path/to/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  --worker /path/to/prism-eval-worker \
  --expected-worker-sha256 YOUR_BUILD_RECEIPT_SHA256 \
  --prism-source /path/to/pinned-prism \
  --tokenizer-dir /path/to/pinned-qwen-tokenizer \
  --expected-tokenizer-sha256 YOUR_TOKENIZER_DIRECTORY_SHA256 \
  --output runs/bonsai-cpu-preflight
```

The preflight saves frozen inputs before scoring, per-call progress/results,
separate process logs, final scores and a failure/success report. It requires
fewer than 100 scored targets, at most 17 generated tokens including the
optional stop-rule probe, and bounded request timeouts. It downloads nothing.
The separate inventory command executes no model.

[Validation record](../validation-bonsai-reference.json) pins the source files,
runtime, tokenizer, input manifest and results. Windows/CUDA preflight and a
same-input, same-protocol BF16/OpenBonsai/Bonsai comparison remain outstanding.
Keep quality claims separate from successful loading and deterministic scoring.
