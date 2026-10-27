# Official Bonsai ternary storage audit

`tools/audit_bonsai_ternary.py` scans the official 9B PTQ1_0 checkpoint without
constructing a model or running inference. It measures every tensor's code
occupancy, exact stored FP16-scale distribution, and hypothetical alternative
packing costs. It does not evaluate quality or reconstruct Prism's training
method. No checkpoint, downloaded dependency or private experiment log is
included in this repository.

## Inputs and reproducibility

The tool requires explicitly supplied **local** files and downloads nothing:

- `Ternary-Bonsai-2-9B-PTQ1_0.gguf`, HF revision
  `5b2abb02362efbed46bbe204014b02837778b2c2`, 1,997,486,976 bytes, SHA256
  `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`.
- Clean Prism llama.cpp checkout at
  `c9c34bc9c4042c13a31598ebf43461cd0ad8f00d` and its CPU `ggml-base` build.
- The small native decoder wrapper built below.

It verifies the full checkpoint hash, source revision and clean checkout before
scanning; records script/source/native-library hashes; and refuses to overwrite
an output. Optional `--prior-evidence` accepts the project's historical completed
command-output JSON for a separate old-run comparison. Omit it for the complete
public-checkpoint audit; the result then contains a null historical comparison.

Python with NumPy, Git, a C++17 compiler and the pinned source's `gguf-py` are
required. The script supports little-endian **macOS and Linux**, limits numeric
CPU threads to two, and has no GPU code. macOS execution was verified locally;
the Linux build commands are provided but have not been executed here. Windows
is not supported by this audit helper.

If the local CPU runtime is not already built, the repository's existing
`tools/build_prism_eval.py --source /path/to/prism --build-dir /path/to/build
--jobs 2` builds it with CUDA and Metal disabled by default. It downloads no
weights or source. Its static archive is under
`/path/to/build/cmake/prism/ggml/src/libggml-base.a`.

Compile the wrapper against that **same pinned build**. On macOS:

```sh
c++ -O2 -std=c++17 -dynamiclib tools/bonsai_ternary_decoder.cpp \
  -I/path/to/prism/ggml/include -I/path/to/prism/ggml/src \
  /path/to/build/cmake/prism/ggml/src/libggml-base.a \
  -lm -framework Accelerate -o /path/to/bonsai_ternary_decoder.dylib
```

On Linux, use a position-independent CPU static build (`CMAKE_POSITION_INDEPENDENT_CODE=ON`):

```sh
c++ -O2 -std=c++17 -shared -fPIC tools/bonsai_ternary_decoder.cpp \
  -I/path/to/prism/ggml/include -I/path/to/prism/ggml/src \
  /path/to/build/cmake/prism/ggml/src/libggml-base.a \
  -pthread -ldl -lm -o /path/to/bonsai_ternary_decoder.so
```

Run with a new output filename in an existing directory:

```sh
python tools/audit_bonsai_ternary.py \
  --model /path/to/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
  --prism-source /path/to/prism \
  --native-library /path/to/bonsai_ternary_decoder.dylib \
  --output /path/to/new-statistics.json
```

The reader memory-maps the file. Analysis uses at most 16,384 groups per chunk
by default (458,752 packed bytes), small byte lookup tables and 65,536-bin scale
histograms. It does not retain a full decoded matrix. Process RSS still includes
resident mapped pages and tokenizer metadata; this is not a sub-megabyte RAM
claim. `--chunk-blocks` permits 1–32,768 groups.

## Format verification and measured findings

Prism's [PTQ1_0 struct](https://github.com/PrismML-Eng/llama.cpp/blob/c9c34bc9c4042c13a31598ebf43461cd0ad8f00d/ggml/src/ggml-common.h#L209)
stores 24 five-trit bytes, two four-trit bytes and one FP16 scale per 128
weights: 1.625 code bits plus 0.125 scale bits per weight. The last eight trits
are real weights, not padding. The [native decoder](https://github.com/PrismML-Eng/llama.cpp/blob/c9c34bc9c4042c13a31598ebf43461cd0ad8f00d/ggml/src/ggml-quants.c#L2255)
performs uint8 overflow before extracting each base-3 digit. Zero is the middle
digit; an all-zero positive-scale block has `qs=128`, `qh=127`.

Validation checks 260 native-encoded known-pattern blocks, every possible byte
value, a positive-scale all-zero block, and 825 actual blocks across every
tensor, including zero-count extrema and subnormal-scale examples. Native
weights and zero counts match the independent lookup decoder exactly.

The pinned checkpoint contains 202 ternary matrices, including embedding/head,
with **8,946,450,432 weights and 32.75744% zero codes**. All scales are positive;
no zero-scale groups inflate that figure. Global scale median is 0.01422882,
p01 0.00861359 and p99 0.02352905. The 89 positive subnormal scales belong to the
embedding. Every matrix divides into full 128-weight groups.

**96.65090% of groups contain exactly 42 zeros; none has more than 42.** This is
an observed checkpoint pattern, not a packing requirement or proof of a top-k
training rule. The native codec can represent all-zero groups. No such training
constraint was found in the inspected [small-model whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-small-whitepaper.pdf),
[27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
or [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf).

## Actual versus hypothetical size

These payload calculations preserve every code and original FP16 group scale:

| Packing | Code + scale bytes | Additional row-index bytes |
|---|---:|---:|
| Actual PTQ1_0 | 1,957,036,032 | 0 |
| Hypothetical BITCOS | 2,010,072,449 | 15,378,000 with OpenBonsai indexes |
| Hypothetical flat five trits/byte | 1,929,078,480 | 0 |

Flat-five uses exactly `sum(ceil(tensor.numel/5)) = 1,789,290,192` code bytes,
with only final-tensor padding, independent of scale-group boundaries. It saves
**27,957,552 bytes versus PTQ1**, **80,993,969 versus index-free BITCOS**, or
**96,371,969 versus indexed BITCOS** on these official tensors. No matrix has
enough zeros for BITCOS to beat its existing PTQ1 code payload.

Replacing only those payloads while holding the existing 29,411,328 other
tensor bytes and 11,039,616 metadata/alignment bytes fixed yields a hypothetical
**1,969,529,424-byte flat-five file**. This is not a written GGUF or OpenBonsai
artifact: new codec headers, runtime compatibility, speed and resident memory
are separate questions. These are language-only counts; vision and optional
MTP weights are absent. Packing statistics do not establish language quality.
