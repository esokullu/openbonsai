# Lossless base-3 packing

This codec changes the storage of **already learned ternary values**. It does
not quantize BF16 weights, retrain scales, change rotations or recover quality.
The research priority remains broad ternary coverage with useful quality;
passing codec checks alone does not meet that goal.

## Why add another codec?

BITCOS uses a presence bit for every weight and a sign bit for every nonzero
weight. Its approximate code cost is `2 - zero_fraction` bits per weight.
The [official checkpoint audit](bonsai-ternary-statistics.md) measured 32.75744%
zeros in Bonsai's 202 ternary matrices. The previously completed OpenBonsai
rotated decoder experiment had about 30.966% zeros. At these densities, five
ternary codes per byte can use less space than BITCOS. These statistics do not
explain or solve the quality gap.

Base-3 costs about 1.6 code bits per weight. At group size 128, FP16 scales add
0.125 bits, giving about 1.725 bits before headers, signs and other tensors.
Ignoring those overheads, BITCOS wins above roughly 40% zeros. Automatic mixed
export uses actual complete candidate bytes rather than this approximation.

## Byte layout

`openbonsai.ternary-base3.v1` maps zero to digit 0, +1 to digit 1, and -1 to
digit 2. For five consecutive codes, the byte is
`d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4`. Values 243–255 are invalid. The final byte's
unused high digits must be zero. The code stream runs continuously across rows
and scale groups; `ceil(padded_numel / 5)` bytes hold it. Existing final-group
padding remains zero and is validated separately. Scale lookup continues to
use the original group size, independent of byte position.

This is an independent OpenBonsai encoding. Prism PTQ1_0 uses a different byte
mapping and 128-weight block layout. These files are neither GGUF nor compatible
with Prism's kernels.

## Export, loading and repacking

- Legacy `save_compressed`/CLI accepts `auto`, `2bit`, `bitcos`, and `base3`.
  Base-3 modules require the v3 artifact marker. The updated loader retains v1
  and v2 support and rejects a mismatched manifest/module format. Legacy auto
  minimizes code bytes, not complete container bytes; equal costs prefer
  two-bit, then BITCOS, then base-3.
- Mixed candidates use encoding `ternary-base3`. Their auto path compares
  complete serialized files, including scales, bias, explicit rotation signs,
  metadata, checksums and any BITCOS row index. Final export verifies the total
  planned artifact size against actual files, including retained tensors and
  supplied tokenizer assets.
- Head and embedding exporters count bounded chunks first, then allocate only
  the chosen code stream. Base-3 carries at most four cloned trits between chunks;
  no row/group boundary padding is inserted. Existing source, two-pass and
  embedding calibration coverage checks still apply.
- `repack_ternary_candidate` preserves integer codes, stored scale/bias dtype
  and values, rotations and provenance. It does not reconstruct floating weights
  to choose codes. It accepts ternary candidates only. This helper is not a
  whole mixed-artifact migration command.

Reference runtime decodes floating-point weights. Selected embedding rows and
projection chunks can cross byte boundaries without decoding the whole table.
No optimized ternary kernel, GPU throughput improvement or runtime-memory
reduction is claimed. The first learned FP32-to-FP16 scale export can introduce
rounding; subsequent repacking preserves the stored values exactly.

## Measured validation

Run the offline native fixture with a new/empty output directory:

```sh
python tools/probe_base3_packing.py --output runs/base3-native-probe
```

The fixture starts from one random BF16 native four-block Qwen model. It fits
25 rotated decoder projections once, then includes rotated embedding/head
values for 27-module codec coverage. **Those random endpoints are uncalibrated**;
this is a software test, not a language-quality experiment. All four exports
use the same integer values, scales, signs, retained weights and export plan.

| Codec | Candidate files | Fixed files | Complete artifact |
|---|---:|---:|---:|
| Two-bit | 214,409 B | 47,146 B | 261,555 B |
| BITCOS | 231,807 B | 47,146 B | 278,953 B |
| Base-3 | 178,401 B | 47,146 B | 225,547 B |
| Auto | 178,401 B | 47,146 B | 225,547 B |

Every artifact's actual size equals its computed budget. Repacking back to
two-bit recovers every code and metadata field exactly. All four fresh-process
loads produce identical logits, and the original BF16 source remains unchanged.
The complete base-3 fixture is 13.77% smaller than two-bit and 19.15% smaller than
BITCOS; small-row index overhead makes these percentages **not a 9B forecast**.

The full CPU suite passes **1,381 tests**. New checks cover all 243 valid byte
patterns, nonaligned rows/groups/chunks, corrupt values/padding/lengths, mixed
formats, bounded embedding lookup, input gradients, cached/uncached execution,
stored bias dtypes and exact reload. [Validation record](../validation-base3.json).

There is no new real-9B base-3 artifact or pretrained quality result in this
stage. The active 5090 diversity comparison retains its frozen code and codecs.
The official checkpoint's alternative packing costs are hypothetical payload
calculations, kept separate from the actual fixture files above.
