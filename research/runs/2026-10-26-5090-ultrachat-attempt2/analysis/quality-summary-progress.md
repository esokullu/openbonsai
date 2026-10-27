# Paired UltraChat coverage diagnostic quality

Status: **progress_only**. Repeated diagnostic/development subset: 2048 NLL tokens, 20 ARC-Easy questions and four generation prompts. Not the original full 16384/100/10 suite or fresh confirmation.

Both arms use the same rotated quantizer and128 updates/window. Differences compare128 distinct UltraChat samples once against a frozen nested16 repeated eight times; active-token exposure and order need separate reporting, not an equal-token causal claim.

Not performed by this utility. Repetition, whitespace and EOS/cap statistics do not establish intelligibility, task relevance or preserved quality.

Historical BF16 results from the same frozen inputs and scoring helpers. Current-run original-model logit parity is recorded separately by the runner; runtime is historical.

All differences are signed absolute differences (row minus BF16); ARC changes use percentage points.

| Evaluation | NLL | PPL | ARC correct | ARC % | Δ NLL | Δ PPL | Δ ARC pp | Unfinished |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original BF16 (historical) | 2.423777 | 11.288 | 19/20 | 95.0 | +0.000000 | +0.000 | +0.0 | 2/4 |
| rotated16x8: Hard FP32 factors | 8.499410 | 4,911.869 | 8/20 | 40.0 | +6.075633 | +4900.581 | -55.0 | 0/4 |
| rotated16x8: FP16 scales, same wrapper GEMM | 8.500027 | 4,914.900 | 8/20 | 40.0 | +6.076250 | +4903.611 | -55.0 | 0/4 |
| rotated16x8: Packed, fresh reload, decoded cache | 8.498620 | 4,907.991 | 8/20 | 40.0 | +6.074843 | +4896.703 | -55.0 | 0/4 |
| rotated128x1: Hard FP32 factors | 9.121631 | 9,151.112 | 8/20 | 40.0 | +6.697854 | +9139.824 | -55.0 | 0/4 |
| rotated128x1: FP16 scales, same wrapper GEMM | 9.121127 | 9,146.503 | 8/20 | 40.0 | +6.697350 | +9135.214 | -55.0 | 0/4 |

Pending/incomplete stages (no final quality numbers inferred):

- rotated128x1/packed: missing

## Direct paired and representation differences

| Left minus right | Δ NLL | Δ PPL | Δ ARC pp |
|---|---:|---:|---:|
| rotated128x1: Hard FP32 factors minus rotated16x8: Hard FP32 factors | +0.622221 | +4239.243 | +0.0 |
| rotated128x1: FP16 scales, same wrapper GEMM minus rotated16x8: FP16 scales, same wrapper GEMM | +0.621100 | +4231.603 | +0.0 |
| rotated16x8: FP16 scales, same wrapper GEMM minus rotated16x8: Hard FP32 factors | +0.000617 | +3.031 | +0.0 |
| rotated16x8: Packed, fresh reload, decoded cache minus rotated16x8: FP16 scales, same wrapper GEMM | -0.001407 | -6.908 | +0.0 |
| rotated128x1: FP16 scales, same wrapper GEMM minus rotated128x1: Hard FP32 factors | -0.000504 | -4.609 | +0.0 |

## Evaluation runtime and memory

Seconds cover already-resident evaluation. Loading, conversion, training, export and decoded-cache preparation are excluded. Peaks are absolute PyTorch allocated/reserved bytes, reset per evaluation phase. Research-wrapper and deployment-cache memory scopes differ; these timings do not establish a speedup over BF16.

| Evaluation | Seconds | Peak allocated GiB | Peak reserved GiB | Generated tokens | EOS / cap |
|---|---:|---:|---:|---:|---:|
| Original BF16 (historical) | 34.278 | 17.954 | 18.191 | 341 | 2 / 2 |
| rotated16x8: Hard FP32 factors | 37.240 | 19.599 | 19.869 | 7 | 4 / 0 |
| rotated16x8: FP16 scales, same wrapper GEMM | 37.677 | 19.599 | 19.869 | 7 | 4 / 0 |
| rotated16x8: Packed, fresh reload, decoded cache | 21.618 | 19.459 | 19.826 | 7 | 4 / 0 |
| rotated128x1: Hard FP32 factors | 34.019 | 19.599 | 19.869 | 4 | 4 / 0 |
| rotated128x1: FP16 scales, same wrapper GEMM | 34.256 | 19.599 | 19.869 | 4 | 4 / 0 |

| Evaluation phase | Seconds | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|
| Original BF16 (historical) / benchmark | 1.725 | 17.625 | 17.656 |
| Original BF16 (historical) / generation | 30.930 | 17.626 | 17.658 |
| Original BF16 (historical) / nll | 1.426 | 17.954 | 18.191 |
| rotated16x8: Hard FP32 factors / benchmark | 23.742 | 19.495 | 19.602 |
| rotated16x8: Hard FP32 factors / generation | 8.290 | 19.485 | 19.602 |
| rotated16x8: Hard FP32 factors / nll | 4.973 | 19.599 | 19.869 |
| rotated16x8: FP16 scales, same wrapper GEMM / benchmark | 24.132 | 19.495 | 19.602 |
| rotated16x8: FP16 scales, same wrapper GEMM / generation | 8.390 | 19.485 | 19.602 |
| rotated16x8: FP16 scales, same wrapper GEMM / nll | 5.067 | 19.599 | 19.869 |
| rotated16x8: Packed, fresh reload, decoded cache / benchmark | 14.049 | 19.132 | 19.248 |
| rotated16x8: Packed, fresh reload, decoded cache / generation | 4.712 | 19.098 | 19.211 |
| rotated16x8: Packed, fresh reload, decoded cache / nll | 2.782 | 19.459 | 19.826 |
| rotated128x1: Hard FP32 factors / benchmark | 23.925 | 19.495 | 19.602 |
| rotated128x1: Hard FP32 factors / generation | 4.953 | 19.484 | 19.602 |
| rotated128x1: Hard FP32 factors / nll | 5.028 | 19.599 | 19.869 |
| rotated128x1: FP16 scales, same wrapper GEMM / benchmark | 24.156 | 19.495 | 19.602 |
| rotated128x1: FP16 scales, same wrapper GEMM / generation | 4.943 | 19.484 | 19.602 |
| rotated128x1: FP16 scales, same wrapper GEMM / nll | 5.095 | 19.599 | 19.869 |

Memory scopes:

- Original BF16 (historical): Historical untouched BF16 resident model; model loading excluded.
- Hard FP32 factors: Research wrappers retain original BF16 weights and learned FP32 factors/A/B; evaluation only.
- FP16 scales, same wrapper GEMM: Research wrappers with FP16-rounded scales/bias and unchanged code selection/GEMM; evaluation only.
- Packed, fresh reload, decoded cache: Fresh packed deployment model with retained BF16 tensors and decoded BF16 weight cache; evaluation only, loading/cache warmup excluded.

## Automated generation checks

Terminal EOS is excluded from repetition counts. These are mechanical observations; a nonblank answer is not evidence of usefulness. Original prompts, unaltered recorded text, full generated IDs and per-output flags are in `quality-generations.md` (or its progress counterpart) and the JSON.

| Evaluation | Single-token repeated outputs | Empty text | Whitespace-only text | Longest identical-token runs |
|---|---:|---:|---:|---|
| Original BF16 (historical) | 0/4 | 0/4 | 0/4 | 1, 1, 1, 1 |
| rotated16x8: Hard FP32 factors | 0/4 | 2/4 | 0/4 | 0, 0, 1, 1 |
| rotated16x8: FP16 scales, same wrapper GEMM | 0/4 | 2/4 | 0/4 | 0, 0, 1, 1 |
| rotated16x8: Packed, fresh reload, decoded cache | 0/4 | 2/4 | 0/4 | 0, 0, 1, 1 |
| rotated128x1: Hard FP32 factors | 0/4 | 4/4 | 0/4 | 0, 0, 0, 0 |
| rotated128x1: FP16 scales, same wrapper GEMM | 0/4 | 4/4 | 0/4 | 0, 0, 0, 0 |

## Frozen protocol

Input SHA-256: `330ffeeb77fa384492a6494e1e698edd400d7088456dbd0b7bc0fd366b7d838a`.

Model: `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

```json
{
  "do_sample": false,
  "enable_thinking": false,
  "eos_token_ids": [
    248044,
    248046
  ],
  "max_new_tokens": 128,
  "pad_token_id": 248044,
  "seed": 5090,
  "use_cache": false
}
```

NLL is shifted, masks both source and target positions and is aggregated by scored-token count. ARC selects the maximum conditional answer-label log likelihood; it does not generate answers. No retokenization or rescoring occurs in this utility. Raw recorded text is retained, but this stdlib utility does not independently decode token IDs.
