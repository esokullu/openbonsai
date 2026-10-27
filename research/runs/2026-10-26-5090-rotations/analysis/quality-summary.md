# Paired rotation diagnostic quality

Status: **completed**. Repeated diagnostic/development subset: 2048 NLL tokens, 20 ARC-Easy questions and four generation prompts. Not the original full 16384/100/10 suite or fresh confirmation.

Not performed by this utility. Repetition, whitespace and EOS/cap statistics do not establish intelligibility, task relevance or preserved quality.

Historical BF16 results from the same frozen inputs and scoring helpers. Current-run original-model logit parity is recorded separately by the runner; runtime is historical.

All differences are signed absolute differences (row minus BF16); ARC changes use percentage points.

| Evaluation | NLL | PPL | ARC correct | ARC % | Δ NLL | Δ PPL | Δ ARC pp | Unfinished |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original BF16 (historical) | 2.423777 | 11.288 | 19/20 | 95.0 | +0.000000 | +0.000 | +0.0 | 2/4 |
| unrotated16x8: Hard FP32 factors | 14.439949 | 1,867,197.748 | 1/20 | 5.0 | +12.016173 | +1867186.460 | -90.0 | 4/4 |
| unrotated16x8: FP16 scales, same wrapper GEMM | 14.438327 | 1,864,170.316 | 1/20 | 5.0 | +12.014550 | +1864159.027 | -90.0 | 4/4 |
| unrotated16x8: Packed, fresh reload, decoded cache | 14.437115 | 1,861,912.383 | 1/20 | 5.0 | +12.013338 | +1861901.095 | -90.0 | 4/4 |
| rotated16x8: Hard FP32 factors | 9.257334 | 10,481.154 | 8/20 | 40.0 | +6.833557 | +10469.866 | -55.0 | 4/4 |
| rotated16x8: FP16 scales, same wrapper GEMM | 9.253318 | 10,439.146 | 8/20 | 40.0 | +6.829541 | +10427.858 | -55.0 | 4/4 |
| rotated16x8: Packed, fresh reload, decoded cache | 9.255346 | 10,460.334 | 8/20 | 40.0 | +6.831569 | +10449.045 | -55.0 | 4/4 |

## Direct paired and representation differences

| Left minus right | Δ NLL | Δ PPL | Δ ARC pp |
|---|---:|---:|---:|
| rotated16x8: Hard FP32 factors minus unrotated16x8: Hard FP32 factors | -5.182615 | -1856716.594 | +35.0 |
| rotated16x8: FP16 scales, same wrapper GEMM minus unrotated16x8: FP16 scales, same wrapper GEMM | -5.185009 | -1853731.169 | +35.0 |
| rotated16x8: Packed, fresh reload, decoded cache minus unrotated16x8: Packed, fresh reload, decoded cache | -5.181769 | -1851452.049 | +35.0 |
| unrotated16x8: FP16 scales, same wrapper GEMM minus unrotated16x8: Hard FP32 factors | -0.001623 | -3027.433 | +0.0 |
| unrotated16x8: Packed, fresh reload, decoded cache minus unrotated16x8: FP16 scales, same wrapper GEMM | -0.001212 | -2257.933 | +0.0 |
| rotated16x8: FP16 scales, same wrapper GEMM minus rotated16x8: Hard FP32 factors | -0.004016 | -42.008 | +0.0 |
| rotated16x8: Packed, fresh reload, decoded cache minus rotated16x8: FP16 scales, same wrapper GEMM | +0.002028 | +21.188 | +0.0 |

## Evaluation runtime and memory

Seconds cover already-resident evaluation. Loading, conversion, training, export and decoded-cache preparation are excluded. Peaks are absolute PyTorch allocated/reserved bytes, reset per evaluation phase. Research-wrapper and deployment-cache memory scopes differ; these timings do not establish a speedup over BF16.

| Evaluation | Seconds | Peak allocated GiB | Peak reserved GiB | Generated tokens | EOS / cap |
|---|---:|---:|---:|---:|---:|
| Original BF16 (historical) | 34.278 | 17.954 | 18.191 | 341 | 2 / 2 |
| unrotated16x8: Hard FP32 factors | 213.287 | 19.598 | 19.875 | 512 | 0 / 4 |
| unrotated16x8: FP16 scales, same wrapper GEMM | 217.598 | 19.598 | 19.875 | 512 | 0 / 4 |
| unrotated16x8: Packed, fresh reload, decoded cache | 284.163 | 19.454 | 19.750 | 512 | 0 / 4 |
| rotated16x8: Hard FP32 factors | 641.245 | 19.599 | 19.869 | 512 | 0 / 4 |
| rotated16x8: FP16 scales, same wrapper GEMM | 643.455 | 19.599 | 19.869 | 512 | 0 / 4 |
| rotated16x8: Packed, fresh reload, decoded cache | 340.528 | 19.457 | 19.846 | 512 | 0 / 4 |

| Evaluation phase | Seconds | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|
| Original BF16 (historical) / benchmark | 1.725 | 17.625 | 17.656 |
| Original BF16 (historical) / generation | 30.930 | 17.626 | 17.658 |
| Original BF16 (historical) / nll | 1.426 | 17.954 | 18.191 |
| unrotated16x8: Hard FP32 factors / benchmark | 7.909 | 19.489 | 19.648 |
| unrotated16x8: Hard FP32 factors / generation | 203.440 | 19.490 | 19.650 |
| unrotated16x8: Hard FP32 factors / nll | 1.853 | 19.598 | 19.875 |
| unrotated16x8: FP16 scales, same wrapper GEMM / benchmark | 8.185 | 19.489 | 19.648 |
| unrotated16x8: FP16 scales, same wrapper GEMM / generation | 207.499 | 19.490 | 19.650 |
| unrotated16x8: FP16 scales, same wrapper GEMM / nll | 1.823 | 19.598 | 19.875 |
| unrotated16x8: Packed, fresh reload, decoded cache / benchmark | 10.533 | 19.128 | 19.229 |
| unrotated16x8: Packed, fresh reload, decoded cache / generation | 271.109 | 19.130 | 19.230 |
| unrotated16x8: Packed, fresh reload, decoded cache / nll | 2.461 | 19.454 | 19.750 |
| rotated16x8: Hard FP32 factors / benchmark | 23.794 | 19.495 | 19.602 |
| rotated16x8: Hard FP32 factors / generation | 612.239 | 19.496 | 19.604 |
| rotated16x8: Hard FP32 factors / nll | 5.125 | 19.599 | 19.869 |
| rotated16x8: FP16 scales, same wrapper GEMM / benchmark | 25.137 | 19.495 | 19.602 |
| rotated16x8: FP16 scales, same wrapper GEMM / generation | 613.237 | 19.496 | 19.604 |
| rotated16x8: FP16 scales, same wrapper GEMM / nll | 4.968 | 19.599 | 19.869 |
| rotated16x8: Packed, fresh reload, decoded cache / benchmark | 12.626 | 19.133 | 19.248 |
| rotated16x8: Packed, fresh reload, decoded cache / generation | 324.873 | 19.134 | 19.270 |
| rotated16x8: Packed, fresh reload, decoded cache / nll | 2.973 | 19.457 | 19.846 |

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
| unrotated16x8: Hard FP32 factors | 4/4 | 0/4 | 0/4 | 128, 128, 128, 128 |
| unrotated16x8: FP16 scales, same wrapper GEMM | 4/4 | 0/4 | 0/4 | 128, 128, 128, 128 |
| unrotated16x8: Packed, fresh reload, decoded cache | 4/4 | 0/4 | 0/4 | 128, 128, 128, 128 |
| rotated16x8: Hard FP32 factors | 0/4 | 0/4 | 0/4 | 1, 1, 1, 1 |
| rotated16x8: FP16 scales, same wrapper GEMM | 0/4 | 0/4 | 0/4 | 1, 1, 1, 1 |
| rotated16x8: Packed, fresh reload, decoded cache | 0/4 | 0/4 | 0/4 | 1, 1, 1, 1 |

## Historical corrected16x1 context

One epoch, exp mapping, MSE, factor LR 0.001; current pair uses eight epochs, sigmoid, Huber and factor LR 0.005. Historical packed evaluation did not use the current decoded-weight cache. Differences cannot be attributed to rotation.

Historical packed NLL 11.191338, perplexity 72,499.733, ARC 1/20, unfinished 4/4. This is context from a different recipe, not a matched rotation control.

Source: `D:\openbonsai\runs\2026-10-26-5090-corrected\corrected16x1\packed-evaluation.json`.

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
