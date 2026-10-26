# Sequential original-target result: alignment improved, quality did not

The 26 October 2026 RTX 5090 run completed all 4,096 calibration updates, export and independent reload. **Useful language quality did not recover.** Original-prefix teacher targets reduced measured activation error, but worsened NLL against the historical conditional-prefix control. Serialization passed. The OpenBonsai quality and approximately 2 GB full-language deployment goals remain unmet.

## Controlled change

The source was untouched [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a), not a fine-tune. Both recipes used 128 frozen calibration records, one epoch per window, sequence length 512, group 128, block-4096 signed Hadamard rotations, rank-64 corrections, sigmoid factors and Huber loss. All 200 large decoder projections were selected; embedding, head, small dynamics and vision stayed floating point.

The changed variable was the [teacher target](teacher-target-control.md). Previously, the original window received the compressed prefix's input. Here, an independent original-weight prefix produced the original window target, while the student still received its current compressed-prefix input. Fitting started fresh from BF16. Window 0 reproduced the parent's learned state and native boundary exactly; source and bypass checks passed. Historical results were reused after identity gates, not rerun as a new paired arm.

Each window completed 102 soft and 26 hard updates: 3,264 soft plus 832 hard overall. Recorded gradient/update evidence was finite; the saved Jacobian matched the actual last-soft pre-update point. Four preflight updates were discarded with exact rollback. Lower cached hard MSE in all 32 windows did not establish useful quality.

## Development results

The unchanged legacy evaluator used FP32 cross-entropy/log-softmax on 2,048 next-token targets, 20 ARC-Easy option-letter questions and four greedy prompts capped at 128 tokens. These repeatedly inspected development inputs are not unseen confirmation data.

| Representation | Mean NLL | Perplexity | ARC |
|---|---:|---:|---:|
| Original BF16, reused | 2.423776811 | 11.29 | 19/20 |
| Conditional128 hard, reused | 9.121630685 | 9,151.11 | 8/20 |
| Conditional128 packed, reused | 9.123807641 | 9,171.06 | 8/20 |
| Original-prefix hard | 11.827691902 | 136,993.93 | 8/20 |
| Original-prefix fresh packed | 11.849104019 | 139,958.89 | 8/20 |

Both new representations selected A for every question. All four answers were repetitive task failures: no requested kitchen tips, animal-count solution, Python function or Turkish rewrite. Each reached 128 tokens without EOS. The parent's four empty EOS-only outputs and the new nonempty repetitions are both failures.

On 16 separate search records with identical original targets, Y/T0 MSE changed as follows:

| Boundary | Conditional128 | Original-prefix | Reduction |
|---|---:|---:|---:|
| 0 | 0.000311311 | 0.000311311 | 0% |
| 16 | 0.196574567 | 0.104790041 | 46.692% |
| 31 | 18.120259124 | 6.858338666 | 62.151% |

Every record improved at boundaries 16 and 31. Final-boundary cosine improved from 0.2014 to 0.5474; the best scalar rescaling reduced remaining MSE by only 0.0251%. Improved alignment and worse language quality coexist. These observations establish neither convergence nor a simple amplitude correction.

## Representation, storage and runtime

Fresh reload verified all 200 projection payloads and 564 retained tensors, with **zero fixed-probe logit difference** against the stored-scale packed reference. Cached/uncached packed execution was also exact. FP32-scale→FP16 same-wrapper and FP16-wrapper→packed maximum differences were 0.134765625 and 0.25 respectively. These arithmetic/rounding differences are separate from serialization. A full same-wrapper FP16 quality pass was not measured; its fixed probe cannot apportion the hard-to-packed NLL increase.

All candidates selected BITCOS. Total deployment files occupy **6,597,192,129 bytes**, including **19,998,761 tokenizer bytes**. Candidate tensor payload is 1,581,405,910 bytes; retained tensors are 4,995,201,904 bytes. The latter already include BF16 embedding/head at **4,068,474,880 bytes combined**, vision at **912,021,104 tensor bytes** (912,058,304 file bytes), and 400 generated-buffer bytes. Equivalent persistent tensors total **6,576,607,414 versus 18,819,627,488 original bytes**, a 65.054529% reduction. Containers, configuration and tokenizer are additional. The separate **13,824,425,984-byte decoded runtime cache** is not deployment storage.

Model-process time was **6,577.657 seconds (109.63 minutes)**, within 14,400 seconds. All children exited and were reaped. Peak allocated/reserved VRAM was 21,708,011,520/22,693,281,792 bytes. This includes loading, preflight, fitting, evaluation, export and reload; it is not evidence of ternary inference acceleration.

## Provenance and limits

Run `2026-10-26-5090-original-prefix-sequential-v2` is sealed: SHA256 `66e8712d99046fe80235a56b0a22fee2120fa0579afdc87a2bae7687aa72dc52`, 1,936 inventoried files / 8,111,555,727 archive bytes. The [validation receipt](../validation-sequential-original-target.json) binds executed source, inputs, result and parity receipts. Weights and the full archive are not included here.

The [official Bonsai result](official-bonsai-development-result.md) uses a different FP64 scorer/runtime; its scalar NLL is not a matched retention comparison. Saved update-path analysis remains pending; this report does not authorize a 512-update control or another experiment.
