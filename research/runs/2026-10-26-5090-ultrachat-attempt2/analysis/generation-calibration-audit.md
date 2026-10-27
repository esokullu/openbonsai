# Frozen UltraChat calibration audit

Passed: 128 accepted traces from exactly the 128 frozen primary prompts, in their original order; no reserve used. The source, token prefix, masks, per-row hashes, all 128 trace files, and frozen generation settings match.

The frozen selector chose 16 records with 7,482 active positions. Eight epochs expose 59,856 positions versus 59,855 for 128 records at one epoch: **+1 position (+0.001670704%)**, within the predefined ±5% gate. An independent suffix-feasibility check confirms the exact optimum and lexicographic tie break. The one-position mismatch is the minimum possible integer difference.

| Measure | Nested16, per epoch | All128, per epoch |
|---|---:|---:|
| Samples | 16 | 128 |
| Prompt tokens | 1,204 | 8,815 |
| Generated raw tokens | 7,205 | 58,089 |
| Raw total tokens | 8,409 | 66,904 |
| Consumed active positions | 7,482 | 59,855 |
| Consumed continuation positions | 6,278 | 51,040 |
| Padding positions | 710 | 5,681 |
| Ended with EOS | 3 | 26 |
| Capped/unfinished | 13 | 102 |
| Right-truncated records | 13 | 105 |
| Right-truncated tokens | 927 | 7,049 |
| EOS retained in consumed tokens | 3 | 23 |

| Exposure component | 16×8 | 128×1 |
|---|---:|---:|
| All active positions | 59,856 | 59,855 |
| Prompt positions | 9,632 | 8,815 |
| Continuation positions | 50,224 | 51,040 |
| Padding positions | 5,680 | 5,681 |

| Length statistic | Nested16 | All128 |
|---|---:|---:|
| Prompt: min / median / max | 26 / 85 / 140 | 21 / 71 / 155 |
| Prompt: mean | 75.250000 | 68.867188 |
| Raw generated: min / median / max | 79 / 512 / 512 | 21 / 512 / 512 |
| Raw generated: mean | 450.312500 | 453.820312 |
| Consumed active: min / median / max | 117 / 512 / 512 | 80 / 512 / 512 |
| Consumed active: mean | 467.625000 | 467.617188 |

Generation retained the official BF16 checkpoint/revision and tokenizer: greedy, nonthinking, SDPA, KV cache enabled, seed5090, 512 new-token limit, no prompt truncation, EOS IDs248044/248046. Consumed sequences use right truncation/padding to512, pad248044, with padding excluded by the mask.

Preparation took 2217.662s in this attempt; charging the prior failed process gives 2223.595s, below5,400s. The128 generation phases total2193.709s (26.480 generated tokens/s); model loading took17.424s.

The audit uses IDs, hashes, token counts and masks. It does not judge response semantics or inspect evaluation examples. Raw EOS completion and truncation for training are distinct. Matching total active positions does not exactly match the prompt/continuation split and does not establish model quality.

Full per-record statistics, histograms, source/settings hashes, trace hashes and memory measurements are saved in generation-calibration-audit.json.
