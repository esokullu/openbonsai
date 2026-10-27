Small-pilot results: both matched UltraChat rotated arms completed, including exact independent packed reloads. Results on the repeated 2,048-token / 20-question / four-prompt development subset:

| Representation | Equivalent-scope tensor bytes | NLL | Perplexity | ARC | Evaluation peak allocated / reserved GiB | Evaluation seconds | Generated tokens |
|---|---|---|---|---|---|---|---|
| Original BF16 (historical) | 18,819,627,488 | 2.423777 | 11.288 | 19/20 | 17.954 / 18.191 | 34.278 | 341 |
| rotated16x8 (fresh packed) | 6,576,199,974 | 8.498620 | 4907.991 | 8/20 | 19.459 / 19.826 | 21.618 | 7 |
| rotated128x1 (fresh packed) | 6,576,245,389 | 9.123808 | 9171.055 | 8/20 | 19.460 / 19.846 | 18.230 | 4 |

Tensor bytes use equivalent represented scope; the baseline is loaded persistent tensors, not an original deployment folder. Cached packed evaluation retains decoded BF16 weights. Different generated-token counts preclude an equal-work speed claim.

rotated16x8: No quality recovery:0/4 complete useful answers; two empty EOS-only replies and two nonanswers (Write a and #), despite4/4 EOS endings.

rotated128x1: No quality recovery:0/4 complete useful answers; all four prompts receive immediate EOS-only empty replies.

| Representation | Predicted letters | Answer keys |
|---|---|---|
| Original BF16 (historical) | A=7, B=7, C=4, D=2 | A=8, B=7, C=4, D=1 |
| rotated16x8: Packed, fresh reload, decoded cache | A=20, B=0, C=0, D=0 | A=8, B=7, C=4, D=1 |
| rotated128x1: Packed, fresh reload, decoded cache | A=20, B=0, C=0, D=0 | A=8, B=7, C=4, D=1 |

rotated16x8 predicts A on all 20 questions: its 8/20 equals the 8 A-labeled answer keys. This is an answer-letter collapse, not broad retained QA competence. rotated128x1 predicts A on all 20 questions: its 8/20 equals the 8 A-labeled answer keys. This is an answer-letter collapse, not broad retained QA competence.

| Arm | Model bytes | Tokenizer bytes | Full deployable bytes | Strict BF16 tensor reduction |
|---|---|---|---|---|
| rotated16x8 | 6,576,785,938 | 19,998,761 | 6,596,784,699 | 65.056694% |
| rotated128x1 | 6,576,831,354 | 19,998,761 | 6,596,830,115 | 65.056453% |

Both retain embeddings, output head and vision. No endpoint prototype is included. The 13,824,425,984-byte decoded BF16 cache is runtime-only; FP32 research files are outside deployment. Exact serialization/reload does not establish useful model quality or a speedup.

Calibration exposure per window was 59,856 versus 59,855 active positions. Initial factors/signs match; both have 4,096 updates. The only attempt-2 runtime correction concerns metadata validation and budget/path wiring, not training math. The earlier 5.924203-second failure is counted once.

Measured generation, both preflights and four arm/reload processes plus that prior debit total 2.805 hours; CPU analysis and human gaps are excluded. Generation lengths differ, so evaluation durations are not equal-token speed comparisons.

[Full report](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/report.md>) · [All generation texts](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/quality-generations.md>) · [Manual review](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/analysis/manual-quality-review.md>) · [Reproduction and hashes](<D:/openbonsai/runs/2026-10-26-5090-ultrachat-attempt2/reproduction.md>)
