Attempt 2 accounting adds the prior failed preparation process exactly once: **5.924203 seconds**. The copied reporting tools remain unchanged.

Generation plus both arm drivers, including the parent debit: **9715.833690 seconds**. Including both resource preflights: **10098.334384 seconds**. Nested phase timings are excluded from these sums.

| Bytes | Rotated 16 × 8 | Rotated 128 × 1 |
|---|---:|---:|
| Model excluding tokenizer | 6,576,785,938 | 6,576,831,354 |
| Tokenizer assets | 19,998,761 | 19,998,761 |
| Complete deployable directory | 6,596,784,699 | 6,596,830,115 |
| Candidate presence | 864,026,624 | 864,026,624 |
| Candidate rotation_signs | 1,081,344 | 1,081,344 |
| Candidate row_sign_offsets | 11,404,864 | 11,404,864 |
| Candidate scales | 108,003,328 | 108,003,328 |
| Candidate signs | 596,482,310 | 596,527,725 |
| Retained linear_attention_convolutions | 1,572,864 | 1,572,864 |
| Retained linear_attention_dynamics | 3,072 | 3,072 |
| Retained linear_attention_gates | 12,582,912 | 12,582,912 |
| Retained output_head | 2,034,237,440 | 2,034,237,440 |
| Retained text_normalization | 546,816 | 546,816 |
| Retained text_rotary_buffers | 256 | 256 |
| Retained token_embeddings | 2,034,237,440 | 2,034,237,440 |
| Retained vision_encoder_including_vision_rotary_buffers | 912,021,104 | 912,021,104 |
| Decoded BF16 cache (runtime only) | 13,824,425,984 | 13,824,425,984 |
| FP32 research file (outside deployment) | 1,291,183,353 | 1,291,183,353 |

No separate endpoint prototype and no endpoint quantization. Retained embedding and output-head tensors remain fully included in deployment bytes; the zero endpoint fields refer only to additional prototype/quantized parameters.

Strict equivalent-source comparisons use 18,819,635,168 raw checkpoint bytes or 18,819,627,488 BF16-loaded persistent bytes. The 48 retained FP32→BF16 casts explain the 7,680-byte difference. Unused MTP tensors (486,581,248 bytes) are excluded from both denominators. Generated rotary buffers (400 bytes) are excluded from strict tensor numerators. Candidate signs, scales and indexes remain included. Complete deployment also includes config/container overhead and tokenizer assets; it is not the strict tensor numerator.

Do not add parent duration to the recorded cumulative timer. Its final sample occurs after output hashing; process accounting below uses outer command timers. Deadline checks occur between token steps/records, not within a CUDA call or loading operation.

Shared generation counted once. These sums exclude CPU preparation/tests/final reporting and historical quality runs; nested trainer/evaluation phase durations are not added again. They are sums of measured processes, not complete human elapsed time.
