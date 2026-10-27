The rotated16x8 artifact audit passed after both main and fresh-reload processes exited successfully. All 769 recorded payload file hashes, 200 candidate internal checksums, 560 persistent retained value hashes and 200 explicit rotation-sign arrays were verified. Fresh-reload and cached/uncached probes are bitwise exact.

| Component | Bytes |
|---|---:|
| Model excluding tokenizer | 6,576,785,938 |
| Supplementary tokenizer assets | 19,998,761 |
| Complete deployable directory | 6,596,784,699 |
| Candidate presence | 864,026,624 |
| Candidate rotation_signs | 1,081,344 |
| Candidate row_sign_offsets | 11,404,864 |
| Candidate scales | 108,003,328 |
| Candidate signs | 596,482,310 |
| Retained linear_attention_convolutions | 1,572,864 |
| Retained linear_attention_dynamics | 3,072 |
| Retained linear_attention_gates | 12,582,912 |
| Retained output_head | 2,034,237,440 |
| Retained text_normalization | 546,816 |
| Retained text_rotary_buffers | 256 |
| Retained token_embeddings | 2,034,237,440 |
| Retained vision_encoder_including_vision_rotary_buffers | 912,021,104 |
| All model metadata/config/container overhead | 585,564 |
| Decoded cache (runtime only) | 13,824,425,984 |
| FP32 research file (outside deployment) | 1,291,183,353 |

Strict raw represented source: 6,576,199,974 / 18,819,635,168 bytes; 65.056709% reduction, 2.861780× smaller.
Strict BF16-loaded persistent source: 6,576,199,974 / 18,819,627,488 bytes; 65.056694% reduction, 2.861778× smaller.

The 486,581,248 unused MTP bytes are excluded from both source denominators. Forty-eight FP32→BF16 casts explain their 7,680-byte difference; generated rotary buffers (400 bytes) are excluded from strict compressed tensor numerators. All candidate signs/scales/indexes remain included. Tokenizer assets appear once in the complete directory total. Retained embedding/head/vision are included; there is no endpoint prototype.

Payload equality and fresh-reload parity are distinct from FP32 versus exported/runtime logit changes. See the JSON's preserved representation comparisons. No quality conclusion is inferred from this audit.
