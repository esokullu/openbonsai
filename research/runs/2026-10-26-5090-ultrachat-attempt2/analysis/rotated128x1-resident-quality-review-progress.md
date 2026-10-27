# UltraChat rotated128x1 resident evaluations (provisional)

Both resident representations give four immediate EOS-only empty replies; ARC8/20 is all-A prediction. No useful answer or quality recovery on these four prompts. Final paired summary waits for fresh packed reload.

| Stage | NLL | Perplexity | ARC | Empty /4 | EOS /4 | Useful /4 |
|---|---:|---:|---:|---:|---:|---:|
| hard_fp32 | 9.121630685 | 9151.112039 | 8/20 (all A) | 4 | 4 | 0 |
| exported_scales | 9.121126852 | 9146.502564 | 8/20 (all A) | 4 | 4 | 0 |

All eight answers were inspected in full. Every exact generated ID list is `[248046]`, decoded text is `""`, and special-token text is `<|im_end|>`. There is no response to the kitchen, animal-count reasoning, Python function, or Turkish-rewrite task. EOS stopping is successful; answering is not.

FP16 minus hard-FP32 NLL is -0.000503833. This small rounding difference does not change any answer or ARC prediction.

Raw text, full prompts and IDs are preserved in the companion JSON and quality-generations-progress.md. These are repeated development diagnostics (2,048 NLL tokens, 20 ARC questions, four prompts). No model execution or recipe changes were made for this review.
