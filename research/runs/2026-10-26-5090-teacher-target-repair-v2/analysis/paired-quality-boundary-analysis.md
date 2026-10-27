# Independent quality and boundary analysis

The original-prefix target improves the selected window on all 16 search records and passes the preregistered local criterion. Model quality remains unusable: the conditional arm returns empty EOS-only outputs; the original-prefix arm returns only `#` and EOS for all four tasks. Neither improves ARC accuracy.

Quality is measured on the assembled hard-quantized research model with FP32 factors/A/B: 2,048 scored WikiText tokens, 20 ARC-Easy questions and four fixed development prompts. These are not the original 16,384/100/10 full-pilot populations or an unused confirmation set. Original BF16 and initial results are authenticated legacy results, supported by fresh exact probes; the two repaired results are newly measured.

| Research state | NLL | Perplexity | ARC | Useful complete | EOS / cap | Empty |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.4237768106 | 11.288413 | 19/20 | 2/4 | 2 / 2 | 0 |
| initial | 9.1216306852 | 9151.112039 | 8/20 | 0/4 | 4 / 0 | 4 |
| conditional | 9.1216917017 | 9151.670425 | 8/20 | 0/4 | 4 / 0 | 4 |
| original_prefix | 8.2791870161 | 3940.989118 | 8/20 | 0/4 | 4 / 0 | 0 |

BF16 also gives two relevant but truncated reasoning/code answers. Every compressed state chooses A on all 20 ARC items; eight reference answers are A. Compared with BF16, this loses 12 previously correct answers and gains one previously incorrect answer. EOS termination is distinct from usefulness.

| Difference | NLL | Perplexity | ARC percentage points |
|---|---:|---:|---:|
| conditional_minus_initial | +0.0000610165 | +0.558386 | +0.0 |
| original_prefix_minus_initial | -0.8424436692 | -5210.122921 | +0.0 |
| original_prefix_minus_conditional | -0.8425046856 | -5210.681306 | +0.0 |
| original_prefix_minus_baseline | +5.8554102054 | +3929.700705 | -55.0 |

The boundary comparison uses the same 16 search records and 7,203 active positions, with fixed quantized-prefix input, conditional target Tq, original-prefix target T0, native arguments and masks. FP64 sufficient statistics are pooled by valid elements. The quality loss continues to use FP32 cross-entropy with FP64 summation.

| Boundary state | T0 MSE | T0 relative RMSE | T0 cosine | Output / T0 RMS | Conditional MSE | Signed cross mean |
|---|---:|---:|---:|---:|---:|---:|
| initial | 0.196574566897 | 0.760928664 | 0.670894394 | 0.841515958 | 0.001402226641 | -0.001707701643 |
| conditional/final_hard | 0.196725874517 | 0.761221459 | 0.670532455 | 0.841037244 | 0.001184585214 | -0.001338752596 |
| original_prefix/final_hard | 0.132229789576 | 0.624086709 | 0.781715482 | 0.805449751 | 0.062466821353 | -0.127117073676 |

Original-prefix pooled T0 MSE decreases 32.733012% versus initial and 32.784749% versus the conditional control (both exceed 5%). All 16 records improve versus both references; none regresses, so the 12/16 and maximum 10% regression criteria pass. The conditional control slightly worsens common-T0 MSE on all 16 records: pooled +0.076972%.

The unchanged target-gap MSE is 0.19688004189915323. Its signed cross term becomes more negative in the original-prefix arm, while conditional error against Tq increases. This is a descriptive error decomposition, not a causal attribution. Relative RMSE against Tq and T0 uses different denominators.

| Arm / reference | Saved last-soft parameters, soft | Same parameters, hard | Final hard | Hard minus soft | Final minus saved hard |
|---|---:|---:|---:|---:|---:|
| conditional / y/t0 | 0.196779880091 | 0.196918048817 | 0.196725874517 | +0.000138168725 | -0.000192174300 |
| conditional / y/tq | 0.001221686626 | 0.001334092308 | 0.001184585214 | +0.000112405682 | -0.000149507095 |
| original_prefix / y/t0 | 0.133738962795 | 0.133728812382 | 0.132229789576 | -0.000010150413 | -0.001499022806 |
| original_prefix / y/tq | 0.058631474296 | 0.061036278677 | 0.062466821353 | +0.002404804381 | +0.001430542676 |

The saved soft state is the actual pre-update102 state, progress102/128 and sharpness29.8828125. Its soft and hard evaluations use identical saved parameter values. Final-versus-saved-hard includes update102 plus all26hard updates; it cannot isolate the effect of the hard stage alone. No last-soft full-model quality evaluation was run.

Runtime and peak memory rows are retained in the JSON with their evaluation-only scope. Generation takes fewer steps because outputs are degenerate and stop early; this is not an inference speedup. Selected FP32-state and packed-payload fresh reload parity is handled separately and does not turn these measurements into whole packed-artifact quality results.

# Exact saved generation outputs

## baseline

**manual-heldout-00 (instruction)**

Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

Judgment: `complete_useful`; EOS=True; capped=False; tokens=61.

Exact text as a JSON string:
```json
"1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.\n2. Wipe down countertops and appliances with disinfectant before and after every cooking session.\n3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests."
```

**manual-heldout-02 (reasoning)**

A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

Judgment: `relevant_but_truncated`; EOS=False; capped=True; tokens=128.

Exact text as a JSON string:
```json
"To solve this, we can use a simple logical deduction based on the number of legs:\n\n1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \\times 2 = 20$ legs.\n2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.\n3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have"
```

**manual-heldout-04 (coding)**

Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

Judgment: `relevant_but_truncated`; EOS=False; capped=True; tokens=128.

Exact text as a JSON string:
```json
"```python\ndef is_balanced(text: str) -> bool:\n    \"\"\"\n    Check whether parentheses in the given text are balanced.\n    Only '(' and ')' are considered; all other characters are ignored.\n    An empty string is considered balanced.\n\n    Examples:\n        is_balanced(\"()\") -> True\n        is_balanced(\"(]\") -> False\n        is_balanced(\"((()))\") -> True\n        is_balanced(\"(()\") -> False\n        is_balanced(\"\") -> True\n        is_balanced(\"a(b)c\") -> True\n    \"\"\"\n    count = 0\n   "
```

**manual-heldout-07 (turkish)**

Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

Judgment: `complete_useful`; EOS=True; capped=False; tokens=24.

Exact text as a JSON string:
```json
"Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?"
```

## initial

**manual-heldout-00 (instruction)**

Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

**manual-heldout-02 (reasoning)**

A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

**manual-heldout-04 (coding)**

Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

**manual-heldout-07 (turkish)**

Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

## conditional

**manual-heldout-00 (instruction)**

Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

**manual-heldout-02 (reasoning)**

A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

**manual-heldout-04 (coding)**

Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

**manual-heldout-07 (turkish)**

Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

Judgment: `empty_eos_only`; EOS=True; capped=False; tokens=1.

Exact text as a JSON string:
```json
""
```

IDs: `[248046]`.

## original_prefix

**manual-heldout-00 (instruction)**

Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

Judgment: `irrelevant_or_degenerate`; EOS=True; capped=False; tokens=2.

Exact text as a JSON string:
```json
"#"
```

IDs: `[2, 248046]`.

**manual-heldout-02 (reasoning)**

A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

Judgment: `irrelevant_or_degenerate`; EOS=True; capped=False; tokens=2.

Exact text as a JSON string:
```json
"#"
```

IDs: `[2, 248046]`.

**manual-heldout-04 (coding)**

Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

Judgment: `irrelevant_or_degenerate`; EOS=True; capped=False; tokens=2.

Exact text as a JSON string:
```json
"#"
```

IDs: `[2, 248046]`.

**manual-heldout-07 (turkish)**

Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

Judgment: `irrelevant_or_degenerate`; EOS=True; capped=False; tokens=2.

Exact text as a JSON string:
```json
"#"
```

IDs: `[2, 248046]`.

All generated IDs, special-token text, exact prompts, result hashes, per-record errors, all ARC score files and provenance are preserved in the adjacent JSON and its bound input files.
