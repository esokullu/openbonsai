# Final paired UltraChat manual quality review

Neither compressed arm recovers useful answers on these four prompts. The 16x8 arm yields two empty replies and two fragments; 128x1 yields four empty replies. Both ARC 8/20 results are entirely all-A predictions. Exact serialization and EOS stopping passed independently of failed task performance.

The baseline supplies two complete useful answers and two relevant but capped replies. All four prompts and all seven representations were inspected in full. This review supplements the numerical diagnostics and does not change the completed experiment.

| Condition / stage | Useful complete | Relevant but capped | Empty | Other nonanswers | EOS | Cap | ARC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline / baseline | 2 | 2 | 0 | 0 | 2 | 2 | 19/20 |
| rotated16x8 / hard_fp32 | 0 | 0 | 2 | 2 | 4 | 0 | 8/20 |
| rotated16x8 / exported_scales | 0 | 0 | 2 | 2 | 4 | 0 | 8/20 |
| rotated16x8 / packed | 0 | 0 | 2 | 2 | 4 | 0 | 8/20 |
| rotated128x1 / hard_fp32 | 0 | 0 | 4 | 0 | 4 | 0 | 8/20 |
| rotated128x1 / exported_scales | 0 | 0 | 4 | 0 | 4 | 0 | 8/20 |
| rotated128x1 / packed | 0 | 0 | 4 | 0 | 4 | 0 | 8/20 |

Packed 128x1 minus 16x8: NLL +0.625187576, ARC +0.0pp; both have zero complete useful answers. Original BF16 NLL is 2.423776811, ARC 19/20. Both compressed models choose A on all 20 questions; the eight A-labeled items account for all eight correct answers.

## All saved answers and judgments

Texts below use JSON string notation to preserve exact newlines and trailing whitespace. Token IDs include the terminal EOS where present.

### manual-heldout-00 — instruction

Prompt: Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

**baseline / baseline** — 61 generated tokens; EOS=True; cap=False.

```json
"1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.\n2. Wipe down countertops and appliances with disinfectant before and after every cooking session.\n3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests."
```

Exact token IDs: `[16, 13, 5797, 321, 43675, 660, 24958, 321, 78145, 8405, 6849, 1238, 958, 310, 5190, 5144, 32944, 25979, 13, 198, 17, 13, 457, 3334, 1441, 81306, 321, 32585, 440, 61590, 3624, 505, 1518, 321, 1238, 1396, 16725, 3669, 13, 198, 18, 13, 11440, 680, 279, 22143, 321, 4147, 279, 32757, 9261, 7071, 310, 5471, 61364, 10468, 1052, 321, 72404, 13, 248046]`.

Special-token text: `"1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.\n2. Wipe down countertops and appliances with disinfectant before and after every cooking session.\n3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests.<|im_end|>"`.

Judgment: Exactly three numbered, one-sentence kitchen tips; complete and relevant.

**rotated16x8 / hard_fp32** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS; none of the requested kitchen tips is supplied.

**rotated16x8 / exported_scales** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS; none of the requested kitchen tips is supplied.

**rotated16x8 / packed** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS; none of the requested kitchen tips is supplied.

**rotated128x1 / hard_fp32** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested instruction task.

**rotated128x1 / exported_scales** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested instruction task.

**rotated128x1 / packed** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested instruction task.


### manual-heldout-02 — reasoning

Prompt: A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

**baseline / baseline** — 128 generated tokens; EOS=False; cap=True.

```json
"To solve this, we can use a simple logical deduction based on the number of legs:\n\n1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \\times 2 = 20$ legs.\n2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.\n3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have"
```

Exact token IDs: `[1206, 11290, 411, 11, 567, 628, 958, 264, 4145, 19214, 37557, 3018, 383, 279, 1324, 314, 13795, 25, 271, 16, 13, 220, 2972, 5439, 3754, 660, 220, 16, 15, 9605, 513, 51727, 159029, 8439, 51727, 599, 220, 17, 13795, 11, 411, 1000, 1067, 303, 393, 16, 15, 1088, 14695, 220, 17, 283, 220, 17, 15, 3, 13795, 13, 198, 17, 13, 220, 2972, 26487, 440, 279, 4880, 2392, 1698, 159029, 561, 35197, 682, 220, 17, 23, 13795, 11, 864, 369, 393, 17, 23, 471, 220, 17, 15, 283, 220, 23, 3, 777, 13795, 1056, 1004, 23571, 13, 198, 18, 13, 220, 2972, 7133, 364, 279, 6463, 159029, 8618, 854, 567, 13881, 264, 15683, 364, 264, 37442, 11, 279, 2392, 1698, 12339, 539, 220, 17, 318, 10957, 66803, 599]`.

Special-token text: `"To solve this, we can use a simple logical deduction based on the number of legs:\n\n1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \\times 2 = 20$ legs.\n2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.\n3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have"`.

Judgment: Relevant correct setup, but cut at 128 tokens before giving the final counts (6 chickens and 4 rabbits); not a complete answer.

**rotated16x8 / hard_fp32** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS; no animal counts or explanation.

**rotated16x8 / exported_scales** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS; no animal counts or explanation.

**rotated16x8 / packed** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS; no animal counts or explanation.

**rotated128x1 / hard_fp32** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested reasoning task.

**rotated128x1 / exported_scales** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested reasoning task.

**rotated128x1 / packed** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested reasoning task.


### manual-heldout-04 — coding

Prompt: Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

**baseline / baseline** — 128 generated tokens; EOS=False; cap=True.

```json
"```python\ndef is_balanced(text: str) -> bool:\n    \"\"\"\n    Check whether parentheses in the given text are balanced.\n    Only '(' and ')' are considered; all other characters are ignored.\n    An empty string is considered balanced.\n\n    Examples:\n        is_balanced(\"()\") -> True\n        is_balanced(\"(]\") -> False\n        is_balanced(\"((()))\") -> True\n        is_balanced(\"(()\") -> False\n        is_balanced(\"\") -> True\n        is_balanced(\"a(b)c\") -> True\n    \"\"\"\n    count = 0\n   "
```

Exact token IDs: `[71093, 12305, 198, 727, 369, 54547, 4718, 7008, 25, 592, 8, 1411, 1744, 25, 198, 262, 4071, 198, 262, 4109, 3315, 71458, 303, 279, 2574, 1414, 513, 23086, 13, 198, 262, 8020, 36629, 321, 76647, 513, 6306, 26, 660, 975, 5587, 513, 11693, 13, 198, 262, 1473, 4147, 886, 369, 6306, 23086, 13, 271, 262, 24508, 25, 198, 285, 369, 54547, 4718, 437, 363, 871, 1411, 2912, 198, 285, 369, 54547, 4718, 30690, 84271, 1411, 3439, 198, 285, 369, 54547, 4718, 437, 1148, 9831, 871, 1411, 2912, 198, 285, 369, 54547, 4718, 437, 4903, 871, 1411, 3439, 198, 285, 369, 54547, 4718, 37754, 1411, 2912, 198, 285, 369, 54547, 4718, 437, 64, 1818, 45417, 871, 1411, 2912, 198, 262, 4071, 198, 262, 1698, 283, 220, 15, 198, 262]`.

Special-token text: `"```python\ndef is_balanced(text: str) -> bool:\n    \"\"\"\n    Check whether parentheses in the given text are balanced.\n    Only '(' and ')' are considered; all other characters are ignored.\n    An empty string is considered balanced.\n\n    Examples:\n        is_balanced(\"()\") -> True\n        is_balanced(\"(]\") -> False\n        is_balanced(\"((()))\") -> True\n        is_balanced(\"(()\") -> False\n        is_balanced(\"\") -> True\n        is_balanced(\"a(b)c\") -> True\n    \"\"\"\n    count = 0\n   "`.

Judgment: Relevant signature, documentation and examples, but cut at 128 tokens before implementing the checking logic; not a usable complete function.

**rotated16x8 / hard_fp32** — 3 generated tokens; EOS=True; cap=False.

```json
"Write a"
```

Exact token IDs: `[7734, 264, 248046]`.

Special-token text: `"Write a<|im_end|>"`.

Judgment: Only a request fragment, Write a; no Python function or empty-string case.

**rotated16x8 / exported_scales** — 3 generated tokens; EOS=True; cap=False.

```json
"Write a"
```

Exact token IDs: `[7734, 264, 248046]`.

Special-token text: `"Write a<|im_end|>"`.

Judgment: Only a request fragment, Write a; no Python function or empty-string case.

**rotated16x8 / packed** — 3 generated tokens; EOS=True; cap=False.

```json
"Write a"
```

Exact token IDs: `[7734, 264, 248046]`.

Special-token text: `"Write a<|im_end|>"`.

Judgment: Only a request fragment, Write a; no Python function or empty-string case.

**rotated128x1 / hard_fp32** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested coding task.

**rotated128x1 / exported_scales** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested coding task.

**rotated128x1 / packed** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested coding task.


### manual-heldout-07 — turkish

Prompt: Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

**baseline / baseline** — 24 generated tokens; EOS=True; cap=False.

```json
"Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?"
```

Exact token IDs: `[56, 277, 15642, 237023, 197619, 245279, 176026, 36069, 26, 215074, 3649, 190796, 193966, 14804, 159262, 377, 220, 13424, 3078, 161301, 9217, 219883, 30, 248046]`.

Special-token text: `"Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?<|im_end|>"`.

Judgment: Complete formal Turkish rewrite preserving the intended request.

**rotated16x8 / hard_fp32** — 2 generated tokens; EOS=True; cap=False.

```json
"#"
```

Exact token IDs: `[2, 248046]`.

Special-token text: `"#<|im_end|>"`.

Judgment: Only a hash character; no Turkish rewrite.

**rotated16x8 / exported_scales** — 2 generated tokens; EOS=True; cap=False.

```json
"#"
```

Exact token IDs: `[2, 248046]`.

Special-token text: `"#<|im_end|>"`.

Judgment: Only a hash character; no Turkish rewrite.

**rotated16x8 / packed** — 2 generated tokens; EOS=True; cap=False.

```json
"#"
```

Exact token IDs: `[2, 248046]`.

Special-token text: `"#<|im_end|>"`.

Judgment: Only a hash character; no Turkish rewrite.

**rotated128x1 / hard_fp32** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested turkish task.

**rotated128x1 / exported_scales** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested turkish task.

**rotated128x1 / packed** — 1 generated tokens; EOS=True; cap=False.

```json
""
```

Exact token IDs: `[248046]`.

Special-token text: `"<|im_end|>"`.

Judgment: Immediate EOS-only empty reply; no content answers the requested turkish task.

## Limits

- Repeated development diagnostics: 2,048 NLL tokens, 20 ARC -Easy questions and four prompts, not the original full suite or independent confirmation.
- Manual usefulness is a small descriptive review, not an automatically scored general capability estimate.
- EOS termination is a stopping condition; empty EOS-only outputs are not useful completions.
- Representation-dependent generated-token counts prevent inference-speed conclusions from total evaluation time.
- Almost matched total active-token exposure does not isolate a pure causal effect of diversity versus repetition/order.
- No quality gate, calibration reselection, hyperparameter change or additional training was triggered by this review.
