# UltraChat rotated128x1 hard-FP32 review (provisional)

All four prompts receive immediate EOS with no answer; ARC8/20 is all-A guessing on this fixed20-question subset. No quality recovery in this completed hard-FP32 stage. FP16 and fresh packed stages remain pending.

NLL 9.121630685, PPL 9151.112039; NLL difference versus16x8 hard-FP32 +0.622220882, versus original BF16 +6.697853875. Four EOS completions, zero useful answers. EOS is a stopping condition, not a task-completion measure.

Each complete saved answer is reproduced below; IDs are exactly [248046] in all four cases.

manual-heldout-00 — instruction

Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

Exact output: `""`; exact special-token output: `<|im_end|>`.

Immediate EOS-only reply; no content addressing the requested instruction task.

manual-heldout-02 — reasoning

A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

Exact output: `""`; exact special-token output: `<|im_end|>`.

Immediate EOS-only reply; no content addressing the requested reasoning task.

manual-heldout-04 — coding

Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

Exact output: `""`; exact special-token output: `<|im_end|>`.

Immediate EOS-only reply; no content addressing the requested coding task.

manual-heldout-07 — turkish

Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

Exact output: `""`; exact special-token output: `<|im_end|>`.

Immediate EOS-only reply; no content addressing the requested turkish task.

This is the repeated diagnostic set of2,048 NLL tokens,20 ARC questions and4 prompts. The other two128x1 representations are excluded until completed; no final paired result is claimed.
