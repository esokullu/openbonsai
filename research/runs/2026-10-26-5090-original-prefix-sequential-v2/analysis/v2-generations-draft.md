Exact saved generations on the four fixed development prompts. EOS completion is not a usefulness judgment.

Prompt `manual-heldout-00`: Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.

**Untouched BF16 (saved)** — 61tokens; EOS=True; cap=False.

> 1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.
> 2. Wipe down countertops and appliances with disinfectant before and after every cooking session.
> 3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests.

Token IDs: `[16, 13, 5797, 321, 43675, 660, 24958, 321, 78145, 8405, 6849, 1238, 958, 310, 5190, 5144, 32944, 25979, 13, 198, 17, 13, 457, 3334, 1441, 81306, 321, 32585, 440, 61590, 3624, 505, 1518, 321, 1238, 1396, 16725, 3669, 13, 198, 18, 13, 11440, 680, 279, 22143, 321, 4147, 279, 32757, 9261, 7071, 310, 5471, 61364, 10468, 1052, 321, 72404, 13, 248046]`

**Historical conditional128 hard FP32** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**Historical conditional128 packed** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**New original-prefix hard FP32** — 128tokens; EOS=False; cap=True.

> Here is a **Ded- A Comprehensive, A "Ded- A Ds" (Ded- 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

Token IDs: `[8160, 369, 264, 2972, 35, 290, 12, 357, 64605, 11, 357, 328, 35, 290, 12, 357, 414, 82, 1, 318, 35, 290, 12, 220, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]`

**New original-prefix fresh packed** — 128tokens; EOS=False; cap=True.

> Here is a **Ded- A Comprehensive, A "Ded- A Ds" (Ded- 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

Token IDs: `[8160, 369, 264, 2972, 35, 290, 12, 357, 64605, 11, 357, 328, 35, 290, 12, 357, 414, 82, 1, 318, 35, 290, 12, 220, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]`

Prompt `manual-heldout-02`: A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.

**Untouched BF16 (saved)** — 128tokens; EOS=False; cap=True.

> To solve this, we can use a simple logical deduction based on the number of legs:
> 
> 1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \times 2 = 20$ legs.
> 2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.
> 3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have

Token IDs: `[1206, 11290, 411, 11, 567, 628, 958, 264, 4145, 19214, 37557, 3018, 383, 279, 1324, 314, 13795, 25, 271, 16, 13, 220, 2972, 5439, 3754, 660, 220, 16, 15, 9605, 513, 51727, 159029, 8439, 51727, 599, 220, 17, 13795, 11, 411, 1000, 1067, 303, 393, 16, 15, 1088, 14695, 220, 17, 283, 220, 17, 15, 3, 13795, 13, 198, 17, 13, 220, 2972, 26487, 440, 279, 4880, 2392, 1698, 159029, 561, 35197, 682, 220, 17, 23, 13795, 11, 864, 369, 393, 17, 23, 471, 220, 17, 15, 283, 220, 23, 3, 777, 13795, 1056, 1004, 23571, 13, 198, 18, 13, 220, 2972, 7133, 364, 279, 6463, 159029, 8618, 854, 567, 13881, 264, 15683, 364, 264, 37442, 11, 279, 2392, 1698, 12339, 539, 220, 17, 318, 10957, 66803, 599]`

**Historical conditional128 hard FP32** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**Historical conditional128 packed** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**New original-prefix hard FP32** — 128tokens; EOS=False; cap=True.

> Here is a **sight** and the **s**
> 
> The **Sight** The **Sight** (The "Sight"")**
> The **Sight** (The "Sight"")**
> The **Sight** (The "Sight"")**
> *   **Sight** (The "Sight"")**
> *   **Sight** (The "Sight"")**
> *   **Sight** (The "Sight"")**
> *   **Sight** (The "Sight"")**
> *   **Sight** (S

Token IDs: `[8160, 369, 264, 2972, 82, 481, 332, 321, 279, 2972, 82, 332, 271, 760, 2972, 50, 481, 332, 561, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 760, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 760, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 9, 256, 2972, 50, 481, 332, 318, 50]`

**New original-prefix fresh packed** — 128tokens; EOS=False; cap=True.

> Here is a **sight** and the **s**
> 
> The **Sight** The **Sight** (The "Sight"")**
> The **Sight** (The "Sight"")**
> The **Sight** (The "Sight"")**
> *   **Sight** (The Suggest)
> *   **Sight** (The Suggest)
> *   **Sight** (The Suggest)
> *   **Sight** (Sight)
> *   **Sight** (Sight)
> *   **Sight** (Sight)

Token IDs: `[8160, 369, 264, 2972, 82, 481, 332, 321, 279, 2972, 82, 332, 271, 760, 2972, 50, 481, 332, 561, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 760, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 760, 2972, 50, 481, 332, 318, 760, 328, 50, 481, 1, 871, 332, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 326, 3671, 8, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 326, 3671, 8, 198, 9, 256, 2972, 50, 481, 332, 318, 760, 326, 3671, 8, 198, 9, 256, 2972, 50, 481, 332, 318, 50, 481, 8, 198, 9, 256, 2972, 50, 481, 332, 318, 50, 481, 8, 198, 9, 256, 2972, 50, 481, 332, 318, 50, 481, 8]`

Prompt `manual-heldout-04`: Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.

**Untouched BF16 (saved)** — 128tokens; EOS=False; cap=True.

> ```python
> def is_balanced(text: str) -> bool:
>     """
>     Check whether parentheses in the given text are balanced.
>     Only '(' and ')' are considered; all other characters are ignored.
>     An empty string is considered balanced.
> 
>     Examples:
>         is_balanced("()") -> True
>         is_balanced("(]") -> False
>         is_balanced("((()))") -> True
>         is_balanced("(()") -> False
>         is_balanced("") -> True
>         is_balanced("a(b)c") -> True
>     """
>     count = 0
>    

Token IDs: `[71093, 12305, 198, 727, 369, 54547, 4718, 7008, 25, 592, 8, 1411, 1744, 25, 198, 262, 4071, 198, 262, 4109, 3315, 71458, 303, 279, 2574, 1414, 513, 23086, 13, 198, 262, 8020, 36629, 321, 76647, 513, 6306, 26, 660, 975, 5587, 513, 11693, 13, 198, 262, 1473, 4147, 886, 369, 6306, 23086, 13, 271, 262, 24508, 25, 198, 285, 369, 54547, 4718, 437, 363, 871, 1411, 2912, 198, 285, 369, 54547, 4718, 30690, 84271, 1411, 3439, 198, 285, 369, 54547, 4718, 437, 1148, 9831, 871, 1411, 2912, 198, 285, 369, 54547, 4718, 437, 4903, 871, 1411, 3439, 198, 285, 369, 54547, 4718, 37754, 1411, 2912, 198, 285, 369, 54547, 4718, 437, 64, 1818, 45417, 871, 1411, 2912, 198, 262, 4071, 198, 262, 1698, 283, 220, 15, 198, 262]`

**Historical conditional128 hard FP32** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**Historical conditional128 packed** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**New original-prefix hard FP32** — 128tokens; EOS=False; cap=True.

> Here is a simple, **s**
> 
> **The "Sight" (Sight)
> **Sight: A simple, non- **Sight**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **

Token IDs: `[8160, 369, 264, 4145, 11, 2972, 82, 332, 271, 332, 760, 328, 50, 481, 1, 318, 50, 481, 8, 198, 332, 50, 481, 25, 357, 4145, 11, 2397, 12, 2972, 50, 481, 332, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972]`

**New original-prefix fresh packed** — 128tokens; EOS=False; cap=True.

> Here is a simple, **s**
> 
> **The "Sight" (Sight)
> **The "Sight" (Sight)
> **Sight: The "Sight" (Sight)
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**
> *   **Sight:**

Token IDs: `[8160, 369, 264, 4145, 11, 2972, 82, 332, 271, 332, 760, 328, 50, 481, 1, 318, 50, 481, 8, 198, 332, 760, 328, 50, 481, 1, 318, 50, 481, 8, 198, 332, 50, 481, 25, 561, 328, 50, 481, 1, 318, 50, 481, 8, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700, 198, 9, 256, 2972, 50, 481, 64700]`

Prompt `manual-heldout-07`: Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?

**Untouched BF16 (saved)** — 24tokens; EOS=True; cap=False.

> Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?

Token IDs: `[56, 277, 15642, 237023, 197619, 245279, 176026, 36069, 26, 215074, 3649, 190796, 193966, 14804, 159262, 377, 220, 13424, 3078, 161301, 9217, 219883, 30, 248046]`

**Historical conditional128 hard FP32** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**Historical conditional128 packed** — 1tokens; EOS=True; cap=False.

> 
(Empty decoded text.)

Token IDs: `[248046]`

**New original-prefix hard FP32** — 128tokens; EOS=False; cap=True.

> Here is a **sight** (The "Cul" A Cyl" A Cyl" (Cyl)
> 
> **The Cyl: The Cyl- A Cyl- (Cyl)
> *   **Cyl- The Cyl- (Cyl)
> *   **Cyl- The Cyl- (Cyl)
> *   **Cyl- The Cyl- (Cyl-)**
> *   **Cyl- The Cyl- (Cyl-)**
> *   **Cyl- Cyl- (Cyl-)**
> *   **Cyl- C

Token IDs: `[8160, 369, 264, 2972, 82, 481, 332, 318, 760, 328, 34, 355, 1, 357, 351, 3791, 1, 357, 351, 3791, 1, 318, 34, 3791, 8, 271, 332, 760, 351, 3791, 25, 561, 351, 3791, 12, 357, 351, 3791, 12, 318, 34, 3791, 8, 198, 9, 256, 2972, 34, 3791, 12, 561, 351, 3791, 12, 318, 34, 3791, 8, 198, 9, 256, 2972, 34, 3791, 12, 561, 351, 3791, 12, 318, 34, 3791, 8, 198, 9, 256, 2972, 34, 3791, 12, 561, 351, 3791, 12, 318, 34, 3791, 12, 31230, 198, 9, 256, 2972, 34, 3791, 12, 561, 351, 3791, 12, 318, 34, 3791, 12, 31230, 198, 9, 256, 2972, 34, 3791, 12, 351, 3791, 12, 318, 34, 3791, 12, 31230, 198, 9, 256, 2972, 34, 3791, 12, 351]`

**New original-prefix fresh packed** — 128tokens; EOS=False; cap=True.

> Here is a **sight** (The "C" A"R"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"O"

Token IDs: `[8160, 369, 264, 2972, 82, 481, 332, 318, 760, 328, 34, 1, 357, 1, 49, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1, 46, 1]`

