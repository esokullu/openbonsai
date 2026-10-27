# Recorded diagnostic generations

Status: **completed**. Raw text is HTML-escaped for display only; original strings and full IDs are preserved in the JSON. No manual quality judgments are assigned.

## manual-heldout-00 (instruction)

Original frozen prompt:

<pre>Give exactly three numbered tips for keeping a shared kitchen clean. Each tip must be one sentence.</pre>

### baseline

Tokens: 61; ended with EOS: True; unfinished: False; hit 128-token cap: False; elapsed: 6.522464 s.

Recorded text:

<pre>1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.
2. Wipe down countertops and appliances with disinfectant before and after every cooking session.
3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests.</pre>

Recorded text including special tokens:

<pre>1. Wash and sanitize all dishes and utensils immediately after use to prevent cross-contamination.
2. Wipe down countertops and appliances with disinfectant before and after every cooking session.
3. Take out the trash and empty the recycling bin daily to avoid lingering odors and pests.&lt;|im_end|&gt;</pre>

Generated IDs:

```json
[16, 13, 5797, 321, 43675, 660, 24958, 321, 78145, 8405, 6849, 1238, 958, 310, 5190, 5144, 32944, 25979, 13, 198, 17, 13, 457, 3334, 1441, 81306, 321, 32585, 440, 61590, 3624, 505, 1518, 321, 1238, 1396, 16725, 3669, 13, 198, 18, 13, 11440, 680, 279, 22143, 321, 4147, 279, 32757, 9261, 7071, 310, 5471, 61364, 10468, 1052, 321, 72404, 13, 248046]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 60,
  "unique_content_token_ids": 46,
  "dominant_token_id": 13,
  "dominant_token_count": 6,
  "dominant_token_fraction": 0.1,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 50.875489 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 52.146453 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 68.292218 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 152.900655 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 153.462651 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 81.302846 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

## manual-heldout-02 (reasoning)

Original frozen prompt:

<pre>A farmer has chickens and rabbits. There are 10 heads and 28 legs. How many of each animal are there? Explain briefly.</pre>

### baseline

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 12.564387 s.

Recorded text:

<pre>To solve this, we can use a simple logical deduction based on the number of legs:

1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \times 2 = 20$ legs.
2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.
3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have</pre>

Recorded text including special tokens:

<pre>To solve this, we can use a simple logical deduction based on the number of legs:

1.  **Assume all 10 animals are chickens.** Since chickens have 2 legs, this would result in $10 \times 2 = 20$ legs.
2.  **Compare with the actual leg count.** The farmer has 28 legs, which is $28 - 20 = 8$ more legs than our assumption.
3.  **Account for the difference.** Each time we swap a chicken for a rabbit, the leg count increases by 2 (since rabbits have</pre>

Generated IDs:

```json
[1206, 11290, 411, 11, 567, 628, 958, 264, 4145, 19214, 37557, 3018, 383, 279, 1324, 314, 13795, 25, 271, 16, 13, 220, 2972, 5439, 3754, 660, 220, 16, 15, 9605, 513, 51727, 159029, 8439, 51727, 599, 220, 17, 13795, 11, 411, 1000, 1067, 303, 393, 16, 15, 1088, 14695, 220, 17, 283, 220, 17, 15, 3, 13795, 13, 198, 17, 13, 220, 2972, 26487, 440, 279, 4880, 2392, 1698, 159029, 561, 35197, 682, 220, 17, 23, 13795, 11, 864, 369, 393, 17, 23, 471, 220, 17, 15, 283, 220, 23, 3, 777, 13795, 1056, 1004, 23571, 13, 198, 18, 13, 220, 2972, 7133, 364, 279, 6463, 159029, 8618, 854, 567, 13881, 264, 15683, 364, 264, 37442, 11, 279, 2392, 1698, 12339, 539, 220, 17, 318, 10957, 66803, 599]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 73,
  "dominant_token_id": 220,
  "dominant_token_count": 11,
  "dominant_token_fraction": 0.0859375,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 51.108211 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 52.758645 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 69.596654 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 152.795270 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 153.483133 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 81.565578 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

## manual-heldout-04 (coding)

Original frozen prompt:

<pre>Write a Python function is_balanced(text) that checks whether parentheses are balanced, ignoring other characters. Include an empty-string case.</pre>

### baseline

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 10.447771 s.

Recorded text:

<pre>```python
def is_balanced(text: str) -&gt; bool:
    &quot;&quot;&quot;
    Check whether parentheses in the given text are balanced.
    Only &#x27;(&#x27; and &#x27;)&#x27; are considered; all other characters are ignored.
    An empty string is considered balanced.

    Examples:
        is_balanced(&quot;()&quot;) -&gt; True
        is_balanced(&quot;(]&quot;) -&gt; False
        is_balanced(&quot;((()))&quot;) -&gt; True
        is_balanced(&quot;(()&quot;) -&gt; False
        is_balanced(&quot;&quot;) -&gt; True
        is_balanced(&quot;a(b)c&quot;) -&gt; True
    &quot;&quot;&quot;
    count = 0
   </pre>

Recorded text including special tokens:

<pre>```python
def is_balanced(text: str) -&gt; bool:
    &quot;&quot;&quot;
    Check whether parentheses in the given text are balanced.
    Only &#x27;(&#x27; and &#x27;)&#x27; are considered; all other characters are ignored.
    An empty string is considered balanced.

    Examples:
        is_balanced(&quot;()&quot;) -&gt; True
        is_balanced(&quot;(]&quot;) -&gt; False
        is_balanced(&quot;((()))&quot;) -&gt; True
        is_balanced(&quot;(()&quot;) -&gt; False
        is_balanced(&quot;&quot;) -&gt; True
        is_balanced(&quot;a(b)c&quot;) -&gt; True
    &quot;&quot;&quot;
    count = 0
   </pre>

Generated IDs:

```json
[71093, 12305, 198, 727, 369, 54547, 4718, 7008, 25, 592, 8, 1411, 1744, 25, 198, 262, 4071, 198, 262, 4109, 3315, 71458, 303, 279, 2574, 1414, 513, 23086, 13, 198, 262, 8020, 36629, 321, 76647, 513, 6306, 26, 660, 975, 5587, 513, 11693, 13, 198, 262, 1473, 4147, 886, 369, 6306, 23086, 13, 271, 262, 24508, 25, 198, 285, 369, 54547, 4718, 437, 363, 871, 1411, 2912, 198, 285, 369, 54547, 4718, 30690, 84271, 1411, 3439, 198, 285, 369, 54547, 4718, 437, 1148, 9831, 871, 1411, 2912, 198, 285, 369, 54547, 4718, 437, 4903, 871, 1411, 3439, 198, 285, 369, 54547, 4718, 37754, 1411, 2912, 198, 285, 369, 54547, 4718, 437, 64, 1818, 45417, 871, 1411, 2912, 198, 262, 4071, 198, 262, 1698, 283, 220, 15, 198, 262]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 59,
  "dominant_token_id": 198,
  "dominant_token_count": 14,
  "dominant_token_fraction": 0.109375,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 50.627928 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 51.372318 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 66.650552 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 153.114559 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 153.218991 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 81.134409 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

## manual-heldout-07 (turkish)

Original frozen prompt:

<pre>Aşağıdaki cümleyi daha resmi bir Türkçeyle yeniden yaz: Yarın gelemiyorum, toplantıyı başka güne alsak olur mu?</pre>

### baseline

Tokens: 24; ended with EOS: True; unfinished: False; hit 128-token cap: False; elapsed: 1.388005 s.

Recorded text:

<pre>Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?</pre>

Recorded text including special tokens:

<pre>Yarın katılım sağlayamayacağım; toplantıyı başka bir tarihe erteleyebilir miyiz?&lt;|im_end|&gt;</pre>

Generated IDs:

```json
[56, 277, 15642, 237023, 197619, 245279, 176026, 36069, 26, 215074, 3649, 190796, 193966, 14804, 159262, 377, 220, 13424, 3078, 161301, 9217, 219883, 30, 248046]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 23,
  "unique_content_token_ids": 23,
  "dominant_token_id": 56,
  "dominant_token_count": 1,
  "dominant_token_fraction": 0.043478260869565216,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 50.821571 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 51.214341 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### unrotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 66.563155 s.

Recorded text:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Recorded text including special tokens:

<pre> number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number number</pre>

Generated IDs:

```json
[1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324, 1324]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 1,
  "dominant_token_id": 1324,
  "dominant_token_count": 128,
  "dominant_token_fraction": 1.0,
  "longest_identical_token_run": 128,
  "single_token_repeated_all_content": true,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/hard_fp32

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 153.422277 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/exported_scales

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 153.065520 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```

### rotated16x8/packed

Tokens: 128; ended with EOS: False; unfinished: True; hit 128-token cap: True; elapsed: 80.862990 s.

Recorded text:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Recorded text including special tokens:

<pre>Here

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

This

</pre>

Generated IDs:

```json
[8160, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271, 1919, 271]
```

Automated repetition statistics:

```json
{
  "content_tokens_excluding_terminal_eos": 128,
  "unique_content_token_ids": 3,
  "dominant_token_id": 271,
  "dominant_token_count": 64,
  "dominant_token_fraction": 0.5,
  "longest_identical_token_run": 1,
  "single_token_repeated_all_content": false,
  "empty_text": false,
  "whitespace_only_text": false
}
```
