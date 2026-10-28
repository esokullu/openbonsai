# Compressing your own Qwen3.5-9B fine-tune

This guide takes a fine-tune of Qwen3.5-9B to a ternary PTQ1_0 GGUF of about 2 GB that runs in Prism's llama.cpp fork. It uses the recipe behind the published model, [webbrain-one/openbonsai-qwen-3.5-9b](https://huggingface.co/webbrain-one/openbonsai-qwen-3.5-9b): `examples/qwen3_5-9b-reasoning.yaml`.

Independent project, not affiliated with or endorsed by Prism ML. Bonsai is Prism ML's product.

**Research preview.** Only base Qwen3.5-9B has been compressed and measured so far. No fine-tune has been run yet, so the numbers below are for the base model and the published recipe, and what your fine-tune keeps is for you to measure (step 5).

Your model is the teacher at every step. It answers prompts in its own chat template (the traces), those answers become calibration and distillation data, it is the BF16 reference for every metric, and its tokenizer and chat template go into the GGUF.

## Other ways to adapt a Bonsai model

Prism publishes Ternary Bonsai 2 as packed ternary files only (GGUF and MLX), and its compression pipeline is not public. So far, community fine-tunes therefore adapt the compressed model itself:

| Approach | Example | Result |
|---|---|---|
| LoRA on the frozen ternary base | [Bonsai-JEV](https://huggingface.co/dakebakery/Bonsai-JEV): rank 16 on Ternary Bonsai 2 27B, trained against the base dequantized to BF16, about 24 h on 2× RTX PRO 6000 (96 GB). [tri-bonsai](https://github.com/athatheo/tri-bonsai): GRPO on Ternary Bonsai 1.7B, in MLX | Prism's ternary model plus an FP16 adapter, loaded with llama.cpp's `--lora` |
| Train only the group scales and norms, codes frozen | tri-bonsai's Scale-Norm Tuning (1.7B) | still fully ternary; small gains there (GSM8K +1.7 points). OpenBonsai's distillation step is of this kind |
| Ternary quantization-aware training | [ternary_QAT](https://pypi.org/project/ternary-QAT/) on Prism's `-unpacked` checkpoints (first-generation Ternary Bonsai, 1.7B to 27B; none exist for Bonsai 2) | ternary weights trained further; its README advises a learning rate 10-50× the usual |
| Compress your own BF16 fine-tune | this guide | the whole fine-tune as a ternary PTQ1_0 file, no adapter |

The first three start from Prism's compression, which is still stronger than OpenBonsai's (with thinking off, GSM8K 77/100 against 69/100, HumanEval 45/50 against 19/50). But the fine-tune has to be redone on the Bonsai model, and an adapter adds FP16 weights and two small matrix products per adapted layer at inference. This guide starts from the fine-tune you already have.

## What you need

- **One 32 GB GPU.** Everything was measured on one RTX 5090; smaller cards are untested. Generation with long answers comes close to the limit (28.7 GB peak at 8 × 4,096 tokens in flight), so run nothing else on the card.
- **The software** from the README's [Install](../README.md#install) section: Python 3.12, PyTorch 2.14.1, transformers 5.20.2, and Prism's llama.cpp fork to run the result.
- **Disk.** About 18-20 GB for your BF16 checkpoint, a few hundred MB of datasets in the Hugging Face cache, and 10-15 GB for the run directory. Plan for 40-50 GB free.
- **Time.** The published recipe took 4 h 23 min end to end on one RTX 5090, including two restarts of the traces stage. Measured per stage, from `research/runs/2026-10-27-reasoning-mix-v1/logs/`:

  | Stage | Measured |
  |---|---|
  | traces: 1,520 prompts, 1.54M generated tokens | 2 h 34 min (2.6 h of generation) |
  | calibrate | under 1 min |
  | quantize: sequential GPTQ on 256 × 2,048 tokens | 16 min |
  | endpoints: head and embedding | 2 min |
  | distill: teacher pass on 1,536 sequences, then 192 training steps | 6 min + 34 min |
  | export and verification | 2 min |
  | eval: dev, extended, compare, GSM8K and HumanEval in both thinking modes | 29 min (25 of them with thinking on) |

  Your own prompts add trace time on top (estimate it in step 3).

**Sizes.** Qwen3.5-9B has a separate embedding and LM head, which the GGUF needs. The 0.8B, 2B and 4B models tie the two (one shared table), and they cannot be written as a GGUF yet: a fine-tune of one of them is not supported. Other Qwen3.5 sizes have not been run.

## 1. Prepare the checkpoint

You need a full, merged checkpoint in safetensors:

- **Merged.** A LoRA adapter folder is not enough. Merge the adapter into the base weights first (with PEFT, `merge_and_unload()`), and save the result with safetensors.
- **Qwen3.5.** `config.json` has `model_type` `qwen3_5` (the original layout, with the vision tower) or `qwen3_5_text` (text only). The vision tower is dropped either way; the GGUF holds the language model.
- **Untied.** `tie_word_embeddings` is false (top level and `text_config`).
- **Not quantized.** A config with `quantization_config` is refused. Start from BF16 weights, not from a 4-bit or 8-bit export.
- **safetensors.** The GGUF export reads `model.safetensors` or `model.safetensors.index.json`, so a run with a `.bin`-only checkpoint is refused before the first stage. A Hugging Face repo whose only weights are a single `pytorch_model.bin` is not detected until the export stage, so convert it first.
- **Tokenizer and chat template.** The folder holds the tokenizer files and the chat template your fine-tune uses. The traces stage refuses a model without a chat template.

A quick check of the first four:

```
python -c "import json; c = json.load(open('my-finetune/config.json')); t = c.get('text_config', {}); print(c['model_type'], c.get('tie_word_embeddings'), t.get('tie_word_embeddings'), 'quantization_config' in c)"
```

The output should read `qwen3_5 False False False` (or `qwen3_5_text ...`; a missing key prints `None`, which also counts as untied).

A Hugging Face id works in place of a folder. Private or gated repositories need `hf auth login` first.

## 2. Copy the recipe

```
cp examples/qwen3_5-9b-reasoning.yaml my-finetune.yaml
```

Point it at your model. The example pins Qwen's commit, so clear the revision (or give your own repo's commit):

```yaml
model:
  path: path/to/my-finetune     # a local folder or a Hugging Face id
  revision: null
```

Two settings differ between the example and the run that made the published model (`research/runs/2026-10-27-reasoning-mix-v1/run.yaml`):

- `traces.batch_tokens: 14000` in the run, 32,768 in the example. At 32,768 the first GSM8K batches held 16 prompts and decoded at about 113 tokens/s. The run was restarted with 14,000: its GSM8K batches held 8 prompts and decoded at 127-215 tokens/s (mean 173). Use 14,000 on a 32 GB card.
- `distill.train.small: false` in the run: it trained only the group scales. The example's `true` also trains the small high-precision tensors (norms, `conv1d`, `A_log`, `dt_bias`, `in_proj_a/b`); on the generic recipe that moved dev NLL from 2.892 to 2.884, within noise. Set `false` to match the published model.

The rest of the recipe (rotation, GPTQ, endpoints, distillation schedule, export) applies to a fine-tune unchanged. Read the comments at the top of the recipe: they explain every data choice below.

## 3. Choose the data

### Your task's prompts: the main lever

Add your fine-tune's task prompts as one more `traces.sources` entry of kind `jsonl`. Your fine-tune answers them itself, greedily, in its own chat template, and those answers go into the calibration blocks and the distillation data. The ScaleQ paper calls this AYOT ("attend to your own thoughts"): the compressed model is fitted on the text the model actually produces.

This is the lever with the most evidence behind it. Calibrated on generic text, the 9B model kept perplexity but scored 30/100 on GSM8K and 6/50 on HumanEval (thinking off). Calibrated on its own answers to 1,520 math and code training prompts, the same pipeline scored 69/100 and 19/50. Distillation only sees its data, so a skill your fine-tune added that is not in the data is not protected.

The prompt file has one JSON object per line, with `prompt` (or `question`):

```
{"prompt": "Summarize this support ticket in one sentence: the printer on floor 3 jams after every second page."}
```

or `messages`, for a system prompt or a multi-turn context:

```
{"messages": [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "Translate 'good morning' into Turkish."}]}
```

The messages are rendered as written, with the generation prompt appended, so they should end with a user turn. Put only prompts in this file, not answers: the model writes the answers. A file with another field name works with `column: <field>`.

Then add the source at the end of `traces.sources`:

```yaml
traces:
  seed: 0
  batch_tokens: 14000
  sources:
    # ... the recipe's seven sources ...
    - {kind: jsonl, name: my_task, path: data/my-task-prompts.jsonl, n: 400, seed: 7, thinking: off, max_new_tokens: 1024, batch: 16}
```

- **Prompts.** Use prompts like the ones your fine-tune will see: the user side of its training data, or real requests. Leave your evaluation prompts out; nothing checks your own test sets (GSM8K and HumanEval test problems are refused everywhere, in your files too).
- **`n`.** The number of prompts, drawn in a seeded order (`seed`). Duplicates are dropped, so `n` can be at most the number of distinct prompts. `shuffle: false` takes them in file order. Longer prompts can be skipped with `max_chars`.
- **`thinking`.** Match how the fine-tune is used. If it is used both ways, add the file twice with different `name`s and seeds, one `on` and one `off`, as the recipe does for GSM8K and MBPP.
- **`max_new_tokens`.** A little above your fine-tune's usual answer length. An answer cut at the cap still counts, but it loses its ending. `traces/summary.json` lists `hit_cap` per source; with thinking on, a cut answer often has no final answer at all.
- **`batch`.** Prompts per generate call; `traces.batch_tokens` lowers it for long prompts.

**Keep the recipe's sources?** They are what raised GSM8K from 30 to 69 and HumanEval from 6 to 19. Keep them if your fine-tune should still reason and write code. If it does not need math or code, you can shrink or drop them, and your task then gets a larger share of the trace blocks. Either choice is untested on a fine-tune.

**Time.** A batch runs until its longest answer stops or reaches the cap, so the time is about the number of batches × the cap ÷ the decode rate. In the published run, batches of 8 decoded 17-24 steps/s (a 1,536-token batch took 65-90 s). 400 prompts with a 1,024-token cap in batches of about 11 (what 14,000 tokens allows) are 37 batches × 1,024 ÷ 20 ≈ 32 minutes. The log shows tokens/s and an ETA after every batch.

### Block arithmetic

Calibration and distillation read blocks of 2,048 tokens (`calibrate.seq_len`, `distill.teacher.seq_len`). The traces of all sources are interleaved in `traces/traces.jsonl` in a seeded order, concatenated whole, and cut into blocks. Calibrate's `{kind: traces, n: 128}` and the teacher's `{kind: traces, n: 640}` both read from the start of that stream. Three rules follow:

1. **Supply.** The traces must make at least as many blocks as the largest trace count: trace tokens ÷ 2,048 ≥ 640 in the recipe. The published traces held 1,736,153 tokens, or 847 blocks. Blocks are never capped silently: a run that cannot supply them stops right after the traces stage, before quantize, and names each count.
2. **Share.** Every prefix of the stream holds each source in proportion to its tokens, so your task's share of the trace blocks is its share of the trace tokens.
3. **Calibration total.** The total `n` of `calibrate.sources` must be a multiple of `quantize.batch` (4). A total that is not is refused before the first stage, with the `--set` that fixes it.

A worked example: 400 prompts of about 200 tokens with answers of about 600 make 400 × 800 = 320,000 tokens, about 156 blocks. With the recipe's sources (1,736,153 tokens) that is 2.06M tokens, about 1,000 blocks, and your task is 16% of them: about 20 of calibrate's 128 trace blocks and 100 of the teacher's 640 trace sequences. To give it more weight, add prompts, shrink the other sources, or move teacher sequences from UltraChat to traces (for example traces 768 and UltraChat 512, if the traces make at least 768 blocks).

You do not have to guess the token counts. Run the traces stage alone first (step 4): `traces/summary.json` has `prompt_tokens` and `generated_tokens` per source, and the log ends with one line per trace count, such as `distill.teacher.sources.2.n: 640 of the 847 blocks of 2048 tokens the traces make`. Then set the counts and continue; the traces are not generated again.

### Calibration

The recipe's 256 calibration blocks are 64 WikiText-2 train, 64 UltraChat and 128 traces. Your task enters through the traces, so this list can stay as it is.

GPTQ holds the inputs of the layer being rounded for every block, about 4.3 GB per copy at 256 blocks (two copies coexist), so keep the total near 256: swap blocks between sources instead of adding them.

### Distillation

The recipe's teacher data is 1,536 sequences: 256 WikiText-2 train, 640 UltraChat and 640 traces. The teacher's targets are your BF16 model's own top-64 predictions on those sequences, so what matters is which text it sees. Each extra 8 sequences add one training step (about 10.5 s) and about 2 s of teacher pass. `distill.teacher.n_seq` is used only when `sources` is unset; keep it equal to their sum so the file stays readable.

### Your own reference data (optional)

If you have conversations with reference answers (your SFT data), you can also add them to calibration and distillation as `kind: jsonl`. Each line holds `messages` (rendered with your chat template, thinking off), `text` or `input_ids`; the lines are read in file order (`shuffle: true` shuffles them with the source's `seed`), concatenated and cut into blocks, so `n` blocks need `n` × 2,048 tokens. For example, swapping 32 UltraChat calibration blocks and 128 UltraChat teacher sequences for your data:

```yaml
calibrate:
  seq_len: 2048
  seed: 0
  sources:
    - {kind: wikitext2, n: 64}
    - {kind: ultrachat, n: 32, min_row: 6000}
    - {kind: jsonl, path: data/my-sft.jsonl, n: 32, shuffle: true}   # 65,536 tokens
    - {kind: traces, n: 128}
distill:
  teacher:
    n_seq: 1536
    seq_len: 2048
    seed: 1
    shard: 64
    topk: 64
    sources:
      - {kind: wikitext2, n: 256}
      - {kind: ultrachat, n: 512, min_row: 6000}
      - {kind: jsonl, path: data/my-sft.jsonl, n: 128, shuffle: true}   # 262,144 tokens
      - {kind: traces, n: 640}
```

Setting `distill.teacher.sources` replaces the whole list, so list every source. This is a complement to the trace source, not a replacement: the traces hold what your model actually writes.

### A held-out set for step 5

Keep some of your task's conversations out of all of the above, as a JSONL file in the same format, and add it to the config now, so the run's own eval scores it:

```yaml
eval:
  compare:
    seq_len: 512
    n: 8
    seed: 7
    topk: 64
    full: true
    sources:
      - {kind: jsonl, path: data/my-task-heldout.jsonl, n: 8}   # 8 blocks of 512 tokens: about 4,100 tokens
```

## 4. Run, check and resume

Generate the traces first and check them:

```
openbonsai run --config my-finetune.yaml --out runs/mine --until traces
```

Look at `traces/summary.json` (tokens and `hit_cap` per source) and at a few records of `traces/traces.jsonl` (`prompt` and `completion` show what your model wrote). Fix the counts if the supply lines say so, then run the rest:

```
openbonsai run --config my-finetune.yaml --out runs/mine
```

- **Resume.** Rerun the same command, or `openbonsai run --out runs/mine`, which reads the run's saved `run.yaml`. Stages that are current are skipped; an interrupted stage resumes (traces per batch, quantize per layer, distill per teacher shard and checkpoint).
- **Edit and rerun.** After a config edit, only the stages whose inputs changed run again. Trace batches are cached by content, so adding a source generates only that source. Changing `traces.batch_tokens` re-plans the batches, and batches made under the old budget are generated again.
- **`--until STAGE`** stops after a stage; `--until export` skips the eval.
- **`--force STAGE`** reruns a stage from scratch, and every later one. `--force traces` deletes the trace cache: hours of generation.
- **`--set SECTION.KEY=VALUE`** overrides one value; a number indexes a list (`--set calibrate.sources.2.n=128` sets the recipe's trace blocks). Overrides are saved in `run.yaml`.
- **`openbonsai inspect runs/mine`** shows which stages have finished, with their results; `openbonsai inspect runs/mine/export/<name>.gguf` summarizes the file.
- **Errors** print one line; `--debug` shows the traceback, which a failed stage also writes to `runs/mine/logs/<stage>.log`. The run checks many settings before the first stage starts.

The result is `runs/mine/export/OpenBonsai-<folder name>-PTQ1_0.gguf`, verified after it is written.

## 5. Evaluate against your own BF16 model

The run's eval stage scores the distilled model with the recipe's suites:

- **`compare`** scores it against your BF16 model on held-out blocks: the NLL of both, top-1 agreement, full-vocabulary KL and expected acceptance rate ([docs/evaluation.md](evaluation.md#teacher-vs-student-metrics)). It works with any tokenizer. With `eval.compare.sources` set (step 3) it scores your task's data; without it, WikiText-2 test. Its BF16 logits are kept on disk, about 2 GB for 8 × 512 tokens.
- **`gsm8k` and `humaneval`** generate with your chat template on frozen test subsets (100 and 50 problems with thinking off; the first 50 and 25 with thinking on). They work for any model with a chat template. HumanEval runs model-written code, and its limits are not a security sandbox ([docs/evaluation.md](evaluation.md#reasoning-suites-gsm8k-humaneval)).
- **`dev` and `extended`** run only if your tokenizer is Qwen3.5's, unchanged; otherwise they are skipped. `extended`'s chat NLL scores base Qwen3.5-9B's answers, not your model's.

Score your BF16 model on the same reasoning suites, so each number has its reference next to it (with thinking on, this takes longer than for the compressed model):

```
openbonsai eval --run runs/mine --from bf16 --suite gsm8k,humaneval
```

Score the GGUF file itself, as written:

```
openbonsai eval --run runs/mine --from runs/mine/export/OpenBonsai-my-finetune-PTQ1_0.gguf --suite compare,gsm8k,humaneval
```

Results are in `runs/mine/eval/<source>.<suite>.json`. For reference, base Qwen3.5-9B with this recipe: GSM8K 69/100 and HumanEval 19/50 with thinking off (BF16: 77/100 and 44/50), 25/50 and 10/25 with thinking on (BF16: 36/50 and 22/25).

**Your own task.** OpenBonsai has no suite for your task's generations yet. Generate answers to your held-out prompts with both models and score them with your own checker: the GGUF through `llama-server` (step 6), your BF16 model with the tool you already use. Use greedy decoding (temperature 0) and the same prompts for both. Prism's runtime quantizes activations to 8 bits, so its output differs slightly from PyTorch's.

## 6. Run the GGUF in Prism's llama.cpp

The file needs Prism's fork ([PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), tested at `c9c34bc`); upstream llama.cpp does not know PTQ1_0. Build it as the README's [Install](../README.md#install) section shows, then:

```
B=third_party/prism-llama.cpp-build/cpu/bin
$B/llama-cli -m runs/mine/export/OpenBonsai-my-finetune-PTQ1_0.gguf                  # chat
$B/llama-server -m runs/mine/export/OpenBonsai-my-finetune-PTQ1_0.gguf --port 8080   # OpenAI-compatible HTTP API
```

On Windows the binaries are `build\bin\Release\llama-cli.exe` and `llama-server.exe` in the Prism checkout. The GGUF carries your tokenizer and chat template.

## 7. Publish

- **Name.** The file is named `OpenBonsai-<folder name>-PTQ1_0.gguf`. Set your own name with `--set export.name=My-Model-PTQ1_0` (it names the file and `general.name`); some base licences require a name. Do not present it as a Prism ML model: no "Ternary Bonsai" in the name, no Prism branding.
- **Licence.** The compressed model is a derivative of your fine-tune and of Qwen3.5-9B (Apache-2.0), and keeps their licences.
- **Model card.** Say what the file is and what it needs: a ternary PTQ1_0 GGUF of your fine-tune that runs only in Prism's llama.cpp fork. Name the recipe and the data, including your prompt source. Put the compressed model's scores next to your BF16 model's on the same suites, with the caveats (small subsets, greedy decoding, one run). Include the attribution and the non-affiliation line:

  ```
  Compressed with OpenBonsai (https://github.com/esokullu/openbonsai) to Prism ML's PTQ1_0 ternary format; runs in Prism's llama.cpp fork.
  OpenBonsai is an independent project, not affiliated with or endorsed by Prism ML. Bonsai is Prism ML's product.
  ```

- **Upload.** For example `hf upload <you>/<repo> runs/mine/export/My-Model-PTQ1_0.gguf`.

## When a run refuses

| Message | What to do |
|---|---|
| `calibrate.sources ask for N blocks in all, not a multiple of quantize.batch (4)` | Change one count; the message gives the `--set` |
| `traces/traces.jsonl is too short: ... N blocks ... but ... make M` | Lower the trace counts to at most M (move the difference to another source), or make more traces |
| `traces.sources.N: only K usable prompts, n requested` | Lower that source's `n`, or add prompts |
| `start from unquantized weights (config has quantization_config)` | Use BF16 weights |
| `tied embedding/head` or `ternary for a tied table` | A tied model (0.8B, 2B, 4B): not supported yet |
| `the model's tokenizer has no chat template` | Add your chat template to the tokenizer files |
