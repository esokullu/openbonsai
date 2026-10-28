# OpenBonsai

**Research preview.** The method was measured end to end on Qwen3.5-9B, first with the research scripts and then with the package's own `openbonsai run`. The results are early and the evaluation is small. Calibrating on the model's own math and code answers brought GSM8K close to the official Ternary Bonsai; code is still well below it (see [Results](#results)). The current model is on Hugging Face: [webbrain-one/openbonsai-qwen-3.5-9b](https://huggingface.co/webbrain-one/openbonsai-qwen-3.5-9b).

Independent project, not affiliated with or endorsed by Prism ML. Bonsai is Prism ML's product.

OpenBonsai is an open (Apache-2.0) framework that compresses a dense language model to ternary weights in the format of Prism ML's Ternary Bonsai models: Prism's PTQ1_0 GGUF, which runs in Prism's llama.cpp fork. For Qwen3.5-9B the output is a file of about 2 GB, with the same tensor layout as Ternary-Bonsai-2-9B.

**Why.** Prism publishes finished Bonsai weights for the base models it chooses, and an inference runtime. It has described the format (ternary weights, FP16 group scales, a blockwise Hadamard rotation) but, as far as we know, not a pipeline to compress a model of your own, and Ternary Bonsai 2 ships packed ternary files only, with no BF16 weights. So community fine-tunes adapt Bonsai itself: LoRA adapters on the frozen ternary base, tuning of its scales and norms, ternary quantization-aware training ([overview](docs/your-own-model.md#other-ways-to-adapt-a-bonsai-model)). They keep Prism's stronger compression, but the fine-tune has to be redone on Bonsai, and an adapter adds FP16 weights at inference. OpenBonsai is the other route, in the open: it compresses your own BF16 model or fine-tune, with its own tokenizer and chat template, to a fully ternary file in Bonsai's format, and measures what compression cost against your own BF16 model. The same pipeline can reach models Prism has not compressed, such as Meta's Muse Glimmer 30B, once they have an adapter ([Other models](#other-models)).

**Status.**
- Qwen3.5-9B: measured end to end, with the research scripts and with the package's own `openbonsai run` (results below).
- Dense Llama, Mistral, Qwen2/2.5 and Qwen3: implemented in a generic adapter, tested only on tiny random models on CPU, including loading tiny Llama and Qwen3 GGUFs in Prism's runtime. There are no real-model results for them yet. Whether Prism's runtime can run the result depends on the architecture (see [Other models](#other-models)).
- ScaleQ/CAT-Q refinement is in `research/`, not in the pipeline: it has not beaten GPTQ plus distillation yet.
- An optional `refine` stage that trains GPTQ's codes and scales layer by layer ([docs/refine.md](docs/refine.md)) is implemented and off by default. On 9B it lowered the error before distillation, but after distillation its reasoning scores were within noise of the model without it (below), so it stays off.

## How it works

```
your BF16 model ──► traces: it answers prompts in its own chat template (AYOT)
        │
        ▼
calibration blocks: WikiText-2 train + UltraChat + its own traces (+ your JSONL)
        │
   1 rotate      W·S·H: one ±1 sign vector per input width, Hadamard blocks of 4096
   2 round       sequential GPTQ to {-1, 0, +1} codes, fp16 scale per 128 weights
  (3 refine)     ScaleQ/CAT-Q: research/ only, not in the default pipeline
     endpoints   ternary LM head and token embedding (GPTQ)
    [refine]     optional, off: train codes + scales per layer (docs/refine.md)
   4 distill     group scales against the BF16 model's top-64, codes frozen
        │
        ▼
export: PTQ1_0 GGUF (~2 GB for 9B) ──► Prism's llama.cpp fork
```

**1. Rotation.** Ternary codes with one scale per group cannot represent a few large weights next to many small ones. A signed Hadamard rotation spreads every input's weight across a block, so the values in a group look alike. We use Prism's layout exactly: the file stores W·S·H (S holds one ±1 sign vector per input width, shared by every tensor of that width; H is the normalized Hadamard matrix in blocks of 4096). The runtime rotates the activations to H(s⊙x), so the rotation adds no weights.

**2. Sequential GPTQ ternary rounding.** Layer by layer, in data-flow order inside each layer, every linear layer is rounded to codes {-1, 0, +1} in groups of 128 rotated input weights, with one fp16 scale per row and group. GPTQ pushes each column's rounding error onto the columns not yet rounded, using the second moment (Hessian) of the layer's real inputs. Those inputs come from the already-quantized layers below, on calibration data that includes the model's own answers (AYOT, "attend to your own thoughts", from the ScaleQ paper). This step does most of the work: dev NLL goes from 8.94 (rotated, rounded without calibration data) to 3.32.

**3. ScaleQ/CAT-Q refinement (research only).** CAT-Q learns a center, a threshold and a scale per group by block-wise reconstruction, as in the ScaleQ-1.58 paper. In our runs on small budgets it did not reach usable quality on its own, and it has not yet beaten GPTQ plus distillation on the same data, so it stays in [research/](research/README.md).

**4. Scale distillation.** The codes stay frozen. Every (row, group) scale gets a trainable multiplier, trained through the whole model against the BF16 model's top-64 next-token probabilities plus one bucket for the remaining mass. This reduces error that builds up across layers, which GPTQ's layer-by-layer objective cannot see. On 4.2M tokens it takes the all-ternary model from dev NLL 3.48 to 2.89 and ARC-Easy from 13/20 to 19/20.

**Endpoints and export.** The "endpoints" are the LM head and the token embedding. The LM head is rounded by GPTQ with the Hessian of the final hidden states. The token embedding has no input Hessian (its inputs are one-hot), so its GPTQ weights each dimension by the square of layer 0's norm gain, which is what the first layer sees. The export writes Prism's PTQ1_0 GGUF: 28 bytes per 128 weights (1.75 bits per weight, scale included), the `prism.hadamard.*` metadata, and the model's tokenizer and chat template. Every file is verified after it is written.

Details, evidence and what did not work: [docs/method.md](docs/method.md). The file format: [docs/formats.md](docs/formats.md).

## Quickstart

### Install

Python 3.12, PyTorch 2.14.1 and transformers 5.20.2 (the versions the results were measured with). Linux with an NVIDIA GPU:

```
scripts/setup_linux.sh               # venv, CUDA PyTorch, this package (editable), optional FLA kernels
scripts/setup_linux.sh --prism       # also clone Prism's llama.cpp at the pinned commit and build it (CPU)
source .venv/bin/activate            # puts `openbonsai` on the PATH
```

`scripts/setup_linux.sh --help` lists the options (CUDA index, `--prism-cuda`, `NO_FLA=1`, and more). By hand:

```
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
pip install transformers==5.20.2 -e ".[test]"
```

**Windows.** The measured runs used Windows 11 with one RTX 5090. In PowerShell:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
.venv\Scripts\python -m pip install transformers==5.20.2 -e ".[test]"
.venv\Scripts\Activate.ps1           # puts `openbonsai` on the PATH
```

- To hide the GPU, set `CUDA_VISIBLE_DEVICES=-1`. An empty value unsets the variable on Windows.
- flash-linear-attention is optional. Without it, transformers runs Qwen3.5's Gated DeltaNet layers (its linear-attention layers, which carry a recurrent state) in plain PyTorch: slower, same results up to rounding.
- Prism's llama.cpp builds with the Visual Studio 2022 Build Tools and CMake, from a developer prompt. The pinned commit is `c9c34bc` of [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp):
  ```
  git clone https://github.com/PrismML-Eng/llama.cpp prism-llama.cpp
  cd prism-llama.cpp
  git checkout --detach c9c34bc9c4042c13a31598ebf43461cd0ad8f00d
  cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
  cmake --build build --config Release
  ```
  The binaries land in `build\bin\Release\` (`llama-cli.exe`, `llama-server.exe`, `llama-perplexity.exe`). The `B=...` example below is bash; in PowerShell, call `build\bin\Release\llama-cli.exe -m <file>.gguf` directly.
- The HumanEval suite limits the memory of each test program with a Windows Job Object.

### Try it on CPU first

A tiny random Qwen3.5 runs the whole pipeline on a CPU in under a minute (18 s on our machine), with no downloads:

```
python -c "import sys; sys.path.insert(0, 'tests'); from tiny_models import save_tiny_qwen3_5; save_tiny_qwen3_5('runs/tiny-model')"
openbonsai run --model runs/tiny-model --config examples/tiny-qwen3_5.yaml --out runs/tiny
openbonsai inspect runs/tiny/export/OpenBonsai-tiny-model-PTQ1_0.gguf
```

### Qwen3.5-9B to a 2 GB GGUF

```
openbonsai run --config examples/qwen3_5-9b.yaml --out runs/qwen3_5-9b
```

This downloads Qwen3.5-9B and the calibration data at pinned revisions, then runs every stage into one run directory: traces, calibrate, quantize, endpoints, distill, export and eval. The result is `runs/qwen3_5-9b/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf`. Each stage is skipped when its output is current and resumes after an interruption (per layer, per teacher shard, per checkpoint). `--until STAGE`, `--force STAGE` and `--set SECTION.KEY=VALUE` control a run. A run checks many settings that a later stage would refuse before the first stage starts, and errors print one line (`--debug` shows the traceback, which a failed stage also writes to `logs/<stage>.log` in the run directory); `openbonsai inspect runs/qwen3_5-9b` shows what has finished.

### Calibrating on the model's own math and code answers

```
openbonsai run --config examples/qwen3_5-9b-reasoning.yaml --out runs/qwen3_5-9b-reasoning
```

The same pipeline, with traces from several prompt sources (`traces.sources`), each with its own sample count, seed, thinking switch, token cap and batch: about 1,520 prompts from training splits (GSM8K, MATH, MBPP, OpenCodeInstruct with thinking on and a 1,536-token cap; some GSM8K and MBPP with thinking off; UltraChat with thinking off). Half of the calibration blocks and 42% of the distillation tokens come from these traces. These are the settings of the run behind the Hugging Face model. Measured on one RTX 5090: 1.54M generated tokens in 2.6 h, and about 4.1 h for the whole run (4 h 23 min with two restarts of the traces stage); the config's comments break it down, and `traces.batch_tokens` keeps each batch within the card's memory. The traces stage caches every batch, so an interrupted run resumes; test problems and near duplicates of them are kept out, but not reworded versions of a test task ([docs/evaluation.md](docs/evaluation.md#keeping-test-data-out)). The prompt kinds and record fields are documented in `openbonsai/stages/traces.py`. Results are in the table below (the "self-generated math and code" row) and in `research/runs/2026-10-27-reasoning-mix-v1/`. `examples/qwen3_5-9b-reasoning-code.yaml` adds more code to the same data and scores 34/50 on HumanEval instead of 19/50, at about 1 h more GPU time ([below](#more-code-in-the-data)).

### Your own fine-tune

```
openbonsai run --model path/to/my-finetune --config examples/qwen3_5-9b-reasoning.yaml --out runs/mine
```

Start from the reasoning recipe: it made the Hugging Face model. Your model is the teacher: it writes its own traces with its own chat template, it is the BF16 reference for every metric, and its tokenizer and chat template go into the GGUF. To keep the skill your fine-tune added, copy the recipe and add your task's prompts as one more trace source, such as `- {kind: jsonl, name: my_task, path: my-prompts.jsonl, n: 400, thinking: off, max_new_tokens: 1024}`. Your model then answers them itself, and those answers go into calibration and distillation.

It needs a full, merged Qwen3.5-9B checkpoint in safetensors (not a LoRA adapter, not quantized), one 32 GB GPU and 4 to 4.5 hours (4 h 23 min for the run behind the Hugging Face model). The tied sizes (0.8B, 2B, 4B) cannot be written as a GGUF yet. For a Hugging Face id instead of a local folder, also pass `--set model.revision=null` (or your commit), because the recipe pins Qwen's commit. `compare` scores the result against your own BF16 model; `gsm8k` and `humaneval` work for any model with a chat template.

Step by step, from the checkpoint to the model card: [docs/your-own-model.md](docs/your-own-model.md).

### Run the GGUF in Prism's llama.cpp

The file needs Prism's fork; upstream llama.cpp does not know PTQ1_0. With the build from `scripts/setup_linux.sh --prism`:

```
B=third_party/prism-llama.cpp-build/cpu/bin
$B/llama-cli -m runs/qwen3_5-9b/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf          # chat
$B/llama-server -m runs/qwen3_5-9b/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf --port 8080
$B/llama-perplexity -m runs/qwen3_5-9b/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf -f wiki.test.raw -c 512 --chunks 8
```

`wiki.test.raw` is the WikiText-2 raw test file, as upstream llama.cpp's `scripts/get-wikitext-2.sh` downloads it. The official model to compare against is `Ternary-Bonsai-2-9B-PTQ1_0.gguf` from [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) (we used revision `5b2abb0`).

### Evaluate

```
openbonsai eval --run runs/qwen3_5-9b --from distill --suite dev,extended,compare
openbonsai eval --run runs/qwen3_5-9b --from distill --suite gsm8k,humaneval   # humaneval executes model-written code: not a sandbox
openbonsai eval --config examples/qwen3_5-9b.yaml --from runs/qwen3_5-9b/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf --suite dev,compare
python -m pytest            # CPU test suite: tiny random Qwen3.5, Llama, Qwen3 and Qwen2 run the whole pipeline
```

`--from` takes a stage (`bf16`, `quantize`, `endpoints`, `distill`) or a file: a state, or a PTQ1_0 GGUF of an architecture the adapters cover (`qwen35`, `llama`, `qwen3`), with that model's config; the official Ternary Bonsai 2 9B file works with the 9B example config. `gsm8k` and `humaneval` generate with the model's chat template; `humaneval` executes model-written code, and its limits are not a security sandbox (see [docs/evaluation.md](docs/evaluation.md#reasoning-suites-gsm8k-humaneval)). `dev` and `extended` need the Qwen3.5 tokenizer; `compare` scores any model against its own BF16 original (NLL, top-1 agreement, full-vocabulary KL and expected acceptance rate). Suites, metrics and the test-data guard: [docs/evaluation.md](docs/evaluation.md).

## Results

Qwen3.5-9B. The dev suite is frozen: 2,048 WikiText-2 test tokens and 20 ARC-Easy questions. The extended suite: 16,384 WikiText-2 test tokens, 100 ARC-Easy questions, and NLL on 16 held-out answers the BF16 model generated itself ("chat"). Lower NLL is better.

| Model | Endpoints | Dev NLL | ARC /20 | Wiki 16K NLL | ARC /100 | Chat NLL | GSM8K off / on | HumanEval off / on |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B BF16 | BF16 | 2.423 | 19 | 2.218 | 97 | 0.554 | 77/100 / 36/50 | 44/50 / 22/25 |
| Ternary Bonsai 2 9B (official, Prism ML) | ternary | 3.191 | 17 | | | | 77/100 / 46/50 | 45/50 / 24/25 |
| Plain ternary rounding (TWN), no rotation | BF16 | 10.2 | 9 | | | | | |
| Rotation + least-squares rounding (RTN) | BF16 | 8.94 | 4 | | | | | |
| Rotation (signs per module) + GPTQ, 128 / 256 calibration blocks | BF16 | 3.32 / 3.30 | 9 / 13 | | | | | |
| Same, one sign vector per width (Prism's layout) | BF16 | 3.318 | 15 | 3.103 | 78 | 1.964 | | |
| + ternary head | head ternary | 3.383 | 16 | 3.167 | 74 | 2.051 | | |
| + ternary embedding: all ternary, ~2 GB | ternary | 3.480 | 13 | 3.281 | 62 | 2.225 | | |
| **+ scale distillation (OpenBonsai)** | ternary | **2.892** | **19** | **2.626** | **87** | **1.318** | **30/100 / 19/50** | **6/50 / 3/25** |
| + small high-precision tensors trained too (norms, `conv1d`, `A_log`, `dt_bias`, `in_proj_a/b`; `distill.train.small`) | ternary | 2.884 | 19 | 2.616 | 87 | 1.308 | | |
| **Calibrated on self-generated math and code** (`examples/qwen3_5-9b-reasoning.yaml`; the Hugging Face model) | ternary | 2.960 | 17 | 2.698 | 84 | 1.439 | **69/100 / 25/50** | **19/50 / 10/25** |
| + all 32 layers refined before distillation (`refine.enabled`; [below](#refining-codes-not-only-scales)) | ternary | 2.950 | 18 | 2.694 | 89 | 1.443 | 66/100 / 24/50 | 19/50 / 10/25 |
| More code in the traces, calibration and distillation (`examples/qwen3_5-9b-reasoning-code.yaml`; [below](#more-code-in-the-data)) | ternary | 3.012 | 18 | 2.763 | 89 | 1.459 | 65/100 / **32/50** | **34/50 / 14/25** |

In Prism's own llama.cpp runtime (CPU build at c9c34bc, upstream llama.cpp's WikiText-2 `wiki.test.raw`, 8 chunks of 512 tokens), perplexity is 17.21 for the official Ternary Bonsai 2 9B and 12.88 for ours (the bold scale-distillation row, exported as a GGUF). Prism's runtime agrees with our PyTorch reference of the same weights, mean KL 0.0028 nats and 95% top-1 agreement over 508 tokens, on the exporter's synthetic test state (not a usable model; `research/runs/gguf-test/runtime-parity/`); that check has not been run on the distilled model. Prism quantizes activations to 8 bits, so the two are not identical. The self-generated math and code model scores 13.61 in the same runtime. All three GGUF files are 1,997,486,976 bytes. The official model's dev NLL is 3.19 in Prism's CPU runtime with ARC 16/20, and 3.191 with ARC 17/20 in our PyTorch harness. The export report is in `research/runs/2026-10-27-gguf-shared-v1/`. The perplexity logs there were run on a copy of the test file with the blank lines removed and CRLF line endings, which gave 16.51, 11.93 and 12.59 (the numbers this README gave before; the order of the models is the same). The logs on the upstream file are in `research/runs/2026-10-28-prism-ppl-upstream/`.

**Read these numbers with the caveats:**
- **WikiText is partly in-domain for us.** Our calibration and distillation data include WikiText-2 *train*; the dev, Wiki 16K and Prism perplexity numbers are on WikiText-2 *test*. Same source, different articles. We do not know Prism's training data, so on WikiText the comparison with the official model favors us. ARC is in neither our calibration nor our distillation data. Our data also contains other answers generated by the same model, so the chat NLL is in-distribution by design. (Trace prompts and training data that contain one of the 16 chat-NLL prompts or answers are refused, and the measured runs used a disjoint trace set. See [docs/evaluation.md](docs/evaluation.md#keeping-test-data-out).)
- **The evaluation is small.** One ARC question is 5 percentage points on the dev suite. The dev suite was used throughout development, so it is not a fresh held-out set.
- **Single runs, no variance.** Every row is one run. Treat differences of one or two ARC questions as noise: re-scoring the same 256-block GPTQ state with the package moved ARC from 13 to 14 (below).
- **Code and non-English output are the main weakness; math improved a lot.** Calibrated on generic text (WikiText, UltraChat, a few general self-generated answers), the model kept perplexity but lost math and code: 30/100 GSM8K and 6/50 HumanEval with thinking off, against 77/100 and 45/50 for the official Ternary Bonsai 2 9B and 77/100 and 44/50 for BF16. This matches the ScaleQ paper's finding. Recalibrating on 1,520 of the model's own answers to math and code training prompts (the AYOT idea) raised that to 69/100 and 19/50, at a small cost in WikiText NLL and ARC. Code is still far behind (more code in the data raises HumanEval to 34/50, [below](#more-code-in-the-data)), the Turkish dev prompt still collapses into repetition, and with thinking on the model often reasons past the 4,096-token cap (23/50 GSM8K answers).
- **Reasoning evals are small and capped.** GSM8K uses a frozen 100-problem subset and HumanEval 50 (thinking off, 512-token cap); with thinking on, the first 50 and 25 of the same subsets, 4,096-token cap, batch 8. With thinking on, 14/50 BF16 GSM8K answers and 4/50 official-model answers ran out of tokens mid-thought, so the cap understates thinking-mode accuracy (most for BF16). The generic-calibration model ran out on 19/25 HumanEval problems with thinking on, mostly by looping; the self-generated math and code model on 5/25. Greedy decoding, the model's own chat template, one run each. Every GSM8K and HumanEval score, the official model's included, comes from OpenBonsai's PyTorch harness on the same weights, not from Prism's runtime. Qwen3.5's chat template turns thinking on unless the caller turns it off, and each score says which mode it used. Results: `research/runs/2026-10-27-reasoning-v1/` and, for the self-generated math and code model, `research/runs/2026-10-27-reasoning-mix-v1/eval/`.
- **Only Qwen3.5-9B has run on real weights.** The runtime checks in Prism's llama.cpp (perplexity, KL, file layout) exist only for the 9B. No real dense model (Llama, Mistral, Qwen3) has been compressed, and no real dense GGUF has been run in Prism.
- **Where the numbers come from.** They were measured with the research scripts in `research/experiments/`. The package ports their arithmetic, and tests check the ports bit for bit on CPU. Given the measured state, the package's exporter writes a GGUF byte-identical to the measured one (`research/validation/p2-export-parity.md`). GPU numbers can still differ slightly: a package re-score of the 256-block GPTQ state gave dev NLL 3.293 and ARC 14/20, against 3.295 and 13/20 from the research script (`research/runs/2026-10-27-accept-256/eval/`). The package's own `openbonsai run` of the 9B example (traces off, the measured run's calibration data) reproduced it on the GPU: its quantize stage is bit-identical to the research GPTQ state (0 code or scale differences in 6.9B weights; dev NLL 3.3183, ARC 15/20), and the full run through endpoints, distillation, verified export and eval gave dev NLL 2.887, ARC 19/20, Wiki 16K 2.630, ARC 89/100, chat NLL 1.322 (`research/runs/2026-10-27-framework-acceptance/`).

### Refining codes, not only scales

The optional `refine` stage ([docs/refine.md](docs/refine.md)) trains each decoder layer's ternary codes and scales against the original model's block outputs, starting from GPTQ's codes. All rows below start from the self-generated math and code model's GPTQ state. Before distillation, refining only layers 0-7 (about 4 minutes, 0.08-0.9% of codes changed per layer) improved every measure. Refining all 32 layers (16 minutes) gave about the same dev NLL as layers 0-7 alone, a lower KL and a higher EAR, but lower ARC and GSM8K. `original` is the default target; refine itself is off by default, and when it is enabled it refines all layers unless `refine.layers` says otherwise. Allocated VRAM peaks at 17.8 GiB while the model is loaded and the inputs are captured, and stays under 10 GiB while the layers train, but the allocator keeps 17.5 GiB reserved throughout, so plan for about 18.5 GiB:

| Before distillation | Dev NLL | ARC /20 | Full-vocab KL vs BF16 | EAR | GSM8K (first 50, thinking off) |
|---|---|---|---|---|---|
| No refine | 3.686 | 13 | 1.373 | 0.536 | 10/50 |
| Layers 0-7, original-model target | **3.396** | **16** | 0.987 | 0.597 | **16/50** |
| Layers 0-7, quantized-prefix target | 3.553 | **16** | 1.171 | 0.567 | 13/50 |
| All 32 layers, original-model target | 3.401 | 13 | **0.955** | **0.633** | 7/50 |

Distillation removed most of the difference. After it, the 32-layer refine is nominally ahead on dev and WikiText NLL, KL, EAR and ARC (chat NLL is 0.004 worse), by margins one run cannot separate from noise, and within noise on the reasoning suites:

| After distillation | Dev NLL | ARC /100 | Full-vocab KL vs BF16 | EAR | GSM8K off / on | HumanEval off / on |
|---|---|---|---|---|---|---|
| No refine (the Hugging Face model) | 2.960 | 84 | 0.554 | 0.706 | **69/100** / **25/50** | **19/50** / **10/25** |
| All 32 layers refined | **2.950** | **89** | **0.543** | **0.710** | 66/100 / 24/50 | **19/50** / **10/25** |

On GSM8K with thinking off, 13 problems were solved only without refine and 10 only with it; on HumanEval with thinking off, 2 and 2. Refine therefore stays off by default. GSM8K before distillation is near the floor (±3 problems is one standard error), so read it only as a tiebreaker. Refine is not bit-reproducible on the GPU. The 32-layer run repeated the pilot's layers 0-7 with the same inputs and settings: layers 0-2 came out identical, but from layer 3 on they differed (later layers also inherit the difference through their inputs). In layers 3-7, 0.004-0.03% of the codes (8,000 to 58,000 per layer) differed between the two runs, against 0.08-0.31% that refine flipped in those layers. Logs: `research/runs/2026-10-28-refine-pilot-*` and `research/runs/2026-10-28-reasoning-mix-refine32/`.

### More code in the data

The self-generated math and code model misses 25 of the 44 HumanEval problems BF16 solves (thinking off), mostly with wrong but well-formed code. `examples/qwen3_5-9b-reasoning-code.yaml` makes three changes:
- it adds 880 OpenCodeInstruct prompts to the traces, 280 with thinking on and 600 with thinking off, so code is 50% of the generated tokens instead of 32%;
- it gives the traces 192 of the 256 calibration blocks;
- it gives the traces 1,088 of the 1,856 distillation sequences.

The run took HumanEval with thinking off from 19/50 to 34/50. 18 problems were solved only by the new model and 3 only by the published one (two-sided sign test p = 0.0015). Two ablations separate the data from the length of distillation:

| Data | Epochs | Dev NLL | KL vs BF16 | GSM8K off / on | HumanEval off / on | Thinking-on GSM8K: mean tokens, cut at 4,096 |
|---|---|---|---|---|---|---|
| Self-generated math and code (the Hugging Face model) | 1 | 2.960 | 0.554 | 69/100 / 25/50 | 19/50 / 10/25 | 2,652, 23/50 |
| Same | 2 | **2.945** | **0.532** | 71/100 / 20/50 | 21/50 / 10/25 | 3,043, 28/50 |
| More code | 2 | 2.998 | 0.581 | **74/100** / 18/50 | 29/50 / **15/25** | 3,097, 32/50 |
| **More code** (`examples/qwen3_5-9b-reasoning-code.yaml`) | 1 | 3.012 | 0.607 | 65/100 / **32/50** | **34/50** / 14/25 | **2,346**, **18/50** |

The data moves HumanEval; a second epoch does not (21/50 on the published data). With either data set, a second epoch lowers NLL and KL but makes the model think longer with thinking on, so more GSM8K answers are cut off. Nearly every finished answer is correct in all four runs. GSM8K with thinking off is within noise: 14 problems solved only by the published model, 10 only by the new one.

The new config is about 1 h slower than the published one, about 5 h in all, because its traces take 3,744 s more. The Hugging Face model and `examples/qwen3_5-9b-reasoning.yaml` are unchanged. HumanEval is still 10 problems short of BF16. One run per row, greedy decoding. Logs and a per-problem comparison script: `research/runs/2026-10-28-humaneval-gap/`.

Two caveats. The 1-epoch config was picked from these four runs on the same 50 HumanEval and 100 GSM8K problems it is reported on, so 34/50 probably overstates its gain; a comparison with the published model on the HumanEval and GSM8K test problems outside the frozen subsets is running, with its decision rule fixed in advance. And OpenCodeInstruct and MBPP train contain reworded versions of six of the 50 HumanEval tasks (HumanEval/13, 20, 23, 35, 61 and 90), which the 13-word guard does not catch; the published model's distillation data already had two of them (23 and 61). Without those six and two related tasks (111, 156), the split is 16 to 3 (p = 0.0044).

### Starting from Prism's weights (separate experiment)

This is not a from-scratch result and not what `openbonsai run` does. Prism ML releases the Ternary Bonsai 2 weights under Apache-2.0. In one experiment we kept Prism's ternary codes, converted from the official 9B GGUF, and distilled only their group scales against BF16 Qwen3.5-9B on 2.1M tokens. Dev NLL went from 3.191 to 2.706 and ARC from 17/20 to 18/20. The codes are Prism's work; only the scale distillation is ours. Logs: `research/runs/2026-10-27-bonsai-start-*`.

## Hardware and time

One RTX 5090 (32 GB), Qwen3.5-9B, the example config:

| Stage | Time |
|---|---|
| quantize: sequential GPTQ on 256 × 2,048 tokens | about 10 min |
| endpoints: head and embedding GPTQ | about 1 min |
| distill, teacher pass: BF16 top-64 on 2,048 × 2,048 tokens | about 8 min |
| distill, training: 1 epoch over the same 4.2M tokens | about 45 min |
| export and verification (CPU) | about 2 min |

The timed stages add up to about 65 minutes. Traces, calibrate and eval were not timed separately. Traces generate 160 answers of up to 512 tokens in 20 batches of 8. We measured 26.5 decode steps/s for batch-1 generation, and a batch costs about the same per step, so the estimate is at most 20 × 512 ÷ 26.5 ≈ 6.5 minutes. Model loading comes on top.

**GPU memory.** Only one 32 GB card has been used; smaller cards are untested. Estimates from the code, not measurements: quantize keeps the BF16 model (about 18 GB) on the GPU, plus the inputs of the layer being rounded for all 256 × 2,048 calibration tokens (about 4.3 GB per copy in BF16; two copies coexist while the next layer's inputs are computed). Fewer calibration blocks is the lever: 128 blocks gave dev NLL 3.32 against 3.30 for 256. For distillation training, the one measurement is the scales-only run of the separate experiment above, on the same 9B model: a peak of 16.4 GB (`research/runs/2026-10-27-bonsai-start-scales/log.jsonl`).

**Disk.** Estimates for the 9B example:
- Hugging Face cache: about 18-20 GB for the BF16 model, plus the datasets (set `HF_HOME` to put it elsewhere).
- The run directory: about 10-15 GB. `quantize/` took 3.6 GB in one run (the state plus its per-layer copies); the teacher's top-64 shards about 1.6 GB; the distillation checkpoint about 0.8 GB; the GGUF 2 GB; the `compare` suite's cached BF16 logits about 2 GB.

## Other models

The pipeline does not know about any one architecture. An adapter says which linear layers to round and in what order, which to keep in high precision, and how the GGUF is laid out.

| Model type | Adapter | Export | Real-model results |
|---|---|---|---|
| `qwen3_5` (Qwen3.5, untied) | `qwen3_5` | PTQ1_0 GGUF, Ternary-Bonsai-2 layout | 9B, above |
| `llama` (Llama 2/3.x, Llama-architecture models such as Nanbeige4-style ones) | `dense` | PTQ1_0 GGUF with a byte-level BPE tokenizer (Llama 3.x); latent with SentencePiece (Llama 2) | none yet |
| `mistral` | `dense` | PTQ1_0 GGUF with a byte-level BPE tokenizer, else latent | none yet |
| `qwen3` | `dense` | PTQ1_0 GGUF; a tied table stays BF16 | none yet |
| `qwen2` (Qwen2, Qwen2.5) | `dense` | latent checkpoint only: Prism's loader refuses `prism.hadamard` for this architecture | none yet |

A latent checkpoint is a Hugging Face folder with the dequantized weights plus the exact codes and scales, for when Prism cannot run the architecture. A dense model of a known type needs no code, but it needs a few settings on top of the 9B example config (revision, evaluation suites, tied endpoints): see [the dense worked example](docs/adding-a-model.md#worked-example-2-a-dense-llamananbeige-style-model). A new dense type is one table row; a new architecture is one adapter file. How, and how to choose what stays in high precision with a sensitivity scan: [docs/adding-a-model.md](docs/adding-a-model.md).

**Models Prism has not compressed.** This is where an open pipeline matters most. Meta's [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache-2.0) is one: a dense decoder with an untied head, whose language model would be about 5.8 GB as PTQ1_0 against about 53 GB in BF16 (our estimate from its config). It is not supported yet. It needs an adapter (`model_type` `muse_glimmer`, sliding-window attention in three of every four layers). Its BF16 weights do not fit one 32 GB card, and quantize keeps the whole model on the GPU. Its widths (6,656 and 19,968) give a Hadamard block of 512, untested on real weights. And the GGUF needs a Prism runtime that knows the architecture; until then, the export would be a latent checkpoint.

**Base model without a chat template.** The traces stage cannot run: the model has no chat template to answer prompts in, so `openbonsai run` refuses it before the first stage and shows these settings. Turn traces off and replace the trace data with UltraChat, whose conversations are then joined as plain text. Add to the command:

```
--set traces.enabled=false --set 'calibrate.sources=[{kind: wikitext2, n: 128}, {kind: ultrachat, n: 128, min_row: 6000}]' --set 'distill.teacher.sources=[{kind: wikitext2, n: 512}, {kind: ultrachat, n: 1536, min_row: 6000}]'
```

**Hadamard block.** The block is derived from the model's widths. Prism's runtime has run real models only at block 4,096 (Qwen3.5-9B, ours and the official one), and tiny random models at block 128 in our tests. Other blocks, such as 2,048 for Llama 3.1 8B, are untested on real weights.

**Licence of the output.** The compressed model is a derivative of your base model and keeps its licence. Llama 3.x, for example, requires a model name that starts with "Llama": set it with `--set export.name=Llama-3.1-8B-OpenBonsai-PTQ1_0` or similar (it names the file and `general.name`).

## Repository layout

| Path | Contents |
|---|---|
| `openbonsai/` | the package: `adapters/` (Qwen3.5, dense), `core/` (Hadamard, GPTQ, ternary modules, losses), `stages/` (one module per stage), `formats/` (run state, PTQ1_0 codec, GGUF reader and writer), `eval/` (suites, frozen inputs), `cli.py` |
| `examples/` | `qwen3_5-9b.yaml` (closest to the measured settings; the differences, such as the trace prompts and the head's calibration blocks, are in its comments), tiny CPU configs, sample prompts and data |
| `docs/` | [your own fine-tune](docs/your-own-model.md), [method](docs/method.md), [formats](docs/formats.md), [adding a model](docs/adding-a-model.md), [evaluation](docs/evaluation.md) |
| `tests/` | CPU tests; tiny random models run the whole pipeline |
| `tools/prism/` | optional worker that scores the frozen dev suite inside Prism's llama.cpp |
| `scripts/setup_linux.sh` | Linux setup, optional Prism build |
| `research/` | history: experiment scripts, CAT-Q code, findings, run reports ([research/README.md](research/README.md)) |
| `web/` | the project's landing page |

## Research history

The method above came out of a longer search. Before GPTQ, we tried CAT-Q-only calibration at small budgets; those models scored dev NLL 8.3 to 13.3 and could not answer the dev prompts. That code, the findings documents, 27 validation receipts and every run report are in [research/](research/README.md). The tag `legacy-staged-pipeline` keeps the old pipeline runnable.

## License and acknowledgements

Apache License 2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) lists third-party data, formats and their licenses.

OpenBonsai builds on the work below. None of these authors or organizations is affiliated with OpenBonsai or endorses it.
- **ScaleQ-1.58 and CAT-Q.** Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan and Anbang Yao (Intel Labs China), "Attend to Your Own Thoughts: Breaking the Barrier for Post-Training Quantization of Reasoning LLMs through the Lens of 1.58-Bit Quantization", [arXiv:2608.01078](https://arxiv.org/abs/2608.01078) (the paper is in [research/papers](research/papers/README.md)). The AYOT traces come from this paper, and the CAT-Q code in `research/catq/` is an independent implementation of it, informed by the [BitTern](https://github.com/IntelChina-AI/BitTern) CAT-Q release.
- **Prism ML's Bonsai format and runtime.** The PTQ1_0 block layout and the `prism.hadamard.*` conventions follow [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (MIT), a fork of llama.cpp by the ggml authors. Ternary Bonsai is Prism ML's model family; we use it only as a baseline and, in the separate experiment above, as a starting point.
- **GPTQ.** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
- **Hadamard rotations.** QuaRot ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) and SpinQuant ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)).
- **Ternary language models.** BitNet ([arXiv:2310.11453](https://arxiv.org/abs/2310.11453)) and BitNet b1.58 ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)); Ternary Weight Networks ([arXiv:1605.04711](https://arxiv.org/abs/1605.04711)) for the plain-rounding baseline.
- **Models and data.** Qwen3.5 by the Qwen team (Apache-2.0); WikiText-2 (CC BY-SA 3.0), ARC (CC BY-SA 4.0), UltraChat 200k (MIT), GSM8K (MIT) and HumanEval (MIT); the reasoning recipes' trace prompts also come from MATH (MIT), MBPP (CC BY 4.0) and OpenCodeInstruct (CC BY 4.0). Revisions and licenses are in [NOTICE](NOTICE).
- **Software.** PyTorch, Hugging Face transformers, safetensors and the Hub, and flash-linear-attention.
