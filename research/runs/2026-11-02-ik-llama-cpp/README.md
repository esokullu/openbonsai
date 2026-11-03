# ik_llama.cpp on the Hugging Face model (2 November)

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) runs Prism's PTQ1_0 format since its PR 2550 (merged 29 September 2026). [Issue #13](https://github.com/esokullu/openbonsai/issues/13), from the PR's author, reported that the Hugging Face model runs there as is. This record checks that it also decodes the file correctly, against Prism's runtime on the same machine, text and settings.

- **ik_llama.cpp:** commit `4955387a25114a0dededcede84d27bf9ddf85333` (main on 2 November). CPU-only Release build: Windows 11, MSVC 19.44, Ninja, `-DGGML_CUDA=OFF -DGGML_IQK_FA_ALL_QUANTS=OFF` (`scripts/build_ik.bat`). The MSVC probe enables AVX2 (`/arch:AVX2`, IQK matmul, OpenMP) but not AVX-VNNI, which MSVC does not probe; a GCC or clang `-march=native` build on the same CPU would use it. `GGML_IQK_FA_ALL_QUANTS=OFF` only drops flash-attention kernels for quantized KV caches; every run used the default F16 cache.
- **Prism:** the CPU build at `c9c34bc` behind the README's numbers.
- **Files:**
  - the Hugging Face model, `OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf`, sha256 `818d8783bc39d92b2777a0333d38f7a25021a7b906f47344b83b7109eba6f7a0`, the file Hugging Face serves (HF commit `f06bf4d`);
  - the export-parity file of `../2026-10-27-gguf-shared-v1/` (sha256 `39a319bd...`);
  - text: upstream llama.cpp's `wiki.test.raw` (md5 `7c0137fc034ddbc56a296bce31b4f7fb`).
- **Machine:** i9-14900KF (8 P-cores with 2 threads each, 16 E-cores), 256 GB. Each run was pinned one thread per P-core (`scripts/run_pinned.ps1`) with the GPUs hidden. Another workload kept about 16 cores busy throughout, so every speed below is a lower bound. Perplexity does not depend on load.

ik_llama.cpp reads the file's rotation metadata and its own sign vectors (`prism.hadamard.sign_values`, one vector per input width). Ours differ from the official model's. Its log: `llm_load_hadamard: PrismML Hadamard enabled: 201 rotated weights, 2 sign widths, 1 inverse rows, block 4096`. With `-rtr 1` it repacks the 201 rotated matrices into its 8-row `PTQ1_0_R8` layout; the token embedding is only looked up by row.

## Perplexity

`llama-perplexity -m <gguf> -f wiki.test.raw -c 512 --chunks 8 -b 512 -t 8 [-rtr]`:

| Runtime | File | Perplexity | Log |
|---|---|---|---|
| Prism `c9c34bc` | Hugging Face model | 14.1514 ± 0.893 | `logs/prism_ppl_a1.*`, the same chunk for chunk as `../2026-10-28-prism-ppl-upstream/code-gap-1epoch.log` |
| ik_llama.cpp | same | 14.1392 ± 0.893 | `logs/ik_ppl_a1.*` |
| ik_llama.cpp, `-rtr 1` | same | 14.1463 ± 0.893 | `logs/ik_ppl_a1_rtr.*` |
| FP32 PyTorch, same weights | same | 14.146 | `../2026-10-29-runtime-parity-9b/` |
| Prism `c9c34bc` | export-parity file | 12.8753 ± 0.807 | `../2026-10-28-prism-ppl-upstream/shared-v1.log` |
| ik_llama.cpp, `-rtr 1` | same | 12.8859 ± 0.808 | `logs/ik_ppl_sv1_rtr.*` |
| ik_llama.cpp, `-rtr 1`, `IK_HADAMARD_DISABLE=1`, 2 chunks | Hugging Face model | 3,874,824 | `logs/ik_ppl_a1_nohad_rtr.*` |

ik_llama.cpp is within 0.1% of Prism on both files, in both modes. Per chunk the differences change sign, and each figure's standard error is about 6%, so the gaps are arithmetic noise. With the rotation switched off the model collapses, so the agreement depends on ik_llama.cpp applying the file's rotation and signs.

## Against Prism, position by position

`llama-perplexity --kl-divergence-base <Prism's log-probs> --kl-divergence -c 512 --chunks 8 -b 512 -t 8 -rtr` (`logs/ik_kld_a1_rtr.*`). The base file is Prism's own output on the same 8 chunks, written for `../2026-10-29-runtime-parity-9b/` (1,013,178,324 bytes, not copied). Both forks read that format with byte-identical code; their writers differ only in the window kept below the top log-prob (Prism 16 nats, ik_llama.cpp 24), and only Prism's file was read here.

| ik_llama.cpp with `-rtr 1` against Prism, 2,040 positions | |
|---|---|
| same top token | 98.97% ± 0.22% (2,019) |
| ik_llama.cpp's perplexity on Prism's stored tokens | 14.146334; each chunk equals its run on its own tokenization, so both runtimes tokenize the text the same way |
| mean KLD (llama-perplexity's estimator: Prism as base, terms below −16 nats dropped) | 0.000350 ± 0.000014 nats; median 0.00024, 99th percentile 0.0020, maximum 0.022 |
| RMS Δp of the target token | 0.44% |

Prism and the FP32 PyTorch reference agree on the top token at 99.2% (2,024 of 2,040) in `../2026-10-29-runtime-parity-9b/`. Two runtimes that each quantize activations to 8 bits differ from each other a little more than either does from FP32. The KLD is llama-perplexity's estimator, not comparable with the KL rows of that record; compare by top-1 agreement and the perplexity gap.

## Greedy replies

The same three questions went to both runtimes: Qwen3.5's chat template with thinking off, greedy decoding, 4 threads. ik_llama.cpp ran `llama-server -rtr 1`, sent the raw prompt `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (`scripts/greedy_client.py`, `logs/samples_ik_rtr.json`). Prism ran `llama-cli -st --reasoning off --temp 0 --top-k 1`, because the Prism build has no server binary (`scripts/prism_cli_samples.sh`, `logs/prism_cli_*.out`). The replies are identical character for character (`scripts/compare_samples.py`):
- "What is 17 * 24?": `468` in both. The answer is 408; the error is the model's.
- the cupcake word problem: 72, 38 tokens;
- a Python `fib(n)`: 128 tokens, cut at the limit.

## Speed and memory

`llama-bench -m <Hugging Face model> -p 512 -n 128 -t N -rtr R` (`logs/bench_*.out`):

| Threads | `-rtr` | Repetitions | pp512 (t/s) | tg128 (t/s) | Peak working set | Peak private memory |
|---|---|---|---|---|---|---|
| 4 | 1 | 3 | 23.10 ± 0.39 | 3.91 ± 0.17 | 1,996 MiB | 1,997 MiB |
| 4 | 0 | 1 | 5.29 | 1.53 | 1,946 MiB | 861 MiB |
| 8 | 1 | 3 | 27.55 ± 7.00 | 0.69 ± 0.26 | 1,996 MiB | 1,997 MiB |

- **`-rtr 1` speedup at 4 threads:** 4.4 times at prompt processing, 2.6 times at generation. PR 2550 reports 4.7 and 3.4 times for PTQ1_0 on the 27B model (pp256 and tg32).
- **8 threads ran slower than 4.** With the other workload, every logical CPU was 85-100% busy, so the 8-thread figures measure contention. The 8-thread run without `-rtr` was stopped after 48 s for that reason, and the greedy replies used 4 threads.
- **Not comparable with issue #13's report** of about 10 t/s of generation (i5-13420H, 4 threads, `-rtr 1`).
- **Memory is about 2 GB.** The model takes 1,894 MiB. With `-rtr 1` the model is read into private memory (no mmap); without it, it is memory-mapped. llama-perplexity peaks at about 2.0 GB (its 512-token batch keeps a 493 MiB compute buffer for the 248K vocabulary). ik_llama.cpp's `llama-server` at `-c 4096` peaked at 3.2 GB of private memory because it keeps a prompt cache in RAM by default (`--cache-ram`).
- **Label only:** `llama-bench` calls the file `qwen35 9B all F32 (guessed)`. Prism's PTQ1_0 files carry `general.file_type` 143, which ik_llama.cpp does not know (its PTQ1_0 is 158). It falls back to the most common tensor type, and the file has 225 F32 tensors against 202 PTQ1_0 ones. Inference does not use this value.

## Files

- `logs/summary.txt`: every run with its start and end, exit code, CPU mask and peak memory. It also notes two slips:
  - stopping the first perplexity batch left its loop alive, and that loop ran `ik_ppl_a1_rtr` before it was stopped; the next batch waited for it, so no two measurements overlapped;
  - a Prism `llama-server` run was attempted, but the build has no server binary.
- `logs/`: the output (`.out`) and log (`.log`) of every run. Empty stderr files are left out. `build-ik.log` keeps the configure step and the end of the build. `smoke.*` is the first 8-thread `llama-cli` run without `-rtr`.
- `scripts/`: the scripts as run, with this machine's absolute paths.
  - Batches: `ppl_runs.sh` was replaced by the trimmed `ppl_runs2.sh`, then `bench_runs.sh`, then `after_ppl.sh`.
  - Runners: `run_pinned.ps1` (CPU mask, hidden GPUs, peak memory) and `serve_and_query.ps1` with `greedy_client.py`.
  - `gguf_kv.py` prints a GGUF's metadata.
