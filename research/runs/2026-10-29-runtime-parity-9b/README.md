# Runtime parity of the Hugging Face model (29 October)

Prism's llama.cpp runtime against our PyTorch reference, on the same weights: the current Hugging Face model, the 1-epoch code-gap run of `../2026-10-28-humaneval-gap/` (GGUF sha256 818d8783..., 1,997,486,976 bytes). The check is the Qwen3-8B's, `../2026-10-28-qwen3-8b-reasoning/runtime_parity.py`, run unchanged from this repository at 17aca8a (the command is in `run.sh`, the commit in `launcher.log`):

- **Prism:** `llama-perplexity --kl-divergence-base` (CPU build at c9c34bc, 8 threads) writes its full-vocabulary log-probs at the scored positions of the first 8 chunks of 512 tokens of upstream llama.cpp's `wiki.test.raw` (md5 7c0137fc034ddbc56a296bce31b4f7fb).
- **PyTorch:** Qwen3.5-9B (revision c202236) in FP32 on the CPU, with the distilled state's dequantized weights, scores the same token chunks.
- **Scored:** the second half of each chunk, positions 256-510 (0-indexed), each predicting the token after it: 255 per chunk, the positions `llama-perplexity` scores and the only ones it writes, 2,040 in all.

| | This model | Qwen3-8B (`../2026-10-28-qwen3-8b-reasoning/`) |
|---|---|---|
| top-1 agreement | **99.2%** (2,024 of 2,040) | 98.8% |
| perplexity: PyTorch / Prism (`llama-perplexity`) / Prism from its stored log-probs | 14.146 / 14.151 / 14.145 | 14.934 / 14.936 / 14.920 |
| KL(PyTorch ‖ Prism), floored tokens lumped into one bucket: a lower bound | 0.00015 nats (max 0.0034) | 0.00020 (max 0.0050) |
| KL(PyTorch ‖ Prism), Prism renormalized: biased high | 0.010 nats (max 0.027) | 0.008 (max 0.017) |
| KL(Prism renormalized ‖ PyTorch), the synthetic check's estimator | 0.042 nats (max 0.25) | 0.044 (max 0.31) |
| mean \|log-prob difference\| where Prism stores it | 0.019 nats | 0.023 |

Prism stores each log-prob as a uint16 over a 16-nat window below the position's top log-prob, and quantizes activations to 8 bits, so the two sides are not identical. The Qwen3-8B report explains the three KL rows: the floor makes the plain KL slightly negative (here −0.0012), lumping the floored tokens gives a lower bound, and renormalizing gives an estimate that runs high. The synthetic state's 0.0028 (`../gguf-test/runtime-parity/`) is not comparable with any of them, not even the last KL row, which uses its formula. On a confident model, the tokens below Prism's 16-nat window dominate that estimator, because renormalizing gives them more mass than they have; the synthetic state (perplexity about 13,700) has much flatter distributions, so its floored tokens carry far less mass (`../2026-10-28-qwen3-8b-reasoning/report.md`, "Why three KL numbers"). Compare the checks by top-1 agreement and by the gap between the two perplexities instead, keeping in mind that the synthetic state's much flatter distributions make its top token easier to flip.

The script computes Prism's perplexity from the stored log-probs, which store a target token below the 16-nat window at the floor and so cap its loss. One of the 2,040 targets here is floored (in the 7th chunk; counted from the log-prob file), so that figure is slightly low. `llama-perplexity` scores the same 2,040 positions from its full logits: 14.1514 for this file (`../2026-10-28-prism-ppl-upstream/code-gap-1epoch.log`) and 14.9356 for the Qwen3-8B (`../2026-10-28-qwen3-8b-reasoning/prism/prism_ppl_qwen3-8b_wiki.test.log`), slightly above PyTorch for both. Rounded, every figure for this model is 14.15, the number the repository README and the model card give.

The script reads the GGUF path from the run's `export/DONE.json`, which records where the export first wrote the file (the `gguf` field of the output). That file is the published one: sha256 818d8783bc39d92b2777a0333d38f7a25021a7b906f47344b83b7109eba6f7a0, the same as the Hugging Face file.

Files:

- `run.sh`: the command as run (CPU only, `D:/` paths).
- `launcher.log`: `run.sh`'s own log: the start time, the commit, and where `openbonsai` was imported from. Its last line gives the exit status of `date`, not of the script (a slip in `run.sh`); the script writes `runtime_parity.json` only when it finishes.
- `runtime_parity.json`: the output.
- `runtime_parity.log`: the script's console output (the model-loading progress bar and the same JSON).

Prism's raw log-prob file (1,013,178,324 bytes, sha256 85e974230d7a39fee78283189feb7951fba5ba3b036180cf4b01b008a5367c73) is not committed. Prism took 357 s including its model load; PyTorch's 8 forward passes took 264 s after it had loaded the model (`prism_seconds`, `torch_seconds`).
