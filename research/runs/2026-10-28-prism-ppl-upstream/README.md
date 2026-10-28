# Prism perplexity on upstream WikiText-2 test (28 October)

All runs use Prism's llama.cpp CPU build at c9c34bc (Release, 8 threads):

```
llama-perplexity -m <model>.gguf -f wiki.test.raw -c 512 --chunks 8 -b 512 -t 8
```

The text is `wiki.test.raw` from upstream llama.cpp's `scripts/get-wikitext-2.sh` (`wikitext-2-raw-v1.zip`): 1,290,590 bytes, md5 7c0137fc034ddbc56a296bce31b4f7fb, used unchanged. `-b 512` (one sequence per pass) and the default batch (4 sequences) give the same numbers, chunk for chunk (`*-default-batch.log`).

| Log | GGUF (sha256) | PPL |
|---|---|---|
| `official.log` | Prism ML's Ternary-Bonsai-2-9B-PTQ1_0.gguf (3561df7a...) | 17.2108 +/- 1.222 |
| `shared-v1.log` | `../2026-10-27-gguf-shared-v1/` export (39a319bd...; measured by a separate run with the default batch, 4 sequences per pass) | 12.8753 +/- 0.807 |
| `reasoning-mix-v1.log` | `../2026-10-27-reasoning-mix-v1/`, the Hugging Face model (4f2f3d10...) | 13.6124 +/- 0.866 |
| `code-gap-2epoch.log` | `../2026-10-28-humaneval-gap/code-gap-2epoch/` (f26e74bc...) | 13.9288 +/- 0.883 |
| `code-gap-1epoch.log` | `../2026-10-28-humaneval-gap/code-gap-1epoch/`, `examples/qwen3_5-9b-reasoning-code.yaml` (818d8783...) | 14.1514 +/- 0.893 |
| `official-old-copy.log` | the official file, on the old copy of the text (below) | 16.5087 +/- 1.167 |

The earlier numbers (16.51 official, 11.93 shared-v1, 12.59 reasoning mix, in `../2026-10-27-gguf-shared-v1/prism_ppl_*.log` and `../2026-10-27-reasoning-mix-v1/export/prism_ppl.txt`) came from a copy of the test file with the blank lines removed and CRLF line endings (1,290,547 bytes). The official file on that copy reproduces 16.5087 (`official-old-copy.log`). The order of the models is the same on both texts.

The logs keep llama-perplexity's output from thread setup on; the model-loading lines before it are not copied.
