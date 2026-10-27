# OpenBonsai

OpenBonsai is a framework for compressing dense language models to ternary
weights in Prism's PTQ1_0 layout (the format of the Ternary Bonsai models),
starting with Qwen3.5-9B. The method: signed block-Hadamard rotation,
sequential GPTQ ternary rounding on calibration data that includes the
model's own traces, ternary embedding and head, then distillation of the
group scales against the original model.

```
pip install -e .[test]
openbonsai run --config examples/qwen3_5-9b.yaml --out runs/qwen3_5-9b
#   traces -> calibrate -> quantize -> endpoints -> distill -> export -> eval
#   export: runs/qwen3_5-9b/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf (~2 GB, Prism's llama.cpp)
openbonsai run --model <your fine-tune> --config examples/qwen3_5-9b.yaml --out runs/mine
openbonsai eval --run runs/mine --from distill --suite compare
openbonsai export --run runs/mine --out mine.gguf          # verified on write
openbonsai eval --from mine.gguf --model <your fine-tune> --suite compare   # or any PTQ1_0 GGUF
python -m pytest            # CPU only; a tiny random Qwen3.5 runs the whole pipeline
```

This branch holds phases P1 and P2 of [FRAMEWORK_PLAN.md](FRAMEWORK_PLAN.md)
(P2: the PTQ1_0 GGUF export and the GGUF loader, research/validation/p2-export-parity.md).
The README and docs rewrite is phase P4.

- Package: `openbonsai/` (`adapters/`, `core/`, `formats/`, `stages/`, `eval/`, `cli.py`)
- Research history, experiment scripts and run reports: [research/README.md](research/README.md)
- Optional Prism CPU evaluation runner: [tools/prism/](tools/prism/)

OpenBonsai is independent and not affiliated with the Bonsai model developers,
the Qwen team or the authors of the papers it builds on.

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
third-party data, formats and references.
