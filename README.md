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
#   traces -> calibrate -> quantize -> endpoints -> distill -> (export, phase P2) -> eval
openbonsai run --model <your fine-tune> --config examples/qwen3_5-9b.yaml --out runs/mine
openbonsai eval --run runs/mine --from distill --suite compare
python -m pytest            # CPU only; a tiny random Qwen3.5 runs the whole pipeline
```

This branch is phase P1 of [FRAMEWORK_PLAN.md](FRAMEWORK_PLAN.md). The PTQ1_0
GGUF export is phase P2; the README and docs rewrite is phase P4.

- Package: `openbonsai/` (`adapters/`, `core/`, `formats/`, `stages/`, `eval/`, `cli.py`)
- Research history, experiment scripts and run reports: [research/README.md](research/README.md)
- Optional Prism CPU evaluation runner: [tools/prism/](tools/prism/)

OpenBonsai is independent and not affiliated with the Bonsai model developers,
the Qwen team or the authors of the papers it builds on.

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
third-party data, formats and references.
