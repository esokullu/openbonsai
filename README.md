# OpenBonsai

OpenBonsai is being rebuilt as a reusable framework for compressing dense
language models to ternary weights in Bonsai's PTQ1_0 GGUF format, starting
with Qwen3.5-9B. The method: Hadamard rotation, sequential GPTQ ternary
rounding, ternary embedding and head, then distillation of the group scales.

This branch is mid-restructure. The package currently holds only the modules
being carried into the framework (`hadamard`, `packing`, `calibration`,
`prepare`, `evaluation`, `eval_backends`); the CLI is not built yet.

- Plan and phases: [FRAMEWORK_PLAN.md](FRAMEWORK_PLAN.md)
- Research history, experiment scripts and run reports: [research/README.md](research/README.md)
- Optional Prism CPU evaluation runner: [tools/prism/](tools/prism/)
- Tests (CPU only): `python -m pytest`

OpenBonsai is independent and not affiliated with the Bonsai model developers,
the Qwen team or the authors of the papers it builds on.

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for
third-party data, formats and references.
