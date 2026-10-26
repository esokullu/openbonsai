"""Explicit local entry points: no model download in smoke or package import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _empty_output(path):
    destination = Path(path)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {destination}")
    return destination


def smoke(output, seed=2):
    import torch
    from .adapter import tiny_model, replace_projections, iter_ternary_modules, set_progress
    from .export import save_compressed, load_compressed
    from .quantizer import restore_ternary_linear
    from .trainer import train, TrainingConfig

    output = _empty_output(output)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = tiny_model(seed=seed)
    original_parameters = sum(p.numel() for p in model.parameters())
    original_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    names = replace_projections(model)
    generator = torch.Generator().manual_seed(seed)
    samples = [{"input_ids": torch.randint(3, 240, (1, 12), generator=generator),
                "attention_mask": torch.ones((1, 12), dtype=torch.long)} for _ in range(2)]
    samples[1]["attention_mask"][:, -3:] = 0
    samples[1]["input_ids"][:, -3:] = 0
    config = TrainingConfig(epochs=3, batch_size=1, window_size=4, stride=4, seed=seed,
                            gradient_checkpointing=True)
    report = train(model, samples, config)
    set_progress(model, 1)
    # FP16 export scales introduce small rounding versus FP32 learned scales.
    with torch.no_grad():
        hard_logits = model(**samples[0], use_cache=False).logits
    manifest = save_compressed(model, output, metadata={"fixture": "random tiny native Qwen3.5", "training": report})
    loaded = load_compressed(output)
    # Compare serialization with the SAME FP16 packed representation, separately
    # from the original hard weights, so a tolerance cannot hide a save bug.
    for name, module in list(iter_ternary_modules(model)):
        model.set_submodule(name, restore_ternary_linear(module.packed_state(), dtype=module.weight.dtype))
    with torch.no_grad():
        packed_logits = model(**samples[0], use_cache=False).logits
        restored_logits = loaded(**samples[0], use_cache=False).logits
    torch.testing.assert_close(restored_logits, packed_logits, atol=0, rtol=0)
    if not bool(torch.isfinite(restored_logits).all()):
        raise FloatingPointError("Nonfinite restored logits")
    result = {"test": "offline native hybrid training and packed round trip", "seed": seed,
              "real_9b_weights_loaded": False, "real_model_quality_evaluated": False,
              "original_parameters": original_parameters, "original_parameter_bytes": original_bytes,
              "converted_projections": len(names), "packed_roundtrip_max_logit_error": 0.0,
              "fp16_scale_rounding_max_logit_error": float((hard_logits - restored_logits).abs().max()),
              "storage": manifest["storage"], "training": report}
    # Kept outside the artifact's checksummed accounting.
    report_path = output.parent / (output.name + "-report.json")
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def train_command(args):
    from transformers import AutoTokenizer
    from .adapter import load_base_model, replace_projections
    from .calibration import load_calibration
    from .export import save_compressed
    from .trainer import TrainingConfig, train

    output = _empty_output(args.output)
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
        window_size=args.window_size, stride=args.stride, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing)
    config.validate()
    # Read local input before loading any model weights.
    if not Path(args.calibration).is_file():
        raise FileNotFoundError(args.calibration)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype,
                           revision=args.revision, local_files_only=args.local_files_only)
    revision = getattr(model.config, "_commit_hash", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision or args.revision,
                      local_files_only=args.local_files_only, trust_remote_code=False)
    samples, calibration = load_calibration(args.calibration, model_id=args.model,
        sequence_length=args.sequence_length, pad_token_id=tokenizer.pad_token_id or 0,
        max_samples=args.max_samples, expected_revision=revision)
    vocab_size = model.get_input_embeddings().num_embeddings
    if any(int(sample["input_ids"].max()) >= vocab_size for sample in samples):
        raise ValueError("Calibration contains tokens outside this model's vocabulary")
    replaced = replace_projections(model, group_size=args.group_size, aggressive_gates=args.aggressive_gates)

    def progress(entry):
        if entry["step"] == 1 or entry["step"] % args.log_every == 0 or entry["step"] == entry["total_steps"]:
            print(json.dumps(entry, allow_nan=False), flush=True)

    report = train(model, samples, config, progress_callback=progress,
                   cache_directory=args.cache_dir)
    metadata = {"base_model": args.model, "resolved_revision": revision,
        "method": "independent ScaleQ-style Qwen3.5 research adaptation",
        "calibration": calibration, "training": report,
        "aggressive_recurrent_gates": args.aggressive_gates, "converted_projections": replaced}
    manifest = save_compressed(model, output, metadata=metadata)
    # Tokenizer assets are supplementary and excluded from manifest storage totals.
    tokenizer.save_pretrained(output / "tokenizer")
    return {"output": str(output.resolve()), "storage": manifest["storage"],
            "tokenizer": "tokenizer/ (additional bytes excluded from model artifact totals)",
            "quality_evaluation": "not performed; use held-out data before deployment"}


def parser():
    from . import prepare
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prepare.add_arguments(sub.add_parser("prepare", help="Generate self-teacher calibration (loads model)"))
    demo = sub.add_parser("smoke", help="Offline test on random tiny Qwen3.5 (no downloads)")
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=2)
    run = sub.add_parser("train", help="Compress original dense Qwen3.5 weights")
    run.add_argument("--model", default="Qwen/Qwen3.5-9B")
    run.add_argument("--revision", default="main")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--calibration", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    run.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    run.add_argument("--sequence-length", type=int, default=2048)
    run.add_argument("--max-samples", type=int)
    run.add_argument("--group-size", type=int, default=128)
    run.add_argument("--epochs", type=int, default=60)
    run.add_argument("--batch-size", type=int, default=3)
    run.add_argument("--window-size", type=int, default=4)
    run.add_argument("--stride", type=int, default=2)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--grad-clip", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=2)
    run.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--aggressive-gates", action="store_true")
    run.add_argument("--cache-dir", help="Disk-backed per-window cache; otherwise all samples stay in RAM")
    run.add_argument("--log-every", type=int, default=50)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from .prepare import run
        result = run(args)
    elif args.command == "smoke":
        result = smoke(args.output, args.seed)
    else:
        if args.log_every < 1:
            raise ValueError("log-every must be positive")
        result = train_command(args)
    print(json.dumps(result, indent=2, allow_nan=False))
