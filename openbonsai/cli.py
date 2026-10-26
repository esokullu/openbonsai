"""OpenBonsai: experimental ScaleQ-style compression for dense Qwen3.5.

The smoke command and package import do not download model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _empty_output(path):
    destination = Path(path)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {destination}")
    return destination


def smoke(output, seed=2, packing="auto", *, factor_mapping="exp",
          reconstruction_loss="mse", huber_delta=1.0):
    import torch
    from .adapter import tiny_model, replace_projections, iter_ternary_modules, set_progress
    from .export import save_compressed, load_compressed
    from .quantizer import restore_ternary_linear
    from .trainer import train, TrainingConfig

    output = _empty_output(output)
    config = TrainingConfig(epochs=3, batch_size=1, window_size=4, stride=4, seed=seed,
                            gradient_checkpointing=True, reconstruction_loss=reconstruction_loss,
                            huber_delta=huber_delta)
    config.validate()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = tiny_model(seed=seed)
    original_parameters = sum(p.numel() for p in model.parameters())
    original_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    torch.manual_seed(seed)
    names = replace_projections(model, factor_mapping=factor_mapping)
    generator = torch.Generator().manual_seed(seed)
    samples = [{"input_ids": torch.randint(3, 240, (1, 12), generator=generator),
                "attention_mask": torch.ones((1, 12), dtype=torch.long)} for _ in range(2)]
    samples[1]["attention_mask"][:, -3:] = 0
    samples[1]["input_ids"][:, -3:] = 0
    report = train(model, samples, config)
    set_progress(model, 1)
    # FP16 export scales introduce small rounding versus FP32 learned scales.
    with torch.no_grad():
        hard_logits = model(**samples[0], use_cache=False).logits
    manifest = save_compressed(model, output, packing=packing,
                               metadata={"fixture": "random tiny native Qwen3.5", "training": report})
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
    result = {"project": "OpenBonsai", "test": "offline native hybrid training and packed round trip", "seed": seed,
              "real_9b_weights_loaded": False, "real_model_quality_evaluated": False,
              "original_parameters": original_parameters, "original_parameter_bytes": original_bytes,
              "converted_projections": len(names), "packed_roundtrip_max_logit_error": 0.0,
              "fp16_scale_rounding_max_logit_error": float((hard_logits - restored_logits).abs().max()),
              "packing": packing, "storage": manifest["storage"], "training": report}
    # Kept outside the artifact's checksummed accounting.
    report_path = output.parent / (output.name + "-report.json")
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def train_command(args):
    import torch
    from transformers import AutoTokenizer
    from .adapter import load_base_model, replace_projections
    from .calibration import load_calibration
    from .export import save_compressed
    from .trainer import TrainingConfig, train

    output = _empty_output(args.output)
    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
        window_size=args.window_size, stride=args.stride, learning_rate=args.learning_rate,
        teacher_prefix=args.teacher_prefix,
        lora_learning_rate=args.lora_learning_rate,
        reconstruction_loss=args.reconstruction_loss, huber_delta=args.huber_delta,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing)
    config.validate()
    if args.lora_rank < 0:
        raise ValueError("lora-rank must be nonnegative")
    if args.factor_mapping not in ("exp", "sigmoid"):
        raise ValueError("factor-mapping must be exp or sigmoid")
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
    torch.manual_seed(args.seed)
    replaced = replace_projections(model, group_size=args.group_size,
        aggressive_gates=args.aggressive_gates, lora_rank=args.lora_rank,
        factor_mapping=args.factor_mapping)

    def progress(entry):
        if entry["step"] == 1 or entry["step"] % args.log_every == 0 or entry["step"] == entry["total_steps"]:
            print(json.dumps(entry, allow_nan=False), flush=True)

    report = train(model, samples, config, progress_callback=progress,
                   cache_directory=args.cache_dir)
    metadata = {"project": "OpenBonsai", "base_model": args.model, "resolved_revision": revision,
        "method": "independent ScaleQ-style Qwen3.5 research adaptation",
        "calibration": calibration, "training": report,
        "lora_rank": args.lora_rank, "lora_initialization_seed": args.seed,
        "factor_mapping": args.factor_mapping,
        "aggressive_recurrent_gates": args.aggressive_gates, "converted_projections": replaced}
    manifest = save_compressed(model, output, metadata=metadata, packing=args.packing)
    # Tokenizer assets are supplementary and excluded from manifest storage totals.
    tokenizer.save_pretrained(output / "tokenizer")
    return {"output": str(output.resolve()), "storage": manifest["storage"],
            "tokenizer": "tokenizer/ (additional bytes excluded from model artifact totals)",
            "quality_evaluation": "not performed; use held-out data before deployment"}


def repack_command(args):
    from .export import repack_compressed

    source, destination = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().absolute()
    manifest = repack_compressed(source, destination, packing=args.packing)
    original = json.loads((source / "manifest.json").read_text())
    before, after = original["storage"], manifest["storage"]
    return {
        "input": str(source), "output": str(destination),
        "packing": args.packing, "before": before, "after": after,
        "artifact_bytes_saved": before["artifact_bytes"] - after["artifact_bytes"],
        "artifact_reduction_percent": 100 * (1 - after["artifact_bytes"] / before["artifact_bytes"]),
        "retrained": False,
        "quality": "exactly preserves existing ternary codes, scales, biases, and retained tensors",
        "tokenizer": "copied if present; supplementary bytes excluded from model artifact totals",
    }


def _packing_argument(command):
    command.add_argument("--packing", choices=["auto", "bitcos", "2bit", "base3"], default="auto",
        help="Storage codec: auto chooses the smallest code payload among 2bit, BITCOS and base3")


def _reconstruction_arguments(command):
    command.add_argument("--factor-mapping", choices=["exp", "sigmoid"], default="exp",
        help="exp keeps the original mappings; sigmoid uses CAT-Q mappings and its scale epsilon")
    command.add_argument("--reconstruction-loss", choices=["mse", "huber"], default="mse",
        help="Calibration objective (default: mse); hard reconstruction diagnostics always report MSE")
    command.add_argument("--huber-delta", type=float, default=1.0,
        help="Positive Huber transition threshold; 1.0 is an independent choice, not an authors' default")


def parser():
    from . import prepare
    root = argparse.ArgumentParser(prog="openbonsai", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prepare.add_arguments(sub.add_parser("prepare", help="Generate self-teacher calibration (loads model)"))
    demo = sub.add_parser("smoke", help="Offline test on random tiny Qwen3.5 (no downloads)")
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=2)
    _packing_argument(demo)
    _reconstruction_arguments(demo)
    repack = sub.add_parser("repack", help="Repack an existing artifact losslessly on CPU; no training or model download")
    repack.add_argument("--input", required=True, help="Existing OpenBonsai artifact directory")
    repack.add_argument("--output", required=True, help="New or empty destination directory")
    _packing_argument(repack)
    run = sub.add_parser("train", help="Compress original dense Qwen3.5 weights")
    _packing_argument(run)
    _reconstruction_arguments(run)
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
    run.add_argument("--teacher-prefix", choices=["quantized", "original"], default="quantized",
        help="Prefix used for original-weight teacher targets; original is an experimental accumulated-error control")
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--lora-rank", type=int, default=64,
        help="Rank of the learned pre-quantization correction (default: 64; 0 disables it)")
    run.add_argument("--lora-learning-rate", type=float, default=0.001,
        help="Learning rate for low-rank correction matrices, separate from group factors")
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
        result = smoke(args.output, args.seed, args.packing, factor_mapping=args.factor_mapping,
                       reconstruction_loss=args.reconstruction_loss, huber_delta=args.huber_delta)
    elif args.command == "repack":
        result = repack_command(args)
    else:
        if args.log_every < 1:
            raise ValueError("log-every must be positive")
        result = train_command(args)
    print(json.dumps(result, indent=2, allow_nan=False))
