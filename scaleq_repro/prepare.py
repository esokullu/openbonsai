"""Prepare AYOT calibration traces using the original target model itself.

The original chat-formatted prompt and every generated token are retained.
This module does not load Hugging Face or download anything until ``run``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .calibration import _positive_integer, _token_ids


CAUSAL_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3", "qwen2", "llama"})


def read_prompts(path: str | Path, max_samples: int | None = None) -> list[str]:
    """Read user-supplied JSONL objects with a text ``prompt`` or ``question``."""
    if max_samples is not None:
        _positive_integer(max_samples, "max_samples")
    prompts: list[str] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if max_samples is not None and len(prompts) >= max_samples:
                break
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected a JSON object")
            if "prompt" in record and "question" in record and record["prompt"] != record["question"]:
                raise ValueError(f"{location}: conflicting prompt and question fields")
            prompt = record.get("prompt", record.get("question"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{location}: prompt or question must be a nonempty string")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def generate_record(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    model_id: str,
    device: str,
    max_new_tokens: int = 2048,
    enable_thinking: bool = True,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate and serialize one complete token sequence, including reasoning.

    The tokenized template goes straight to ``generate``; no decode/re-encode
    round trip can lose special tokens. An oversized prompt raises instead of
    silently removing chat delimiters or the assistant reasoning prefix.
    """
    import torch

    _positive_integer(max_new_tokens, "max_new_tokens")
    if max_prompt_tokens is not None:
        _positive_integer(max_prompt_tokens, "max_prompt_tokens")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = _token_ids(prompt_ids, "chat template")
    if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
        raise ValueError(
            f"Chat-formatted prompt has {len(prompt_ids)} tokens, exceeding "
            f"max_prompt_tokens={max_prompt_tokens}; shorten the prompt or increase the limit"
        )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
    }
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
    ids = _token_ids(output[0].detach().cpu().tolist(), "model generation")
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Generation did not retain the input prompt prefix; expected a decoder-only model")
    generated_ids = ids[len(prompt_ids):]
    if not generated_ids:
        raise ValueError("Model generated no continuation tokens")
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
    return {
        "model_id": model_id,
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "prompt_token_count": len(prompt_ids),
        "generated_token_count": len(generated_ids),
        "enable_thinking": enable_thinking,
        "hit_generation_limit": len(generated_ids) >= max_new_tokens and generated_ids[-1] not in eos_ids,
        "prompt": prompt,
        "formatted_prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--prompts", type=Path, required=True, help="Local JSONL: {prompt: text} or {question: text}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="HF revision; recorded resolved commit when available")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=None, help="Reject longer chat prompts; never silently truncate")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached/local weights only")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cpu":
        return requested
    if requested == "mps" and torch.backends.mps.is_available():
        return requested
    if (requested == "cuda" or requested.startswith("cuda:")) and torch.cuda.is_available():
        device = torch.device(requested)
        if device.index is None or device.index < torch.cuda.device_count():
            return requested
    raise ValueError(f"Requested device {requested!r} is invalid or unavailable")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load original target weights once and generate a local calibration JSONL."""
    _positive_integer(args.max_new_tokens, "max_new_tokens")
    if args.max_prompt_tokens is not None:
        _positive_integer(args.max_prompt_tokens, "max_prompt_tokens")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    prompts = read_prompts(args.prompts, args.max_samples)
    output = Path(args.output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"Output or manifest already exists: {output}; use --overwrite to replace")
    if Path(args.prompts).resolve() in {output.resolve(), manifest_path.resolve()}:
        raise ValueError("Calibration output and manifest must differ from the source prompt file")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, set_seed

    device = _resolve_device(args.device, torch)
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False
    )
    model_type = config.model_type
    if model_type == "qwen3_5":
        model_class = AutoModelForImageTextToText
    elif model_type in CAUSAL_MODEL_TYPES:
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported model_type {model_type!r}; supported: qwen3_5, {', '.join(sorted(CAUSAL_MODEL_TYPES))}")
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("Calibration must be generated from original unquantized target weights")
    resolved_revision = getattr(config, "_commit_hash", None) or args.revision
    common = {
        "revision": resolved_revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = model_class.from_pretrained(args.model, config=config, dtype=dtype, **common).to(device).eval()
    manifest: dict[str, Any] = {
        "method": "AYOT: original target model self-generated traces",
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "model_type": model_type,
        "model_class": model_class.__name__,
        "device": device,
        "requested_dtype": args.dtype,
        "loaded_dtype": str(model.dtype),
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_source": str(Path(args.prompts).resolve()),
        "num_samples": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "total_tokens": 0,
        "records_hitting_generation_limit": 0,
        "truncated_tokens": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            for index, prompt in enumerate(prompts, 1):
                record = generate_record(
                    tokenizer, model, prompt, model_id=args.model, device=device,
                    max_new_tokens=args.max_new_tokens, enable_thinking=args.enable_thinking,
                    do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                record["resolved_revision"] = resolved_revision
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest["num_samples"] += 1
                manifest["prompt_tokens"] += record["prompt_token_count"]
                manifest["generated_tokens"] += record["generated_token_count"]
                manifest["total_tokens"] += len(record["input_ids"])
                manifest["records_hitting_generation_limit"] += int(record["hit_generation_limit"])
                print(f"Prepared {index}/{len(prompts)}: {record['generated_token_count']} generated tokens", flush=True)
        os.replace(temporary_path, output)
        temporary_path = None
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
