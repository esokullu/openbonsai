"""Generate content-bound AYOT records from one original local checkpoint.

This is an attributable execution receipt, not cryptographic proof that an
arbitrary third-party receipt is honest. Neither the output nor its quality is
promoted automatically into a training experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

from .eval_backends import _identity, local_directory_identity
from .pipeline_sources import (checkpoint_tensors, code_identity, load_original_source,
                               package_tokenizer, tokenizer_identity)
from .prepare import generate_record, read_prompts
from .stage_runtime import write_json

FORMAT = "openbonsai-original-ayot.v1"


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_generation(value):
    fields = {"seed", "max_new_tokens", "max_prompt_tokens", "max_samples",
              "enable_thinking", "do_sample", "temperature", "top_p"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Generation requires every explicit field and no unknown fields")
    for key in ("max_new_tokens", "max_prompt_tokens", "max_samples"):
        _positive(value[key], key)
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("Generation seed must be a nonnegative integer")
    for key in ("enable_thinking", "do_sample"):
        if type(value[key]) is not bool:
            raise ValueError(f"{key} must be a bool")
    import math
    for key in ("temperature", "top_p"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if value["top_p"] > 1:
        raise ValueError("top_p must be at most one")


def prepare_bound(*, model_directory, expected_model_sha256, tokenizer_directory,
                  expected_tokenizer_sha256, prompts, expected_prompts_sha256,
                  generation, role, output, device="cpu", operation=None,
                  progress_callback=None):
    """Generate complete local traces; a failed attempt has no producer.json.

    Tokenizer hashes identify the supported raw-byte subset from
    ``pipeline_sources.tokenizer_identity``, not the complete checkpoint folder.
    Every generated token and its original prompt are retained. Training's
    separately declared padding/truncation is not applied here.
    """
    validate_generation(generation)
    if role not in {"calibration", "search_validation", "evaluation"}:
        raise ValueError("Unknown trace role")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("Pipeline device must be cpu or cuda:0")
    operation = operation or (lambda _label, fn: fn())
    output = Path(output).absolute()
    for path in (model_directory, tokenizer_directory, prompts, Path(__file__).parent):
        if output.resolve() == Path(path).resolve() or output.resolve().is_relative_to(Path(path).resolve()):
            raise ValueError("Trace output cannot be inside any input source")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    code = code_identity()
    source = operation("prepare-source-identity", lambda: local_directory_identity(model_directory))
    tokens = tokenizer_identity(tokenizer_directory)
    prompt_source = _identity(prompts, expected_prompts_sha256)
    if source["sha256"] != expected_model_sha256 or tokens["sha256"] != expected_tokenizer_sha256:
        raise ValueError("Original source or tokenizer hash mismatch")
    tensor_table = checkpoint_tensors(model_directory)
    from .pipeline_sources import _read_json
    config = _read_json(Path(model_directory) / "config.json")
    text_config = config.get("text_config", config)
    context_limit = text_config.get("max_position_embeddings")
    if (type(context_limit) is not int or context_limit < 1
            or generation["max_prompt_tokens"] + generation["max_new_tokens"] > context_limit):
        raise ValueError("Declared prompt plus generation limits exceed the original text context")
    prompt_values = read_prompts(prompts, generation["max_samples"])
    output.mkdir(parents=True, exist_ok=False)
    packaged = package_tokenizer(tokenizer_directory, output / "tokenizer")
    from transformers import AutoTokenizer, set_seed
    tokenizer = AutoTokenizer.from_pretrained(packaged["path"], local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("Bound preparation requires the native fast tokenizer")
    original = {"model_directory": source, "checkpoint_tensors": tensor_table,
                "tokenizer_directory": packaged}
    model = operation("prepare-load-original", lambda: load_original_source(original, device=device))
    set_seed(generation["seed"])
    # Explicitly align this reference path with no-KV-cache calibration. This
    # changes only the freshly loaded object's runtime generation option.
    model.generation_config.use_cache = False
    source_id = "sha256:" + source["sha256"]
    counts = {"records": 0, "prompt_tokens": 0, "generated_tokens": 0,
              "total_tokens": 0, "records_hitting_generation_limit": 0}
    trace_path = output / "traces.jsonl"
    with trace_path.open("xb") as stream:
        for index, prompt in enumerate(prompt_values):
            def generate():
                return generate_record(tokenizer, model, prompt, model_id=source_id,
                    device=device, max_new_tokens=generation["max_new_tokens"],
                    max_prompt_tokens=generation["max_prompt_tokens"],
                    enable_thinking=generation["enable_thinking"], do_sample=generation["do_sample"],
                    temperature=generation["temperature"], top_p=generation["top_p"])
            record = operation(f"prepare-record-{index:06d}", generate)
            record.update(id=f"{role}-{index:06d}", resolved_revision=source["sha256"],
                          source_content_sha256=source["sha256"], tokenizer_content_sha256=tokens["sha256"])
            stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
            stream.flush()
            counts["records"] += 1
            counts["prompt_tokens"] += record["prompt_token_count"]
            counts["generated_tokens"] += record["generated_token_count"]
            counts["total_tokens"] += len(record["input_ids"])
            counts["records_hitting_generation_limit"] += int(record["hit_generation_limit"])
            if progress_callback:
                progress_callback({"stage": "prepare", "record": index + 1,
                                   "records": len(prompt_values), "counts": dict(counts)})
    def preserve():
        if (local_directory_identity(model_directory) != source
                or tokenizer_identity(tokenizer_directory) != tokens
                or tokenizer_identity(packaged["path"]) != packaged
                or _identity(prompts) != prompt_source or code_identity() != code):
            raise RuntimeError("Preparation inputs or implementation changed")
    operation("prepare-final-preservation", preserve)
    file = _identity(trace_path)
    receipt = {"kind": FORMAT, "status": "complete", "role": role,
               "source_content_sha256": source["sha256"],
               "tokenizer_content_sha256": tokens["sha256"],
               "source_jsonl_sha256": file["sha256"],
               "description": "Original local checkpoint self-generated traces; preserved prompt and continuation IDs",
               "model_source": source, "tokenizer_source": tokens,
               "packaged_tokenizer": packaged, "prompt_source": prompt_source,
               "generation": generation, "generation_cache_policy": "use_cache=False; full-prefix recomputation",
               "source_load": getattr(model, "_openbonsai_source_receipt", None),
               "code": code, "file": file, "counts": counts,
               "quality_claim": False, "external_receipt_authenticity": "Hashes bind content; producer attribution requires trusted execution"}
    write_json(output / "producer.json", receipt)
    return receipt
