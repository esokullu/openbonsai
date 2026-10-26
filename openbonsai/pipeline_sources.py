"""Sealed local-source and completed-stage handoffs for the research pipeline.

No hub access, optimizer resume, checkpoint copying, or data provenance inference.
The producer of token records must explicitly declare their original checkpoint
and tokenizer identities. Hashes establish identity, not quality or absence of
semantic overlap with evaluation data.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

from .eval_backends import _identity, local_directory_identity

JOB_FORMAT = "openbonsai.pipeline-job.v1"
STAGE_FORMAT = "openbonsai.pipeline-stage.v1"
TRACE_FORMAT = "openbonsai.pipeline-traces.v1"
ROLES = {"calibration", "search_validation", "evaluation"}
STAGES = {"decoder", "head", "embedding"}
_TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def json_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open("rb") as stream:
        return json.load(stream, object_pairs_hook=unique,
                         parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Expected a portable relative file path")
    path = Path(relative)
    if path.is_absolute() or relative != path.as_posix() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Path escapes owned directory")
    result = Path(root)
    for part in (None, *path.parts):
        if part is not None:
            result = result / part
        if result.is_symlink():
            raise ValueError("Symlink paths are unsupported")
    return result


def code_identity():
    """Bind installed pipeline/library Python sources, without Git assumptions."""
    root = Path(__file__).parent
    files = {str(path.relative_to(root).as_posix()): _identity(path)["sha256"]
             for path in sorted(root.rglob("*.py"))}
    return {"files": files, "sha256": json_sha256(files)}


def tokenizer_identity(directory):
    """Hash the exact supported tokenizer subset; ignore checkpoint weights."""
    root = Path(directory).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Tokenizer source must be a materialized local directory")
    for required in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / required).is_file() or (root / required).is_symlink():
            raise ValueError(f"Tokenizer requires a regular {required}")
    config = _read_json(root / "tokenizer_config.json")
    if not isinstance(config, dict) or config.get("auto_map") or config.get("fast_tokenizer_files"):
        raise ValueError("Custom or versioned tokenizer files are outside this narrow package contract")
    selected = [root / name for name in sorted(_TOKENIZER_FILES) if (root / name).exists()]
    templates = root / "chat_templates"
    if templates.exists():
        if templates.is_symlink() or not templates.is_dir():
            raise ValueError("chat_templates must be a materialized directory")
        selected += sorted(templates.glob("*.jinja"))
    files = {}
    for path in selected:
        info = _identity(path)
        files[path.relative_to(root).as_posix()] = {"sha256": info["sha256"], "bytes": info["bytes"]}
    return {"kind": "tokenizer-subset-sha256", "path": str(root), "files": files,
            "sha256": json_sha256(files), "bytes": sum(v["bytes"] for v in files.values())}


def package_tokenizer(directory, output):
    """Copy only recognized tokenizer assets byte-for-byte to a new directory."""
    identity = tokenizer_identity(directory)
    output = Path(output).absolute()
    if output.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("Tokenizer package must not modify its original source directory")
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in identity["files"].items():
        source = _inside(identity["path"], name)
        destination = _inside(output, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        actual = _identity(destination)
        if any(actual[k] != expected[k] for k in ("sha256", "bytes")):
            raise RuntimeError("Tokenizer changed during packaging")
    result = tokenizer_identity(output)
    if result["sha256"] != identity["sha256"] or tokenizer_identity(directory) != identity:
        raise RuntimeError("Tokenizer identity changed during packaging")
    return result


def checkpoint_tensors(directory):
    """Inspect safetensors headers only; never deserialize a second full model."""
    from safetensors import safe_open
    root = Path(directory)
    config = _read_json(root / "config.json")
    if config.get("model_type") not in {"qwen3_5", "qwen3_5_text"} or config.get("quantization_config") or config.get("auto_map"):
        raise ValueError("Pipeline requires an original native dense Qwen3.5 checkpoint")
    if (root / "adapter_config.json").exists() or list(root.glob("*.bin")):
        raise ValueError("Use merged original BF16 safetensors, not adapter-only or pickle checkpoints")
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("No original safetensors weights found")
    tensors = {}
    for shard in shards:
        if shard.is_symlink():
            raise ValueError("Materialize checkpoint shards before sealing")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                view = handle.get_slice(name)
                dtype, shape = view.get_dtype(), view.get_shape()
                if dtype not in {"BF16", "F32", "I64", "I32", "BOOL"}:
                    raise ValueError(f"Unsupported original checkpoint dtype {dtype}: {name}")
                if len(shape) == 2 and dtype != "BF16":
                    raise ValueError(f"Original matrix must already be BF16, not cast from {dtype}: {name}")
                tensors[name] = {"shape": shape, "dtype": dtype, "file": shard.name}
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = _read_json(index).get("weight_map")
        if weight_map != {name: info["file"] for name, info in tensors.items()}:
            raise ValueError("Safetensors shard index does not exactly cover source tensors")
    elif len(shards) != 1 or shards[0].name != "model.safetensors":
        raise ValueError("Expected model.safetensors or an exact shard index")
    return tensors


def _trace_records(path, *, sequence_length, max_samples, pad_token_id, vocab_size):
    """Strict IDs/masks without silently stripping input provenance fields."""
    import torch
    _positive(sequence_length, "sequence_length")
    if max_samples is not None:
        _positive(max_samples, "max_samples")
    records, hashes, raw_hashes = [], [], []
    consumed = hashlib.sha256()
    counts = {"source_tokens": 0, "active_tokens": 0, "padding_tokens": 0, "truncated_tokens": 0}
    with Path(path).open("rb") as stream:
        for line in stream:
            if max_samples is not None and len(records) == max_samples:
                break
            consumed.update(line)
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("Trace record must be an object")
            ids = raw.get("input_ids")
            if not isinstance(ids, list) or not ids or any(type(v) is not int or not 0 <= v < vocab_size for v in ids):
                raise ValueError("Trace IDs must be nonempty integers within the original vocabulary")
            mask = raw.get("attention_mask", [1] * len(ids))
            if not isinstance(mask, list) or len(mask) != len(ids) or any(type(v) is not int or v not in (0, 1) for v in mask):
                raise ValueError("Trace mask must match IDs with integer 0/1 values")
            kept = min(len(ids), sequence_length)
            missing = sequence_length - kept
            final_ids, final_mask = ids[:kept] + [pad_token_id] * missing, mask[:kept] + [0] * missing
            if not any(final_mask) or not any(a and b for a, b in zip(final_mask, final_mask[1:])):
                raise ValueError("Trace must retain at least one valid adjacent next-token position")
            record = {"input_ids": torch.tensor([final_ids], dtype=torch.long),
                      "attention_mask": torch.tensor([final_mask], dtype=torch.long)}
            records.append(record)
            hashes.append(json_sha256({"input_ids": final_ids, "attention_mask": final_mask}))
            raw_hashes.append(json_sha256({"input_ids": ids, "attention_mask": mask}))
            counts["source_tokens"] += len(ids)
            counts["active_tokens"] += sum(final_mask)
            counts["padding_tokens"] += missing
            counts["truncated_tokens"] += len(ids) - kept
    if not records:
        raise ValueError("No trace records")
    return records, {**counts, "num_records": len(records), "record_sha256s": hashes,
                     "raw_token_record_sha256s": raw_hashes, "consumed_jsonl_sha256": consumed.hexdigest()}


@dataclass
class PipelineContext:
    job: dict
    job_path: Path
    job_sha256: str
    source_provenance: dict
    records_by_role: dict
    record_receipts_by_role: dict

    @property
    def calibration_records(self):
        return self.records_by_role["calibration"]

    @property
    def calibration_receipt(self):
        return self.record_receipts_by_role["calibration"]

    @property
    def tokenizer_directory(self):
        return Path(self.job["source"]["tokenizer_directory"]["path"])


def _seal_digest(value):
    result = dict(value)
    result["sha256"] = json_sha256(value)
    return result


def _check_seal(value, name):
    if not isinstance(value, dict) or "sha256" not in value:
        raise ValueError(f"Missing {name} seal")
    unsigned = {k: v for k, v in value.items() if k != "sha256"}
    if _digest(value["sha256"], name) != json_sha256(unsigned):
        raise ValueError(f"{name} seal mismatch")
    return unsigned


def _check_job_seal(job):
    if not isinstance(job, dict) or "job_sha256" not in job:
        raise ValueError("Missing job seal")
    if _digest(job["job_sha256"], "job") != json_sha256({k: v for k, v in job.items() if k != "job_sha256"}):
        raise ValueError("Job seal mismatch")


def _validate_recipes(decoder, head, embedding, assembly):
    from dataclasses import fields
    from .trainer import TrainingConfig
    from .head_calibration import HeadTrainingConfig
    from .embedding_calibration import EmbeddingTrainingConfig
    quant_common = {"group_size", "block_size", "rotation_seed", "lora_rank", "factor_mapping",
                    "hard_phase", "soft_fraction", "max_sharpness"}
    for stage, config, cls in (("decoder", decoder, TrainingConfig), ("head", head, HeadTrainingConfig),
                                ("embedding", embedding, EmbeddingTrainingConfig)):
        required = {"quantization", "training", "packing"} | ({"cache"} if stage == "head" else set())
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f"Explicit {stage} recipe fields required: {sorted(required)}")
        if set(config["training"]) != {f.name for f in fields(cls)}:
            raise ValueError(f"All {stage} training options must be explicit")
        cls(**config["training"]).validate()
        quant = config["quantization"]
        required_quant = quant_common | ({"aggressive_gates"} if stage == "decoder" else {"row_chunk_size", "initialization_seed"})
        if not isinstance(quant, dict) or set(quant) != required_quant:
            raise ValueError(f"All {stage} quantization options must be explicit")
        for key in ("group_size", "block_size", "row_chunk_size"):
            if key in quant:
                _positive(quant[key], key)
        if quant["block_size"] & (quant["block_size"] - 1):
            raise ValueError("Rotation block_size must be a power of two")
        for key in ("rotation_seed", "initialization_seed", "lora_rank"):
            if key in quant and (type(quant[key]) is not int or (key == "lora_rank" and quant[key] < 0)):
                raise ValueError(f"Invalid {key}")
        if (quant["factor_mapping"] not in {"exp", "sigmoid"}
                or quant["hard_phase"] not in {"soft_surrogate", "frozen_jacobian"}
                or type(quant.get("aggressive_gates", False)) is not bool):
            raise ValueError("Invalid quantization mapping/phase/gate policy")
        import math
        for key in ("soft_fraction", "max_sharpness"):
            if type(quant[key]) not in (int, float) or not math.isfinite(quant[key]):
                raise ValueError(f"Invalid {key}")
        if not 0 < quant["soft_fraction"] < 1 or quant["max_sharpness"] <= 0:
            raise ValueError("Invalid soft/hard schedule")
        if config["packing"] not in {"auto", "2bit", "bitcos", "base3"}:
            raise ValueError("Unsupported ternary packing")
        if stage == "head":
            cache = config["cache"]
            if not isinstance(cache, dict) or set(cache) != {"max_positions", "shard_positions"}:
                raise ValueError("Explicit head cache limits required")
            _positive(cache["shard_positions"], "shard_positions")
            if cache["max_positions"] is not None:
                _positive(cache["max_positions"], "max_positions")
    if not isinstance(assembly, dict) or set(assembly) != {"coverage", "output_scope", "budget_bytes", "row_chunk_size", "cache_decoded_weights"}:
        raise ValueError("All assembly options must be explicit")
    if (assembly["coverage"] not in {"all", "decoder"}
            or assembly["output_scope"] not in {"language-only", "retain-vision"}
            or type(assembly["cache_decoded_weights"]) is not bool):
        raise ValueError("Invalid assembly coverage/scope/cache policy")
    _positive(assembly["budget_bytes"], "budget_bytes")
    _positive(assembly["row_chunk_size"], "row_chunk_size")


def seal_job(*, model_directory, tokenizer_directory, traces_by_role, decoder, head,
             embedding, assembly, output):
    """Seal explicit source/data/config identities; never generate or infer data.

    Each trace spec has path, sequence_length, max_samples, and producer. The
    producer identifies a JSON receipt via its exact path and SHA256. It must
    declare kind/source_content_sha256/tokenizer_content_sha256/source_jsonl_sha256
    and a description. This is an attributable caller assertion, not proof
    that arbitrary externally supplied token IDs came from that tokenizer.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(output.resolve().is_relative_to(Path(path).resolve()) for path in (model_directory, tokenizer_directory)):
        raise ValueError("Job output must be outside original source/tokenizer directories")
    source = local_directory_identity(model_directory)
    tensors = checkpoint_tensors(model_directory)
    tokenizer_source = tokenizer_identity(tokenizer_directory)
    if not isinstance(traces_by_role, dict) or "calibration" not in traces_by_role or set(traces_by_role) - ROLES:
        raise ValueError("Supply calibration and optional distinct search_validation/evaluation roles")
    _validate_recipes(decoder, head, embedding, assembly)
    config = _read_json(Path(model_directory) / "config.json")
    config = config["text_config"] if config["model_type"] == "qwen3_5" else config
    vocab_size = _positive(config["vocab_size"], "vocab_size")
    pad_id = config.get("pad_token_id")
    if type(pad_id) is not int or not 0 <= pad_id < vocab_size:
        raise ValueError("Original text config must specify an in-vocabulary pad_token_id")
    prepared = {}
    for role, spec in traces_by_role.items():
        if not isinstance(spec, dict) or set(spec) != {"path", "sequence_length", "max_samples", "producer"}:
            raise ValueError("Trace spec requires exactly path/sequence_length/max_samples/producer")
        producer_file = spec["producer"]
        if not isinstance(producer_file, dict) or set(producer_file) != {"path", "sha256"}:
            raise ValueError("Producer must identify a materialized JSON receipt by path and SHA256")
        producer_identity = _identity(producer_file["path"], _digest(producer_file["sha256"], "producer"))
        producer = _read_json(producer_identity["path"])
        file = _identity(spec["path"])
        if (not isinstance(producer, dict)
                or not isinstance(producer.get("kind"), str) or not producer["kind"].strip()
                or producer.get("source_content_sha256") != source["sha256"]
                or producer.get("tokenizer_content_sha256") != tokenizer_source["sha256"]
                or producer.get("source_jsonl_sha256") != file["sha256"]
                or not isinstance(producer.get("description"), str) or not producer["description"].strip()):
            raise ValueError("Trace producer must bind this exact JSONL, original checkpoint and tokenizer")
        if "role" in producer and producer["role"] != role:
            raise ValueError("Producer role differs from requested trace role")
        if producer["kind"] == "openbonsai-original-ayot.v1" and producer.get("status") != "complete":
            raise ValueError("AYOT producer receipt is incomplete")
        _, stats = _trace_records(spec["path"], sequence_length=spec["sequence_length"],
                                  max_samples=spec["max_samples"], pad_token_id=pad_id, vocab_size=vocab_size)
        if _identity(spec["path"]) != file:
            raise RuntimeError("Trace file changed while sealing")
        prepared[role] = (spec, file, stats, producer_identity, producer)
    for role, (_, _, stats, _, _) in prepared.items():
        for other, (_, _, other_stats, _, _) in prepared.items():
            if role < other and (set(stats["record_sha256s"]) & set(other_stats["record_sha256s"])
                                 or set(stats["raw_token_record_sha256s"]) & set(other_stats["raw_token_record_sha256s"])):
                raise ValueError("Trace roles overlap in exact token records")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = package_tokenizer(tokenizer_directory, output / "tokenizer")
    receipts = {}
    for role, (spec, file, stats, producer_identity, producer) in prepared.items():
        path = output / f"{role}.jsonl"
        with Path(file["path"]).open("rb") as src, path.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        copied = _identity(path, file["sha256"])
        producer_path = output / f"{role}.producer.json"
        with Path(producer_identity["path"]).open("rb") as src, producer_path.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        copied_producer = _identity(producer_path, producer_identity["sha256"])
        receipts[role] = _seal_digest({"format": TRACE_FORMAT, "role": role, "file": copied,
            "producer": producer, "producer_file": copied_producer,
            "provenance_status": "authenticated-receipt; producer claims are not independent generation proof",
            "sequence_length": spec["sequence_length"],
            "max_samples": spec["max_samples"], "pad_token_id": pad_id, "vocab_size": vocab_size,
            "padding_side": "right", "truncation_side": "right", **stats})
    provenance = {"source_checkpoint": "sha256:" + source["sha256"],
                  "source_content_sha256": source["sha256"], "source_revision": source["sha256"]}
    job = {"format": JOB_FORMAT,
        "source": {"model_directory": source, "tokenizer_directory": tokenizer, "checkpoint_tensors": tensors},
        "source_provenance": provenance, "traces_by_role": receipts, "decoder": decoder, "head": head,
        "embedding": embedding, "assembly": assembly, "code": code_identity()}
    job["job_sha256"] = json_sha256(job)
    _write_json(output / "job.json", job)
    return load_context(output / "job.json")


def verify_context(context):
    """Fail before reuse if any sealed job, source, tokenizer, code, or data changed."""
    job = _read_json(context.job_path)
    _check_job_seal(job)
    if job != context.job or job["job_sha256"] != context.job_sha256:
        raise ValueError("Context differs from sealed job")
    if code_identity() != job["code"]:
        raise ValueError("Pipeline implementation differs from sealed code")
    for key in ("model_directory", "tokenizer_directory"):
        expected = job["source"][key]
        actual = local_directory_identity(expected["path"]) if key == "model_directory" else tokenizer_identity(expected["path"])
        if actual != expected:
            raise ValueError(f"Sealed {key} changed")
        if key == "tokenizer_directory" and local_directory_identity(expected["path"])["files"] != expected["files"]:
            raise ValueError("Sealed tokenizer-only package contains unexpected files")
    for receipt in job["traces_by_role"].values():
        _check_seal(receipt, "trace")
        if _identity(receipt["file"]["path"]) != receipt["file"]:
            raise ValueError("Sealed trace bytes changed")
        if (_identity(receipt["producer_file"]["path"]) != receipt["producer_file"]
                or _read_json(receipt["producer_file"]["path"]) != receipt["producer"]):
            raise ValueError("Sealed producer receipt changed")
    if context.records_by_role:
        import torch
        if set(context.records_by_role) != set(job["traces_by_role"]):
            raise ValueError("In-memory trace roles differ from sealed job")
        for role, records in context.records_by_role.items():
            receipt = job["traces_by_role"][role]
            if context.record_receipts_by_role.get(role) != receipt or len(records) != receipt["num_records"]:
                raise ValueError("In-memory trace receipt differs from job")
            hashes = []
            for record in records:
                if set(record) != {"input_ids", "attention_mask"}:
                    raise ValueError("Unexpected in-memory trace fields")
                if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                       or value.dtype != torch.long or list(value.shape) != [1, receipt["sequence_length"]]
                       for value in record.values()):
                    raise ValueError("In-memory trace tensor schema changed")
                hashes.append(json_sha256({key: value.reshape(-1).tolist() for key, value in record.items()}))
            if hashes != receipt["record_sha256s"]:
                raise ValueError("In-memory trace records changed")
    return True


def load_context(job_path):
    job_path = Path(job_path).absolute()
    if job_path.is_symlink():
        raise ValueError("Job must be a regular owned file")
    job = _read_json(job_path)
    _check_job_seal(job)
    if job.get("format") != JOB_FORMAT:
        raise ValueError("Unsupported pipeline job")
    _validate_recipes(*(job[key] for key in ("decoder", "head", "embedding", "assembly")))
    expected = job["source"]["model_directory"]["sha256"]
    if job["source_provenance"] != {"source_checkpoint": "sha256:" + expected,
            "source_content_sha256": expected, "source_revision": expected}:
        raise ValueError("Job source provenance is not its sealed original checkpoint")
    context = PipelineContext(job, job_path, job["job_sha256"], job["source_provenance"], {}, {})
    verify_context(context)
    if "calibration" not in job["traces_by_role"] or set(job["traces_by_role"]) - ROLES:
        raise ValueError("Invalid trace roles")
    for role, receipt in job["traces_by_role"].items():
        if receipt["role"] != role or receipt["format"] != TRACE_FORMAT:
            raise ValueError("Trace role mismatch")
        producer = receipt["producer"]
        if (producer.get("source_content_sha256") != expected
                or producer.get("tokenizer_content_sha256") != job["source"]["tokenizer_directory"]["sha256"]
                or producer.get("source_jsonl_sha256") != receipt["file"]["sha256"]):
            raise ValueError("Trace producer differs from job source/tokenizer")
        records, stats = _trace_records(receipt["file"]["path"], sequence_length=receipt["sequence_length"],
            max_samples=receipt["max_samples"], pad_token_id=receipt["pad_token_id"], vocab_size=receipt["vocab_size"])
        if any(receipt[key] != value for key, value in stats.items()):
            raise ValueError("Trace record content differs from receipt")
        context.records_by_role[role], context.record_receipts_by_role[role] = records, receipt
    for role in context.record_receipts_by_role:
        for other in context.record_receipts_by_role:
            if role < other:
                for field in ("record_sha256s", "raw_token_record_sha256s"):
                    if set(context.record_receipts_by_role[role][field]) & set(context.record_receipts_by_role[other][field]):
                        raise ValueError("Trace roles overlap")
    return context


def load_original_source(source, device="cpu"):
    """Load one native source and restore bounded saved FP32 exceptions exactly.

    The normal Transformers BF16 load can cast small retained FP32 tensors. Read
    only each exceptional original tensor back from its sealed shard; never load
    a second shard/model. Unsupported large dtype exceptions fail closed. Native
    unused ``mtp.*`` state is recorded separately and is not compression savings.
    """
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
    from .adapter import load_base_model
    from .text_export import _check_source
    expected_identity = source["model_directory"]
    if local_directory_identity(expected_identity["path"]) != expected_identity:
        raise ValueError("Original checkpoint differs from sealed source")
    expected = source["checkpoint_tensors"]
    if checkpoint_tensors(expected_identity["path"]) != expected:
        raise ValueError("Checkpoint tensor headers differ from source receipt")
    model = load_base_model(expected_identity["path"], device=device, dtype="auto",
                            revision=expected_identity["sha256"], local_files_only=True)
    if type(model) is Qwen3_5ForConditionalGeneration:
        _check_source(model)
    elif type(model) is Qwen3_5ForCausalLM:
        if (model.config.quantization_config if hasattr(model.config, "quantization_config") else None):
            raise ValueError("Original causal model cannot be quantized")
        if (model.config.tie_word_embeddings or model.config._attn_implementation != "eager"
                or model.config.use_cache or model.config.rope_parameters.get("rope_type") != "default"):
            raise ValueError("Original causal model requires untied/eager/no-cache/default-RoPE configuration")
        if any(m.training or "forward" in vars(m) or hasattr(m, "_hf_hook") or m._forward_hooks
               or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks for m in model.modules()):
            raise ValueError("Original causal model contains custom hooks or execution state")
    else:
        raise ValueError("Expected exact native dense Qwen3.5 model class")
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64,
              "I32": torch.int32, "BOOL": torch.bool}
    loaded = model.state_dict(keep_vars=True)
    unused = set(expected) - set(loaded)
    if set(loaded) - set(expected) or any(not name.startswith("mtp.") for name in unused):
        raise ValueError("Native tensor names differ from checkpoint beyond explicitly unused mtp.*")
    restored = {}
    for name, tensor in loaded.items():
        spec = expected[name]
        if tensor.is_meta or list(tensor.shape) != spec["shape"]:
            raise ValueError(f"Native load changed original tensor shape: {name}")
        if tensor.dtype != dtypes[spec["dtype"]]:
            if spec["dtype"] != "F32" or tensor.numel() > 1048576:
                raise ValueError(f"Unsupported original dtype conversion: {name}")
            with safe_open(Path(expected_identity["path"]) / spec["file"], framework="pt", device="cpu") as handle:
                original = handle.get_tensor(name)
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            value = original.to(device=tensor.device)
            if leaf in parent._parameters:
                parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
            elif leaf in parent._buffers:
                parent._buffers[leaf] = value
            else:
                raise ValueError(f"Unsupported native tensor location: {name}")
            restored[name] = {"dtype": "float32", "bytes": value.numel() * value.element_size()}
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Original model must remain frozen")
    if local_directory_identity(expected_identity["path"]) != expected_identity:
        raise RuntimeError("Original checkpoint changed during loading")
    model._openbonsai_source_receipt = {"source_content_sha256": expected_identity["sha256"],
        "restored_saved_dtype_tensors": restored,
        "unused_mtp_tensors": {name: expected[name] for name in sorted(unused)},
        "unused_mtp_is_not_compression": True}
    return model


def load_original(context, device="cpu"):
    """Load the sealed source once; no deep copy or second complete weight master."""
    verify_context(context)
    model = load_original_source(context.job["source"], device=device)
    verify_context(context)
    return model


def load_tokenizer(context):
    from transformers import AutoTokenizer
    verify_context(context)
    tokenizer = AutoTokenizer.from_pretrained(context.tokenizer_directory, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("Pipeline requires the sealed native fast tokenizer")
    return tokenizer


def candidate_provenance(context, stage, name, **extra):
    if stage not in STAGES or not isinstance(name, str) or not name:
        raise ValueError("Invalid pipeline candidate stage/name")
    reserved = {**context.source_provenance, "pipeline_job_sha256": context.job_sha256,
        "pipeline_stage": stage, "original_module_name": name,
        "pipeline_stage_config_sha256": json_sha256(context.job[stage]),
        "pipeline_code_sha256": context.job["code"]["sha256"],
        "pipeline_calibration_sha256": context.calibration_receipt["sha256"]}
    if set(extra) & set(reserved):
        raise ValueError("Cannot override pipeline provenance")
    return {**reserved, **extra}


def validate_stage_output(context, output):
    """Reject input-tree mutations before any stage creates its owned directory."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    protected = (context.job["source"]["model_directory"]["path"],
                 context.job["source"]["tokenizer_directory"]["path"],
                 context.job_path.parent, Path(__file__).parent)
    if any(output.resolve().is_relative_to(Path(path).resolve()) for path in protected):
        raise ValueError("Stage output must be outside sealed source/tokenizer/job/code directories")
    return output


def _tensor_sha256(value):
    import torch
    tensor = value.detach().to("cpu").contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _candidate_description(context, stage, name, candidate):
    from .mixed_formats import validate_candidate
    from .text_export import text_module_name
    validate_candidate(candidate)
    if name.startswith("model.language_model.") or name == "lm_head":
        text_module_name(name)
    elif not (name.startswith("model.layers.") or name == "model.embed_tokens"):
        raise ValueError("Expected an original native language module name")
    if not candidate.encoding.startswith("ternary-"):
        raise ValueError("This staged pipeline only publishes learned ternary candidates")
    provenance = candidate.metadata["provenance"]
    if any(provenance.get(k) != v for k, v in candidate_provenance(context, stage, name).items()):
        raise ValueError("Candidate provenance must be bound before packing/cost selection")
    source_hash = _digest(provenance.get("source_weight_sha256"), "source_weight_sha256")
    rotation = candidate.metadata.get("rotation")
    if rotation is None or "rotation_signs" not in candidate.tensors:
        raise ValueError("Rotated pipeline candidate is missing authoritative signs")
    return {"shape": list(candidate.shape), "module_kind": candidate.module_kind,
        "encoding": candidate.encoding, "source_weight_sha256": source_hash,
        "rotation": rotation, "rotation_signs_sha256": _tensor_sha256(candidate.tensors["rotation_signs"])}


def publish_candidates(context, stage, candidates, output, *, report):
    """Publish one candidate at a time; no receipt means an incomplete stage."""
    from .mixed_formats import save_candidate
    if stage not in STAGES or not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("Expected a nonempty lazy candidate mapping for one supported stage")
    output = validate_stage_output(context, output)
    verify_context(context)
    output.mkdir(parents=True, exist_ok=False)
    entries = {}
    for index, name in enumerate(sorted(candidates)):
        candidate = candidates[name]
        description = _candidate_description(context, stage, name, candidate)
        path = output / f"candidate-{index:05d}.obc"
        save_candidate(candidate, path)
        info = _identity(path)
        entries[name] = {**description, "file": path.name, "sha256": info["sha256"], "bytes": info["bytes"]}
        del candidate
    verify_context(context)
    receipt = _seal_digest({"format": STAGE_FORMAT, "status": "complete", "stage": stage,
        "job_sha256": context.job_sha256, "source_provenance": context.source_provenance,
        "stage_config_sha256": json_sha256(context.job[stage]), "code_sha256": context.job["code"]["sha256"],
        "calibration_sha256": context.calibration_receipt["sha256"], "candidates": entries, "report": report})
    _write_json(output / "stage.json", receipt)
    return receipt


class CandidateStage(Mapping):
    def __init__(self, context, root, receipt):
        self.context, self.root, self.receipt = context, root, receipt

    def __iter__(self):
        return iter(self.receipt["candidates"])

    def __len__(self):
        return len(self.receipt["candidates"])

    def __getitem__(self, name):
        from .mixed_formats import load_candidate
        entry = self.receipt["candidates"][name]
        path = _inside(self.root, entry["file"])
        info = _identity(path, entry["sha256"])
        if info["bytes"] != entry["bytes"]:
            raise ValueError("Candidate file size changed")
        candidate = load_candidate(path)
        actual = _candidate_description(self.context, self.receipt["stage"], name, candidate)
        if any(entry.get(key) != value for key, value in actual.items()):
            raise ValueError("Candidate payload differs from stage receipt")
        return candidate


def load_candidate_stage(context, path, *, expected_stage=None):
    verify_context(context)
    root = Path(path).absolute()
    receipt = _read_json(_inside(root, "stage.json"))
    _check_seal(receipt, "stage")
    stage = receipt.get("stage")
    if (receipt.get("format") != STAGE_FORMAT or receipt.get("status") != "complete"
            or stage not in STAGES or (expected_stage is not None and stage != expected_stage)):
        raise ValueError("Unsupported or incomplete candidate stage")
    expected = {"job_sha256": context.job_sha256, "source_provenance": context.source_provenance,
        "stage_config_sha256": json_sha256(context.job[stage]), "code_sha256": context.job["code"]["sha256"],
        "calibration_sha256": context.calibration_receipt["sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Candidate stage was produced by a different job/source/config/data/code")
    if not isinstance(receipt.get("candidates"), dict) or not receipt["candidates"]:
        raise ValueError("Empty candidate stage")
    for entry in receipt["candidates"].values():
        info = _identity(_inside(root, entry["file"]), entry["sha256"])
        if info["bytes"] != entry["bytes"]:
            raise ValueError("Candidate file size differs from receipt")
    return CandidateStage(context, root, receipt), receipt
