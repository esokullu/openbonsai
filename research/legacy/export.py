"""Standalone research artifact with packed ternary weights and retained floats.

This is not GGUF and has no optimized inference kernel. Loading uses the native
Qwen3.5 architecture and a slow reference linear operator that dequantizes packed
weights. Both the format and every stored byte are explicit in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import copy
from pathlib import Path
import tempfile

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from accelerate import init_empty_weights
from transformers.initialization import no_init_weights

from .adapter import config_from_dict, get_text_layers, iter_ternary_modules, model_from_config
from .quantizer import PackedTernaryLinear, TernaryLinear
from .packing import BASE3_FORMAT, BITCOS_FORMAT, FORMAT as GROUPED_FORMAT, repack_grouped_weight, validate_packed_state


# Stable on-disk identifier: preserve compatibility with pre-OpenBonsai artifacts.
FORMAT = "scaleq-qwen35.research.v1"
BITCOS_ARTIFACT_FORMAT = "openbonsai-qwen35.research.v2"
BASE3_ARTIFACT_FORMAT = "openbonsai-qwen35.research.v3"
_CHUNK_BYTES = 64 * 1024 * 1024


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_description(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "tensor_bytes": tensor.numel() * tensor.element_size()}


def _all_tensors(model):
    """Include persistent state and nonpersistent buffers, without alias loss."""
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    state = dict(model.state_dict(keep_vars=True))
    persistent_names = set(state)
    for name, tensor in buffers.items():
        state.setdefault(name, tensor)
    for name, tensor in parameters.items():
        state.setdefault(name, tensor)
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Cannot export non-tensor model state: {name}")
        if name in parameters:
            kind = "parameter"
        elif name in buffers:
            kind = "buffer"
        else:
            raise ValueError(f"Unrecognized state entry: {name}")
        yield name, tensor, kind, name in persistent_names


def _under_module(name, module_names):
    return any(name.startswith(prefix + ".") for prefix in module_names)


def _write_tensor_file(directory, filename, tensors, files):
    # Reuse unaliased CPU storage (including large retained embeddings). Only
    # tensors sharing storage inside this file need a copy for safetensors.
    # GPU-to-CPU transfers already create independent storage.
    payload, seen_storage = {}, set()
    for name, tensor in tensors.items():
        value = tensor.detach().to("cpu").contiguous()
        storage = value.untyped_storage().data_ptr()
        if value.numel() and storage in seen_storage:
            value = value.clone()
            storage = value.untyped_storage().data_ptr()
        if value.numel():
            seen_storage.add(storage)
        payload[name] = value
    path = directory / filename
    save_file(payload, str(path))
    files[filename] = {"bytes": path.stat().st_size, "sha256": _digest(path)}


def _destination(path):
    destination = Path(path).expanduser().absolute()
    if destination.is_symlink() or (destination.exists() and (
            not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {destination}")
    return destination


def _packing_option(packing):
    if packing not in ("2bit", "bitcos", "base3", "auto"):
        raise ValueError("packing must be '2bit', 'bitcos', 'base3', or 'auto'")


def _artifact_format(module_formats):
    formats = set(module_formats)
    if BASE3_FORMAT in formats:
        return BASE3_ARTIFACT_FORMAT
    return BITCOS_ARTIFACT_FORMAT if BITCOS_FORMAT in formats else FORMAT


def _finalize_manifest(staging, manifest):
    """Resolve the manifest's self-inclusive byte count before publication."""
    manifest["format"] = _artifact_format(
        item["state"]["format"] for item in manifest["packed_modules"].values())
    payload_bytes = sum(entry["bytes"] for entry in manifest["files"].values())
    supplementary = manifest.get("supplementary_files")
    for _ in range(16):
        encoded = _json_bytes(manifest)
        expected = {"manifest_bytes": len(encoded), "artifact_bytes": payload_bytes + len(encoded)}
        if supplementary is not None:
            extra = sum(item["bytes"] for item in supplementary.values())
            expected.update(supplementary_bytes=extra, directory_bytes=payload_bytes + len(encoded) + extra)
        if all(manifest["storage"].get(key) == value for key, value in expected.items()):
            (staging / "manifest.json").write_bytes(encoded)
            return
        manifest["storage"].update(expected)
    raise RuntimeError("Could not finalize artifact size accounting")


def _publish(staging, destination):
    if destination.exists():
        # rmdir only succeeds if the preexisting directory is still empty.
        destination.rmdir()
    os.replace(staging, destination)


def save_compressed(model, path, metadata=None, *, packing="auto"):
    """Atomically save a standalone artifact; never overwrite a nonempty folder.

    Packed codes and fp16 group scales replace all state inside ternary modules.
    Every other parameter and registered buffer retains its original dtype.
    """
    _packing_option(packing)
    get_text_layers(model)  # Fail closed on other architecture families.
    destination = _destination(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    modules = dict(iter_ternary_modules(model))
    if not modules:
        raise ValueError("Model contains no ternary modules to export")
    metadata = {} if metadata is None else metadata
    _json_bytes(metadata)  # Validate metadata before creating any output.
    manifest = {
        "format": FORMAT,
        "runtime": "native-qwen3.5-with-reference-dequantization",
        "optimized_kernels": False,
        "gguf": False,
        "metadata": metadata,
        "files": {},
        "packed_modules": {},
        "retained_tensors": {},
        "storage": {"packed_tensor_bytes": 0, "retained_tensor_bytes": 0,
                    "manifest_bytes": 0, "artifact_bytes": 0},
    }
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        config_path = staging / "config.json"
        config_path.write_bytes(_json_bytes(model.config.to_dict()))
        manifest["files"]["config.json"] = {"bytes": config_path.stat().st_size, "sha256": _digest(config_path)}

        for index, (name, module) in enumerate(sorted(modules.items())):
            packed = module.packed_state(packing=packing)
            tensors = {key: value for key, value in packed.items() if isinstance(value, torch.Tensor)}
            state_metadata = {key: value for key, value in packed.items() if not isinstance(value, torch.Tensor)}
            _json_bytes(state_metadata)
            filename = f"ternary-{index:05d}.safetensors"
            _write_tensor_file(staging, filename, tensors, manifest["files"])
            # The input dtype is separate from stored fp16 group scales.
            reference = module._compute_reference if isinstance(module, PackedTernaryLinear) else module.weight
            dtype = str(reference.dtype).removeprefix("torch.")
            descriptions = {key: _tensor_description(value) for key, value in tensors.items()}
            manifest["packed_modules"][name] = {
                "file": filename, "compute_dtype": dtype,
                "state": state_metadata, "tensors": descriptions,
            }
            manifest["storage"]["packed_tensor_bytes"] += sum(x["tensor_bytes"] for x in descriptions.values())

        chunk, chunk_bytes, chunk_index = {}, 0, 0
        chunk_names = []

        def flush():
            nonlocal chunk, chunk_bytes, chunk_index, chunk_names
            if not chunk:
                return
            filename = f"retained-{chunk_index:05d}.safetensors"
            _write_tensor_file(staging, filename, chunk, manifest["files"])
            for name in chunk_names:
                manifest["retained_tensors"][name]["file"] = filename
            chunk, chunk_names, chunk_bytes = {}, [], 0
            chunk_index += 1

        for name, tensor, kind, persistent in sorted(_all_tensors(model), key=lambda item: item[0]):
            if _under_module(name, modules):
                continue
            description = _tensor_description(tensor)
            if chunk and chunk_bytes + description["tensor_bytes"] > _CHUNK_BYTES:
                flush()
            chunk[name] = tensor
            chunk_names.append(name)
            chunk_bytes += description["tensor_bytes"]
            manifest["retained_tensors"][name] = {**description, "kind": kind, "persistent": persistent}
            manifest["storage"]["retained_tensor_bytes"] += description["tensor_bytes"]
        flush()
        _finalize_manifest(staging, manifest)
        _publish(staging, destination)
    return manifest


def _artifact_file(root, name, description):
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid artifact path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError(f"Invalid artifact path: {name}")
    if (not isinstance(description, dict) or type(description.get("bytes")) is not int
            or description["bytes"] < 0 or not isinstance(description.get("sha256"), str)
            or len(description["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in description["sha256"])):
        raise ValueError(f"Invalid artifact file metadata: {name}")
    if any((root / Path(*relative.parts[:index])).is_symlink()
           for index in range(1, len(relative.parts) + 1)):
        raise ValueError(f"Artifact path cannot be a symbolic link: {name}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes its directory: {name}") from exc
    if not path.is_file() or path.stat().st_size != description["bytes"]:
        raise ValueError(f"Missing or size-mismatched artifact file: {name}")
    if _digest(path) != description["sha256"]:
        raise ValueError(f"Checksum mismatch for artifact file: {name}")
    return path


def _check_tensor(name, tensor, description):
    if _tensor_description(tensor) != {key: description[key] for key in ("shape", "dtype", "tensor_bytes")}:
        raise ValueError(f"Tensor metadata mismatch: {name}")


def _torch_dtype(name):
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
               "float64": torch.float64}
    if name not in choices:
        raise ValueError(f"Unsupported packed compute dtype: {name}")
    return choices[name]


def _read_artifact(root):
    """Validate file and manifest contracts without constructing a model."""
    try:
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Artifact manifest cannot be a symbolic link")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or manifest.get("format") not in (FORMAT, BITCOS_ARTIFACT_FORMAT, BASE3_ARTIFACT_FORMAT):
            raise ValueError(f"Unsupported compressed artifact format: {manifest.get('format') if isinstance(manifest, dict) else None!r}")
        _json_bytes(manifest)  # Reject JSON NaN and other noncanonical metadata.
        files, modules, retained, storage = (manifest[key] for key in (
            "files", "packed_modules", "retained_tensors", "storage"))
        if not all(isinstance(item, dict) for item in (files, modules, retained, storage)) or not modules:
            raise ValueError("Invalid artifact manifest mappings")
        if "config.json" not in files or "manifest.json" in files:
            raise ValueError("Artifact must contain a separate config.json")
        checked = {name: _artifact_file(root, name, description) for name, description in files.items()}
        payload_files = {"config.json"}
        packed_bytes, retained_bytes = 0, 0
        module_formats = set()
        for name, info in modules.items():
            if not isinstance(name, str) or not all(name.split(".")):
                raise ValueError("Invalid packed module name")
            if info["file"] in payload_files or info["file"] not in files:
                raise ValueError(f"Duplicate or missing packed module file: {name}")
            payload_files.add(info["file"])
            _torch_dtype(info["compute_dtype"])
            if not isinstance(info["state"], dict) or not isinstance(info["tensors"], dict):
                raise ValueError(f"Invalid packed module metadata: {name}")
            if set(info["state"]) & set(info["tensors"]):
                raise ValueError(f"Packed state metadata shadows tensor keys: {name}")
            if info["state"].get("format") not in (GROUPED_FORMAT, BITCOS_FORMAT, BASE3_FORMAT):
                raise ValueError(f"Unsupported packed module format: {name}")
            module_formats.add(info["state"]["format"])
            packed_bytes += sum(_description_bytes(item) for item in info["tensors"].values())
        expected_format = _artifact_format(module_formats)
        if manifest["format"] != expected_format:
            raise ValueError("Artifact format does not match its packed module formats")
        for name, description in retained.items():
            if not isinstance(name, str) or not all(name.split(".")) or _under_module(name, modules):
                raise ValueError(f"Invalid retained tensor name: {name}")
            if description["file"] not in files or description["file"] in payload_files:
                raise ValueError(f"Invalid retained tensor file: {name}")
            if description["kind"] not in ("buffer", "parameter") or type(description["persistent"]) is not bool:
                raise ValueError(f"Invalid retained tensor metadata: {name}")
            retained_bytes += _description_bytes(description)
        payload_files.update(item["file"] for item in retained.values())
        if payload_files != set(files):
            raise ValueError("Artifact contains unreferenced payload files")
        expected_storage = {
            "packed_tensor_bytes": packed_bytes, "retained_tensor_bytes": retained_bytes,
            "manifest_bytes": manifest_path.stat().st_size,
            "artifact_bytes": sum(item["bytes"] for item in files.values()) + manifest_path.stat().st_size,
        }
        supplementary = manifest.get("supplementary_files")
        if supplementary is not None:
            if not isinstance(supplementary, dict) or set(supplementary) & (set(files) | {"manifest.json"}):
                raise ValueError("Invalid supplementary file mapping")
            for name, description in supplementary.items():
                _artifact_file(root, name, description)
            extra = sum(item["bytes"] for item in supplementary.values())
            expected_storage.update(supplementary_bytes=extra, directory_bytes=expected_storage["artifact_bytes"] + extra)
        elif "supplementary_bytes" in storage or "directory_bytes" in storage:
            raise ValueError("Supplementary byte accounting requires supplementary_files")
        if any(type(storage.get(key)) is not int or storage[key] != value for key, value in expected_storage.items()):
            raise ValueError("Artifact total byte accounting mismatch")
        config_from_dict(json.loads(checked["config.json"].read_text()))
        return manifest, checked
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid compressed artifact metadata") from exc


def _description_bytes(description):
    shape, dtype = description["shape"], getattr(torch, description["dtype"], None)
    if (not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(dtype, torch.dtype) or type(description.get("tensor_bytes")) is not int):
        raise ValueError("Invalid tensor metadata")
    count = 1
    for dim in shape:
        count *= dim
    expected = count * torch.empty(0, dtype=dtype).element_size()
    if description["tensor_bytes"] != expected:
        raise ValueError("Tensor byte accounting mismatch")
    return expected


def _packed_state(checked, name, info):
    payload = load_file(str(checked[info["file"]]), device="cpu")
    if set(payload) != set(info["tensors"]):
        raise ValueError(f"Packed tensor keys mismatch: {name}")
    for key, tensor in payload.items():
        _check_tensor(f"{name}.{key}", tensor, info["tensors"][key])
    state = {**info["state"], **payload}
    validate_packed_state(state)
    return state


def _validate_retained_headers(checked, retained):
    """Check shape/dtype/key metadata without materializing retained weights."""
    by_file = {}
    for name, description in retained.items():
        by_file.setdefault(description["file"], {})[name] = description
    dtype_names = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32",
                   "I64": "int64", "U16": "uint16", "U32": "uint32", "U64": "uint64",
                   "F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
    for filename, descriptions in by_file.items():
        with safe_open(str(checked[filename]), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(descriptions):
                raise ValueError(f"Retained tensor keys mismatch in {filename}")
            for name, description in descriptions.items():
                tensor_slice = handle.get_slice(name)
                if (tensor_slice.get_shape() != description["shape"]
                        or dtype_names.get(tensor_slice.get_dtype()) != description["dtype"]):
                    raise ValueError(f"Retained tensor metadata mismatch: {name}")


def _supplementary_files(root, manifest):
    descriptions = copy.deepcopy(manifest.get("supplementary_files", {}))
    tokenizer = root / "tokenizer"
    if tokenizer.is_symlink():
        raise ValueError("Tokenizer directory cannot be a symbolic link")
    if tokenizer.exists():
        if not tokenizer.is_dir():
            raise ValueError("Tokenizer assets must be a directory")
        for directory, dirs, files in os.walk(tokenizer, followlinks=False):
            for child in dirs + files:
                if (Path(directory) / child).is_symlink():
                    raise ValueError("Tokenizer assets cannot contain symbolic links")
            for filename in files:
                path = Path(directory) / filename
                name = path.relative_to(root).as_posix()
                if name not in descriptions:
                    descriptions[name] = {"bytes": path.stat().st_size, "sha256": _digest(path)}
    return descriptions


def _copy_checked_file(source, destination, description):
    """Copy with bounded memory and detect source changes after validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    if count != description["bytes"] or digest.hexdigest() != description["sha256"]:
        raise ValueError(f"Source changed while copying artifact file: {source.name}")


def repack_compressed(source, path, *, packing="auto"):
    """Repack an artifact one projection at a time on CPU, without a model.

    The retained safetensors, configuration, metadata, compute dtypes and
    tokenizer assets are preserved. Only ternary code storage is changed.
    Model-artifact accounting excludes supplementary tokenizer bytes, which
    are recorded separately when present.
    """
    _packing_option(packing)
    root = Path(source).expanduser().resolve()
    destination = Path(path).expanduser().absolute()
    resolved_destination = destination.resolve()
    if (root == resolved_destination or root in resolved_destination.parents
            or resolved_destination in root.parents):
        raise ValueError("Source and output must be separate, non-nested artifact directories")
    destination = _destination(destination)
    manifest, checked = _read_artifact(root)
    _validate_retained_headers(checked, manifest["retained_tensors"])
    supplementary = _supplementary_files(root, manifest)
    if set(supplementary) & (set(manifest["files"]) | {"manifest.json"}):
        raise ValueError("Supplementary files overlap model artifact files")
    result = copy.deepcopy(manifest)
    result["storage"]["packed_tensor_bytes"] = 0
    if supplementary:
        result["supplementary_files"] = supplementary
    destination.parent.mkdir(parents=True, exist_ok=True)
    packed_files = {item["file"] for item in manifest["packed_modules"].values()}
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        for filename, original in checked.items():
            if filename not in packed_files:
                _copy_checked_file(original, staging / filename, manifest["files"][filename])
        for name, info in manifest["packed_modules"].items():
            original = _packed_state(checked, name, info)
            packed = repack_grouped_weight(original, packing=packing)
            del original
            tensors = {key: value for key, value in packed.items() if isinstance(value, torch.Tensor)}
            state_metadata = {key: value for key, value in packed.items() if not isinstance(value, torch.Tensor)}
            filename = info["file"]
            (staging / filename).parent.mkdir(parents=True, exist_ok=True)
            _write_tensor_file(staging, filename, tensors, result["files"])
            descriptions = {key: _tensor_description(value) for key, value in tensors.items()}
            result["packed_modules"][name] = {**info, "state": state_metadata, "tensors": descriptions}
            result["storage"]["packed_tensor_bytes"] += sum(item["tensor_bytes"] for item in descriptions.values())
            del packed, tensors
        for filename, description in supplementary.items():
            original = _artifact_file(root, filename, description)
            _copy_checked_file(original, staging / filename, description)
        _finalize_manifest(staging, result)
        _publish(staging, destination)
    return result


def load_compressed(path, device="cpu"):
    """Load a self-contained, checksummed research artifact without network I/O."""
    root = Path(path).expanduser().absolute()
    manifest, checked = _read_artifact(root)
    files = manifest["files"]
    config = config_from_dict(json.loads(checked["config.json"].read_text()))
    # Parameters are immediately replaced by artifact tensors: do not allocate
    # a full FP32 copy of the original model first. Keep buffers on CPU so native
    # constructor-defined rotary buffers and their nonregistered aliases remain
    # initialized; the artifact restores all registered buffers below.
    with no_init_weights(), init_empty_weights(include_buffers=False):
        model = model_from_config(config)

    consumed_files = {"config.json"}
    for name, info in manifest["packed_modules"].items():
        native = model.get_submodule(name)
        if not isinstance(native, nn.Linear):
            raise ValueError(f"Packed target is not a native linear projection: {name}")
        filename = info["file"]
        state = _packed_state(checked, name, info)
        if list(state["shape"]) != [native.out_features, native.in_features]:
            raise ValueError(f"Packed projection shape mismatch: {name}")
        replacement = TernaryLinear.from_packed_state(state, dtype=_torch_dtype(info["compute_dtype"]))
        model.set_submodule(name, replacement)
        consumed_files.add(filename)

    module_names = set(manifest["packed_modules"])
    expected = {name: (tensor, kind, persistent) for name, tensor, kind, persistent in _all_tensors(model)
                if not _under_module(name, module_names)}
    retained = manifest["retained_tensors"]
    if set(expected) != set(retained):
        missing = sorted(set(expected) - set(retained))
        extra = sorted(set(retained) - set(expected))
        raise ValueError(f"Retained state is incomplete: missing={missing[:5]}, extra={extra[:5]}")
    by_file = {}
    for name, description in retained.items():
        by_file.setdefault(description["file"], {})[name] = description
    for filename, descriptions in by_file.items():
        payload = load_file(str(checked[filename]), device="cpu")
        if set(payload) != set(descriptions):
            raise ValueError(f"Retained tensor keys mismatch in {filename}")
        for name, tensor in payload.items():
            description = descriptions[name]
            _check_tensor(name, tensor, description)
            original, kind, persistent = expected[name]
            if list(original.shape) != list(tensor.shape) or description["kind"] != kind or description["persistent"] != persistent:
                raise ValueError(f"Retained native-state contract mismatch: {name}")
            module_name, _, attribute = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            if kind == "parameter":
                module._parameters[attribute] = nn.Parameter(tensor, requires_grad=False)
            else:
                module._buffers[attribute] = tensor
        consumed_files.add(filename)
    if consumed_files != set(files):
        raise ValueError("Artifact contains unreferenced payload files")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
