"""Standalone research artifact with packed ternary weights and retained floats.

This is not GGUF and has no optimized inference kernel. Loading uses the native
Qwen3.5 architecture and a slow reference linear operator that dequantizes packed
weights. Both the format and every stored byte are explicit in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers.initialization import no_init_weights

from .adapter import config_from_dict, get_text_layers, iter_ternary_modules, model_from_config
from .quantizer import PackedTernaryLinear, TernaryLinear


FORMAT = "scaleq-qwen35.research.v1"
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
    # Independent CPU copies break shared aliases, which safetensors rejects.
    # A GPU-to-CPU transfer already makes a copy: avoid a second large clone.
    # Only one bounded chunk (or one larger tensor) is copied at a time.
    payload = {
        name: (tensor.detach().contiguous().clone() if tensor.device.type == "cpu"
               else tensor.detach().to("cpu").contiguous())
        for name, tensor in tensors.items()
    }
    path = directory / filename
    save_file(payload, str(path))
    files[filename] = {"bytes": path.stat().st_size, "sha256": _digest(path)}


def save_compressed(model, path, metadata=None):
    """Atomically save a standalone artifact; never overwrite a nonempty folder.

    Packed codes and fp16 group scales replace all state inside ternary modules.
    Every other parameter and registered buffer retains its original dtype.
    """
    get_text_layers(model)  # Fail closed on other architecture families.
    destination = Path(path).expanduser().absolute()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {destination}")
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
            packed = module.packed_state()
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
        payload_bytes = sum(entry["bytes"] for entry in manifest["files"].values())
        for _ in range(12):
            encoded = _json_bytes(manifest)
            if manifest["storage"]["manifest_bytes"] == len(encoded):
                break
            manifest["storage"]["manifest_bytes"] = len(encoded)
            manifest["storage"]["artifact_bytes"] = payload_bytes + len(encoded)
        encoded = _json_bytes(manifest)
        if len(encoded) != manifest["storage"]["manifest_bytes"]:
            raise RuntimeError("Could not finalize artifact size accounting")
        (staging / "manifest.json").write_bytes(encoded)
        if destination.exists():
            # rmdir only succeeds if the preexisting directory is still empty.
            destination.rmdir()
        os.replace(staging, destination)
    return manifest


def _artifact_file(root, name, description):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid artifact path: {name}")
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


def load_compressed(path, device="cpu"):
    """Load a self-contained, checksummed research artifact without network I/O."""
    root = Path(path).expanduser().absolute()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Unsupported compressed artifact format: {manifest.get('format')!r}")
    files = manifest["files"]
    checked = {name: _artifact_file(root, name, description) for name, description in files.items()}
    actual_bytes = sum(item["bytes"] for item in files.values()) + manifest_path.stat().st_size
    if actual_bytes != manifest["storage"]["artifact_bytes"] or manifest_path.stat().st_size != manifest["storage"]["manifest_bytes"]:
        raise ValueError("Artifact total byte accounting mismatch")
    config = config_from_dict(json.loads(checked["config.json"].read_text()))
    # Avoid filling large original weight tensors just to replace them. Native
    # constructor-defined rotary buffers still initialize correctly on CPU.
    with no_init_weights():
        model = model_from_config(config)

    consumed_files = {"config.json"}
    for name, info in manifest["packed_modules"].items():
        native = model.get_submodule(name)
        if not isinstance(native, nn.Linear):
            raise ValueError(f"Packed target is not a native linear projection: {name}")
        filename = info["file"]
        payload = load_file(str(checked[filename]), device="cpu")
        if set(payload) != set(info["tensors"]):
            raise ValueError(f"Packed tensor keys mismatch: {name}")
        for key, tensor in payload.items():
            _check_tensor(f"{name}.{key}", tensor, info["tensors"][key])
        state = {**info["state"], **payload}
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
