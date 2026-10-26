"""Self-contained mixed-format artifacts with a size contract known before search.

Candidate files carry their own complete format metadata. Manifest byte counts
are fixed-width decimal strings, so changing an assignment never changes the
common cost. Export verifies predicted bytes against every actual file, including
tokenizer/config and retained vision weights. Execution reconstructs float weights.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import GenerationConfig
from transformers.initialization import no_init_weights

from .adapter import config_from_dict, model_from_config
from .candidates import source_weight_sha256
from .export import (_all_tensors, _artifact_file, _check_tensor, _copy_checked_file,
                     _destination, _digest, _json_bytes, _publish, _tensor_description,
                     _torch_dtype, _under_module, _write_tensor_file)
from .mixed_formats import (MixedLinear, MixedEmbedding, candidate_nbytes,
                            encode_candidate, decode_candidate)

FORMAT = "openbonsai.mixed.qwen35.v1"


def _count(value):
    if type(value) is not int or not 0 <= value < 10**20:
        raise ValueError("Byte count must be a nonnegative integer below 10**20")
    return f"{value:020d}"


def _parse_count(value):
    if not isinstance(value, str) or len(value) != 20 or not value.isascii() or not value.isdigit():
        raise ValueError("Invalid fixed-width byte count")
    return int(value)


def _module_kind(module):
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Embedding):
        # Native Qwen embeddings have ordinary gather semantics.
        if module.max_norm is not None or module.scale_grad_by_freq or module.sparse:
            raise ValueError("Only plain embedding lookup is supported")
        return "embedding"
    raise ValueError("Candidates require original native Linear/Embedding modules")


def _bias_hash(bias):
    return source_weight_sha256(bias.detach().to(torch.bfloat16).reshape(1, -1))


def _check_source(candidate, info):
    if candidate.metadata['provenance'].get('source_weight_sha256') != info['source_weight_sha256']:
        raise ValueError('Candidate original BF16 provenance mismatch')
    bias = candidate.tensors.get('bias')
    allowed = info['allowed_bias_compute_sha256']
    if (bias is None) != (not allowed) or (bias is not None and _bias_hash(bias) not in allowed):
        raise ValueError('Candidate bias differs from original or documented FP16 export rounding')


@dataclass
class MixedExportPlan:
    """Common files in an owned temporary directory; call close when finished."""
    directory: Path
    manifest: dict
    fixed_bytes: int
    _temporary: object

    def close(self):
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepare_mixed_export(model, module_names, *, metadata, tokenizer_dir=None, work_dir=None):
    """Freeze all common deployment bytes before allocation (no candidate fitting).

    ``metadata`` must identify the original checkpoint/revision or local content
    hash, including any intended BF16 fine-tuning. Research results belong outside
    the artifact: adding them later would change the predeclared size budget.
    """
    names = sorted(module_names)
    if not names or len(names) != len(set(names)) or any(not n or not all(n.split('.')) for n in names):
        raise ValueError("Expected unique nonempty candidate module names")
    if any(a != b and b.startswith(a + '.') for a in names for b in names):
        raise ValueError("Nested candidate modules are unsupported")
    if not isinstance(metadata, dict) or not metadata.get("source_checkpoint"):
        raise ValueError("Original source_checkpoint provenance is required")
    if getattr(model.config, "quantization_config", None) is not None:
        raise ValueError("Export plan requires an original uncompressed model")
    modules = {}
    seen_weights = set()
    for index, name in enumerate(names):
        native = model.get_submodule(name)
        kind = _module_kind(native)
        if native.weight.dtype != torch.bfloat16:
            raise ValueError("Mixed candidates must start from original BF16 weights")
        if id(native.weight) in seen_weights:
            raise ValueError("Tied candidate weights need an explicit shared allocation policy")
        seen_weights.add(id(native.weight))
        bias = getattr(native, 'bias', None)
        allowed_bias = [] if bias is None else sorted({_bias_hash(bias), _bias_hash(bias.to(torch.float16))})
        modules[name] = {"file": f"candidate-{index:05d}.obc", "kind": kind,
                         "shape": list(native.weight.shape), "compute_dtype": "bfloat16",
                         "source_weight_sha256": source_weight_sha256(native.weight),
                         "allowed_bias_compute_sha256": allowed_bias,
                         "bytes": _count(0), "sha256": "0" * 64}
    temporary = tempfile.TemporaryDirectory(prefix="openbonsai-mixed-", dir=work_dir)
    root = Path(temporary.name)
    manifest = {"format": FORMAT, "metadata": copy.deepcopy(metadata), "modules": modules,
                "files": {}, "retained_tensors": {}, "tokenizer_files": [],
                "runtime": "reference floating-point reconstruction; no low-bit kernel",
                "storage": {key: _count(0) for key in (
                    "fixed_bytes", "candidate_bytes", "artifact_bytes", "manifest_bytes",
                    "vision_file_bytes", "retained_tensor_bytes", "tokenizer_bytes")}}
    try:
        config_path = root / "config.json"
        config_path.write_bytes(_json_bytes(model.config.to_dict()))
        manifest['files']['config.json'] = {'bytes': config_path.stat().st_size, 'sha256': _digest(config_path)}
        generation_path = root / 'generation_config.json'
        generation_path.write_bytes(_json_bytes(model.generation_config.to_dict()))
        manifest['files']['generation_config.json'] = {'bytes': generation_path.stat().st_size,
                                                       'sha256': _digest(generation_path)}
        vision_bytes = retained_bytes = 0
        for index, (name, tensor, kind, persistent) in enumerate(sorted(_all_tensors(model))):
            if _under_module(name, names):
                continue
            filename = f"retained-{index:05d}.safetensors"
            _write_tensor_file(root, filename, {name: tensor}, manifest['files'])
            desc = _tensor_description(tensor)
            manifest['retained_tensors'][name] = {**desc, 'file': filename, 'kind': kind,
                                                   'persistent': persistent}
            retained_bytes += desc['tensor_bytes']
            if name.startswith('model.visual.'):
                vision_bytes += manifest['files'][filename]['bytes']
        tokenizer_bytes = 0
        if tokenizer_dir is not None:
            source = Path(tokenizer_dir)
            if source.is_symlink() or not source.is_dir():
                raise ValueError("Tokenizer directory must be a real directory")
            for path in sorted(source.rglob('*')):
                if path.is_symlink():
                    raise ValueError("Tokenizer symlinks are unsupported")
                if not path.is_file():
                    continue
                name = 'tokenizer/' + path.relative_to(source).as_posix()
                description = {'bytes': path.stat().st_size, 'sha256': _digest(path)}
                _copy_checked_file(path, root / name, description)
                manifest['files'][name] = description
                manifest['tokenizer_files'].append(name)
                tokenizer_bytes += description['bytes']
        storage = manifest['storage']
        storage.update(vision_file_bytes=_count(vision_bytes), retained_tensor_bytes=_count(retained_bytes),
                       tokenizer_bytes=_count(tokenizer_bytes))
        size = len(_json_bytes(manifest))
        fixed = size + sum(f['bytes'] for f in manifest['files'].values())
        storage.update(fixed_bytes=_count(fixed), manifest_bytes=_count(size), artifact_bytes=_count(fixed))
        assert len(_json_bytes(manifest)) == size
        return MixedExportPlan(root, manifest, fixed, temporary)
    except BaseException:
        temporary.cleanup()
        raise


def save_mixed(plan, assignment, path, *, budget_bytes):
    """Export one selected candidate per tensor and enforce the complete budget."""
    destination = _destination(path)
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget_bytes must be a nonnegative integer")
    if set(assignment) != set(plan.manifest['modules']):
        raise ValueError("Assignment must cover exactly the planned modules")
    predicted = plan.fixed_bytes + sum(candidate_nbytes(c) for c in assignment.values())
    if predicted > budget_bytes:
        raise ValueError(f"Selected artifact exceeds budget: {predicted} > {budget_bytes}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}-', dir=destination.parent) as directory:
        staging = Path(directory) / 'artifact'
        staging.mkdir()
        manifest = copy.deepcopy(plan.manifest)
        for name, desc in manifest['files'].items():
            _copy_checked_file(_artifact_file(plan.directory, name, desc), staging / name, desc)
        for name, candidate in assignment.items():
            info = manifest['modules'][name]
            if list(candidate.shape) != info['shape'] or candidate.module_kind != info['kind']:
                raise ValueError(f"Candidate shape/kind mismatch: {name}")
            _check_source(candidate, info)
            payload = encode_candidate(candidate)
            output = staging / info['file']
            output.write_bytes(payload)
            info.update(bytes=_count(len(payload)), sha256=_digest(output))
        manifest['storage'].update(candidate_bytes=_count(predicted - plan.fixed_bytes), artifact_bytes=_count(predicted))
        encoded = _json_bytes(manifest)
        if len(encoded) != _parse_count(manifest['storage']['manifest_bytes']):
            raise ValueError("Manifest size changed after allocation")
        (staging / 'manifest.json').write_bytes(encoded)
        actual = sum(p.stat().st_size for p in staging.rglob('*') if p.is_file())
        if actual != predicted or actual > budget_bytes:
            raise ValueError("Actual artifact bytes disagree with allocation")
        _publish(staging, destination)
    return manifest


def load_mixed(path, device='cpu'):
    """Offline load of complete native Qwen3.5, validating all retained state."""
    root = Path(path).absolute()
    manifest_path = root / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError("Manifest cannot be a symlink")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError("Unsupported mixed artifact format")
    _json_bytes(manifest)
    modules, files, retained = (manifest[k] for k in ('modules', 'files', 'retained_tensors'))
    if not modules or not {'config.json', 'generation_config.json'}.issubset(files):
        raise ValueError("Incomplete mixed artifact")
    checked = {name: _artifact_file(root, name, desc) for name, desc in files.items()}
    used = {'manifest.json', *files}
    candidates = {}
    for name, info in modules.items():
        filename = info['file']
        if filename in used or not name or not all(name.split('.')):
            raise ValueError("Duplicate candidate file or invalid module path")
        used.add(filename)
        file = _artifact_file(root, filename, {'bytes': _parse_count(info['bytes']), 'sha256': info['sha256']})
        candidate = decode_candidate(file.read_bytes())
        if list(candidate.shape) != info['shape'] or candidate.module_kind != info['kind']:
            raise ValueError("Candidate shape/kind differs from manifest")
        _check_source(candidate, info)
        candidates[name] = candidate
    # Refuse unaccounted deployment files and all symlinks, not only listed paths.
    entries = list(root.rglob('*'))
    if any(p.is_symlink() for p in entries) or {p.relative_to(root).as_posix() for p in entries if p.is_file()} != used:
        raise ValueError("Artifact contains unaccounted files or symlinks")
    storage = {k: _parse_count(v) for k, v in manifest['storage'].items()}
    fixed = manifest_path.stat().st_size + sum(d['bytes'] for d in files.values())
    candidate_bytes = sum(_parse_count(x['bytes']) for x in modules.values())
    if (storage['fixed_bytes'] != fixed or storage['candidate_bytes'] != candidate_bytes
            or storage['artifact_bytes'] != fixed + candidate_bytes
            or storage['manifest_bytes'] != manifest_path.stat().st_size):
        raise ValueError("Mixed artifact byte accounting mismatch")
    if set(manifest['tokenizer_files']) != {x for x in files if x.startswith('tokenizer/')}:
        raise ValueError("Tokenizer file accounting mismatch")
    config = config_from_dict(json.loads(checked['config.json'].read_text()))
    with no_init_weights():
        model = model_from_config(config)
    model.generation_config = GenerationConfig.from_dict(json.loads(checked['generation_config.json'].read_text()))
    for name, candidate in candidates.items():
        native = model.get_submodule(name)
        if list(native.weight.shape) != list(candidate.shape) or _module_kind(native) != candidate.module_kind:
            raise ValueError(f"Native candidate contract mismatch: {name}")
        if (getattr(native, 'bias', None) is None) != (candidate.tensors.get('bias') is None):
            raise ValueError(f"Native candidate bias presence mismatch: {name}")
        cls = MixedLinear if candidate.module_kind == 'linear' else MixedEmbedding
        model.set_submodule(name, cls(candidate, dtype=_torch_dtype(modules[name]['compute_dtype'])))
    expected = {n: (t, k, p) for n, t, k, p in _all_tensors(model) if not _under_module(n, modules)}
    if set(expected) != set(retained):
        raise ValueError("Mixed retained state is incomplete")
    consumed = {'config.json', 'generation_config.json', *manifest['tokenizer_files']}
    retained_bytes = vision_bytes = 0
    for name, desc in retained.items():
        filename = desc['file']
        if filename in consumed:
            raise ValueError("Duplicate retained tensor file")
        consumed.add(filename)
        payload = load_file(str(checked[filename]), device='cpu')
        if set(payload) != {name}:
            raise ValueError("Retained tensor file keys mismatch")
        tensor = payload[name]
        _check_tensor(name, tensor, desc)
        original, kind, persistent = expected[name]
        if tensor.shape != original.shape or kind != desc['kind'] or persistent != desc['persistent']:
            raise ValueError("Retained native tensor contract mismatch")
        parent, _, key = name.rpartition('.')
        target = model.get_submodule(parent) if parent else model
        if kind == 'parameter':
            target._parameters[key] = nn.Parameter(tensor, requires_grad=False)
        else:
            target._buffers[key] = tensor
        retained_bytes += desc['tensor_bytes']
        if name.startswith('model.visual.'):
            vision_bytes += files[filename]['bytes']
    if consumed != set(files):
        raise ValueError("Unreferenced common file")
    if (storage['retained_tensor_bytes'] != retained_bytes or storage['vision_file_bytes'] != vision_bytes
            or storage['tokenizer_bytes'] != sum(files[f]['bytes'] for f in manifest['tokenizer_files'])):
        raise ValueError("Retained/vision/tokenizer byte accounting mismatch")
    model.to(device)
    model.requires_grad_(False)
    return model.eval()
