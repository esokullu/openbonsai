"""Separate original-hidden-state calibration for a bounded ternary LM head.

The teacher is the unchanged BF16 checkpoint, including its decoder. Training
fits a full-vocabulary KL at temperature one. Native head logits retain their
computation dtype; probability normalization/reduction explicitly uses FP64.
This is a reference endpoint experiment, not a ScaleQ/Bonsai reproduction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from safetensors.torch import load_file, save_file
import transformers

from .adapter import _text_module
from .candidates import source_weight_sha256
from .export import _artifact_file, _destination, _digest, _json_bytes, _publish

CACHE_FORMAT = "openbonsai.original-head-inputs.v1"
_ROLES = {"calibration", "search_validation", "evaluation"}


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _is_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _provenance(value):
    if not isinstance(value, Mapping):
        raise ValueError("source_provenance must identify the original decoder checkpoint")
    value = json.loads(_json_bytes(dict(value)))
    if not isinstance(value.get('source_checkpoint'), str) or not value['source_checkpoint']:
        raise ValueError("source_checkpoint is required")
    revision = value.get('source_revision', value.get('resolved_revision'))
    if 'source_revision' in value and 'resolved_revision' in value and value['source_revision'] != value['resolved_revision']:
        raise ValueError("Conflicting source revisions")
    content = value.get('source_content_sha256')
    if content is not None and not _is_digest(content):
        raise ValueError("source_content_sha256 must be a SHA256 of the original checkpoint")
    if revision is not None:
        if not isinstance(revision, str) or not revision or revision in {'main', 'master', 'HEAD', 'latest'}:
            raise ValueError("source_revision must identify a resolved immutable checkpoint")
        value['source_revision'] = revision
        value.pop('resolved_revision', None)
    if revision is None and content is None:
        raise ValueError("Original decoder source_revision or source_content_sha256 is required")
    return value


def _tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(_json_bytes({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bias_hash(bias):
    return None if bias is None else _tensor_hash(bias)


def _trainable_state_sha256(head, *, chunk_elements=1048576):
    """Hash sorted named trainables with bounded CPU copies, excluding masters.

    Each tensor contributes a canonical JSON name/dtype/shape header followed
    by contiguous row-major bytes. Copies are at most chunk_elements or one
    parameter row, whichever is larger; the head's matrix row width is small.
    """
    _positive_int(chunk_elements, 'chunk_elements')
    digest = hashlib.sha256()
    for name, parameter in sorted(head.named_parameters()):
        digest.update(_json_bytes({'name': name, 'dtype': str(parameter.dtype), 'shape': list(parameter.shape)}))
        values = parameter.detach() if parameter.ndim else parameter.detach().reshape(1)
        row_width = math.prod(values.shape[1:])
        rows = max(1, chunk_elements // max(1, row_width))
        for start in range(0, values.shape[0], rows):
            part = values[start:start + rows].cpu().contiguous()
            digest.update(part.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class HeadInputCache:
    directory: Path
    manifest: dict

    def __len__(self):
        """Number of input records, independent of the disk shard size."""
        return len(self.manifest['records'])

    @property
    def positions(self):
        return self.manifest['positions']

    def iter_record(self, index, *, token_microbatch):
        """Yield CPU valid-position rows without joining whole records/shards."""
        _positive_int(token_microbatch, 'token_microbatch')
        record = self.manifest['records'][index]
        for name in record['shards']:
            desc = self.manifest['files'][name]
            path = _artifact_file(self.directory, name, desc)
            tensors = load_file(str(path), device='cpu')
            _validate_shard(tensors, desc['positions'], self.manifest['width'], self.manifest['vocabulary'])
            for start in range(0, desc['positions'], token_microbatch):
                yield {key: value[start:start + token_microbatch] for key, value in tensors.items()}


def _validate_shard(tensors, positions, width, vocabulary):
    if set(tensors) != {'hidden', 'labels', 'positions'}:
        raise ValueError("Invalid head cache shard fields")
    hidden, labels, coordinates = (tensors[k] for k in ('hidden', 'labels', 'positions'))
    if hidden.shape != (positions, width) or hidden.dtype != torch.bfloat16 or not bool(torch.isfinite(hidden).all()):
        raise ValueError("Head cache must contain finite original BF16 hidden states")
    if labels.shape != (positions,) or labels.dtype != torch.int64 or not bool(((labels >= 0) & (labels < vocabulary)).all()):
        raise ValueError("Invalid cached next-token labels")
    if coordinates.shape != (positions, 2) or coordinates.dtype != torch.int64 or bool((coordinates < 0).any()):
        raise ValueError("Invalid original batch/token coordinates")


def load_head_input_cache(directory) -> HeadInputCache:
    """Validate all cache files/checksums, including roles and content identity."""
    root = Path(directory).absolute()
    path = root / 'manifest.json'
    if root.is_symlink() or path.is_symlink():
        raise ValueError("Head cache symlinks are unsupported")
    manifest = json.loads(path.read_text())
    _json_bytes(manifest)
    if manifest.get('format') != CACHE_FORMAT or manifest.get('role') not in _ROLES:
        raise ValueError("Unsupported head cache format/role")
    if manifest.get('position_policy') != 'valid_next_token' or manifest.get('dtype') != 'bfloat16':
        raise ValueError("Unsupported cached position policy")
    _provenance(manifest.get('source_provenance'))
    for field in ('positions', 'width', 'vocabulary'):
        _positive_int(manifest.get(field), field)
    for field in ('source_weight_sha256', 'config_sha256'):
        if not _is_digest(manifest.get(field)):
            raise ValueError(f"Invalid {field}")
    if manifest.get('source_bias_sha256') is not None and not _is_digest(manifest['source_bias_sha256']):
        raise ValueError("Invalid source bias hash")
    records, files = manifest.get('records'), manifest.get('files')
    if not isinstance(records, list) or not records or not isinstance(files, dict) or not files:
        raise ValueError("Incomplete head cache")
    seen, count = set(), 0
    for record in records:
        _positive_int(record.get('positions'), 'record positions')
        if not isinstance(record.get('record_id'), str) or not _is_digest(record.get('input_sha256')):
            raise ValueError("Invalid cache input provenance")
        sequences = record.get('sequence_sha256')
        if not isinstance(sequences, list) or not sequences or not all(_is_digest(x) for x in sequences):
            raise ValueError("Invalid cache sequence provenance")
        subtotal = 0
        for name in record.get('shards', []):
            if name in seen or name not in files:
                raise ValueError("Duplicate or missing head cache shard")
            seen.add(name)
            desc = files[name]
            _positive_int(desc.get('positions'), 'shard positions')
            path = _artifact_file(root, name, desc)
            tensors = load_file(str(path), device='cpu')
            _validate_shard(tensors, desc['positions'], manifest['width'], manifest['vocabulary'])
            del tensors
            subtotal += desc['positions']
        if subtotal != record['positions']:
            raise ValueError("Head cache position accounting mismatch")
        count += subtotal
    if seen != set(files) or count != manifest['positions']:
        raise ValueError("Head cache file/position accounting mismatch")
    entries = list(root.rglob('*'))
    if any(p.is_symlink() for p in entries) or {p.relative_to(root).as_posix() for p in entries if p.is_file()} != {'manifest.json', *files}:
        raise ValueError("Unaccounted head cache files or symlinks")
    return HeadInputCache(root, manifest)


class _CapturedHead(Exception):
    pass


def _original_head(model):
    _, text_model = _text_module(model)
    from .quantizer import TernaryLinear, PackedTernaryLinear
    from .mixed_formats import MixedLinear, MixedEmbedding
    wrappers = (TernaryLinear, PackedTernaryLinear, MixedLinear, MixedEmbedding)
    if (getattr(model.config, 'quantization_config', None) is not None
            or getattr(text_model.config, 'quantization_config', None) is not None
            or getattr(model, 'peft_config', None)
            or any(isinstance(m, wrappers) or type(m).__name__ == 'BoundedTernaryHead'
                   or hasattr(m, 'parametrizations') for m in model.modules())):
        raise ValueError("Head input cache requires the original model, not a quantized/adapted student")
    # Native BF16 checkpoints may retain recurrent/norm parameters in FP32.
    # Preserve their original precision: forcing them to BF16 changes the
    # teacher. The head and captured activations are checked separately.
    if any(p.is_floating_point() and p.dtype not in (torch.bfloat16, torch.float32) for p in model.parameters()):
        raise ValueError("Head input cache supports original BF16/FP32 model parameters")
    head, embedding = model.lm_head, model.get_input_embeddings()
    if type(head) is not nn.Linear or head.weight.dtype != torch.bfloat16:
        raise ValueError("lm_head must be an original BF16 nn.Linear")
    if head.bias is not None and head.bias.dtype != torch.bfloat16:
        raise ValueError("Original head bias must be BF16")
    if (getattr(text_model.config, 'tie_word_embeddings', False)
            or head.weight is embedding.weight
            or (head.weight.device == embedding.weight.device
                and head.weight.untyped_storage().data_ptr() == embedding.weight.untyped_storage().data_ptr())):
        raise ValueError("Tied embedding/head calibration is unsupported")
    return head


def _record_inputs(record, vocabulary):
    if not isinstance(record, Mapping):
        raise ValueError("Cache records must be mappings of original token inputs")
    allowed = {'input_ids', 'attention_mask', 'labels', 'position_ids', 'record_id'}
    if set(record) - allowed:
        raise ValueError(f"Unsupported head cache record fields: {sorted(set(record) - allowed)}")
    ids = record.get('input_ids')
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or ids.shape[1] < 2:
        raise ValueError("input_ids must be integer [batch, sequence>=2]")
    ids = ids.detach().cpu().to(torch.int64)
    if not bool(((ids >= 0) & (ids < vocabulary)).all()):
        raise ValueError("input_ids are outside the source vocabulary")
    mask = record.get('attention_mask', torch.ones_like(ids))
    if not isinstance(mask, Tensor) or mask.shape != ids.shape or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("attention_mask must match input_ids and contain zero/one")
    mask = mask.detach().cpu().to(torch.bool)
    labels = record.get('labels', ids)
    if not isinstance(labels, Tensor) or labels.shape != ids.shape or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be integer original next-token labels with optional -100 masking")
    labels = labels.detach().cpu().to(torch.int64)
    if not bool(((labels == ids) | (labels == -100)).all()):
        raise ValueError("Nonignored labels must equal the original input token IDs before shifting")
    inputs = {'input_ids': ids, 'attention_mask': mask.to(torch.int64)}
    if 'position_ids' in record:
        position_ids = record['position_ids']
        if not isinstance(position_ids, Tensor) or position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must be an integer tensor")
        inputs['position_ids'] = position_ids.detach().cpu().to(torch.int64)
    valid = mask[:, :-1] & mask[:, 1:] & (labels[:, 1:] != -100)
    coordinates = valid.nonzero(as_tuple=False)
    token_hashes = [_tensor_hash(row[keep]) for row, keep in zip(ids, mask) if bool(keep.any())]
    input_hash = hashlib.sha256(_json_bytes({key: _tensor_hash(value) for key, value in {**inputs, 'labels': labels}.items()})).hexdigest()
    record_id = record.get('record_id', input_hash)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a nonempty string")
    return inputs, labels, coordinates, {'record_id': record_id, 'input_sha256': input_hash,
                                        'sequence_sha256': token_hashes}


@torch.no_grad()
def cache_original_head_inputs(original_model, records, directory, source_provenance, *,
                               role='calibration', max_positions=None, shard_positions=512) -> HeadInputCache:
    """Capture final-normalized original BF16 hidden rows before vocabulary logits.

    Records are text token inputs, optionally labels with -100 masking. Each
    selected t has valid t/t+1 and predicts input_ids[t+1]; EOS remains valid.
    Caller supplies checkpoint identity and owns dataset/benchmark separation.
    Explicit roles and exact content hashes catch reuse, not semantic leakage.
    """
    provenance = _provenance(source_provenance)
    if role not in _ROLES:
        raise ValueError(f"role must be one of {sorted(_ROLES)}")
    _positive_int(shard_positions, 'shard_positions')
    if max_positions is not None:
        _positive_int(max_positions, 'max_positions')
    destination = _destination(directory)
    head = _original_head(original_model)
    fingerprint = source_weight_sha256(head.weight)
    config_bytes = _json_bytes(original_model.config.to_dict())
    parameter_dtypes = {}
    for parameter in original_model.parameters():
        counts = parameter_dtypes.setdefault(str(parameter.dtype).removeprefix('torch.'),
                                             {'tensors': 0, 'numel': 0, 'bytes': 0})
        counts['tensors'] += 1
        counts['numel'] += parameter.numel()
        counts['bytes'] += parameter.numel() * parameter.element_size()
    manifest = {'format': CACHE_FORMAT, 'role': role, 'position_policy': 'valid_next_token',
                'source_provenance': provenance, 'source_weight_sha256': fingerprint,
                'source_bias_sha256': _bias_hash(head.bias), 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                'width': head.in_features, 'vocabulary': head.out_features, 'positions': 0,
                'dtype': 'bfloat16', 'records': [], 'files': {}, 'records_seen': 0,
                'source_parameter_dtypes': parameter_dtypes,
                'max_positions': max_positions, 'shard_positions': shard_positions,
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'capture': 'original lm_head input after native final norm; logits_to_keep=0; use_cache=False',
                'separation': 'caller-declared roles; exact sequence/record reuse checks cannot detect semantic benchmark leakage'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    modes = [(module, module.training) for module in original_model.modules()]
    hook = None
    captured = None

    def capture(_module, args):
        nonlocal captured
        if len(args) != 1 or not isinstance(args[0], Tensor):
            raise ValueError("Native lm_head did not receive one hidden-state tensor")
        captured = args[0].detach()
        raise _CapturedHead

    try:
        original_model.eval()
        hook = head.register_forward_pre_hook(capture)
        with tempfile.TemporaryDirectory(prefix=f'.{destination.name}-', dir=destination.parent) as directory:
            staging = Path(directory) / 'cache'
            staging.mkdir()
            for record in records:
                inputs, labels, coordinates, info = _record_inputs(record, head.out_features)
                manifest['records_seen'] += 1
                if not len(coordinates):
                    continue
                remaining = len(coordinates) if max_positions is None else max_positions - manifest['positions']
                coordinates = coordinates[:remaining]
                captured = None
                try:
                    original_model(**{key: value.to(head.weight.device) for key, value in inputs.items()},
                                   use_cache=False, logits_to_keep=0)
                except _CapturedHead:
                    pass
                if captured is None:
                    raise RuntimeError("Original lm_head capture hook did not fire")
                if (captured.shape != (*inputs['input_ids'].shape, head.in_features)
                        or captured.dtype != torch.bfloat16):
                    raise ValueError("Captured head inputs must be original BF16 [batch,sequence,width]")
                info.update(positions=len(coordinates), shards=[])
                for offset in range(0, len(coordinates), shard_positions):
                    coords = coordinates[offset:offset + shard_positions]
                    batch, position = coords.unbind(-1)
                    hidden = captured[batch.to(captured.device), position.to(captured.device)].cpu().contiguous()
                    payload = {'hidden': hidden, 'labels': labels[batch, position + 1].contiguous(),
                               'positions': coords.contiguous()}
                    _validate_shard(payload, len(coords), head.in_features, head.out_features)
                    name = f"shard-{len(manifest['files']):06d}.safetensors"
                    path = staging / name
                    save_file(payload, str(path))
                    manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': _digest(path), 'positions': len(coords)}
                    info['shards'].append(name)
                captured = None
                manifest['records'].append(info)
                manifest['positions'] += info['positions']
                if max_positions is not None and manifest['positions'] >= max_positions:
                    break
            if not manifest['positions']:
                raise ValueError("No valid next-token positions; all-padding/ignored cache is refused")
            (staging / 'manifest.json').write_bytes(_json_bytes(manifest))
            _publish(staging, destination)
    finally:
        captured = None
        if hook is not None:
            hook.remove()
        for module, training in modes:
            module.training = training
    return load_head_input_cache(destination)


def _cache(value):
    return load_head_input_cache(value.directory if isinstance(value, HeadInputCache) else value)


def _check_source(head, cache, source_provenance):
    if _provenance(source_provenance) != cache.manifest['source_provenance']:
        raise ValueError("Head cache original decoder provenance mismatch")
    if (head.weight.dtype != torch.bfloat16 or list(head.weight.shape) != [cache.manifest['vocabulary'], cache.manifest['width']]
            or source_weight_sha256(head.weight) != cache.manifest['source_weight_sha256']
            or _bias_hash(head.bias) != cache.manifest['source_bias_sha256']):
        raise ValueError("Head cache original BF16 weight/bias source mismatch")


def _separation(calibration, others):
    prior = [calibration]
    for cache, role in others:
        if cache.manifest['role'] != role:
            raise ValueError(f"Expected {role} cache")
        if cache.manifest['source_provenance'] != calibration.manifest['source_provenance'] or cache.manifest['config_sha256'] != calibration.manifest['config_sha256']:
            raise ValueError("Held-out cache original decoder/config provenance mismatch")
        ids = {r['record_id'] for r in cache.manifest['records']}
        hashes = {h for r in cache.manifest['records'] for h in r['sequence_sha256']}
        for other in prior:
            if (ids & {r['record_id'] for r in other.manifest['records']}
                    or hashes & {h for r in other.manifest['records'] for h in r['sequence_sha256']}):
                raise ValueError("Calibration/search/evaluation caches contain overlapping records or token sequences")
        prior.append(cache)


def chunked_full_vocab_kl(student: Callable, teacher: Callable, *, vocabulary: int,
                          row_chunk: int, mask: Tensor, denominator: int | None = None,
                          backward=True) -> Tensor:
    """Return actual KL(P_teacher || Q_student); optionally accumulate gradients.

    Callbacks independently reconstruct [start:end] logits. This uses global
    FP64 normalization and exact dL/dlogits=(Q-P)/denominator. Parameter values,
    progress and snapshots must stay fixed across both passes. Only first-order
    derivatives are supported. Call optimizer.step only after all microbatches.
    """
    _positive_int(vocabulary, 'vocabulary')
    _positive_int(row_chunk, 'row_chunk')
    if not isinstance(mask, Tensor) or mask.ndim < 1 or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask must contain zero/one token-position values")
    count = int(mask.sum())
    if count <= 0:
        raise ValueError("No valid KL positions")
    denominator = count if denominator is None else denominator
    _positive_int(denominator, 'denominator')
    if denominator < count:
        raise ValueError("Logical-batch denominator cannot be smaller than this microbatch")

    def check(logits, start, end):
        if (not isinstance(logits, Tensor) or not logits.is_floating_point()
                or logits.shape != (*mask.shape, end - start) or logits.device != mask.device
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("Expected finite chunk logits matching mask/device/vocabulary range")

    teacher_lse = student_lse = None
    with torch.no_grad():
        for start in range(0, vocabulary, row_chunk):
            end = min(start + row_chunk, vocabulary)
            t, s = teacher(start, end), student(start, end)
            check(t, start, end); check(s, start, end)
            tl, sl = t.double().logsumexp(-1), s.double().logsumexp(-1)
            teacher_lse = tl if teacher_lse is None else torch.logaddexp(teacher_lse, tl)
            student_lse = sl if student_lse is None else torch.logaddexp(student_lse, sl)
    result = student_lse.new_zeros(())
    for start in range(0, vocabulary, row_chunk):
        end = min(start + row_chunk, vocabulary)
        with torch.set_grad_enabled(backward):
            s = student(start, end)
        check(s, start, end)
        with torch.no_grad():
            t = teacher(start, end)
            check(t, start, end)
            log_p = t.double() - teacher_lse[..., None]
            log_q = s.detach().double() - student_lse[..., None]
            p, q = log_p.exp(), log_q.exp()
            result += ((p * (log_p - log_q)).sum(-1) * mask).sum() / denominator
            derivative = (q - p) * mask[..., None] / denominator
        if backward:
            if not s.requires_grad:
                raise ValueError("Student chunk has no trainable graph")
            s.backward(derivative)
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("Nonfinite full-vocabulary KL")
    return result.detach()


@dataclass
class HeadTrainingConfig:
    epochs: int = 2
    batch_size: int = 1
    token_microbatch: int = 64
    vocabulary_chunk: int = 2048
    learning_rate: float = .001
    lora_learning_rate: float = .001
    weight_decay: float = 0.
    grad_clip: float = 1.
    seed: int = 2

    def validate(self):
        for name in ('epochs', 'batch_size', 'token_microbatch', 'vocabulary_chunk'):
            _positive_int(getattr(self, name), name)
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ('learning_rate', 'lora_learning_rate', 'weight_decay', 'grad_clip'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (name != 'weight_decay' and value == 0)):
                raise ValueError("Learning rates/clipping must be positive and weight decay nonnegative")


@dataclass
class HeadFitResult:
    report: dict
    trainable_state: dict[str, Tensor]


def _head_device_and_chunk(head, vocabulary_chunk):
    parameters = tuple(head.trainable_parameters())
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Bounded head trainable parameters must be FP32")
    device = parameters[0].device
    if any(p.device != device for p in parameters):
        raise ValueError("Head trainables must share one compute device")
    chunk = min(vocabulary_chunk, head.row_chunk_size)
    _positive_int(chunk, 'effective vocabulary_chunk')
    return device, chunk


def fit_head_from_cache(head, cache, config: HeadTrainingConfig, *, source_provenance,
                        validation_cache=None, evaluation_cache=None, progress_callback=None) -> HeadFitResult:
    """Fit a bounded head only; never forwards a decoder or fits held-out caches.

    batch_size counts original input records, independent of disk sharding.
    Validation/evaluation caches, when supplied, are checked for provenance and
    exact sequence/record overlap but never optimized or automatically selected.
    Caller remains responsible for semantic dataset/benchmark separation.
    """
    config.validate()
    if not getattr(head, 'quant_enabled', True) or getattr(head, 'rotation_only', False):
        raise ValueError("Head calibration requires quantization enabled and rotation_only=False")
    cache = _cache(cache)
    if cache.manifest['role'] != 'calibration':
        raise ValueError("fit_head_from_cache requires a calibration-role cache")
    _check_source(head, cache, source_provenance)
    held_out = []
    for value, role in ((validation_cache, 'search_validation'), (evaluation_cache, 'evaluation')):
        if value is not None:
            other = _cache(value)
            _check_source(head, other, source_provenance)
            held_out.append((other, role))
    _separation(cache, held_out)
    total_steps = config.epochs * math.ceil(len(cache) / config.batch_size)
    if getattr(head, 'hard_phase', 'frozen_jacobian') == 'frozen_jacobian' and 1 / total_steps > head.soft_fraction:
        raise ValueError("Head calibration needs at least one soft update before the hard phase")
    device, chunk = _head_device_and_chunk(head, config.vocabulary_chunk)
    parameters = tuple(head.trainable_parameters())
    groups = [{'params': list(head.factor_parameters()), 'lr': config.learning_rate, 'initial_lr': config.learning_rate}]
    lora = list(head.lora_parameters())
    if lora:
        groups.append({'params': lora, 'lr': config.lora_learning_rate, 'initial_lr': config.lora_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay, foreach=False)
    signs = getattr(head, 'rotation_signs', None)
    head_configuration = {name: getattr(head, name) for name in (
        'group_size', 'factor_mapping', 'lora_rank', 'hard_phase', 'soft_fraction',
        'max_sharpness', 'row_chunk_size')}
    head_configuration['rotation'] = (None if signs is None else {
        'convention': 'row-DH', 'normalization': 'orthonormal', 'axis': -1,
        'block_size': head.block_size, 'signs_sha256': _tensor_hash(signs),
        'sign_hash_encoding': 'canonical dtype/shape JSON header then contiguous raw tensor bytes'})
    report = {'stage': 'head-only-original-hidden', 'config': asdict(config), 'steps': [],
              'positions': cache.positions, 'records': len(cache), 'role': 'calibration',
              'source_provenance': cache.manifest['source_provenance'],
              'source_weight_sha256': cache.manifest['source_weight_sha256'],
              'head_configuration': head_configuration,
              'initial_trainable_state_sha256': _trainable_state_sha256(head),
              'seed_scope': 'config.seed controls cached record order only; caller initializes LoRA and rotation signs',
              'cache_manifest_sha256': _digest(cache.directory / 'manifest.json'),
              'objective': {'name': 'KL(original_teacher || student)', 'temperature': 1,
                            'reduction': 'mean over valid next-token positions in each logical batch',
                            'logits': 'native original BF16 computation', 'normalization': 'global FP64 across full vocabulary',
                            'reported_loss': 'actual KL; not the external-gradient surrogate'},
              'effective_vocabulary_chunk': chunk, 'held_out_roles_checked': [role for _, role in held_out],
              'separation': 'held-out inputs are never fitted; caller owns semantic benchmark separation',
              'quality_claim': False, 'torch': torch.__version__}
    flags = [(p, p.requires_grad) for p in parameters]
    training = head.training
    step = 0
    rng = random.Random(config.seed)
    try:
        head.eval()
        for p in parameters:
            p.requires_grad_(True)
            p.grad = None
        head.clear_soft_snapshot()
        for epoch in range(config.epochs):
            order = list(range(len(cache)))
            rng.shuffle(order)
            for offset in range(0, len(order), config.batch_size):
                indices = order[offset:offset + config.batch_size]
                denominator = sum(cache.manifest['records'][i]['positions'] for i in indices)
                optimizer.zero_grad(set_to_none=True)
                head.begin_step((step + 1) / total_steps)
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * (1 - step / total_steps)
                actual_kl = 0.
                for index in indices:
                    for record in cache.iter_record(index, token_microbatch=config.token_microbatch):
                        hidden = record['hidden'].to(device)
                        mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=device)
                        value = chunked_full_vocab_kl(
                            lambda start, end: head.forward_rows(hidden, start, end),
                            lambda start, end: head.teacher_rows(hidden, start, end),
                            vocabulary=head.out_features, row_chunk=chunk, mask=mask, denominator=denominator)
                        actual_kl += float(value)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                event = {'step': step, 'epoch': epoch + 1, 'total_steps': total_steps,
                         'positions': denominator, 'progress': step / total_steps,
                         'kl': actual_kl, 'grad_norm': float(grad_norm)}
                report['steps'].append(event)
                if progress_callback is not None:
                    progress_callback(dict(event))
        if source_weight_sha256(head.weight) != cache.manifest['source_weight_sha256'] or _bias_hash(head.bias) != cache.manifest['source_bias_sha256']:
            raise RuntimeError("Head calibration changed the original BF16 source")
        state = {name: p.detach().cpu().clone() for name, p in head.named_parameters()}
        head.clear_soft_snapshot()
        return HeadFitResult(report, state)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        head.training = training


@torch.no_grad()
def evaluate_head_from_cache(head, cache, *, source_provenance, token_microbatch=64, vocabulary_chunk=2048):
    """Evaluate the current head stage without changing progress/snapshot/state."""
    _positive_int(token_microbatch, 'token_microbatch')
    _positive_int(vocabulary_chunk, 'vocabulary_chunk')
    cache = _cache(cache)
    _check_source(head, cache, source_provenance)
    device, chunk = _head_device_and_chunk(head, vocabulary_chunk)
    total = 0.
    for index in range(len(cache)):
        for record in cache.iter_record(index, token_microbatch=token_microbatch):
            hidden = record['hidden'].to(device)
            mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=device)
            total += float(chunked_full_vocab_kl(
                lambda start, end: head.forward_rows(hidden, start, end),
                lambda start, end: head.teacher_rows(hidden, start, end), vocabulary=head.out_features,
                row_chunk=chunk, mask=mask, denominator=cache.positions, backward=False))
    return {'kl': total, 'positions': cache.positions, 'role': cache.manifest['role'],
            'source_provenance': cache.manifest['source_provenance'],
            'cache_manifest_sha256': _digest(cache.directory / 'manifest.json'),
            'objective': 'actual full-vocabulary KL(original_teacher || student), temperature 1',
            'normalization': 'FP64', 'effective_vocabulary_chunk': chunk,
            'head_progress': getattr(head, 'progress', None), 'quality_claim': False}
