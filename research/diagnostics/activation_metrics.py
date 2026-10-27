"""Bounded CPU statistics for a fixed five-stream activation diagnostic.

``ActivationMetrics(chunk_elements=65536).update(xq, x0, tq, t0, y, mask)``
accepts matching floating tensors [B, T, H] and a binary mask [B, T]. Each
batch row is one independent record; optional ``record_ids`` label those rows.
Repeated updates may have different sequence lengths. ``summary()`` returns
per-record and element-weighted pooled results composed only of JSON values.

Padding is removed before arithmetic, so NaNs in padding are irrelevant.
Detached input slices are copied to CPU, selected, and promoted to FP64 in
bounded chunks. No model, optimizer, correction or rescaling is applied. The
best scalar is a diagnostic of amplitude versus direction, not a proposed fix.
Empty records, malformed input and unrepresentable FP64 moments fail closed.
"""
from __future__ import annotations

import math

import torch


_PAIRS = {'xq/x0': ('xq', 'x0'), 'tq/t0': ('tq', 't0'),
          'y/tq': ('y', 'tq'), 'y/t0': ('y', 't0')}
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_MASK_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64, *_FLOAT_DTYPES)


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Activation statistics exceed the finite FP64 numeric range')
    return value


class _Sum:
    """Neumaier compensation across bounded chunk reductions."""
    def __init__(self):
        self.value = self.correction = 0.0

    def add(self, value):
        value = _finite(value)
        total = _finite(self.value + value)
        if abs(self.value) >= abs(value):
            adjustment = (self.value - total) + value
        else:
            adjustment = (value - total) + self.value
        self.correction = _finite(self.correction + adjustment)
        self.value = total

    def result(self):
        return _finite(self.value + self.correction)


def _pair_summary(raw, count):
    aa, bb, dot, error = (raw[key] for key in ('a2', 'ref2', 'dot', 'error2'))
    norm_a, norm_b = math.sqrt(aa), math.sqrt(bb)
    status = ('both_zero' if aa == bb == 0 else 'activation_zero' if aa == 0
              else 'reference_zero' if bb == 0 else 'nonzero')
    if aa == 0 or dot <= 0:
        scale, residual = 0.0, bb
    else:
        scale = _finite(dot / aa)
        # These equivalent forms avoid subtracting two large nearly equal
        # energies when a and ref are almost identical. The directly accumulated
        # a*(ref-a) retains small differences lost by subtracting dot - a2.
        if error < bb:
            change_dot = raw['a_dot_ref_minus_a']
            residual = error - change_dot * _finite(change_dot / aa)
        else:
            residual = bb - dot * scale
        if residual < -64 * math.ulp(max(error, bb)):
            raise ValueError('Scalar residual violates its nonnegative FP64 bound')
        residual = max(0.0, _finite(residual))
    return {
        'valid_elements': count, 'norm_status': status,
        'activation_rms': _finite(math.sqrt(aa / count)),
        'reference_rms': _finite(math.sqrt(bb / count)),
        'rms_ratio': None if bb == 0 else _finite(norm_a / norm_b),
        'cosine': None if aa == 0 or bb == 0 else max(-1.0, min(1.0, _finite(dot / norm_a / norm_b))),
        'mse': _finite(error / count),
        'relative_rmse': None if bb == 0 else _finite(math.sqrt(error) / norm_b),
        'best_nonnegative_scale': scale,
        'scale_is_unique': aa != 0,
        'rescaled_mse': _finite(residual / count),
        'rescaled_relative_rmse': None if bb == 0 else _finite(math.sqrt(residual) / norm_b),
        'scale_interpretation': 'diagnostic multiplier on activation; no model change',
    }


def _result(raw, tokens, count):
    pairs = {name: _pair_summary(raw[name], count) for name in _PAIRS}
    total = raw['y/t0']['error2']
    conditional = raw['y/tq']['error2']
    gap = raw['tq/t0']['error2']
    cross = raw['cross_term']
    rhs = _finite(math.fsum((conditional, gap, cross)))
    terms = {'total_error': total, 'conditional_error': conditional,
             'target_gap': gap, 'cross_term': cross}
    target_energy = raw['y/t0']['ref2']
    closure = _finite(total - rhs)
    closure_scale = max(total, _finite(math.fsum((conditional, gap, abs(cross)))))
    return {
        'valid_tokens': tokens, 'valid_elements': count, 'pairs': pairs,
        'decomposition': {
            'identity': '||Y-T0||^2 = ||Y-Tq||^2 + ||Tq-T0||^2 + 2 dot(Y-Tq,Tq-T0)',
            'sum_terms': terms,
            'mean_terms': {key: _finite(value / count) for key, value in terms.items()},
            'original_target_energy': target_energy,
            'terms_divided_by_original_target_energy': None if target_energy == 0 else
                {key: _finite(value / target_energy) for key, value in terms.items()},
            'terms_divided_by_total_error': None if total == 0 else
                {key: _finite(value / total) for key, value in terms.items()},
            'total_error_normalization_status': 'undefined_zero_total_error' if total == 0 else 'defined',
            'target_gap_over_total_error_rms': None if total == 0 else _finite(math.sqrt(gap) / math.sqrt(total)),
            'closure_residual_sum': closure,
            'relative_closure_residual': 0.0 if closure_scale == 0 else _finite(closure / closure_scale),
            'interpretation': 'The cross term prevents causal attribution of final error to the target gap.',
        },
    }


class ActivationMetrics:
    """Accumulate independent records without retaining their activations.

    All five streams must describe the same unshifted [B,T,H] positions. A
    failed update adds no records. Duplicate IDs and entirely masked records
    are rejected. FP64 accumulation is on CPU even when inputs live elsewhere.
    ``chunk_elements`` bounds each copied/decoded activation slice, not the
    input tensors owned by the caller or the small per-record result history.
    """
    def __init__(self, chunk_elements=65536):
        if type(chunk_elements) is not int or chunk_elements <= 0:
            raise ValueError('chunk_elements must be a positive integer')
        self.chunk_elements = chunk_elements
        self._records = []
        self._ids = set()

    def update(self, xq, x0, tq, t0, y, mask, *, record_ids=None):
        streams = dict(xq=xq, x0=x0, tq=tq, t0=t0, y=y)
        if any(not isinstance(value, torch.Tensor) or value.dtype not in _FLOAT_DTYPES
               or value.ndim != 3 for value in streams.values()):
            raise ValueError('Activations must be real floating tensors with shape [B,T,H]')
        shape = tuple(xq.shape)
        if any(tuple(value.shape) != shape for value in streams.values()) or any(size <= 0 for size in shape):
            raise ValueError('Activations must have matching nonempty [B,T,H] shapes')
        batch, length, width = shape
        if (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (batch, length)
                or mask.dtype not in _MASK_DTYPES):
            raise ValueError('mask must be a binary tensor with shape [B,T]')
        if record_ids is None:
            record_ids = [f'record-{len(self._records) + index:06d}' for index in range(batch)]
        if (not isinstance(record_ids, (list, tuple)) or len(record_ids) != batch
                or any(not isinstance(value, str) or not value for value in record_ids)):
            raise ValueError('record_ids must contain one nonempty string per batch row')
        if len(set(record_ids)) != batch or self._ids.intersection(record_ids):
            raise ValueError('record_ids must be unique')
        additions = []
        token_chunk = max(1, self.chunk_elements // width)
        hidden_chunk = min(width, self.chunk_elements)
        for record, record_id in enumerate(record_ids):
            moments = {name: {key: _Sum() for key in ('a2', 'ref2', 'dot', 'error2', 'a_dot_ref_minus_a')}
                       for name in _PAIRS}
            cross = _Sum()
            tokens = 0
            for first in range(0, length, token_chunk):
                last = min(first + token_chunk, length)
                chunk_mask = mask[record, first:last].detach().to(device='cpu')
                if not bool(((chunk_mask == 0) | (chunk_mask == 1)).all()):
                    raise ValueError('mask must contain only zero and one')
                selected = torch.nonzero(chunk_mask, as_tuple=True)[0]
                tokens += selected.numel()
                if not selected.numel():
                    continue
                for hidden in range(0, width, hidden_chunk):
                    end = min(hidden + hidden_chunk, width)
                    values = {name: value[record, first:last, hidden:end].detach().to(device='cpu')
                              .index_select(0, selected).to(dtype=torch.float64)
                              for name, value in streams.items()}
                    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
                        raise ValueError('Valid activations must be finite')
                    for name, (left, right) in _PAIRS.items():
                        a, ref = values[left], values[right]
                        difference = ref - a
                        chunk = {'a2': (a*a).sum(), 'ref2': (ref*ref).sum(), 'dot': (a*ref).sum(),
                                 'error2': (difference*difference).sum(),
                                 'a_dot_ref_minus_a': (a*difference).sum()}
                        for key, value in chunk.items():
                            moments[name][key].add(value.item())
                    cross.add((2 * ((values['y'] - values['tq']) * (values['tq'] - values['t0'])).sum()).item())
            if tokens == 0:
                raise ValueError('Each record must contain at least one valid token')
            raw = {name: {key: value.result() for key, value in values.items()}
                   for name, values in moments.items()}
            raw['cross_term'] = cross.result()
            # Validate derived metrics before atomically accepting this update.
            _result(raw, tokens, tokens * width)
            additions.append((record_id, tokens, tokens * width, raw))
        self._records.extend(additions)
        self._ids.update(record_ids)
        return self

    def summary(self):
        if not self._records:
            raise ValueError('No valid activation records have been accumulated')
        raw = {name: {key: _finite(math.fsum(item[3][name][key] for item in self._records))
                      for key in self._records[0][3][name]} for name in _PAIRS}
        raw['cross_term'] = _finite(math.fsum(item[3]['cross_term'] for item in self._records))
        tokens = sum(item[1] for item in self._records)
        count = sum(item[2] for item in self._records)
        return {
            'format': 'openbonsai.activation-metrics.v1',
            'arithmetic': 'detached CPU FP64 reductions with compensated chunk sums',
            'pooling': 'sum of valid element moments; not an average of record metrics',
            'padding': 'binary mask selects tokens before activation arithmetic',
            'record_count': len(self._records),
            'records': [dict(record_id=name, **_result(values, nt, ne)) for name, nt, ne, values in self._records],
            'pooled': _result(raw, tokens, count),
            'quality_claim': False,
        }
