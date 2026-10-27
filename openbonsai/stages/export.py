"""PTQ1_0 GGUF export (phase P2).

The PTQ1_0 block codec is in openbonsai.formats.ptq1_0; the GGUF writer and
the qwen35 tensor map arrive in P2. Until then this stage is a placeholder
that ``openbonsai run`` skips and ``openbonsai export`` reports clearly.
"""
from __future__ import annotations


class ExportNotAvailable(RuntimeError):
    pass


MESSAGE = ("PTQ1_0 GGUF export is phase P2 and not implemented yet. The run state "
           "(quantize/endpoints state.safetensors + distill/scales.safetensors) holds "
           "everything the exporter needs; the block codec is openbonsai.formats.ptq1_0.")


def run(ctx) -> dict:
    raise ExportNotAvailable(MESSAGE)
