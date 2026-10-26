"""Bounded phase observations for experimental compression stages.

The external process supervisor enforces wall time. CUDA observations account
for this process's allocator peaks and before/after device free memory, not
unobserved transient allocations by other processes or a total host RAM cap.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time


def write_json(path, value, *, replace=False):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    temporary = Path(str(path) + ".tmp") if replace else path
    with temporary.open("xb") as stream:
        stream.write(data)
    if replace:
        os.replace(temporary, path)


def check_deadline(deadline):
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("Compression stage deadline exhausted")
    return remaining


class StageMeter:
    def __init__(self, output, device, deadline, minimum_free_cuda_bytes):
        import torch
        self.torch, self.device, self.deadline = torch, torch.device(device), deadline
        if type(minimum_free_cuda_bytes) is not int or minimum_free_cuda_bytes < 0:
            raise ValueError("minimum_free_cuda_bytes must be a nonnegative integer")
        if self.device.type == "cuda" and minimum_free_cuda_bytes < 1 << 30:
            raise ValueError("CUDA stages require at least 1 GiB free-memory margin")
        self.minimum_free, self.index = minimum_free_cuda_bytes, 0
        self.output = Path(output)
        self.output.mkdir(exist_ok=False)

    def __call__(self, label, function):
        check_deadline(self.deadline)
        torch, device = self.torch, self.device
        cuda = device.type == "cuda"
        self.index += 1
        path = self.output / f"{self.index:06d}.json"
        row = {"label": label, "status": "started", "device": str(device),
               "minimum_free_cuda_bytes": self.minimum_free if cuda else None,
               "headroom_method": "before/after device free plus own peak allocator reservation"}
        write_json(path, row)
        started, begun = time.monotonic(), False
        try:
            if cuda:
                torch.cuda.synchronize(device)
                free, total = map(int, torch.cuda.mem_get_info(device))
                row.update(initial_free_bytes=free, total_device_bytes=total,
                           initial_reserved_bytes=int(torch.cuda.memory_reserved(device)))
                if free < self.minimum_free:
                    raise RuntimeError("Insufficient initial CUDA headroom")
                torch.cuda.reset_peak_memory_stats(device)
                begun = True
            result = function()
            if cuda:
                torch.cuda.synchronize(device)
                free_after, _ = torch.cuda.mem_get_info(device)
                peak = int(torch.cuda.max_memory_reserved(device))
                conservative = row["initial_free_bytes"] + row["initial_reserved_bytes"] - peak
                row.update(ending_free_bytes=int(free_after), peak_reserved_bytes=peak,
                           peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                           conservative_free_bytes=conservative)
                if min(row["initial_free_bytes"], int(free_after), conservative) < self.minimum_free:
                    raise RuntimeError("CUDA peak headroom gate failed")
            check_deadline(self.deadline)
            row["status"] = "complete"
            return result
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            if cuda and begun:
                try:
                    row.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                               peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
                except BaseException:
                    pass
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            write_json(path, row, replace=True)
