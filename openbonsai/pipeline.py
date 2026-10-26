"""Run one explicit, source-bound research compression stage in an owned child.

No downloads, automatic next stage, resume, or quality acceptance. Each stage
gets a separate whole-process deadline. Chaining requires a pinned successful
outer result, not merely a candidate written before a failed resource gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from .eval_backends import _identity
from .pipeline_sources import _digest, _read_json, code_identity
from .stage_runtime import check_deadline, write_json

REQUEST_FORMAT = "openbonsai.pipeline-request.v1"
RESULT_FORMAT = "openbonsai.pipeline-result.v1"
STAGES = {"prepare", "seal", "decoder", "cache-head", "head", "embedding", "assemble"}
_SELECTION_FIELDS = {"num_records", "consumed_jsonl_sha256", "record_sha256s",
                     "raw_token_record_sha256s"}


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} requires exactly {sorted(keys)}")


def _absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Pipeline paths must be absolute")
    return Path(value)


def _pinned(reference):
    _exact(reference, {"path", "sha256"}, "reference")
    path = _absolute(reference["path"])
    identity = _identity(path, reference["sha256"])
    value = _read_json(path)
    if _identity(path) != identity:
        raise RuntimeError("Pinned JSON changed while reading")
    return value


def _validate_trace_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Trace specification must be an object")
    if "historical" not in spec:
        _exact(spec, {"prepared", "sequence_length", "max_samples"}, "prepared trace")
        return
    _exact(spec, {"historical", "sequence_length", "max_samples"}, "historical trace")
    historical = spec["historical"]
    _exact(historical, {"traces", "producer", "expected_selection"}, "historical references")
    for key in ("traces", "producer"):
        reference = historical[key]
        _exact(reference, {"path", "sha256"}, "historical " + key)
        _absolute(reference["path"])
        _digest(reference["sha256"], "historical " + key)
    expected = historical["expected_selection"]
    _exact(expected, _SELECTION_FIELDS, "expected selection")
    count = expected["num_records"]
    if type(count) is not int or count < 1:
        raise ValueError("Expected selection num_records must be a positive integer")
    if type(spec["sequence_length"]) is not int or spec["sequence_length"] < 1:
        raise ValueError("Historical sequence_length must be a positive integer")
    maximum = spec["max_samples"]
    if maximum is not None and (type(maximum) is not int or maximum < count):
        raise ValueError("Historical max_samples must be null or at least expected num_records")
    _digest(expected["consumed_jsonl_sha256"], "consumed_jsonl_sha256")
    for key in ("record_sha256s", "raw_token_record_sha256s"):
        if not isinstance(expected[key], list) or len(expected[key]) != count:
            raise ValueError(f"Expected {key} must contain exactly num_records hashes")
        for digest in expected[key]:
            _digest(digest, key)


def _historical_pins(spec):
    """Pin raw JSONL as bytes; only the producer is a single JSON document."""
    historical = spec["historical"]
    traces = _identity(historical["traces"]["path"], historical["traces"]["sha256"])
    producer = _identity(historical["producer"]["path"], historical["producer"]["sha256"])
    _pinned(historical["producer"])
    return traces, producer


def completed_stage(reference, expected_stage):
    """Resolve only a successful supervised stage and its unchanged payload."""
    result = _pinned(reference)
    if (result.get("format") != RESULT_FORMAT or result.get("status") != "complete"
            or result.get("stage") != expected_stage
            or result.get("supervision", {}).get("status") != "complete"
            or result.get("supervision", {}).get("exit_code") != 0
            or result.get("supervision", {}).get("cleanup", {}).get("no_live_owned_processes") is not True):
        raise ValueError("Stage requires a complete, cleanly supervised producer result")
    expected_name = {"prepare": "producer.json", "seal": "job.json"}.get(expected_stage, "stage.json")
    path = _absolute(result["payload"]["path"])
    owned = Path(reference["path"]).parent / "stage" / expected_name
    if path != owned or path.is_symlink():
        raise ValueError("Stage payload must be its owned inner handoff")
    if _identity(path, result["payload"]["sha256"]) != result["payload"]:
        raise ValueError("Stage payload changed after supervised completion")
    return path, _read_json(path)


def validate_request(request):
    _exact(request, {"format", "stage", "parameters"}, "request")
    if request["format"] != REQUEST_FORMAT or request["stage"] not in STAGES:
        raise ValueError("Unsupported pipeline request")
    stage, p = request["stage"], request["parameters"]
    keys = {
        "prepare": {"model_directory", "expected_model_sha256", "tokenizer_directory",
                    "expected_tokenizer_sha256", "prompts", "expected_prompts_sha256", "generation", "role"},
        "seal": {"model_directory", "tokenizer_directory", "traces_by_role", "decoder", "head", "embedding", "assembly"},
        "decoder": {"job"}, "embedding": {"job"}, "cache-head": {"job", "role"},
        "head": {"job", "calibration_cache", "search_cache", "evaluation_cache"},
        "assemble": {"job", "candidate_stages"},
    }
    expected = keys[stage]
    if stage == "assemble" and isinstance(p, dict) and "assembly_control" in p:
        expected = expected | {"assembly_control"}
    _exact(p, expected, stage + " parameters")
    if stage == "assemble" and "assembly_control" in p:
        from .pipeline_decoder import validate_assembly_control
        validate_assembly_control(p["assembly_control"])
    for key in ("model_directory", "tokenizer_directory", "prompts"):
        if key in p:
            _absolute(p[key])
    if stage == "prepare":
        from .pipeline_prepare import validate_generation
        validate_generation(p["generation"])
    if stage == "seal":
        if not isinstance(p["traces_by_role"], dict) or "calibration" not in p["traces_by_role"]:
            raise ValueError("Seal requires calibration traces")
        for spec in p["traces_by_role"].values():
            _validate_trace_spec(spec)
    if stage == "assemble" and (not isinstance(p["candidate_stages"], list) or not p["candidate_stages"]):
        raise ValueError("Assembly requires candidate stage references")
    return request


def protected_paths(request):
    """Discover sealed roots before the supervisor creates any output files."""
    stage, p = request["stage"], request["parameters"]
    roots = [Path(__file__).parent]
    roots += [_absolute(p[key]) for key in ("model_directory", "tokenizer_directory", "prompts") if key in p]
    if "job" in p:
        job_path, job = completed_stage(p["job"], "seal")
        if stage == "assemble" and "assembly_control" in p:
            from .pipeline_decoder import validate_assembly_control
            validate_assembly_control(p["assembly_control"], sealed_assembly=job["assembly"])
        roots += [job_path.parent.parent, Path(job["source"]["model_directory"]["path"]),
                  Path(job["source"]["tokenizer_directory"]["path"])]
    refs = []
    if stage == "seal":
        for spec in p["traces_by_role"].values():
            if "historical" in spec:
                # Protect the owning input directories before creating output.
                # Historical evidence is attributed, never a prepare result.
                roots += [Path(info["path"]).parent for info in _historical_pins(spec)]
            else:
                refs.append((spec["prepared"], "prepare"))
    if stage == "head":
        refs += [(p[key], "cache-head") for key in ("calibration_cache", "search_cache", "evaluation_cache") if p[key] is not None]
    if stage == "assemble":
        for ref in p["candidate_stages"]:
            value = _pinned(ref)
            if value.get("stage") not in {"decoder", "head", "embedding"}:
                raise ValueError("Assembly accepts only calibrated candidate stages")
            refs.append((ref, value["stage"]))
    for ref, kind in refs:
        path, body = completed_stage(ref, kind)
        roots.append(path.parent.parent)
        if kind == "prepare":
            roots += [Path(body["model_source"]["path"]), Path(body["tokenizer_source"]["path"]),
                      Path(body["prompt_source"]["path"])]
    return roots


def execute_stage(request, output, device, meter, progress):
    """Internal dispatch; CLI callers must use supervisor_main, not this API."""
    from . import pipeline_sources as sources
    from .pipeline_decoder import assemble_stage, fit_decoder_stage
    from .pipeline_endpoints import cache_head_stage, fit_embedding_stage, fit_head_stage
    stage, p = request["stage"], copy.deepcopy(request["parameters"])
    output = Path(output)
    if stage == "prepare":
        from .pipeline_prepare import prepare_bound
        prepare_bound(**p, output=output, device=device, operation=meter, progress_callback=progress)
        return output / "producer.json"
    if stage == "seal":
        historical_inputs = {}
        for role, spec in list(p["traces_by_role"].items()):
            if "historical" in spec:
                traces, producer = _historical_pins(spec)
                historical_inputs[role] = (traces, producer, spec["historical"]["expected_selection"])
                trace_path, producer_ref = traces["path"], spec["historical"]["producer"]
            else:
                path, producer = completed_stage(spec["prepared"], "prepare")
                if producer["role"] != role:
                    raise ValueError("Prepared trace role differs from sealed role")
                trace_path = str(path.parent / "traces.jsonl")
                producer_ref = {"path": str(path), "sha256": _identity(path)["sha256"]}
            p["traces_by_role"][role] = {"path": trace_path, "producer": producer_ref,
                "sequence_length": spec["sequence_length"], "max_samples": spec["max_samples"]}
        context = meter("seal-job", lambda: sources.seal_job(**p, output=output))
        for role, (traces, producer, expected) in historical_inputs.items():
            receipt = context.record_receipts_by_role[role]
            if any(receipt[key] != expected[key] for key in _SELECTION_FIELDS):
                raise ValueError(f"Historical selection differs from expected records: {role}")
            for original, copied in ((traces, receipt["file"]), (producer, receipt["producer_file"])):
                if (_identity(original["path"]) != original
                        or any(copied[key] != original[key] for key in ("sha256", "bytes"))
                        or _identity(copied["path"]) != copied):
                    raise RuntimeError(f"Historical input changed while sealing: {role}")
        return output / "job.json"
    job_path, _ = completed_stage(p.pop("job"), "seal")
    context = meter("load-sealed-context", lambda: sources.load_context(job_path))
    if stage == "decoder":
        fit_decoder_stage(context, output, device=device, operation=meter, progress_callback=progress)
    elif stage == "embedding":
        fit_embedding_stage(context, output, device=device, operation=meter, progress_callback=progress)
    elif stage == "cache-head":
        cache_head_stage(context, p["role"], output, device=device, operation=meter, progress_callback=progress)
    elif stage == "head":
        caches = {key: None if ref is None else completed_stage(ref, "cache-head")[0].parent
                  for key, ref in p.items()}
        fit_head_stage(context, output=output, device=device, operation=meter,
                       progress_callback=progress, **caches)
    elif stage == "assemble":
        if device != "cpu":
            raise ValueError("Assembly is a CPU-only stage")
        directories = []
        for ref in p["candidate_stages"]:
            kind = _pinned(ref)["stage"]
            if kind not in {"decoder", "head", "embedding"}:
                raise ValueError("Invalid candidate stage")
            directories.append(completed_stage(ref, kind)[0].parent)
        report = assemble_stage(context, directories, output, operation=meter,
                                assembly_control=p.get("assembly_control"))
        write_json(output / "stage.json", {"format": sources.STAGE_FORMAT, "status": "complete", **report})
    meter("final-context-preservation", lambda: sources.verify_context(context))
    return output / "stage.json"


def child_main(job_path):
    job = _read_json(job_path)
    if os.getppid() != job["supervisor_pid"]:
        raise RuntimeError("Child must run under its owning supervisor")
    output, deadline = Path(job["output"]), job["deadline_monotonic"]
    status = {"status": "starting", "stage": job["stage"], "quality_claim": False}
    write_json(output / "child-status.json", status)
    try:
        check_deadline(deadline)
        if code_identity() != job["code"]:
            raise RuntimeError("Implementation changed after freeze")
        request = validate_request(_pinned(job["request"]))
        import torch
        import platform
        import importlib.metadata
        from .stage_runtime import StageMeter
        torch.set_num_threads(job["threads"])
        status["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "psutil")},
            "torch_cuda": torch.version.cuda, "device": job["device"], "threads": job["threads"]}
        meter = StageMeter(output / "phases", job["device"], deadline, job["minimum_free_cuda_bytes"])
        def progress(event):
            check_deadline(deadline)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        payload = execute_stage(request, output / "stage", job["device"], meter, progress)
        check_deadline(deadline)
        if code_identity() != job["code"] or _pinned(job["request"]) != request:
            raise RuntimeError("Implementation or request changed during stage")
        status.update(status="complete", payload=_identity(payload))
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        status["elapsed_from_supervisor_start_seconds"] = time.monotonic() - job["started_monotonic"]
        write_json(output / "child-status.json", status, replace=True)
    return 0 if status["status"] == "complete" else 1


def supervisor_main(args):
    if (type(args.max_seconds) not in (int, float) or not math.isfinite(args.max_seconds)
            or not 0 < args.max_seconds <= 14400):
        raise ValueError("max-seconds must be in (0,14400]")
    if type(args.threads) is not int or not 1 <= args.threads <= 64:
        raise ValueError("threads must be in [1,64]")
    if args.device not in {"cpu", "cuda:0"}:
        raise ValueError("device must be cpu or cuda:0")
    if (type(args.minimum_free_cuda_bytes) is not int or args.minimum_free_cuda_bytes < 0
            or args.device == "cuda:0" and args.minimum_free_cuda_bytes < 1 << 30):
        raise ValueError("Explicit nonnegative headroom is required; CUDA minimum is1GiB")
    started = time.monotonic()
    deadline = started + args.max_seconds
    reference = {"path": str(Path(args.request).absolute()), "sha256": args.expected_request_sha256}
    request = validate_request(_pinned(reference))
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for root in [Path(reference["path"]), *protected_paths(request)]:
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError("Output cannot modify a sealed input or implementation directory")
    if request["stage"] in {"seal", "assemble"} and args.device != "cpu":
        raise ValueError("Seal and assembly run on CPU")
    check_deadline(deadline)
    output.mkdir(parents=True, exist_ok=False)
    try:
        # Preserve the exact caller bytes. Semantic JSON serialization is not a
        # replacement for the pinned request file's content hash.
        frozen = output / "request.json"
        with frozen.open("xb") as stream:
            stream.write(Path(reference["path"]).read_bytes())
        _identity(frozen, reference["sha256"])
        job = {"stage": request["stage"], "request": {"path": str(frozen), "sha256": reference["sha256"]},
               "original_request": reference, "output": str(output), "supervisor_pid": os.getpid(),
               "started_monotonic": started, "deadline_monotonic": deadline,
               "maximum_whole_process_seconds": args.max_seconds, "threads": args.threads,
               "device": args.device, "minimum_free_cuda_bytes": args.minimum_free_cuda_bytes,
               "code": code_identity()}
        write_json(output / "attempt.json", job)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        from .evaluation_supervisor import supervise
        supervision = supervise([sys.executable, "-m", "openbonsai.pipeline", "--child-job", str(output / "attempt.json")],
                                output, deadline, env=env)
        result = {"format": RESULT_FORMAT, "stage": request["stage"], "status": "failed",
                  "supervision": supervision, "request": reference, "code": job["code"],
                  "maximum_whole_process_seconds": args.max_seconds, "quality_claim": False,
                  "whole_process_elapsed_seconds": time.monotonic() - started}
        if supervision["status"] == "complete":
            status = _read_json(output / "child-status.json")
            if status["status"] != "complete" or code_identity() != job["code"]:
                raise RuntimeError("Missing complete child receipt or changed code")
            payload = status["payload"]
            if _identity(payload["path"], payload["sha256"]) != payload:
                raise RuntimeError("Published payload differs")
            result.update(status="complete", payload=payload, environment=status["environment"])
        write_json(output / "result.json", result)
        return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()})
        raise


def add_arguments(parser):
    parser.add_argument("--run", action="store_true", help="Explicitly execute one stage; no automatic successor")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--expected-request-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--minimum-free-cuda-bytes", type=int)
    return parser


def run(args):
    if not args.run or any(getattr(args, key) is None for key in (
            "request", "expected_request_sha256", "output", "threads", "max_seconds", "minimum_free_cuda_bytes")):
        raise ValueError("Supply --run, pinned request, new output and explicit thread/time/headroom limits")
    return supervisor_main(args)


def main(argv=None):
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--child-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_job is not None:
        return child_main(args.child_job)
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output.absolute())}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
