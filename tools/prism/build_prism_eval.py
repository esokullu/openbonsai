#!/usr/bin/env python3
"""Build the reference JSONL worker against an exact clean Prism source revision.

Requires git, CMake and a C++17 compiler; downloads neither dependencies nor weights.
Clone Prism separately, then pass --source and an out-of-tree --build-dir.
CPU is the default (including Metal OFF on macOS); --cuda opts into a CUDA build.
The generated build receipt records effective CMake settings and file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

PIN = "c9c34bc9c4042c13a31598ebf43461cd0ad8f00d"
ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(argv, check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cmake_path(path: Path) -> str:
    value = path.as_posix()
    if any(character in value for character in ('"', ';', '\n', '\r', '$')):
        raise ValueError("CMake paths cannot contain quotes, semicolons, newlines or dollar signs")
    return '"' + value + '"'


def verify_source(source: Path) -> None:
    actual = run(["git", "-C", str(source), "rev-parse", "HEAD"], capture=True)
    if actual != PIN:
        raise ValueError(f"Prism source must be pinned to {PIN}; found {actual}")
    if run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], capture=True):
        raise ValueError("Prism source has modifications or untracked files; a clean pinned source is required")
    for relative in ("include/llama.h", "vendor/nlohmann/json.hpp", "LICENSE"):
        if not (source / relative).is_file():
            raise ValueError(f"missing Prism source file: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--cuda", action="store_true", help="explicitly compile CUDA; default is CPU only")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    source = args.source.resolve()
    build = args.build_dir.resolve()
    if build == source or source in build.parents:
        parser.error("--build-dir must be outside the pinned source tree")
    verify_source(source)
    build.mkdir(parents=True, exist_ok=True)
    project = build / "worker-project"
    project.mkdir(exist_ok=True)
    worker = ROOT / "tools/prism_eval_worker.cpp"
    unit = ROOT / "tests/prism_worker_unit.cpp"
    text = f'''cmake_minimum_required(VERSION 3.20)
project(openbonsai_prism_eval LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${{CMAKE_BINARY_DIR}}/bin")
add_subdirectory({cmake_path(source)} prism)
foreach(target prism-eval-worker prism-worker-unit)
  if(target STREQUAL "prism-eval-worker")
    add_executable(${{target}} {cmake_path(worker)})
  else()
    add_executable(${{target}} {cmake_path(unit)})
  endif()
  target_link_libraries(${{target}} PRIVATE llama)
  target_include_directories(${{target}} PRIVATE {cmake_path(source / "vendor")})
  target_compile_definitions(${{target}} PRIVATE OPENBONSAI_PRISM_SHA="{PIN}" OPENBONSAI_CUDA_BUILD={int(args.cuda)})
endforeach()
'''
    (project / "CMakeLists.txt").write_text(text, encoding="utf-8")
    flags = {
        "CMAKE_BUILD_TYPE": "Release", "BUILD_SHARED_LIBS": "OFF",
        "LLAMA_BUILD_COMMON": "OFF", "LLAMA_BUILD_TESTS": "OFF",
        "LLAMA_BUILD_TOOLS": "OFF", "LLAMA_BUILD_EXAMPLES": "OFF",
        "LLAMA_BUILD_SERVER": "OFF", "LLAMA_BUILD_APP": "OFF",
        "LLAMA_BUILD_UI": "OFF", "LLAMA_OPENSSL": "OFF",
        "GGML_NATIVE": "OFF", "GGML_METAL": "OFF",
        "GGML_CUDA": "ON" if args.cuda else "OFF",
        "GGML_VULKAN": "OFF", "GGML_BLAS": "OFF",
        "GGML_HIP": "OFF", "GGML_MUSA": "OFF", "GGML_SYCL": "OFF",
        "GGML_OPENCL": "OFF", "GGML_OPENVINO": "OFF", "GGML_CANN": "OFF",
        "GGML_WEBGPU": "OFF", "GGML_VIRTGPU": "OFF", "GGML_VIRTGPU_BACKEND": "OFF",
        "GGML_HEXAGON": "OFF", "GGML_RPC": "OFF",
        "GGML_OPENMP": "OFF", "GGML_BACKEND_DL": "OFF",
        "GGML_ACCELERATE": "ON" if sys.platform == "darwin" else "OFF",
    }
    configure = ["cmake", "-S", str(project), "-B", str(build / "cmake")]
    configure += [f"-D{key}={value}" for key, value in flags.items()]
    run(configure)
    run(["cmake", "--build", str(build / "cmake"), "--config", "Release", "--target",
         "prism-eval-worker", "prism-worker-unit", "--parallel", str(args.jobs)])
    suffix = ".exe" if os.name == "nt" else ""
    executable = build / "cmake/bin" / ("prism-eval-worker" + suffix)
    unit_executable = build / "cmake/bin" / ("prism-worker-unit" + suffix)
    unit_result = json.loads(run([str(unit_executable)], capture=True))
    verify_source(source)
    cache = build / "cmake/CMakeCache.txt"
    effective = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(("CMAKE_", "GGML_", "LLAMA_", "BUILD_SHARED_LIBS:")) and "=" in line:
            key, value = line.split("=", 1)
            effective[key] = value
    receipt = {
        "format": "openbonsai.prism-worker-build.v1", "runtime_sha": PIN,
        "source": str(source), "executable": str(executable),
        "executable_sha256": sha256(executable), "worker_source_sha256": sha256(worker),
        "unit_source_sha256": sha256(unit), "build_helper_sha256": sha256(Path(__file__)),
        "prism_license_sha256": sha256(source / "LICENSE"), "cuda": args.cuda,
        "platform": platform.platform(), "cmake_version": run(["cmake", "--version"], capture=True),
        "configure_argv": configure, "effective_cmake_cache": effective,
        "unit_result": unit_result,
        "compiler_configuration": {
            str(path.relative_to(build / "cmake")): path.read_text(encoding="utf-8")
            for path in sorted((build / "cmake/CMakeFiles").glob("*/CMake*Compiler.cmake"))
        },
    }
    receipt_path = build / "build-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"executable": str(executable), "receipt": str(receipt_path), "unit_result": unit_result}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"build_prism_eval: {error}", file=sys.stderr)
        raise SystemExit(1)
