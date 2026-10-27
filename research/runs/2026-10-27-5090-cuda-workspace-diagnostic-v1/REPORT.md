# Synthetic CUDA workspace diagnostic — one bounded fixture

Outcome: **passed**. Retained library workspace supported for this synthetic fixture only

This is a synthetic allocation diagnostic, not a compression or quality experiment. The failed head preflight remains failed and unchanged. This fixture does not retrospectively prove its cause. No model/checkpoint/token/activation-cache inputs, optimizer, calibration, retry, or successor were used.

One-child accounting: 1 launched record(s), 2.672000 seconds against the external 120-second cap. All owned children reaped: True. The supervisor measured the whole launch/wait interval; preparation and final CPU review are separate.

Measured peak allocated: 257950208 bytes (246.000 MiB; cap 1,024 MiB). Peak reserved: 276824064 bytes (264.000 MiB). Minimum conservative headroom: 32081182720 bytes (30595.000 MiB; floor 4,096 MiB).

Headroom at each checkpoint is the minimum of ending free bytes and previous free bytes + previous reserved bytes − cumulative peak reserved bytes. Cumulative peaks make this conservative; current allocations are recorded separately.

| Checkpoint | Current allocated MiB | Current reserved MiB | Peak allocated MiB | Peak reserved MiB | Device free MiB | Conservative free MiB | Source dead | Allocated <16 MiB | Conjunction |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| initial_empty | 0.000 | 0.000 | 0.000 | 0.000 | 30991.000 | 30991.000 | False | True | False |
| sentinel_created | 32.000 | 32.000 | 32.000 | 32.000 | 30845.000 | 30845.000 | False | False | False |
| sentinel_after_first_clear | 32.000 | 32.000 | 32.000 | 32.000 | 30845.000 | 30845.000 | False | False | False |
| synthetic_refs_released_before_second_clear | 96.000 | 96.000 | 246.000 | 264.000 | 30763.000 | 30613.000 | True | False | False |
| after_second_workspace_clear | 32.000 | 32.000 | 246.000 | 264.000 | 30827.000 | 30595.000 | True | False | False |
| after_sentinel_deleted | 0.000 | 0.000 | 246.000 | 264.000 | 30859.000 | 30595.000 | True | True | True |

“Source dead” is false before a synthetic source reference exists; the complete per-reference observations are in each phase JSON. No ending allocation is inferred from a peak.

| Reference | Before second clear | After second clear | After sentinel deletion |
|---|---|---|---|
| input | dead | dead | dead |
| source_weight | dead | dead | dead |
| output | dead | dead | dead |
| loss | dead | dead | dead |
| input_gradient | dead | dead | dead |
| source_weight_gradient | dead | dead | dead |
| sentinel | live | live | dead |

Second synchronized workspace clear removed exactly **67108864 bytes** (64.000 MiB) of current allocation. All registered references finally dead: True; current allocation exactly back to the empty baseline: True.

The deliberate sentinel is 33,554,432 bytes of uint8 value 17. Its identity, exact storage size, weak reference, and first/middle/last values are recorded. Its live allocation must keep the unchanged strict 16 MiB gate false across both clears. Only its deletion may allow that gate to pass.

Synthetic operation counts: `{"backward_completed": 1, "backward_started": 1, "checkpoints": 6, "clear_completed": 2, "clear_started": 2, "forward_completed": 1, "forward_started": 1}`. The sole forward uses BF16 input [256,4096] filled with 1/16 and weight [4096,4096] filled with 1/256. Both are gradient-enabled leaves; loss is output.float().mean(), followed by one backward without graph retention. Only weak references and CPU scalar metadata leave that function.

Observed forward/gradient checks: `{"input_grad_first": 1.52587890625e-05, "input_gradient_finite": true, "loss": 1.0, "output_first": 1.0, "weight_grad_first": 1.52587890625e-05, "weight_gradient_finite": true}`.

Runtime: `{"cuda_version": "13.0", "distribution_root": "D:\\openbonsai\\.venv\\Lib\\site-packages", "distribution_version": "2.14.1+cu130", "python_base_executable": "C:\\Users\\esoku\\AppData\\Roaming\\uv\\python\\cpython-3.12.11-windows-x86_64-none\\python.exe", "python_base_prefix": "C:\\Users\\esoku\\AppData\\Roaming\\uv\\python\\cpython-3.12.11-windows-x86_64-none", "python_executable": "D:\\openbonsai\\.venv\\Scripts\\python.exe", "python_prefix": "D:\\openbonsai\\.venv", "python_version": "3.12.11 (main, Aug 18 2025, 19:17:54) [MSC v.1944 64 bit (AMD64)]", "torch_extension_file": "D:\\openbonsai\\.venv\\Lib\\site-packages\\torch\\_C.cp312-win_amd64.pyd", "torch_git_version": "5c4886908584029761b579af026dcfb627c84070", "torch_package_file": "D:\\openbonsai\\.venv\\Lib\\site-packages\\torch\\__init__.py", "torch_version": "2.14.1+cu130"}`.

Workspace-size getter results (configuration sizes, not counts of allocations): `{"cublas_bytes": 33554432, "cublaslt_bytes": 1048576, "status": "queried_after_existing_cuda_initialization"}`. They were skipped before initialization because the pinned getter can query device properties. No current-BLAS-handle getter or workspace-size setter was called.

Actual settings: `{"allow_bf16_reduced_precision_reduction": true, "allow_bf16_reduced_precision_reduction_split_k": true, "allow_fp16_accumulation": false, "allow_fp16_reduced_precision_reduction": true, "allow_fp16_reduced_precision_reduction_split_k": true, "cudnn_allow_tf32": true, "cudnn_benchmark": false, "cudnn_deterministic": false, "cudnn_enabled": true, "deterministic_algorithms": false, "deterministic_warn_only": false, "float32_matmul_precision": "highest", "matmul_allow_tf32": false, "matmul_fp32_precision": "none", "preferred_blas_library": "_BlasBackend.Cublas"}`. Allocator backend: `native`. Environment: `{"CUBLASLT_WORKSPACE_SIZE": null, "CUBLAS_WORKSPACE_CONFIG": null, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_LAUNCH_BLOCKING": null, "CUDA_MODULE_LOADING": null, "CUDA_VISIBLE_DEVICES": "GPU-6d908ed7-d2e6-1c70-8a1e-28754a8cbfae", "CUDNN_CONV_WSCAP_DBG": null, "MKL_NUM_THREADS": null, "NVIDIA_TF32_OVERRIDE": null, "OMP_NUM_THREADS": null, "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_ALLOC_CONF": null, "PYTORCH_CUDA_ALLOC_CONF": null, "PYTORCH_NO_CUDA_MEMORY_CACHING": null, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": null, "TORCH_BLAS_PREFER_CUBLASLT": null, "TORCH_CUBLASLT_UNIFIED_WORKSPACE": null, "TORCH_CUDNN_V8_API_DISABLED": null, "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": null}`. Runtime settings unchanged: True.

The pinned [CublasHandlePool.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cuda/CublasHandlePool.cpp) retains workspaces keyed by handle and stream and clears the workspace maps through the tested API. Its open-source CUDA path defaults to unified cuBLAS/cuBLASLt workspace storage when TORCH_CUBLASLT_UNIFIED_WORKSPACE is unset. The measured allocation delta does not count owners or establish two independent 32 MiB pools. Even an exact match to the prior 64 MiB residual cannot prove the parent cause.

Preservation review: passed. Exact parent seal: `5a2be8b083356fe6f0e764805d5a5f8da9cbc4781e46c8bac3b7d5ec41a0f47f` (83,772 bytes). All 372 parent file metadata entries match; 18 used small source/evidence files were rehashed. The 39-file dirty-main boundary, HEAD, status, and diff match. Metadata equality for unread payloads supplements the sealed hashes; it is not a new payload byte verification. Ancestor checkpoint/cache payloads were not rescanned.

Supplemental prior-seals-before.json and prior-seals-after.json freshly verify 11 existing seal identities and three bounded intermediary receipts. These are small evidence files; this is not a repeated full artifact scan.

Read-only before/after NVIDIA telemetry is saved in analysis/gpu-before.json and analysis/gpu-after.json. Final worker observations confirm no owned interpreter or launcher remains; desktop graphics contexts are reported separately by NVIDIA.

Post-child RTX 5090 telemetry: `NVIDIA GeForce RTX 5090, GPU-6d908ed7-d2e6-1c70-8a1e-28754a8cbfae, 596.21, 32607 MiB, 137 MiB, 32051 MiB, 0 %`.

Reproduction and audit: inputs/request.txt, inputs/config.json, analysis/commands.json, runtime-source-freeze.json, workspace-supervisor.json, results.json, phases/*.json, logs/workspace-child.log, analysis/focused-cpu-tests.json, analysis/independent-review.json, and analysis/report-review.json. The final-validation.json seal inventories every source/result/phase file with exact bytes and SHA-256. The exclusive attempt markers deliberately prevent replay in this directory.
