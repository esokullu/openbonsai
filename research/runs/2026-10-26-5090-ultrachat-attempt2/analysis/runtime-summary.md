# Measured paired-run time and CUDA memory

Total measured driver time: **124.86 minutes** across four successful processes.

Shared AYOT generation process: **36.98 minutes** (timed load/generation: 36.96); generation plus both arms: **161.83 minutes**. Generation is counted once.

| Condition | Process/scope | Phase | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---:|---:|---:|
| rotated16x8 | real | load_original_bf16 | 15.017 | 17.538 | 17.588 |
| rotated16x8 | real | baseline_probe_and_hidden_reference | 0.548 | 17.705 | 17.812 |
| rotated16x8 | real | prepare_original_export_plan | 24.791 | 17.569 | 17.619 |
| rotated16x8 | real | conversion_rank64 | 3.199 | 19.366 | 19.412 |
| rotated16x8 | real | initial_state_fingerprints | 1.363 | 19.183 | 19.207 |
| rotated16x8 | real | pretraining_distributions | 1.502 | 19.307 | 19.354 |
| rotated16x8 | real | training | 1303.809 | 20.221 | 20.297 |
| rotated16x8 | real | save_research_state_and_distributions | 3.021 | 19.219 | 19.244 |
| rotated16x8 | real | construct_hard_candidates | 596.268 | 19.434 | 19.492 |
| rotated16x8 | real | save_mixed_artifact | 135.856 | 19.214 | 19.238 |
| rotated16x8 | real | verify_export_and_install_reference | 88.080 | 19.218 | 19.240 |
| rotated16x8 | real | export-reference_decoded_cache_preparation | 26.057 | 19.168 | 19.266 |
| rotated16x8 | real | export-reference_full_nll_shape_memory_gate | 0.761 | 19.497 | 19.811 |
| rotated16x8 | real | verify_cached_reference_exact | 0.671 | 19.134 | 19.215 |
| rotated16x8 | fresh reload | load_mixed_without_decoded_cache_for_profiling | 69.893 | 6.168 | 6.254 |
| rotated16x8 | fresh reload | fresh-reload_decoded_cache_preparation | 26.691 | 19.099 | 19.221 |
| rotated16x8 | fresh reload | fresh-reload_full_nll_shape_memory_gate | 1.395 | 19.459 | 19.826 |
| rotated16x8 | fresh reload | exact_reference_parity | 0.667 | 19.096 | 19.207 |
| rotated16x8 | within training | cache | 323.431 | 19.533 | 20.297 |
| rotated16x8 | within training | hard_evaluation_before | 27.480 | 19.529 | 20.297 |
| rotated16x8 | within training | audit_boundary_before | 0.454 | 19.564 | 20.297 |
| rotated16x8 | within training | optimization | 926.978 | 20.221 | 20.297 |
| rotated16x8 | within training | freeze_window | 0.007 | 19.383 | 20.297 |
| rotated16x8 | within training | hard_evaluation_after | 24.421 | 19.605 | 20.297 |
| rotated16x8 | within training | audit_boundary_after | 0.424 | 19.641 | 20.297 |
| rotated16x8 | within training | cleanup | 0.510 | 19.297 | 20.297 |
| rotated16x8 | evaluation | hard_fp32 | 37.240 | 19.599 | 19.869 |
| rotated16x8 | evaluation | fp16_same_wrapper | 37.677 | 19.599 | 19.869 |
| rotated16x8 | evaluation | packed_fresh_reload | 21.618 | 19.459 | 19.826 |
| rotated128x1 | real | load_original_bf16 | 16.978 | 17.538 | 17.588 |
| rotated128x1 | real | baseline_probe_and_hidden_reference | 0.556 | 17.705 | 17.812 |
| rotated128x1 | real | prepare_original_export_plan | 24.979 | 17.569 | 17.619 |
| rotated128x1 | real | conversion_rank64 | 3.625 | 19.366 | 19.412 |
| rotated128x1 | real | initial_state_fingerprints | 1.428 | 19.183 | 19.207 |
| rotated128x1 | real | pretraining_distributions | 1.618 | 19.307 | 19.354 |
| rotated128x1 | real | training | 3965.794 | 20.219 | 20.297 |
| rotated128x1 | real | save_research_state_and_distributions | 3.081 | 19.219 | 19.244 |
| rotated128x1 | real | construct_hard_candidates | 576.950 | 19.434 | 19.492 |
| rotated128x1 | real | save_mixed_artifact | 133.677 | 19.214 | 19.238 |
| rotated128x1 | real | verify_export_and_install_reference | 88.328 | 19.218 | 19.240 |
| rotated128x1 | real | export-reference_decoded_cache_preparation | 25.659 | 19.168 | 19.266 |
| rotated128x1 | real | export-reference_full_nll_shape_memory_gate | 0.755 | 19.497 | 19.811 |
| rotated128x1 | real | verify_cached_reference_exact | 0.686 | 19.134 | 19.215 |
| rotated128x1 | fresh reload | load_mixed_without_decoded_cache_for_profiling | 68.238 | 6.168 | 6.273 |
| rotated128x1 | fresh reload | fresh-reload_decoded_cache_preparation | 26.588 | 19.100 | 19.240 |
| rotated128x1 | fresh reload | fresh-reload_full_nll_shape_memory_gate | 1.200 | 19.460 | 19.846 |
| rotated128x1 | fresh reload | exact_reference_parity | 0.663 | 19.096 | 19.227 |
| rotated128x1 | within training | cache | 2604.822 | 19.533 | 20.297 |
| rotated128x1 | within training | hard_evaluation_before | 223.278 | 19.528 | 20.297 |
| rotated128x1 | within training | audit_boundary_before | 0.471 | 19.563 | 20.297 |
| rotated128x1 | within training | optimization | 935.228 | 20.219 | 20.297 |
| rotated128x1 | within training | freeze_window | 0.007 | 19.380 | 20.297 |
| rotated128x1 | within training | hard_evaluation_after | 197.007 | 19.605 | 20.297 |
| rotated128x1 | within training | audit_boundary_after | 0.441 | 19.641 | 20.297 |
| rotated128x1 | within training | cleanup | 4.436 | 19.297 | 20.297 |
| rotated128x1 | evaluation | hard_fp32 | 34.019 | 19.599 | 19.869 |
| rotated128x1 | evaluation | fp16_same_wrapper | 34.256 | 19.599 | 19.869 |
| rotated128x1 | evaluation | packed_fresh_reload | 18.230 | 19.460 | 19.846 |

Cautions:

- Nested trainer phases overlap the enclosing training phase; do not sum all table rows.
- Console enclosing CUDA counters were reset by subphases; metrics.json uses the corrected maximum across synchronized trainer phases.
- Generation lengths/stopping may differ by representation. Recorded token counts accompany throughput; total durations are not equal-token kernel benchmarks.
- Cached packed inference includes 13,824,425,984 nonpersistent decoded BF16 bytes, absent from serialized artifact.
- CPU hashing and final reporting occur after measured GPU execution and are excluded.
- Process totals include setup, probes, Python work and other gaps not individually timed. Preflights and earlier runs are excluded.
- Shared original-model generation of128 UltraChat traces is counted once at run scope; the frozen nested16 are reused by the repeated-sample arm.
- The90-minute generation deadline is checked between token steps/records. Loading or an individual CUDA call cannot be interrupted exactly at that deadline.
