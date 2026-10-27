# Measured paired-run time and CUDA memory

Total measured driver time: **104.55 minutes** across four successful processes.

| Condition | Process/scope | Phase | Seconds | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---:|---:|---:|
| unrotated | real | load_original_bf16 | 15.625 | 17.538 | 17.588 |
| unrotated | real | baseline_probe_and_hidden_reference | 0.566 | 17.705 | 17.812 |
| unrotated | real | prepare_original_export_plan | 24.335 | 17.569 | 17.619 |
| unrotated | real | conversion_rank64 | 1.123 | 19.271 | 19.334 |
| unrotated | real | pretraining_distributions | 1.223 | 19.309 | 19.348 |
| unrotated | real | training | 731.303 | 20.207 | 20.297 |
| unrotated | real | save_research_state_and_distributions | 2.826 | 19.217 | 19.258 |
| unrotated | real | construct_hard_candidates | 441.147 | 19.401 | 19.510 |
| unrotated | real | save_mixed_artifact | 141.338 | 19.213 | 19.254 |
| unrotated | real | verify_export_and_install_reference | 90.323 | 19.216 | 19.254 |
| unrotated | real | export-reference_decoded_cache_preparation | 25.803 | 19.162 | 19.262 |
| unrotated | real | export-reference_full_nll_shape_memory_gate | 0.599 | 19.491 | 19.750 |
| unrotated | real | verify_cached_reference_exact | 0.522 | 19.129 | 19.203 |
| unrotated | fresh reload | load_mixed_without_decoded_cache_for_profiling | 70.529 | 6.162 | 6.273 |
| unrotated | fresh reload | fresh-reload_decoded_cache_preparation | 26.685 | 19.093 | 19.221 |
| unrotated | fresh reload | fresh-reload_full_nll_shape_memory_gate | 0.949 | 19.454 | 19.750 |
| unrotated | fresh reload | exact_reference_parity | 0.520 | 19.090 | 19.207 |
| unrotated | within training | cache | 116.959 | 19.520 | 20.297 |
| unrotated | within training | hard_evaluation_before | 14.478 | 19.515 | 20.297 |
| unrotated | within training | audit_boundary_before | 0.312 | 19.468 | 20.297 |
| unrotated | within training | optimization | 586.979 | 20.207 | 20.297 |
| unrotated | within training | freeze_window | 0.008 | 19.375 | 20.297 |
| unrotated | within training | hard_evaluation_after | 11.719 | 19.591 | 20.297 |
| unrotated | within training | audit_boundary_after | 0.277 | 19.546 | 20.297 |
| unrotated | within training | cleanup | 0.465 | 19.295 | 20.297 |
| unrotated | evaluation | hard_fp32 | 213.287 | 19.598 | 19.875 |
| unrotated | evaluation | fp16_same_wrapper | 217.598 | 19.598 | 19.875 |
| unrotated | evaluation | packed_fresh_reload | 284.163 | 19.454 | 19.750 |
| rotated | real | load_original_bf16 | 17.890 | 17.538 | 17.588 |
| rotated | real | baseline_probe_and_hidden_reference | 0.565 | 17.705 | 17.812 |
| rotated | real | prepare_original_export_plan | 25.654 | 17.569 | 17.619 |
| rotated | real | conversion_rank64 | 3.988 | 19.366 | 19.412 |
| rotated | real | pretraining_distributions | 1.601 | 19.311 | 19.354 |
| rotated | real | training | 1309.048 | 20.221 | 20.297 |
| rotated | real | save_research_state_and_distributions | 3.041 | 19.219 | 19.244 |
| rotated | real | construct_hard_candidates | 584.969 | 19.434 | 19.492 |
| rotated | real | save_mixed_artifact | 132.435 | 19.214 | 19.238 |
| rotated | real | verify_export_and_install_reference | 84.218 | 19.218 | 19.240 |
| rotated | real | export-reference_decoded_cache_preparation | 25.751 | 19.168 | 19.266 |
| rotated | real | export-reference_full_nll_shape_memory_gate | 0.754 | 19.497 | 19.811 |
| rotated | real | verify_cached_reference_exact | 0.677 | 19.134 | 19.215 |
| rotated | fresh reload | load_mixed_without_decoded_cache_for_profiling | 67.280 | 6.166 | 6.273 |
| rotated | fresh reload | fresh-reload_decoded_cache_preparation | 26.096 | 19.097 | 19.240 |
| rotated | fresh reload | fresh-reload_full_nll_shape_memory_gate | 1.259 | 19.457 | 19.846 |
| rotated | fresh reload | exact_reference_parity | 0.712 | 19.094 | 19.227 |
| rotated | within training | cache | 324.059 | 19.533 | 20.297 |
| rotated | within training | hard_evaluation_before | 28.104 | 19.529 | 20.297 |
| rotated | within training | audit_boundary_before | 0.441 | 19.564 | 20.297 |
| rotated | within training | optimization | 930.732 | 20.221 | 20.297 |
| rotated | within training | freeze_window | 0.008 | 19.382 | 20.297 |
| rotated | within training | hard_evaluation_after | 24.643 | 19.605 | 20.297 |
| rotated | within training | audit_boundary_after | 0.406 | 19.642 | 20.297 |
| rotated | within training | cleanup | 0.556 | 19.298 | 20.297 |
| rotated | evaluation | hard_fp32 | 641.245 | 19.599 | 19.869 |
| rotated | evaluation | fp16_same_wrapper | 643.455 | 19.599 | 19.869 |
| rotated | evaluation | packed_fresh_reload | 340.528 | 19.457 | 19.846 |

Cautions:

- Nested trainer phases overlap the enclosing training phase; do not sum all table rows.
- Console enclosing CUDA counters were reset by subphases; metrics.json uses the corrected maximum across synchronized trainer phases.
- Baseline stops after 341 generated tokens; compressed stages produce 512. Total evaluation durations are not equal-token kernel benchmarks.
- Cached packed inference includes 13,824,425,984 nonpersistent decoded BF16 bytes, absent from serialized artifact.
- CPU hashing and final reporting occur after measured GPU execution and are excluded.
- Process totals include setup, probes, Python work and other gaps not individually timed. Preflights and earlier runs are excluded.
