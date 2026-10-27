Integrated only activation_capture.py and activation_metrics.py from reviewed commit 419d20ef43a0576f2f182842b88fbdb3c995aea0. The statistics module is byte-identical. The collector calls the inherited conditional-only cache_window without the newer teacher_prefix keyword. All 25 inherited core Python files remain byte-identical to the sealed parent.

The exact production/test diff and both source hashes are in collector-integration.patch/.json. The compressed-embedding rejection fixture uses the existing rotated embedding class; no endpoint implementation was imported. The upstream random-model training probe was not imported or executed.

64 CPU tests passed in 25.34 seconds. These include native BF16/FP32 hybrid independent forward oracles, first-window negative control, repeat and reverse ordering, restoration of flags/hooks/modes/gradients/snapshots on each failure, finite padding guards, bounded FP64 scratch at widths 4096/12288, release of previous activation records, and both cross-term signs. No optimizer or training driver was invoked.

These are software tests. Real saved-state restoration, exact hard-FP32 probe parity, device memory and wall-time gates remain the orchestration driver responsibility.
