The independent final auditor is `analysis/audit_completed_activation.py`. It uses only the Python standard library and refuses to write a completed audit until preparation and activation supervisors both report successful completion and the child measurement seal exists.

Run after the coordinator confirms completion:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONDONTWRITEBYTECODE='1'
& D:/openbonsai/.venv/Scripts/python.exe -B analysis/audit_completed_activation.py --verify-parent-inventory
```

The audit checks the explicitly frozen source SHA `08830dd39e3275b8686a189ce9f385621baefe3ed0ddf532851b4033601e1f8c`, every frozen file and child-sealed result, original source and retained tensor schemas, all original tensor fingerprints before/after wrapping, all restored tensor fingerprints before/after measurement, 1,600 regenerated initial fingerprints, 1,000 learned FP32 tensors, 200 explicit sign arrays, constructor-storage and RNG receipts, and all five original/bypass/hard probe results against the authenticated parent references.

It independently verifies the two fixed windows, 16 records each, active token/element counts, first-window negative control, 12 native oracle checks, A–B–A and reverse-order receipts, signed decomposition closure, pooled sums, the fixed noise threshold and 12-of-16 selector. It recomputes each recorded conservative memory bound and the continuation budget estimate, then validates the parent/main preservation receipt and current main hashes/status. The optional flag performs another streaming verification of all 2,034 parent files (16,192,227,790 bytes).

Five small CPU tests passed, including refusal of an incomplete process, signed cross-term/count corruption, a forged memory-pass flag, nonfinite arithmetic and path escape. No actual final measurement audit has run yet.

Read-only source review of the frozen measurement driver and shared binding helpers found no substantive blocker. Discarded activations cannot be independently recomputed from scalar/checksum receipts; exact native-oracle comparisons remain authenticated execution evidence. Memory sampling is discrete and deadline checks cannot interrupt an individual CUDA kernel. The diagnostic does not establish quality recovery or authorize training.
