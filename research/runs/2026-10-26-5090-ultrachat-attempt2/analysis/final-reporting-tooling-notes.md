The final report and sealing helpers are prepared outside the frozen checkout. Neither has been executed. Their code was syntax-checked and independently reviewed for schema, accounting and quality-interpretation errors; no blocking issue was found. The copied artifact/training/quality/runtime tools remain unchanged.

`report_attempt2.py` requires both completed arm/reload processes and the final paired streamed artifact audit, paired training audit, quality summary, manual review, runtime summary, attempt-2 accounting, generation/calibration audit, both independent learned-state audits and final historical-preservation evidence. It validates frozen checkout/archive hashes, current source maps, the parent/historical anchor seals and referenced evidence hashes before creating `report.md`, `summary.md`, `reproduction.md` and `analysis/report-input-manifest.json`. Outputs must be new. Current partial training summaries cannot satisfy its completion gate; the final paired training audit must refresh them after both arms finish.

The report includes a three-way original BF16/new16×8/new128×1 overview with equivalent-scope tensor bytes, quality metrics, evaluation memory/time and generated-token counts. It explicitly distinguishes tensor storage from deployment folders, the decoded runtime cache from serialized bytes, and original historical evaluation timing from current measurements. All-A ARC collapse, EOS-only/fragments, FP16 rounding, packed row-GEMM differences and exact fresh reload are separate findings. Full candidate/container/retained/tokenizer/research components and measured phases are retained, with source/calibration/protocol hashes and recorded launch commands. Historical diagnosis/repack results are linked as context with distinct protocols.

Expected final inputs under `analysis`:

- `paired-artifact-audit.json` with full payload hash verification.
- `paired-training-audit.json` plus completed `{arm}-training-audit.json` files.
- `quality-summary.json`, `quality-generations.md`, `manual-quality-review.json` and `.md`.
- `runtime-summary.json`, `attempt2-accounting.json`, `generation-calibration-audit.json`.
- `{arm}-real-completed-training-audit.json` and `.md`, covering all saved factor/LoRA/sign tensors.
- `historical-preservation-final.json` and `.md`, and the preserved `rotated16x8-failure-analysis.json` and `.md`.

`seal_attempt2.py --seal` is a separate final action. It repeats completion/source/report checks, validates every report link and hashes all current-run evidence while detecting file/set changes. It compares the resulting artifact and learned-checkpoint hashes with their final audit identities. It excludes only its own output manifest (`final-validation.json`), Git internals and Python/test/tool caches; the helper's source is included. It records manual quality assessments independently from software/serialization success. It does not rehash every historical payload itself; the authenticated final preservation audit supplies that separately stated scope.

Invoke the sealer directly after review and explicit authorization, without redirecting output to a file in this run or creating a still-running command JSON inside it. Such concurrent writes deliberately fail its idle/stability checks. The manifest cannot include its own hash; the helper prints that hash after exclusive creation. No automatic follow-up experiment, GPU launch, upload or branch operation is implemented.
