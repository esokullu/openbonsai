# Corrected compression: small-pilot results

The corrected 16×1 pipeline completed with finite training and exact fresh-reload parity. Its packed diagnostic NLL is 11.191338, 2.066354 nats below the original failed pilot, but ARC is 1/20. All four answers consist of 128 repetitions of “box” and hit the cap without EOS: **0/4 answers are task-relevant or usable. No useful quality recovery was demonstrated.**

All reported real-model experiments start from the pinned official BF16 model. No task fine-tuning, packed source, or least-squares-recovered source is used. The table uses the fixed **2,048-token / 20-question / four-generation diagnostic subset**; it is not the complete pilot evaluation.

| State | NLL | ΔNLL vs BF16 | Perplexity | ARC correct | ΔARC | Unfinished |
|---|---:|---:|---:|---:|---:|---:|
| Original BF16 | 2.423777 | +0.000000 | 11.288413 | 19/20 | +0.0 pp | 2/4 |
| Original failed pilot | 13.257692 | +10.833916 | 572,456.680041 | 1/20 | -90.0 pp | 4/4 |
| Corrected 16×1 packed | 11.191338 | +8.767561 | 72,499.732925 | 1/20 | -90.0 pp | 4/4 |

Saved model artifact: **6,561,490,248 bytes**; tokenizer: 19,998,761; deployable total: **6,581,489,009 bytes**. The packed diagnostic evaluation took 419.03 seconds with 6.560/6.840 GiB peak allocated/reserved VRAM, excluding loading. The real-run driver completed in 1617.238147 seconds (26.95 minutes), including its training/export/evaluation, reload, and size-check subprocesses. Research-state checkpoints are excluded from deployable size; file size does not imply inference speedup.

Bounded 16×8 launched: **False**. Full frozen evaluation launched: **False**. Conditional 128×3 launched: **False**. Recorded decision: The latest delegated user instruction superseded automatic escalation and required stopping after the active corrected16x1 run. That run completed with finite updates and exact saved/reloaded parity, but did not recover useful quality. No unstarted training or search stage was launched.

The latest instruction supersedes the earlier expansion plan: finish the active 16×1 boundary and stop. Unrun experiments and gates are not reported as failures.

The recorded decision includes manual assessment of actual outputs. Successful serialization is distinct from preserved BF16 quality.

Detailed evidence: [report.md](report.md). Raw side-by-side outputs: [generations.md](generations.md).

