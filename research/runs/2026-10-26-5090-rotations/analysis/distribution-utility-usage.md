The stdlib-only `../compare_distributions.py` passed eight tests and validated the unrotated JSON across 200 projections, ten projection families and 32 hidden-state layers. Both conditions have now completed, and the final paired report was generated successfully with all strict matching checks passing.

After both conditions have their final completion markers, run:

```powershell
& 'D:\openbonsai\.venv\Scripts\python.exe' 'D:\openbonsai\runs\2026-10-26-5090-rotations\compare_distributions.py'
```

This writes `analysis/distribution-comparison.json` and `.md`. The utility rejects incomplete pairs or mismatched source/training provenance, row selections, original weight/teacher-input statistics, and original hidden-probe references. It reads only saved JSON and has no watcher.

The JSON includes sampled weight kurtosis/outlier fractions before and after correction, full summary ranges, projection-family medians, teacher inputs before/after the runtime rotation, all 32 hidden-state drift records, and descriptive linear-attention/full-attention block summaries. The Markdown shows compact tables. Statistics retain their original/rotated coordinate labels; correction RMS ratios stay within a condition's same basis. Projection-family hidden drift and trained student activation distributions were not captured and are explicitly unavailable. All aggregation is over per-module sample statistics, without treating those medians as pooled moments or claiming a causal quality explanation.
