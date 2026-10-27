# Head rotation preflight commands

All commands use the local machine and `D:/openbonsai/.venv/Scripts/python.exe`; model execution is exclusively through the supervisor. The completed C/B/ancestor runs and dirty main remain unchanged.

The isolated checkout was initialized and fetched at the exact reviewed commit:

```powershell
git init runs/2026-10-27-5090-head-rotation-preflight-v1/checkout
git -C runs/2026-10-27-5090-head-rotation-preflight-v1/checkout remote add origin https://github.com/esokullu/openbonsai.git
git -C runs/2026-10-27-5090-head-rotation-preflight-v1/checkout fetch --depth 1 origin 77e150e1edf974d753cc8fbdcfabd0332424fa20
git -C runs/2026-10-27-5090-head-rotation-preflight-v1/checkout checkout --detach FETCH_HEAD
```

Exact CPU test commands, environment overrides and logs are bound by `analysis/focused-cpu-tests.json` and its component receipts. CPU input attribution copied historical bytes and verified the official materialized source, without loading a model. No new generation or historical prepare stage is claimed.

After the CPU source review and environment receipt, freeze and launch once:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe -B runs/2026-10-27-5090-head-rotation-preflight-v1/tools/freeze_head.py --freeze
D:/openbonsai/.venv/Scripts/python.exe -B D:/openbonsai/runs/2026-10-27-5090-head-rotation-preflight-v1/tools/head_supervise.py
```

The supervisor records the exact child command, absolute deadline, environment, PIDs, elapsed duration and exit state. Its exclusive claim prevents retries. It allows one model process and zero scientific fitting. The child uses reviewed low-level original-head cache and KL APIs, with the fixed configuration in `inputs/config.json`.

After the supervisor and child exit, the CPU-only `tools/final_preserve.py` checks frozen sources, used lineage, original input attribution, dirty main and live worker absence. Saved-scalar analysis, report and final inventory sealing occur afterward; no model process is launched by those operations.
