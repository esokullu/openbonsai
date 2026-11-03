#!/usr/bin/env bash
# After the perplexity batch: llama-bench, then add llama-server to the ik build, then the greedy samples. Strictly sequential.
set -u
SP='C:\Users\esoku\AppData\Local\Temp\claude\D--openbonsai\14b566d0-5e53-4238-9262-1246932509e5\scratchpad'
SPU=/c/Users/esoku/AppData/Local/Temp/claude/D--openbonsai/14b566d0-5e53-4238-9262-1246932509e5/scratchpad
SUMMARY=$SPU/results/summary.txt
bash $SPU/scripts/bench_runs.sh
echo "$(date '+%F %T') start build llama-server" >> "$SUMMARY"
cmd.exe //c "$SP\\scripts\\build_ik_server.bat" "$SP\\ik\\build-cpu" > $SPU/ik/build-server.log 2>&1
echo "$(date '+%F %T') done  build_server: rc=$? $(tail -1 $SPU/ik/build-server.log | tr -d '\r')" >> "$SUMMARY"
[ -f $SPU/ik/build-cpu/bin/llama-server.exe ] && bash $SPU/scripts/samples_runs.sh
echo "$(date '+%F %T') AFTER_PPL_DONE" >> "$SUMMARY"
