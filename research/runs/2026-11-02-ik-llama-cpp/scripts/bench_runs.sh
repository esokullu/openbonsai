#!/usr/bin/env bash
# llama-bench (pp512, tg128) of the Hugging Face GGUF in ik_llama.cpp:
# 4 threads on P-cores 0-3 (mask 0x55) and 8 threads on P-cores 0-7 (mask 0x5555), one thread per core,
# without and with run-time repacking (-rtr 1). GPUs hidden.
set -u
SP='C:\Users\esoku\AppData\Local\Temp\claude\D--openbonsai\14b566d0-5e53-4238-9262-1246932509e5\scratchpad'
RUN="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SP\\scripts\\run_pinned.ps1"
OUT="$SP\\results"
IK="$SP\\ik\\build-cpu\\bin"
A1='D:/openbonsai/runs/2026-10-28-code-gap-1epoch/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf'
SUMMARY=/c/Users/esoku/AppData/Local/Temp/claude/D--openbonsai/14b566d0-5e53-4238-9262-1246932509e5/scratchpad/results/summary.txt

run() {  # name mask args...
  local name=$1 mask=$2; shift 2
  local line
  echo "$(date '+%F %T') start $name: llama-bench $*" >> "$SUMMARY"
  line=$($RUN -Exe "$IK\\llama-bench.exe" -Mask $mask -Out "$OUT\\$name.out" -Err "$OUT\\$name.log" -ArgLine "$*" | tr -d '\r')
  echo "$(date '+%F %T') done  $name: $line" >> "$SUMMARY"
}

# The default (non-repacked) path generates at well under 1 token/s here, so it gets one repetition.
BENCH="-m $A1 -p 512 -n 128 -o md"
run bench_t4_rtr  0x55   $BENCH -r 3 -t 4 -rtr 1
run bench_t8_rtr  0x5555 $BENCH -r 3 -t 8 -rtr 1
run bench_t8      0x5555 $BENCH -r 1 -t 8 -rtr 0
run bench_t4      0x55   $BENCH -r 1 -t 4 -rtr 0
echo "$(date '+%F %T') BENCH_DONE" >> "$SUMMARY"
