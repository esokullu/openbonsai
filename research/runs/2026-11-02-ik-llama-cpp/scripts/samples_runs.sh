#!/usr/bin/env bash
# Greedy samples of the Hugging Face GGUF: ik_llama.cpp (-rtr 1) and Prism's llama.cpp, same raw prompts, 8 threads on P-cores.
set -u
SP='C:\Users\esoku\AppData\Local\Temp\claude\D--openbonsai\14b566d0-5e53-4238-9262-1246932509e5\scratchpad'
SPU=/c/Users/esoku/AppData/Local/Temp/claude/D--openbonsai/14b566d0-5e53-4238-9262-1246932509e5/scratchpad
SQ="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SP\\scripts\\serve_and_query.ps1"
OUT="$SP\\results"
IK="$SP\\ik\\build-cpu\\bin"
PR='D:\openbonsai\runs\third_party\prism-llama.cpp\build-cpu\bin'
A1='D:/openbonsai/runs/2026-10-28-code-gap-1epoch/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf'
SUMMARY=$SPU/results/summary.txt

serve() {  # name exe-dir port args...
  local name=$1 dir=$2 port=$3; shift 3
  local line
  echo "$(date '+%F %T') start $name: llama-server $*" >> "$SUMMARY"
  line=$($SQ -Exe "$dir\\llama-server.exe" -Mask 0x55 -Port $port -OutPrefix "$OUT\\$name" -ArgLine "$*" | tr -d '\r' | tail -1)
  echo "$(date '+%F %T') done  $name: $line" >> "$SUMMARY"
}

# 4 threads on P-cores 0-3: the VM saturates the machine, and 8 threads were slower than 4 under that load.
serve samples_ik_rtr "$IK" 8091 -m $A1 -t 4 -c 4096 -rtr
serve samples_prism  "$PR" 8092 -m $A1 -t 4 -c 4096
echo "$(date '+%F %T') SAMPLES_DONE" >> "$SUMMARY"
