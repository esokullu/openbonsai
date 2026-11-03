#!/usr/bin/env bash
# Perplexity and KL checks of the OpenBonsai PTQ1_0 GGUFs in ik_llama.cpp and Prism's llama.cpp.
# Sequential, 8 threads pinned one per P-core (mask 0x5555), GPUs hidden.
set -u
SP='C:\Users\esoku\AppData\Local\Temp\claude\D--openbonsai\14b566d0-5e53-4238-9262-1246932509e5\scratchpad'
RUN="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SP\\scripts\\run_pinned.ps1"
OUT="$SP\\results"
IK="$SP\\ik\\build-cpu\\bin"
PR='D:\openbonsai\runs\third_party\prism-llama.cpp\build-cpu\bin'
A1='D:/openbonsai/runs/2026-10-28-code-gap-1epoch/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf'
SV1='D:/openbonsai/runs/2026-10-27-gguf-shared-v1/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf'
WIKI='D:/openbonsai/runs/qwen3-8b/wikitext/wiki.test.raw'
PLOGITS='D:/openbonsai/runs/2026-10-29-runtime-parity-9b/runtime_parity.prism-logits'
PPL="-c 512 --chunks 8 -b 512 -t 8"
SUMMARY=/c/Users/esoku/AppData/Local/Temp/claude/D--openbonsai/14b566d0-5e53-4238-9262-1246932509e5/scratchpad/results/summary.txt
mkdir -p /c/Users/esoku/AppData/Local/Temp/claude/D--openbonsai/14b566d0-5e53-4238-9262-1246932509e5/scratchpad/results

run() {  # name exe-dir exe envset args...
  local name=$1 dir=$2 exe=$3 envset=$4; shift 4
  local line
  echo "$(date '+%F %T') start $name: $exe $*" >> "$SUMMARY"
  line=$($RUN -Exe "$dir\\$exe" -Mask 0x5555 -Out "$OUT\\$name.out" -Err "$OUT\\$name.log" -ArgLine "$*" -EnvSet "$envset" | tr -d '\r')
  echo "$(date '+%F %T') done  $name: $line" >> "$SUMMARY"
}

run ik_ppl_a1          "$IK" llama-perplexity.exe "" -m $A1 -f $WIKI $PPL
run ik_ppl_a1_rtr      "$IK" llama-perplexity.exe "" -m $A1 -f $WIKI $PPL -rtr
run ik_kld_a1          "$IK" llama-perplexity.exe "" -m $A1 --kl-divergence-base $PLOGITS --kl-divergence $PPL
run ik_kld_a1_rtr      "$IK" llama-perplexity.exe "" -m $A1 --kl-divergence-base $PLOGITS --kl-divergence $PPL -rtr
run prism_ppl_a1       "$PR" llama-perplexity.exe "" -m $A1 -f $WIKI $PPL
run ik_ppl_sv1         "$IK" llama-perplexity.exe "" -m $SV1 -f $WIKI $PPL
run prism_ppl_sv1      "$PR" llama-perplexity.exe "" -m $SV1 -f $WIKI $PPL
run ik_ppl_a1_nohad    "$IK" llama-perplexity.exe "IK_HADAMARD_DISABLE=1" -m $A1 -f $WIKI $PPL
echo "$(date '+%F %T') ALL_DONE" >> "$SUMMARY"
