#!/usr/bin/env bash
# Greedy single-turn replies from Prism's llama-cli (its build has no llama-server), thinking off, 4 threads on P-cores 0-3.
# Same questions and token limits as greedy_client.py's chat prompts.
set -u
SP='C:\Users\esoku\AppData\Local\Temp\claude\D--openbonsai\14b566d0-5e53-4238-9262-1246932509e5\scratchpad'
SPU=/c/Users/esoku/AppData/Local/Temp/claude/D--openbonsai/14b566d0-5e53-4238-9262-1246932509e5/scratchpad
RUN="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SP\\scripts\\run_pinned.ps1"
OUT="$SP\\results"
PR='D:\openbonsai\runs\third_party\prism-llama.cpp\build-cpu\bin'
A1='D:/openbonsai/runs/2026-10-28-code-gap-1epoch/export/OpenBonsai-Qwen3.5-9B-PTQ1_0.gguf'
SUMMARY=$SPU/results/summary.txt
COMMON="-m $A1 -t 4 -c 4096 -st --reasoning off --temp 0 --top-k 1 --verbose-prompt --simple-io"

run() {  # name n_predict question
  local name=$1 n=$2 q=$3 line
  echo "$(date '+%F %T') start $name: llama-cli $COMMON -n $n -p \"$q\"" >> "$SUMMARY"
  line=$($RUN -Exe "$PR\\llama-cli.exe" -Mask 0x55 -Out "$OUT\\$name.out" -Err "$OUT\\$name.log" -ArgLine "$COMMON -n $n -p \"$q\"" | tr -d '\r')
  echo "$(date '+%F %T') done  $name: $line" >> "$SUMMARY"
}

run prism_cli_arith        16  "What is 17 * 24? Reply with just the number."
run prism_cli_word_problem 128 "A baker makes 48 cupcakes in the morning and half as many in the afternoon. How many cupcakes does the baker make in total? Answer briefly."
run prism_cli_code         128 "Write a Python function fib(n) that returns the n-th Fibonacci number, with fib(0) = 0."
echo "$(date '+%F %T') PRISM_CLI_DONE" >> "$SUMMARY"
