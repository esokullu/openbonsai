#!/bin/bash
# CPU-only checks of the exported Qwen3-8B GGUF in Prism's runtime (no gpu.lock: CUDA hidden). Resumable:
# each step is skipped when its output already exists.
R=D:/openbonsai/runs; D=$R/2026-10-28-qwen3-8b-reasoning; O=$R/qwen3-8b/prism; W=$R/qwen3-8b/wikitext
BIN=$R/third_party/prism-llama.cpp/build-cpu/bin; GGUF=$D/export/OpenBonsai-Qwen3-8B-PTQ1_0.gguf
REPO=D:/openbonsai/.claude/worktrees/vigorous-carson-a7d117
until [ -e $D/export/DONE.json ]; do sleep 30; done
export CUDA_VISIBLE_DEVICES=-1 HF_HOME=$R/hf-cache HF_HUB_OFFLINE=1 PYTHONUTF8=1 OMP_NUM_THREADS=8
for f in wiki.test.raw wiki.test.join.raw; do
  out=$O/prism_ppl_qwen3-8b_${f%.raw}.log
  [ -s $out ] && grep -q "Final estimate" $out && continue
  echo "$(date '+%H:%M:%S') perplexity $f"
  $BIN/llama-perplexity.exe -m $GGUF -f $W/$f -c 512 --chunks 8 -t 8 > $out 2>&1
  grep "Final estimate" $out
done
if [ ! -s $O/runtime_parity.json ]; then
  echo "$(date '+%H:%M:%S') runtime parity"
  PYTHONPATH=$REPO D:/openbonsai/.venv-fla/Scripts/python.exe $REPO/research/runs/2026-10-28-qwen3-8b-reasoning/runtime_parity.py \
    --run $D --bin $BIN --text $W/wiki.test.raw --out $O/runtime_parity.json > $O/runtime_parity.log 2>&1
  tail -5 $O/runtime_parity.log
fi
echo "$(date '+%H:%M:%S') cpu checks done"
