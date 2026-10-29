#!/bin/bash
# Prism ML's Ternary Bonsai 8B, unpacked (F16 safetensors), scored as an ordinary Hugging Face model: the F16
# check (f16_check.py), then GSM8K and HumanEval through `openbonsai eval --from bf16` with the weights loaded
# in F16. One gpu.lock hold, taken after the download and any other holder. Every step resumes (eval caches
# each generated batch), so after a crash launch this script again (detached: PowerShell Start-Process with
# Git bash).
#
# Prism's chat template has no thinking switch: it always opens the answer with an empty <think></think>
# block, Qwen3's thinking-off prompt. So thinking off runs on the checkpoint as published (run/), and thinking
# on, which eval would otherwise render with that same empty block, runs only when thinking-on.ok exists:
# on model-qwen3-template/, Prism's config and weights (hard links) with Qwen3-8B's tokenizer files and chat
# template (run-qwen3-template/).
R=D:/openbonsai/runs; D=$R/2026-10-29-ternary-bonsai-8b-unpacked; L=$D/launcher.log; LOCK=$R/gpu.lock
REPO=prism-ml/Ternary-Bonsai-8B-unpacked; REV=ac20f03fc62e872399218b659c8e949dfca05769
TAG=eager-driscoll-6e0245; JOB=bonsai8b-unpacked-eval
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $L; }
gpu_used() { nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'; }
WINPID=$(cat /proc/$$/winpid 2>/dev/null || echo $$)
HAVE=0
release() { if [ $HAVE = 1 ]; then rm -f $LOCK/owner.txt; rmdir $LOCK 2>/dev/null; log "released gpu.lock"; fi; }
trap release EXIT

log "launcher start (pid $WINPID)"
until grep -q 'DONE$' $D/download.log 2>/dev/null; do sleep 30; done
if grep -qs " $TAG $JOB " $LOCK/owner.txt; then
  # A restart (after a crash or a manual stop) keeps the lock this job still holds.
  echo "$WINPID $TAG $JOB $(date '+%Y-%m-%d %H:%M:%S')" > $LOCK/owner.txt; HAVE=1
  log "kept gpu.lock from the previous launch"
else
  while true; do
    if [ "$(gpu_used)" -lt 2048 ] && mkdir $LOCK 2>/dev/null; then
      echo "$WINPID $TAG $JOB $(date '+%Y-%m-%d %H:%M:%S')" > $LOCK/owner.txt; HAVE=1; break
    fi
    sleep 15
  done
  log "acquired gpu.lock"
fi

export HF_HOME=$R/hf-cache HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 \
  OMP_NUM_THREADS=4
PY=D:/openbonsai/.venv-fla/Scripts/python.exe
SNAP=$R/hf-cache/hub/models--prism-ml--Ternary-Bonsai-8B-unpacked/snapshots/$REV
WIKI=$R/hf-cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet

if [ ! -f $D/f16_check.json ]; then
  log "f16 check start"
  $PY -I $D/f16_check.py $SNAP $WIKI $D/f16_check.json >> $D/f16_check.log 2>&1; rc=$?
  if [ $rc = 3 ]; then log "f16 check: non-finite values in F16; stopping before eval"; exit 1; fi
  if [ $rc != 0 ]; then log "f16 check FAILED (exit $rc)"; exit 1; fi
  log "f16 check done"
fi
grep -q '"f16_all_finite": true' $D/f16_check.json || { log "f16_check.json: not finite in F16; stopping"; exit 1; }

cd /d/openbonsai/work/qwen3-8b-snapshot      # the package at 8c150da, as the Qwen3-8B run
export PYTHONPATH=D:/openbonsai/work/qwen3-8b-snapshot
if [ -f $D/run/eval/bf16.gsm8k.think-off.json ] && [ -f $D/run/eval/bf16.humaneval.think-off.json ]; then
  log "thinking off: results already there; skipped"
else
  log "eval thinking off start (Prism's checkpoint and chat template as published)"
  $PY -m openbonsai eval --run $D/run --config $D/config.yaml --set model.path=$REPO --set model.revision=$REV \
    --set model.dtype=float16 --set model.local_files_only=true --set 'eval.reasoning.thinking=[off]' \
    --from bf16 --suite gsm8k,humaneval >> $D/eval.stdout 2>&1 && log "eval thinking off done" \
    || { log "eval thinking off FAILED"; exit 1; }
fi
if [ -f $D/thinking-on.ok ] && [ -f $D/model-qwen3-template/READY ]; then
  log "eval thinking on start (Prism's weights with Qwen3-8B's chat template)"
  $PY -m openbonsai eval --run $D/run-qwen3-template --config $D/config.yaml --set model.path=$D/model-qwen3-template \
    --set model.revision=null --set model.dtype=float16 --set model.local_files_only=true \
    --set 'eval.reasoning.thinking=[on]' --from bf16 --suite gsm8k,humaneval >> $D/eval.stdout 2>&1 \
    && log "eval thinking on done" || { log "eval thinking on FAILED"; exit 1; }
else
  log "thinking on: not approved or model-qwen3-template/ not ready; skipped"
fi
log "all done"
