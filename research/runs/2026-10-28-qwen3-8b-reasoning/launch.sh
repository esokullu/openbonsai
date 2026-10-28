#!/bin/bash
# Qwen3-8B (dense adapter) with the reasoning recipe (examples/qwen3-8b-reasoning.yaml), then its BF16
# reasoning baseline and compare checks of the intermediate stages and the exported GGUF. One gpu.lock
# hold for all of it, taken after the queued qwen35-0.8b jobs (run + BF16 baseline). Every stage resumes,
# so after a crash launch this script again (detached: PowerShell Start-Process with Git bash).
R=D:/openbonsai/runs; L=$R/qwen3_8b.log; D=$R/2026-10-28-qwen3-8b-reasoning; LOCK=$R/gpu.lock
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $L; }
gpu_used() { nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'; }
WINPID=$(cat /proc/$$/winpid 2>/dev/null || echo $$)
HAVE=0
release() { if [ $HAVE = 1 ]; then rm -f $LOCK/owner.txt; rmdir $LOCK 2>/dev/null; log "released gpu.lock"; fi; }
trap release EXIT

log "launcher start (pid $WINPID)"
until grep -q '^DONE' $R/qwen3-8b/download.log 2>/dev/null; do sleep 30; done
if grep -qs qwen3-8b-session $LOCK/owner.txt; then
  # A restart (after a crash or a manual stop) keeps the lock this job still holds.
  echo "$WINPID qwen3-8b-session qwen3-8b $(date '+%Y-%m-%d %H:%M:%S')" > $LOCK/owner.txt; HAVE=1
  log "kept gpu.lock from the previous launch"
else
  # Let the qwen35-0.8b queue finish: the LAST line of its log is a final state for 10 minutes while no qwen35
  # job holds gpu.lock (a relaunch appends a new start line after an old FAILED). If the GPU sits idle and
  # unlocked for 10 minutes, that queue is gone anyway.
  final() { tail -1 $R/qwen35_08b.log 2>/dev/null | grep -qE 'bf16 baseline (done|FAILED)|run FAILED'; }
  stable=0; idle=0
  while [ $stable -lt 20 ] && [ $idle -lt 20 ]; do
    if final && ! grep -qs qwen35 $LOCK/owner.txt; then stable=$((stable + 1)); else stable=0; fi
    if [ "$(gpu_used)" -lt 2048 ] && [ ! -d $LOCK ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 30
  done
  log "done waiting for the qwen35-0.8b queue ($(tail -1 $R/qwen35_08b.log); idle checks $idle)"
  while true; do
    if [ "$(gpu_used)" -lt 2048 ] && mkdir $LOCK 2>/dev/null; then
      echo "$WINPID qwen3-8b-session qwen3-8b $(date '+%Y-%m-%d %H:%M:%S')" > $LOCK/owner.txt; HAVE=1; break
    fi
    sleep 15
  done
  log "acquired gpu.lock"
fi

cd /d/openbonsai/work/qwen3-8b-snapshot
export PYTHONPATH=D:/openbonsai/work/qwen3-8b-snapshot HF_HOME=$R/hf-cache HF_HUB_OFFLINE=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=4
PY=D:/openbonsai/.venv-fla/Scripts/python.exe
U=$R/2026-10-26-5090-ultrachat-attempt2/source/data/train_sft-00000-of-00003-a3ecf92756993583.parquet
GGUF=$D/export/OpenBonsai-Qwen3-8B-PTQ1_0.gguf

log "run start"
$PY -m openbonsai run --config $D/config.yaml --out $D --set model.local_files_only=true \
  --set traces.sources.4.path=$U --set calibrate.sources.1.path=$U --set distill.teacher.sources.1.path=$U \
  >> $R/qwen3_8b.stdout 2>&1 && log "run done" || { log "run FAILED"; exit 1; }
log "bf16 baseline start"
$PY -m openbonsai eval --run $D --from bf16 --suite gsm8k,humaneval >> $R/qwen3_8b.stdout 2>&1 \
  && log "bf16 baseline done" || log "bf16 baseline FAILED"
for src in quantize endpoints $GGUF; do
  $PY -m openbonsai eval --run $D --from $src --suite compare >> $R/qwen3_8b.stdout 2>&1 \
    && log "compare $(basename $src) done" || log "compare $(basename $src) FAILED"
done
log "all done"
