#!/bin/bash
# Confirming eval of P (published reasoning-mix-v1) vs A1 (code-gap, 1 epoch) on fresh problems, in copied run dirs.
# Waits for the frozen pre-registration (PREREGISTRATION.sha256) and then for the shared gpu.lock; holds the lock
# for all six evals (another session takes it within a second of any release). Each eval is its own process, so a
# failure is logged and the next one still runs; cached generation batches make a rerun cheap.
C=D:/openbonsai/runs/code-gap/confirm; L=$C/confirm_eval.log; LOCK=D:/openbonsai/runs/gpu.lock
PY=D:/openbonsai/.venv-fla/Scripts/python.exe
log() { echo "$(date '+%F %T') $*" >> $L; }
WINPID=$(cat /proc/$$/winpid 2>/dev/null || echo $$)
HAVE=0
release() { if [ $HAVE = 1 ]; then rm -f $LOCK/owner.txt; rmdir $LOCK 2>/dev/null; log "released gpu.lock"; fi; }
trap release EXIT
log "launcher start (winpid $WINPID); waiting for PREREGISTRATION.sha256"
until [ -s $C/PREREGISTRATION.sha256 ]; do sleep 30; done
log "pre-registration present: $(cat $C/PREREGISTRATION.sha256)"
while true; do
  used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  if [ "$used" -lt 2048 ] && mkdir $LOCK 2>/dev/null; then
    echo "$WINPID a378e74a-openbonsai confirm-eval $(date '+%Y-%m-%d %H:%M:%S')" > $LOCK/owner.txt; HAVE=1; break
  fi
  sleep 15
done
log "acquired gpu.lock"
export PYTHONPATH=D:/openbonsai/work/code-gap-snapshot HF_HOME=D:/openbonsai/runs/hf-cache HF_HUB_OFFLINE=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=4
cd /d/openbonsai/work/code-gap-snapshot
mkdir -p $C/results
one() {  # one <model> <name> <suite> <mode> <sets...>
  local m=$1 name=$2 suite=$3 mode=$4; shift 4
  local R=$C/runs/$m
  log "start $m $name"
  if $PY -m openbonsai eval --run $R --from distill --suite $suite --set "eval.reasoning.thinking=[$mode]" "$@" \
       >> $C/eval.$m.$name.stdout 2>&1; then
    cp $R/eval/distill.$suite.think-$mode.json $C/results/$m.$name.json && log "done $m $name"
  else
    log "FAILED $m $name (exit $?)"
  fi
}
one P  humaneval164.think-off humaneval off --set eval.reasoning.humaneval.n=164
one A1 humaneval164.think-off humaneval off --set eval.reasoning.humaneval.n=164
one P  humaneval75.think-on   humaneval on  --set eval.reasoning.humaneval.n_thinking=75
one A1 humaneval75.think-on   humaneval on  --set eval.reasoning.humaneval.n_thinking=75
one P  gsm8k1319.think-off    gsm8k     off --set eval.reasoning.gsm8k.n=1319
one A1 gsm8k1319.think-off    gsm8k     off --set eval.reasoning.gsm8k.n=1319
log "all done"
