#!/bin/bash
# The two ablations of the code-gap run: which change moved HumanEval, the data or the second epoch?
#   code-gap-1epoch       the code-gap run's stages (traces, calibration, quantize, endpoints, teacher), 1 epoch
#   reasoning-mix-2epoch  the published run's stages and teacher (reasoning-mix-v1), 2 epochs
# Each copies the source run's finished stages up to the distillation teacher into a new run dir (the
# sources are only read), so only distillation training, export and eval run. Each takes the shared
# GPU through runs/gpu_job.sh (a mkdir lock) and releases it between the two runs.
#
# As run on 2026-10-28 (the scratchpad paths replaced by variables). SNAPSHOT is a checkout of a67e004,
# the package that ran both the published and the code-gap runs:
#   SNAPSHOT=/path/to/a67e004 PY=.venv/Scripts/python.exe bash research/runs/2026-10-28-humaneval-gap/ablations.sh
# The original also waited for another job queue before taking the lock; that wait is left out.
set -u
R=$(pwd)/runs                            # the source runs and gpu_job.sh (run from the repo root)
W=${W:-$R}                               # where the ablation runs go (absolute)
S=${SNAPSHOT:?a checkout of a67e004}
PY=${PY:?python with openbonsai and torch}
L=$W/ablations.log
mkdir -p $W
log() { echo "$(date +%H:%M:%S) $*" >> $L; }

seed() {   # seed <src run> <dest run>: copy the finished stages up to the distillation teacher
  local SRC=$1 D=$2
  mkdir -p $D/traces $D/quantize $D/eval $D/distill
  cp $SRC/run.yaml $SRC/signs.safetensors $D/
  find $SRC/traces -maxdepth 1 -type f -exec cp {} $D/traces/ \;
  cp -r $SRC/calib $SRC/endpoints $D/
  cp $SRC/quantize/*.json $SRC/quantize/state.safetensors $D/quantize/
  cp -r $SRC/distill/teacher $D/distill/ && cp $SRC/distill/TEACHER_INPUTS.json $D/distill/
  cp -r $SRC/eval/compare-reference-* $D/eval/ 2>/dev/null
  log "seeded $D from $SRC"
}

run() {    # run <label> <dest run> <epochs>: resume from the copied run.yaml with one value changed
  local LABEL=$1 D=$2 EPOCHS=$3
  log "$LABEL start (distill.epochs=$EPOCHS)"
  (cd $S && PYTHONPATH=$S bash $R/gpu_job.sh $LABEL $PY -m openbonsai run --out $D \
    --set distill.epochs=$EPOCHS) > $D.stdout 2>&1 && log "$LABEL done" || log "$LABEL FAILED"
}

A1=$W/2026-10-28-code-gap-1epoch
A2=$W/2026-10-28-reasoning-mix-2epoch
[ -f $A1/run.yaml ] || seed $R/2026-10-28-code-gap $A1
[ -f $A2/run.yaml ] || seed $R/2026-10-27-reasoning-mix-v1 $A2
[ "${1:-}" = seed ] && exit 0
run code-gap-1epoch $A1 1
run reasoning-mix-2epoch $A2 2
log "all done"
