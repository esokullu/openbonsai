#!/bin/bash
# Score the 32-layer refine state before distillation (dev, compare, GSM8K 50 thinking off), as the pilot arms were scored.
R=D:/openbonsai/runs; D=$R/2026-10-28-reasoning-mix-refine32; L=$R/refine_full.log
log() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd /d/openbonsai/work/refine-pilot-snapshot
export PYTHONPATH=D:/openbonsai/work/refine-pilot-snapshot HF_HOME=$R/hf-cache HF_HUB_OFFLINE=1
log "pre-distill eval start"
bash $R/gpu_job.sh refine32-predistill D:/openbonsai/.venv-fla/Scripts/python.exe -m openbonsai eval --run $D --from refine \
  --suite dev,compare,gsm8k --set model.local_files_only=true --set 'eval.reasoning.thinking=[off]' --set eval.reasoning.gsm8k.n=50 \
  > $R/refine32-predistill.stdout 2>&1 && log "pre-distill eval done" || log "pre-distill eval FAILED"
