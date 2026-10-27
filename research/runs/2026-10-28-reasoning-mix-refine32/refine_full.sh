#!/bin/bash
# Full refine (all 32 layers, target original) on the reasoning-mix model, then distill, export and the full evals.
R=D:/openbonsai/runs; L=$R/refine_full.log
log() { echo "$(date +%H:%M:%S) $*" >> $L; }
until grep -q "pilot2 all done" $R/refine_pilot.log; do sleep 30; done
SRC=$R/2026-10-27-reasoning-mix-v1; D=$R/2026-10-28-reasoning-mix-refine32
mkdir -p $D/traces $D/quantize $D/eval $D/distill
cp $SRC/run.yaml $SRC/signs.safetensors $D/
find $SRC/traces -maxdepth 1 -type f -exec cp {} $D/traces/ \;
cp -r $SRC/calib $SRC/endpoints $D/
cp $SRC/quantize/*.json $SRC/quantize/state.safetensors $D/quantize/
cp -r $SRC/distill/teacher $D/distill/ && cp $SRC/distill/TEACHER_INPUTS.json $D/distill/
cp -r $SRC/eval/compare-reference-* $D/eval/ 2>/dev/null
log "copied reasoning-mix artifacts into $D"
cd /d/openbonsai/work/refine-pilot-snapshot
export PYTHONPATH=D:/openbonsai/work/refine-pilot-snapshot HF_HOME=$R/hf-cache HF_HUB_OFFLINE=1
U=$R/2026-10-26-5090-ultrachat-attempt2/source/data/train_sft-00000-of-00003-a3ecf92756993583.parquet
log "run start (refine all layers, target original)"
bash $R/gpu_job.sh refine32 D:/openbonsai/.venv-fla/Scripts/python.exe -m openbonsai run --config examples/qwen3_5-9b-reasoning.yaml --out $D \
  --set model.local_files_only=true --set distill.train.small=false --set traces.batch_tokens=14000 \
  --set traces.sources.4.path=$U --set calibrate.sources.1.path=$U --set distill.teacher.sources.1.path=$U \
  --set refine.enabled=true --set refine.target=original > $R/refine32.stdout 2>&1 && log "run done" || log "run FAILED"
