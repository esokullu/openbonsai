#!/bin/bash
# The published 9B's runtime parity with main's script (research/runs/2026-10-28-qwen3-8b-reasoning/runtime_parity.py), CPU only.
set -u
sleep 5
R=D:/openbonsai/runs/2026-10-29-runtime-parity-9b
W=D:/openbonsai/work/publish-parity
export CUDA_VISIBLE_DEVICES=-1 HF_HOME=D:/openbonsai/runs/hf-cache HF_HUB_OFFLINE=1 PYTHONPATH=$W
PY=D:/openbonsai/.venv-fla/Scripts/python.exe
echo "$(date '+%F %T') start, code $(git -C $W rev-parse HEAD)" > $R/launcher.log
$PY -X utf8 -c "import openbonsai; print('openbonsai from', openbonsai.__file__)" >> $R/launcher.log 2>&1
$PY -X utf8 $W/research/runs/2026-10-28-qwen3-8b-reasoning/runtime_parity.py \
  --run D:/openbonsai/runs/2026-10-28-code-gap-1epoch \
  --bin D:/openbonsai/runs/third_party/prism-llama.cpp/build-cpu/bin \
  --text D:/openbonsai/runs/qwen3-8b/wikitext/wiki.test.raw \
  --threads 8 --out $R/runtime_parity.json > $R/runtime_parity.log 2>&1
echo "$(date '+%F %T') exit $?" >> $R/launcher.log
