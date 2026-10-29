#!/bin/bash
# Download Prism ML's Ternary Bonsai 8B, unpacked (F16 safetensors, ~16.4 GB) at a pinned revision into the
# shared HF cache. Resumable: run it again after an interruption. Detached: PowerShell Start-Process with Git bash.
R=D:/openbonsai/runs; D=$R/2026-10-29-ternary-bonsai-8b-unpacked; L=$D/download.log
REPO=prism-ml/Ternary-Bonsai-8B-unpacked; REV=ac20f03fc62e872399218b659c8e949dfca05769
PY=D:/openbonsai/.venv-fla/Scripts/python.exe
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $L; }
export HF_HOME=$R/hf-cache PYTHONUTF8=1 HF_HUB_DISABLE_TELEMETRY=1
log "download start (pid $(cat /proc/$$/winpid 2>/dev/null || echo $$)) $REPO@$REV"
cd $D
if $PY -I -c "
import sys
from huggingface_hub import HfApi, snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
patterns = ['*.json', '*.jinja', '*.txt', '*.safetensors', '*.md']
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
print('sha', info.sha, flush=True)
for s in info.siblings:
    print(f'{s.rfilename}\t{s.size}', flush=True)
path = snapshot_download(repo, revision=rev, allow_patterns=patterns, max_workers=2)
print('path', path, flush=True)
" $REPO $REV >> $L 2>&1; then log "DONE"; else log "download FAILED"; exit 1; fi
