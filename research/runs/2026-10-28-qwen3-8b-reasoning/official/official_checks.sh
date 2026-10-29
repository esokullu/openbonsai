#!/bin/bash
# Prism ML's Ternary-Bonsai-8B (PQ2_0, group 128), scored the way our Qwen3-8B GGUF was (../prism/cpu_checks.sh).
# CPU only, no gpu.lock: CUDA hidden. Resumable: each step is skipped when its output already exists.
R=D:/openbonsai/runs; M=$R/ternary-bonsai-8b-gguf; O=$R/qwen3-8b/official; W=$R/qwen3-8b/wikitext
BIN=$R/third_party/prism-llama.cpp/build-cpu/bin; PY=D:/openbonsai/.venv-fla/Scripts/python.exe
REPO=D:/openbonsai/.claude/worktrees/modest-chandrasekhar-f8f56d
F=Ternary-Bonsai-8B-PQ2_0.gguf; REV=c2aefbeb4b24469cd11579c3384b990404c17a30
SHA=1376f942aa90e60f7b570c1d81b3916fea1315ff85aa1c4d19006af68fb4b922
export CUDA_VISIBLE_DEVICES=-1 HF_HOME=$R/hf-cache PYTHONUTF8=1 OMP_NUM_THREADS=8
mkdir -p $M $O
cd $O
if [ ! -s $O/sha256.ok ]; then
  echo "$(date '+%H:%M:%S') download $F at $REV"
  $PY -I -c "import sys; from huggingface_hub import hf_hub_download as d; print(d('prism-ml/Ternary-Bonsai-8B-gguf', sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))" \
    $F $REV $M || exit 1
  got=$(sha256sum $M/$F | cut -d' ' -f1)
  [ "$got" = "$SHA" ] || { echo "sha256 mismatch: $got"; exit 1; }
  echo "$got  $F" > $O/sha256.ok
fi
# the file as published (YaRN rope scaling in its metadata), then with rope scaling off, as our file runs
for run in wiki.test: wiki.test.join: wiki.test:rope-none; do
  f=${run%%:*}.raw; tag=${run##*:}
  out=$O/prism_ppl_ternary-bonsai-8b_${f%.raw}${tag:+.$tag}.log
  [ -s $out ] && grep -q "Final estimate" $out && continue
  echo "$(date '+%H:%M:%S') perplexity $f ${tag:-as published}"
  $BIN/llama-perplexity.exe -m $M/$F -f $W/$f -c 512 --chunks 8 -t 8 ${tag:+--rope-scaling none} > $out 2>&1
  grep "Final estimate" $out
done
# can openbonsai read the file? inspect lists it; eval --from builds the model from it (the reasoning suites
# would run next, so CUDA stays hidden: the load is expected to fail)
if [ ! -s $O/openbonsai_eval.log ]; then
  echo "$(date '+%H:%M:%S') openbonsai load check"
  { echo "\$ openbonsai inspect $F"; PYTHONPATH=$REPO $PY -m openbonsai inspect $M/$F; echo "exit $?"; } \
    > $O/openbonsai_inspect.log 2>&1
  { echo "\$ openbonsai eval --run eval --config examples/qwen3-8b-reasoning.yaml --set model.local_files_only=true --from $F --suite gsm8k,humaneval"
    HF_HUB_OFFLINE=1 PYTHONPATH=$REPO $PY -m openbonsai eval --run $O/eval --config $REPO/examples/qwen3-8b-reasoning.yaml \
      --set model.local_files_only=true --from $M/$F --suite gsm8k,humaneval; echo "exit $?"; } > $O/openbonsai_eval.log 2>&1
  tail -3 $O/openbonsai_eval.log
fi
# the file's metadata and tensor shapes, and how often each 2-bit code occurs
H=$REPO/research/runs/2026-10-28-qwen3-8b-reasoning/official
[ -s $O/header.json ] || $PY -I $H/gguf_header.py $M/$F $O/header.json > /dev/null
[ -s $O/pq2_codes.json ] || $PY -I $H/pq2_codes.py $M/$F $O/pq2_codes.json
echo "$(date '+%H:%M:%S') official checks done"
