#!/usr/bin/env bash
# Set up OpenBonsai on Linux with an NVIDIA GPU; optionally build Prism's llama.cpp fork.
#
#   scripts/setup_linux.sh                  # venv, CUDA PyTorch, the package (editable), FLA kernels
#   scripts/setup_linux.sh --prism          # also clone Prism's llama.cpp at the pinned commit; CPU build
#   scripts/setup_linux.sh --prism-cuda     # ... CPU and CUDA builds
#   scripts/setup_linux.sh --prism-only     # only the Prism clone and builds (no Python setup)
#
# Environment overrides (defaults in brackets):
#   VENV           [<repo>/.venv]   virtual environment; created with uv when it is on PATH, else python -m venv
#   PYTHON         [python3.12]     interpreter for python -m venv (uv fetches 3.12 itself)
#   TORCH_VERSION  [2.14.1]         the version the 9B results were measured with (torch 2.14.1+cu130)
#   TORCH_INDEX    [https://download.pytorch.org/whl/cu130]
#                  PyTorch's wheel index picks the CUDA build. cu130 (CUDA 13.0) is what the
#                  measured runs used; it needs NVIDIA driver >= 580 and supports Turing through
#                  Blackwell (sm_75..sm_120, e.g. RTX 5090). Use .../whl/cu128 for drivers 570-579 or for
#                  Volta (V100), which CUDA 13 dropped; .../whl/cpu for no GPU. The CUDA runtime comes
#                  inside the wheels; only the driver must be installed.
#   FLA            [flash-linear-attention]   pip spec of the Gated DeltaNet Triton kernels, which
#                  transformers uses when installed; pin it for reproducible installs, e.g.
#                  FLA=flash-linear-attention==<version>. It is installed under a constraints file
#                  that holds torch and transformers at the versions above, so its resolver cannot
#                  move them. NO_FLA=1 skips it (then the slower PyTorch path runs, as in the measured
#                  runs; results agree up to kernel rounding); a CPU TORCH_INDEX always skips it.
#                  The kernels need CUDA tensors: on the CPU (model.device: cpu, or
#                  CUDA_VISIBLE_DEVICES=-1) openbonsai imports the model with them hidden, and the CPU
#                  test suite blocks them itself. causal-conv1d (optional, built with nvcc) would speed
#                  up the short convolution; it is not installed here.
#   PRISM_DIR      [<repo>/third_party/prism-llama.cpp]        the clone (kept clean: builds go elsewhere)
#   PRISM_BUILD    [<repo>/third_party/prism-llama.cpp-build]  cpu/ and cuda/ build trees
#   CUDA_ARCHS     [llama.cpp's default list]   e.g. 120 for RTX 5090, 90 for H100, 86 for RTX 30xx
#   JOBS           [nproc]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-python3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TRANSFORMERS_VERSION="5.20.2"
FLA="${FLA:-flash-linear-attention}"
PRISM_REPO="https://github.com/PrismML-Eng/llama.cpp"
PRISM_COMMIT="c9c34bc9c4042c13a31598ebf43461cd0ad8f00d"   # = tools/prism/build_prism_eval.py PIN
PRISM_DIR="${PRISM_DIR:-$REPO/third_party/prism-llama.cpp}"
PRISM_BUILD="${PRISM_BUILD:-$REPO/third_party/prism-llama.cpp-build}"
JOBS="${JOBS:-$(nproc)}"

python_setup=1 prism_cpu=0 prism_cuda=0
for arg in "$@"; do
  case "$arg" in
    --prism) prism_cpu=1 ;;
    --prism-cuda) prism_cpu=1; prism_cuda=1 ;;
    --prism-only) python_setup=0; prism_cpu=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }

setup_python() {
  if command -v uv >/dev/null 2>&1; then
    say "venv $VENV (uv)"
    [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
    pip_install() { uv pip install --python "$VENV/bin/python" "$@"; }
  else
    say "venv $VENV ($PYTHON -m venv)"
    [ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    pip_install() { "$VENV/bin/python" -m pip install "$@"; }
  fi

  # PyTorch alone from its own index (which also serves torch's dependencies, triton included),
  # so the CUDA build is the one installed; everything else then comes from PyPI.
  say "torch==$TORCH_VERSION from $TORCH_INDEX"
  pip_install --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"

  say "transformers==$TRANSFORMERS_VERSION and openbonsai (editable, with test extras)"
  pip_install "transformers==$TRANSFORMERS_VERSION" -e "$REPO[test]"

  case "$TORCH_INDEX" in
    */cpu|*/cpu/) cpu_torch=1 ;;
    *) cpu_torch=0 ;;
  esac
  if [ "${NO_FLA:-0}" = "1" ]; then
    say "flash-linear-attention skipped (NO_FLA=1)"
  elif [ "$cpu_torch" = 1 ]; then
    say "flash-linear-attention skipped: CPU PyTorch ($TORCH_INDEX); its kernels need CUDA"
  else
    say "$FLA (Gated DeltaNet kernels), torch and transformers held by constraints"
    local constraints
    constraints="$(mktemp)"
    # The installed torch (with its local tag, e.g. +cu130) and transformers stay as they are.
    "$VENV/bin/python" -c 'import torch, transformers; print(f"torch=={torch.__version__}"); print(f"transformers=={transformers.__version__}")' > "$constraints"
    # Linux CUDA torch wheels already depend on a matching triton; install it only if missing.
    "$VENV/bin/python" -c "import triton" 2>/dev/null || pip_install -c "$constraints" triton
    pip_install -c "$constraints" "$FLA"
    rm -f "$constraints"
  fi

  say "check"
  "$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("transformers", transformers.__version__)
for dist in ("triton", "flash-linear-attention", "fla-core", "openbonsai"):
    try:
        print(dist, md.version(dist))
    except md.PackageNotFoundError:
        print(dist, "not installed")
try:
    import fla  # noqa: F401
    print("fla import ok")
except Exception as error:  # optional: transformers falls back to PyTorch
    print("fla import failed:", error)
EOF
  echo "Activate with: source $VENV/bin/activate"
  echo "CPU test suite (it hides the GPUs itself): $VENV/bin/python -m pytest -q"
}

clone_prism() {
  say "Prism llama.cpp at $PRISM_COMMIT in $PRISM_DIR"
  if [ ! -d "$PRISM_DIR/.git" ]; then
    mkdir -p "$(dirname "$PRISM_DIR")"
    git clone "$PRISM_REPO" "$PRISM_DIR"
  fi
  if ! git -C "$PRISM_DIR" cat-file -e "$PRISM_COMMIT^{commit}" 2>/dev/null; then
    git -C "$PRISM_DIR" fetch origin "$PRISM_COMMIT"
  fi
  git -C "$PRISM_DIR" checkout --detach "$PRISM_COMMIT"
  test "$(git -C "$PRISM_DIR" rev-parse HEAD)" = "$PRISM_COMMIT"
}

build_prism() {   # $1 = cpu | cuda
  local kind="$1" dir="$PRISM_BUILD/$1"
  local flags=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF)
  if [ "$kind" = cuda ]; then
    flags+=(-DGGML_CUDA=ON)
    [ -n "${CUDA_ARCHS:-}" ] && flags+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS")
  else
    flags+=(-DGGML_CUDA=OFF)
  fi
  say "Prism build ($kind) in $dir"
  cmake -S "$PRISM_DIR" -B "$dir" "${flags[@]}"
  cmake --build "$dir" --config Release -j "$JOBS"
  echo "binaries: $dir/bin (llama-cli, llama-server, llama-perplexity, ...)"
}

[ "$python_setup" = 1 ] && setup_python
if [ "$prism_cpu" = 1 ]; then
  command -v cmake >/dev/null || { echo "cmake is required for the Prism build" >&2; exit 1; }
  clone_prism
  build_prism cpu
  [ "$prism_cuda" = 1 ] && build_prism cuda
  echo "Manifest scorer worker: python tools/prism/build_prism_eval.py --source $PRISM_DIR" \
       "--build-dir $PRISM_BUILD/eval-worker [--cuda]"
fi
exit 0
