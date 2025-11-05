#!/usr/bin/env bash
set -euo pipefail

# AutoDL container helper: set sane CPU threading for llama.cpp (OpenMP) and friends.
# Targets: 18 host cores, run llama.cpp with 16 threads by default.
#
# Usage patterns:
#   1) Interactive shell with env set (recommended):
#        bash run_autodl.sh           # drops you into a shell; then run: python3 example.py
#   2) Print exports to eval in current shell:
#        eval "$(bash run_autodl.sh print-env)"
#   3) Run a specific script immediately:
#        bash run_autodl.sh example.py [args...]

PYTHON=${PYTHON:-python3}

# Threads for llama.cpp and OpenMP. Default to 16 on an 18-core host.
LLAMA_THREADS=${LLAMA_THREADS:-${THREADS:-16}}

mode=${1:-shell}
if [[ $# -ge 1 ]]; then
  shift || true
fi

set_env() {
  # Cap OpenMP threads and bind behavior
  export OMP_NUM_THREADS="${LLAMA_THREADS}"
  export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
  export OMP_PLACES=${OMP_PLACES:-cores}
  export OMP_DYNAMIC=${OMP_DYNAMIC:-false}

  # Prevent math libs from adding more threads
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

  # Keep PyTorch single-threaded on CPU to avoid oversubscription
  export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
  export NANOVLLM_TORCH_THREADS=${NANOVLLM_TORCH_THREADS:-1}

  # Hint Config to set llama_cpp threads if not passed explicitly in code
  export NANOVLLM_LLAMA_CPP_THREADS=${NANOVLLM_LLAMA_CPP_THREADS:-${LLAMA_THREADS}}

  # For Intel/LLVM OpenMP runtimes (no-op on others)
  export KMP_AFFINITY=${KMP_AFFINITY:-granularity=fine,compact,1,0}
  export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
}

print_env() {
  cat <<EOF
export OMP_NUM_THREADS="${LLAMA_THREADS}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export NANOVLLM_TORCH_THREADS="${NANOVLLM_TORCH_THREADS:-1}"
export NANOVLLM_LLAMA_CPP_THREADS="${NANOVLLM_LLAMA_CPP_THREADS:-${LLAMA_THREADS}}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
EOF
}

case "${mode}" in
  print-env)
    print_env
    ;;
  shell)
    set_env
    echo "[run_autodl] Env set (llama.cpp threads=${LLAMA_THREADS}). Starting interactive shell..." >&2
    exec "${SHELL:-/bin/bash}"
    ;;
  *)
    ENTRY="${mode}"
    if [[ ! -f "${ENTRY}" ]]; then
      echo "Entry script not found: ${ENTRY}" >&2
      exit 1
    fi
    set_env
    exec "$PYTHON" "$ENTRY" "$@"
    ;;
esac
