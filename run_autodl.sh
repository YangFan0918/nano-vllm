#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for autoDL containers to cap CPU threads for OpenMP/BLAS/PyTorch
# Usage:
#   bash run_autodl.sh [entry.py] [args...]
#   THREADS=10 bash run_autodl.sh example.py

THREADS=${THREADS:-10}
PYTHON=${PYTHON:-python}

ENTRY=${1:-example.py}
if [[ $# -ge 1 ]]; then
  shift
fi

if [[ ! -f "$ENTRY" ]]; then
  echo "Entry script not found: $ENTRY" >&2
  exit 1
fi

# Cap OpenMP threads and bind behavior
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}

# Prevent math libs from adding more threads
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

# Keep PyTorch single-threaded on CPU to avoid oversubscription
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}

# For Intel/LLVM OpenMP runtimes (no-op on others)
export KMP_AFFINITY=${KMP_AFFINITY:-granularity=fine,compact,1,0}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}

exec "$PYTHON" "$ENTRY" "$@"

