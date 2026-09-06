#!/usr/bin/env bash
# Discover CUDA kernels and library usage in a project. Read-only; prints a summary.
# Usage: scan_cuda.sh [project-root]   (default: current dir)
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

echo "== CUDA source files =="
find . -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | grep -v -E '/(build|\.git)/' | sort || true

echo
echo "== __global__ / __device__ definitions =="
grep -rInE '__global__|__device__' --include='*.cu' --include='*.cuh' --include='*.cpp' --include='*.h' --include='*.hpp' . 2>/dev/null \
  | grep -v -E '/(build|\.git)/' || echo "(none found)"

echo
echo "== Kernel launches (<<<...>>>) =="
grep -rInE '<<<[^;]*>>>' --include='*.cu' --include='*.cuh' --include='*.cpp' . 2>/dev/null \
  | grep -v -E '/(build|\.git)/' || echo "(none found)"

echo
echo "== Warp / cooperative-group primitives =="
grep -rInE '__shfl_|__ballot_sync|__any_sync|__all_sync|cooperative_groups|cg::' \
  --include='*.cu' --include='*.cuh' . 2>/dev/null | grep -v -E '/(build|\.git)/' || echo "(none)"

echo
echo "== Shared memory / atomics =="
grep -rInE '__shared__|atomicAdd|atomicCAS|atomicMax|atomicMin|atomicExch' \
  --include='*.cu' --include='*.cuh' . 2>/dev/null | grep -v -E '/(build|\.git)/' || echo "(none)"

echo
echo "== CUDA libraries in use =="
for lib in cublas cudnn thrust cub cufft cusparse cusolver curand; do
  hits=$(grep -rIl -E "${lib}" --include='*.cu' --include='*.cuh' --include='*.cpp' --include='*.h' --include='*.hpp' . 2>/dev/null | grep -v -E '/(build|\.git)/' | head -20 || true)
  [[ -n "$hits" ]] && { echo "-- $lib --"; echo "$hits"; }
done

echo
echo "== Build system hints =="
ls -1 CMakeLists.txt Makefile makefile *.mk 2>/dev/null || true
grep -rIl -E 'cuda|nvcc|CUDA' CMakeLists.txt 2>/dev/null || true

echo
echo "Done. Confirm results by reading the sources — templates/generated launchers can hide kernels."
