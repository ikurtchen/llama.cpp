#!/usr/bin/env bash
# Enumerate the LAST-MILE BACKEND INTEGRATION surfaces of a CUDA project.
# Read-only; prints a summary the agent turns into .sycl/state/integration/index.json.
# Usage: scan_integration.sh [project-root]   (default: current dir)
# NOTE: deliberately not `set -e` — a grep with no match must not truncate the scan.
set -uo pipefail

ROOT="${1:-.}"
cd "$ROOT"

EX='/(build|\.git|\.sycl|third_party/.*/test|node_modules)/'
HOST_INC=(--include='*.cc' --include='*.cpp' --include='*.cxx' --include='*.c'
          --include='*.h' --include='*.hpp' --include='*.hxx' --include='*.inl')

hits() {  # hits <regex> [extra find/grep includes...]
  grep -rInE "$1" "${HOST_INC[@]}" --include='*.cu' --include='*.cuh' --include='*.py' \
       --include='CMakeLists.txt' --include='*.cmake' --include='Makefile' --include='*.mk' \
       . 2>/dev/null | grep -vE "$EX" || true
}

section() { echo; echo "== $1 =="; }

echo "== Integration scan: $(pwd) =="

# ---------------------------------------------------------------- surface: runtime -------------
section "surface 'runtime' — CUDA runtime/driver API call sites"
for api in cudaSetDevice cudaGetDevice cudaGetDeviceCount cudaDeviceProp cudaDeviceSynchronize \
           cudaStreamCreate cudaStreamSynchronize cudaStreamWaitEvent cudaEventCreate \
           cudaEventRecord cudaEventElapsedTime cudaGraph cuCtx cuDevice; do
  n=$(hits "\\b${api}" | wc -l)
  [[ "$n" -gt 0 ]] && printf '  %-28s %5d call sites\n' "$api" "$n"
done
echo "  (files touching the CUDA runtime API:"
hits 'cuda_runtime|cuda\.h|cuda_runtime_api' | cut -d: -f1 | sort -u | sed 's/^/    /' | head -40
echo "   )"

# ---------------------------------------------------------------- surface: memory --------------
section "surface 'memory' — allocation / memory manager"
for api in cudaMalloc cudaMallocHost cudaMallocManaged cudaMallocAsync cudaFree cudaMemcpy \
           cudaMemcpyAsync cudaMemset cudaHostRegister; do
  n=$(hits "\\b${api}" | wc -l)
  [[ "$n" -gt 0 ]] && printf '  %-28s %5d call sites\n' "$api" "$n"
done
for mm in 'rmm::' 'device_memory_resource' 'MemoryPool' 'CachingAllocator' 'caching_allocator'; do
  n=$(hits "$mm" | wc -l)
  [[ "$n" -gt 0 ]] && printf '  %-28s %5d hits (project memory manager)\n' "$mm" "$n"
done

# ---------------------------------------------------------------- surface: hostlib -------------
section "surface 'hostlib' — host-side CUDA library call sites"
declare -A LIBMAP=(
  [thrust]='oneDPL' [cub]='oneDPL (device-wide) / group collectives (block-level)'
  [cublas]='oneMKL BLAS' [cublasLt]='oneMKL BLAS' [cudnn]='oneDNN' [cufft]='oneMKL DFT'
  [curand]='oneMKL RNG' [cusparse]='oneMKL sparse' [cusolver]='oneMKL LAPACK'
  [nccl]='oneCCL' [nvml]='device info / xpu-smi' [cutlass]='sycl-tla'
)
for lib in "${!LIBMAP[@]}"; do
  files=$(grep -rIl -E "\\b${lib}" "${HOST_INC[@]}" --include='*.cu' --include='*.cuh' . 2>/dev/null \
          | grep -vE "$EX" || true)
  n=$(printf '%s' "$files" | grep -c . || true)
  [[ "$n" -gt 0 ]] && printf '  %-10s %4d files  ->  %s\n' "$lib" "$n" "${LIBMAP[$lib]}"
done

# ---------------------------------------------------------------- surface: dispatch ------------
section "surface 'dispatch' — registration / dispatch points"
hits 'TORCH_LIBRARY|TORCH_LIBRARY_IMPL|torch::RegisterOperators|REGISTER_OP\(|REGISTER_KERNEL_BUILDER' \
  | head -30 || true
hits 'DeviceAPI|DEVICE_TYPE|kDL[A-Z]+|DLDeviceType|dispatch_|Registry::|LayerFactory|create_pipeline|register_layer' \
  | head -30 || true
echo "  (nothing above => dispatch is direct calls from the driver: archetype A)"

# ---------------------------------------------------------------- surface: build ---------------
section "surface 'build' — build system + host TU count"
ls -1 CMakeLists.txt Makefile makefile setup.py pyproject.toml meson.build 2>/dev/null | sed 's/^/  /' || true
grep -rInE 'CUDA|nvcc|cuda_add|enable_language\(CUDA|CUDAToolkit|CUDAExtension|cpp_extension' \
  --include='CMakeLists.txt' --include='*.cmake' --include='Makefile' --include='*.mk' \
  --include='setup.py' . 2>/dev/null | grep -vE "$EX" | head -30 || true
host_tus=$( { grep -rIl --include='*.cc' --include='*.cpp' --include='*.cxx' \
             -E 'cuda_runtime|cuda\.h|thrust/|cub/|cublas|__host__|__device__' . 2>/dev/null \
             || true; } | grep -vE "$EX" | wc -l)
echo "  HOST translation units that currently see CUDA: ${host_tus}"
echo "  -> this count sizes the icpx host-port sweep (see references/host-port-sweep.md)"

# ---------------------------------------------------------------- surface: package -------------
section "surface 'package' — user-facing entry"
ls -1 setup.py pyproject.toml setup.cfg 2>/dev/null | sed 's/^/  /' || true
grep -rInE 'find_package|install\(TARGETS|export\(' --include='CMakeLists.txt' . 2>/dev/null \
  | grep -vE "$EX" | head -10 || true
grep -rInE '^\s*(import|from)\s+\w+' --include='__init__.py' . 2>/dev/null \
  | grep -vE "$EX" | head -10 || true

# ---------------------------------------------------------------- surface: nvonly --------------
section "surface 'nvonly' — NVIDIA-only dependencies (waiver candidates)"
for dep in optix OptiX dlss DLSS nvrtc NVRTC tensorrt TensorRT nvjpeg nvToolsExt nvtx OpenXR \
           cudaGraphicsGL cudaD3D; do
  n=$(hits "\\b${dep}" | wc -l)
  [[ "$n" -gt 0 ]] && printf '  %-16s %5d hits\n' "$dep" "$n"
done

# ---------------------------------------------------------------- entrypoints ------------------
section "entrypoint candidates (the L3 gate runs one of these)"
grep -rInE '^\s*int\s+main\s*\(' --include='*.c' --include='*.cc' --include='*.cpp' --include='*.cu' . 2>/dev/null \
  | grep -vE "$EX" | head -20 || true
ls -1d examples benchmarks scripts tools 2>/dev/null | sed 's/^/  dir: /' || true
grep -rInE 'console_scripts|entry_points' --include='setup.py' --include='pyproject.toml' . 2>/dev/null | head -5 || true

# ---------------------------------------------------------------- reference artifacts ----------
section "reference artifacts for e2e validation (L3)"
find . -maxdepth 3 -type f \( -name '*debug_state*' -o -name '*golden*' -o -name '*reference*' \
     -o -name '*expected*' -o -name '*.npz' -o -name '*.npy' \) 2>/dev/null \
  | grep -vE "$EX" | head -20 || echo "  (none found — see references/e2e-validation.md for fallbacks)"

echo
echo "Done. Confirm every finding by reading the sources: generated files, macros and templates hide"
echo "call sites, and an absent grep hit is not evidence that a surface does not exist."
