# Host-Code Port Sweep — compiling the project's own `.cc/.cpp` with `icpx`

On a real library, the kernels are a minority of the code. Hundreds of **host** translation units
include CUDA headers, use CUDA types in signatures, and are compiled by `nvcc` today. Until they
compile with `icpx`, there is no `libfoo.so` and therefore no L1 — regardless of how many kernels are
green.

This work is **broad but shallow**: many small mechanical fixes, few hard problems. It is the single
most under-estimated item in a last-mile plan, and the most predictable once measured.

---

## Size it first (at `scaffold`, in 5 minutes)

```bash
# how many host TUs currently see CUDA?
grep -rIl --include='*.cc' --include='*.cpp' --include='*.cxx' \
     -E 'cuda_runtime|cuda\.h|thrust/|cub/|cublas|__host__|__device__' . | wc -l
```
That count **is** the estimate. Roughly: ≤20 TUs → days; 50–200 → 2–4 weeks; 500+ → its own phase.
Put the number in the `build` surface's `effort` field so nobody re-litigates it later.

---

## The checklist

Work file-by-file, but fix **classes** of problem across the tree in one pass each — one `git commit`
per class, so a regression is attributable.

### 1. Function decorations
| Found | Fix |
|---|---|
| `__host__ __device__` on a shared helper | make it a plain `inline` function in a backend-neutral header; SYCL device code can call ordinary functions |
| `__host__` alone | delete |
| `__device__` on something a host TU includes | move it into the SYCL kernel source, or make it a plain function usable from both |
| `__forceinline__` | `inline` (add `__attribute__((always_inline))` only if measured) |
| `__restrict__` | keep — icpx accepts it |
| `__launch_bounds__` in a host header | move to the kernel definition as `[[sycl::reqd_work_group_size(...)]]` |

**Rule:** a function used inside a SYCL kernel must not capture globals, must not use function
pointers, virtuals, RTTI, exceptions, or `std::` facilities that are not device-safe (`std::string`,
`std::vector`, iostreams, `printf` beyond `sycl::ext::oneapi::experimental::printf`).

### 2. CUDA types leaking into host signatures and public headers
| Found | Fix |
|---|---|
| `cudaStream_t` in a public API | replace with the project's own stream/queue handle type (defined by the backend layer) |
| `cudaError_t` returns | the project's own error enum, translated at the layer boundary |
| `dim3`, `float4`, `__half`, `__nv_bfloat16` | backend-neutral aliases: `sycl::float4`, `sycl::half`, `sycl::ext::oneapi::bfloat16` behind a project typedef |
| `thrust::device_vector` / `cub::` in an **exported** header | remove or guard — it forces CUDA on every consumer |
| `#include <cuda_runtime.h>` in a host TU | replace with the backend header |

This is the highest-value class: fixing the public headers usually unblocks dozens of TUs at once.
Do it **first**.

### 3. nvcc-isms the host compiler never had to accept
- `#pragma unroll` outside a loop, or with nvcc-specific forms → `#pragma unroll N` / drop.
- nvcc-only extended lambdas (`__device__` lambdas, `--expt-extended-lambda`) → plain lambdas.
- nvcc's tolerance for missing `typename`/`template` disambiguation → `icpx` (Clang-based) is
  stricter; add them.
- Two-phase lookup: templates referring to base-class members without `this->` compile under nvcc's
  MSVC-ish path but not under Clang → add `this->`.
- Implicit narrowing in brace-init, missing `#include`s that nvcc pulled in transitively.
- `-Werror` sets tuned for nvcc will fire new warnings; fix them, do not blanket-disable — a new
  warning class here is frequently a real bug.

### 4. Build-system surgery
- Add the SYCL option/target beside the CUDA one; **do not** delete the CUDA path.
- Host TUs compile with `icpx` (no `-fsycl` needed unless they contain SYCL); kernel TUs compile with
  `icpx -fsycl`. Mixing is fine and keeps compile times down.
- Separate compilation with `-fsycl` needs the device link step — let the build system drive it
  (`icpx -fsycl` as the linker), and prefer JIT during the sweep (AOT only for the final artifact and
  never for profiling builds).
- Keep generated-file rules (protobuf, flatbuffers, codegen) untouched; only the compiler changes.
- Preserve the project's existing `-O`/LTO settings; if a specific TU miscompiles, pin **that TU** to
  a lower `-O` with a comment naming the bug, rather than lowering the whole build.

### 5. Known compiler-level traps
- **IGC optimizer bugs** at higher `-O` on a specific kernel: isolate to the single TU, pin its
  optimization level, record it as a `nvonly`/`build` note with the reproducer, and file it upstream.
  A whole-project `-O0` is not an acceptable resolution — it silently invalidates every performance
  number afterwards.
- **Long compile times** on heavily templated code: `-fsycl-device-code-split=per_kernel` helps link
  time; AOT for every dev build does not pay for itself.
- **Missing `sycl::` device-side `std::` support**: replace with `sycl::` math builtins.

---

## Order of work (fastest path to L1)

1. Public headers — remove CUDA types from the exported surface.
2. Backend layer header — so every TU has something to include.
3. Shared helpers — de-decorate `__host__ __device__` utilities.
4. Bulk sweep — compile, fix, repeat, in dependency order (leaf libraries first).
5. Link the project's own artifact. **That link is the L1 gate.**

Track it as a single `build` surface with a running count in `notes`
(`"142/190 host TUs compiling"`), so progress is visible in PROGRESS.md without inventing sub-items.

---

## What *not* to do

- **Do not** convert host code to SYCL. Host code stays host code; it just needs a compiler that
  accepts it and a backend header that gives it types.
- **Do not** use `dpct`/SYCLomatic on host files either. Its output is unreviewable and it will
  rewrite call sites you were deliberately keeping identical.
- **Do not** delete the CUDA path to make the build pass. The whole value of the mirrored structure
  is that both backends stay live and comparable.
- **Do not** mark the surface `done` when it compiles but does not link, or links but the artifact is
  never loaded. The gate is the project's own artifact, produced by the project's own build.
