# Device Runtime Layer — CUDA runtime/driver → SYCL

The kernels are the part everyone plans for. The **device runtime layer** is the part that is always
missing when a migration ends at L0: who owns the device, who owns the queue, what a "stream" is,
where memory comes from, and how the project's existing `cudaMalloc`/`cudaMemcpyAsync`/`cudaEvent`
call sites keep compiling.

Build it at `scaffold`, before migrating kernels. It is a few hundred lines, it is the same shape in
every project, and retrofitting it is what turns a two-week last mile into a two-month one.

---

## Design rule: keep the project's own abstraction

Do **not** expose `sycl::queue` to the whole codebase. Every project already has a device
abstraction — a `Handle`, `Resources`, `Context`, `DeviceAPI`, or a set of `gpuXxx` macros. Implement
**that** interface over SYCL. Call sites then need no change, the diff stays reviewable, and the CUDA
backend keeps working.

```cpp
// backend.hpp — ONE place selects the backend. Call sites never see #ifdef.
#if defined(FOO_BACKEND_SYCL)
  #include "backend_sycl.hpp"
#else
  #include "backend_cuda.hpp"
#endif
```

---

## Core mapping

| CUDA | SYCL 2020 | Notes |
|---|---|---|
| `cudaSetDevice(i)` / current device | pick `sycl::device` from `sycl::platform::get_devices()`; keep a per-thread "current device" index in your layer | SYCL has **no implicit current device** — you must own that state |
| `cudaGetDeviceCount` | `device::get_devices(info::device_type::gpu).size()` | filter by backend (Level Zero) to avoid duplicate enumeration through multiple platforms |
| CUDA context | `sycl::context` (one per process, shared by all queues on the device) | create **one** context and reuse it; a context per queue defeats USM sharing and is a classic perf cliff |
| `cudaStream_t` | `sycl::queue` (in-order: `property::queue::in_order{}`) | an **in-order queue is the faithful stream analogue**. Use out-of-order only where the CUDA code genuinely relied on independent streams |
| default/null stream | one designated in-order queue in your layer | SYCL has no implicit global stream; do not create a fresh queue per call — queue creation is not free |
| `cudaStreamSynchronize(s)` | `q.wait()` (or `q.wait_and_throw()`) | prefer `wait_and_throw` so async device errors surface where they happened |
| `cudaDeviceSynchronize()` | `wait()` on every queue your layer owns | keep a registry of live queues in the layer |
| `cudaEvent_t` + `cudaEventRecord` | `sycl::event` returned by each submit; store the last event per queue | to time, submit on a queue with `property::queue::enable_profiling{}` and read `info::event_profiling::command_{start,end}` |
| `cudaEventSynchronize(e)` | `e.wait()` | |
| `cudaStreamWaitEvent(s, e)` | `q.submit([&](handler& h){ h.depends_on(e); … })`, or `ext_oneapi_submit_barrier({e})` | this is how cross-stream ordering is preserved |
| `cudaEventElapsedTime` | profiling-info delta (ns) | different clock domain from CUDA's — never compare absolute numbers across backends |
| `cudaMalloc` | `sycl::malloc_device(bytes, dev, ctx)` | |
| `cudaMallocHost` (pinned) | `sycl::malloc_host(bytes, ctx)` | |
| `cudaMallocManaged` | `sycl::malloc_shared(bytes, dev, ctx)` | shared USM performs differently from CUDA UM; prefer explicit device USM on the hot path |
| `cudaFree` / `cudaFreeHost` | `sycl::free(ptr, ctx)` | **the context must be the same one that allocated** — a mismatched free is a silent corruption |
| `cudaMemcpy*` | `q.memcpy(dst, src, bytes)` (+ `.wait()` for the sync form) | |
| `cudaMemset*` | `q.memset(ptr, value, bytes)` / `q.fill<T>(…)` | |
| `cudaMemcpy2D` / strided | a small copy kernel or a loop of `memcpy` rows | there is no direct 2D copy; write it once in the layer |
| `cudaPointerGetAttributes` | `sycl::get_pointer_type(ptr, ctx)` / `get_pointer_device` | useful for asserting a pointer belongs to the right context |
| `cudaGetLastError` / `cudaError_t` | C++ exceptions (`sycl::exception`) | translate at the layer boundary into the project's existing error enum — do **not** let SYCL exceptions escape into C call sites |
| async error handler | pass an `async_handler` to the queue/context | without one, async errors are reported at an unrelated later point and are very hard to attribute |
| `cudaDeviceProp` | `device.get_info<info::device::…>()` | `max_work_group_size`, `local_mem_size`, `global_mem_size`, `max_compute_units`, sub-group sizes |
| `cudaGraph` | `sycl_ext_oneapi_graph` (command graph extension) | optional; only worth it where CUDA graphs were load-bearing |
| `__launch_bounds__` | `[[sycl::reqd_work_group_size(...)]]`, `[[intel::reqd_sub_group_size(N)]]` | |
| `cudaOccupancyMaxActiveBlocks…` | no equivalent; derive from device info + measured occupancy | put a fixed, tuned value in the layer and let `sycl-optimization` search it |

### Minimal layer sketch
```cpp
struct Backend {                     // one per process
  sycl::device  dev;
  sycl::context ctx;                 // ONE context, shared by every queue
  std::vector<sycl::queue> streams;  // in-order queues; streams[0] == "default stream"

  void*  alloc(size_t n)                     { return sycl::malloc_device(n, dev, ctx); }
  void   free_(void* p)                      { sycl::free(p, ctx); }
  void   h2d(void* d, const void* h, size_t n, int s) { streams[s].memcpy(d, h, n); }
  void   sync(int s)                         { streams[s].wait_and_throw(); }
  sycl::event record(int s)                  { return streams[s].ext_oneapi_submit_barrier(); }
  void   wait_event(int s, sycl::event e)    { streams[s].ext_oneapi_submit_barrier({e}); }
};
```
Keep this in **one** header/source pair named after the project's existing backend files.

---

## Memory management

Replacing `cudaMalloc` one-to-one is correct but usually **slow**: projects that used a caching
allocator (RMM, a framework caching allocator, a pool) were relying on it for hot-path allocation.

- **Mirror the interface, not the implementation.** Implement the project's own allocator interface
  (`rmm::device_memory_resource`-shaped, or the framework's) over USM so call sites are untouched.
- **Add a simple pool early.** A size-bucketed free-list over `malloc_device` removes the per-call
  allocation cost, which otherwise shows up as a mysterious e2e regression that no kernel profile
  explains.
- **Stream-ordered semantics.** If the CUDA code used stream-ordered allocation
  (`cudaMallocAsync`/RMM streams), the SYCL replacement must not recycle a block before the work
  using it has completed — track the last event per block.
- **Framework interop (archetype C):** do **not** allocate independently. Take the pointer from the
  framework tensor and submit on the framework's own XPU queue, so its caching allocator's lifetime
  and stream ordering remain valid.
- **Context identity is a correctness invariant.** Every pointer must be freed on the context it was
  allocated from; assert it in debug builds.

---

## Host libraries

These are **host-side call sites**, not kernels — they are integration work, and they are usually
what actually blocks the build.

| CUDA-side | Intel replacement | Notes |
|---|---|---|
| Thrust (`thrust::sort`, `reduce`, `scan`, `transform`, vectors) | **oneDPL** (`oneapi::dpl::execution::make_device_policy(q)` + `std::sort`/`reduce`/`inclusive_scan`) | closest 1:1 mapping; `thrust::device_vector` → USM allocator + `std::vector`, or keep raw pointers |
| CUB device-wide primitives (`DeviceScan`, `DeviceRadixSort`, `DeviceReduce`) | oneDPL device policies | CUB *block-level* primitives are kernel work (group collectives) — that is `sycl-migration`'s job, not this one |
| cuBLAS / cuBLASLt | **oneMKL** (`oneapi::mkl::blas::…`, column-major aware) | mind the layout convention; wrap once |
| cuDNN | **oneDNN** primitives | fused conv/norm/attention shapes |
| cuFFT | oneMKL DFT | callbacks (`cuFFT Xt`) have no direct analogue — waive or restructure |
| cuRAND | oneMKL RNG (`oneapi::mkl::rng`) | bit-exact streams differ; regenerate reference data on both sides rather than comparing sequences |
| cuSPARSE / cuSOLVER | oneMKL sparse / LAPACK | coverage is not 1:1 — check the specific routine early |
| NCCL | **oneCCL** | single-device can stub the collective layer; multi-device is a separate estimate |
| NVML | `sycl` device info + `xpu-smi` | telemetry only; usually waivable |
| CUB/Thrust in **public headers** | must be removed or guarded | a Thrust type in an exported header forces every consumer to have CUDA |
| OptiX / DLSS / OpenXR / TensorRT | **no equivalent** | record a waiver with the blast radius |

Link/flag details (oneMKL/oneDNN link lines, AOT targets, common icpx errors) are in
`sycl-reference` → `references/sycl-build-guide.md`.

---

## Multi-device and collectives

- Enumerate devices once in the layer; map the project's device index onto it (do not assume ordinal
  equality with anything).
- Peer access (`cudaDeviceEnablePeerAccess`) → devices in the **same context** can share USM
  pointers; cross-context sharing is not valid.
- Collectives → oneCCL, with its own communicator lifecycle. Keep the collective layer behind the
  project's existing interface so a single-device stub and the real implementation are swappable.
- If the target workload is single-device, **say so** and put multi-device in the residual package
  with its own estimate.

---

## Verification for this layer (the L1 smoke test)

The scaffold smoke test must exercise the layer, not a kernel:
1. enumerate + select a device, print its name;
2. allocate device USM, `memset`, copy H2D, run one trivial kernel, copy D2H, verify on the host;
3. record an event, wait on it from a second queue, verify ordering;
4. free everything and destroy the queues cleanly (no leak, no async exception at teardown).

Capture it once green:
```bash
.sycl/scripts/evidence.sh record integration runtime --label scaffold-smoke \
    --reference "host-computed expected buffer" -- "<smoke command through the project's own API>"
```
Drive it **through the project's own API** (its `Handle`/`Resources`/`DeviceAPI`), not through raw
SYCL — otherwise it proves SYCL works, which was never in doubt, rather than that the project's
runtime abstraction works.
