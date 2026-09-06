# Integration Archetypes

Six shapes cover essentially every CUDA project. Pick one at `scaffold` (S1), and it tells you the
**surfaces**, the **order**, the **realistic target level**, and the **specific trap** that shape is
prone to. Record the choice and a one-line reason in `readiness.json` `archetype` /
`archetype_reason`.

Identify the archetype from three questions:
1. **Who owns the entrypoint?** The project (A/B/F) or a host framework (C/D/E)?
2. **Where does device code come from?** Written ahead of time (A/B/C/F) or generated at runtime (D/E)?
3. **How does a call reach a kernel?** Direct call (A), library API (B), framework dispatcher (C/D),
   graph/parser (F), or generated + JIT-compiled (E)?

---

## A — C/C++ application with a backend seam
**Shape:** one driver (`main`, a training loop, an inference loop) + per-kernel headers or a kernel
directory. CUDA is reached through inline launcher functions.
**Reachable level:** **L4**. Cheapest archetype by a wide margin.

**Surfaces:** `build`, `runtime`, `memory` (thin), `dispatch` (trivial — it is the header swap),
`nvonly` (rare). No `package`.

**Recipe**
1. Backend header (`backend.hpp`) that includes either the `.cuh` or the `.hpp` kernel set and
   aliases the memory/stream primitives (`gpuMalloc`, `gpuMemcpy`, `gpuStream_t`, `gpuSync`).
2. Keep **launcher signatures byte-identical** to the CUDA ones — the driver must not know which
   backend it links against.
3. Add a sibling build target (`make train_foo_sycl` beside `make train_foo`), same objects, same
   driver source, `icpx -fsycl` instead of `nvcc`.
4. Run the real workload; diff against the project's own reference artifact.

**Trap:** the seam gets built and the workload is never run, because the per-kernel tests are green
and look like completion. **The per-kernel tests are L0.** Budget the workload run explicitly — it is
usually a day, and it is the single most valuable day in the project.

**Signals you are here:** a `dev/cuda/` per-kernel harness; a `train_*.c` / `main.cpp` driver;
kernels in `.cuh` headers with inline host launchers.

---

## B — C++ library with CMake and its own test suite
**Shape:** builds `libfoo.so`, ships `gtest`/`ctest`/pytest, consumed by other programs. Often has a
resource/handle object (`StandardGpuResources`, `Context`, `Handle`) that owns device, stream, and
memory.
**Reachable level:** **L3** (L4 if it ships a benchmark suite).

**Surfaces:** `build`, `runtime` (the resource object), `memory`, `dispatch`, `hostlib`, `package`
(CMake export / `find_package`), `nvonly`.

**Recipe**
1. Add a `FOO_ENABLE_SYCL` CMake option beside `FOO_ENABLE_CUDA`; a SYCL source list mirroring the
   CUDA one (same file names, same launcher signatures) so the option flips the source set.
2. Implement the **resource object** over SYCL: `sycl::queue` per stream, USM allocations, an
   optional temp-memory arena mirroring the CUDA one, a device scope/guard type.
3. Sweep the **host** translation units with `icpx` (`host-port-sweep.md`). On a big library this is
   the bulk of the calendar time — broad, shallow, and highly parallelisable.
4. Replace host-side Thrust/CUB/cuBLAS with oneDPL/oneMKL (`device-runtime-layer.md` §Host libraries).
5. Run the project's **own** unmodified test suite as the L2 gate; that is the whole point of
   mirroring the CUDA structure.

**Trap:** stopping after the infrastructure (option + dispatch macros + device-type enum) without
compiling the host tree. Count the host TUs at `scaffold` — the number is the estimate.

**Signals:** `CMakeLists.txt` with a CUDA option; `tests/` with gtest; a `Resources`/`Handle` class;
`rmm::`, `thrust::`, `cub::` in host code.

---

## C — Framework custom-op extension (PyTorch / TensorFlow)
**Shape:** kernels compiled by `setuptools`/`CMake` into an extension loaded by a Python framework;
tensors come from the framework's allocator; ops are registered into its dispatcher.
**Reachable level:** **L3**.

**Surfaces:** `build` (extension), `runtime` (framework device/stream interop), `memory` (framework
allocator interop — **do not** allocate independently), `dispatch` (op registration), `package`
(`pip install -e .`), `nvonly`.

**Recipe**
1. Replace the CUDA extension build with an **XPU extension**: `icpx -fsycl` compile of the SYCL
   sources, link against the framework's XPU runtime, keep the same module name and the same
   Python-visible symbols.
2. **Interop, do not allocate.** Take the raw pointer out of the framework tensor and use the
   framework's own XPU queue/stream for submission, so caching-allocator lifetime and stream
   ordering stay correct. Allocating your own USM behind the framework's back is the #1 source of
   corruption and mysterious hangs here.
3. Register ops on the XPU dispatch key with the **same schema strings** as the CUDA ones, so
   existing Python call sites are unchanged.
4. Gate L2 on the project's own pytest suite with the device set to `xpu`; gate L3 on a real
   model forward+backward (or an inference generation) vs the framework's CPU or reference result.

**Traps:** (a) a kernel that segfaults on the **backward** path blocks every training workload — fix
prerequisites before claiming a level; (b) a second kernel source (e.g. a set of Triton kernels)
that was never enumerated silently caps coverage — enumerate it and put it in the residual package;
(c) dtype/layout mismatches (channels-last, non-contiguous) that the CUDA path handled implicitly.

**Signals:** `setup.py` with `CUDAExtension`; `TORCH_LIBRARY` / `REGISTER_OP`; `torch.utils.cpp_extension`.

---

## D — Python array / JIT library
**Shape:** a Python package whose device code is **source strings compiled at runtime** (raw-kernel
templates, JIT modules). `import pkg` is itself a substantial integration surface.
**Reachable level:** **L2–L3**; the long tail of the project's own test suite is usually out of scope
and must be *declared* out of scope.

**Surfaces:** `package` (make `import pkg` work — often blocked by pre-existing, unrelated import
bugs), `runtime` (device/stream shim), `memory` (memory pool over USM), `dispatch` (module-level
routing of array ops to the SYCL JIT path), `hostlib`, `nvonly`.

**Recipe**
1. Fix the import path first — it is usually hours and it blocks *everything* downstream.
2. Memory pool over USM with the same interface the package's own allocator exposes.
3. Stream/event shim mapped to `sycl::queue`.
4. Wire the SYCL compile path into the core module dispatch, then run the package's own pytest suite
   as the acceptance gate.

**Trap — the denominator.** The unit of migration is the *template*, not the kernel, so "8/8
migrated" can hide ~90 first-party kernels living as **CUDA-C++ source strings inside `.py`
modules**. A best-effort source translator is not coverage: if **no test exercises** a single one of
them, coverage is **unknown**, and unknown is what the report must say. Enumerate the embedded-kernel
modules at `scaffold` and put them in the residual package with an explicit count.

**Second trap — re-cutting the inventory.** Regenerating the kernel index under a coarser taxonomy
turns pending items into a smaller denominator and reports 100%. Never re-cut: carry old ids forward
or mark them explicitly dropped with a reason.

**Signals:** `RawKernel`/`RawModule`-style APIs; kernel source in Python string literals; a
`_core`/`_driver` compiled core; a memory pool module.

---

## E — Runtime / JIT-codegen system
**Shape:** the project **generates device source at runtime** from user code (a Python-to-kernel
compiler) and ships its own device runtime (a large `.cu` runtime file, a ctypes layer, graph
capture, device/stream management).
**Reachable level:** **L1–L2**. A codegen backend is a systems project, not a port — say so up front.

**Surfaces:** `runtime` (a whole new ctypes/device runtime layer), `dispatch` (a **SYCL codegen
target** in the compiler), `build`, `memory`, `package`.

**Recipe**
1. Be explicit at `scaffold`: the migrated kernels are the *library* half; the *compiler/runtime*
   half is the residual package. Set `target_level` accordingly and estimate it.
2. If any level is targeted: implement the device runtime layer (device/stream/event/alloc/graph)
   over SYCL first — it is reusable regardless of when codegen lands.
3. The codegen target itself (emit SYCL source, compile with `icpx`, cache the modules) goes in the
   residual package with a `high` risk.

**Trap:** presenting kernel-level completion as backend support. The correct framing is "kernel layer
de-risked; device backend scoped separately" — quote it separately in every rollup.

**Signals:** a `codegen.py`; runtime kernel-source generation; a multi-thousand-line device runtime
`.cu`; a ctypes/cffi binding layer.

---

## F — Graph / parser-driven engine
**Shape:** a model description (JSON/graph/config) is parsed into a pipeline of layers/operators by a
factory; the production path never calls kernels directly.
**Reachable level:** **L2–L3**.

**Surfaces:** `dispatch` (the layer/op **factory** and the pipeline builder — this is the whole
game), `runtime`, `memory`, `build`, `hostlib` (collectives!), `nvonly`.

**Recipe**
1. Register the SYCL layer/op implementations in the project's **factory**, keyed exactly as the
   CUDA ones are.
2. Make the parser build a pipeline of SYCL layers from an unmodified model config — the L3 gate is
   "the config the users write, run unchanged".
3. Collectives: single-device can stub the collective layer; multi-device needs oneCCL and is its own
   estimate — keep it separate.

**Trap — bypassing the production path.** A manual forward/backward loop that constructs layers by
hand *works*, uses the real objects, and is **not** integration: users go through the parser. That is
L1/L2, not L3. Also investigate parity anomalies (e.g. a mixed-precision path with no speedup) before
they are reported as results — they usually mean a path is not actually being taken.

**Signals:** a JSON/YAML model parser; a `create_pipeline`/`LayerFactory`; NCCL in the layer set.

---

## Cross-archetype prerequisite rule

A **broken kernel on the path the workload needs** blocks the level; it does not lower it. If a
backward-path kernel segfaults or is numerically wrong, fix it (back-edge to `migrate`) *before*
claiming L3 for any training workload — a level claimed over a known-broken required kernel is worse
than no claim, because it will fail on the reader's first run.
