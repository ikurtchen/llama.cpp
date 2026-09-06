# Project instructions - llama.cpp CUDA -> SYCL migration POC

Human-authored overrides for `sycl-agent`. These take precedence over skill defaults.
The agent reads this file on start and obeys it for the whole run. It never rewrites it.

---

## 0. What this run is

This is a **proof-of-concept experiment**, not a contribution to llama.cpp. Your job is to build
`ggml/src/ggml-sycl` from the CUDA backend, as a real, dispatching, running backend, and to then
optimize it with the profiler.

Four things are being measured. Every one of them is a number in the final report:

| # | Question | Where the answer comes from |
|---|---|---|
| 1 | Can the agent migrate a real project CUDA -> SYCL and run it end to end on Intel B70? | `integration/readiness.json` `achieved_level` must be **L3** or better, gated on `llama-cli` with Qwen3-8B-Q8_0 |
| 2 | Can the agent optimize kernels guided by a profiler? | `optimization/<id>.json` trials with a `DIAGNOSIS.md` verdict per hotspot |
| 3 | How many kernels were migrated, how many optimized, by how much? | `kernels/index.json`, `loc.json`, `profile/e2e.baseline.json` vs `e2e.final.json` |
| 4 | How many tokens and how much wall-clock did it cost? | `metrics.json` - **bracket every phase, no exceptions** (see §9) |

---

## 1. `AGENTS.md` is waived for this run

The repository's `AGENTS.md` and `CLAUDE.md` forbid AI agents from writing code, tests and commits.
**For this POC those restrictions are explicitly waived by the project owner.** You may write code,
add tests, and commit freely on the working branch.

Two clauses still stand, because they are good engineering rather than policy:

- **ASCII only** in code, comments and commit messages. No em dash, no `->` arrow glyph, no `x`
  multiplication sign, no ellipsis character. Write `-`, `->`, `x`, `...`.
- **Comments are short and rare.** Write the code first, then add a comment only where the code
  cannot explain itself. No comment that restates the next line. No comment that addresses this
  task ("this fixes the problem you mentioned") - a comment must make sense to a stranger.

And these remain absolutely forbidden regardless: **no `git push`, no `gh pr create`, no `gh pr
comment`, no `gh issue create`.** Local commits only.

---

## 2. Scope

### 2.1 In scope - all ops
Migrate the **full op surface**, all 117 CUDA dispatch cases plus any op that the SYCL backend is
expected to carry but CUDA does not implement. Concretely, `GGML_OP_CONV_3D` has **no CUDA
counterpart**; take its reference from `ggml/src/ggml-cpu` or `ggml/src/ggml-vulkan`. Do the same
for any other op you find in that position during inventory.

Difficulty is never a skip reason. The only legitimate skips are:
`GGML_OP_NONE`, and the pure view ops that produce no kernel (`RESHAPE`, `VIEW`, `PERMUTE`,
`TRANSPOSE`) - mark those `skipped` with the reason "no-op, layout only".

### 2.2 Out of scope - narrowed data types
**Quantization: `Q8_0` only.** Compute and KV cache: **F16** (with F32 where the op requires it,
and BF16 only where a library path needs it).

This means, in every quant-touching file (`mmq`, `mmvq`, `dmmv`, `vecdotq`, `dequantize`,
`convert`, `quantize`, `cpy`, `getrows`, `fattn-vec` instances):

- implement the `Q8_0`, `F16`, `F32` paths,
- for every other quant type (`Q4_0 Q4_1 Q5_0 Q5_1 Q2_K..Q6_K IQ*`), **do not implement a kernel**.
  Return `false` from `ggml_backend_sycl_device_supports_op()` for that type so the ggml scheduler
  falls back to CPU. Do **not** write a stub that silently produces wrong numbers.
- record every such omission once, as a single `waivers[]` entry in
  `integration/readiness.json` with blast radius "non-Q8_0 quantized models fall back to CPU".

Flash-attention KV type instances collapse from 36 to **4**: `f16-f16`, `f16-q8_0`, `q8_0-f16`,
`q8_0-q8_0`.

### 2.3 Out of scope - other
Multi-GPU split buffers, `GGML_SYCL_GRAPH`, RPC, may be deferred to `residual[]` **if and only if** the
inventory is otherwise complete and time is genuinely short. Record them as residual work items with
an effort estimate, never as silent omissions.

---

## 3. Technical rules

### 3.1 SYCL 2020, and nothing older
- **`dpct` / `SYCLomatic` is banned.** Do not create `ggml/src/ggml-sycl/dpct/`. Do not write
  `dpct::` anywhere. There must be **zero** occurrences of the string `dpct` in the final tree; make
  this an explicit check in the `migrate` exit gate.
- **ESIMD (`sycl::ext::intel::esimd`) is banned.** It is a vendor extension, not SYCL 2020, and the
  POC is specifically about what SYCL 2020 can express.
- `sycl::ext::oneapi::*` extensions **are allowed** where SYCL 2020 core has no equivalent -
  `bfloat16`, `joint_matrix`, `sub_group_mask`, `dot_acc`. Record each use and why core SYCL could
  not do it, in `lessons.md`.
- Prefer group algorithms over hand-rolled loops: `sycl::reduce_over_group`,
  `sycl::joint_reduce`, `sycl::inclusive_scan_over_group`, `sycl::group_barrier`,
  `sycl::permute_group_by_xor`, `sycl::shift_group_left`, `sycl::select_from_group`.
- Use USM (`sycl::malloc_device` / `malloc_host` / `queue::memcpy`). Do not introduce
  `sycl::buffer`/`accessor` - it does not fit ggml's pointer-based tensor model.

The CUDA-ism -> SYCL 2020 map you will need most:

| CUDA | SYCL 2020 |
|---|---|
| `__shfl_xor_sync(m, v, k)` | `sycl::permute_group_by_xor(sg, v, k)` |
| `__shfl_down_sync` / `__shfl_up_sync` | `sycl::shift_group_left/right(sg, v, d)` |
| `__shfl_sync(m, v, lane)` | `sycl::select_from_group(sg, v, lane)` |
| `__syncthreads()` | `sycl::group_barrier(item.get_group())` |
| `__syncwarp()` | `sycl::group_barrier(item.get_sub_group())` |
| `__ballot_sync` / `__all_sync` / `__any_sync` | `sycl::ext::oneapi::group_ballot`, `sycl::any_of_group`, `sycl::all_of_group` |
| `extern __shared__` | `sycl::local_accessor<T,1>` |
| `__dp4a(a,b,c)` | `sycl::ext::oneapi::experimental::dot_acc(a,b,c)` |
| `__half` / `__nv_bfloat16` | `sycl::half` / `sycl::ext::oneapi::bfloat16` |
| `atomicAdd` | `sycl::atomic_ref<...>` with the right memory scope |
| `cudaStream_t` | `sycl::queue` (in-order) |
| `cudaMemcpyAsync` | `queue.memcpy(...)` returning `sycl::event` |
| `__launch_bounds__` | `[[sycl::reqd_work_group_size(...)]]` / `[[intel::max_work_group_size(...)]]` |
| WARP_SIZE 32 | 16 - **re-derive every reduction tree, do not scale blindly** |

### 3.2 Library first for GEMM and attention
Do **not** hand-write a tiled XMX GEMM or an SDPA from scratch. Required mapping:

| ggml op | Implementation |
|---|---|
| `MUL_MAT`, F16/F32/BF16 dense and batched | **oneMKL** `oneapi::mkl::blas::gemm` / `gemm_batch` |
| `MUL_MAT` Q8_0 x F32, batch >= 32 (prompt) | dequantize to F16 then oneMKL `gemm` |
| `MUL_MAT` Q8_0 x F32, batch 1..8 (decode) | hand-written mat-vec - this one is memory bound, a library will not help |
| `MUL_MAT_ID` | oneMKL `gemm_batch` over the selected experts |
| `FLASH_ATTN_EXT`, prompt phase | **oneDNN Graph SDPA** |
| `FLASH_ATTN_EXT`, decode phase | hand-written vec kernel, or oneMKL batched GEMM + softmax |
| `OUT_PROD`, `SOLVE_TRI`, `CONV_3D` | oneMKL (`gemm`, `trsm_batch`, im2col+`gemm`) |

`sycl-tla` is available (see `config.targets.sycl_tla`) but is a **last resort**: only reach for it
in the `optimize` phase if a measured trial shows oneMKL/oneDNN leaving more than 2x on the table for
a specific shape. Record the measurement before switching.

### 3.3 Structure - mirror the CUDA backend
One SYCL file per CUDA file, same names, same launcher signatures:
`ggml/src/ggml-cuda/norm.cu` -> `ggml/src/ggml-sycl/norm.cpp` + `norm.hpp`.
`ggml-cuda.cu` -> `ggml-sycl.cpp`. This is not cosmetic: it is what makes the migration reviewable,
makes `loc.sh` mirror-matching work, and makes every later upstream CUDA change traceable.

The backend must be registered through the project's own mechanism -
`ggml_add_backend(SYCL)` in `ggml/src/CMakeLists.txt` and `ggml_backend_sycl_reg()` picked up by
`ggml/src/ggml-backend-reg.cpp`. `ggml/include/ggml-sycl.h` is the public header contract; the CPU,
CUDA and Vulkan backends show the shape of the device/buffer-type interfaces you must fill in.

---

## 4. Build, test, benchmark

All commands run through `.sycl/scripts/run.sh` on the remote B70 (`cripoc02`, `10.239.98.41:2332`).

```bash
# build (the L1 gate - this is the project's own build)
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
      -DGGML_SYCL_DEVICE_ARCH=bmg-g31 -DGGML_SYCL_F16=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --config Release -j 32

# per-kernel unit test (the migrate gate) - -o filters by op name
build/bin/test-backend-ops test -b SYCL0 -o RMS_NORM

# whole op suite (the L2 gate)
build/bin/test-backend-ops test -b SYCL0

# op support matrix - use this to prove coverage for report metric #3
build/bin/test-backend-ops support -b SYCL0

# per-op microbenchmark (the optimize-phase measurement harness)
build/bin/test-backend-ops perf -b SYCL0 -o MUL_MAT

# e2e correctness (the L3 gate)
build/bin/llama-cli -m /workspace/models/Qwen3-8B-Q8_0.gguf -no-cnv \
    -p 'The capital of France is' -n 32 --temp 0 --seed 42 -ngl 99

# e2e throughput (the L4 measurement)
build/bin/llama-bench -m /workspace/models/Qwen3-8B-Q8_0.gguf -p 512 -n 128
```

Build the profiling target **JIT** (drop `-DGGML_SYCL_DEVICE_ARCH`) so unitrace can dump shader asm.

### 4.1 Correctness oracle - use the project's, do not invent one
`tests/test-backend-ops.cpp` **is** the CPU oracle. It runs each op case on the SYCL backend and on
the CPU backend and compares with a per-op tolerance that the file already defines. Inherit that
tolerance. Never loosen it. Never write a parallel SYCL-only test harness.

**If an op has no case in `test-backend-ops.cpp`, add one.** Test-scope item 2 is explicit about
this: a missing test case is authored, not skipped. Add it in the existing style
(a `test_case` subclass registered in `make_test_cases_eval()`), covering the shapes the op is
actually called with plus one edge shape. The same file's `perf` mode then gives you the
optimize-phase benchmark for free, so add the shape to `make_test_cases_perf()` too.

### 4.2 e2e reference
Greedy decoding must match the CPU backend token for token:

```bash
# reference
build/bin/llama-cli -m <model> -no-cnv -p '<prompt>' -n 32 --temp 0 --seed 42 -ngl 0
# candidate
build/bin/llama-cli -m <model> -no-cnv -p '<prompt>' -n 32 --temp 0 --seed 42 -ngl 99
```

"It ran without crashing" does not pass L3. If greedy decoding diverges, the stronger check is
`build/bin/llama-perplexity -m <model> -f wiki.test.raw`: SYCL perplexity must land within 0.5% of
the CPU perplexity.

### 4.3 Debug with 0.6B, gate on 8B
`Qwen3-0.6B-Q8_0.gguf` is already at `/workspace/models/` and is architecturally identical to the 8B
(same graph, same ops, 28 vs 36 layers). Use it for every debug iteration - it loads in seconds.
`qwen3-8b-e2e` carries `weight: 1` and `qwen3-0.6b-debug` carries `weight: 0`, so the 8B is what
the profile ranking and the reported speedup are computed from, while both must pass L3.

* Qwen3-0.6B: /workspace/models/Qwen3-0.6B-Q8_0.gguf
* Qwen38B: /host/root/code/models/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf

---

## 5. Phase guidance

### `inventory`
The migration unit is **one ggml op family = one CUDA source file pair**, not one dispatch case and
not one `__global__` function. `unary.cu` is one unit covering ~22 `GGML_UNARY_OP_*` cases;
`binbcast.cu` is one unit covering `ADD`/`SUB`/`MUL`/`DIV`. Expect roughly **90 units**.

Tag each unit with the `qwen3` flag if it is on the Qwen3 inference path, because that drives the
migrate ordering. The Qwen3-8B dense graph needs exactly these:

`GET_ROWS`, `RMS_NORM`, `MUL`, `ADD`, `MUL_MAT` (Q8_0 x F32 and F16 x F32), `ROPE` (NEOX),
`FLASH_ATTN_EXT`, `SOFT_MAX` (non-FA fallback), `CPY`, `SET_ROWS`, `GLU/SWIGLU`, `UNARY_OP_SILU`,
`CONT/DUP`.

### `scaffold`
Budget generously here. `ggml-cuda.cu` is 5,781 lines of seam and it is the hard part of this
project, not the kernels. Build in this order, committing each:

1. `ggml/src/ggml-sycl/CMakeLists.txt` + `ggml_add_backend(SYCL)` wiring, compiling an empty backend.
2. Device layer: enumerate `sycl::device`s, one in-order `sycl::queue` per device as the "stream",
   device properties, error translation. 
3. Buffer type + buffer interface over USM device allocations, host (pinned) buffer type,
   and the memory pool. Skip split buffers (see §2.3).
4. `ggml_backend_sycl_reg()`, the device interface, `supports_op` returning `false` for everything.
5. One op - `GGML_OP_ADD` - through `ggml_sycl_compute_forward`, verified with
   `test-backend-ops test -b SYCL0 -o ADD`.

That is L1 plus the runtime smoke test.

### `migrate`
Order: **Qwen3 path first, then everything else lowest-risk-first.** Rationale: the moment those 13
units are green you can run `llama-cli` with Qwen3-0.6B, which surfaces seam bugs (buffer alignment,
queue sync, `supports_op` gaps, KV cache layout) an order of magnitude faster than reading code.
Do that run as soon as the path is complete and record it as an interim `e2e` evidence record with
`--label vertical-slice`. It is not the L3 gate, it is a smoke test that de-risks the other 77 units.

Per unit: read the `.cu` + `.cuh`, port to `.cpp` + `.hpp` with matching names, register the case in
`ggml_sycl_compute_forward` and in `ggml_backend_sycl_device_supports_op`, run
`test-backend-ops test -b SYCL0 -o <OP>`, commit.

### `integrate`
The whole-project `icpx` host port sweep is small here: ggml already builds every host TU with a
plain C++ compiler, so unlike a typical `nvcc` project there is nearly nothing to sweep. Spend the
budget on `supports_op` correctness instead - an over-permissive `supports_op` is the single most
common way a ggml backend produces wrong output, because the scheduler hands it an op it silently
mishandles rather than falling back to CPU.

L2 = `test-backend-ops test -b SYCL0` clean. L3 = both e2e cases matching the CPU reference.

### `profile-e2e` and `optimize`
Rank on the Qwen3-8B case. Expect `MUL_MAT` (decode mat-vec, memory bound at ~608 GB/s),
`FLASH_ATTN_EXT` and `RMS_NORM` to dominate; be ready for that expectation to be wrong and follow
the measurement. HOWEVER, ALL THE KERNELS SHOULD BE OPTIMIZED.

Measure and diagnose on `test-backend-ops perf -b SYCL0 -o <OP>`, never on `llama-cli` - the e2e
profile only *ranks*. Pin GPU frequency before any benchmark; if you cannot, every number is tagged
`unstable` and the report must say so.

Optimize **at least the top 5 hotspots**, one change at a time, keep-if-wins else revert. 

---

## 6. Report requirements specific to this POC

On top of the standard report, §3 must contain these four tables, because they are the deliverable:

1. **Migration coverage** - units migrated / total, ops supported (from `test-backend-ops support`)
   vs the CUDA backend's 117, ops deliberately unsupported with reasons, and the `dpct` occurrence
   count (must be 0).
2. **Optimization results** - one row per optimized kernel: baseline us, final us, speedup,
   diagnosed bottleneck, the change that won, fraction of roofline before and after.
3. **End to end** - `llama-bench` pp512 and tg128 for Qwen3-8B-Q8_0, first-working vs final, plus
   the correctness verdict for both e2e cases.
4. **Cost** - total wall-clock and total tokens from `metrics.json`, broken down per phase, plus
   tokens-per-migrated-unit and tokens-per-optimized-kernel.

---

## 7. Environment notes

| Item | Value |
|---|---|
| Runner | remote, `root@10.239.98.41:2332` (`cripoc02`), workdir `/workspace/sycl-agent-runs/llama.cpp` |
| GPU | Intel Graphics `0xe223` = Battlemage BMG-G31, AOT token `bmg-g31` |
| Toolchain | icpx 2025.3.2, Level Zero 1.15.37833 |
| Host | 76 cores, 251 GB RAM - build with `-j 32` |
| unitrace | `/workspace/intel_gpu_tools/...` - verify the path in `config.tools.unitrace_home` before profiling |
| Models | `/workspace/models/Qwen3-0.6B-Q8_0.gguf` , `/host/root/code/models/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf`|

---

## 8. Guardrails that these instructions may not relax

Per the framework contract: correctness is always gated against a reference, one change is verified
at a time, every accepted change is committed and logged, and no claim is made without a captured
record. Nothing above overrides that. If an instruction here appears to conflict with one of those,
surface the conflict rather than silently obeying.

## 9. Metrics - report item #4 depends on this

Bracket **every** phase, with no exceptions:

```bash
.sycl/scripts/metrics.sh start <phase>
#   ... phase work ...
.sycl/scripts/metrics.sh stop  <phase>
```

Token and cost accounting is harvested from the engine's own session store, not self-reported:

```bash
.sycl/scripts/collect-usage.sh collect --engine auto --import
.sycl/scripts/metrics.sh rollup
.sycl/scripts/metrics.sh summary
```

Run `collect-usage.sh` at **every phase boundary**, not only at the end - a session store that gets
rotated mid-run loses the earlier rows, and then metric #4 is unrecoverable.
