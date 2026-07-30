---
name: sycl-profiler
description: >
  Detect the Intel GPU execution environment, benchmark SYCL kernels and end-to-end workloads
  with stable methodology, classify kernels as memory- or compute-bound via roofline, and
  attribute end-to-end wall-clock to kernels to set optimization priority. Use when: detecting
  icpx/oneAPI/GPU arch, pinning GPU frequency for stable benchmarks, measuring kernel bandwidth
  or throughput, running unitrace/VTune, computing arithmetic intensity vs the target's ridge point,
  or ranking kernels by their share of e2e runtime. All execution routes through the remote/local
  runner (.sycl/scripts/run.sh).
---

# SYCL Profiler

Measure and classify — reliably. Everything here runs on an Intel GPU host through
`.sycl/scripts/run.sh`, so it works whether or not the local machine has a GPU.

## Required reading
- `sycl-reference` → `references/intel-gpu-hardware.md` (envelope, ridge point, roofline).
- `sycl-reference` → `references/intel-gpu-profiling-tools.md` (the `unitrace`/VTune/IGC-dump tool
  reference: collection modes, scoped collection, JIT-vs-AOT dump timing).
- `sycl-reference` → `references/intel-gpu-hardware-metrics-bmg.md` /
  `references/intel-gpu-hardware-metrics-cri.md` (per-metric-group counter catalog, units, derived ratios,
  empirical thresholds, and the `SendStall`/`SbidStall`/… ↔ `XVE_STALL_*` crosswalk) — load the shard
  for the target arch (`xe2`→BMG, `xe3p`→CRI) when reading counter CSVs in §4.

## 1. Environment detection (phase: detect)
```bash
.sycl/scripts/run.sh env                 # host, icpx version, sycl-ls, unitrace presence
.sycl/scripts/run.sh exec "sycl-ls --verbose"
```
Record in `.sycl/state/project.json.env`: `icpx`, `oneapi`, `gpu`, `arch` (expect `xe2`/`xe3p`),
and read EU count / max WG size / `local_mem_size` / sub-group sizes off the device. Confirm the
board's FP/bandwidth peaks against the official spec sheet (see the knowledge shard) and note them in
`.sycl/state/lessons.md`.

> Deep profiling's asm annotation (`analyzeperfmetrics.py`) needs **pandas + matplotlib**. `run.sh`
> provisions the sycl-agent Python deps on the runner automatically — no action needed here. If they
> are ever unavailable (e.g. offline), the metric-counter and stall CSVs are still produced; only the
> asm-annotation/PDF step is skipped.

### GPU frequency pinning (required before benchmarking)
Unpinned GPU clocks make benchmarks unreproducible. Check and, if allowed, pin to max.
`$XPU_SMI` / `$UNITRACE` / `$VTUNE` are exported by `run.sh` from `.sycl/config.json.tools` so the
same command works on servers that install these tools in different locations:
```bash
.sycl/scripts/run.sh exec '"$XPU_SMI" discovery' || true
.sycl/scripts/run.sh exec '"$XPU_SMI" config -d 0 -t 0 --frequencyrange <max>,<max>' || true
```
If frequency cannot be pinned, set `env.freq_pinned: false` and treat benchmark numbers as noisy
(report variance; avoid ranking on small differences).

## 2. Stable benchmark methodology
- Warm-up runs (default 3) then timed runs (default 20; see `config.json.benchmark`).
- Report **median + p90 + stddev**, not a single number.
- Pin the backend: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
- Keep inputs fixed and large enough to saturate; exclude host<->device transfer unless that is the
  thing being measured.
- Helper: `scripts/bench.sh` wraps warm-up/iters and prints JSON stats.

### Kernel-level benchmark (the optimization measurement target)
The **unit of optimization is a single kernel**, so the thing you benchmark and profile during
optimization is a **standalone single-kernel benchmark driver** — *not* the full end-to-end workload.
Running the whole e2e app to time one kernel is slow, noisy (other kernels/host code in the way), and
is what makes `unitrace` overflow and core-dump. A small single-kernel driver is fast, deterministic,
and safe to profile.

**If the project has no kernel-level benchmark, write one** (`sycl-migration` scaffolds it from
`templates/kernel_bench.cpp`; if it is missing at optimization time, author it then). Requirements:
- Drives **one kernel** in isolation: allocate USM inputs, regenerate them from the kernel detail's
  seed+shape+dtype, run the kernel, `q.wait()`, time device execution over warm-up + iters.
- **Choose representative shapes** — benchmark and report each, because the best config can differ by
  shape:
  - **Popular model configs** — sizes from real models the kernel serves (e.g. hidden dims 768/1024/
    2048/4096/8192, head dim 64/128, seq len 512/1024/2048/4096, typical batch/token counts). Pull
    the actual shapes the source project uses where known.
  - **General sizes** — a couple of round, saturating sizes to reflect steady-state throughput.
  - **Edge cases** — tiny inputs, non-power-of-2 and non-multiple-of-sub-group sizes, masked/boundary
    tiles, and a large size near memory limits — these expose tail effects and correctness-adjacent
    perf cliffs.
- Emit per-shape median/p90/stddev (drive it with `scripts/bench.sh`, one shape per invocation, or
  loop shapes inside the driver and print JSON per shape).
- Build it **JIT (no AOT)** when it will be the shader-dump/stall target — see §4.

This kernel benchmark is what `sycl-optimization` uses for every MEASURE and every DIAGNOSE.

## 3. Roofline classification (per kernel)
Compute arithmetic intensity `AI = FLOPs / bytes_moved` and compare to the device ridge point
(`peak_FLOPS / peak_bandwidth` from the knowledge shard):
- `AI < ridge` → **memory-bound**; expected metric is achieved bandwidth (GB/s) vs peak.
- `AI > ridge` → **compute-bound**; expected metric is achieved FLOPS vs peak.
Set `performance.bound`, `arithmetic_intensity`, and `meets_expectation` (e.g. ≥60–70% of relevant
peak) in the kernel detail. Helper: `scripts/roofline.py`.

## 4. Deep profiling (for underperformers)
Roofline + timing tells you *whether* a kernel is memory- or compute-bound; deep profiling tells you
*why* and *where*, so `sycl-optimization` can pick the right fix instead of guessing. Gather **three
complementary lenses** and combine them. Tooling is resolved deterministically from
`.sycl/config.json.tools.unitrace_home` (exported as `$UNITRACE_HOME` / `$UNITRACE` /
`$METRICS_PLATFORM`, plus `$VTUNE`) — the same install-vs-source layout as the reference
`run_unitrace.sh`, so the binary, `analyzeperfmetrics.py`, and the metrics config all come from one
known root instead of `PATH`.

Helper (resolves paths, runs the counter groups + stall sampling, and annotates the asm). It also
**transposes every metrics CSV** to a readable `*_transposed.csv` (one row per kernel instance), and
after the asm annotation overlays per-IP stall counts onto the disassembly as `*.asm.ip.stall`:
```bash
.sycl/scripts/run.sh profile \
  "bash framework/skills/sycl-profiler/scripts/profile_metrics.sh <build-dir>/test_<id>"
# knobs: PROFILE_MODE=all|metrics|stall, METRICS_GROUPS="VectorEngineStalls,ComputeBasic,MemoryProfile" (comma-separated),
#        METRIC_MODE=query|sampling (query=aggregate per-kernel counters; sampling=--metric-sampling timeline),
#        KERNEL_NAME=<k>, SHADER_DUMP=on|off
# fast follow-up (only re-query a group, skip stall/asm), e.g. after the first full run:
.sycl/scripts/run.sh profile \
  "PROFILE_MODE=metrics METRICS_GROUPS=MemoryProfile \
   bash framework/skills/sycl-profiler/scripts/profile_metrics.sh <build-dir>/test_<id>"
```
`PROFILE_MODE` gates the phases: `all` (default) = counters + stall + asm; `metrics` = only the
`--metric-query` group pass (cheap follow-up to inspect memory/cache/etc. after a first run);
`stall` = only the stall-sampling + asm-annotation pass.

1. **Hardware metric counters** — `unitrace --metric-query --group <G>`. Query more than one group:
   start with **`VectorEngineStalls`** for the *full aggregate stall mix* of the whole kernel (Active
   % vs each `*Stall` %), then **`ComputeBasic`** for EU active/stall %, thread occupancy, achieved
   bandwidth and L1/L2 hit rates; add `MemoryProfile`, `DeviceCacheProfile`, `ComputeExtended` or `L3`
   when the diagnosis needs a closer look. Reveals **which resource** is the limiter; cross-check
   achieved bandwidth/FLOPS against the roofline peaks. (`profile_metrics.sh` loops `METRICS_GROUPS`;
   set `METRIC_MODE=sampling` to collect a `--metric-sampling` counter timeline instead of one
   aggregate value per kernel.)
   For the counters inside each group, their units/derived ratios, alarm thresholds, and the raw
   `XVE_STALL_*` names behind the aliases below, see the arch metrics shard
   (`intel-gpu-hardware-metrics-bmg.md` / `-cri.md`) — note CRI uses **`L1Profile`** in place of BMG
   `MemoryProfile`.
2. **Stall sampling** — `unitrace --stall-sampling`. Where the aggregate group says the stall *mix*,
   sampling attributes stalls to individual **instruction pointers** and a **reason**, so you know
   *what* is stalling and *where*:

   | Stall reason | Meaning | Points to |
   |---|---|---|
   | `SendStall` | waiting on a memory/SLM message (load/store/atomic) | memory-latency bound → coalesce, prefetch, cut traffic, raise occupancy to hide latency |
   | `SbidStall` | scoreboard dep — waiting on an outstanding async result | low ILP → more in-flight loads, unroll, software pipeline |
   | `PipeStall` / `DistStall` | back-to-back ALU dependency / register distance | low ILP → unroll, interleave independent work, reorder |
   | `ControlStall` | branch / control flow | divergence → reduce branching, uniformize |
   | `SyncStall` | barrier wait | over-synchronization → fewer/rebalanced barriers |
   | `InstrFetchStall` | instruction-cache miss | code too large → unroll less, shrink kernel |

3. **IGC shader / asm dump** — capture the generated assembly to see register spills/fills, SIMD
   width, DPAS density, and uncoalesced sends directly. `profile_metrics.sh` captures this automatically
   on the stall run into a **freshly-wiped** dir (`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=…`) so it
   holds exactly one dump set, then maps the stall CSV onto the asm by IP via unitrace's
   `analyzeperfmetrics.py -k <kernel> -s <dump_dir>` (emits `<kernel>.asm.ip` + a PDF). Three
   requirements the helper enforces:
   - **JIT build (not AOT).** The runtime dump only works for **JIT** compilation. An **AOT** build
     (`-fsycl-targets=spir64_gen`) compiles the kernel at *build* time, so the runtime dump is empty.
     Profile a **JIT-built** kernel benchmark (the optimization target should be built without AOT),
     **or** capture the dump at build time and pass it in with `SHADER_DUMP=prebuilt
     SHADER_DUMP_DIR=<build-dump>` (the helper then skips the runtime dump and reuses it).
   - **Correct kernel name.** Auto-detected when the dump holds a single kernel; otherwise set
     `KERNEL_NAME=<kernel>` (the annotation is per-kernel).
   - **One shader dump per kernel.** Profile a single-kernel benchmark (`test_<id>`/`bench_<id>`) and
     let the helper wipe the dump dir each run — multiple kernels in the dir make the IP→asm mapping
     ambiguous and the helper will refuse rather than annotate the wrong asm.

   **Register spills** in the dump are a top red flag on Xe — they silently kill occupancy and inflate
   `SendStall`.

Combine: e.g. *memory-bound roofline + high `SendStall` + no spills* → coalescing/traffic problem;
*compute-bound + high `PipeStall`/`SbidStall`* → ILP problem; *counters show spills + high `SendStall`*
→ register pressure. Record the diagnosis (bottleneck class + dominant stall + spill flag) in the
kernel detail and feed it to `sycl-optimization`.

**If `unitrace` cannot be resolved on the runner** (check `detect_env.sh` → `unitrace_available`,
which honors `.tools.unitrace_home`; `profile_metrics.sh` exits non-zero with a clear message), fall back
gracefully — do not block the workflow:
- Use **VTune** (`"$VTUNE" -collect gpu-hotspots`, GPU stall/occupancy) if available, else
- Derive the bottleneck from **roofline + timing alone**: measured bandwidth/FLOPS vs peak and the
  arithmetic intensity already tell you memory- vs compute-bound. This is enough to drive most of the
  optimization checklist (you just lose per-IP stall/spill precision).
- Record the tool actually used in the profiling report; note the missing tool in
  `.sycl/state/lessons.md` so the environment gap is visible.

## 5. End-to-end profiling → optimization priority (phase: profile-e2e)
Profile the whole workload and attribute wall-clock to kernels, capturing **both host overhead and
device (kernel) time**. Write `.sycl/state/profile/e2e.json` (validate against
`e2e-profile.schema.json`) with a `ranking` sorted by descending `wallclock_pct`. This ranking is the
optimization order — **optimize the biggest contributors first.** Low-impact kernels can be deferred.
Set `basis: "e2e-workload"`.

**Library projects (no single workload).** When `project.kind == "library"` there is nothing to
attribute wall-clock to. Instead build the per-kernel benchmarks (the `bench_<id>` drivers) at
representative shapes and rank by each kernel's **own runtime** (× a per-kernel call-weight if
`.sycl/config.json` supplies one). Write the same `e2e.json` shape with `basis: "kernel-suite"` and
`wallclock_pct` = each kernel's share of the suite total. Every exported kernel is a deliverable, so
all get optimized; the ranking only orders effort. The unitrace scoping notes below do not apply —
single-kernel drivers don't crash unitrace.

Helper: `scripts/profile_e2e.sh` runs unitrace with `--host-timing` (host API overhead) **and**
`--device-timing`/`--device-timeline` (per-kernel device time), so one trace yields host-vs-device
split and a per-kernel ranking. It has two capture modes:
```bash
# Small/short workload — one-shot overview (will crash on big runs):
.sycl/scripts/run.sh profile \
  "bash framework/skills/sycl-profiler/scripts/profile_e2e.sh overview ./app --iters 3"

# Real workload — crash-safe TIME SLICE via unitrace sessions. Start the app PAUSED (async), let it
# reach steady state, resume for one layer/iteration, then stop:
.sycl/scripts/run.sh exec \
  "bash framework/skills/sycl-profiler/scripts/profile_e2e.sh slice mysess ./app" &   # background
.sycl/scripts/run.sh exec "bash framework/.../profile_e2e.sh resume mysess"           # begin collecting
.sycl/scripts/run.sh exec "bash framework/.../profile_e2e.sh stop   mysess"           # flush the slice
```

**Tracing the full workload can crash unitrace** (huge trace → OOM / core dump). Do not trace
start-to-end; scope the collection (see `intel-gpu-profiling-tools.md` → Scoped collection):
- Prefer the **`slice` mode**: trace only a **representative window** (one layer / one iteration) with
  `--start-paused --session <name>` bracketed by `resume`/`stop` (the helper wraps these), or wrap the
  region in-app with `PTI_ENABLE_COLLECTION`/`__itt_resume`.
- Set `INCLUDE_KERNELS=<substrs>` (or `EXCLUDE_KERNELS`) to collect just the kernels you care about —
  a second lever for shrinking the trace.
- If unitrace still crashes or is unavailable, fall back to **lightweight in-app device timers**
  (per-kernel `sycl::event` profiling or wall-clock around each launch) to build the ranking — an
  acceptable, documented substitute. Note the method used in the report.

Parse the `=== Device Timing ===` per-kernel totals from the trace into the `ranking`; keep the
`=== Host Timing ===` totals to flag host-bound stretches (submission/sync overhead) worth a note.

Note: e2e profiling is only for **ranking**. Per-kernel optimization does **not** re-run the e2e
workload — it benchmarks and profiles the standalone **kernel-level benchmark** (§2).

## Reporting & state
- Write benchmark/profiling reports to `.sycl/reports/`.
- Update kernel `performance` blocks and `profile/e2e.json`; regenerate PROGRESS.md.
- `.sycl/scripts/log.sh sycl-agent <phase> info "<benchmark/verdict>"`.

## Note on cross-vendor comparison
Benchmark SYCL against its **own baseline** and the **target device's roofline** (per
`target.platform`). Do not compare SYCL-on-Intel
wall-clock against CUDA-on-NVIDIA — different architectures are not directly comparable.

## Assets
- `scripts/detect_env.sh` — probe icpx/oneAPI/GPU/unitrace/xpu-smi on the runner (JSON-ish output).
- `scripts/bench.sh` — warm-up + timed iterations, prints median/p90/stddev.
- `scripts/profile_metrics.sh` — single-kernel: metric counters (+ per-CSV transpose) + stall sampling +
  IGC shader dump + asm annotation (`analyzeperfmetrics.py`) + per-IP stall overlay.
- `scripts/profile_e2e.sh` — whole-workload host+device timing; `overview` (one-shot) or crash-safe
  `slice` (session `--start-paused`/`resume`/`stop`) for the kernel ranking.
- `scripts/transpose_csv.py` — transpose a unitrace metrics CSV (used by `profile_metrics.sh`).
- `scripts/annotate_asm_stalls.py` — overlay per-IP stall counts + top-N ranks onto `.asm.ip`.
- `scripts/roofline.py` — AI, ridge point, bound classification, fraction-of-peak.
- `sycl-migration/templates/kernel_bench.cpp` — standalone single-kernel benchmark driver (shape sets:
  model configs / general / edge) used as the optimization measurement + profiling target.
