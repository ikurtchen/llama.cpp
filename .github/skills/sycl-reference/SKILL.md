---
name: sycl-reference
description: >
  Authoritative, sharded knowledge base for CUDA→SYCL and Triton→SYCL migration and Intel Xe2/Xe3
  optimization (Arc Pro B60/B70 on Battlemage, Crescent Island on Xe3P). Use when you need canonical
  facts: SYCL 2020 API mappings, Triton→SYCL mappings, when/how to target sycl-tla, the
  target-platform catalog + hardware envelope (peaks, ridge point, sub-group width, SLM, XMX), the
  optimization catalog, or proven anti-patterns. Other skills cite this one. Triggers: "how do I map
  CUDA X to SYCL", "how do I map Triton tl.X to SYCL", "should this target sycl-tla or plain SYCL",
  "what is the B70 ridge point", "what AOT device token for Crescent Island", "what work-group size
  on Intel GPU", "is this a known anti-pattern", "how does oneDNN/PyTorch-XPU/vLLM do this kernel",
  "show me a reference SYCL implementation".
---

# SYCL / Xe2·Xe3 Knowledge Base

A **citable, sharded** reference. Do **not** read it all — load only the shard you need via the
INDEX. Each fact is tagged with provenance so you know how much to trust it.

## How to use
1. Open `references/INDEX.md` and pick the shard that matches your question.
2. Read only that shard. Shards are kept small and focused so context stays bounded even as the
   knowledge base grows.
3. When you learn something durable during a run (a measured peak, an optimization that helped or
   regressed on a specific kernel), append it to `.sycl/state/lessons.md` **with provenance** — never
   silently edit canonical shards.

## Provenance tags (correctness policy)
Every non-trivial claim carries a tag so canonical and empirical facts are not confused:

- `[SPEC]` — from the SYCL 2020 spec or Intel/oneAPI documentation. Cite the URL. Stable.
- `[ARCH]` — architectural fact (sub-group width, SLM model, XMX behavior). Stable across boards of
  the same arch; note where Xe3P differs from Xe2.
- `[VERIFY]` — a device peak or limit that MUST be confirmed on the running device / official spec
  sheet before it is trusted. Replace with a measured value + source + date when confirmed.
- `[EMPIRICAL: <gpu>, <driver>, <oneapi>, <date>]` — a measured result (achieved bandwidth, an
  optimization outcome). Only valid for that environment; re-verify elsewhere.
- `[REF: <repo>@<relative/path>]` — a pattern observed in a local reference repo (see
  `references/intel-gpu-software-repos.md`). Illustrative, not authoritative; a local checkout may drift from
  upstream. For API *semantics* prefer `[SPEC]`/`intel-llvm` over any application repo.

Rule: **`[VERIFY]` and `[EMPIRICAL]` facts are never trusted as permanent.** Canonical `[SPEC]`/
`[ARCH]` facts are cited to their source. This keeps the base correct as hardware/drivers change.

## Verifying device peaks
Use `sycl-profiler`'s detection (`.sycl/scripts/run.sh env`, `sycl-ls --verbose`, `zeinfo`) to read
EU count, max work-group size, `local_mem_size`, and sub-group sizes off the real device, and confirm
FP/bandwidth peaks against the official spec sheet. Record confirmed values in `.sycl/state/lessons.md`
and update the relevant shard's `[VERIFY]` rows.

## Shards (see references/INDEX.md)
- `references/sycl-kernel-patterns.md` — CUDA→SYCL 2020 mapping tables + canonical kernel patterns.
- `references/triton-patterns.md` — Triton→SYCL 2020 mapping (program/tile model, block pointers +
  masks, `tl.dot`→XMX, reductions, atomics, autotune) + patterns.
- `references/sycl-tla-patterns.md` — when/how to migrate to **sycl-tla** (CUTLASS-style tiled
  GEMM/attention) instead of plain SYCL: decision rule, programming model, build wiring.
- `references/intel-gpu-hardware.md` — target platform catalog (B60/B70/CRI), architecture facts,
  device envelope, roofline method.
- `references/sycl-build-guide.md` — icpx/oneAPI build & compile flag reference (JIT vs AOT,
  device tokens, oneMKL/oneDNN/sycl-tla linking, env, common failures).
- `references/intel-gpu-profiling-tools.md` — Intel GPU profiling tool inventory: `unitrace`
  collection modes (device timing, metric query/sampling, stall sampling), scoped collection to avoid
  crashing the tracer, the IGC shader/asm dump (JIT vs AOT), VTune/Advisor, and the Level Zero backend
  selector. The *tool* companion to the metrics shards; cited by `sycl-profiler` §4.
- `references/sycl-optimization-catalog.md` — profiling-signal → strategy lookup table + proven
  anti-patterns (feeds the `sycl-optimization` search loop).
- `references/intel-gpu-hardware-metrics-bmg.md` / `references/intel-gpu-hardware-metrics-cri.md` —
  `unitrace`/Metrics Discovery metric-group + counter catalog (units, derived ratios, empirical
  thresholds, and the stall-name crosswalk to this framework's `SendStall`/`SbidStall`/… aliases) for
  Battlemage (Xe2) and Crescent Island (Xe3P). Cited by `sycl-profiler` §4 and the optimization
  catalog's trigger vocabulary.
- `references/intel-gpu-software-repos.md` — catalog of local upstream repos (oneDNN, sycl-tla, intel-llvm,
  PyTorch XPU, vLLM XPU, Triton, IGC, compute-runtime, …) to consult for how real Intel-GPU SYCL
  code is written. Paths are per-machine in `.sycl/config.json` → `reference_repos`.

## Local reference repositories
When the shards, `.sycl/state/lessons.md`, and your own reasoning don't settle a design question,
consult the local upstream repos configured in `.sycl/config.json` → `reference_repos`. Load
`references/intel-gpu-software-repos.md` for the catalog (each repo's role + what to grep for) and the rules:
**read-only, extract patterns not verbatim code (respect licenses), skip repos whose path is empty or
missing, and log anything learned to `.sycl/state/lessons.md` as `[REF: <repo>@<path>]`.** For API
*semantics* prefer `intel-llvm`/the SYCL 2020 spec; application repos are illustrative, not canonical.
