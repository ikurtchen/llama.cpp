# Backend Readiness Levels (BRL) — L0 … L4

A migration's *depth* is the single fact readers most want and reports most often blur. This ladder
makes it a measured, evidence-backed number: **declared** before the work (`target_level`),
**computed** after it (`achieved_level`), and printed side by side in the final report.

> The rule that makes it useful: **`achieved_level` is derived from gates that have run records.**
> It is never asserted. A level whose gate has no record is not achieved, however convincing the
> prose is.

---

## The ladder

### L0 — kernel-standalone
> "The kernels are migrated and validated on Intel GPU."

- SYCL kernels exist and pass their own unit tests against a CPU-oracle reference.
- Exercised by **standalone binaries** (`dev/sycl/<k>`), not by the project.
- **Gate:** all non-skipped kernels `migrated` with `unit_test.result: pass` + test records.
- **This is a real deliverable.** It de-risks the hardest technical part (the math is proven correct
  on silicon). It is *not* "the project runs on Intel GPU".

### L1 — build-through
> "…and the project builds with a SYCL backend."

- The project's **own build system** produces the project's **own artifact** with the SYCL/XPU
  backend selected (`libfoo.so`, the app binary, the Python extension `.so`).
- A backend switch exists in one place; the SYCL path is not a fork.
- **Gate:** a `build` record, exit 0, of the project's own build command with the backend enabled.
- **Common false positive:** a test binary compiled. That is L0. The artifact must be the one the
  project ships.

### L2 — runtime-live
> "…and the project's own test suite passes on Intel GPU."

- The device runtime layer is implemented (device/context/queue/stream/event/sync, USM allocation
  and copies) and the project's **own dispatch path** reaches SYCL kernels.
- The project's **own** test suite runs with the backend selected — not the migrated-kernel unit
  tests.
- **Gate:** an `integration` record (`--label l2-suite`), exit 0, of the project's own test command.
  A partial pass counts **only** with every failure enumerated in `gates.L2.exceptions` with a
  reason.
- **Common false positive:** dispatch macros, a device-type enum and a CMake module committed, but
  no host code compiled and no test run. Infrastructure is not a runtime.

### L3 — workload-correct
> "**<project> runs on Intel GPU**, producing correct results on <workload>."

- The project's **real entrypoint** completes a **real workload** on the Intel GPU, and the result
  matches a reference within a tolerance the project itself defines.
- Every resolved e2e case passes (config `benchmark.e2e_cases` if non-empty, else `project.json`
  `e2e_cases`).
- **Gate:** one `e2e` record per case, exit 0, each carrying `--reference` and `--tolerance`.
- **This is the level that licenses the headline claim.** Below it, "runs on Intel GPU" is false.
- **Common false positives:** (a) the workload ran and nothing was compared — liveness is not
  correctness; (b) the production entry path was bypassed by a hand-built driver loop; (c) one case
  of several passed.

### L4 — workload-fast
> "…and is X.X× faster end-to-end than the initial SYCL baseline."

- L3 plus a measured end-to-end **baseline → final** comparison on the real workload, both records
  sharing an env fingerprint, taken after the `optimize` phase.
- **Gate:** `profile/e2e.baseline.json` and `profile/e2e.final.json` each backed by e2e records, on
  the same env fingerprint (this is check E7).
- Set at `done`, not at `integrate`.
- **Note on the comparison:** the denominator is the *SYCL baseline*, not CUDA. Any claim relative
  to an NVIDIA GPU requires an actual A/B measurement and must say which GPU, which driver, which
  workload — otherwise do not make it.

---

## Choosing a target

Set `target_level` at `scaffold`, from the archetype and what the project's structure allows:

| Situation | Target |
|---|---|
| Archetype A/B and a backend seam already exists or is cheap | **L4** |
| Archetype C/F — framework extension or graph engine, single-device | **L3** |
| Archetype D — Python array/JIT library with a huge own test suite | **L3**, with the suite's long tail explicitly out of scope |
| Archetype E — the project *generates* device code at runtime (a codegen backend is a separate systems project) | **L1–L2**, with the codegen backend as the residual package |
| Any archetype where a prerequisite is broken (a segfaulting kernel on the required path, a missing numeric fix) | fix the prerequisite first; it blocks the level, it does not lower it |
| Multi-GPU / collectives required by the workload | one level lower unless oneCCL is in scope |

Rules:
- **Never target below L1 without a recorded blocker.** L0 is an outcome, not a plan.
- **Lower the target early and loudly**, at `scaffold`, when the skeleton will not stand. Lowering it
  at report time is how a programme ends with seven silent L0s.
- `config.integration.target_level`, if set, overrides this judgement (the user is asserting scope).

---

## Claim rules (what the report may say)

| Claim | Minimum level |
|---|---|
| "N kernels migrated and validated on Intel GPU" | L0 |
| "builds with a SYCL/XPU backend" | L1 |
| "the project's tests pass on Intel GPU" | L2 |
| "**runs on Intel GPU**" / "supports Intel GPU" / "X now works on Arc" | **L3** |
| "X.X× end-to-end speedup" | **L4** |
| "faster than / comparable to <NVIDIA GPU>" | L4 **plus** an actual A/B record on both devices |

The report's integration section is **generated from `readiness.json`**, not written as prose, for
the same reason the known-issues section is: prose goes stale the moment a gate changes, and a stale
overclaim is the most expensive kind of error in this whole workflow.

---

## `readiness.json` shape

```jsonc
{
  "schema": "sycl-agent/integration-readiness",
  "archetype": "A",                       // A..F, see integration-archetypes.md
  "archetype_reason": "single C driver (train_gpt2.c) + per-kernel headers; backend seam exists",
  "target_level": "L4",
  "target_reason": "backend seam already present; every prerequisite kernel is green",
  "achieved_level": "L3",                 // COMPUTED from gates with records — never asserted
  "entrypoint": "./train_gpt2_sycl",
  "backend_switch": "make BACKEND=sycl  (llmc/backend.hpp selects the kernel headers)",
  "gates": {
    "L0": { "status": "pass", "evidence": ["<test run ids>"], "note": "19/19 kernel families" },
    "L1": { "status": "pass", "evidence": ["build-backend-2026...-a1b2c3"] },
    "L2": { "status": "pass", "evidence": ["integration-suite-2026...-d4e5f6"],
            "exceptions": [] },
    "L3": { "status": "pass", "evidence": ["e2e-baseline-2026...-99aa11"],
            "reference": "loss curve vs gpt2_124M_debug_state.bin", "tolerance": "|Δloss| < 1e-3" },
    "L4": { "status": "pending", "evidence": [] }
  },
  "waivers": [
    { "id": "nvonly-optix", "what": "OptiX ray-tracing path", "reason": "NVIDIA-only API, no SYCL equivalent",
      "blast_radius": "hardware ray tracing disabled; CPU/compute fallback used", "level_impact": "none" }
  ],
  "residual": [
    { "surface": "hostlib", "what": "batched/strided-batched GEMM via oneMKL", "unlocks": "L4 on the batched path",
      "effort_weeks": [1, 2], "risk": "low" }
  ],
  "updated_at": "…"
}
```

`gates.<L>.status` is one of `pass | partial | fail | pending | waived`. Only `pass` (and, for L2,
`partial` with a non-empty `exceptions` list) counts toward `achieved_level`, and **only if
`evidence` resolves to a record with exit code 0**.

---

## Residual work package

Everything not reached is written down as a scoped package. This is the difference between a
credibility risk ("they hid it") and a business outcome ("here is the next engagement").

Per entry: **surface**, **what** (one line of concrete work), **unlocks** (which level or capability),
**effort_weeks** (a range, for one experienced SYCL/C++ engineer, assuming the migrated kernels stay
as-is and GPU CI is available), **risk** (`low|medium|high` — the chance the estimate is materially
exceeded), and any **prerequisite** (e.g. "fix kernel t021 first — it segfaults on the backward path").

Effort rubric (calibrated against completed integrations, use as a starting point and adjust for the
project's actual size):

| Shape of the work | Range | Risk |
|---|---|---|
| Backend seam exists; build a target and run the workload | 1–2 wk | low |
| CMake option + resource manager over `sycl::queue`/USM; reuse the existing test suite | 3–4 wk | low |
| Graph/parser factory wiring for an existing layer registry, single-device | 4–5 wk | medium |
| Host `icpx` sweep over a large library (100+ TUs) + a `DeviceAPI` implementation | 4–6 wk | medium |
| Framework custom-op extension (PyTorch XPU): build + `torch.ops` bindings + allocator interop | 5–7 wk | medium |
| Same, plus an unported second kernel source (e.g. a Triton set) or a numerically broken kernel | +3–4 wk / +1–2 wk | high |
| Python array/JIT library: module-level dispatch + memory pool + stream shim + own pytest suite | 6–8 wk | high |
| Fused-MLP / tensor-core-oriented dependency re-expressed on XMX | 8–10 wk | high |
| Memory-manager replacement + host-level Thrust/CUB removal across a very large library | 12–16 wk | high |
| A new device backend for a runtime that JIT-generates kernel source | 12–16 wk | high |

Add the prerequisites separately; do not fold them into the surface estimate.

---

## Why the ladder, and not a percentage

A percentage ("83% integrated") averages incomparable things and always rounds toward the flattering
reading. A level is a **claim licence**: it says exactly which sentence is now true, and each one is
falsifiable by re-running a single recorded command. That is the property the report needs, because
the first thing a maintainer or a customer does is try to run it.
