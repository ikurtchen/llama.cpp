<!-- Fill this in during the `report` phase from the authoritative state under .sycl/state/ (project.json,
     kernels/index.json + per-kernel details, profile/e2e.baseline.json + e2e.final.json,
     optimization/<id>.json, metrics.json, lessons.md) and the git history. Keep every number traceable
     to state — never invent figures. Delete guidance in <!-- --> comments as you go. Save the finished
     report to .sycl/reports/FINAL_REPORT.md.

     Audience is two-tiered, so the document is too:
       • §1 Executive summary — for a manager/boss: clear objective, persuasive, simple, accurate.
       • §2+ Details — for the team: the why / what / how, reproducible and specific. -->

# <Project> — CUDA/Triton → SYCL Migration & Optimization Report

_<one sentence: what was done and the single headline result, e.g. "Ported the N attention/norm
kernels of <project> to SYCL 2020 for Intel Arc Pro B60 and cut end-to-end GPU time by X.X×.">_

**Date:** <YYYY-MM-DD> · **Target GPU:** <b60|b70|cri> (<arch>) · **Prepared by:** sycl-agent

---

## 1. Executive summary
<!-- For the boss. 4–6 lines, no jargon. Lead with the outcome and the business value. Must be
     accurate and defensible — every claim is backed by §3/§4. -->

- **Objective.** <Why this work existed, in one line — e.g. "Run <project> on Intel GPUs at
  competitive performance, removing the CUDA/NVIDIA dependency.">
- **Result.** <Headline metric, e.g. "End-to-end GPU time improved X.X× (from A ms to B ms) on
  <GPU>; M of N kernels reach ≥P% of hardware peak.">
- **Correctness.** <e.g. "All N migrated kernels match the reference within tolerance; end-to-end
  output validated.">
- **Status.** <Done / done with documented shortfalls.> <If shortfalls, one line naming them.>
- **Bottom line.** <One persuasive sentence a manager can repeat upward.>

| Metric | Baseline | Final | Change |
|--------|---------:|------:|-------:|
| End-to-end GPU time | <A> | <B> | **<X.X×>** |
| Kernels migrated | — | <M>/<N> | — |
| Kernels ≥ target % of peak | — | <k>/<N> | — |
| Accuracy (vs reference) | — | <PASS> | — |

<!-- Cost line for the boss — one sentence, from metrics.json totals. Frame it as the price of the
     result above. Lead with the dollar figure if cost_usd is populated (clearest for a manager); else
     use AI credits / premium requests / tokens. -->
- **Cost to deliver.** <Total agent time T, and $ cost (or AI credits / premium requests / tokens).>

---

## 2. Scope & objectives  (the *why* and *what*)
<!-- For the team. What was in scope and the goals/success criteria. -->

- **Sources migrated:** <CUDA and/or Triton>; <N> kernels enumerated, <M> migrated, <S> skipped,
  <R> needs-reference.
- **Target:** <platform/arch>, <plain-SYCL and/or sycl-tla>. <Why this target style per kernel class.>
- **Success criteria:** correctness within tolerance vs a CPU-oracle reference, and performance judged
  vs the SYCL baseline and the hardware roofline (target ≥ <P>% of relevant peak).
- **Out of scope / deferred:** <skipped kernels and why; deferred sycl-tla re-targets; etc.>

## 3. Results  (the *what*, measured)
<!-- The evidence. Pull straight from state; keep it a scannable table + a few sentences. -->

### 3.1 End-to-end performance
<!-- Application: from profile/e2e.baseline.json vs e2e.final.json; state host-vs-device split if
     relevant. Multiple e2e cases: one row per case (baseline→final) + a geomean. Library
     (basis: kernel-suite): report the per-kernel suite baseline→final and its geomean speedup —
     there is no single workload number. -->
<Baseline A → Final B, X.X× overall. One line on where the time went / what dominated.>

### 3.2 Per-kernel outcomes
<!-- One row per non-skipped kernel. impact% from e2e ranking; speedup + %peak from the kernel detail. -->

| Kernel | Source | Impact % (e2e) | Baseline | Optimized | Speedup | % of peak | Bound | Accuracy |
|--------|--------|---------------:|---------:|----------:|--------:|----------:|-------|----------|
| <id>   | <cuda\|triton> | <p> | <t0> | <t1> | <x.x×> | <q%> | <mem\|compute> | <PASS (tol)> |

### 3.3 Accuracy validation
<!-- How correctness was proven, and the result. -->
<CPU-oracle reference method (live regen from seed+shape+dtype), tolerances used, e2e validation
result. Note any needs-reference kernels.>

## 4. Approach & methodology  (the *how*)
<!-- For the team to trust and reproduce the numbers. Concise, factual. -->

1. **Migration.** Hand-written SYCL 2020 (no dpct/SYCLomatic). <Target-style decisions.> Each kernel
   validated against a CPU-oracle reference before acceptance; every accepted change is a git commit.
2. **Ranking.** End-to-end profiled with unitrace (host + device timing), scoped to representative
   iterations to stay crash-safe; kernels ranked by wall-clock share to set optimization priority.
3. **Optimization.** Highest-impact first, against a **standalone single-kernel benchmark** across
   representative shapes. Each kernel **diagnosed** first (HW metric counters + stall sampling + IGC
   asm) to find the true bottleneck, then **one** change per trial, kept only if it wins across shapes
   (else reverted). Trials recorded in `.sycl/state/optimization/<id>.json`.
4. **Verification.** Correctness re-checked after every optimization; final e2e re-profiled for the
   baseline→final comparison in §3.1.

## 5. Key optimizations  (deep dive — *why each win worked*)
<!-- For the team / knowledge sharing. Pick the top 2–4 kernels by impact. For each: the diagnosis
     (bottleneck + dominant stall + spills), the change applied, and the measured result. This is the
     transferable engineering insight. -->

### <kernel id> — <headline, e.g. "coalesced loads + higher occupancy → 2.3×">
- **Why (diagnosis):** <bound class; dominant stall reason; spills; roofline gap.>
- **What (change):** <the single optimization applied.>
- **How it helped:** <mechanism, tied to the metric that moved.>
- **Result:** <t0 → t1, x.x×, q% of peak.>

## 6. Risks, shortfalls & known gaps
<!-- Honest and specific — builds trust. -->
- <Kernels below target % of peak and why; documented stop_reason.>
- <Skipped / needs-reference kernels and the reason.>
- <Environment gaps (e.g. tool unavailable, freq not pinned) from lessons.md.>

## 7. Lessons learned & recommendations
<!-- What to carry forward. From lessons.md + optimization notes. -->
- <Reusable insight / pattern.>
- <Recommended next step (e.g. sycl-tla re-target of kernel X, tackle deferred kernel Y).>

## 8. Agent efficiency & cost  (what the migration *cost to run*)
<!-- Straight from .sycl/state/metrics.json (regenerated by metrics.sh from the append-only event
     log). Lets a reader judge the agent's ROI: the engineering result above vs the AI time/spend it
     took. Wall-clock time is exact; $/token/credit/request figures are what the engine reported.
     Show the total, then the per-phase breakdown so the expensive stage is visible. -->

- **Total:** <T agent wall-clock> · <$cost_usd if reported> · <tokens_total (in / out / cache r / cache w)> · <requests> premium requests · <credits> AI credits.
- **Where it went:** <one line — e.g. "optimize dominated at ~P% of time and $; migration was N kernels × ~t each.">

| Phase | Time | $ USD | Tokens (in / out / cache) | Premium reqs | AI credits |
|-------|-----:|------:|--------------------------:|-------------:|-----------:|
| detect | <t> | <$> | <in/out/cache> | <r> | <c> |
| inventory | <t> | <$> | <in/out/cache> | <r> | <c> |
| migrate | <t> | <$> | <in/out/cache> | <r> | <c> |
| integrate | <t> | <$> | <in/out/cache> | <r> | <c> |
| profile-e2e | <t> | <$> | <in/out/cache> | <r> | <c> |
| optimize | <t> | <$> | <in/out/cache> | <r> | <c> |
| done + report | <t> | <$> | <in/out/cache> | <r> | <c> |
| **Total** | **<T>** | **<$>** | **<in/out/cache>** | **<R>** | **<C>** |

<!-- Record only the cost dimensions your engine exposes: a direct $ figure (best), token counts
     (split uncached vs cache-read/-write since vendors like Anthropic price them differently), and/or
     GitHub AI credits / premium requests (Copilot). Omit columns that are all-zero. -->

---

## Appendix
- **Environment:** icpx <ver>, oneAPI <ver>, GPU <device>, freq pinned <yes/no>, runner <local|remote>.
- **Tooling:** unitrace (metrics + stall sampling), roofline, CPU-oracle references.
- **Artifacts:** benchmark/profile reports under `.sycl/reports/`; authoritative state under
  `.sycl/state/`; per-kernel optimization logs under `.sycl/state/optimization/`.
- **Audit trail:** git history (one commit per accepted change); `.sycl/logs/` JSONL decision log.
