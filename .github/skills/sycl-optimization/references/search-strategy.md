# Optimization Search Strategy — finding good combinations

The optimization space is combinatorial and the effects interact, so exhaustive search is infeasible
and blind stacking hides regressions. This is the disciplined middle path: a **profile-guided greedy
search with combination trials and checkpoint branching**, all tracked in git and the trial log.

> **This file is the *procedure*; the catalog is the *data*.** `sycl-reference` →
> `references/sycl-optimization-catalog.md` supplies *what* to try for a given profiling signal
> (triggers, priority, and the concrete synergy/conflict/enabling edges). This file supplies *how* to
> drive the loop with that data: greedy → combination → branching, the multi-shape definition of
> "faster", the correctness gate, and the stop criteria. Read the catalog to pick candidates; read this
> to sequence and gate them. Do not duplicate the catalog's per-strategy pairs here — cite them.

## Why not just try everything?
- N optimizations → 2^N combinations; each needs a build + correctness + benchmark. Infeasible.
- Optimizations interact: two individual wins can conflict (e.g. large WG + high register pressure →
  spills), or compound (coalescing + vec4). You cannot infer the combination from the parts.
- Blind stacking makes regressions unattributable. One-change-at-a-time keeps causality clear.

## What "faster" means here
Every build → test → **bench** runs the standalone **kernel-level benchmark** (`bench_<id>`) across its
representative shape set (model configs / general / edge) — never the full e2e workload. A change is a
**win only if it holds across the shape set**, not just on one favorable size: a config that helps the
large model shape but regresses the edge/tiny shapes is a lateral move, not a win. When shapes disagree,
prefer the change that helps the shapes the workload actually uses and does no harm elsewhere; log the
per-shape split in the trial.

## The recipe

### Phase 1 — Greedy single-optimization pass
1. From the profiler's bottleneck class, list applicable candidate optimizations (catalog order).
2. Apply each, one at a time, from the current best commit:
   - build → test (gate) → bench.
   - **keep** (git commit) if faster and correct; **revert** otherwise, logging the root cause.
3. Result: a chain of individually-winning commits and a set of known-neutral/known-bad changes.

> **Don't run blind on a stale diagnosis.** A single greedy pass reuses one diagnosis across many
> trials — correct, but the limiter can move under you. Alongside the condition-based re-diagnosis
> triggers in `SKILL.md`, apply a **deterministic cadence fallback**: force a re-diagnosis after **5
> trials since the last profile** or **3 consecutive rejects**, whichever comes first (and always after
> a checkpoint-branch restore). Use the cheap `metrics`-only pass for these forced checks; escalate to a
> full deep profile only if the limiter actually shifted.

### Phase 2 — Combination trials
1. Pick 2–3 kept winners that plausibly interact (same resource: memory, registers, ILP).
2. Apply them together as a single `combination: true` trial from the best commit.
3. Keep only if the combination beats the best single result. Interactions are often **sub-additive**
   (don't expect the product of speedups) and occasionally **super-additive**.
4. Do not combine a memory-bound fix with an unrelated compute-bound fix unless the kernel is
   genuinely balanced (AI near the ridge point).

### Phase 3 — Checkpoint branching (escape local optima)
1. When greedy + combination plateau, `git branch opt/<id>/<strategy>` from the best commit.
2. Explore a *different strategy family* (e.g. switch from scalar+unroll to float4+dot, or from
   hand-rolled reduction to `reduce_over_group`, or offload to oneMKL/XMX).
3. Benchmark the branch tip; keep the branch (merge/reset best to it) only if it beats the current
   best. Otherwise discard the branch and keep the previous best.
4. This lets you leave a local optimum without losing the known-good kernel.

### Simplicity criterion
At equal performance, prefer the simpler/more-maintainable kernel. Reject small gains (< ~2%) that add
significant complexity. You may keep a neutral change if it is simpler and opens new optimization
paths — but never keep a regression.

## Decision table
| Situation | Action |
|-----------|--------|
| Faster + correct | KEEP (commit) |
| Faster + incorrect | REVERT (correctness gate) |
| Neutral + simpler | may KEEP (simplicity) |
| Neutral + more complex | REVERT |
| Slower | REVERT + log root cause (candidate anti-pattern) |
| Plateau after greedy+combination | checkpoint-branch a new strategy family |
| All remaining ideas break correctness | stop: `correctness-block`, send back to migrate |

## When to stop (and say why)
- `target-met`: ≥ roofline expectation (e.g. 60–70% of the relevant peak).
- `diminishing-returns`: k consecutive trials each < ~2% improvement.
- `checklist-exhausted`: no untried applicable strategy remains.
Always write `stop_reason` and the achieved fraction of peak into `optimization/<id>.json`.

## What to log per trial
hypothesis · bottleneck class · commit · build ok/fail · correctness pass/fail · metric before/after ·
decision keep/revert · root cause (especially for reverts — these grow the anti-pattern catalog).
