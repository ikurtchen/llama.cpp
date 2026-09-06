---
name: sycl-review
description: >
  Adversarial phase-boundary review for a sycl-agent run: verify that a phase actually happened and
  that its exit criteria are met before the workflow is allowed to advance. Runs the mechanical
  phase gate (`evidence.sh gate <phase>`, checks E15/E16/E17), then adjudicates what a machine
  cannot: whether a waiver's reason is real or a restatement of "it was hard", whether the CPU
  oracle is genuinely independent of the SYCL under test, whether a tolerance was loosened until it
  passed, whether a "100%" is a denominator artifact of a re-cut inventory, and whether the report's
  prose outruns the Backend Readiness Level. Produces a verdict (accept / accept-with-findings /
  reject) with findings and drives the refine-then-re-gate loop. Use when: closing a workflow phase,
  before writing FINAL_REPORT.md, when a run looks finished but under-evidenced, when a phase may
  have been skipped, when auditing a migration result. Never migrates or rewrites — it reports and
  blocks.
---

# SYCL Phase Review (did this actually happen?)

Every other check in this framework asks the same shape of question: *you claimed X — show the
record.* Each one fires on **a sentence someone wrote**. None of them can fire on a sentence nobody
wrote, and the most expensive failures in this framework's history were **omissions**, not lies.

> **The failure this skill exists to prevent.** A run migrated 19 kernel families, passed 20/20
> regression tests, and had **19/19 claims mechanically evidence-verified** — the cleanest evidence
> record in the whole programme. It also deferred the entire optimization phase and never once ran
> the project end-to-end. Nothing in the audit noticed, because a phase that is skipped makes no
> claim to falsify. Silence was free.

This skill makes silence expensive.

## Hard rules

1. **The mechanical gate runs first, and you may not override it.** `evidence.sh gate <phase>`
   decides whether the criteria are met. Your judgement adds findings on top; it never converts a
   FAIL into a pass. If you find yourself constructing an argument for why a failed criterion is
   acceptable here, you are writing a waiver — write it as one, with a blast radius.
2. **Review with fresh context.** Run as a subagent that reads only state + evidence, not as a
   continuation of the conversation that did the work. The agent that skips a phase is the agent at
   hour five, with a full context window and a result that already looks finished. A reviewer that
   inherits that context inherits the same blind spot.
3. **You report; you do not repair.** A reviewer who fixes what they find becomes a co-author and
   stops being a check. Hand findings back; let the main agent fix and re-gate.
4. **Absence is a finding.** "There is no optimization record for the kernel that is 61% of
   wall-clock" is a stronger finding than anything you will say about the records that do exist.
   Look first at what is *missing*, then at what is present.
5. **A waiver needs a reason and a blast radius.** "Deferred", "out of scope" and "time-boxed" are
   not reasons — they are restatements of the fact that it did not happen. A reason names the
   obstacle ("unitrace segfaults on this driver, see run id X"); a blast radius names what the
   reader now does not know ("no kernel is profiled, so every performance statement is a bandwidth
   estimate, not a measurement").
6. **Never loosen a gate to make a phase close.** Tolerance, hotspot threshold, target level and
   inventory granularity are inputs, not dials. Moving one to reach a pass is the single most
   damaging thing that can happen in this workflow, because it destroys the comparability of every
   number that came before it.
7. **Route execution through `run.sh`.** A review that re-runs anything re-runs it on the same
   machine, through the same runner, and records it like any other run.

## Procedure

### R1 — Run the mechanical gate

```bash
.sycl/scripts/evidence.sh gate <phase>          # criteria + evidence checks; records the attempt
```

Read the output before reading any state. It tells you what is already known to be wrong, so you
spend your judgement on what is not. Note `gate.attempts`: a phase passing on attempt 3 was refined
twice, and the two things that had to be fixed are usually where the interesting findings are.

### R2 — Ask the questions the gate cannot

The gate checks that a thing **exists**. You check whether it means what it appears to mean. Work
through [references/review-checklist.md](references/review-checklist.md) for the phase; the
cross-cutting ones are always in play:

| # | Question | The failure it catches |
|---|---|---|
| 1 | Is the reference **independent** of the code under test? | A CPU oracle derived from the SYCL implementation validates a typo against itself. Self-consistency is the weakest possible oracle and it caps what may be claimed. |
| 2 | Was the tolerance **inherited**, or chosen after seeing the error? | `--tolerance-source` exists precisely so this is answerable. A tolerance set to just above the observed error is a pass that means nothing. |
| 3 | Does the denominator still mean what it meant at inventory? | A "100%" produced by re-cutting the inventory coarser is not a completion; it is arithmetic. Check `phases.json scope`. |
| 4 | Is the unit of the headline number the same across items? | Kernel families, JIT templates and concrete kernels do not add up. If they were summed, say so. |
| 5 | Does the prose outrun the evidence? | "Runs on Intel GPU" needs L3; "N× faster" needs L4; any NVIDIA comparison needs an A/B record. |
| 6 | Is a passing test actually **exercising** the new path? | A test that would pass with the backend disabled tests nothing. Ask what it would do if you deleted the SYCL kernel. |
| 7 | Is a failure recorded as a failure? | A known-wrong kernel carried quietly to `done` is worse than a missing one, because it will be trusted. |

### R3 — Classify each finding

| Severity | Meaning | Effect |
|---|---|---|
| `blocker` | The phase's stated outcome is not true. | Verdict `reject`. The phase does not close. |
| `concern` | True but weaker than it appears; a reader would draw a wrong conclusion. | Verdict `accept-with-findings`. Must be reflected in the report. |
| `note` | Worth knowing, changes nothing. | Recorded only. |

Write findings into `state/phases.json → phases.<phase>.review`. Be specific: name the id, the file
and the run id. "Coverage looks thin" is not a finding; "k007 is 22% of wall-clock and has no
optimization record" is.

### R4 — Refine, then re-gate

A failed gate is not a verdict on the run, it is a work item. For each unmet criterion, exactly one
of:

- **fix it** — do the missing work, then re-run the gate; or
- **waive it** — add to `phases.<phase>.waivers` with `criterion`, `reason`, `blast_radius`, and the
  run id of the failed attempt where one exists; then re-run the gate.

Then re-run `evidence.sh gate <phase>`. Repeat until it passes or `max_gate_attempts` is reached.

> **When the limit is reached, stop.** Record the remaining unmet criteria as a blocker, set the
> phase to `waived` with its waivers, and continue the workflow. A gate loop that never terminates
> burns the budget the remaining phases need, and "we spent it all trying to close phase 4" is a
> worse outcome than a phase honestly marked incomplete. Escalating is a legitimate exit; looping is
> not.

### R5 — Record the verdict

```jsonc
"review": {
  "verdict": "accept-with-findings",
  "reviewed_at": "2026-08-15T04:10:00Z",
  "reviewer": "subagent",
  "findings": [
    { "id": "F1", "severity": "concern",
      "what": "k007 is 22% of e2e wall-clock but its only trial reverted; the report calls the phase 'complete'",
      "resolution": "accepted-risk",
      "evidence": ["bench-k007-20260814T221000Z-9f2a01"] }
  ]
}
```

`reject` means the phase does not close and the main agent goes back to work. Say what would change
the verdict — a reviewer who rejects without naming the remedy has produced an obstacle, not a
review.

## What a good review is not

| Anti-pattern | Why it fails |
|---|---|
| Re-reading the code for style | The gate is about whether the work happened, not how it reads. |
| Approving because the numbers look plausible | Plausibility is what the evidence model exists to replace. |
| Rejecting for missing polish | `note`-level findings do not block phases. |
| Rewriting the work to fix a finding | You become the author; the check disappears. |
| Accepting "deferred" as a reason | It is the restatement of the problem, not its cause. |
| Reviewing only the phase's own artifacts | Most real findings are cross-phase: a `migrate` that never landed in the dispatch path only shows up as an `integrate` surface nobody opened. |

## Honest limitation

A reviewer drawn from the same model as the author shares its blind spots, and can be argued into
"the deferral was reasonable here" by the same reasoning that produced the deferral. That is exactly
why the mechanical gate is authoritative and this skill is the **second** net: E15/E16/E17 catch the
omissions deterministically and cannot be persuaded; this skill catches the judgement calls that
survive them. If you ever find the two disagreeing, the gate wins.

## References

- [references/phase-exit-criteria.md](references/phase-exit-criteria.md) — every phase's criteria,
  what each one is protecting against, and how to satisfy it.
- [references/review-checklist.md](references/review-checklist.md) — the per-phase adversarial
  questions, and the evidence that answers each.
