# Review checklist

What to ask at each boundary **after** `evidence.sh gate <phase>` has passed. The gate proves an
artifact exists; these questions ask whether it means what it appears to mean. Every one of them
comes from a result that was mechanically clean and substantively weaker than it looked.

Findings go to `state/phases.json → phases.<phase>.review.findings` with a severity
(`blocker` / `concern` / `note`). Name the id, the file and the run id — a finding a reader cannot
locate will not be acted on.

---

## Cross-cutting (ask at every boundary)

- **What is missing?** Read the inventory and the surface index before reading any result. The
  strongest findings are absences, and absences do not appear in the artifacts you are handed.
- **Would this pass with the backend turned off?** If yes, it is not testing the backend. This one
  question retires a surprising number of green test suites.
- **Who chose the number?** A tolerance, a threshold or a target chosen after seeing the result is
  not a criterion, it is a rationalisation. `--tolerance-source` is the audit trail.
- **Does the unit hold?** Kernel families, JIT templates and concrete kernels are different units.
  If a total sums them, the total needs a sentence saying so.
- **Is the run's own history consistent?** `gate.attempts > 1` means something had to be fixed. What
  was it, and was it fixed or waived?

---

## detect

- Is the named entrypoint the **real** one users invoke, or a convenience script that will not
  exercise the dispatch path?
- Does the archetype match what the repository actually is? An archetype chosen optimistically
  (a JIT-codegen runtime classified as a library with a backend seam) sets a target level the
  project cannot reach, and every later phase inherits the error.

## inventory

- Are there CUDA sources **outside** the enumerated set? Ask specifically about kernels embedded as
  strings in `.py`/`.cpp` files — a text-search inventory does not see them, and they will not
  appear as a gap later either.
- Is anything counted at a different granularity from its neighbours?
- Is the frozen denominator the one the report will eventually divide by?

## scaffold

- Does the walking skeleton go through the project's **own** dispatch, or a test harness that calls
  the kernel directly? The second proves nothing about the first.
- Is the target level justified by the archetype, or by optimism?
- If L1 passed: did it build the **whole** project, or only the new SYCL subtree? A subtree that
  compiles under `icpx` says nothing about the several hundred host translation units that have
  never seen the compiler.

## migrate

- Pick the two highest-risk kernels and read the SYCL. Does it implement the algorithm, or a
  simplified case that satisfies the test's shapes?
- Is the reference oracle independent of the implementation under test?
- Are the test shapes representative, or all small/aligned/power-of-two?
- Is a kernel marked `migrated` actually **reachable** from the project, or does it live in a
  self-contained subtree the project never loads? If unreachable, it belongs to an open `dispatch`
  surface — say so.

## integrate

- Does the workload go through the production path, or a documented manual fallback that bypasses
  the parser/factory/dispatcher the real path uses? A fallback is a legitimate result and an
  illegitimate silence.
- Is `import <pkg>` actually tested, or is the packaging surface closed on inspection alone?
- Do the waivers describe **blast radius**, or only scope?

## profile-e2e

- Is the profiled workload the one the report will talk about?
- Was attribution measured, or estimated from kernel benchmarks and presented as e2e share? The
  `method` field must say which.
- Does the ranking cover enough of total wall-clock to be a basis for prioritisation? If the top
  entries sum to 30%, the profile is not telling you where the time goes.

## optimize

- For each hotspot: was a hypothesis formed from the **diagnosis**, or was a checklist applied?
- Do the kept trials' speedups add up to the reported e2e improvement? If kernel time fell 40% and
  e2e fell 2%, the kernel was not the bottleneck and the profile mis-ranked it — that is a finding
  about the profile, not the optimization.
- Was anything kept without a post-change correctness run?
- Is `stop_reason` a real criterion (`target-met`, `diminishing-returns`, `checklist-exhausted`) or a
  euphemism for running out of room?

## done

- Does `achieved_level` follow from records, or from the plan?
- Is every known-broken kernel visible at this level, or has one been quietly folded into a count?
- Does the residual work package let someone else pick this up — surface, effort, what it unlocks?

## report

- Read the report **alone**, as a reader with no access to the state. What would you conclude? Now
  compare that to `readiness.json`. The gap between the two is the review's most valuable output.
- Does every headline number carry a run id?
- Are the skips presented with the same prominence as the successes?
- Does a percentage anywhere use a denominator that changed mid-run?
- Is there a sentence a journalist could quote that the evidence does not support? That is the
  sentence to fix.
