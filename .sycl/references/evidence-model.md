# Evidence Model

`.sycl/state/*.json` records **what the agent concluded**. This document defines the second half:
**why anyone should believe it.** Every verdict and every number the agent writes must be traceable
to raw, immutable, machine-checkable evidence produced by an actual command on an actual machine.

> **The rule: no claim without a record.**
> `unit_test.result: "pass"` is an *assertion*. A run record holding the exact command, the raw
> stdout, the exit code, the tolerance, the measured error, the git commit and the machine it ran on
> is *evidence*. State may only assert what a record proves.

This exists because a migration report is presented to people who did not watch it happen. They will
ask exactly four questions, and each maps to one evidence class:

| The question a reviewer asks | Evidence class | Where it lives |
|---|---|---|
| "How do I know the tests really passed?" | **correctness** | `.sycl/reports/evidence/test/<kernel>/` |
| "How do I know these benchmark numbers are real?" | **performance** | `.sycl/reports/evidence/bench/<kernel>/`, `…/e2e/<label>/` |
| "How do I know the optimization went the right way?" | **diagnostic** | `.sycl/reports/opt_<id>/{baseline,final}/` + `DIAGNOSIS.md` |
| "How do I know the project actually **runs** on the Intel GPU?" | **integration** | `.sycl/reports/evidence/build/<subject>/`, `…/integration/<subject>/`, `…/e2e/<label>/` |

The fourth is the one a kernel-level migration cannot answer. Its records are what turn a Backend
Readiness Level from an assertion into a computation: L1 needs a `build` record of the **project's
own** build command, L2 an `integration` record of the **project's own** test suite, L3 an `e2e`
record of the **real entrypoint** compared against a reference.

## Layout

Evidence lives under `.sycl/reports/` — the tree the runner already treats as
runner-produced and protected (never pruned by a push, always pulled back after a remote command),
so evidence captured on the GPU host arrives locally by itself.

```
.sycl/reports/
├── FINAL_REPORT.md              # the deliverable (a narrative; cites evidence, is not evidence)
├── AUDIT.md                     # GENERATED evidence index: every claim -> its record -> its raw log
├── evidence/
│   ├── manifest.json            # GENERATED index of every artifact: sha256, bytes, producer, run_id
│   ├── env/
│   │   └── env-<sha8>.json      # machine fingerprint (GPU, driver, icpx, clocks, freq_pinned)
│   ├── test/<kernel-id>/
│   │   ├── <run-id>.json        # run record: cmd, exit code, tolerance, measured error, commit
│   │   └── <run-id>.log         # RAW stdout+stderr, byte for byte
│   ├── bench/<kernel-id>/
│   │   ├── <run-id>.json        # run record: per-shape stats + raw samples, warmup/iters, env ref
│   │   └── <run-id>.log
│   ├── e2e/<label>/             # label = baseline | final | case-<name>
│   │   ├── <run-id>.json
│   │   └── <run-id>.log
│   ├── build/<subject>/         # the PROJECT's own build with the SYCL backend enabled (the L1 gate)
│   │   ├── <run-id>.json        #   subject = backend | <component>; a test binary does NOT count
│   │   └── <run-id>.log
│   └── integration/<subject>/   # runtime smoke + the PROJECT's own test suite (the L2 gate)
│       ├── <run-id>.json        #   subject = runtime | suite | <surface>
│       └── <run-id>.log
└── opt_<kernel-id>/
    ├── baseline/                # raw deep profile BEFORE optimization (counters, stall CSV, asm)
    ├── final/                   # raw deep profile AFTER  optimization (same lenses)
    └── DIAGNOSIS.md             # GENERATED before/after delta: did the metric we blamed actually move?
```

**Nothing in `evidence/` is ever edited or overwritten.** Records are append-only and named by run
id; a re-run produces a new record, so a regression is visible as history rather than erased.

## Run records

A run record is the atomic unit of evidence: one command, one machine, one commit, one verdict.
Schema: `.sycl/schemas/run-record.schema.json`. It is produced by
`.sycl/scripts/evidence.sh record`, never hand-written.

```jsonc
{
  "schema": "sycl-agent/run-record",
  "run_id": "test-layernorm_001-20250806T142233Z-9f3a1c",
  "class": "test",                       // test | bench | e2e | build | integration | profile
  "subject": "layernorm_001",            // kernel id, or e2e case label
  "label": "post-opt-trial-3",           // which point in the story this run belongs to
  "phase": "migrate",
  "ts": "2025-08-06T14:22:33Z",
  "duration_s": 12.4,

  "command": "./dev/sycl/layernorm 3",   // EXACT command, copy-pasteable to reproduce
  "via": "run.sh test",                  // how it was routed
  "runner": {"mode": "remote", "host": "10.239.98.41", "workdir": "/workspace/proj"},
  "env_fingerprint": "env-4b1e7a2c",     // -> evidence/env/env-4b1e7a2c.json
  "commit": "a1b2c3d", "dirty": false,   // exact code state under test

  "exit_code": 0,                        // the machine's verdict, not the agent's
  "verdict": "pass",
  "log": "evidence/test/layernorm_001/test-…-9f3a1c.log",
  "log_sha256": "…", "log_bytes": 4213,
  "artifacts": [{"path": "…", "sha256": "…", "bytes": 1024, "kind": "csv"}],

  "measurements": {                      // parsed from the log, or annotated by the agent
    "tolerance": {"rtol": 1e-5, "atol": 1e-6},
    "max_abs_err": 3.8e-7, "max_rel_err": 1.1e-6,
    "shape": "B=8,T=1024,C=768", "dtype": "fp32", "seed": 137,
    "reference": "cpu-oracle (fp64 accum, dev/common.hpp:validate_result)"
  },
  "extracted": { "...": "auto-parsed key/values, kept verbatim" }
}
```

Three fields carry the audit weight and may never be synthesised:

- **`exit_code`** — the verdict comes from the process, not from the agent's reading of the output.
  `verdict: "pass"` requires `exit_code == 0`; an agent cannot talk its way past a non-zero exit.
- **`log_sha256`** — the raw log is hashed at capture. `evidence.sh verify` re-hashes it, so a record
  whose log was later edited (or lost in a sync) is detected, not trusted.
- **`commit` + `dirty`** — the code state that produced the number. A benchmark captured with a dirty
  tree is flagged: nobody can reproduce it.

### Why raw logs are kept, not summaries
A summary is the agent's interpretation, and the interpretation is exactly what is in question. The
log is the primary source: a reviewer reads the actual "PASS, max abs err 3.8e-7 (tol 1e-6)" line
that the harness printed. Logs are small (KBs); the audit value is disproportionate.

## Correctness evidence — "did the tests really pass?"

A passing test is only meaningful if a reviewer can see **what was compared, against what, and how
closely**. Every migrated kernel therefore carries a test record that answers all five:

| Question | Field |
|---|---|
| What ran? | `command`, `commit`, `runner` |
| What was it compared against? | `measurements.reference` (CPU-oracle description + source path) |
| On what inputs? | `measurements.seed` / `shape` / `dtype` — enough to regenerate them exactly |
| How close did it get? | `measurements.max_abs_err`, `max_rel_err` |
| Was that close enough? | `measurements.tolerance` — the **inherited** tolerance, plus its source |
| Did the machine agree? | `exit_code`, raw log |

The measured error next to the tolerance is the single most valuable line in the whole audit trail:
it converts "PASS" into "PASS with 3× margin" or "PASS with 2% margin", and the second one is worth
a conversation. `evidence.sh verify` refuses a `pass` verdict whose record has a non-zero exit code,
and flags a `pass` whose `max_abs_err` is within 10% of `atol` as `margin-thin`.

Required at: every `migrate` unit test, every `optimize` correctness re-check (step 3 of the loop),
and the `integrate` end-to-end accuracy gate.

## Performance evidence — "are the benchmark numbers real?"

A benchmark number is worthless without its conditions. Every bench/e2e record pins them:

- **Raw samples, not just a summary.** `measurements.samples_ms` keeps every timed iteration, so a
  reviewer can recompute the median, see the spread, and spot a bimodal or drifting run that a lone
  median hides. `median_ms` / `p90_ms` / `stddev_ms` / `min_ms` are derived and stored beside it.
- **The machine state.** `env_fingerprint` points at a snapshot of the GPU, driver, icpx version and
  **`freq_pinned`**. An unpinned-clock benchmark is not evidence of an optimization — `verify` marks
  every such record `unstable` and `AUDIT.md` renders the warning next to the number.
- **Both endpoints on the same footing.** Baseline and final must share an env fingerprint and the
  same shape set, else the comparison is between two different experiments. `verify` reports a
  baseline/final env mismatch as a failure.
- **Shape coverage.** `measurements.shape` is per-record; a kernel's speedup claim must be backed by
  records covering the model/general/edge shape set, not one flattering size.
- **Reproduction.** `command` + `commit` + `runner` is a copy-pasteable recipe. The reviewer's real
  test of a benchmark is running it again.

## Diagnostic evidence — "did the optimization go the right way?"

This is the question raw CSVs cannot answer on their own. The optimization loop already retains a
full deep profile at two points (`opt_<id>/baseline/` and `opt_<id>/final/`), but a reviewer will not
diff two 400-column unitrace CSVs by hand. `DIAGNOSIS.md` (generated by
`evidence.sh diagnosis <kernel-id>`) turns them into a falsifiable claim check:

```markdown
## layernorm_001 — claim: memory-bound, dominated by SendStall (uncoalesced loads)
Baseline  a1b2c3d   Final  d4e5f6a   Env  env-4b1e7a2c (freq pinned)

| Signal              | Baseline | Final |  Δ     | Direction predicted | Verdict |
|---------------------|---------:|------:|-------:|---------------------|---------|
| Time (median, ms)   |    0.412 | 0.181 | -56.1% | down                | ok      |
| Achieved BW (GB/s)  |      212 |   486 | +129%  | up                  | ok      |
| % of peak BW        |      35% |   80% | +45pp  | up                  | ok      |
| SendStall %         |     61.2 |  18.4 | -42.8pp| down (the blamed metric) | ok |
| EU active %         |     31.0 |  74.5 | +43.5pp| up                  | ok      |
| Register spills     |        0 |     0 |   0    | unchanged           | ok      |
| SbidStall %         |      8.1 |  29.7 | +21.6pp| —                   | **new limiter** |

**Verdict: supported.** The metric named in the diagnosis moved in the predicted direction and the
time moved with it. The kernel is now limited by `SbidStall` (ILP), recorded as the next lead.
```

The point is not the table, it is the **falsifiability**: the optimization stated a hypothesis
("`SendStall` is the limiter"), and the before/after counters either confirm it or expose that the
speedup came from somewhere else. A kernel that got faster while the blamed metric did not move is
flagged `unexplained` — a real and common outcome (noise, a different change, a measurement error)
that should be visible rather than narrated away.

`verify` requires, for every kernel with kept trials: a non-empty `baseline/`, a non-empty `final/`,
and a `DIAGNOSIS.md` whose verdict is `supported` or an explicitly recorded `unexplained`.

## Claims must point at their evidence

Evidence that nobody can find is not evidence. Each state file that makes a claim carries an
`evidence` pointer to the run id(s) backing it, so `AUDIT.md` and `verify` can walk claim → record →
raw log mechanically:

| Claim in state | Backing pointer |
|---|---|
| `kernels/<id>.json` → `unit_test.result` | `unit_test.evidence` (test run id) |
| `kernels/<id>.json` → `performance.baseline` / `optimized` | `performance.evidence.baseline` / `.optimized` (bench run ids) |
| `kernels/<id>.json` → `reference.crosscheck` | `reference.evidence` (test run id) |
| `optimization/<id>.json` → each trial | `trials[].evidence.{test,bench_before,bench_after}` |
| `optimization/<id>.json` → `diagnosis` | `diagnosis.artifacts.{baseline,final,diagnosis_md}` |
| `profile/e2e.baseline.json` / `e2e.final.json` | `evidence` (e2e run ids, one per case) |
| `integration/readiness.json` → `achieved_level` / each passing gate | `gates.<L>.evidence` (L1 → build run id; L2 → integration run id; L3 → e2e run ids; L4 → both e2e snapshots) |
| `integration/index.json` → each `status: done` surface | `surfaces[].evidence` (build / integration / e2e / test run ids) |
| `project.json` → `env` | `env.evidence` (env fingerprint id) |

## The audit tools

```bash
.sycl/scripts/evidence.sh env                          # snapshot the machine; prints the fingerprint id
.sycl/scripts/evidence.sh record test  <kernel> [opts] -- <cmd>   # capture a run + its raw log
.sycl/scripts/evidence.sh record bench <kernel> --shape "M=4096" -- <cmd>
.sycl/scripts/evidence.sh record e2e   baseline -- <cmd>
.sycl/scripts/evidence.sh record build backend --label scaffold-l1 -- <the project's own build>
.sycl/scripts/evidence.sh record integration suite --label l2-suite -- <the project's own test suite>
.sycl/scripts/evidence.sh annotate <run-id> --set max_abs_err=3.8e-7 --set tolerance.atol=1e-6
.sycl/scripts/evidence.sh diagnosis <kernel>           # build opt_<id>/DIAGNOSIS.md from baseline vs final
.sycl/scripts/evidence.sh manifest                     # (re)hash every artifact into manifest.json
.sycl/scripts/evidence.sh verify [--strict]            # mechanical audit; non-zero exit on any gap
.sycl/scripts/evidence.sh audit                        # regenerate .sycl/reports/AUDIT.md
.sycl/scripts/evidence.sh enter <phase>                # stamp phase entry (E16 measures the window)
.sycl/scripts/evidence.sh gate  <phase>                # phase-exit gate; the only way to exit a phase
```

`record` routes through `.sycl/scripts/run.sh` like everything else, so evidence is captured on the
machine that did the work — and, being under `.sycl/reports/`, is pulled back automatically.

### `verify` — the checks
Run at every phase exit and again before `report`. `--strict` turns warnings into failures.

| # | Check | Severity |
|---|---|---|
| E1 | every `unit_test.result: pass` has an existing test record with `exit_code == 0` | fail |
| E2 | every non-empty `performance.baseline` / `optimized` has a bench record | fail |
| E3 | every kept optimization trial has a correctness record **and** a bench record at that commit | fail |
| E4 | `e2e.baseline.json` / `e2e.final.json` each have an e2e record per case | fail |
| E5 | every record's log exists and its sha256 matches the manifest | fail |
| E6 | every kernel with kept trials has non-empty `opt_<id>/baseline/` and `final/` | fail |
| E7 | baseline and final bench/e2e records share an env fingerprint | fail |
| E8 | numbers quoted in `FINAL_REPORT.md` appear in some record | fail |
| E9 | bench records ran with `freq_pinned: true` | warn (`unstable`) |
| E10 | records captured with `dirty: true` | warn (`unreproducible`) |
| E11 | `max_abs_err` within 10% of `atol` | warn (`margin-thin`) |
| E12 | `DIAGNOSIS.md` verdict is `supported` | warn (`unexplained`) |
| E13 | `readiness.achieved_level` is supported by its gates: ≥L1 needs a passing **build** record, ≥L2 a passing **integration** record, ≥L3 an **e2e** record per case with a reference, L4 both e2e snapshots | fail |
| E14 | `FINAL_REPORT.md` makes no claim above the achieved level (e.g. "runs on Intel GPU" without L3, an e2e speedup without L4, a "vs NVIDIA" number with no A/B record) | fail |
| E15 | every phase marked `exited`/`waived` in `phases.json` actually met its exit criteria (and a `waived` phase lists its waivers) | fail |
| E16 | no execution-bearing phase (`scaffold`, `migrate`, `integrate`, `profile-e2e`, `optimize`) was entered and left with **zero run records** in its window | fail |
| E17 | no work item left `kernels/index.json` (or a surface left the integration index) without a matching entry in `phases.json` → `scope.changes` | fail |

E13 and E14 are the pair that make the last mile auditable. E13 stops the *state* from over-declaring
depth; E14 stops the *prose* from doing it anyway. Both are fixed by lowering the sentence or raising
the evidence — never by deleting the level.

**E15–E17 answer a different question from all the checks above them.** E1–E14 each fire on a claim:
you said something, so show the record. That means they are blind to *omission* — a phase that was
skipped makes no claim, so it cannot be falsified, and a real run once passed every check above with
**19/19 claims mechanically verified** while having deferred its entire optimization phase and never
run the project end-to-end. E15 asks *were the phase's criteria met?*, E16 asks *did anything happen
inside the phase at all?* (a crashed profiler quietly turning into a skipped phase), and E17 asks *is
this still the same question we started with?* (an inventory re-cut coarser until nothing is
pending). A run can pass any two and fail the third.

The output is the sentence a reviewer wants: *"14 claims, 14 backed, 0 unbacked, 2 warnings
(1 unstable-clock bench, 1 thin margin) — all log hashes verified."*

### `gate` — the phase-exit contract
`verify` asks *"is what you said true?"*. `gate` asks *"did you do the thing this phase exists to
do?"* — and is the **only** thing that may mark a phase `exited` in `.sycl/state/phases.json`.

```bash
.sycl/scripts/evidence.sh gate optimize
[PASS] optimize.hotspots_identified
[FAIL] optimize.hotspots_accounted — 2 hotspot kernel(s) with no optimization outcome: k001 (no optimization record); k002 (no optimization record)
```

Each phase's criteria are enumerated in skill `sycl-review` →
`references/phase-exit-criteria.md`. A criterion that genuinely cannot be met is not deleted — it is
**waived** in `phases.json` with a `reason` and a `blast_radius`, which converts an invisible skip
into a reviewable decision. Every attempt is counted, so a phase that passed on the third try stays
visibly a phase that was refined twice; at `max_gate_attempts` the agent stops refining and records
the shortfall rather than burning the remaining budget on one boundary.

The criterion that matters most: `optimize` will not close while a kernel worth ≥5% of end-to-end
wall-clock has neither a measured trial nor a recorded reason for having none. **A trial that was
measured, found ineffective and reverted passes** — the criterion is about whether the question was
asked, never about whether the answer was favourable.

### `AUDIT.md` — the human entry point
One page, generated, that a reviewer reads **beside** `FINAL_REPORT.md`: the report says what was
achieved, `AUDIT.md` says where to check it. Every row links a claim to its record and its raw log,
with the verification verdict, so nothing has to be taken on trust.

## What each store proves

| Store | Proves | Can the agent forge it? |
|---|---|---|
| `state/*.json` | what the agent concluded | yes — it is authorship, not evidence |
| `reports/evidence/**/*.log` | what the machine actually printed | no (captured by wrapper, hashed) |
| `reports/evidence/**/*.json` | the conditions of that run | only by not using `evidence.sh` — which `verify` detects as a missing record |
| `reports/opt_<id>/{baseline,final}` | the hardware's view before/after | no (raw tool output) |
| git history | what code produced it | no (commit hashes) |
| `logs/*.jsonl` | when decisions were taken, in order | no (append-only, timestamped) |

The agent's honesty is therefore not a required assumption: a claim without a record is *absent*, and
absence is what `verify` reports.
