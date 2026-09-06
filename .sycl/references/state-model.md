# State Model

`sycl-agent` coordinates through files, not chat history. Five stores, each with a distinct job:

| Store | Format | Role | Source of truth? |
|-------|--------|------|------------------|
| `.sycl/state/*.json` | structured JSON | machine-readable progress that drives the workflow | **yes** |
| `.sycl/PROGRESS.md` | Markdown | human-readable dashboard, **generated** from state | no (a view) |
| git history | commits | *what code changed*; enables revert | audit (code) |
| `.sycl/logs/*.jsonl` | JSONL | *decisions, results, timings*; append-only | audit (process) |
| `.sycl/reports/evidence/` | JSON + raw logs | *what the machine actually printed* — hashed run records behind every claim | **audit (results)** |
| `.sycl/state/metrics.json` | structured JSON | **analytics** rollup (efficiency & cost), **generated** from `logs/metrics.jsonl` | no (a view) |

The JSON is authoritative. PROGRESS.md is regenerated from it after every change so the two never
drift. Never hand-edit PROGRESS.md — edit the JSON and run `.sycl/scripts/gen-progress.sh`.

State and evidence answer different questions and neither substitutes for the other: state says the
agent **concluded** a kernel passes at 486 GB/s; evidence says which command was run, on which
machine, at which commit, and what it printed. See `evidence-model.md`.

## Layout
```
.sycl/
├── config.json                     # runner (local|remote) + toolchain + benchmark policy
├── instructions.md                 # OPTIONAL human-authored project overrides (agent reads, never edits)
├── PROGRESS.md                     # GENERATED human dashboard (do not edit)
├── state/
│   ├── project.json                # env, build system, target GPU, current phase
│   ├── phases.json                 # the phase-exit contract: per-phase status, gate result +
│   │                             #   attempts + unmet criteria, waivers, review verdict, and the
│   │                             #   FROZEN scope denominator (kernel/surface ids at inventory exit)
│   ├── kernels/
│   │   ├── index.json              # list: id, source, status, risk, -> detail
│   │   ├── <id>.json               # per-kernel detail (algorithm, IO, test, performance)
│   │   └── <id>/ref/               # golden input/output tensors for accuracy checks
│   ├── profile/e2e.json            # LIVE per-kernel wall-clock share -> optimization priority
│   ├── profile/e2e.baseline.json   # immutable pre-optimization snapshot (audit)
│   ├── profile/e2e.final.json      # immutable post-optimization snapshot (audit)
│   ├── integration/
│   │   ├── index.json              # the last-mile surfaces: build, runtime, memory, dispatch,
│   │   │                         #   hostlib, package, nvonly -- status + what level each blocks
│   │   ├── <surface>.json          # OPTIONAL per-surface detail (plan, files, effort, waiver)
│   │   └── readiness.json          # Backend Readiness Level: declared target vs COMPUTED achieved,
│   │                             #   gate evidence, waivers, residual work package
│   ├── optimization/<id>.json      # trial log: hypothesis, commit, correctness, keep/revert
│   ├── loc.json                    # GENERATED code-volume rollup: CUDA/Triton lines migrated -> SYCL lines
│   ├── metrics.json                # GENERATED analytics rollup (efficiency & cost; do not edit)
│   └── lessons.md                  # durable, provenance-tagged learnings (also shown in PROGRESS)
├── schemas/                        # JSON Schemas validating state/*.json
├── reports/                         # generated benchmark / profiling reports
│   ├── FINAL_REPORT.md              # the deliverable (cites evidence; is not evidence)
│   ├── AUDIT.md                     # GENERATED evidence index: every claim -> its record + raw log
│   ├── evidence/
│   │   ├── manifest.json            # GENERATED sha256 index of every artifact
│   │   ├── env/env-<sha8>.json      # machine fingerprint (toolchain, GPU, clock pinning)
│   │   ├── test/<id>/<run-id>.{json,log}   # correctness runs: exit code + measured error vs tolerance
│   │   ├── bench/<id>/<run-id>.{json,log}  # benchmarks: every timed iteration, not just the median
│   │   ├── build/<subject>/<run-id>.{json,log}      # the PROJECT's own build with the SYCL backend (L1)
│   │   ├── integration/<subject>/<run-id>.{json,log} # runtime smoke + the project's own test suite (L2)
│   │   └── e2e/<label>/<run-id>.{json,log} # workload snapshots (baseline|final|case-<name>)
│   └── opt_<id>/{baseline,final}/   # raw deep profiles + GENERATED DIAGNOSIS.md
└── logs/*.jsonl                     # append-only audit logs: metrics.jsonl (cost/time events),
                                    #   progress.jsonl (gen-progress heartbeat: history + timing cross-check)
                                    #   usage.import.json (last normalized usage export from the engine log)
└── scripts/                         # run.sh, log.sh, metrics.sh, collect-usage.sh, rotate-logs.sh,
                                    #   gen-progress.sh, init-state.sh, evidence.sh, evidence_audit.py,
                                    #   loc.sh, loc.py
```

## Rules
1. **Read `project.json` first** — it holds the current `phase` and the runner/env facts.
2. **Keep detail out of the index.** Per-kernel detail lives in `kernels/<id>.json`; load only what
   you need. This keeps each subagent's context focused.
3. **Regenerate PROGRESS.md** after any state write (`gen-progress.sh`). Consistency is required.
4. **Validate against `schemas/`.** State files should conform; schemas are lean (no version
   lockstep) — treat them as guardrails, not ceremony.
5. **Stamp `updated_at`** (ISO-8601 UTC) whenever you write a state file.
6. **Commit accepted changes** (code + state together) so git history is the code audit trail.
7. **No claim without a record.** Any `unit_test.result`, `performance.*` figure or e2e total written
   to state must carry an `evidence` pointer to a run record captured by `.sycl/scripts/evidence.sh
   record`. Write the pointer in the same edit as the claim. `evidence.sh verify` re-hashes the raw
   logs and reports unbacked claims as **absent** — so the reviewer never has to assume good faith.
8. **Readiness is computed, never asserted.** `integration/readiness.json` `achieved_level` is the
   highest level whose gate — and every gate below it — is `pass` with a resolvable run record. The
   declared `target_level` may be lowered **only** at the moment the blocker is found, with the reason
   recorded; never at report time.
9. **Integration surfaces are never deleted from the index.** A surface the project does not have is
   marked with a reason (`status: waived` or a `notes` line). Deleting it shrinks the denominator and
   turns an incomplete backend into a fake 100% — the same failure as re-cutting a kernel inventory
   under a coarser taxonomy mid-run. Whatever is not done is a `waiver` (with a blast radius) or a
   `residual` item (with an effort range and what it unlocks).
10. **Resumability.** Never rely on in-memory context. A fresh run reconstructs everything from
    `.sycl/state/` + git.
11. **A phase is exited only by its gate.** `phases.json` is written by `evidence.sh enter` and
    `evidence.sh gate <phase>`; nothing else sets a phase to `exited`. `project.json.phase` says
    where the run *is*, `phases.json` says what each phase *delivered* — the second is the one an
    auditor reads, because it is the only place a phase that never happened leaves a mark. A
    criterion that cannot be met is waived in place, with a `reason` and a `blast_radius`.
12. **The denominator is frozen at inventory exit.** `phases.json` → `scope.baseline` records the
    kernel and surface ids the run committed to. Any later removal needs a `scope.changes` entry
    with a `reason`; check **E17** fails otherwise. Re-cutting an inventory at a coarser granularity
    until `pending: 0` is arithmetic, not progress.

## Execution routing
No agent calls `icpx`, tests, `unitrace`, or benchmarks directly. Everything goes through
`.sycl/scripts/run.sh`, which reads `config.json` and runs locally or over SSH on the remote GPU
host (syncing sources first). This keeps the workflow identical whether or not this machine has an
Intel GPU.

## Logging
```bash
.sycl/scripts/log.sh sycl-agent <phase> <info|warn|error> "<message>"
```
JSONL, one object per line (`ts, agent, phase, level, msg`), auto-rotated by size. Log at least:
phase transitions, per-kernel start/finish, test verdicts, benchmark results, optimizations
kept/reverted, and blockers.

## Evidence (results audit)
A log entry saying `test passed` is the agent's own testimony. Evidence is the machine's. Every
command whose outcome is later cited is wrapped:
```bash
.sycl/scripts/evidence.sh env                                       # -> env-<sha8>
.sycl/scripts/evidence.sh record <test|bench|e2e|build> <subject> [opts] -- <command>
.sycl/scripts/evidence.sh diagnosis <kernel-id>                     # -> opt_<id>/DIAGNOSIS.md
.sycl/scripts/evidence.sh manifest && .sycl/scripts/evidence.sh verify
.sycl/scripts/evidence.sh audit                                     # -> reports/AUDIT.md
```
`record` routes the command through `run.sh`, keeps the **unedited log**, and writes a run record
holding the command, exit code, duration, `log_sha256`, commit + dirty flag, and env fingerprint,
plus whatever measurements it can extract from the output (measured error vs tolerance for tests;
every timed iteration for benchmarks). It prints a `run_id`, which goes into the `evidence` field of
the state it justifies.

Three fields carry the audit weight and are never synthesised: the **exit code** (so the verdict
comes from the process, not the agent), the **log hash** (re-checked by `verify`, so an edited log is
detected), and the **commit + dirty flag** (so the code behind a number can be checked out again).

Evidence lives under `.sycl/reports/`, which `run.sh` pulls back from the remote host after every
command and never prunes — so a measurement taken on the GPU machine is auditable locally with no
extra plumbing. Full design, field reference, and the E1–E12 verification checks:
`.sycl/references/evidence-model.md`.

## Code volume (how much was actually migrated)
"We ported the kernels" is not a size. `.sycl/scripts/loc.sh` measures it, so the report can state
**how many lines of CUDA/Triton the migration covered** and **how much SYCL replaced them**:
```bash
.sycl/scripts/loc.sh all             # measure every non-skipped kernel, then roll up
.sycl/scripts/loc.sh kernel <id>     # one kernel (after writing its SYCL)
.sycl/scripts/loc.sh count <file[:120-198|:sym1,sym2]>...   # ad-hoc
.sycl/scripts/loc.sh scan            # project-wide CUDA/Triton/SYCL totals
```
- **Measured at inventory** (source side, from the `source` symbols each analysis skill recorded) and
  **refreshed after migrate** (SYCL side, from `sycl_impl`) — not reconstructed at report time.
- Each kernel symbol is resolved to its **line span** (brace-balanced for C/CUDA, indentation-delimited
  for Triton, decorators included), so a 40-line kernel inside a 3000-line header counts as 40.
- `code` excludes comments and blank lines; `raw` keeps them. `code` is the headline figure.
- Per-kernel results land in `kernels/<id>.json` `loc`; the rollup is `state/loc.json`, which **merges
  overlapping spans per file** so kernels sharing a header are counted once, and compares the total
  against a whole-project scan for a coverage percentage.
- `loc.unresolved` lists symbols the script could not find — those lines are **not** counted. Fix the
  symbol name, or pin the source explicitly in `loc.source_spec` (hand-added specs survive
  re-measurement). Silence here means undercounting, so check it.

## Analytics (efficiency & cost)
The agent's own **running time and cost** are tracked so a human can judge ROI — how much AI time
and spend the migration took for the performance it delivered. Same view/audit split as PROGRESS.md:
```bash
.sycl/scripts/metrics.sh start <phase>          # open a timing interval at a phase boundary
.sycl/scripts/metrics.sh stop  <phase>          # close it; records elapsed wall-clock seconds
.sycl/scripts/metrics.sh usage <phase> \        # record cost for work done in <phase>
    --tokens-in N --tokens-out N \              #   uncached prompt / completion tokens, and/or
    --cache-read N --cache-write N \            #   cached tokens (Anthropic etc. price these apart), and/or
    --requests N --credits X \                  #   GitHub premium requests / AI credits (Copilot), and/or
    --cost-usd X \                              #   a direct dollar figure if your tool reports one (best)
    --model NAME --kernel ID --note "..."
.sycl/scripts/metrics.sh summary                # print totals + per-phase breakdown
```
Events append to `.sycl/logs/metrics.jsonl` (append-only audit); `metrics.sh` regenerates the
authoritative rollup `.sycl/state/metrics.json` (grand total + per-phase breakdown) after every
write — resumable, never drifts. **Time is always exact** (from timestamps). Prefer `--cost-usd` when
the tool gives a real dollar figure; otherwise record tokens (splitting cached vs uncached, since
caching is priced differently) and/or credits. gen-progress surfaces it in PROGRESS.md, and the
`report` phase renders it as §8 of the final report.

### Token usage comes from the engine's session store (`collect-usage.sh`)
Cost cannot be self-reported reliably: most engines never hand their token accounting back to the
model, so an agent asked to record it either reports 0 or invents numbers — and the ROI half of the
report is lost. Every engine does persist its own session log on disk, so the framework reads that
instead:
```bash
.sycl/scripts/collect-usage.sh detect            # which engine session stores hold data for this project
.sycl/scripts/collect-usage.sh collect --import  # extract -> attribute to phases -> metrics.jsonl
.sycl/scripts/collect-usage.sh stats save|diff   # opencode only: cross-check the total against `opencode stats`
.sycl/scripts/collect-usage.sh bench             # emit the sycl-agent-bench `usage` block from the rollup
```
| Engine | Session store | What it yields |
| --- | --- | --- |
| Claude Code | `~/.claude/**/*.jsonl` (`projects/<slug>/`, incl. `subagents/`) | **exact** per-request input/output/cache-read/cache-write tokens, model, `costUSD` when present |
| opencode | `~/.local/share/opencode/opencode.db` | **exact** per-message input/output/reasoning/cache tokens, USD cost, model |
| Copilot CLI / agent | `~/.copilot/session-store.db` (`assistant_usage_events`) | **exact** per-request input/output/cache-read/cache-write tokens, model and premium-request multiplier, scoped by the session's `cwd`. Falls back to the `session.shutdown` totals in `~/.copilot/session-state/<id>/events.jsonl` (still exact, but one row per cli run) |
| VS Code Copilot chat | `workspaceStorage/*/GitHub.copilot-chat/transcripts/*.jsonl` | last resort — the chat transcript logs **no** tokens, only premium requests (1 per user prompt) + round-trips; tokens stay 0 unless `--estimate-tokens` is passed (then flagged as an estimate). Used only when the `~/.copilot` stores hold nothing for the project |
| Trae | `~/.config/Trae[ CN]/logs/<stamp>/window*/renderer*.log` | **exact** user prompts + model round-trips; Trae **encrypts** its conversation store and logs no token counts, and the message text never reaches the log, so tokens cannot even be estimated |

- **Phase attribution** — each record has a timestamp, so it is bucketed into the phase that was open
  then, reconstructed from the bracketed intervals in `metrics.jsonl` plus the denser `progress.jsonl`
  heartbeat. Records outside any interval clamp to the nearest phase.
- **Idempotent** — every row carries a `uid` taken from the engine's own record id; `metrics.sh import`
  skips uids already in the log, so collecting after every phase (and again at the end) never
  double-counts.
- **Scoped** — only sessions whose working directory is inside the project are read, and by default
  only records at or after the run's first event (`--all-time` lifts that floor).
- **Cross-check (opencode)** — `collect-usage.sh stats save baseline` before the run and
  `collect-usage.sh stats diff baseline` after it snapshot the engine's own lifetime counters
  (`opencode stats`) and print the delta next to what was imported. `opencode stats` rounds to three
  significant digits and covers all projects, so it is a sanity check, not an audit.
- Overrides: `SYCL_CLAUDE_HOME`, `SYCL_OPENCODE_DB`, `SYCL_COPILOT_HOME`, `SYCL_COPILOT_DB`,
  `SYCL_VSCODE_USER_DIRS`, `SYCL_TRAE_DIRS`, `SYCL_TRAE_LOG_MAX`, `SYCL_COPILOT_CTX_CAP`.

When the collector finds nothing for the engine in use, fall back to `metrics.sh usage` with whatever
the engine exposes to the agent — and if it exposes nothing, leave the fields at 0 and say so in the
report rather than guessing.

### Independent timing cross-check (`logs/progress.jsonl`)
Because `metrics.sh start/stop` depends on the agent remembering to bracket every phase (unreliable
on some engines — an agent that stops calling it silently under-reports time), `gen-progress.sh`
keeps an **independent heartbeat**. It runs after *every* state change, and each run appends one
timestamped snapshot (phase + kernel counts) to the append-only `.sycl/logs/progress.jsonl`. This
trail serves two purposes:
- **Cross-check** — the first→last span is a trustworthy *lower bound* on real wall-clock even when
  `metrics.sh` was never called; a variant sums only the gaps ≤ `SYCL_PROGRESS_IDLE_CAP` (default
  1800 s) as `observed working` time, excluding overnight/idle pauses. PROGRESS.md shows both next to
  the metrics figures and **flags** when `metrics.sh` elapsed is far below the observed span.
- **History** — the snapshots are never overwritten (unlike PROGRESS.md, a regenerated view), so the
  run's trajectory is retained; PROGRESS.md renders the recent rows as a *Progress history* table.
`progress.jsonl` is a `*.jsonl` file, so `rotate-logs.sh` (which rotates only `*.log`) never truncates
its earliest anchor.

## Why this shape
Structured JSON gives reliable, queryable state for kernels, specs, and especially optimization-trial
history (which a flat markdown file cannot represent well). git gives free reversibility and a code
audit trail. Logs give a process audit trail. Evidence gives a **results** audit trail — the one a
reviewer needs, because the other three record what the agent did and said, not what the hardware
actually reported. PROGRESS.md gives humans a single consistent glance.
Each store does one job; none duplicates another.
