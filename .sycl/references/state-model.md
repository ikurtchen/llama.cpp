# State Model

`sycl-agent` coordinates through files, not chat history. Four stores, each with a distinct job:

| Store | Format | Role | Source of truth? |
|-------|--------|------|------------------|
| `.sycl/state/*.json` | structured JSON | machine-readable progress that drives the workflow | **yes** |
| `.sycl/PROGRESS.md` | Markdown | human-readable dashboard, **generated** from state | no (a view) |
| git history | commits | *what code changed*; enables revert | audit (code) |
| `.sycl/logs/*.jsonl` | JSONL | *decisions, results, timings*; append-only | audit (process) |
| `.sycl/state/metrics.json` | structured JSON | **analytics** rollup (efficiency & cost), **generated** from `logs/metrics.jsonl` | no (a view) |

The JSON is authoritative. PROGRESS.md is regenerated from it after every change so the two never
drift. Never hand-edit PROGRESS.md — edit the JSON and run `.sycl/scripts/gen-progress.sh`.

## Layout
```
.sycl/
├── config.json                     # runner (local|remote) + toolchain + benchmark policy
├── instructions.md                 # OPTIONAL human-authored project overrides (agent reads, never edits)
├── PROGRESS.md                     # GENERATED human dashboard (do not edit)
├── state/
│   ├── project.json                # env, build system, target GPU, current phase
│   ├── kernels/
│   │   ├── index.json              # list: id, source, status, risk, -> detail
│   │   ├── <id>.json               # per-kernel detail (algorithm, IO, test, performance)
│   │   └── <id>/ref/               # golden input/output tensors for accuracy checks
│   ├── profile/e2e.json            # LIVE per-kernel wall-clock share -> optimization priority
│   ├── profile/e2e.baseline.json   # immutable pre-optimization snapshot (audit)
│   ├── profile/e2e.final.json      # immutable post-optimization snapshot (audit)
│   ├── optimization/<id>.json      # trial log: hypothesis, commit, correctness, keep/revert
│   ├── metrics.json                # GENERATED analytics rollup (efficiency & cost; do not edit)
│   └── lessons.md                  # durable, provenance-tagged learnings (also shown in PROGRESS)
├── schemas/                        # JSON Schemas validating state/*.json
├── reports/                        # generated benchmark / profiling reports
└── logs/*.jsonl                    # append-only audit logs: metrics.jsonl (cost/time events),
                                    #   progress.jsonl (gen-progress heartbeat: history + timing cross-check)
└── scripts/                        # run.sh, log.sh, metrics.sh, rotate-logs.sh, gen-progress.sh, init-state.sh
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
7. **Resumability.** Never rely on in-memory context. A fresh run reconstructs everything from
   `.sycl/state/` + git.

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
write — resumable, never drifts. **Time is always exact** (from timestamps); **$/token/credit counts
are self-reported** by the agent, because only the model/engine sees its own usage — record whatever
your engine exposes (all cost fields optional). Prefer `--cost-usd` when the tool gives a real dollar
figure; otherwise record tokens (splitting cached vs uncached, since caching is priced differently)
and/or credits. gen-progress surfaces it in PROGRESS.md, and the `report` phase renders it as §8 of
the final report.

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
audit trail. Logs give a process audit trail. PROGRESS.md gives humans a single consistent glance.
Each store does one job; none duplicates another.
