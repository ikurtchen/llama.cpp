# Phase exit criteria

The complete contract, as enforced by `.sycl/scripts/evidence.sh gate <phase>` and audited
afterwards by check **E15**. Every criterion is mechanical: it reads state and evidence, and does
not require anyone's judgement. That is the point — a reviewer can be persuaded that a missing
phase was reasonable; a missing file cannot be.

Each criterion may be **waived**, never ignored. A waiver lives in
`state/phases.json → phases.<phase>.waivers` and needs `criterion`, `reason` and `blast_radius`.

---

## Why these exist

Every criterion below is a real failure from the 11-project migration programme, written down so it
cannot recur silently:

| Observed outcome | Criterion that now catches it |
|---|---|
| "The optimization phase was deferred entirely" | `optimize.hotspots_accounted` |
| "Optimizations applied but no measured speedup deltas recorded" | `optimize.trials_measured` |
| "No optimization — the profiler segfaulted" | **E16** (a phase that produced no run record) |
| Inventory regenerated at coarser granularity until `pending: 0` | **E17** (scope monotonicity) |
| Kernels ported but never wired into the project's dispatch path | `integrate.surfaces_terminal` |
| The report claims more than the run earned | `report.*` + **E14** |

---

## detect

| Criterion | Requires | Protects against |
|---|---|---|
| `detect.project_identified` | `project.json` has a real `name` and a resolved `build_system` | Starting work against a project nobody characterised |
| `detect.runner_verified` | `env.gpu` is resolved | Discovering at profile time that the runner never worked |
| `detect.archetype_classified` | an integration archetype A–F | `scaffold` having no basis to choose a target level |
| `detect.entrypoint_named` | `integration.entrypoint` | Reaching the end with nothing to run end-to-end |

## inventory

| Criterion | Requires | Protects against |
|---|---|---|
| `inventory.not_empty` | ≥1 kernel enumerated | An empty plan that trivially completes |
| `inventory.records_complete` | every kernel has `id` + `source` + `status` | Work items that cannot be tracked or counted |
| `inventory.scope_frozen` | `scope.baseline.kernel_ids` populated | The denominator moving later without anyone noticing (**E17**) |

Freeze the denominator as the last act of the phase:

```bash
# after the inventory is final
python3 - <<'PY'
import json, datetime
ph = json.load(open(".sycl/state/phases.json"))
idx = json.load(open(".sycl/state/kernels/index.json"))
ph["scope"]["baseline"] = {
    "captured_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "captured_in_phase": "inventory",
    "kernel_ids": [k["id"] for k in idx.get("kernels", [])],
    "surface_ids": [s["id"] for s in json.load(open(".sycl/state/integration/index.json"))["surfaces"]],
}
json.dump(ph, open(".sycl/state/phases.json", "w"), indent=2)
PY
```

## scaffold

| Criterion | Requires | Protects against |
|---|---|---|
| `scaffold.target_declared` | `readiness.target_level` ∈ L0–L4 | Retrofitting the goal to whatever was achieved |
| `scaffold.backend_switch` | a recorded way to select SYCL | A backend the project has no way to turn on |
| `scaffold.surfaces_enumerated` | integration surfaces with statuses | Discovering the integration work after migrating |
| `scaffold.l1_resolved` | gate L1 not `pending` | Migrating into a build nobody has tried |
| `scaffold.skeleton_blocker_recorded` | when L1 fails: a waiver or residual item | A skeleton that will not stand, discovered at report time |

## migrate

| Criterion | Requires | Protects against |
|---|---|---|
| `migrate.all_terminal` | no kernel left `pending` | Quiet partial completion |
| `migrate.skips_justified` | every skip/needs-reference has notes | Skips indistinguishable from failures |
| `migrate.all_tested` | every migrated kernel has a test result | Code that was written but never run |

## integrate

| Criterion | Requires | Protects against |
|---|---|---|
| `integrate.surfaces_terminal` | every surface `done`/`waived`/`deferred`/`blocked` | The last mile silently left open |
| `integrate.deferrals_costed` | deferred/blocked surfaces appear in `residual` | "Not finished" with no idea what finishing costs |
| `integrate.l2_resolved` | gate L2 not `pending` | Never running the project's own test suite |

## profile-e2e

| Criterion | Requires | Protects against |
|---|---|---|
| `profile.baseline_captured` | `e2e.baseline.json` has a `ranking` | An optimize phase with no denominator — and therefore no way to check it |
| `profile.baseline_backed` | the ranking resolves to an e2e record | A hotspot list assembled by intuition |
| `profile.method_stated` | `method` names how wall-clock was attributed | Overclaiming precision when a fallback method was used |

## optimize

The phase this contract was written for.

| Criterion | Requires | Protects against |
|---|---|---|
| `optimize.hotspots_identified` | a baseline ranking | Optimizing without knowing what matters |
| `optimize.hotspots_accounted` | every kernel ≥ `hotspot_pct_threshold` of wall-clock has **either** a trial with a measured before/after **or** `status: skipped` with a `stop_reason` | The phase being skipped in whole or in part |
| `optimize.trials_measured` | every trial has numeric `metric_before` and `metric_after` | "We optimized it" with nothing to show it worked |
| `optimize.stop_reason_given` | closed optimizations state why they stopped | Stopping because the context ran out, presented as completion |

**A reverted trial passes.** Three measured trials that all made it slower, honestly reverted, is a
complete optimize phase and a useful result — it says the kernel is at its practical limit. Zero
trials with no reason is not. The criterion is about whether the question was **asked**, never about
whether the answer was favourable.

**When the profiler will not run**, that is a legitimate exit — recorded:

```jsonc
// state/optimization/<id>.json
{ "kernel_id": "k007", "status": "skipped", "trials": [],
  "stop_reason": "unitrace segfaults on driver 24.35 (run id bench-k007-...-3f10ab); no attribution
                  obtainable, so no hypothesis can be formed or tested" }
```

## done

| Criterion | Requires | Protects against |
|---|---|---|
| `done.no_open_kernels` / `done.no_open_surfaces` | nothing still in progress | Declaring completion over open work |
| `done.readiness_settled` | `achieved_level` computed | An unearned readiness level |
| `done.readiness_mirrored` | `project.json` agrees with `readiness.json` | A dashboard showing a stale level |
| `done.phases_closed` | every earlier phase `exited` or `waived` | **The core check**: reaching the end without ever gating a phase |

## report

| Criterion | Requires | Protects against |
|---|---|---|
| `report.exists` | `FINAL_REPORT.md` | — |
| `report.cites_evidence` | ≥1 run id in the prose | Numbers no reader can trace |
| `report.backend_section` | a backend integration status section | The reader assuming the project runs on the GPU |
| `report.residual_stated` | residual work appears in the prose | Known remaining work living only in JSON |
| `report.audit_generated` | `AUDIT.md` | No link from a number to its log |

---

## The three audit checks

| Check | Fires when | Why it is separate from the gate |
|---|---|---|
| **E15** | a phase is marked `exited`/`waived` but its criteria are not met | The gate can be bypassed by hand-editing `phases.json`; E15 re-derives the verdict from state at audit time, so the edit is caught |
| **E16** | an execution-bearing phase was entered and left with **zero run records in its window** | Catches the shape the criteria cannot: a phase that was technically closed but did nothing, e.g. a tool crashed and the phase was abandoned |
| **E17** | a work item leaves the inventory with no `scope.changes` entry | The denominator is the basis of every percentage; moving it silently invalidates the whole report |

E15 asks *were the criteria met?*, E16 asks *did anything happen at all?*, E17 asks *is this still
the same question we started with?* A run can pass any two and fail the third.
