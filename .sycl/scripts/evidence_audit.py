#!/usr/bin/env python3
"""Audit side of the sycl-agent evidence model: manifest, verify, AUDIT.md, DIAGNOSIS.md.

Called through `.sycl/scripts/evidence.sh {manifest,verify,audit,diagnosis,list}`.

The capture side (evidence.sh) answers "what happened". This answers the reviewer's question:
"is every number in the report backed by something a machine produced, and can I still check it?"
A claim in state with no run record behind it is reported as UNBACKED — the audit never has to
decide whether to trust the agent, only whether the evidence exists and still hashes.

See .sycl/references/evidence-model.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import sys

# ---------------------------------------------------------------------------- io helpers ---------


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Ctx:
    """Everything the audit reads, loaded once."""

    def __init__(self, sycl_dir: str):
        self.sycl = os.path.abspath(sycl_dir)
        self.state = os.path.join(self.sycl, "state")
        self.reports = os.path.join(self.sycl, "reports")
        self.evid = os.path.join(self.reports, "evidence")

        self.project = load_json(os.path.join(self.state, "project.json")) or {}
        self.index = load_json(os.path.join(self.state, "kernels", "index.json")) or {}
        self.kernels = {}
        kdir = os.path.join(self.state, "kernels")
        if os.path.isdir(kdir):
            for fn in sorted(os.listdir(kdir)):
                if fn.endswith(".json") and fn != "index.json":
                    d = load_json(os.path.join(kdir, fn))
                    if isinstance(d, dict):
                        self.kernels[d.get("id") or fn[:-5]] = d
        self.optim = {}
        odir = os.path.join(self.state, "optimization")
        if os.path.isdir(odir):
            for fn in sorted(os.listdir(odir)):
                if fn.endswith(".json"):
                    d = load_json(os.path.join(odir, fn))
                    if isinstance(d, dict):
                        self.optim[d.get("kernel_id") or fn[:-5]] = d
        self.e2e_baseline = load_json(os.path.join(self.state, "profile", "e2e.baseline.json"))
        self.e2e_final = load_json(os.path.join(self.state, "profile", "e2e.final.json"))
        self.readiness = load_json(os.path.join(self.state, "integration", "readiness.json")) or {}
        self.integration = load_json(os.path.join(self.state, "integration", "index.json")) or {}
        self.phases_path = os.path.join(self.state, "phases.json")
        self.phases = load_json(self.phases_path) or {}

        self.records = self._load_records()
        self.by_id = {r["run_id"]: r for r in self.records if r.get("run_id")}
        self.envs = {}
        edir = os.path.join(self.evid, "env")
        if os.path.isdir(edir):
            for fn in sorted(os.listdir(edir)):
                if fn.endswith(".json"):
                    d = load_json(os.path.join(edir, fn))
                    if isinstance(d, dict):
                        self.envs[d.get("id") or fn[:-5]] = d

    def _load_records(self):
        out = []
        if not os.path.isdir(self.evid):
            return out
        for cls in sorted(os.listdir(self.evid)):
            cdir = os.path.join(self.evid, cls)
            if cls == "env" or not os.path.isdir(cdir):
                continue
            for subj in sorted(os.listdir(cdir)):
                sdir = os.path.join(cdir, subj)
                if not os.path.isdir(sdir):
                    continue
                for fn in sorted(os.listdir(sdir)):
                    if not fn.endswith(".json"):
                        continue
                    d = load_json(os.path.join(sdir, fn))
                    if isinstance(d, dict) and d.get("schema") == "sycl-agent/run-record":
                        d["_path"] = os.path.join(sdir, fn)
                        out.append(d)
        out.sort(key=lambda r: r.get("ts", ""))
        return out

    # -- lookups ----------------------------------------------------------------------------------
    def records_for(self, cls=None, subject=None, label=None):
        res = self.records
        if cls:
            res = [r for r in res if r.get("class") == cls]
        if subject:
            res = [r for r in res if r.get("subject") == subject]
        if label:
            res = [r for r in res if (r.get("label") or "") == label]
        return res

    def resolve(self, ref):
        """Resolve an evidence pointer: a run id, a list of them, or a record path."""
        if not ref:
            return []
        if isinstance(ref, str):
            ref = [ref]
        out = []
        for item in ref:
            if not isinstance(item, str):
                continue
            rec = self.by_id.get(item)
            if rec is None and item.endswith(".json"):
                rec = self.by_id.get(os.path.basename(item)[:-5])
            if rec:
                out.append(rec)
        return out

    def rel(self, path):
        return os.path.relpath(path, self.reports).replace(os.sep, "/")


# ---------------------------------------------------------------------------- manifest -----------

KIND_BY_EXT = {
    ".log": "log", ".json": "json", ".csv": "csv", ".asm": "asm", ".ip": "asm",
    ".stall": "asm", ".pdf": "pdf", ".md": "md", ".txt": "text", ".png": "image",
}


def cmd_manifest(ctx: Ctx, args) -> int:
    """Hash every artifact under reports/ so a later edit or a lost sync is detectable."""
    arts, total, by_class = [], 0, {}
    if os.path.isdir(ctx.reports):
        for root, dirs, files in os.walk(ctx.reports):
            dirs[:] = [d for d in dirs if d not in (".pull-backups", "__pycache__")]
            for fn in sorted(files):
                full = os.path.join(root, fn)
                if os.path.islink(full) or fn == "manifest.json":
                    continue
                try:
                    size = os.path.getsize(full)
                    entry = {
                        "path": ctx.rel(full),
                        "sha256": sha256_file(full),
                        "bytes": size,
                        "kind": KIND_BY_EXT.get(os.path.splitext(fn)[1].lower(), "other"),
                        "mtime": _dt.datetime.fromtimestamp(
                            os.path.getmtime(full), _dt.timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                except OSError:
                    continue
                total += size
                arts.append(entry)

    stem_to_rec = {}
    for r in ctx.records:
        stem_to_rec[os.path.splitext(ctx.rel(r["_path"]))[0]] = r
        by_class[r.get("class", "?")] = by_class.get(r.get("class", "?"), 0) + 1
    for a in arts:
        rec = stem_to_rec.get(os.path.splitext(a["path"])[0])
        if rec:
            a["run_id"] = rec.get("run_id")
            a["class"] = rec.get("class")
            a["subject"] = rec.get("subject")
            if a["kind"] == "json":
                a["kind"] = "record"

    manifest = {
        "schema": "sycl-agent/evidence-manifest",
        "project": ctx.project.get("name", ""),
        "totals": {
            "records": len(ctx.records),
            "artifacts": len(arts),
            "bytes": total,
            "by_class": by_class,
        },
        "env_fingerprints": [
            {
                "id": e.get("id"),
                "path": f"evidence/env/{e.get('id')}.json",
                "gpu": e.get("gpu"),
                "freq_pinned": e.get("freq_pinned"),
                "used_by": sum(1 for r in ctx.records if r.get("env_fingerprint") == e.get("id")),
            }
            for e in ctx.envs.values()
        ],
        "artifacts": arts,
        "updated_at": utcnow(),
    }
    os.makedirs(ctx.evid, exist_ok=True)
    out = os.path.join(ctx.evid, "manifest.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(
        f"manifest: {len(arts)} artifacts, {len(ctx.records)} records, "
        f"{total/1e6:.1f} MB -> {ctx.rel(out)}"
    )
    return 0


# ---------------------------------------------------------------------------- verify -------------

class Finding:
    __slots__ = ("check", "severity", "subject", "message")

    def __init__(self, check, severity, subject, message):
        self.check, self.severity, self.subject, self.message = check, severity, subject, message


def _kernel_active(k):
    return k.get("status") not in ("skipped", "needs-reference")


# ------------------------------------------------------------------- phase-exit contract ---------
# E1-E14 all ask the same shape of question: "you claimed X — show the record". None of them can ask
# "did you do the thing this phase exists to do?", so a phase that was skipped in silence passes a
# clean audit. Every failure below is an OMISSION observed in a real run:
#   optimize deferred entirely (no trials, no reason)      -> optimize.hotspots_accounted
#   optimizations applied but no measured deltas recorded  -> optimize.trials_measured
#   profiling tool crashed, phase abandoned, nothing said  -> E16 (no-op phase)
#   inventory re-cut coarser until pending hit 0           -> E17 (scope monotonicity)
# The criteria are deliberately mechanical: a reviewer can be talked into "deferral was reasonable
# here", a missing file cannot.

PHASE_ORDER = ["detect", "inventory", "scaffold", "migrate", "integrate",
               "profile-e2e", "optimize", "done", "report"]

# Phases whose whole purpose is to make something run. Entering and leaving one of these without
# producing a single run record means the work did not happen (E16).
EXEC_PHASES = {"scaffold", "migrate", "integrate", "profile-e2e", "optimize"}

TERMINAL_KERNEL = ("migrated", "skipped", "needs-reference")
TERMINAL_SURFACE = ("done", "waived", "deferred", "blocked")


class Criterion:
    __slots__ = ("id", "ok", "detail")

    def __init__(self, cid, ok, detail=""):
        self.id, self.ok, self.detail = cid, bool(ok), detail


def _waived(ctx: Ctx, phase: str, cid: str):
    """A criterion may be deliberately not met — but only out loud, with a stated blast radius."""
    ph = ((ctx.phases.get("phases") or {}).get(phase) or {})
    for w in ph.get("waivers") or []:
        if w.get("criterion") == cid and (w.get("reason") or "").strip() and \
                (w.get("blast_radius") or "").strip():
            return True
    return False


def hotspots(ctx: Ctx):
    """Kernels the baseline profile says are worth optimizing. This is the optimize denominator."""
    thr = fnum(ctx.phases.get("hotspot_pct_threshold"))
    thr = 5.0 if thr is None else thr
    rank = (ctx.e2e_baseline or {}).get("ranking") or []
    out = []
    for r in rank:
        pct = fnum(r.get("wallclock_pct"))
        if r.get("id") and pct is not None and pct >= thr:
            out.append(r["id"])
    return out


def phase_criteria(ctx: Ctx, phase: str):
    """Completion criteria for one phase. Pure: reads state, never calls the evidence checks."""
    c = []
    proj = ctx.project or {}
    integ_on = bool(ctx.readiness) or bool(ctx.integration)
    gates = (ctx.readiness or {}).get("gates") or {}
    surfaces = (ctx.integration or {}).get("surfaces") or []

    if phase == "detect":
        name = (proj.get("name") or "").strip()
        c.append(Criterion("detect.project_identified",
                           name and name != "REPLACE_ME" and proj.get("build_system") not in (None, "", "unknown"),
                           "project.json needs a real name and a resolved build_system"))
        c.append(Criterion("detect.runner_verified",
                           (proj.get("env") or {}).get("gpu") not in (None, "", "unknown"),
                           "run.sh env has not resolved a GPU — the runner was never exercised"))
        if integ_on:
            ig = proj.get("integration") or {}
            c.append(Criterion("detect.archetype_classified",
                               (ig.get("archetype") or (ctx.readiness or {}).get("archetype") or "").strip(),
                               "no integration archetype (A-F) — scaffold cannot pick a target level"))
            c.append(Criterion("detect.entrypoint_named",
                               (ig.get("entrypoint") or (ctx.readiness or {}).get("entrypoint") or "").strip(),
                               "no real entrypoint named — nothing to run end-to-end later"))

    elif phase == "inventory":
        ks = ctx.index.get("kernels") if isinstance(ctx.index.get("kernels"), list) else []
        n = len(ks) or len(ctx.kernels)
        c.append(Criterion("inventory.not_empty", n > 0, "no kernels enumerated"))
        bad = [k.get("id") or "?" for k in (ks or list(ctx.kernels.values()))
               if not (k.get("id") and k.get("source") and k.get("status"))]
        c.append(Criterion("inventory.records_complete", not bad,
                           f"incomplete kernel records (need id+source+status): {', '.join(bad[:8])}"))
        base = ((ctx.phases.get("scope") or {}).get("baseline") or {})
        c.append(Criterion("inventory.scope_frozen", base.get("kernel_ids"),
                           "scope.baseline.kernel_ids is empty — freeze the denominator so later "
                           "percentages stay comparable to the plan (check E17)"))

    elif phase == "scaffold":
        if not integ_on:
            return [Criterion("scaffold.skipped", True, "integration disabled in config")]
        tgt = (ctx.readiness.get("target_level") or "").strip()
        c.append(Criterion("scaffold.target_declared", tgt in ("L0", "L1", "L2", "L3", "L4"),
                           "no target_level declared — declare the level you are aiming for BEFORE "
                           "migrating, so falling short is visible rather than retrofitted"))
        c.append(Criterion("scaffold.backend_switch",
                           (ctx.readiness.get("backend_switch") or "").strip(),
                           "no backend switch recorded — the project has no way to select SYCL"))
        c.append(Criterion("scaffold.surfaces_enumerated",
                           surfaces and all((s.get("status") or "").strip() for s in surfaces),
                           "integration surfaces not enumerated or left status-less"))
        l1 = (gates.get("L1") or {}).get("status") or "pending"
        c.append(Criterion("scaffold.l1_resolved", l1 != "pending",
                           "gate L1 is still 'pending' — the project's own build with the backend "
                           "enabled must either pass, or fail with the blocker recorded"))
        if l1 in ("fail", "waived"):
            c.append(Criterion("scaffold.skeleton_blocker_recorded",
                               _waived(ctx, "scaffold", "scaffold.l1_resolved")
                               or (ctx.readiness.get("residual") or []),
                               "L1 did not pass and nothing says why or what it costs — lower the "
                               "target level now, loudly, and record the blocker"))

    elif phase == "migrate":
        pend = [kid for kid, k in ctx.kernels.items() if k.get("status") not in TERMINAL_KERNEL]
        c.append(Criterion("migrate.all_terminal", not pend,
                           f"{len(pend)} kernel(s) not in a terminal state: {', '.join(sorted(pend)[:8])}"))
        noreason = [kid for kid, k in ctx.kernels.items()
                    if k.get("status") in ("skipped", "needs-reference")
                    and not (k.get("notes") or "").strip()]
        c.append(Criterion("migrate.skips_justified", not noreason,
                           f"skipped with no reason: {', '.join(sorted(noreason)[:8])}"))
        untested = [kid for kid, k in ctx.kernels.items()
                    if k.get("status") == "migrated"
                    and (k.get("unit_test") or {}).get("result") not in ("pass", "fail")]
        c.append(Criterion("migrate.all_tested", not untested,
                           f"migrated but never tested: {', '.join(sorted(untested)[:8])}"))

    elif phase == "integrate":
        if not integ_on:
            return [Criterion("integrate.skipped", True, "integration disabled in config")]
        open_s = [s.get("id") or "?" for s in surfaces if s.get("status") not in TERMINAL_SURFACE]
        c.append(Criterion("integrate.surfaces_terminal", not open_s,
                           f"integration surface(s) still open: {', '.join(open_s)}"))
        resid = {(r.get("surface") or "") for r in (ctx.readiness.get("residual") or [])}
        unpriced = [s.get("id") or "?" for s in surfaces
                    if s.get("status") in ("deferred", "blocked") and (s.get("id") or "") not in resid]
        c.append(Criterion("integrate.deferrals_costed", not unpriced,
                           f"deferred/blocked with no residual work item: {', '.join(unpriced)} — "
                           "say what is left, what it unlocks and what it costs"))
        l2 = (gates.get("L2") or {}).get("status") or "pending"
        c.append(Criterion("integrate.l2_resolved", l2 != "pending",
                           "gate L2 is still 'pending' — run the project's own test suite, or "
                           "record why it cannot be run"))

    elif phase == "profile-e2e":
        base = ctx.e2e_baseline or {}
        c.append(Criterion("profile.baseline_captured", (base.get("ranking") or []),
                           "profile/e2e.baseline.json has no ranking — without it the optimize "
                           "phase has no denominator and cannot be checked at all"))
        c.append(Criterion("profile.baseline_backed",
                           ctx.resolve(base.get("evidence")) or ctx.records_for("e2e", "baseline"),
                           "the baseline ranking has no e2e run record behind it"))
        c.append(Criterion("profile.method_stated", (base.get("method") or "").strip(),
                           "no attribution method stated — a fallback method must be named so the "
                           "report does not overclaim precision"))

    elif phase == "optimize":
        hs = hotspots(ctx)
        c.append(Criterion("optimize.hotspots_identified", (ctx.e2e_baseline or {}).get("ranking"),
                           "no baseline ranking — profile-e2e did not produce the hotspot list"))
        missing = []
        for kid in hs:
            o = ctx.optim.get(kid)
            if not o:
                missing.append(f"{kid} (no optimization record)")
                continue
            trials = o.get("trials") or []
            measured = [t for t in trials
                        if fnum(t.get("metric_before")) is not None and fnum(t.get("metric_after")) is not None]
            if measured:
                continue
            if o.get("status") == "skipped" and (o.get("stop_reason") or "").strip():
                continue
            if trials:
                missing.append(f"{kid} (trials with no measured before/after)")
            else:
                missing.append(f"{kid} (no trial, no stop_reason)")
        c.append(Criterion("optimize.hotspots_accounted", not missing,
                           "every hotspot needs a measured trial or a recorded reason for having "
                           f"none — unaccounted: {'; '.join(missing[:6])}"))
        unmeasured = []
        for kid, o in ctx.optim.items():
            for t in o.get("trials") or []:
                if fnum(t.get("metric_before")) is None or fnum(t.get("metric_after")) is None:
                    unmeasured.append(f"{kid}#{t.get('n')}")
        c.append(Criterion("optimize.trials_measured", not unmeasured,
                           "a trial with no before/after number is not a trial — an optimization "
                           f"applied but never measured cannot be kept or reverted: {', '.join(unmeasured[:8])}"))
        nostop = [kid for kid, o in ctx.optim.items()
                  if o.get("status") == "done" and not (o.get("stop_reason") or "").strip()]
        c.append(Criterion("optimize.stop_reason_given", not nostop,
                           f"optimization closed with no stop_reason: {', '.join(sorted(nostop)[:8])}"))

    elif phase == "done":
        pend = [kid for kid, k in ctx.kernels.items() if k.get("status") not in TERMINAL_KERNEL]
        c.append(Criterion("done.no_open_kernels", not pend,
                           f"{len(pend)} kernel(s) still open"))
        if integ_on:
            open_s = [s.get("id") or "?" for s in surfaces if s.get("status") not in TERMINAL_SURFACE]
            c.append(Criterion("done.no_open_surfaces", not open_s,
                               f"integration surface(s) still open: {', '.join(open_s)}"))
            ach = ctx.readiness.get("achieved_level") or ""
            c.append(Criterion("done.readiness_settled", ach in ("L0", "L1", "L2", "L3", "L4"),
                               "achieved_level not settled"))
            c.append(Criterion("done.readiness_mirrored",
                               (proj.get("integration") or {}).get("achieved_level") == ach,
                               "project.json integration.achieved_level disagrees with "
                               "integration/readiness.json — the dashboard would show a stale level"))
        earlier = [p for p in PHASE_ORDER[:PHASE_ORDER.index("done")]
                   if ((ctx.phases.get("phases") or {}).get(p) or {}).get("status") not in ("exited", "waived")]
        c.append(Criterion("done.phases_closed", not earlier,
                           f"phase(s) never gated: {', '.join(earlier)} — a phase left without "
                           "passing its gate is the failure mode this contract exists to stop"))

    elif phase == "report":
        fr = os.path.join(ctx.reports, "FINAL_REPORT.md")
        c.append(Criterion("report.exists", os.path.exists(fr), "no FINAL_REPORT.md"))
        if os.path.exists(fr):
            txt = open(fr, errors="replace").read()
            c.append(Criterion("report.cites_evidence",
                               re.search(r"\b[a-z0-9]+-[A-Za-z0-9_.-]+-\d{8}T\d{6}Z-[0-9a-f]{6}\b", txt),
                               "cites no run ids — no number is traceable"))
            if integ_on:
                c.append(Criterion("report.backend_section",
                                   "backend integration" in txt.lower(),
                                   "no backend integration status section — the reader cannot tell "
                                   "whether the project actually runs on the Intel GPU"))
            c.append(Criterion("report.residual_stated",
                               not (ctx.readiness.get("residual") or [])
                               or re.search(r"residual|remaining work|what is left", txt, re.I),
                               "residual work exists in state but the report never mentions it"))
        c.append(Criterion("report.audit_generated",
                           os.path.exists(os.path.join(ctx.reports, "AUDIT.md")),
                           "no AUDIT.md — the reviewer has no link from a number to its log"))

    return [x if x.ok or not _waived(ctx, phase, x.id)
            else Criterion(x.id, True, f"WAIVED — {x.detail}")
            for x in c]


def collect_findings(ctx: Ctx):
    """The mechanical audit. Each finding names a claim that evidence does not (yet) support."""
    f = []
    manifest = load_json(os.path.join(ctx.evid, "manifest.json")) or {}
    man_by_path = {a["path"]: a for a in manifest.get("artifacts", [])}

    # E1 — a passing unit test must be backed by a record whose PROCESS exited 0.
    for kid, k in ctx.kernels.items():
        ut = k.get("unit_test") or {}
        if ut.get("result") == "pass":
            recs = ctx.resolve(ut.get("evidence")) or ctx.records_for("test", kid)
            if not recs:
                f.append(Finding("E1", "fail", kid, "unit_test.result=pass with no test run record"))
            else:
                latest = recs[-1]
                if latest.get("exit_code") != 0:
                    f.append(Finding("E1", "fail", kid,
                                     f"claims pass but record {latest['run_id']} exited "
                                     f"{latest.get('exit_code')}"))
                # E11 — a pass that only just cleared the tolerance deserves a human's eyes.
                m = latest.get("measurements") or {}
                err, tol = fnum(m.get("max_abs_err")), fnum((m.get("tolerance") or {}).get("atol"))
                if err is not None and tol and tol > 0 and err > 0.9 * tol:
                    f.append(Finding("E11", "warn", kid,
                                     f"margin-thin: max_abs_err {err:g} vs atol {tol:g}"))
                if not (m.get("tolerance") and m.get("reference")):
                    f.append(Finding("E1b", "warn", kid,
                                     "test record lacks tolerance and/or reference description"))
        elif ut.get("result") == "fail":
            f.append(Finding("E1c", "warn", kid, "unit_test.result=fail is still recorded"))

    # E2 — every performance figure must come from a benchmark someone can re-run.
    for kid, k in ctx.kernels.items():
        if not _kernel_active(k):
            continue
        perf = k.get("performance") or {}
        ev = perf.get("evidence") or {}
        for slot in ("baseline", "optimized"):
            if not perf.get(slot):
                continue
            recs = ctx.resolve(ev.get(slot)) or ctx.records_for("bench", kid, label=slot)
            if not recs:
                f.append(Finding("E2", "fail", kid,
                                 f"performance.{slot}={perf[slot]!r} with no bench run record"))
            else:
                r = recs[-1]
                envid = r.get("env_fingerprint")
                envd = ctx.envs.get(envid) or {}
                if envd.get("freq_pinned") is not True:
                    f.append(Finding("E9", "warn", kid,
                                     f"unstable: {slot} bench {r['run_id']} ran with "
                                     f"freq_pinned={envd.get('freq_pinned')!r}"))
                if r.get("dirty"):
                    f.append(Finding("E10", "warn", kid,
                                     f"unreproducible: {slot} bench {r['run_id']} captured on a dirty tree"))
        # E7 — a speedup is only a speedup if both ends ran on the same machine, in the same state.
        b = ctx.resolve(ev.get("baseline")) or ctx.records_for("bench", kid, label="baseline")
        o = ctx.resolve(ev.get("optimized")) or ctx.records_for("bench", kid, label="final")
        if b and o and b[-1].get("env_fingerprint") != o[-1].get("env_fingerprint"):
            f.append(Finding("E7", "fail", kid,
                             "baseline and optimized benchmarks ran on different env fingerprints "
                             f"({b[-1].get('env_fingerprint')} vs {o[-1].get('env_fingerprint')})"))

    # E3 — a KEPT optimization changed the shipped code: it needs both gates on record.
    for kid, o in ctx.optim.items():
        for t in o.get("trials") or []:
            if t.get("decision") != "keep":
                continue
            ev = t.get("evidence") or {}
            n = t.get("n")
            if not (ctx.resolve(ev.get("test")) or ctx.resolve(ev.get("correctness"))):
                f.append(Finding("E3", "fail", kid, f"kept trial #{n} has no correctness record"))
            if not ctx.resolve(ev.get("bench_after")):
                f.append(Finding("E3", "fail", kid, f"kept trial #{n} has no post-change bench record"))

    # E6/E12 — the before/after profile is what shows the optimization went the RIGHT way.
    for kid, o in ctx.optim.items():
        kept = [t for t in (o.get("trials") or []) if t.get("decision") == "keep"]
        if not kept:
            continue
        optdir = os.path.join(ctx.reports, f"opt_{kid}")
        for side in ("baseline", "final"):
            d = os.path.join(optdir, side)
            if not (os.path.isdir(d) and any(os.scandir(d))):
                f.append(Finding("E6", "fail", kid, f"missing deep-profile artifacts: opt_{kid}/{side}/"))
        diag = os.path.join(optdir, "DIAGNOSIS.md")
        if not os.path.exists(diag):
            f.append(Finding("E12", "warn", kid,
                             f"no opt_{kid}/DIAGNOSIS.md — the before/after claim check is missing"))
        else:
            txt = open(diag, errors="replace").read().lower()
            if "verdict: supported" not in txt:
                f.append(Finding("E12", "warn", kid,
                                 "DIAGNOSIS.md verdict is not 'supported' — the speedup is not "
                                 "explained by the diagnosed bottleneck"))

    # E4 — the headline end-to-end number needs its own record, per case.
    for label, snap in (("baseline", ctx.e2e_baseline), ("final", ctx.e2e_final)):
        if not snap:
            continue
        cases = snap.get("cases") or [{"name": label}]
        recs = ctx.resolve(snap.get("evidence")) or ctx.records_for("e2e", label)
        if not recs:
            f.append(Finding("E4", "fail", f"e2e/{label}",
                             f"profile/e2e.{label}.json has no e2e run record"))
        elif len(recs) < len(cases):
            f.append(Finding("E4", "warn", f"e2e/{label}",
                             f"{len(cases)} e2e case(s) but only {len(recs)} record(s)"))

    # E5 — evidence that no longer hashes is not evidence.
    for r in ctx.records:
        logrel = r.get("log")
        if not logrel:
            f.append(Finding("E5", "fail", r.get("subject", "?"), f"record {r['run_id']} has no log"))
            continue
        full = os.path.join(ctx.reports, logrel)
        if not os.path.exists(full):
            f.append(Finding("E5", "fail", r.get("subject", "?"),
                             f"log missing for {r['run_id']}: {logrel}"))
            continue
        want = r.get("log_sha256")
        if want:
            got = sha256_file(full)
            if got != want:
                f.append(Finding("E5", "fail", r.get("subject", "?"),
                                 f"log for {r['run_id']} was modified after capture (sha mismatch)"))
        man = man_by_path.get(logrel)
        if man and want and man.get("sha256") != want:
            f.append(Finding("E5", "fail", r.get("subject", "?"),
                             f"manifest hash disagrees with record for {r['run_id']}"))
        if not r.get("env_fingerprint"):
            f.append(Finding("E5b", "warn", r.get("subject", "?"),
                             f"record {r['run_id']} has no env fingerprint"))

    # E13 — the Backend Readiness Level is COMPUTED, never asserted. A migration that ends with the
    # project unable to build or run is L0 no matter how many kernels passed their unit tests, so
    # each level above L0 must resolve to a run record from the PROJECT'S OWN artifact.
    if ctx.readiness:
        levels = ["L0", "L1", "L2", "L3", "L4"]
        achieved = ctx.readiness.get("achieved_level") or "L0"
        target = ctx.readiness.get("target_level") or ""
        gates = ctx.readiness.get("gates") or {}
        if achieved not in levels:
            f.append(Finding("E13", "fail", "readiness",
                             f"achieved_level '{achieved}' is not one of {'/'.join(levels)}"))
            achieved = "L0"

        def passing(level, cls, fallback_subjects=()):
            """A gate counts only if its evidence resolves to a record that actually exited 0."""
            g = gates.get(level) or {}
            if g.get("status") not in ("pass", "partial"):
                return False, f"gate {level} is '{g.get('status', 'pending')}'"
            recs = ctx.resolve(g.get("evidence"))
            if not recs:
                for subj in fallback_subjects:
                    recs = ctx.records_for(cls, subj)
                    if recs:
                        break
            recs = [r for r in recs if r.get("class") == cls] or recs
            if not recs:
                return False, f"gate {level} claims '{g.get('status')}' with no {cls} record"
            if not any(r.get("exit_code") == 0 for r in recs):
                return False, f"gate {level} evidence has no {cls} record with exit_code 0"
            if g.get("status") == "partial" and level == "L2" and not (g.get("exceptions") or []):
                return False, "gate L2 is 'partial' but lists no exceptions — enumerate each failure"
            return True, ""

        supported = "L0"
        for level, cls, subjects in (
            ("L1", "build", ("backend", "project")),
            ("L2", "integration", ("suite", "runtime")),
            ("L3", "e2e", ("baseline",)),
            ("L4", "e2e", ("final",)),
        ):
            ok, why = passing(level, cls, subjects)
            if not ok:
                if levels.index(achieved) >= levels.index(level):
                    f.append(Finding("E13", "fail", "readiness",
                                     f"achieved_level {achieved} not supported: {why}"))
                break
            supported = level

        # L3 additionally needs a reference — "it ran without crashing" is liveness, not correctness.
        if levels.index(supported) >= 3:
            g3 = gates.get("L3") or {}
            if not (g3.get("reference") or "").strip():
                f.append(Finding("E13", "fail", "readiness",
                                 "gate L3 names no reference — liveness is not correctness"))
        # L4 must compare like with like.
        if levels.index(supported) >= 4:
            fps = {r.get("env_fingerprint") for r in ctx.resolve((gates.get("L4") or {}).get("evidence"))}
            fps.discard(None)
            if len(fps) > 1:
                f.append(Finding("E13", "fail", "readiness",
                                 "gate L4 baseline and final came from different machines"))
        # Falling short of target is a legitimate result -- but only with a costed work package, so
        # "we did not finish" becomes a next step. Only asked for once the run is meant to be over.
        if (target and target in levels and levels.index(achieved) < levels.index(target)
                and ctx.project.get("phase") in ("done", "report")
                and not (ctx.readiness.get("residual") or [])):
            f.append(Finding("E13", "warn", "readiness",
                             f"achieved {achieved} below target {target} with no residual work "
                             f"package — say what is left, what it unlocks, and what it costs"))

    # E14 — the state may be honest while the prose is not. Bind the report's sentences to the level.
    final_report = os.path.join(ctx.reports, "FINAL_REPORT.md")
    if ctx.readiness and os.path.exists(final_report):
        levels = ["L0", "L1", "L2", "L3", "L4"]
        achieved = ctx.readiness.get("achieved_level") or "L0"
        ach_i = levels.index(achieved) if achieved in levels else 0
        low = open(final_report, errors="replace").read().lower()
        claim_rules = [
            (1, r"(builds?|compiles?) (with|against|using) the sycl backend", "the project builds with the SYCL backend"),
            (2, r"backend is (now )?live|running in the project'?s runtime", "the SYCL backend is live in the runtime"),
            (3, r"runs? on (the )?intel (gpu|arc|xe)|running on (the )?intel gpu|fully functional (sycl|xpu) backend",
                "<project> runs on Intel GPU"),
            (4, r"end-to-end speedup|e2e speedup|\bspeedup of\b.*end.to.end", "an end-to-end speedup"),
        ]
        for need, pat, sentence in claim_rules:
            if ach_i < need and re.search(pat, low):
                f.append(Finding("E14", "fail", "FINAL_REPORT.md",
                                 f'claims "{sentence}" but achieved_level is {achieved} '
                                 f"(needs {levels[need]}) — lower the sentence or raise the evidence"))
        if re.search(r"\b(vs\.?|versus|compared (to|with))\s+(nvidia|cuda|a100|h100|rtx)", low):
            if not ctx.records_for("bench", "nvidia") and "nvidia" not in json.dumps(
                    [r.get("label", "") for r in ctx.records]).lower():
                f.append(Finding("E14", "warn", "FINAL_REPORT.md",
                                 "makes an NVIDIA comparison with no A/B record on NVIDIA hardware"))

    # E8 — every number the REPORT shows to a human must exist in some record.
    final_report = os.path.join(ctx.reports, "FINAL_REPORT.md")
    if os.path.exists(final_report):
        txt = open(final_report, errors="replace").read()
        cited = set(re.findall(r"\b([a-z0-9]+-[A-Za-z0-9_.-]+-\d{8}T\d{6}Z-[0-9a-f]{6})\b", txt))
        unknown = sorted(c for c in cited if c not in ctx.by_id)
        for c in unknown:
            f.append(Finding("E8", "fail", "FINAL_REPORT.md", f"cites unknown run id {c}"))
        if not cited:
            f.append(Finding("E8", "warn", "FINAL_REPORT.md",
                             "cites no run ids — numbers are not traceable to evidence"))

    # E15 — a phase marked exited must actually have met its exit criteria. This is the check that
    # asks about SILENCE rather than about claims: nothing else in this file can notice a phase that
    # was skipped, because a phase that was skipped makes no claim to falsify.
    ph_all = (ctx.phases.get("phases") or {})
    for name in PHASE_ORDER:
        ph = ph_all.get(name) or {}
        if ph.get("status") not in ("exited", "waived"):
            continue
        for crit in phase_criteria(ctx, name):
            if not crit.ok:
                f.append(Finding("E15", "fail", f"phase/{name}",
                                 f"exited without meeting {crit.id}: {crit.detail}"))
        if ph.get("status") == "waived" and not (ph.get("waivers") or []):
            f.append(Finding("E15", "fail", f"phase/{name}",
                             "phase is 'waived' but lists no waiver — a skipped phase must say why "
                             "it was skipped and what is unknown as a result"))

    # E16 — a phase whose job is to make something run, entered and left without producing a single
    # run record, did not happen. A tool that crashed is a blocker to record, not a phase to drop.
    for name in sorted(EXEC_PHASES):
        ph = ph_all.get(name) or {}
        if ph.get("status") != "exited":
            continue
        t0, t1 = ph.get("entered_at") or "", ph.get("exited_at") or ""
        if not (t0 and t1):
            f.append(Finding("E16", "warn", f"phase/{name}",
                             "exited without entered_at/exited_at — the phase cannot be shown to "
                             "have done any work"))
            continue
        during = [r for r in ctx.records if t0 <= (r.get("ts") or "") <= t1]
        if not during and not (ph.get("waivers") or []):
            f.append(Finding("E16", "fail", f"phase/{name}",
                             f"produced no run record between {t0} and {t1} — the phase was a no-op; "
                             "either do the work or record a waiver with its blast radius"))

    # E17 — the denominator may move, but never quietly. Re-cutting an inventory at a coarser
    # granularity until nothing is 'pending' changes every percentage in the report.
    scope = ctx.phases.get("scope") or {}
    base_ids = set((scope.get("baseline") or {}).get("kernel_ids") or [])
    if base_ids:
        removed_ok = set()
        for ch in scope.get("changes") or []:
            if (ch.get("reason") or "").strip():
                removed_ok |= set(ch.get("removed") or [])
            else:
                f.append(Finding("E17", "fail", "scope",
                                 f"scope change removing {', '.join(ch.get('removed') or []) or '(nothing)'} "
                                 "gives no reason"))
        vanished = sorted(base_ids - set(ctx.kernels) - removed_ok)
        if vanished:
            f.append(Finding("E17", "fail", "scope",
                             f"{len(vanished)} work item(s) left the inventory with no scope change "
                             f"record: {', '.join(vanished[:8])}"))
        base_surf = set((scope.get("baseline") or {}).get("surface_ids") or [])
        cur_surf = {s.get("id") for s in ((ctx.integration or {}).get("surfaces") or [])}
        gone = sorted(base_surf - cur_surf - removed_ok)
        if gone:
            f.append(Finding("E17", "fail", "scope",
                             f"integration surface(s) deleted from the index: {', '.join(gone)}"))
    return f


def cmd_verify(ctx: Ctx, args) -> int:
    findings = collect_findings(ctx)
    fails = [x for x in findings if x.severity == "fail"]
    warns = [x for x in findings if x.severity == "warn"]

    claims = backed = 0
    for kid, k in ctx.kernels.items():
        ut = k.get("unit_test") or {}
        if ut.get("result") in ("pass", "fail"):
            claims += 1
            if ctx.resolve(ut.get("evidence")) or ctx.records_for("test", kid):
                backed += 1
        perf = k.get("performance") or {}
        ev = perf.get("evidence") or {}
        for slot in ("baseline", "optimized"):
            if perf.get(slot):
                claims += 1
                if ctx.resolve(ev.get(slot)) or ctx.records_for("bench", kid, label=slot):
                    backed += 1

    print(f"evidence verify — {ctx.project.get('name','(project)')} @ {utcnow()}")
    print(f"  records: {len(ctx.records)}  env fingerprints: {len(ctx.envs)}  "
          f"kernels: {len(ctx.kernels)}")
    if ctx.readiness:
        surfaces = (ctx.integration or {}).get("surfaces") or []
        closed = sum(1 for s in surfaces if s.get("status") in ("done", "waived"))
        print(f"  backend: readiness {ctx.readiness.get('achieved_level','L0')} "
              f"-> target {ctx.readiness.get('target_level') or '(undeclared)'}"
              + (f"  surfaces {closed}/{len(surfaces)} closed" if surfaces else ""))
    if ctx.phases:
        ph_all = ctx.phases.get("phases") or {}
        gated = [p for p in PHASE_ORDER if (ph_all.get(p) or {}).get("status") in ("exited", "waived")]
        cur = ctx.project.get("phase") or "?"
        print(f"  phases:  {len(gated)}/{len(PHASE_ORDER)} gated, now in '{cur}'"
              + ("  [ungated exit]" if cur in PHASE_ORDER and any(
                  (ph_all.get(p) or {}).get("status") not in ("exited", "waived")
                  for p in PHASE_ORDER[:PHASE_ORDER.index(cur)]) else ""))
    print(f"  claims:  {claims} total, {backed} backed, {claims-backed} unbacked")
    for x in fails:
        print(f"  FAIL [{x.check}] {x.subject}: {x.message}")
    for x in warns:
        print(f"  WARN [{x.check}] {x.subject}: {x.message}")
    if not findings:
        print("  OK — every claim is backed and every log hash verified.")
    print(f"  => {len(fails)} failure(s), {len(warns)} warning(s)")

    if fails:
        return 1
    if warns and getattr(args, "strict", False):
        return 1
    return 0


# ---------------------------------------------------------------------------- gate ---------------

def cmd_gate(ctx: Ctx, args) -> int:
    """Evaluate one phase's exit criteria. This is the ONLY way a phase may be marked exited.

    Every evaluation is recorded — including the failed ones — so 'passed on attempt 3' stays
    visible as 'was refined twice', and a phase can never be exited by simply not asking.
    """
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"unknown phase '{phase}' (expected one of: {', '.join(PHASE_ORDER)})")
        return 2
    if not ctx.phases:
        print(f"no {ctx.rel(ctx.phases_path)} — run init-state.sh to install the phase contract")
        return 2

    crits = phase_criteria(ctx, phase)
    unmet = [c for c in crits if not c.ok]

    # The evidence checks are part of the exit gate too: a phase may not be left with a failing
    # claim behind it. E15 is excluded to avoid grading this phase against a status it has not
    # been given yet.
    ev_fails = [x for x in collect_findings(ctx) if x.severity == "fail" and x.check != "E15"]

    print(f"phase gate — {phase} @ {utcnow()}")
    for c in crits:
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.id}" + (f": {c.detail}" if not c.ok or
              c.detail.startswith("WAIVED") else ""))
    for x in ev_fails:
        print(f"  [FAIL] evidence {x.check} {x.subject}: {x.message}")

    ok = not unmet and not ev_fails
    ph = (ctx.phases.setdefault("phases", {}).setdefault(phase, {}))
    g = ph.setdefault("gate", {})
    g["attempts"] = int(g.get("attempts") or 0) + 1
    g["checked_at"] = utcnow()
    g["result"] = "pass" if ok else "fail"
    g["unmet"] = [f"{c.id}: {c.detail}" for c in unmet] + \
                 [f"evidence {x.check} {x.subject}: {x.message}" for x in ev_fails]
    if ok:
        ph["status"] = "exited"
        ph["exited_at"] = ph.get("exited_at") or utcnow()
    elif ph.get("status") in (None, "", "pending"):
        ph["status"] = "in-progress"
        if not ph.get("entered_at"):
            ph["entered_at"] = utcnow()

    ctx.phases["updated_at"] = utcnow()
    if not getattr(args, "dry_run", False):
        with open(ctx.phases_path, "w") as fh:
            json.dump(ctx.phases, fh, indent=2)
            fh.write("\n")

    if ok:
        print(f"  => PASS ({len(crits)} criteria) — phase '{phase}' marked exited on attempt "
              f"{g['attempts']}")
        return 0

    limit = int(ctx.phases.get("max_gate_attempts") or 2)
    print(f"  => FAIL: {len(unmet)} unmet criterion/criteria, {len(ev_fails)} evidence failure(s) "
          f"(attempt {g['attempts']}/{limit})")
    if g["attempts"] >= limit:
        print("  => attempt limit reached — STOP REFINING. Record the unmet criteria as a waiver "
              "(reason + blast radius) or as a blocker, and say so in the report. Do not loop.")
    else:
        print("  => fix the gap and re-run the gate, or record a waiver with a reason and a blast "
              "radius. Leaving the phase without doing either is check E15.")
    return 1


def cmd_enter(ctx: Ctx, args) -> int:
    """Stamp entry into a phase, so E16 can later tell whether anything happened inside it."""
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"unknown phase '{phase}'")
        return 2
    ph = ctx.phases.setdefault("phases", {}).setdefault(phase, {})
    if ph.get("status") in ("exited", "waived"):
        print(f"phase '{phase}' is already {ph['status']} — re-entering (back-edge)")
    ph["status"] = "in-progress"
    ph["entered_at"] = utcnow()
    ph["exited_at"] = ""
    ctx.phases["updated_at"] = utcnow()
    with open(ctx.phases_path, "w") as fh:
        json.dump(ctx.phases, fh, indent=2)
        fh.write("\n")
    print(f"phase '{phase}' entered at {ph['entered_at']}")
    return 0


# ---------------------------------------------------------------------------- diagnosis ----------

# Signals worth putting in front of a reviewer, in the order a diagnosis reads them.
# Counter column names differ between tools and arches ("XVE Stall Send[%]", "SendStall",
# "XVE_STALL_SEND"), so each pattern is order-agnostic on the "<reason> stall" pair.
def _stall(reason):
    return rf"(?:xve[_ ]?)?(?:stall.*{reason}|{reason}.*stall)"


SIGNAL_PATTERNS = [
    ("EU active %", r"(?:xve|eu)[_ ]?active"),
    ("SendStall %", _stall("send")),
    ("SbidStall %", _stall("sbid")),
    ("PipeStall %", _stall("pipe")),
    ("DistStall %", _stall("dist")),
    ("SyncStall %", _stall("sync")),
    ("ControlStall %", _stall("control")),
    ("InstrFetchStall %", _stall("instr")),
    ("Occupancy %", r"occupancy"),
    ("GPU read BW (GB/s)", r"(?:gpu|memory).*read.*(?:bandwidth|bytes|gb/s)"),
    ("GPU write BW (GB/s)", r"(?:gpu|memory).*write.*(?:bandwidth|bytes|gb/s)"),
    ("L1 hit %", r"\bl1\b.*(?:hit|miss)"),
    ("L3 hit %", r"\b(?:l3|llc)\b.*(?:hit|miss)"),
]

# Which direction a fix for each dominant stall should push the metric it blamed.
BLAME_DIRECTION = {
    "sendstall": ("SendStall %", "down"),
    "sbidstall": ("SbidStall %", "down"),
    "pipestall": ("PipeStall %", "down"),
    "diststall": ("DistStall %", "down"),
    "syncstall": ("SyncStall %", "down"),
    "controlstall": ("ControlStall %", "down"),
    "instrfetchstall": ("InstrFetchStall %", "down"),
    "occupancy": ("Occupancy %", "up"),
}


def read_metric_csvs(d):
    """Average each numeric metric across kernel instances in every *_transposed.csv under d."""
    vals = {}
    if not os.path.isdir(d):
        return vals
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, newline="", errors="replace") as fh:
                    rows = list(csv.reader(fh))
            except OSError:
                continue
            if not rows:
                continue
            # Find the header row: the first row where >1 cell is a plausible column name.
            hdr_i = next((i for i, r in enumerate(rows[:8]) if len(r) > 1), 0)
            hdr = [c.strip() for c in rows[hdr_i]]
            acc = {}
            for r in rows[hdr_i + 1:]:
                for i, cell in enumerate(r[: len(hdr)]):
                    v = fnum(cell.strip().rstrip("%"))
                    if v is None:
                        continue
                    acc.setdefault(hdr[i], []).append(v)
            for name, xs in acc.items():
                if not name or not xs:
                    continue
                prev = vals.get(name)
                merged = (prev or []) + xs
                vals[name] = merged
    return {k: sum(v) / len(v) for k, v in vals.items() if v}


def pick_signals(raw):
    out = {}
    for label, pat in SIGNAL_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        hits = [(k, v) for k, v in raw.items() if rx.search(k)]
        if hits:
            hits.sort(key=lambda kv: len(kv[0]))
            out[label] = hits[0][1]
    return out


def count_spills(d):
    if not os.path.isdir(d):
        return None
    n = 0
    found = False
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if not (fn.endswith(".asm") or ".asm" in fn):
                continue
            found = True
            try:
                txt = open(os.path.join(root, fn), errors="replace").read()
            except OSError:
                continue
            n += len(re.findall(r"\bspill\b", txt, re.IGNORECASE))
    return n if found else None


def cmd_diagnosis(ctx: Ctx, args) -> int:
    """Turn opt_<id>/baseline vs final into a falsifiable claim check a human can read."""
    kids = [args.kernel] if args.kernel else sorted(ctx.optim.keys())
    if not kids:
        print("diagnosis: no optimization state found", file=sys.stderr)
        return 0
    rc = 0
    for kid in kids:
        optdir = os.path.join(ctx.reports, f"opt_{kid}")
        base_raw = read_metric_csvs(os.path.join(optdir, "baseline"))
        fin_raw = read_metric_csvs(os.path.join(optdir, "final"))
        if not base_raw and not fin_raw:
            print(f"diagnosis: {kid}: no metric CSVs under {ctx.rel(optdir)}/ — skipped", file=sys.stderr)
            rc = max(rc, 1)
            continue
        base, fin = pick_signals(base_raw), pick_signals(fin_raw)

        o = ctx.optim.get(kid) or {}
        diag = o.get("diagnosis") or {}
        dom = (diag.get("dominant_stall") or "").strip()
        blamed, direction = BLAME_DIRECTION.get(re.sub(r"[^a-z]", "", dom.lower()), (None, None))

        b_bench = ctx.records_for("bench", kid, label="baseline")
        f_bench = ctx.records_for("bench", kid, label="final")
        rows = []
        t0 = t1 = None
        if b_bench and f_bench:
            t0 = fnum((b_bench[-1].get("measurements") or {}).get("median_ms"))
            t1 = fnum((f_bench[-1].get("measurements") or {}).get("median_ms"))
        if t0 is None:
            t0 = fnum(o.get("baseline"))
        if t1 is None:
            t1 = fnum((o.get("best") or {}).get("value"))
        if t0 is not None and t1 is not None:
            rows.append(("Time / metric (median)", t0, t1, "down"))

        for label in [s[0] for s in SIGNAL_PATTERNS]:
            if label in base or label in fin:
                want = direction if label == blamed else ""
                rows.append((label, base.get(label), fin.get(label), want))

        sb, sf = count_spills(os.path.join(optdir, "baseline")), count_spills(os.path.join(optdir, "final"))
        if sb is not None or sf is not None:
            rows.append(("Register spill mentions (asm)", sb, sf, ""))

        # Verdict: did the metric the diagnosis BLAMED actually move the way it predicted?
        verdict, why = "unexplained", ""
        moved = None
        if blamed and blamed in base and blamed in fin:
            d = fin[blamed] - base[blamed]
            moved = (d < -1.0) if direction == "down" else (d > 1.0)
        faster = (t0 is not None and t1 is not None and t1 < t0 * 0.98)
        if blamed is None:
            verdict, why = "unchecked", (
                "no `diagnosis.dominant_stall` recorded, so there is no hypothesis to test — "
                "the before/after numbers are shown but nothing was predicted."
            )
        elif moved and faster:
            verdict, why = "supported", (
                f"`{blamed}` moved {direction} as predicted and the measured time fell with it."
            )
        elif moved and not faster:
            verdict, why = "unexplained", (
                f"`{blamed}` moved {direction} as predicted, but the measured time did not improve — "
                "the blamed signal was not the binding constraint."
            )
        elif faster:
            verdict, why = "unexplained", (
                f"the kernel got faster, but `{blamed}` did not move as predicted — the speedup came "
                "from somewhere other than the diagnosed bottleneck. Re-diagnose before generalising."
            )
        else:
            verdict, why = "unexplained", "neither the blamed signal nor the runtime moved materially."

        out = [
            f"# {kid} — optimization diagnosis check",
            "",
            "<!-- GENERATED by `.sycl/scripts/evidence.sh diagnosis`. Do not hand-edit: regenerate. -->",
            "",
            f"**Claim under test:** bottleneck `{diag.get('bottleneck_class','unknown')}`"
            + (f", dominated by `{dom}`" if dom else "")
            + ".",
            "",
            f"- Baseline profile: `{ctx.rel(os.path.join(optdir,'baseline'))}/`"
            + (f" · bench `{b_bench[-1]['run_id']}`" if b_bench else " · _no bench record_"),
            f"- Final profile: `{ctx.rel(os.path.join(optdir,'final'))}/`"
            + (f" · bench `{f_bench[-1]['run_id']}`" if f_bench else " · _no bench record_"),
            f"- Kept trials: {sum(1 for t in (o.get('trials') or []) if t.get('decision')=='keep')}"
            f" of {len(o.get('trials') or [])} · stop reason: `{o.get('stop_reason','-')}`",
            f"- Best commit: `{(o.get('best') or {}).get('commit','-')}`",
            "",
            "| Signal | Baseline | Final | Δ | Predicted | Note |",
            "|---|---:|---:|---:|---|---|",
        ]
        for label, a, b, want in rows:
            fa = "—" if a is None else f"{a:,.4g}"
            fb = "—" if b is None else f"{b:,.4g}"
            if a is None or b is None:
                delta = "—"
            elif "%" in label:
                delta = f"{b-a:+.2f} pp"
            elif a:
                delta = f"{(b-a)/a*100:+.1f}%"
            else:
                delta = f"{b-a:+.4g}"
            note = "**the blamed signal**" if (blamed and label == blamed) else ""
            out.append(f"| {label} | {fa} | {fb} | {delta} | {want or '—'} | {note} |")

        out += [
            "",
            f"**Verdict: {verdict}.** {why}",
            "",
            "_Signals are averaged across kernel instances from the `*_transposed.csv` counter files "
            "in each profile directory; spill counts are occurrences of `spill` in the IGC asm dump. "
            "The raw files beside this report are the primary evidence — this table is a reading aid._",
            "",
        ]
        os.makedirs(optdir, exist_ok=True)
        dst = os.path.join(optdir, "DIAGNOSIS.md")
        with open(dst, "w") as fh:
            fh.write("\n".join(out))
        print(f"diagnosis: {kid}: {verdict} -> {ctx.rel(dst)}")
    return rc


# ---------------------------------------------------------------------------- audit --------------

def _fmt_rec_link(ctx: Ctx, r):
    if not r:
        return "—"
    log = r.get("log", "")
    return f"[`{r['run_id']}`]({log})" if log else f"`{r['run_id']}`"


def cmd_audit(ctx: Ctx, args) -> int:
    """AUDIT.md — the page a reviewer reads beside FINAL_REPORT.md to check every claim."""
    findings = collect_findings(ctx)
    fails = [x for x in findings if x.severity == "fail"]
    warns = [x for x in findings if x.severity == "warn"]
    proj = ctx.project.get("name", "(project)")
    env = ctx.project.get("env") or {}

    L = [
        f"# {proj} — Evidence & Audit Index",
        "",
        "<!-- GENERATED by `.sycl/scripts/evidence.sh audit`. Do not hand-edit: regenerate. -->",
        "",
        "`FINAL_REPORT.md` says what was achieved. **This page says where to check it.** Every claim "
        "below links to the run record that produced it and to the raw, unedited log the machine "
        "printed. Nothing here has to be taken on trust: re-run "
        "`.sycl/scripts/evidence.sh verify` to re-hash every log and re-check every claim.",
        "",
        f"**Generated:** {utcnow()} · **Phase:** `{ctx.project.get('phase','-')}` · "
        f"**Records:** {len(ctx.records)}",
        "",
        "## 0. Verification status",
        "",
        f"- **{len(fails)} failure(s)**, **{len(warns)} warning(s)** across "
        f"{len(ctx.kernels)} kernel(s).",
    ]
    if fails:
        L.append("- Failures (a claim with no evidence, or evidence that no longer hashes):")
        L += [f"  - **[{x.check}] {x.subject}** — {x.message}" for x in fails]
    if warns:
        L.append("- Warnings (evidence exists but is weaker than it should be):")
        L += [f"  - [{x.check}] {x.subject} — {x.message}" for x in warns]
    if not findings:
        L.append("- **All claims backed; all log hashes verified.**")

    # --- environment -----------------------------------------------------------------------------
    L += [
        "",
        "## 1. Where the numbers were measured",
        "",
        "Two numbers are only comparable when they carry the same fingerprint. An unpinned GPU clock "
        "makes a benchmark unreproducible, so it is called out here rather than buried.",
        "",
        "| Fingerprint | GPU | arch | icpx | Freq pinned | Runs |",
        "|---|---|---|---|---|---:|",
    ]
    if ctx.envs:
        for e in ctx.envs.values():
            used = sum(1 for r in ctx.records if r.get("env_fingerprint") == e.get("id"))
            pinned = e.get("freq_pinned")
            mark = "yes" if pinned is True else (f"**{pinned}**" if pinned is not None else "?")
            L.append(f"| [`{e.get('id')}`](evidence/env/{e.get('id')}.json) | {e.get('gpu','?')} | "
                     f"{e.get('arch','?')} | {e.get('icpx','?')} | {mark} | {used} |")
    else:
        L.append(f"| _none captured_ | {env.get('gpu','?')} | {env.get('arch','?')} | "
                 f"{env.get('icpx','?')} | {env.get('freq_pinned','?')} | 0 |")

    # --- correctness -----------------------------------------------------------------------------
    L += [
        "",
        "## 2. Correctness — how do I know the tests really passed?",
        "",
        "The verdict column is the **process exit code**, not the agent's opinion. `Measured err` "
        "next to `Tolerance` is the line worth reading: it turns PASS into *PASS with margin*.",
        "",
        "| Kernel | Verdict | Exit | Measured err | Tolerance | Reference oracle | Commit | Raw log |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for kid in sorted(ctx.kernels):
        k = ctx.kernels[kid]
        if not _kernel_active(k):
            L.append(f"| {kid} | _{k.get('status')}_ | — | — | — | — | — | — |")
            continue
        ut = k.get("unit_test") or {}
        recs = ctx.resolve(ut.get("evidence")) or ctx.records_for("test", kid)
        r = recs[-1] if recs else None
        if not r:
            L.append(f"| {kid} | **{ut.get('result','-')}** | — | — | {ut.get('tolerance','-')} | "
                     f"— | — | **NO RECORD** |")
            continue
        m = r.get("measurements") or {}
        tol = m.get("tolerance") or {}
        toltxt = ", ".join(f"{k2}={v}" for k2, v in tol.items() if k2 != "source") or \
            (ut.get("tolerance") or "—")
        err = m.get("max_abs_err")
        L.append(
            f"| {kid} | {'PASS' if r.get('exit_code')==0 else '**FAIL**'} | {r.get('exit_code')} | "
            f"{err if err is not None else '—'} | {toltxt} | {m.get('reference','—')} | "
            f"`{r.get('commit','—')}`{' *(dirty)*' if r.get('dirty') else ''} | "
            f"[log]({r.get('log','')}) |"
        )

    # --- performance -----------------------------------------------------------------------------
    L += [
        "",
        "## 3. Performance — are the benchmark numbers real?",
        "",
        "Each row is backed by a record holding **every timed iteration**, the warm-up/iteration "
        "policy, the exact command, and the machine fingerprint — so the median can be recomputed "
        "and the run repeated.",
        "",
        "| Kernel | Baseline | Final | Change | Shapes | Iters | Baseline record | Final record |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for kid in sorted(ctx.kernels):
        k = ctx.kernels[kid]
        if not _kernel_active(k):
            continue
        perf = k.get("performance") or {}
        ev = perf.get("evidence") or {}
        b = ctx.resolve(ev.get("baseline")) or ctx.records_for("bench", kid, label="baseline")
        o = ctx.resolve(ev.get("optimized")) or ctx.records_for("bench", kid, label="final")
        shapes = {(r.get("measurements") or {}).get("shape") for r in ctx.records_for("bench", kid)}
        shapes.discard(None)
        t0 = fnum((b[-1].get("measurements") or {}).get("median_ms")) if b else None
        t1 = fnum((o[-1].get("measurements") or {}).get("median_ms")) if o else None
        chg = f"{t0/t1:.2f}x" if (t0 and t1) else "—"
        iters = (o[-1].get("measurements") or {}).get("iters") if o else None
        L.append(
            f"| {kid} | {perf.get('baseline','—')} | {perf.get('optimized','—')} | {chg} | "
            f"{len(shapes) or '—'} | {iters or '—'} | "
            f"{_fmt_rec_link(ctx, b[-1] if b else None)} | {_fmt_rec_link(ctx, o[-1] if o else None)} |"
        )

    L += ["", "### End-to-end", "",
          "| Snapshot | Total wall-clock | Cases | Record |", "|---|---:|---:|---|"]
    for label, snap in (("baseline", ctx.e2e_baseline), ("final", ctx.e2e_final)):
        if not snap:
            L.append(f"| {label} | _not captured_ | — | — |")
            continue
        recs = ctx.resolve(snap.get("evidence")) or ctx.records_for("e2e", label)
        L.append(f"| {label} | {snap.get('total_wallclock_ms','—')} ms | "
                 f"{len(snap.get('cases') or []) or 1} | "
                 f"{', '.join(_fmt_rec_link(ctx, r) for r in recs) or '**NO RECORD**'} |")

    # --- optimization direction ------------------------------------------------------------------
    L += [
        "",
        "## 4. Optimization — did it go in the right direction?",
        "",
        "Each optimization stated a hypothesis about the bottleneck. `DIAGNOSIS.md` puts the "
        "before/after hardware counters beside that hypothesis, so the claim is **falsifiable**: a "
        "kernel that got faster while the blamed metric never moved is marked `unexplained` rather "
        "than narrated away.",
        "",
        "| Kernel | Diagnosed bottleneck | Trials (kept) | Stop reason | Before/after profile | Check |",
        "|---|---|---:|---|---|---|",
    ]
    for kid in sorted(ctx.optim):
        o = ctx.optim[kid]
        d = o.get("diagnosis") or {}
        trials = o.get("trials") or []
        kept = sum(1 for t in trials if t.get("decision") == "keep")
        optdir = f"opt_{kid}"
        has_b = os.path.isdir(os.path.join(ctx.reports, optdir, "baseline"))
        has_f = os.path.isdir(os.path.join(ctx.reports, optdir, "final"))
        prof = " / ".join(
            [f"[baseline]({optdir}/baseline)" if has_b else "**missing**",
             f"[final]({optdir}/final)" if has_f else "**missing**"])
        diag_path = os.path.join(ctx.reports, optdir, "DIAGNOSIS.md")
        if os.path.exists(diag_path):
            txt = open(diag_path, errors="replace").read()
            m = re.search(r"\*\*Verdict:\s*(\w+)", txt)
            check = f"[{m.group(1) if m else 'see'}]({optdir}/DIAGNOSIS.md)"
        else:
            check = "**no DIAGNOSIS.md**"
        L.append(f"| {kid} | {d.get('bottleneck_class','?')}"
                 f"{' / ' + d['dominant_stall'] if d.get('dominant_stall') else ''} | "
                 f"{len(trials)} ({kept}) | `{o.get('stop_reason','-')}` | {prof} | {check} |")
    if not ctx.optim:
        L.append("| _no optimization state yet_ | — | — | — | — | — |")

    # --- backend integration ---------------------------------------------------------------------
    # The question a kernel-level audit cannot answer: does the PROJECT run on the Intel GPU?
    if ctx.readiness or ctx.integration:
        LEVELS = ["L0", "L1", "L2", "L3", "L4"]
        MEANING = {
            "L0": "kernels pass their own tests; the project does not build or run with them",
            "L1": "the project's own build produces an artifact with the SYCL backend",
            "L2": "the project's own test suite passes with the backend selected",
            "L3": "the real entrypoint runs and matches a reference",
            "L4": "end-to-end baseline -> final measured on the same machine",
        }
        GATE_CLASS = {"L1": "build", "L2": "integration", "L3": "e2e", "L4": "e2e"}
        achieved = ctx.readiness.get("achieved_level", "L0")
        target = ctx.readiness.get("target_level") or "(undeclared)"
        L += [
            "",
            "## 5. Backend integration — does the project actually run on the Intel GPU?",
            "",
            "Ported kernels are not a backend. This section is the ladder from *the kernels compile* "
            "to *the workload runs correctly and fast*, and each rung is a run record of the "
            "**project's own** artifact — not of a standalone test binary.",
            "",
            f"- **Achieved: {achieved}** ({MEANING.get(achieved, '?')}) · **Target: {target}**",
            "",
            "| Level | Gate | Status | Evidence |",
            "|---|---|---|---|",
        ]
        for lv in LEVELS:
            g = (ctx.readiness.get("gates") or {}).get(lv)
            if not g:
                continue
            recs = ctx.resolve(g.get("evidence"))
            ev = ", ".join(_fmt_rec_link(ctx, r) for r in recs) or "—"
            L.append(f"| {lv} | {g.get('name', GATE_CLASS.get(lv, '-'))} | "
                     f"{g.get('status', 'pending')} | {ev} |")
        surfaces = (ctx.integration or {}).get("surfaces") or []
        if surfaces:
            L += [
                "",
                "**Surfaces** — every one is listed whether or not it applies, so coverage cannot be "
                "improved by shrinking the list.",
                "",
                "| Surface | Status | Blocks | Effort (wk) | Evidence |",
                "|---|---|---|---|---|",
            ]
            for s in surfaces:
                ew = s.get("effort_weeks") or []
                ew_s = f"{ew[0]:g}–{ew[1]:g}" if len(ew) == 2 and any(ew) else "—"
                ev = ", ".join(_fmt_rec_link(ctx, r) for r in ctx.resolve(s.get("evidence"))) or "—"
                L.append(f"| {s.get('id','?')} | {s.get('status','-')} | "
                         f"{s.get('blocks_level','-')} | {ew_s} | {ev} |")
        waivers = ctx.readiness.get("waivers") or []
        if waivers:
            L += ["", "**Waived** — recorded, not hidden. A silent stub is a defect; a declared "
                      "waiver is a scoped engagement.", ""]
            L += [f"- **{w.get('what','?')}** — {w.get('reason','?')} · "
                  f"blast radius: {w.get('blast_radius','?')}" for w in waivers]
        residual = ctx.readiness.get("residual") or []
        if residual:
            total_wk = sum((r.get("effort_weeks") or [0, 0])[-1] for r in residual)
            L += ["", f"**Residual work** — what is left, and what it costs (≈{total_wk:g} "
                      f"engineer-weeks total).", "",
                  "| What | Unlocks | Effort (wk) | Risk |", "|---|---|---|---|"]
            for r in residual:
                ew = r.get("effort_weeks") or []
                ew_s = f"{ew[0]:g}–{ew[1]:g}" if len(ew) == 2 else "—"
                L.append(f"| {r.get('what','?')} | {r.get('unlocks','?')} | {ew_s} | "
                         f"{r.get('risk','-')} |")

    # --- phase contract --------------------------------------------------------------------------
    if ctx.phases:
        ph_all = ctx.phases.get("phases") or {}
        L += [
            "",
            "## 6. Was every step actually taken?",
            "",
            "The claim checks above can only ask whether what the run *said* is true. They cannot "
            "notice a phase that was skipped, because a skipped phase makes no claim. This section "
            "is the answer to the other question: each phase has exit criteria a machine checks, "
            "and no phase may be left until they are met or explicitly waived.",
            "",
            "| Phase | Status | Gate | Attempts | Unmet / waived |",
            "|---|---|---|---:|---|",
        ]
        for name in PHASE_ORDER:
            ph = ph_all.get(name) or {}
            g = ph.get("gate") or {}
            st = ph.get("status", "pending")
            mark = {"exited": "done", "waived": "**waived**", "in-progress": "_in progress_"}.get(st, st)
            unmet = g.get("unmet") or []
            wv = ph.get("waivers") or []
            note = ""
            if unmet:
                note = "; ".join(u.split(":")[0] for u in unmet[:3])
            if wv:
                note = (note + " · " if note else "") + f"waived: {', '.join(w.get('criterion','?') for w in wv)}"
            L.append(f"| `{name}` | {mark} | {g.get('result','pending')} | "
                     f"{g.get('attempts',0)} | {note or '—'} |")
        skipped = [n for n in PHASE_ORDER if (ph_all.get(n) or {}).get("status") == "waived"]
        if skipped:
            L += ["", "**Phases skipped by waiver** — each states why and what is unknown as a "
                      "result. A waived phase is a scoped decision; an unrecorded one is a hole.", ""]
            for n in skipped:
                for w in (ph_all.get(n) or {}).get("waivers") or []:
                    L.append(f"- `{n}` / {w.get('criterion','?')} — {w.get('reason','?')} · "
                             f"blast radius: {w.get('blast_radius','?')}")
        retried = [(n, (ph_all.get(n) or {}).get("gate", {}).get("attempts", 0))
                   for n in PHASE_ORDER if ((ph_all.get(n) or {}).get("gate") or {}).get("attempts", 0) > 1]
        if retried:
            L += ["", "Gates that took more than one attempt (the phase was refined before it was "
                      "allowed to close): "
                      + ", ".join(f"`{n}` ×{a}" for n, a in retried) + ".", ""]
        changes = (ctx.phases.get("scope") or {}).get("changes") or []
        if changes:
            L += ["", "**Scope changes** — the work-item denominator moved during the run, so "
                      "percentages before and after these points are not comparable.", ""]
            for ch in changes:
                L.append(f"- {ch.get('ts','?')} ({ch.get('kind','?')}): removed "
                         f"{', '.join(ch.get('removed') or []) or '—'}, added "
                         f"{', '.join(ch.get('added') or []) or '—'} — {ch.get('reason','?')}")

    # --- reproduce -------------------------------------------------------------------------------
    L += [
        "",
        "## 7. Reproduce it yourself",
        "",
        "Every record stores the exact command, the commit and the runner it used. To repeat any "
        "measurement:",
        "",
        "```bash",
        ".sycl/scripts/evidence.sh show <run-id>      # command, commit, env, verdict",
        ".sycl/scripts/evidence.sh verify             # re-hash every log, re-check every claim",
        "git checkout <commit>                        # the exact code that produced the number",
        ".sycl/scripts/run.sh bench \"<command>\"       # run it again on the same machine",
        "```",
        "",
        "## 8. Full record index",
        "",
        "| Run id | Class | Subject | Label | When | Verdict | Commit | Log |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ctx.records:
        L.append(f"| `{r['run_id']}` | {r.get('class')} | {r.get('subject')} | "
                 f"{r.get('label','—')} | {r.get('ts','')} | "
                 f"{r.get('verdict')} ({r.get('exit_code')}) | `{r.get('commit','—')}` | "
                 f"[log]({r.get('log','')}) |")
    if not ctx.records:
        L.append("| _no run records captured — nothing in this report is independently verifiable_ "
                 "| | | | | | | |")
    L.append("")

    os.makedirs(ctx.reports, exist_ok=True)
    dst = os.path.join(ctx.reports, "AUDIT.md")
    with open(dst, "w") as fh:
        fh.write("\n".join(L))
    print(f"audit: {len(ctx.records)} record(s), {len(fails)} failure(s), {len(warns)} warning(s) "
          f"-> {ctx.rel(dst)}")
    return 0


def cmd_list(ctx: Ctx, args) -> int:
    for r in ctx.records_for(args.cls, args.subject):
        m = r.get("measurements") or {}
        extra = m.get("shape") or m.get("median_ms") or ""
        print(f"{r['run_id']}\t{r.get('class')}\t{r.get('subject')}\t{r.get('label','-')}\t"
              f"{r.get('verdict')}({r.get('exit_code')})\t{r.get('commit','-')}\t{extra}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="evidence_audit.py", add_help=True)
    p.add_argument("command", choices=["manifest", "verify", "audit", "diagnosis", "list",
                                       "gate", "enter"])
    p.add_argument("--sycl-dir", default=".sycl")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--class", dest="cls")
    p.add_argument("--subject")
    p.add_argument("--kernel")
    p.add_argument("--phase")
    p.add_argument("--dry-run", action="store_true",
                   help="gate: evaluate the criteria without recording the attempt")
    # argparse will not reliably bind a trailing positional that follows an option, so the optional
    # kernel/phase argument of `diagnosis` and `gate` is taken from the leftovers.
    args, extra = p.parse_known_args()
    if not args.kernel and extra:
        args.kernel = extra[0]
    if args.command in ("gate", "enter") and not args.phase and extra:
        args.phase = extra[0]

    ctx = Ctx(args.sycl_dir)
    return {
        "manifest": cmd_manifest,
        "verify": cmd_verify,
        "audit": cmd_audit,
        "diagnosis": cmd_diagnosis,
        "list": cmd_list,
        "gate": cmd_gate,
        "enter": cmd_enter,
    }[args.command](ctx, args)


if __name__ == "__main__":
    sys.exit(main())
