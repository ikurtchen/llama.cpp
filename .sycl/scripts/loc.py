#!/usr/bin/env python3
"""Measure the lines of CUDA/Triton source a migration covered, and the SYCL written for it.

Answers the question a reviewer always asks about a port: *how much code was actually migrated?*
LOC is captured in the **inventory** phase (source side, from the kernel spans the analysis skill
recorded) and topped up in **migrate** (SYCL side, from `sycl_impl`), then rolled up for the report.

Counting rules (so the number means something):
  - **code**    = lines with executable/declarative content after comments and blank lines are removed.
                  This is the headline LOC.
  - **comment** = comment-only lines.  **blank** = whitespace-only.  **raw** = every physical line.
  - Kernel spans (not whole files) are counted when the symbol can be resolved, so a 3000-line
    header does not inflate a 40-line kernel.
  - Spans are merged per file before totalling, so two kernels sharing a header are never
    double-counted in the project total.

Usage (normally via .sycl/scripts/loc.sh):
  loc.py count  <spec>...                 # print counts for files/spans/symbols
  loc.py spans  <file> <sym1,sym2,...>    # resolve symbol -> line span
  loc.py kernel <id> [--source SPEC]... [--sycl SPEC]... [--no-mirror] [--dry-run]
  loc.py all    [--no-mirror]             # measure every kernel in the index, then roll up
  loc.py rollup                           # aggregate state/kernels/*.json -> state/loc.json
  loc.py scan                             # project-wide CUDA/Triton/SYCL totals

Spec forms:
  path/to/file.cu            whole file
  path/to/file.cu:120-198    explicit line span (1-based, inclusive)
  path/to/file.cuh:foo,bar   symbol list (kernel/function names) resolved to spans
"""

import argparse
import datetime
import json
import os
import re
import sys

C_EXT = {".cu", ".cuh", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".ipp", ".inl"}
PY_EXT = {".py"}
SKIP_DIRS = {".git", ".sycl", ".github", ".claude", ".opencode", ".agents", ".vscode",
             "build", "Build", "out", "dist", ".venv", "venv", "env",
             "__pycache__", "node_modules", "site-packages", ".mypy_cache", ".pytest_cache"}
MAX_BYTES = 4 * 1024 * 1024


# --------------------------------------------------------------------------- helpers
def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lang_of(path):
    ext = os.path.splitext(path)[1]
    if ext in C_EXT:
        return "c"
    if ext in PY_EXT:
        return "py"
    return None


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def analyze_c(lines):
    """-> list of (kind, code_text) with comments and string bodies removed from code_text."""
    out = []
    in_block = False
    for line in lines:
        code = []
        had_comment = False
        i, n = 0, len(line)
        while i < n:
            if in_block:
                had_comment = True
                j = line.find("*/", i)
                if j < 0:
                    i = n
                else:
                    in_block = False
                    i = j + 2
            elif line.startswith("//", i):
                had_comment = True
                i = n
            elif line.startswith("/*", i):
                had_comment = True
                in_block = True
                i += 2
            elif line[i] in "\"'":
                q = line[i]
                code.append(q)
                i += 1
                while i < n:
                    if line[i] == "\\":
                        i += 2
                        continue
                    if line[i] == q:
                        code.append(q)
                        i += 1
                        break
                    i += 1
            else:
                code.append(line[i])
                i += 1
        text = "".join(code).strip()
        out.append(("code" if text else ("comment" if had_comment else "blank"), text))
    return out


TRIPLE = ('"""', "'''")


def analyze_py(lines):
    """-> list of (kind, code_text). Statement-level triple-quoted blocks count as comments."""
    out = []
    in_tq = None          # the active triple-quote delimiter
    tq_is_doc = False     # opened with no code before it -> docstring, i.e. comment
    for line in lines:
        code = []
        had_comment = False
        i, n = 0, len(line)
        while i < n:
            if in_tq:
                j = line.find(in_tq, i)
                if tq_is_doc:
                    had_comment = True
                if j < 0:
                    i = n
                else:
                    i = j + 3
                    in_tq = None
                    tq_is_doc = False
                continue
            ch = line[i]
            if ch == "#":
                had_comment = True
                i = n
            elif line.startswith(TRIPLE[0], i) or line.startswith(TRIPLE[1], i):
                delim = line[i:i + 3]
                end = line.find(delim, i + 3)
                opened_bare = not "".join(code).strip()
                if end < 0:
                    in_tq = delim
                    tq_is_doc = opened_bare
                    if opened_bare:
                        had_comment = True
                    else:
                        code.append("s")
                    i = n
                else:
                    if opened_bare:
                        had_comment = True
                    else:
                        code.append("s")
                    i = end + 3
            elif ch in "\"'":
                code.append("s")
                i += 1
                while i < n:
                    if line[i] == "\\":
                        i += 2
                        continue
                    if line[i] == ch:
                        i += 1
                        break
                    i += 1
            else:
                code.append(ch)
                i += 1
        text = "".join(code).strip()
        out.append(("code" if text else ("comment" if had_comment else "blank"), text))
    return out


def analyze(path):
    lines = read_lines(path)
    kind = lang_of(path)
    if kind == "py":
        return lines, analyze_py(lines)
    return lines, analyze_c(lines)


# --------------------------------------------------------------------------- symbol spans
def c_symbol_spans(rows, sym):
    """Resolve a C/CUDA function/kernel definition to a 1-based inclusive line span."""
    pat = re.compile(r"\b" + re.escape(sym) + r"\s*\(")
    spans = []
    for idx, (kind, text) in enumerate(rows):
        if kind != "code" or not pat.search(text):
            continue
        # Must be a definition: a '{' must appear before any ';' within a short window.
        buf, open_at, semi = "", None, False
        for j in range(idx, min(idx + 20, len(rows))):
            buf = rows[j][1]
            pos_b, pos_s = buf.find("{"), buf.find(";")
            if pos_s >= 0 and (pos_b < 0 or pos_s < pos_b):
                semi = True
                break
            if pos_b >= 0:
                open_at = j
                break
        if semi or open_at is None:
            continue
        # Walk up over a multi-line signature (`__global__ void __launch_bounds__(...)` etc).
        start = idx
        for j in range(idx - 1, max(idx - 6, -1), -1):
            k, t = rows[j]
            if k != "code" or t.endswith((";", "}", "{", ":")):
                break
            start = j
        # Balance braces from the opening one.
        depth, end = 0, None
        for j in range(open_at, len(rows)):
            t = rows[j][1]
            for ch in t:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            if end is not None:
                break
        if end is None:
            continue
        spans.append((start + 1, end + 1))
    return spans


def py_symbol_spans(lines, rows, sym):
    """Resolve a Python (Triton) `def` — including its decorators — to a 1-based span."""
    pat = re.compile(r"^\s*def\s+" + re.escape(sym) + r"\s*\(")
    spans = []
    for idx, line in enumerate(lines):
        if not pat.match(line):
            continue
        indent = len(line) - len(line.lstrip())
        start = idx
        for j in range(idx - 1, -1, -1):
            s = lines[j].strip()
            if s.startswith("@") or (s and rows[j][0] == "code" and start - 1 == j and s.endswith((",", "("))):
                start = j
            elif not s:
                continue
            else:
                break
        end = len(lines) - 1
        for j in range(idx + 1, len(lines)):
            s = lines[j]
            if not s.strip():
                continue
            if len(s) - len(s.lstrip()) <= indent:
                end = j - 1
                break
        while end > idx and not lines[end].strip():
            end -= 1
        spans.append((start + 1, end + 1))
    return spans


def symbol_spans(path, sym):
    lines, rows = analyze(path)
    if lang_of(path) == "py":
        return py_symbol_spans(lines, rows, sym)
    return c_symbol_spans(rows, sym)


_SRC_CACHE = {}


def source_files(root):
    """Every C/CUDA/Python source in the project — used when a symbol names the wrong file."""
    if root in _SRC_CACHE:
        return _SRC_CACHE[root]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if lang_of(fn) is None:
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) <= MAX_BYTES:
                    out.append(os.path.relpath(p, root))
            except OSError:
                pass
    _SRC_CACHE[root] = out
    return out


def find_symbol(root, sym):
    """-> (relpath, spans) for the first project file that actually defines `sym`, else (None, None)."""
    for rel in source_files(root):
        try:
            spans = symbol_spans(os.path.join(root, rel), sym)
        except OSError:
            continue
        if spans:
            return rel, spans
    return None, None


# --------------------------------------------------------------------------- spec parsing
SPAN_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
PATH_RE = re.compile(r"[\w./\-]*\w+\.(?:cu|cuh|c|h|cpp|hpp|cc|cxx|hxx|inl|ipp|py)\b")
IDENT_RE = re.compile(r"[A-Za-z_]\w*")
RANGE_RE = re.compile(r"\b([A-Za-z_]\w*?)(\d+)\s*-\s*(\d+)\b")


def expand_ranges(text):
    """`foo_kernel1-8` is common shorthand in hand-written state -> foo_kernel1 ... foo_kernel8."""
    def rep(m):
        base, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        if b < a or b - a > 64:
            return m.group(0)
        return " ".join(f"{base}{i}" for i in range(a, b + 1))
    return RANGE_RE.sub(rep, text)


def resolve_spec(root, spec):
    """One spec -> (relpath, spans|None for whole file, unresolved[]).

    Accepts `path`, `path:START-END`, and `path:sym1,sym2`. Symbols that do not resolve in the named
    file are looked up project-wide (state files often name the header, not the .cu that defines the
    variant). A file whose symbols ALL fail is not counted at all — silently falling back to the whole
    file would inflate the migration size, which is the one thing this measurement must not do.
    """
    spec = (spec or "").strip()
    if not spec:
        return None, None, [], {}
    path, tail = spec, None
    if ":" in spec:
        head, maybe = spec.split(":", 1)
        head, maybe = head.strip(), maybe.strip()
        if head and maybe:
            path, tail = head, maybe
    abspath = path if os.path.isabs(path) else os.path.join(root, path)
    rel = os.path.relpath(abspath, root)
    known = os.path.isfile(abspath) and lang_of(abspath) is not None

    if not tail:
        return (rel, None, [], {}) if known else (None, None, [spec], {})

    m = SPAN_RE.match(tail)
    if m:
        return (rel, [(int(m.group(1)), int(m.group(2)))], [], {}) if known else (None, None, [spec], {})

    extra, spans, missing = {}, [], []
    for sym in IDENT_RE.findall(expand_ranges(re.sub(r"\([^)]*\)", " ", tail))):
        found = symbol_spans(abspath, sym) if known else []
        if found:
            spans.extend(found)
            continue
        alt_rel, alt_spans = find_symbol(root, sym)
        if alt_spans:
            extra.setdefault(alt_rel, []).extend(alt_spans)
        else:
            missing.append(f"{rel}:{sym}")
    return (rel if spans else None), (spans or None), missing, extra


def parse_field(text):
    """Split a free-form `source` / `sycl_impl` field into specs.

    These fields are written by hand and look like
      "llmc/attention.cuh: permute_kernel, unpermute_kernel"
      "dev/cuda/x.cu (k1-k9) / llmc/x.cuh (k10)"
      "dev/sycl/m.cpp (kernels 1-4) + dev/sycl/m_mkl.hpp (oneMKL)"
    so anchor on the file paths and attach whatever follows each one to it.
    """
    text = text or ""
    hits = list(PATH_RE.finditer(text))
    specs = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        seg = text[m.end():end]
        stripped = seg.lstrip()
        if stripped.startswith(":"):
            body = re.sub(r"\([^)]*\)", " ", stripped[1:])
            syms = IDENT_RE.findall(expand_ranges(body))
            specs.append(f"{m.group(0)}:{','.join(syms)}" if syms else m.group(0))
        else:
            specs.append(m.group(0))
    return specs


# --------------------------------------------------------------------------- counting
def merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def count(root, entries):
    """entries: {relpath: spans|None}. -> dict with code/comment/blank/raw + per-file detail."""
    total = {"code": 0, "comment": 0, "blank": 0, "raw": 0}
    files = []
    for rel, spans in sorted(entries.items()):
        abspath = os.path.join(root, rel)
        if not os.path.isfile(abspath) or os.path.getsize(abspath) > MAX_BYTES:
            continue
        lines, rows = analyze(abspath)
        sel = merge(spans) if spans else [(1, len(lines))]
        c = {"code": 0, "comment": 0, "blank": 0, "raw": 0}
        for s, e in sel:
            s, e = max(1, s), min(len(lines), e)
            for i in range(s - 1, e):
                c[rows[i][0]] += 1
                c["raw"] += 1
        for k in total:
            total[k] += c[k]
        files.append({"path": rel, "lang": "triton" if lang_of(abspath) == "py" else "c",
                      "spans": [f"{s}-{e}" for s, e in sel] if spans else "whole-file", **c})
    total["files"] = len(files)
    return total, files


# --------------------------------------------------------------------------- state access
def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def mirror_source(root, sycl_rel, already):
    """The framework mandates the SYCL port mirror the CUDA layout (dev/cuda/x.cu -> dev/sycl/x.cpp).
    Find the same-named original so variant files the analysis skill did not name are still counted."""
    base = os.path.splitext(os.path.basename(sycl_rel))[0]
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if os.sep + "sycl" in dirpath.replace(root, "", 1):
            continue
        for fn in filenames:
            if os.path.splitext(fn)[0] != base:
                continue
            ext = os.path.splitext(fn)[1]
            if ext in (".cu", ".cuh") or ext == ".py":
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                if rel not in already:
                    hits.append(rel)
    return hits


def add(entries, rel, spans):
    if rel is None:
        return
    if rel in entries:
        if entries[rel] is None or spans is None:
            entries[rel] = None
        else:
            entries[rel] = entries[rel] + spans
    else:
        entries[rel] = spans


# --------------------------------------------------------------------------- commands
def cmd_count(root, args):
    entries, missing = {}, []
    for spec in args.specs:
        rel, spans, miss, extra = resolve_spec(root, spec)
        add(entries, rel, spans)
        for r, s in extra.items():
            add(entries, r, s)
        missing += miss
    total, files = count(root, entries)
    print(json.dumps({"total": total, "files": files, "unresolved": missing}, indent=2))
    return 0


def cmd_spans(root, args):
    path = args.file if os.path.isabs(args.file) else os.path.join(root, args.file)
    out = {}
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        out[sym] = [f"{s}-{e}" for s, e in symbol_spans(path, sym)]
    print(json.dumps(out, indent=2))
    return 0


def measure_kernel(root, state, kid, extra_source, extra_sycl, mirror=True):
    detail_path = os.path.join(state, "kernels", f"{kid}.json")
    detail = load(detail_path)
    if detail is None:
        return None, f"no detail file for kernel '{kid}'"

    src_entries, sycl_entries, missing = {}, {}, []
    keep = (detail.get("loc") or {})
    for spec in (keep.get("source_spec") or []) + parse_field(detail.get("source", "")) + list(extra_source):
        rel, spans, miss, extra = resolve_spec(root, spec)
        add(src_entries, rel, spans)
        for r, s in extra.items():
            add(src_entries, r, s)
        missing += miss
    for spec in (keep.get("sycl_spec") or []) + parse_field(detail.get("sycl_impl", "")) + list(extra_sycl):
        rel, spans, miss, extra = resolve_spec(root, spec)
        add(sycl_entries, rel, spans)
        for r, s in extra.items():
            add(sycl_entries, r, s)
        missing += miss

    mirrored = []
    if mirror:
        for rel in list(sycl_entries):
            for hit in mirror_source(root, rel, set(src_entries) | set(sycl_entries)):
                add(src_entries, hit, None)
                mirrored.append(hit)

    src_total, src_files = count(root, src_entries)
    sycl_total, sycl_files = count(root, sycl_entries)
    loc = {
        "source_lang": detail.get("source_lang", "cuda"),
        "source": {k: src_total[k] for k in ("code", "comment", "blank", "raw", "files")},
        "sycl": {k: sycl_total[k] for k in ("code", "comment", "blank", "raw", "files")},
        "source_files": src_files,
        "sycl_files": sycl_files,
        "mirrored_sources": mirrored,
        "unresolved": sorted(set(missing)),
        "method": "span" if any(f["spans"] != "whole-file" for f in src_files) else "file",
        "measured_at": now(),
    }
    for key in ("source_spec", "sycl_spec"):
        if keep.get(key):
            loc[key] = keep[key]
    return (detail_path, detail, loc), None


def cmd_kernel(root, state, args):
    res, err = measure_kernel(root, state, args.id, args.source, args.sycl, mirror=not args.no_mirror)
    if err:
        print(f"loc: {err}", file=sys.stderr)
        return 1
    detail_path, detail, loc = res
    if not args.dry_run:
        detail["loc"] = loc
        detail["updated_at"] = now()
        save(detail_path, detail)
        idx_path = os.path.join(state, "kernels", "index.json")
        idx = load(idx_path)
        if idx:
            for k in idx.get("kernels", []):
                if k.get("id") == args.id:
                    k["loc"] = {"source_code": loc["source"]["code"], "sycl_code": loc["sycl"]["code"]}
            idx["updated_at"] = now()
            save(idx_path, idx)
    print(json.dumps({"id": args.id, "loc": loc}, indent=2))
    return 0


def scan_project(root):
    out = {"cuda": {"code": 0, "raw": 0, "files": 0},
           "triton": {"code": 0, "raw": 0, "files": 0},
           "sycl": {"code": 0, "raw": 0, "files": 0}}
    paths = {"cuda": [], "triton": [], "sycl": []}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1]
            abspath = os.path.join(dirpath, fn)
            rel = os.path.relpath(abspath, root)
            try:
                if os.path.getsize(abspath) > MAX_BYTES:
                    continue
            except OSError:
                continue
            bucket = None
            if ext in (".cu", ".cuh"):
                bucket = "cuda"
            elif ext == ".py":
                try:
                    head = open(abspath, "r", encoding="utf-8", errors="replace").read(8192)
                except OSError:
                    continue
                if "triton" in head and ("@triton.jit" in head or "triton.language" in head):
                    bucket = "triton"
            elif ext in (".cpp", ".hpp", ".cc", ".cxx"):
                try:
                    head = open(abspath, "r", encoding="utf-8", errors="replace").read(8192)
                except OSError:
                    continue
                if "sycl/sycl.hpp" in head or "CL/sycl.hpp" in head or "sycl::" in head:
                    bucket = "sycl"
            if not bucket:
                continue
            total, _ = count(root, {rel: None})
            out[bucket]["code"] += total["code"]
            out[bucket]["raw"] += total["raw"]
            out[bucket]["files"] += 1
            paths[bucket].append(rel)
    return out, paths


def cmd_scan(root, args):
    totals, _ = scan_project(root)
    print(json.dumps(totals, indent=2))
    return 0


def cmd_rollup(root, state, args):
    idx = load(os.path.join(state, "kernels", "index.json"), {}) or {}
    kernels = idx.get("kernels", [])
    per_kernel, by_lang = [], {}
    uniq_src, uniq_sycl = {}, {}
    for k in kernels:
        kid = k.get("id")
        if k.get("status") == "skipped":
            continue
        detail = load(os.path.join(state, "kernels", f"{kid}.json"), {}) or {}
        loc = detail.get("loc")
        if not loc:
            per_kernel.append({"id": kid, "measured": False})
            continue
        lang = loc.get("source_lang", "cuda")
        b = by_lang.setdefault(lang, {"code": 0, "raw": 0, "kernels": 0})
        b["code"] += loc["source"]["code"]
        b["raw"] += loc["source"]["raw"]
        b["kernels"] += 1
        for f in loc.get("source_files", []):
            spans = None if f["spans"] == "whole-file" else [tuple(int(x) for x in s.split("-")) for s in f["spans"]]
            add(uniq_src, f["path"], spans)
        for f in loc.get("sycl_files", []):
            add(uniq_sycl, f["path"], None)
        per_kernel.append({
            "id": kid, "measured": True, "source_lang": lang,
            "status": k.get("status", "-"),
            "source_code": loc["source"]["code"], "source_raw": loc["source"]["raw"],
            "sycl_code": loc["sycl"]["code"], "sycl_raw": loc["sycl"]["raw"],
            "unresolved": loc.get("unresolved", []),
        })
    src_unique, _ = count(root, uniq_src)
    sycl_unique, _ = count(root, uniq_sycl)
    project, project_paths = scan_project(root)
    orig_total = project["cuda"]["code"] + project["triton"]["code"]
    unattributed = sorted(set(project_paths["sycl"]) - set(uniq_sycl))
    rollup = {
        "schema": "sycl-agent/loc",
        "generated_at": now(),
        "totals": {
            "source_migrated": {"code": src_unique["code"], "raw": src_unique["raw"],
                                "files": src_unique["files"]},
            "sycl_written": {"code": sycl_unique["code"], "raw": sycl_unique["raw"],
                             "files": sycl_unique["files"]},
            "by_source_lang": by_lang,
            "kernels_measured": sum(1 for p in per_kernel if p.get("measured")),
            "kernels_total": len(per_kernel),
            "sycl_per_source_ratio": round(sycl_unique["code"] / src_unique["code"], 2) if src_unique["code"] else 0,
            "coverage_pct_of_project": round(100.0 * src_unique["code"] / orig_total, 1) if orig_total else 0,
        },
        "project": project,
        "unattributed_sycl_files": unattributed,
        "kernels": per_kernel,
        "note": ("code = lines excluding comments and blanks; source_migrated merges overlapping "
                 "kernel spans per file so shared headers are counted once. "
                 "unattributed_sycl_files are SYCL sources in the tree that no kernel's sycl_impl "
                 "claims — fix the state rather than the number."),
    }
    save(os.path.join(state, "loc.json"), rollup)
    t = rollup["totals"]
    print(f"wrote {os.path.join(state, 'loc.json')}")
    print(f"  migrated source : {t['source_migrated']['code']} code lines "
          f"({t['source_migrated']['files']} files, {t['source_migrated']['raw']} raw)")
    print(f"  SYCL written    : {t['sycl_written']['code']} code lines "
          f"({t['sycl_written']['files']} files, {t['sycl_written']['raw']} raw)")
    print(f"  by source lang  : " + ", ".join(f"{k}={v['code']}" for k, v in by_lang.items()))
    print(f"  measured        : {t['kernels_measured']}/{t['kernels_total']} kernels  ·  "
          f"{t['coverage_pct_of_project']}% of project CUDA/Triton code lines")
    if unattributed:
        print(f"  unattributed    : {len(unattributed)} SYCL file(s) no kernel claims — "
              f"{', '.join(unattributed[:5])}{' …' if len(unattributed) > 5 else ''}")
    return 0


def cmd_all(root, state, args):
    idx = load(os.path.join(state, "kernels", "index.json"), {}) or {}
    for k in idx.get("kernels", []):
        if k.get("status") == "skipped":
            continue
        kid = k.get("id")
        res, err = measure_kernel(root, state, kid, [], [], mirror=not args.no_mirror)
        if err:
            print(f"loc: {kid}: {err}", file=sys.stderr)
            continue
        detail_path, detail, loc = res
        detail["loc"] = loc
        detail["updated_at"] = now()
        save(detail_path, detail)
        k["loc"] = {"source_code": loc["source"]["code"], "sycl_code": loc["sycl"]["code"]}
        print(f"  {kid}: source {loc['source']['code']} -> sycl {loc['sycl']['code']} code lines"
              + (f"  (unresolved: {len(loc['unresolved'])})" if loc["unresolved"] else ""))
    idx["updated_at"] = now()
    save(os.path.join(state, "kernels", "index.json"), idx)
    return cmd_rollup(root, state, args)


def main():
    ap = argparse.ArgumentParser(description="Count migrated CUDA/Triton LOC and the SYCL written for it.")
    ap.add_argument("--sycl-dir", default=None, help="path to .sycl/ (default: parent of this script)")
    ap.add_argument("--root", default=None, help="project root (default: parent of .sycl/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("count"); p.add_argument("specs", nargs="+")
    p = sub.add_parser("spans"); p.add_argument("file"); p.add_argument("symbols")
    p = sub.add_parser("kernel")
    p.add_argument("id")
    p.add_argument("--source", action="append", default=[])
    p.add_argument("--sycl", action="append", default=[])
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("all"); p.add_argument("--no-mirror", action="store_true")
    sub.add_parser("rollup")
    sub.add_parser("scan")

    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sycl_dir = os.path.abspath(args.sycl_dir) if args.sycl_dir else os.path.dirname(here)
    root = os.path.abspath(args.root) if args.root else os.path.dirname(sycl_dir)
    state = os.path.join(sycl_dir, "state")

    if args.cmd == "count":
        return cmd_count(root, args)
    if args.cmd == "spans":
        return cmd_spans(root, args)
    if args.cmd == "kernel":
        return cmd_kernel(root, state, args)
    if args.cmd == "all":
        return cmd_all(root, state, args)
    if args.cmd == "rollup":
        return cmd_rollup(root, state, args)
    if args.cmd == "scan":
        return cmd_scan(root, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
