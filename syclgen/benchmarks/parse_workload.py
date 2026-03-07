#!/usr/bin/env python3
"""Parse llama-bench workload benchmark log files into structured JSON.

Handles the markdown table format output by llama-bench, including tables
with and without the n_ubatch column (omitted when ub >= prompt length).
"""

import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_size_to_gib(size_str: str) -> float:
    """Convert size string like '604.15 MiB' or '8.11 GiB' to GiB float."""
    size_str = size_str.strip()
    m = re.match(r'([\d.]+)\s*(MiB|GiB)', size_str)
    if not m:
        raise ValueError(f"Cannot parse size: {size_str!r}")
    val = float(m.group(1))
    unit = m.group(2)
    if unit == 'MiB':
        val /= 1024.0
    return round(val, 2)


def parse_params(params_str: str) -> str:
    """Normalize params string like '596.05 M' -> '596.05M', '8.19 B' -> '8.19B'."""
    params_str = params_str.strip()
    # Remove spaces between number and suffix
    return re.sub(r'\s+', '', params_str)


def format_model_name(model_col: str) -> str:
    """Convert model column like 'qwen3 0.6B Q8_0' to 'Qwen3-0.6B-Q8_0'."""
    parts = model_col.strip().split()
    if len(parts) >= 3:
        # Capitalize first part, join with hyphens
        parts[0] = parts[0].capitalize()
        return '-'.join(parts)
    return '-'.join(parts)


def parse_table_row(line: str):
    """Parse a data row from a llama-bench markdown table.

    Returns a dict with parsed fields, or None if not a data row.
    Handles both 8-column (with n_ubatch) and 7-column (without n_ubatch) formats.
    """
    line = line.strip()
    if not line.startswith('|'):
        return None

    # Split by pipe, strip each cell
    cells = [c.strip() for c in line.split('|')]
    # Remove empty first/last elements from leading/trailing pipes
    cells = cells[1:-1]

    # Skip header/separator rows
    if len(cells) == 0:
        return None
    if cells[0].startswith('---') or cells[0].startswith(' ---'):
        return None
    # Skip if first cell is a header keyword
    if cells[0].strip().lower() == 'model':
        return None

    # Determine format: 8 cells = has n_ubatch, 7 cells = no n_ubatch
    if len(cells) == 8:
        # model | size | params | backend | ngl | n_ubatch | test | t/s
        model_raw = cells[0]
        size_str = cells[1]
        params_str = cells[2]
        backend = cells[3].strip()
        ngl = int(cells[4].strip())
        n_ubatch = int(cells[5].strip())
        test_str = cells[6].strip()
        ts_str = cells[7].strip()
    elif len(cells) == 7:
        # model | size | params | backend | ngl | test | t/s  (n_ubatch omitted)
        model_raw = cells[0]
        size_str = cells[1]
        params_str = cells[2]
        backend = cells[3].strip()
        ngl = int(cells[4].strip())
        n_ubatch = None  # Will be inferred from CMD line
        test_str = cells[5].strip()
        ts_str = cells[6].strip()
    else:
        return None

    # Parse throughput: "90.78 ± 0.65" or "9.53 ± 0.00"
    ts_match = re.match(r'([\d.]+)\s*[±]\s*([\d.]+)', ts_str)
    if not ts_match:
        return None

    throughput = float(ts_match.group(1))
    stddev = float(ts_match.group(2))

    return {
        'model_raw': model_raw.strip(),
        'size_str': size_str.strip(),
        'params_str': params_str.strip(),
        'backend': backend,
        'ngl': ngl,
        'n_ubatch': n_ubatch,
        'test': test_str,
        'throughput_ts': throughput,
        'stddev_ts': stddev,
    }


def parse_log_file(log_path: str) -> dict:
    """Parse a llama-bench workload log file into structured data."""

    with open(log_path, 'r') as f:
        content = f.read()

    # Strip ANSI codes for cleaner parsing
    content = strip_ansi(content)

    lines = content.splitlines()

    # Extract build hash from first occurrence of "build: XXXXX"
    build = None
    for line in lines:
        m = re.match(r'^build:\s+(\S+)', line.strip())
        if m:
            build = m.group(1)
            break

    # Extract timestamp from log filename
    log_basename = os.path.basename(log_path)
    ts_match = re.search(r'(\d{8}_\d{6})', log_basename)
    if ts_match:
        ts_str = ts_match.group(1)
        timestamp = datetime.strptime(ts_str, '%Y%m%d_%H%M%S').replace(
            tzinfo=timezone.utc
        ).isoformat()
    else:
        timestamp = datetime.now(timezone.utc).isoformat()

    # Parse line by line, tracking current CMD context for n_ubatch
    models = OrderedDict()  # model_name -> {size_gib, params, results: []}
    current_ub = None
    total_tests = 0

    for line in lines:
        stripped = strip_ansi(line).strip()

        # Track CMD lines to get ub value for rows missing n_ubatch
        cmd_match = re.search(r'-ub\s+(\d+)', stripped)
        if cmd_match:
            current_ub = int(cmd_match.group(1))

        # Try to parse as table data row
        row = parse_table_row(stripped)
        if row is None:
            continue

        model_name = format_model_name(row['model_raw'])
        size_gib = parse_size_to_gib(row['size_str'])
        params = parse_params(row['params_str'])

        # Use CMD-derived ub if table didn't have n_ubatch column
        n_ubatch = row['n_ubatch'] if row['n_ubatch'] is not None else current_ub

        if model_name not in models:
            models[model_name] = {
                'name': model_name,
                'size_gib': size_gib,
                'params': params,
                'results': [],
            }

        models[model_name]['results'].append({
            'test': row['test'],
            'n_ubatch': n_ubatch,
            'throughput_ts': row['throughput_ts'],
            'stddev_ts': row['stddev_ts'],
        })
        total_tests += 1

    # Count PASSED tests from log (each invocation produces pp + tg rows)
    passed_count = len(re.findall(r'\[SUCCESS\].*PASSED', strip_ansi(content)))

    # Build per-model test counts for notes
    model_list = list(models.values())

    # Determine note about 14B model
    notes_parts = []
    for m in model_list:
        if '14B' in m['name']:
            # Check which prompt lengths completed
            pp_tests = sorted(set(
                r['test'] for r in m['results'] if r['test'].startswith('pp')
            ))
            if pp_tests:
                completed_pp = ', '.join(pp_tests)
                notes_parts.append(
                    f"14B model ({m['size_gib']:.2f} GiB) exceeds B60 VRAM (~12 GiB); "
                    f"only pp128 batch sweep completed before timeout"
                )

    result = {
        'timestamp': timestamp,
        'hardware': 'Intel Arc Pro B60',
        'server': 'b60',
        'build': build,
        'log_file': log_basename,
        'models': model_list,
        'summary': {
            'total_tests': passed_count,
            'models_benchmarked': len(model_list),
            'notes': '; '.join(notes_parts) if notes_parts else '',
        },
    }

    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Default log file
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = os.path.join(repo_root, 'logs', 'b60_workload_20260307_201055.log')

    if not os.path.isfile(log_path):
        print(f"Error: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    output_path = os.path.join(script_dir, 'workload_baseline_results.json')

    print(f"Parsing: {log_path}")
    result = parse_log_file(log_path)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Output:  {output_path}")
    print(f"Models:  {result['summary']['models_benchmarked']}")
    print(f"Tests:   {result['summary']['total_tests']}")
    for m in result['models']:
        pp_tests = [r for r in m['results'] if r['test'].startswith('pp')]
        tg_tests = [r for r in m['results'] if r['test'].startswith('tg')]
        print(f"  {m['name']}: {len(m['results'])} results "
              f"({len(pp_tests)} pp, {len(tg_tests)} tg), "
              f"size={m['size_gib']} GiB, params={m['params']}")
    if result['summary']['notes']:
        print(f"Notes:   {result['summary']['notes']}")


if __name__ == '__main__':
    main()
