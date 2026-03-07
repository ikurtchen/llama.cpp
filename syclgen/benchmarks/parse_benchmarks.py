#!/usr/bin/env python3
"""Parse benchmark log files and create baseline_results.json."""

import json
import re
import sys
import os
from datetime import datetime, timezone
from collections import defaultdict

LOG_FILES = [
    "../../logs/b60_benchmark_20260307_194801.log",   # MUL_MAT
    "../../logs/b60_benchmark_20260307_195257.log",   # SOFT_MAX, ROPE, norms, FLASH_ATTN_EXT
    "../../logs/b60_benchmark_20260307_195258.log",   # ADD, SUB, MUL, DIV, CPY, etc.
    "../../logs/b60_benchmark_20260307_195524.log",   # MUL_MAT_ID, GET_ROWS, CONCAT, etc.
    "../../logs/b60_benchmark_20260307_195527.log",   # Activation/unary ops
    "../../logs/b60_benchmark_20260307_200131.log",   # Misc ops
    "../../logs/b60_benchmark_20260307_200133.log",   # Gated linear, backprop, RWKV, SSM
    "../../logs/b60_benchmark_20260307_200439.log",   # Batch 8: ops with no perf tests (DUP, CONT, etc.)
    "../../logs/b60_benchmark_20260307_200441.log",   # Batch 9: ROPE_BACK, TOP_K
]

# Pattern for successful benchmark lines:
# Two formats observed:
# 1) FLOPS-based:  OP_NAME(params):    N runs -   T us/run - X MFLOP/run - Y GFLOPS
# 2) BW-based:     OP_NAME(params):    N runs -   T us/run - X kB/run - Y GB/s
# Also: "not supported" lines

# Regex for successful run with FLOPS metric
re_flops = re.compile(
    r'^\s+(\w+)\(([^)]*)\):\s+'
    r'(\d+)\s+runs\s+-\s+'
    r'([\d.]+)\s+us/run\s+-\s+'
    r'([\d.]+)\s+([A-Za-z/]+)/run\s+-\s+'
    r'\x1b\[1;34m\s*([\d.]+)\s+(GFLOPS|TFLOPS)\x1b\[0m'
)

# Regex for successful run with bandwidth metric
re_bw = re.compile(
    r'^\s+(\w+)\(([^)]*)\):\s+'
    r'(\d+)\s+runs\s+-\s+'
    r'([\d.]+)\s+us/run\s+-\s+'
    r'([\d.]+)\s+(kB|MB|GB)/run\s+-\s+'
    r'\x1b\[1;34m\s*([\d.]+)\s+(GB/s|TB/s)\x1b\[0m'
)

# Regex for "not supported" lines
re_not_supported = re.compile(
    r'^\s+(\w+)\(([^)]*)\):\s+not supported'
)

def parse_log_file(filepath):
    """Parse a single log file and return list of results."""
    results = []
    not_supported = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Try FLOPS pattern
            m = re_flops.search(line)
            if m:
                op_name = m.group(1)
                params = m.group(2)
                runs = int(m.group(3))
                time_us = float(m.group(4))
                compute_val = float(m.group(5))
                compute_unit = m.group(6)
                perf_val = float(m.group(7))
                perf_unit = m.group(8)
                
                # Normalize to GFLOPS
                throughput_gflops = perf_val
                if perf_unit == "TFLOPS":
                    throughput_gflops = perf_val * 1000.0
                
                results.append({
                    "kernel": op_name,
                    "name": f"{op_name}({params})",
                    "time_us": time_us,
                    "runs": runs,
                    "throughput_gflops": round(throughput_gflops, 2),
                    "bandwidth_gbs": 0.0,
                    "metric_type": "flops"
                })
                continue
            
            # Try bandwidth pattern
            m = re_bw.search(line)
            if m:
                op_name = m.group(1)
                params = m.group(2)
                runs = int(m.group(3))
                time_us = float(m.group(4))
                data_val = float(m.group(5))
                data_unit = m.group(6)
                bw_val = float(m.group(7))
                bw_unit = m.group(8)
                
                # Normalize to GB/s
                bandwidth_gbs = bw_val
                if bw_unit == "TB/s":
                    bandwidth_gbs = bw_val * 1000.0
                
                results.append({
                    "kernel": op_name,
                    "name": f"{op_name}({params})",
                    "time_us": time_us,
                    "runs": runs,
                    "throughput_gflops": 0.0,
                    "bandwidth_gbs": round(bandwidth_gbs, 2),
                    "metric_type": "bandwidth"
                })
                continue
            
            # Try not supported pattern
            m = re_not_supported.search(line)
            if m:
                op_name = m.group(1)
                params = m.group(2)
                not_supported.append({
                    "kernel": op_name,
                    "name": f"{op_name}({params})",
                    "status": "not_supported"
                })
                continue
    
    return results, not_supported


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_results = []
    all_not_supported = []
    log_file_paths = []
    
    for log_rel in LOG_FILES:
        log_path = os.path.join(script_dir, log_rel)
        if not os.path.exists(log_path):
            print(f"WARNING: Log file not found: {log_path}")
            continue
        
        log_file_paths.append(os.path.abspath(log_path))
        results, not_supported = parse_log_file(log_path)
        all_results.extend(results)
        all_not_supported.extend(not_supported)
        print(f"Parsed {log_path}: {len(results)} results, {len(not_supported)} not supported")
    
    # Group by kernel name
    kernel_groups = defaultdict(list)
    for r in all_results:
        kernel_groups[r["kernel"]].append({
            "name": r["name"],
            "time_us": r["time_us"],
            "runs": r["runs"],
            "throughput_gflops": r["throughput_gflops"],
            "bandwidth_gbs": r["bandwidth_gbs"],
            "metric_type": r["metric_type"]
        })
    
    not_supported_groups = defaultdict(list)
    for ns in all_not_supported:
        not_supported_groups[ns["kernel"]].append(ns["name"])
    
    # Build output
    kernel_results = []
    for kernel_name in sorted(kernel_groups.keys()):
        test_cases = kernel_groups[kernel_name]
        times = [tc["time_us"] for tc in test_cases]
        kernel_results.append({
            "kernel": kernel_name,
            "num_test_cases": len(test_cases),
            "min_time_us": round(min(times), 2),
            "max_time_us": round(max(times), 2),
            "test_cases": test_cases
        })
    
    not_supported_list = []
    for kernel_name in sorted(not_supported_groups.keys()):
        not_supported_list.append({
            "kernel": kernel_name,
            "unsupported_cases": not_supported_groups[kernel_name]
        })
    
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "Intel Arc Pro B60",
        "device": "Intel Graphics [0xe211]",
        "server": "b60",
        "driver_version": "1.13.35563+7",
        "compute_units": 160,
        "memory_mb": 23256,
        "log_files": [os.path.basename(p) for p in log_file_paths],
        "summary": {
            "total_kernels_benchmarked": len(kernel_groups),
            "total_test_cases": len(all_results),
            "total_not_supported": len(all_not_supported),
            "kernels_with_unsupported": len(not_supported_groups)
        },
        "results": kernel_results,
        "not_supported": not_supported_list
    }
    
    output_path = os.path.join(script_dir, "baseline_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nWrote baseline_results.json with:")
    print(f"  {len(kernel_groups)} unique kernels")
    print(f"  {len(all_results)} total test cases")
    print(f"  {len(all_not_supported)} not supported cases")
    print(f"  Output: {output_path}")
    
    # Print summary table
    print(f"\n{'Kernel':<25} {'Cases':>6} {'Min (us)':>12} {'Max (us)':>12}")
    print("-" * 60)
    for kr in kernel_results:
        print(f"{kr['kernel']:<25} {kr['num_test_cases']:>6} {kr['min_time_us']:>12.2f} {kr['max_time_us']:>12.2f}")


if __name__ == "__main__":
    main()
