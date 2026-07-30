#!/usr/bin/env python3
"""Roofline classification for one kernel.

Given FLOPs and bytes moved for a kernel, plus device peaks, classify the kernel as
memory- or compute-bound and report the achieved fraction of the relevant peak.

Usage:
  roofline.py --flops 2.1e9 --bytes 4.2e9 \
              --peak-flops 1.2e13 --peak-bw 4.5e11 \
              [--achieved-gbps 320] [--achieved-gflops 900]

All peaks in base units (FLOP/s, bytes/s). Prints JSON.
"""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", type=float, required=True)
    ap.add_argument("--bytes", type=float, required=True)
    ap.add_argument("--peak-flops", type=float, required=True, help="peak FLOP/s")
    ap.add_argument("--peak-bw", type=float, required=True, help="peak bytes/s")
    ap.add_argument("--achieved-gbps", type=float, default=None)
    ap.add_argument("--achieved-gflops", type=float, default=None)
    a = ap.parse_args()

    if a.bytes <= 0:
        print("bytes must be > 0", file=sys.stderr); return 2
    ai = a.flops / a.bytes
    ridge = a.peak_flops / a.peak_bw
    bound = "memory" if ai < ridge else "compute"

    out = {
        "arithmetic_intensity": ai,
        "ridge_point": ridge,
        "bound": bound,
    }
    # Attainable performance at this AI (min of the two roofs), in FLOP/s.
    attainable = min(a.peak_flops, ai * a.peak_bw)
    out["attainable_gflops"] = attainable / 1e9

    if bound == "memory" and a.achieved_gbps is not None:
        out["fraction_of_peak"] = (a.achieved_gbps * 1e9) / a.peak_bw
        out["metric"] = "bandwidth"
    elif bound == "compute" and a.achieved_gflops is not None:
        out["fraction_of_peak"] = (a.achieved_gflops * 1e9) / a.peak_flops
        out["metric"] = "flops"

    if "fraction_of_peak" in out:
        out["meets_expectation"] = out["fraction_of_peak"] >= 0.60

    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
