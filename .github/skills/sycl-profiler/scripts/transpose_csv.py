#!/usr/bin/env python3
import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(description="Transpose rows and columns of a CSV file.")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output CSV file (default: stdout)")
    parser.add_argument("-s", "--skip", type=int, default=0, help="Number of lines to skip at the beginning")
    parser.add_argument("-d", "--delimiter", default=",", help="CSV delimiter (default: comma)")
    args = parser.parse_args()

    with open(args.input, "r", newline="") as f:
        skipped_lines = [f.readline() for _ in range(args.skip)]
        reader = csv.reader(f, delimiter=args.delimiter)
        rows = list(reader)

    if not rows:
        print("No data to transpose.", file=sys.stderr)
        sys.exit(1)

    max_cols = max(len(r) for r in rows)
    # Pad shorter rows so zip_longest isn't needed
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    transposed = list(zip(*padded))

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        for line in skipped_lines:
            out.write(line)
        writer = csv.writer(out, delimiter=args.delimiter)
        for row in transposed:
            writer.writerow(row)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
