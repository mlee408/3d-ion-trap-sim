#!/usr/bin/env python3
"""
Rescore sweep cases from existing run_case.stdout.txt files.
Run from inside a sweep result folder: runs/sweeps/<sweep_name>/

Usage:
    python ../../../scripts/rescore_sweep_cases.py --case-start 0 --case-end 49
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_stdout(path: Path) -> dict:
    """Parse run_case.stdout.txt and extract scoring metrics."""
    result = {
        "depth_z_eV": None,
        "radial_depth_core_eV": None,
        "strong_freq_MHz": None,  # list
        "warning": "",
    }

    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        result["warning"] = f"cannot read file: {e}"
        return result

    # 1. depth_z_eV from paper-comparison line
    m = re.search(
        r"\[sweep \| paper-comparison eV\]\s+depth_z=([\d.eE+\-]+)", text
    )
    if m:
        result["depth_z_eV"] = float(m.group(1))

    # 2. radial_depth_core_eV
    m = re.search(
        r"\[sweep \| local confinement eV\]\s+radial_depth_core=([\d.eE+\-]+)", text
    )
    if m:
        result["radial_depth_core_eV"] = float(m.group(1))

    # 3. strong_freq_MHz from local secular line
    m = re.search(
        r"\[sweep \| local secular MHz\]\s+strong_freq=\[([\d.,\s]+)\]", text
    )
    if m:
        freqs = [float(f.strip()) for f in m.group(1).split(",") if f.strip()]
        if freqs:
            result["strong_freq_MHz"] = freqs

    # Fallback: derive from all_freq_hz if strong_freq missing
    if result["strong_freq_MHz"] is None:
        m = re.search(
            r"\[sweep\]\s+h_used=[\d.eE+\-]+\s+all_freq_hz=\[([^\]]+)\]", text
        )
        if m:
            raw = m.group(1)
            # values may be quoted strings or plain numbers
            tokens = re.findall(r"[\d.eE+\-]+", raw)
            hz_vals = [float(t) for t in tokens if float(t) > 0]
            if len(hz_vals) >= 2:
                hz_vals_sorted = sorted(hz_vals, reverse=True)
                top2_mhz = [v / 1e6 for v in hz_vals_sorted[:2]]
                result["strong_freq_MHz"] = top2_mhz
                result["warning"] += "used all_freq_hz fallback; "

    if result["depth_z_eV"] is None:
        result["warning"] += "depth_z_eV missing; "
    if result["strong_freq_MHz"] is None:
        result["warning"] += "strong_freq_MHz missing; "

    return result


def compute_score(depth_z_eV: float, strong_freq_MHz: list) -> tuple:
    """Return (new_score, min_freq, max_freq, spread_MHz, spread_excess_MHz)."""
    min_f = min(strong_freq_MHz)
    max_f = max(strong_freq_MHz)
    spread = max_f - min_f
    spread_excess = max(0.0, spread - 0.5)
    score = 2.0 * depth_z_eV + 2.0 * min_f - 0.3 * spread_excess
    return score, min_f, max_f, spread, spread_excess


def main():
    parser = argparse.ArgumentParser(description="Rescore sweep cases from stdout files.")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-end", type=int, default=49)
    parser.add_argument("--out", default="rescored_cases.csv")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    sweep_dir = Path.cwd()
    rows = []

    for idx in range(args.case_start, args.case_end + 1):
        case_id = f"case_{idx:04d}"
        stdout_path = sweep_dir / case_id / "run_case.stdout.txt"

        row = {
            "case_id": case_id,
            "success_parse": False,
            "depth_z_eV": "",
            "radial_depth_core_eV": "",
            "min_strong_freq_MHz": "",
            "max_strong_freq_MHz": "",
            "spread_MHz": "",
            "spread_excess_MHz": "",
            "new_score": "",
            "warning": "",
            "stdout_path": str(stdout_path),
        }

        if not stdout_path.exists():
            row["warning"] = "stdout file not found"
            rows.append(row)
            continue

        parsed = parse_stdout(stdout_path)
        row["warning"] = parsed["warning"].strip().rstrip(";")

        if parsed["radial_depth_core_eV"] is not None:
            row["radial_depth_core_eV"] = parsed["radial_depth_core_eV"]

        if parsed["depth_z_eV"] is not None and parsed["strong_freq_MHz"] is not None:
            score, min_f, max_f, spread, spread_excess = compute_score(
                parsed["depth_z_eV"], parsed["strong_freq_MHz"]
            )
            row["success_parse"] = True
            row["depth_z_eV"] = parsed["depth_z_eV"]
            row["min_strong_freq_MHz"] = min_f
            row["max_strong_freq_MHz"] = max_f
            row["spread_MHz"] = spread
            row["spread_excess_MHz"] = spread_excess
            row["new_score"] = score
        else:
            if parsed["depth_z_eV"] is not None:
                row["depth_z_eV"] = parsed["depth_z_eV"]

        rows.append(row)

    # Sort: successful cases by score desc, failures at bottom
    successful = sorted(
        [r for r in rows if r["success_parse"]],
        key=lambda r: r["new_score"],
        reverse=True,
    )
    failed = [r for r in rows if not r["success_parse"]]
    sorted_rows = successful + failed

    # Write CSV
    out_path = sweep_dir / args.out
    fieldnames = [
        "case_id",
        "success_parse",
        "depth_z_eV",
        "radial_depth_core_eV",
        "min_strong_freq_MHz",
        "max_strong_freq_MHz",
        "spread_MHz",
        "spread_excess_MHz",
        "new_score",
        "warning",
        "stdout_path",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Wrote {len(sorted_rows)} rows to {out_path}")
    print(f"  Parsed successfully: {len(successful)}")
    print(f"  Failed / missing:    {len(failed)}")
    print()

    # Print top N
    top_n = min(args.top, len(successful))
    if top_n == 0:
        print("No successfully parsed cases to display.")
        return

    col_w = [10, 10, 12, 26, 22]
    header = (
        f"{'case_id':<{col_w[0]}}  "
        f"{'new_score':>{col_w[1]}}  "
        f"{'depth_z_eV':>{col_w[2]}}  "
        f"{'strong_freq_MHz [min, max]':<{col_w[3]}}  "
        f"{'radial_depth_core_eV':>{col_w[4]}}"
    )
    print(f"Top {top_n} cases by new_score:")
    print(header)
    print("-" * len(header))

    for r in successful[:top_n]:
        freq_str = f"[{r['min_strong_freq_MHz']:.3f}, {r['max_strong_freq_MHz']:.3f}]"
        rdc = f"{r['radial_depth_core_eV']:.4f}" if r["radial_depth_core_eV"] != "" else "n/a"
        print(
            f"{r['case_id']:<{col_w[0]}}  "
            f"{r['new_score']:>{col_w[1]}.4f}  "
            f"{r['depth_z_eV']:>{col_w[2]}.4f}  "
            f"{freq_str:<{col_w[3]}}  "
            f"{rdc:>{col_w[4]}}"
        )


if __name__ == "__main__":
    main()
