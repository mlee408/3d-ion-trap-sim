#!/usr/bin/env python3
"""
Rescore sweep cases from existing JSON reports and run_case.stdout.txt files.
Run from inside a sweep result folder: runs/sweeps/<sweep_name>/

Usage:
    python ../../../scripts/rescore_sweep_cases.py --case-start 0 --case-end 49

Scoring matches automate.py score_case_metrics():
  +2.0 × depth_z_eV
  +2.0 × min_strong_freq_MHz
  −0.3 × max(0, spread_MHz − 5.0)
  −50.0 × transport_barrier_eV  (when available from JSON report)
  ×0.85 if hessian_status == "borderline_numeric"
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Scoring constants (must match automate.py score_case_metrics) ─────────────
_SPREAD_TARGET_MHZ = 5.0    # free band; excess beyond this is penalised
_SPREAD_COEFF      = 0.3    # pts per MHz of excess spread
_TRANSPORT_COEFF   = 50.0   # pts per eV of transport barrier
_HESSIAN_MULT      = 0.85   # applied when hessian_status == "borderline_numeric"


# ── JSON report helpers ────────────────────────────────────────────────────────

def _find_json_report(case_dir: Path) -> Optional[Path]:
    for pat in ["*_sweep.json", "*_report.json", "report.json"]:
        hits = [p for p in case_dir.glob(pat) if p.name != "params.json"]
        if hits:
            return hits[0]
    jsons = [p for p in case_dir.glob("*.json") if p.name != "params.json"]
    return jsons[0] if len(jsons) == 1 else None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def read_json_metrics(case_dir: Path) -> Dict[str, Any]:
    """Read transport_barrier, physical_min_ok, and hessian_status from the JSON report."""
    out: Dict[str, Any] = {
        "transport_barrier_eV": None,
        "physical_min_ok": None,
        "hessian_status": None,
        "json_found": False,
    }
    report_path = _find_json_report(case_dir)
    if report_path is None:
        return out
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return out

    out["json_found"] = True
    out["transport_barrier_eV"] = _safe_float(report.get("transport_barrier_xscan_eV"))
    pmo = report.get("physical_min_ok")
    if pmo is not None:
        out["physical_min_ok"] = bool(pmo)
    out["hessian_status"] = report.get("hessian_status")
    return out


# ── Stdout parser ─────────────────────────────────────────────────────────────

def parse_stdout(path: Path) -> Dict[str, Any]:
    """Parse run_case.stdout.txt and extract depth and frequency metrics."""
    result: Dict[str, Any] = {
        "depth_z_eV": None,
        "radial_depth_core_eV": None,
        "strong_freq_MHz": None,
        "warning": "",
    }

    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        result["warning"] = f"cannot read file: {e}"
        return result

    m = re.search(
        r"\[sweep \| paper-comparison eV\]\s+depth_z=([\d.eE+\-]+)", text
    )
    if m:
        result["depth_z_eV"] = float(m.group(1))

    m = re.search(
        r"\[sweep \| local confinement eV\]\s+radial_depth_core=([\d.eE+\-]+)", text
    )
    if m:
        result["radial_depth_core_eV"] = float(m.group(1))

    m = re.search(
        r"\[sweep \| local secular MHz\]\s+strong_freq=\[([\d.,\s]+)\]", text
    )
    if m:
        freqs = [float(f.strip()) for f in m.group(1).split(",") if f.strip()]
        if freqs:
            result["strong_freq_MHz"] = freqs

    # Fallback: derive from all_freq_hz
    if result["strong_freq_MHz"] is None:
        m = re.search(
            r"\[sweep\]\s+h_used=[\d.eE+\-]+\s+all_freq_hz=\[([^\]]+)\]", text
        )
        if m:
            tokens = re.findall(r"[\d.eE+\-]+", m.group(1))
            hz_vals = [float(t) for t in tokens if float(t) > 0]
            if len(hz_vals) >= 2:
                top2_mhz = [v / 1e6 for v in sorted(hz_vals, reverse=True)[:2]]
                result["strong_freq_MHz"] = top2_mhz
                result["warning"] += "used all_freq_hz fallback; "

    if result["depth_z_eV"] is None:
        result["warning"] += "depth_z_eV missing; "
    if result["strong_freq_MHz"] is None:
        result["warning"] += "strong_freq_MHz missing; "

    return result


# ── Scorer ────────────────────────────────────────────────────────────────────

def compute_score(
    depth_z_eV: float,
    strong_freq_MHz: List[float],
    transport_barrier_eV: Optional[float] = None,
    hessian_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Return score and breakdown dict matching automate.py score_case_metrics."""
    min_f  = min(strong_freq_MHz)
    max_f  = max(strong_freq_MHz)
    spread = max_f - min_f

    term_depth        = 2.0 * depth_z_eV
    term_freq         = 2.0 * min_f
    spread_excess     = max(0.0, spread - _SPREAD_TARGET_MHZ)
    term_spread_pen   = _SPREAD_COEFF * spread_excess
    term_transport_pen = _TRANSPORT_COEFF * transport_barrier_eV if transport_barrier_eV is not None else 0.0

    score = term_depth + term_freq - term_spread_pen - term_transport_pen

    if hessian_status == "borderline_numeric":
        score *= _HESSIAN_MULT

    return {
        "score":               score,
        "min_freq_MHz":        min_f,
        "max_freq_MHz":        max_f,
        "spread_MHz":          spread,
        "spread_excess_MHz":   spread_excess,
        "term_depth":          round(term_depth, 4),
        "term_freq":           round(term_freq, 4),
        "term_spread_pen":     round(-term_spread_pen, 4),
        "term_transport_pen":  round(-term_transport_pen, 4),
        "transport_barrier_meV": round(transport_barrier_eV * 1e3, 3) if transport_barrier_eV is not None else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore sweep cases from JSON reports and stdout files."
    )
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-end",   type=int, default=49)
    parser.add_argument("--out",  default="rescored_cases.csv")
    parser.add_argument("--top",  type=int, default=10)
    args = parser.parse_args()

    sweep_dir = Path.cwd()
    rows: List[Dict[str, Any]] = []

    for idx in range(args.case_start, args.case_end + 1):
        case_id   = f"case_{idx:04d}"
        case_dir  = sweep_dir / case_id
        stdout_path = case_dir / "run_case.stdout.txt"

        row: Dict[str, Any] = {
            "case_id":              case_id,
            "success_parse":        False,
            "rejection_reason":     "",
            "depth_z_eV":           "",
            "radial_depth_core_eV": "",
            "min_strong_freq_MHz":  "",
            "max_strong_freq_MHz":  "",
            "spread_MHz":           "",
            "spread_excess_MHz":    "",
            "transport_barrier_meV": "",
            "physical_min_ok":      "",
            "hessian_status":       "",
            "score":                "",
            "term_depth":           "",
            "term_freq":            "",
            "term_spread_pen":      "",
            "term_transport_pen":   "",
            "warning":              "",
            "stdout_path":          str(stdout_path),
        }

        if not case_dir.is_dir():
            row["warning"] = "case directory not found"
            rows.append(row)
            continue

        # Read JSON report for validity / transport fields
        jm = read_json_metrics(case_dir)
        row["physical_min_ok"] = jm["physical_min_ok"] if jm["physical_min_ok"] is not None else ""
        row["hessian_status"]  = jm["hessian_status"]  or ""

        # Hard reject: invalid physical minimum
        if jm["physical_min_ok"] is False:
            row["rejection_reason"] = "invalid_physical"
            rows.append(row)
            continue

        if not stdout_path.exists():
            row["warning"] = "stdout file not found"
            rows.append(row)
            continue

        parsed = parse_stdout(stdout_path)
        row["warning"] = parsed["warning"].strip().rstrip(";")

        if parsed["radial_depth_core_eV"] is not None:
            row["radial_depth_core_eV"] = parsed["radial_depth_core_eV"]

        if parsed["depth_z_eV"] is None or parsed["strong_freq_MHz"] is None:
            rows.append(row)
            continue

        row["depth_z_eV"] = parsed["depth_z_eV"]

        bd = compute_score(
            depth_z_eV          = parsed["depth_z_eV"],
            strong_freq_MHz     = parsed["strong_freq_MHz"],
            transport_barrier_eV = jm["transport_barrier_eV"],
            hessian_status      = jm["hessian_status"],
        )

        row["success_parse"]       = True
        row["min_strong_freq_MHz"] = bd["min_freq_MHz"]
        row["max_strong_freq_MHz"] = bd["max_freq_MHz"]
        row["spread_MHz"]          = bd["spread_MHz"]
        row["spread_excess_MHz"]   = bd["spread_excess_MHz"]
        row["transport_barrier_meV"] = bd["transport_barrier_meV"] if bd["transport_barrier_meV"] is not None else ""
        row["score"]               = bd["score"]
        row["term_depth"]          = bd["term_depth"]
        row["term_freq"]           = bd["term_freq"]
        row["term_spread_pen"]     = bd["term_spread_pen"]
        row["term_transport_pen"]  = bd["term_transport_pen"]

        rows.append(row)

    # Sort: successful by score desc, failures at bottom
    successful = sorted(
        [r for r in rows if r["success_parse"]],
        key=lambda r: r["score"],
        reverse=True,
    )
    failed = [r for r in rows if not r["success_parse"]]
    sorted_rows = successful + failed

    # Write CSV
    fieldnames = [
        "case_id", "success_parse", "rejection_reason",
        "score", "term_depth", "term_freq", "term_spread_pen", "term_transport_pen",
        "depth_z_eV", "radial_depth_core_eV",
        "min_strong_freq_MHz", "max_strong_freq_MHz",
        "spread_MHz", "spread_excess_MHz",
        "transport_barrier_meV", "physical_min_ok", "hessian_status",
        "warning", "stdout_path",
    ]
    out_path = sweep_dir / args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Wrote {len(sorted_rows)} rows to {out_path}")
    print(f"  Scored successfully : {len(successful)}")
    print(f"  Failed / missing    : {len(failed)}")
    print()

    top_n = min(args.top, len(successful))
    if top_n == 0:
        print("No successfully scored cases to display.")
        return

    # Table header
    has_barrier = any(r["transport_barrier_meV"] != "" for r in successful[:top_n])
    col_w = [10, 9, 10, 26, 14, 14]
    header = (
        f"{'case_id':<{col_w[0]}}  "
        f"{'score':>{col_w[1]}}  "
        f"{'depth_z_eV':>{col_w[2]}}  "
        f"{'freq [min, max] MHz':<{col_w[3]}}  "
        f"{'spread_MHz':>{col_w[4]}}"
    )
    if has_barrier:
        header += f"  {'barrier_meV':>{col_w[5]}}"

    print(f"Top {top_n} cases by score:")
    print(header)
    print("-" * len(header))

    for r in successful[:top_n]:
        freq_str = f"[{r['min_strong_freq_MHz']:.3f}, {r['max_strong_freq_MHz']:.3f}]"
        line = (
            f"{r['case_id']:<{col_w[0]}}  "
            f"{r['score']:>{col_w[1]}.4f}  "
            f"{r['depth_z_eV']:>{col_w[2]}.4f}  "
            f"{freq_str:<{col_w[3]}}  "
            f"{r['spread_MHz']:>{col_w[4]}.3f}"
        )
        if has_barrier:
            bval = f"{r['transport_barrier_meV']:.3f}" if r["transport_barrier_meV"] != "" else "n/a"
            line += f"  {bval:>{col_w[5]}}"
        print(line)


if __name__ == "__main__":
    main()
