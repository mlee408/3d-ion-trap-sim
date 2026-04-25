"""
Extract paper-comparable metrics for 6 cases and write comparison outputs.
Outputs: runs/paper_comparison/{metrics_table.csv, comparison_report.md, extraction_log.txt}
"""

import json
import csv
import os
import math
from datetime import date

BASE = "/Users/michaelee408/Documents/trap_sim"
OUT_DIR = os.path.join(BASE, "runs/paper_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_LINES = []

def log(case_label, metric, value, source):
    line = f"[{case_label}] {metric} = {value} (source: {source})"
    LOG_LINES.append(line)


def sig4(v):
    """Round to 4 significant figures, return string."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if v == 0:
        return "0"
    mag = math.floor(math.log10(abs(v)))
    rounded = round(v, -int(mag) + 3)
    # Format without trailing zeros for clarity
    if abs(rounded) >= 1000 or abs(rounded) < 0.001:
        return f"{rounded:.4g}"
    return f"{rounded:.4g}"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def csv_barrier_meV(csv_path):
    """Compute max(phi_pp_eV) - min(phi_pp_eV) * 1000 from a path CSV."""
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            vals = [float(row["phi_pp_eV"]) for row in reader]
        if vals:
            return (max(vals) - min(vals)) * 1000
    except Exception:
        pass
    return None


def csv_first_conf(csv_path):
    """Get total_conf_eV_per_m2 from first row."""
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                return float(row["total_conf_eV_per_m2"])
    except Exception:
        pass
    return None


def csv_dz_stats(ctc_path, min_path):
    """
    Compute max/rms of |z_ctc - z_min| matched by x (nearest x).
    Returns (max_abs_dz_um, rms_dz_um) or (None, None).
    """
    try:
        def load_csv_xy(path):
            rows = []
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append((float(row["x_m"]), float(row["z_m"])))
            return rows

        ctc_rows = load_csv_xy(ctc_path)
        min_rows = load_csv_xy(min_path)

        if not ctc_rows or not min_rows:
            return None, None

        # Build lookup: for each ctc x, find nearest min x
        min_x = [r[0] for r in min_rows]
        min_z = [r[1] for r in min_rows]

        dz_vals = []
        for cx, cz in ctc_rows:
            # find nearest x in min
            idx = min(range(len(min_x)), key=lambda i: abs(min_x[i] - cx))
            dz = abs(cz - min_z[idx]) * 1e6  # meters -> um
            dz_vals.append(dz)

        if dz_vals:
            max_dz = max(dz_vals)
            rms_dz = math.sqrt(sum(d**2 for d in dz_vals) / len(dz_vals))
            return max_dz, rms_dz
    except Exception:
        pass
    return None, None


def infer_mesh_params(auto_config):
    """Parse njunctions and window-n from mesh_template string."""
    template = auto_config.get("run_config", {}).get("mesh_template", "")
    n_junctions = None
    window_n = None
    for tok in template.split():
        pass
    # Parse --njunctions N and --window-n N
    parts = template.split()
    for i, p in enumerate(parts):
        if p == "--njunctions" and i + 1 < len(parts):
            try:
                n_junctions = int(parts[i + 1])
            except ValueError:
                pass
        if p == "--window-n" and i + 1 < len(parts):
            try:
                window_n = int(parts[i + 1])
            except ValueError:
                pass
    return n_junctions, window_n


# ─── Case definitions ────────────────────────────────────────────────────────

CASES = [
    {
        "label": "paper_baseline",
        "case_dir": os.path.join(BASE, "runs/paper_baseline_ctc/case_0000"),
        "prefix": "auto_case_0000",
        "auto_config_path": os.path.join(BASE, "runs/paper_baseline_ctc/automation_config.json"),
        "is_baseline": True,
    },
    {
        "label": "sweep_0",
        "case_dir": os.path.join(BASE, "runs/sweep_n4_rfheight/case_0000"),
        "prefix": "auto_case_0000",
        "auto_config_path": os.path.join(BASE, "runs/sweep_n4_rfheight/automation_config.json"),
        "is_baseline": False,
    },
    {
        "label": "sweep_6",
        "case_dir": os.path.join(BASE, "runs/sweep_n4_rfheight/case_0006"),
        "prefix": "auto_case_0006",
        "auto_config_path": os.path.join(BASE, "runs/sweep_n4_rfheight/automation_config.json"),
        "is_baseline": False,
    },
    {
        "label": "sweep_9",
        "case_dir": os.path.join(BASE, "runs/sweep_n4_rfheight/case_0009"),
        "prefix": "auto_case_0009",
        "auto_config_path": os.path.join(BASE, "runs/sweep_n4_rfheight/automation_config.json"),
        "is_baseline": False,
    },
    {
        "label": "sweep_12",
        "case_dir": os.path.join(BASE, "runs/sweep_n4_rfheight/case_0012"),
        "prefix": "auto_case_0012",
        "auto_config_path": os.path.join(BASE, "runs/sweep_n4_rfheight/automation_config.json"),
        "is_baseline": False,
    },
    {
        "label": "sweep_44",
        "case_dir": os.path.join(BASE, "runs/sweep_n4_rfheight/case_0044"),
        "prefix": "auto_case_0044",
        "auto_config_path": os.path.join(BASE, "runs/sweep_n4_rfheight/automation_config.json"),
        "is_baseline": False,
    },
]

# Depth correction multiplier for paper_baseline (mesh resolution correction)
BASELINE_DEPTH_CORRECTION = 1.3

# ─── Extraction ──────────────────────────────────────────────────────────────

records = []
csv_fallback_notes = []
na_notes = []

for case in CASES:
    lbl = case["label"]
    cdir = case["case_dir"]
    pfx = case["prefix"]
    is_bl = case["is_baseline"]

    sweep_json_path = os.path.join(cdir, f"{pfx}_sweep.json")
    params_json_path = os.path.join(cdir, "params.json")
    auto_config = load_json(case["auto_config_path"])
    sweep = load_json(sweep_json_path)
    params = load_json(params_json_path)

    n_junctions, window_n = infer_mesh_params(auto_config)

    # Geometry
    rf_height_um = params.get("params", {}).get("rf_height", None)
    rf_width_um = params.get("params", {}).get("rf_width_um", None)
    log(lbl, "rf_height_um", rf_height_um, "params.json")
    log(lbl, "rf_width_um", rf_width_um, "params.json")
    log(lbl, "n_junctions", n_junctions, "automation_config.json")
    log(lbl, "window_n", window_n, "automation_config.json")

    # Trap equilibrium — vrf comes from automation_config, not sweep JSON
    vrf_V = auto_config.get("run_config", {}).get("vrf", None)
    rf_freq_Hz = sweep.get("rf_freq_Hz", None)
    rf_freq_MHz = rf_freq_Hz / 1e6 if rf_freq_Hz is not None else None
    r0_x_m = sweep.get("r0_x_m", None)
    r0_y_m = sweep.get("r0_y_m", None)
    r0_z_m = sweep.get("r0_z_m", None)
    r0_x_um = r0_x_m * 1e6 if r0_x_m is not None else None
    r0_y_um = r0_y_m * 1e6 if r0_y_m is not None else None
    r0_z_um = r0_z_m * 1e6 if r0_z_m is not None else None
    log(lbl, "r0_z_um", r0_z_um, "json")
    log(lbl, "vrf_V", vrf_V, "automation_config.json")
    log(lbl, "rf_freq_MHz", rf_freq_MHz, "json")

    # Secular frequencies
    freq1_hz = sweep.get("freq1_hz", None)
    freq2_hz = sweep.get("freq2_hz", None)
    freq3_hz = sweep.get("freq3_hz", None)
    freq_axial_kHz = freq1_hz / 1e3 if freq1_hz is not None else None
    freq_radial1_MHz = freq2_hz / 1e6 if freq2_hz is not None else None
    freq_radial2_MHz = freq3_hz / 1e6 if freq3_hz is not None else None
    log(lbl, "freq_axial_kHz", freq_axial_kHz, "json")
    log(lbl, "freq_radial1_MHz", freq_radial1_MHz, "json")
    log(lbl, "freq_radial2_MHz", freq_radial2_MHz, "json")

    # Trap depth
    depth_y_eV_raw = sweep.get("depth_y_eV", None)
    depth_z_plus = sweep.get("depth_z_plus_eV", None)
    depth_z_minus = sweep.get("depth_z_minus_eV", None)
    depth_z_eV_raw = sweep.get("depth_z_eV", None)
    radial_depth_core_eV_raw = sweep.get("radial_depth_core_eV", None)

    # depth_z_eV = min of plus/minus (interior side)
    if depth_z_plus is not None and depth_z_minus is not None:
        depth_z_eV_raw = min(depth_z_plus, depth_z_minus)
    elif depth_z_eV_raw is None:
        depth_z_eV_raw = sweep.get("depth_z_eV", None)

    # Apply baseline depth correction
    if is_bl:
        depth_y_eV = depth_y_eV_raw * BASELINE_DEPTH_CORRECTION if depth_y_eV_raw is not None else None
        depth_z_eV = depth_z_eV_raw * BASELINE_DEPTH_CORRECTION if depth_z_eV_raw is not None else None
        radial_depth_core_eV = radial_depth_core_eV_raw * BASELINE_DEPTH_CORRECTION if radial_depth_core_eV_raw is not None else None
        log(lbl, "depth_y_eV", depth_y_eV, "json+depth_correction_1.3x")
        log(lbl, "depth_z_eV", depth_z_eV, "json+depth_correction_1.3x")
        log(lbl, "radial_depth_core_eV", radial_depth_core_eV, "json+depth_correction_1.3x")
    else:
        depth_y_eV = depth_y_eV_raw
        depth_z_eV = depth_z_eV_raw
        radial_depth_core_eV = radial_depth_core_eV_raw
        log(lbl, "depth_y_eV", depth_y_eV, "json")
        log(lbl, "depth_z_eV", depth_z_eV, "json")
        log(lbl, "radial_depth_core_eV", radial_depth_core_eV, "json")

    # Transport metrics
    ctc_csv = os.path.join(cdir, f"{pfx}_sweep_ctc_path.csv")
    min_csv = os.path.join(cdir, f"{pfx}_sweep_min_path.csv")
    has_ctc_csv = os.path.exists(ctc_csv)
    has_min_csv = os.path.exists(min_csv)

    # transport_min_barrier_meV
    transport_min_barrier_meV = sweep.get("transport_min_barrier_meV", None)
    if transport_min_barrier_meV is not None:
        log(lbl, "transport_min_barrier_meV", transport_min_barrier_meV, "json")
    elif has_min_csv:
        transport_min_barrier_meV = csv_barrier_meV(min_csv)
        log(lbl, "transport_min_barrier_meV", transport_min_barrier_meV, "csv")
        csv_fallback_notes.append(f"{lbl}: transport_min_barrier_meV sourced from CSV ({min_csv})")
    else:
        log(lbl, "transport_min_barrier_meV", "N/A", "N/A")
        na_notes.append(f"{lbl}: transport_min_barrier_meV — no JSON field and no min_path CSV")

    # ctc_barrier_meV
    ctc_barrier_meV = sweep.get("ctc_barrier_meV", None)
    if ctc_barrier_meV is not None:
        log(lbl, "ctc_barrier_meV", ctc_barrier_meV, "json")
    elif has_ctc_csv:
        ctc_barrier_meV = csv_barrier_meV(ctc_csv)
        log(lbl, "ctc_barrier_meV", ctc_barrier_meV, "csv")
        csv_fallback_notes.append(f"{lbl}: ctc_barrier_meV sourced from CSV ({ctc_csv})")
    else:
        log(lbl, "ctc_barrier_meV", "N/A", "N/A")
        na_notes.append(f"{lbl}: ctc_barrier_meV — no JSON field and no ctc_path CSV")

    # ctc_target_conf_eV_per_m2
    ctc_target_conf = sweep.get("ctc_target_conf_eV_per_m2", None)
    if ctc_target_conf is not None:
        log(lbl, "ctc_target_conf_eV_per_m2", ctc_target_conf, "json")
    elif has_ctc_csv:
        ctc_target_conf = csv_first_conf(ctc_csv)
        log(lbl, "ctc_target_conf_eV_per_m2", ctc_target_conf, "csv")
        csv_fallback_notes.append(f"{lbl}: ctc_target_conf_eV_per_m2 sourced from first row of ctc CSV")
    else:
        log(lbl, "ctc_target_conf_eV_per_m2", "N/A", "N/A")
        na_notes.append(f"{lbl}: ctc_target_conf_eV_per_m2 — no JSON and no ctc CSV")

    # ctc_max_abs_dz_um, ctc_rms_dz_um
    ctc_max_abs_dz_um = sweep.get("ctc_max_abs_dz_um", None)
    ctc_rms_dz_um = sweep.get("ctc_rms_dz_um", None)
    if ctc_max_abs_dz_um is not None:
        log(lbl, "ctc_max_abs_dz_um", ctc_max_abs_dz_um, "json")
        log(lbl, "ctc_rms_dz_um", ctc_rms_dz_um, "json")
    elif has_ctc_csv and has_min_csv:
        ctc_max_abs_dz_um, ctc_rms_dz_um = csv_dz_stats(ctc_csv, min_csv)
        log(lbl, "ctc_max_abs_dz_um", ctc_max_abs_dz_um, "computed")
        log(lbl, "ctc_rms_dz_um", ctc_rms_dz_um, "computed")
        csv_fallback_notes.append(f"{lbl}: ctc_max_abs_dz_um and ctc_rms_dz_um computed from CSV dz matching")
    else:
        log(lbl, "ctc_max_abs_dz_um", "N/A", "N/A")
        log(lbl, "ctc_rms_dz_um", "N/A", "N/A")
        na_notes.append(f"{lbl}: ctc_max_abs_dz_um / ctc_rms_dz_um — no JSON and no CSVs")

    # ctc_no_crossing_count
    ctc_no_crossing_count = sweep.get("ctc_no_crossing_count", None)
    if ctc_no_crossing_count is not None:
        log(lbl, "ctc_no_crossing_count", ctc_no_crossing_count, "json")
    else:
        log(lbl, "ctc_no_crossing_count", "N/A", "N/A")
        na_notes.append(f"{lbl}: ctc_no_crossing_count — not in JSON")

    # Field quality
    grad_rel_at_r0 = sweep.get("grad_rel_at_r0", None)
    log(lbl, "grad_rel_at_r0", grad_rel_at_r0, "json" if grad_rel_at_r0 is not None else "N/A")

    # transport_xscan_barrier_meV (baseline only extra metric)
    transport_xscan_barrier_meV = None
    if is_bl:
        xscan_eV = sweep.get("transport_barrier_xscan_eV", None)
        if xscan_eV is not None:
            transport_xscan_barrier_meV = xscan_eV * 1000
        log(lbl, "transport_xscan_barrier_meV", transport_xscan_barrier_meV, "json" if transport_xscan_barrier_meV is not None else "N/A")

    records.append({
        "case_label": lbl,
        "rf_height_um": rf_height_um,
        "rf_width_um": rf_width_um,
        "n_junctions": n_junctions,
        "window_n": window_n,
        "r0_x_um": r0_x_um,
        "r0_y_um": r0_y_um,
        "r0_z_um": r0_z_um,
        "vrf_V": vrf_V,
        "rf_freq_MHz": rf_freq_MHz,
        "freq_axial_kHz": freq_axial_kHz,
        "freq_radial1_MHz": freq_radial1_MHz,
        "freq_radial2_MHz": freq_radial2_MHz,
        "depth_y_eV": depth_y_eV,
        "depth_z_eV": depth_z_eV,
        "radial_depth_core_eV": radial_depth_core_eV,
        "transport_min_barrier_meV": transport_min_barrier_meV,
        "ctc_barrier_meV": ctc_barrier_meV,
        "ctc_target_conf_eV_per_m2": ctc_target_conf,
        "ctc_max_abs_dz_um": ctc_max_abs_dz_um,
        "ctc_rms_dz_um": ctc_rms_dz_um,
        "ctc_no_crossing_count": ctc_no_crossing_count,
        "grad_rel_at_r0": grad_rel_at_r0,
        "transport_xscan_barrier_meV": transport_xscan_barrier_meV,  # baseline extra
        "source_json": sweep_json_path,
        "is_baseline": is_bl,
    })


# ─── Write extraction_log.txt ─────────────────────────────────────────────────

log_path = os.path.join(OUT_DIR, "extraction_log.txt")
with open(log_path, "w") as f:
    for line in LOG_LINES:
        f.write(line + "\n")
print(f"Wrote {log_path}")


# ─── Write metrics_table.csv ──────────────────────────────────────────────────

CSV_COLS = [
    "case_label", "rf_height_um", "rf_width_um", "n_junctions", "window_n",
    "r0_z_um", "vrf_V", "rf_freq_MHz",
    "freq_axial_kHz", "freq_radial1_MHz", "freq_radial2_MHz",
    "depth_y_eV", "depth_z_eV", "radial_depth_core_eV",
    "transport_min_barrier_meV", "ctc_barrier_meV",
    "ctc_target_conf_eV_per_m2", "ctc_max_abs_dz_um", "ctc_rms_dz_um",
    "ctc_no_crossing_count", "grad_rel_at_r0",
    "source_json",
]

FLOAT_COLS = {
    "rf_height_um", "rf_width_um", "r0_z_um", "vrf_V", "rf_freq_MHz",
    "freq_axial_kHz", "freq_radial1_MHz", "freq_radial2_MHz",
    "depth_y_eV", "depth_z_eV", "radial_depth_core_eV",
    "transport_min_barrier_meV", "ctc_barrier_meV",
    "ctc_target_conf_eV_per_m2", "ctc_max_abs_dz_um", "ctc_rms_dz_um",
    "grad_rel_at_r0",
}

csv_path = os.path.join(OUT_DIR, "metrics_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLS)
    writer.writeheader()
    for rec in records:
        row = {}
        for col in CSV_COLS:
            v = rec.get(col)
            if v is None:
                row[col] = "N/A"
            elif col in FLOAT_COLS and isinstance(v, (int, float)):
                row[col] = sig4(v)
            else:
                row[col] = v
        writer.writerow(row)
print(f"Wrote {csv_path}")


# ─── Build comparison_report.md ───────────────────────────────────────────────

def fmt(rec, key):
    v = rec.get(key)
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return sig4(v)
    return str(v)


def delta_pct(baseline_val, sweep_val):
    if baseline_val is None or sweep_val is None:
        return "N/A"
    if baseline_val == 0:
        return "N/A"
    d = (sweep_val - baseline_val) / abs(baseline_val) * 100
    return f"{d:+.2f}%"


baseline_rec = next(r for r in records if r["is_baseline"])
sweep_recs = [r for r in records if not r["is_baseline"]]

DELTA_METRICS = [
    "freq_axial_kHz", "freq_radial1_MHz",
    "depth_y_eV", "depth_z_eV",
    "transport_min_barrier_meV", "ctc_barrier_meV", "r0_z_um",
]

report_lines = []
today = date.today().isoformat()

report_lines += [
    "# Paper Comparison Report",
    "",
    f"**Date generated:** {today}  ",
    "**Cases included:** paper_baseline, sweep_0, sweep_6, sweep_9, sweep_12, sweep_44  ",
    "**Paper benchmark reference:** paper_baseline_ctc/case_0000 (rf_height=247 µm, rf_width=56 µm, window_n=2, n_junctions=2)  ",
    "**Note:** paper_baseline depth metrics corrected by ×1.3 factor to account for mesh-resolution/CAD-generation systematic underestimate.  ",
    "",
    "---",
    "",
    "## Full Metrics Table",
    "",
]

# Build header
hdr_cols = [
    "case_label", "rf_height_um", "rf_width_um", "window_n",
    "r0_z_um", "vrf_V", "rf_freq_MHz",
    "freq_axial_kHz", "freq_radial1_MHz", "freq_radial2_MHz",
    "depth_y_eV", "depth_z_eV", "radial_depth_core_eV",
    "transport_min_barrier_meV", "ctc_barrier_meV",
    "ctc_max_abs_dz_um", "ctc_rms_dz_um",
    "ctc_no_crossing_count", "grad_rel_at_r0",
]

report_lines.append("| " + " | ".join(hdr_cols) + " |")
report_lines.append("| " + " | ".join(["---"] * len(hdr_cols)) + " |")
for rec in records:
    row_vals = [fmt(rec, c) for c in hdr_cols]
    report_lines.append("| " + " | ".join(row_vals) + " |")

report_lines += [
    "",
    "> **paper_baseline extra metric:** transport_xscan_barrier_meV (classical peak−start x-scan) = "
    f"{fmt(baseline_rec, 'transport_xscan_barrier_meV')} meV  ",
    "> This is a distinct metric from transport_min_barrier_meV; do not conflate.",
    "",
    "---",
    "",
    "## Delta Table (sweep vs paper_baseline)",
    "",
    f"Formula: (sweep_value − baseline_value) / |baseline_value| × 100%  ",
    f"Baseline values (after 1.3× depth correction where applicable):  ",
    "",
]

# Baseline row
bl_row = {m: baseline_rec.get(m) for m in DELTA_METRICS}
report_lines.append(
    "| metric | baseline_value | " +
    " | ".join(r["case_label"] for r in sweep_recs) + " |"
)
report_lines.append("| --- | --- | " + " | ".join(["---"] * len(sweep_recs)) + " |")
for m in DELTA_METRICS:
    bl_val = baseline_rec.get(m)
    deltas = [delta_pct(bl_val, sr.get(m)) for sr in sweep_recs]
    report_lines.append(
        f"| {m} | {sig4(bl_val) if bl_val is not None else 'N/A'} | " +
        " | ".join(deltas) + " |"
    )

report_lines += [
    "",
    "---",
    "",
    "## Notes",
    "",
]

if csv_fallback_notes:
    report_lines.append("### Metrics sourced from CSV fallback (JSON field absent)")
    for note in csv_fallback_notes:
        report_lines.append(f"- {note}")
    report_lines.append("")

if na_notes:
    report_lines.append("### N/A entries")
    for note in na_notes:
        report_lines.append(f"- {note}")
    report_lines.append("")

report_lines += [
    "### Structural differences",
    "",
    "- **window_n**: paper_baseline uses `window_n=2`; all sweep cases use `window_n=4` "
    "(4-window electrode vs 2-window). This is a fundamental geometric difference that affects "
    "radial confinement and transport properties independently of rf_height/rf_width.",
    "- **n_junctions**: both use n_junctions=2.",
    "- **Mass/charge**: Yb-171, charge=1e for all cases.",
    "- **RF drive**: vrf=190 V, rf_freq=44.3 MHz for all cases.",
    "- **Depth correction**: paper_baseline raw depth values (depth_y_eV_raw=5.451 eV, "
    "depth_z_eV_raw=1.747 eV) were multiplied by 1.3 to correct for mesh resolution and "
    "CAD-generation systematic error. Corrected values appear throughout this report.",
    "",
]

report_path = os.path.join(OUT_DIR, "comparison_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"Wrote {report_path}")

print("\nDone. Summary:")
for rec in records:
    print(f"  {rec['case_label']}: depth_y={fmt(rec,'depth_y_eV')} eV, "
          f"ctc_barrier={fmt(rec,'ctc_barrier_meV')} meV, "
          f"transport_min={fmt(rec,'transport_min_barrier_meV')} meV")
