#!/usr/bin/env python3
"""
Hybrid mesh experiment — compare FEM accuracy vs runtime for three mesh strategies.

Cases
-----
  0  baseline   : lc_far=0.030  (current default — reference point)
  1  graded_far : lc_far=0.080  (coarser far vacuum, same near-electrode refinement)
  2  hybrid_box : lc_far=0.100 + Gmsh Box field (fine mesh forced inside inner-trap
                  box; coarse outside) — Option-A single conformal mesh, no stitching.

Acceptance thresholds (relative to baseline)
--------------------------------------------
  r0_z shift        < 1 µm
  strong freq shift < 2 %
  depth_z shift     < 5 %
  no tag corruption, no solver instability

Usage
-----
  cd /path/to/trap_sim
  python experiments/run_hybrid_mesh_experiment.py [options]

  # Quick run (fast FEM, skip transport + y-depth):
  python experiments/run_hybrid_mesh_experiment.py --fast-fem

  # Skip mesh generation (meshes already built):
  python experiments/run_hybrid_mesh_experiment.py --skip-meshgen

  # Run only one case (0=baseline, 1=graded, 2=hybrid):
  python experiments/run_hybrid_mesh_experiment.py --only-case 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths (relative to repo root, where this script is invoked from)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
ASSEMBLE  = REPO_ROOT / "geometry" / "assemble_mesh.py"
SOLVER    = REPO_ROOT / "src"      / "run_sweep_metrics.py"

# CAD files — base single-junction geometry
RF_STEP     = REPO_ROOT / "cad" / "base" / "rf.step"
DC_STEP     = REPO_ROOT / "cad" / "base" / "dc.step"
GROUND_STEP = REPO_ROOT / "cad" / "base" / "ground.step"

# ---------------------------------------------------------------------------
# Physics parameters — identical across all cases (paper Yb-171 settings)
# ---------------------------------------------------------------------------
PHYSICS = {
    "rf_tags"    : [1],
    "ground_tags": [2, 3],
    "outer_tags" : [4],
    "degree"     : 2,
    "rf_freq"    : 44.3e6,
    "mass_amu"   : 171.0,
    "charge_e"   : 1.0,
    "vrf"        : 190.0,
    "coord_unit" : 1e-3,
    "r0_z_min"   : 0.03,
    "r0_z_max"   : 0.12,
}

# Reference results (paper_171Yb_sweep.json, 2-junction mesh).
# Single-junction results should be very close in the linear arm.
REFERENCE = {
    "r0_z_m"           : 8.268e-05,
    "strong_freq_min_hz": 2_225_095.9,
    "strong_freq_max_hz": 2_438_098.1,
    "radial_depth_core_eV": 0.6068,
}

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
CASES = [
    {
        "name"            : "baseline",
        "lc_electrode"    : 0.003,
        "lc_center"       : 0.005,
        "lc_far"          : 0.030,
        # No inner-box field
        "box_lc_in"       : None,
    },
    {
        "name"            : "graded_far",
        "lc_electrode"    : 0.003,
        "lc_center"       : 0.005,
        "lc_far"          : 0.080,   # 2.7× coarser far field
        "box_lc_in"       : None,
    },
    {
        "name"            : "hybrid_box",
        "lc_electrode"    : 0.003,
        "lc_center"       : 0.005,
        "lc_far"          : 0.100,   # very coarse far vacuum
        # Box field: keep fine mesh inside inner-trap box
        "box_lc_in"       : 0.005,
        "box_margin_xy"   : 0.150,   # 150 µm xy margin beyond electrode bbox
        "box_z_top_offset": 0.250,   # box top = electrode_zmax + 250 µm (covers ion at ~82 µm)
        "box_thickness"   : 0.020,   # 20 µm smooth transition wall
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str, log_path: Path) -> tuple[int, float]:
    """Run a subprocess, tee output to log_path, return (returncode, elapsed_s)."""
    print(f"\n{'='*60}")
    print(f"[run] {label}")
    print(f"[cmd] {' '.join(str(c) for c in cmd)}")
    print(f"[log] {log_path}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as fh:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()

    elapsed = time.perf_counter() - t0
    status  = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"[{label}] {status}  elapsed={elapsed:.1f}s")
    return proc.returncode, elapsed


def build_meshgen_cmd(case: dict, mesh_path: Path) -> list[str]:
    cmd = [
        sys.executable, str(ASSEMBLE),
        "--rf",     str(RF_STEP),
        "--dc",     str(DC_STEP),
        "--ground", str(GROUND_STEP),
        "--out",    str(mesh_path),
        "--nopopup",
        f"--lc-electrode={case['lc_electrode']}",
        f"--lc-center={case['lc_center']}",
        f"--lc-far={case['lc_far']}",
    ]
    if case.get("box_lc_in") is not None:
        cmd += [
            f"--box-lc-in={case['box_lc_in']}",
            f"--box-margin-xy={case.get('box_margin_xy', 0.150)}",
            f"--box-z-top-offset={case.get('box_z_top_offset', 0.250)}",
            f"--box-thickness={case.get('box_thickness', 0.020)}",
        ]
    return cmd


def build_solver_cmd(mesh_path: Path, outdir: Path, prefix: str,
                     fast_fem: bool) -> list[str]:
    p = PHYSICS
    cmd = [
        sys.executable, str(SOLVER),
        "--mesh",    str(mesh_path),
        "--outdir",  str(outdir),
        "--prefix",  prefix,
        "--degree",  str(p["degree"]),
        "--rf-freq", str(p["rf_freq"]),
        "--mass-amu", str(p["mass_amu"]),
        "--charge-e", str(p["charge_e"]),
        "--vrf",      str(p["vrf"]),
        "--coord-unit", str(p["coord_unit"]),
        "--r0-z-min",   str(p["r0_z_min"]),
        "--r0-z-max",   str(p["r0_z_max"]),
    ]
    for t in p["rf_tags"]:
        cmd += ["--rf-tags", str(t)]
    for t in p["ground_tags"]:
        cmd += ["--ground-tags", str(t)]
    for t in p["outer_tags"]:
        cmd += ["--outer-tags", str(t)]
    if fast_fem:
        cmd += ["--fast-metrics"]
    return cmd


def count_msh_tets(msh_path: Path) -> Optional[int]:
    """Count tetrahedra in a Gmsh .msh file without importing gmsh."""
    if not msh_path.exists():
        return None
    try:
        import meshio
        m = meshio.read(str(msh_path))
        for blk in m.cells:
            if blk.type == "tetra":
                return len(blk.data)
    except Exception:
        pass
    return None


def load_result(outdir: Path, prefix: str) -> Optional[dict]:
    """Find and load the solver JSON output."""
    for pat in [f"{prefix}_*_sweep.json", f"{prefix}_sweep.json", "*.json"]:
        matches = sorted(outdir.glob(pat))
        if matches:
            try:
                return json.loads(matches[0].read_text())
            except Exception:
                pass
    return None


def pct_diff(val: float, ref: float) -> str:
    if ref == 0:
        return "n/a"
    return f"{100*(val-ref)/ref:+.2f}%"


def abs_diff_um(val_m: float, ref_m: float) -> str:
    return f"{(val_m - ref_m)*1e6:+.2f} µm"


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path,
                    default=REPO_ROOT / "experiments" / "hybrid_mesh_results",
                    help="Root output directory (default: experiments/hybrid_mesh_results)")
    ap.add_argument("--fast-fem", action="store_true",
                    help="Pass --fast-metrics to solver (12 rays, skip transport/y-depth). "
                         "Saves ~50%% solver time; use for quick feasibility check.")
    ap.add_argument("--skip-meshgen", action="store_true",
                    help="Skip mesh generation; reuse existing .msh files in outdir.")
    ap.add_argument("--only-case", type=int, default=None, choices=[0, 1, 2],
                    help="Run only one case index (0=baseline, 1=graded, 2=hybrid).")
    args = ap.parse_args()

    outdir   = args.outdir.resolve()
    fast_fem = args.fast_fem

    cases_to_run = CASES if args.only_case is None else [CASES[args.only_case]]

    print(f"\n{'#'*70}")
    print("# Hybrid mesh experiment")
    print(f"# outdir : {outdir}")
    print(f"# fast   : {fast_fem}")
    print(f"# cases  : {[c['name'] for c in cases_to_run]}")
    print(f"{'#'*70}\n")

    timings: dict[str, dict[str, float]] = {}
    results: dict[str, dict]             = {}
    cell_counts: dict[str, Optional[int]] = {}

    # ── Part 1: mesh generation ────────────────────────────────────────────
    for case in cases_to_run:
        name     = case["name"]
        case_dir = outdir / name
        mesh_path = case_dir / "mesh.msh"

        timings.setdefault(name, {})

        if args.skip_meshgen and mesh_path.exists():
            print(f"[{name}] Skipping meshgen — using existing {mesh_path}")
        else:
            cmd = build_meshgen_cmd(case, mesh_path)
            rc, elapsed = _run(cmd, f"meshgen/{name}",
                               case_dir / "meshgen.log")
            timings[name]["meshgen_s"] = elapsed
            if rc != 0:
                print(f"[ERROR] meshgen failed for {name} — aborting this case.")
                continue

        n_tet = count_msh_tets(mesh_path)
        cell_counts[name] = n_tet
        size_mb = mesh_path.stat().st_size / 1e6 if mesh_path.exists() else 0
        print(f"[{name}] mesh: {n_tet} tets, {size_mb:.1f} MB")

    # ── Part 2: FEM solve ─────────────────────────────────────────────────
    for case in cases_to_run:
        name      = case["name"]
        case_dir  = outdir / name
        mesh_path = case_dir / "mesh.msh"

        if not mesh_path.exists():
            print(f"[{name}] No mesh found — skipping FEM.")
            continue

        prefix = f"hybrid_exp_{name}"
        cmd    = build_solver_cmd(mesh_path, case_dir, prefix, fast_fem)
        rc, elapsed = _run(cmd, f"fem/{name}",
                           case_dir / "fem.log")
        timings[name]["fem_s"] = elapsed

        if rc != 0:
            print(f"[ERROR] FEM failed for {name}.")
            continue

        result = load_result(case_dir, prefix)
        if result:
            results[name] = result
        else:
            print(f"[WARN] Could not find JSON result for {name}.")

    # ── Part 3: Summary report ─────────────────────────────────────────────
    if not results:
        print("\n[WARN] No FEM results collected. Check logs above.")
        return

    ref_name = "baseline"
    ref      = results.get(ref_name, {})

    print(f"\n{'#'*70}")
    print("# EXPERIMENT RESULTS")
    print(f"{'#'*70}\n")

    # Header
    col_w = 18
    names = [c["name"] for c in cases_to_run if c["name"] in results]
    hdr   = f"{'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in names)
    sep   = "-" * len(hdr)
    print(hdr)
    print(sep)

    def row(label: str, vals: list[str]) -> None:
        print(f"{label:<28}" + "".join(f"{v:>{col_w}}" for v in vals))

    def get(name: str, key: str) -> Optional[float]:
        return results.get(name, {}).get(key)

    # Cell counts
    row("tets (×1000)",
        [f"{(cell_counts.get(n) or 0)/1e3:.1f}" for n in names])

    # Mesh + FEM timing
    row("meshgen (s)",
        [f"{timings.get(n,{}).get('meshgen_s',float('nan')):.0f}" for n in names])
    row("fem (s)",
        [f"{timings.get(n,{}).get('fem_s',float('nan')):.0f}" for n in names])
    row("total (s)",
        [f"{timings.get(n,{}).get('meshgen_s',0)+timings.get(n,{}).get('fem_s',0):.0f}"
         for n in names])
    print(sep)

    # r0_z
    vals_r0z = [get(n, "r0_z_m") for n in names]
    row("r0_z (µm)", [f"{(v or 0)*1e6:.2f}" for v in vals_r0z])
    if ref.get("r0_z_m") and len(names) > 1:
        diffs = []
        for v in vals_r0z:
            if v is None:
                diffs.append("n/a")
            else:
                diffs.append(abs_diff_um(v, ref["r0_z_m"]))
        row("  Δr0_z vs baseline", diffs)

    # Frequencies
    vals_fmin = [get(n, "strong_freq_min_hz") for n in names]
    vals_fmax = [get(n, "strong_freq_max_hz") for n in names]
    row("f_min (MHz)", [f"{(v or 0)/1e6:.4f}" for v in vals_fmin])
    row("f_max (MHz)", [f"{(v or 0)/1e6:.4f}" for v in vals_fmax])
    if ref.get("strong_freq_min_hz") and len(names) > 1:
        row("  Δf_min vs baseline",
            ["—"] + [pct_diff(v or 0, ref["strong_freq_min_hz"])
                     for v in vals_fmin[1:]])
        row("  Δf_max vs baseline",
            ["—"] + [pct_diff(v or 0, ref["strong_freq_max_hz"])
                     for v in vals_fmax[1:]])

    # Depth
    vals_d = [get(n, "radial_depth_core_eV") for n in names]
    row("depth_z (eV)", [f"{v:.4f}" if v else "n/a" for v in vals_d])
    if ref.get("radial_depth_core_eV") and len(names) > 1:
        row("  Δdepth vs baseline",
            ["—"] + [pct_diff(v or 0, ref["radial_depth_core_eV"])
                     for v in vals_d[1:]])

    # Hessian status
    row("hessian_status",
        [results.get(n, {}).get("hessian_status", "n/a") for n in names])
    print(sep)

    # ── Acceptance check ──────────────────────────────────────────────────
    print("\n## Acceptance check (vs baseline)\n")
    THRESH_R0Z_UM   = 1.0   # µm
    THRESH_FREQ_PCT = 2.0   # %
    THRESH_DEPTH_PCT= 5.0   # %

    all_pass = True
    for name in names:
        if name == ref_name:
            continue
        r  = results[name]
        ok_r0z  = True
        ok_freq = True
        ok_dep  = True

        if ref.get("r0_z_m") and r.get("r0_z_m"):
            delta_um = abs((r["r0_z_m"] - ref["r0_z_m"]) * 1e6)
            ok_r0z   = delta_um < THRESH_R0Z_UM
            print(f"  [{name}] r0_z shift: {delta_um:.3f} µm  "
                  f"(threshold {THRESH_R0Z_UM} µm)  → {_pass_fail(ok_r0z)}")

        for fkey, label in [("strong_freq_min_hz", "f_min"),
                             ("strong_freq_max_hz", "f_max")]:
            if ref.get(fkey) and r.get(fkey):
                dp = abs(100 * (r[fkey] - ref[fkey]) / ref[fkey])
                ok = dp < THRESH_FREQ_PCT
                ok_freq &= ok
                print(f"  [{name}] {label} shift: {dp:.3f}%  "
                      f"(threshold {THRESH_FREQ_PCT}%)  → {_pass_fail(ok)}")

        if ref.get("radial_depth_core_eV") and r.get("radial_depth_core_eV"):
            dp    = abs(100 * (r["radial_depth_core_eV"] - ref["radial_depth_core_eV"])
                        / ref["radial_depth_core_eV"])
            ok_dep = dp < THRESH_DEPTH_PCT
            print(f"  [{name}] depth shift:  {dp:.3f}%  "
                  f"(threshold {THRESH_DEPTH_PCT}%)  → {_pass_fail(ok_dep)}")

        h_ok = r.get("hessian_status") in ("valid", "borderline_numeric")
        print(f"  [{name}] hessian_status: {r.get('hessian_status')}  "
              f"→ {_pass_fail(h_ok)}")

        case_pass = ok_r0z and ok_freq and ok_dep and h_ok
        all_pass  = all_pass and case_pass
        verdict   = "PASS — within thresholds" if case_pass else "FAIL — exceeds threshold"
        print(f"  [{name}] Overall: {verdict}\n")

    # ── Recommendation ────────────────────────────────────────────────────
    print("## Recommendation\n")

    hybrid_res = results.get("hybrid_box")
    graded_res = results.get("graded_far")
    base_res   = results.get("baseline")

    if hybrid_res and base_res:
        tet_ratio = (cell_counts.get("hybrid_box") or 1) / max(cell_counts.get("baseline") or 1, 1)
        time_ratio = (
            (timings.get("hybrid_box", {}).get("fem_s", 0) +
             timings.get("hybrid_box", {}).get("meshgen_s", 0)) /
            max(timings.get("baseline", {}).get("fem_s", 0) +
                timings.get("baseline", {}).get("meshgen_s", 0), 1)
        )
        print(f"  hybrid_box vs baseline: {tet_ratio:.2f}× tets, {time_ratio:.2f}× total time")
        if all_pass:
            print("  → ADOPT: hybrid_box mesh meets all accuracy thresholds with fewer cells.")
        else:
            print("  → REJECT hybrid approach.  Examine which metric failed first (above).")
            print("     Consider adopting graded_far only if it passes all thresholds.")
    elif graded_res and base_res:
        print("  Only graded_far result available.")
    else:
        print("  Insufficient data for recommendation.")

    # Save summary JSON
    summary = {
        "cases"       : [c["name"] for c in cases_to_run],
        "cell_counts" : cell_counts,
        "timings_s"   : timings,
        "results"     : results,
        "reference"   : REFERENCE,
        "thresholds"  : {
            "r0_z_um"   : THRESH_R0Z_UM,
            "freq_pct"  : THRESH_FREQ_PCT,
            "depth_pct" : THRESH_DEPTH_PCT,
        },
    }
    summary_path = outdir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] Written to {summary_path}")


if __name__ == "__main__":
    main()
