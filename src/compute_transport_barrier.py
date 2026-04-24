#!/usr/bin/env python3
"""compute_transport_barrier.py  (dolfinx 0.10.x)

Compute the pseudopotential transport barrier for completed sweep cases without
re-running the full FEM pipeline.

Two solve modes (selected automatically):
  (a) Checkpoint (fast): load phi_rf from {prefix}_phi_rf_dofs.npy saved by
      run_sweep_metrics.py --save-solution.  Near-instantaneous — no FEM solve.
  (b) Re-solve (slow): reload mesh from the path in the JSON report and
      re-run the Laplace solve.  Takes the same time as the original solve
      (~1–5 min per case depending on mesh size).  Falls back to this when no
      checkpoint file is found.

Transport scan algorithm
------------------------
1. Locate nearest junction center from r0:
      junc_x = round(r0_x / pitch) * pitch   (pitch default 600 µm)
2. Scan from r0_x outward to junc_x + 0.2*|dx|, using n_steps points.
3. At each x step, minimise Ψ(x, y, z) over (y, z) with scipy Nelder-Mead,
   chaining the initial guess from the previous step (bidirectional from r0).
4. Barrier = max(Ψ_scan) − Ψ(r0).

Usage
-----
  # single case
  python src/compute_transport_barrier.py --case-dir runs/sweeps/.../case_0003

  # whole sweep
  python src/compute_transport_barrier.py --sweep-dir runs/sweeps/n3_...

  # force re-scan even if already present
  python src/compute_transport_barrier.py --sweep-dir ... --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
_E_CHARGE = 1.602176634e-19   # C
_AMU      = 1.66053906660e-27  # kg


# ── Helper: find the JSON report inside a case directory ──────────────────────

def _find_report(case_dir: Path, hint_prefix: Optional[str] = None) -> Optional[Path]:
    """Return the path to the JSON report in case_dir, or None."""
    candidates: List[Path] = []
    if hint_prefix:
        candidates += [
            case_dir / f"{hint_prefix}_sweep.json",
            case_dir / f"{hint_prefix}_report.json",
        ]
    candidates += [case_dir / "report.json"]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: sole JSON file that is not params.json
    jsons = [p for p in sorted(case_dir.glob("*.json")) if p.name != "params.json"]
    return jsons[0] if len(jsons) == 1 else None


def _load_report(case_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    p = _find_report(case_dir)
    if p is None:
        return None, None
    try:
        return p, json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return p, None


# ── Helper: load automation_config from sweep parent ─────────────────────────

def _load_automation_config(case_dir: Path) -> Optional[Dict[str, Any]]:
    """Walk up from case_dir to find automation_config.json."""
    for parent in [case_dir.parent, case_dir.parent.parent]:
        cfg_path = parent / "automation_config.json"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


# ── Core: load phi_rf from checkpoint or re-solve ────────────────────────────

def _load_phi_rf_from_checkpoint(
    mesh_path: Path,
    checkpoint_path: Path,
    degree: int,
    comm,
) -> Tuple[Any, Any]:
    """Load mesh and restore phi_rf from a saved DOF-array checkpoint.

    Returns (domain, phi_rf_function).
    """
    from run_case import load_case_mesh
    from dolfinx import fem

    domain, facet_tags, _ = load_case_mesh(mesh_path)
    V = fem.functionspace(domain, ("CG", degree))
    phi_rf = fem.Function(V)
    phi_rf.name = "phi_rf"

    dofs = np.load(str(checkpoint_path))
    if dofs.shape != phi_rf.x.array.shape:
        raise ValueError(
            f"Checkpoint DOF array shape {dofs.shape} does not match "
            f"function space shape {phi_rf.x.array.shape}. "
            "The mesh or degree may have changed since the checkpoint was saved."
        )
    phi_rf.x.array[:] = dofs
    return domain, phi_rf


def _resolve_phi_rf(
    case_dir: Path,
    report: Dict[str, Any],
    *,
    rf_tags: List[int],
    ground_tags: List[int],
    outer_tags: List[int],
    comm,
) -> Tuple[Any, Any, bool]:
    """Return (domain, phi_rf, used_checkpoint).

    Tries checkpoint first; falls back to a fresh Laplace solve.
    """
    from run_case import load_case_mesh, solve_laplace_tagged
    from dolfinx import fem

    degree = int(report.get("degree", 2))
    mesh_str = report.get("mesh", "")
    mesh_path = Path(mesh_str) if mesh_str else case_dir / "mesh.msh"
    if not mesh_path.is_absolute():
        mesh_path = (case_dir / mesh_path).resolve()

    # Check for checkpoint
    prefix = report.get("prefix", "")
    ckpt_rel = report.get("phi_rf_checkpoint")
    ckpt_path: Optional[Path] = None

    if ckpt_rel:
        p = case_dir / ckpt_rel
        if p.exists():
            ckpt_path = p
    if ckpt_path is None and prefix:
        p = case_dir / f"{prefix}_phi_rf_dofs.npy"
        if p.exists():
            ckpt_path = p

    if ckpt_path is not None:
        domain, phi_rf = _load_phi_rf_from_checkpoint(mesh_path, ckpt_path, degree, comm)
        return domain, phi_rf, True

    # Re-solve
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    print(f"[ctb] loading mesh {mesh_path.name} ...", flush=True)
    domain, facet_tags, _ = load_case_mesh(mesh_path)
    if facet_tags is None:
        raise RuntimeError("Mesh has no facet tags — cannot apply boundary conditions.")
    print(f"[ctb] mesh loaded  (degree={degree})", flush=True)

    outer_set = set(outer_tags)
    gnd_tags_bc = [t for t in ground_tags if t not in outer_set]
    bc_map: Dict[int, float] = {t: 1.0 for t in rf_tags}
    bc_map.update({t: 0.0 for t in gnd_tags_bc})

    print(f"[ctb] solving Laplace  rf_tags={rf_tags}  gnd_tags={gnd_tags_bc} ...", flush=True)
    phi_rf = solve_laplace_tagged(
        domain, facet_tags, bc_map,
        degree=degree,
        petsc_prefix="ctb_rf_",
    )
    phi_rf.name = "phi_rf"
    print("[ctb] Laplace solve done", flush=True)

    # Save checkpoint so future runs skip the re-solve entirely
    ckpt_name = f"{prefix}_phi_rf_dofs.npy" if prefix else "phi_rf_dofs.npy"
    ckpt_out = case_dir / ckpt_name
    np.save(str(ckpt_out), phi_rf.x.array)
    report["phi_rf_checkpoint"] = ckpt_name   # picked up when JSON is patched later

    return domain, phi_rf, False


# ── Core: compute Psi from phi_rf ─────────────────────────────────────────────

def _build_psi(phi_rf, report: Dict[str, Any]):
    """Return the RF pseudopotential Function computed from phi_rf."""
    import metrics

    degree = int(report.get("degree", 2))
    rf_freq = float(report.get("rf_freq_Hz", 40e6))
    mass    = float(report.get("mass_amu", 40.0)) * _AMU
    charge  = float(report.get("charge_e", 1.0)) * _E_CHARGE
    omega   = 2.0 * np.pi * rf_freq

    print("[ctb] computing pseudopotential ...", flush=True)
    Psi = metrics.compute_rf_pseudopotential(
        phi_rf, omega_rf=omega, q_C=charge, m_kg=mass, degree=degree,
    )
    Psi.name = "Psi_rf"
    print("[ctb] pseudopotential ready", flush=True)
    return Psi


# ── Transport barrier scan ────────────────────────────────────────────────────

def _scan_transport_barrier(
    Psi,
    report: Dict[str, Any],
    *,
    junction_pitch: float = 600e-6,
    n_steps: int = 60,
    scan_both_axes: bool = False,
    optim_method: str = "coord_descent",   # kept for CLI compat; unused
    comm=None,
) -> Dict[str, Any]:
    """Scan toward the nearest junction center and compute the transport barrier.

    Algorithm mirrors metrics._ctc_scan: walk in x, at each step use multi-scale
    coordinate descent in (y, z) to track the Ψ valley floor.  This is more robust
    than scipy Nelder-Mead because the step size is derived from the mesh cell size
    rather than a fixed heuristic.

    Returns a dict ready to be merged into the JSON report.
    """
    import metrics

    domain     = Psi.function_space.mesh
    coord_unit = float(report.get("coord_unit_m_per_mesh", 1e-3))
    vrf        = float(report.get("vrf_V", 1.0))
    r0_x_m    = float(report["r0_x_m"])
    r0_y_m    = float(report["r0_y_m"])
    r0_z_m    = float(report["r0_z_m"])

    # r0 in mesh units
    r0 = np.array([r0_x_m, r0_y_m, r0_z_m]) / coord_unit

    phys_scale = vrf ** 2       # Psi stored per (V²); multiply to get SI Joules
    to_eV      = phys_scale / _E_CHARGE

    # Step size for coordinate descent in transverse plane — use mesh cell size
    h_mesh = metrics._estimate_cell_h(domain)
    h_yz_base = h_mesh * 1.5    # matches metrics._ctc_scan

    # Psi at r0
    psi0_raw = float(metrics.eval_function_at_points(
        Psi, np.array([r0], dtype=np.float64), comm=comm
    )[0])
    if not np.isfinite(psi0_raw):
        raise ValueError(f"Psi is NaN at r0 (mesh units {r0.tolist()}). r0 may be outside mesh.")
    psi0_raw = max(psi0_raw, 0.0)

    def _eval_psi(pt: np.ndarray) -> float:
        v = float(metrics.eval_function_at_points(
            Psi, np.array([pt], dtype=np.float64), comm=comm
        )[0])
        return v if np.isfinite(v) else 1e30

    def _yz_descent(trial: np.ndarray, cur_v: float) -> Tuple[np.ndarray, float]:
        """Multi-scale coordinate descent in y,z axes (axes 1 and 2)."""
        for h_scale in (2.0, 1.0):
            h = h_yz_base * h_scale
            for _ in range(2):
                for ax in (1, 2):
                    for sign in (+1.0, -1.0):
                        cand = trial.copy(); cand[ax] += sign * h
                        v = _eval_psi(cand)
                        if v < cur_v:
                            cur_v = v; trial = cand
        return trial, cur_v

    # ── Find nearest junction center ─────────────────────────────────────────
    pitch_mesh  = junction_pitch / coord_unit
    junc_x_mesh = round(r0[0] / pitch_mesh) * pitch_mesh

    dx_mesh      = junc_x_mesh - r0[0]
    scan_end     = junc_x_mesh + 0.2 * abs(dx_mesh)
    scan_back    = r0[0] - 0.2 * abs(dx_mesh)

    x_fwd = np.linspace(r0[0], scan_end,  n_steps)
    x_bwd = np.linspace(r0[0], scan_back, max(n_steps // 5, 5))

    # ── Forward scan (r0 → junction + 20%) ───────────────────────────────────
    print(f"[ctb] scanning x: r0={r0[0]:.4f} → junction={junc_x_mesh:.4f} "
          f"(+20%={scan_end:.4f})  n_steps={n_steps}  h_yz={h_yz_base:.4f}", flush=True)

    cur = r0.copy()
    cur_v = psi0_raw
    psi_fwd = np.full(len(x_fwd), np.nan)
    for i, xi in enumerate(x_fwd):
        trial = cur.copy(); trial[0] = xi
        tv = _eval_psi(trial)
        if tv > 1e29:
            break
        trial, tv = _yz_descent(trial, tv)
        cur = trial; cur_v = tv
        psi_fwd[i] = max(tv, 0.0)
        if (i + 1) % 10 == 0:
            print(f"[ctb]   step {i+1}/{len(x_fwd)}  x={xi:.4f}  psi={tv:.3e}", flush=True)

    # ── Backward scan (r0 → r0 - 20%) ────────────────────────────────────────
    cur = r0.copy(); cur_v = psi0_raw
    psi_bwd = np.full(len(x_bwd) - 1, np.nan)
    for i, xi in enumerate(x_bwd[1:]):
        trial = cur.copy(); trial[0] = xi
        tv = _eval_psi(trial)
        if tv > 1e29:
            break
        trial, tv = _yz_descent(trial, tv)
        cur = trial; cur_v = tv
        psi_bwd[i] = max(tv, 0.0)

    # Combine: backward reversed + forward
    x_full   = np.concatenate([x_bwd[1:][::-1], x_fwd])
    psi_full = np.concatenate([psi_bwd[::-1],   psi_fwd])

    valid = np.isfinite(psi_full)
    if valid.sum() < 3:
        raise RuntimeError("Transport scan: fewer than 3 valid Psi evaluations — scan may have left the mesh.")

    x_v = x_full[valid]
    p_v = psi_full[valid]

    # ── Barrier ───────────────────────────────────────────────────────────────
    psi_max     = float(np.max(p_v))
    peak_idx    = int(np.argmax(p_v))
    peak_x_mesh = float(x_v[peak_idx])
    barrier_raw = max(psi_max - psi0_raw, 0.0)

    barrier_eV = barrier_raw * to_eV
    psi_max_eV = psi_max     * to_eV
    psi_r0_eV  = psi0_raw    * to_eV
    peak_x_m   = peak_x_mesh * coord_unit
    junc_x_m   = junc_x_mesh * coord_unit

    fwd_valid = np.isfinite(psi_fwd)
    reached_junction = (fwd_valid.sum() > 0 and
                        float(x_fwd[np.where(fwd_valid)[0][-1]]) >= junc_x_mesh - 1e-8 * pitch_mesh)

    print(f"[ctb] barrier={barrier_eV*1e3:.3f} meV  "
          f"peak_x={peak_x_m*1e6:.1f} µm  junction_x={junc_x_m*1e6:.1f} µm  "
          f"reached={'yes' if reached_junction else 'NO'}", flush=True)

    result: Dict[str, Any] = {
        "transport_barrier_xscan_eV":           round(barrier_eV, 8),
        "transport_xscan_psi_max_eV":           round(psi_max_eV, 8),
        "transport_xscan_psi_r0_eV":            round(psi_r0_eV, 8),
        "transport_xscan_peak_x_m":             round(peak_x_m, 10),
        "transport_xscan_nearest_junction_x_m": round(junc_x_m, 10),
        "transport_xscan_n_points":             int(valid.sum()),
        "transport_xscan_reached_junction":     bool(reached_junction),
        "transport_xscan_junction_pitch_m":     junction_pitch,
    }

    # ── Optional y-scan ───────────────────────────────────────────────────────
    if scan_both_axes:
        y_end = r0[1] + 0.5 * pitch_mesh
        y_pts = np.linspace(r0[1], y_end + 0.2 * abs(y_end - r0[1]), n_steps)
        cur_y = r0.copy(); cur_v_y = psi0_raw
        psi_y = np.full(len(y_pts), np.nan)
        for i, yi in enumerate(y_pts):
            trial = cur_y.copy(); trial[1] = yi
            tv = _eval_psi(trial)
            if tv > 1e29:
                break
            # Coordinate descent in x, z
            for h_scale in (2.0, 1.0):
                h = h_yz_base * h_scale
                for _ in range(2):
                    for ax in (0, 2):
                        for sign in (+1.0, -1.0):
                            cand = trial.copy(); cand[ax] += sign * h
                            v = _eval_psi(cand)
                            if v < tv:
                                tv = v; trial = cand
            cur_y = trial; cur_v_y = tv
            psi_y[i] = max(tv, 0.0)

        valid_y = np.isfinite(psi_y)
        if valid_y.sum() >= 3:
            psi_y_max     = float(np.max(psi_y[valid_y]))
            barrier_y_raw = max(psi_y_max - psi0_raw, 0.0)
            result["transport_barrier_yscan_eV"] = round(barrier_y_raw * to_eV, 8)
        else:
            result["transport_barrier_yscan_eV"] = None

    return result


# ── Single-case entry point ───────────────────────────────────────────────────

def process_case(
    case_dir: Path,
    *,
    junction_pitch: float,
    n_steps: int,
    scan_both_axes: bool,
    optim_method: str,
    overwrite: bool,
    dry_run: bool,
    output_mode: str,
    rf_tags: Optional[List[int]] = None,
    ground_tags: Optional[List[int]] = None,
    outer_tags: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run the transport barrier scan for one case directory.

    Returns a status dict:
      {"status": "ok"|"skipped"|"error", "case_dir": str,
       "barrier_eV": float|None, "peak_x_um": float|None,
       "junction_x_um": float|None, "used_checkpoint": bool|None,
       "message": str|None}
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    def _skip(msg: str) -> Dict[str, Any]:
        return {"status": "skipped", "case_dir": str(case_dir), "message": msg,
                "barrier_eV": None, "peak_x_um": None, "junction_x_um": None,
                "used_checkpoint": None}

    def _error(msg: str) -> Dict[str, Any]:
        return {"status": "error", "case_dir": str(case_dir), "message": msg,
                "barrier_eV": None, "peak_x_um": None, "junction_x_um": None,
                "used_checkpoint": None}

    report_path, report = _load_report(case_dir)
    if report is None:
        return _skip("no report JSON found" if report_path is None else "JSON parse error")

    if report.get("success") is False:
        return _skip("report.success == False")

    # Already computed and not overwriting?
    existing_val = report.get("transport_barrier_xscan_eV")
    if existing_val is not None and not overwrite:
        return {
            "status": "skipped", "case_dir": str(case_dir),
            "message": "already present (use --overwrite to re-run)",
            "barrier_eV": float(existing_val),
            "peak_x_um": _um_or_none(report.get("transport_xscan_peak_x_m")),
            "junction_x_um": _um_or_none(report.get("transport_xscan_nearest_junction_x_m")),
            "used_checkpoint": None,
        }

    if dry_run:
        has_ckpt = _find_checkpoint(case_dir, report) is not None
        return {
            "status": "dry_run", "case_dir": str(case_dir),
            "message": f"would compute (checkpoint={'yes' if has_ckpt else 'no — will re-solve'})",
            "barrier_eV": None, "peak_x_um": None, "junction_x_um": None,
            "used_checkpoint": None,
        }

    # Resolve electrode tags (CLI > automation_config > report defaults)
    auto_cfg = _load_automation_config(case_dir)
    run_cfg  = (auto_cfg or {}).get("run_config", {})

    _rf_tags     = rf_tags     or run_cfg.get("rf_tags")     or [1]
    _ground_tags = ground_tags or run_cfg.get("ground_tags") or [2, 3]
    _outer_tags  = outer_tags  or run_cfg.get("outer_tags")  or [4]

    t0 = time.perf_counter()
    try:
        domain, phi_rf, used_ckpt = _resolve_phi_rf(
            case_dir, report,
            rf_tags=_rf_tags, ground_tags=_ground_tags, outer_tags=_outer_tags,
            comm=comm,
        )
        Psi = _build_psi(phi_rf, report)

        transport = _scan_transport_barrier(
            Psi, report,
            junction_pitch=junction_pitch,
            n_steps=n_steps,
            scan_both_axes=scan_both_axes,
            optim_method=optim_method,
            comm=comm,
        )
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t0

    # ── Write results ─────────────────────────────────────────────────────────
    if output_mode == "patch":
        report.update(transport)
        # Remove from skipped_metrics if it was there
        skipped = report.get("skipped_metrics", [])
        for k in list(transport.keys()):
            if k in skipped:
                skipped.remove(k)
        report["skipped_metrics"] = skipped
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:  # sidecar
        sidecar = report_path.with_suffix("").name + "_transport.json"
        sidecar_path = case_dir / sidecar
        sidecar_path.write_text(json.dumps(transport, indent=2, sort_keys=True, default=str))

    barrier_eV = transport["transport_barrier_xscan_eV"]
    peak_x_m   = transport.get("transport_xscan_peak_x_m")
    junc_x_m   = transport.get("transport_xscan_nearest_junction_x_m")
    return {
        "status": "ok",
        "case_dir": str(case_dir),
        "message": f"elapsed {elapsed:.1f}s {'(checkpoint)' if used_ckpt else '(re-solved)'}",
        "barrier_eV": float(barrier_eV),
        "peak_x_um": _um_or_none(peak_x_m),
        "junction_x_um": _um_or_none(junc_x_m),
        "used_checkpoint": used_ckpt,
    }


def _find_checkpoint(case_dir: Path, report: Dict[str, Any]) -> Optional[Path]:
    prefix = report.get("prefix", "")
    ckpt_rel = report.get("phi_rf_checkpoint")
    if ckpt_rel:
        p = case_dir / ckpt_rel
        if p.exists():
            return p
    if prefix:
        p = case_dir / f"{prefix}_phi_rf_dofs.npy"
        if p.exists():
            return p
    return None


def _um_or_none(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val) * 1e6, 3)
    except (TypeError, ValueError):
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute transport barrier for completed sweep cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-dir", type=Path, metavar="DIR",
                       help="Single case directory to process.")
    group.add_argument("--sweep-dir", type=Path, metavar="DIR",
                       help="Sweep workdir; processes all case_* subdirectories.")

    ap.add_argument("--junction-pitch", type=float, default=600e-6, metavar="M",
                    help="Junction center-to-center pitch in metres.")
    ap.add_argument("--n-steps", type=int, default=60,
                    help="Number of x sample points per scan direction.")
    ap.add_argument("--scan-both-axes", action="store_true",
                    help="Also scan along y through the junction (default: x only).")
    ap.add_argument("--yz-optim-method", type=str, default="Nelder-Mead",
                    dest="optim_method",
                    help="scipy optimizer for transverse (y,z) minimisation at each step.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run even if transport_barrier_xscan_eV already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without modifying any files.")
    ap.add_argument("--output-mode", choices=["patch", "sidecar"], default="patch",
                    help="'patch': add fields to existing JSON in-place. "
                         "'sidecar': write a separate *_transport.json file.")

    # Electrode tags (optional — auto-detected from automation_config.json otherwise)
    ap.add_argument("--rf-tags", type=int, nargs="+", default=None,
                    help="RF electrode facet tags. Auto-detected from automation_config.json "
                         "if not supplied.")
    ap.add_argument("--ground-tags", type=int, nargs="+", default=None,
                    help="Ground electrode facet tags.")
    ap.add_argument("--outer-tags", type=int, nargs="+", default=None,
                    help="Outer (Neumann) boundary tags.")

    return ap


def _add_src_to_path() -> None:
    """Ensure src/ is on sys.path so metrics, run_case etc. can be imported."""
    src = Path(__file__).parent.resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    _add_src_to_path()

    ap = build_argparser()
    args = ap.parse_args()

    shared_kwargs = dict(
        junction_pitch = args.junction_pitch,
        n_steps        = args.n_steps,
        scan_both_axes = args.scan_both_axes,
        optim_method   = args.optim_method,
        overwrite      = args.overwrite,
        dry_run        = args.dry_run,
        output_mode    = args.output_mode,
        rf_tags        = args.rf_tags,
        ground_tags    = args.ground_tags,
        outer_tags     = args.outer_tags,
    )

    # ── Single case ───────────────────────────────────────────────────────────
    if args.case_dir is not None:
        r = process_case(args.case_dir.resolve(), **shared_kwargs)
        _print_row(r)
        sys.exit(0 if r["status"] in ("ok", "skipped", "dry_run") else 1)

    # ── Batch sweep ───────────────────────────────────────────────────────────
    sweep_dir = args.sweep_dir.resolve()
    case_dirs = sorted(sweep_dir.glob("case_*"))
    if not case_dirs:
        print(f"[batch] No case_* directories found in {sweep_dir}")
        sys.exit(1)

    print(f"[batch] {len(case_dirs)} cases in {sweep_dir}")
    rows: List[Dict[str, Any]] = []
    for cd in case_dirs:
        if not cd.is_dir():
            continue
        r = process_case(cd, **shared_kwargs)
        _print_row(r)
        rows.append(r)

    # Summary table
    n_ok      = sum(1 for r in rows if r["status"] == "ok")
    n_skipped = sum(1 for r in rows if r["status"] == "skipped")
    n_error   = sum(1 for r in rows if r["status"] == "error")
    print(f"\n{'─'*72}")
    print(f"  ok={n_ok}  skipped={n_skipped}  error={n_error}  "
          f"total={len(rows)}")

    # Print table header
    print(f"\n{'case_id':<14}{'barrier_meV':>12}{'peak_x_µm':>11}"
          f"{'junction_x_µm':>15}{'status'}")
    print("─" * 60)
    for r in rows:
        cid    = Path(r["case_dir"]).name
        bar    = f"{r['barrier_eV']*1e3:.2f}" if r["barrier_eV"] is not None else "—"
        peak   = f"{r['peak_x_um']:.1f}"      if r["peak_x_um"]  is not None else "—"
        junc   = f"{r['junction_x_um']:.1f}"  if r["junction_x_um"] is not None else "—"
        print(f"{cid:<14}{bar:>12}{peak:>11}{junc:>15}  {r['status']}")

    # Write aggregate CSV
    csv_path = sweep_dir / "transport_barriers.csv"
    if not args.dry_run and n_ok > 0:
        fieldnames = ["case_id", "status", "barrier_eV", "barrier_meV",
                      "peak_x_um", "junction_x_um", "used_checkpoint", "message"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row = dict(r)
                row["case_id"]    = Path(r["case_dir"]).name
                row["barrier_meV"] = (
                    round(r["barrier_eV"] * 1e3, 4) if r["barrier_eV"] is not None else None
                )
                w.writerow(row)
        print(f"\n[batch] wrote {csv_path}")


def _print_row(r: Dict[str, Any]) -> None:
    cid  = Path(r["case_dir"]).name
    stat = r["status"]
    bar  = f"{r['barrier_eV']*1e3:.2f} meV" if r["barrier_eV"] is not None else "—"
    msg  = r.get("message") or ""
    ckpt = ""
    if r.get("used_checkpoint") is True:
        ckpt = " [ckpt]"
    elif r.get("used_checkpoint") is False:
        ckpt = " [re-solved]"
    print(f"[{cid}] {stat:8s}  barrier={bar}  {msg}{ckpt}")


if __name__ == "__main__":
    main()
