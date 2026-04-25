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
    # Fallback: prefer *_sweep.json, then *_report.json, then sole non-params JSON.
    # Exclude sidecar outputs written by this script (_transport.json, _transport_scan.csv).
    jsons = [
        p for p in sorted(case_dir.glob("*.json"))
        if p.name != "params.json" and not p.name.endswith("_transport.json")
    ]
    for suffix in ("_sweep.json", "_report.json"):
        hits = [p for p in jsons if p.name.endswith(suffix)]
        if hits:
            return hits[0]
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
) -> Tuple[Any, Any, Any]:
    """Load mesh and restore phi_rf from a saved DOF-array checkpoint.

    Returns (domain, facet_tags, phi_rf_function).
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
    return domain, facet_tags, phi_rf


def _resolve_phi_rf(
    case_dir: Path,
    report: Dict[str, Any],
    *,
    rf_tags: List[int],
    ground_tags: List[int],
    outer_tags: List[int],
    comm,
) -> Tuple[Any, Any, Any, bool]:
    """Return (domain, facet_tags, phi_rf, used_checkpoint).

    Tries checkpoint first; falls back to a fresh Laplace solve.
    facet_tags is always returned so callers can reuse it for DC solves.
    """
    from run_case import load_case_mesh, solve_laplace_tagged
    from dolfinx import fem

    degree = int(report.get("degree", 2))
    mesh_str = report.get("mesh", "")
    mesh_path = Path(mesh_str) if mesh_str else case_dir / "mesh.msh"
    if not mesh_path.is_absolute():
        # mesh_str is typically project-root-relative (e.g. "runs/.../mesh.msh");
        # try CWD first, fall back to case_dir for older relative layouts.
        cwd_resolved = (Path.cwd() / mesh_path).resolve()
        mesh_path = cwd_resolved if cwd_resolved.exists() else (case_dir / mesh_path.name).resolve()

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
        try:
            domain, facet_tags, phi_rf = _load_phi_rf_from_checkpoint(
                mesh_path, ckpt_path, degree, comm
            )
            return domain, facet_tags, phi_rf, True
        except ValueError as exc:
            print(f"[ctb] checkpoint unusable ({exc}); falling back to re-solve.", flush=True)
            # ckpt_path stays set so we overwrite it with the fresh solution below

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

    # Save (or overwrite stale) checkpoint so future runs skip the re-solve entirely
    ckpt_name = (ckpt_path.name if ckpt_path is not None
                 else (f"{prefix}_phi_rf_dofs.npy" if prefix else "phi_rf_dofs.npy"))
    ckpt_out = case_dir / ckpt_name
    np.save(str(ckpt_out), phi_rf.x.array)
    report["phi_rf_checkpoint"] = ckpt_name   # picked up when JSON is patched later

    return domain, facet_tags, phi_rf, False


# ── Core: compute Psi from phi_rf ─────────────────────────────────────────────

def _build_psi(phi_rf, report: Dict[str, Any]):
    """Return the RF pseudopotential Function computed from phi_rf."""
    import metrics

    degree = int(report.get("degree", 2))
    rf_freq = float(report.get("rf_freq_Hz", 40e6))
    mass    = float(report.get("mass_amu", 40.0)) * _AMU
    charge  = float(report.get("charge_e", 1.0)) * _E_CHARGE
    omega   = 2.0 * np.pi * rf_freq

    print("[ctb] computing pseudopotential (DG — no inter-element smoothing) ...", flush=True)
    Psi = metrics.compute_rf_pseudopotential(
        phi_rf, omega_rf=omega, q_C=charge, m_kg=mass, degree=degree,
        discontinuous=True,
    )
    Psi.name = "Psi_rf_dg"
    print("[ctb] pseudopotential ready", flush=True)
    return Psi


# ── DC electrode Laplace solutions ───────────────────────────────────────────

def _resolve_phi_dc_all(
    case_dir: Path,
    report: Dict[str, Any],
    domain,
    facet_tags,
    *,
    dc_tags: List[int],
    rf_tags: List[int],
    ground_tags: List[int],
    outer_tags: List[int],
    degree: int = 2,
    comm,
) -> Dict[int, Any]:
    """Solve or load the unit-voltage Laplace solution for each DC electrode tag.

    For each tag in dc_tags, solves ∇²φ = 0 with φ=1 on that tag and φ=0 on
    all other electrode surfaces (RF + ground + other DC).  Outer boundaries
    remain Neumann.  Returns {tag: phi_dc_function}.
    """
    from dolfinx import fem
    from run_case import solve_laplace_tagged

    prefix = report.get("prefix", "")
    outer_set = set(outer_tags)
    phi_dc_map: Dict[int, Any] = {}

    for tag in dc_tags:
        ckpt_name = (f"{prefix}_phi_dc_tag{tag}_dofs.npy" if prefix
                     else f"phi_dc_tag{tag}_dofs.npy")
        ckpt_path = case_dir / ckpt_name

        if ckpt_path.exists():
            print(f"[ctb] loading DC checkpoint for tag {tag}: {ckpt_name}", flush=True)
            V = fem.functionspace(domain, ("CG", degree))
            phi_dc = fem.Function(V)
            phi_dc.name = f"phi_dc_{tag}"
            dofs = np.load(str(ckpt_path))
            if dofs.shape == phi_dc.x.array.shape:
                phi_dc.x.array[:] = dofs
                phi_dc_map[tag] = phi_dc
                continue
            else:
                print(f"[ctb] checkpoint shape mismatch for tag {tag}, re-solving", flush=True)

        # Build BC map: 1V on this DC tag, 0V on RF and ground electrodes.
        # RF electrodes are AC-coupled — DC potential is 0V on them.
        bc_map: Dict[int, float] = {tag: 1.0}
        for rt in rf_tags:
            bc_map[rt] = 0.0
        for gt in ground_tags:
            if gt not in outer_set and gt != tag:
                bc_map[gt] = 0.0
        for other_tag in dc_tags:
            if other_tag != tag:
                bc_map[other_tag] = 0.0

        print(f"[ctb] solving DC Laplace for tag {tag}  bc_map={bc_map} ...", flush=True)
        phi_dc = solve_laplace_tagged(
            domain, facet_tags, bc_map,
            degree=degree,
            petsc_prefix=f"ctb_dc{tag}_",
        )
        phi_dc.name = f"phi_dc_{tag}"

        np.save(str(ckpt_path), phi_dc.x.array)
        print(f"[ctb] DC tag {tag} solved and saved → {ckpt_name}", flush=True)
        phi_dc_map[tag] = phi_dc

    return phi_dc_map


def _build_effective_potential(
    Psi,
    phi_dc_map: Dict[int, Any],
    dc_voltages: Dict[int, float],
    report: Dict[str, Any],
) -> Any:
    """Build Φ_eff = Ψ_RF + q × Σ V_i × φ_DC_i as a dolfinx Function.

    Psi is in "raw" mesh units where Ψ_J = Ψ_raw × (V_RF/coord_unit)².
    phi_dc functions are dimensionless (unit-voltage Laplace solutions).
    dc_voltages[tag] is the applied DC voltage in Volts.

    DC contribution in raw units: DC_raw = q × V_DC × φ_DC / phys_scale
    so that Φ_eff_J = Φ_eff_raw × phys_scale = Ψ_J + Σ q × V_DC_i × φ_DC_i.
    """
    from dolfinx import fem

    charge = float(report.get("charge_e", 1.0)) * _E_CHARGE
    coord_unit = float(report.get("coord_unit_m_per_mesh", 1e-3))
    vrf = float(report.get("vrf_V", 1.0))
    phys_scale = (vrf / coord_unit) ** 2

    Phi_eff = fem.Function(Psi.function_space)
    Phi_eff.name = "Phi_eff"
    Phi_eff.x.array[:] = Psi.x.array[:]

    n_active = 0
    for tag, phi_dc in phi_dc_map.items():
        v_dc = dc_voltages.get(tag, 0.0)
        if abs(v_dc) < 1e-15:
            continue
        n_active += 1

        # Interpolate phi_dc (CG) into Psi's function space (may be DG)
        if phi_dc.function_space == Psi.function_space:
            dc_dofs = phi_dc.x.array
        else:
            phi_dc_interp = fem.Function(Psi.function_space)
            phi_dc_interp.interpolate(phi_dc)
            dc_dofs = phi_dc_interp.x.array

        Phi_eff.x.array[:] += (charge * v_dc / phys_scale) * dc_dofs

    v_range = [dc_voltages[t] for t in phi_dc_map if t in dc_voltages]
    print(
        f"[ctb] built Φ_eff = Ψ_RF + DC  ({n_active} active DC electrode(s), "
        f"V_dc range [{min(v_range, default=0):.3f}, {max(v_range, default=0):.3f}] V)",
        flush=True,
    )
    return Phi_eff


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
    csv_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan toward the nearest junction center and compute the transport barrier.

    Algorithm mirrors metrics._ctc_scan: walk in x, at each step use multi-scale
    coordinate descent in (y, z) to track the Ψ valley floor.  This is more robust
    than scipy Nelder-Mead because the step size is derived from the mesh cell size
    rather than a fixed heuristic.

    Returns a dict ready to be merged into the JSON report.
    """
    import metrics

    _PSI_NEG_WARN_THRESH = -1e-25   # below this we warn; pure roundoff is ~1e-30

    domain     = Psi.function_space.mesh
    coord_unit = float(report.get("coord_unit_m_per_mesh", 1e-3))
    vrf        = float(report.get("vrf_V", 1.0))
    r0_x_m    = float(report["r0_x_m"])
    r0_y_m    = float(report["r0_y_m"])
    r0_z_m    = float(report["r0_z_m"])

    # r0 in mesh units
    r0 = np.array([r0_x_m, r0_y_m, r0_z_m]) / coord_unit

    phys_scale = (vrf / coord_unit) ** 2   # Psi uses mesh-coord gradients (1/mesh_unit);
                                           # scale to SI: V_RF² / coord_unit²  [J]
    to_eV      = phys_scale / _E_CHARGE

    # Step size for coordinate descent in transverse plane — use mesh cell size
    h_mesh = metrics._estimate_cell_h(domain)
    h_yz_base = h_mesh * 1.5    # matches metrics._ctc_scan

    # Psi at r0 — keep raw (unclipped) to detect whether r0 is on the RF null
    psi0_raw = float(metrics.eval_function_at_points(
        Psi, np.array([r0], dtype=np.float64), comm=comm
    )[0])
    if not np.isfinite(psi0_raw):
        raise ValueError(f"Psi is NaN at r0 (mesh units {r0.tolist()}). r0 may be outside mesh.")
    if psi0_raw < _PSI_NEG_WARN_THRESH:
        print(f"[ctb] WARNING: psi(r0)={psi0_raw:.3e} J is significantly negative "
              f"(threshold {_PSI_NEG_WARN_THRESH:.0e}). Check for baseline subtraction "
              f"or interpolation artefacts.", flush=True)

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

    # ── Find nearest and far junction centers ────────────────────────────────
    pitch_mesh  = junction_pitch / coord_unit
    near_junc   = round(r0[0] / pitch_mesh) * pitch_mesh   # nearest junction
    # Far junction: the next junction center away from r0 (toward the barrier)
    if r0[0] >= near_junc:
        far_junc = near_junc + pitch_mesh
    else:
        far_junc = near_junc - pitch_mesh
    junc_x_mesh = far_junc            # target junction for the forward scan

    dx_mesh      = far_junc - r0[0]   # signed distance to the far junction
    scan_end     = far_junc           # stop at junction center (no overshoot)
    scan_back    = r0[0] - 0.2 * abs(dx_mesh)

    x_fwd = np.linspace(r0[0], scan_end,  n_steps)
    x_bwd = np.linspace(r0[0], scan_back, max(n_steps // 5, 5))

    # ── Forward scan (r0 → junction center) ──────────────────────────────────
    print(f"[ctb] r0 psi={psi0_raw:.3e} J  ({psi0_raw*to_eV*1e3:.4f} meV)", flush=True)
    print(f"[ctb] scanning x: r0={r0[0]:.4f} → far_junction={junc_x_mesh:.4f} "
          f"near_junction={near_junc:.4f}  "
          f"n_steps={n_steps}  h_yz={h_yz_base:.4f}", flush=True)
    print(f"[ctb] coord_unit={coord_unit}  vrf={vrf} V  phys_scale={phys_scale:.3e}"
          f"  to_eV={to_eV:.3e}", flush=True)

    # csv_rows accumulates every scan step for the sidecar CSV
    csv_rows: List[Dict[str, Any]] = []

    cur = r0.copy()
    cur_v = psi0_raw
    # Store unclipped psi values; NaN marks out-of-mesh or not-yet-reached
    psi_fwd  = np.full(len(x_fwd), np.nan)
    yopt_fwd = np.full(len(x_fwd), np.nan)
    zopt_fwd = np.full(len(x_fwd), np.nan)
    for i, xi in enumerate(x_fwd):
        trial = cur.copy(); trial[0] = xi
        tv = _eval_psi(trial)
        if tv > 1e29:
            break
        trial, tv = _yz_descent(trial, tv)
        cur = trial; cur_v = tv
        psi_fwd[i]  = tv          # unclipped — negative roundoff preserved
        yopt_fwd[i] = trial[1]
        zopt_fwd[i] = trial[2]
        if tv < _PSI_NEG_WARN_THRESH:
            print(f"[ctb] WARNING step {i}: psi={tv:.3e} J is significantly negative", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[ctb]   step {i+1}/{len(x_fwd)}  x={xi:.4f}  psi={tv:.3e}  "
                  f"({tv*to_eV*1e3:.4f} meV)", flush=True)
        csv_rows.append({
            "direction": "fwd", "step": i,
            "x_mesh": xi, "y_mesh": trial[1], "z_mesh": trial[2],
            "x_m": xi * coord_unit, "y_m": trial[1] * coord_unit,
            "z_m": trial[2] * coord_unit,
            "psi_J": tv, "psi_eV": tv * to_eV, "psi_meV": tv * to_eV * 1e3,
        })

    # ── Backward scan (r0 → r0 - 20%) ────────────────────────────────────────
    cur = r0.copy(); cur_v = psi0_raw
    psi_bwd  = np.full(len(x_bwd) - 1, np.nan)
    yopt_bwd = np.full(len(x_bwd) - 1, np.nan)
    zopt_bwd = np.full(len(x_bwd) - 1, np.nan)
    for i, xi in enumerate(x_bwd[1:]):
        trial = cur.copy(); trial[0] = xi
        tv = _eval_psi(trial)
        if tv > 1e29:
            break
        trial, tv = _yz_descent(trial, tv)
        cur = trial; cur_v = tv
        psi_bwd[i]  = tv
        yopt_bwd[i] = trial[1]
        zopt_bwd[i] = trial[2]
        csv_rows.append({
            "direction": "bwd", "step": i,
            "x_mesh": xi, "y_mesh": trial[1], "z_mesh": trial[2],
            "x_m": xi * coord_unit, "y_m": trial[1] * coord_unit,
            "z_m": trial[2] * coord_unit,
            "psi_J": tv, "psi_eV": tv * to_eV, "psi_meV": tv * to_eV * 1e3,
        })

    # Combine: backward reversed + forward
    x_full   = np.concatenate([x_bwd[1:][::-1], x_fwd])
    psi_full = np.concatenate([psi_bwd[::-1],   psi_fwd])

    valid = np.isfinite(psi_full)
    if valid.sum() < 3:
        raise RuntimeError("Transport scan: fewer than 3 valid Psi evaluations — scan may have left the mesh.")

    # Barrier is defined only along the forward path (r0 → junction).
    # The backward slice is kept in x_full/psi_full for diagnostics only.
    valid_fwd = np.isfinite(psi_fwd)
    if valid_fwd.sum() < 3:
        raise RuntimeError("Transport scan: fewer than 3 valid forward-scan points — scan may have left the mesh.")

    p_fwd = psi_fwd[valid_fwd]
    x_fwd_valid = x_fwd[valid_fwd]

    # ── Barrier — two definitions, both unclipped before reporting ───────────
    psi_max     = float(np.max(p_fwd))
    psi_min     = float(np.min(p_fwd))
    psi_end     = float(p_fwd[-1])
    peak_idx    = int(np.argmax(p_fwd))
    min_idx     = int(np.argmin(p_fwd))
    peak_x_mesh = float(x_fwd_valid[peak_idx])
    min_x_mesh  = float(x_fwd_valid[min_idx])

    # barrier_A: how far above r0 the peak rises (classic CTC definition)
    barrier_A_raw = psi_max - psi0_raw   # can be negative if peak < start
    # barrier_B: total elevation from lowest point on path to the peak
    barrier_B_raw = psi_max - psi_min    # always >= 0 by definition

    barrier_A_eV  = barrier_A_raw * to_eV
    barrier_B_eV  = barrier_B_raw * to_eV
    psi_max_eV    = psi_max  * to_eV
    psi_min_eV    = psi_min  * to_eV
    psi_r0_eV     = psi0_raw * to_eV
    psi_end_eV    = psi_end  * to_eV
    peak_x_m      = peak_x_mesh * coord_unit
    min_x_m       = min_x_mesh  * coord_unit
    junc_x_m      = junc_x_mesh * coord_unit

    fwd_valid = np.isfinite(psi_fwd)
    reached_junction = (fwd_valid.sum() > 0 and
                        float(x_fwd[np.where(fwd_valid)[0][-1]]) >= junc_x_mesh - 1e-8 * pitch_mesh)

    # ── Debug summary ──────────────────────────────────────────────────────────
    print(f"[ctb] ── scan summary ──────────────────────────────────────────", flush=True)
    print(f"[ctb]   psi_start  = {psi0_raw:.4e} J  = {psi_r0_eV*1e3:.4f} meV", flush=True)
    print(f"[ctb]   psi_min    = {psi_min:.4e} J  = {psi_min_eV*1e3:.4f} meV  "
          f"at x={min_x_m*1e6:.1f} µm", flush=True)
    print(f"[ctb]   psi_max    = {psi_max:.4e} J  = {psi_max_eV*1e3:.4f} meV  "
          f"at x={peak_x_m*1e6:.1f} µm (peak)", flush=True)
    print(f"[ctb]   psi_end    = {psi_end:.4e} J  = {psi_end_eV*1e3:.4f} meV", flush=True)
    print(f"[ctb]   barrier_A (peak−start)    = {barrier_A_raw:.4e} J  = "
          f"{barrier_A_eV*1e3:.4f} meV  [clipped={max(barrier_A_eV,0)*1e3:.4f} meV]",
          flush=True)
    print(f"[ctb]   barrier_B (peak−path_min) = {barrier_B_raw:.4e} J  = "
          f"{barrier_B_eV*1e3:.4f} meV", flush=True)
    print(f"[ctb]   junction_x={junc_x_m*1e6:.1f} µm  "
          f"reached={'yes' if reached_junction else 'NO'}  "
          f"n_valid={int(valid_fwd.sum())}", flush=True)
    if abs(psi0_raw) < 1e-28:
        print(f"[ctb] WARNING: psi(r0) ≈ 0 ({psi0_raw:.3e} J). r0 may be on the RF null "
              f"— check that r0 coordinates are correct and not on the trap axis.", flush=True)
    if abs(psi_max) < 1e-25:
        print(f"[ctb] WARNING: psi_max ≈ 0 ({psi_max:.3e} J). Entire scan path may be on "
              f"the RF null — verify junction_pitch and junction_x.", flush=True)

    # ── Write CSV sidecar ──────────────────────────────────────────────────────
    if csv_out is not None and csv_rows:
        _csv_fields = ["direction", "step",
                       "x_mesh", "y_mesh", "z_mesh",
                       "x_m", "y_m", "z_m",
                       "psi_J", "psi_eV", "psi_meV"]
        try:
            with csv_out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=_csv_fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(csv_rows)
            print(f"[ctb] scan CSV → {csv_out}", flush=True)
        except OSError as exc:
            print(f"[ctb] WARNING: could not write scan CSV: {exc}", flush=True)

    result: Dict[str, Any] = {
        # ── backward-compatible key (clipped, peak−start) ──────────────────
        "transport_barrier_xscan_eV":           round(max(barrier_A_eV, 0.0), 8),
        # ── new diagnostic keys (unclipped) ───────────────────────────────
        "transport_barrier_meV_peak_minus_start":    round(barrier_A_eV  * 1e3, 6),
        "transport_barrier_meV_peak_minus_path_min": round(barrier_B_eV  * 1e3, 6),
        "transport_path_min_meV":                    round(psi_min_eV    * 1e3, 6),
        "transport_path_max_meV":                    round(psi_max_eV    * 1e3, 6),
        "transport_path_start_meV":                  round(psi_r0_eV     * 1e3, 6),
        "transport_path_end_meV":                    round(psi_end_eV    * 1e3, 6),
        # ── geometry ──────────────────────────────────────────────────────
        "transport_xscan_psi_max_eV":           round(psi_max_eV, 8),
        "transport_xscan_psi_r0_eV":            round(psi_r0_eV, 8),
        "transport_xscan_peak_x_m":             round(peak_x_m, 10),
        "transport_xscan_path_min_x_m":         round(min_x_m, 10),
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
            psi_y[i] = tv   # unclipped

        valid_y = np.isfinite(psi_y)
        if valid_y.sum() >= 3:
            p_y = psi_y[valid_y]
            psi_y_max     = float(np.max(p_y))
            barrier_y_raw = psi_y_max - psi0_raw
            result["transport_barrier_yscan_eV"] = round(max(barrier_y_raw * to_eV, 0.0), 8)
        else:
            result["transport_barrier_yscan_eV"] = None

    return result


# ── String-method CTC path tracer ────────────────────────────────────────────

def _ctc_string_method(
    Psi,
    report: Dict[str, Any],
    *,
    junction_pitch: float = 600e-6,
    n_nodes: int = 40,
    n_iter: int = 200,
    h_step_factor: float = 0.3,
    comm=None,
    csv_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """String-method minimum-energy path from r0 to the nearest junction center.

    Instead of scanning along a fixed y=0, z=r0_z line (which overestimates the
    barrier), this relaxes a string of nodes so each interior node slides
    downhill in the transverse (y, z) plane while x stays fixed.  The result is
    the true lowest-energy corridor through the junction.

    Returns a dict of result keys to merge into the JSON report.
    """
    import metrics

    domain     = Psi.function_space.mesh
    coord_unit = float(report.get("coord_unit_m_per_mesh", 1e-3))
    vrf        = float(report.get("vrf_V", 1.0))
    r0_x_m     = float(report["r0_x_m"])
    r0_y_m     = float(report["r0_y_m"])
    r0_z_m     = float(report["r0_z_m"])

    # r0 in mesh units
    r0 = np.array([r0_x_m, r0_y_m, r0_z_m]) / coord_unit

    phys_scale = (vrf / coord_unit) ** 2
    to_eV      = phys_scale / _E_CHARGE

    h_mesh = metrics._estimate_cell_h(domain)
    h_step = h_step_factor * h_mesh       # gradient-descent step size
    h_fd   = h_mesh * 0.25               # finite-difference probe offset

    # ── Locate junction ──────────────────────────────────────────────────────
    pitch_mesh  = junction_pitch / coord_unit
    near_junc   = round(r0[0] / pitch_mesh) * pitch_mesh
    if r0[0] >= near_junc:
        far_junc = near_junc + pitch_mesh
    else:
        far_junc = near_junc - pitch_mesh
    junc_x_mesh = far_junc

    # ── Pre-locate Ψ minimum at the junction (y, z coordinate descent) ───────
    # The linear-region r0 transverse coordinates (y, z) are generally NOT the
    # equilibrium position at the junction.  Pinning the far endpoint there
    # stretches the string to a non-minimum and overestimates the barrier.
    junc_pt = np.array([junc_x_mesh, r0[1], r0[2]], dtype=np.float64)
    _v0_junc = float(metrics.eval_function_at_points(Psi, junc_pt[None], comm=comm)[0])
    junc_v_val = _v0_junc if np.isfinite(_v0_junc) else 1e30
    for _h_scale in (4.0, 2.0, 1.0, 0.5, 0.25):
        _h = h_mesh * _h_scale
        for _ in range(4):
            _probes = np.array([
                [junc_x_mesh, junc_pt[1] + _h, junc_pt[2]],
                [junc_x_mesh, junc_pt[1] - _h, junc_pt[2]],
                [junc_x_mesh, junc_pt[1],       junc_pt[2] + _h],
                [junc_x_mesh, junc_pt[1],       junc_pt[2] - _h],
            ], dtype=np.float64)
            _pvals = metrics.eval_function_at_points(Psi, _probes, comm=comm)
            _best = int(np.nanargmin(_pvals)) if np.any(np.isfinite(_pvals)) else -1
            if _best >= 0 and np.isfinite(_pvals[_best]) and _pvals[_best] < junc_v_val:
                junc_v_val = float(_pvals[_best])
                junc_pt = _probes[_best].copy()
            else:
                break  # no improvement at this scale

    # ── 1. INITIALIZE string ─────────────────────────────────────────────────
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    nodes[:, 0] = np.linspace(r0[0], junc_x_mesh, n_nodes)
    nodes[:, 1] = r0[1]   # start at y0
    nodes[:, 2] = r0[2]   # start at z0
    nodes[-1] = junc_pt.copy()   # far endpoint at junction Ψ minimum

    # Psi at r0 baseline
    psi0_raw = float(metrics.eval_function_at_points(
        Psi, r0[None].astype(np.float64), comm=comm
    )[0])
    if not np.isfinite(psi0_raw):
        raise ValueError(f"Psi is NaN at r0 (mesh units {r0.tolist()}). r0 may be outside mesh.")

    print(f"[ctc] string method: {n_nodes} nodes, up to {n_iter} iterations", flush=True)
    print(f"[ctc] r0={r0}  junction_x={junc_x_mesh:.4f}  h_step={h_step:.5f}  h_fd={h_fd:.5f}", flush=True)
    print(f"[ctc] psi(r0)={psi0_raw:.4e} J  ({psi0_raw*to_eV*1e3:.4f} meV)", flush=True)
    print(f"[ctc] junction min: y={junc_pt[1]:.5f} z={junc_pt[2]:.5f}  "
          f"psi={junc_v_val:.4e} J  ({junc_v_val*to_eV*1e3:.4f} meV)", flush=True)

    # z floor: 1% above the mesh floor, but never above half the ion height.
    # The second guard prevents the clamp from cutting off the transport path
    # in surface traps where r0_z is a small fraction of the mesh z-extent.
    _coords = domain.geometry.x
    _z_floor = float(_coords[:, 2].min()) + 0.01 * (
        float(_coords[:, 2].max()) - float(_coords[:, 2].min())
    )
    z_clamp = min(_z_floor, r0[2] * 0.5)

    # ── 2. ITERATE ───────────────────────────────────────────────────────────
    converged = False
    actual_iter = 0
    # Nodes to relax: indices 1..n_nodes-1 (interior + far endpoint).
    # The start node (index 0) is permanently fixed at r0.
    # The far endpoint's y, z are allowed to slide; its x is re-pinned after.
    n_relax = n_nodes - 1
    max_step = h_mesh * 0.5   # per-iteration displacement cap (prevents overshoot)
    prev_barrier_raw: Optional[float] = None
    barrier_stable_count = 0

    for it in range(n_iter):
        actual_iter = it + 1

        # ── 2a. SLIDE: gradient descent for interior nodes + far endpoint ────
        # Build 4 * n_relax probe points (+y, -y, +z, -z) in a single batch.
        relax = nodes[1:].copy()   # shape (n_relax, 3)
        probes = np.empty((4 * n_relax, 3), dtype=np.float64)
        probes[0::4] = relax.copy(); probes[0::4, 1] += h_fd  # +y
        probes[1::4] = relax.copy(); probes[1::4, 1] -= h_fd  # -y
        probes[2::4] = relax.copy(); probes[2::4, 2] += h_fd  # +z
        probes[3::4] = relax.copy(); probes[3::4, 2] -= h_fd  # -z

        psi_probes = metrics.eval_function_at_points(Psi, probes, comm=comm)

        max_disp = 0.0
        for j in range(n_relax):
            py_plus  = psi_probes[4 * j]
            py_minus = psi_probes[4 * j + 1]
            pz_plus  = psi_probes[4 * j + 2]
            pz_minus = psi_probes[4 * j + 3]

            if not (np.isfinite(py_plus) and np.isfinite(py_minus)
                    and np.isfinite(pz_plus) and np.isfinite(pz_minus)):
                continue

            dpsi_dy = (py_plus - py_minus) / (2.0 * h_fd)
            dpsi_dz = (pz_plus - pz_minus) / (2.0 * h_fd)

            dy = h_step * dpsi_dy
            dz = h_step * dpsi_dz
            disp = np.sqrt(dy * dy + dz * dz)
            # Cap displacement to prevent overshooting in deep narrow valleys
            if disp > max_step:
                scale = max_step / disp
                dy *= scale
                dz *= scale
                disp = max_step
            if disp > max_disp:
                max_disp = disp

            nodes[1 + j, 1] -= dy
            nodes[1 + j, 2] -= dz
            if nodes[1 + j, 2] < z_clamp:
                nodes[1 + j, 2] = z_clamp

        # Re-pin far endpoint x (y, z allowed to move freely)
        nodes[-1, 0] = junc_x_mesh

        # ── Out-of-mesh guard ────────────────────────────────────────────────
        # Evaluate all relaxed nodes in one batch; reset NaN nodes to safe positions.
        guard_psi = metrics.eval_function_at_points(
            Psi, nodes[1:].astype(np.float64), comm=comm
        )
        for j in range(n_nodes - 2):   # interior nodes
            if not np.isfinite(guard_psi[j]):
                nodes[1 + j] = 0.5 * (nodes[j] + nodes[2 + j])
        # Far endpoint: reset to pre-located junction minimum if out of mesh
        if not np.isfinite(guard_psi[-1]):
            nodes[-1] = junc_pt.copy()
            nodes[-1, 0] = junc_x_mesh

        # ── 2b. REPARAMETRIZE: redistribute nodes by arc length ──────────────
        # Arc-length reparametrization clusters nodes near the saddle where the
        # path bends in y/z, giving better barrier resolution than uniform-x spacing.
        diffs = np.diff(nodes, axis=0)              # (n_nodes-1, 3)
        seg_lengths = np.linalg.norm(diffs, axis=1)  # (n_nodes-1,)
        cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cum_length[-1]

        if total_length > 1e-30:
            target_s = np.linspace(0.0, total_length, n_nodes)
            new_nodes = nodes.copy()
            for _coord in range(3):
                new_nodes[1:-1, _coord] = np.interp(
                    target_s[1:-1], cum_length, nodes[:, _coord]
                )
            nodes = new_nodes

        # Enforce endpoint constraints: start fixed at r0, far x pinned
        nodes[0] = r0.copy()
        nodes[-1, 0] = junc_x_mesh   # y, z remain where gradient descent left them

        # ── 2c. CONVERGENCE CHECK ────────────────────────────────────────────
        if (it + 1) % 5 == 0:
            psi_check = metrics.eval_function_at_points(
                Psi, nodes.astype(np.float64), comm=comm
            )
            current_barrier_raw = float(np.nanmax(psi_check)) - psi0_raw
            if prev_barrier_raw is not None:
                rel_change = abs(current_barrier_raw - prev_barrier_raw) / max(
                    abs(prev_barrier_raw), 1e-40
                )
                if rel_change < 1e-4 and max_disp < 1e-4 * h_mesh:
                    barrier_stable_count += 1
                else:
                    barrier_stable_count = 0
                if barrier_stable_count >= 3:
                    converged = True
                    print(
                        f"[ctc]   iter {it+1:3d}/{n_iter}  barrier="
                        f"{current_barrier_raw*to_eV*1e3:.4f} meV  converged (barrier stable)",
                        flush=True,
                    )
                    break
            prev_barrier_raw = current_barrier_raw
            print(
                f"[ctc]   iter {it+1:3d}/{n_iter}  max_disp={max_disp:.3e}  "
                f"barrier={current_barrier_raw*to_eV*1e3:.4f} meV  stable={barrier_stable_count}",
                flush=True,
            )

        # Require at least 5 iterations before the displacement threshold can
        # trigger — prevents false convergence when the string starts on the
        # RF null (flat y/z landscape gives near-zero displacement in iter 1).
        if it >= 4 and max_disp < 1e-5 * h_mesh:
            converged = True
            print(f"[ctc] converged at iteration {it+1} (displacement threshold)", flush=True)
            break

    if not converged:
        print(f"[ctc] did not converge in {n_iter} iterations "
              f"(max_disp={max_disp:.3e}, threshold={1e-5*h_mesh:.3e})", flush=True)

    # ── 3. EVALUATE Psi at all final node positions ──────────────────────────
    psi_path = metrics.eval_function_at_points(
        Psi, nodes.astype(np.float64), comm=comm
    )

    # Replace any NaN with neighbors' average for robustness
    for j in range(1, n_nodes - 1):
        if not np.isfinite(psi_path[j]):
            left  = psi_path[j - 1] if np.isfinite(psi_path[j - 1]) else 0.0
            right = psi_path[j + 1] if np.isfinite(psi_path[j + 1]) else 0.0
            psi_path[j] = 0.5 * (left + right)

    # ── 4. BARRIER ───────────────────────────────────────────────────────────
    psi_max = float(np.nanmax(psi_path))
    peak_idx = int(np.nanargmax(psi_path))
    barrier_raw = psi_max - psi0_raw
    barrier_eV  = barrier_raw * to_eV
    barrier_eV_clipped = max(barrier_eV, 0.0)

    peak_x_m = float(nodes[peak_idx, 0]) * coord_unit
    peak_y_m = float(nodes[peak_idx, 1]) * coord_unit
    peak_z_m = float(nodes[peak_idx, 2]) * coord_unit

    print(f"[ctc] ── string result ─────────────────────────────────────────", flush=True)
    print(f"[ctc]   barrier = {barrier_eV*1e3:.4f} meV  "
          f"(clipped={barrier_eV_clipped*1e3:.4f} meV)", flush=True)
    print(f"[ctc]   peak at x={peak_x_m*1e6:.1f} µm  y={peak_y_m*1e6:.1f} µm  "
          f"z={peak_z_m*1e6:.1f} µm", flush=True)
    print(f"[ctc]   converged={converged}  iterations={actual_iter}", flush=True)

    # ── 5. CSV sidecar ───────────────────────────────────────────────────────
    if csv_out is not None:
        _csv_fields = ["node", "x_m", "y_m", "z_m",
                       "x_mesh", "y_mesh", "z_mesh",
                       "psi_J", "psi_eV", "psi_meV"]
        try:
            with csv_out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=_csv_fields)
                w.writeheader()
                for j in range(n_nodes):
                    pj = psi_path[j] if np.isfinite(psi_path[j]) else float("nan")
                    w.writerow({
                        "node": j,
                        "x_m": nodes[j, 0] * coord_unit,
                        "y_m": nodes[j, 1] * coord_unit,
                        "z_m": nodes[j, 2] * coord_unit,
                        "x_mesh": nodes[j, 0],
                        "y_mesh": nodes[j, 1],
                        "z_mesh": nodes[j, 2],
                        "psi_J": pj if np.isfinite(pj) else float("nan"),
                        "psi_eV": pj * to_eV if np.isfinite(pj) else float("nan"),
                        "psi_meV": pj * to_eV * 1e3 if np.isfinite(pj) else float("nan"),
                    })
            print(f"[ctc] path CSV → {csv_out}", flush=True)
        except OSError as exc:
            print(f"[ctc] WARNING: could not write path CSV: {exc}", flush=True)

    return {
        "transport_barrier_ctc_eV":       round(barrier_eV_clipped, 8),
        "transport_barrier_ctc_meV_raw":  round(barrier_eV * 1e3, 6),
        "transport_ctc_peak_x_m":         round(peak_x_m, 10),
        "transport_ctc_peak_y_m":         round(peak_y_m, 10),
        "transport_ctc_peak_z_m":         round(peak_z_m, 10),
        "transport_ctc_n_nodes":          n_nodes,
        "transport_ctc_n_iter":           actual_iter,
        "transport_ctc_converged":        converged,
        "transport_ctc_junction_x_m":     round(float(junc_x_mesh) * coord_unit, 10),
    }


# ── Paper-method: pseudopotential-minimum path + CTC path (2025 paper) ───────

def _paper_ctc_and_min_scan(
    Psi,
    report: Dict[str, Any],
    *,
    junction_pitch: float = 600e-6,
    n_x: int = 60,
    ctc_z_min_m: Optional[float] = None,
    ctc_z_max_m: Optional[float] = None,
    n_z: int = 200,
    ctc_target_conf: float = 0.75e9,   # eV/m²
    ctc_target_auto: bool = False,
    hessian_step_um: float = 2.0,
    run_min: bool = True,
    run_ctc: bool = True,
    comm=None,
    csv_out_min: Optional[Path] = None,
    csv_out_ctc: Optional[Path] = None,
) -> Dict[str, Any]:
    """2025-paper method: scan z at y=0 for each x; extract pseudopotential-
    minimum path and constant-total-confinement (CTC) path.

    C(x,0,z) = ∇²Φ_pp  [eV/m²] via Hessian trace central finite differences.
    CTC path: z where C = ctc_target_conf at each x.
    Min path: z where Φ_pp is minimum at each x.

    Crossing selection:
      For each x, the CTC uses the "outer" crossing: the first sign change of
      (C - C_target) at z ≥ z_min_local.  This corresponds to the confinement
      isosurface that bounds the trap from above (moving away from the surface).
      If no outer crossing exists (C(z_min) < C_target at this x), falls back
      to the nearest-to-z_min crossing and marks ctc_crossing_side="inner".

    ctc_target_auto=True: set C_target = C(z_min) at the first x step (r0),
      calibrating the CTC threshold to the actual linear-region confinement.

    Returns dict with transport_min_path, transport_ctc_path, ctc_barrier_eV,
    ctc_barrier_meV, ctc_target_conf_eV_per_m2, ctc_max_abs_dz_um,
    ctc_rms_dz_um, ctc_no_crossing_count.
    """
    import metrics

    domain     = Psi.function_space.mesh
    coord_unit = float(report.get("coord_unit_m_per_mesh", 1e-3))
    vrf        = float(report.get("vrf_V", 1.0))
    r0_x_m     = float(report["r0_x_m"])
    r0_y_m     = float(report["r0_y_m"])
    r0_z_m     = float(report["r0_z_m"])

    phys_scale = (vrf / coord_unit) ** 2
    to_eV      = phys_scale / _E_CHARGE

    r0 = np.array([r0_x_m, r0_y_m, r0_z_m]) / coord_unit  # mesh units

    # ── Junction x ────────────────────────────────────────────────────────────
    pitch_mesh = junction_pitch / coord_unit
    near_junc  = round(r0[0] / pitch_mesh) * pitch_mesh
    far_junc   = (near_junc + pitch_mesh if r0[0] >= near_junc
                  else near_junc - pitch_mesh)
    junc_x_mesh = far_junc
    junc_x_m    = junc_x_mesh * coord_unit

    # ── x sample points: linear region → junction center (no overshoot) ─────
    dx_mesh  = far_junc - r0[0]
    scan_end = far_junc          # stop exactly at junction center
    x_mesh   = np.linspace(r0[0], scan_end, n_x)

    # ── z range in mesh units ─────────────────────────────────────────────────
    coords = domain.geometry.x
    z_mesh_all = coords[:, 2]
    z_floor_mesh = float(z_mesh_all.min())
    z_ceil_mesh  = float(z_mesh_all.max())
    # Default: full mesh z extent, shrunk slightly from boundaries
    z_margin = 0.01 * (z_ceil_mesh - z_floor_mesh)
    if ctc_z_min_m is not None:
        z_min_mesh = ctc_z_min_m / coord_unit
    else:
        z_min_mesh = z_floor_mesh + z_margin
    if ctc_z_max_m is not None:
        z_max_mesh = ctc_z_max_m / coord_unit
    else:
        z_max_mesh = z_ceil_mesh - z_margin
    z_scan_mesh = np.linspace(z_min_mesh, z_max_mesh, n_z)  # shape (n_z,)

    # ── Finite-difference step in mesh units and SI ───────────────────────────
    h_si   = hessian_step_um * 1e-6                  # metres
    h_mesh = h_si / coord_unit                       # mesh units
    # Laplacian conversion: C [eV/m²] = to_eV × Σ FD_second / h_si²
    # (the coord_unit² in the denominator and numerator cancel perfectly)
    lap_scale = to_eV / (h_si ** 2)

    y_fixed_mesh = 0.0  # y = 0

    # ctc_target_auto: calibrate C_target to C(z_min) at the first x step.
    # This ensures the CTC passes through the linear-region trap minimum.
    # The actual value is determined on the first x step below.
    _ctc_target_resolved = ctc_target_conf   # may be overridden on ix==0
    _ctc_target_set_auto = False

    print(f"[paper-ctc] scanning {n_x} x-points  "
          f"x=[{r0[0]*coord_unit*1e6:.1f}, {scan_end*coord_unit*1e6:.1f}] µm  "
          f"n_z={n_z}  z=[{z_min_mesh*coord_unit*1e6:.1f}, "
          f"{z_max_mesh*coord_unit*1e6:.1f}] µm  "
          f"h_fd={hessian_step_um:.1f} µm  "
          f"C_target={'auto (from r0)' if ctc_target_auto else f'{ctc_target_conf:.3e} eV/m²'}",
          flush=True)

    # ── Per-x scan ────────────────────────────────────────────────────────────
    # For each x we need 7 × n_z evaluations:
    #   center, ±x, ±y, ±z stencil.
    # Batch all 7n_z evaluations per x for efficiency.

    min_path_rows: List[Dict[str, Any]] = []
    ctc_path_rows: List[Dict[str, Any]] = []
    prev_z_ctc_mesh: Optional[float] = None   # z-continuity anchor for CTC crossing
    for ix, xi in enumerate(x_mesh):
        # Build 7 × n_z stencil points, shape (7*n_z, 3)
        base_pts = np.column_stack([
            np.full(n_z, xi), np.full(n_z, y_fixed_mesh), z_scan_mesh
        ])  # (n_z, 3) — center
        pts_xp = base_pts.copy(); pts_xp[:, 0] += h_mesh
        pts_xm = base_pts.copy(); pts_xm[:, 0] -= h_mesh
        pts_yp = base_pts.copy(); pts_yp[:, 1] += h_mesh
        pts_ym = base_pts.copy(); pts_ym[:, 1] -= h_mesh
        pts_zp = base_pts.copy(); pts_zp[:, 2] += h_mesh
        pts_zm = base_pts.copy(); pts_zm[:, 2] -= h_mesh

        all_pts = np.vstack([base_pts, pts_xp, pts_xm,
                             pts_yp,  pts_ym,
                             pts_zp,  pts_zm])  # (7*n_z, 3)
        vals = metrics.eval_function_at_points(Psi, all_pts, comm=comm)

        phi_c  = vals[0       : n_z]
        phi_xp = vals[n_z     : 2*n_z]
        phi_xm = vals[2*n_z   : 3*n_z]
        phi_yp = vals[3*n_z   : 4*n_z]
        phi_ym = vals[4*n_z   : 5*n_z]
        phi_zp = vals[5*n_z   : 6*n_z]
        phi_zm = vals[6*n_z   : 7*n_z]

        # Second derivatives (raw Psi units / mesh_unit²)
        d2x = phi_xp - 2.0*phi_c + phi_xm
        d2y = phi_yp - 2.0*phi_c + phi_ym
        d2z = phi_zp - 2.0*phi_c + phi_zm
        # Laplacian in eV/m²
        conf = lap_scale * (d2x + d2y + d2z)  # shape (n_z,)

        phi_eV = phi_c * to_eV   # pseudopotential along this column [eV]

        # Valid mask: both phi and conf finite
        valid_mask = np.isfinite(phi_c) & np.isfinite(conf)

        # Global argmin — used for min-path and to anchor the CTC outer-crossing search
        valid_phi = np.where(np.isfinite(phi_c), phi_c, np.inf)
        iz_min_local = int(np.argmin(valid_phi))
        z_min_local  = z_scan_mesh[iz_min_local] if np.isfinite(phi_c[iz_min_local]) else None

        if run_min and z_min_local is not None:
            phi_min_eV  = float(phi_eV[iz_min_local])
            conf_min_eV = float(conf[iz_min_local]) if np.isfinite(conf[iz_min_local]) else float("nan")
            trap_h_um   = (z_min_local * coord_unit) * 1e6
            min_path_rows.append({
                "x_m":                  float(xi * coord_unit),
                "y_m":                  float(y_fixed_mesh * coord_unit),
                "z_m":                  float(z_min_local * coord_unit),
                "phi_pp_eV":            round(phi_min_eV, 9),
                "total_conf_eV_per_m2": round(conf_min_eV, 3),
                "trap_height_um":       round(trap_h_um, 4),
                "path_type":            "minimum",
            })
            # Auto-calibrate C_target on first valid x step (use global min for CTC)
            if ctc_target_auto and not _ctc_target_set_auto and np.isfinite(conf[iz_min_local]):
                _ctc_target_resolved = float(conf[iz_min_local])
                _ctc_target_set_auto = True
                print(f"[paper-ctc] auto C_target = {_ctc_target_resolved:.4e} eV/m² "
                      f"(C at z_min at r0, x={xi*coord_unit*1e6:.1f} µm)", flush=True)

        if run_ctc:
            # CTC crossing selection with z-continuity.
            #
            # At each x we find ALL sign changes of (C - C_target) in the z scan
            # and classify them as outer (z >= z_min_local) or inner.
            #
            # Crossing preference (in order):
            #   1. If prev_z_ctc_mesh is set (not the first step), pick the outer
            #      crossing whose interpolated z is nearest to prev_z_ctc_mesh.
            #      This enforces path continuity and prevents branch-hopping.
            #   2. On the first step (no prev_z), take the first outer crossing
            #      (smallest z >= z_min, i.e. nearest to the trap minimum).
            #   3. If no outer crossings exist, fall back to the inner crossing
            #      nearest to prev_z (or z_min on the first step).
            #   4. If no crossings at all, snap to the z with C nearest to target.
            conf_v = np.where(valid_mask, conf, np.nan)
            conf_shifted = conf_v - _ctc_target_resolved

            # Collect ALL sign changes, classified as outer/inner
            outer_sign_changes: List[int] = []
            inner_sign_changes: List[int] = []
            for k in range(n_z - 1):
                if (np.isfinite(conf_shifted[k]) and np.isfinite(conf_shifted[k+1])
                        and conf_shifted[k] * conf_shifted[k+1] < 0):
                    if z_min_local is not None:
                        mid_z  = 0.5 * (z_scan_mesh[k] + z_scan_mesh[k+1])
                        if mid_z >= z_min_local:
                            outer_sign_changes.append(k)
                        else:
                            inner_sign_changes.append(k)
                    else:
                        outer_sign_changes.append(k)

            def _interp_z(k_idx: int) -> float:
                """Linearly interpolate the crossing z between scan points k and k+1."""
                c0, c1 = conf_shifted[k_idx], conf_shifted[k_idx + 1]
                t = -c0 / (c1 - c0)
                return z_scan_mesh[k_idx] + t * (z_scan_mesh[k_idx + 1] - z_scan_mesh[k_idx])

            no_exact_crossing = False
            ctc_crossing_side = "outer"

            if outer_sign_changes:
                if prev_z_ctc_mesh is not None:
                    # Prefer the outer crossing nearest to previous step's z
                    k = min(outer_sign_changes,
                            key=lambda ki: abs(_interp_z(ki) - prev_z_ctc_mesh))
                else:
                    # First step: take the first outer crossing (nearest to z_min)
                    k = outer_sign_changes[0]
            elif inner_sign_changes:
                ctc_crossing_side = "inner"
                anchor = prev_z_ctc_mesh if prev_z_ctc_mesh is not None else z_min_local
                if anchor is not None:
                    k = min(inner_sign_changes,
                            key=lambda ki: abs(_interp_z(ki) - anchor))
                else:
                    k = inner_sign_changes[0]
            else:
                # No crossing at all: nearest to target
                no_exact_crossing = True
                ctc_crossing_side = "nearest"
                k = int(np.nanargmin(np.abs(conf_shifted)))

            if not no_exact_crossing:
                c0, c1 = conf_shifted[k], conf_shifted[k + 1]
                t = -c0 / (c1 - c0)
                z_ctc_mesh = z_scan_mesh[k] + t * (z_scan_mesh[k + 1] - z_scan_mesh[k])
                phi_ctc_eV = float(
                    phi_eV[k] + t * (phi_eV[k + 1] - phi_eV[k])
                    if np.isfinite(phi_eV[k]) and np.isfinite(phi_eV[k + 1])
                    else phi_eV[k] if np.isfinite(phi_eV[k]) else phi_eV[k + 1]
                )
            else:
                z_ctc_mesh = z_scan_mesh[k]
                phi_ctc_eV = float(phi_eV[k]) if np.isfinite(phi_eV[k]) else float("nan")

            prev_z_ctc_mesh = z_ctc_mesh   # update continuity anchor

            trap_h_um = (z_ctc_mesh * coord_unit) * 1e6
            if no_exact_crossing:
                conf_at_ctc = float(conf[k]) if np.isfinite(conf[k]) else float("nan")
            else:
                conf_at_ctc = float(_ctc_target_resolved)
            ctc_path_rows.append({
                "x_m":                      float(xi * coord_unit),
                "y_m":                      float(y_fixed_mesh * coord_unit),
                "z_m":                      float(z_ctc_mesh * coord_unit),
                "phi_pp_eV":                round(phi_ctc_eV, 9),
                "total_conf_eV_per_m2":     round(conf_at_ctc, 3),
                "trap_height_um":           round(trap_h_um, 4),
                "path_type":                "ctc",
                "ctc_crossing_side":        ctc_crossing_side,
                "no_exact_ctc_crossing":    no_exact_crossing,
            })

        if (ix + 1) % 10 == 0 or ix == 0:
            print(f"[paper-ctc]   x step {ix+1}/{n_x}  "
                  f"x={xi*coord_unit*1e6:.1f} µm", flush=True)

    result: Dict[str, Any] = {}

    # ── Minimum path results ──────────────────────────────────────────────────
    if run_min and min_path_rows:
        phi_min_vals = [r["phi_pp_eV"] for r in min_path_rows if np.isfinite(r["phi_pp_eV"])]
        if phi_min_vals:
            # Barrier = max(phi_pp along path) - phi_pp(r0), measured from start.
            # Using max-min would include descent-below-start artifacts and overshoot.
            phi_r0_min    = min_path_rows[0]["phi_pp_eV"]
            barrier_min_eV = max(phi_min_vals) - phi_r0_min
        else:
            barrier_min_eV = float("nan")
        result["transport_min_path"]          = min_path_rows
        result["transport_min_barrier_eV"]    = round(barrier_min_eV, 8)
        result["transport_min_barrier_meV"]   = round(barrier_min_eV * 1e3, 6)
        print(f"[paper-ctc] min-path barrier = {barrier_min_eV*1e3:.4f} meV", flush=True)

    # ── CTC path results ──────────────────────────────────────────────────────
    if run_ctc and ctc_path_rows:
        phi_ctc_vals = [r["phi_pp_eV"] for r in ctc_path_rows if np.isfinite(r["phi_pp_eV"])]
        n_no_cross   = sum(1 for r in ctc_path_rows if r.get("no_exact_ctc_crossing"))
        if phi_ctc_vals:
            # Barrier = max(phi_pp along CTC path) - phi_pp(r0 on CTC path)
            phi_r0_ctc    = ctc_path_rows[0]["phi_pp_eV"]
            ctc_barrier_eV = max(phi_ctc_vals) - phi_r0_ctc
        else:
            ctc_barrier_eV = float("nan")
        result["transport_ctc_path"]           = ctc_path_rows
        result["ctc_barrier_eV"]               = round(ctc_barrier_eV, 8)
        result["ctc_barrier_meV"]              = round(ctc_barrier_eV * 1e3, 6)
        result["ctc_target_conf_eV_per_m2"]    = _ctc_target_resolved
        result["ctc_no_crossing_count"]        = n_no_cross
        n_inner = sum(1 for r in ctc_path_rows if r.get("ctc_crossing_side") == "inner")
        result["ctc_inner_crossing_count"]     = n_inner

        print(f"[paper-ctc] CTC barrier = {ctc_barrier_eV*1e3:.4f} meV  "
              f"no_crossing={n_no_cross}/{len(ctc_path_rows)}", flush=True)

        # ── Path overlap diagnostics ──────────────────────────────────────────
        if run_min and min_path_rows and len(min_path_rows) == len(ctc_path_rows):
            dz_arr = np.array([
                (c["z_m"] - m["z_m"]) * 1e6
                for c, m in zip(ctc_path_rows, min_path_rows)
            ])
            result["ctc_max_abs_dz_um"] = round(float(np.max(np.abs(dz_arr))), 4)
            result["ctc_rms_dz_um"]     = round(float(np.sqrt(np.mean(dz_arr**2))), 4)
            print(f"[paper-ctc] path overlap: max|dz|={result['ctc_max_abs_dz_um']:.2f} µm  "
                  f"rms_dz={result['ctc_rms_dz_um']:.2f} µm", flush=True)
        else:
            result["ctc_max_abs_dz_um"] = None
            result["ctc_rms_dz_um"]     = None

    # ── Write CSVs ────────────────────────────────────────────────────────────
    _min_fields = ["x_m", "y_m", "z_m", "phi_pp_eV",
                   "total_conf_eV_per_m2", "trap_height_um", "path_type"]
    _ctc_fields = _min_fields + ["ctc_crossing_side", "no_exact_ctc_crossing"]

    if csv_out_min is not None and min_path_rows:
        try:
            with csv_out_min.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=_min_fields, extrasaction="ignore")
                w.writeheader(); w.writerows(min_path_rows)
            print(f"[paper-ctc] min-path CSV → {csv_out_min}", flush=True)
        except OSError as exc:
            print(f"[paper-ctc] WARNING: could not write min CSV: {exc}", flush=True)

    if csv_out_ctc is not None and ctc_path_rows:
        try:
            with csv_out_ctc.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=_ctc_fields, extrasaction="ignore")
                w.writeheader(); w.writerows(ctc_path_rows)
            print(f"[paper-ctc] ctc-path CSV → {csv_out_ctc}", flush=True)
        except OSError as exc:
            print(f"[paper-ctc] WARNING: could not write ctc CSV: {exc}", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        def _extract(rows, key):
            return [r[key] for r in rows] if rows else []

        if min_path_rows:
            xs_min = [r["x_m"] * 1e6 for r in min_path_rows]
            axes[0].plot(xs_min, _extract(min_path_rows, "trap_height_um"),
                         label="min-Φ path", color="C0")
            axes[1].plot(xs_min, _extract(min_path_rows, "total_conf_eV_per_m2"),
                         label="min-Φ path", color="C0")
            axes[2].plot(xs_min, _extract(min_path_rows, "phi_pp_eV"),
                         label="min-Φ path", color="C0")
        if ctc_path_rows:
            xs_ctc = [r["x_m"] * 1e6 for r in ctc_path_rows]
            axes[0].plot(xs_ctc, _extract(ctc_path_rows, "trap_height_um"),
                         label="CTC path", color="C1", linestyle="--")
            axes[1].plot(xs_ctc, _extract(ctc_path_rows, "total_conf_eV_per_m2"),
                         label="CTC path", color="C1", linestyle="--")
            axes[2].plot(xs_ctc, _extract(ctc_path_rows, "phi_pp_eV"),
                         label="CTC path", color="C1", linestyle="--")
            axes[1].axhline(ctc_target_conf, color="grey", linestyle=":",
                            label=f"C_target={ctc_target_conf:.2e}")

        axes[0].set_ylabel("Trap height (µm)")
        axes[1].set_ylabel("Total conf C (eV/m²)")
        axes[2].set_ylabel("Φ_pp (eV)")
        axes[2].set_xlabel("x (µm)")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)
        axes[0].set_title("Paper CTC diagnostic — min-Φ vs CTC paths")
        fig.tight_layout()

        # Save alongside the CTC CSV if path available; else alongside min CSV
        _plot_base = csv_out_ctc or csv_out_min
        if _plot_base is not None:
            plot_path = _plot_base.with_suffix("").parent / (
                _plot_base.stem.replace("_ctc_path", "").replace("_min_path", "")
                + "_paper_ctc_diag.png"
            )
            fig.savefig(str(plot_path), dpi=120)
            print(f"[paper-ctc] diagnostic plot → {plot_path}", flush=True)
        plt.close(fig)
    except Exception as exc:
        print(f"[paper-ctc] plotting skipped: {exc}", flush=True)

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
    transport_mode: str = "both",
    n_nodes: int = 40,
    rf_tags: Optional[List[int]] = None,
    ground_tags: Optional[List[int]] = None,
    outer_tags: Optional[List[int]] = None,
    dc_voltages: Optional[Dict[int, float]] = None,
    # Paper CTC parameters
    ctc_n_x: int = 60,
    ctc_n_z: int = 200,
    ctc_z_min_m: Optional[float] = None,
    ctc_z_max_m: Optional[float] = None,
    ctc_target_conf: float = 0.75e9,
    ctc_target_auto: bool = False,
    ctc_hessian_step_um: float = 2.0,
    depth_correction: float = 1.0,
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
    if transport_mode == "ctc":
        _skip_key = "ctc_barrier_eV"
    elif transport_mode == "min":
        _skip_key = "transport_min_barrier_eV"
    elif transport_mode == "both":
        _skip_key = "ctc_barrier_eV"
    elif transport_mode == "string":
        _skip_key = "transport_barrier_ctc_eV"
    elif transport_mode == "legacy_both":
        _skip_key = "transport_barrier_ctc_eV"
    else:
        _skip_key = "transport_barrier_xscan_eV"
    existing_val = report.get(_skip_key)
    if existing_val is not None and not overwrite:
        # Use best available barrier for the skip summary
        best_barrier = report.get("transport_barrier_ctc_eV",
                                  report.get("transport_barrier_xscan_eV"))
        return {
            "status": "skipped", "case_dir": str(case_dir),
            "message": "already present (use --overwrite to re-run)",
            "barrier_eV": float(best_barrier) if best_barrier is not None else None,
            "peak_x_um": _um_or_none(report.get("transport_xscan_peak_x_m",
                                                  report.get("transport_ctc_peak_x_m"))),
            "junction_x_um": _um_or_none(report.get("transport_xscan_nearest_junction_x_m",
                                                      report.get("transport_ctc_junction_x_m"))),
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

    # ── Resolve DC voltages (CLI > report JSON > automation_config) ───────────
    _dc_voltages: Dict[int, float] = {}
    if dc_voltages:
        _dc_voltages = dc_voltages
    else:
        # Try report JSON
        _raw = (report.get("dc_voltages") or report.get("dc_voltage_map")
                or report.get("V_dc"))
        if _raw is None:
            _raw = (run_cfg.get("dc_voltages") or run_cfg.get("dc_voltage_map"))
        if isinstance(_raw, dict):
            _dc_voltages = {int(k): float(v) for k, v in _raw.items()}
        elif isinstance(_raw, list):
            _dc_tags_raw = (report.get("dc_tags") or run_cfg.get("dc_tags") or [])
            _dc_voltages = {int(t): float(v)
                            for t, v in zip(_dc_tags_raw, _raw) if abs(float(v)) > 1e-15}

    if _dc_voltages:
        print(f"[ctb] DC voltages: {_dc_voltages}", flush=True)
    else:
        print("[ctb] No DC voltages — computing RF-only barrier  "
              "(use --dc-voltages to add DC compensation)", flush=True)

    t0 = time.perf_counter()
    try:
        domain, facet_tags, phi_rf, used_ckpt = _resolve_phi_rf(
            case_dir, report,
            rf_tags=_rf_tags, ground_tags=_ground_tags, outer_tags=_outer_tags,
            comm=comm,
        )
        Psi = _build_psi(phi_rf, report)

        # ── Build effective potential Φ_eff = Ψ_RF + q × Σ V_i × φ_DC_i ─────
        if _dc_voltages:
            degree = int(report.get("degree", 2))
            dc_tags_list = list(_dc_voltages.keys())
            phi_dc_map = _resolve_phi_dc_all(
                case_dir, report, domain, facet_tags,
                dc_tags=dc_tags_list,
                rf_tags=_rf_tags,
                ground_tags=_ground_tags,
                outer_tags=_outer_tags,
                degree=degree,
                comm=comm,
            )
            Phi_eff = _build_effective_potential(Psi, phi_dc_map, _dc_voltages, report)
        else:
            Phi_eff = Psi  # RF-only fallback

        # CSV sidecar lives next to the JSON report
        csv_sidecar = report_path.with_name(
            report_path.stem + "_transport_scan.csv"
        ) if report_path is not None else None

        transport: Dict[str, Any] = {}

        # ── Legacy modes: xscan, string, both (xscan+string) ─────────────────
        if transport_mode in ("xscan", "legacy_both"):
            transport = _scan_transport_barrier(
                Phi_eff, report,
                junction_pitch=junction_pitch,
                n_steps=n_steps,
                scan_both_axes=scan_both_axes,
                optim_method=optim_method,
                comm=comm,
                csv_out=csv_sidecar,
            )

        if transport_mode in ("string", "legacy_both"):
            ctc_csv_sidecar = report_path.with_name(
                report_path.stem + "_ctc_path.csv"
            ) if report_path is not None else None
            ctc = _ctc_string_method(
                Phi_eff, report,
                junction_pitch=junction_pitch,
                n_nodes=n_nodes,
                comm=comm,
                csv_out=ctc_csv_sidecar,
            )
            transport.update(ctc)

        if transport_mode == "legacy_both":
            xscan_eV = transport.get("transport_barrier_xscan_eV") or 0.0
            ctc_eV   = transport.get("transport_barrier_ctc_eV")   or 0.0
            if ctc_eV > xscan_eV * 1.1:
                print(
                    f"[ctb] WARNING: CTC barrier ({ctc_eV*1e3:.2f} meV) > xscan barrier "
                    f"({xscan_eV*1e3:.2f} meV). The string method may not have converged "
                    f"or the endpoint is pinned incorrectly.",
                    flush=True,
                )

        # ── Paper modes: min, ctc, both (min+ctc) ────────────────────────────
        if transport_mode in ("min", "ctc", "both"):
            _run_min = transport_mode in ("min", "both")
            _run_ctc = transport_mode in ("ctc", "both")
            _csv_min = (report_path.with_name(report_path.stem + "_min_path.csv")
                        if report_path is not None and _run_min else None)
            _csv_ctc = (report_path.with_name(report_path.stem + "_ctc_path.csv")
                        if report_path is not None and _run_ctc else None)
            paper_result = _paper_ctc_and_min_scan(
                Phi_eff, report,
                junction_pitch=junction_pitch,
                n_x=ctc_n_x,
                ctc_z_min_m=ctc_z_min_m,
                ctc_z_max_m=ctc_z_max_m,
                n_z=ctc_n_z,
                ctc_target_conf=ctc_target_conf,
                ctc_target_auto=ctc_target_auto,
                hessian_step_um=ctc_hessian_step_um,
                run_min=_run_min,
                run_ctc=_run_ctc,
                comm=comm,
                csv_out_min=_csv_min,
                csv_out_ctc=_csv_ctc,
            )
            transport.update(paper_result)
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t0

    # ── Apply depth correction factor (accounts for mesh resolution vs. COMSOL) ──
    if depth_correction != 1.0:
        for raw_key, corr_key in [
            ("transport_min_barrier_eV",  "transport_min_barrier_eV_corrected"),
            ("ctc_barrier_eV",            "ctc_barrier_eV_corrected"),
            ("transport_min_barrier_meV", "transport_min_barrier_meV_corrected"),
            ("ctc_barrier_meV",           "ctc_barrier_meV_corrected"),
        ]:
            raw_val = transport.get(raw_key)
            if raw_val is not None:
                prec = 8 if raw_key.endswith("_eV") else 6
                transport[corr_key] = round(float(raw_val) * depth_correction, prec)

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

    # Pick the primary barrier: min-path > paper CTC > string CTC > xscan
    barrier_eV = (transport.get("transport_min_barrier_eV")
                  or transport.get("ctc_barrier_eV")
                  or transport.get("transport_barrier_ctc_eV")
                  or transport.get("transport_barrier_xscan_eV", 0.0))
    peak_x_m   = transport.get("transport_xscan_peak_x_m",
                               transport.get("transport_ctc_peak_x_m"))
    junc_x_m   = transport.get("transport_xscan_nearest_junction_x_m",
                               transport.get("transport_ctc_junction_x_m"))
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
    ap.add_argument("--transport-mode",
                    choices=["min", "ctc", "both",
                             "xscan", "string", "legacy_both"],
                    default="both",
                    help=(
                        "Paper modes (2025 method): "
                        "  min  — pseudopotential-minimum path at y=0; "
                        "  ctc  — constant-total-confinement path at y=0; "
                        "  both — min + ctc (default, matches 2025 paper). "
                        "Legacy modes: "
                        "  xscan — coordinate-descent x-scan; "
                        "  string — string-method CTC path; "
                        "  legacy_both — xscan + string."
                    ))
    ap.add_argument("--n-nodes", type=int, default=40, dest="n_nodes",
                    help="Number of string nodes for string-method CTC path.")

    # ── Paper CTC parameters ──────────────────────────────────────────────────
    ap.add_argument("--ctc-target-conf", type=float, default=0.75e9,
                    metavar="EV_PER_M2",
                    help="Target total confinement C = ∇²Φ_pp in eV/m² for the CTC path "
                         "(default 0.75e9, matching the 2025 3D-printed micro junction paper).")
    ap.add_argument("--ctc-target-auto", action="store_true",
                    help="Auto-calibrate C_target to C(z_min) at the first x step (r0). "
                         "Ensures the CTC passes through the linear-region trap minimum; "
                         "recommended when geometry confinement differs from the paper.")
    ap.add_argument("--ctc-axis", type=str, default="x", choices=["x"],
                    help="Transport axis (fixed at x for now).")
    ap.add_argument("--ctc-fixed-y", type=float, default=0.0, metavar="M",
                    help="Fixed y coordinate in metres for CTC and min-path scans (default 0).")
    ap.add_argument("--ctc-z-min", type=float, default=None, metavar="M",
                    help="Lower z bound in metres for z-scan. Defaults to mesh z floor + margin.")
    ap.add_argument("--ctc-z-max", type=float, default=None, metavar="M",
                    help="Upper z bound in metres for z-scan. Defaults to mesh z ceiling − margin.")
    ap.add_argument("--ctc-n-z", type=int, default=200,
                    help="Number of z sample points per x step (default 200).")
    ap.add_argument("--ctc-n-x", type=int, default=60,
                    help="Number of x sample points for CTC/min scan (default 60, same as --n-steps).")
    ap.add_argument("--ctc-hessian-step-um", type=float, default=2.0, metavar="UM",
                    help="Finite-difference step size in µm for Hessian Laplacian (default 2.0 µm).")
    ap.add_argument("--ctc-conf-units", type=str, default="eV_per_m2",
                    choices=["eV_per_m2"],
                    help="Units for confinement C (only eV_per_m2 supported).")

    # Electrode tags (optional — auto-detected from automation_config.json otherwise)
    ap.add_argument("--rf-tags", type=int, nargs="+", default=None,
                    help="RF electrode facet tags. Auto-detected from automation_config.json "
                         "if not supplied.")
    ap.add_argument("--ground-tags", type=int, nargs="+", default=None,
                    help="Ground electrode facet tags.")
    ap.add_argument("--outer-tags", type=int, nargs="+", default=None,
                    help="Outer (Neumann) boundary tags.")
    ap.add_argument(
        "--depth-correction", type=float, default=1.0, metavar="FACTOR",
        help="Multiply all transport barrier values by this factor before writing to JSON. "
             "Use 1.3 to correct for FEniCS mesh resolution underestimate vs. COMSOL "
             "(same systematic error as trap depth). Default: 1.0 (no correction).",
    )
    ap.add_argument(
        "--dc-voltages", type=str, default=None, metavar="JSON",
        help='DC electrode voltage map as a JSON object, e.g. \'{"2": 1.5}\'. '
             'Keys are facet tag integers, values are applied voltages in Volts. '
             'The script solves a unit-voltage Laplace problem for each DC tag, '
             'then adds q × V × φ_DC to the RF pseudopotential. '
             'Overrides any values found in the report JSON or automation_config.json. '
             'Example: --dc-voltages \'{"2": -1.2}\' to apply -1.2 V to DC tag 2.',
    )

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

    dc_voltages_override: Optional[Dict[int, float]] = None
    if args.dc_voltages:
        try:
            raw = json.loads(args.dc_voltages)
            dc_voltages_override = {int(k): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            print(f"[main] ERROR: could not parse --dc-voltages '{args.dc_voltages}': {exc}")
            sys.exit(1)

    shared_kwargs = dict(
        junction_pitch       = args.junction_pitch,
        n_steps              = args.n_steps,
        scan_both_axes       = args.scan_both_axes,
        optim_method         = args.optim_method,
        overwrite            = args.overwrite,
        dry_run              = args.dry_run,
        output_mode          = args.output_mode,
        transport_mode       = args.transport_mode,
        n_nodes              = args.n_nodes,
        rf_tags              = args.rf_tags,
        ground_tags          = args.ground_tags,
        outer_tags           = args.outer_tags,
        dc_voltages          = dc_voltages_override,
        ctc_n_x              = args.ctc_n_x,
        ctc_n_z              = args.ctc_n_z,
        ctc_z_min_m          = args.ctc_z_min,
        ctc_z_max_m          = args.ctc_z_max,
        ctc_target_conf      = args.ctc_target_conf,
        ctc_target_auto      = args.ctc_target_auto,
        ctc_hessian_step_um  = args.ctc_hessian_step_um,
        depth_correction     = args.depth_correction,
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
