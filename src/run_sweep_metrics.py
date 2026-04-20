#!/usr/bin/env python3
"""run_sweep_metrics.py  (dolfinx 0.10.x)

Lightweight sweep-oriented entry point.  Physics pipeline is identical to
run_case.py (same bounds detection, same minimum-finder, same Hessian step,
same fast transport).  Only the output is lighter: no XDMF, no debug z-scan,
no CTC diagnostics, one-line terminal summary, compact JSON record.

Design rule: every step that touches the physics (Laplace solve, pseudopotential,
r0 search bounds, find_minimum_cg1, secular_frequencies, depth rays) is a
direct copy of the validated run_case.py logic.  Do not diverge.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
from mpi4py import MPI

import metrics
from run_case import load_case_mesh, solve_laplace_tagged, compute_post_r0_metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep-mode trap metrics — fast transport, compact JSON output."
    )
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="sweep")
    ap.add_argument("--rf-tags", type=int, nargs="+", required=True)
    ap.add_argument("--ground-tags", type=int, nargs="+", required=True)
    ap.add_argument(
        "--outer-tags", type=int, nargs="*", default=[4],
        help="Facet tags for the outer (far-field) Neumann boundary (default: [4]).",
    )
    ap.add_argument(
        "--degree", type=int, default=2,
        help="FEM polynomial degree (default 2 = CG2 for accurate secular frequencies).",
    )
    ap.add_argument("--rf-freq", type=float, default=40e6)
    ap.add_argument("--mass-amu", type=float, default=40.0)
    ap.add_argument("--charge-e", type=float, default=1.0)
    ap.add_argument("--vrf", type=float, default=1.0,
                    help="RF voltage amplitude in volts (default 1 = normalised).")
    ap.add_argument("--depth-nrays", type=int, default=48)
    # ── r0 search bounds (same semantics as run_case.py) ────────────────────
    ap.add_argument("--r0-x-min", type=float, default=None)
    ap.add_argument("--r0-x-max", type=float, default=None)
    ap.add_argument("--r0-y-min", type=float, default=None)
    ap.add_argument("--r0-y-max", type=float, default=None)
    ap.add_argument("--r0-z-min", type=float, default=None)
    ap.add_argument("--r0-z-max", type=float, default=None)
    ap.add_argument(
        "--r0-search-margin", type=float, default=5.0e-5,
        help="Physical margin in metres for z_max auto-detect (default 5e-5 m = 50 µm). "
             "For 3D blade/rail traps (electrode top > this margin in metres): margin is "
             "SUBTRACTED from electrode top to exclude the electrode-tip saddle-point artifact. "
             "For surface traps (electrode top ≈ 0): margin is added above the electrode. "
             "Override entirely with explicit --r0-z-max (e.g. --r0-z-max 0.12 for mm-unit "
             "mesh with ~82 µm expected trap height).",
    )
    ap.add_argument(
        "--r0-x-auto", action="store_true",
        help="Auto-detect x bounds from RF electrode geometry (RF x-extent centre ±25%%). "
             "Same as run_case.py --r0-x-auto.  Use for multi-junction meshes when "
             "explicit --r0-x-min/max are not provided.",
    )
    ap.add_argument(
        "--coord-unit", type=float, default=None,
        help="Mesh coordinate unit in metres, e.g. 1e-3 for mm (default: auto-detect).",
    )
    args = ap.parse_args()

    # ── Load mesh ────────────────────────────────────────────────────────────
    domain, facet_tags, _cell_tags = load_case_mesh(args.mesh)
    comm = domain.comm
    rank = comm.rank

    if facet_tags is None:
        raise RuntimeError("facet_tags is None — mesh must include facet markers.")

    coords = domain.geometry.x
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_span = float(np.linalg.norm(bbox_max - bbox_min))
    h_mesh = metrics._estimate_cell_h(domain)

    # ── Degree warning (same as run_case.py) ─────────────────────────────────
    if args.degree == 1 and rank == 0:
        warnings.warn(
            "[degree=1] CG1 elements give piecewise-constant gradients; secular "
            "frequencies from the Hessian will be 10–300× too low on typical meshes. "
            "Use --degree 2 for accurate secular frequencies.",
            UserWarning, stacklevel=2,
        )

    # ── Auto-scale h and ray_length (identical logic to run_case.py) ─────────
    _h_multiplier = 4.0 if args.degree == 1 else 3.0
    h = 2e-6     # sentinel: will be replaced if mesh is not in SI metres
    ray_length = 200e-6

    if h_mesh > 1e-2:
        h = h_mesh * _h_multiplier
        ray_length = bbox_span

    # ── Coordinate unit auto-detect (same heuristic as run_case.py) ──────────
    coord_unit = args.coord_unit
    if coord_unit is None:
        if bbox_span > 100.0:
            coord_unit = 1e-6
        elif bbox_span > 0.1:
            coord_unit = 1e-3
        else:
            coord_unit = 1.0

    if rank == 0:
        print(f"[sweep] mesh={args.mesh.name}  coord_unit={coord_unit:.0e}  "
              f"h={h:.3e}  ray_length={ray_length:.3e}  h_mesh={h_mesh:.3e}")

    # ── Boundary conditions ──────────────────────────────────────────────────
    outer_tags = set(args.outer_tags) if args.outer_tags else set()
    bad_gnd = set(args.ground_tags) & outer_tags
    if bad_gnd and rank == 0:
        warnings.warn(
            f"[BC] Tag(s) {sorted(bad_gnd)} appear in both --ground-tags and "
            "--outer-tags. Outer-boundary tags are Neumann — remove them from "
            "--ground-tags.",
            UserWarning,
        )

    gnd_electrode_tags = [t for t in args.ground_tags if t not in outer_tags]

    if rank == 0:
        print(f"[BC] RF Dirichlet tags : {args.rf_tags}  (φ = 1 V)")
        print(f"[BC] GND Dirichlet tags: {gnd_electrode_tags}  (φ = 0 V)")
        print(f"[BC] Neumann (outer)   : {sorted(outer_tags)}  (∂φ/∂n = 0)")

    bc_map_rf: Dict[int, float] = {tag: 1.0 for tag in args.rf_tags}
    bc_map_rf.update({tag: 0.0 for tag in gnd_electrode_tags})

    # ── Laplace solve ────────────────────────────────────────────────────────
    phi_rf = solve_laplace_tagged(
        domain, facet_tags, bc_map_rf,
        degree=args.degree, petsc_prefix=f"{args.prefix}_rf_",
    )
    phi_rf.name = "phi_rf"

    # ── RF pseudopotential (unclipped — same as run_case.py) ─────────────────
    e_charge = 1.602176634e-19
    amu = 1.66053906660e-27
    q = args.charge_e * e_charge
    m = args.mass_amu * amu

    Psi = metrics.compute_rf_pseudopotential(
        phi_rf, omega_rf=2.0 * np.pi * args.rf_freq, q_C=q, m_kg=m,
        degree=args.degree,
    )
    Psi.name = "Psi_rf"

    # ── r0 search bounds — identical logic to run_case.py ────────────────────
    # IMPORTANT: keep this section a verbatim mirror of run_case.py so that
    # both scripts always find the same trap minimum.
    r0_z_max = args.r0_z_max
    r0_z_min = args.r0_z_min
    rf_nodes = None   # reused for x/y auto-detect

    # z_max: auto-detect from RF electrode top + margin
    if r0_z_max is None and domain.topology.dim == 3:
        try:
            tdim_local = domain.topology.dim
            fdim_local = tdim_local - 1
            domain.topology.create_connectivity(fdim_local, 0)
            f2v = domain.topology.connectivity(fdim_local, 0)
            rf_facet_indices = np.concatenate(
                [facet_tags.find(t) for t in args.rf_tags]
            )
            rf_nodes = np.unique(
                np.concatenate([f2v.links(int(fi)) for fi in rf_facet_indices])
            )
            z_electrode_top = float(domain.geometry.x[rf_nodes, 2].max())
            margin_mesh = args.r0_search_margin / coord_unit   # metres → mesh units
            # 3D blade/rail traps: ion is trapped INSIDE the electrode structure
            # (ion height << electrode_top).  The electrode-tip region just below
            # the blade top has a saddle-point artifact where |∇φ|→0 from the
            # geometry transition.  Adding margin above pushes z_max further into
            # this artifact zone.  For 3D traps we must SUBTRACT the margin to
            # restrict the search below the electrode top.
            # Surface traps: electrode near z=0, ion above → add margin.
            # Heuristic: electrode_top in metres > r0_search_margin → 3D trap.
            _elec_top_m = z_electrode_top * coord_unit  # physical metres
            if _elec_top_m > args.r0_search_margin:
                r0_z_max = z_electrode_top - margin_mesh  # 3D trap: exclude tip artifact
                _margin_dir = "below electrode top"
            else:
                r0_z_max = z_electrode_top + margin_mesh  # surface trap: above electrode
                _margin_dir = "above electrode top"
            if rank == 0:
                print(f"[r0 search] z_electrode_top={z_electrode_top:.4g} mesh units "
                      f"({_elec_top_m*1e6:.1f} µm)  margin={margin_mesh:.4g} ({_margin_dir})"
                      f"  → z_max={r0_z_max:.4g} mesh units")
        except Exception as _e:
            if rank == 0:
                print(f"[r0 search] z_max auto-detect failed ({_e}); no z bound applied.")

    # z_min: default to 0.0 (DC surface) if not user-specified
    if r0_z_min is None and domain.topology.dim == 3:
        r0_z_min = 0.0
        if rank == 0:
            print("[r0 search] z_min auto-set to 0.0 (DC surface). "
                  "Override with --r0-z-min if your trap sits below z=0.")

    # x/y bounds: explicit flags take priority; --r0-x-auto fills x if not supplied
    r0_x_min = args.r0_x_min
    r0_x_max = args.r0_x_max
    r0_y_min = args.r0_y_min
    r0_y_max = args.r0_y_max

    if args.r0_x_auto and r0_x_min is None and r0_x_max is None:
        try:
            if rf_nodes is None:
                fdim_local = domain.topology.dim - 1
                domain.topology.create_connectivity(fdim_local, 0)
                f2v = domain.topology.connectivity(fdim_local, 0)
                rf_facet_indices = np.concatenate(
                    [facet_tags.find(t) for t in args.rf_tags]
                )
                rf_nodes = np.unique(
                    np.concatenate([f2v.links(int(fi)) for fi in rf_facet_indices])
                )
            x_rf = domain.geometry.x[rf_nodes, 0]
            x_rf_min = float(x_rf.min())
            x_rf_max = float(x_rf.max())
            x_rf_mid = (x_rf_min + x_rf_max) / 2.0
            x_rf_half = (x_rf_max - x_rf_min) / 4.0   # ±25% of full span
            r0_x_min = x_rf_mid - x_rf_half
            r0_x_max = x_rf_mid + x_rf_half
            if rank == 0:
                print(f"[r0 search] x_auto: RF x=[{x_rf_min:.4g}, {x_rf_max:.4g}], "
                      f"centre±25%=[{r0_x_min:.4g}, {r0_x_max:.4g}] mesh units")
                print(f"[r0 search] hint: override with "
                      f"--r0-x-min {r0_x_min:.4g} --r0-x-max {r0_x_max:.4g}")
        except Exception as _e:
            if rank == 0:
                print(f"[r0 search] x_auto failed ({_e}); no x bound applied.")

    if rank == 0:
        print(f"[r0 search] active bounds (mesh units): "
              f"x=[{r0_x_min},{r0_x_max}]  "
              f"y=[{r0_y_min},{r0_y_max}]  "
              f"z=[{r0_z_min},{r0_z_max}]")

    # ── Debug: z-distribution of low-Ψ interior DOFs in the search window ───
    # Print where the bottom-5% Ψ cluster sits in z BEFORE the minimum finder
    # runs.  A cluster near z_electrode_top indicates an electrode-tip artifact;
    # a cluster near the expected trap height indicates a healthy search domain.
    if rank == 0:
        try:
            _V = Psi.function_space
            _n_owned = _V.dofmap.index_map.size_local
            _raw_coords = _V.tabulate_dof_coordinates().reshape(-1, 3)
            _vals = Psi.x.array[:_n_owned]
            _z_all = _raw_coords[:_n_owned, 2]
            _zmask = (
                (_vals >= 0)
                & (_z_all >= (r0_z_min if r0_z_min is not None else -np.inf))
                & (_z_all <= (r0_z_max if r0_z_max is not None else np.inf))
            )
            if _zmask.any():
                _z_cand = _z_all[_zmask]
                _v_cand = _vals[_zmask]
                _p5 = float(np.percentile(_v_cand, 5))
                _low_z = _z_cand[_v_cand <= _p5]
                _pct = np.percentile(_low_z, [5, 25, 50, 75, 95])
                print(f"[r0 debug] z of bottom-5%% Ψ DOFs in window "
                      f"[{r0_z_min},{r0_z_max}] mesh units:")
                print(f"  z percentiles [5,25,50,75,95] = "
                      f"{[f'{v:.4g}' for v in _pct]} mesh units")
                print(f"  → in µm: {[f'{v*coord_unit*1e6:.1f}' for v in _pct]}")
                print(f"  (target: cluster near expected trap height "
                      f"~{82:.0f} µm = {82e-6/coord_unit:.4g} mesh units)")
        except Exception as _de:
            print(f"[r0 debug] z-distribution check failed: {_de}")

    # ── Find trap minimum ────────────────────────────────────────────────────
    mininfo = metrics.find_minimum_cg1(
        Psi, comm=comm,
        x_min=r0_x_min, x_max=r0_x_max,
        y_min=r0_y_min, y_max=r0_y_max,
        z_min=r0_z_min, z_max=r0_z_max,
    )

    _r0_SI_coarse = np.array(mininfo.r_min) * coord_unit

    if rank == 0:
        print(f"[trap min] r0 (coarse)={mininfo.r_min.tolist()}, Psi_min={mininfo.psi_min:.4e} J")
        print(f"[trap min] r0_SI (coarse, pre-refinement)={_r0_SI_coarse.tolist()} m")

    # Hard-reject minimum below DC surface
    if mininfo.r_min.shape[0] >= 3 and float(mininfo.r_min[2]) < 0.0:
        raise RuntimeError(
            f"[trap min] r0.z = {float(mininfo.r_min[2]):.4g} mesh units — "
            "minimum below DC surface (z < 0). "
            "Ensure --r0-z-min >= 0 and --r0-z-max is tight enough. "
            f"Current bounds: z=[{r0_z_min}, {r0_z_max}]"
        )

    # Warn if r0.z is suspiciously close to z_max (far-field drift)
    if rank == 0 and r0_z_max is not None and mininfo.r_min.shape[0] >= 3:
        z_r0 = float(mininfo.r_min[2])
        if z_r0 > 0.85 * r0_z_max:
            warnings.warn(
                f"[trap min] r0.z={z_r0:.4g} is within 15% of z_max={r0_z_max:.4g} — "
                "centroid may have drifted into the Neumann far-field vacuum. "
                f"Try --r0-z-max {z_r0 * 0.25:.3g}  (aim ~1.5× expected trap height).",
                UserWarning, stacklevel=2,
            )

    # ── Post-r0 metrics via the shared canonical helper ──────────────────────
    # compute_post_r0_metrics lives in run_case.py and is the single authoritative
    # implementation used by both scripts.  Calling it here guarantees identical
    # physics, scaling, and conversion factors — no divergence possible.
    if rank == 0:
        print(f"[consistency] charge_C       = {q:.6e} C  (charge_e={args.charge_e})")
    post = compute_post_r0_metrics(
        Psi, m_kg=m, r0=mininfo.r_min, h=h,
        coord_unit=coord_unit, vrf=args.vrf,
        ray_length=ray_length, nrays=args.depth_nrays,
        transport_mode="fast",
        compute_depth=True,
        comm=comm,
    )
    sec   = post["sec"]
    depth = post["depth"]
    hessian_status = post.get("hessian_status", "unknown")

    # Use the extra-refined r0 (updated inside compute_post_r0_metrics) for
    # all output quantities — it may differ slightly from mininfo.r_min.
    r0_final = np.array(post.get("r0_refined", mininfo.r_min.tolist()), dtype=np.float64)
    r0_SI    = r0_final * coord_unit

    # ── Compute strong-mode frequencies (top 2 by magnitude) ────────────────
    # Sort all positive-eigenvalue frequencies descending and take the top two.
    # This excludes the weak axial/transport mode (lowest frequency) so that
    # strong_freq_min/max always refer to the two radial confinement modes.
    if rank == 0:
        freqs   = sec["freq_hz"]
        eigvals = np.array(sec["eigvals"])

        all_pos = sorted(
            [float(f) for f, ev in zip(freqs, eigvals) if ev > 0],
            reverse=True,   # descending: largest first
        )
        top2          = all_pos[:2]                             # two strongest modes
        strong_fmin   = float(min(top2)) if len(top2) >= 2 else (top2[0] if top2 else float("nan"))
        strong_fmax   = float(max(top2)) if top2 else float("nan")
        n_strong      = len(all_pos)                            # total positive modes
        strong_modes_ok = n_strong >= 2

        z_um = float(r0_final[2]) * coord_unit * 1e6 if r0_final.shape[0] >= 3 else float("nan")
        physical_min_ok = (
            r0_final.shape[0] >= 3
            and float(r0_final[2]) > 0.0
            and (r0_z_max is None or float(r0_final[2]) < 0.85 * r0_z_max)
            and hessian_status in ("valid", "borderline_numeric")
        )

        print(f"[sweep] h_used={sec['h']:.3e}  all_freq_hz={[f'{f:.4e}' for f in freqs]}")
        print(f"[sweep result]  r0.z={z_um:.2f} µm  "
              f"strong_freqs=[{strong_fmin/1e6:.3f}, {strong_fmax/1e6:.3f}] MHz  "
              f"radial_depth={depth.get('radial_depth_core_eV')} eV  "
              f"depth_z={depth.get('depth_z_eV')} eV (paper-comparable)  "
              f"depth_y={depth.get('depth_y_eV')} eV  "
              f"xscan_barrier={depth.get('transport_barrier_xscan_eV')} eV")

    # ── Build compact JSON record ────────────────────────────────────────────
    if rank == 0:
        record = {
            "mesh": str(args.mesh),
            "prefix": args.prefix,
            "r0_x_m": float(r0_SI[0]) if len(r0_SI) > 0 else None,
            "r0_y_m": float(r0_SI[1]) if len(r0_SI) > 1 else None,
            "r0_z_m": float(r0_SI[2]) if len(r0_SI) > 2 else None,
            "freq1_hz": float(freqs[0]) if len(freqs) > 0 else None,
            "freq2_hz": float(freqs[1]) if len(freqs) > 1 else None,
            "freq3_hz": float(freqs[2]) if len(freqs) > 2 else None,
            # top-2 radial confinement modes (axial excluded)
            "strong_freq_min_hz": strong_fmin,
            "strong_freq_max_hz": strong_fmax,
            # ── Axis-specific depth (paper-comparable; no electrode-hit exclusion) ──
            "depth_z_eV": depth.get("depth_z_eV"),          # min(+z, -z) barrier
            "depth_z_plus_eV": depth.get("depth_z_plus_eV"),
            "depth_z_minus_eV": depth.get("depth_z_minus_eV"),
            "depth_z_interior": depth.get("depth_z_interior"),
            "depth_y_eV": depth.get("depth_y_eV"),          # min(+y, -y) barrier
            "depth_y_plus_eV": depth.get("depth_y_plus_eV"),
            "depth_y_minus_eV": depth.get("depth_y_minus_eV"),
            "depth_y_interior": depth.get("depth_y_interior"),
            # ── Fibonacci-sphere radial depth (electrode-hit rays excluded) ──
            "radial_depth_core_eV": depth.get("radial_depth_core_eV"),
            "transport_barrier_xscan_eV": depth.get("transport_barrier_xscan_eV"),
            "transport_barrier_eigvec_eV": depth.get("transport_barrier_eigvec_eV"),
            "h_used": float(sec["h"]),
            # status flags for sweep filtering
            "physical_min_ok": physical_min_ok,
            "strong_modes_ok": strong_modes_ok,
            # Hessian validity: "valid" | "borderline_numeric" | "invalid_saddle"
            # "borderline_numeric" means a tiny negative eigenvalue within the
            # numerical-noise tolerance (|λ_neg|/max(λ_pos) < 1e-3); results are
            # accepted but flagged.  "invalid_saddle" never appears here because
            # compute_post_r0_metrics raises RuntimeError before reaching this code.
            "hessian_status": hessian_status,
            "success": True,
            "notes": None,
            # sweep parameters for traceability
            "rf_freq_Hz": args.rf_freq,
            "vrf_V": args.vrf,
            "mass_amu": args.mass_amu,
            "charge_e": args.charge_e,
            "degree": args.degree,
            "coord_unit_m_per_mesh": coord_unit,
            "r0_search_bounds_mesh": {
                "x": [r0_x_min, r0_x_max],
                "y": [r0_y_min, r0_y_max],
                "z": [r0_z_min, r0_z_max],
            },
        }

        args.outdir.mkdir(parents=True, exist_ok=True)
        out_path = args.outdir / f"{args.prefix}_sweep.json"
        out_path.write_text(json.dumps(record, indent=2))
        print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
