#!/usr/bin/env python3
"""plot_field_slices.py — COMSOL-style slice plots for a trap design.

Solves the electrostatic problem ∇²φ = 0 with φ=1 V on RF facets and φ=0 V on
grounded facets (Neumann ∂φ/∂n=0 on the outer boundary), then samples three
fields on the xy, xz, and yz planes through a chosen point (default: the RF
null):

    φ      potential                       (V)
    |E|    field magnitude  = |∇φ|         (V/m)
    Ψ      RF pseudopotential               (eV)

One PNG per plane is written to --outdir, each with three colour-mapped panels.

Example
-------
    python scripts/plot_field_slices.py \
        --mesh runs/sweep_n2_j2_transport_v3/case_0003/mesh_xdmf/mesh.xdmf \
        --rf-tags 1 2 --ground-tags 3 \
        --outdir runs/baseline/field_slices/

The script is MPI-aware: launch with mpirun -n N for parallel solves. Plotting
runs only on rank 0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import laplace          # noqa: E402
import metrics          # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Mesh loading — same convention as src/run_case.py
# ─────────────────────────────────────────────────────────────────────────────
def load_mesh_xdmf(mesh_path: Path):
    comm = MPI.COMM_WORLD
    with XDMFFile(comm, str(mesh_path), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        tdim = domain.topology.dim
        domain.topology.create_entities(tdim - 1)

    facet_tags = None
    facets_path = mesh_path.parent / "facets.xdmf"
    if facets_path.exists():
        with XDMFFile(comm, str(facets_path), "r") as xdmf:
            facet_tags = xdmf.read_meshtags(domain, name="Grid")

    domain.topology.create_connectivity(tdim - 1, tdim)
    return domain, facet_tags


# ─────────────────────────────────────────────────────────────────────────────
# Field construction
# ─────────────────────────────────────────────────────────────────────────────
def project_field_magnitude(phi: fem.Function, *, degree: int = 1) -> fem.Function:
    """Return a CG scalar Function holding |∇φ| (mesh-unit⁻¹)."""
    V = fem.functionspace(phi.function_space.mesh, ("CG", degree))
    expr = ufl.sqrt(ufl.dot(ufl.grad(phi), ufl.grad(phi)) + 1e-300)
    Emag = metrics.project(expr, V, prefix="Emag_")
    Emag.name = "E_mag"
    return Emag


# ─────────────────────────────────────────────────────────────────────────────
# Slice sampling
# ─────────────────────────────────────────────────────────────────────────────
PLANE_AXES = {
    "xy": (0, 1, 2),   # vary x,y; hold z = slice_centre[2]
    "xz": (0, 2, 1),   # vary x,z; hold y
    "yz": (1, 2, 0),   # vary y,z; hold x
}


def build_plane_grid(
    plane: str,
    centre: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_u: int,
    n_v: int,
    pad_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (U, V, pts) where U,V are 2-D mesh-units grids and pts is (N,3)."""
    u_axis, v_axis, fixed_axis = PLANE_AXES[plane]
    u_lo, u_hi = bbox_min[u_axis], bbox_max[u_axis]
    v_lo, v_hi = bbox_min[v_axis], bbox_max[v_axis]
    if pad_frac:
        u_lo += pad_frac * (u_hi - u_lo); u_hi -= pad_frac * (u_hi - u_lo)
        v_lo += pad_frac * (v_hi - v_lo); v_hi -= pad_frac * (v_hi - v_lo)
    u = np.linspace(u_lo, u_hi, n_u)
    v = np.linspace(v_lo, v_hi, n_v)
    U, V = np.meshgrid(u, v, indexing="xy")
    pts = np.empty((U.size, 3), dtype=np.float64)
    pts[:, u_axis] = U.ravel()
    pts[:, v_axis] = V.ravel()
    pts[:, fixed_axis] = centre[fixed_axis]
    return U, V, pts


def sample_on_plane(f: fem.Function, pts: np.ndarray, shape, comm) -> np.ndarray:
    """Evaluate f at pts (MPI-aware); return masked 2-D array (NaN outside mesh)."""
    vals = metrics.eval_function_at_points(f, pts, comm=comm)
    return vals.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────
def _plane_axis_labels(plane: str) -> Tuple[str, str]:
    return {"xy": ("x (µm)", "y (µm)"),
            "xz": ("x (µm)", "z (µm)"),
            "yz": ("y (µm)", "z (µm)")}[plane]


def _robust_clim(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


def plot_plane(
    plane: str,
    U_um: np.ndarray, V_um: np.ndarray,
    phi: np.ndarray, Emag_Vpm: np.ndarray, Psi_eV: np.ndarray,
    centre_um: np.ndarray,
    out_path: Path,
    vrf: float,
    log_E: bool = True,
    log_Psi: bool = True,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    xlabel, ylabel = _plane_axis_labels(plane)
    fixed_axis = PLANE_AXES[plane][2]
    fixed_label = "xyz"[fixed_axis]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    fig.suptitle(
        f"Trap electrostatics on the {plane}-plane through "
        f"{fixed_label} = {centre_um[fixed_axis]:.2f} µm  "
        f"(V_RF = {vrf:g} V applied to RF electrodes)",
        fontsize=12,
    )

    # --- φ panel -------------------------------------------------------------
    ax = axes[0]
    pcm = ax.pcolormesh(U_um, V_um, phi * vrf, cmap="RdBu_r",
                        vmin=-vrf, vmax=vrf, shading="auto")
    ax.set_title("Potential  φ  (V)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_aspect("equal")
    cs = ax.contour(U_um, V_um, phi * vrf, levels=11,
                    colors="k", linewidths=0.4, alpha=0.5)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)

    # --- |E| panel -----------------------------------------------------------
    ax = axes[1]
    if log_E:
        lo, hi = _robust_clim(Emag_Vpm, 1.0, 99.5)
        lo = max(lo, hi * 1e-6)
        norm = LogNorm(vmin=lo, vmax=hi)
    else:
        norm = Normalize(*_robust_clim(Emag_Vpm))
    pcm = ax.pcolormesh(U_um, V_um, Emag_Vpm, cmap="inferno",
                        norm=norm, shading="auto")
    ax.set_title("Field magnitude  |E| = |∇φ|  (V/m)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)

    # --- Ψ panel -------------------------------------------------------------
    ax = axes[2]
    if log_Psi:
        lo, hi = _robust_clim(Psi_eV, 1.0, 99.5)
        lo = max(lo, hi * 1e-6)
        norm = LogNorm(vmin=lo, vmax=hi)
    else:
        norm = Normalize(*_robust_clim(Psi_eV))
    pcm = ax.pcolormesh(U_um, V_um, Psi_eV, cmap="viridis",
                        norm=norm, shading="auto")
    ax.set_title("RF pseudopotential  Ψ  (eV)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", type=Path, required=True,
                    help="Path to mesh.xdmf (a facets.xdmf sibling must exist).")
    ap.add_argument("--rf-tags", type=int, nargs="+", required=True,
                    help="Facet tag IDs for RF electrodes (φ=1 V).")
    ap.add_argument("--ground-tags", type=int, nargs="+", required=True,
                    help="Facet tag IDs for grounded electrodes (φ=0 V).")
    ap.add_argument("--outer-tags", type=int, nargs="*", default=[4],
                    help="Outer-boundary tags receiving Neumann BC.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--degree", type=int, default=2,
                    help="CG polynomial degree (2 recommended; CG1 gives "
                         "piecewise-constant |E| per element).")

    ap.add_argument("--rf-freq", type=float, default=40e6, help="RF frequency (Hz).")
    ap.add_argument("--mass-amu", type=float, default=40.0)
    ap.add_argument("--charge-e", type=float, default=1.0)
    ap.add_argument("--vrf", type=float, default=1.0,
                    help="RF amplitude scaling for the φ / Ψ plots (volts).")
    ap.add_argument("--coord-unit", type=float, default=None,
                    help="Metres per mesh unit (auto-detected if omitted).")

    ap.add_argument("--slice-x", type=float, default=None,
                    help="x of slice intersection (mesh units). Default: RF-null.")
    ap.add_argument("--slice-y", type=float, default=None,
                    help="y of slice intersection (mesh units). Default: RF-null.")
    ap.add_argument("--slice-z", type=float, default=None,
                    help="z of slice intersection (mesh units). Default: RF-null.")
    ap.add_argument("--n-grid", type=int, default=240,
                    help="Samples per axis on each slice plane.")
    ap.add_argument("--pad-frac", type=float, default=0.02,
                    help="Fractional bbox padding removed from each plane edge.")
    ap.add_argument("--planes", type=str, nargs="+",
                    default=["xy", "xz", "yz"], choices=["xy", "xz", "yz"])

    ap.add_argument("--linear-E", action="store_true",
                    help="Use linear (not log) scale for |E|.")
    ap.add_argument("--linear-Psi", action="store_true",
                    help="Use linear (not log) scale for Ψ.")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ── 1. Load mesh ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    domain, facet_tags = load_mesh_xdmf(args.mesh)
    if facet_tags is None:
        if rank == 0:
            print(f"[error] facets.xdmf not found next to {args.mesh}", file=sys.stderr)
        return 2

    coords = domain.geometry.x
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_span = float(np.linalg.norm(bbox_max - bbox_min))

    coord_unit = args.coord_unit
    if coord_unit is None:
        if bbox_span > 100.0:
            coord_unit = 1e-6
        elif bbox_span > 0.1:
            coord_unit = 1e-3
        else:
            coord_unit = 1.0
    if rank == 0:
        print(f"[mesh] tdim={domain.topology.dim} bbox=[{bbox_min.tolist()}, "
              f"{bbox_max.tolist()}] coord_unit={coord_unit:.0e} m/mesh_unit")

    # ── 2. Solve Laplace (RF=1, GND=0) ──────────────────────────────────────
    outer = set(args.outer_tags or [])
    gnd = [t for t in args.ground_tags if t not in outer]
    bc_map: Dict[int, float] = {t: 1.0 for t in args.rf_tags}
    bc_map.update({t: 0.0 for t in gnd})
    if rank == 0:
        print(f"[BC] RF={args.rf_tags}  GND={gnd}  Neumann={sorted(outer)}")

    res = laplace.solve_laplace_tagged(
        domain, facet_tags, bc_map,
        degree=args.degree, petsc_options_prefix="slices_phi_",
    )
    phi = res.phi
    if rank == 0:
        print(f"[solve] φ range=[{phi.x.array.min():.4f}, {phi.x.array.max():.4f}]"
              f"  ({time.perf_counter()-t0:.2f}s)")

    # ── 3. Build |E| and Ψ FEM fields ───────────────────────────────────────
    Emag = project_field_magnitude(phi, degree=args.degree)

    e_charge = 1.602176634e-19
    amu = 1.66053906660e-27
    q = args.charge_e * e_charge
    m = args.mass_amu * amu
    Psi = metrics.compute_rf_pseudopotential(
        phi, omega_rf=2.0 * np.pi * args.rf_freq, q_C=q, m_kg=m,
        degree=args.degree,
    )

    # ── 4. Slice centre — default to RF null ────────────────────────────────
    centre = np.array([
        0.5 * (bbox_min[0] + bbox_max[0]),
        0.5 * (bbox_min[1] + bbox_max[1]),
        0.5 * (bbox_min[2] + bbox_max[2]),
    ], dtype=np.float64)

    user_centre = [args.slice_x, args.slice_y, args.slice_z]
    if any(c is None for c in user_centre):
        try:
            mininfo = metrics.find_minimum_cg1(
                Psi, comm=comm,
                z_min=0.0 if domain.topology.dim == 3 else None,
            )
            for i in range(3):
                if user_centre[i] is None:
                    centre[i] = float(mininfo.r_min[i]) if i < mininfo.r_min.shape[0] else centre[i]
            if rank == 0:
                print(f"[slice] RF-null estimate r0={mininfo.r_min.tolist()} mesh units")
        except Exception as ex:  # noqa: BLE001
            if rank == 0:
                print(f"[slice] RF-null search failed ({ex}); using bbox centre.")
    for i, c in enumerate(user_centre):
        if c is not None:
            centre[i] = float(c)
    if rank == 0:
        print(f"[slice] slice intersection (mesh units) = {centre.tolist()}")
        print(f"[slice] slice intersection (µm)         = "
              f"{(centre * coord_unit * 1e6).tolist()}")

    # ── 5. Per-plane sample + plot ──────────────────────────────────────────
    args.outdir.mkdir(parents=True, exist_ok=True)
    centre_um = centre * coord_unit * 1e6

    for plane in args.planes:
        if rank == 0:
            print(f"[plane {plane}] sampling on a {args.n_grid}×{args.n_grid} grid...")
        U, V, pts = build_plane_grid(
            plane, centre, bbox_min, bbox_max,
            n_u=args.n_grid, n_v=args.n_grid, pad_frac=args.pad_frac,
        )

        phi_grid = sample_on_plane(phi, pts, U.shape, comm)
        Emag_grid_mu = sample_on_plane(Emag, pts, U.shape, comm)   # 1/mesh_unit
        Psi_grid_J = sample_on_plane(Psi, pts, U.shape, comm)      # J / V²

        # Convert to physical SI / eV with V_RF scaling.
        # φ is dimensionless (0..1) for unit RF; multiply by V_RF in plot.
        Emag_Vpm = Emag_grid_mu * args.vrf / coord_unit              # V/m
        Psi_eV   = Psi_grid_J * (args.vrf ** 2) / (coord_unit ** 2) / e_charge

        if rank != 0:
            continue

        U_um = U * coord_unit * 1e6
        V_um = V * coord_unit * 1e6
        out_path = args.outdir / f"field_slice_{plane}.png"
        plot_plane(
            plane, U_um, V_um,
            phi_grid, Emag_Vpm, Psi_eV,
            centre_um=centre_um,
            out_path=out_path,
            vrf=args.vrf,
            log_E=not args.linear_E,
            log_Psi=not args.linear_Psi,
        )
        print(f"[plane {plane}] wrote {out_path}")

    if rank == 0:
        print(f"[done] total {time.perf_counter()-t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
