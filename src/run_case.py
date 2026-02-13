#!/usr/bin/env python3
"""
run_case.py  (dolfinx 0.10.x)

End-to-end single-case runner:
  mesh -> Laplace solve(s) -> RF pseudopotential -> trap metrics -> outputs

This script is intentionally "batteries included": it can use your laplace.py + mesh_io.py
if their APIs match, but it also includes a fallback Laplace solver so you can run immediately.

Typical usage (example):
  mpirun -n 4 python run_case.py \
    --mesh outputs/toy/mesh.xdmf \
    --rf-tags 10 \
    --ground-tags 1 2 3 \
    --rf-freq 40e6 \
    --mass-amu 40 \
    --outdir outputs/toy

Notes:
- You must provide facet tag integers for RF and ground (Dirichlet) boundaries.
- If you want multiple electrode basis solutions, pass --basis-tags TAG1 TAG2 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl

# ---- Your project modules ----
# metrics_improved.py is the updated version I produced for you.
# Rename it to metrics.py in your repo if you prefer, and update this import accordingly.
from metrics_improved import (
    compute_rf_pseudopotential,
    find_minimum_cg1,
    secular_frequencies_from_pseudopotential,
    estimate_trap_depth_by_rays,
)

# Optional imports: use your mesh_io / laplace if available.
try:
    import mesh_io  # type: ignore
except Exception:
    mesh_io = None

try:
    import laplace  # type: ignore
except Exception:
    laplace = None


def _load_mesh_xdmf(mesh_path: Path):
    """
    Minimal XDMF loader fallback.

    Expects an XDMF containing:
      - mesh topology/geometry
      - optional MeshTags named 'facet_tags' (or any single tags set on facets)
      - optional MeshTags named 'cell_tags'  (not required here)
    """
    comm = MPI.COMM_WORLD
    with XDMFFile(comm, str(mesh_path), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        # Try common tag names; tolerate missing tags.
        facet_tags = None
        cell_tags = None
        for name in ("facet_tags", "facets", "ft", "facet_markers"):
            try:
                facet_tags = xdmf.read_meshtags(domain, name=name)
                break
            except Exception:
                pass
        for name in ("cell_tags", "cells", "ct", "cell_markers"):
            try:
                cell_tags = xdmf.read_meshtags(domain, name=name)
                break
            except Exception:
                pass
    return domain, facet_tags, cell_tags


def load_case_mesh(mesh_path: Path):
    """
    Uses your mesh_io.py if it provides a loader; otherwise falls back to XDMFFile.
    """
    if mesh_io is not None:
        # Try a few likely function names without forcing you to match one exact API.
        for fname in ("load_mesh", "read_mesh", "load_xdmf", "read_xdmf"):
            if hasattr(mesh_io, fname):
                obj = getattr(mesh_io, fname)(str(mesh_path))
                # Accept common return patterns.
                if isinstance(obj, tuple) and len(obj) >= 2:
                    # (domain, facet_tags, cell_tags) or similar
                    domain = obj[0]
                    facet_tags = obj[1]
                    cell_tags = obj[2] if len(obj) > 2 else None
                    return domain, facet_tags, cell_tags
    # Fallback
    return _load_mesh_xdmf(mesh_path)


def build_dirichlet_bcs(
    V: fem.FunctionSpace,
    facet_tags,
    *,
    tag_to_value: Dict[int, float],
) -> List[fem.DirichletBC]:
    """
    Build Dirichlet BCs on facet tag sets.
    """
    if facet_tags is None:
        raise RuntimeError(
            "facet_tags is None. Your mesh must include facet markers/tags for electrode boundaries."
        )

    bcs: List[fem.DirichletBC] = []
    tdim = V.mesh.topology.dim
    fdim = tdim - 1

    for tag, value in tag_to_value.items():
        facets = facet_tags.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bc_val = fem.Constant(V.mesh, np.array(value, dtype=np.float64))
        bcs.append(fem.dirichletbc(bc_val, dofs, V))
    return bcs


def solve_laplace_dirichlet(
    V: fem.FunctionSpace,
    bcs: List[fem.DirichletBC],
    *,
    petsc_prefix: str = "lap_",
    petsc_options: Optional[Dict[str, str]] = None,
) -> fem.Function:
    """
    Fallback Laplace solver:
      -Δφ = 0 with Dirichlet BCs.
    """
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-10",
        }

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(V.mesh, np.array(0.0, dtype=np.float64)) * v * ufl.dx

    from dolfinx.fem.petsc import LinearProblem

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=petsc_prefix,
        petsc_options=petsc_options,
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


def solve_laplace_with_project_module(
    V: fem.FunctionSpace,
    bcs: List[fem.DirichletBC],
    *,
    petsc_prefix: str,
) -> fem.Function:
    """
    Attempt to use your laplace.py if it exposes a compatible solver;
    otherwise use fallback.
    """
    if laplace is not None:
        for fname in ("solve_laplace", "solve", "solve_dirichlet", "solve_laplace_dirichlet"):
            if hasattr(laplace, fname):
                try:
                    phi = getattr(laplace, fname)(V, bcs=bcs, prefix=petsc_prefix)
                    if isinstance(phi, fem.Function):
                        return phi
                except TypeError:
                    # Signature mismatch; try next.
                    pass
                except Exception:
                    # If your solver exists but errors, fall back so you can still run.
                    break
    return solve_laplace_dirichlet(V, bcs, petsc_prefix=petsc_prefix)


def write_xdmf(out_path: Path, domain, fields: Sequence[fem.Function]):
    comm = domain.comm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(comm, str(out_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in fields:
            xdmf.write_function(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True, help="Path to mesh XDMF.")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    ap.add_argument("--rf-tags", type=int, nargs="+", required=True, help="Facet tag(s) at +1V RF.")
    ap.add_argument("--ground-tags", type=int, nargs="+", required=True, help="Facet tag(s) at 0V ground.")
    ap.add_argument("--basis-tags", type=int, nargs="*", default=[], help="Optional additional electrode tags to solve as 1V basis.")
    ap.add_argument("--degree", type=int, default=1, help="CG degree for phi/Psi output.")
    ap.add_argument("--rf-freq", type=float, default=40e6, help="RF frequency (Hz).")
    ap.add_argument("--mass-amu", type=float, default=40.0, help="Ion mass in atomic mass units.")
    ap.add_argument("--charge-e", type=float, default=1.0, help="Ion charge in units of elementary charge.")
    ap.add_argument("--h", type=float, default=2e-6, help="Step size for Hessian finite difference (m).")
    ap.add_argument("--depth-ray-length", type=float, default=200e-6, help="Ray length for depth estimation (m).")
    ap.add_argument("--depth-nrays", type=int, default=48, help="Number of rays for depth estimation.")
    ap.add_argument("--no-depth", action="store_true", help="Skip trap depth estimation.")
    ap.add_argument("--prefix", type=str, default="case", help="Name prefix for outputs.")

    args = ap.parse_args()

    domain, facet_tags, cell_tags = load_case_mesh(args.mesh)
    comm = domain.comm
    rank = comm.rank

    # Function space for Laplace potential
    V = fem.functionspace(domain, ("CG", args.degree))

    # ---- RF solve: set RF facets to 1V, ground facets to 0V ----
    tag_to_value = {tag: 1.0 for tag in args.rf_tags}
    tag_to_value.update({tag: 0.0 for tag in args.ground_tags})
    bcs_rf = build_dirichlet_bcs(V, facet_tags, tag_to_value=tag_to_value)

    phi_rf = solve_laplace_with_project_module(V, bcs_rf, petsc_prefix=f"{args.prefix}_rf_")
    phi_rf.name = "phi_rf"

    # ---- Optional: additional basis electrode solves ----
    basis_fields: List[fem.Function] = []
    for tag in args.basis_tags:
        tag_to_value_b = {tag: 1.0}
        # Keep ground at 0 if provided
        for gt in args.ground_tags:
            tag_to_value_b[gt] = 0.0
        bcs_b = build_dirichlet_bcs(V, facet_tags, tag_to_value=tag_to_value_b)
        phi_b = solve_laplace_with_project_module(V, bcs_b, petsc_prefix=f"{args.prefix}_b{tag}_")
        phi_b.name = f"phi_basis_{tag}"
        basis_fields.append(phi_b)

    # ---- Compute RF pseudopotential ----
    # Ion parameters
    e = 1.602176634e-19
    amu = 1.66053906660e-27
    q = args.charge_e * e
    m = args.mass_amu * amu

    Psi = compute_rf_pseudopotential(phi_rf, omega_rf=2.0 * np.pi * args.rf_freq, q_C=q, m_kg=m)
    Psi.name = "Psi_rf"

    # ---- Trap minimum ----
    mininfo = find_minimum_cg1(Psi, comm=comm)

    # ---- Secular frequencies (from Hessian at r0) ----
    sec = secular_frequencies_from_pseudopotential(Psi, m_kg=m, r0=mininfo.r_min, h=args.h, comm=comm)

    # ---- Trap depth (optional) ----
    depth = None
    if not args.no_depth:
        depth = estimate_trap_depth_by_rays(
            Psi,
            r0=mininfo.r_min,
            ray_length=args.depth_ray_length,
            nrays=args.depth_nrays,
            comm=comm,
        )

    # ---- Write outputs ----
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Fields XDMF
    xdmf_path = outdir / f"{args.prefix}_fields.xdmf"
    write_xdmf(xdmf_path, domain, [phi_rf, Psi, *basis_fields])

    # Metrics JSON (rank 0 writes)
    report = {
        "mesh": str(args.mesh),
        "rf_tags": args.rf_tags,
        "ground_tags": args.ground_tags,
        "rf_freq_Hz": args.rf_freq,
        "mass_amu": args.mass_amu,
        "charge_e": args.charge_e,
        "trap_min": {
            "r0_m": mininfo.r_min.tolist(),
            "Psi_min_J": float(mininfo.psi_min),
            "rank_found": int(mininfo.rank),
            "dof_index": int(mininfo.dof_index),
        },
        "secular": sec,  # should already be JSON-serializable dict from metrics_improved
        "depth": depth,  # None or dict
    }

    if rank == 0:
        (outdir / f"{args.prefix}_report.json").write_text(json.dumps(report, indent=2))
        print("\n=== Trap report ===")
        print(json.dumps(report, indent=2))
        print(f"\nWrote: {xdmf_path}")
        print(f"Wrote: {outdir / f'{args.prefix}_report.json'}")


if __name__ == "__main__":
    main()
