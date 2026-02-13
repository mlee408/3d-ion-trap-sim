#!/usr/bin/env python3
"""run_case.py  (dolfinx 0.10.x)

End-to-end single-case runner:
  mesh -> Laplace solve(s) -> RF pseudopotential -> trap metrics -> outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl

import metrics

try:
    import mesh_io  # type: ignore
except Exception:
    mesh_io = None

try:
    import laplace  # type: ignore
except Exception:
    laplace = None


def _load_mesh_xdmf(mesh_path: Path):
    comm = MPI.COMM_WORLD
    with XDMFFile(comm, str(mesh_path), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        facet_tags = None
        cell_tags = None
        for name in ("facet_tags", "facets", "ft", "facet_markers", "name_to_read"):
            try:
                facet_tags = xdmf.read_meshtags(domain, name=name)
                break
            except Exception:
                pass
        for name in ("cell_tags", "cells", "ct", "cell_markers", "name_to_read"):
            try:
                cell_tags = xdmf.read_meshtags(domain, name=name)
                break
            except Exception:
                pass
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    return domain, facet_tags, cell_tags


def load_case_mesh(mesh_path: Path):
    if mesh_io is not None:
        for fname in ("load_mesh", "load_xdmf", "read_xdmf", "load_msh"):
            if hasattr(mesh_io, fname):
                obj = getattr(mesh_io, fname)(str(mesh_path))
                if hasattr(obj, "domain"):
                    return obj.domain, obj.facet_tags, obj.cell_tags
                if isinstance(obj, tuple) and len(obj) >= 2:
                    domain = obj[0]
                    facet_tags = obj[1]
                    cell_tags = obj[2] if len(obj) > 2 else None
                    return domain, facet_tags, cell_tags
    return _load_mesh_xdmf(mesh_path)


def solve_laplace_tagged(
    domain,
    facet_tags,
    boundary_values: Dict[int, float],
    *,
    degree: int,
    petsc_prefix: str,
) -> fem.Function:
    if laplace is not None and hasattr(laplace, "solve_laplace_tagged"):
        res = laplace.solve_laplace_tagged(
            domain,
            facet_tags,
            boundary_values,
            degree=degree,
            petsc_options_prefix=petsc_prefix,
        )
        return res.phi

    # fallback
    V = fem.functionspace(domain, ("CG", degree))
    tdim = domain.topology.dim
    fdim = tdim - 1
    bcs: List[fem.DirichletBC] = []
    for tag, value in boundary_values.items():
        facets = facet_tags.find(tag)
        if facets.size == 0:
            continue
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bcs.append(fem.dirichletbc(np.array(value, dtype=np.float64), dofs, V))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, np.array(0.0, dtype=np.float64)) * v * ufl.dx

    from dolfinx.fem.petsc import LinearProblem
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=petsc_prefix,
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
    )
    phi = problem.solve()
    phi.name = "phi"
    return phi


def write_xdmf(out_path: Path, domain, fields: Sequence[fem.Function]):
    comm = domain.comm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(comm, str(out_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in fields:
            xdmf.write_function(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--rf-tags", type=int, nargs="+", required=True)
    ap.add_argument("--ground-tags", type=int, nargs="+", required=True)
    ap.add_argument("--basis-tags", type=int, nargs="*", default=[])
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--rf-freq", type=float, default=40e6)
    ap.add_argument("--mass-amu", type=float, default=40.0)
    ap.add_argument("--charge-e", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=2e-6)
    ap.add_argument("--depth-ray-length", type=float, default=200e-6)
    ap.add_argument("--depth-nrays", type=int, default=48)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--prefix", type=str, default="case")
    args = ap.parse_args()

    domain, facet_tags, _cell_tags = load_case_mesh(args.mesh)
    comm = domain.comm
    rank = comm.rank

    if facet_tags is None:
        raise RuntimeError("facet_tags is None. Your mesh must include facet markers/tags.")

    bc_map_rf: Dict[int, float] = {tag: 1.0 for tag in args.rf_tags}
    bc_map_rf.update({tag: 0.0 for tag in args.ground_tags})

    phi_rf = solve_laplace_tagged(
        domain, facet_tags, bc_map_rf, degree=args.degree, petsc_prefix=f"{args.prefix}_rf_"
    )
    phi_rf.name = "phi_rf"

    basis_fields: List[fem.Function] = []
    for tag in args.basis_tags:
        bc_map_b: Dict[int, float] = {tag: 1.0}
        for gt in args.ground_tags:
            bc_map_b[gt] = 0.0
        phi_b = solve_laplace_tagged(
            domain, facet_tags, bc_map_b, degree=args.degree, petsc_prefix=f"{args.prefix}_b{tag}_"
        )
        phi_b.name = f"phi_basis_{tag}"
        basis_fields.append(phi_b)

    e = 1.602176634e-19
    amu = 1.66053906660e-27
    q = args.charge_e * e
    m = args.mass_amu * amu

    Psi = metrics.compute_rf_pseudopotential(phi_rf, omega_rf=2.0 * np.pi * args.rf_freq, q_C=q, m_kg=m)
    Psi.name = "Psi_rf"

    mininfo = metrics.find_minimum_cg1(Psi, comm=comm)
    sec = metrics.secular_frequencies_from_pseudopotential(Psi, m_kg=m, r0=mininfo.r_min, h=args.h, comm=comm)

    depth = None
    if not args.no_depth:
        depth = metrics.estimate_trap_depth_by_rays(
            Psi, r0=mininfo.r_min, ray_length=args.depth_ray_length, nrays=args.depth_nrays, comm=comm
        )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / f"{args.prefix}_fields.xdmf"
    write_xdmf(xdmf_path, domain, [phi_rf, Psi, *basis_fields])

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
        "secular": sec,
        "depth": depth,
    }

    if rank == 0:
        (outdir / f"{args.prefix}_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"Wrote: {xdmf_path}")


if __name__ == "__main__":
    main()
