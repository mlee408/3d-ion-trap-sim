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

        # Create facet entities before reading meshtags
        tdim = domain.topology.dim
        domain.topology.create_entities(tdim - 1)

        # Cell tags live in mesh.xdmf; try the attribute name written by meshio.
        cell_tags = None
        for name in ("cell_tags", "cells", "ct", "cell_markers", "name_to_read"):
            try:
                cell_tags = xdmf.read_meshtags(domain, name=name)
                break
            except Exception:
                pass

    # Facet tags are written to a *separate* facets.xdmf file by load_msh /
    # meshio.  They are never stored inside mesh.xdmf, so we must open the
    # companion file explicitly.
    facet_tags = None
    facets_path = mesh_path.parent / "facets.xdmf"
    if facets_path.exists():
        try:
            with XDMFFile(comm, str(facets_path), "r") as xdmf:
                facet_tags = xdmf.read_meshtags(domain, name="Grid")
        except Exception:
            pass

    domain.topology.create_connectivity(tdim - 1, tdim)
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
    # Physical unit parameters
    ap.add_argument(
        "--vrf", type=float, default=1.0,
        help="RF electrode voltage amplitude in volts (default 1 = normalised). "
             "Scale Ψ and secular frequencies to physical values."
    )
    ap.add_argument(
        "--coord-unit", type=float, default=None,
        help="Mesh coordinate unit expressed in metres, e.g. 1e-6 for µm, "
             "1e-3 for mm (default: auto-detected from bounding box)."
    )
    args = ap.parse_args()

    domain, facet_tags, _cell_tags = load_case_mesh(args.mesh)
    comm = domain.comm
    rank = comm.rank

    if facet_tags is None:
        raise RuntimeError("facet_tags is None. Your mesh must include facet markers/tags.")

    # ── Mesh diagnostics ────────────────────────────────────────────────────
    h_mesh = metrics._estimate_cell_h(domain)
    coords = domain.geometry.x
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_span = float(np.linalg.norm(bbox_max - bbox_min))
    if rank == 0:
        print(f"[mesh] tdim={domain.topology.dim}, gdim={domain.geometry.dim}")
        print(f"[mesh] bbox min={bbox_min.tolist()}, max={bbox_max.tolist()}")
        print(f"[mesh] estimated cell size h_mesh={h_mesh:.4e}")

    # Auto-scale h and ray_length when the user left the defaults (2e-6 m / 200e-6 m)
    # but the mesh is clearly not in SI metres.
    h = args.h
    ray_length = args.depth_ray_length
    if h_mesh > 1e-2:          # mesh appears to be in mm, cm, or larger units
        if args.h == 2e-6:
            h = h_mesh * 2.0   # must span multiple CG1 cells to get non-zero Hessian
        if args.depth_ray_length == 200e-6:
            ray_length = h_mesh * 20.0
        if rank == 0:
            print(f"[auto-scale] h={h:.4e}, ray_length={ray_length:.4e} "
                  f"(mesh unit ~{h_mesh:.4e})")

    # Detect or use the specified coordinate unit (metres per mesh unit).
    coord_unit = args.coord_unit
    if coord_unit is None:
        # Heuristic: if the bounding-box span is > 1 mm, assume µm; if > 1 m, warn.
        if bbox_span > 1.0:          # likely µm
            coord_unit = 1e-6
        elif bbox_span > 1e-3:       # likely mm
            coord_unit = 1e-3
        else:                        # assume metres
            coord_unit = 1.0
        if rank == 0:
            print(f"[auto-detect] coord_unit={coord_unit:.0e} m/mesh_unit "
                  f"(bbox span={bbox_span:.3g} mesh units)")
    # ────────────────────────────────────────────────────────────────────────

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
    if rank == 0:
        print(f"[trap min] r0={mininfo.r_min.tolist()}, Psi_min={mininfo.psi_min:.4e} J")

    sec = metrics.secular_frequencies_from_pseudopotential(
        Psi, m_kg=m, r0=mininfo.r_min, h=h, comm=comm,
        coord_scale=coord_unit, v_rf=args.vrf,
    )

    depth = None
    if not args.no_depth:
        depth = metrics.estimate_trap_depth_by_rays(
            Psi, r0=mininfo.r_min, ray_length=ray_length, nrays=args.depth_nrays, comm=comm
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
        "h_used": h,
        "ray_length_used": ray_length,
        "h_mesh_estimate": h_mesh,
        "coord_unit_m_per_mesh": coord_unit,
        "vrf_V": args.vrf,
        "r0_SI_m": (np.array(mininfo.r_min) * coord_unit).tolist(),
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
