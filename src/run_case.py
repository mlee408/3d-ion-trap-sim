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
    """Write fields to XDMF, interpolating higher-degree functions to CG1.

    DOLFINx XDMFFile requires functions to be in the same polynomial space as
    the mesh geometry (CG1 for a standard linear mesh).  When --degree 2 is
    used the Laplace solution and pseudopotential are in CG2; we interpolate
    them down to CG1 purely for visualisation — the computation (Hessian,
    depth) has already been performed on the full-degree fields.
    """
    comm = domain.comm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    V1 = fem.functionspace(domain, ("CG", 1))
    with XDMFFile(comm, str(out_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in fields:
            # Check if the function is already CG1 (degree 1).
            # Attribute path differs across DOLFINx versions; fall back to
            # always interpolating if introspection is unavailable.
            try:
                deg = f.function_space.element.basix_element.degree
            except AttributeError:
                deg = None   # unknown — interpolate to be safe
            if deg == 1:
                xdmf.write_function(f)
            else:
                f_out = fem.Function(V1)
                f_out.interpolate(f)
                f_out.name = f.name
                xdmf.write_function(f_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--rf-tags", type=int, nargs="+", required=True)
    ap.add_argument("--ground-tags", type=int, nargs="+", required=True)
    ap.add_argument("--basis-tags", type=int, nargs="*", default=[])
    ap.add_argument(
        "--degree", type=int, default=1,
        help="FEM polynomial degree for the Laplace solve (default 1 = CG1). "
             "IMPORTANT: CG1 gives piecewise-*constant* gradients, so the "
             "pseudopotential Ψ = |∇φ|² is piecewise-constant per element.  "
             "Its Hessian (needed for secular frequencies) can only be resolved "
             "if the mesh is extremely fine near the RF null — in practice the "
             "computed frequencies are 10–300× too low with CG1 on typical meshes.  "
             "Use --degree 2 (CG2, piecewise-*linear* gradients) for accurate "
             "secular frequencies: the Laplace solve is ~4–8× more expensive but "
             "the Hessian is accurate on the same mesh.  Always use --degree 2 "
             "for production runs."
    )
    ap.add_argument("--rf-freq", type=float, default=40e6)
    ap.add_argument("--mass-amu", type=float, default=40.0)
    ap.add_argument("--charge-e", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=2e-6)
    ap.add_argument("--depth-ray-length", type=float, default=200e-6)
    ap.add_argument("--depth-nrays", type=int, default=48)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument(
        "--transport-mode", type=str, default="fast", choices=["fast", "full"],
        help="Transport barrier evaluation mode. 'fast' (default): x-scan + eigvec only, "
             "no verbose ray listings. 'full': adds CTC height-following path scan and "
             "full diagnostic output. Use 'full' for validation runs only.",
    )
    ap.add_argument("--prefix", type=str, default="case")
    # Minimum-finder spatial bounds (mesh units).
    # With Neumann outer BC the pseudopotential decays smoothly to zero far
    # from the electrodes, creating a large near-zero region that confuses
    # the cluster-centroid finder.  Restricting the search to a thin slab
    # above the electrode surface fixes this.
    ap.add_argument(
        "--r0-z-max", type=float, default=None,
        help="Upper z bound for the RF-null search (mesh units). "
             "Default: auto-detected as z_electrode_top + --r0-search-margin."
    )
    ap.add_argument(
        "--r0-z-min", type=float, default=None,
        help="Lower z bound for the RF-null search (mesh units, default: no bound)."
    )
    ap.add_argument(
        "--r0-search-margin", type=float, default=5.0e-5,
        help="Physical margin (metres) added above the RF electrode top when "
             "auto-setting --r0-z-max (default 5.0e-5 m = 50 µm). "
             "For 3D pillar traps the ion sits BELOW the RF pillar tops, so "
             "this margin is intentionally small to exclude the Neumann far-field "
             "vacuum region above the pillars, which contains many near-zero Ψ "
             "DOFs that would otherwise drag the cluster centroid away from the "
             "true RF null.  If the auto-detected minimum looks wrong, override "
             "with --r0-z-max directly (e.g. --r0-z-max 0.12 for a mm-unit mesh "
             "with ~82 µm trap height)."
    )
    # ── Lateral (x/y) search bounds ─────────────────────────────────────────
    # For multi-junction meshes the RF null is diffuse across the entire array.
    # Restricting x (and optionally y) to a single arm keeps the 5%-low-Ψ
    # cluster centroid inside the intended linear region or junction pocket.
    #
    # Example for a 2-junction 0.6 mm-pitch mesh (junctions near x=0 & x=0.6):
    #   Right outer arm:   --r0-x-min 0.65 --r0-x-max 1.05
    #   Left outer arm:    --r0-x-max -0.05
    #   Inter-junction:    --r0-x-min 0.05 --r0-x-max 0.55
    #
    # Auto-detect (--r0-x-auto): computes x_electrode_centre ± 0.25 × pitch
    # for the linear-region midpoint between junctions, and prints the bounds
    # so you can override them with explicit flags on the next run.
    ap.add_argument(
        "--r0-x-min", type=float, default=None,
        help="Lower x bound for the RF-null search (mesh units). "
             "Default: no bound (all x)."
    )
    ap.add_argument(
        "--r0-x-max", type=float, default=None,
        help="Upper x bound for the RF-null search (mesh units). "
             "Default: no bound."
    )
    ap.add_argument(
        "--r0-y-min", type=float, default=None,
        help="Lower y bound for the RF-null search (mesh units). "
             "Default: no bound."
    )
    ap.add_argument(
        "--r0-y-max", type=float, default=None,
        help="Upper y bound for the RF-null search (mesh units). "
             "Default: no bound."
    )
    ap.add_argument(
        "--r0-x-auto", action="store_true",
        help="Auto-detect x bounds for the RF-null search from the RF electrode "
             "geometry.  Computes the RF surface x-extent, then restricts the "
             "search to the central half of that extent (electrode_x_mid ± 0.25 × span). "
             "Useful for 2-junction runs without explicit --r0-x-min/max.  "
             "Printed bounds can be copied as explicit flags for subsequent runs."
    )
    ap.add_argument(
        "--outer-tags", type=int, nargs="*", default=[4],
        help="Facet tag IDs that are the outer (far-field) boundary. "
             "These receive Neumann BC (∂φ/∂n = 0) — do NOT include them in "
             "--ground-tags. Default: [4]."
    )
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

    # ── Degree warning ──────────────────────────────────────────────────────
    # CG1 (degree=1) gives piecewise-constant gradients → pseudopotential Ψ is
    # piecewise-constant per element.  Its Hessian (secular frequencies) is then
    # dominated by discretisation noise — frequencies come out 10–300× too low
    # on production-quality meshes.  CG2 (degree=2) gives piecewise-linear
    # gradients → Ψ ∝ |∇φ|² is piecewise-quadratic → Hessian is accurate.
    if args.degree == 1 and rank == 0:
        import warnings as _w
        _w.warn(
            "[degree=1] Using CG1 elements.  Secular frequencies computed from "
            "the Hessian of Ψ = |∇φ|² will be 10–300× too low on typical meshes "
            "because CG1 cannot accurately represent the quadratic RF null. "
            "Re-run with --degree 2 (CG2) for accurate secular frequencies.  "
            "The Laplace solve is ~4–8× more expensive but the Hessian is "
            "accurate on the same mesh.",
            UserWarning,
            stacklevel=2,
        )

    # Auto-scale h and ray_length when the user left the defaults (2e-6 m / 200e-6 m)
    # but the mesh is clearly not in SI metres.
    #
    # Hessian step scaling depends on element degree:
    #   CG1 (degree=1): Ψ is piecewise-constant per element; the finite-difference
    #     step must span several elements to see a signal above rounding noise.
    #     Use h = 4 × h_mesh (was 2×, but 2× is often at the noise floor).
    #   CG2 (degree=2): Ψ is piecewise-quadratic; curvature is resolved within
    #     a single element.  Use h = 3 × h_mesh for a good balance of accuracy
    #     and truncation error.
    _h_multiplier = 4.0 if args.degree == 1 else 3.0
    h = args.h
    ray_length = args.depth_ray_length
    if h_mesh > 1e-2:          # mesh appears to be in mm, cm, or larger units
        if args.h == 2e-6:
            h = h_mesh * _h_multiplier
        if args.depth_ray_length == 200e-6:
            # Use the full bbox diagonal so rays always reach the outer boundary.
            # Points outside the mesh return NaN and are skipped automatically.
            ray_length = bbox_span
        if rank == 0:
            print(f"[auto-scale] h={h:.4e} ({_h_multiplier}×h_mesh, degree={args.degree}), "
                  f"ray_length={ray_length:.4e} (mesh unit ~{h_mesh:.4e})")

    # Detect or use the specified coordinate unit (metres per mesh unit).
    coord_unit = args.coord_unit
    if coord_unit is None:
        # Heuristic based on bounding-box span in mesh units:
        #   span > 100  → coordinates are in µm  (coord_unit = 1e-6 m/mu)
        #   span > 0.1  → coordinates are in mm  (coord_unit = 1e-3 m/mu)
        #   else        → coordinates are in m   (coord_unit = 1.0 m/mu)
        if bbox_span > 100.0:
            coord_unit = 1e-6
        elif bbox_span > 0.1:
            coord_unit = 1e-3
        else:
            coord_unit = 1.0
        if rank == 0:
            print(f"[auto-detect] coord_unit={coord_unit:.0e} m/mesh_unit "
                  f"(bbox span={bbox_span:.3g} mesh units)")
    # ────────────────────────────────────────────────────────────────────────

    outer_tags = set(args.outer_tags) if args.outer_tags else set()

    # Warn if the user accidentally put outer-boundary tags in --ground-tags.
    # Those should be Neumann (no Dirichlet), not grounded.
    bad_gnd = set(args.ground_tags) & outer_tags
    if bad_gnd and rank == 0:
        import warnings
        warnings.warn(
            f"[BC] Tag(s) {sorted(bad_gnd)} appear in both --ground-tags and "
            "--outer-tags. Outer-boundary tags are Neumann (∂φ/∂n=0) — "
            "remove them from --ground-tags to avoid imposing φ=0 at the "
            "far-field boundary (which artificially truncates the escape barrier).",
            UserWarning,
        )

    # Electrode ground tags minus any outer-boundary tags
    gnd_electrode_tags = [t for t in args.ground_tags if t not in outer_tags]

    if rank == 0:
        print(f"[BC] RF Dirichlet tags : {args.rf_tags}")
        print(f"[BC] GND Dirichlet tags: {gnd_electrode_tags}")
        print(f"[BC] Neumann (outer)   : {sorted(outer_tags)}")

    bc_map_rf: Dict[int, float] = {tag: 1.0 for tag in args.rf_tags}
    bc_map_rf.update({tag: 0.0 for tag in gnd_electrode_tags})

    phi_rf = solve_laplace_tagged(
        domain, facet_tags, bc_map_rf, degree=args.degree, petsc_prefix=f"{args.prefix}_rf_"
    )
    phi_rf.name = "phi_rf"

    basis_fields: List[fem.Function] = []
    for tag in args.basis_tags:
        bc_map_b: Dict[int, float] = {tag: 1.0}
        for gt in gnd_electrode_tags:
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

    # Compute pseudopotential WITHOUT clipping — preserves curvature near the
    # RF null so the Hessian / secular-frequency calculation sees real signal
    # instead of a flat plateau.
    Psi = metrics.compute_rf_pseudopotential(
        phi_rf, omega_rf=2.0 * np.pi * args.rf_freq, q_C=q, m_kg=m,
        degree=args.degree,
    )
    Psi.name = "Psi_rf"

    if rank == 0:
        psi_arr = Psi.x.array
        print(f"[phi_rf] min={phi_rf.x.array.min():.4f}  max={phi_rf.x.array.max():.4f}")
        n_neg = int(np.sum(psi_arr < 0))
        print(f"[psi_raw] min={psi_arr.min():.4e}  max={psi_arr.max():.4e}  "
              f"n_negative={n_neg}/{psi_arr.size}")

    # ── RF-null search bounds ────────────────────────────────────────────────
    # The search box restricts the 5%-low-Ψ cluster centroid so the minimum
    # finder cannot drift into:
    #   z: the Neumann far-field (Ψ → 0 smoothly far from electrodes)
    #   x: a remote junction's RF null in multi-junction meshes
    #   y: the outer vacuum padding on the transverse axis
    #
    # z bound (auto-detect from RF facet top + margin) — always active.
    r0_z_max = args.r0_z_max
    r0_z_min = args.r0_z_min
    rf_nodes = None   # will be re-used for x/y auto-detect below

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
            r0_z_max = z_electrode_top + margin_mesh
            if rank == 0:
                print(f"[r0 search] z_electrode_top={z_electrode_top:.4g} mesh units, "
                      f"margin={margin_mesh:.4g} → z_max={r0_z_max:.4g}")
        except Exception as _e:
            if rank == 0:
                print(f"[r0 search] z_max auto-detect failed ({_e}); no z bound applied.")

    # z_min: if not user-specified, default to 0.0 (DC surface).
    # The mesh extends to z_min_bbox ≈ -0.22 mm (vacuum below the chip) which
    # has many near-zero Ψ DOFs.  Without a lower z bound the centroid finder
    # can drift below the electrode surface.
    if r0_z_min is None and domain.topology.dim == 3:
        r0_z_min = 0.0
        if rank == 0:
            print("[r0 search] z_min auto-set to 0.0 (DC surface). "
                  "Override with --r0-z-min if your trap sits below z=0.")

    # x/y bounds — explicit flags take priority; --r0-x-auto fills in x if
    # neither --r0-x-min nor --r0-x-max was supplied.
    r0_x_min = args.r0_x_min
    r0_x_max = args.r0_x_max
    r0_y_min = args.r0_y_min
    r0_y_max = args.r0_y_max

    if args.r0_x_auto and r0_x_min is None and r0_x_max is None:
        try:
            # Reuse rf_nodes gathered during z auto-detect; fall back to
            # computing them now if z auto-detect was skipped or failed.
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
            x_rf_half = (x_rf_max - x_rf_min) / 4.0  # ±25 % of full span
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

    # Print active bounds summary for reproducibility.
    if rank == 0:
        bounds_str = (
            f"x=[{r0_x_min},{r0_x_max}]  "
            f"y=[{r0_y_min},{r0_y_max}]  "
            f"z=[{r0_z_min},{r0_z_max}]"
        )
        print(f"[r0 search] active bounds (mesh units): {bounds_str}")

    # Find the RF null (centroid of low-|Ψ| cluster) using unclipped field.
    mininfo = metrics.find_minimum_cg1(
        Psi, comm=comm,
        x_min=r0_x_min, x_max=r0_x_max,
        y_min=r0_y_min, y_max=r0_y_max,
        z_min=r0_z_min, z_max=r0_z_max,
    )
    if rank == 0:
        print(f"[trap min] r0={mininfo.r_min.tolist()}, Psi_min={mininfo.psi_min:.4e} J")
        r0_SI = np.array(mininfo.r_min) * coord_unit
        print(f"[trap min] r0_SI={r0_SI.tolist()} m")
        bbox_center = ((bbox_min + bbox_max) / 2).tolist()
        print(f"[trap min] bbox center={bbox_center}")

    # ── Validate minimum location ─────────────────────────────────────────
    # Hard-reject any minimum below the DC surface (z < 0).  This is always
    # non-physical for a trap fabricated on a flat chip.
    if mininfo.r_min.shape[0] >= 3 and float(mininfo.r_min[2]) < 0.0:
        raise RuntimeError(
            f"[trap min] r0.z = {float(mininfo.r_min[2]):.4g} mesh units — "
            "the minimum finder landed below the DC surface (z < 0). "
            "Ensure --r0-z-min >= 0 (auto-set to 0.0 by default) and that "
            "--r0-z-max is tight enough to exclude the vacuum below the chip. "
            f"Current bounds: z=[{r0_z_min}, {r0_z_max}]"
        )

    if rank == 0:
        # Warn if the minimum z is suspiciously close to z_max — a sign the
        # search window is too wide and the centroid drifted into the far-field.
        if r0_z_max is not None and mininfo.r_min.shape[0] >= 3:
            z_r0 = float(mininfo.r_min[2])
            if z_r0 > 0.85 * r0_z_max:
                import warnings as _w
                _w.warn(
                    f"[trap min] r0.z={z_r0:.4g} is within 15% of the search ceiling "
                    f"z_max={r0_z_max:.4g} (mesh units).  This strongly suggests the "
                    "5th-percentile cluster centroid drifted into the Neumann far-field "
                    "vacuum instead of the true RF null.  Re-run with a tighter "
                    f"--r0-z-max, e.g.  --r0-z-max {z_r0 * 0.25:.3g}  "
                    "(aim for ~1.5× the expected trap height).",
                    UserWarning,
                    stacklevel=2,
                )

    # ── Debug: scan Ψ(z) along the z-axis through r0 (x,y fixed) ───────────
    # Prints Psi values from z=0 to z=0.25 mm so you can verify a real
    # minimum exists near the expected trap height and confirm r0 is correct.
    if rank == 0:
        _scan_x = float(mininfo.r_min[0])
        _scan_y = float(mininfo.r_min[1]) if mininfo.r_min.shape[0] >= 2 else 0.0
        _z_scan = np.linspace(0.0, 0.25, 26)  # 0 to 0.25 mesh units (mm)
        _pts = np.column_stack([
            np.full_like(_z_scan, _scan_x),
            np.full_like(_z_scan, _scan_y),
            _z_scan,
        ])
        _psi_scan = metrics.eval_function_at_points(Psi, _pts, comm=comm)
        print("[debug Ψ scan] z (mm) → Ψ_FEM:")
        for _zv, _pv in zip(_z_scan, _psi_scan):
            print(f"  z={_zv:.3f}  Psi={_pv:.4e}")

    # Secular frequencies from Hessian of the UNCLIPPED Ψ at the RF null.
    sec = metrics.secular_frequencies_from_pseudopotential(
        Psi, m_kg=m, r0=mininfo.r_min, h=h, comm=comm,
        coord_scale=coord_unit, v_rf=args.vrf,
    )
    if rank == 0:
        print(f"[secular] h_requested={h:.3e}  h_used={sec['h']:.3e} mesh units"
              f"  (h_mesh={h_mesh:.3e})")
        print(f"[secular] freq_hz={sec['freq_hz']}")

    _eigvals = sec["eigvals"]
    _n_neg = sum(1 for ev in _eigvals if ev < 0)
    if _n_neg > 0 and rank == 0:
        import warnings as _w
        _w.warn(
            f"[secular] Hessian has {_n_neg} negative eigenvalue(s): {_eigvals}. "
            "r0 is not a true local Ψ minimum (it may be a saddle point or an "
            "electrode-adjacent artifact). "
            "Tighten --r0-z-min / --r0-z-max / --r0-x-min / --r0-x-max so the "
            "search stays inside the physical trapping corridor.  "
            f"Current r0={mininfo.r_min.tolist()} (mesh units).",
            RuntimeWarning,
        )
    # Treat all-negative Hessian as a hard failure — frequencies are meaningless.
    if all(ev < 0 for ev in _eigvals):
        raise RuntimeError(
            "[secular] All Hessian eigenvalues are negative — r0 is not a local "
            "minimum at all.  Re-run with tighter search bounds, e.g.: "
            f"--r0-z-min 0.02 --r0-z-max 0.12  (current r0.z = "
            f"{float(mininfo.r_min[2]) if mininfo.r_min.shape[0] >= 3 else 'N/A':.4g})"
        )

    # Clipped copy for depth estimation (negative artefacts would deflate the
    # barrier height) and for visualisation / XDMF export.
    Psi_clipped = fem.Function(Psi.function_space)
    Psi_clipped.x.array[:] = np.maximum(Psi.x.array, 0.0)
    Psi_clipped.name = "Psi_rf"

    # Canonical Ψ₀ at r0 from the CLIPPED field.  Using the clipped field here
    # ensures trap_min.Psi_min_J and depth.Psi0_J are consistent (both come
    # from the same Psi_clipped evaluated at the same r0, without ambiguity
    # from Gibbs-ringing negative values in the raw field).
    psi_min_clipped = float(metrics.eval_function_at_points(
        Psi_clipped, np.array([mininfo.r_min], dtype=np.float64), comm=comm
    )[0])

    # ── Transport direction from weakest secular mode ───────────────────────
    # The eigenvectors in sec["eigvecs"] are columns, sorted by eigenvalue
    # (ascending).  The first column is the weakest mode — typically the axial
    # transport / junction direction with the smallest confinement.
    _eigvals_arr = np.array(sec["eigvals"])
    _eigvecs_arr = np.array(sec["eigvecs"])
    _weak_idx = int(np.argmin(_eigvals_arr))
    transport_eigvec = _eigvecs_arr[:, _weak_idx]
    if rank == 0:
        print(f"[depth] weak secular mode idx={_weak_idx}  "
              f"eigval={_eigvals_arr[_weak_idx]:.4e}  "
              f"eigvec={transport_eigvec.round(4).tolist()}")

    depth = None
    if not args.no_depth:
        depth = metrics.estimate_trap_depth_by_rays(
            Psi_clipped, r0=mininfo.r_min, ray_length=ray_length, nrays=args.depth_nrays, comm=comm,
            coord_scale=coord_unit, v_rf=args.vrf,
            transport_dir=transport_eigvec,
            transport_mode=args.transport_mode,
        )
        if rank == 0 and depth is not None:
            print("[depth summary] ── Sweep metrics ──")
            print(f"[depth summary]   r0.z              = {float(mininfo.r_min[2]) * coord_unit * 1e6:.2f} µm")
            print(f"[depth summary]   radial_depth_core = {depth.get('radial_depth_core_eV')} eV  "
                  f"({depth.get('n_core_radial_rays')} rays)")
            print(f"[depth summary]   transport_xscan   = {depth.get('transport_barrier_xscan_eV')} eV  "
                  f"interior={depth.get('transport_barrier_xscan_interior')}")
            print(f"[depth summary]   eigvec-scan       = {depth.get('transport_barrier_eigvec_eV')} eV  "
                  f"(dir: {depth.get('transport_dir_source')})")
            if args.transport_mode == "full":
                print("[depth summary] ── Full transport (CTC) ──")
                print(f"[depth summary]   ctc_like          = "
                      f"{depth.get('transport_barrier_ctc_like_eV')} eV  "
                      f"interior={depth.get('transport_barrier_ctc_interior')}  "
                      f"barrier at x={depth.get('ctc_barrier_x_mesh')} z={depth.get('ctc_barrier_z_mesh')} (mesh)")
                print("[depth summary] ── Broad radial depth ──")
                print(f"[depth summary]   min={depth.get('radial_depth_broad_min_eV')} eV  "
                      f"median={depth.get('radial_depth_broad_median_eV')} eV  "
                      f"max={depth.get('radial_depth_broad_max_eV')} eV  "
                      f"({depth.get('n_broad_radial_rays')} rays)")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_path = outdir / f"{args.prefix}_fields.xdmf"
    write_xdmf(xdmf_path, domain, [phi_rf, Psi_clipped, *basis_fields])

    report = {
        "mesh": str(args.mesh),
        "rf_tags": args.rf_tags,
        "ground_tags": args.ground_tags,
        "rf_freq_Hz": args.rf_freq,
        "mass_amu": args.mass_amu,
        "charge_e": args.charge_e,
        "h_used": sec["h"],        # actual step after auto-scaling inside numerical_hessian
        "h_requested": h,          # step that was passed in (may have been overridden)
        "ray_length_used": ray_length,
        "h_mesh_estimate": h_mesh,
        "coord_unit_m_per_mesh": coord_unit,
        "vrf_V": args.vrf,
        "r0_search_bounds_mesh": {
            "x": [r0_x_min, r0_x_max],
            "y": [r0_y_min, r0_y_max],
            "z": [r0_z_min, r0_z_max],
        },
        "r0_SI_m": (np.array(mininfo.r_min) * coord_unit).tolist(),
        "trap_min": {
            "r0_m": mininfo.r_min.tolist(),
            "Psi_min_J": psi_min_clipped,       # clipped; matches depth.Psi0_J
            "Psi_min_raw_J": float(mininfo.psi_min),  # unclipped (may be slightly < 0)
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
