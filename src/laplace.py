# src/laplace.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh


@dataclass
class LaplaceResult:
    phi: fem.Function
    V: fem.FunctionSpace
    bcs_applied: Dict[int, float]


def solve_laplace_tagged(
    domain: dmesh.Mesh,
    facet_tags: fem.MeshTags,
    boundary_values: Dict[int, float],
    *,
    degree: int = 1,
    petsc_options_prefix: str = "laplace_",
    petsc_options: Optional[Dict[str, str]] = None,
) -> LaplaceResult:
    """
    Solve Laplace equation ∇²phi = 0 on 'domain' with Dirichlet boundary conditions
    applied by facet tag IDs.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational mesh.
    facet_tags : dolfinx.mesh.MeshTags
        Facet (boundary) tags. Must tag the boundary facets you want to constrain.
    boundary_values : dict[int, float]
        Mapping {tag_id: value} where value is the Dirichlet potential on facets with that tag.
        Example: {RF_TAG: 1.0, GND_TAG: 0.0}
    degree : int
        CG polynomial degree (default 1).
    petsc_options_prefix : str
        Required by some dolfinx builds (your install). Use a unique-ish prefix per problem.
    petsc_options : dict[str, str] | None
        PETSc linear solver options. If None, uses a safe default.

    Returns
    -------
    LaplaceResult
        Contains phi (solution), V (function space), and the applied BC map.
    """
    if petsc_options is None:
        # Safe baseline that works on most installs
        petsc_options = {"ksp_type": "cg", "pc_type": "jacobi"}

    # Function space (scalar potential)
    V = fem.functionspace(domain, ("CG", degree))

    # Build Dirichlet BCs from facet tags
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Basic validation: ensure requested tags exist
    existing = set(np.unique(facet_tags.values))
    missing = [tag for tag in boundary_values.keys() if tag not in existing]
    if missing:
        raise ValueError(
            f"Requested boundary tag(s) {missing} not found in facet_tags. "
            f"Existing tags: {sorted(existing)}"
        )

    bcs = []
    for tag_id, val in boundary_values.items():
        facets = facet_tags.find(tag_id)
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        bc = fem.dirichletbc(PETSc.ScalarType(val), dofs, V)
        bcs.append(bc)

    # Variational problem for Laplace: ∫ grad(u)·grad(v) dx = 0
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    # Import LinearProblem from the correct place for your build
    from dolfinx.fem.petsc import LinearProblem

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=petsc_options_prefix,
        petsc_options=petsc_options,
    )

    phi = problem.solve()
    phi.name = "phi"

    return LaplaceResult(phi=phi, V=V, bcs_applied=dict(boundary_values))


def export_xdmf(
    domain: dmesh.Mesh,
    functions: Tuple[fem.Function, ...],
    xdmf_path: str,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """
    Export mesh + one or more Functions to XDMF for ParaView/PyVista inspection.
    """
    from dolfinx import io

    with io.XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in functions:
            xdmf.write_function(f)
