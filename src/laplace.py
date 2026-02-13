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
    """Return object for Laplace solves."""
    phi: fem.Function
    V: fem.FunctionSpace
    bcs_applied: Dict[int, float]


def solve_laplace_tagged(
    domain: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
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
    domain
        The computational mesh.
    facet_tags
        Facet (boundary) tags. Must tag the boundary facets you want to constrain.
    boundary_values
        Mapping {tag_id: value} where value is the Dirichlet potential (volts) on facets with that tag.
        Example: {RF_TAG: 1.0, GND_TAG: 0.0}
    degree
        CG polynomial degree (default 1).
    petsc_options_prefix
        PETSc options prefix (use a unique-ish prefix per problem).
    petsc_options
        PETSc linear solver options. If None, uses a safe default.

    Returns
    -------
    LaplaceResult
        Contains phi (solution), V (function space), and the applied BC map.
    """
    if petsc_options is None:
        petsc_options = {"ksp_type": "cg", "pc_type": "jacobi"}

    if facet_tags is None:
        raise ValueError("facet_tags is None. Cannot apply boundary conditions by tag.")

    V = fem.functionspace(domain, ("CG", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1

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

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

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
    """Export mesh + one or more Functions to XDMF for ParaView/PyVista."""
    from dolfinx import io

    with io.XDMFFile(comm, xdmf_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        for f in functions:
            xdmf.write_function(f)
