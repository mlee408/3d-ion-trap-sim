# src/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh as dmesh


# -----------------------------
# Helpers: projection utilities
# -----------------------------

def _linear_problem(a, L, *, bcs=None, prefix="proj_", petsc_options=None):
    """Compatibility wrapper for your dolfinx build."""
    from dolfinx.fem.petsc import LinearProblem
    if petsc_options is None:
        petsc_options = {"ksp_type": "cg", "pc_type": "jacobi"}
    return LinearProblem(
        a, L,
        bcs=bcs if bcs is not None else [],
        petsc_options_prefix=prefix,
        petsc_options=petsc_options,
    )


def project(expr, V: fem.FunctionSpace, *, prefix="proj_", petsc_options=None) -> fem.Function:
    """
    L2 project UFL expression 'expr' into FunctionSpace V.

    This avoids relying on legacy 'project' APIs and works in dolfinx.
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(expr, v) * ufl.dx

    uh = fem.Function(V)
    problem = _linear_problem(a, L, prefix=prefix, petsc_options=petsc_options)
    uh = problem.solve()
    return uh


# -----------------------------
# Field and pseudopotential
# -----------------------------

def compute_electric_field(
    phi: fem.Function,
    *,
    degree: int = 1,
    prefix: str = "E_",
    petsc_options: Optional[Dict[str, str]] = None,
) -> fem.Function:
    """
    Compute E = -grad(phi) as a projected vector field.

    Returns a dolfinx fem.Function in a CG vector function space.
    """
    domain = phi.function_space.mesh
    gdim = domain.geometry.dim

    # Vector CG space
    Vvec = fem.functionspace(domain, ("CG", degree, (gdim,)))
    E_expr = -ufl.grad(phi)
    E = project(E_expr, Vvec, prefix=prefix, petsc_options=petsc_options)
    E.name = "E"
    return E


def compute_field_magnitude_sq(
    E: fem.Function,
    *,
    degree: int = 1,
    prefix: str = "Emag2_",
    petsc_options: Optional[Dict[str, str]] = None,
) -> fem.Function:
    """
    Compute |E|^2 as a projected scalar field.
    """
    domain = E.function_space.mesh
    V = fem.functionspace(domain, ("CG", degree))
    Emag2_expr = ufl.dot(E, E)
    Emag2 = project(Emag2_expr, V, prefix=prefix, petsc_options=petsc_options)
    Emag2.name = "E_mag2"
    return Emag2


def compute_rf_pseudopotential(
    phi_rf_basis: fem.Function,
    *,
    q_coulomb: float,
    m_kg: float,
    Omega_rad_s: float,
    Vrf_volts: float,
    degree: int = 1,
    prefix: str = "PsiRF_",
    petsc_options: Optional[Dict[str, str]] = None,
) -> fem.Function:
    """
    Compute RF pseudopotential in Joules from a *basis* RF potential solution.

    Assumptions:
    - phi_rf_basis solves Laplace with RF electrode set to 1.0 (unit volts)
      and other electrodes 0.0 (or appropriate basis definition).
    - Actual RF potential is: phi_rf = Vrf_volts * phi_rf_basis
    - Electric field: E_rf = -grad(phi_rf) = -Vrf * grad(phi_basis)

    Formula:
        Psi_RF = (q^2 / (4 m Omega^2)) * |E_rf|^2

    Returns:
        Psi_RF (Joules) as a scalar CG Function.
    """
    domain = phi_rf_basis.function_space.mesh
    V = fem.functionspace(domain, ("CG", degree))

    # E_rf = -grad(Vrf * phi_basis) = -Vrf * grad(phi_basis)
    E_rf_expr = -Vrf_volts * ufl.grad(phi_rf_basis)
    Emag2_expr = ufl.dot(E_rf_expr, E_rf_expr)

    coeff = (q_coulomb ** 2) / (4.0 * m_kg * (Omega_rad_s ** 2))
    Psi_expr = coeff * Emag2_expr

    Psi = project(Psi_expr, V, prefix=prefix, petsc_options=petsc_options)
    Psi.name = "Psi_RF_J"
    return Psi


# -----------------------------
# Sampling / evaluation utilities
# -----------------------------

def eval_function_at_points(
    f: fem.Function,
    points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate fem.Function at physical points.

    points: shape (N, gdim)
    Returns: values shape (N,) for scalar f, or (N, gdim) for vector f.

    Points outside the mesh return np.nan.
    """
    domain = f.function_space.mesh
    gdim = domain.geometry.dim
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != gdim:
        raise ValueError(f"points must have shape (N, {gdim})")

    # Build bounding box tree and locate points
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    # For each point, pick first colliding cell (if any)
    cell_indices = np.full((pts.shape[0],), -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        cells = colliding.links(i)
        if len(cells) > 0:
            cell_indices[i] = cells[0]

    # Prepare output buffer
    bs = f.function_space.dofmap.index_map_bs  # block size (1 for scalar, gdim for vector CG)
    if bs == 1:
        out = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        tmp = np.zeros((1,), dtype=np.float64)
        for i, c in enumerate(cell_indices):
            if c >= 0:
                f.eval(tmp, pts[i:i+1], np.array([c], dtype=np.int32))
                out[i] = float(tmp[0])
        return out
    else:
        out = np.full((pts.shape[0], bs), np.nan, dtype=np.float64)
        tmp = np.zeros((1, bs), dtype=np.float64)
        for i, c in enumerate(cell_indices):
            if c >= 0:
                f.eval(tmp, pts[i:i+1], np.array([c], dtype=np.int32))
                out[i, :] = tmp[0, :]
        return out


# -----------------------------
# Trap minimum / height
# -----------------------------

@dataclass
class TrapMinimum:
    r_min: np.ndarray
    psi_min: float
    local_index: int

def find_minimum_cg1(Psi, comm=None):
    # --- pick comm safely ---
    if comm is None:
        if MPI is None:
            comm = None
        else:
            comm = MPI.COMM_WORLD

    # --- pull local values from your CG1 function ---
    V = Psi.x.array  # dolfinx vector (local chunk)
    local_idx = int(np.argmin(V))
    local_min = float(V[local_idx])

    # Get coordinate for that local dof (you likely already have this part)
    # r_local = ... (shape (gdim,)) corresponding to local_idx
    # I'll assume you already compute `r_local` below.

    # --- SERIAL EARLY RETURN (fixes your crash) ---
    if (comm is None) or (getattr(comm, "size", 1) == 1):
        r_local = dof_coordinate_from_index(Psi, local_idx)  # your existing helper
        return MinInfo(r_min=r_local, psi_min=local_min, dof_index=local_idx, rank=0)

    # --- PARALLEL: avoid MINLOC entirely; do an allgather of scalars ---
    rank = comm.rank
    candidates = comm.allgather((local_min, rank, local_idx))

    # pick smallest by value, then rank as tie-breaker (deterministic)
    best_min, best_rank, best_lidx = min(candidates, key=lambda t: (t[0], t[1]))

    # now best_rank computes the coordinate and broadcasts it
    if rank == best_rank:
        r_best = dof_coordinate_from_index(Psi, best_lidx)  # your existing helper
    else:
        r_best = None

    r_best = comm.bcast(r_best, root=best_rank)

    return MinInfo(r_min=np.array(r_best), psi_min=float(best_min), dof_index=int(best_lidx), rank=int(best_rank))

# -----------------------------
# Hessian + secular frequencies
# -----------------------------

@dataclass
class SecularFrequencies:
    r0: np.ndarray                 # point used
    h: float                       # step size
    hessian: np.ndarray            # (gdim, gdim) numeric Hessian of Psi
    eigvals: np.ndarray            # (gdim,) principal curvatures
    eigvecs: np.ndarray            # (gdim, gdim) columns are eigenvectors
    omega_rad_s: np.ndarray        # (gdim,) sqrt(eigvals/m) if Psi is energy
    freq_hz: np.ndarray            # (gdim,) omega/(2pi)


def numerical_hessian(
    Psi: fem.Function,
    r0: np.ndarray,
    h: float,
) -> np.ndarray:
    """
    Compute numeric Hessian of scalar field Psi at r0 using central differences.

    Mixed derivative:
      d2f/dx_i dx_j ≈ (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4h^2)
    Diagonal:
      d2f/dx_i^2 ≈ (f(x+h_i) - 2f(x) + f(x-h_i)) / h^2

    Returns Hessian (gdim, gdim). Points outside mesh become nan → may break.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    gdim = r0.shape[0]

    def f_at(pt):
        val = eval_function_at_points(Psi, np.array([pt]))[0]
        return val

    f0 = f_at(r0)
    if not np.isfinite(f0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")

    H = np.zeros((gdim, gdim), dtype=np.float64)

    # Diagonal
    for i in range(gdim):
        e = np.zeros(gdim)
        e[i] = 1.0
        fp = f_at(r0 + h * e)
        fm = f_at(r0 - h * e)
        if not (np.isfinite(fp) and np.isfinite(fm)):
            H[i, i] = np.nan
        else:
            H[i, i] = (fp - 2.0 * f0 + fm) / (h ** 2)

    # Mixed
    for i in range(gdim):
        for j in range(i + 1, gdim):
            ei = np.zeros(gdim); ei[i] = 1.0
            ej = np.zeros(gdim); ej[j] = 1.0
            fpp = f_at(r0 + h*ei + h*ej)
            fpm = f_at(r0 + h*ei - h*ej)
            fmp = f_at(r0 - h*ei + h*ej)
            fmm = f_at(r0 - h*ei - h*ej)
            if not all(np.isfinite(x) for x in [fpp, fpm, fmp, fmm]):
                val = np.nan
            else:
                val = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
            H[i, j] = val
            H[j, i] = val

    return H


def secular_frequencies_from_pseudopotential(
    Psi_energy: fem.Function,
    *,
    m_kg: float,
    r0: np.ndarray,
    h: float,
) -> SecularFrequencies:
    """
    Compute secular frequencies from a pseudopotential that is already in energy units (Joules).

    Around minimum:
      Psi ≈ Psi0 + 1/2 * (x^T H x)
    For a 1D harmonic oscillator:
      Psi ≈ 1/2 * m * omega^2 * x^2
    So principal curvatures λ_i relate to omega via:
      omega_i = sqrt(λ_i / m)

    Returns omega (rad/s) and f (Hz).
    """
    H = numerical_hessian(Psi_energy, r0=r0, h=h)

    # Eigen-decomposition (handle nan by raising)
    if np.any(~np.isfinite(H)):
        raise ValueError("Hessian contains non-finite entries. Try smaller h or a point deeper inside mesh.")

    eigvals, eigvecs = np.linalg.eigh(H)
    # Guard: negative curvatures mean not a stable minimum in that direction
    omega = np.sqrt(np.clip(eigvals, 0.0, None) / m_kg)
    freq = omega / (2.0 * np.pi)

    return SecularFrequencies(
        r0=np.array(r0, dtype=np.float64),
        h=float(h),
        hessian=H,
        eigvals=eigvals,
        eigvecs=eigvecs,
        omega_rad_s=omega,
        freq_hz=freq,
    )


# -----------------------------
# Trap depth (approx via rays)
# -----------------------------

@dataclass
class TrapDepthEstimate:
    r0: np.ndarray
    psi0: float
    depth_joules: float
    depth_ev: float
    worst_direction: np.ndarray  # direction that gives minimum depth
    ray_max_psi: float
    ray_samples_used: int


_E_CHARGE = 1.602176634e-19  # Coulomb, also J/eV


def estimate_trap_depth_by_rays(
    Psi_energy: fem.Function,
    r0: np.ndarray,
    *,
    directions: Optional[Sequence[np.ndarray]] = None,
    r_max: float = 5e-4,
    n: int = 200,
) -> TrapDepthEstimate:
    """
    Approximate trap depth by sampling Psi along multiple rays from r0 and
    taking min over directions of (max(Psi along ray) - Psi0).

    This is a heuristic but works well as a first benchmark metric.

    - r_max: maximum distance to march along ray
    - n: number of samples along each ray
    """
    domain = Psi_energy.function_space.mesh
    gdim = domain.geometry.dim

    r0 = np.asarray(r0, dtype=np.float64)
    psi0 = float(eval_function_at_points(Psi_energy, np.array([r0]))[0])
    if not np.isfinite(psi0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")

    if directions is None:
        # Default: ±axes
        dirs = []
        for i in range(gdim):
            e = np.zeros(gdim); e[i] = 1.0
            dirs.append(e)
            dirs.append(-e)
        directions = dirs

    best_depth = np.inf
    best_dir = None
    best_max = np.nan
    best_used = 0

    ts = np.linspace(0.0, r_max, n)
    for d in directions:
        d = np.asarray(d, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-30)

        pts = r0[None, :] + ts[:, None] * d[None, :]
        vals = eval_function_at_points(Psi_energy, pts)

        # keep finite segment
        finite = np.isfinite(vals)
        if finite.sum() < 5:
            continue

        vals_f = vals[finite]
        max_psi = float(np.max(vals_f))
        depth = max_psi - psi0

        if depth < best_depth:
            best_depth = depth
            best_dir = d.copy()
            best_max = max_psi
            best_used = int(finite.sum())

    if best_dir is None:
        raise RuntimeError("Could not sample any valid rays inside the mesh. Check r0 and r_max.")

    depth_ev = best_depth / _E_CHARGE

    return TrapDepthEstimate(
        r0=r0,
        psi0=psi0,
        depth_joules=float(best_depth),
        depth_ev=float(depth_ev),
        worst_direction=best_dir,
        ray_max_psi=float(best_max),
        ray_samples_used=int(best_used),
    )


# -----------------------------
# Line cuts for benchmarking
# -----------------------------

@dataclass
class LineCut:
    t: np.ndarray
    points: np.ndarray
    values: np.ndarray


def line_cut(
    f: fem.Function,
    r0: np.ndarray,
    direction: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    n: int,
) -> LineCut:
    """
    Sample f along a line: r(t) = r0 + t * direction
    """
    r0 = np.asarray(r0, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-30)

    t = np.linspace(t_min, t_max, n)
    pts = r0[None, :] + t[:, None] * d[None, :]
    vals = eval_function_at_points(f, pts)
    return LineCut(t=t, points=pts, values=vals)
