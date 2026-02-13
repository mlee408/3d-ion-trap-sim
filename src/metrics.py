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
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray:
    """
    Evaluate fem.Function at physical points in a way that is correct under MPI.

    Each rank attempts to evaluate points that fall within its local mesh partition.
    Values are then combined so that, for each point, the (unique) owning rank supplies
    the result. Points outside the global mesh return np.nan.

    points: shape (N, gdim)
    Returns: values shape (N,) for scalar f, or (N, bs) for vector/tensor fields.
    """
    domain = f.function_space.mesh
    gdim = domain.geometry.dim
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != gdim:
        raise ValueError(f"points must have shape (N, {gdim})")

    if comm is None:
        comm = domain.comm

    # Build bounding box tree and locate points (local candidates)
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    # For each point, pick first colliding cell (if any) on this rank
    cell_indices = np.full((pts.shape[0],), -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        cells = colliding.links(i)
        if len(cells) > 0:
            cell_indices[i] = cells[0]

    bs = f.function_space.dofmap.index_map_bs  # 1 for scalar; gdim for vector CG

    if bs == 1:
        local = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        tmp = np.zeros((1,), dtype=np.float64)
        for i, c in enumerate(cell_indices):
            if c >= 0:
                f.eval(tmp, pts[i:i+1], np.array([c], dtype=np.int32))
                local[i] = float(tmp[0])

        # Combine across ranks: sum(valid values) / sum(mask)
        mask = np.isfinite(local).astype(np.float64)
        vals = np.where(np.isfinite(local), local, 0.0)
        vals_g = np.empty_like(vals)
        mask_g = np.empty_like(mask)
        comm.Allreduce(vals, vals_g, op=MPI.SUM)
        comm.Allreduce(mask, mask_g, op=MPI.SUM)
        out = np.full_like(vals_g, np.nan)
        ok = mask_g > 0.5
        out[ok] = vals_g[ok] / mask_g[ok]
        return out

    else:
        local = np.full((pts.shape[0], bs), np.nan, dtype=np.float64)
        tmp = np.zeros((1, bs), dtype=np.float64)
        for i, c in enumerate(cell_indices):
            if c >= 0:
                f.eval(tmp, pts[i:i+1], np.array([c], dtype=np.int32))
                local[i, :] = tmp[0, :]

        mask = np.isfinite(local).all(axis=1).astype(np.float64)  # one mask per point
        vals = np.where(np.isfinite(local), local, 0.0)

        vals_g = np.empty_like(vals)
        mask_g = np.empty_like(mask)
        comm.Allreduce(vals, vals_g, op=MPI.SUM)
        comm.Allreduce(mask, mask_g, op=MPI.SUM)

        out = np.full_like(vals_g, np.nan)
        ok = mask_g > 0.5
        out[ok, :] = vals_g[ok, :] / mask_g[ok, None]
        return out


# -----------------------------
# Trap minimum / height
# -----------------------------

@dataclass
class TrapMinimum:
    """Global trap minimum information."""
    r_min: np.ndarray
    psi_min: float
    dof_index: int
    rank: int

    @property
    def local_index(self) -> int:  # backward-compat alias
        return self.dof_index

def dof_coordinate_from_index(f: fem.Function, local_dof: int) -> np.ndarray:
    """Return physical coordinate of a (scalar) CG1 dof given its *local* dof index.

    Notes (dolfinx 0.10):
    - For CG1 Lagrange, dofs coincide with mesh vertices.
    - `tabulate_dof_coordinates()` returns coordinates for local dofs (including ghosts) in
      the same ordering as the dofmap.
    """
    V = f.function_space
    gdim = V.mesh.geometry.dim
    coords = V.tabulate_dof_coordinates().reshape((-1, gdim))
    if local_dof < 0 or local_dof >= coords.shape[0]:
        raise IndexError(f"local_dof {local_dof} out of range [0, {coords.shape[0]})")
    return np.array(coords[local_dof], dtype=np.float64)


def find_minimum_cg1(Psi: fem.Function, comm: MPI.Comm | None = None) -> TrapMinimum:
    """Find the global minimum of a scalar CG1 field Psi.

    - Uses owned dofs only (avoids ghost dofs in parallel).
    - Avoids MPI.MINLOC datatype pitfalls by using allgather + deterministic tie-break.
    """
    if comm is None:
        comm = Psi.function_space.mesh.comm

    # Owned dofs only (avoid ghosts)
    bs = Psi.function_space.dofmap.index_map_bs
    if bs != 1:
        raise ValueError("find_minimum_cg1 expects a scalar (bs=1) function.")
    imap = Psi.function_space.dofmap.index_map
    n_owned = imap.size_local * bs
    local_vals = Psi.x.array[:n_owned]

    if local_vals.size == 0:
        # Degenerate partition; treat as +inf
        local_min = float("inf")
        local_idx = -1
    else:
        local_idx = int(np.argmin(local_vals))
        local_min = float(local_vals[local_idx])

    rank = comm.rank
    candidates = comm.allgather((local_min, rank, local_idx))

    # Pick smallest value, then smaller rank as deterministic tie-break
    best_min, best_rank, best_lidx = min(candidates, key=lambda t: (t[0], t[1]))

    if best_lidx < 0 or not np.isfinite(best_min):
        raise RuntimeError("Could not determine a finite minimum (check Psi field / partitions).")

    if rank == best_rank:
        r_best = dof_coordinate_from_index(Psi, best_lidx)
    else:
        r_best = None

    r_best = comm.bcast(r_best, root=best_rank)

    return TrapMinimum(
        r_min=np.array(r_best, dtype=np.float64),
        psi_min=float(best_min),
        dof_index=int(best_lidx),
        rank=int(best_rank),
    )


# -----------------------------
# Hessian + secular frequencies
 + secular frequencies
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
    *,
    max_tries: int = 6,
) -> np.ndarray:
    """Compute numeric Hessian of scalar field Psi at r0 using central differences.

    This routine is mesh-aware in the sense that it will automatically shrink `h` if
    any of the probe points land outside the mesh (leading to NaNs).
    """
    r0 = np.asarray(r0, dtype=np.float64)
    gdim = int(r0.shape[0])

    def f_at(pt: np.ndarray) -> float:
        return float(eval_function_at_points(Psi, np.array([pt], dtype=np.float64))[0])

    f0 = f_at(r0)
    if not np.isfinite(f0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")

    for k in range(max_tries):
        hk = float(h) * (0.5 ** k)

        H = np.zeros((gdim, gdim), dtype=np.float64)

        # Diagonal
        ok = True
        for i in range(gdim):
            e = np.zeros(gdim); e[i] = 1.0
            fp = f_at(r0 + hk * e)
            fm = f_at(r0 - hk * e)
            if not (np.isfinite(fp) and np.isfinite(fm)):
                ok = False
                break
            H[i, i] = (fp - 2.0 * f0 + fm) / (hk ** 2)

        if not ok:
            continue

        # Mixed
        for i in range(gdim):
            for j in range(i + 1, gdim):
                ei = np.zeros(gdim); ei[i] = 1.0
                ej = np.zeros(gdim); ej[j] = 1.0
                fpp = f_at(r0 + hk*ei + hk*ej)
                fpm = f_at(r0 + hk*ei - hk*ej)
                fmp = f_at(r0 - hk*ei + hk*ej)
                fmm = f_at(r0 - hk*ei - hk*ej)
                if not all(np.isfinite(x) for x in (fpp, fpm, fmp, fmm)):
                    ok = False
                    break
                val = (fpp - fpm - fmp + fmm) / (4.0 * hk * hk)
                H[i, j] = val
                H[j, i] = val
            if not ok:
                break

        if ok and np.all(np.isfinite(H)):
            return H

    raise ValueError("Hessian contains non-finite entries after shrinking h. r0 may be too close to the boundary, or the mesh may be too small.")


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
