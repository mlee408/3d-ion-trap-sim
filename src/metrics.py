from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, geometry


def _linear_problem(a, L, *, bcs=None, prefix="proj_", petsc_options=None):
    from dolfinx.fem.petsc import LinearProblem
    if petsc_options is None:
        petsc_options = {"ksp_type": "cg", "pc_type": "jacobi"}
    return LinearProblem(
        a,
        L,
        bcs=bcs if bcs is not None else [],
        petsc_options_prefix=prefix,
        petsc_options=petsc_options,
    )


def project(expr, V: fem.FunctionSpace, *, prefix="proj_", petsc_options=None) -> fem.Function:
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(expr, v) * ufl.dx
    return _linear_problem(a, L, prefix=prefix, petsc_options=petsc_options).solve()


def compute_rf_pseudopotential(
    phi_rf: fem.Function,
    *,
    omega_rf: float,
    q_C: float,
    m_kg: float,
    degree: int = 1,
    prefix: str = "PsiRF_",
    petsc_options: Optional[Dict[str, str]] = None,
) -> fem.Function:
    """RF pseudopotential (J) from RF potential φ (V): Ψ = q²|∇φ|²/(4 m ω²)."""
    domain = phi_rf.function_space.mesh
    V = fem.functionspace(domain, ("CG", degree))
    E_expr = -ufl.grad(phi_rf)
    Emag2_expr = ufl.dot(E_expr, E_expr)
    coeff = (q_C ** 2) / (4.0 * m_kg * (omega_rf ** 2))
    Psi = project(coeff * Emag2_expr, V, prefix=prefix, petsc_options=petsc_options)
    Psi.name = "Psi_RF_J"
    return Psi


def eval_function_at_points(
    f: fem.Function,
    points: np.ndarray,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray:
    domain = f.function_space.mesh
    gdim = domain.geometry.dim
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != gdim:
        raise ValueError(f"points must have shape (N, {gdim})")
    if comm is None:
        comm = domain.comm

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    cell_indices = np.full((pts.shape[0],), -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        cells = colliding.links(i)
        if len(cells) > 0:
            cell_indices[i] = cells[0]

    bs = f.function_space.dofmap.index_map_bs

    if bs == 1:
        local = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        tmp = np.zeros((1,), dtype=np.float64)
        for i, c in enumerate(cell_indices):
            if c >= 0:
                f.eval(tmp, pts[i:i+1], np.array([c], dtype=np.int32))
                local[i] = float(tmp[0])
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

    local = np.full((pts.shape[0], bs), np.nan, dtype=np.float64)
    tmp = np.zeros((1, bs), dtype=np.float64)
    for i, c in enumerate(cell_indices):
        if c >= 0:
            f.eval(tmp, pts[i:i+1], np.array([c], dtype=np.int32))
            local[i, :] = tmp[0, :]
    mask = np.isfinite(local).all(axis=1).astype(np.float64)
    vals = np.where(np.isfinite(local), local, 0.0)
    vals_g = np.empty_like(vals)
    mask_g = np.empty_like(mask)
    comm.Allreduce(vals, vals_g, op=MPI.SUM)
    comm.Allreduce(mask, mask_g, op=MPI.SUM)
    out = np.full_like(vals_g, np.nan)
    ok = mask_g > 0.5
    out[ok, :] = vals_g[ok, :] / mask_g[ok, None]
    return out


@dataclass
class TrapMinimum:
    r_min: np.ndarray
    psi_min: float
    dof_index: int
    rank: int

    @property
    def local_index(self) -> int:
        return self.dof_index


def dof_coordinate_from_index(f: fem.Function, local_dof: int) -> np.ndarray:
    V = f.function_space
    gdim = V.mesh.geometry.dim
    coords = V.tabulate_dof_coordinates().reshape((-1, gdim))
    if local_dof < 0 or local_dof >= coords.shape[0]:
        raise IndexError(f"local_dof {local_dof} out of range [0, {coords.shape[0]})")
    return np.array(coords[local_dof], dtype=np.float64)


def find_minimum_cg1(Psi: fem.Function, comm: MPI.Comm | None = None) -> TrapMinimum:
    if comm is None:
        comm = Psi.function_space.mesh.comm
    bs = Psi.function_space.dofmap.index_map_bs
    if bs != 1:
        raise ValueError("find_minimum_cg1 expects a scalar (bs=1) function.")
    imap = Psi.function_space.dofmap.index_map
    n_owned = imap.size_local * bs
    local_vals = Psi.x.array[:n_owned]

    if local_vals.size == 0:
        local_min = float("inf")
        local_idx = -1
    else:
        local_idx = int(np.argmin(local_vals))
        local_min = float(local_vals[local_idx])

    rank = comm.rank
    candidates = comm.allgather((local_min, rank, local_idx))
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


@dataclass
class SecularFrequencies:
    r0: np.ndarray
    h: float
    hessian: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    omega_rad_s: np.ndarray
    freq_hz: np.ndarray

    def to_jsonable(self) -> Dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def numerical_hessian(
    Psi: fem.Function,
    r0: np.ndarray,
    h: float,
    *,
    comm: MPI.Comm | None = None,
    max_tries: int = 6,
) -> np.ndarray:
    r0 = np.asarray(r0, dtype=np.float64)
    gdim = int(r0.shape[0])

    def f_at(pt: np.ndarray) -> float:
        return float(eval_function_at_points(Psi, np.array([pt], dtype=np.float64), comm=comm)[0])

    f0 = f_at(r0)
    if not np.isfinite(f0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")

    for k in range(max_tries):
        hk = float(h) * (0.5 ** k)
        H = np.zeros((gdim, gdim), dtype=np.float64)

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

    raise ValueError(
        "Hessian contains non-finite entries after shrinking h. "
        "r0 may be too close to the boundary, or the mesh may be too small."
    )


def secular_frequencies_from_pseudopotential(
    Psi: fem.Function,
    *,
    m_kg: float,
    r0: np.ndarray,
    h: float,
    comm: MPI.Comm | None = None,
) -> Dict:
    H = numerical_hessian(Psi, r0=r0, h=h, comm=comm)
    eigvals, eigvecs = np.linalg.eigh(H)
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
    ).to_jsonable()


_E_CHARGE = 1.602176634e-19


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.clip(1 - z * z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def estimate_trap_depth_by_rays(
    Psi: fem.Function,
    *,
    r0: np.ndarray,
    ray_length: float = 200e-6,
    nrays: int = 48,
    nsamples: int = 200,
    comm: MPI.Comm | None = None,
) -> Dict:
    domain = Psi.function_space.mesh
    gdim = domain.geometry.dim
    r0 = np.asarray(r0, dtype=np.float64)
    psi0 = float(eval_function_at_points(Psi, np.array([r0]), comm=comm)[0])
    if not np.isfinite(psi0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")

    if gdim == 3:
        dirs = _fibonacci_sphere(max(6, int(nrays)))
    elif gdim == 2:
        ang = np.linspace(0.0, 2.0 * np.pi, max(6, int(nrays)), endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        dirs = np.array([[1.0], [-1.0]], dtype=np.float64)

    ts = np.linspace(0.0, float(ray_length), int(nsamples))
    best_depth = np.inf
    best_dir = None
    best_max = np.nan
    best_used = 0

    for d in dirs:
        d = np.asarray(d, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-30)
        pts = r0[None, :] + ts[:, None] * d[None, :]
        vals = eval_function_at_points(Psi, pts, comm=comm)
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
        raise RuntimeError("Could not sample any valid rays inside the mesh. Check r0 and ray_length.")

    return {
        "r0_m": r0.tolist(),
        "Psi0_J": float(psi0),
        "depth_J": float(best_depth),
        "depth_eV": float(best_depth / _E_CHARGE),
        "worst_direction": best_dir.tolist(),
        "ray_max_Psi_J": float(best_max),
        "ray_samples_used": int(best_used),
        "ray_length_m": float(ray_length),
        "nrays": int(nrays),
        "nsamples": int(nsamples),
    }
