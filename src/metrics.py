from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional
import warnings

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, geometry, mesh as dmesh


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
    """RF pseudopotential (normalised, J/V²) from unit RF potential φ (0–1).

    The returned Ψ is computed for φ normalised to 1 V (i.e., with actual
    V_RF = 1 V).  Multiply by V_RF² to get the physical pseudopotential.

    Ψ = q² |∇φ|² / (4 m ω²)

    Notes
    -----
    * φ is dimensionless (boundary conditions φ=1 on RF, φ=0 on ground),
      so |∇φ|² has units of [mesh_unit]⁻².  The secular frequencies derived
      from this Ψ must be scaled by V_RF / coord_scale (see
      `secular_frequencies_from_pseudopotential`).
    * The CG1 L2-projection of |∇φ|² can produce small negative nodal values
      (Gibbs-like undershoot near discontinuities).  These are clipped to zero
      here; they are non-physical artefacts of the projection.
    """
    domain = phi_rf.function_space.mesh
    V = fem.functionspace(domain, ("CG", degree))
    E_expr = -ufl.grad(phi_rf)
    Emag2_expr = ufl.dot(E_expr, E_expr)
    coeff = (q_C ** 2) / (4.0 * m_kg * (omega_rf ** 2))
    Psi = project(coeff * Emag2_expr, V, prefix=prefix, petsc_options=petsc_options)
    # Clip non-physical negative values produced by the L2-projection overshoot.
    Psi.x.array[:] = np.maximum(Psi.x.array, 0.0)
    Psi.name = "Psi_RF_J"
    return Psi


def eval_function_at_points(
    f: fem.Function,
    points: np.ndarray,
    *,
    comm: MPI.Comm | None = None,
) -> np.ndarray:
    """Evaluate a FEniCSx Function at arbitrary points, MPI-aware.

    Parameters
    ----------
    f      : scalar or vector fem.Function
    points : shape (N, gdim)
    comm   : MPI communicator (defaults to domain.comm)

    Returns
    -------
    np.ndarray
        Shape (N,) for scalar functions, (N, bs) for vector functions.
        Points not found in the local mesh partition return NaN; values
        are reduced with a weighted sum across all ranks.
    """
    domain = f.function_space.mesh
    gdim = domain.geometry.dim
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != gdim:
        raise ValueError(f"points must have shape (N, {gdim}), got {pts.shape}")
    if comm is None:
        comm = domain.comm

    N = pts.shape[0]

    # DOLFINx ≥ 0.8: geometry routines and f.eval() always require 3-component
    # coordinates regardless of the mesh's geometric dimension.
    if gdim < 3:
        pts3 = np.pad(pts, ((0, 0), (0, 3 - gdim)), constant_values=0.0)
    else:
        pts3 = np.ascontiguousarray(pts)

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts3)

    cell_indices = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        cells = colliding.links(i)
        if len(cells) > 0:
            cell_indices[i] = int(cells[0])

    bs = f.function_space.dofmap.index_map_bs
    local = np.full((N, bs), np.nan, dtype=np.float64)

    valid = cell_indices >= 0
    if valid.any():
        # DOLFINx ≥ 0.8 API: result = f.eval(x, cells)  →  shape (npts, bs)
        # (the old 3-arg form  f.eval(out, x, cells)  was removed in 0.8)
        result = f.eval(pts3[valid], cell_indices[valid])
        result = np.asarray(result, dtype=np.float64).reshape(int(valid.sum()), bs)
        local[valid] = result

    # MPI reduction: weighted average so every rank gets the correct value
    mask = np.isfinite(local).all(axis=-1).astype(np.float64)   # (N,)
    vals = np.where(np.isfinite(local), local, 0.0)              # (N, bs)

    vals_g = np.empty_like(vals)
    mask_g = np.empty_like(mask)
    comm.Allreduce(vals, vals_g, op=MPI.SUM)
    comm.Allreduce(mask, mask_g, op=MPI.SUM)

    out = np.full((N, bs), np.nan, dtype=np.float64)
    ok = mask_g > 0.5
    out[ok] = vals_g[ok] / mask_g[ok, None]

    # Return 1-D array for scalar functions (preserves caller expectations)
    return out[:, 0] if bs == 1 else out


def _estimate_cell_h(domain) -> float:
    """Return a rough characteristic cell size from the mesh bounding box.

    Used to auto-scale finite-difference step sizes (h) that depend on the
    physical units of the mesh coordinates.
    """
    coords = domain.geometry.x  # always (num_nodes, 3) in DOLFINx
    if coords.shape[0] < 2:
        return 1.0
    span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    num_cells = max(domain.topology.index_map(domain.topology.dim).size_global, 1)
    tdim = max(domain.topology.dim, 1)
    return span / float(num_cells ** (1.0 / tdim))


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
    # DOLFINx always stores dof coordinates as (ndofs, 3) regardless of gdim.
    # Reshaping by gdim would miscount rows for gdim < 3.
    raw = V.tabulate_dof_coordinates()
    coords = raw.reshape(-1, 3)[:, :gdim]   # trim z (and y) for lower-dim meshes
    if local_dof < 0 or local_dof >= coords.shape[0]:
        raise IndexError(f"local_dof {local_dof} out of range [0, {coords.shape[0]})")
    return np.array(coords[local_dof], dtype=np.float64)


def find_minimum_cg1(Psi: fem.Function, comm: MPI.Comm | None = None) -> TrapMinimum:
    """Find the DOF with the smallest Ψ value among interior (non-boundary) DOFs.

    Boundary DOFs sit on electrode surfaces where Ψ may be near zero due to
    the boundary condition on φ.  Using them as the starting point for the
    Hessian would place r0 on the mesh boundary, causing all finite-difference
    stencil points to fall outside the domain.
    """
    if comm is None:
        comm = Psi.function_space.mesh.comm
    bs = Psi.function_space.dofmap.index_map_bs
    if bs != 1:
        raise ValueError("find_minimum_cg1 expects a scalar (bs=1) function.")

    domain = Psi.function_space.mesh
    V = Psi.function_space
    tdim = domain.topology.dim
    fdim = tdim - 1
    imap = V.dofmap.index_map
    n_owned = imap.size_local          # number of locally owned DOFs

    # Build a boolean mask that is False for any DOF touching a boundary facet.
    domain.topology.create_connectivity(fdim, tdim)
    bnd_facets = dmesh.exterior_facet_indices(domain.topology)
    bnd_dofs = fem.locate_dofs_topological(V, fdim, bnd_facets)
    interior_mask = np.ones(n_owned, dtype=bool)
    owned_bnd = bnd_dofs[bnd_dofs < n_owned]
    if owned_bnd.size > 0:
        interior_mask[owned_bnd] = False

    local_vals = Psi.x.array[:n_owned]
    interior_idx = np.where(interior_mask)[0]

    if interior_idx.size > 0:
        best = int(np.argmin(local_vals[interior_idx]))
        local_idx = int(interior_idx[best])
        local_min = float(local_vals[local_idx])
    elif local_vals.size > 0:
        # Degenerate mesh partition — fall back to all DOFs
        local_idx = int(np.argmin(local_vals))
        local_min = float(local_vals[local_idx])
    else:
        local_idx = -1
        local_min = float("inf")

    rank = comm.rank
    candidates = comm.allgather((local_min, rank, local_idx))
    best_min, best_rank, best_lidx = min(candidates, key=lambda t: (t[0], t[1]))
    if best_lidx < 0 or not np.isfinite(best_min):
        raise RuntimeError(
            "Could not find a finite interior minimum. "
            "Check that the Psi field is non-trivial and the mesh is valid."
        )

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
    max_tries: int = 12,
) -> np.ndarray:
    """Compute the Hessian of Psi at r0 by central finite differences.

    h is auto-scaled if it looks inappropriate for the mesh's physical units
    (e.g. mesh in mm but h specified in metres).  Each failed attempt halves h;
    up to max_tries halvings are tried before raising.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    gdim = int(r0.shape[0])
    domain = Psi.function_space.mesh

    # Auto-scale h to mesh geometry if the supplied value is wildly off.
    h_mesh = _estimate_cell_h(domain)
    h = float(h)
    if h <= 0.0 or not np.isfinite(h) or h < h_mesh * 0.5 or h > h_mesh * 100.0:
        h_new = h_mesh * 2.0
        if comm is None or comm.rank == 0:
            warnings.warn(
                f"h={h:.3e} is outside a reasonable range for this mesh "
                f"(estimated cell size {h_mesh:.3e}). Auto-scaling to h={h_new:.3e}.",
                RuntimeWarning,
                stacklevel=3,
            )
        h = h_new

    def f_at(pt: np.ndarray) -> float:
        return float(eval_function_at_points(Psi, np.array([pt], dtype=np.float64), comm=comm)[0])

    f0 = f_at(r0)
    if not np.isfinite(f0):
        raise ValueError(
            f"Psi is NaN/inf at r0={r0.tolist()}. "
            "The trap minimum lies outside the mesh, or boundary DOF exclusion failed."
        )

    for k in range(max_tries):
        hk = h * (0.5 ** k)
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
                fpp = f_at(r0 + hk * ei + hk * ej)
                fpm = f_at(r0 + hk * ei - hk * ej)
                fmp = f_at(r0 - hk * ei + hk * ej)
                fmm = f_at(r0 - hk * ei - hk * ej)
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
        f"Hessian could not be computed at r0={r0.tolist()}. "
        f"Tried {max_tries} step sizes from h={h:.2e} down to {h * 0.5**(max_tries-1):.2e} "
        f"(mesh cell size ~{h_mesh:.2e}). "
        "r0 may still be too close to a mesh boundary — check that the interior "
        "minimum is well inside the domain."
    )


def secular_frequencies_from_pseudopotential(
    Psi: fem.Function,
    *,
    m_kg: float,
    r0: np.ndarray,
    h: float,
    comm: MPI.Comm | None = None,
    coord_scale: float = 1.0,
    v_rf: float = 1.0,
) -> Dict:
    """Compute secular trap frequencies from the pseudopotential Hessian.

    Parameters
    ----------
    Psi         : pseudopotential computed with normalised φ (V_RF = 1 V)
    m_kg        : ion mass in kg
    r0          : trap minimum coordinates in mesh units
    h           : finite-difference step in mesh units (auto-scaled internally)
    coord_scale : mesh unit expressed in metres, e.g. 1e-6 for µm, 1e-3 for mm.
                  Required to convert the Hessian from J/[mesh_unit]² to J/m².
    v_rf        : actual RF voltage amplitude in volts.  The normalised Ψ
                  (computed at V_RF=1) is scaled by v_rf² before frequencies
                  are extracted.

    Returns
    -------
    dict with omega_rad_s and freq_hz in physical SI units.
    """
    H = numerical_hessian(Psi, r0=r0, h=h, comm=comm)
    eigvals, eigvecs = np.linalg.eigh(H)

    # Convert Hessian from J/[mesh_unit]² → J/m², then scale for actual V_RF.
    # Ψ_physical = v_rf² × Ψ_normalised
    # H_physical [J/m²] = v_rf² × H_mesh [J/mesh_unit²] / coord_scale²
    # ω² = H_physical / m  →  ω = v_rf / coord_scale × sqrt(H_mesh / m)
    scale = v_rf / coord_scale
    omega = scale * np.sqrt(np.clip(eigvals, 0.0, None) / m_kg)
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
