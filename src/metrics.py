from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
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
      (Gibbs-like undershoot near discontinuities).  The raw (unclipped) field
      is returned so that curvature information near the RF null is preserved
      for Hessian / secular-frequency computation.  Callers that need a
      non-negative field (e.g. depth estimation, visualisation) should clip
      explicitly with ``np.maximum(Psi.x.array, 0)``.
    """
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


def refine_minimum(
    Psi: fem.Function,
    r0_init: np.ndarray,
    *,
    bounds_lo: Optional[np.ndarray] = None,
    bounds_hi: Optional[np.ndarray] = None,
    comm: MPI.Comm | None = None,
    h: Optional[float] = None,
    n_rounds: int = 5,
) -> Tuple[np.ndarray, float]:
    """Coordinate-descent polishing of an initial Ψ minimum estimate.

    Performs n_rounds of coordinate descent, halving h each round.
    Bounds are enforced by hard clamping. Requires ~2*gdim*n_rounds Ψ evals.

    Parameters
    ----------
    r0_init   : initial guess in mesh units (from centroid finder)
    bounds_lo : lower bound per axis (mesh units); None means no bound
    bounds_hi : upper bound per axis (mesh units); None means no bound
    h         : initial step size in mesh units; defaults to 0.4×h_mesh
    n_rounds  : number of halving rounds (each halves h)
    """
    r = np.asarray(r0_init, dtype=np.float64).copy()
    gdim = len(r)

    if h is None:
        h = _estimate_cell_h(Psi.function_space.mesh) * 0.4

    lo = bounds_lo if bounds_lo is not None else np.full(gdim, -np.inf)
    hi = bounds_hi if bounds_hi is not None else np.full(gdim, +np.inf)

    def psi_at(pt: np.ndarray) -> float:
        v = eval_function_at_points(Psi, np.array([pt], dtype=np.float64), comm=comm)[0]
        return float(v) if np.isfinite(v) else np.inf

    f_cur = psi_at(r)

    for _round in range(n_rounds):
        for i in range(gdim):
            e = np.zeros(gdim); e[i] = 1.0
            r_p = np.clip(r + h * e, lo, hi)
            r_m = np.clip(r - h * e, lo, hi)
            f_p = psi_at(r_p)
            f_m = psi_at(r_m)
            if f_p < f_cur and f_p <= f_m:
                r = r_p; f_cur = f_p
            elif f_m < f_cur:
                r = r_m; f_cur = f_m
        h *= 0.5

    return r, f_cur


def find_minimum_cg1(
    Psi: fem.Function,
    comm: MPI.Comm | None = None,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    refine: bool = True,
) -> TrapMinimum:
    """Find the RF null — the trap centre where Ψ ≈ 0.

    The CG1 L2-projection of |∇φ|² can make thousands of DOFs
    indistinguishable from zero (Gibbs ringing + floating-point limits).
    A naïve ``argmin(|Ψ|)`` picks an arbitrary near-zero DOF, often at a
    mesh boundary.

    Strategy: collect all interior DOFs in the bottom 5 % by ``|Ψ|``,
    compute their spatial centroid (which clusters around the true RF null),
    then select the interior DOF nearest that centroid.

    Parameters
    ----------
    x_min, x_max : float, optional
        Restrict the search to DOFs with x-coordinate in [x_min, x_max]
        (mesh units).  For multi-junction meshes this lets you pin the search
        to a specific arm or linear segment.  Example for the right outer arm
        of a 2-junction 0.6 mm-pitch mesh: x_min=0.65, x_max=1.05.
    y_min, y_max : float, optional
        Analogous y-coordinate restriction (mesh units).
    z_min, z_max : float, optional
        Restrict the low-Ψ cluster search to DOFs whose z-coordinate
        (3rd axis, index 2) lies within [z_min, z_max] (mesh units).
        **Required when using Neumann outer BC with a large domain** — the RF
        pseudopotential decays smoothly to zero far from the electrodes, so
        without a z-bound the 5th-percentile threshold captures the diffuse
        far-field cloud instead of the tight RF-null cluster near the trap.
        Typical usage: z_max = z_electrode_top + margin (in mesh units), where
        margin ≈ 0.15 mm for single-junction, 0.20 mm for multi-junction.
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
    gdim = domain.geometry.dim
    imap = V.dofmap.index_map
    n_owned = imap.size_local

    # Exclude DOFs on boundary facets (electrode surfaces).
    domain.topology.create_connectivity(fdim, tdim)
    bnd_facets = dmesh.exterior_facet_indices(domain.topology)
    bnd_dofs = fem.locate_dofs_topological(V, fdim, bnd_facets)
    interior_mask = np.ones(n_owned, dtype=bool)
    owned_bnd = bnd_dofs[bnd_dofs < n_owned]
    if owned_bnd.size > 0:
        interior_mask[owned_bnd] = False

    local_vals = Psi.x.array[:n_owned]
    interior_idx = np.where(interior_mask)[0]

    if interior_idx.size == 0:
        raise RuntimeError(
            "No interior DOFs found. Check mesh and boundary markers."
        )

    raw_coords = V.tabulate_dof_coordinates().reshape(-1, 3)[:, :gdim]

    # ── Centroid of the low-|Ψ| cluster ──────────────────────────────────
    # Only consider non-negative interior DOFs for the cluster.
    # Gibbs undershoots near grounded electrode surfaces produce Ψ < 0;
    # the true RF null has Ψ → 0⁺.  Including negative DOFs drags the
    # centroid toward electrode surfaces when DC electrodes are grounded.
    nonneg_mask = local_vals[:n_owned] >= 0
    cluster_mask = interior_mask & nonneg_mask

    # Spatial restriction — critical with Neumann outer BC and multi-junction meshes.
    # Without bounds the far-field Ψ → 0 region (or a remote junction's null)
    # contaminates the 5th-percentile threshold and pulls the centroid away from
    # the intended trap minimum.
    #
    # Axis index mapping (all in mesh units):
    #   0 → x  (along junction array / linear-arm direction)
    #   1 → y  (transverse to array)
    #   2 → z  (normal to electrode plane; most critical bound)
    _spatial_bounds = [
        (0, x_min, x_max),
        (1, y_min, y_max),
        (2, z_min, z_max),
    ]
    for _axis, _lo, _hi in _spatial_bounds:
        if _axis >= gdim:
            continue
        if _lo is None and _hi is None:
            continue
        _c = raw_coords[:n_owned, _axis]
        if _lo is not None:
            cluster_mask &= _c >= _lo
        if _hi is not None:
            cluster_mask &= _c <= _hi

    cluster_idx = np.where(cluster_mask)[0]
    if cluster_idx.size == 0:
        # Fallback: relax spatial constraint but keep non-negative filter
        warnings.warn(
            "find_minimum_cg1: no DOFs in spatially-restricted non-negative cluster "
            f"(x=[{x_min},{x_max}], y=[{y_min},{y_max}], z=[{z_min},{z_max}]). "
            "Falling back to all interior DOFs.",
            RuntimeWarning,
            stacklevel=2,
        )
        cluster_idx = interior_idx

    cluster_abs = np.abs(local_vals[cluster_idx])

    # Gather the global 5th-percentile threshold across all MPI ranks.
    local_sorted = np.sort(cluster_abs)
    all_sorted = comm.allgather(local_sorted)
    global_abs = np.concatenate(all_sorted)
    threshold = float(np.percentile(global_abs, 5))
    # Guard against a degenerate threshold of exactly zero.
    threshold = max(threshold, float(global_abs.max()) * 1e-10,
                    float(np.finfo(np.float64).tiny))

    low_mask = cluster_abs <= threshold
    low_dofs = cluster_idx[low_mask]
    low_coords = raw_coords[low_dofs]

    local_sum = low_coords.sum(axis=0).astype(np.float64)
    local_count = np.array(float(low_coords.shape[0]), dtype=np.float64)
    global_sum = np.empty(gdim, dtype=np.float64)
    global_count = np.array(0.0, dtype=np.float64)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_count, global_count, op=MPI.SUM)

    if global_count < 1.0:
        raise RuntimeError("No DOFs below threshold — Psi field may be trivial.")
    centroid = global_sum / global_count

    # ── Evaluate Ψ at the centroid ────────────────────────────────────────
    psi_at_centroid = eval_function_at_points(
        Psi, np.array([centroid], dtype=np.float64), comm=comm
    )[0]

    # ── Optional coordinate-descent refinement ────────────────────────────
    # The cluster centroid is a robust initial guess but is not in general the
    # exact Ψ minimum.  Polishing it with a few rounds of coordinate descent
    # (with the same spatial bounds) ensures r0 sits at the true local Ψ dip,
    # which makes the Hessian finite-difference accurate.
    r0 = centroid.copy()
    psi_r0 = float(psi_at_centroid)

    if refine:
        _lo_list = [x_min, y_min, z_min][:gdim]
        _hi_list = [x_max, y_max, z_max][:gdim]
        _lo = np.array([v if v is not None else -np.inf for v in _lo_list])
        _hi = np.array([v if v is not None else +np.inf for v in _hi_list])
        r0_ref, psi_ref = refine_minimum(
            Psi, centroid, bounds_lo=_lo, bounds_hi=_hi, comm=comm
        )
        # Accept refined result only if it genuinely reduces Ψ.
        if np.isfinite(psi_ref) and psi_ref <= psi_r0:
            r0 = r0_ref
            psi_r0 = psi_ref

    # ── Bounds validation ─────────────────────────────────────────────────
    # Reject the minimum if it lies outside the requested search box.  This
    # catches edge cases where the refinement stepped slightly outside a bound
    # (shouldn't happen with hard clamping, but guard anyway).
    _bounds_check = [
        (0, x_min, x_max, "x"),
        (1, y_min, y_max, "y"),
        (2, z_min, z_max, "z"),
    ]
    for _ax, _lo_v, _hi_v, _name in _bounds_check:
        if _ax >= gdim:
            continue
        _coord = float(r0[_ax])
        if _lo_v is not None and _coord < _lo_v - 1e-10:
            warnings.warn(
                f"find_minimum_cg1: refined r0.{_name}={_coord:.4g} is below "
                f"{_name}_min={_lo_v:.4g} (mesh units).  "
                "Clamping to bound — consider tightening the search window.",
                RuntimeWarning, stacklevel=2,
            )
            r0[_ax] = _lo_v
        if _hi_v is not None and _coord > _hi_v + 1e-10:
            warnings.warn(
                f"find_minimum_cg1: refined r0.{_name}={_coord:.4g} is above "
                f"{_name}_max={_hi_v:.4g} (mesh units).  "
                "Clamping to bound — consider tightening the search window.",
                RuntimeWarning, stacklevel=2,
            )
            r0[_ax] = _hi_v

    # Re-evaluate Ψ after any clamping.
    psi_r0 = float(eval_function_at_points(
        Psi, np.array([r0], dtype=np.float64), comm=comm
    )[0])

    # ── Nearest-DOF bookkeeping (diagnostics only) ────────────────────────
    interior_coords = raw_coords[interior_idx]
    dists = np.linalg.norm(interior_coords - r0, axis=1)
    local_best = int(np.argmin(dists))
    local_min_dist = float(dists[local_best])
    local_idx = int(interior_idx[local_best])

    rank = comm.rank
    candidates = comm.allgather((local_min_dist, rank, local_idx))
    _, best_rank, best_lidx = min(candidates, key=lambda t: (t[0], t[1]))

    return TrapMinimum(
        r_min=np.array(r0, dtype=np.float64),
        psi_min=float(psi_r0),
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
) -> Tuple[np.ndarray, float]:
    """Compute the Hessian of Psi at r0 by central finite differences.

    h is auto-scaled if it looks inappropriate for the mesh's physical units
    (e.g. mesh in mm but h specified in metres).  Each failed attempt halves h;
    up to max_tries halvings are tried before raising.

    Returns
    -------
    (H, h_used) : (ndarray, float)
        H       — symmetric Hessian matrix
        h_used  — the actual finite-difference step that was used (may differ
                  from the input h if auto-scaling was applied)
    """
    r0 = np.asarray(r0, dtype=np.float64)
    gdim = int(r0.shape[0])
    domain = Psi.function_space.mesh

    # Auto-scale h to mesh geometry if the supplied value is wildly off.
    h_mesh = _estimate_cell_h(domain)
    h = float(h)
    # Allow steps as small as 0.05 × h_mesh (was 0.5 × h_mesh).
    # For sub-cell trap heights (e.g. 82 µm ion with 50 µm cells) a step of
    # ~0.1 × h_mesh is the right order of magnitude; the old 0.5× lower bound
    # would silently inflate h to 2 × h_mesh, spanning well outside the
    # quadratic regime of Ψ.
    if h <= 0.0 or not np.isfinite(h) or h < h_mesh * 0.05 or h > h_mesh * 100.0:
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
            return H, hk

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
    H, h_used = numerical_hessian(Psi, r0=r0, h=h, comm=comm)
    eigvals, eigvecs = np.linalg.eigh(H)

    # Derivation:
    #   Ψ_phys(x_SI) = (v_rf/coord_scale)² × Ψ_norm(x_mesh)
    #   ∂/∂x_SI = (1/coord_scale) ∂/∂x_mesh
    #   H_phys [J/m²] = (v_rf/coord_scale)² × (1/coord_scale²) × H_mesh
    #                 = v_rf² × H_mesh / coord_scale⁴
    #   ω² = H_phys / m  →  ω = (v_rf / coord_scale²) × sqrt(H_mesh / m)
    scale = v_rf / coord_scale ** 2
    omega = scale * np.sqrt(np.clip(eigvals, 0.0, None) / m_kg)
    freq = omega / (2.0 * np.pi)
    return SecularFrequencies(
        r0=np.array(r0, dtype=np.float64),
        h=float(h_used),           # actual step used, after any auto-scaling
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
    coord_scale: float = 1.0,
    v_rf: float = 1.0,
) -> Dict:
    """Estimate trap depth as the minimum Ψ barrier across all ray directions.

    With Neumann outer BC the pseudopotential decays smoothly to zero beyond the
    electrode region.  Rays that point toward the outer boundary see Ψ
    **decreasing from the very first step** — the maximum is at t = 0 (r0 itself),
    giving a spurious "depth" of zero or negative.  These are **open-boundary
    rays** and are excluded from the depth estimate: a ray is counted only if its
    Ψ maximum is found at an interior sample (index > 0), i.e., the ray first
    climbs before (optionally) falling back.

    If every ray is an open-boundary ray a warning is emitted and the result
    contains ``depth_eV = null`` rather than a misleading 0.
    """
    domain = Psi.function_space.mesh
    gdim = domain.geometry.dim
    r0 = np.asarray(r0, dtype=np.float64)
    psi0 = float(eval_function_at_points(Psi, np.array([r0]), comm=comm)[0])
    if not np.isfinite(psi0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")
    # CG2 interpolation can yield tiny negative values even on a clipped field;
    # clamp here so depth = max_psi - psi0 is never inflated by a negative base.
    psi0 = max(psi0, 0.0)

    if gdim == 3:
        dirs = _fibonacci_sphere(max(6, int(nrays)))
    elif gdim == 2:
        ang = np.linspace(0.0, 2.0 * np.pi, max(6, int(nrays)), endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        dirs = np.array([[1.0], [-1.0]], dtype=np.float64)

    # Start at t = one step past r0 so the first sample is not r0 itself.
    # This prevents open-boundary misclassification for rays that immediately
    # exit the mesh (only r0 = ts[0] would be finite → argmax at index 0).
    dt = float(ray_length) / max(int(nsamples) - 1, 1)
    ts = np.linspace(dt, float(ray_length), int(nsamples))

    # ── Per-ray record: (depth_FEM, direction_unit_vec, direction_class, is_phys)
    # direction_class: "upward" | "lateral" | "downward"
    # is_phys:  True  → local max in interior (real saddle-point barrier)
    #           False → argmax at last finite sample (ray exits mesh ascending,
    #                   i.e. hits solid electrode — not a physical escape route)
    _RayRec = tuple   # (depth_FEM, dir_vec, dir_class, is_phys)
    all_rays: list = []

    n_open = 0
    n_electrode_hit = 0
    n_physical = 0

    # Physical-barrier trackers (used for depth_eV)
    best_phys_depth = np.inf
    best_phys_dir: Optional[np.ndarray] = None
    best_phys_max = np.nan
    best_phys_used = 0

    for d in dirs:
        d = np.asarray(d, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-30)
        pts = r0[None, :] + ts[:, None] * d[None, :]
        vals = eval_function_at_points(Psi, pts, comm=comm)
        finite = np.isfinite(vals)
        n_fin = int(finite.sum())
        if n_fin < 3:
            continue

        vals_f = vals[finite]
        finite_indices = np.where(finite)[0]
        argmax_local = int(np.argmax(vals_f))
        argmax_global = int(finite_indices[argmax_local])
        last_finite_idx = int(finite_indices[-1])
        max_psi = float(vals_f[argmax_local])

        # ── Open-boundary ray ────────────────────────────────────────────────
        # argmax at the very first finite sample means Ψ only decreases along
        # this direction — pointing toward the Neumann far-field.  No barrier.
        if argmax_global == 0:
            n_open += 1
            continue

        depth = max_psi - psi0

        # ── Direction classification ─────────────────────────────────────────
        # Based on the z-component (electrode-normal axis).
        # Lateral rays are the primary transport/escape directions.
        if gdim >= 3:
            if d[2] > 0.5:
                dir_class = "upward"      # toward RF pillar tops
            elif d[2] < -0.5:
                dir_class = "downward"    # toward DC substrate
            else:
                dir_class = "lateral"     # along electrode plane / junction
        else:
            dir_class = "lateral"

        # ── Physical-barrier vs electrode-hit ────────────────────────────────
        # If the argmax is at the LAST finite sample, the ray exited the mesh
        # while Ψ was still at its maximum (i.e., it hit a solid electrode or
        # domain boundary while ascending).  Ions cannot escape through solid
        # metal, so these barriers are not physical escape routes.
        # If the argmax is at an interior point (finite samples exist after it),
        # Ψ rose then fell — a genuine saddle-point barrier.
        is_phys = (argmax_global < last_finite_idx)

        all_rays.append((float(depth), d.copy(), dir_class, is_phys))

        if is_phys:
            n_physical += 1
            if depth < best_phys_depth:
                best_phys_depth = depth
                best_phys_dir = d.copy()
                best_phys_max = max_psi
                best_phys_used = n_fin
        else:
            n_electrode_hit += 1

    # ── Compute physical scale once ──────────────────────────────────────────
    # Ψ_physical [J] = (V_RF / coord_scale)² × Ψ_FEM
    # coord_scale has units of m/mesh_unit (e.g. 1e-3 for mm mesh).
    phys_scale = (v_rf / coord_scale) ** 2
    psi0_phys_J = psi0 * phys_scale

    # ── Diagnostics ─────────────────────────────────────────────────────────
    # Print per-family sorted barrier distributions.
    phys_rays = [(dep, dv, dc) for dep, dv, dc, ip in all_rays if ip]
    elec_rays  = [(dep, dv, dc) for dep, dv, dc, ip in all_rays if not ip]
    phys_rays.sort(key=lambda x: x[0])
    elec_rays.sort(key=lambda x: x[0])

    _n_total = n_open + n_electrode_hit + n_physical
    print(f"[depth] psi0_FEM={psi0:.3e}  psi0_phys={psi0_phys_J / _E_CHARGE:.3e} eV  "
          f"(v_rf={v_rf} V, coord_scale={coord_scale:.0e} m/mesh_unit)")
    print(f"[depth] ray counts: total={_n_total}  open={n_open}  "
          f"electrode-hit={n_electrode_hit}  physical-barrier={n_physical}")

    def _print_rays(label: str, rays: list, show: int = 5) -> None:
        nd = len(rays)
        if nd == 0:
            print(f"  {label}: (none)")
            return
        print(f"  {label} ({nd} rays, sorted by barrier height):")
        for i, (dep, dv, dc) in enumerate(rays[:show]):
            print(f"    [{i+1}/{nd}] {dep * phys_scale / _E_CHARGE:.4f} eV "
                  f"({dep:.3e} FEM)  class={dc}  dir={dv.round(3).tolist()}")
        if nd > 2 * show:
            print(f"    ... ({nd - 2*show} middle rays omitted) ...")
        for i, (dep, dv, dc) in enumerate(rays[max(0, nd - show):]):
            print(f"    [{nd - show + i + 1}/{nd}] {dep * phys_scale / _E_CHARGE:.4f} eV "
                  f"({dep:.3e} FEM)  class={dc}  dir={dv.round(3).tolist()}")

    print("[depth] ── Physical barriers (local max in interior) ──")
    _print_rays("physical", phys_rays)
    print("[depth] ── Electrode-hit rays (monotonic ascent, excluded from depth_eV) ──")
    _print_rays("electrode-hit", elec_rays)

    # ── Warn if no physical barriers found ──────────────────────────────────
    if n_physical == 0:
        warnings.warn(
            f"estimate_trap_depth_by_rays: no physical-barrier rays found "
            f"({n_electrode_hit} electrode-hit, {n_open} open-boundary). "
            "depth_eV will be null.  Increase ray density (--depth-nrays) or "
            "check that r0 is well inside the electrode structure.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "r0_m": r0.tolist(),
            "Psi0_J": float(psi0_phys_J),
            "depth_J": None,
            "depth_eV": None,
            "depth_raw_eV": None,
            "worst_direction": None,
            "ray_max_Psi_J": None,
            "ray_samples_used": None,
            "ray_length_mesh": float(ray_length),
            "nrays": int(nrays),
            "nsamples": int(nsamples),
            "n_open_boundary_rays": n_open,
            "n_electrode_hit_rays": n_electrode_hit,
            "n_physical_barrier_rays": n_physical,
        }

    # ── Depth values ─────────────────────────────────────────────────────────
    depth_phys_J = best_phys_depth * phys_scale
    max_phys_J = best_phys_max * phys_scale

    # Raw minimum across ALL valid rays (including electrode hits) — for debug.
    best_raw_depth = min(dep for dep, _, _, _ in all_rays)
    best_raw_eV = float(best_raw_depth * phys_scale / _E_CHARGE)

    phys_min_eV = float(depth_phys_J / _E_CHARGE)
    phys_max_eV = float(phys_rays[-1][0] * phys_scale / _E_CHARGE)

    print(f"[depth] physical depth (worst escape) = {phys_min_eV:.4f} eV")
    print(f"[depth] physical max barrier           = {phys_max_eV:.4f} eV")
    print(f"[depth] raw min (incl. electrode hits) = {best_raw_eV:.4f} eV  "
          f"(not used for reported depth)")

    return {
        "r0_m": r0.tolist(),
        "Psi0_J": float(psi0_phys_J),
        "depth_J": float(depth_phys_J),
        "depth_eV": float(phys_min_eV),          # min barrier among physical rays
        "depth_max_eV": float(phys_max_eV),       # max barrier among physical rays
        "depth_raw_eV": float(best_raw_eV),       # unfiltered min (for diagnostics)
        "worst_direction": best_phys_dir.tolist(),
        "ray_max_Psi_J": float(max_phys_J),
        "ray_samples_used": int(best_phys_used),
        "ray_length_mesh": float(ray_length),
        "nrays": int(nrays),
        "nsamples": int(nsamples),
        "n_open_boundary_rays": n_open,
        "n_electrode_hit_rays": n_electrode_hit,
        "n_physical_barrier_rays": n_physical,
    }
