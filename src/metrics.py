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
    tol_rel: float = 1e-5,
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
    _stagnant = 0
    _stop_reason = "max_rounds"

    for _round in range(n_rounds):
        f_before_round = f_cur
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
        # Adaptive early stopping: stop if per-round improvement is negligible.
        # tol_rel=1e-5 is conservative — won't trigger unless already converged.
        _improvement = f_before_round - f_cur
        if abs(f_cur) > 0 and _improvement < tol_rel * abs(f_cur):
            _stagnant += 1
        else:
            _stagnant = 0
        if _stagnant >= 2:
            _stop_reason = "tiny_improvement"
            break

    _eff_comm = comm if comm is not None else Psi.function_space.mesh.comm
    if _eff_comm.rank == 0 and _stop_reason != "max_rounds":
        print(f"[refine] early stop ({_stop_reason}) after {_round + 1}/{n_rounds} rounds  "
              f"f_cur={f_cur:.4e}")

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
    transport_dir: Optional[np.ndarray] = None,
    transport_mode: str = "fast",
    skip_depth_y: bool = False,
    skip_transport_scan: bool = False,
) -> Dict:
    """Estimate radial trap depth and transport barrier as separate metrics.

    Radial trap depth — two definitions
    ------------------------------------
    Broad: interior-barrier rays with d[z] ≥ 0, |d·t_dir| < 0.7,
           barrier z > 10 % of trap height, Tukey-fence IQR outlier rejection.
    Core : as broad, plus d[z] > 0.3 (strongly upward) AND
           |d·t_dir| < 0.3 (mostly perpendicular to trap axis).
           ``radial_depth_core_eV`` (median of core rays) is the primary
           paper-comparison metric corresponding to the linear-region radial
           well depth.

    Transport / junction barrier — tiered by transport_mode
    --------------------------------------------------------
    transport_mode="fast" (default):
        eigvec-scan + x-scan only.  ~2×nsamples Ψ evals per direction.
        Suppresses verbose ray listings and CTC path diagnostics.
        Use for routine runs and parameter sweeps.
    transport_mode="full":
        eigvec-scan + x-scan + CTC-like height-following path.
        CTC cost is capped: 30 x-steps, 2 transverse scales × 2 rounds,
        early stopping after 10 consecutive descending steps post-peak.
        Prints full ray listings and CTC path diagnostics for validation.

    Parameters
    ----------
    transport_dir : optional unit vector (mesh-gdim)
        Eigenvector of the weakest secular mode; used for the eigvec scan and
        for excluding transport-like rays from radial metrics.  Defaults to x.
    transport_mode : "fast" | "full"
        Controls which transport metrics are computed and verbosity level.
    """
    domain = Psi.function_space.mesh
    gdim = domain.geometry.dim
    r0 = np.asarray(r0, dtype=np.float64)
    psi0 = float(eval_function_at_points(Psi, np.array([r0]), comm=comm)[0])
    if not np.isfinite(psi0):
        raise ValueError("r0 is outside mesh or Psi could not be evaluated at r0.")
    psi0 = max(psi0, 0.0)

    if gdim == 3:
        dirs = _fibonacci_sphere(max(6, int(nrays)))
    elif gdim == 2:
        ang = np.linspace(0.0, 2.0 * np.pi, max(6, int(nrays)), endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        dirs = np.array([[1.0], [-1.0]], dtype=np.float64)

    dt = float(ray_length) / max(int(nsamples) - 1, 1)
    ts = np.linspace(dt, float(ray_length), int(nsamples))

    # Normalised transport direction
    if transport_dir is not None:
        t_dir = np.asarray(transport_dir, dtype=np.float64)[:gdim].copy()
        t_dir_source = "eigvec"
    else:
        t_dir = np.zeros(gdim, dtype=np.float64); t_dir[0] = 1.0
        t_dir_source = "x-axis"
    t_dir = t_dir / (np.linalg.norm(t_dir) + 1e-30)

    z_r0 = float(r0[2]) if gdim >= 3 else 0.0
    z_barrier_min = max(0.0, z_r0 * 0.1)

    # ── 1-D straight-line scan ────────────────────────────────────────────────
    def _scan_barrier_1d(direction: np.ndarray) -> Tuple[float, bool, float, Optional[np.ndarray]]:
        """Return (barrier_FEM, is_interior, t_at_max, r_at_max)."""
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-30)
        pts = r0[None, :] + ts[:, None] * d[None, :]
        vals = eval_function_at_points(Psi, pts, comm=comm)
        finite = np.isfinite(vals)
        if int(finite.sum()) < 3:
            return np.nan, False, np.nan, None
        vals_f = vals[finite]
        finite_indices = np.where(finite)[0]
        argmax_local = int(np.argmax(vals_f))
        argmax_global = int(finite_indices[argmax_local])
        last_finite_idx = int(finite_indices[-1])
        if argmax_global == 0:
            return np.nan, False, np.nan, None
        max_psi = float(vals_f[argmax_local])
        t_max = float(ts[argmax_global])
        return max_psi - psi0, (argmax_global < last_finite_idx), t_max, r0 + t_max * d

    # ── CTC-like height-following scan ────────────────────────────────────────
    # Walks in x; at each x step uses multi-scale coordinate descent in the
    # transverse plane (y, z for 3-D) to track the Ψ valley floor.
    # Returns (barrier_FEM, is_interior, path_data, argmax_idx).
    # path_data: list of (x, y, z, psi_FEM) tuples including the starting point.
    def _ctc_scan(x_sign: int, n_steps: int = 30) -> Tuple[float, bool, list, int]:
        h_mesh = _estimate_cell_h(domain)
        h_yz_base = h_mesh * 1.5
        x_step = float(x_sign) * ray_length / n_steps
        cur_pos = r0.copy()

        def _entry(pos: np.ndarray, psi: float) -> tuple:
            return (float(pos[0]),
                    float(pos[1]) if gdim > 1 else 0.0,
                    float(pos[2]) if gdim > 2 else 0.0,
                    float(psi))

        path: list = [_entry(r0, psi0)]
        peak_idx = 0
        steps_past_peak = 0

        for k in range(1, n_steps + 1):
            trial = cur_pos.copy()
            trial[0] = r0[0] + k * x_step
            cur_v = float(eval_function_at_points(Psi, trial[None, :], comm=comm)[0])
            if not np.isfinite(cur_v):
                break   # left the mesh

            # Reduced: 2 scales × 2 rounds (was 3 × 3)
            for h_scale in (2.0, 1.0):
                h = h_yz_base * h_scale
                for _ in range(2):
                    for ax in range(1, gdim):
                        for sign in (+1.0, -1.0):
                            cand = trial.copy(); cand[ax] += sign * h
                            v = float(eval_function_at_points(Psi, cand[None, :], comm=comm)[0])
                            if np.isfinite(v) and v < cur_v:
                                cur_v = v; trial = cand

            path.append(_entry(trial, cur_v))
            cur_pos = trial.copy()

            # Early stopping: break after 10 consecutive descending steps past peak
            if cur_v > path[peak_idx][3]:
                peak_idx = len(path) - 1
                steps_past_peak = 0
            else:
                steps_past_peak += 1
                if steps_past_peak >= 10:
                    break

        if len(path) < 3:
            return np.nan, False, path, 0

        psi_arr = np.array([e[3] for e in path], dtype=np.float64)
        argmax = int(np.argmax(psi_arr))
        if argmax == 0:
            return np.nan, False, path, 0
        return float(psi_arr[argmax]) - psi0, (argmax < len(path) - 1), path, argmax

    # ── Ray scan loop ─────────────────────────────────────────────────────────
    # entry: (depth_FEM, d_unit, dir_class, is_phys, t_barrier, r_barrier)
    all_rays: list = []
    n_open = 0; n_electrode_hit = 0; n_physical = 0

    for d in dirs:
        d = np.asarray(d, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-30)
        pts = r0[None, :] + ts[:, None] * d[None, :]
        vals = eval_function_at_points(Psi, pts, comm=comm)
        finite = np.isfinite(vals)
        if int(finite.sum()) < 3:
            continue
        vals_f = vals[finite]
        finite_indices = np.where(finite)[0]
        argmax_local = int(np.argmax(vals_f))
        argmax_global = int(finite_indices[argmax_local])
        last_finite_idx = int(finite_indices[-1])
        max_psi = float(vals_f[argmax_local])
        if argmax_global == 0:
            n_open += 1; continue
        depth = max_psi - psi0
        t_bar = float(ts[argmax_global])
        r_bar = r0 + t_bar * d
        if gdim >= 3:
            if d[2] > 0.5:    dir_class = "upward"
            elif d[2] < -0.5: dir_class = "downward"
            else:              dir_class = "lateral"
        else:
            dir_class = "lateral"
        is_phys = (argmax_global < last_finite_idx)
        all_rays.append((float(depth), d.copy(), dir_class, is_phys, t_bar, r_bar.copy()))
        if is_phys: n_physical += 1
        else:       n_electrode_hit += 1

    # ── Physical scale ────────────────────────────────────────────────────────
    phys_scale = (v_rf / coord_scale) ** 2
    psi0_phys_J = psi0 * phys_scale

    # ── Radial ray filters ────────────────────────────────────────────────────
    # Broad: interior, non-downward, non-transport, barrier above substrate floor
    # Core:  broad + strongly upward (d[z]>0.3) + tighter transport exclusion (<0.3)
    _BROAD_T_THR = 0.7   # cos threshold for broad transport exclusion (~46°)
    _CORE_DZ    = 0.3    # minimum d[z] for core rays
    _CORE_T_THR = 0.3    # cos threshold for core transport exclusion (~17°)

    def _broad_ok(e) -> bool:
        _, dv, _, ip, _, r_b = e
        if not ip: return False
        if gdim >= 3 and dv[2] < 0: return False
        if abs(float(np.dot(dv, t_dir))) > _BROAD_T_THR: return False
        if gdim >= 3 and float(r_b[2]) < z_barrier_min: return False
        return True

    def _core_ok(e) -> bool:
        _, dv, _, _, _, _ = e
        if not _broad_ok(e): return False
        if gdim >= 3 and dv[2] <= _CORE_DZ: return False
        if abs(float(np.dot(dv, t_dir))) > _CORE_T_THR: return False
        return True

    def _iqr_filter(rays: list) -> Tuple[list, int]:
        if len(rays) < 4:
            return rays, 0
        ds = np.array([e[0] for e in rays])
        q25, q75 = float(np.percentile(ds, 25)), float(np.percentile(ds, 75))
        cut = q75 + 5.0 * (q75 - q25)
        kept = [e for e in rays if e[0] <= cut]
        return kept, len(rays) - len(kept)

    broad_rays, n_broad_out = _iqr_filter([e for e in all_rays if _broad_ok(e)])
    core_rays,  n_core_out  = _iqr_filter([e for e in all_rays if _core_ok(e)])
    broad_sorted = sorted(broad_rays, key=lambda e: e[0])
    core_sorted  = sorted(core_rays,  key=lambda e: e[0])

    down_phys = sorted([e for e in all_rays if e[3] and e[2] == "downward"], key=lambda e: e[0])
    phys_rays = sorted([e for e in all_rays if e[3]],                        key=lambda e: e[0])
    elec_rays = sorted([e for e in all_rays if not e[3]],                    key=lambda e: e[0])
    _n_total = n_open + n_electrode_hit + n_physical

    # ── Print helpers ─────────────────────────────────────────────────────────
    def _print_rays(label: str, rays: list, show: int = 5) -> None:
        nd = len(rays)
        if nd == 0:
            print(f"  {label}: (none)"); return
        print(f"  {label} ({nd} rays, sorted by barrier height):")
        for i, e in enumerate(rays[:show]):
            dep, dv, dc = e[0], e[1], e[2]
            extra = (f"  t_bar={e[4]:.4g}  z_bar={float(e[5][2]):.4g}"
                     if len(e) >= 6 and gdim >= 3 else "")
            print(f"    [{i+1}/{nd}] {dep*phys_scale/_E_CHARGE:.4f} eV "
                  f"({dep:.3e} FEM)  class={dc}  "
                  f"dir={np.asarray(dv).round(3).tolist()}{extra}")
        if nd > 2 * show:
            print(f"    ... ({nd - 2*show} middle omitted) ...")
        for i, e in enumerate(rays[max(0, nd - show):]):
            dep, dv, dc = e[0], e[1], e[2]
            print(f"    [{nd-show+i+1}/{nd}] {dep*phys_scale/_E_CHARGE:.4f} eV "
                  f"({dep:.3e} FEM)  class={dc}  dir={np.asarray(dv).round(3).tolist()}")

    def _print_ctc_path(label: str, barrier: float, interior: bool,
                        path: list, argmax_idx: int) -> None:
        n = len(path)
        b_eV = barrier * phys_scale / _E_CHARGE if np.isfinite(barrier) else float("nan")
        print(f"  CTC-{label}: {n} steps  barrier={b_eV:.6f} eV  interior={interior}")
        if np.isnan(barrier) or n < 2:
            return
        stride = max(1, (n - 1) // 10)
        rows = sorted(set(list(range(0, n, stride)) + [argmax_idx]))
        for k in rows:
            if k >= n: continue
            px, py, pz, pv = path[k]
            tag = " <-- BARRIER" if k == argmax_idx else ""
            print(f"    k={k:3d}: x={px:.4f} y={py:.4f} z={pz:.4f} "
                  f"Psi={pv:.3e} ({pv*phys_scale/_E_CHARGE:.4f} eV){tag}")

    # ── Diagnostics: ray families (full mode only) ───────────────────────────
    if transport_mode == "full":
        print(f"[depth] psi0_FEM={psi0:.3e}  psi0_phys={psi0_phys_J/_E_CHARGE:.3e} eV  "
              f"(v_rf={v_rf} V, coord_scale={coord_scale:.0e} m/mesh_unit)")
        print(f"[depth] ray counts: total={_n_total}  open={n_open}  "
              f"electrode-hit={n_electrode_hit}  physical-barrier={n_physical}")
        n_broad_cands = len(broad_rays) + n_broad_out
        n_core_cands  = len(core_rays)  + n_core_out
        print(f"[depth] broad filter: {len(broad_rays)}/{n_broad_cands} kept  "
              f"{n_broad_out} IQR-outliers excluded  "
              f"(d[z]≥0, |d·t|<{_BROAD_T_THR}, z_bar≥{z_barrier_min:.4g})")
        print(f"[depth] core  filter: {len(core_rays)}/{n_core_cands} kept  "
              f"{n_core_out} IQR-outliers excluded  "
              f"(d[z]>{_CORE_DZ}, |d·t|<{_CORE_T_THR})")
        print("[depth] ── Core radial rays (strongly upward, perpendicular to transport) ──")
        _print_rays("core", core_sorted)
        print("[depth] ── Broad radial rays (d[z]≥0, non-transport, interior barrier) ──")
        _print_rays("broad", broad_sorted)
        print("[depth] ── Downward rays (substrate-facing, excluded) ──")
        _print_rays("downward-phys", down_phys)
        print("[depth] ── Electrode-hit / geometry-blocked rays ──")
        _print_rays("electrode-hit", elec_rays)

    # ── Transport barriers ────────────────────────────────────────────────────
    # skip_transport_scan=True omits these 4 line scans (~800 Psi evals saved)
    # and writes null into the JSON output for traceability.
    if not skip_transport_scan:
        # 1. Eigvec scan
        _ev_p, _ev_ip, _, _ = _scan_barrier_1d(t_dir)
        _ev_n, _ev_in, _, _ = _scan_barrier_1d(-t_dir)
        _ev_c = [(b, bi) for b, bi in ((_ev_p, _ev_ip), (_ev_n, _ev_in)) if np.isfinite(b)]
        if _ev_c:
            _ev_FEM, _ev_int = min(_ev_c, key=lambda x: x[0])
            tb_eigvec_eV: Optional[float] = float(_ev_FEM * phys_scale / _E_CHARGE)
            tb_eigvec_int: bool = bool(_ev_int)
        else:
            tb_eigvec_eV = None; tb_eigvec_int = False

        # 2. Pure x-scan
        _xd = np.zeros(gdim); _xd[0] = 1.0
        _xp, _xip, _, _ = _scan_barrier_1d(_xd)
        _xn, _xin, _, _ = _scan_barrier_1d(-_xd)
        _x_c = [(b, bi) for b, bi in ((_xp, _xip), (_xn, _xin)) if np.isfinite(b)]
        if _x_c:
            _x_FEM, _x_int = min(_x_c, key=lambda x: x[0])
            tb_xscan_eV: Optional[float] = float(_x_FEM * phys_scale / _E_CHARGE)
            tb_xscan_int: bool = bool(_x_int)
        else:
            tb_xscan_eV = None; tb_xscan_int = False
    else:
        tb_eigvec_eV = None; tb_eigvec_int = False
        tb_xscan_eV = None; tb_xscan_int = False

    # 3. Pure z-scan (+z / -z) — reports regardless of electrode-hit classification
    if gdim >= 3:
        _zd = np.zeros(gdim); _zd[2] = 1.0
        _zp, _zip, _, _ = _scan_barrier_1d(_zd)
        _zn, _zin, _, _ = _scan_barrier_1d(-_zd)
        _zp_eV: Optional[float] = float(_zp * phys_scale / _E_CHARGE) if np.isfinite(_zp) else None
        _zn_eV: Optional[float] = float(_zn * phys_scale / _E_CHARGE) if np.isfinite(_zn) else None
        _z_finite = [(b, bi) for b, bi in ((_zp, _zip), (_zn, _zin)) if np.isfinite(b)]
        if _z_finite:
            _z_FEM, _z_int = min(_z_finite, key=lambda x: x[0])
            depth_z_eV: Optional[float] = float(_z_FEM * phys_scale / _E_CHARGE)
            depth_z_interior: bool = bool(_z_int)
        else:
            depth_z_eV = None; depth_z_interior = False
        print(f"[depth] z-axis: +z={(_zp_eV if _zp_eV is not None else float('nan')):.4f} eV "
              f"(interior={_zip})  "
              f"-z={(_zn_eV if _zn_eV is not None else float('nan')):.4f} eV "
              f"(interior={_zin})  → depth_z={depth_z_eV} eV  *** paper-comparable ***")
    else:
        _zp_eV = None; _zn_eV = None; depth_z_eV = None; depth_z_interior = False

    # 4. Pure y-scan (+y / -y) — paper-comparable; skip with skip_depth_y=True
    # skip_depth_y saves ~400 Psi evals; result is null in JSON for traceability.
    if gdim >= 2 and not skip_depth_y:
        _yd = np.zeros(gdim); _yd[1] = 1.0
        _yp, _yip, _, _ = _scan_barrier_1d(_yd)
        _yn, _yin, _, _ = _scan_barrier_1d(-_yd)
        _yp_eV: Optional[float] = float(_yp * phys_scale / _E_CHARGE) if np.isfinite(_yp) else None
        _yn_eV: Optional[float] = float(_yn * phys_scale / _E_CHARGE) if np.isfinite(_yn) else None
        _y_finite = [(b, bi) for b, bi in ((_yp, _yip), (_yn, _yin)) if np.isfinite(b)]
        if _y_finite:
            _y_FEM, _y_int = min(_y_finite, key=lambda x: x[0])
            depth_y_eV: Optional[float] = float(_y_FEM * phys_scale / _E_CHARGE)
            depth_y_interior: bool = bool(_y_int)
        else:
            depth_y_eV = None; depth_y_interior = False
        print(f"[depth] y-axis: +y={(_yp_eV if _yp_eV is not None else float('nan')):.4f} eV "
              f"(interior={_yip})  "
              f"-y={(_yn_eV if _yn_eV is not None else float('nan')):.4f} eV "
              f"(interior={_yin})  → depth_y={depth_y_eV} eV")
    else:
        _yp_eV = None; _yn_eV = None; depth_y_eV = None; depth_y_interior = False
        if skip_depth_y:
            print("[depth] y-axis: skipped (skip_depth_y=True)")

    # 5. CTC-like height-following scan (full mode only)
    if transport_mode == "full":
        print("[depth] ── CTC-like path scan (height-following) ──")
        _cp, _cip, _cpd, _cpai = _ctc_scan(+1)
        _cn, _cin, _cnd, _cnai = _ctc_scan(-1)
        _print_ctc_path("+x", _cp, _cip, _cpd, _cpai)
        _print_ctc_path("-x", _cn, _cin, _cnd, _cnai)

        _c_c = [(b, bi) for b, bi in ((_cp, _cip), (_cn, _cin)) if np.isfinite(b)]
        if _c_c:
            _c_FEM, _c_int = min(_c_c, key=lambda x: x[0])
            tb_ctc_eV: Optional[float] = float(_c_FEM * phys_scale / _E_CHARGE)
            tb_ctc_int: bool = bool(_c_int)
            _best_path = _cpd if (np.isfinite(_cp) and (not np.isfinite(_cn) or _cp <= _cn)) else _cnd
            _best_ai   = _cpai if (np.isfinite(_cp) and (not np.isfinite(_cn) or _cp <= _cn)) else _cnai
            _ctc_bx = float(_best_path[_best_ai][0]) if _best_ai < len(_best_path) else None
            _ctc_bz = float(_best_path[_best_ai][2]) if _best_ai < len(_best_path) else None
        else:
            tb_ctc_eV = None; tb_ctc_int = False
            _ctc_bx = None; _ctc_bz = None
    else:
        tb_ctc_eV = None; tb_ctc_int = False
        _ctc_bx = None; _ctc_bz = None

    def _fmt_tb(label: str, eV: Optional[float], interior: bool) -> str:
        return f"  {label}: n/a" if eV is None else f"  {label}: {eV:.6f} eV  interior={interior}"

    if not skip_transport_scan:
        print(f"[depth] transport axis = {t_dir.round(3).tolist()} (source: {t_dir_source})")
        print("[depth] ── Transport barrier summary ──")
        print(_fmt_tb("eigvec-scan ", tb_eigvec_eV, tb_eigvec_int))
        print(_fmt_tb("x-scan      ", tb_xscan_eV,  tb_xscan_int))
        if transport_mode == "full":
            print(_fmt_tb("ctc-like    ", tb_ctc_eV,    tb_ctc_int))
            if _ctc_bx is not None:
                print(f"  ctc barrier at: x={_ctc_bx:.4f}  z={_ctc_bz:.4f} (mesh units)")
    else:
        print("[depth] transport scan: skipped (skip_transport_scan=True)")

    # ── Radial depth statistics ───────────────────────────────────────────────
    def _ray_stats(rays: list, label: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not rays:
            return None, None, None
        evs = np.array([e[0] * phys_scale / _E_CHARGE for e in rays])
        mn, md, mx = float(evs.min()), float(np.median(evs)), float(evs.max())
        print(f"[depth] {label}: min={mn:.4f} eV  median={md:.4f} eV  max={mx:.4f} eV  "
              f"({len(rays)} rays)")
        return mn, md, mx

    if transport_mode == "full":
        core_min_eV,  core_med_eV,  core_max_eV  = _ray_stats(core_rays,  "radial_core ")
        broad_min_eV, broad_med_eV, broad_max_eV = _ray_stats(broad_rays, "radial_broad")
    else:
        def _ray_stats_silent(rays):
            if not rays:
                return None, None, None
            evs = np.array([e[0] * phys_scale / _E_CHARGE for e in rays])
            return float(evs.min()), float(np.median(evs)), float(evs.max())
        core_min_eV,  core_med_eV,  core_max_eV  = _ray_stats_silent(core_rays)
        broad_min_eV, broad_med_eV, broad_max_eV = _ray_stats_silent(broad_rays)

    if not core_rays:
        warnings.warn(
            "estimate_trap_depth_by_rays: no core radial rays found after filtering. "
            "radial_depth_core_eV will be null.  Increase --depth-nrays or check r0.",
            RuntimeWarning, stacklevel=2,
        )
    if not broad_rays:
        warnings.warn(
            "estimate_trap_depth_by_rays: no broad radial rays found after filtering. "
            "radial_depth_broad_* will be null.",
            RuntimeWarning, stacklevel=2,
        )

    # ── Debug-only values ─────────────────────────────────────────────────────
    best_legacy_eV: Optional[float] = (
        float(phys_rays[0][0] * phys_scale / _E_CHARGE) if phys_rays else None
    )
    best_legacy_dir: Optional[list] = phys_rays[0][1].tolist() if phys_rays else None
    best_raw_eV: Optional[float] = (
        float(min(e[0] for e in all_rays) * phys_scale / _E_CHARGE) if all_rays else None
    )

    return {
        "r0_m": r0.tolist(),
        "Psi0_J": float(psi0_phys_J),
        # ── Axis-specific depth (paper-comparable) ──
        "depth_z_eV": depth_z_eV,
        "depth_z_plus_eV": _zp_eV,
        "depth_z_minus_eV": _zn_eV,
        "depth_z_interior": depth_z_interior,
        "depth_y_eV": depth_y_eV,
        "depth_y_plus_eV": _yp_eV,
        "depth_y_minus_eV": _yn_eV,
        "depth_y_interior": depth_y_interior,
        # ── Paper-comparison metrics ──
        "radial_depth_core_eV": core_med_eV,
        "radial_depth_core_min_eV": core_min_eV,
        "radial_depth_core_max_eV": core_max_eV,
        "n_core_radial_rays": len(core_rays),
        "n_core_outliers_excluded": n_core_out,
        "transport_barrier_ctc_like_eV": tb_ctc_eV,
        "transport_barrier_ctc_interior": tb_ctc_int,
        "ctc_barrier_x_mesh": _ctc_bx,
        "ctc_barrier_z_mesh": _ctc_bz,
        # ── Secondary transport metrics ──
        "transport_barrier_eigvec_eV": tb_eigvec_eV,
        "transport_barrier_eigvec_interior": tb_eigvec_int,
        "transport_barrier_xscan_eV": tb_xscan_eV,
        "transport_barrier_xscan_interior": tb_xscan_int,
        "transport_dir_used": t_dir.tolist(),
        "transport_dir_source": t_dir_source,
        # ── Broad radial depth (diagnostics) ──
        "radial_depth_broad_min_eV": broad_min_eV,
        "radial_depth_broad_median_eV": broad_med_eV,
        "radial_depth_broad_max_eV": broad_max_eV,
        "n_broad_radial_rays": len(broad_rays),
        "n_broad_outliers_excluded": n_broad_out,
        # ── Debug-only ──
        "depth_all_phys_min_eV": best_legacy_eV,
        "depth_raw_eV": best_raw_eV,
        "worst_direction": best_legacy_dir,
        "ray_length_mesh": float(ray_length),
        "nrays": int(nrays),
        "nsamples": int(nsamples),
        "n_open_boundary_rays": n_open,
        "n_electrode_hit_rays": n_electrode_hit,
        "n_physical_barrier_rays": n_physical,
        "n_downward_physical_rays": len(down_phys),
    }
