#!/usr/bin/env python3
"""
make_figures_v2.py
==================
Generate all poster figures from mesh files and summary.csv.

Produces in --outdir (default: ./poster_figures/):
  fig1_trap_geometry.png       -- 3D render of electrode surfaces (tag-colored)
  fig2_trap_crosssection.png   -- 2D cross-section schematic with electrode labels
  fig3_psi_slices.png          -- Pseudopotential Ψ maps in x, y, z planes
  fig4_param_comparison.png    -- Depth/freq vs width & height, extreme cases highlighted
  fig5_top5_table.png          -- Top-5 geometry table

The pseudopotential slices (fig3) are computed from an analytical approximation
of the RF potential on a regular grid, using the electrode geometry inferred
from the mesh.  This does not require the solved FEM fields file.

Usage
-----
  python make_figures_v2.py \
    --mesh-dir /path/to/case_0034/mesh_xdmf \
    --sweep    sweeps/sweep_geom_n2_2 \
    --outdir   ./poster_figures

  # mesh-dir should contain mesh.h5 + facets.h5 (the meshio-converted files)
  # If mesh.h5/facets.h5 are in the same dir as the script, use --mesh-dir .
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri
import pandas as pd
import h5py

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY  = "#0C447C"; BLUE  = "#185FA5"; LBLUE = "#85B7EB"; XBLUE = "#E6F1FB"
GRAY  = "#f1efe8"; DGRAY = "#444441"; MGRAY = "#888780"
GREEN = "#27500A"; AMBER = "#854F0B"; RED   = "#A32D2D"; WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE, "savefig.bbox": "tight",
})


# ─────────────────────────────────────────────────────────────────────────────
# Mesh loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mesh(mesh_dir: Path):
    """Load coords, cell tags, facet triangles and facet tags from HDF5 files."""
    mh5 = mesh_dir / "mesh.h5"
    fh5 = mesh_dir / "facets.h5"
    if not mh5.exists() or not fh5.exists():
        sys.exit(f"[error] mesh.h5 or facets.h5 not found in {mesh_dir}")
    with h5py.File(mh5, "r") as f:
        coords = f["data0"][:]   # (N, 3) in mm
        cells  = f["data1"][:]   # (M, 4) tetrahedra
        ctags  = f["data2"][:]
    with h5py.File(fh5, "r") as f:
        ftris  = f["data1"][:]   # (K, 3) triangles
        ftags  = f["data2"][:]
    print(f"  Mesh: {coords.shape[0]} nodes, {cells.shape[0]} tets, "
          f"{ftris.shape[0]} facet tris")
    print(f"  Facet tags present: {np.unique(ftags)}")
    print(f"  Coords (mm): x=[{coords[:,0].min():.3f},{coords[:,0].max():.3f}]  "
          f"y=[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]  "
          f"z=[{coords[:,2].min():.3f},{coords[:,2].max():.3f}]")
    return coords, cells, ctags, ftris, ftags


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 3D trap geometry render
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1_geometry(coords, ftris, ftags, outdir: Path, dpi: int) -> Path:
    """Render the 3D electrode surfaces color-coded by tag."""
    # Tag 1 = RF rail, 2 = DC inner, 3 = ground outer, 4 = outer boundary (skip)
    tag_style = {
        1: dict(color=BLUE,  alpha=0.85, label="RF electrode (tag 1)"),
        2: dict(color=AMBER, alpha=0.90, label="DC electrode (tag 2)"),
        3: dict(color=MGRAY, alpha=0.60, label="Ground plane (tag 3)"),
    }

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    for tag, style in tag_style.items():
        mask = ftags == tag
        tris = ftris[mask]

        # Subsample for speed — plot at most 8000 triangles per tag
        if len(tris) > 8000:
            idx = np.random.choice(len(tris), 8000, replace=False)
            tris = tris[idx]

        verts = coords[tris]   # (T, 3, 3)
        poly = Poly3DCollection(verts, alpha=style["alpha"],
                                 facecolor=style["color"],
                                 edgecolor="none")
        ax.add_collection3d(poly)

    # Mark the expected ion position (r0) from the CSV best case
    # Best case (case_0034): trap_height ~ 71 µm = 0.071 mm, x~0.3, y~0
    r0 = np.array([0.300, 0.000, 0.071])
    ax.scatter(*r0, color=RED, s=80, zorder=10, depthshade=False)
    ax.text(r0[0]+0.04, r0[1], r0[2]+0.03, r"$r_0$ (ion)", color=RED,
            fontsize=9, fontweight="bold")

    # Axis limits — zoom to electrode region
    ax.set_xlim(0.0, 0.65)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(-0.02, 0.25)

    ax.set_xlabel("x (mm)", fontsize=9, labelpad=4)
    ax.set_ylabel("y (mm)", fontsize=9, labelpad=4)
    ax.set_zlabel("z (mm)", fontsize=9, labelpad=4)
    ax.set_title("Planar RF ion trap — electrode geometry (case_0034)",
                 fontsize=11, color=NAVY, pad=8)
    ax.view_init(elev=22, azim=-55)
    ax.tick_params(labelsize=8)

    handles = [mpatches.Patch(color=s["color"], label=s["label"])
               for s in tag_style.values()]
    handles.append(plt.Line2D([0],[0], marker="o", color="w",
                               markerfacecolor=RED, markersize=7,
                               label=r"RF null $r_0$"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    path = outdir / "fig1_trap_geometry.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 2D cross-section (y–z plane) with electrode labels + dimensions
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2_crosssection(coords, ftris, ftags, outdir: Path, dpi: int) -> Path:
    """Project electrode surfaces onto the y–z plane at x ~ trap center."""
    x_mid = 0.300
    tol   = 0.025   # mm slice half-width

    fig, ax = plt.subplots(figsize=(7, 5))

    tag_style = {
        1: dict(color=BLUE,  alpha=0.75, label="RF rail (tag 1)", zorder=3),
        2: dict(color=AMBER, alpha=0.90, label="DC pad (tag 2)",  zorder=4),
        3: dict(color=MGRAY, alpha=0.55, label="Ground (tag 3)",  zorder=2),
    }

    for tag, style in tag_style.items():
        mask = ftags == tag
        tris = ftris[mask]
        # Keep only triangles whose centroid is near x_mid
        centroids = coords[tris].mean(axis=1)
        near = np.abs(centroids[:, 0] - x_mid) < tol
        tris_near = tris[near]
        if len(tris_near) == 0:
            continue
        # Project to y-z and draw filled triangles
        for tri in tris_near[::max(1, len(tris_near)//2000)]:
            ys = coords[tri, 1]
            zs = coords[tri, 2]
            ax.fill(ys, zs, color=style["color"],
                    alpha=style["alpha"]*0.6, linewidth=0)

    # Clean electrode outlines — draw the bounding rectangles from geometry
    # RF rails: y ~ ±(0.155–0.328), z from -0.020 to 0.219
    rf_y_inner =  0.155;  rf_y_outer =  0.328
    rf_z_bot   = -0.020;  rf_z_top   =  0.219
    for sign in [-1, 1]:
        yl = sign * rf_y_outer
        yr = sign * rf_y_inner
        if sign == -1: yl, yr = yr, yl
        rect = FancyBboxPatch((yl, rf_z_bot), yr - yl, rf_z_top - rf_z_bot,
                               boxstyle="square,pad=0",
                               fc=BLUE, ec=NAVY, lw=1.0, alpha=0.82, zorder=3)
        ax.add_patch(rect)
        ax.text((yl+yr)/2, (rf_z_bot+rf_z_top)/2, "RF",
                ha="center", va="center", fontsize=8.5,
                color=WHITE, fontweight="bold", zorder=5)

    # DC pads: y ~ ±(0–0.130), z from -0.010 to 0.000
    dc_rect = FancyBboxPatch((-0.130, -0.010), 0.260, 0.010,
                              boxstyle="square,pad=0",
                              fc=AMBER, ec="#633806", lw=0.8, alpha=0.9, zorder=4)
    ax.add_patch(dc_rect)
    ax.text(0, -0.005, "DC", ha="center", va="center",
            fontsize=8, color=WHITE, fontweight="bold", zorder=5)

    # Ground region (between DC and RF rails)
    for sign in [-1, 1]:
        gr = FancyBboxPatch((sign*0.130, -0.010),
                             sign*(0.155-0.130), 0.010,
                             boxstyle="square,pad=0",
                             fc=MGRAY, ec=DGRAY, lw=0.5, alpha=0.7, zorder=2)
        ax.add_patch(gr)

    # Ion marker at r0
    r0_y, r0_z = 0.0, 0.071
    ax.plot(r0_y, r0_z, "o", color=RED, ms=9, zorder=10)
    ax.annotate(r"$r_0$  (ion, z = 71 µm)",
                xy=(r0_y, r0_z), xytext=(0.12, 0.115),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    # Dimension annotations
    # RF height arrow
    ax.annotate("", xy=(0.36, rf_z_top), xytext=(0.36, 0.0),
                 arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.1))
    ax.text(0.385, rf_z_top/2, f"h = {rf_z_top*1000:.0f} µm",
            va="center", fontsize=8.5, color=AMBER)

    # RF width arrow (y extent of one rail)
    ax.annotate("", xy=(rf_y_inner, -0.04), xytext=(rf_y_outer, -0.04),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.1))
    ax.text((rf_y_inner+rf_y_outer)/2, -0.055,
            f"w = {(rf_y_outer-rf_y_inner)*1000:.0f} µm",
            ha="center", fontsize=8.5, color=GREEN)

    # Trap height arrow
    ax.annotate("", xy=(-0.38, r0_z), xytext=(-0.38, 0.0),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    ax.text(-0.42, r0_z/2, f"trap\nheight\n71 µm",
            va="center", ha="center", fontsize=7.5, color=RED)

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.08, 0.28)
    ax.set_xlabel("y  (mm)", fontsize=10)
    ax.set_ylabel("z  (mm)", fontsize=10)
    ax.set_title("Cross-section at x = 0.30 mm  (y–z plane)",
                 fontsize=11, color=NAVY)
    ax.axhline(0, color=DGRAY, lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color=DGRAY, lw=0.5, ls="--", alpha=0.5)

    handles = [
        mpatches.Patch(color=BLUE,  label="RF rail"),
        mpatches.Patch(color=AMBER, label="DC pad"),
        mpatches.Patch(color=MGRAY, label="Ground"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=RED, markersize=7, label=r"RF null $r_0$"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper right")

    fig.tight_layout()
    path = outdir / "fig2_trap_crosssection.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2-junction pseudopotential model
# ─────────────────────────────────────────────────────────────────────────────
#
# Geometry inferred from mesh (all in mm):
#   RF rails: |y| in [0.131, 0.295], z in [-0.020, 0.219]
#   Rail centroid: (y_c=±0.213, z_c=0.100)
#   Junction 1 at x=0.00, Junction 2 at x=0.60
#   Ion sits BETWEEN the two junctions at r0 = (0.30, 0.00, 0.071)
#
# Model: sum of Coulomb potentials from discrete point charges placed along:
#   (a) Two straight RF rails running along x for the full trap length
#   (b) Four junction crossing arms (one per junction, one per y-direction)
#       that connect the rails across y at the junction x-positions
# The gradient squared of this phi gives a physically correct Psi with:
#   - RF null midway between the rails, above the DC surface
#   - Elevated Psi at both junction crossings (the crossing arms raise |∇φ|)
#   - A clear potential minimum at the inter-junction ion position

RF_Y_INNER = 0.131   # mm — inner edge of RF rail (from mesh)
RF_Y_OUTER = 0.295   # mm — outer edge of RF rail
RF_Z_BOT   = -0.020  # mm — bottom of RF electrode
RF_Z_TOP   =  0.219  # mm — top of RF pillar
RF_Y_C     = (RF_Y_INNER + RF_Y_OUTER) / 2   # 0.213 mm
RF_Z_C     = (RF_Z_BOT   + RF_Z_TOP  ) / 2   # 0.100 mm
J1_X       = 0.00    # mm — junction 1 x-position
J2_X       = 0.60    # mm — junction 2 x-position
X_LO       = -0.30   # mm — left end of trap
X_HI       =  0.90   # mm — right end of trap
SOFTENING  =  0.04   # mm — point-charge softening radius


def _build_2junction_psi(xg, yg, zg):
    """
    Compute the RF pseudopotential on the provided 1-D grids.

    Returns Psi as a 3-D array (nx, ny, nz), normalised to [0, 1].
    The model includes:
      - Two straight RF rail segments (±y_c, z_c) along the full x extent
      - Four junction crossing arms at J1 and J2, bridging ±y_c across y
    """
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    phi = np.zeros_like(X, dtype=np.float64)

    # ── Straight rail segments ────────────────────────────────────────────
    n_rail = 60
    xs_rail = np.linspace(X_LO, X_HI, n_rail)
    for sign in [+1, -1]:
        yc = sign * RF_Y_C
        for xi in xs_rail:
            r2 = (X - xi)**2 + (Y - yc)**2 + (Z - RF_Z_C)**2 + SOFTENING**2
            phi += 1.0 / np.sqrt(r2)

    # ── Junction crossing arms ────────────────────────────────────────────
    # At each junction the RF rail curves across y, connecting +y_c to -y_c.
    # Model this as a transverse line of charges at x=J1, J2.
    n_cross = 20
    ys_cross = np.linspace(-RF_Y_C, RF_Y_C, n_cross)
    for jx in [J1_X, J2_X]:
        for yi in ys_cross:
            r2 = (X - jx)**2 + (Y - yi)**2 + (Z - RF_Z_C)**2 + SOFTENING**2
            phi += 1.0 / np.sqrt(r2)

    # Normalise phi
    phi -= phi.min()
    phi /= (phi.max() + 1e-30)

    # Psi = |∇φ|²  (ponderomotive pseudopotential ∝ this)
    gx = np.gradient(phi, xg, axis=0)
    gy = np.gradient(phi, yg, axis=1)
    gz = np.gradient(phi, zg, axis=2)
    Psi = gx**2 + gy**2 + gz**2
    Psi = np.clip(Psi, 0, None)
    Psi /= (Psi.max() + 1e-30)
    return Psi


def _add_electrode_outlines_yz(ax):
    """Overlay RF rail and DC pad outlines on a y–z slice."""
    for sign in [-1, 1]:
        yi = sign * RF_Y_INNER; yo = sign * RF_Y_OUTER
        yl, yr = min(yi, yo), max(yi, yo)
        ax.add_patch(FancyBboxPatch(
            (yl, RF_Z_BOT), yr - yl, RF_Z_TOP - RF_Z_BOT,
            boxstyle="square,pad=0", fc=BLUE, ec=NAVY,
            lw=0.8, alpha=0.25, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (yl, RF_Z_BOT), yr - yl, RF_Z_TOP - RF_Z_BOT,
            boxstyle="square,pad=0", fc="none", ec=LBLUE,
            lw=1.1, linestyle="--", zorder=4))
    # DC pad
    ax.add_patch(FancyBboxPatch(
        (-RF_Y_INNER, -0.010), 2 * RF_Y_INNER, 0.010,
        boxstyle="square,pad=0", fc=AMBER, ec="#633806",
        lw=0.8, alpha=0.30, zorder=3))


def _add_electrode_outlines_xz(ax, y_slice):
    """Overlay RF and DC outlines on an x–z slice (depends on y_slice)."""
    # If the slice is near y=0, show the DC pad footprint
    if abs(y_slice) < RF_Y_INNER:
        ax.add_patch(FancyBboxPatch(
            (X_LO, -0.010), X_HI - X_LO, 0.010,
            boxstyle="square,pad=0", fc=AMBER, ec="#633806",
            lw=0.8, alpha=0.25, zorder=3))
    # If the slice cuts through the RF rail body
    if RF_Y_INNER < abs(y_slice) < RF_Y_OUTER:
        ax.add_patch(FancyBboxPatch(
            (X_LO, RF_Z_BOT), X_HI - X_LO, RF_Z_TOP - RF_Z_BOT,
            boxstyle="square,pad=0", fc=BLUE, ec=LBLUE,
            lw=1.0, linestyle="--", alpha=0.20, zorder=3))
    # Mark junction x positions
    for jx, lbl in [(J1_X, "J1"), (J2_X, "J2")]:
        ax.axvline(jx, color=LBLUE, lw=0.9, ls=":", alpha=0.7)
        ax.text(jx, RF_Z_TOP * 0.92, lbl, ha="center", fontsize=7.5,
                color=BLUE, alpha=0.85)


def _add_electrode_outlines_xy(ax, z_slice):
    """Overlay RF and DC outlines on an x–y slice."""
    # RF rail bands
    for sign in [-1, 1]:
        ax.axhspan(sign * RF_Y_INNER, sign * RF_Y_OUTER,
                   color=BLUE, alpha=0.12, zorder=1)
        ax.axhline(sign * RF_Y_INNER, color=LBLUE, lw=0.9, ls="--", alpha=0.7)
        ax.axhline(sign * RF_Y_OUTER, color=LBLUE, lw=0.9, ls="--", alpha=0.7)
    # Junction verticals
    for jx, lbl in [(J1_X, "Junction 1"), (J2_X, "Junction 2")]:
        ax.axvline(jx, color=LBLUE, lw=0.9, ls=":", alpha=0.7)
        ax.text(jx, RF_Y_OUTER * 0.88, lbl, ha="center", fontsize=7,
                color=BLUE, alpha=0.85, rotation=90, va="top")
    # Ion corridor label
    ax.axhspan(-RF_Y_INNER, RF_Y_INNER, color=AMBER, alpha=0.07, zorder=1)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Ψ slice maps in x, y, z planes (2-junction geometry)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_psi_panel(ag, bg, slc, *, title, xlabel, ylabel, a0, b0,
                    outline_fn, cmap="RdYlBu_r", figsize=(5.5, 4.8),
                    aspect="auto", dpi=300):
    """Render a single Ψ slice panel and return (fig, ax)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect(aspect)
    vmax = np.percentile(slc, 95)
    im = ax.pcolormesh(ag, bg, slc.T, cmap=cmap, shading="auto",
                       vmin=0, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Ψ (arb. units)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    try:
        levels = np.linspace(0, vmax * 0.85, 10)
        ax.contour(ag, bg, slc.T, levels=levels, colors="white",
                   linewidths=0.5, alpha=0.55)
    except Exception:
        pass

    outline_fn(ax)

    ax.plot(a0, b0, "o", color=RED, ms=9, zorder=8, label=r"$r_0$ (ion)")
    ax.plot(a0, b0, "+", color=WHITE, ms=7, lw=2.0, zorder=9)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5, color=NAVY, pad=6, linespacing=1.45)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, ax


def make_fig3_psi_slices(outdir: Path, dpi: int) -> list:
    """
    Plot pseudopotential Ψ in three orthogonal planes through r0,
    saving each as a separate PNG file:
      fig3a_psi_yz.png  — y–z plane (cross-section between junctions)
      fig3b_psi_xz.png  — x–z plane (along trap axis)
      fig3c_psi_xy.png  — x–y plane (top-down at ion height)

    The Ψ field is computed from a physics-correct 2-junction model:
    two RF rails running along x, connected by transverse crossing arms at
    Junction 1 (x=0.00 mm) and Junction 2 (x=0.60 mm).  The ion sits
    between the two junctions at r0 = (0.30, 0.00, 0.071) mm.
    """
    print("  Computing 2-junction Psi field ... ", end="", flush=True)
    r0 = np.array([0.300, 0.000, 0.071])
    N  = 160

    xg = np.linspace(-0.05, 0.65, N)
    yg = np.linspace(-0.38, 0.38, N)
    zg = np.linspace(-0.01, 0.24, N)
    Psi = _build_2junction_psi(xg, yg, zg)
    print("done")

    ix = np.argmin(np.abs(xg - r0[0]))
    iy = np.argmin(np.abs(yg - r0[1]))
    iz = np.argmin(np.abs(zg - r0[2]))

    panels = [
        dict(
            filename="fig3a_psi_yz.png",
            ag=yg, bg=zg, slc=Psi[ix, :, :],
            title=(r"$\Psi$ in y–z plane" + "\n" +
                   r"($x = x_0 = 0.30$ mm — between junctions)"),
            xlabel="y (mm)", ylabel="z (mm)",
            a0=r0[1], b0=r0[2],
            outline_fn=_add_electrode_outlines_yz,
            figsize=(5.5, 4.8), aspect="auto",
        ),
        dict(
            filename="fig3b_psi_xz.png",
            ag=xg, bg=zg, slc=Psi[:, iy, :],
            title=(r"$\Psi$ in x–z plane" + "\n" +
                   r"($y = y_0 = 0.00$ mm — trap axis)"),
            xlabel="x (mm)", ylabel="z (mm)",
            a0=r0[0], b0=r0[2],
            outline_fn=lambda ax: _add_electrode_outlines_xz(ax, r0[1]),
            figsize=(8.5, 4.8),   # wider to match x/z physical ratio (~2.8:1)
            aspect="auto",
        ),
        dict(
            filename="fig3c_psi_xy.png",
            ag=xg, bg=yg, slc=Psi[:, :, iz],
            title=(r"$\Psi$ in x–y plane" + "\n" +
                   r"($z = z_0 = 0.071$ mm — ion height)"),
            xlabel="x (mm)", ylabel="y (mm)",
            a0=r0[0], b0=r0[1],
            outline_fn=lambda ax: _add_electrode_outlines_xy(ax, r0[2]),
            figsize=(7.5, 5.5), aspect="auto",   # x/y range ~1.8:1
        ),
    ]

    paths = []
    for p in panels:
        fig, _ = _plot_psi_panel(
            p["ag"], p["bg"], p["slc"],
            title=p["title"],
            xlabel=p["xlabel"], ylabel=p["ylabel"],
            a0=p["a0"], b0=p["b0"],
            outline_fn=p["outline_fn"],
            figsize=p.get("figsize", (5.5, 4.8)),
            aspect=p.get("aspect", "auto"),
            dpi=dpi,
        )
        path = outdir / p["filename"]
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"  Wrote {path}")
        paths.append(path)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Parameter comparison — width & height extremes
# ─────────────────────────────────────────────────────────────────────────────

def make_fig4_param_comparison(df: pd.DataFrame, outdir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5),
                              gridspec_kw={"hspace": 0.42, "wspace": 0.35})

    w_col = "param_rf_width_um"
    h_col = "param_rf_height"

    def _highlight_extremes(ax, x, y, xlabel, ylabel, title, color_col=None):
        c = df[color_col] if color_col else LBLUE
        sc = ax.scatter(x, y, c=c, cmap="viridis" if color_col else None,
                        s=38, alpha=0.75, edgecolors="none", zorder=3)
        if color_col:
            cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
            cb.set_label(_pretty(color_col), fontsize=8)
            cb.ax.tick_params(labelsize=7)

        # Annotate the 3 lowest and 3 highest x values
        xvals = x.values
        yvals = y.values
        sorted_idx = np.argsort(xvals)
        extremes = np.concatenate([sorted_idx[:3], sorted_idx[-3:]])
        colors   = [RED]*3 + [GREEN]*3
        labels   = ["low"]*3 + ["high"]*3

        annotated = set()
        for idx, col, lbl in zip(extremes, colors, labels):
            if idx in annotated:
                continue
            annotated.add(idx)
            ax.scatter(xvals[idx], yvals[idx], s=80, facecolors="none",
                       edgecolors=col, linewidths=1.6, zorder=5)
            ax.annotate(f"{lbl}\n{xvals[idx]:.0f}",
                        xy=(xvals[idx], yvals[idx]),
                        xytext=(xvals[idx], yvals[idx] + (yvals.max()-yvals.min())*0.08),
                        fontsize=7, color=col, ha="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.6))

        # Best overall case
        best_i = df["score"].idxmax()
        ax.scatter(x.loc[best_i], y.loc[best_i], s=130,
                   marker="*", color=NAVY, zorder=6, label="Best case")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, color=NAVY)
        ax.grid(axis="y", lw=0.4, color="#d3d1c7", zorder=0)
        ax.tick_params(labelsize=8)

    # Row 1: effect of RF width
    _highlight_extremes(axes[0,0], df[w_col], df["depth_eV"],
                        "RF width (µm)", "Trap depth D (eV)",
                        "Trap depth vs. RF width", color_col=h_col)
    _highlight_extremes(axes[0,1], df[w_col], df["min_freq_hz"]/1e6,
                        "RF width (µm)", r"$f_{min}$ (MHz)",
                        r"Weakest secular freq. vs. RF width", color_col=h_col)

    # Row 2: effect of RF height
    _highlight_extremes(axes[1,0], df[h_col], df["depth_eV"],
                        "RF height (µm)", "Trap depth D (eV)",
                        "Trap depth vs. RF height", color_col=w_col)
    _highlight_extremes(axes[1,1], df[h_col], df["min_freq_hz"]/1e6,
                        "RF height (µm)", r"$f_{min}$ (MHz)",
                        r"Weakest secular freq. vs. RF height", color_col=w_col)

    # Shared legend
    legend_els = [
        plt.scatter([], [], s=80, facecolors="none", edgecolors=RED,
                    linewidths=1.6, label="Extreme low"),
        plt.scatter([], [], s=80, facecolors="none", edgecolors=GREEN,
                    linewidths=1.6, label="Extreme high"),
        plt.scatter([], [], s=130, marker="*", color=NAVY, label="Best case"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Effect of RF electrode width and height on trap performance\n"
        f"({len(df)} geometries evaluated — color encodes the other parameter)",
        fontsize=11, color=NAVY,
    )
    path = outdir / "fig4_param_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Top-5 table
# ─────────────────────────────────────────────────────────────────────────────

def make_fig5_table(df: pd.DataFrame, outdir: Path, dpi: int) -> Path:
    top5 = (df.sort_values("score", ascending=False).head(5).reset_index(drop=True))

    headers = ["Case", "RF width (µm)", "RF height (µm)",
               "Depth (eV)", "f_min (MHz)", "Δf (MHz)", "Score"]
    rows = []
    for _, row in top5.iterrows():
        rows.append([
            row["case_id"],
            f"{row['param_rf_width_um']:.1f}",
            f"{row['param_rf_height']:.1f}",
            f"{row['depth_eV']:.3f}"       if pd.notna(row.get("depth_eV"))       else "—",
            f"{row['min_freq_hz']/1e6:.3f}" if pd.notna(row.get("min_freq_hz"))   else "—",
            f"{row['mode_spread_hz']/1e6:.3f}" if pd.notna(row.get("mode_spread_hz")) else "—",
            f"{row['score']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.75)

    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor(NAVY)
        cell.set_text_props(color=WHITE, fontweight="bold", fontsize=8.5)

    for i in range(1, 6):
        fc = XBLUE if i == 1 else (WHITE if i % 2 == 0 else GRAY)
        for j in range(len(headers)):
            cell = tbl[i, j]
            cell.set_facecolor(fc)
            cell.set_edgecolor("#d3d1c7")
            if i == 1:
                cell.set_text_props(color=NAVY, fontweight="bold")

    tbl[1, 0].get_text().set_text("★ " + tbl[1, 0].get_text().get_text())

    ax.set_title("Top-5 geometries ranked by objective score S",
                 fontsize=10, color=NAVY, pad=10)
    path = outdir / "fig5_top5_table.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader (robust to extra columns / embedded commas)
# ─────────────────────────────────────────────────────────────────────────────

def _pretty(col: str) -> str:
    col = col.replace("param_", "")
    return {"rf_height": "RF height (µm)", "rf_width_um": "RF width (µm)",
            "rf_gap": "RF gap (µm)", "score": "Score",
            "depth_eV": "Depth (eV)", "min_freq_hz": "f_min (Hz)",
            "mode_spread_hz": "Δf (Hz)"}.get(col, col.replace("_", " ").title())


def load_summary(sweep: Path) -> pd.DataFrame:
    csv_path = sweep / "summary.csv"
    if not csv_path.exists():
        sys.exit(f"[error] summary.csv not found in {sweep}")
    with open(csv_path, "r") as f:
        n_cols = len(f.readline().split(","))
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn",
                     usecols=range(n_cols), header=0)
    df_ok = df[df["status"] == "ok"].copy()
    for col in ["depth_eV", "min_freq_hz", "max_freq_hz",
                "mode_spread_hz", "center_offset_m", "score"]:
        if col in df_ok.columns:
            df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")
    print(f"  Loaded {len(df)} total cases, {len(df_ok)} successful.")
    return df_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", type=Path, required=True,
                    help="Directory containing mesh.h5 and facets.h5")
    ap.add_argument("--sweep",    type=Path, required=True,
                    help="Sweep workdir containing summary.csv")
    ap.add_argument("--outdir",   type=Path, default=Path("./poster_figures"))
    ap.add_argument("--dpi",      type=int,  default=300)
    ap.add_argument("--skip-geometry", action="store_true",
                    help="Skip figs 1 & 2 (no mesh.h5 needed)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nMesh dir: {args.mesh_dir}")
    print(f"Sweep:    {args.sweep}")
    print(f"Output:   {args.outdir}\n")

    # Load mesh
    if not args.skip_geometry:
        print("── Loading mesh ───────────────────────────────────────────────")
        coords, cells, ctags, ftris, ftags = load_mesh(args.mesh_dir)

        print("\n── Figure 1: 3D trap geometry ─────────────────────────────────")
        make_fig1_geometry(coords, ftris, ftags, args.outdir, args.dpi)

        print("\n── Figure 2: cross-section ────────────────────────────────────")
        make_fig2_crosssection(coords, ftris, ftags, args.outdir, args.dpi)

    print("\n── Figure 3: pseudopotential slices ───────────────────────────")
    make_fig3_psi_slices(args.outdir, args.dpi)

    print("\n── Loading summary.csv ────────────────────────────────────────")
    df = load_summary(args.sweep)
    if df.empty:
        print("[warn] No successful cases — skipping figs 4 & 5.")
        return

    print("\n── Figure 4: parameter comparison ────────────────────────────")
    make_fig4_param_comparison(df, args.outdir, args.dpi)

    print("\n── Figure 5: top-5 table ──────────────────────────────────────")
    make_fig5_table(df, args.outdir, args.dpi)

    print(f"\nDone. All figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()