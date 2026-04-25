#!/usr/bin/env python3
"""
make_figures.py
===============
Generate all poster figures from a completed automate.py sweep directory.

Produces four files in --outdir (default: ./poster_figures/):
  fig1_electrode_schematic.png   -- labeled cross-section schematic (drawn, no mesh needed)
  fig2_pipeline.png              -- computational pipeline flowchart
  fig3_pseudopotential.png       -- Psi_rf colormap for the best case (needs pyvista)
  fig4_depth_scatter.png         -- trap depth vs. RF height scatter, colored by RF gap
  fig5_top5_table.png            -- rendered table of top-5 geometries

Usage
-----
  python make_figures.py --sweep ./sweep_001 --prefix auto --outdir ./poster_figures

  # If you only want the CSV-based figures (no FEniCSx/pyvista needed):
  python make_figures.py --sweep ./sweep_001 --no-field-plot

Arguments
---------
  --sweep     Path to the sweep workdir (must contain summary.csv)
  --prefix    Case prefix used in run_case.py (default: auto)
  --outdir    Where to write output PNGs (default: ./poster_figures)
  --dpi       Resolution for saved figures (default: 300)
  --param-x   CSV column for x-axis of scatter plot (default: auto-detected)
  --param-c   CSV column for scatter plot color axis (default: auto-detected)
  --no-field-plot
              Skip fig3 (pseudopotential colormap). Use this if pyvista /
              dolfinx are not installed in this environment.
  --best-case Override which case_id to use for fig3 (default: highest score)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────
NAVY   = "#0C447C"
BLUE   = "#185FA5"
LBLUE  = "#85B7EB"
XBLUE  = "#E6F1FB"
GRAY   = "#f1efe8"
DGRAY  = "#444441"
MGRAY  = "#888780"
WHITE  = "#ffffff"
GREEN  = "#27500A"
AMBER  = "#854F0B"
RED    = "#791F1F"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.facecolor": WHITE,
    "axes.facecolor":   WHITE,
    "savefig.facecolor":WHITE,
    "savefig.bbox":     "tight",
    "savefig.dpi":      300,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Electrode cross-section schematic (purely drawn — no mesh needed)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1_schematic(outdir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Substrate
    sub = FancyBboxPatch((0, -0.5), 10, 0.5, boxstyle="square,pad=0",
                          fc="#d3d1c7", ec=DGRAY, lw=0.8)
    ax.add_patch(sub)
    ax.text(5, -0.25, "Substrate", ha="center", va="center",
            fontsize=8, color=DGRAY)

    # Ground electrodes (left pair + right pair)
    for x0 in [0.3, 1.5, 6.5, 7.7]:
        gnd = FancyBboxPatch((x0, 0), 1.0, 0.55, boxstyle="square,pad=0",
                              fc=MGRAY, ec=DGRAY, lw=0.7)
        ax.add_patch(gnd)
    ax.text(1.3, 0.28, "GND", ha="center", va="center", fontsize=7.5,
            color=WHITE, fontweight="bold")
    ax.text(7.2, 0.28, "GND", ha="center", va="center", fontsize=7.5,
            color=WHITE, fontweight="bold")

    # RF electrodes (centre pair)
    for x0 in [3.0, 5.5]:
        rf = FancyBboxPatch((x0, 0), 1.5, 0.9, boxstyle="square,pad=0",
                             fc=BLUE, ec=NAVY, lw=0.9)
        ax.add_patch(rf)
    ax.text(3.75, 0.45, "RF", ha="center", va="center", fontsize=8.5,
            color=WHITE, fontweight="bold")
    ax.text(6.25, 0.45, "RF", ha="center", va="center", fontsize=8.5,
            color=WHITE, fontweight="bold")

    # RF null marker
    ax.plot(5.0, 2.35, "o", color=RED, ms=9, zorder=5)
    ax.text(5.3, 2.35, r"$\mathbf{r_0}$ (RF null)", fontsize=9,
            va="center", color=RED)

    # Dimension arrows — RF height
    ax.annotate("", xy=(2.4, 0.9), xytext=(2.4, 0),
                 arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.2))
    ax.text(2.15, 0.45, "h", ha="center", va="center", fontsize=9,
            color=AMBER, fontstyle="italic")

    # Dimension arrows — RF gap
    ax.annotate("", xy=(5.5, -0.15), xytext=(3.0+1.5, -0.15),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.2))
    ax.text(5.0, -0.35, "gap", ha="center", va="center", fontsize=9,
            color=GREEN, fontstyle="italic")

    # Pseudopotential contours (schematic ellipses)
    for lev, alpha in [(0.4, 0.18), (0.65, 0.13), (0.9, 0.09)]:
        ell = matplotlib.patches.Ellipse((5.0, 2.35), width=lev*3.5,
                                          height=lev*2.2, fc="none",
                                          ec=LBLUE, lw=0.9, alpha=alpha+0.3,
                                          linestyle="--")
        ax.add_patch(ell)

    # Psi label
    ax.text(6.9, 3.5, r"$\Psi$ contours", fontsize=8, color=BLUE,
            fontstyle="italic")

    # V_RF label
    ax.annotate("$V_{RF}$cos($\Omega t$)", xy=(6.25, 0.9),
                 xytext=(7.8, 1.6),
                 fontsize=8.5, color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9))

    ax.set_title("Planar surface-electrode RF ion trap — cross-section schematic",
                 fontsize=10, color=NAVY, pad=6)

    path = outdir / "fig1_electrode_schematic.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Pipeline flowchart
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2_pipeline(outdir: Path, dpi: int) -> Path:
    steps = [
        ("1. Mesh\ngeneration", "Gmsh\nPhysical Groups", XBLUE, BLUE),
        ("2. Laplace\nsolve", "∇²φ = 0\nCG2 FEM", XBLUE, BLUE),
        ("3. Pseudo-\npotential", "Ψ = q²|∇φ|²\n4mΩ²", XBLUE, BLUE),
        ("4. RF null\nlocation", "5th-pct cluster\n+ coord. descent", XBLUE, BLUE),
        ("5. Secular\nfreqs.", "Hessian eigen-\ndecomposition", XBLUE, BLUE),
        ("6. Trap\ndepth", "48-ray cast\nFibonacci sphere", XBLUE, BLUE),
        ("7. Score\n& rank", "Physics-informed\nobjective S", GRAY, MGRAY),
    ]

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(-0.1, 2.7)
    ax.axis("off")

    box_w, box_h = 1.12, 2.0
    gap = 0.18
    x0 = 0.1

    for i, (title, subtitle, fc, ec) in enumerate(steps):
        xc = x0 + i * (box_w + gap)
        box = FancyBboxPatch((xc, 0.35), box_w, box_h,
                              boxstyle="round,pad=0.05",
                              fc=fc, ec=ec, lw=1.0)
        ax.add_patch(box)
        ax.text(xc + box_w/2, 0.35 + box_h*0.68, title,
                ha="center", va="center", fontsize=7.8,
                color=NAVY, fontweight="bold", linespacing=1.4)
        ax.text(xc + box_w/2, 0.35 + box_h*0.28, subtitle,
                ha="center", va="center", fontsize=7.0,
                color=DGRAY, linespacing=1.35)

        # Arrow to next box
        if i < len(steps) - 1:
            ax_next = xc + box_w
            ax.annotate("", xy=(ax_next + gap, 0.35 + box_h/2),
                         xytext=(ax_next, 0.35 + box_h/2),
                         arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    # Repeat annotation
    last_x = x0 + (len(steps)-1) * (box_w + gap)
    ax.annotate("", xy=(x0 + box_w/2, 0.3),
                 xytext=(last_x + box_w/2, 0.3),
                 arrowprops=dict(arrowstyle="->", color=MGRAY, lw=1.0,
                                 connectionstyle="arc3,rad=-0.35"))
    ax.text((x0 + last_x + box_w) / 2, 0.06,
            "repeat for N candidate geometries (parallel workers)",
            ha="center", fontsize=7.5, color=MGRAY, fontstyle="italic")

    ax.set_title("Automated evaluation pipeline", fontsize=10, color=NAVY, pad=4)

    path = outdir / "fig2_pipeline.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Pseudopotential colormap (best case, via pyvista)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig3_pseudopotential(
    best_case_dir: Path,
    prefix: str,
    r0_SI: list,
    outdir: Path,
    dpi: int,
) -> Path:
    try:
        import pyvista as pv
    except ImportError:
        print("  [fig3] pyvista not installed — skipping pseudopotential colormap.")
        print("         Install with: pip install pyvista")
        return None

    # The xdmf written by run_case.py is at {case_dir}/{prefix}_{case_id}_fields.xdmf
    # but when called from automate the prefix already includes the case_id fragment.
    # We search for the first *_fields.xdmf we find in the case dir.
    xdmf_candidates = sorted(best_case_dir.glob("*_fields.xdmf"))
    if not xdmf_candidates:
        print(f"  [fig3] No *_fields.xdmf found in {best_case_dir} — skipping.")
        return None

    xdmf_path = xdmf_candidates[0]
    print(f"  [fig3] Reading {xdmf_path}")

    try:
        mesh = pv.read(str(xdmf_path))
    except Exception as e:
        print(f"  [fig3] pyvista could not read XDMF: {e} — skipping.")
        return None

    # Find the Psi_rf array (name written by run_case.py)
    psi_key = None
    for name in mesh.array_names:
        if "psi" in name.lower() or "Psi" in name:
            psi_key = name
            break
    if psi_key is None:
        print(f"  [fig3] Could not find Psi array in {xdmf_path}. "
              f"Available: {mesh.array_names} — skipping.")
        return None

    plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
    plotter.set_background("white")

    plotter.add_mesh(
        mesh,
        scalars=psi_key,
        cmap="coolwarm",
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Ψ (J)",
            "title_font_size": 14,
            "label_font_size": 12,
            "color": DGRAY,
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.25,
            "width": 0.08,
            "height": 0.5,
        },
        opacity=0.92,
    )

    # Mark r0
    if r0_SI and len(r0_SI) >= 2:
        r0_pt = np.array(r0_SI + [0.0] if len(r0_SI) == 2 else r0_SI,
                          dtype=float)
        r0_cloud = pv.PolyData(r0_pt.reshape(1, 3))
        plotter.add_mesh(r0_cloud, color=RED, point_size=18,
                          render_points_as_spheres=True)
        plotter.add_point_labels(
            r0_cloud, ["r₀ (RF null)"],
            font_size=13, text_color=RED,
            point_color=RED, point_size=1,
            always_visible=True, fill_shape=False,
        )

    plotter.view_isometric()
    plotter.add_title("RF pseudopotential Ψ — best geometry", font_size=12,
                       color=NAVY)

    path = outdir / "fig3_pseudopotential.png"
    plotter.screenshot(str(path))
    plotter.close()
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Scatter — depth vs. primary parameter, colored by secondary
# ─────────────────────────────────────────────────────────────────────────────

def make_fig4_scatter(
    df: pd.DataFrame,
    param_x: str,
    param_c: str,
    outdir: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"wspace": 0.38})

    # ── Left: depth vs param_x ────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(
        df[param_x], df["depth_eV"],
        c=df[param_c], cmap="viridis",
        s=42, alpha=0.82, edgecolors="none", zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(_pretty(param_c), fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Highlight best case
    best_idx = df["score"].idxmax()
    ax.scatter(df.loc[best_idx, param_x], df.loc[best_idx, "depth_eV"],
               s=120, facecolors="none", edgecolors=RED, linewidths=1.8,
               zorder=5, label="Best score")
    ax.legend(fontsize=8, frameon=False)

    ax.set_xlabel(_pretty(param_x), fontsize=9)
    ax.set_ylabel("Trap depth D (eV)", fontsize=9)
    ax.set_title(f"Trap depth vs. {_pretty(param_x)}", fontsize=10, color=NAVY)
    ax.grid(axis="y", lw=0.4, color="#d3d1c7", zorder=0)

    # ── Right: score distribution histogram ───────────────────────────────
    ax2 = axes[1]
    ax2.hist(df["score"].dropna(), bins=20, color=LBLUE, edgecolor=BLUE,
             linewidth=0.6, zorder=3)
    ax2.axvline(df["score"].max(), color=RED, lw=1.5, ls="--",
                label=f"Best = {df['score'].max():.3g}")
    ax2.set_xlabel("Objective score S", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title("Score distribution across all cases", fontsize=10, color=NAVY)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", lw=0.4, color="#d3d1c7", zorder=0)

    fig.suptitle(
        f"Random search results  —  {len(df)} evaluated geometries",
        fontsize=11, color=NAVY, y=1.01,
    )

    path = outdir / "fig4_depth_scatter.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Top-5 table rendered as a figure
# ─────────────────────────────────────────────────────────────────────────────

def make_fig5_table(df: pd.DataFrame, param_cols: list, outdir: Path, dpi: int) -> Path:
    top5 = (
        df[df["status"] == "ok"]
        .sort_values("score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    display_cols = param_cols + ["depth_eV", "min_freq_hz", "mode_spread_hz", "score"]
    headers = (
        [_pretty(c) for c in param_cols]
        + ["Depth (eV)", "f_min (MHz)", "Δf (MHz)", "Score"]
    )

    rows = []
    for _, row in top5.iterrows():
        r = []
        for c in param_cols:
            r.append(f"{row[c]:.1f}" if c in row and pd.notna(row[c]) else "—")
        r.append(f"{row['depth_eV']:.3f}"   if pd.notna(row.get("depth_eV"))       else "—")
        r.append(f"{row['min_freq_hz']/1e6:.2f}" if pd.notna(row.get("min_freq_hz")) else "—")
        r.append(f"{row['mode_spread_hz']/1e6:.2f}" if pd.notna(row.get("mode_spread_hz")) else "—")
        r.append(f"{row['score']:.4f}"       if pd.notna(row.get("score"))          else "—")
        rows.append(r)

    ncols = len(headers)
    fig_w = max(6.5, ncols * 1.15)
    fig, ax = plt.subplots(figsize=(fig_w, 2.4))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor(NAVY)
        cell.set_text_props(color=WHITE, fontweight="bold", fontsize=8.5)

    for i in range(1, 6):
        fc = XBLUE if i == 1 else (WHITE if i % 2 == 0 else GRAY)
        for j in range(ncols):
            cell = tbl[i, j]
            cell.set_facecolor(fc)
            cell.set_edgecolor("#d3d1c7")
            if i == 1:
                cell.set_text_props(color=NAVY, fontweight="bold")

    # Star on best row
    tbl[1, 0].get_text().set_text("★ " + tbl[1, 0].get_text().get_text())

    ax.set_title("Top-5 geometries ranked by objective score S",
                 fontsize=10, color=NAVY, pad=10)

    path = outdir / "fig5_top5_table.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pretty(col: str) -> str:
    """Human-readable column label."""
    col = col.replace("param_", "")
    replacements = {
        "rf_height":    "RF height (µm)",
        "rf_width_um":  "RF width (µm)",
        "rf_gap":       "RF gap (µm)",
        "junction_rounding": "Rounding (µm)",
        "depth_ev":     "Depth (eV)",
        "min_freq_hz":  "f_min (Hz)",
        "mode_spread_hz": "Δf (Hz)",
        "score":        "Score",
        "center_offset_m": "|r₀| (m)",
    }
    return replacements.get(col, col.replace("_", " ").title())


def _detect_param_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("param_")]


def _load_summary(sweep: Path) -> pd.DataFrame:
    csv_path = sweep / "summary.csv"
    if not csv_path.exists():
        sys.exit(f"[error] summary.csv not found in {sweep}")

    # Use the Python engine which handles rows with more fields than the header
    # (caused by unquoted commas inside junction_report_path / junction_score
    # values written by extended versions of automate.py).
    # on_bad_lines='warn' skips genuinely truncated rows; extra trailing fields
    # are silently ignored because we name only the header columns.
    with open(csv_path, "r", newline="") as f:
        header_line = f.readline()
    n_cols = len(header_line.split(","))

    df = pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="warn",
        usecols=range(n_cols),   # ignore any extra trailing fields per row
        header=0,
    )

    df_ok = df[df["status"] == "ok"].copy()
    for col in ["depth_eV", "min_freq_hz", "max_freq_hz", "mode_spread_hz",
                "center_offset_m", "score"]:
        if col in df_ok.columns:
            df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")

    n_total = len(df)
    n_ok    = len(df_ok)
    print(f"  Loaded {n_total} total cases, {n_ok} successful.")
    return df_ok


def _best_case_info(df: pd.DataFrame, sweep: Path) -> tuple[Path, list]:
    """Return (case_dir, r0_SI_m) for the highest-scoring case."""
    best_row = df.loc[df["score"].idxmax()]
    case_dir = Path(best_row["case_dir"])
    r0_SI = []
    report_path = best_row.get("report_path")
    if report_path and Path(str(report_path)).exists():
        try:
            report = json.loads(Path(str(report_path)).read_text())
            r0_SI = report.get("r0_SI_m", [])
        except Exception:
            pass
    return case_dir, r0_SI


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate poster figures from a sweep directory.")
    ap.add_argument("--sweep",    type=Path, required=True,
                    help="Path to the sweep workdir containing summary.csv")
    ap.add_argument("--prefix",   type=str, default="auto",
                    help="Case prefix used in run_case.py (default: auto)")
    ap.add_argument("--outdir",   type=Path, default=Path("./poster_figures"),
                    help="Output directory for PNGs (default: ./poster_figures)")
    ap.add_argument("--dpi",      type=int, default=300)
    ap.add_argument("--param-x",  type=str, default=None,
                    help="CSV column for scatter x-axis (default: first param_ column)")
    ap.add_argument("--param-c",  type=str, default=None,
                    help="CSV column for scatter color axis (default: second param_ column)")
    ap.add_argument("--no-field-plot", action="store_true",
                    help="Skip fig3 (pseudopotential colormap). No pyvista needed.")
    ap.add_argument("--best-case", type=str, default=None,
                    help="Override case_id for fig3 (default: highest score)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nReading sweep: {args.sweep}")
    print(f"Output dir:    {args.outdir}\n")

    df = _load_summary(args.sweep)

    if df.empty:
        sys.exit("[error] No successful cases found in summary.csv.")

    param_cols = _detect_param_cols(df)
    if not param_cols:
        sys.exit("[error] No param_* columns found in summary.csv.")

    param_x = args.param_x or param_cols[0]
    param_c = args.param_c or (param_cols[1] if len(param_cols) > 1 else param_cols[0])

    print("── Figure 1: electrode schematic ─────────────────────────────")
    make_fig1_schematic(args.outdir, args.dpi)

    print("── Figure 2: pipeline flowchart ──────────────────────────────")
    make_fig2_pipeline(args.outdir, args.dpi)

    if not args.no_field_plot:
        print("── Figure 3: pseudopotential colormap ────────────────────────")
        if args.best_case:
            best_dir = args.sweep / args.best_case
            best_row = df[df["case_id"] == args.best_case].iloc[0]
            r0_SI = []
            if pd.notna(best_row.get("report_path", None)):
                try:
                    report = json.loads(Path(best_row["report_path"]).read_text())
                    r0_SI = report.get("r0_SI_m", [])
                except Exception:
                    pass
        else:
            best_dir, r0_SI = _best_case_info(df, args.sweep)

        make_fig3_pseudopotential(best_dir, args.prefix, r0_SI, args.outdir, args.dpi)
    else:
        print("── Figure 3: skipped (--no-field-plot) ───────────────────────")

    print("── Figure 4: depth scatter + score histogram ─────────────────")
    make_fig4_scatter(df, param_x, param_c, args.outdir, args.dpi)

    print("── Figure 5: top-5 table ─────────────────────────────────────")
    make_fig5_table(df, param_cols, args.outdir, args.dpi)

    print(f"\nDone. All figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()