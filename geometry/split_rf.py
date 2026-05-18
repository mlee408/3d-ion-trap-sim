#!/usr/bin/env python3
"""
Parameterized 2D RF X-junction electrode generator (v2).

Generates the RF electrode polygons for a surface-electrode ion trap
X-junction. Each branch has two RF rails; at the junction the inner
edges taper into a bowtie shape while the outer edges sweep around
the corner with a fillet.

Topology (Q1 view, +x right, +y up):

    The quadrant contains TWO partial rail pieces:
      A) The outer rail of the +x branch (upper rail, y > 0)
      B) The outer rail of the +y branch (right rail, x > 0)
    These merge at the outer corner with a fillet, forming one
    connected L-shaped polygon per quadrant.

    Inner edges taper from (taper_start, half_gap) toward a tip
    near the junction center at (tip_inset, tip_inset).

Parameters:
    rail_width    - width of each RF rail in the linear region
    rail_gap      - center-to-center separation between inner edges of the two rails
    arm_length    - length from center to the end of each branch
    taper_start   - distance from center where the inner taper begins
    tip_inset     - how close the inner RF tip gets to the junction center
    tip_radius    - fillet radius at the RF tip
    fillet_radius - fillet at outer corner where branches meet
    dc_gap        - RF-to-DC gap (for DC outline generation)
"""

import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path as FilePath


DEFAULTS = dict(
    rail_width=120.0,
    rail_gap=71.5,
    arm_length=300.0,
    taper_start=150.0,
    tip_inset=23.0,
    tip_radius=8.0,
    fillet_radius=12.0,
    dc_gap=10.0,
    n_fillet_pts=16,
)


def fillet_arc(center, radius, angle_start, angle_end, n_pts):
    """Generate points along a circular arc."""
    angles = np.linspace(angle_start, angle_end, n_pts)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])


def build_quadrant_polygon(
    rail_width, rail_gap, arm_length, taper_start,
    tip_inset, tip_radius, fillet_radius, dc_gap,
    n_fillet_pts, **_kw
):
    """
    Build one quadrant (Q1: +x, +y) of the RF electrode.

    This quadrant is an L-shaped polygon formed by the upper rail
    of the +x branch merging with the right rail of the +y branch.

    Trace (CCW):
      P1 → P2: +x branch, outer edge (y = half_gap + rail_width)
      P2 → P3: outer corner fillet
      P3 → P4: +y branch, outer edge (x = half_gap + rail_width)
      P4 → P5: +y branch, inner edge going down (x = half_gap)
      P5 → tip: inner taper of +y branch toward center
      tip arc:  fillet at junction center tip
      tip → P6: inner taper of +x branch away from center
      P6 → P7: +x branch, inner edge (y = half_gap)
      P7 → P1: close at arm end
    """
    hg = rail_gap / 2.0          # half of rail gap
    outer = hg + rail_width      # outer edge position
    pts = []

    # ── P1: +x arm far end, inner edge ──
    pts.append([arm_length, hg])

    # ── P7→P1 already handled by closing; start trace from outer ──
    # Actually let's trace starting from the +x arm far end, outer edge, CCW

    # Start: +x branch, far end, outer edge
    pts = []
    pts.append([arm_length, outer])          # A: top-right corner of +x arm

    # ── Outer edge of +x branch going left toward junction ──
    pts.append([taper_start, outer])         # B: where outer edge reaches taper zone
    # Outer edge continues to the corner where +x and +y branches meet
    # The corner is at approximately (outer, outer) with a fillet

    # Outer corner fillet: arc from angle=0 (pointing +x from center)
    # to angle=π/2 (pointing +y from center)
    # Fillet center is at (outer - fillet_radius, outer - fillet_radius)
    fc = [outer - fillet_radius, outer - fillet_radius]
    arc = fillet_arc(fc, fillet_radius, 0, np.pi / 2, n_fillet_pts)
    for p in arc:
        pts.append(p.tolist())

    # ── Outer edge of +y branch going up ──
    pts.append([outer, taper_start])         # C
    pts.append([outer, arm_length])          # D: top of +y arm, outer edge

    # ── +y arm far end, inner edge ──
    pts.append([hg, arm_length])             # E: top of +y arm, inner edge

    # ── Inner edge of +y branch coming back toward junction ──
    pts.append([hg, taper_start])            # F: taper start on +y inner edge

    # ── Inner taper of +y branch: from F toward tip ──
    # Taper from (hg, taper_start) to near (tip_inset, tip_inset+tip_radius)
    # Use a straight line (the taper)
    pts.append([tip_inset, tip_inset + tip_radius])  # G: just above tip

    # ── Tip fillet arc ──
    # Arc centered at (tip_inset + tip_radius, tip_inset + tip_radius)
    # going from π (pointing -x) to 3π/2 (pointing -y)
    tc = [tip_inset + tip_radius, tip_inset + tip_radius]
    tip_arc = fillet_arc(tc, tip_radius, np.pi, 3 * np.pi / 2, n_fillet_pts)
    for p in tip_arc:
        pts.append(p.tolist())

    # ── Inner taper of +x branch: from tip toward F' ──
    pts.append([tip_inset + tip_radius, tip_inset])  # H: just right of tip
    pts.append([taper_start, hg])            # I: taper start on +x inner edge

    # ── Inner edge of +x branch going right ──
    pts.append([arm_length, hg])             # J: far end, inner edge

    return np.array(pts)


def mirror_to_all_quadrants(q1):
    """Mirror Q1 polygon to Q2 (-x,+y), Q3 (-x,-y), Q4 (+x,-y)."""
    q2 = q1.copy()
    q2[:, 0] *= -1
    q2 = q2[::-1]  # reverse for consistent winding

    q3 = q1.copy()
    q3 *= -1  # both axes flipped, winding preserved

    q4 = q1.copy()
    q4[:, 1] *= -1
    q4 = q4[::-1]

    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}


def plot_junction(quadrants, params, save_path=None):
    """Visualize with labeled dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax_idx, (ax, title) in enumerate(zip(axes, ["Full Junction", "Junction Center Detail"])):
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, fontweight="bold")

        for label, pts in quadrants.items():
            poly = plt.Polygon(pts, closed=True, fc="#c0392b", ec="white",
                               lw=0.6, alpha=0.9)
            ax.add_patch(poly)

        # Center marker
        ax.axhline(0, color="#888", lw=0.4, ls="--", alpha=0.4)
        ax.axvline(0, color="#888", lw=0.4, ls="--", alpha=0.4)
        ax.plot(0, 0, "+", ms=10, mew=1.5, color="black")

        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

        if ax_idx == 0:
            # Full view
            margin = 30
            lim = params["arm_length"] + margin
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            p = params
            hg = p["rail_gap"] / 2
            outer = hg + p["rail_width"]
            fs = 7
            ap = dict(arrowstyle="<->", color="navy", lw=1.0)

            # Rail width annotation
            xp = p["arm_length"] - 20
            ax.annotate("", xy=(xp, outer), xytext=(xp, hg), arrowprops=ap)
            ax.text(xp + 5, (outer + hg) / 2, f'rail_width\n{p["rail_width"]}',
                    fontsize=fs, color="navy", va="center")

            # Rail gap annotation
            xp2 = p["arm_length"] - 60
            ax.annotate("", xy=(xp2, hg), xytext=(xp2, -hg), arrowprops=ap)
            ax.text(xp2 + 5, 0, f'rail_gap\n{p["rail_gap"]}',
                    fontsize=fs, color="navy", va="center")

            # Arm length
            ax.annotate("", xy=(p["arm_length"], outer + 12),
                        xytext=(0, outer + 12), arrowprops=ap)
            ax.text(p["arm_length"] / 2, outer + 20,
                    f'arm_length = {p["arm_length"]}', fontsize=fs,
                    color="navy", ha="center")

            # Taper start
            ax.annotate("", xy=(p["taper_start"], -outer - 12),
                        xytext=(0, -outer - 12), arrowprops=ap)
            ax.text(p["taper_start"] / 2, -outer - 22,
                    f'taper_start = {p["taper_start"]}', fontsize=fs,
                    color="navy", ha="center")

        else:
            # Zoomed junction center
            zoom = max(params["taper_start"], params["rail_gap"] / 2 + params["rail_width"]) + 20
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)

            # Mark tip positions
            ti = params["tip_inset"]
            tr = params["tip_radius"]
            for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
                ax.plot(sx * (ti + tr), sy * (ti + tr), "^", ms=6,
                        color="blue", zorder=5)
            ax.text(ti + tr + 5, ti + tr + 5,
                    f'tip_inset={ti}\ntip_r={tr}',
                    fontsize=7, color="blue")

            # Fillet radius label
            hg = params["rail_gap"] / 2
            outer = hg + params["rail_width"]
            fr = params["fillet_radius"]
            ax.annotate(f'fillet_r={fr}',
                        xy=(outer - fr / 2, outer - fr / 2),
                        fontsize=7, color="green",
                        arrowprops=dict(arrowstyle="->", color="green"),
                        xytext=(outer + 20, outer + 20))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def export_json(quadrants, params, path):
    """Export polygons + params as JSON for downstream CAD/FEM."""
    data = {
        "parameters": params,
        "units": "micrometers",
        "description": "RF electrode polygons for surface-electrode X-junction trap",
        "quadrants": {k: v.tolist() for k, v in quadrants.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON → {path}")


def export_dxf(quadrants, path):
    """Export as DXF if ezdxf is available."""
    try:
        import ezdxf
        doc = ezdxf.new()
        msp = doc.modelspace()
        for label, pts in quadrants.items():
            closed = list(map(tuple, pts)) + [tuple(pts[0])]
            msp.add_lwpolyline(closed, dxfattribs={"layer": f"RF_{label}"})
        doc.saveas(path)
        print(f"Saved DXF → {path}")
    except ImportError:
        print("ezdxf not installed — skipping DXF. Install with: pip install ezdxf")


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameterized RF X-junction electrode geometry"
    )
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}",
                            type=type(default), default=default,
                            help=f"(default: {default})")
    parser.add_argument("--output-dir", type=str, default="/home/claude")
    parser.add_argument("--no-dxf", action="store_true")
    args = parser.parse_args()

    params = {k: getattr(args, k) for k in DEFAULTS}
    out = FilePath(args.output_dir)

    print("═" * 50)
    print("RF X-Junction Parameterizer v2")
    print("═" * 50)
    for k, v in params.items():
        print(f"  {k:20s}: {v}")
    print()

    q1 = build_quadrant_polygon(**params)
    quadrants = mirror_to_all_quadrants(q1)

    plot_path = str(out / "rf_junction_v2.png")
    plot_junction(quadrants, params, save_path=plot_path)
    print(f"Saved plot → {plot_path}")

    json_path = str(out / "rf_junction_v2.json")
    export_json(quadrants, params, json_path)

    if not args.no_dxf:
        export_dxf(quadrants, str(out / "rf_junction_v2.dxf"))

    return plot_path, json_path


if __name__ == "__main__":
    main()
