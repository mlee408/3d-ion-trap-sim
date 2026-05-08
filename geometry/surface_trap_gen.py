#!/usr/bin/env python3
"""
surface_trap_gen.py — Parametric STEP generator for a surface X-junction ion trap.

Generates RF, DC, and Ground electrode STEP files for a surface-electrode
X-junction trap.  All user-facing dimensions are in micrometres; CadQuery uses mm.

RF geometry (cross / plus shape)
---------------------------------
The RF electrode is a 28-vertex cross solid with four arms extending in ±x/±y.
Each arm has width w_rf (120 µm baseline) and tapers at the junction to a narrow
neck of width ``gap`` (10 µm, fixed).  The taper is defined by two parameters:

    d_tip       distance from centre to the start of the neck  [µm]
    theta_taper half-angle of the taper  [deg]

From these two, the taper end (where the arm reaches its full width) is computed:

    taper_end = d_tip + (w_rf/2 - gap/2) / tan(theta_taper)

Baseline values extracted from cad/base/rf_surface.step:
    cell_half=300, w_rf=120, gap=10, d_tip=23.07, taper_end=95.27 → theta≈37.3°

Optional junction features (disabled by default)
-------------------------------------------------
    r_tip       arc radius at the 8 inner neck tip vertices  (0 = sharp)
    r_fillet    fillet radius at the 4 ninety-degree centre corners  (0 = none)
    o1_offset, o1_radius   first set of 4×45° junction RF lobes
    o2_offset, o2_radius   second set of 4×45° junction RF lobes

DC geometry
-----------
Rectangular pads covering two zones:
  • Quadrant corner zone (near junction):  one square corner piece + row of
    segments along each arm direction, per quadrant.  Four-fold symmetry →
    4 corner pieces + 4×2 series of arm segments.
  • Linear flank zone (straight arm section):  pads beside the arm outside the
    quadrant corner, providing axial confinement in the transport channels.
    Two flanks per arm × 4 arms → 8 flank series.

Ground geometry
---------------
Outer square ring frame from cell_half to (cell_half + frame_width) with
rectangular slots cut where the four RF arms exit the cell.

Usage
-----
    python geometry/surface_trap_gen.py                        # baseline defaults
    python geometry/surface_trap_gen.py --w-rf 100 --d-tip 20 --theta-taper 40
    python geometry/surface_trap_gen.py --o1-offset 30 --o1-radius 8 \\
                                        --out-dir cad/generated/case_001
    python geometry/surface_trap_gen.py --r-tip 3 --r-fillet 5 --out-dir cad/generated/rounded
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cadquery as cq
from cadquery import exporters

# ─────────────────────────────────────────────────────────────────────────────
# Default parameters (µm unless noted)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS: dict = dict(
    # Fixed process / layout constraints (normally do not sweep these)
    cell_half       = 300.0,   # half cell size [µm]
    gap             = 10.0,    # electrode–electrode gap [µm]  (FIXED = 10 µm)
    thickness       = 10.0,    # electrode layer thickness [µm]

    # ── Core RF parameters ───────────────────────────────────────────────────
    w_rf            = 120.0,   # RF arm width in linear sections  [µm]  (paper: 120)
    d_tip           = 23.07,   # depth of narrow neck from junction centre  [µm]
    theta_taper     = 37.3,    # taper half-angle  [deg]
                               #   → baseline taper_end ≈ 95.27 µm

    # ── Tip and fillet rounding ───────────────────────────────────────────────
    r_tip           = 0.0,     # arc radius at 8 inner neck tip vertices  [µm]
    r_fillet        = 0.0,     # fillet at 4 right-angle centre corners  [µm]

    # ── Junction optimisation lobes O1 and O2 ────────────────────────────────
    # Four-fold symmetric circular RF additions at the 45° diagonals.
    # Position: (o_offset / √2, o_offset / √2) from centre (and permutations).
    # Set o_radius > 0 to enable.
    o1_offset       = 0.0,     # O1 distance from centre along diagonal  [µm]
    o1_radius       = 0.0,     # O1 disk radius  [µm]  (0 = disabled)
    o2_offset       = 0.0,     # O2 distance from centre along diagonal  [µm]
    o2_radius       = 0.0,     # O2 disk radius  [µm]  (0 = disabled)

    # ── DC electrode layout ──────────────────────────────────────────────────
    dc_corner_size  = 60.0,    # square corner DC pad side length  [µm]
    dc_seg_len      = 35.0,    # arm-direction DC segment length  [µm]
    dc_seg_gap      = 10.0,    # gap between adjacent DC segments  [µm]
    n_dc_junction   = 3,       # segments per arm direction in quadrant corner zone
    dc_flank_width  = 60.0,    # transverse width of linear-flank DC pads  [µm]
    dc_flank_len    = 35.0,    # axial length of linear-flank DC pads  [µm]
    n_dc_flank      = 4,       # pads per arm flank side in linear zone

    # ── Ground frame ─────────────────────────────────────────────────────────
    frame_width     = 50.0,    # ground ring thickness beyond cell_half  [µm]
)


def mm(x: float) -> float:
    """µm → mm (CadQuery native unit)."""
    return x * 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _taper_end(p: dict) -> float:
    """Return the taper-end distance from centre [µm]."""
    w = p['w_rf'] / 2
    g = p['gap'] / 2
    return p['d_tip'] + (w - g) / math.tan(math.radians(p['theta_taper']))


def _validate(p: dict) -> None:
    te = _taper_end(p)
    C = p['cell_half']
    d = p['d_tip']
    g = p['gap'] / 2
    if te >= C:
        raise ValueError(
            f"taper_end = {te:.2f} µm ≥ cell_half = {C} µm.  "
            "Decrease theta_taper or d_tip."
        )
    if d <= g:
        raise ValueError(
            f"d_tip = {d} µm must be > gap/2 = {g} µm."
        )
    if p['w_rf'] / 2 <= p['gap'] / 2:
        raise ValueError("w_rf must be > gap.")


def _box_solid(
    x0: float, y0: float, x1: float, y1: float, thickness: float
) -> cq.Workplane:
    """Return a box solid (in µm) at z = -thickness to z = 0."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    t  = mm(thickness)
    return (
        cq.Workplane("XY", origin=(mm(cx), mm(cy), -t))
        .box(mm(dx), mm(dy), t, centered=(True, True, False))
    )


def _rotate_pt(x: float, y: float, deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * x - s * y, s * x + c * y


def _rotate_boxes(
    boxes: list[tuple[float, float, float, float]], deg: float
) -> list[tuple[float, float, float, float]]:
    """Rotate axis-aligned boxes by deg degrees (0/90/180/270 only)."""
    result = []
    for x0, y0, x1, y1 in boxes:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        rx = [_rotate_pt(x, y, deg)[0] for x, y in corners]
        ry = [_rotate_pt(x, y, deg)[1] for x, y in corners]
        result.append((min(rx), min(ry), max(rx), max(ry)))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RF electrode
# ─────────────────────────────────────────────────────────────────────────────
def _rf_polygon_verts(p: dict) -> list[tuple[float, float]]:
    """
    Return the 28 counter-clockwise vertices of the RF cross polygon [µm].

    Layout (one arm shown, –y direction):

         ┌──────────────────────────────────────────┐  y = 0
         │    g          g                          │
         │ (g,−g)────(g,−d)   ← inner neck right   │
         │                ╲                         │
         │  taper          ╲                        │
         │                  (w,−te)                 │
         │                  │                       │
         │                  │  arm straight          │
         │                  (w,−C)                  │
         └──────────────────────────────────────────┘  y = −C

    Vertex indices  0–27 in CCW order; see comments in code.
    """
    C  = p['cell_half']
    w  = p['w_rf'] / 2
    g  = p['gap'] / 2
    d  = p['d_tip']
    te = _taper_end(p)

    return [
        ( g, -g),   #  0 – centre corner
        ( g, -d),   #  1 – –y arm right inner tip
        ( w, -te),  #  2 – –y arm right taper end
        ( w, -C),   #  3 – –y arm outer right
        (-w, -C),   #  4 – –y arm outer left
        (-w, -te),  #  5 – –y arm left taper end
        (-g, -d),   #  6 – –y arm left inner tip
        (-g, -g),   #  7 – centre corner
        (-d, -g),   #  8 – –x arm bottom inner tip
        (-te, -w),  #  9 – –x arm bottom taper end
        (-C, -w),   # 10 – –x arm outer bottom
        (-C,  w),   # 11 – –x arm outer top
        (-te,  w),  # 12 – –x arm top taper end
        (-d,  g),   # 13 – –x arm top inner tip
        (-g,  g),   # 14 – centre corner
        (-g,  d),   # 15 – +y arm left inner tip
        (-w,  te),  # 16 – +y arm left taper end
        (-w,  C),   # 17 – +y arm outer left
        ( w,  C),   # 18 – +y arm outer right
        ( w,  te),  # 19 – +y arm right taper end
        ( g,  d),   # 20 – +y arm right inner tip
        ( g,  g),   # 21 – centre corner
        ( d,  g),   # 22 – +x arm top inner tip
        ( te,  w),  # 23 – +x arm top taper end
        ( C,  w),   # 24 – +x arm outer top
        ( C, -w),   # 25 – +x arm outer bottom
        ( te, -w),  # 26 – +x arm bottom taper end
        ( d, -g),   # 27 – +x arm bottom inner tip
    ]


def build_rf(p: dict) -> cq.Workplane:
    """Build the RF cross electrode solid."""
    _validate(p)
    verts_mm = [(mm(x), mm(y)) for x, y in _rf_polygon_verts(p)]
    t = mm(p['thickness'])

    rf = (
        cq.Workplane("XY", origin=(0, 0, -t))
        .polyline(verts_mm)
        .close()
        .extrude(t)
    )

    # ── O1 / O2 junction lobes ────────────────────────────────────────────────
    for prefix in ('o1', 'o2'):
        r      = p.get(f'{prefix}_radius', 0.0)
        offset = p.get(f'{prefix}_offset', 0.0)
        if r > 0.0 and offset > 0.0:
            d45 = offset / math.sqrt(2.0)
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                lobe = (
                    cq.Workplane("XY", origin=(mm(sx * d45), mm(sy * d45), -t))
                    .circle(mm(r))
                    .extrude(t)
                )
                rf = rf.union(lobe)

    # ── Tip rounding ─────────────────────────────────────────────────────────
    # The 8 inner tip vertices are vertical (Z-direction) edges at the neck tips.
    # Select them by their XY neighbourhood and fillet.
    if p.get('r_tip', 0.0) > 0.0:
        r_mm = mm(p['r_tip'])
        g_mm = mm(p['gap'] / 2)
        d_mm = mm(p['d_tip'])
        # Build a small selector box around each of the 8 tip edges
        tip_xy = [
            ( g_mm, -d_mm), (-g_mm, -d_mm),   # –y arm
            (-d_mm, -g_mm), (-d_mm,  g_mm),   # –x arm
            (-g_mm,  d_mm), ( g_mm,  d_mm),   # +y arm
            ( d_mm,  g_mm), ( d_mm, -g_mm),   # +x arm
        ]
        eps = r_mm * 0.5
        for cx, cy in tip_xy:
            try:
                rf = (
                    rf
                    .edges(
                        cq.selectors.BoxSelector(
                            (cx - eps, cy - eps, -t - eps),
                            (cx + eps, cy + eps,  t + eps),
                        )
                    )
                    .fillet(r_mm)
                )
            except Exception:
                pass   # skip if edge not found (already filleted or param too large)

    # ── Arm/junction corner fillet ───────────────────────────────────────────
    # The 4 right-angle centre corners are vertical edges at (±g, ±g).
    if p.get('r_fillet', 0.0) > 0.0:
        r_mm = mm(p['r_fillet'])
        g_mm = mm(p['gap'] / 2)
        corner_xy = [( g_mm,  g_mm), (-g_mm,  g_mm),
                     (-g_mm, -g_mm), ( g_mm, -g_mm)]
        eps = r_mm * 0.5
        for cx, cy in corner_xy:
            try:
                rf = (
                    rf
                    .edges(
                        cq.selectors.BoxSelector(
                            (cx - eps, cy - eps, -t - eps),
                            (cx + eps, cy + eps,  t + eps),
                        )
                    )
                    .fillet(r_mm)
                )
            except Exception:
                pass

    return rf


# ─────────────────────────────────────────────────────────────────────────────
# DC electrodes
# ─────────────────────────────────────────────────────────────────────────────
def _dc_quadrant_boxes(p: dict) -> list[tuple[float, float, float, float]]:
    """
    DC pad bounding boxes (x0,y0,x1,y1) [µm] for the TOP-RIGHT quadrant (x>0, y>0).

    Returns:
      • one square corner piece
      • n_dc_junction rectangular segments along the +x arm direction (y fixed)
      • n_dc_junction rectangular segments along the +y arm direction (x fixed)

    Four-fold rotation covers all quadrants.
    """
    C   = p['cell_half']
    w   = p['w_rf'] / 2
    gap = p['gap']
    cs  = p['dc_corner_size']
    sl  = p['dc_seg_len']
    sg  = p['dc_seg_gap']
    n   = p['n_dc_junction']

    # Inner boundary of DC zone in this quadrant
    dc_in  = w + gap           # = 70 µm baseline
    dc_out = dc_in + cs        # = 130 µm baseline

    boxes: list[tuple[float, float, float, float]] = []

    # ── Square corner piece ─────────────────────────────────────────────────
    boxes.append((dc_in, dc_in, dc_out, dc_out))

    # ── Segments along the +x arm (y ∈ [dc_in, dc_out], x marching outward) ─
    x = dc_out + sg
    for _ in range(n):
        if x + sl > C:
            sl = C - x          # clip last pad to cell boundary
        if sl <= 0:
            break
        boxes.append((x, dc_in, x + sl, dc_out))
        x += sl + sg

    # ── Segments along the +y arm (x ∈ [dc_in, dc_out], y marching outward) ─
    y = dc_out + sg
    sl = p['dc_seg_len']        # reset to original
    for _ in range(n):
        if y + sl > C:
            sl = C - y
        if sl <= 0:
            break
        boxes.append((dc_in, y, dc_out, y + sl))
        y += sl + sg

    return boxes


def _dc_linear_flank_boxes(p: dict) -> list[tuple[float, float, float, float]]:
    """
    DC pad bounding boxes for the RIGHT flank of the –y arm in the LINEAR section
    (the region outside the quadrant corner, beside the arm straight section).

    Pads at:  x ∈ [w+gap, w+gap+dc_flank_width],  y ∈ [–C, –dc_out–gap]

    Four-fold rotation (and separate left-flank mirror) covers all 8 flank zones.
    """
    C   = p['cell_half']
    w   = p['w_rf'] / 2
    gap = p['gap']
    cs  = p['dc_corner_size']
    fw  = p['dc_flank_width']
    fl  = p['dc_flank_len']
    sg  = p['dc_seg_gap']
    n   = p['n_dc_flank']

    dc_in  = w + gap            # right edge inner boundary  = 70 µm
    dc_out = dc_in + cs         # right edge of quadrant corner pad  = 130 µm

    # Flank starts just below the quadrant corner pad bottom edge (which is at –dc_out)
    # The quadrant corner pad is at y ∈ [dc_in, dc_out].
    # After 270° rotation for the bottom-right quadrant the corner pad will be at
    # y ∈ [–dc_out, –dc_in].  The flank pads must start below –dc_out–gap.
    y_top = -(dc_out + sg)       # uppermost flank pad top edge (closest to junction)
    y_bot = -C                   # cell outer boundary

    available = abs(y_bot - y_top)
    seg_len = fl
    total = n * seg_len + (n - 1) * sg
    if total > available:
        seg_len = max(gap, (available - (n - 1) * sg) / n)

    boxes: list[tuple[float, float, float, float]] = []
    y = y_top
    for _ in range(n):
        y1 = y
        y0 = y - seg_len
        if y0 < y_bot:
            y0 = y_bot
        if y1 <= y0:
            break
        boxes.append((dc_in, y0, dc_in + fw, y1))
        y = y0 - sg

    return boxes


def build_dc(p: dict) -> list[cq.Workplane]:
    """
    Build all DC electrode pads and return a list of CadQuery Workplane objects.

    Layout:
      • Quadrant corner zone  – generated for Q1 (x>0, y>0), rotated ×4
      • Linear flank zone     – right flank of –y arm, rotated ×4;
                                left flank of –y arm, rotated ×4
    """
    _validate(p)
    t   = p['thickness']
    pads: list[cq.Workplane] = []

    q_boxes = _dc_quadrant_boxes(p)
    f_right = _dc_linear_flank_boxes(p)

    # Left flank = right flank mirrored: x → –x  (i.e. negate x coords)
    f_left = [(-x1, y0, -x0, y1) for (x0, y0, x1, y1) in f_right]

    for angle in (0.0, 90.0, 180.0, 270.0):
        for x0, y0, x1, y1 in _rotate_boxes(q_boxes, angle):
            pads.append(_box_solid(x0, y0, x1, y1, t))
        for x0, y0, x1, y1 in _rotate_boxes(f_right, angle):
            pads.append(_box_solid(x0, y0, x1, y1, t))
        for x0, y0, x1, y1 in _rotate_boxes(f_left, angle):
            pads.append(_box_solid(x0, y0, x1, y1, t))

    return pads


# ─────────────────────────────────────────────────────────────────────────────
# Ground electrode
# ─────────────────────────────────────────────────────────────────────────────
def build_ground(p: dict) -> cq.Workplane:
    """
    Ground frame: square ring from cell_half to (cell_half + frame_width).

    Rectangular slots are cut on all four sides where the RF arms exit the cell:
        slot half-width = w_rf/2 + gap
    """
    C   = p['cell_half']
    fw  = p['frame_width']
    w   = p['w_rf'] / 2
    gap = p['gap']
    t   = mm(p['thickness'])

    outer      = C + fw
    slot_half  = w + gap     # arm exit slot half-width (with gap clearance)

    # Full outer square
    frame = (
        cq.Workplane("XY", origin=(0, 0, -t))
        .box(mm(2 * outer), mm(2 * outer), t, centered=(True, True, False))
    )

    # Hollow out the inner cell area
    inner_cut = (
        cq.Workplane("XY", origin=(0, 0, -t))
        .box(mm(2 * C), mm(2 * C), t, centered=(True, True, False))
    )
    frame = frame.cut(inner_cut)

    # Arm exit slots (4 sides)
    slot_depth = mm(2 * fw)
    slot_width = mm(2 * slot_half)
    for cx_um, cy_um, horiz in (
        (0.0,    C + fw / 2,  True),   # +y arm
        (0.0,  -(C + fw / 2), True),   # –y arm
        (C + fw / 2,   0.0,   False),  # +x arm
        (-(C + fw / 2), 0.0,  False),  # –x arm
    ):
        sw = slot_width if horiz else slot_depth
        sl = slot_depth if horiz else slot_width
        slot = (
            cq.Workplane("XY", origin=(mm(cx_um), mm(cy_um), -t))
            .box(sw, sl, t, centered=(True, True, False))
        )
        frame = frame.cut(slot)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Build and export
# ─────────────────────────────────────────────────────────────────────────────
def build_and_export(p: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    te = _taper_end(p)
    print(
        f"[surface_trap_gen] Parameters:\n"
        f"  cell={2*p['cell_half']:.0f}×{2*p['cell_half']:.0f} µm²  "
        f"gap={p['gap']:.0f} µm  thickness={p['thickness']:.0f} µm\n"
        f"  w_rf={p['w_rf']:.1f} µm  d_tip={p['d_tip']:.2f} µm  "
        f"theta={p['theta_taper']:.1f}°  taper_end={te:.2f} µm",
        flush=True,
    )

    # ── RF ───────────────────────────────────────────────────────────────────
    print("[surface_trap_gen] Building RF …", flush=True)
    rf      = build_rf(p)
    rf_path = out_dir / "rf.step"
    exporters.export(rf, str(rf_path))
    print(f"  ✓  {rf_path}")

    # ── DC ───────────────────────────────────────────────────────────────────
    print("[surface_trap_gen] Building DC pads …", flush=True)
    dc_pads = build_dc(p)
    print(f"  {len(dc_pads)} pads generated", flush=True)

    # Export all DC pads in a single STEP file as a compound
    try:
        compound = cq.Compound.makeCompound([pad.val() for pad in dc_pads])
        dc_wp    = cq.Workplane().newObject([compound])
    except Exception:
        # Fallback: union all pads (slower but always works)
        dc_wp = dc_pads[0]
        for pad in dc_pads[1:]:
            dc_wp = dc_wp.union(pad)

    dc_path = out_dir / "dc.step"
    exporters.export(dc_wp, str(dc_path))
    print(f"  ✓  {dc_path}")

    # ── Ground ───────────────────────────────────────────────────────────────
    print("[surface_trap_gen] Building ground frame …", flush=True)
    gnd      = build_ground(p)
    gnd_path = out_dir / "ground.step"
    exporters.export(gnd, str(gnd_path))
    print(f"  ✓  {gnd_path}")

    print(
        f"\n[surface_trap_gen] Done.  Files written to {out_dir}/",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1].strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for key, default in DEFAULTS.items():
        flag = f"--{key.replace('_', '-')}"
        t    = type(default)
        ap.add_argument(flag, type=t, default=default,
                        metavar=str(default),
                        help=f"[µm or deg]")
    ap.add_argument(
        "--out-dir", default="cad/generated/surface",
        help="Output directory for the three STEP files",
    )
    return ap


def main(argv=None) -> int:
    args = _make_parser().parse_args(argv)
    p    = {k: getattr(args, k) for k in DEFAULTS}
    try:
        build_and_export(p, Path(args.out_dir))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
