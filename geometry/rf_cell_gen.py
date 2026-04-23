#!/usr/bin/env python3
"""
rf_cell_gen.py  –  Standalone parametric CAD/mesh generator for the 3D RF lattice cell.

All Gmsh coordinates are in millimetres (Gmsh default unit).
User-facing parameters are in micrometres; the script converts internally.

Elementary structures
---------------------
  1. Support beam   – rectangular prism, 56 µm × 56 µm cross-section; height spans
                      from the surface RF top (z = 0 in the final assembled mesh) up
                      to the top vertex of the lattice beam diamond cross-section.
  2. Lattice beam   – prism with diamond cross section, length 656 µm
                      diamond: 56 µm horizontal × 82 µm vertical (baseline)

rf_height convention
--------------------
  rf_height is the vertical distance [µm] from the TOP of the surface RF electrode
  to the CENTROID (centre) of the 3D RF beam diamond cross-section.

  In the final assembled mesh (z = 0 at the surface RF electrode top):
    z_top_surface_rf = 0                       (reference plane)
    z_center_beam    = z_top_surface_rf + rf_height
    z_bottom_beam    = z_center_beam - beam_height / 2   (beam_height = dv)
    z_top_beam       = z_center_beam + beam_height / 2

  Do NOT interpret rf_height as total beam height, bottom-of-beam to substrate,
  or bottom-of-beam to surface RF top.

Geometry layout
---------------
  • Four support beams at corners, centre-to-centre spacing = 600 µm in x and y.
  • (window_n + 1) lattice ribs in each direction, creating window_n × window_n windows.
  • Outer rib positions coincide with support beam centres (±300 µm from origin).
  • Top of every lattice beam = top of support beam = z_top_beam (in final mesh coords).

Usage
-----
  python rf_cell_gen.py [options]

  --rf_height    H   vertical distance from surface RF top to beam cross-section
                     centre [µm]  (default 290)
  --rf_thickness T   scale factor for lattice beam cross-section (default 1.0)
  --window_n     N   windows per side (default 2 → 2×2 = 4 windows)
  --step             also write .step
  --mesh             also generate and write .msh
  --no-brep          suppress .brep output
  --gui              launch Gmsh GUI after build

Baseline example
----------------
  python rf_cell_gen.py --rf_height 290 --rf_thickness 1.0 --window_n 2

Smoke-test set (n = 1, 2, 3)
-----------------------------
  python rf_cell_gen.py --window_n 1 --rf_height 290 --rf_thickness 1.0
  python rf_cell_gen.py --window_n 2 --rf_height 290 --rf_thickness 1.0
  python rf_cell_gen.py --window_n 3 --rf_height 290 --rf_thickness 1.0
"""

import argparse
from typing import Optional
import sys

# ---------------------------------------------------------------------------
# Baseline geometric constants (all in µm)
# ---------------------------------------------------------------------------
SUPPORT_SIDE_UM    = 56.0    # support beam square cross-section side
SUPPORT_SPACING_UM = 600.0   # centre-to-centre distance between support beams
LATTICE_LENGTH_UM  = 656.0   # lattice beam length (fixed in v1)
LATTICE_DH_BASE_UM = 56.0    # baseline horizontal diamond diagonal
LATTICE_DV_BASE_UM = 82.0    # baseline vertical diamond diagonal


def um(x: float) -> float:
    """Convert µm → mm (Gmsh model unit)."""
    return x * 1e-3


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_rf_cell(
    rf_height: float,
    rf_thickness: float,
    window_n: int,
    rf_width_um: Optional[float] = None,
    njunctions: int = 2,
    junction_pitch_um: float = 600.0,
    out_brep: bool = True,
    out_step: bool = False,
    out_mesh: bool = False,
    gui: bool = False,
    base_step_path: Optional[str] = None,
) -> None:
    """
    Build one RF lattice cell and export requested file formats.

    Parameters
    ----------
    rf_height      : vertical distance [µm] from the TOP of the surface RF electrode
                     (z = 0 in the final assembled mesh) to the CENTROID of the 3D RF
                     beam diamond cross-section.  This is NOT total beam height, NOT
                     bottom-of-beam to substrate, and NOT bottom-of-beam to surface RF.
                     The beam cross-section then spans:
                       z_bottom_beam = z_top_surface_rf + rf_height - dv/2
                       z_top_beam    = z_top_surface_rf + rf_height + dv/2
    rf_thickness   : uniform scale factor applied to lattice beam cross-section
    window_n       : number of windows per side (total windows = window_n²)
    base_step_path : optional path to rf_base.step (the original RF base plate
                     with X-shaped cutout at z=[-0.01, 0]).  When provided it
                     is imported and fused with the parametric cell so the
                     complete RF electrode matches the original footprint.
    """
    import gmsh

    # ── rf_width_um overrides rf_thickness ───────────────────────────────────
    # When rf_width_um is provided, compute rf_thickness from it so that the
    # horizontal diamond diagonal equals rf_width_um.  This lets the sweep
    # work in physical µm units (10, 15, 20 ... µm) rather than scale factors.
    if rf_width_um is not None:
        rf_thickness = rf_width_um / LATTICE_DH_BASE_UM

    # ── derived dimensions (µm) ──────────────────────────────────────────────
    half_sp = SUPPORT_SPACING_UM / 2.0               # 300 µm from origin to corner

    dh = LATTICE_DH_BASE_UM * rf_thickness           # scaled horizontal diagonal
    dv = LATTICE_DV_BASE_UM * rf_thickness           # scaled vertical diagonal
    half_l = LATTICE_LENGTH_UM / 2.0                 # 328 µm half-length of lattice beam
    s_half  = dh / 2.0   # pillar half-width = rail width (keeps pillar == rail)

    # ── z coordinates of the lattice beam diamond cross-section ─────────────────
    #
    # Convention: rf_height = vertical distance from TOP of the surface RF electrode
    # to the CENTROID of the 3D RF beam cross-section (in final assembled mesh coords,
    # where z = 0 is the surface RF top).
    #
    # Implementation note: rf_cell_gen.py applies occ.translate(fused, 0, 0, -0.02 mm)
    # after building.  This shifts everything down by 20 µm so the support beam
    # bottoms align with the surface RF plate (z ∈ [−0.02, 0] mm in final mesh).
    # The build ("pre-translation") coordinate system therefore sits 20 µm ABOVE the
    # final mesh coordinate system.  All Gmsh addPoint/addBox calls below use
    # pre-translation values.
    #
    # Final-mesh z values (z = 0 at surface RF top):
    _TRANSLATE_DOWN_MM  = 0.02          # must match occ.translate below — do not change
    _TRANSLATE_OFFSET_UM = _TRANSLATE_DOWN_MM * 1e3   # 20.0 µm

    z_top_surface_rf = 0.0                          # final mesh: surface RF electrode top
    z_center_beam    = z_top_surface_rf + rf_height # final mesh: center of beam cross-section
    z_bottom_beam    = z_center_beam - dv / 2.0     # final mesh: bottom vertex of diamond
    z_top_beam       = z_center_beam + dv / 2.0     # final mesh: top vertex of diamond

    # Debug output: show all key z positions in final-mesh coordinates.
    print(
        f"  [z-layout] z_top_surface_rf = {z_top_surface_rf:.1f} µm  "
        f"(reference: top of surface RF, z=0 in assembled mesh)\n"
        f"  [z-layout] z_center_beam    = {z_center_beam:.1f} µm  "
        f"(= z_top_surface_rf + rf_height = {rf_height:.1f})\n"
        f"  [z-layout] z_bottom_beam    = {z_bottom_beam:.1f} µm  "
        f"(= z_center_beam - dv/2 = {z_center_beam:.1f} - {dv/2:.1f})\n"
        f"  [z-layout] z_top_beam       = {z_top_beam:.1f} µm  "
        f"(= z_center_beam + dv/2 = {z_center_beam:.1f} + {dv/2:.1f})"
    )

    if z_bottom_beam < 0.0:
        print(
            f"WARNING: lattice beam bottom vertex is {abs(z_bottom_beam):.1f} µm BELOW "
            "the surface RF electrode top (z = 0 in assembled mesh). "
            "Increase rf_height or reduce rf_thickness.",
            file=sys.stderr,
        )

    # Pre-translation build coordinates (add _TRANSLATE_OFFSET_UM to all final-mesh z):
    z_top    = z_top_beam    + _TRANSLATE_OFFSET_UM  # pre-translation diamond top vertex
    z_centre = z_center_beam + _TRANSLATE_OFFSET_UM  # pre-translation diamond centroid
    z_bot    = z_bottom_beam + _TRANSLATE_OFFSET_UM  # pre-translation diamond bottom vertex

    # ── rib positions along x and y (µm from origin) ────────────────────────
    if window_n < 1:
        raise ValueError(f"window_n must be ≥ 1, got {window_n}")

    rib_spacing = SUPPORT_SPACING_UM / window_n   # µm between adjacent ribs

    # Check that adjacent ribs do not overlap (minimum clear gap > 0)
    min_gap = rib_spacing - dh
    if min_gap <= 0.0:
        raise ValueError(
            f"Invalid geometry: rib spacing {rib_spacing:.1f} µm ≤ lattice beam "
            f"horizontal width {dh:.1f} µm (gap = {min_gap:.1f} µm). "
            "Reduce window_n or rf_thickness."
        )

    # Positions of ribs: n+1 equally spaced from -half_sp to +half_sp
    rib_pos = [-half_sp + i * rib_spacing for i in range(window_n + 1)]

    # ── Gmsh / OCC setup ─────────────────────────────────────────────────────
    gmsh.initialize()
    gmsh.model.add("rf_cell")
    occ = gmsh.model.occ

    solids: list[tuple[int, int]] = []   # (dim=3, tag) collected before fuse

    # ── 1. Support beams (pedestals) ────────────────────────────────────────
    # Four rectangular prisms at the corners of the 600 µm × 600 µm square.
    # Each pedestal spans from z=0 (pre-translation build coords) up to the top
    # vertex of the diamond beam (z_top, pre-translation), so the pedestal provides
    # a continuous conductive path from the surface RF plate up through the full
    # height of the diamond cross-section.
    # In final-mesh coordinates the pedestal top is at z_top_beam = rf_height + dv/2.
    corners = [
        (-half_sp, -half_sp),
        ( half_sp, -half_sp),
        ( half_sp,  half_sp),
        (-half_sp,  half_sp),
    ]
    for cx_um, cy_um in corners:
        tag = occ.addBox(
            um(cx_um - s_half), um(cy_um - s_half), um(0.0),
            um(dh), um(dh), um(z_top),   # z_top is pre-translation diamond top vertex
        )
        solids.append((3, tag))

    # ── 2. Lattice beam primitives ───────────────────────────────────────────
    # Each beam is built by:
    #   a) drawing the 4-vertex diamond in the cross-section plane at the
    #      near end of the beam, then
    #   b) extruding along the beam axis for LATTICE_LENGTH_UM.

    def _diamond_prism_along_x(y_um: float) -> int:
        """Create a diamond-section prism running along X, centred at y=y_um [µm]."""
        x0 = um(-half_l)        # starting x position (far left)
        # Diamond vertices in the Y-Z plane (all at x = x0)
        p = [
            occ.addPoint(x0, um(y_um),          um(z_top)),     # top
            occ.addPoint(x0, um(y_um + dh / 2), um(z_centre)),  # +y (right)
            occ.addPoint(x0, um(y_um),          um(z_bot)),     # bottom
            occ.addPoint(x0, um(y_um - dh / 2), um(z_centre)),  # -y (left)
        ]
        lines = [
            occ.addLine(p[0], p[1]),
            occ.addLine(p[1], p[2]),
            occ.addLine(p[2], p[3]),
            occ.addLine(p[3], p[0]),
        ]
        loop = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([loop])
        ext  = occ.extrude([(2, surf)], um(LATTICE_LENGTH_UM), 0.0, 0.0)
        vols = [tag for dim, tag in ext if dim == 3]
        return vols[0]

    def _diamond_prism_along_y(x_um: float) -> int:
        """Create a diamond-section prism running along Y, centred at x=x_um [µm]."""
        y0 = um(-half_l)        # starting y position (far front)
        # Diamond vertices in the X-Z plane (all at y = y0)
        p = [
            occ.addPoint(um(x_um),          y0, um(z_top)),     # top
            occ.addPoint(um(x_um + dh / 2), y0, um(z_centre)),  # +x (right)
            occ.addPoint(um(x_um),          y0, um(z_bot)),     # bottom
            occ.addPoint(um(x_um - dh / 2), y0, um(z_centre)),  # -x (left)
        ]
        lines = [
            occ.addLine(p[0], p[1]),
            occ.addLine(p[1], p[2]),
            occ.addLine(p[2], p[3]),
            occ.addLine(p[3], p[0]),
        ]
        loop = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([loop])
        ext  = occ.extrude([(2, surf)], 0.0, um(LATTICE_LENGTH_UM), 0.0)
        vols = [tag for dim, tag in ext if dim == 3]
        return vols[0]

    # Beams running along X at each rib Y position
    for y in rib_pos:
        tag = _diamond_prism_along_x(y)
        solids.append((3, tag))

    # Beams running along Y at each rib X position
    for x in rib_pos:
        tag = _diamond_prism_along_y(x)
        solids.append((3, tag))

    # ── 3. Boolean fuse ──────────────────────────────────────────────────────
    occ.synchronize()

    if len(solids) > 1:
        fused, _ = occ.fuse(
            [solids[0]], solids[1:],
            removeObject=True,
            removeTool=True,
        )
    else:
        fused = solids

    # ── Tile junctions along x ───────────────────────────────────────────────
    # For njunctions > 1, copy and translate the single-cell fused solid
    # by junction_pitch_um in x for each additional junction, then fuse.
    # This creates the linear region between junctions where ions are stored.
    if njunctions > 1:
        occ.synchronize()
        all_junctions = list(fused)
        for i in range(1, njunctions):
            copies = occ.copy(fused)
            occ.translate(copies, um(junction_pitch_um * i), 0.0, 0.0)
            all_junctions.extend(copies)
        occ.synchronize()
        fused, _ = occ.fuse(
            [all_junctions[0]], all_junctions[1:],
            removeObject=True, removeTool=True,
        )
        print(f"  Tiled {njunctions} junctions at pitch {junction_pitch_um} µm")

    # Shift the entire RF structure down by _TRANSLATE_DOWN_MM so the support beam
    # bottoms (currently at z=0 in build coords) land at z=-_TRANSLATE_DOWN_MM in the
    # final assembled mesh, aligning with the bottom of the surface RF plate.
    # After this translation: z=0 in the final mesh = z=_TRANSLATE_OFFSET_UM in build coords
    # = top of the surface RF electrode.  This must match _TRANSLATE_DOWN_MM above.
    occ.translate(fused, 0.0, 0.0, -_TRANSLATE_DOWN_MM)
    occ.synchronize()

    # ── 3b. Import and fuse RF base plate ────────────────────────────────────
    # The base plate (z=[-0.01, 0]) has an X-shaped cutout matching the
    # DC pad and ground corner positions — it must come from the original STEP
    # because reconstructing that cutout analytically is impractical.
    if base_step_path is not None:
        import os
        if not os.path.exists(base_step_path):
            raise FileNotFoundError(
                f"rf_base.step not found at: {base_step_path}\n"
                "Extract it from rf.step by running:\n"
                "  python extract_rf_base.py"
            )
        before = set(t for _, t in occ.getEntities(3))
        occ.importShapes(base_step_path)
        occ.synchronize()
        after  = set(t for _, t in occ.getEntities(3))
        base_vols = [(3, t) for t in sorted(after - before)]
        if base_vols:
            print(f"  Imported RF base plate: {len(base_vols)} vol(s) from {base_step_path}")
            # fused currently holds the translated parametric cell dimtags
            all_vols = [(d, t) for d, t in fused if d == 3] + base_vols
            if len(all_vols) > 1:
                fused, _ = occ.fuse(
                    all_vols[:1], all_vols[1:],
                    removeObject=True, removeTool=True,
                )
                occ.synchronize()
                print(f"  Fused into {len([x for x in fused if x[0]==3])} RF vol(s)")
        else:
            print("  WARNING: no volumes found in base_step_path — skipping base plate.")

    # ── 4. Export ─────────────────────────────────────────────────────────────
    t_int = int(round(rf_thickness * 100))
    stem  = f"rfcell_h{int(rf_height)}_t{t_int:03d}_n{window_n}_j{njunctions}"

    if out_brep:
        fname = stem + ".brep"
        gmsh.write(fname)
        print(f"  Wrote  {fname}")

    if out_step:
        fname = stem + ".step"
        gmsh.write(fname)
        print(f"  Wrote  {fname}")

    if out_mesh:
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)        # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",  um(8.0))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", um(40.0))
        gmsh.model.mesh.generate(3)
        fname = stem + ".msh"
        gmsh.write(fname)
        print(f"  Wrote  {fname}")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF lattice cell – parametric CAD/mesh generator (v1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rf_height", type=float, default=290.0,
        help="Vertical distance from the TOP of the surface RF electrode to the "
             "CENTROID of the 3D RF beam diamond cross-section [µm]. "
             "The beam spans ±(dv/2) around this centre in z, where dv = "
             "rf_thickness × 82 µm (baseline vertical diamond diagonal). "
             "Do NOT interpret as total beam height or as a substrate-referenced "
             "top-of-beam coordinate.",
    )
    parser.add_argument(
        "--rf_thickness", type=float, default=1.0,
        help="Uniform scale factor for lattice beam cross-section",
    )
    parser.add_argument(
        "--window_n", type=int, default=2,
        help="Number of windows per side (total = window_n²)",
    )
    parser.add_argument(
        "--njunctions", type=int, default=2,
        help="Number of junctions to tile end-to-end (default 2). "
             "Use 2 for the standard two-junction assembly with linear region.",
    )
    parser.add_argument(
        "--junction_pitch_um", type=float, default=600.0,
        help="Centre-to-centre junction pitch in µm (default 600).",
    )
    parser.add_argument(
        "--rf_width_um", type=float, default=None,
        help="Electrode width in µm (sets horizontal diamond diagonal directly). "
             "When provided, overrides --rf_thickness. "
             "Minimum recommended: 10 µm, step size: 5 µm.",
    )
    parser.add_argument(
        "--no-brep", dest="brep", action="store_false", default=True,
        help="Suppress .brep output",
    )
    parser.add_argument(
        "--step", action="store_true", default=False,
        help="Also export .step",
    )
    parser.add_argument(
        "--mesh", action="store_true", default=False,
        help="Also generate and export .msh (3D mesh)",
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Launch Gmsh GUI after build",
    )
    parser.add_argument(
        "--base-step", type=str, default=None,
        dest="base_step",
        help="Path to rf_base.step (original RF base plate with X-shaped cutout). "
             "When provided, fused with the parametric cell for a complete RF electrode.",
    )
    args = parser.parse_args()

    _eff_thickness = (args.rf_width_um / LATTICE_DH_BASE_UM
                      if args.rf_width_um is not None else args.rf_thickness)
    _eff_width_um  = args.rf_width_um if args.rf_width_um is not None                      else args.rf_thickness * LATTICE_DH_BASE_UM
    _dv_eff = 82.0 * _eff_thickness   # effective vertical diamond diagonal [µm]
    print(
        f"\nRF lattice cell\n"
        f"  rf_height (beam-centre above surface RF top) = {args.rf_height} µm\n"
        f"  rf_width_um  = {_eff_width_um:.1f} µm  "
        f"(rf_thickness = {_eff_thickness:.4f})\n"
        f"  window_n     = {args.window_n}  "
        f"({args.window_n}×{args.window_n} = {args.window_n**2} windows, "
        f"{args.window_n + 1} ribs/side)\n"
        f"  beam z-layout (in assembled mesh, z=0 at surface RF top):\n"
        f"    z_top_surface_rf = 0.0 µm\n"
        f"    z_center_beam    = {args.rf_height:.1f} µm\n"
        f"    z_bottom_beam    = {args.rf_height - _dv_eff/2:.1f} µm\n"
        f"    z_top_beam       = {args.rf_height + _dv_eff/2:.1f} µm\n"
    )

    try:
        build_rf_cell(
            rf_height         = args.rf_height,
            rf_thickness      = args.rf_thickness,
            window_n          = args.window_n,
            rf_width_um       = args.rf_width_um,
            njunctions        = args.njunctions,
            junction_pitch_um = args.junction_pitch_um,
            out_brep          = args.brep,
            out_step          = args.step,
            out_mesh          = args.mesh,
            gui               = args.gui,
            base_step_path    = args.base_step,
        )
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.\n")


if __name__ == "__main__":
    main()