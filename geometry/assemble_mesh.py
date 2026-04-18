#!/usr/bin/env python3
import argparse
import gmsh
import sys


RF_TAG = 1
DC_TAG = 2
GROUND_TAG = 3
OUTER_TAG = 4
VACUUM_TAG = 100


def only_dim(dimtags, dim):
    return [dt for dt in dimtags if dt[0] == dim]


def just_tags(dimtags, dim=None):
    if dim is None:
        return [t for _, t in dimtags]
    return [t for d, t in dimtags if d == dim]


def union_bbox(dimtags):
    if not dimtags:
        raise RuntimeError("No entities given for bbox union.")
    xmins, ymins, zmins, xmaxs, ymaxs, zmaxs = [], [], [], [], [], []
    for d, t in dimtags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(d, t)
        xmins.append(xmin)
        ymins.append(ymin)
        zmins.append(zmin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        zmaxs.append(zmax)
    return min(xmins), min(ymins), min(zmins), max(xmaxs), max(ymaxs), max(zmaxs)


def import_step_volumes(path):
    ents = gmsh.model.occ.importShapes(path)
    vols = only_dim(ents, 3)
    if not vols:
        raise RuntimeError(f"{path} did not import any 3D volumes. Are these closed solids?")
    return vols


def classify_vacuum_surfaces(vacuum_vols, rf_vols, dc_vols, ground_vols):
    vac_set = set(vacuum_vols)
    rf_set = set(rf_vols)
    dc_set = set(dc_vols)
    gnd_set = set(ground_vols)

    rf_surfs = set()
    dc_surfs = set()
    gnd_surfs = set()
    outer_surfs = set()
    unknown_surfs = set()

    for v in vacuum_vols:
        bnd = gmsh.model.getBoundary([(3, v)], combined=False, oriented=False, recursive=False)
        for dim, s in bnd:
            if dim != 2:
                continue

            up, down = gmsh.model.getAdjacencies(2, s)
            up = set(int(x) for x in up)

            if not (up & vac_set):
                continue

            other = up - vac_set

            if not other:
                outer_surfs.add(s)
            elif other & rf_set:
                rf_surfs.add(s)
            elif other & dc_set:
                dc_surfs.add(s)
            elif other & gnd_set:
                gnd_surfs.add(s)
            else:
                unknown_surfs.add(s)

    return rf_surfs, dc_surfs, gnd_surfs, outer_surfs, unknown_surfs


def clip_volumes_to_x(vol_tags, x_lo, x_hi, ymin, ymax, zmin, zmax, margin=1.0):
    """Clip 3-D volumes to x ∈ [x_lo, x_hi] via OCC intersect with a slab box.

    Volumes already fully inside [x_lo, x_hi] pass through unchanged.
    Parts outside are removed.  Returns list of (3, tag) pairs.

    A flat-plane clip is far more robust than fusing overlapping solids:
    the resulting faces are co-planar, so tiled copies meet at flush faces
    that OCC fuse can merge without creating degenerate topology.
    """
    clip = gmsh.model.occ.addBox(
        x_lo, ymin - margin, zmin - margin,
        x_hi - x_lo,
        (ymax - ymin) + 2 * margin,
        (zmax - zmin) + 2 * margin,
    )
    result, _ = gmsh.model.occ.intersect(
        [(3, t) for t in vol_tags],
        [(3, clip)],
        removeObject=True,
        removeTool=True,
    )
    return [(d, t) for d, t in result if d == 3]


def tile_and_fuse_volumes(base_vols, n_junctions, pitch_x):
    """Copy, translate, and fuse *base_vols* along x for n_junctions unit cells.

    Assumes base_vols have already been clipped to exactly pitch_x wide so
    copies meet at flush co-planar faces.  OCC fuse of touching flat faces is
    clean and produces a single continuous solid with no degenerate seams.

    Returns the list of (3, tag) pairs after the fuse.
    """
    if n_junctions == 1:
        return [(3, t) for t in base_vols]

    all_dimtags = [(3, t) for t in base_vols]
    for i in range(1, n_junctions):
        copies = gmsh.model.occ.copy([(3, v) for v in base_vols])
        gmsh.model.occ.translate(copies, pitch_x * i, 0.0, 0.0)
        all_dimtags.extend((d, t) for d, t in copies if d == 3)

    fused, _ = gmsh.model.occ.fuse(
        all_dimtags[:1], all_dimtags[1:],
        removeObject=True, removeTool=True,
    )
    return [(d, t) for d, t in fused if d == 3]


def add_mesh_size_fields(
    rf_surfs, dc_surfs, gnd_surfs,
    lc_electrode, lc_center, lc_far,
    dist_min, dist_max,
    center_radius, trap_centers,
):
    """Set mesh sizes via Distance+Threshold from electrode surfaces + Ball(s) at trap centres.

    Uses Gmsh background mesh fields exclusively:
      1. Distance field measuring distance from all electrode surfaces (geometry-based).
      2. Threshold field that interpolates from lc_electrode (at the surface) to
         lc_far (beyond dist_max).
      3. One Ball field per entry in *trap_centers* for additional local refinement.
         Pass a list of [x, y, z] points — one per linear region and/or junction centre.
      4. Min field combining all above — every point gets the finest applicable size.

    This approach covers the full electrode surface area, not just geometric corner
    points, and avoids the circular dependency of older Distance-from-mesh approaches.

    Parameters
    ----------
    trap_centers : list of [x, y, z] or None
        Positions (mesh units) of all trap-centre Ball fields.  Pass an empty
        list or None to disable Ball refinement.
    """
    all_electrode_surfs = sorted(rf_surfs | dc_surfs | gnd_surfs)

    fields = []

    # Distance from electrode surfaces (Gmsh 4.8+ evaluates against CAD geometry)
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", all_electrode_surfs)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", lc_electrode)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", lc_far)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", dist_max)
    fields.append(f_thresh)
    print(f"[fields] Distance+Threshold: lc={lc_electrode} at surfaces → "
          f"lc={lc_far} beyond dist={dist_max} (surfaces: {all_electrode_surfs})")

    # Ball field(s) around expected trap centre(s).
    # Each entry is a (xyz, radius) tuple so different positions can carry
    # their own radius (e.g. outer arms vs inter-junction midpoint).
    if trap_centers:
        for tc, r_ball in trap_centers:
            f_ball = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(f_ball, "VIn",     lc_center)
            gmsh.model.mesh.field.setNumber(f_ball, "VOut",    lc_far)
            gmsh.model.mesh.field.setNumber(f_ball, "Radius",  r_ball)
            gmsh.model.mesh.field.setNumber(f_ball, "XCenter", tc[0])
            gmsh.model.mesh.field.setNumber(f_ball, "YCenter", tc[1])
            gmsh.model.mesh.field.setNumber(f_ball, "ZCenter", tc[2])
            fields.append(f_ball)
            print(f"[fields] Ball(lc={lc_center}) centred at {tc}, r={r_ball}")

    # Min field: every point gets the finest size from all active fields
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # Background field is the sole size source; disable other mechanisms
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints",         0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",      0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True, help="RF STEP file")
    ap.add_argument("--dc", required=True, help="DC STEP file")
    ap.add_argument("--ground", required=True, help="Ground STEP file")
    ap.add_argument("--out", required=True, help="Output .msh file")

    # ── Vacuum domain padding ──────────────────────────────────────────────
    # Independent x/y and z controls so the domain is deep enough above the
    # electrode plane (where the trap minimum lives at z ≈ 80–130 µm) without
    # blowing up unnecessarily in x/y.
    ap.add_argument("--pad", type=float, default=None,
                    help="Uniform padding override (overrides --pad-xy/z-bot/z-top)")
    ap.add_argument("--pad-xy", type=float, default=0.200,
                    help="Symmetric x/y padding beyond conductor bbox (mm, default 0.200)")
    ap.add_argument("--pad-z-bot", type=float, default=0.200,
                    help="Padding below conductor bbox in z (mm, default 0.200)")
    ap.add_argument("--pad-z-top", type=float, default=0.600,
                    help="Padding above conductor bbox in z (mm, default 0.600). "
                         "Must be large enough that the Neumann outer BC does not "
                         "distort the field at the trap minimum (~80–130 µm above "
                         "the electrode surface).  0.6 mm gives ~5× the trap height "
                         "of clearance and is the recommended default for both single- "
                         "and multi-junction meshes.")

    # ── Mesh size field parameters (mm) ───────────────────────────────────
    ap.add_argument("--lc-electrode", type=float, default=0.003,
                    help="Target element size on electrode surfaces (mm, default 0.003)")
    ap.add_argument("--lc-center", type=float, default=0.005,
                    help="Target element size inside trap-centre ball (mm, default 0.005)")
    ap.add_argument("--lc-far", type=float, default=0.030,
                    help="Target element size in the far field (mm, default 0.030)")
    ap.add_argument("--dist-min", type=float, default=0.0,
                    help="Distance from electrode surface where fine mesh starts (mm)")
    ap.add_argument("--dist-max", type=float, default=0.050,
                    help="Distance from electrode surface where far-field size is reached (mm)")
    ap.add_argument("--center-radius", type=float, default=0.060,
                    help="Radius of the fine-mesh ball around the primary trap centre "
                         "(mm, default 0.060).  Covers ±60 µm around the estimated centre, "
                         "enough to contain the trap even if the z-offset estimate is 30–40 µm "
                         "off.  Outer-arm balls use --center-radius-junction (default 0.070).")
    ap.add_argument("--trap-center", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Primary expected trap-centre position in mesh units (mm). "
                         "Adds one Ball refinement field at this point. "
                         "Default: auto-detected from RF surface geometry (see below).")
    ap.add_argument("--trap-center-z-offset", type=float, default=0.082,
                    help="Z offset above the RF electrode top surface used when "
                         "auto-estimating the trap centre for the PRIMARY (inter-junction "
                         "or single-junction) ball field (mm, default 0.082 ≈ paper "
                         "linear-region trap height of 82.3 µm).")
    ap.add_argument("--trap-center-z-offset-junction", type=float, default=0.082,
                    help="Z offset used for the automatically-added outer-arm Ball fields "
                         "in multi-junction meshes (mm, default 0.082 ≈ paper linear-region "
                         "trap height of 82.3 µm).  The outer arms of a multi-junction array "
                         "have the same electrode cross-section as the single-junction linear "
                         "region, so 0.082 is the correct default.  The inter-junction arm "
                         "(primary ball, --trap-center-z-offset) has a lower trap height "
                         "(~45 µm observed) but is covered by the primary ball + z-offset.")
    ap.add_argument("--center-radius-junction", type=float, default=0.070,
                    help="Ball radius for the auto-added outer-arm Ball fields in "
                         "multi-junction meshes (mm, default 0.070).  Larger than the "
                         "primary --center-radius (0.040 mm) to tolerate ±30 µm uncertainty "
                         "in the exact trap height, ensuring the fine-mesh region always "
                         "covers the actual RF null location.")
    ap.add_argument("--no-center-ball", action="store_true",
                    help="Disable all trap-centre Ball fields (use only Distance/Threshold).")
    ap.add_argument("--linear-region-x", type=float, nargs="+", default=None,
                    metavar="X",
                    help="One or more x-coordinates (mm) of linear-region trap centres "
                         "for extra Ball refinement.  Useful for multi-junction meshes "
                         "where you want fine mesh at each linear arm midpoint AND each "
                         "junction crossing.  Y is set to 0; Z is RF-top + "
                         "--trap-center-z-offset. "
                         "Example for 2-junction 0.6 mm pitch: "
                         "--linear-region-x -0.15 0.30 0.75  "
                         "(left arm, inter-junction midpoint, right arm).")

    # ── Multi-junction tiling ──────────────────────────────────────────────
    ap.add_argument("--njunctions", type=int, default=1,
                    help="Number of x-junctions to tile end-to-end (default 1). "
                         "Use 2 for two connected x-junctions.")
    ap.add_argument("--junction-pitch", type=float, default=None,
                    help="Center-to-center junction pitch in x for tiling (mm). "
                         "Default: 0.600 mm (the standard cell pitch). "
                         "The physical cell is wider than this, so outer RF arms will "
                         "overlap — the fuse merges them into one continuous beam.")

    ap.add_argument("--nopopup", action="store_true")
    args = ap.parse_args()

    # Resolve padding values
    if args.pad is not None:
        pad_xy = args.pad
        pad_z_bot = args.pad
        pad_z_top = args.pad
    else:
        pad_xy = args.pad_xy
        pad_z_bot = args.pad_z_bot
        pad_z_top = args.pad_z_top

    gmsh.initialize([sys.argv[0]])  # pass only the program name; argparse owns the rest
    gmsh.option.setNumber("General.Terminal", 1)
    # Safety bounds — the background field will drive actual sizes.
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.lc_electrode * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.lc_far)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.add("trap_vacuum")

    # 1) import conductor solids
    rf_in = import_step_volumes(args.rf)
    dc_in = import_step_volumes(args.dc)
    gnd_in = import_step_volumes(args.ground)

    gmsh.model.occ.synchronize()

    # 1b) tile junctions in x if requested
    if args.njunctions > 1:
        all_cond_single = rf_in + dc_in + gnd_in
        xmin0, ymin0, zmin0, xmax0, ymax0, zmax0 = union_bbox(all_cond_single)
        cell_span_x = xmax0 - xmin0
        pitch_x = args.junction_pitch if args.junction_pitch is not None else 0.600
        print(f"[tiling] {args.njunctions} junctions, pitch_x = {pitch_x:.4f} mm, "
              f"cell bbox x-span = {cell_span_x:.4f} mm "
              f"({'overlap' if cell_span_x > pitch_x else 'gap'} = "
              f"{abs(cell_span_x - pitch_x):.4f} mm)")

        # If the physical cell is wider than the pitch, clip each electrode to
        # [x_c - pitch/2, x_c + pitch/2] before tiling.  This converts an
        # overlap into a flush face — OCC fuse of co-planar flat faces is
        # trivially robust, whereas fusing overlapping solids creates degenerate
        # topology (AlertAcquiredSelfIntersection, wire-fix failures, etc.).
        if cell_span_x > pitch_x + 1e-6:
            x_c = (xmin0 + xmax0) / 2.0
            x_lo = x_c - pitch_x / 2.0
            x_hi = x_c + pitch_x / 2.0
            trim = (cell_span_x - pitch_x) / 2.0
            print(f"[tiling] clipping {trim * 1e3:.2f} µm per side "
                  f"→ x ∈ [{x_lo:.4f}, {x_hi:.4f}] mm")

            rf_in  = clip_volumes_to_x(just_tags(rf_in,  3), x_lo, x_hi,
                                        ymin0, ymax0, zmin0, zmax0)
            dc_in  = clip_volumes_to_x(just_tags(dc_in,  3), x_lo, x_hi,
                                        ymin0, ymax0, zmin0, zmax0)
            gnd_in = clip_volumes_to_x(just_tags(gnd_in, 3), x_lo, x_hi,
                                        ymin0, ymax0, zmin0, zmax0)
            gmsh.model.occ.synchronize()
            print(f"[tiling] after clip: rf={len(rf_in)}, "
                  f"dc={len(dc_in)}, gnd={len(gnd_in)}")

        # Tile at pitch_x.  Copies now meet at flush co-planar faces; fuse
        # merges them into one continuous body per electrode type.
        rf_in  = tile_and_fuse_volumes(just_tags(rf_in,  3), args.njunctions, pitch_x)
        dc_in  = tile_and_fuse_volumes(just_tags(dc_in,  3), args.njunctions, pitch_x)
        gnd_in = tile_and_fuse_volumes(just_tags(gnd_in, 3), args.njunctions, pitch_x)
        gmsh.model.occ.synchronize()
        print(f"[tiling] after tile+fuse: rf={len(rf_in)} vol(s), "
              f"dc={len(dc_in)} vol(s), gnd={len(gnd_in)} vol(s)")

    all_cond_in = rf_in + dc_in + gnd_in
    xmin, ymin, zmin, xmax, ymax, zmax = union_bbox(all_cond_in)

    # 2) build vacuum box — asymmetric z padding keeps the outer boundary
    #    well away from the trap minimum (z ≈ 80 µm above electrode plane)
    box = gmsh.model.occ.addBox(
        xmin - pad_xy,   ymin - pad_xy,   zmin - pad_z_bot,
        (xmax - xmin) + 2 * pad_xy,
        (ymax - ymin) + 2 * pad_xy,
        (zmax - zmin) + pad_z_bot + pad_z_top,
    )
    print(f"[conductors] bbox x=[{xmin:.4f}, {xmax:.4f}]  "
          f"y=[{ymin:.4f}, {ymax:.4f}]  z=[{zmin:.4f}, {zmax:.4f}]  (mm)")
    print(f"[vacuum box] x=[{xmin - pad_xy:.4f}, {xmax + pad_xy:.4f}]  "
          f"y=[{ymin - pad_xy:.4f}, {ymax + pad_xy:.4f}]  "
          f"z=[{zmin - pad_z_bot:.4f}, {zmax + pad_z_top:.4f}]  (mm)")

    # 3) fragment box with conductors
    # out_map contains one entry per input entity:
    # [box, rf_1, rf_2, ..., dc_1, ..., gnd_1, ...]
    objects = [(3, box)]
    tools = rf_in + dc_in + gnd_in
    out, out_map = gmsh.model.occ.fragment(objects, tools, removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    n_rf = len(rf_in)
    n_dc = len(dc_in)
    n_gnd = len(gnd_in)

    rf_maps = out_map[1:1 + n_rf]
    dc_maps = out_map[1 + n_rf:1 + n_rf + n_dc]
    gnd_maps = out_map[1 + n_rf + n_dc:1 + n_rf + n_dc + n_gnd]

    rf_vols = set()
    for m in rf_maps:
        rf_vols.update(just_tags(m, 3))

    dc_vols = set()
    for m in dc_maps:
        dc_vols.update(just_tags(m, 3))

    gnd_vols = set()
    for m in gnd_maps:
        gnd_vols.update(just_tags(m, 3))

    all_out_vols = set(just_tags(only_dim(out, 3)))
    conductor_vols = rf_vols | dc_vols | gnd_vols
    vacuum_vols = all_out_vols - conductor_vols

    if not vacuum_vols:
        raise RuntimeError("No vacuum volume found after fragment.")

    # 4) classify vacuum boundary surfaces by adjacency
    rf_surfs, dc_surfs, gnd_surfs, outer_surfs, unknown_surfs = classify_vacuum_surfaces(
        vacuum_vols, rf_vols, dc_vols, gnd_vols
    )

    print("=== volume summary ===")
    print("vacuum volumes :", sorted(vacuum_vols))
    print("rf volumes     :", sorted(rf_vols))
    print("dc volumes     :", sorted(dc_vols))
    print("ground volumes :", sorted(gnd_vols))

    print("=== surface summary ===")
    print("rf surfaces     :", len(rf_surfs))
    print("dc surfaces     :", len(dc_surfs))
    print("ground surfaces :", len(gnd_surfs))
    print("outer surfaces  :", len(outer_surfs))
    print("unknown surfaces:", len(unknown_surfs))

    if unknown_surfs:
        print("Unknown surface IDs:", sorted(unknown_surfs))
        raise RuntimeError("Some vacuum boundary faces could not be classified.")

    # 5) create physical groups
    gmsh.model.addPhysicalGroup(3, sorted(vacuum_vols), VACUUM_TAG)
    gmsh.model.setPhysicalName(3, VACUUM_TAG, "vacuum")

    gmsh.model.addPhysicalGroup(2, sorted(rf_surfs), RF_TAG)
    gmsh.model.setPhysicalName(2, RF_TAG, "rf")

    gmsh.model.addPhysicalGroup(2, sorted(dc_surfs), DC_TAG)
    gmsh.model.setPhysicalName(2, DC_TAG, "dc")

    gmsh.model.addPhysicalGroup(2, sorted(gnd_surfs), GROUND_TAG)
    gmsh.model.setPhysicalName(2, GROUND_TAG, "ground")

    gmsh.model.addPhysicalGroup(2, sorted(outer_surfs), OUTER_TAG)
    gmsh.model.setPhysicalName(2, OUTER_TAG, "outer")

    # 6) mesh size fields
    # Build the list of (centre, radius) pairs for Ball refinement fields.
    # For multi-junction meshes this includes:
    #   • primary ball at the inter-junction midpoint (z-offset = trap height there)
    #   • arm balls at each outer arm (z-offset = outer-arm trap height)
    # Using separate z-offsets for inter-junction vs outer-arm positions ensures
    # the fine-mesh region is actually centred on the expected trap minimum in
    # each location, not 30–40 µm above it.
    trap_centers = []   # list of (xyz, radius) tuples
    if not args.no_center_ball:
        rf_bbox = union_bbox([(2, s) for s in rf_surfs])
        rf_x_min, rf_y_min, _, rf_x_max, rf_y_max, rf_z_top = rf_bbox
        rf_x_mid = (rf_x_min + rf_x_max) / 2.0
        rf_y_mid = (rf_y_min + rf_y_max) / 2.0

        # Primary z-offset: inter-junction arm height or single-junction height
        z_ball_primary = rf_z_top + args.trap_center_z_offset
        # Outer-arm z-offset (may differ — outer arm has different RF geometry)
        z_ball_arm     = rf_z_top + args.trap_center_z_offset_junction
        # Per-arm ball radius (default: same as primary)
        r_arm = args.center_radius_junction if args.center_radius_junction is not None \
                else args.center_radius

        if args.trap_center is not None:
            # Explicit single override — honour it exactly
            trap_centers.append((args.trap_center, args.center_radius))
            print(f"[trap centre] explicit override: {args.trap_center}, r={args.center_radius}")
        else:
            # Auto-estimate primary centre from RF surface centroid.
            primary = [rf_x_mid, rf_y_mid, z_ball_primary]
            trap_centers.append((primary, args.center_radius))
            print(f"[trap centre] auto-estimated primary: {primary}, r={args.center_radius}")

            # For multi-junction meshes add outer-arm balls with the correct
            # per-arm z-offset so the fine-mesh region covers the actual trap
            # height in those arms.
            if args.njunctions > 1:
                pitch_x = args.junction_pitch if args.junction_pitch is not None else 0.600
                # Left outer arm: quarter-pitch inward from left RF edge
                x_left  = rf_x_min + pitch_x * 0.25
                # Right outer arm: quarter-pitch inward from right RF edge
                x_right = rf_x_max - pitch_x * 0.25
                for x_extra in [x_left, x_right]:
                    extra = [x_extra, rf_y_mid, z_ball_arm]
                    trap_centers.append((extra, r_arm))
                    print(f"[trap centre] auto-added arm ball: {extra}, r={r_arm}")

        # Additional explicit linear-region x positions (use primary z-offset)
        if args.linear_region_x:
            for x_lr in args.linear_region_x:
                lr_centre = [float(x_lr), rf_y_mid, z_ball_primary]
                trap_centers.append((lr_centre, args.center_radius))
                print(f"[trap centre] --linear-region-x extra ball: {lr_centre}")

    add_mesh_size_fields(
        rf_surfs, dc_surfs, gnd_surfs,
        lc_electrode=args.lc_electrode,
        lc_center=args.lc_center,
        lc_far=args.lc_far,
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        center_radius=args.center_radius,
        trap_centers=trap_centers,
    )

    # 7) generate and write mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(args.out)

    if not args.nopopup:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    main()
