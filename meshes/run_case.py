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


def add_mesh_size_fields(
    rf_surfs, dc_surfs, gnd_surfs,
    lc_electrode, lc_center, lc_far,
    dist_min, dist_max,
    center_radius, trap_center,
):
    """Set mesh sizes via Distance+Threshold from electrode surfaces + Ball at trap centre.

    Uses Gmsh background mesh fields exclusively:
      1. Distance field measuring distance from all electrode surfaces (geometry-based).
      2. Threshold field that interpolates from lc_electrode (at the surface) to
         lc_far (beyond dist_max).
      3. Optional Ball field around the expected trap centre for additional refinement.
      4. Min field combining (2) and (3) — every point gets the finest applicable size.

    This approach covers the full electrode surface area, not just geometric corner
    points, and avoids the circular dependency of older Distance-from-mesh approaches.
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

    # Ball field around expected trap centre
    if trap_center is not None:
        f_ball = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(f_ball, "VIn",     lc_center)
        gmsh.model.mesh.field.setNumber(f_ball, "VOut",    lc_far)
        gmsh.model.mesh.field.setNumber(f_ball, "Radius",  center_radius)
        gmsh.model.mesh.field.setNumber(f_ball, "XCenter", trap_center[0])
        gmsh.model.mesh.field.setNumber(f_ball, "YCenter", trap_center[1])
        gmsh.model.mesh.field.setNumber(f_ball, "ZCenter", trap_center[2])
        fields.append(f_ball)
        print(f"[fields] Ball(lc={lc_center}) centred at {trap_center}, r={center_radius}")

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
    # electrode plane (where the trap minimum lives at z ≈ 80 µm) without
    # blowing up unnecessarily in x/y.
    ap.add_argument("--pad", type=float, default=None,
                    help="Uniform padding override (overrides --pad-xy/z-bot/z-top)")
    ap.add_argument("--pad-xy", type=float, default=0.200,
                    help="Symmetric x/y padding beyond conductor bbox (mm, default 0.200)")
    ap.add_argument("--pad-z-bot", type=float, default=0.200,
                    help="Padding below conductor bbox in z (mm, default 0.200)")
    ap.add_argument("--pad-z-top", type=float, default=0.400,
                    help="Padding above conductor bbox in z (mm, default 0.400). "
                         "Needs to be large: trap minimum is ~0.080 mm above electrode "
                         "surface and the outer BC must not distort the field there.")

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
    ap.add_argument("--center-radius", type=float, default=0.040,
                    help="Radius of the fine-mesh ball around the trap centre (mm, default 0.040)")
    ap.add_argument("--trap-center", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Expected trap-centre position in mesh units (mm). "
                         "Default: RF surface centroid + --trap-center-z-offset.")
    ap.add_argument("--trap-center-z-offset", type=float, default=0.080,
                    help="Z offset added to the RF surface centroid to estimate the "
                         "trap centre when --trap-center is not given (mm, default 0.080)")
    ap.add_argument("--no-center-ball", action="store_true",
                    help="Disable the trap-centre Ball field (use only Distance/Threshold)")

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
    if not args.no_center_ball:
        if args.trap_center is not None:
            trap_center = args.trap_center
        else:
            # Auto-estimate: RF surface bbox centroid + z offset.
            # The RF surfaces define the quadrupole null axis; the trap minimum
            # is typically ~80 µm above the electrode plane in z.
            rf_bbox = union_bbox([(2, s) for s in rf_surfs])
            trap_center = [
                (rf_bbox[0] + rf_bbox[3]) / 2.0,
                (rf_bbox[1] + rf_bbox[4]) / 2.0,
                rf_bbox[5] + args.trap_center_z_offset,  # above RF top surface
            ]
            print(f"[trap centre] auto-estimated from RF surface centroid: {trap_center}")
    else:
        trap_center = None

    add_mesh_size_fields(
        rf_surfs, dc_surfs, gnd_surfs,
        lc_electrode=args.lc_electrode,
        lc_center=args.lc_center,
        lc_far=args.lc_far,
        dist_min=args.dist_min,
        dist_max=args.dist_max,
        center_radius=args.center_radius,
        trap_center=trap_center,
    )

    # 7) generate and write mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(args.out)

    if not args.nopopup:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    main()
