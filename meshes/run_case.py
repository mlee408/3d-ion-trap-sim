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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True, help="RF STEP file")
    ap.add_argument("--dc", required=True, help="DC STEP file")
    ap.add_argument("--ground", required=True, help="Ground STEP file")
    ap.add_argument("--out", required=True, help="Output .msh file")
    ap.add_argument("--pad", type=float, default=100.0,
                    help="Padding added to conductor bbox to build vacuum box")
    ap.add_argument("--lc", type=float, default=20.0,
                    help="Global target mesh size")
    ap.add_argument("--nopopup", action="store_true")
    args = ap.parse_args()

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.lc)
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    gmsh.model.add("trap_vacuum")

    # 1) import conductor solids
    rf_in = import_step_volumes(args.rf)
    dc_in = import_step_volumes(args.dc)
    gnd_in = import_step_volumes(args.ground)

    gmsh.model.occ.synchronize()

    all_cond_in = rf_in + dc_in + gnd_in
    xmin, ymin, zmin, xmax, ymax, zmax = union_bbox(all_cond_in)

    # 2) build vacuum box
    pad = args.pad
    box = gmsh.model.occ.addBox(
        xmin - pad, ymin - pad, zmin - pad,
        (xmax - xmin) + 2 * pad,
        (ymax - ymin) + 2 * pad,
        (zmax - zmin) + 2 * pad
    )

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

    box_map = out_map[0]
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

    # 6) mesh only the physical groups we kept
    gmsh.model.mesh.generate(3)
    gmsh.write(args.out)

    if not args.nopopup:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    main()
