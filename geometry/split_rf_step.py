import gmsh
import os
import sys

INPUT_STEP = "rf.step"
OUTPUT_SURFACE = "rf_surface.step"
OUTPUT_3D = "rf_3d.step"

Z_SPLIT = 0.01   # mm = 20 um
EPS = 1e-6       # mm
PAD_XY = 1.0     # mm
PAD_Z = 1.0      # mm


def get_combined_bbox(dimtags):
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for dim, tag in dimtags:
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(dim, tag)
        xmin = min(xmin, x0)
        ymin = min(ymin, y0)
        zmin = min(zmin, z0)
        xmax = max(xmax, x1)
        ymax = max(ymax, y1)
        zmax = max(zmax, z1)
    return xmin, ymin, zmin, xmax, ymax, zmax


def split_and_export(input_step, output_step, keep_lower):
    gmsh.clear()
    gmsh.model.add("split_model")

    gmsh.model.occ.importShapes(input_step)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    if not vols:
        raise RuntimeError("No 3D volumes found in input STEP.")

    xmin, ymin, zmin, xmax, ymax, zmax = get_combined_bbox(vols)

    dx = (xmax - xmin) + 2 * PAD_XY
    dy = (ymax - ymin) + 2 * PAD_XY

    if keep_lower:
        box_z0 = zmin - PAD_Z
        box_dz = (Z_SPLIT + EPS) - box_z0
    else:
        box_z0 = Z_SPLIT - EPS
        box_dz = (zmax + PAD_Z) - box_z0

    cutter = gmsh.model.occ.addBox(
        xmin - PAD_XY,
        ymin - PAD_XY,
        box_z0,
        dx,
        dy,
        box_dz
    )
    gmsh.model.occ.synchronize()

    result, _ = gmsh.model.occ.intersect(
        vols,
        [(3, cutter)],
        removeObject=True,
        removeTool=True
    )
    gmsh.model.occ.synchronize()

    result_vols = [dt for dt in result if dt[0] == 3]
    if not result_vols:
        raise RuntimeError(f"No output volume generated for {output_step}")

    gmsh.write(output_step)
    print(f"Wrote {output_step}")


def main():
    if not os.path.exists(INPUT_STEP):
        print(f"ERROR: {INPUT_STEP} not found")
        sys.exit(1)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    try:
        split_and_export(INPUT_STEP, OUTPUT_SURFACE, keep_lower=True)
        split_and_export(INPUT_STEP, OUTPUT_3D, keep_lower=False)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
