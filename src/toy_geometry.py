import gmsh
import os

def generate_strip_trap_2d(
    filename="meshes/toy/strip_2d.msh",
    width=200e-6,
    height=200e-6,
    rf_width=80e-6,
    gap=20e-6,
    mesh_size=10e-6,
):
    """
    Generates a simple 2D strip surface trap cross-section.
    Bottom boundary is split into:
        RF electrode | gap | GND electrode
    """

    gmsh.initialize()
    gmsh.model.add("strip_trap_2d")

    # Coordinates
    x0 = -width / 2
    x1 = width / 2
    y0 = 0
    y1 = height

    # Points
    p1 = gmsh.model.geo.addPoint(x0, y0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(x1, y0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(x1, y1, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(x0, y1, 0, mesh_size)

    # Rectangle edges
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Split bottom boundary into RF | gap | GND
    rf_left = -rf_width/2
    rf_right = rf_width/2
    gap_left = rf_right
    gap_right = rf_right + gap

    p_rf_left = gmsh.model.geo.addPoint(rf_left, y0, 0, mesh_size)
    p_rf_right = gmsh.model.geo.addPoint(rf_right, y0, 0, mesh_size)
    p_gap_right = gmsh.model.geo.addPoint(gap_right, y0, 0, mesh_size)

    # Split original bottom line
    gmsh.model.geo.remove([(1, l1)])

    l_rf = gmsh.model.geo.addLine(p1, p_rf_left)
    l_rf2 = gmsh.model.geo.addLine(p_rf_left, p_rf_right)
    l_gap = gmsh.model.geo.addLine(p_rf_right, p_gap_right)
    l_gnd = gmsh.model.geo.addLine(p_gap_right, p2)

    # Create surface
    cl = gmsh.model.geo.addCurveLoop([l_rf, l_rf2, l_gap, l_gnd, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Tag boundaries
    RF_TAG = 10
    GND_TAG = 1

    gmsh.model.addPhysicalGroup(1, [l_rf2], RF_TAG)
    gmsh.model.setPhysicalName(1, RF_TAG, "RF")

    gmsh.model.addPhysicalGroup(1, [l_rf, l_gap, l_gnd, l2, l3, l4], GND_TAG)
    gmsh.model.setPhysicalName(1, GND_TAG, "GND")

    gmsh.model.addPhysicalGroup(2, [surface], 100)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

    print(f"Mesh written to {filename}")
