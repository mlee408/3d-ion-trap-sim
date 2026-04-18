import gmsh

gmsh.initialize()
gmsh.model.add("model")

gmsh.model.occ.importShapes("rfcell_h290_t100_n3.brep")
gmsh.model.occ.synchronize()

gmsh.write("output.step")   # or output.stp
gmsh.finalize()
