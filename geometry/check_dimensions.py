import cadquery as cq

shape = cq.importers.importStep("ground.step")

bbox = shape.val().BoundingBox()

print("X length:", bbox.xlen)
print("Y length:", bbox.ylen)
print("Z length:", bbox.zlen)

print("xmin xmax:", bbox.xmin, bbox.xmax)
print("ymin ymax:", bbox.ymin, bbox.ymax)
print("zmin zmax:", bbox.zmin, bbox.zmax)
