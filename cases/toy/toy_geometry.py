import numpy as np
from mpi4py import MPI
from dolfinx import mesh

def create_toy_mesh(L=1e-3, nx=32):
    """
    Cube domain [-L, L]^3
    """
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([-L, -L, -L]), np.array([L, L, L])],
        [nx, nx, nx],
        cell_type=mesh.CellType.tetrahedron,
    )
    return domain
