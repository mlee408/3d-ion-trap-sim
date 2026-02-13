from mpi4py import MPI
from dolfinx import fem
import ufl

from toy_geometry import create_toy_mesh
from toy_bc import apply_toy_bcs
from laplace import solve_laplace  # your existing function

# 1️⃣ Mesh
domain = create_toy_mesh(L=1e-3, nx=32)

# 2️⃣ Function space
V = fem.FunctionSpace(domain, ("CG", 1))

# 3️⃣ Apply BCs
bcs = apply_toy_bcs(domain, V, V_rf=1.0)

# 4️⃣ Solve Laplace
phi = solve_laplace(domain, V, bcs)

print("Toy simulation complete.")
