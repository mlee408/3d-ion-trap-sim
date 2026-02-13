from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh


@dataclass
class MeshData:
    domain: dmesh.Mesh
    cell_tags: Optional[dmesh.MeshTags]
    facet_tags: Optional[dmesh.MeshTags]


def summarize_tags(tags: Optional[dmesh.MeshTags]) -> Dict[int, int]:
    """Return {tag_value: count} for a MeshTags object."""
    if tags is None:
        return {}
    uniq, cnt = np.unique(tags.values, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt)}


def load_msh(msh_path: str, *, comm=MPI.COMM_WORLD) -> MeshData:
    """
    Load a Gmsh .msh via meshio -> XDMF (works even when dolfinx has no gmshio).

    Requirements:
    - In Gmsh you must define Physical Groups for BOTH:
        * the volume (cells)
        * the boundary facets
    """
    import meshio
    from dolfinx import io
    from pathlib import Path

    msh = meshio.read(msh_path)
    cell_types = [c.type for c in msh.cells]

    if "triangle" in cell_types and "line" in cell_types:
        vol_type, facet_type = "triangle", "line"
    elif "tetra" in cell_types and "triangle" in cell_types:
        vol_type, facet_type = "tetra", "triangle"
    else:
        raise ValueError(f"Unsupported element types in {msh_path}: {cell_types}")

    vol_cells = msh.get_cells_type(vol_type)
    facet_cells = msh.get_cells_type(facet_type)

    def get_phys(ctype: str):
        for block, data in zip(msh.cells, msh.cell_data.get("gmsh:physical", [])):
            if block.type == ctype:
                return np.array(data, dtype=np.int32)
        return None

    vol_tags = get_phys(vol_type)
    facet_tags_arr = get_phys(facet_type)

    if vol_tags is None or facet_tags_arr is None:
        raise ValueError(
            "Missing gmsh physical tags. In Gmsh you must define Physical Groups "
            "for both the domain and the boundaries."
        )

    msh_path = Path(msh_path)
    out_dir = msh_path.parent / (msh_path.stem + "_xdmf")
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_xdmf = out_dir / "mesh.xdmf"
    facet_xdmf = out_dir / "facets.xdmf"

    tag_name = "name_to_read"
    meshio.write(
        mesh_xdmf,
        meshio.Mesh(
            points=msh.points,
            cells=[(vol_type, vol_cells)],
            cell_data={tag_name: [vol_tags]},
        ),
    )
    meshio.write(
        facet_xdmf,
        meshio.Mesh(
            points=msh.points,
            cells=[(facet_type, facet_cells)],
            cell_data={tag_name: [facet_tags_arr]},
        ),
    )

    with io.XDMFFile(comm, str(mesh_xdmf), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    with io.XDMFFile(comm, str(mesh_xdmf), "r") as xdmf:
        cell_tags = xdmf.read_meshtags(domain, name=tag_name)

    with io.XDMFFile(comm, str(facet_xdmf), "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name=tag_name)

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    return MeshData(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)


def load_xdmf(
    xdmf_path: str,
    *,
    mesh_name: str = "Grid",
    cell_tags_name: Optional[str] = None,
    facet_tags_name: Optional[str] = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MeshData:
    """Load mesh (and optionally cell/facet tags) from an XDMF file."""
    from dolfinx import io

    with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
        domain = xdmf.read_mesh(name=mesh_name)

        cell_tags = None
        facet_tags = None

        if cell_tags_name is not None:
            cell_tags = xdmf.read_meshtags(domain, name=cell_tags_name)

        if facet_tags_name is not None:
            facet_tags = xdmf.read_meshtags(domain, name=facet_tags_name)

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    return MeshData(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)


def load_mesh(path: str, *, comm: MPI.Comm = MPI.COMM_WORLD) -> MeshData:
    """Convenience loader: chooses based on extension (.msh or .xdmf)."""
    if path.lower().endswith(".msh"):
        return load_msh(path, comm=comm)
    if path.lower().endswith(".xdmf"):
        return load_xdmf(path, comm=comm)
    raise ValueError(f"Unsupported mesh file extension for: {path}")


def print_mesh_report(md: MeshData) -> None:
    """Print a quick report of mesh size and available tag IDs."""
    comm = md.domain.comm
    rank = comm.rank

    local_num_cells = md.domain.topology.index_map(md.domain.topology.dim).size_local
    local_num_verts = md.domain.topology.index_map(0).size_local

    if rank == 0:
        print("=== Mesh report ===")
        print(f"gdim: {md.domain.geometry.dim}, tdim: {md.domain.topology.dim}")
        print(f"local vertices: {local_num_verts}, local cells: {local_num_cells}")

        ct = summarize_tags(md.cell_tags)
        ft = summarize_tags(md.facet_tags)

        print(f"cell tags present: {sorted(ct.keys()) if ct else 'None'}")
        if ct:
            print(f"cell tag counts: {ct}")

        print(f"facet tags present: {sorted(ft.keys()) if ft else 'None'}")
        if ft:
            print(f"facet tag counts: {ft}")
        print("===================")
