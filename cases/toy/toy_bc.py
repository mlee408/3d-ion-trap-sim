import numpy as np
from dolfinx import mesh

# Tag IDs
TAG_GND = 1
TAG_RF_PLUS = 2
TAG_RF_MINUS = 3

def make_toy_facet_tags(domain, L=1e-3, patch_halfwidth=3e-4):
    """
    Tag boundary facets:
      - RF patch on +x face centered at (y,z)=(0,0)
      - RF patch on -x face centered at (y,z)=(0,0)
      - All other outer boundary facets as ground
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # Helper to find all boundary facets (outer boundary)
    def is_boundary(x):
        return (np.isclose(x[0], -L) | np.isclose(x[0],  L) |
                np.isclose(x[1], -L) | np.isclose(x[1],  L) |
                np.isclose(x[2], -L) | np.isclose(x[2],  L))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, is_boundary)

    # RF patches on +/-x faces, restricted in y,z
    def rf_plus_patch(x):
        return (np.isclose(x[0], L) &
                (np.abs(x[1]) <= patch_halfwidth) &
                (np.abs(x[2]) <= patch_halfwidth))

    def rf_minus_patch(x):
        return (np.isclose(x[0], -L) &
                (np.abs(x[1]) <= patch_halfwidth) &
                (np.abs(x[2]) <= patch_halfwidth))

    rf_plus_facets = mesh.locate_entities_boundary(domain, fdim, rf_plus_patch)
    rf_minus_facets = mesh.locate_entities_boundary(domain, fdim, rf_minus_patch)

    # Start all boundary facets as GND
    facet_indices = np.array(boundary_facets, dtype=np.int32)
    facet_values = np.full(len(facet_indices), TAG_GND, dtype=np.int32)

    # Overwrite with RF tags where applicable
    # (We do it by matching facet ids)
    rf_plus_set = set(rf_plus_facets.tolist())
    rf_minus_set = set(rf_minus_facets.tolist())

    for i, f in enumerate(facet_indices):
        if f in rf_plus_set:
            facet_values[i] = TAG_RF_PLUS
        elif f in rf_minus_set:
            facet_values[i] = TAG_RF_MINUS

    # Create meshtags
    facet_tags = mesh.meshtags(domain, fdim, facet_indices, facet_values)
    return facet_tags


def toy_boundary_values(V_rf=1.0):
    """
    Map tag -> Dirichlet value.
    """
    return {
        TAG_GND: 0.0,
        TAG_RF_PLUS: V_rf,
        TAG_RF_MINUS: V_rf,
    }
