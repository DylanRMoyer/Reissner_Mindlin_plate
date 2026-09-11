import numpy as np
from dolfinx import fem, mesh
from weak_form import collapse_subspace



# Boundary of the plate

def make_border_marker(length, width):
    """Return a marker function flagging points on the rectangular
    plate's outer boundary, closed over the given length/width."""
    def border(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], length)),
            np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], width)),
        )
    return border

def dirichlet_boundary_conditions(
        domain, function_space, length: float, width: float, phi_0: float = 0
    ):

    topological_dimension_mesh = domain.topology.dim
    facet_dim = topological_dimension_mesh - 1
    border = make_border_marker(length, width)

    clamped_facets = mesh.locate_entities_boundary(domain, facet_dim, border)

    function_space_w, _ = collapse_subspace(function_space, 0)
    function_space_theta, _ = collapse_subspace(function_space, 1)

    u0_w = fem.Function(function_space_w)
    u0_w.x.array[:] = phi_0 # zero by default, clamped displacement; nonzero for shaker BC
    u0_t = fem.Function(function_space_theta)   # rotation stays zero, no boundary rotation for rigid translation

    dofs_w = fem.locate_dofs_topological((function_space.sub(0), function_space_w), facet_dim, clamped_facets)
    dofs_t = fem.locate_dofs_topological((function_space.sub(1), function_space_theta), facet_dim, clamped_facets)

    bc_w = fem.dirichletbc(u0_w, dofs_w, function_space.sub(0))
    bc_t = fem.dirichletbc(u0_t, dofs_t, function_space.sub(1))

    bcs = [bc_w, bc_t]

    return bcs

