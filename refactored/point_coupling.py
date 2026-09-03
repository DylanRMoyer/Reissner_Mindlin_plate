import numpy as np
import basix
import dolfinx
from weak_form import collapse_w_subspace
from dataclasses import dataclass


# --- Implementing point mass ---

# Point mass parameters
@dataclass
class VAMMConfig:
    target_x: float
    target_y: float
    point_stiffness: float
    point_mass: float

def compute_point_eigenfrequency(point_stiffness: float, point_mass: float):
    return np.sqrt(point_stiffness/point_mass)/(2*np.pi)


# --- Find cell containing the chosen point

def locate_cell_and_reference_coords(domain, target_point_x, target_point_y):
    """Find the mesh cell containing (target_point_x, target_point_y) and
    pull the point back into that cell's reference coordinates."""
    target_point_vector = np.array([[target_point_x, target_point_y, 0.0]])

    bounding_box_tree = dolfinx.geometry.bb_tree(domain, 2)
    possible_boxes = dolfinx.geometry.compute_collisions_points(bounding_box_tree, target_point_vector)
    target_cells = dolfinx.geometry.compute_colliding_cells(domain, possible_boxes, target_point_vector)

    # Step A — take one definite cell
    cell = target_cells.links(0)[0]

    # Step B — pull the physical target point back into this cell's reference coordinates
    cmap = domain.geometry.cmaps[0]
    geom_dofs = np.array(domain.geometry.dofmaps[0][cell])
    cell_geom = np.array(domain.geometry.x[geom_dofs])
    x_ref = cmap.pull_back(target_point_vector[:, :domain.geometry.dim], cell_geom)

    return cell, x_ref

def evaluate_basis_weights_and_dofs(function_space, cell, x_ref):
    # Get w subspace
    function_space_w, w_to_parent = collapse_w_subspace(function_space)

    # Step C — evaluate the w-subspace's local basis functions at that reference point
    basix_element_w = function_space_w.element.basix_element
    tab = basix_element_w.tabulate(0, x_ref)
    phi = tab[0, 0, :, 0]

    # Step D — local dofs of this cell (in w-space numbering) -> parent (mixed-space) numbering
    local_to_global_w = function_space_w.dofmap.cell_dofs(cell)
    global_dofs_parent = w_to_parent[local_to_global_w]

    return phi, global_dofs_parent

def locate_target_cell_degrees_of_freedom(domain, function_space,
                                            target_point_x, target_point_y):
    """Combine cell location and basis evaluation into one call."""
    cell, x_ref = locate_cell_and_reference_coords(domain, target_point_x, target_point_y)
    phi, global_dofs_parent = evaluate_basis_weights_and_dofs(
        function_space, cell, x_ref
    )
    return phi, global_dofs_parent