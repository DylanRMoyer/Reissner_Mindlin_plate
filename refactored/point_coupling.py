import numpy as np
import dolfinx
from weak_form import collapse_w_subspace
from dataclasses import dataclass


# --- Implementing point mass ---

# Point mass parameters
@dataclass
class VAMM:
    x: float
    y: float
    stiffness: float
    mass: float
    gamma: float = 0.0
    index: int | None = None

    def __post_init__(self):
        if self.stiffness <= 0 or self.mass <= 0:
            raise ValueError(f"VAMM stiffness/ mass/ damping must be positive, got "
                              f"stiffness={self.stiffness}, mass={self.mass}")
        if self.gamma < 0:
            raise ValueError(f"VAMM damping (gamma) must be non-negative, got gamme={self.gamma}")

def check_separation(current_vamm_list, x, y, min_distance):
    for vamm in current_vamm_list: # Only skip over already placed VAMMs
        if abs(vamm.x - x) < min_distance and abs(vamm.y - y) < min_distance:
            raise ValueError(f"{x, y} is too close to {vamm.x, vamm.y}. Creation skipped.")
    return True

def register_vamm_indices(vamm_list):
    """Assign dense, order-matching indices to vamm_list in place. Always
    safe to re-run after any mutation (place_vamm/remove_vamm -> Future extensions)."""
    for i, vamm in enumerate(vamm_list):
        vamm.index = i
    return vamm_list

def assert_vamm_indices_registered(vamm_list):
    """Fail loudly if vamm_list's indices aren't densely order-matched.
    Call this inside augmentation functions as a defensive guard."""
    assert all(vamm.index == i for i, vamm in enumerate(vamm_list)), \
        "vamm_list is not densely index-ordered — call register_vamm_indices first"

def create_vamm_list_and_assign_indices(coords_stiffnesses_masses, min_distance):
    vamm_list = []
    for entry in coords_stiffnesses_masses:
        x, y, k, m, *rest = entry
        gamma = rest[0] if rest else 0.0
        try:
            check_separation(vamm_list, x, y, min_distance)
        except ValueError as e:
            print(f"Skipping VAMM at ({x}, {y}): {e}")
            continue
        vamm_list.append(VAMM(x=x, y=y, stiffness=k, mass=m, gamma=gamma))
    vamm_list = register_vamm_indices(vamm_list)
    return vamm_list

def compute_point_eigenfrequency(vamm: VAMM):
    return np.sqrt(vamm.stiffness/vamm.mass)/(2*np.pi)


# --- Find cell containing the chosen point

def locate_cell_and_reference_coords(domain, vamm: VAMM):
    """Find the mesh cell containing (target_point_x, target_point_y) and
    pull the point back into that cell's reference coordinates."""
    target_point_vector = np.array([[vamm.x, vamm.y, 0.0]])

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

    return phi, global_dofs_parent, local_to_global_w

def locate_target_cell_degrees_of_freedom(domain, function_space, vamm):
    """Combine cell location and basis evaluation into one call."""
    cell, x_ref = locate_cell_and_reference_coords(domain, vamm)
    phi, global_dofs_parent, local_to_global_w = evaluate_basis_weights_and_dofs(
        function_space, cell, x_ref
    )
    return phi, global_dofs_parent, local_to_global_w

def compute_phi_and_dofs_for_vamm_list(domain, function_space, vamm_list):
    """Run locate_target_cell_degrees_of_freedom once per VAMM, in vamm_list order."""
    phi_list = []
    global_dofs_parent_list = []
    local_to_global_w_list = []
    for vamm in vamm_list:
        phi, global_dofs_parent, local_to_global_w = locate_target_cell_degrees_of_freedom(domain, function_space, vamm)
        phi_list.append(phi)
        global_dofs_parent_list.append(global_dofs_parent)
        local_to_global_w_list.append(local_to_global_w)
    return phi_list, global_dofs_parent_list, local_to_global_w_list