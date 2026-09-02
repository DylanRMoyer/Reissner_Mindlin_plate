import numpy as np
import basix
import dolfinx


# --- Implementing point mass ---

# Point mass parameters

target_point_mass_x = 0.137
target_point_mass_y = 0.912


point_stiffness = 100 # Newton per meter, keeping it SI
point_mass = 1e4 # kilogram, also keeping it SI

def compute_point_eigenfrequency(point_stiffness: float, point_mass: float):
    return np.sqrt(point_stiffness/point_mass)/(2*np.pi)


# --- Find cell containing the chosen point
def locate_target_cell_degrees_of_freedom(domain,
                                          function_space_w,
                                          w_to_parent,
                                          target_point_x: float,
                                          target_point_y: float):

    #Step 0 - find candidates for a cell containing the target point
    target_point_vector = np.array([[target_point_x, target_point_y, 0.0]])

    bounding_box_tree = dolfinx.geometry.bb_tree(domain,2)
    possible_boxes = dolfinx.geometry.compute_collisions_points(bounding_box_tree, target_point_vector)
    target_cells = dolfinx.geometry.compute_colliding_cells(domain, possible_boxes, target_point_vector)

    # Step A — take one definite cell
    cell = target_cells.links(0)[0]

    # Step B — pull the physical target point back into this cell's reference coordinates
    cmap = domain.geometry.cmaps[0]
    geom_dofs = np.array(domain.geometry.dofmaps[0][cell])
    cell_geom = np.array(domain.geometry.x[geom_dofs])
    x_ref = cmap.pull_back(target_point_vector[:, :domain.geometry.dim], cell_geom)
    print("x_ref =", x_ref)

    # Step C — evaluate the w-subspace's local basis functions at that reference point
    basix_element_w = function_space_w.element.basix_element
    tab = basix_element_w.tabulate(0, x_ref)      # 0th derivative = values only
    phi = tab[0, 0, :, 0]                         # one weight per local dof of this cell

    # Step D — local dofs of this cell (in w-space numbering) -> parent (mixed-space) numbering
    local_to_global_w = function_space_w.dofmap.cell_dofs(cell)
    global_dofs_parent = w_to_parent[local_to_global_w]

    print("basis weights at target point:", phi)
    print("parent dofs receiving the point load:", global_dofs_parent)

    return phi, global_dofs_parent
