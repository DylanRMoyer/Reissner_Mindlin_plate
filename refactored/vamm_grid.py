from copy import deepcopy
from dataclasses import dataclass
from point_coupling import create_vamm_list_and_assign_indices

@dataclass
class GridConfig:
    n_x: int
    n_y: int
    stiffness: float
    mass: float
    gamma: float = 0.0
    offset_x: float = 0.5
    offset_y: float = 0.5

    def __post_init__(self):
        if self.n_x <= 0 or self.n_y <= 0:
            raise ValueError(f"n_x and n_y must be at least 1, got n_x = {self.n_x} and n_y = {self.n_y}")
        if self.stiffness <= 0 or self.mass <= 0:
            raise ValueError(f"VAMM stiffness/ mass must be positive, got "
                             f"stiffness={self.stiffness}, mass={self.mass}")
        if self.gamma < 0:
            raise ValueError(f"VAMM damping (gamma) must be non-negative, got gamma={self.gamma}")
        if self.offset_x < 0 or self.offset_y < 0 or self.offset_x > 1 or self.offset_y > 1:
            raise ValueError("Relative offset inside unit cell must be between 0 and 1. \n"
                             f"Currently: offset_x ={self.offset_x}, offset_y = {self.offset_y}")


def build_grid_positions(
        length: float,
        width: float,
        grid_config: GridConfig):


    cell_length = length / grid_config.n_x
    cell_width = width / grid_config.n_y

    vamm_grid_coords = []

    for i in range(grid_config.n_x):
        for j in range(grid_config.n_y):
            x_coord = (i + grid_config.offset_x) * cell_length
            y_coord = (j + grid_config.offset_y) * cell_width
            vamm_grid_coords.append([x_coord, y_coord])

    return vamm_grid_coords

def state_to_vamm_list(state, grid_config: GridConfig, length, width):

    if len(state) != grid_config.n_x * grid_config.n_y:
       raise ValueError(
           f"Length of bitstring {len(state)} does not match amount of cells {grid_config.n_x * grid_config.n_y}.")
    invalid = [(i, b) for i, b in enumerate(state) if b not in (0, 1)]
    if invalid:
        raise ValueError(
            f"Entries of bitstring must be 0 or 1, found invalid (position, value) pairs: {invalid}")

    coords_stiffnesses_masses = []

    vamm_grid_coords = build_grid_positions(length, width, grid_config)

    for idx, bit in enumerate(state):
        if bit == 1:
            x, y = vamm_grid_coords[idx][0], vamm_grid_coords[idx][1]
            coords_stiffnesses_masses.append((x, y, grid_config.stiffness, grid_config.mass, grid_config.gamma))

    vamm_list = create_vamm_list_and_assign_indices(coords_stiffnesses_masses)

    return vamm_list

def vamm_list_to_state(vamm_list, grid_config: GridConfig, length, width):
    n_x, n_y = grid_config.n_x, grid_config.n_y
    cell_length, cell_width = length / n_x, width / n_y
    tol = 1e-9 * max(length, width)
    state = [0] * (n_x * n_y)

    for idx, vamm in enumerate(vamm_list):
        if (vamm.stiffness != grid_config.stiffness or vamm.mass != grid_config.mass
                or vamm.gamma != grid_config.gamma):
            raise ValueError(f"vamm_list[{idx}] parameters differ from grid_config; not representable as a state.")
        i = round(vamm.x / cell_length - grid_config.offset_x)
        j = round(vamm.y / cell_width - grid_config.offset_y)
        if not (0 <= i < n_x and 0 <= j < n_y):
            raise ValueError(f"vamm_list[{idx}] at ({vamm.x}, {vamm.y}) is outside the grid.")
        x_site = (i + grid_config.offset_x) * cell_length
        y_site = (j + grid_config.offset_y) * cell_width
        if abs(vamm.x - x_site) > tol or abs(vamm.y - y_site) > tol:
            raise ValueError(f"vamm_list[{idx}] at ({vamm.x}, {vamm.y}) is not on a grid site.")
        k = i * n_y + j
        if state[k] == 1:
            raise ValueError(f"vamm_list[{idx}] duplicates a VAMM already at grid site {k}.")
        state[k] = 1
    return state

def alter_state(state, index: int):
    """
    Alter a state by removing/ adding a VAMM to grid site index with/ without a VAMM present at site index.
    Takes a state and an index at which the altering takes place.
    Returns the altered state as a copy, leaves original state as is.

    Will raise an error if index is out of range or if state[index] is an invalid entry, i. e., not 0 or 1.
    """
    if not 0 <= index < len(state) - 1:
        raise IndexError("VAMM index out of range. Please choose a VAMM from the given state.")
    new_state = list(state)
    if new_state[index] == 1:
        new_state[index] = 0
    elif new_state[index] == 0:
        new_state[index] = 1
    else:
        raise ValueError(f"Invalid entry: Only 0 and 1 are valid entries, got {new_state[index]}.")
    return new_state

if __name__ == "__main__":
    grid_config = GridConfig(n_x=2, n_y=3, stiffness=85878.35366, mass=1.0, gamma=0.02)
    empty_state = [0] * grid_config.n_x * grid_config.n_y
    state = [1,0,1,0,1,1]

    vamm_list = state_to_vamm_list(state, grid_config, 1.1, 1)
    #nudge_vamm_list = deepcopy(vamm_list)

    #for idx, vamm in enumerate(nudge_vamm_list):
    #    if idx == 0:
    #        vamm.x += 1e-6

    #print(vamm_list)

    #error_state = vamm_list_to_state(nudge_vamm_list, grid_config, 1, 1)
    return_state = vamm_list_to_state(vamm_list, grid_config, 1.1, 1)
    print(return_state)

    first_altered_state = alter_state(return_state, 0)
    print(first_altered_state)
    second_altered_state = alter_state(first_altered_state, 1)
    print(second_altered_state)
    error_altered_state = alter_state(second_altered_state, 1)
