from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import VAMMConfig, locate_target_cell_degrees_of_freedom
from solve import solve_evp
from postprocessing import plot_eigenmode

from boundary_conditions import make_border_marker
from frequency_sweep import frequency_sweep_plate
from postprocessing import plot_frequency_response

if __name__ == "__main__":

    do_frequency_sweep = True

    plate_config = PlateConfig(
    length = 1.1,
    width = 1,
    thickness = 0.02,
    nx = 60,
    ny = 50,
    rho = 7850,
    mu = 77e9,
    lambda_ = 115e9,
    # constant_force = 1 # Use if wanting to prescribe a custom constant force across the plate
    )

    domain, function_space, bcs, plate_problem_constants = build_plate_problem(plate_config, deg=2)
    # Add degree custom degree (default: deg = 2) or function type (default: el_type = "S") if needed

# --- Compute eigenfrequencies and -modes with an optional VAMM and plot result ---

    vamm_config = VAMMConfig(
        target_x = 0.137,
        target_y = 0.912,
        point_stiffness = 100,
        point_mass = 1
    )

    phi, global_dofs_parent, local_to_global_w\
        = locate_target_cell_degrees_of_freedom(domain, function_space, vamm_config)

    eigenfrequencies, eigenmodes = solve_evp(
        domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
        vamm_config = None, phi = phi, global_dofs_parent = global_dofs_parent, eigenmode_number = 10)

    for i, freq in enumerate(eigenfrequencies):
        print(f"mode {i}: {freq:.4f} Hz")

    plot_eigenmode(
        domain = domain, function_space = function_space, eigenmodes = eigenmodes,
        eigenmode_index = 3, length = plate_config.length,
        vamm_config = vamm_config, phi = phi, local_to_global_w = local_to_global_w)
              #     include_theta=True, include_mass=True,
              #     view_vector=(2, 2, -1), save_path=None):


# --- Perform frequency sweep and compare with computed eigenfrequencies if desired ---
    if do_frequency_sweep:
        border = make_border_marker(plate_config.length, plate_config.width)

        f_values, w_max_plate = (frequency_sweep_plate
                                 (domain = domain, function_space = function_space, problem = plate_problem_constants,
                                  plate_config = plate_config, border = border, f_start = 0.0, f_end = 1000.0))

        plot_frequency_response(
            f_values = f_values, w_max_plate = w_max_plate, eigenfrequencies = eigenfrequencies)