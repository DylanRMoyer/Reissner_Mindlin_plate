from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import VAMM, locate_target_cell_degrees_of_freedom
from shaker_force import ShakerParameters
from solve import solve_evp
from postprocessing import plot_eigenmode

from frequency_sweep import frequency_sweep_plate
from postprocessing import plot_frequency_response

from petsc4py import PETSc

if __name__ == "__main__":

    do_frequency_sweep = True
    free_plate = True  # switch: free (shaker-driven) vs. clamped-edge plate
    excitation = "force"  # "force" (needs free_plate=True or False) or "motion" (needs free_plate=False)

    plate_config = PlateConfig(
    length = 1.1,
    width = 1,
    thickness = 0.02,
    nx = 60,
    ny = 50,
    rho = 7850,
    mu = 77e9,
    lambda_ = 115e9,
    # constant_force = 0.0 # Use if wanting to prescribe a custom constant force across the plate
    )

    domain, function_space, bcs_clamped, plate_problem_constants = build_plate_problem(plate_config, deg=2)
    # Add degree custom degree (default: deg = 2) or function type (default: el_type = "S") if needed

    # Use the real clamped BCs only if we're not testing the free plate.
    bcs = [] if free_plate else bcs_clamped

# --- Compute eigenfrequencies and -modes with an optional VAMM and plot result ---

    vamm = VAMM(
        x = 0.137,
        y = 0.912,
        stiffness = 100,
        mass = 1
    )

    phi, global_dofs_parent, local_to_global_w\
        = locate_target_cell_degrees_of_freedom(domain, function_space, vamm)

    eigenfrequencies, eigenmodes = solve_evp(
        domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
        vamm = vamm, phi = phi, global_dofs_parent = global_dofs_parent, eigenmode_number = 11)

    for i, freq in enumerate(eigenfrequencies):
        print(f"mode {i}: {freq:.4f} Hz")

    plot_eigenmode(
        domain = domain, function_space = function_space, eigenmodes = eigenmodes,
        eigenmode_index = 8, length = plate_config.length,
        vamm = vamm, phi = phi, local_to_global_w = local_to_global_w)
              #     include_theta=True, include_mass=True,
              #     view_vector=(2, 2, -1), save_path=None):


# --- Perform frequency sweep and compare with computed eigenfrequencies if desired ---
    if do_frequency_sweep:

        shaker_config = ShakerParameters(
            x=plate_config.length/3,
            y=plate_config.width/3,
            force_amplitude=1.0,
            phase=0.0
        ) if excitation == "force" else None

        f_values, w_max_plate, rms_velocity = frequency_sweep_plate(
            domain=domain, function_space=function_space, problem=plate_problem_constants,
            plate_config=plate_config,
            f_start=25, f_end=300, Omega_size=20, gamma=0.04,
            free_plate=free_plate, excitation=excitation, shaker_config=shaker_config)
        # Peak width scales Delta_f = gamma * f_res, so choose Delta_f > (f_end - f_start)/Omega_size
        # for resonance frequencies of interest!

        plot_frequency_response(
            f_values = f_values, rms_velocity=rms_velocity, eigenfrequencies = eigenfrequencies)