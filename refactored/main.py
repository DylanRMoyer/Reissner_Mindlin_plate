import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
from shaker_force import ShakerParameters
from solve import solve_evp
from postprocessing import plot_eigenmode, plot_frequency_response, plot_mobility_bode, PlateLayout
from vamm_grid import GridConfig, state_to_vamm_list
from probe import make_probe, build_probe_coordinates
from sweep_cache import CACHE_DIR, make_cache_key, get_or_compute_sweep

from frequency_sweep import frequency_sweep_plate, build_frequency_grid, warn_if_grid_too_coarse

if __name__ == "__main__":

    do_frequency_sweep = True
    free_plate = True # switch: free (shaker-driven) vs. clamped-edge plate
    excitation = "force"  # "force" (allows both free_plate=True or False) or "motion" (needs free_plate=False)
    vamm_mode = "manual"  # "grid" or "manual"

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

    if vamm_mode == "grid":
        grid_config = GridConfig(n_x=2, n_y=3, stiffness=85878.35366, mass=1.0, gamma=0.02)
        # state = [0] * 6
        state = [1, 0, 1, 0, 1, 1]  # length n_x * n_y
        vamm_list = state_to_vamm_list(state, grid_config, plate_config.length, plate_config.width)
    elif vamm_mode == "manual":
        vamm_list = create_vamm_list_and_assign_indices(
            [
                (1.1,0.0,858783.5366,1.0,0.02)
                #,
                #(0.55, 0.5, 10042816.24, 10.0, 0.05)
                #,
                #(0.1, 0.1, 500, 5, 0.01)
            ])
        vamm_list2 = create_vamm_list_and_assign_indices([
            (0.275, 0.16666666666666666, 85878.35366, 1.0, 0.02),
            (0.275, 0.8333333333333333, 85878.35366, 1.0, 0.02),
            (0.8250000000000001, 0.5, 85878.35366, 1.0, 0.02),
            (0.8250000000000001, 0.8333333333333333, 85878.35366, 1.0, 0.02)
        ])
    else:
        raise ValueError(f"unknown vamm_mode: {vamm_mode!r}")


    phi_list, global_dofs_parent_list, local_to_global_w_list = compute_phi_and_dofs_for_vamm_list(
        domain=domain, function_space=function_space, vamm_list=vamm_list
    )

    eigenfrequencies_bare, eigenmodes_bare = solve_evp(
        domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
        vamm_list = None, phi_list = phi_list, global_dofs_parent_list = global_dofs_parent_list,
        eigenmode_number = 40)

    eigenfrequencies_vamm, eigenmodes_vamm = solve_evp(
        domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
        vamm_list = vamm_list, phi_list = phi_list, global_dofs_parent_list = global_dofs_parent_list,
        eigenmode_number = 40)

    for i, freq in enumerate(eigenfrequencies_bare):
        print(f"mode {i}: {freq:.4f} Hz")

    for i, freq in enumerate(eigenfrequencies_vamm):
        print(f"mode {i}: {freq:.4f} Hz")

    plot_eigenmode(
        domain = domain, function_space = function_space, eigenmodes = eigenmodes_bare,
        eigenmode_index = 34, length = plate_config.length,
        vamm_list = vamm_list, phi_list = phi_list, local_to_global_w_list = local_to_global_w_list)
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

        shaker_probe = make_probe(domain, function_space, shaker_config.x, shaker_config.y)
        probe_coords = build_probe_coordinates(plate_config=plate_config, shaker_config=shaker_config,
                                               amount=5, vamm_list=vamm_list)
        probes = [make_probe(domain, function_space, x, y) for x, y in probe_coords]
        probes.insert(0, shaker_probe)
        f_start = 100
        f_end = 300
        gamma_plate = 0.04
        grid = build_frequency_grid(f_start=f_start, f_end=f_end, gamma=0.02)  # smallest gamma in either run
        layout = PlateLayout(
            plate_config=plate_config, shaker_config=shaker_config, probes=probes, vamm_list=vamm_list
        )

    settings = dict(
        plate_config=plate_config, shaker_config=shaker_config,
        probe_coords=[(p.x, p.y) for p in probes],
        grid=np.asarray(grid), gamma_plate=gamma_plate,
        free_plate=free_plate, excitation=excitation,
        code_version="v1")  # bump this string if you change the sweep code itself

    key_vamm, blob_vamm = make_cache_key(vamm_list=vamm_list, **settings)
    key_bare, blob_bare = make_cache_key(vamm_list=None, **settings)

    sweep_vamm = get_or_compute_sweep(
        CACHE_DIR / f"sweep_vamm_{key_vamm}.npz",
        compute=lambda: frequency_sweep_plate(
            domain=domain, function_space=function_space, problem=plate_problem_constants,
            plate_config=plate_config, f_start=f_start, f_end=f_end,
            vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
            frequency_grid=grid, gamma=gamma_plate,
            free_plate=free_plate, excitation=excitation, shaker_config=shaker_config,
            probes=probes),
        settings_blob=blob_vamm)

    sweep_bare = get_or_compute_sweep(
        CACHE_DIR / f"sweep_bare_{key_bare}.npz",
        compute=lambda: frequency_sweep_plate(
            domain=domain, function_space=function_space, problem=plate_problem_constants,
            plate_config=plate_config, f_start=f_start, f_end=f_end,
            vamm_list=None, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
            frequency_grid=grid, gamma=gamma_plate,
            free_plate=free_plate, excitation=excitation, shaker_config=shaker_config,
            probes=probes),
        settings_blob=blob_bare)

    # --- Driving point (probe 0 = shaker location) ---
    plot_mobility_bode(
        f_values=sweep_bare.f_values,
        mobilities=[sweep_bare.probe_mobility[0], sweep_vamm.probe_mobility[0]],
        labels=["bare plate", "with VAMMs"],
        eigenfrequencies=[eigenfrequencies_bare, eigenfrequencies_vamm],
        title="Driving-point mobility",
        layout=layout, highlight_probe=None
        #, save_path=r"/home/local/CSI/dm27demu/Desktop/Mobilities/Y_dp.pdf"
    )

    # --- Transfer mobilities (probes 1..5) ---
    for i in range(1, len(probes)):
        d = np.hypot(probes[i].x - shaker_config.x, probes[i].y - shaker_config.y)
        plot_mobility_bode(
            f_values=sweep_bare.f_values,
            mobilities=[sweep_bare.probe_mobility[i], sweep_vamm.probe_mobility[i]],
            labels=["bare plate", "with VAMMs"],
            eigenfrequencies=[eigenfrequencies_bare, eigenfrequencies_vamm],
            title=f"Transfer mobility, probe {i} ({d:.2f} m from shaker)",
            layout=layout, highlight_probe=i
            #, save_path=rf"/home/local/CSI/dm27demu/Desktop/Mobilities/Y_{i}.pdf"
        )

    plot_frequency_response(
        f_values=sweep_vamm.f_values, rms_velocity=sweep_vamm.rms_velocity,
        eigenfrequencies=eigenfrequencies_bare)