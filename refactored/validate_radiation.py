import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
from shaker_force import ShakerParameters
from solve import solve_evp
from postprocessing import (plot_eigenmode, plot_frequency_response, plot_mobility_bode, PlateLayout,
                            plot_insertion_loss_bode, draw_plate_layout)
from vamm_grid import GridConfig, state_to_vamm_list, cell_center_positions
from probe import (make_probe, build_probe_coordinates,
                   mobility_at_probes, mobility_on_points, build_interpolation_matrix)
from sweep_cache import CACHE_DIR, make_cache_key, get_or_compute_sweep
from insertion_loss import assert_comparable_sweeps, compute_insertion_loss

from frequency_sweep import frequency_sweep_plate, build_frequency_grid, warn_if_grid_too_coarse

from radiation import (build_radiation_points, radiation_resistance_matrix,
                       radiated_power, radiation_efficiency, power_level_db)

from insertion_loss import (input_power, global_insertion_loss_radiated_db,
                            global_insertion_loss_radiated_equal_power_db)
from scipy.signal import find_peaks

def assert_allclose_detailed(array, target=0.0, rtol=1e-05, atol=1e-08):
    close_mask = np.isclose(array, target, rtol=rtol, atol=atol)
    if not np.all(close_mask):
        bad_idx = np.flatnonzero(~close_mask)[0]
        raise AssertionError(
            f"Element {bad_idx} should be value {target}, is {array[bad_idx]}"
        )

def assert_allclose_two_arrays(array, target_array, rtol=1e-05, atol=1e-08):
    array = np.asarray(array)
    target_array = np.asarray(target_array)

    # 1. Compare the two arrays element-wise
    close_mask = np.isclose(array, target_array, rtol=rtol, atol=atol)

    # 2. Check if any elements failed the tolerance check
    if not np.all(close_mask):
        bad_flat_idx = np.flatnonzero(~close_mask)[0]

        # Unravel the flat index into multidimensional coordinates (if applicable)
        bad_idx = np.unravel_index(bad_flat_idx, array.shape)
        if len(bad_idx) == 1:
            bad_idx = bad_idx[0]  # Keep it clean for 1D arrays

        raise AssertionError(
            f"Element {bad_idx} should be value {target_array.flat[bad_flat_idx]}, "
            f"is {array.flat[bad_flat_idx]}"
        )



# assert_allclose_detailed(a, b)
# Raises: AssertionError: Element (1, 0) should be value 3.05, is 3.0
if __name__ == "__main__":

    do_frequency_sweep = True
    free_plate = True # switch: free (shaker-driven) vs. clamped-edge plate
    excitation = "force"  # "force" (allows both free_plate=True or False) or "motion" (needs free_plate=False)
    vamm_mode = "manual"  # "grid" or "manual"

    eigenmode_number = 20 # the total number of eigenmodes to be computed by eigensolver
    eigenmode_to_plot = 13 # the desired eigenmode INDEX to be plotted. Can be no larger than (eigenmode_number - 1).
    target_frequency_hz = 0.0

    deg = 2
    el_type = "S"

    ncv_factor = 4 # integer, default: 2, increase if suspected that not all modes are found by the solver

    plate_config = PlateConfig(
        length = 1.1,
        width = 1,
        thickness = 0.02,
        nx = 50,
        ny = 50,
        rho = 7850,
        mu = 77e9,
        lambda_ = 115e9,
        # constant_force = 0.0 # Use if wanting to prescribe a custom constant force across the plate
    )

    domain, function_space, bcs_clamped, plate_problem_constants = build_plate_problem(
        plate_config, deg=deg, el_type=el_type)
    # Add degree custom degree (default: deg = 2) or function type (default: el_type = "S") if needed

    # Use the real clamped BCs only if we're not testing the free plate.
    bcs = [] if free_plate else bcs_clamped

    # --- Compute eigenfrequencies and -modes with an optional VAMM and plot result ---

    if vamm_mode == "grid":
        grid_config = GridConfig(n_x=2, n_y=3, stiffness=85878.35366, mass=1.0, gamma=0.02)
        state = [0] * 6
        # state = [1, 0, 1, 0, 1, 1]  # length n_x * n_y
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

    if do_frequency_sweep:

        shaker_config = ShakerParameters(
            x=plate_config.length/3,
            y=plate_config.width/3,
            force_amplitude=1.0,
            phase=0.0
        ) if excitation == "force" else None

        shaker_probe = make_probe(domain, function_space, shaker_config.x, shaker_config.y)
        probe_coords = build_probe_coordinates(plate_config=plate_config, shaker_config=shaker_config,
                                               amount=3, vamm_list=vamm_list)

        probes = [make_probe(domain, function_space, x, y) for x, y in probe_coords]
        probes.insert(0, shaker_probe)

        probes_mesh, dS, r = build_radiation_points(domain=domain, function_space=function_space, plate_config=plate_config)

        f_start = 100
        f_end = 300
        gamma_plate = 0.04
        grid = build_frequency_grid(f_start=f_start, f_end=f_end, gamma=0.02)  # smallest gamma in either run
        layout = PlateLayout(
            plate_config=plate_config, shaker_config=shaker_config, probes=probes, vamm_list=vamm_list
        )

        settings = dict(
            plate_config=plate_config, shaker_config=shaker_config,
            grid=np.asarray(grid), gamma_plate=gamma_plate,
            free_plate=free_plate, excitation=excitation,
            deg=deg, el_type=el_type,
            code_version="v2")  # bump this string if you change the sweep code itself

        key_vamm, blob_vamm = make_cache_key(vamm_list=vamm_list, **settings)
        key_bare, blob_bare = make_cache_key(vamm_list=None, **settings)

        sweep_vamm = get_or_compute_sweep(
            CACHE_DIR / f"sweep_vamm_{key_vamm}.npz",
            compute=lambda: frequency_sweep_plate(
                domain=domain, function_space=function_space, problem=plate_problem_constants,
                plate_config=plate_config, f_start=f_start, f_end=f_end,
                vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
                frequency_grid=grid, gamma=gamma_plate,
                free_plate=free_plate, excitation=excitation, shaker_config=shaker_config),
            settings_blob=blob_vamm)

        sweep_bare = get_or_compute_sweep(
            CACHE_DIR / f"sweep_bare_{key_bare}.npz",
            compute=lambda: frequency_sweep_plate(
                domain=domain, function_space=function_space, problem=plate_problem_constants,
                plate_config=plate_config, f_start=f_start, f_end=f_end,
                vamm_list=None, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
                frequency_grid=grid, gamma=gamma_plate,
                free_plate=free_plate, excitation=excitation, shaker_config=shaker_config),
            settings_blob=blob_bare)

        bare_probe_mobility = mobility_at_probes(
            sweep=sweep_bare, probes=probes_mesh, force_amplitude=shaker_config.force_amplitude)

        vamm_probe_mobility = mobility_at_probes(
            sweep=sweep_vamm, probes=probes_mesh, force_amplitude=shaker_config.force_amplitude)

        bare_mobility_check = np.sqrt(np.mean(np.abs(bare_probe_mobility)**2, axis=0))
        vamm_mobility_check = np.sqrt(np.mean(np.abs(vamm_probe_mobility)**2, axis=0))

        assert np.allclose(bare_mobility_check, sweep_bare.rms_velocity, rtol=0.01)
        assert np.allclose(vamm_mobility_check, sweep_vamm.rms_velocity, rtol=0.01)

        few = probes_mesh[:5]  # any handful of probes
        Phi_few = build_interpolation_matrix(few, sweep_bare.w_field.shape[1])

        Y_matrix = mobility_on_points(sweep_bare, Phi_few, shaker_config.force_amplitude)
        Y_loop = mobility_at_probes(sweep_bare, few, shaker_config.force_amplitude)

        assert np.allclose(Y_matrix, Y_loop, rtol=1e-12, atol=0)

        coords = cell_center_positions(plate_config.length, plate_config.width, n_x=20, n_y=18)  # 360 points
        xy = np.asarray(coords)
        r_small = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        dS_small = (plate_config.length / 20) * (plate_config.width / 18)

        for f in (50.0, 150.0, 300.0, 1000.0):
            R = radiation_resistance_matrix(r_small, dS_small, 2 * np.pi * f)
            assert np.allclose(R, R.T, rtol=1e-12, atol=0)
            eig = np.linalg.eigvalsh(R)
            assert eig.min() >= -1e-10 * eig.max(), f"R not PSD at {f} Hz: min eigenvalue {eig.min():.3e}"

        import numpy as np
        from radiation import radiation_resistance_matrix, radiated_power, radiation_efficiency, rho_air, c_air

        L, W = 1.1, 1.0
        S = L * W


        def points(n_x, n_y):
            coords = np.asarray(cell_center_positions(L, W, n_x, n_y))
            r = np.hypot(coords[:, None, 0] - coords[None, :, 0],
                         coords[:, None, 1] - coords[None, :, 1])
            return coords, r, S / (n_x * n_y)


        coords_syn, r_syn, dS_syn = points(50, 50)

        # Low frequency

        f = np.array([1.0])  # kL ~ 0.02
        Y = np.ones((len(coords_syn), 1), dtype=complex)  # v = 1 m/s everywhere
        P = radiated_power(Y_grid=Y, r=r_syn, dS=dS_syn, f_values=f, force_amplitude=1.0)
        Q = 1.0 * S
        P_expected = rho_air * (2 * np.pi * f) ** 2 * Q ** 2 / (4 * np.pi * c_air)
        assert np.allclose(P, P_expected, rtol=1e-3)

        # High frequency, uniform velocity

        f = np.array([2000.0, 4000.0])  # kL ~ 40-80
        Y = np.ones((len(coords_syn), len(f)), dtype=complex)
        P = radiated_power(Y_grid=Y, r=r_syn, dS=dS_syn, f_values=f, force_amplitude=1.0)
        sigma = radiation_efficiency(P, Y, S, force_amplitude=1.0)
        print(sigma)  # should be close to 1, approaching it as f increases

        # Synthetic pattern

        def power_for_grid(n, m=3, nn=2, f_values=(100.0, 300.0, 1000.0)):
            coords, r, dS = points(n, n)
            v = np.sin(m * np.pi * coords[:, 0] / L) * np.sin(nn * np.pi * coords[:, 1] / W)
            Y = np.repeat(v[:, None], len(f_values), axis=1).astype(complex)
            return radiated_power(Y_grid=Y, r=r, dS=dS, f_values=np.array(f_values), force_amplitude=1.0)


        P12, P25, P50 = (power_for_grid(n) for n in (12, 25, 50))
        print("rel. diff 12->25:", np.abs(P12 - P25) / P50)
        print("rel. diff 25->50:", np.abs(P25 - P50) / P50)



        P_bare = radiated_power(
            Y_grid=bare_probe_mobility, r=r, dS=dS,
            f_values=sweep_bare.f_values, force_amplitude=shaker_config.force_amplitude)

        P_vamm = radiated_power(
            Y_grid=vamm_probe_mobility, r=r, dS=dS,
            f_values=sweep_bare.f_values, force_amplitude=shaker_config.force_amplitude)


        for idx, power in enumerate(P_bare):
            assert power >= 0, (f"Power should not be negative, found P[{idx}] = {power}")
        for idx, power in enumerate(P_vamm):
            assert power >= 0, (f"Power should not be negative, found P[{idx}] = {power}")

        surface_area = plate_config.length * plate_config.width
        shaker_force = shaker_config.force_amplitude

        sigma_bare = radiation_efficiency(P_bare, bare_probe_mobility, surface_area, force_amplitude=shaker_force)
        sigma_vamm = radiation_efficiency(P_vamm, vamm_probe_mobility, surface_area, force_amplitude=shaker_force)

        # driving-point mobility from the shaker probe (probes[0])
        Y_dp_bare = mobility_at_probes(sweep_bare, [shaker_probe], shaker_force)[0]
        Y_dp_vamm = mobility_at_probes(sweep_vamm, [shaker_probe], shaker_force)[0]
        ratio_bare = P_bare / input_power(Y_dp_bare, shaker_force)
        ratio_vamm = P_vamm / input_power(Y_dp_vamm, shaker_force)


        no_loss = global_insertion_loss_radiated_db(P_bare, P_bare)
        assert_allclose_detailed(array=no_loss, target=0.0, rtol=1e-3)

        P_vamm_test = 0.5 * P_bare
        half_insertion_loss_db = global_insertion_loss_radiated_db(P_bare, P_vamm_test)
        assert_allclose_detailed(array=half_insertion_loss_db, target=10*np.log10(2), rtol=1e-3)

        no_loss_power = global_insertion_loss_radiated_equal_power_db(P_bare, P_bare, Y_dp_bare, Y_dp_bare)
        assert_allclose_detailed(array=no_loss_power, target=0.0, rtol=1e-3)

        Y_dp_vamm_test = 2 * Y_dp_bare
        half_insertion_loss_db_power = global_insertion_loss_radiated_equal_power_db(
            P_bare, P_bare, Y_dp_bare, Y_dp_vamm_test
        )
        assert_allclose_detailed(array=half_insertion_loss_db_power, target=10*np.log10(2), rtol=1e-3)

        rms_bare = bare_mobility_check * shaker_force
        rms_vamm = vamm_mobility_check * shaker_force
        insertion_loss_db = global_insertion_loss_radiated_db(P_bare, P_vamm)
        insertion_loss_db_power = global_insertion_loss_radiated_equal_power_db(
            P_bare, P_vamm, Y_dp_bare, Y_dp_vamm, force_amplitude=shaker_force)

        control_insertion_loss_db =(
                20*np.log10(rms_bare/rms_vamm) + 10*np.log10(sigma_bare/sigma_vamm))
        assert_allclose_two_arrays(
            array=insertion_loss_db, target_array=control_insertion_loss_db, rtol=1e-9, atol=1e-9)

        sigma_term = 10*np.log10(sigma_bare/sigma_vamm)
        assert_allclose_two_arrays(insertion_loss_db, 20*np.log10(rms_bare/rms_vamm) + sigma_term, rtol=1e-9, atol=1e-9)

        max_insertion_loss_db = np.max(insertion_loss_db)
        max_insertion_loss_db_power = np.max(insertion_loss_db_power)
        print(f"max il_rad = {max_insertion_loss_db}, max il_rad_pow = {max_insertion_loss_db_power}.")

        for name, sig, ratio in (("bare", sigma_bare, ratio_bare), ("vamm", sigma_vamm, ratio_vamm)):
            print(f"{name}: sigma in [{sig.min():.2e}, {sig.max():.2e}], "
                  f"P_rad/P_in in [{ratio.min():.2e}, {ratio.max():.2e}]")
            assert ratio.max() < 1.0, f"{name}: radiated power exceeds input power"

        # peaks of P should coincide with peaks of the structural response
        f = sweep_bare.f_values
        p_peaks, _ = find_peaks(P_bare)
        v_peaks, _ = find_peaks(sweep_bare.rms_velocity)
        print("P peaks (Hz):  ", np.round(f[p_peaks], 1))
        print("rms-v peaks (Hz):", np.round(f[v_peaks], 1))

        i = int(np.argmin(np.abs(f - 280.1)))
        j = int(np.argmin(np.abs(f - 296.2)))
        for label, idx in (("280 Hz", i), ("296 Hz", j)):
            v = bare_probe_mobility[:, idx]
            monopole_content = np.abs(np.mean(v)) / np.sqrt(np.mean(np.abs(v) ** 2))
            print(f"{label}: sigma = {sigma_bare[idx]:.3e}, |mean v|/rms v = {monopole_content:.3f}")

        print("P around 280 Hz:", P_bare[i - 3:i + 4])  # shoulder or flat?



        from matplotlib import pyplot as plt
        plt.plot(f, P_bare)
        #plt.show()


