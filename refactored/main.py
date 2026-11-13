import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
from shaker_force import ShakerParameters
from solve import solve_evp
from postprocessing import (plot_eigenmode, plot_frequency_response, plot_mobility_bode, PlateLayout,
                            plot_insertion_loss_bode, draw_plate_layout, plot_radiation_bode)
from vamm_grid import GridConfig, state_to_vamm_list
from probe import make_probe, build_probe_coordinates, mobility_at_probes
from sweep_cache import (CACHE_DIR, make_cache_key, get_or_compute_sweep, get_or_compute_radiated_power,
                         get_or_compute_eigenfrequencies, get_or_compute_dispersion)
from insertion_loss import (assert_comparable_sweeps, compute_insertion_loss,
                            global_insertion_loss_radiated_db, global_insertion_loss_radiated_equal_power_db)
from radiation import radiated_power, build_radiation_points, radiation_efficiency, rho_air, c_air

from frequency_sweep import frequency_sweep_plate, build_frequency_grid, warn_if_grid_too_coarse
from dispersion import (unit_cell_element_counts, build_wavevector_path, dispersion_sweep, analytic_dispersion,
                        analytic_resonator_dispersion, find_complete_gaps, describe_band_gap)
from postprocessing_dispersion import plot_dispersion, plot_tracked_dispersion
import dataclasses

if __name__ == "__main__":

    do_frequency_sweep = True
    do_dispersion = True  # Bloch band diagram of one 0.1 m cell (with do_frequency_sweep: insertion loss side panel)
    do_parameter_study = False  # stage 7: gap maps over f_r / f_X and mass ratio, finite-plate IL at 3 tunings (~35 min uncached)
    free_plate = True # switch: free (shaker-driven) vs. clamped-edge plate
    excitation = "force"  # "force" (allows both free_plate=True or False) or "motion" (needs free_plate=False)
    vamm_mode = "grid"  # "grid" or "manual"

    eigenmode_number = 500 # all modes up to ~5 kHz for the eigenfrequency lines (was 20); the total number of eigenmodes to be computed by eigensolver
    plot_mode_shape = True # separate small solve (eigenmode_to_plot + 1 modes) for the mode plot of the bare plate
    eigenmode_to_plot = 13 # the desired eigenmode INDEX to be plotted (0-based, rigid-body modes included).
    target_frequency_hz = 0.0

    deg = 2
    el_type = "Lagrange"

    ncv_factor = 2 # 2 for the 500-mode solve (memory and time; was 4); integer, default: 2, increase if suspected that not all modes are found by the solver

    plate_config = PlateConfig(
    length = 1.0,
    width = 1.0,
    thickness = 0.005,
    nx = 100,  # 100 elements per metre, as in the Bloch unit cell; cell centres of the 0.1 m grid lie on nodes
    ny = 100,
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

# --- Compute eigenfrequencies and -modes with optional resonators and plot result ---

    if vamm_mode == "grid":
        # 0.1 m lattice as in the Bloch unit cell: one resonator per cell, f_r = 1.2 kHz,
        # resonator mass = mass ratio 0.25 times the plate mass of one cell (rho h a^2)
        grid_cell_length = 0.1
        grid_resonator_frequency_hz = 1200.0
        grid_resonator_mass = 0.25 * plate_config.rho * plate_config.thickness * grid_cell_length ** 2
        grid_config = GridConfig(
            n_x=round(plate_config.length / grid_cell_length), n_y=round(plate_config.width / grid_cell_length),
            stiffness=grid_resonator_mass * (2 * np.pi * grid_resonator_frequency_hz) ** 2,
            mass=grid_resonator_mass, gamma=0.02)
        state = np.repeat(1, grid_config.n_x * grid_config.n_y)  # every cell occupied
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

    # Eigenfrequencies are cached like the sweeps (frequencies only: the mode shapes of 500 modes are ~1 GB).
    # ncv_factor is left out of the key on purpose: it changes how the solver converges, not the result.
    eigen_settings = dict(plate_config=plate_config, free_plate=free_plate, deg=deg, el_type=el_type,
                          eigenmode_number=eigenmode_number, target_hz=target_frequency_hz,
                          code_version="v1")  # bump this string if you change the eigensolver code itself
    key_eigen_bare, blob_eigen_bare = make_cache_key(vamm_list=None, **eigen_settings)
    key_eigen_vamm, blob_eigen_vamm = make_cache_key(vamm_list=vamm_list, **eigen_settings)

    eigenfrequencies_bare = get_or_compute_eigenfrequencies(
        CACHE_DIR / f"eigenfrequencies_bare_{key_eigen_bare}.npz",
        compute=lambda: solve_evp(
            domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
            vamm_list = None, phi_list = phi_list, global_dofs_parent_list = global_dofs_parent_list,
            eigenmode_number = eigenmode_number, target_hz=target_frequency_hz, ncv_factor=ncv_factor)[0],
        settings_blob=blob_eigen_bare)

    eigenfrequencies_vamm = get_or_compute_eigenfrequencies(
        CACHE_DIR / f"eigenfrequencies_vamm_{key_eigen_vamm}.npz",
        compute=lambda: solve_evp(
            domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
            vamm_list = vamm_list, phi_list = phi_list, global_dofs_parent_list = global_dofs_parent_list,
            eigenmode_number = eigenmode_number, target_hz=target_frequency_hz, ncv_factor=ncv_factor)[0],
        settings_blob=blob_eigen_vamm)

    for i, freq in enumerate(eigenfrequencies_bare):
        print(f"mode {i}: {freq:.4f} Hz")


    for i, freq in enumerate(eigenfrequencies_vamm):
        print(f"mode {i}: {freq:.4f} Hz")

    if plot_mode_shape:
        _, eigenmodes_bare = solve_evp(
            domain = domain, function_space = function_space, bcs = bcs, problem = plate_problem_constants,
            vamm_list = None, phi_list = phi_list, global_dofs_parent_list = global_dofs_parent_list,
            eigenmode_number = eigenmode_to_plot + 1, target_hz=target_frequency_hz, ncv_factor=4)
        plot_eigenmode(
            domain = domain, function_space = function_space, eigenmodes = eigenmodes_bare,
            eigenmode_index = eigenmode_to_plot, length = plate_config.length,
            vamm_list = vamm_list
            , phi_list = phi_list, local_to_global_w_list = local_to_global_w_list)
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
                                               amount=3, vamm_list=vamm_list)
        probes = [make_probe(domain, function_space, x, y) for x, y in probe_coords]
        probes.insert(0, shaker_probe)
        f_start = 500
        f_end = 5000
        gamma_plate = 0.04
        grid = build_frequency_grid(f_start=f_start, f_end=f_end, gamma=0.02)  # smallest gamma in either run
        layout = PlateLayout(
            plate_config=plate_config, shaker_config=shaker_config, probes=probes, vamm_list=vamm_list
        )

        surface_area = plate_config.length * plate_config.width
        shaker_force = shaker_config.force_amplitude

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
                free_plate=free_plate, excitation=excitation, shaker_config=shaker_config,
                factorization_package="mumps"),  # solver choice is not part of the cache key
            settings_blob=blob_vamm)

        sweep_bare = get_or_compute_sweep(
            CACHE_DIR / f"sweep_bare_{key_bare}.npz",
            compute=lambda: frequency_sweep_plate(
                domain=domain, function_space=function_space, problem=plate_problem_constants,
                plate_config=plate_config, f_start=f_start, f_end=f_end,
                vamm_list=None, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
                frequency_grid=grid, gamma=gamma_plate,
                free_plate=free_plate, excitation=excitation, shaker_config=shaker_config,
                factorization_package="mumps"),
            settings_blob=blob_bare)

        bare_probe_mobility = mobility_at_probes(
            sweep=sweep_bare, probes=probes, force_amplitude=shaker_force)

        vamm_probe_mobility = mobility_at_probes(
            sweep=sweep_vamm, probes=probes, force_amplitude=shaker_force)


        probes_mesh, dS, r = build_radiation_points(
            domain=domain, function_space=function_space, plate_config=plate_config)

        bare_probe_mobility_radiation = mobility_at_probes(
            sweep=sweep_bare, probes=probes_mesh, force_amplitude=shaker_force)

        vamm_probe_mobility_radiation = mobility_at_probes(
            sweep=sweep_vamm, probes=probes_mesh, force_amplitude=shaker_force)


        # cached like the sweeps: the key contains the sweep's key plus the air constants and the force amplitude
        radiation_settings = dict(rho_air=rho_air, c_air=c_air, force_amplitude=shaker_force,
                                  code_version="v1")  # bump this string if you change the radiation code itself
        key_radiation_bare, blob_radiation_bare = make_cache_key(sweep_key=key_bare, **radiation_settings)
        key_radiation_vamm, blob_radiation_vamm = make_cache_key(sweep_key=key_vamm, **radiation_settings)

        P_bare = get_or_compute_radiated_power(
            CACHE_DIR / f"radiation_bare_{key_radiation_bare}.npz", f_values=sweep_bare.f_values,
            compute=lambda: radiated_power(
                Y_grid=bare_probe_mobility_radiation, r=r, dS=dS,
                f_values=sweep_bare.f_values, force_amplitude=shaker_force),
            settings_blob=blob_radiation_bare)

        P_vamm = get_or_compute_radiated_power(
            CACHE_DIR / f"radiation_vamm_{key_radiation_vamm}.npz", f_values=sweep_bare.f_values,
            compute=lambda: radiated_power(
                Y_grid=vamm_probe_mobility_radiation, r=r, dS=dS,
                f_values=sweep_bare.f_values, force_amplitude=shaker_force),
            settings_blob=blob_radiation_vamm)

        sigma_bare = radiation_efficiency(
            P_bare, bare_probe_mobility_radiation, surface_area, force_amplitude=shaker_force)
        sigma_vamm = radiation_efficiency(
            P_vamm, vamm_probe_mobility_radiation, surface_area, force_amplitude=shaker_force)

        Y_dp_bare = mobility_at_probes(sweep_bare, [shaker_probe], shaker_force)[0]
        Y_dp_vamm = mobility_at_probes(sweep_vamm, [shaker_probe], shaker_force)[0]

        insertion_loss_db = global_insertion_loss_radiated_db(P_bare, P_vamm)
        insertion_loss_db_power = global_insertion_loss_radiated_equal_power_db(
            P_bare, P_vamm, Y_dp_bare, Y_dp_vamm, force_amplitude=shaker_force)

        assert_comparable_sweeps(sweep_vamm, sweep_bare)

        il_pt_rad, dphi_rad, il_f_rad, il_p_rad = compute_insertion_loss(
            bare_probe_mobility=bare_probe_mobility, vamm_probe_mobility=vamm_probe_mobility,
            sweep_bare=sweep_bare, sweep_vamm=sweep_vamm)

        radiation_title = (f"Radiated sound power, {len(vamm_list)} resonators on a 0.1 m grid "
                           f"(f_r = 1.2 kHz, mass ratio 0.25)")
        # full band: hundreds of modes, so no eigenfrequency lines but the mode count per 50 Hz
        plot_radiation_bode(
            f_values=sweep_bare.f_values, P_bare=P_bare, P_vamm=P_vamm, sigma_bare=sigma_bare, sigma_vamm=sigma_vamm,
            il_rad=insertion_loss_db, il_rad_pow=insertion_loss_db_power, il_force=il_f_rad,
            eigenfrequencies_bare=eigenfrequencies_bare, eigenfrequencies_vamm=eigenfrequencies_vamm,
            title=radiation_title, show_eigenfrequency_lines=False, mode_count_bin_hz=50,
            save_path="results/radiation_1x1m_100resonators_1200hz_500to5000hz_modecount.pdf")
        # zoom around the band gap with the eigenfrequency lines
        zoom = (sweep_bare.f_values >= 900) & (sweep_bare.f_values <= 1500)
        plot_radiation_bode(
            f_values=sweep_bare.f_values[zoom], P_bare=P_bare[zoom], P_vamm=P_vamm[zoom],
            sigma_bare=sigma_bare[zoom], sigma_vamm=sigma_vamm[zoom], il_rad=insertion_loss_db[zoom],
            il_rad_pow=insertion_loss_db_power[zoom], il_force=il_f_rad[zoom],
            eigenfrequencies_bare=eigenfrequencies_bare, eigenfrequencies_vamm=eigenfrequencies_vamm,
            title=radiation_title + ", 900-1500 Hz",
            save_path="results/radiation_1x1m_100resonators_1200hz_900to1500hz_eigenfrequencies.pdf")

        il_pt, dphi, il_f, il_p = compute_insertion_loss(
            bare_probe_mobility=bare_probe_mobility, vamm_probe_mobility=vamm_probe_mobility,
            sweep_bare=sweep_bare, sweep_vamm=sweep_vamm)

        plot_insertion_loss_bode(
            f_values=sweep_bare.f_values,
            il_pointwise=il_pt, dphi_deg=dphi, il_force=il_f, il_power=il_p,
            eigenfrequencies_vamm=eigenfrequencies_vamm, eigenfrequencies_bare=eigenfrequencies_bare,
            layout=layout, draw_layout=lambda ax, layout: draw_plate_layout(
                ax=ax, plate_config=layout.plate_config, shaker_config=layout.shaker_config, probes=probes,
                vamm_list=layout.vamm_list),
            title=f"Insertion loss, {len(vamm_list)} resonators on a 0.1 m grid")

        # --- Driving point (probe 0 = shaker location) ---
        plot_mobility_bode(
            f_values=sweep_bare.f_values,
            mobilities=[bare_probe_mobility[0], vamm_probe_mobility[0]],
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
                mobilities=[bare_probe_mobility[i], vamm_probe_mobility[i]],
                labels=["bare plate", "with VAMMs"],
                eigenfrequencies=[eigenfrequencies_bare, eigenfrequencies_vamm],
                title=f"Transfer mobility, probe {i} ({d:.2f} m from shaker)",
                layout=layout, highlight_probe=i
                #, save_path=rf"/home/local/CSI/dm27demu/Desktop/Mobilities/Y_{i}.pdf"
            )

        plot_frequency_response(
            f_values=sweep_vamm.f_values, rms_velocity=sweep_vamm.rms_velocity,
            eigenfrequencies=eigenfrequencies_bare)


# --- Dispersion diagram of the unit cell (optionally with one resonator at the cell centre) ---
    if do_dispersion:
        unit_cell_length = 0.1          # = VAMM spacing; same material and thickness as plate_config
        maximum_frequency_hz = 5000.0   # bands below this frequency (critical frequency of 5 mm steel: ~2.45 kHz)
        include_resonator = True
        resonator_frequency_hz = 1200.0
        resonator_mass_ratio = 0.25     # resonator mass / plate mass per cell (rho h a^2)
        track_bands = True              # branch tracking: lines coloured by plate share instead of sorted markers

        unit_cell_elements_x, unit_cell_elements_y = unit_cell_element_counts(unit_cell_length, unit_cell_length)
        unit_cell_config = dataclasses.replace(plate_config, length=unit_cell_length, width=unit_cell_length,
                                               nx=unit_cell_elements_x, ny=unit_cell_elements_y)
        _, unit_cell_function_space, _, unit_cell_problem = build_plate_problem(
            unit_cell_config, deg=2, el_type="Lagrange")
        wavevector_path = build_wavevector_path(unit_cell_length, unit_cell_length, points_per_segment=40)

        resonator_mass = resonator_mass_ratio * plate_config.rho * plate_config.thickness * unit_cell_length ** 2
        resonator_stiffness = resonator_mass * (2 * np.pi * resonator_frequency_hz) ** 2
        unit_cell_vamm_list = create_vamm_list_and_assign_indices(
            [(unit_cell_length / 2, unit_cell_length / 2, resonator_stiffness, resonator_mass)]
        ) if include_resonator else None

        computed_limit_hz = 1.1 * maximum_frequency_hz if track_bands else maximum_frequency_hz
        # tracked: compute 10 % above the plotted range, so branches enter and leave outside the plot
        dispersion_settings = dict(
            cell_config=unit_cell_config, deg=2, el_type="Lagrange", wavevectors=wavevector_path.wavevectors,
            maximum_frequency_hz=computed_limit_hz, vamm_list=unit_cell_vamm_list, track_bands=track_bands,
            minimum_match_quality=0.5,
            code_version="v1")  # bump this string if you change the Bloch solver or the tracking code itself
        dispersion_key, dispersion_blob = make_cache_key(**dispersion_settings)
        dispersion_result = get_or_compute_dispersion(
            CACHE_DIR / f"dispersion_{'vamm' if include_resonator else 'bare'}_{dispersion_key}.npz",
            compute=lambda: dispersion_sweep(
                function_space=unit_cell_function_space, problem=unit_cell_problem, plate_config=unit_cell_config,
                wavevector_path=wavevector_path, maximum_frequency_hz=computed_limit_hz,
                vamm_list=unit_cell_vamm_list, track_bands=track_bands, minimum_match_quality=0.5),
            settings_blob=dispersion_blob)

        # complete gaps from the sorted frequencies (never from the tracked branches)
        band_gaps = find_complete_gaps(dispersion_result.frequencies, dispersion_result.path)
        print(f"Complete band gaps below {computed_limit_hz:.0f} Hz: {len(band_gaps)}")
        for gap in band_gaps:
            print("  " + describe_band_gap(gap))

        # Side panel with the finite-plate insertion loss, if the forced-response block ran in this process.
        # It is only meaningful if that plate carries the same resonators on the same lattice (grid mode above:
        # 0.1 m, 1.2 kHz, mass ratio 0.25, equal to the cell settings here).
        insertion_loss_curves = None
        if do_frequency_sweep and include_resonator:
            insertion_loss_curves = [
                (r"$IL_{rad}$ (equal force)", sweep_bare.f_values, insertion_loss_db),
                (r"$IL_{rad}$ (equal input power)", sweep_bare.f_values, insertion_loss_db_power)]
        figure_name = ("dispersion_cell{:g}m_".format(unit_cell_length)
                       + (f"{resonator_frequency_hz:.0f}hz_mass{resonator_mass_ratio:g}_" if include_resonator
                          else "bare_")
                       + f"0to{maximum_frequency_hz:.0f}hz" + ("_with_il" if insertion_loss_curves else ""))
        dispersion_save_path = f"results/{figure_name}.pdf"

        bare_analytic = analytic_dispersion(wavevector_path, unit_cell_config,
                                            maximum_frequency_hz=1.1 * maximum_frequency_hz)
        if track_bands:
            plot_tracked_dispersion(
                result=dispersion_result, colour_by="plate_share" if include_resonator else "branch",
                frequency_limit=maximum_frequency_hz,
                resonator_frequencies_hz=[resonator_frequency_hz] if include_resonator else None,
                title=f"Unit cell {unit_cell_length} m" + (
                    f" with resonator: f_r = {resonator_frequency_hz:.0f} Hz, mass ratio {resonator_mass_ratio}"
                    if include_resonator else " (bare)") + f", h = {plate_config.thickness * 1e3:.0f} mm",
                band_gaps=band_gaps, insertion_loss_curves=insertion_loss_curves,
                save_path=dispersion_save_path)
        elif include_resonator:
            plot_dispersion(
                result=dispersion_result,
                analytic_frequencies=analytic_resonator_dispersion(
                    wavevector_path, unit_cell_config, resonator_stiffness, resonator_mass,
                    maximum_frequency_hz=1.1 * maximum_frequency_hz),
                analytic_label="analytic (plane-wave expansion)",
                bare_frequencies=bare_analytic, resonator_frequencies_hz=[resonator_frequency_hz],
                title=f"Unit cell {unit_cell_length} m with resonator: f_r = {resonator_frequency_hz:.0f} Hz, "
                      f"mass ratio {resonator_mass_ratio}, h = {plate_config.thickness * 1e3:.0f} mm",
                band_gaps=band_gaps, insertion_loss_curves=insertion_loss_curves,
                save_path=dispersion_save_path)
        else:
            plot_dispersion(
                result=dispersion_result, analytic_frequencies=bare_analytic,
                title=f"Bare unit cell {unit_cell_length} x {unit_cell_length} m, "
                      f"h = {plate_config.thickness * 1e3:.0f} mm",
                band_gaps=band_gaps, insertion_loss_curves=insertion_loss_curves,
                save_path=dispersion_save_path)


# --- Parameter study (stage 7): complete gaps vs tuning and mass ratio, finite-plate insertion loss at 3 tunings ---
    if do_parameter_study:
        from parameter_study import (gap_parameter_sweep, get_or_compute_gap_study, FinitePlateSetup,
                                     cached_bare_radiated_power, windowed_insertion_loss, cell_band_gaps,
                                     bragg_frequency_x, FREQUENCY_RATIOS, MASS_RATIOS, LIMIT_OVER_BRAGG,
                                     POINTS_PER_SEGMENT)
        from postprocessing_dispersion import plot_gap_map, plot_gap_width, plot_insertion_loss_vs_gaps
        from compare_cool_2024 import cool_plate_config

        study_cell_length = 0.1
        study_cell_config = dataclasses.replace(plate_config, length=study_cell_length, width=study_cell_length,
                                                nx=10, ny=10)   # plane-wave model: the mesh does not enter
        gap_studies = {}
        for study_name, cell_config in [("cell0.1m", study_cell_config), ("cool_cell0.05m", cool_plate_config())]:
            key_study, blob_study = make_cache_key(
                cell_config=cell_config, frequency_ratios=FREQUENCY_RATIOS, mass_ratios=MASS_RATIOS,
                limit_over_bragg=LIMIT_OVER_BRAGG, points_per_segment=POINTS_PER_SEGMENT,
                code_version="v1")  # bump this string if you change the gap study code itself
            gap_studies[study_name] = get_or_compute_gap_study(
                CACHE_DIR / f"gap_study_{study_name}_{key_study}.npz",
                compute=lambda: gap_parameter_sweep(cell_config), settings_blob=blob_study)

        for study_name, study in gap_studies.items():
            widest = study.widest_relative_width()
            print(f"Gap study {study_name} (f_X = {study.bragg_frequency_hz:.1f} Hz): widest complete gap per mass ratio")
            for mass_index, mass_ratio in enumerate(study.mass_ratios):
                best = int(np.argmax(widest[mass_index]))
                print(f"  mu = {mass_ratio:g}: {100 * widest[mass_index, best]:.1f} % at f_r / f_X = "
                      f"{study.frequency_ratios[best]:.2f}")
        study_labels = ["0.1 m cell, h/a = 0.05 (plane-wave model)",
                        "0.05 m cell with Cool's Table 1 parameters, h/a = 0.1 (plane-wave model)"]
        # FE gap of the production design (validate_band_gaps.py): 0.1 m cell, f_r = 1.2 kHz, mu = 0.25
        reference_designs = [dict(label="1.2 kHz design, FE (1031-1195 Hz)", mass_ratio=0.25,
                                  frequency_ratio=1200.0 / gap_studies["cell0.1m"].bragg_frequency_hz,
                                  lower_ratio=1031.4 / gap_studies["cell0.1m"].bragg_frequency_hz,
                                  upper_ratio=1195.3 / gap_studies["cell0.1m"].bragg_frequency_hz)]
        plot_gap_map([gap_studies["cell0.1m"], gap_studies["cool_cell0.05m"]], labels=study_labels,
                     title="Complete band gaps vs resonator tuning (plane-wave model, centred resonator)",
                     save_path="results/gap_map_cell0.1m_vs_cool0.05m.pdf", reference_designs=reference_designs)
        plot_gap_width([gap_studies["cell0.1m"], gap_studies["cool_cell0.05m"]], labels=study_labels,
                       title="Widest complete gap vs resonator tuning (plane-wave model)",
                       save_path="results/gap_width_cell0.1m_vs_cool0.05m.pdf")

        # Finite 1x1 m plate with 10x10 resonators (mass ratio 0.25) at three tunings. 1.2 kHz reuses the full
        # sweep of the forced-response block (main's vamm_list, require_cached); the others sweep only a window.
        finite_plate_setup = FinitePlateSetup(
            domain=domain, function_space=function_space, problem=plate_problem_constants, plate_config=plate_config,
            shaker_config=ShakerParameters(x=plate_config.length / 3, y=plate_config.width / 3,
                                           force_amplitude=1.0, phase=0.0),
            frequency_grid=build_frequency_grid(f_start=500, f_end=5000, gamma=0.02), gamma_plate=0.04,
            free_plate=free_plate, excitation=excitation, deg=deg, el_type=el_type)
        bare_radiated_power = cached_bare_radiated_power(finite_plate_setup)
        study_mass_ratio = 0.25
        study_path = build_wavevector_path(study_cell_length, study_cell_length, points_per_segment=POINTS_PER_SEGMENT)
        study_bragg_hz = bragg_frequency_x(study_cell_config)
        insertion_loss_cases = []
        for resonator_frequency_study_hz, frequency_window_hz, plot_window_hz in [
                (600.0, (500.0, 900.0), (500.0, 900.0)),
                (1200.0, (0.0, np.inf), (900.0, 1500.0)),
                (2000.0, (1500.0, 2700.0), (1500.0, 2700.0))]:
            if resonator_frequency_study_hz == 1200.0:
                assert vamm_mode == "grid", "the 1.2 kHz case reuses the grid vamm_list of the forced-response block"
                study_vamm_list = vamm_list
            else:
                study_resonator_mass = study_mass_ratio * plate_config.rho * plate_config.thickness * study_cell_length ** 2
                study_grid_config = GridConfig(
                    n_x=round(plate_config.length / study_cell_length), n_y=round(plate_config.width / study_cell_length),
                    stiffness=study_resonator_mass * (2 * np.pi * resonator_frequency_study_hz) ** 2,
                    mass=study_resonator_mass, gamma=0.02)
                study_vamm_list = state_to_vamm_list(np.repeat(1, study_grid_config.n_x * study_grid_config.n_y),
                                                     study_grid_config, plate_config.length, plate_config.width)
            f_values_study, insertion_loss_study_db = windowed_insertion_loss(
                finite_plate_setup, study_vamm_list, frequency_window_hz, bare_radiated_power,
                require_cached=(resonator_frequency_study_hz == 1200.0))
            in_plot = (f_values_study >= plot_window_hz[0]) & (f_values_study <= plot_window_hz[1])
            study_gaps = cell_band_gaps(study_cell_config, study_path, resonator_frequency_study_hz, study_mass_ratio,
                                        maximum_frequency_hz=LIMIT_OVER_BRAGG * study_bragg_hz)
            print(f"f_r = {resonator_frequency_study_hz:.0f} Hz (f_r / f_X = {resonator_frequency_study_hz / study_bragg_hz:.2f}):"
                  f" gaps {[(round(gap.lower_edge_hz), round(gap.upper_edge_hz)) for gap in study_gaps]} Hz,"
                  f" IL max {insertion_loss_study_db[in_plot].max():+.1f} dB at"
                  f" {f_values_study[in_plot][np.argmax(insertion_loss_study_db[in_plot])]:.0f} Hz,"
                  f" min {insertion_loss_study_db[in_plot].min():+.1f} dB at"
                  f" {f_values_study[in_plot][np.argmin(insertion_loss_study_db[in_plot])]:.0f} Hz", flush=True)
            insertion_loss_cases.append(dict(
                label=f"$f_r$ = {resonator_frequency_study_hz:.0f} Hz ($f_r/f_X$ = {resonator_frequency_study_hz / study_bragg_hz:.2f})",
                f_values=f_values_study[in_plot], insertion_loss_db=insertion_loss_study_db[in_plot],
                gaps=[(gap.lower_edge_hz, gap.upper_edge_hz) for gap in study_gaps],
                resonator_frequency_hz=resonator_frequency_study_hz))
        plot_insertion_loss_vs_gaps(
            insertion_loss_cases, save_path="results/insertion_loss_vs_gaps_1x1m_100resonators_mass0.25.pdf",
            title="1x1 m plate, 10x10 resonators (mass ratio 0.25): insertion loss vs Bloch gaps of the 0.1 m cell")
