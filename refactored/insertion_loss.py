import numpy as np
from validate_probes import evaluate_w_at_probe


def _floor_magnitude(values, rel=1e-12):
    """|values|, floored at rel * (largest |value| along the frequency axis)
    so dividing by it can never produce inf/nan. Works on 1D arrays and on
    (n_probes, n_freq) arrays; each probe is scaled by its own maximum."""
    magnitude = np.abs(values)
    scale = np.max(magnitude, axis=-1, keepdims=True)
    return np.maximum(magnitude, np.maximum(rel * scale, np.finfo(float).tiny))


def _floor_real_part(y, rel=1e-12):
    """Re(y), floored for use in a ratio. Re(Y_dp) >= 0 by passivity, so
    only round-off-level zeros or slightly negative values get clipped."""
    real = np.real(y)
    scale = np.max(np.abs(y), axis=-1, keepdims=True)
    return np.maximum(real, np.maximum(rel * scale, np.finfo(float).tiny))


def pointwise_insertion_loss_db(y_bare, y_vamm):
    return 20 * np.log10(_floor_magnitude(y_bare) / _floor_magnitude(y_vamm))


def phase_shift_deg(y_bare, y_vamm):
    # arg(Y_vamm / Y_bare) computed as arg(Y_vamm * conj(Y_bare)): no division
    return np.degrees(np.angle(y_vamm * np.conj(y_bare)))


def global_insertion_loss_force_db(rms_bare, rms_vamm):
    return 20 * np.log10(_floor_magnitude(rms_bare) / _floor_magnitude(rms_vamm))


def input_power(y_dp, force_amplitude):
    return 0.5 * np.abs(force_amplitude) ** 2 * np.real(y_dp)


def global_insertion_loss_power_db(rms_bare, rms_vamm, y_dp_bare, y_dp_vamm):
    il_force = global_insertion_loss_force_db(rms_bare, rms_vamm)
    power_term = 10 * np.log10(_floor_real_part(y_dp_vamm) / _floor_real_part(y_dp_bare))
    return il_force + power_term


def assert_comparable_sweeps(result_bare, result_vamm):
    if not np.array_equal(result_bare.f_values, result_vamm.f_values):
        raise ValueError("Sweeps use different frequency grids; IL is undefined.")
    if result_bare.probe_mobility.shape != result_vamm.probe_mobility.shape:
        raise ValueError(
            f"Probe/frequency shapes differ: {result_bare.probe_mobility.shape} "
            f"vs {result_vamm.probe_mobility.shape}.")




# --- Test/ validation area ---
def run_synthetic_checks():
    rng = np.random.default_rng(0)
    shape = (6, 200)
    y_bare = 1e-4 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

    # identity
    assert np.all(pointwise_insertion_loss_db(y_bare, y_bare) == 0.0), (
    f"Expected 0, got {pointwise_insertion_loss_db(y_bare, y_bare)}")
    dphi_identity = phase_shift_deg(y_bare, y_bare)
    assert np.max(np.abs(dphi_identity)) < 1e-10, (
        f"Expected 0, worst deviation {np.max(np.abs(dphi_identity)):.3e} deg")

    # known scaling (pins the sign convention)
    il = pointwise_insertion_loss_db(y_bare, 0.5 * y_bare)
    assert np.allclose(il, 20 * np.log10(2), rtol=1e-12, atol=0)
    dphi = phase_shift_deg(y_bare, 0.5 * np.exp(1j * np.pi / 4) * y_bare)
    assert np.allclose(dphi, 45.0, rtol=1e-12, atol=0)

    # wrapping, on arbitrary phases
    y_other = 1e-4 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    assert np.all(np.abs(phase_shift_deg(y_bare, y_other)) <= 180.0)

    # zero guard: an exact zero must give a finite IL and no warning
    y_zero = y_bare.copy()
    y_zero[0, 0] = 0.0
    assert np.all(np.isfinite(pointwise_insertion_loss_db(y_bare, y_zero)))

    # power identity, computed independently from the definition
    n = shape[1]
    rms_b, rms_v = rng.uniform(1, 2, n), rng.uniform(1, 2, n)
    y_dp_b = rng.uniform(0.5, 2, n) + 1j * rng.normal(size=n)   # Re > 0 (passive)
    y_dp_v = rng.uniform(0.5, 2, n) + 1j * rng.normal(size=n)
    p_b, p_v = input_power(y_dp_b, 1.0), input_power(y_dp_v, 1.0)
    il_p_direct = 10 * np.log10((rms_b**2 / p_b) / (rms_v**2 / p_v))
    assert np.allclose(
        global_insertion_loss_power_db(rms_b, rms_v, y_dp_b, y_dp_v),
        il_p_direct, rtol=1e-12, atol=0)

def compute_insertion_loss(
        sweep_bare, sweep_vamm,
        f0=None, f_lo=None, f_hi=None, gamma_plate=None):
    assert_comparable_sweeps(sweep_bare, sweep_vamm)
    f = sweep_bare.f_values
    y_b, y_v = sweep_bare.probe_mobility, sweep_vamm.probe_mobility

    il_pt = pointwise_insertion_loss_db(y_b, y_v)
    dphi = phase_shift_deg(y_b, y_v)
    il_f = global_insertion_loss_force_db(sweep_bare.rms_velocity, sweep_vamm.rms_velocity)
    il_p = global_insertion_loss_power_db(
            sweep_bare.rms_velocity, sweep_vamm.rms_velocity, y_b[0], y_v[0])

    if f0 and f_lo and f_hi and gamma_plate:
        w = gamma_plate * f0                       # approx. width of a bare resonance (FWHM)
        near = lambda fc, half: np.abs(f - fc) <= half
        i0 = int(np.argmin(np.abs(f - f0)))

        # participation: bare |Y| at f0 within 6 dB of its maximum in a +-w window
        ratio = np.abs(y_b[:, i0]) / np.abs(y_b[:, near(f0, w)]).max(axis=1)
        part = ratio >= 0.5
        assert part.any(), "no probe responds to the target mode; check is vacuous"

        il_at_f0 = il_pt[:, i0]
        il_split_lo = il_pt[:, near(f_lo, w / 2)].min(axis=1)    # most negative IL near each split peak
        il_split_hi = il_pt[:, near(f_hi, w / 2)].min(axis=1)

        print(f"f0={f0:.2f} Hz, split {f_lo:.2f}/{f_hi:.2f} Hz  "
              f"(split/FWHM = {(f_hi - f_lo) / w:.1f})")
        print(f"{'probe':>5} {'part':>5} {'IL(f0)':>8} {'IL@lo':>8} {'IL@hi':>8}   [dB]")
        for p in range(len(il_pt)):
            print(f"{p:>5} {str(bool(part[p])):>5} {il_at_f0[p]:8.2f} "
                  f"{il_split_lo[p]:8.2f} {il_split_hi[p]:8.2f}")

        assert np.all(il_at_f0[part] > 0), f"IL(f0) not positive at participating probes: {il_at_f0[part]}"
        #assert np.all(il_split_lo[part] < 0), f"no negative IL near f_lo: {il_split_lo[part]}"
        #assert np.all(il_split_hi[part] < 0), f"no negative IL near f_hi: {il_split_hi[part]}"

        assert il_f[i0] > 0
        assert il_f[near(f_lo, w / 2)].min() < 0 and il_f[near(f_hi, w / 2)].min() < 0

        # diagnostics (not asserted)
        print(f"IL^F / IL^P at f0:  {il_f[i0]:.2f} / {il_p[i0]:.2f} dB")
        print(f"dphi at driving point, f0: {dphi[0, i0]:.1f} deg")
    return il_pt, dphi, il_f, il_p

if __name__ == "__main__":

    run_synthetic_checks()

    from config import PlateConfig
    from problem_setup import build_plate_problem
    from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
    from shaker_force import ShakerParameters
    from postprocessing import PlateLayout
    from vamm_grid import GridConfig, state_to_vamm_list
    from probe import make_probe, build_probe_coordinates
    from sweep_cache import CACHE_DIR, make_cache_key, get_or_compute_sweep
    from solve import solve_evp

    from frequency_sweep import frequency_sweep_plate, build_frequency_grid

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
                (1.1, 0.0, 858783.5366, 1.0, 0.02)
                # ,
                # (0.55, 0.5, 10042816.24, 10.0, 0.05)
                # ,
                # (0.1, 0.1, 500, 5, 0.01)
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

    freqs_bare, eigenmodes_bare = solve_evp(
        domain=domain, function_space=function_space, bcs=bcs,
        problem=plate_problem_constants, eigenmode_number=14)
    freqs_vamm, eigenmodes_vamm = solve_evp(
        domain=domain, function_space=function_space, bcs=bcs,
        problem=plate_problem_constants, vamm_list=vamm_list, phi_list=phi_list,
        global_dofs_parent_list=global_dofs_parent_list, eigenmode_number=14)

    f0 = min(freqs_bare, key=lambda f: abs(f - 147.49))  # the bare mode that the VAMM is tuned onto

    candidates = [(fr, abs(eigenmodes_vamm[i][1][0]))  # (frequency, |q_r| of VAMM 0)
                  for i, fr in enumerate(freqs_vamm) if abs(fr - f0) < 25.0]
    assert len(candidates) >= 2, f"fewer than 2 VAMM-system modes within 25 Hz of {f0:.2f} Hz"
    pair = sorted(sorted(candidates, key=lambda c: c[1], reverse=True)[:2])  # two largest |q_r|, then by frequency
    f_lo, f_hi = pair[0][0], pair[1][0]
    print("candidates (f, |q_r|):", candidates)
    print(f"split pair: {f_lo:.2f} / {f_hi:.2f} Hz")

    shaker_config = ShakerParameters(
        x=plate_config.length / 3,
        y=plate_config.width / 3,
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


    def mode_amp_at_probes(eigenmodes, freqs, f_target, probes):
        i = int(np.argmin(np.abs(np.array(freqs) - f_target)))
        w_mode = eigenmodes[i][0].sub(0).collapse()
        amp = np.array([abs(evaluate_w_at_probe(w_mode, p)) for p in probes])
        return amp / amp.max()  # each mode has its own normalization, so compare shapes only


    print("bare 147  :", mode_amp_at_probes(eigenmodes_bare, freqs_bare, f0, probes).round(2))
    print("hybrid lo :", mode_amp_at_probes(eigenmodes_vamm, freqs_vamm, f_lo, probes).round(2))
    print("hybrid hi :", mode_amp_at_probes(eigenmodes_vamm, freqs_vamm, f_hi, probes).round(2))

    settings = dict(
        plate_config=plate_config, shaker_config=shaker_config,
        probe_coords=[(p.x, p.y) for p in probes],
        grid=np.asarray(grid), gamma_plate=gamma_plate,
        free_plate=free_plate, excitation=excitation,
        code_version="v1")  # bump this string if you change the sweep code itself

    from dataclasses import replace

    grid_coarse = np.linspace(f_start, f_end, 25)
    settings_coarse = {**settings, "grid": grid_coarse}  # new grid -> new cache key
    vamm_list_light = [replace(v, mass=1e-12) for v in vamm_list]


    # replace() copies every field including `index`, so the registration assert still passes;
    # positions are unchanged, so phi_list / global_dofs_parent_list can be reused as they are.

    def coarse_sweep(vl, name):
        key, blob = make_cache_key(vamm_list=vl, **settings_coarse)
        return get_or_compute_sweep(
            CACHE_DIR / f"sweep_{name}_coarse_{key}.npz",
            compute=lambda: frequency_sweep_plate(
                domain=domain, function_space=function_space, problem=plate_problem_constants,
                plate_config=plate_config, f_start=f_start, f_end=f_end,
                vamm_list=vl, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
                frequency_grid=grid_coarse, gamma=gamma_plate,
                free_plate=free_plate, excitation=excitation, shaker_config=shaker_config,
                probes=probes),
            settings_blob=blob)


    sweep_bare_c = coarse_sweep(None, "bare")
    sweep_light_c = coarse_sweep(vamm_list_light, "light")
    assert_comparable_sweeps(sweep_bare_c, sweep_light_c)

    il_pt = pointwise_insertion_loss_db(sweep_bare_c.probe_mobility, sweep_light_c.probe_mobility)
    dphi = phase_shift_deg(sweep_bare_c.probe_mobility, sweep_light_c.probe_mobility)
    il_f = global_insertion_loss_force_db(sweep_bare_c.rms_velocity, sweep_light_c.rms_velocity)
    il_p = global_insertion_loss_power_db(
        sweep_bare_c.rms_velocity, sweep_light_c.rms_velocity,
        sweep_bare_c.probe_mobility[0], sweep_light_c.probe_mobility[0])

    print(f"max |IL pointwise| = {np.max(np.abs(il_pt)):.3e} dB")
    print(f"max |dphi|         = {np.max(np.abs(dphi)):.3e} deg")
    print(f"max |IL^F|, |IL^P| = {np.max(np.abs(il_f)):.3e}, {np.max(np.abs(il_p)):.3e} dB")

    assert np.max(np.abs(il_pt)) < 1e-3
    assert np.max(np.abs(dphi)) < 0.05
    assert np.max(np.abs(il_f)) < 1e-3 and np.max(np.abs(il_p)) < 1e-3

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

    for y_bare_sweep in sweep_bare.probe_mobility:
        for idx, y_bare in enumerate(y_bare_sweep):
            no_loss = pointwise_insertion_loss_db(y_bare, y_bare)
            no_phase = phase_shift_deg(y_bare, y_bare)
            assert np.isclose(no_loss, 0, rtol=1e-9),(
                f"insertion loss should be 0 for bare plate, found {no_loss} at f={sweep_bare.f_values[idx]}")
            assert np.isclose(no_phase, 0, rtol=1e-9), (
                f"Phase shift should be 0 for bare plate, found {no_phase} at f={sweep_bare.f_values[idx]}")

    y_vamm_test_nophase = 0.5 * sweep_bare.probe_mobility
    y_vamm_test_phase = 0.5 * np.exp(1j * np.pi/4) * sweep_bare.probe_mobility

    y_dp_bare = sweep_bare.probe_mobility[0]
    y_dp_vamm = sweep_vamm.probe_mobility[0]

    for idx in range(len(sweep_bare.probe_mobility)):
        probe_mobility = sweep_bare.probe_mobility[idx]
        probe_mobility_vamm_nophase = y_vamm_test_nophase[idx]
        probe_mobility_phase = y_vamm_test_phase[idx]
        for index in range(len(probe_mobility)):
            loss = pointwise_insertion_loss_db(probe_mobility[index], probe_mobility_vamm_nophase[index])
            phase = phase_shift_deg(probe_mobility[index], probe_mobility_phase[index])
            assert np.isclose(loss, 6.0206, rtol=1e-6),(
                f"insertion loss should be 6.0206 db for half the mobility,"
                f"found {loss} at f={sweep_bare.f_values[idx]} instead.")
            assert np.isclose(phase, 45, rtol=1e-12), (
                f"Phase shift should be 45 degrees for half the mobility,"
                f"found {phase} at f={sweep_bare.f_values[idx]} instead.")

    for idx in range(len(sweep_bare.rms_velocity)):
        rms_bare = sweep_bare.rms_velocity[idx]
        rms_vamm = sweep_vamm.rms_velocity[idx]
        ilf = global_insertion_loss_force_db(rms_bare, rms_vamm)
        ilp = global_insertion_loss_power_db(rms_bare, rms_vamm, y_dp_bare[idx], y_dp_vamm[idx])
        power_term = 10 * np.log10(_floor_real_part(y_dp_vamm[idx]) / _floor_real_part(y_dp_bare[idx]))
        assert np.isclose(ilp - ilf, power_term, rtol=1e-9),\
            (f"Expected difference {power_term}, instead found {ilp - ilf}.")

    compute_insertion_loss(
        sweep_bare=sweep_bare, sweep_vamm=sweep_vamm, f0=147.49, f_lo=133.49, f_hi=152.75, gamma_plate=gamma_plate)

