import numpy as np
from validate_probes import evaluate_w_at_probe

from radiation import radiated_power, build_radiation_points


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




def _floor_magnitude(values, rel=1e-12):
    """|values|, floored at rel * (largest |value| along the frequency axis)
    so dividing by it can never produce inf/nan. Works on 1D arrays and on
    (n_probes, n_freq) arrays; each probe is scaled by its own maximum."""
    magnitude = np.abs(values)
    scale = np.max(magnitude, axis=-1, keepdims=True)
    return np.maximum(magnitude, np.maximum(rel * scale, np.finfo(float).tiny))

def _floor_scalar(x):
    return max(float(x), np.finfo(float).tiny)

def _floor_real_part(y, rel=1e-12):
    """Re(y), floored for use in a ratio. Re(Y_dp) >= 0 by passivity, so
    only round-off-level zeros or slightly negative values get clipped."""
    real = np.real(y)
    scale = np.max(np.abs(y), axis=-1, keepdims=True)
    return np.maximum(real, np.maximum(rel * scale, np.finfo(float).tiny))

def mobility_at_probes(sweep, probes, force_amplitude):
    Omega = 2*np.pi*sweep.f_values            # (n_freq,)
    Y = np.empty((len(probes), len(Omega)), dtype=complex)
    for p, probe in enumerate(probes):
        w_at = sweep.w_field[:, probe.local_to_global_w] @ probe.phi   # (n_freq,)
        Y[p] = 1j*Omega*w_at/force_amplitude
    return Y


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

def global_insertion_loss_radiated_db(P_bare, P_vamm=1e-12):
    """IL of the radiated power at equal force: 10*log10(P_bare / P_vamm).
       Positive = the VAMMs reduce the radiated sound power."""
    return 10 * np.log10(np.maximum(P_bare, np.finfo(float).tiny) / P_vamm)

def global_insertion_loss_radiated_equal_power_db(
        P_bare, P_vamm, y_dp_bare, y_dp_vamm, force_amplitude=1.0):
    """Same, but both systems compared at equal *input* power: normalize each P_rad by its
    own input power first (P_rad / P_in is the fraction of the input that is radiated)."""
    frac_bare = P_bare / _floor_real_part(input_power(y_dp_bare, force_amplitude))
    frac_vamm = P_vamm / _floor_real_part(input_power(y_dp_vamm, force_amplitude))
    return 10 * np.log10(_floor_magnitude(frac_bare) / _floor_magnitude(frac_vamm))


def assert_comparable_sweeps(sweep_bare, sweep_vamm):
    if sweep_bare.f_values.shape != sweep_vamm.f_values.shape:
        raise ValueError(
            f"Probe shapes differ: {sweep_bare.f_values.shape} "
            f"vs {sweep_vamm.f_values.shape}.")

def _clip_to_band(f_values, y, f_lo, f_hi):
    """Validated, edge-interpolated arrays on [f_lo, f_hi] (same logic as band_integral)."""
    f = np.asarray(f_values, dtype=float)
    y = np.asarray(y, dtype=float)
    if f.shape != y.shape:
        raise ValueError(f"f_values {f.shape} and y {y.shape} must have the same shape")
    if not np.all(np.diff(f) > 0):
        raise ValueError("f_values must be strictly increasing")
    if not f_lo < f_hi:
        raise ValueError(f"need f_lo < f_hi, got {f_lo} and {f_hi}")
    if f_lo < f[0] or f_hi > f[-1]:
        raise ValueError(f"band [{f_lo}, {f_hi}] must lie inside [{f[0]}, {f[-1]}]")
    inside = (f > f_lo) & (f < f_hi)
    f_clip = np.concatenate(([f_lo], f[inside], [f_hi]))
    y_clip = np.concatenate(([np.interp(f_lo, f, y)], y[inside], [np.interp(f_hi, f, y)]))
    return f_clip, y_clip

def band_integral(f_values, y, f_lo, f_hi):
    f_clip, y_clip = _clip_to_band(f_values, y, f_lo, f_hi)
    return np.trapezoid(y_clip, f_clip)

def _band_il_db(f_values, y_bare, y_vamm, f_lo, f_hi):
    num = _floor_scalar(band_integral(f_values, y_bare, f_lo, f_hi))
    den = _floor_scalar(band_integral(f_values, y_vamm, f_lo, f_hi))
    return 10 * np.log10(num / den)

def band_insertion_loss_radiated_db(f_values, P_bare, P_vamm, f_lo, f_hi):
    return _band_il_db(f_values, P_bare, P_vamm, f_lo, f_hi)

def band_insertion_loss_structural_db(f_values, rms_bare, rms_vamm, f_lo, f_hi):
    return _band_il_db(f_values, np.asarray(rms_bare)**2, np.asarray(rms_vamm)**2, f_lo, f_hi)

def _insert_zero_crossings(f, d):
    """Add a node with d = 0 on every interval where d changes sign strictly."""
    s0, s1 = d[:-1], d[1:]
    cross = np.where(s0 * s1 < 0)[0]          # interval indices with a strict sign change
    if cross.size == 0:
        return f, d
    f_star = f[cross] + (f[cross + 1] - f[cross]) * s0[cross] / (s0[cross] - s1[cross])
    f_new = np.insert(f, cross + 1, f_star)
    d_new = np.insert(d, cross + 1, 0.0)
    return f_new, d_new

def penalized_power_reduction(
        f_values, P_bare, P_vamm, f_lo, f_hi, lam=2.0, normalize_full=False):
    """Sign-aware, power-weighted reduction of radiated power over [f_lo, f_hi].
    By default, normalization is against P_bare over [f_lo, f_hi]. If normalize_full is true,
    then normalization is against P_bare over f_values instead, meaning the contribution
    to the full-band reward is computed.

    Delta(f) = P_bare - P_vamm (positive where the VAMM helps, negative where it hurts).
    Additional frequencies inserted at zero-crossings to avoid overestimation of harm
    for lamb > 1.
    R_norm = (int max(Delta,0) df - lam * int max(-Delta,0) df) / int P_bare df
    (integrals over the band). For normalize_full=True, it is
    (int_band P_bare - int_band P_vamm) / int_all P_bare.

    lam = 1 gives 1 - int P_vamm / int P_bare (the linear form of the band IL).
    R_norm <= 1 always; for full normalization it is bounded by the band's share of the
    total; it is negative if the VAMM makes things worse on balance in both cases.
    (and more negative the larger lam is). Uses amplitude convention, flat 1 N spectrum.
    """
    if lam < 1:
        raise ValueError(f"lam must be >= 1, got {lam}")
    f = np.asarray(f_values, dtype=float)
    P_b = np.asarray(P_bare, dtype=float)
    P_v = np.asarray(P_vamm, dtype=float)
    if P_b.shape != P_v.shape:
        raise ValueError(f"P_bare {P_b.shape} and P_vamm {P_v.shape} must have the same shape")

    f_c, delta = _clip_to_band(f_values, P_b - P_v, f_lo, f_hi)  # delta interpolated at the edges
    f_all = np.asarray(f_values, dtype=float)
    if normalize_full:
        total = np.trapezoid(P_b, f_all)  # whole grid
    else:
        _, Pb_c = _clip_to_band(f_values, P_b, f_lo, f_hi)
        total = np.trapezoid(Pb_c, f_c)  # band only
    _, Pb_c = _clip_to_band(f_values, P_b, f_lo, f_hi)
    if total <= 0:
        raise ValueError("bare power integral is not positive; check the spectrum")

    f_r, d_r = _insert_zero_crossings(f_c, delta)
    gain = np.trapezoid(np.maximum(d_r, 0.0), f_r)
    harm = np.trapezoid(np.maximum(-d_r, 0.0), f_r)
    return (gain - lam * harm) / total

# --- Test/ validation area ---
from frequency_sweep import build_frequency_grid

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

    f_values = build_frequency_grid(f_start=100, f_end=300, gamma=0.02)
    size_frequencies = len(f_values)
    p_bare_test, p_vamm_test = 1e-3 * rng.uniform(0.5, 2, size_frequencies), 1e-3 * rng.uniform(0.5,2, size_frequencies)

    # --- Testing band insertion losses ---

    # Check identity property
    no_band_insertion_loss_radiated = band_insertion_loss_radiated_db(
        f_values, p_bare_test, p_bare_test, f_values[0], f_values[-1])
    assert np.isclose(no_band_insertion_loss_radiated, 0, rtol=1e-12, atol=0)

    # Check scaling property
    scaling_band_insertion_loss_radiated = band_insertion_loss_radiated_db(
        f_values, p_bare_test, 0.5*p_bare_test, f_values[0], f_values[-1])
    assert np.isclose(scaling_band_insertion_loss_radiated, 10*np.log10(2), rtol=1e-9, atol=0)

    # Check Lorentzian peak
    def lorentzian(f, f0, fwhm, amplitude=1.0, baseline=0.5):
        """Computes the Lorentzian function.

        Parameters:
        - f: float or array-like, independent variable
        - f0: float, center of the peak
        - fwhm: float, Full Width at Half Maximum
        - amplitude: float, peak height at f = f0 (default 1.0)
        - baseline: float, baseline offset (default 0.5)
        """
        return baseline + amplitude / (1.0 + ((f - f0) / (fwhm / 2)) ** 2)

    def uniform_reduction(frequency, reduction_factor=4, f_window_lo=None, f_window_hi=None):
        if f_window_lo and f_window_hi:
            if frequency < f_window_lo or frequency > f_window_hi:
                return 1
        return 1/(1 + reduction_factor)

    def non_uniform_reduction(frequency, f0, fwhm, reduction_factor=4, f_window_lo=None, f_window_hi=None):
        if f_window_lo and f_window_hi:
            if frequency < f_window_lo or frequency > f_window_hi:
                return 1
        return 1/(1 + reduction_factor*np.exp(-np.abs(frequency-f0)/fwhm))

    f0_test = 200
    fwhm_test = 10
    reduction_factor = 4
    lorentzian_p_bare_test = lorentzian(f_values, f0_test, fwhm_test)
    uniform_reduction_factors = [uniform_reduction(
        freq, reduction_factor=reduction_factor, f_window_lo=180, f_window_hi=220) for freq in f_values]

    lorentzian_p_vamm_test_uniform = [
        lorentzian_p_bare_test[i]*uniform_reduction_factors[i] for i in range(len(uniform_reduction_factors))]

    reduction_factors = [non_uniform_reduction(
        freq, f0_test, fwhm_test,
        reduction_factor=reduction_factor, f_window_lo=180, f_window_hi=220) for freq in f_values]

    lorentzian_p_vamm_test = [lorentzian_p_bare_test[i]*reduction_factors[i] for i in range(len(reduction_factors))]

    # Create finely meshed benchmark to compare the usual frequency grid against
    fine_frequency_grid = np.linspace(100, 300, 1000)
    fine_bare_test = lorentzian(fine_frequency_grid, f0_test, fwhm_test)
    fine_reduction_factors = [non_uniform_reduction(
        freq, f0_test, fwhm_test, reduction_factor=4, f_window_lo=180, f_window_hi=220) for freq in fine_frequency_grid]
    fine_vamm_test = [fine_bare_test[i]*fine_reduction_factors[i] for i in range(len(fine_reduction_factors))]

    # Stay inside reduction window
    lorentzian_insertion_loss_uniform = band_insertion_loss_radiated_db(
        f_values, lorentzian_p_bare_test, lorentzian_p_vamm_test_uniform, 181, 219
    )

    lorentzian_insertion_loss_normal_mesh = band_insertion_loss_radiated_db(
        f_values, lorentzian_p_bare_test, lorentzian_p_vamm_test, f_values[0], f_values[-1])

    lorentzian_insertion_loss_fine_mesh2 = band_insertion_loss_radiated_db(
        fine_frequency_grid, fine_bare_test, fine_vamm_test, fine_frequency_grid[0], fine_frequency_grid[-1])
    lorentzian_insertion_loss_fine_mesh = (
    10*np.log10(np.trapezoid(fine_bare_test, fine_frequency_grid)/np.trapezoid(fine_vamm_test, fine_frequency_grid)))

    assert np.isclose(lorentzian_insertion_loss_uniform, 10*np.log10(reduction_factor+1), rtol=1e-9), (
        f"uniform reduction loss is {lorentzian_insertion_loss_uniform:.9f},"
        f"should be {10*np.log10(reduction_factor+1):.9f}.")
    assert np.isclose(lorentzian_insertion_loss_normal_mesh, lorentzian_insertion_loss_fine_mesh, atol=0.01), (
        f"non-uniform reduction loss is {lorentzian_insertion_loss_normal_mesh}, is "
        f"{lorentzian_insertion_loss_fine_mesh}."
    )

    # --- penalized_power_reduction ---

    f = build_frequency_grid(f_start=100, f_end=300, gamma=0.02)

    def raises_value_error(fn, *args):
        try:
            fn(*args)
        except ValueError:
            return True
        return False

    Ps_b = 1.0 + 0.5 * np.sin(2 * np.pi * f / 150.0)
    Ps_v = Ps_b * (0.5 + 0.3 * np.exp(-0.5 * ((f - 200) / 30.0) ** 2))

    lo, hi = 120.0, 250.0

    # identical spectra -> 0
    assert abs(penalized_power_reduction(f, Ps_b, Ps_b, lo, hi, lam=3.0)) < 1e-12

    # pure reduction (P_vamm = 0.5 P_bare): R = 0.5 for any lam
    for lam in (1.0, 2.0, 5.0):
        assert np.isclose(penalized_power_reduction(f, Ps_b, 0.5 * Ps_b, lo, hi, lam), 0.5, rtol=1e-12)

    # pure harm (P_vamm = 2 P_bare): R = -lam
    for lam in (1.0, 2.0, 5.0):
        assert np.isclose(penalized_power_reduction(f, Ps_b, 2.0 * Ps_b, lo, hi, lam), -lam, rtol=1e-12)

    # lam = 1 must equal 1 - int P_vamm / int P_bare, on mixed-sign data (random sign changes)
    P_b_r = 1e-3 * rng.uniform(0.5, 2, len(f))
    P_v_r = 1e-3 * rng.uniform(0.5, 2, len(f))
    r1 = penalized_power_reduction(f, P_b_r, P_v_r, lo, hi, lam=1.0)
    ref = 1 - band_integral(f, P_v_r, lo, hi) / band_integral(f, P_b_r, lo, hi)
    assert np.isclose(r1, ref, rtol=1e-9, atol=1e-12), f"lam=1 identity failed: {r1} vs {ref}"

    # monotone in lam: more penalty never increases the reward, strictly decreases if any harm exists
    r2 = penalized_power_reduction(f, P_b_r, P_v_r, lo, hi, lam=2.0)
    r4 = penalized_power_reduction(f, P_b_r, P_v_r, lo, hi, lam=4.0)
    assert r1 > r2 > r4

    # upper bound: R_norm <= 1 (cannot save more than all of the bare power)
    assert penalized_power_reduction(f, Ps_b, 1e-9 * Ps_b, lo, hi, lam=2.0) <= 1.0

    # error paths
    assert raises_value_error(penalized_power_reduction, f, Ps_b, Ps_v, 250, 120)
    assert raises_value_error(penalized_power_reduction, f, Ps_b, Ps_v, lo, hi, 0.5)   # lam < 1
    assert raises_value_error(penalized_power_reduction, f, Ps_b, Ps_v[:-1], lo, hi)   # shape mismatch
    print("penalized_power_reduction tests passed")

    # --- Remainder of Task 4 ---
    def raises_value_error(fn, *args):
        try:
            fn(*args)
        except ValueError:
            return True
        return False

    f = f_values
    R = reduction_factor
    il_max = 10 * np.log10(R)
    rms_b = np.sqrt(p_bare_test)

    # Structural: identity and scaling (rms/sqrt(2) -> 3.0103 dB; a missing square would give 1.505)
    assert np.isclose(band_insertion_loss_structural_db(f, rms_b, rms_b, f[0], f[-1]), 0, atol=1e-12)

    # Scaling on several bands, including edges between grid points and a band with no interior points
    bands = [(f[0], f[-1]), (120.37, 180.91), (150.1, 150.4), (200.0, 299.99)]
    for lo, hi in bands:
        il_r = band_insertion_loss_radiated_db(f, p_bare_test, 0.5 * p_bare_test, lo, hi)
        il_s = band_insertion_loss_structural_db(f, rms_b, rms_b / np.sqrt(2), lo, hi)
        assert np.isclose(il_r, 10 * np.log10(2), rtol=1e-9), f"radiated scaling failed on {(lo, hi)}: {il_r}"
        assert np.isclose(il_s, 10 * np.log10(2), rtol=1e-9), f"structural scaling failed on {(lo, hi)}: {il_s}"

    # Uniform hard window: far band -> 0 dB, straddling band strictly in between
    Pb = lorentzian(f, 200, 10)
    Pv_u = np.where((f >= 180) & (f <= 220), Pb / R, Pb)
    il_far = band_insertion_loss_radiated_db(f, Pb, Pv_u, 250, 300)
    il_straddle = band_insertion_loss_radiated_db(f, Pb, Pv_u, 200, 240)
    assert abs(il_far) < 1e-12, f"far band should be 0 dB, got {il_far}"
    assert 0 < il_straddle < il_max, f"straddling band should lie in (0, {il_max:.3f}), got {il_straddle}"

    # Ordering test: smooth non-uniform reduction, ratio-of-integrals vs average-of-dB
    def make_spectra(freq, f0, hw, R_peak):
        p_bare = 1e-3 + 1.0 / (1 + ((freq - f0) / hw) ** 2)
        bump = np.exp(-0.5 * ((freq - f0) / hw) ** 2)
        return p_bare, p_bare / (1 + (R_peak - 1) * bump)

    f0_s, hw_s = 150.0, 0.5 * 0.02 * 150.0
    lo_s, hi_s = f0_s - 3 * hw_s, f0_s + 3 * hw_s
    Pb_s, Pv_s = make_spectra(f, f0_s, hw_s, 10)
    il_code = band_insertion_loss_radiated_db(f, Pb_s, Pv_s, lo_s, hi_s)
    f_fine = np.linspace(lo_s, hi_s, 200_001)
    pb_f, pv_f = make_spectra(f_fine, f0_s, hw_s, 10)
    il_ref = 10 * np.log10(np.trapezoid(pb_f, f_fine) / np.trapezoid(pv_f, f_fine))
    in_band = (f >= lo_s) & (f <= hi_s)
    il_mean_db = np.mean(10 * np.log10(Pb_s / Pv_s)[in_band])
    print(f"ordering test: code={il_code:.3f}, ref={il_ref:.3f}, mean-of-dB={il_mean_db:.3f}")
    assert abs(il_code - il_ref) < 0.05          # loosen/refine grid if this fails marginally
    assert il_ref - il_mean_db > 0.5             # set from the printed gap

    # Grid independence: smooth, broad spectrum, full grid vs every second point
    Ps_b = 1.0 + 0.5 * np.sin(2 * np.pi * f / 150.0)
    Ps_v = Ps_b * (0.5 + 0.3 * np.exp(-0.5 * ((f - 200) / 30.0) ** 2))
    il_full = band_insertion_loss_radiated_db(f, Ps_b, Ps_v, 120, 280)
    il_half = band_insertion_loss_radiated_db(f[::2], Ps_b[::2], Ps_v[::2], 120, 280)
    assert abs(il_full - il_half) < 0.05, f"grid dependence: {il_full:.4f} vs {il_half:.4f}"

    # Edge continuity: tiny shift of f_lo -> tiny change
    il_a = band_insertion_loss_radiated_db(f, Ps_b, Ps_v, 120.0, 250.0)
    il_b = band_insertion_loss_radiated_db(f, Ps_b, Ps_v, 120.0 + 1e-6, 250.0)
    assert abs(il_a - il_b) < 1e-6, f"band edge jump: {abs(il_a - il_b):.2e} dB"

    # Error paths
    assert raises_value_error(band_insertion_loss_radiated_db, f, Pb, Pv_u, 250, 120), "reversed band"
    assert raises_value_error(band_insertion_loss_radiated_db, f, Pb, Pv_u, 150, 150), "empty band"
    assert raises_value_error(band_insertion_loss_radiated_db, f, Pb, Pv_u, 50, 150), "band below grid"
    assert raises_value_error(band_insertion_loss_radiated_db, f, Pb, Pv_u, 150, 400), "band above grid"
    assert raises_value_error(band_insertion_loss_radiated_db, f[::-1], Pb, Pv_u, 150, 200), "unsorted f"
    assert raises_value_error(band_insertion_loss_radiated_db, f, Pb[:-1], Pv_u, 150, 200), "shape mismatch"
    print("Task 4 band-IL tests passed")

    # Test revised penalty function utilizing crossing insertion
    f_t = np.array([100.0, 130.0, 170.0, 200.0])
    Pb_t = np.full(4, 10.0)
    Pv_t = Pb_t - (f_t - 150.0)
    # gain = int_{150}^{200} (f-150) df = 1250 ; harm = int_{100}^{150} (150-f) df = 1250 ; total = 1000
    lam = 3.0
    expected = (1250 - lam * 1250) / 1000
    assert np.isclose(penalized_power_reduction(f_t, Pb_t, Pv_t, 100, 200, lam), expected, rtol=1e-12)

    # --- additivity of normalize_full ---
    edges = [f[0], 120.0, 147.3, 190.55, 250.0, f[-1]]     # edges deliberately between grid points
    for lam in (1.0, 2.0, 4.0):
        parts = [penalized_power_reduction(f, P_b_r, P_v_r, lo, hi, lam, normalize_full=True)
                 for lo, hi in zip(edges[:-1], edges[1:])]
        whole = penalized_power_reduction(f, P_b_r, P_v_r, f[0], f[-1], lam, normalize_full=True)
        assert np.isclose(sum(parts), whole, rtol=1e-9, atol=1e-12), (
            f"additivity failed for lam={lam}: {sum(parts)} vs {whole}")

    # full-band flag has no effect when the band is the whole grid
    assert np.isclose(
        penalized_power_reduction(f, P_b_r, P_v_r, f[0], f[-1], 2.0, normalize_full=True),
        penalized_power_reduction(f, P_b_r, P_v_r, f[0], f[-1], 2.0), rtol=1e-12)


def compute_insertion_loss(
        bare_probe_mobility, vamm_probe_mobility, sweep_bare, sweep_vamm,
        f0=None, f_lo=None, f_hi=None, gamma_plate=None):
    assert_comparable_sweeps(
        sweep_bare=sweep_bare, sweep_vamm=sweep_vamm)
    f = sweep_bare.f_values
    y_b, y_v = bare_probe_mobility, vamm_probe_mobility

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

    print("Can abort")

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

    deg = 2
    el_type = "S"

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

    shaker_force = shaker_config.force_amplitude


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
        grid=np.asarray(grid), gamma_plate=gamma_plate,
        free_plate=free_plate, excitation=excitation,
        deg=deg, el_type=el_type,
        code_version="v2")  # bump this string if you change the sweep code itself

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
                free_plate=free_plate, excitation=excitation, shaker_config=shaker_config,),
            settings_blob=blob)


    sweep_bare_c = coarse_sweep(None, "bare")
    sweep_light_c = coarse_sweep(vamm_list_light, "light")
    assert_comparable_sweeps(sweep_bare_c, sweep_light_c)

    bare_c_probe_mobility = mobility_at_probes(
        sweep=sweep_bare_c, probes=probes, force_amplitude=shaker_config.force_amplitude)

    light_c_probe_mobility = mobility_at_probes(
        sweep=sweep_light_c, probes=probes, force_amplitude=shaker_config.force_amplitude)

    il_pt = pointwise_insertion_loss_db(bare_c_probe_mobility, light_c_probe_mobility)
    dphi = phase_shift_deg(bare_c_probe_mobility, light_c_probe_mobility)
    il_f = global_insertion_loss_force_db(sweep_bare_c.rms_velocity, sweep_light_c.rms_velocity)
    il_p = global_insertion_loss_power_db(
        sweep_bare_c.rms_velocity, sweep_light_c.rms_velocity,
        bare_c_probe_mobility[0], light_c_probe_mobility[0])

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
        sweep=sweep_bare, probes=probes, force_amplitude=shaker_config.force_amplitude
    )

    for y_bare_sweep in bare_probe_mobility:
        for idx, y_bare in enumerate(y_bare_sweep):
            no_loss = pointwise_insertion_loss_db(y_bare, y_bare)
            no_phase = phase_shift_deg(y_bare, y_bare)
            assert np.isclose(no_loss, 0, rtol=1e-9),(
                f"insertion loss should be 0 for bare plate, found {no_loss} at f={sweep_bare.f_values[idx]}")
            assert np.isclose(no_phase, 0, rtol=1e-9), (
                f"Phase shift should be 0 for bare plate, found {no_phase} at f={sweep_bare.f_values[idx]}")


    vamm_probe_mobility = mobility_at_probes(
        sweep=sweep_vamm, probes=probes, force_amplitude=shaker_config.force_amplitude
    )

    y_vamm_test_nophase = 0.5 * bare_probe_mobility
    y_vamm_test_phase = 0.5 * np.exp(1j * np.pi/4) * bare_probe_mobility

    y_dp_bare = bare_probe_mobility[0]
    y_dp_vamm = vamm_probe_mobility[0]

    for idx in range(len(bare_probe_mobility)):
        probe_mobility = bare_probe_mobility[idx]
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
        bare_probe_mobility=bare_probe_mobility, vamm_probe_mobility=vamm_probe_mobility,
        sweep_bare=sweep_bare, sweep_vamm=sweep_vamm,
        f0=147.49, f_lo=133.49, f_hi=152.75, gamma_plate=gamma_plate)

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

    P_bare = radiated_power(
        Y_grid=bare_probe_mobility_radiation, r=r, dS=dS,
        f_values=sweep_bare.f_values, force_amplitude=shaker_force)

    P_vamm = radiated_power(
        Y_grid=vamm_probe_mobility_radiation, r=r, dS=dS,
        f_values=sweep_bare.f_values, force_amplitude=shaker_force)

    f_values = sweep_bare.f_values
    f_values2, P_bare2, P_vamm2 = f_values[::2], P_bare[::2], P_vamm[::2]
    rb, rv = sweep_bare.rms_velocity, sweep_vamm.rms_velocity
    rb2, rv2 = sweep_bare.rms_velocity[::2], sweep_vamm.rms_velocity[::2]

    bands = {"full": (100, 300), "147 mode": (140, 155),
             "hybrid": (130, 155), "low": (100, 110)}
    for name, (lo, hi) in bands.items():
        full_r = band_insertion_loss_radiated_db(f_values, P_bare, P_vamm, lo, hi)
        half_r = band_insertion_loss_radiated_db(f_values2, P_bare2, P_vamm2, lo, hi)
        full_s = band_insertion_loss_structural_db(f_values, rb, rv, lo, hi)
        half_s = band_insertion_loss_structural_db(f_values2, rb2, rv2, lo, hi)
        print(f"{name:9s} rad {full_r:7.3f} vs {half_r:7.3f} (d={half_r - full_r:+.3f})  "
              f"struct {full_s:7.3f} vs {half_s:7.3f} (d={half_s - full_s:+.3f})")

    f2a, Pb2a, Pv2a, rb2a, rv2a = f_values[::2], P_bare[::2], P_vamm[::2], rb[::2], rv[::2]
    f2b, Pb2b, Pv2b, rb2b, rv2b = f_values[1::2], P_bare[1::2], P_vamm[1::2], rb[1::2], rv[1::2]

    f0, f_lo_h, f_hi_h = 147.49, 133.49, 152.75
    fwhm = gamma_plate * f0                      # approx. bare resonance width (about 5.9 Hz)

    bands = {
        "full":       (f_values[0], f_values[-1]),
        "low":        (100, 110),
        "hybrid all": (130, 155),
        "147 mode":   (f0 - fwhm, f0 + fwhm),    # about 147.5 +- 5.9 Hz
        "hybrid lo":  (f_lo_h - fwhm / 2, f_lo_h + fwhm / 2),
        "hybrid hi":  (f_hi_h - fwhm / 2, f_hi_h + fwhm / 2),
        "narrow 147 mode": (f0 - fwhm/4, f0 + fwhm/4),  # about 147.5 +- 5.9 Hz
        "narrow hybrid lo": (f_lo_h - fwhm / 8, f_lo_h + fwhm / 8),
        "narrow hybrid hi": (f_hi_h - fwhm / 8, f_hi_h + fwhm / 8),
    }

    print(f"{'band':11s} {'IL_rad':>8s} {'IL_F':>8s} {'rad-F':>8s} | "
          f"{'max |d| rad':>11s} {'max |d| F':>10s}")
    for name, (lo, hi) in bands.items():
        r_full = band_insertion_loss_radiated_db(f_values, P_bare, P_vamm, lo, hi)
        s_full = band_insertion_loss_structural_db(f_values, rb, rv, lo, hi)

        d_rad, d_str = [], []
        for fx, pb, pv, rbx, rvx in ((f2a, Pb2a, Pv2a, rb2a, rv2a), (f2b, Pb2b, Pv2b, rb2b, rv2b)):
            lo_c, hi_c = max(lo, fx[0]), min(hi, fx[-1])
            # compare against the full grid on the SAME clipped band
            r_ref = band_insertion_loss_radiated_db(f_values, P_bare, P_vamm, lo_c, hi_c)
            s_ref = band_insertion_loss_structural_db(f_values, rb, rv, lo_c, hi_c)
            d_rad.append(band_insertion_loss_radiated_db(fx, pb, pv, lo_c, hi_c) - r_ref)
            d_str.append(band_insertion_loss_structural_db(fx, rbx, rvx, lo_c, hi_c) - s_ref)

        print(f"{name:11s} {r_full:8.3f} {s_full:8.3f} {r_full - s_full:8.3f} | "
              f"{max(map(abs, d_rad)):11.3f} {max(map(abs, d_str)):10.3f}")

    print(f"{'band':17s} {'R(l=1)':>8s} {'R(l=2)':>8s} {'R(l=4)':>8s} | {'max |d| R(l=2)':>14s}")
    for name, (lo, hi) in bands.items():
        vals = [penalized_power_reduction(f_values, P_bare, P_vamm, lo, hi, lam) for lam in (1, 2, 4)]
        diffs = []
        for fx, pb, pv in ((f2a, Pb2a, Pv2a), (f2b, Pb2b, Pv2b)):
            lo_c, hi_c = max(lo, fx[0]), min(hi, fx[-1])
            ref2 = penalized_power_reduction(f_values, P_bare, P_vamm, lo_c, hi_c, 2.0)
            diffs.append(abs(penalized_power_reduction(fx, pb, pv, lo_c, hi_c, 2.0) - ref2))
        print(f"{name:17s} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f} | {max(diffs):14.4f}")

    # --- How much of the bare radiated power does each band carry? ---
    total_bare = band_integral(f_values, P_bare, f_values[0], f_values[-1])
    total_vamm = band_integral(f_values, P_vamm, f_values[0], f_values[-1])
    print(f"{'band':17s} {'bare share':>10s} {'vamm share':>10s} {'band width [Hz]':>16s}")
    for name, (lo, hi) in bands.items():
        share_b = band_integral(f_values, P_bare, lo, hi) / total_bare
        share_v = band_integral(f_values, P_vamm, lo, hi) / total_bare   # relative to BARE total
        print(f"{name:17s} {share_b:10.4f} {share_v:10.4f} {hi - lo:16.1f}")

    # --- Check if this holds for full normalization as well ---
    print(f" Full normalization {'band':17s} {'R(l=1)':>8s} {'R(l=2)':>8s} {'R(l=4)':>8s} | {'max |d| R(l=2)':>14s}")
    for name, (lo, hi) in bands.items():
        vals = [penalized_power_reduction(
            f_values, P_bare, P_vamm, lo, hi, lam, normalize_full=True) for lam in (1, 2, 4)]
        diffs = []
        for fx, pb, pv in ((f2a, Pb2a, Pv2a), (f2b, Pb2b, Pv2b)):
            lo_c, hi_c = max(lo, fx[0]), min(hi, fx[-1])
            ref2 = penalized_power_reduction(f_values, P_bare, P_vamm, lo_c, hi_c, 2.0)
            diffs.append(abs(penalized_power_reduction(fx, pb, pv, lo_c, hi_c, 2.0) - ref2))
        print(f"{name:17s} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f} | {max(diffs):14.4f}")
