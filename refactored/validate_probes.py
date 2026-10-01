import numpy as np
from dolfinx import fem
from config import PlateConfig
from problem_setup import build_plate_problem
from weak_form import collapse_w_subspace
from probe import make_probe, evaluate_w_at_probe
from scipy.signal import find_peaks
from config import PlateConfig

def infinite_plate_mobility(plate_config):
    """Y_inf = 1/(8 sqrt(D * rho * h)) as a plain float, from the Lame parameters."""
    mu, lam, h, rho = plate_config.mu, plate_config.lambda_, plate_config.thickness, plate_config.rho
    E = mu * (3 * lam + 2 * mu) / (mu + lam)
    nu = lam / (2 * (lam + mu))
    D = E * h**3 / (12 * (1 - nu**2))
    return 1.0 / (8.0 * np.sqrt(D * rho * h))


def check_passivity(Y, name, tol_rel=1e-10):
    tol = tol_rel * np.max(np.abs(Y))
    assert (Y.real >= -tol).all(), f"{name}: Re(Y_dp) < 0 somewhere (min {Y.real.min():.3e})"
    max_phase = np.degrees(np.max(np.abs(np.angle(Y))))
    assert max_phase <= 90 + 1e-6, f"{name}: |phase| = {max_phase:.2f} deg > 90"
    print(f"{name}: passivity ok (max |phase| = {max_phase:.1f} deg)")


def check_peak_alignment(f, Y, eigenfrequencies, gamma, name):
    """Every |Y_dp| peak must sit near a known eigenfrequency. The reverse is NOT required:
    modes with a node at the shaker are legitimately invisible."""
    peaks, _ = find_peaks(20 * np.log10(np.abs(Y)))
    eig = np.asarray(eigenfrequencies)
    unmatched = []
    for idx in peaks:
        f_peak = f[idx]
        tol = max(0.5 * gamma * f_peak, 2 * np.max(np.diff(f)[max(idx - 1, 0):idx + 1]))
        if np.min(np.abs(eig - f_peak)) > tol:
            unmatched.append(round(f_peak, 2))
    assert not unmatched, f"{name}: peaks without a nearby eigenfrequency: {unmatched}"
    invisible = [round(e, 2) for e in eig if f[0] <= e <= f[-1]
                 and np.min(np.abs(f[peaks] - e)) > 0.5 * gamma * e]
    print(f"{name}: {len(peaks)} peaks, all matched. Eigenfrequencies without a peak "
          f"(expected if nodal at shaker): {invisible}")


def check_infinite_plate_level(f, Y_bare, y_inf):
    """Band mean of Re(Y) vs infinite-plate value. Uses a trapezoid integral over f, not
    a plain mean over points, since the geometric grid oversamples low frequencies."""
    mean_re = np.trapezoid(Y_bare.real, f) / (f[-1] - f[0])
    delta_db = 20 * np.log10(mean_re / y_inf)
    print(f"band-mean Re(Y_bare) = {mean_re:.3e}, Y_inf = {y_inf:.3e}, difference = {delta_db:+.1f} dB")
    return delta_db


if __name__ == "__main__":
    plate_config = PlateConfig(length=1.1, width=1, thickness=0.02,
                               nx=60, ny=50, rho=7850, mu=77e9, lambda_=115e9)
    domain, function_space, _, _ = build_plate_problem(plate_config)

    # 1. known scalar field on the collapsed w-space
    V_w, w_to_parent = collapse_w_subspace(function_space)
    w_exact = fem.Function(V_w)
    w_exact.interpolate(lambda x: x[0] + 2 * x[1])

    # 2. embed it into the mixed space (theta stays zero)
    u = fem.Function(function_space)
    u.x.array[w_to_parent] = w_exact.x.array

    # 3. extract exactly as solve_linear_system does
    w_point = u.sub(0).collapse()

    # 4. probe points
    hx, hy = plate_config.length / plate_config.nx, plate_config.width / plate_config.ny
    test_points = {
        "interior":        (0.371, 0.613),
        "mesh node":       (10 * hx, 10 * hy),
        "cell edge mid":   (10.5 * hx, 10 * hy),
        "boundary edge":   (0.4, 0.0),
        "corner":          (plate_config.length, 0.0),   # your manual VAMM list uses this one
    }

    for name, (x, y) in test_points.items():
        probe = make_probe(domain, function_space, x, y)
        value = evaluate_w_at_probe(w_point, probe)
        expected = x + 2 * y
        assert abs(value - expected) < 1e-10, f"{name}: got {value}, expected {expected}"
        assert abs(value.imag) < 1e-12, f"{name}: unexpected imaginary part {value.imag}"
        print(f"{name:15s} ok, error = {abs(value - expected):.2e}")


    d = np.load("sweep_dp_100to300.npz")
    f, gamma = d["f"], float(d["gamma"])

    check_passivity(d["Y_bare"], "bare")
    check_passivity(d["Y_vamm"], "vamm")
    check_peak_alignment(f, d["Y_bare"], d["eig_bare"], gamma, "bare")
    check_peak_alignment(f, d["Y_vamm"], d["eig_vamm"], gamma, "vamm")

    y_inf = infinite_plate_mobility(plate_config)
    delta_db = check_infinite_plate_level(f, d["Y_bare"], y_inf)
    assert abs(delta_db) < 6, f"level off the infinite-plate value by {delta_db:.1f} dB: check conventions"
    print("driving-point validation passed")