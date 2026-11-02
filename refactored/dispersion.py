import time
import numpy as np
from dataclasses import dataclass
from scipy.linalg import eigh
from periodic_bc import build_pairs
from bloch_solver import assemble_unit_cell_matrices, reduce_to_bloch, solve_bloch_evp

# Decision of 2026-11-02: 0.1 m cell, h = 5 mm steel, f_max = 5 kHz. Max bare-cell error over all bands below
# 5 kHz: 80/m 0.72 %, 100/m 0.33 %, 120/m 0.17 %. 100/m keeps a factor 3 margin to the 1 % limit (see CLAUDE.md).
PRODUCTION_ELEMENTS_PER_METRE = 100


# --- Unit cell mesh ---

def unit_cell_element_counts(length_x, length_y, elements_per_metre=PRODUCTION_ELEMENTS_PER_METRE):
    """Element counts (nx, ny) giving about elements_per_metre in both directions."""
    return max(1, round(elements_per_metre * length_x)), max(1, round(elements_per_metre * length_y))


# --- Wavevector path through the irreducible Brillouin zone ---

@dataclass
class WavevectorPath:
    wavevectors: np.ndarray                # shape (n_k, 2), rad/m
    path_coordinate: np.ndarray            # shape (n_k,), distance travelled in k-space, rad/m (plot x-axis)
    symmetry_point_positions: np.ndarray   # path_coordinate of Gamma, X, M, (Y,) Gamma
    symmetry_point_labels: list


def build_wavevector_path(length_x, length_y, points_per_segment=40):
    """Gamma-X-M-Gamma for a square cell, Gamma-X-M-Y-Gamma for a rectangular one.

    points_per_segment: number of intervals on Gamma-X; every other segment gets the same spacing in k,
        so longer segments get proportionally more points. Corner points appear exactly once."""
    gamma, x_point = np.array([0.0, 0.0]), np.array([np.pi / length_x, 0.0])
    m_point, y_point = np.array([np.pi / length_x, np.pi / length_y]), np.array([0.0, np.pi / length_y])
    is_square = np.isclose(length_x, length_y, rtol=1e-12)
    if is_square:
        corners, labels = [gamma, x_point, m_point, gamma], ["Γ", "X", "M", "Γ"]
    else:
        corners, labels = [gamma, x_point, m_point, y_point, gamma], ["Γ", "X", "M", "Y", "Γ"]

    spacing = np.linalg.norm(x_point - gamma) / points_per_segment
    wavevectors, symmetry_point_positions = [], [0.0]
    for start, end in zip(corners[:-1], corners[1:]):
        segment_length = np.linalg.norm(end - start)
        number_of_intervals = max(1, round(segment_length / spacing))
        fractions = np.arange(number_of_intervals) / number_of_intervals     # end point belongs to next segment
        wavevectors.extend(start + fraction * (end - start) for fraction in fractions)
        symmetry_point_positions.append(symmetry_point_positions[-1] + segment_length)
    wavevectors.append(corners[-1])
    wavevectors = np.array(wavevectors)

    steps = np.linalg.norm(np.diff(wavevectors, axis=0), axis=1)
    path_coordinate = np.concatenate([[0.0], np.cumsum(steps)])
    return WavevectorPath(wavevectors=wavevectors, path_coordinate=path_coordinate,
                          symmetry_point_positions=np.array(symmetry_point_positions),
                          symmetry_point_labels=labels)


# --- k-sweep over the path ---

@dataclass
class DispersionResult:
    path: WavevectorPath
    frequencies: np.ndarray                # shape (n_k, n_bands), Hz, ascending per k (no branch tracking);
                                           # NaN where a k has fewer bands below maximum_frequency_hz
    eigenvectors: np.ndarray | None        # shape (n_k, n_independent_dofs, n_bands), complex, NaN-padded, or None
    maximum_frequency_hz: float | None


def estimate_band_count(plate_config, maximum_frequency_hz):
    """Folded bands below maximum_frequency_hz: number of G vectors inside the circle |k + G| <= kappa_max,
    about kappa_max^2 * Lx * Ly / (4 pi). kappa_max from the thin-plate (Kirchhoff) relation; the
    Reissner-Mindlin plate is slightly softer, so the true count can be a little higher."""
    constants = plate_constants(plate_config)
    angular_frequency = 2 * np.pi * maximum_frequency_hz
    wavenumber_squared = angular_frequency * np.sqrt(constants["mass_per_area"] / constants["bending_rigidity"])
    return int(np.ceil(wavenumber_squared * plate_config.length * plate_config.width / (4 * np.pi)))


def pad_with_nan(rows, width):
    padded = np.full((len(rows), width) + rows[0].shape[1:], np.nan, dtype=rows[0].dtype)
    for row_index, row in enumerate(rows):
        padded[row_index, :len(row)] = row
    return padded


def dispersion_sweep(function_space, problem, plate_config, wavevector_path, maximum_frequency_hz=None,
                     number_of_bands=None, keep_eigenvectors=False, factorization_package="petsc"):
    """Solve the Bloch eigenproblem at every wavevector of the path for the bare unit cell.

    Give exactly one of
      maximum_frequency_hz: all bands below this frequency. The band count starts from estimate_band_count
          and grows at each k until the highest computed band lies above maximum_frequency_hz, so no band
          below the limit is missed. The count can differ between wavevectors (NaN padding).
      number_of_bands: a fixed number of lowest bands at every k.

    The cell is plate_config.length x plate_config.width; function_space must be Lagrange (nodal DOFs for
    the periodic pairing) and the run needs fenicsx-env-complex. Everything k-independent (K, M, DOF pairs)
    is built once before the loop. keep_eigenvectors stores the reduced eigenvectors for branch tracking."""
    if (maximum_frequency_hz is None) == (number_of_bands is None):
        raise ValueError("give exactly one of maximum_frequency_hz and number_of_bands")
    length_x, length_y = plate_config.length, plate_config.width
    stiffness, mass = assemble_unit_cell_matrices(function_space, problem)
    pairs = build_pairs(function_space, length_x, length_y)

    bands_to_request = number_of_bands if number_of_bands is not None \
        else estimate_band_count(plate_config, maximum_frequency_hz) + 2
    number_of_wavevectors = len(wavevector_path.wavevectors)
    print(f"Dispersion sweep: {number_of_wavevectors} wavevectors, "
          + (f"bands below {maximum_frequency_hz:.0f} Hz" if maximum_frequency_hz else f"{number_of_bands} bands")
          + f", {len(pairs['independent'])} independent DOFs")

    frequency_rows, eigenvector_rows = [], []
    sweep_start = time.perf_counter()
    for step, (wavenumber_x, wavenumber_y) in enumerate(wavevector_path.wavevectors):
        step_start = time.perf_counter()
        stiffness_reduced, mass_reduced = reduce_to_bloch(stiffness, mass, pairs, wavenumber_x, wavenumber_y,
                                                          length_x, length_y)
        while True:
            band_frequencies, band_vectors = solve_bloch_evp(stiffness_reduced, mass_reduced,
                                                             number_of_bands=bands_to_request,
                                                             factorization_package=factorization_package)
            if maximum_frequency_hz is None or band_frequencies[-1] > maximum_frequency_hz:
                break
            bands_to_request += 2           # kept for the following wavevectors as well
        if maximum_frequency_hz is not None:
            below_limit = band_frequencies <= maximum_frequency_hz
            band_frequencies, band_vectors = band_frequencies[below_limit], band_vectors[:, below_limit]
        frequency_rows.append(band_frequencies)
        if keep_eigenvectors:
            eigenvector_rows.append(band_vectors.T)       # (n_bands, n_dofs), transposed back after padding
        if step == 0:
            first_step_seconds = time.perf_counter() - step_start
            print(f"First wavevector took {first_step_seconds:.2f} s; estimated total about "
                  f"{number_of_wavevectors * first_step_seconds / 60:.1f} min")
    print(f"Dispersion sweep finished in {(time.perf_counter() - sweep_start) / 60:.1f} min")

    widest = max(len(row) for row in frequency_rows)
    frequencies = pad_with_nan(frequency_rows, widest)
    eigenvectors = pad_with_nan(eigenvector_rows, widest).transpose(0, 2, 1) if keep_eigenvectors else None
    return DispersionResult(path=wavevector_path, frequencies=frequencies, eigenvectors=eigenvectors,
                            maximum_frequency_hz=maximum_frequency_hz)


# --- Analytic reference: infinite Reissner-Mindlin plate folded into the cell ---

def plate_constants(plate_config):
    """Same definitions as weak_form.reissner_mindlin_constants, evaluated as plain floats."""
    shear_modulus, lame_lambda = plate_config.mu, plate_config.lambda_
    poisson_ratio = lame_lambda / (2 * (lame_lambda + shear_modulus))
    youngs_modulus = shear_modulus * (3 * lame_lambda + 2 * shear_modulus) / (shear_modulus + lame_lambda)
    thickness = plate_config.thickness
    bending_rigidity = youngs_modulus * thickness ** 3 / (12 * (1 - poisson_ratio ** 2))
    shear_stiffness = shear_modulus * thickness * 5.0 / 6.0
    return dict(bending_rigidity=bending_rigidity, shear_stiffness=shear_stiffness, poisson_ratio=poisson_ratio,
                mass_per_area=plate_config.rho * thickness,
                rotary_inertia=plate_config.rho * thickness ** 3 / 12)


def plane_wave_omega_squared(wavenumber_magnitude, constants):
    """All three Reissner-Mindlin branches for a plane wave exp(i kappa e.x) in an infinite plate.

    Rotation split into theta_parallel (along e) and theta_perpendicular. theta_parallel couples to w:
        K = [[F k^2, -F k], [-F k, D k^2 + F]],  M = diag(rho h, rho h^3/12)   (theta_parallel = i * Phi)
    theta_perpendicular is decoupled: omega^2 = (D (1 - nu)/2 k^2 + F) / (rho h^3 / 12)."""
    bending, shear = constants["bending_rigidity"], constants["shear_stiffness"]
    stiffness = np.array([[shear * wavenumber_magnitude ** 2, -shear * wavenumber_magnitude],
                          [-shear * wavenumber_magnitude, bending * wavenumber_magnitude ** 2 + shear]])
    mass = np.diag([constants["mass_per_area"], constants["rotary_inertia"]])
    coupled_branches = eigh(stiffness, mass, eigvals_only=True)
    twist_branch = (bending * (1 - constants["poisson_ratio"]) / 2 * wavenumber_magnitude ** 2 + shear) \
        / constants["rotary_inertia"]
    return np.concatenate([coupled_branches, [twist_branch]])


def analytic_folded_frequencies(wavenumber_x, wavenumber_y, length_x, length_y, constants,
                                number_of_bands, harmonic_range=6):
    """Lowest bands of the infinite plate folded into the cell: kappa = k + G, G = 2 pi (m / Lx, n / Ly)."""
    omega_squared = []
    for m in range(-harmonic_range, harmonic_range + 1):
        for n in range(-harmonic_range, harmonic_range + 1):
            folded_magnitude = np.hypot(wavenumber_x + 2 * np.pi * m / length_x,
                                        wavenumber_y + 2 * np.pi * n / length_y)
            omega_squared.extend(plane_wave_omega_squared(folded_magnitude, constants))
    omega_squared = np.sort(np.clip(omega_squared, 0.0, None))[:number_of_bands]
    return np.sqrt(omega_squared) / (2 * np.pi)


def analytic_dispersion(wavevector_path, plate_config, maximum_frequency_hz=None, number_of_bands=None):
    """Analytic folded bands of the bare cell along the path, Hz, shape (n_k, n_bands).
    Same band selection as dispersion_sweep: either all bands below maximum_frequency_hz (NaN-padded)
    or a fixed number_of_bands."""
    if (maximum_frequency_hz is None) == (number_of_bands is None):
        raise ValueError("give exactly one of maximum_frequency_hz and number_of_bands")
    constants = plate_constants(plate_config)
    bands_to_compute = number_of_bands if number_of_bands is not None \
        else 2 * estimate_band_count(plate_config, maximum_frequency_hz) + 8
    harmonic_range = 6 + int(np.sqrt(bands_to_compute))
    rows = []
    for wavenumber_x, wavenumber_y in wavevector_path.wavevectors:
        band_frequencies = analytic_folded_frequencies(wavenumber_x, wavenumber_y, plate_config.length,
                                                       plate_config.width, constants, bands_to_compute,
                                                       harmonic_range=harmonic_range)
        if maximum_frequency_hz is not None:
            assert band_frequencies[-1] > maximum_frequency_hz, "computed too few analytic bands"
            band_frequencies = band_frequencies[band_frequencies <= maximum_frequency_hz]
        rows.append(band_frequencies)
    return pad_with_nan(rows, max(len(row) for row in rows))


def bare_cell_relative_error(result, analytic_frequencies, frequency_margin=0.05, zero_frequency_hz=1e-2):
    """Max relative FE-vs-analytic error over all bands, the check to run for every new configuration
    (cell size, thickness, material, maximum frequency, mesh density).

    analytic_frequencies must reach frequency_margin above the sweep's limit, e.g.
    analytic_dispersion(path, plate_config, maximum_frequency_hz=1.05 * maximum_frequency_hz): bands close to
    the limit can sit just above it in the (slightly stiffer) FE and just below it analytically, or the other
    way round, and a 4-fold group can do this all at once. Per k, every FE band is compared with the analytic
    band of the same sorted index, and every analytic band below (1 - frequency_margin) * limit must have an
    FE partner, so no band is silently missing. Zero modes are compared in absolute terms."""
    limit = result.maximum_frequency_hz
    worst_relative_error = 0.0
    for fe_row, analytic_row in zip(result.frequencies, analytic_frequencies):
        fe_row, analytic_row = fe_row[~np.isnan(fe_row)], analytic_row[~np.isnan(analytic_row)]
        assert len(analytic_row) >= len(fe_row), "analytic bands do not reach far enough above the limit"
        if limit is not None:
            assert len(fe_row) >= np.sum(analytic_row < (1 - frequency_margin) * limit), "FE band missing"
        analytic_row = analytic_row[:len(fe_row)]
        nonzero = analytic_row > zero_frequency_hz
        assert np.all(np.abs(fe_row - analytic_row)[~nonzero] < zero_frequency_hz), "zero modes differ"
        if nonzero.any():
            relative_error = np.abs(fe_row - analytic_row)[nonzero] / analytic_row[nonzero]
            worst_relative_error = max(worst_relative_error, relative_error.max())
    return worst_relative_error
