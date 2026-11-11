import time
import numpy as np
from dataclasses import dataclass
from scipy.linalg import eigh
from periodic_bc import build_pairs
from bloch_solver import assemble_unit_cell_matrices, reduce_to_bloch, solve_bloch_evp
from band_tracking import band_overlap, resonator_energy_share, track_bands as track_bands_by_overlap

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
    resonator_share: np.ndarray | None = None   # (n_k, n_bands) resonator energy share of the sorted bands (track_bands)
    tracking: object = None                     # band_tracking.TrackedBands (track_bands=True), else None


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
                     number_of_bands=None, vamm_list=None, keep_eigenvectors=False, factorization_package="petsc",
                     track_bands=False, minimum_match_quality=0.5):
    """Solve the Bloch eigenproblem at every wavevector of the path for the unit cell, bare or with resonators.

    Give exactly one of
      maximum_frequency_hz: all bands below this frequency. The band count starts from estimate_band_count
          and grows at each k until the highest computed band lies above maximum_frequency_hz, so no band
          below the limit is missed. The count can differ between wavevectors (NaN padding).
      number_of_bands: a fixed number of lowest bands at every k.

    The cell is plate_config.length x plate_config.width; function_space must be Lagrange (nodal DOFs for
    the periodic pairing) and the run needs fenicsx-env-complex. Everything k-independent (K, M, DOF pairs)
    is built once before the loop. keep_eigenvectors stores the reduced eigenvectors for branch tracking;
    with resonators, the last len(vamm_list) entries of each eigenvector are the resonator displacements.

    vamm_list: optional resonators in the cell (cell coordinates, registered indices); undamped here.

    track_bands: also follow each branch by mass-weighted eigenvector overlap (band_tracking.track_bands) and store
        the result in result.tracking, plus the resonator energy share of every band in result.resonator_share.
        result.frequencies stays sorted per k (use it for gap detection). Track with maximum_frequency_hz a bit
        above the plotted range, so branches entering or leaving at the cut-off do so outside the plot."""
    if (maximum_frequency_hz is None) == (number_of_bands is None):
        raise ValueError("give exactly one of maximum_frequency_hz and number_of_bands")
    length_x, length_y = plate_config.length, plate_config.width
    number_of_resonators = len(vamm_list) if vamm_list else 0
    stiffness, mass = assemble_unit_cell_matrices(function_space, problem, vamm_list=vamm_list)
    pairs = build_pairs(function_space, length_x, length_y)

    bands_to_request = number_of_bands if number_of_bands is not None \
        else estimate_band_count(plate_config, maximum_frequency_hz) + number_of_resonators + 2
    number_of_wavevectors = len(wavevector_path.wavevectors)
    print(f"Dispersion sweep: {number_of_wavevectors} wavevectors, "
          + (f"bands below {maximum_frequency_hz:.0f} Hz" if maximum_frequency_hz else f"{number_of_bands} bands")
          + f", {len(pairs['independent']) + number_of_resonators} independent DOFs"
          + (f" ({number_of_resonators} resonator)" if number_of_resonators else ""))

    frequency_rows, eigenvector_rows = [], []
    overlaps, bridge_overlaps, resonator_share_rows = [], [], []
    previous_vectors = vectors_two_back = None
    sweep_start = time.perf_counter()
    for step, (wavenumber_x, wavenumber_y) in enumerate(wavevector_path.wavevectors):
        step_start = time.perf_counter()
        stiffness_reduced, mass_reduced = reduce_to_bloch(stiffness, mass, pairs, wavenumber_x, wavenumber_y,
                                                          length_x, length_y,
                                                          number_of_extra_dofs=number_of_resonators)
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
        if track_bands:
            if previous_vectors is not None:
                overlaps.append(band_overlap(previous_vectors, band_vectors, mass_reduced))
            if vectors_two_back is not None:
                bridge_overlaps.append(band_overlap(vectors_two_back, band_vectors, mass_reduced))
            resonator_share_rows.append(resonator_energy_share(band_vectors, mass_reduced, number_of_resonators))
            vectors_two_back, previous_vectors = previous_vectors, band_vectors
        if step == 0:
            first_step_seconds = time.perf_counter() - step_start
            print(f"First wavevector took {first_step_seconds:.2f} s; estimated total about "
                  f"{number_of_wavevectors * first_step_seconds / 60:.1f} min")
    print(f"Dispersion sweep finished in {(time.perf_counter() - sweep_start) / 60:.1f} min")

    widest = max(len(row) for row in frequency_rows)
    frequencies = pad_with_nan(frequency_rows, widest)
    eigenvectors = pad_with_nan(eigenvector_rows, widest).transpose(0, 2, 1) if keep_eigenvectors else None
    resonator_share = tracking = None
    if track_bands:
        resonator_share = pad_with_nan(resonator_share_rows, widest)
        tracking = track_bands_by_overlap(frequencies, overlaps, bridge_overlaps=bridge_overlaps,
                                          resonator_share=resonator_share, minimum_quality=minimum_match_quality)
    return DispersionResult(path=wavevector_path, frequencies=frequencies, eigenvectors=eigenvectors,
                            maximum_frequency_hz=maximum_frequency_hz, resonator_share=resonator_share,
                            tracking=tracking)


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


# --- Analytic reference with resonators: plane-wave expansion (PWE), one point resonator per cell ---

def resonator_force_factor(resonator_stiffness, resonator_mass, angular_frequency):
    """Z(omega) = k m omega^2 / (k - m omega^2): force on the plate per unit plate deflection at the attachment
    (= dynamic effective mass m_eff(omega) times omega^2). Derivation: resonator -m w^2 u = k (w0 - u)."""
    return resonator_stiffness * resonator_mass * angular_frequency ** 2 \
        / (resonator_stiffness - resonator_mass * angular_frequency ** 2)


def plane_wave_point_compliance(wavenumber_magnitude, angular_frequency, constants, theory):
    """w response per unit point force for one plane wave (before dividing by the cell area).
    kirchhoff: 1 / (D k^4 - rho h w^2). reissner_mindlin: w-entry of (K(k) - w^2 M)^-1 for the coupled
    (w, theta_parallel) 2x2 system of plane_wave_omega_squared; the twist branch does not couple to a point force."""
    bending, shear = constants["bending_rigidity"], constants["shear_stiffness"]
    if theory == "kirchhoff":
        return 1.0 / (bending * wavenumber_magnitude ** 4 - constants["mass_per_area"] * angular_frequency ** 2)
    dynamic_ww = shear * wavenumber_magnitude ** 2 - constants["mass_per_area"] * angular_frequency ** 2
    dynamic_rotation = bending * wavenumber_magnitude ** 2 + shear - constants["rotary_inertia"] * angular_frequency ** 2
    return dynamic_rotation / (dynamic_ww * dynamic_rotation - (shear * wavenumber_magnitude) ** 2)


def resonator_dispersion_function(angular_frequency, wavenumber_x, wavenumber_y, length_x, length_y, constants,
                                  resonator_stiffness, resonator_mass, theory, harmonic_cutoff):
    """h(omega) = 1 - Z(omega)/A * sum_G g(|k + G|, omega); its roots are the Bloch frequencies of a plate with
    one point resonator per cell. For Reissner-Mindlin the sum diverges like log(cutoff) (point force on a
    shear-deformable plate); harmonic_cutoff (max |m|, |n|) then sets the effective attachment size."""
    harmonics = np.arange(-harmonic_cutoff, harmonic_cutoff + 1)
    folded_x = wavenumber_x + 2 * np.pi * harmonics[:, None] / length_x
    folded_y = wavenumber_y + 2 * np.pi * harmonics[None, :] / length_y
    with np.errstate(divide="ignore"):          # exactly on a pole the term is infinite; h is then not a root
        compliance_sum = plane_wave_point_compliance(np.hypot(folded_x, folded_y), angular_frequency, constants,
                                                     theory).sum()
    cell_area = length_x * length_y
    return 1.0 - resonator_force_factor(resonator_stiffness, resonator_mass, angular_frequency) \
        / cell_area * compliance_sum


def analytic_resonator_frequencies(wavenumber_x, wavenumber_y, plate_config, resonator_stiffness, resonator_mass,
                                   maximum_frequency_hz, theory="reissner_mindlin", harmonic_cutoff=12,
                                   points_per_interval=200):
    """Bloch frequencies (Hz, sorted) below maximum_frequency_hz of the plate with one point resonator per cell.

    Roots of resonator_dispersion_function lie between its poles (bare folded frequencies and the resonator
    frequency); each interval is scanned and sign changes with a small |h| are refined with brentq.
    Degenerate bare frequencies: a point resonator couples to one combination of the degenerate plane waves;
    the other (multiplicity - 1) combinations vanish at the attachment and keep the bare frequency, so they
    are added explicitly."""
    from scipy.optimize import brentq
    constants = plate_constants(plate_config)
    length_x, length_y = plate_config.length, plate_config.width
    harmonics = np.arange(-harmonic_cutoff, harmonic_cutoff + 1)
    folded = np.hypot(wavenumber_x + 2 * np.pi * harmonics[:, None] / length_x,
                      wavenumber_y + 2 * np.pi * harmonics[None, :] / length_y).ravel()
    bare_frequencies = np.array(sorted(np.sqrt(max(plane_wave_omega_squared(magnitude, constants)[0], 0.0))
                                       / (2 * np.pi) for magnitude in folded))
    bare_frequencies = bare_frequencies[bare_frequencies < 1.2 * maximum_frequency_hz]
    resonator_frequency = np.sqrt(resonator_stiffness / resonator_mass) / (2 * np.pi)

    distinct_poles, multiplicities = [], []
    for frequency in bare_frequencies:
        if distinct_poles and np.isclose(frequency, distinct_poles[-1], rtol=1e-9, atol=1e-9):
            multiplicities[-1] += 1
        else:
            distinct_poles.append(frequency)
            multiplicities.append(1)
    poles = np.sort(np.concatenate([distinct_poles, [resonator_frequency]]))

    def dispersion_function(frequency_hz):
        return resonator_dispersion_function(2 * np.pi * frequency_hz, wavenumber_x, wavenumber_y, length_x,
                                             length_y, constants, resonator_stiffness, resonator_mass, theory,
                                             harmonic_cutoff)

    roots = []
    interval_edges = np.concatenate([[0.0], poles, [1.2 * maximum_frequency_hz]])
    for lower, upper in zip(interval_edges[:-1], interval_edges[1:]):
        if upper - lower < 1e-9:
            continue
        span = upper - lower
        grid = np.linspace(lower + 1e-9 * span, upper - 1e-9 * span, points_per_interval)
        values = np.array([dispersion_function(frequency) for frequency in grid])
        for index in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
            root = brentq(dispersion_function, grid[index], grid[index + 1], xtol=1e-10)
            if abs(dispersion_function(root)) < 1e-6:            # a root, not a jump across a pole
                roots.append(root)
    unchanged = [pole for pole, multiplicity in zip(distinct_poles, multiplicities) for _ in range(multiplicity - 1)]
    if distinct_poles[0] < 1e-9:
        # Gamma: plate and resonator translating together (w = u = const) is a zero mode; h(omega) was divided
        # by omega^2 in the derivation, so this root has to be added by hand
        unchanged.append(0.0)
    frequencies = np.sort(np.concatenate([roots, unchanged]))
    return frequencies[frequencies <= maximum_frequency_hz]


def analytic_resonator_dispersion(wavevector_path, plate_config, resonator_stiffness, resonator_mass,
                                  maximum_frequency_hz, theory="reissner_mindlin", harmonic_cutoff=24):
    """analytic_resonator_frequencies along the whole path, NaN-padded, shape (n_k, n_bands), Hz."""
    rows = [analytic_resonator_frequencies(wavenumber_x, wavenumber_y, plate_config, resonator_stiffness,
                                           resonator_mass, maximum_frequency_hz, theory=theory,
                                           harmonic_cutoff=harmonic_cutoff)
            for wavenumber_x, wavenumber_y in wavevector_path.wavevectors]
    return pad_with_nan(rows, max(len(row) for row in rows))


# --- Complete band gaps (on the frequency-sorted bands; never on tracked branches) ---

@dataclass
class BandGap:
    lower_band_index: int                 # column of the sorted frequencies under the gap (0-based); the gap lies
                                          # between sorted bands lower_band_index and lower_band_index + 1
    lower_edge_hz: float                  # maximum of the lower band over all wavevectors
    upper_edge_hz: float                  # minimum of the upper band over all wavevectors
    lower_edge_wavevector: np.ndarray     # (kx, ky) in rad/m where the lower edge sits
    upper_edge_wavevector: np.ndarray
    lower_edge_location: str              # symmetry point ("M") or path segment ("X–M") of the lower edge
    upper_edge_location: str

    @property
    def centre_hz(self):
        return 0.5 * (self.lower_edge_hz + self.upper_edge_hz)

    @property
    def relative_width(self):
        """Gap width over gap centre (the usual figure of merit, e.g. 0.147 for 1031-1195 Hz)."""
        return (self.upper_edge_hz - self.lower_edge_hz) / self.centre_hz


def wavevector_location(wavevector_path, wavevector_index, relative_tolerance=1e-9):
    """Readable location of a wavevector on the path: its symmetry-point label if it sits on one, otherwise the
    segment ("X–M"). For a point set without symmetry points (zone-interior grid) the wavevector itself."""
    positions = np.asarray(wavevector_path.symmetry_point_positions, dtype=float)
    labels = wavevector_path.symmetry_point_labels
    if len(positions) == 0:
        wavenumber_x, wavenumber_y = wavevector_path.wavevectors[wavevector_index]
        return f"k = ({wavenumber_x:.1f}, {wavenumber_y:.1f}) rad/m"
    coordinate = wavevector_path.path_coordinate[wavevector_index]
    tolerance = relative_tolerance * max(wavevector_path.path_coordinate[-1], 1.0)
    on_symmetry_point = np.flatnonzero(np.abs(positions - coordinate) <= tolerance)
    if on_symmetry_point.size:
        return labels[on_symmetry_point[0]]
    segment = int(np.searchsorted(positions, coordinate)) - 1
    return f"{labels[segment]}–{labels[segment + 1]}"


def find_complete_gaps(frequencies, wavevector_path, minimum_relative_width=0.005):
    """Complete band gaps of the frequency-sorted bands, shape (n_k, n_bands) in Hz (DispersionResult.frequencies,
    or an analytic table on the same path).

    A complete gap between sorted bands n and n + 1 exists if max over k of band n < min over k of band n + 1:
    no wavevector on the path has a Bloch wave in between. NaN padding means "above the computed limit at this
    k", so it counts as +infinity: a NaN in band n closes the gap (band n reaches above the limit), a NaN in band
    n + 1 does not open one. A gap whose upper band is NaN everywhere is not reported (upper edge unknown).

    Use the sorted frequencies, not the tracked branches: a gap is a property of the spectrum at each k, and a
    tracked branch can cross another one, so its extremes say nothing about empty frequency ranges.
    minimum_relative_width: gaps narrower than this fraction of their centre frequency are dropped; the default
        0.5 % sits above the FE band error at the production mesh (0.33 % below 5 kHz).
    The path only covers the edge of the irreducible Brillouin zone; band extrema usually lie on it, which
    validate_band_gaps.py checks with a zone-interior grid for the production resonator cell."""
    band_values = np.where(np.isnan(frequencies), np.inf, np.asarray(frequencies, dtype=float))
    gaps = []
    for lower_band_index in range(band_values.shape[1] - 1):
        lower_band, upper_band = band_values[:, lower_band_index], band_values[:, lower_band_index + 1]
        lower_edge_hz, upper_edge_hz = lower_band.max(), upper_band.min()
        if not (np.isfinite(lower_edge_hz) and np.isfinite(upper_edge_hz)) or upper_edge_hz <= lower_edge_hz:
            continue
        if (upper_edge_hz - lower_edge_hz) / (0.5 * (lower_edge_hz + upper_edge_hz)) < minimum_relative_width:
            continue
        lower_edge_index, upper_edge_index = int(np.argmax(lower_band)), int(np.argmin(upper_band))
        gaps.append(BandGap(
            lower_band_index=lower_band_index, lower_edge_hz=float(lower_edge_hz), upper_edge_hz=float(upper_edge_hz),
            lower_edge_wavevector=wavevector_path.wavevectors[lower_edge_index].copy(),
            upper_edge_wavevector=wavevector_path.wavevectors[upper_edge_index].copy(),
            lower_edge_location=wavevector_location(wavevector_path, lower_edge_index),
            upper_edge_location=wavevector_location(wavevector_path, upper_edge_index)))
    return gaps


def describe_band_gap(gap):
    """One-line summary for printing, e.g. '1031.4-1195.2 Hz (width 163.8 Hz, 14.7 %), edges at M and X'."""
    return (f"{gap.lower_edge_hz:.1f}-{gap.upper_edge_hz:.1f} Hz (width {gap.upper_edge_hz - gap.lower_edge_hz:.1f} "
            f"Hz, {gap.relative_width:.1%}), between sorted bands {gap.lower_band_index + 1} and "
            f"{gap.lower_band_index + 2}, edges at {gap.lower_edge_location} and {gap.upper_edge_location}")


def build_irreducible_zone_grid(length_x, length_y, points_per_side=16):
    """Wavevectors filling the irreducible Brillouin zone, boundary included, as a WavevectorPath without symmetry
    points (path_coordinate is the running index). Used to check that band extrema lie on the path.
    Square cell: the triangle Gamma-X-M (0 <= ky <= kx <= pi/a); rectangular cell: [0, pi/Lx] x [0, pi/Ly]."""
    wavenumbers_x = np.linspace(0.0, np.pi / length_x, points_per_side + 1)
    wavenumbers_y = np.linspace(0.0, np.pi / length_y, points_per_side + 1)
    if np.isclose(length_x, length_y, rtol=1e-12):
        wavevectors = [(wavenumbers_x[i], wavenumbers_y[j]) for i in range(points_per_side + 1) for j in range(i + 1)]
    else:
        wavevectors = [(wavenumber_x, wavenumber_y) for wavenumber_x in wavenumbers_x for wavenumber_y in wavenumbers_y]
    return WavevectorPath(wavevectors=np.array(wavevectors), path_coordinate=np.arange(len(wavevectors), dtype=float),
                          symmetry_point_positions=np.array([]), symmetry_point_labels=[])
