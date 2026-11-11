"""Validation of the branch tracking (band_tracking.py, dispersion_sweep(track_bands=True)).

Run in fenicsx-env-complex from refactored/ (about 3 min):
    ~/miniconda3/envs/fenicsx-env-complex/bin/python -u validate_band_tracking.py

A. Synthetic cases with known answers (identity mass matrix, no FE):
   true crossing between k-points, true crossing exactly on a k-point, avoided crossing,
   crossing through a 3-fold degenerate point, fallback for unrelated eigenvectors.
B. Bare FE cell: every tracked branch must follow one analytic plane-wave branch k + G (the folded bands cross
   without interacting); plain sorting must violate this somewhere, otherwise the test would prove nothing.
   Below 5 kHz the folded bands of the 0.1 m cell meet only at Gamma, X and M (the bisector of k + G and k + G'
   meets the path only there for these small G); the first crossing between symmetry points is at
   |k + G| = 2.5 pi / a (~7.5 kHz). B therefore runs to 9 kHz on a 20x20 cell (finer, so the FE bands stay
   within the 0.5 % tolerance of the analytic ones).
C. Resonator FE cell (f_r = 1.2 kHz, mass ratio 0.25): band 1 has an avoided crossing on M-Gamma and must keep its
   label; degenerate groups at Gamma, X, M must be matched; does tracking differ from sorting below 5 kHz?
D. Deliberately coarse path: the fallback must trigger.
Plots go to results/ (tracked diagrams of B and C).
"""
import time
import dataclasses
import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices
from dispersion import (unit_cell_element_counts, build_wavevector_path, dispersion_sweep, plate_constants,
                        plane_wave_omega_squared)
from band_tracking import band_overlap, track_bands, relabelled_bands, degenerate_groups
from postprocessing_dispersion import plot_tracked_dispersion

PLOTTED_LIMIT_HZ = 5000.0
TRACKING_LIMIT_HZ = 5500.0      # track a bit above the plotted range, so branches enter and leave outside the plot
CELL_LENGTH = 0.1


# --- A. synthetic cases ---

def synthetic_tracking(frequency_rows, vector_rows):
    """Sort each k by frequency (like the solver), build the overlaps with an identity mass and track."""
    sorted_frequencies, sorted_vectors = [], []
    for frequencies, vectors in zip(frequency_rows, vector_rows):
        order = np.argsort(frequencies, kind="stable")
        sorted_frequencies.append(np.asarray(frequencies)[order])
        sorted_vectors.append(np.asarray(vectors)[:, order])
    identity = np.eye(sorted_vectors[0].shape[0])
    overlaps = [band_overlap(left, right, identity) for left, right in zip(sorted_vectors[:-1], sorted_vectors[1:])]
    bridges = [band_overlap(left, right, identity) for left, right in zip(sorted_vectors[:-2], sorted_vectors[2:])]
    return track_bands(np.array(sorted_frequencies), overlaps, bridge_overlaps=bridges)


def random_unitary(size, generator):
    matrix = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
    return np.linalg.qr(matrix)[0]


def check_synthetic():
    generator = np.random.default_rng(1)
    basis = np.eye(3, dtype=complex)

    # true crossing between k-points and exactly on one: branch a rises, b falls, eigenvectors fixed
    for name, parameters in (("between k-points", np.arange(0, 10.5, 0.7)), ("on a k-point", np.arange(0, 11.0))):
        rising, falling = 100 + 10 * parameters, 200 - 10 * parameters
        vector_rows = []
        for rising_value, falling_value in zip(rising, falling):
            vectors = basis[:, :2].copy()
            if np.isclose(rising_value, falling_value):            # degenerate: the solver returns any basis
                vectors = vectors @ random_unitary(2, generator)
            vector_rows.append(vectors)
        tracked = synthetic_tracking(np.column_stack([rising, falling]), vector_rows)
        assert np.allclose(tracked.frequencies[:, 0], rising) and np.allclose(tracked.frequencies[:, 1], falling), \
            f"true crossing {name}: branches swapped"
        assert all(status == "tracked" for status in tracked.step_status)
        print(f"A1. true crossing {name}: both branches followed, {relabelled_bands(tracked)} relabelled "
              f"sorted positions (expected > 0)")

    # avoided crossing: two-level model, eigenvectors turn by 90 degrees; labels must stay sorted
    frequency_rows, vector_rows = [], []
    for detuning in np.linspace(-5, 5, 41):
        values, vectors = np.linalg.eigh(np.array([[150 + detuning, 1.0], [1.0, 150 - detuning]]))
        frequency_rows.append(values)
        vector_rows.append(np.vstack([vectors, np.zeros((1, 2))]))
    tracked = synthetic_tracking(np.array(frequency_rows), vector_rows)
    assert relabelled_bands(tracked) == 0 and all(status == "tracked" for status in tracked.step_status)
    print(f"A2. avoided crossing: labels stay sorted, lowest match quality {np.nanmin(tracked.match_quality):.3f}")

    # crossing through a 3-fold degenerate point: order of the branches changes across it
    frequency_rows = [[10, 11, 12], [10.5, 11, 11.5], [11, 11, 11], [10.5, 11, 11.5], [10, 11, 12]]
    vector_rows = [basis, basis, basis @ random_unitary(3, generator), basis[:, [2, 1, 0]], basis[:, [2, 1, 0]]]
    tracked = synthetic_tracking(np.array(frequency_rows, dtype=float), vector_rows)
    expected = np.array([[10, 11, 12], [10.5, 11, 11.5], [11, 11, 11], [11.5, 11, 10.5], [12, 11, 10]], dtype=float)
    assert np.allclose(tracked.frequencies, expected), tracked.frequencies
    print("A3. crossing through a 3-fold degenerate point: bridged, all three branches followed")

    # unrelated eigenvectors with equal band count: fallback to plain sorting
    mixed_vectors = np.array([[0.6, 0.6, 0.53], [0.6, -0.75, 0.28], [0.53, 0.28, -0.8]])   # no MAC above 0.64
    vector_rows = [basis, mixed_vectors]
    tracked = synthetic_tracking(np.array([[10, 11, 12], [10, 11, 12]], dtype=float), vector_rows)
    assert tracked.step_status == ["fallback"] and relabelled_bands(tracked) == 0
    print("A4. unrelated eigenvectors: step flagged 'fallback', plain sorting used")


# --- B-D. FE unit cell ---

def build_unit_cell(elements_per_metre=None):
    base = PlateConfig(length=CELL_LENGTH, width=CELL_LENGTH, thickness=0.005, nx=1, ny=1, rho=7850, mu=77e9,
                       lambda_=115e9)
    elements_x, elements_y = unit_cell_element_counts(CELL_LENGTH, CELL_LENGTH) if elements_per_metre is None \
        else unit_cell_element_counts(CELL_LENGTH, CELL_LENGTH, elements_per_metre=elements_per_metre)
    cell_config = dataclasses.replace(base, nx=elements_x, ny=elements_y)
    _, function_space, _, problem = build_plate_problem(cell_config, deg=2, el_type="Lagrange")
    return cell_config, function_space, problem


def analytic_branch_frequencies(wavevectors, cell_config, harmonic_range=4):
    """Bending branch of every plane wave k + G separately (no sorting): shape (n_k, n_G). These are the
    true identities of the bare-cell bands."""
    constants = plate_constants(cell_config)
    reciprocal = 2 * np.pi / cell_config.length
    columns = []
    for m in range(-harmonic_range, harmonic_range + 1):
        for n in range(-harmonic_range, harmonic_range + 1):
            magnitudes = np.hypot(wavevectors[:, 0] + m * reciprocal, wavevectors[:, 1] + n * reciprocal)
            omega_squared = [min(plane_wave_omega_squared(magnitude, constants)) for magnitude in magnitudes]
            columns.append(np.sqrt(np.clip(omega_squared, 0, None)) / (2 * np.pi))
    return np.column_stack(columns)


def identity_violations(branch_frequencies, analytic_branches, frequency_limit_hz, window=5,
                        relative_tolerance=0.005, absolute_tolerance_hz=2.0):
    """Windows of `window` consecutive k-points where a branch (column) cannot be one and the same plane wave:
    no analytic branch k + G lies within tolerance at all k-points of the window. A single step is not enough,
    because next to a crossing both analytic branches are within tolerance on both sides; a few steps further
    they have separated. Only windows entirely below frequency_limit_hz are checked."""
    violations = []
    for step in range(len(branch_frequencies) - window + 1):
        for column in range(branch_frequencies.shape[1]):
            values = branch_frequencies[step:step + window, column]
            if np.isnan(values).any() or values.max() > frequency_limit_hz:
                continue
            tolerance = np.maximum(absolute_tolerance_hz, relative_tolerance * values)
            within = np.abs(analytic_branches[step:step + window] - values[:, None]) <= tolerance[:, None]
            assert within.any(axis=1).all(), f"no analytic match near step {step}, {values[0]:.1f} Hz"
            if not within.all(axis=0).any():
                violations.append((step, column, values[0], values[-1]))
    return violations


def relabelled_below_limit(tracked, frequencies):
    """Sorted positions below the plotted limit whose branch changes from one step to the next: the places where
    the tracked curves differ from plain sorting (true crossings)."""
    events = []
    for step in range(1, len(frequencies)):
        for position in range(frequencies.shape[1]):
            previous_label, current_label = tracked.labels[step - 1, position], tracked.labels[step, position]
            if min(previous_label, current_label) < 0 or frequencies[step, position] > PLOTTED_LIMIT_HZ:
                continue
            if previous_label != current_label and current_label in tracked.labels[step - 1]:
                events.append((step, position, frequencies[step, position]))
    return events


def symmetry_point_indices(path):
    return {label: int(np.argmin(np.abs(path.path_coordinate - position)))
            for label, position in zip(path.symmetry_point_labels[:-1], path.symmetry_point_positions[:-1])}


BARE_CHECK_LIMIT_HZ, BARE_TRACKING_LIMIT_HZ, BARE_ELEMENTS_PER_METRE = 9000.0, 9500.0, 200


def check_bare_cell(path):
    cell_config, function_space, problem = build_unit_cell(elements_per_metre=BARE_ELEMENTS_PER_METRE)
    start = time.perf_counter()
    result = dispersion_sweep(function_space, problem, cell_config, path,
                              maximum_frequency_hz=BARE_TRACKING_LIMIT_HZ, track_bands=True)
    tracked = result.tracking
    analytic = analytic_branch_frequencies(path.wavevectors, cell_config)
    tracked_violations = identity_violations(tracked.frequencies, analytic, BARE_CHECK_LIMIT_HZ)
    sorted_violations = identity_violations(result.frequencies, analytic, BARE_CHECK_LIMIT_HZ)
    statuses = {status: tracked.step_status.count(status) for status in set(tracked.step_status)}
    print(f"B. bare cell ({time.perf_counter() - start:.0f} s): steps {statuses}, "
          f"lowest match quality {np.nanmin(tracked.match_quality):.3f}")
    print(f"   identity violations below {BARE_CHECK_LIMIT_HZ:.0f} Hz (windows of 5 k-points): tracked {len(tracked_violations)}, "
          f"plain sorting {len(sorted_violations)} (must be > 0 for the test to mean anything)")
    for violation in tracked_violations[:5]:
        print(f"   tracked violation: step {violation[0]}, branch {violation[1]}, {violation[2]:.1f} -> "
              f"{violation[3]:.1f} Hz")
    assert not tracked_violations and sorted_violations
    plot_tracked_dispersion(result, colour_by="branch", frequency_limit=BARE_CHECK_LIMIT_HZ,
                            title=f"Bare cell 0.1 m, h = 5 mm, {BARE_ELEMENTS_PER_METRE} elements/m: "
                                  "tracked branches (one colour per branch)",
                            save_path="results/dispersion_bare_tracked.png")


def check_resonator_cell(cell_config, function_space, problem, path):
    resonator_mass = 0.25 * cell_config.rho * cell_config.thickness * CELL_LENGTH ** 2
    vamm_list = create_vamm_list_and_assign_indices(
        [(CELL_LENGTH / 2, CELL_LENGTH / 2, resonator_mass * (2 * np.pi * 1200.0) ** 2, resonator_mass)])
    start = time.perf_counter()
    result = dispersion_sweep(function_space, problem, cell_config, path, maximum_frequency_hz=TRACKING_LIMIT_HZ,
                              vamm_list=vamm_list, track_bands=True)
    tracked = result.tracking
    statuses = {status: tracked.step_status.count(status) for status in set(tracked.step_status)}
    print(f"C. resonator cell ({time.perf_counter() - start:.0f} s): steps {statuses}, "
          f"lowest match quality {np.nanmin(tracked.match_quality):.3f}")
    assert all(status == "tracked" for status in tracked.step_status)

    # avoided crossing: band 1 keeps label 0 along the whole path
    assert np.all(tracked.labels[:, 0] == 0), "band 1 was relabelled"
    points = symmetry_point_indices(path)
    share_gamma, share_m = tracked.plate_share[points["Γ"], 0], tracked.plate_share[points["M"], 0]
    print(f"   band 1 keeps its label; plate share {share_gamma:.0%} at Γ, {share_m:.0%} at M "
          f"(theory session: 80 % and 5 %)")
    assert 0.75 < share_gamma < 0.85 and share_m < 0.1

    # degenerate groups at the symmetry points and the match quality next to them
    for label, index in points.items():
        frequencies = result.frequencies[index][~np.isnan(result.frequencies[index])]
        groups = [group for group in degenerate_groups(frequencies) if len(group) > 1]
        neighbours = [step for step in (index - 1, index, index + 1) if 0 < step < len(path.wavevectors)]
        nearby_quality = np.nanmin(tracked.match_quality[neighbours])
        description = ", ".join(f"{len(group)}x {frequencies[group[0]]:.0f} Hz" for group in groups) or "none"
        print(f"   {label}: degenerate groups {description}; lowest match quality next to it {nearby_quality:.3f}")

    events = relabelled_below_limit(tracked, result.frequencies)
    print(f"   label changes against plain sorting below {PLOTTED_LIMIT_HZ:.0f} Hz (true crossings): {len(events)}"
          + "".join(f"\n     step {step}, sorted band {position + 1}, {frequency:.0f} Hz"
                    for step, position, frequency in events[:10]))
    # Below 5 kHz this cell has no true crossing; degenerate groups (e.g. the 3-fold group at Gamma, where the path
    # starts) must not cause label changes either: the tie-breaker keeps the sorted order where scores are equal.
    assert not events, "label changes below the limit"
    title = "Unit cell 0.1 m with resonator: f_r = 1200 Hz, mass ratio 0.25, h = 5 mm"
    plot_tracked_dispersion(result, colour_by="plate_share", frequency_limit=PLOTTED_LIMIT_HZ,
                            resonator_frequencies_hz=[1200.0], title=title,
                            save_path="results/dispersion_vamm_1200_tracked.png")
    plot_tracked_dispersion(result, colour_by="branch", frequency_limit=PLOTTED_LIMIT_HZ,
                            resonator_frequencies_hz=[1200.0], title=title + " (branches)",
                            save_path="results/dispersion_vamm_1200_branches.png")
    return vamm_list


def check_fallback(cell_config, function_space, problem, vamm_list):
    coarse_path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=3)
    result = dispersion_sweep(function_space, problem, cell_config, coarse_path,
                              maximum_frequency_hz=TRACKING_LIMIT_HZ, vamm_list=vamm_list, track_bands=True)
    statuses = {status: result.tracking.step_status.count(status) for status in set(result.tracking.step_status)}
    print(f"D. coarse path ({len(coarse_path.wavevectors)} wavevectors): steps {statuses}")
    assert any(status != "tracked" for status in result.tracking.step_status), "fallback never triggered"


if __name__ == "__main__":
    total_start = time.perf_counter()
    check_synthetic()
    path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=40)
    check_bare_cell(path)
    cell_config, function_space, problem = build_unit_cell()
    vamm_list = check_resonator_cell(cell_config, function_space, problem, path)
    check_fallback(cell_config, function_space, problem, vamm_list)
    print(f"All band tracking checks passed ({time.perf_counter() - total_start:.0f} s).")
