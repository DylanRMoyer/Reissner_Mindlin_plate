"""Checks for dispersion.py (run in fenicsx-env-complex):
1. Wavevector path: corners, no duplicates, monotonic path coordinate, segment lengths, square vs rectangle.
2. dispersion_sweep at the symmetry points equals single solve_bloch_evp calls there.
3. All bands below the maximum frequency are found (band count vs analytic and vs the folding estimate).
4. The bare-cell diagram stays within PRODUCTION_MESH_TOLERANCE of the analytic folded bands
   (production cell: 0.1 m, h = 5 mm steel, bands below 5 kHz, 100 elements per metre).
"""
import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from periodic_bc import build_pairs
from bloch_solver import assemble_unit_cell_matrices, reduce_to_bloch, solve_bloch_evp
from dispersion import (build_wavevector_path, dispersion_sweep, analytic_dispersion, unit_cell_element_counts,
                        estimate_band_count, bare_cell_relative_error)

LENGTH_X, LENGTH_Y = 1.1, 1.0                      # rectangular cell, path checks only
UNIT_CELL_LENGTH, THICKNESS = 0.1, 0.005           # production cell, same as main.py
MAXIMUM_FREQUENCY_HZ = 5000.0
PRODUCTION_MESH_TOLERANCE = 1e-2


# 1. path
square_path = build_wavevector_path(1.0, 1.0, points_per_segment=10)
assert square_path.symmetry_point_labels == ["Γ", "X", "M", "Γ"]
assert np.allclose(square_path.symmetry_point_positions, [0, np.pi, 2 * np.pi, 2 * np.pi + np.sqrt(2) * np.pi])
assert len(square_path.wavevectors) == 10 + 10 + 14 + 1          # M-Gamma: sqrt(2) * 10 = 14 intervals

rectangle_path = build_wavevector_path(LENGTH_X, LENGTH_Y, points_per_segment=10)
assert rectangle_path.symmetry_point_labels == ["Γ", "X", "M", "Y", "Γ"]
corners = {"Γ": (0, 0), "X": (np.pi / LENGTH_X, 0), "M": (np.pi / LENGTH_X, np.pi / LENGTH_Y), "Y": (0, np.pi / LENGTH_Y)}
for path in (square_path, rectangle_path):
    assert np.all(np.diff(path.path_coordinate) > 0), "path coordinate must increase strictly (no duplicated points)"
    assert np.isclose(path.path_coordinate[-1], path.symmetry_point_positions[-1])
    for position, label in zip(path.symmetry_point_positions, path.symmetry_point_labels):
        index = np.flatnonzero(np.isclose(path.path_coordinate, position))
        assert len(index) == 1, (label, "symmetry point must be on the path exactly once")
        expected = corners[label] if path is rectangle_path else {"Γ": (0, 0), "X": (np.pi, 0), "M": (np.pi, np.pi)}[label]
        assert np.allclose(path.wavevectors[index[0]], expected), (label, path.wavevectors[index[0]])
print("wavevector path checks passed")

# 2. - 4. bare-cell sweep on the production mesh
elements_x, elements_y = unit_cell_element_counts(UNIT_CELL_LENGTH, UNIT_CELL_LENGTH)
plate_config = PlateConfig(length=UNIT_CELL_LENGTH, width=UNIT_CELL_LENGTH, thickness=THICKNESS,
                           nx=elements_x, ny=elements_y, rho=7850, mu=77e9, lambda_=115e9)
_, function_space, _, problem = build_plate_problem(plate_config, deg=2, el_type="Lagrange")
path = build_wavevector_path(UNIT_CELL_LENGTH, UNIT_CELL_LENGTH, points_per_segment=20)
result = dispersion_sweep(function_space, problem, plate_config, path, maximum_frequency_hz=MAXIMUM_FREQUENCY_HZ,
                          keep_eigenvectors=True)
assert np.nanmax(result.frequencies) <= MAXIMUM_FREQUENCY_HZ
assert result.eigenvectors.shape[0] == len(path.wavevectors)
assert result.eigenvectors.shape[2] == result.frequencies.shape[1]

stiffness, mass = assemble_unit_cell_matrices(function_space, problem)
pairs = build_pairs(function_space, UNIT_CELL_LENGTH, UNIT_CELL_LENGTH)
for position, label in zip(path.symmetry_point_positions, path.symmetry_point_labels):
    index = np.flatnonzero(np.isclose(path.path_coordinate, position))[0]
    swept = result.frequencies[index][~np.isnan(result.frequencies[index])]
    single_frequencies, _ = solve_bloch_evp(*reduce_to_bloch(stiffness, mass, pairs, *path.wavevectors[index],
                                                             UNIT_CELL_LENGTH, UNIT_CELL_LENGTH),
                                            number_of_bands=len(swept) + 4)
    assert np.allclose(swept, single_frequencies[:len(swept)], atol=1e-6), label
    assert single_frequencies[len(swept)] > MAXIMUM_FREQUENCY_HZ, (label, "a band below the limit was dropped")
print("sweep matches single solves at the symmetry points, no band below the limit dropped")

analytic_frequencies = analytic_dispersion(path, plate_config, maximum_frequency_hz=1.05 * MAXIMUM_FREQUENCY_HZ)
band_counts = np.sum(~np.isnan(result.frequencies), axis=1)
analytic_band_counts = np.sum(~np.isnan(analytic_frequencies), axis=1)
print(f"bands below {MAXIMUM_FREQUENCY_HZ:.0f} Hz per k: FE {band_counts.min()}-{band_counts.max()}, "
      f"analytic (to 1.05 x limit) {analytic_band_counts.min()}-{analytic_band_counts.max()}, "
      f"folding estimate {estimate_band_count(plate_config, MAXIMUM_FREQUENCY_HZ)}")

worst_relative_error = bare_cell_relative_error(result, analytic_frequencies)
print(f"max relative error over all bands below {MAXIMUM_FREQUENCY_HZ:.0f} Hz and {len(path.wavevectors)} "
      f"wavevectors: {worst_relative_error:.2e}")
assert worst_relative_error < PRODUCTION_MESH_TOLERANCE, "bare-cell diagram outside the production tolerance"
print("dispersion checks passed")
