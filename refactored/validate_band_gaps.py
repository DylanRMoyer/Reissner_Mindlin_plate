"""Validation of the complete-gap finder (dispersion.find_complete_gaps) and the dispersion cache
(sweep_cache.get_or_compute_dispersion).

Run in fenicsx-env-complex from refactored/ (about 2-3 min):
    ~/miniconda3/envs/fenicsx-env-complex/bin/python -u validate_band_gaps.py

A. Synthetic tables with known answers: a clear gap, overlapping bands, NaN padding in the upper band (must not
   open a gap) and in the lower band (must close it), a gap below the width threshold, the top band.
B. Resonator FE cell (f_r = 1.2 kHz, mass ratio 0.25): the session-3 gap 1031-1195 Hz with edges at M and X.
C. Same cell, plane-wave expansion (independent model): same gap within 1 % per edge.
D. Zone interior: a grid over the whole irreducible zone (triangle Gamma-X-M) must not push any band beyond its
   extremes on the path by more than a tolerance, so the path gap is a true complete gap.
E. Bare FE cell: no complete gap (a homogeneous plate covers every frequency); also reports the widest positive
   "gap" without the width threshold, which would be a sampling artefact of the discrete path.
F. Cache round trip: save, load, identical arrays and identical gaps.
"""
import time
import tempfile
import dataclasses
from pathlib import Path
import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices
from dispersion import (unit_cell_element_counts, build_wavevector_path, dispersion_sweep, find_complete_gaps,
                        describe_band_gap, build_irreducible_zone_grid, analytic_resonator_dispersion)
from sweep_cache import save_dispersion, load_dispersion

CELL_LENGTH = 0.1
LIMIT_HZ = 5500.0                       # as the tracked production diagram (10 % above the plotted 5 kHz)
EXPECTED_GAP_HZ = (1031.0, 1195.0)      # session 3 (rounded to 1 Hz), edges at M and X


# --- A. synthetic tables ---

def check_synthetic():
    path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=4)   # 15 wavevectors
    number_of_wavevectors = len(path.wavevectors)
    x_index, m_index = 4, 8
    lower_band = np.linspace(0.0, 90.0, number_of_wavevectors)
    lower_band[x_index] = 100.0                      # maximum at X
    upper_band = np.full(number_of_wavevectors, 300.0)
    upper_band[m_index] = 150.0                      # minimum at M
    frequencies = np.column_stack([lower_band, upper_band])

    gaps = find_complete_gaps(frequencies, path)
    assert len(gaps) == 1
    gap = gaps[0]
    assert (gap.lower_band_index, gap.lower_edge_hz, gap.upper_edge_hz) == (0, 100.0, 150.0)
    assert (gap.lower_edge_location, gap.upper_edge_location) == ("X", "M")
    assert np.isclose(gap.relative_width, 50.0 / 125.0)
    assert np.allclose(gap.lower_edge_wavevector, [np.pi / CELL_LENGTH, 0.0])

    overlapping = frequencies.copy()
    overlapping[0, 1] = 99.0                          # upper band dips below the lower maximum at Gamma
    assert find_complete_gaps(overlapping, path) == []

    nan_in_upper = frequencies.copy()
    nan_in_upper[2:6, 1] = np.nan                     # upper band above the computed limit there: still a gap
    assert len(find_complete_gaps(nan_in_upper, path)) == 1

    nan_in_lower = frequencies.copy()
    nan_in_lower[10, 0] = np.nan                      # lower band above the limit there: reaches beyond the gap
    assert find_complete_gaps(nan_in_lower, path) == []

    narrow = frequencies.copy()
    narrow[m_index, 1] = 100.4                        # 0.4 % gap: below the default 0.5 % threshold
    assert find_complete_gaps(narrow, path) == []
    assert len(find_complete_gaps(narrow, path, minimum_relative_width=0.0)) == 1

    top_band_only = np.column_stack([lower_band, np.full(number_of_wavevectors, np.nan)])
    assert find_complete_gaps(top_band_only, path) == []

    segment_case = frequencies.copy()
    segment_case[x_index, 0], segment_case[6, 0] = 50.0, 120.0      # lower maximum now between X and M
    assert find_complete_gaps(segment_case, path)[0].lower_edge_location == "X–M"
    print("A. synthetic tables: all cases as expected")


# --- B-F. FE unit cell ---

def build_unit_cell():
    base = PlateConfig(length=CELL_LENGTH, width=CELL_LENGTH, thickness=0.005, nx=1, ny=1, rho=7850, mu=77e9,
                       lambda_=115e9)
    elements_x, elements_y = unit_cell_element_counts(CELL_LENGTH, CELL_LENGTH)
    cell_config = dataclasses.replace(base, nx=elements_x, ny=elements_y)
    _, function_space, _, problem = build_plate_problem(cell_config, deg=2, el_type="Lagrange")
    return cell_config, function_space, problem


def production_resonator(cell_config):
    resonator_mass = 0.25 * cell_config.rho * cell_config.thickness * CELL_LENGTH ** 2
    return resonator_mass * (2 * np.pi * 1200.0) ** 2, resonator_mass


def check_resonator_cell(cell_config, function_space, problem, path, resonator_stiffness, resonator_mass):
    vamm_list = create_vamm_list_and_assign_indices(
        [(CELL_LENGTH / 2, CELL_LENGTH / 2, resonator_stiffness, resonator_mass)])
    result = dispersion_sweep(function_space, problem, cell_config, path, maximum_frequency_hz=LIMIT_HZ,
                              vamm_list=vamm_list, track_bands=True)
    gaps = find_complete_gaps(result.frequencies, result.path)
    print(f"B. resonator FE cell: {len(gaps)} complete gap(s) below {LIMIT_HZ:.0f} Hz")
    for gap in gaps:
        print("   " + describe_band_gap(gap))
    first_gap = gaps[0]
    assert abs(first_gap.lower_edge_hz - EXPECTED_GAP_HZ[0]) < 1.0
    assert abs(first_gap.upper_edge_hz - EXPECTED_GAP_HZ[1]) < 1.0
    assert (first_gap.lower_edge_location, first_gap.upper_edge_location) == ("M", "X")
    return result, gaps, vamm_list


def check_plane_wave_model(cell_config, path, resonator_stiffness, resonator_mass, finite_element_gap):
    start = time.perf_counter()
    analytic = analytic_resonator_dispersion(path, cell_config, resonator_stiffness, resonator_mass,
                                             maximum_frequency_hz=2000.0)
    analytic_gaps = find_complete_gaps(analytic, path)
    analytic_gap = analytic_gaps[0]
    lower_deviation = analytic_gap.lower_edge_hz / finite_element_gap.lower_edge_hz - 1
    upper_deviation = analytic_gap.upper_edge_hz / finite_element_gap.upper_edge_hz - 1
    print(f"C. plane-wave expansion ({time.perf_counter() - start:.0f} s): " + describe_band_gap(analytic_gap))
    print(f"   edge deviation from FE: lower {lower_deviation:+.2%}, upper {upper_deviation:+.2%}")
    assert abs(lower_deviation) < 0.01 and abs(upper_deviation) < 0.01
    assert analytic_gap.lower_band_index == finite_element_gap.lower_band_index
    assert (analytic_gap.lower_edge_location, analytic_gap.upper_edge_location) == ("M", "X")


def check_zone_interior(cell_config, function_space, problem, vamm_list, path_result, path_gaps):
    zone_grid = build_irreducible_zone_grid(CELL_LENGTH, CELL_LENGTH, points_per_side=16)
    zone_result = dispersion_sweep(function_space, problem, cell_config, zone_grid, maximum_frequency_hz=LIMIT_HZ,
                                   vamm_list=vamm_list)
    path_values = np.where(np.isnan(path_result.frequencies), np.inf, path_result.frequencies)
    zone_values = np.where(np.isnan(zone_result.frequencies), np.inf, zone_result.frequencies)
    # Every band whose path maximum is below the limit (finite) must not exceed it inside the zone (a NaN = above
    # the limit inside the zone counts as an infinite excess); every band minimum on the path must not be undercut.
    number_of_bands = min(path_values.shape[1], zone_values.shape[1])
    bands_with_finite_maximum = [band for band in range(number_of_bands) if np.isfinite(path_values[:, band].max())]
    bands_with_finite_minimum = [band for band in range(number_of_bands)
                                 if 1.0 < path_values[:, band].min() < np.inf]      # skip the zero band at Gamma
    maximum_excess = max(zone_values[:, band].max() / path_values[:, band].max() - 1
                         for band in bands_with_finite_maximum)
    minimum_excess = max(1 - zone_values[:, band].min() / path_values[:, band].min()
                         for band in bands_with_finite_minimum)
    print(f"D. zone interior ({len(zone_grid.wavevectors)} wavevectors): the interior exceeds the path maximum by at "
          f"most {maximum_excess:+.2e} ({len(bands_with_finite_maximum)} bands) and undercuts the path minimum by at "
          f"most {minimum_excess:+.2e} ({len(bands_with_finite_minimum)} bands), relative")

    zone_gaps = find_complete_gaps(zone_result.frequencies, zone_grid)
    tolerance = 1e-6
    for path_gap in path_gaps:
        matches = [gap for gap in zone_gaps if gap.lower_band_index == path_gap.lower_band_index]
        assert matches, f"gap above band {path_gap.lower_band_index + 1} closes inside the zone"
        zone_gap = matches[0]
        print(f"   gap above band {path_gap.lower_band_index + 1}: path {path_gap.lower_edge_hz:.2f}-"
              f"{path_gap.upper_edge_hz:.2f} Hz, zone {zone_gap.lower_edge_hz:.2f}-{zone_gap.upper_edge_hz:.2f} Hz "
              f"(zone edges at {zone_gap.lower_edge_location} / {zone_gap.upper_edge_location})")
        assert zone_gap.lower_edge_hz <= path_gap.lower_edge_hz * (1 + tolerance)
        assert zone_gap.upper_edge_hz >= path_gap.upper_edge_hz * (1 - tolerance)


def check_bare_cell(cell_config, function_space, problem, path):
    result = dispersion_sweep(function_space, problem, cell_config, path, maximum_frequency_hz=LIMIT_HZ)
    gaps = find_complete_gaps(result.frequencies, result.path)
    unthresholded = find_complete_gaps(result.frequencies, result.path, minimum_relative_width=0.0)
    widest = max(unthresholded, key=lambda gap: gap.relative_width, default=None)
    print(f"E. bare FE cell: {len(gaps)} complete gap(s) above 0.5 %; without threshold {len(unthresholded)}"
          + (f", widest {describe_band_gap(widest)}" if widest else ""))
    assert gaps == []


def check_cache_round_trip(result, gaps):
    with tempfile.TemporaryDirectory() as directory:
        cache_path = Path(directory) / "dispersion_check.npz"
        save_dispersion(cache_path, result, settings_blob="validation")
        loaded = load_dispersion(cache_path)
    np.testing.assert_array_equal(loaded.frequencies, result.frequencies)
    np.testing.assert_array_equal(loaded.path.wavevectors, result.path.wavevectors)
    np.testing.assert_array_equal(loaded.resonator_share, result.resonator_share)
    assert loaded.path.symmetry_point_labels == result.path.symmetry_point_labels
    assert loaded.maximum_frequency_hz == result.maximum_frequency_hz and loaded.eigenvectors is None
    for field in ("labels", "frequencies", "plate_share", "match_quality"):
        np.testing.assert_array_equal(getattr(loaded.tracking, field), getattr(result.tracking, field))
    assert loaded.tracking.step_status == result.tracking.step_status
    loaded_gaps = find_complete_gaps(loaded.frequencies, loaded.path)
    assert [describe_band_gap(gap) for gap in loaded_gaps] == [describe_band_gap(gap) for gap in gaps]
    print("F. cache round trip: frequencies, path, resonator share, tracking tables and gaps identical")


if __name__ == "__main__":
    total_start = time.perf_counter()
    check_synthetic()
    cell_config, function_space, problem = build_unit_cell()
    path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=40)
    resonator_stiffness, resonator_mass = production_resonator(cell_config)
    resonator_result, resonator_gaps, vamm_list = check_resonator_cell(
        cell_config, function_space, problem, path, resonator_stiffness, resonator_mass)
    check_plane_wave_model(cell_config, path, resonator_stiffness, resonator_mass, resonator_gaps[0])
    check_zone_interior(cell_config, function_space, problem, vamm_list, resonator_result, resonator_gaps)
    check_bare_cell(cell_config, function_space, problem, path)
    check_cache_round_trip(resonator_result, resonator_gaps)
    print(f"All band gap checks passed ({time.perf_counter() - total_start:.0f} s).")
