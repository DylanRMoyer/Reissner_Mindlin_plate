"""Queue Q5: zone-interior check (check D of validate_band_gaps.py) for cells beyond the centred-resonator square.

Check D showed for the square cell with one resonator that no band leaves its path extremes inside the zone, so the
gaps found on Gamma-X-M-Gamma are complete gaps. Here the same test with FE for:
  A. rectangular cell 0.1 x 0.15 m, one resonator (symmetry C2v): path Gamma-X-M-Y-Gamma, interior = the
     rectangle [0, pi/Lx] x [0, pi/Ly].
  B. square cell with two resonators at (0, 0) and (a/2, 0) (Chapter 6, Section "Two resonators per cell"; so far
     plane-wave model only): mirror symmetries x -> -x and y -> -y remain, the 90 degree rotation is lost, so the
     path needs Y and the interior is the quarter zone [0, pi/a]^2.
  C. square cell with two resonators at general positions, (0.02, 0.03) and (0.07, 0.05) m: no mirror symmetry left,
     only k -> -k (time reversal), so the irreducible zone is half the Brillouin zone [-pi/a, pi/a] x [0, pi/a]
     and the path Gamma-X-M-Y-Gamma is no longer its boundary.
One resonator per cell at an off-centre position is not a separate case: moving it is a translation of the
whole lattice, so the bands do not change (validate_bloch_vamm_cell.py, position invariance to 1.8e-9).
Resonators: f_r = 1.2 kHz, mass ratio 0.25 each (relative to the plate mass of the cell); bands below 2.5 kHz.

Run in fenicsx-env-complex (~5-10 min)."""
import time

import numpy as np

from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices
from dispersion import (WavevectorPath, unit_cell_element_counts, dispersion_sweep, find_complete_gaps,
                        build_irreducible_zone_grid, describe_band_gap)
from lattice_plane_wave import make_lattice, lattice_path

RESONATOR_FREQUENCY_HZ = 1200.0
MASS_RATIO = 0.25
LIMIT_HZ = 2500.0
POINTS_PER_SEGMENT = 20
GRID_POINTS_PER_SIDE = 16


def cell_setup(length_x, length_y):
    elements_x, elements_y = unit_cell_element_counts(length_x, length_y)
    cell_config = PlateConfig(length=length_x, width=length_y, thickness=0.005, nx=elements_x, ny=elements_y,
                              rho=7850, mu=77e9, lambda_=115e9)
    _, function_space, _, problem = build_plate_problem(cell_config, deg=2, el_type="Lagrange")
    return cell_config, function_space, problem


def resonators(cell_config, positions):
    mass = MASS_RATIO * cell_config.rho * cell_config.thickness * cell_config.length * cell_config.width
    stiffness = mass * (2 * np.pi * RESONATOR_FREQUENCY_HZ) ** 2
    return create_vamm_list_and_assign_indices([(x, y, stiffness, mass) for x, y in positions])


def rectangle_path(length_x, length_y):
    """Gamma-X-M-Y-Gamma for any rectangular (also square) cell."""
    lattice = make_lattice("rectangle", [[length_x, 0.0], [0.0, length_y]],
                           [[0, 0], [np.pi / length_x, 0], [np.pi / length_x, np.pi / length_y],
                            [0, np.pi / length_y], [0, 0]], ["Γ", "X", "M", "Y", "Γ"])
    return lattice_path(lattice, POINTS_PER_SEGMENT)


def grid_path(wavevectors):
    wavevectors = np.asarray(wavevectors, dtype=float)
    return WavevectorPath(wavevectors=wavevectors, path_coordinate=np.arange(len(wavevectors), dtype=float),
                          symmetry_point_positions=np.array([]), symmetry_point_labels=[])


def quarter_zone_grid(length_x, length_y):
    """[0, pi/Lx] x [0, pi/Ly] (build_irreducible_zone_grid gives the triangle for square cells)."""
    wavenumbers_x = np.linspace(0, np.pi / length_x, GRID_POINTS_PER_SIDE + 1)
    wavenumbers_y = np.linspace(0, np.pi / length_y, GRID_POINTS_PER_SIDE + 1)
    return grid_path([(kx, ky) for kx in wavenumbers_x for ky in wavenumbers_y])


def half_zone_grid(length_x, length_y):
    """[-pi/Lx, pi/Lx] x [0, pi/Ly]: irreducible when only k -> -k is left."""
    wavenumbers_x = np.linspace(-np.pi / length_x, np.pi / length_x, 2 * GRID_POINTS_PER_SIDE + 1)
    wavenumbers_y = np.linspace(0, np.pi / length_y, GRID_POINTS_PER_SIDE + 1)
    return grid_path([(kx, ky) for kx in wavenumbers_x for ky in wavenumbers_y])


def compare_path_and_zone(name, cell_config, function_space, problem, vamm_list, path, zone, tolerance=1e-6):
    """Band extremes and complete gaps on the path vs on the zone grid. Returns (path gaps, zone gaps, report)."""
    start = time.perf_counter()
    path_result = dispersion_sweep(function_space, problem, cell_config, path, maximum_frequency_hz=LIMIT_HZ,
                                   vamm_list=vamm_list)
    zone_result = dispersion_sweep(function_space, problem, cell_config, zone, maximum_frequency_hz=LIMIT_HZ,
                                   vamm_list=vamm_list)
    path_values = np.where(np.isnan(path_result.frequencies), np.inf, path_result.frequencies)
    zone_values = np.where(np.isnan(zone_result.frequencies), np.inf, zone_result.frequencies)
    number_of_bands = min(path_values.shape[1], zone_values.shape[1])
    finite_maxima = [band for band in range(number_of_bands) if np.isfinite(path_values[:, band].max())]
    maximum_excess = max(zone_values[:, band].max() / path_values[:, band].max() - 1 for band in finite_maxima)
    minimum_excess = max(1 - zone_values[:, band].min() / path_values[:, band].min()
                         for band in range(number_of_bands) if path_values[:, band].min() > 1.0)
    path_gaps = find_complete_gaps(path_result.frequencies, path)
    zone_gaps = find_complete_gaps(zone_result.frequencies, zone)
    print(f"\n{name} ({len(path.wavevectors)} path + {len(zone.wavevectors)} zone wavevectors, "
          f"{(time.perf_counter() - start) / 60:.1f} min)")
    print(f"  interior exceeds a path maximum by at most {100 * maximum_excess:.4f} %, undercuts a path minimum by "
          f"at most {100 * minimum_excess:.4f} %  ({len(finite_maxima)} band maxima below {LIMIT_HZ:.0f} Hz)")
    for gap in path_gaps:
        print("  path: " + describe_band_gap(gap))
    for gap in zone_gaps:
        print("  zone: " + describe_band_gap(gap))
    extremes_on_path = maximum_excess <= tolerance and minimum_excess <= tolerance
    print(f"  -> band extremes {'ON the path' if extremes_on_path else 'NOT all on the path'}")
    return path_gaps, zone_gaps, extremes_on_path


if __name__ == "__main__":
    outcomes = {}
    # A. rectangular cell, one resonator at the centre
    config_a, space_a, problem_a = cell_setup(0.1, 0.15)
    outcomes["A"] = compare_path_and_zone(
        "A. rectangle 0.1 x 0.15 m, one resonator", config_a, space_a, problem_a,
        resonators(config_a, [(0.05, 0.075)]), rectangle_path(0.1, 0.15), build_irreducible_zone_grid(0.1, 0.15))

    config_square, space_square, problem_square = cell_setup(0.1, 0.1)
    # B. two resonators (0, 0) + (a/2, 0)
    outcomes["B"] = compare_path_and_zone(
        "B. square 0.1 m, resonators at (0, 0) and (a/2, 0)", config_square, space_square, problem_square,
        resonators(config_square, [(0.0, 0.0), (0.05, 0.0)]), rectangle_path(0.1, 0.1), quarter_zone_grid(0.1, 0.1))
    # C. two resonators at general positions
    outcomes["C"] = compare_path_and_zone(
        "C. square 0.1 m, resonators at (0.02, 0.03) and (0.07, 0.05) m", config_square, space_square,
        problem_square, resonators(config_square, [(0.02, 0.03), (0.07, 0.05)]), rectangle_path(0.1, 0.1),
        half_zone_grid(0.1, 0.1))

    print("\nSummary")
    for case, (path_gaps, zone_gaps, extremes_on_path) in outcomes.items():
        path_edges = [(round(gap.lower_edge_hz, 1), round(gap.upper_edge_hz, 1)) for gap in path_gaps]
        zone_edges = [(round(gap.lower_edge_hz, 1), round(gap.upper_edge_hz, 1)) for gap in zone_gaps]
        print(f"  {case}: extremes on path: {extremes_on_path}; path gaps {path_edges}; zone gaps {zone_edges}")
