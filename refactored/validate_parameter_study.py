"""Validation of the stage 7 gap study (parameter_study.py) on the 0.1 m production cell (5 mm steel, 10 x 10 elements).

A. Plane-wave model vs FE at spot points across the map (corners, around the Bragg fold f_r = f_X): every gap wider
   than 1 % in either method has a partner in the other with both edges within 0.5 %.
B. Gamma limit: the upper hybrid band at Gamma tends to the effective-medium value f_r sqrt(1 + mu) as f_r / f_X -> 0.
   The deviation comes from the plate bending locally around the point force inside each cell (a compliance in
   series with the spring); estimate: it scales with mu (f_r / f_X)^2, so deviation / ratio^2 should level off.
C. Small mass ratio: the gap width tends to zero in proportion to mu (no width threshold here).
D. Zone interior near the Bragg fold (f_r / f_X = 0.9 and 1.1, mu = 0.25): FE on a grid over the irreducible zone
   finds no gap edge beyond the path's edges, so the path is enough there too.
E. Scaling: in f / f_X the gaps depend only on mu, f_r / f_X and (Reissner-Mindlin only) h / a; E / rho scales every
   frequency alike and nu is nearly equal. The 0.1 m cell at 10 mm and Cool's 0.05 m cell at 5 mm (both h / a = 0.1,
   different a, E, rho) must give the same normalised edges; the 0.1 m cell at 5 mm (h / a = 0.05) must not.
Run in fenicsx-env-complex, a few minutes."""
import dataclasses
import time

import numpy as np

from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices
from dispersion import (build_wavevector_path, dispersion_sweep, find_complete_gaps, analytic_resonator_frequencies,
                        build_irreducible_zone_grid)
from parameter_study import (bragg_frequency_x, resonator_parameters, cell_band_gaps, LIMIT_OVER_BRAGG,
                             POINTS_PER_SEGMENT)

CELL_LENGTH = 0.1
EDGE_TOLERANCE = 0.005
SPOT_POINTS = [(0.2, 0.05), (0.2, 1.0), (0.5, 0.5), (0.9, 0.25), (1.0, 0.25), (1.1, 0.25), (2.5, 0.05), (2.5, 1.0)]


def build_cell():
    config = PlateConfig(length=CELL_LENGTH, width=CELL_LENGTH, thickness=0.005, nx=10, ny=10, rho=7850,
                         mu=77e9, lambda_=115e9)
    _, function_space, _, problem = build_plate_problem(config, deg=2, el_type="Lagrange")
    return config, function_space, problem


def matched(gaps, partners, minimum_relative_width=0.01):
    """Every gap of 'gaps' wider than minimum_relative_width has a partner with both edges within EDGE_TOLERANCE;
    returns the worst relative edge difference among the matched pairs."""
    worst = 0.0
    for gap in gaps:
        if gap.relative_width < minimum_relative_width:
            continue
        differences = [max(abs(partner.lower_edge_hz / gap.lower_edge_hz - 1),
                           abs(partner.upper_edge_hz / gap.upper_edge_hz - 1)) for partner in partners]
        assert differences and min(differences) < EDGE_TOLERANCE, \
            f"gap {gap.lower_edge_hz:.1f}-{gap.upper_edge_hz:.1f} Hz has no partner within {EDGE_TOLERANCE:.1%}"
        worst = max(worst, min(differences))
    return worst


def check_plane_wave_against_fe(config, function_space, problem, path, bragg_hz):
    print("A. plane-wave model vs FE at spot points")
    limit_hz = LIMIT_OVER_BRAGG * bragg_hz
    for frequency_ratio, mass_ratio in SPOT_POINTS:
        resonator_frequency_hz = frequency_ratio * bragg_hz
        plane_wave = cell_band_gaps(config, path, resonator_frequency_hz, mass_ratio, limit_hz)
        finite_element = cell_band_gaps(config, path, resonator_frequency_hz, mass_ratio, limit_hz, method="fe",
                                        function_space=function_space, problem=problem)
        worst = max(matched(plane_wave, finite_element), matched(finite_element, plane_wave))
        summary = ", ".join(f"{gap.lower_edge_hz:.0f}-{gap.upper_edge_hz:.0f} ({gap.relative_width:.1%})"
                            for gap in finite_element) or "none"
        print(f"   f_r/f_X = {frequency_ratio:.2f}, mu = {mass_ratio:g}: FE gaps {summary} Hz; "
              f"PWE {len(plane_wave)} gaps; worst edge difference {100 * worst:.3f} %", flush=True)


def check_gamma_limit(config, bragg_hz, mass_ratio=0.25):
    print(f"B. upper hybrid band at Gamma vs f_r sqrt(1 + mu), mu = {mass_ratio}")
    deviations = []
    for frequency_ratio in (0.05, 0.1, 0.2, 0.4, 0.8):
        resonator_frequency_hz = frequency_ratio * bragg_hz
        stiffness, mass = resonator_parameters(config, resonator_frequency_hz, mass_ratio)
        bands = analytic_resonator_frequencies(0.0, 0.0, config, stiffness, mass,
                                               maximum_frequency_hz=2 * resonator_frequency_hz)
        deviation = 1 - bands[1] / (resonator_frequency_hz * np.sqrt(1 + mass_ratio))
        deviations.append(deviation)
        print(f"   f_r/f_X = {frequency_ratio:.2f}: {bands[1]:.2f} Hz, below the estimate by {100 * deviation:.4f} %, "
              f"deviation / ratio^2 = {deviation / frequency_ratio ** 2:.4f}")
    assert np.all(np.diff(deviations) > 0), "deviation should grow with f_r / f_X"
    assert deviations[0] < 1e-3, "effective-medium limit not reached at f_r / f_X = 0.05"


def check_small_mass_ratio(config, path, bragg_hz, frequency_ratio=0.5):
    print(f"C. gap width for small mass ratios, f_r/f_X = {frequency_ratio}")
    resonator_frequency_hz = frequency_ratio * bragg_hz
    widths_per_mass_ratio = []
    for mass_ratio in (0.005, 0.01, 0.02):
        gaps = cell_band_gaps(config, path, resonator_frequency_hz, mass_ratio, LIMIT_OVER_BRAGG * bragg_hz,
                              minimum_relative_width=0.0)
        gap = min(gaps, key=lambda gap: abs(gap.centre_hz - resonator_frequency_hz))
        widths_per_mass_ratio.append((gap.upper_edge_hz - gap.lower_edge_hz) / (mass_ratio * resonator_frequency_hz))
        print(f"   mu = {mass_ratio}: gap {gap.lower_edge_hz:.2f}-{gap.upper_edge_hz:.2f} Hz, "
              f"width / (mu f_r) = {widths_per_mass_ratio[-1]:.4f}")
    spread = max(widths_per_mass_ratio) / min(widths_per_mass_ratio) - 1
    print(f"   spread of width / (mu f_r): {100 * spread:.1f} %")
    assert spread < 0.05, "gap width not proportional to the mass ratio"


def check_zone_interior(config, function_space, problem, path, bragg_hz, mass_ratio=0.25):
    print("D. zone interior near the Bragg fold (FE)")
    zone_grid = build_irreducible_zone_grid(CELL_LENGTH, CELL_LENGTH, points_per_side=16)
    for frequency_ratio in (0.9, 1.1):
        resonator_frequency_hz = frequency_ratio * bragg_hz
        stiffness, mass = resonator_parameters(config, resonator_frequency_hz, mass_ratio)
        vamm_list = create_vamm_list_and_assign_indices([(CELL_LENGTH / 2, CELL_LENGTH / 2, stiffness, mass)])
        limit_hz = LIMIT_OVER_BRAGG * bragg_hz
        path_gaps = find_complete_gaps(dispersion_sweep(function_space, problem, config, path, maximum_frequency_hz=limit_hz,
                                                        vamm_list=vamm_list).frequencies, path)
        zone_gaps = find_complete_gaps(dispersion_sweep(function_space, problem, config, zone_grid,
                                                        maximum_frequency_hz=limit_hz, vamm_list=vamm_list).frequencies,
                                       zone_grid)
        for path_gap in path_gaps:
            zone_matches = [gap for gap in zone_gaps if gap.lower_band_index == path_gap.lower_band_index]
            assert zone_matches, f"gap above band {path_gap.lower_band_index + 1} closes inside the zone"
            zone_gap = zone_matches[0]
            print(f"   f_r/f_X = {frequency_ratio}: path {path_gap.lower_edge_hz:.2f}-{path_gap.upper_edge_hz:.2f} Hz, "
                  f"zone {zone_gap.lower_edge_hz:.2f}-{zone_gap.upper_edge_hz:.2f} Hz", flush=True)
            assert zone_gap.lower_edge_hz <= path_gap.lower_edge_hz * (1 + 1e-6)
            assert zone_gap.upper_edge_hz >= path_gap.upper_edge_hz * (1 - 1e-6)
        if not path_gaps:
            print(f"   f_r/f_X = {frequency_ratio}: no complete gap on the path")


def check_thickness_ratio_scaling(config, points=((0.5, 0.5), (1.5, 0.5), (2.5, 1.0)), tolerance=5e-4):
    from compare_cool_2024 import cool_plate_config
    print("E. scaling with h / a: normalised gap edges of three cells")
    cells = {"0.1 m, 5 mm (h/a 0.05)": config,
             "0.1 m, 10 mm (h/a 0.10)": dataclasses.replace(config, thickness=0.010),
             "0.05 m Cool, 5 mm (h/a 0.10)": cool_plate_config()}
    largest_difference_thin = 0.0
    for frequency_ratio, mass_ratio in points:
        normalised = {}
        for name, cell in cells.items():
            bragg_hz = bragg_frequency_x(cell)
            cell_path = build_wavevector_path(cell.length, cell.width, points_per_segment=POINTS_PER_SEGMENT)
            gaps = cell_band_gaps(cell, cell_path, frequency_ratio * bragg_hz, mass_ratio, LIMIT_OVER_BRAGG * bragg_hz)
            normalised[name] = np.array([(gap.lower_edge_hz, gap.upper_edge_hz) for gap in gaps]) / bragg_hz
            print(f"   f_r/f_X = {frequency_ratio}, mu = {mass_ratio}, {name}: "
                  + ", ".join(f"{lower:.4f}-{upper:.4f}" for lower, upper in normalised[name]), flush=True)
        thin, thick_ours, thick_cool = (normalised[name] for name in cells)
        assert thick_ours.shape == thick_cool.shape and np.allclose(thick_ours, thick_cool, rtol=tolerance), \
            "cells with equal h / a do not collapse"
        if thin.shape == thick_ours.shape:
            largest_difference_thin = max(largest_difference_thin, np.max(np.abs(thin / thick_ours - 1)))
    print(f"   equal h/a: edges agree within {tolerance:.0e}; h/a 0.05 vs 0.10: up to {100 * largest_difference_thin:.1f} %")
    assert largest_difference_thin > 10 * tolerance, "h / a should matter in Reissner-Mindlin"


if __name__ == "__main__":
    start = time.time()
    config, function_space, problem = build_cell()
    path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=POINTS_PER_SEGMENT)
    bragg_hz = bragg_frequency_x(config)
    print(f"0.1 m cell: f_X = {bragg_hz:.1f} Hz")
    check_gamma_limit(config, bragg_hz)
    check_small_mass_ratio(config, path, bragg_hz)
    check_plane_wave_against_fe(config, function_space, problem, path, bragg_hz)
    check_zone_interior(config, function_space, problem, path, bragg_hz)
    check_thickness_ratio_scaling(config)
    print(f"All checks passed ({(time.time() - start) / 60:.1f} min)")
