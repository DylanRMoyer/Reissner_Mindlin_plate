"""Stage 7: reproduce Fig. 16a and 16c of Cool et al. 2024 (MSSP 215, 111393) with the Reissner-Mindlin Bloch cell.

Cool's cell (Table 1): 0.05 x 0.05 m steel plate, 5 mm thick, E = 210 GPa, nu = 0.3, rho = 7800 kg/m^3, path
O-A-B-O (= Gamma-X-M-Gamma), up to 15 kHz. 16a: bare plate. 16c: one mass-spring resonator at the centre,
f_res = 2500 Hz, mass ratio 0.3 (resonator mass / plate mass per cell).

Model difference: Cool meshes the plate with 3D hexahedral solids (10 x 10 x 3), so his diagrams also contain
the in-plane longitudinal (L) and shear (S) waves. The Reissner-Mindlin cell has only w, theta_x, theta_y, so only
the bending branches can be compared. The paper gives no data tables; the reference values below were read off a
400 dpi render of Fig. 16 (about +-50 Hz), except f_lambda/2, which the paper states (Eq. 16, 4933 Hz).

Mesh: 10 x 10 elements on the 0.05 m cell (200 per metre) gives 0.080 % max bare-cell error below 15 kHz
(16 x 16: 0.013 %, 20 x 20: 0.005 %; order about h^4). Run in fenicsx-env-complex, about 1 min."""
import numpy as np

from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices
from dispersion import (build_wavevector_path, dispersion_sweep, analytic_dispersion, analytic_resonator_dispersion,
                        bare_cell_relative_error, find_complete_gaps, describe_band_gap)
from postprocessing_dispersion import plot_dispersion

# --- Cool et al. Table 1 and Fig. 16 ---
COOL_YOUNGS_MODULUS = 210e9
COOL_POISSON_RATIO = 0.3
COOL_DENSITY = 7800.0
COOL_CELL_LENGTH = 0.05
COOL_THICKNESS = 0.005
COOL_FREQUENCY_LIMIT_HZ = 15000.0
COOL_RESONATOR_FREQUENCY_HZ = 2500.0
COOL_MASS_RATIO = 0.3
COOL_HALF_WAVELENGTH_FREQUENCY_HZ = 4933.0   # Eq. 16 (Kirchhoff), stated in the text
# Read off Fig. 16c (about +-50 Hz): lower band top, upper band bottom (at O), directional gap at A
COOL_FIG16C_READ_HZ = dict(complete_gap_lower=2420.0, complete_gap_upper=2800.0,
                           gap_at_x_lower=4900.0, gap_at_x_upper=5400.0)

ELEMENTS_PER_CELL = 10
POINTS_PER_SEGMENT = 40


def cool_plate_config(elements_per_cell=ELEMENTS_PER_CELL):
    """Cool's Table 1 cell in our parameters: Lame constants from E and nu (3D Lame values; weak_form builds the
    plane-stress D from them, so D = E h^3 / (12 (1 - nu^2)) as in Cool's Eq. 16)."""
    shear_modulus = COOL_YOUNGS_MODULUS / (2 * (1 + COOL_POISSON_RATIO))
    lame_lambda = COOL_YOUNGS_MODULUS * COOL_POISSON_RATIO / ((1 + COOL_POISSON_RATIO) * (1 - 2 * COOL_POISSON_RATIO))
    return PlateConfig(length=COOL_CELL_LENGTH, width=COOL_CELL_LENGTH, thickness=COOL_THICKNESS,
                       nx=elements_per_cell, ny=elements_per_cell, rho=COOL_DENSITY,
                       mu=shear_modulus, lambda_=lame_lambda)


def kirchhoff_half_wavelength_frequency(plate_config, youngs_modulus, poisson_ratio):
    """Cool Eq. 16: frequency at which the Kirchhoff bending wavelength equals 2 Lx (Bragg fold at X)."""
    return 2 * np.pi / (2 * plate_config.length) ** 2 * np.sqrt(
        youngs_modulus * plate_config.thickness ** 2 / (12 * (1 - poisson_ratio ** 2) * plate_config.rho))


def symmetry_point_indices(wavevector_path):
    """Row index of Gamma, X, M on the Gamma-X-M-Gamma path."""
    return {label: int(np.argmin(np.abs(wavevector_path.path_coordinate - position)))
            for label, position in zip(["Gamma", "X", "M"], wavevector_path.symmetry_point_positions[:3])}


def print_comparison(name, ours_hz, cool_hz):
    print(f"  {name:<34s} ours {ours_hz:8.1f} Hz   Cool {cool_hz:8.1f} Hz   difference {100 * (ours_hz / cool_hz - 1):+6.2f} %")


if __name__ == "__main__":
    plate_config = cool_plate_config()
    _, function_space, _, problem = build_plate_problem(plate_config, deg=2, el_type="Lagrange")
    wavevector_path = build_wavevector_path(COOL_CELL_LENGTH, COOL_CELL_LENGTH, points_per_segment=POINTS_PER_SEGMENT)
    point_index = symmetry_point_indices(wavevector_path)
    computed_limit_hz = 1.1 * COOL_FREQUENCY_LIMIT_HZ   # bands that leave the plot do so above its top edge

    # --- Fig. 16a: bare cell ---
    bare_result = dispersion_sweep(function_space, problem, plate_config, wavevector_path,
                                   maximum_frequency_hz=computed_limit_hz)
    bare_analytic = analytic_dispersion(wavevector_path, plate_config, maximum_frequency_hz=1.05 * computed_limit_hz)
    print("\nFig. 16a (bare cell)")
    print(f"  FE vs analytic RM bands: max error {100 * bare_cell_relative_error(bare_result, bare_analytic):.3f} %")
    kirchhoff_fold_hz = kirchhoff_half_wavelength_frequency(plate_config, COOL_YOUNGS_MODULUS, COOL_POISSON_RATIO)
    print_comparison("f_lambda/2, Kirchhoff Eq. 16", kirchhoff_fold_hz, COOL_HALF_WAVELENGTH_FREQUENCY_HZ)
    print_comparison("lowest band at X (A), RM FE", bare_result.frequencies[point_index["X"], 0],
                     COOL_HALF_WAVELENGTH_FREQUENCY_HZ)
    print_comparison("lowest band at M (B), RM FE", bare_result.frequencies[point_index["M"], 0],
                     2 * COOL_HALF_WAVELENGTH_FREQUENCY_HZ)   # Kirchhoff: |k| = sqrt(2) pi / a -> 2 f_lambda/2
    bare_gaps = find_complete_gaps(bare_result.frequencies, wavevector_path)
    print(f"  complete gaps below {computed_limit_hz:.0f} Hz: {len(bare_gaps)} (expected 0)")
    plot_dispersion(result=bare_result, analytic_frequencies=bare_analytic, frequency_limit=COOL_FREQUENCY_LIMIT_HZ,
                    analytic_label="analytic RM (folded plate)",
                    title="Cool et al. 2024, Fig. 16a: bare 0.05 m steel cell, h = 5 mm (bending waves only)",
                    save_path="results/cool2024_fig16a_bare_cell0.05m_0to15000hz.pdf")

    # --- Fig. 16c: resonator at the cell centre ---
    resonator_mass = COOL_MASS_RATIO * plate_config.rho * plate_config.thickness * COOL_CELL_LENGTH ** 2
    resonator_stiffness = resonator_mass * (2 * np.pi * COOL_RESONATOR_FREQUENCY_HZ) ** 2
    vamm_list = create_vamm_list_and_assign_indices(
        [(COOL_CELL_LENGTH / 2, COOL_CELL_LENGTH / 2, resonator_stiffness, resonator_mass)])
    resonator_result = dispersion_sweep(function_space, problem, plate_config, wavevector_path,
                                        maximum_frequency_hz=computed_limit_hz, vamm_list=vamm_list)
    plane_wave_frequencies = analytic_resonator_dispersion(wavevector_path, plate_config, resonator_stiffness,
                                                           resonator_mass, maximum_frequency_hz=computed_limit_hz)
    resonator_gaps = find_complete_gaps(resonator_result.frequencies, wavevector_path)
    plane_wave_gaps = find_complete_gaps(plane_wave_frequencies, wavevector_path)
    print("\nFig. 16c (resonator, f_r = 2500 Hz, mass ratio 0.3)")
    for gap in resonator_gaps:
        print("  FE:  " + describe_band_gap(gap))
    for gap in plane_wave_gaps:
        print("  PWE: " + describe_band_gap(gap))
    first_gap = resonator_gaps[0]
    print_comparison("complete gap, lower edge", first_gap.lower_edge_hz, COOL_FIG16C_READ_HZ["complete_gap_lower"])
    print_comparison("complete gap, upper edge", first_gap.upper_edge_hz, COOL_FIG16C_READ_HZ["complete_gap_upper"])
    x_row = resonator_result.frequencies[point_index["X"]]
    print_comparison("directional gap at X, lower edge", x_row[1], COOL_FIG16C_READ_HZ["gap_at_x_lower"])
    print_comparison("directional gap at X, upper edge", x_row[2], COOL_FIG16C_READ_HZ["gap_at_x_upper"])
    effective_medium_upper_hz = COOL_RESONATOR_FREQUENCY_HZ * np.sqrt(1 + COOL_MASS_RATIO)
    print(f"  effective-medium upper edge f_r sqrt(1 + mass ratio) = {effective_medium_upper_hz:.1f} Hz")
    plot_dispersion(result=resonator_result, analytic_frequencies=plane_wave_frequencies,
                    frequency_limit=COOL_FREQUENCY_LIMIT_HZ, analytic_label="analytic (plane-wave expansion)",
                    bare_frequencies=bare_result.frequencies, resonator_frequencies_hz=[COOL_RESONATOR_FREQUENCY_HZ],
                    band_gaps=resonator_gaps,
                    title="Cool et al. 2024, Fig. 16c: 0.05 m cell with resonator, f_r = 2500 Hz, mass ratio 0.3",
                    save_path="results/cool2024_fig16c_cell0.05m_2500hz_mass0.3_0to15000hz.pdf")
