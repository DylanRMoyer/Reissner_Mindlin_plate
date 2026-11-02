"""Bare unit cell (no VAMM): FE Bloch bands vs the analytic folded Reissner-Mindlin dispersion.

Run in fenicsx-env-complex. Checks:
1. SLEPc bands agree with a dense eigensolver (no missed degenerate partners).
2. FE bands match the analytic folded bands at points on Gamma-X-M-Gamma (including points close
   to Gamma) and at a generic k, for a square and a rectangular cell.
3. The production mesh (16 elements per metre) stays within PRODUCTION_MESH_TOLERANCE of the
   analytic bands, and halving the element size reduces the error at close to the expected rate. Q2 eigenvalues converge
   with order 4 asymptotically; full shear integration locks mildly on coarse meshes, so the observed
   order between 16x16 and 32x32 elements per metre is about 3.3 and the meshes are one level finer
   than with the reduced rule.
4. Spurious mode: Lagrange Q2 with the reduced 2x2 shear quadrature has one extra zero-energy
   (hourglass) mode at Gamma whose branch rises steeply out of Gamma. The unit cell therefore uses full
   3x3 shear integration (assemble_unit_cell_matrices default); this check confirms that the extra mode
   exists with 2x2 and is gone with 3x3.
"""
import numpy as np
from scipy.linalg import eigh
from config import PlateConfig
from problem_setup import build_plate_problem
from periodic_bc import build_pairs
from bloch_solver import assemble_unit_cell_matrices, reduce_to_bloch, solve_bloch_evp
from dispersion import plate_constants, analytic_folded_frequencies

NUMBER_OF_BANDS = 10
RELATIVE_TOLERANCE = 5e-3          # FE vs analytic, finest mesh
PRODUCTION_MESH_TOLERANCE = 1e-2   # FE vs analytic, coarse mesh = production density (16 elements per metre)
MINIMUM_CONVERGENCE_ORDER = 3.0    # observed log2(coarse error / fine error); asymptotic theory gives 4
ZERO_FREQUENCY_HZ = 1e-2


def setup_cell(length_x, length_y, elements_x, elements_y, shear_quadrature_degree=None):
    plate_config = PlateConfig(length=length_x, width=length_y, thickness=0.02, nx=elements_x, ny=elements_y,
                               rho=7850, mu=77e9, lambda_=115e9)
    _, function_space, _, problem = build_plate_problem(plate_config, deg=2, el_type="Lagrange")
    stiffness, mass = assemble_unit_cell_matrices(function_space, problem,
                                                  shear_quadrature_degree=shear_quadrature_degree)
    pairs = build_pairs(function_space, length_x, length_y)
    return plate_config, stiffness, mass, pairs


def wavevector_samples(length_x, length_y):
    """Gamma, points on Gamma-X-M-Gamma (including two close to Gamma, where the hourglass branch
    used to mix with the physical bands) and one generic point."""
    gamma, x_point, m_point = np.zeros(2), np.array([np.pi / length_x, 0]), np.array([np.pi / length_x, np.pi / length_y])
    samples = {"Gamma": gamma, "X": x_point, "M": m_point, "generic": np.array([0.3 * np.pi / length_x, 0.7 * np.pi / length_y])}
    for fraction in (0.25, 0.5, 0.75):
        samples[f"Gamma-X {fraction}"] = fraction * x_point
        samples[f"X-M {fraction}"] = x_point + fraction * (m_point - x_point)
        samples[f"M-Gamma {fraction}"] = m_point + fraction * (gamma - m_point)
    for fraction in (0.05, 0.1):
        samples[f"Gamma-X {fraction}"] = fraction * x_point
        samples[f"Gamma-M {fraction}"] = fraction * m_point
    return samples


def compare_cell(length_x, length_y, elements_x, elements_y):
    """Max relative FE-vs-analytic error per wavevector sample (Gamma: over the nonzero bands)."""
    plate_config, stiffness, mass, pairs = setup_cell(length_x, length_y, elements_x, elements_y)
    constants = plate_constants(plate_config)
    errors = {}
    for name, (wavenumber_x, wavenumber_y) in wavevector_samples(length_x, length_y).items():
        stiffness_reduced, mass_reduced = reduce_to_bloch(stiffness, mass, pairs, wavenumber_x, wavenumber_y,
                                                          length_x, length_y)
        fe_frequencies, _ = solve_bloch_evp(stiffness_reduced, mass_reduced, number_of_bands=NUMBER_OF_BANDS + 1)
        analytic_frequencies = analytic_folded_frequencies(wavenumber_x, wavenumber_y, length_x, length_y,
                                                           constants, NUMBER_OF_BANDS + 1)
        if name == "Gamma":
            fe_zero_count = int(np.sum(fe_frequencies < ZERO_FREQUENCY_HZ))
            analytic_zero_count = int(np.sum(analytic_frequencies < ZERO_FREQUENCY_HZ))
            assert fe_zero_count == analytic_zero_count == 1, \
                f"expected only the rigid zero mode, got FE {fe_zero_count} vs analytic {analytic_zero_count}"
            fe_frequencies, analytic_frequencies = fe_frequencies[1:], analytic_frequencies[1:]   # drop rigid mode
        fe_frequencies, analytic_frequencies = fe_frequencies[:NUMBER_OF_BANDS - 1], analytic_frequencies[:NUMBER_OF_BANDS - 1]
        nonzero = analytic_frequencies > ZERO_FREQUENCY_HZ
        absolute_error = np.abs(fe_frequencies - analytic_frequencies)
        assert np.all(absolute_error[~nonzero] < ZERO_FREQUENCY_HZ), (name, fe_frequencies, analytic_frequencies)
        errors[name] = (absolute_error[nonzero] / analytic_frequencies[nonzero]).max()
    return errors


if __name__ == "__main__":
    # 1. SLEPc vs dense solve at a generic k (square cell: degenerate pairs present at symmetric points too)
    _, stiffness, mass, pairs = setup_cell(1.0, 1.0, 4, 4)
    for wavenumber_x, wavenumber_y in [(0.3 * np.pi, 0.7 * np.pi), (np.pi, np.pi), (0.0, 0.0)]:
        stiffness_reduced, mass_reduced = reduce_to_bloch(stiffness, mass, pairs, wavenumber_x, wavenumber_y, 1.0, 1.0)
        slepc_frequencies, _ = solve_bloch_evp(stiffness_reduced, mass_reduced, number_of_bands=NUMBER_OF_BANDS)
        dense_omega_squared = eigh(stiffness_reduced.toarray(), mass_reduced.toarray(), eigvals_only=True)
        dense_frequencies = np.sqrt(np.clip(dense_omega_squared[:NUMBER_OF_BANDS], 0, None)) / (2 * np.pi)
        assert np.allclose(slepc_frequencies, dense_frequencies, atol=1e-3, rtol=1e-8), \
            (wavenumber_x, wavenumber_y, slepc_frequencies, dense_frequencies)
    print("SLEPc matches dense eigh (including degenerate groups)")

    # 4. hourglass mode: two zero modes at Gamma with reduced 2x2 shear integration, one with full 3x3
    for shear_quadrature_degree, expected_zero_modes in [(2, 2), (4, 1)]:
        _, stiffness, mass, pairs = setup_cell(1.0, 1.0, 8, 8, shear_quadrature_degree=shear_quadrature_degree)
        gamma_frequencies, _ = solve_bloch_evp(*reduce_to_bloch(stiffness, mass, pairs, 0.0, 0.0, 1.0, 1.0),
                                               number_of_bands=4)
        zero_mode_count = int(np.sum(gamma_frequencies < ZERO_FREQUENCY_HZ))
        assert zero_mode_count == expected_zero_modes, (shear_quadrature_degree, gamma_frequencies)
    print("hourglass mode present with 2x2 shear quadrature, removed with 3x3")

    # 2. + 3. analytic comparison with refinement, square and rectangular cell
    for length_x, length_y, coarse_elements, fine_elements in [(1.0, 1.0, (16, 16), (32, 32)),
                                                               (1.3, 0.7, (21, 11), (42, 22))]:
        coarse_errors = compare_cell(length_x, length_y, *coarse_elements)
        fine_errors = compare_cell(length_x, length_y, *fine_elements)
        print(f"\ncell {length_x} x {length_y}: max relative error over the lowest {NUMBER_OF_BANDS - 1} bands")
        print(f"{'k point':<16}{str(coarse_elements):>12}{str(fine_elements):>12}{'order':>8}")
        for name in fine_errors:
            convergence_order = np.log2(coarse_errors[name] / fine_errors[name])
            print(f"{name:<16}{coarse_errors[name]:>12.2e}{fine_errors[name]:>12.2e}{convergence_order:>8.2f}")
            assert convergence_order > MINIMUM_CONVERGENCE_ORDER, f"{name}: convergence order {convergence_order:.2f}"
        worst_coarse = max(coarse_errors.values())
        assert worst_coarse < PRODUCTION_MESH_TOLERANCE, f"production mesh error {worst_coarse:.2e} too large"
        worst_fine = max(fine_errors.values())
        assert worst_fine < RELATIVE_TOLERANCE, f"FE vs analytic error {worst_fine:.2e} too large"
    print("\nbare-cell bands match the analytic folded Reissner-Mindlin dispersion")
