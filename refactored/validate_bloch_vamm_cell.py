"""Unit cell with one point resonator (VAMM): structural, limit and position-invariance checks.

Run in fenicsx-env-complex. Production cell: 0.1 m, h = 5 mm steel, 10x10 elements, bands below 5 kHz.
1. Structure: reduced matrices Hermitian, resonator entry of M_r equals m, SLEPc equals dense eigh.
   (validate_augmented_system already runs inside assemble_unit_cell_matrices.)
2. Limits: mass -> 0 (resonator frequency leaves the range, the massless DOF follows the plate) and
   spring -> 0 (resonator decouples: bare bands plus a flat band at f_r) both give the bare bands back,
   and the deviation shrinks in proportion to the small parameter.
3. Position invariance: moving the resonator inside the cell only shifts the lattice origin, so the bands
   must not change. Nodes (centre, corners, edges incl. slave DOFs on the right/top boundary) agree to
   round-off because the mesh repeats every element; a point inside an element agrees to discretization
   accuracy only.
4. Independent analytic reference: FE bands match the plane-wave expansion (Reissner-Mindlin, one point
   resonator per cell) within the production tolerance. Kirchhoff differs by ~1.7 % near 5 kHz (no shear,
   no rotary inertia) and is only reported.
5. Attachment resolution: doubling the mesh (10x10 -> 20x20) changes the bands by less than the production
   tolerance. The resonator-dominated band at Gamma drifts slowly with refinement (1273.5, 1271.7, 1270.5 Hz
   for 10, 20, 40 elements), the logarithmic point-force singularity of Reissner-Mindlin; the plane-wave
   model shows the same drift with its harmonic cutoff.
"""
import numpy as np
from scipy.linalg import eigh
from config import PlateConfig
from problem_setup import build_plate_problem
from periodic_bc import build_pairs
from point_coupling import create_vamm_list_and_assign_indices
from bloch_solver import assemble_unit_cell_matrices, reduce_to_bloch, solve_bloch_evp
from dispersion import (build_wavevector_path, dispersion_sweep, unit_cell_element_counts,
                        analytic_resonator_dispersion, bare_cell_relative_error)

CELL_LENGTH, THICKNESS, DENSITY = 0.1, 0.005, 7850
MAXIMUM_FREQUENCY_HZ = 5000.0
RESONATOR_FREQUENCY_HZ = 1200.0
MASS_RATIO = 0.25                                            # resonator mass / plate mass per cell
CELL_PLATE_MASS = DENSITY * THICKNESS * CELL_LENGTH ** 2     # rho h a^2, about 0.39 kg
RESONATOR_MASS = MASS_RATIO * CELL_PLATE_MASS
RESONATOR_STIFFNESS = RESONATOR_MASS * (2 * np.pi * RESONATOR_FREQUENCY_HZ) ** 2
NODE_TOLERANCE = 1e-8                                       # relative, positions on mesh nodes
PRODUCTION_TOLERANCE = 1e-2
INSIDE_ELEMENT_TOLERANCE = PRODUCTION_TOLERANCE


def resonator_list(x, y, stiffness=RESONATOR_STIFFNESS, mass=RESONATOR_MASS):
    return create_vamm_list_and_assign_indices([(x, y, stiffness, mass)])


def band_difference(frequencies, reference_frequencies):
    """Max relative difference between two NaN-padded band arrays (same NaN pattern required)."""
    assert frequencies.shape == reference_frequencies.shape, (frequencies.shape, reference_frequencies.shape)
    assert np.array_equal(np.isnan(frequencies), np.isnan(reference_frequencies)), "band counts differ"
    finite = ~np.isnan(frequencies) & (reference_frequencies > 1.0)
    return np.max(np.abs(frequencies[finite] - reference_frequencies[finite]) / reference_frequencies[finite])


if __name__ == "__main__":
    elements_x, elements_y = unit_cell_element_counts(CELL_LENGTH, CELL_LENGTH)
    plate_config = PlateConfig(length=CELL_LENGTH, width=CELL_LENGTH, thickness=THICKNESS, nx=elements_x,
                               ny=elements_y, rho=DENSITY, mu=77e9, lambda_=115e9)
    _, function_space, _, problem = build_plate_problem(plate_config, deg=2, el_type="Lagrange")
    path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=8)

    def sweep(vamm_list=None):
        return dispersion_sweep(function_space=function_space, problem=problem, plate_config=plate_config,
                                wavevector_path=path, maximum_frequency_hz=MAXIMUM_FREQUENCY_HZ,
                                vamm_list=vamm_list).frequencies

    # 1. structure at a generic wavevector
    stiffness, mass = assemble_unit_cell_matrices(function_space, problem, vamm_list=resonator_list(0.05, 0.05))
    pairs = build_pairs(function_space, CELL_LENGTH, CELL_LENGTH)
    stiffness_reduced, mass_reduced = reduce_to_bloch(stiffness, mass, pairs, 0.3 * np.pi / CELL_LENGTH,
                                                      0.7 * np.pi / CELL_LENGTH, CELL_LENGTH, CELL_LENGTH,
                                                      number_of_extra_dofs=1)
    for name, matrix in (("K_r", stiffness_reduced), ("M_r", mass_reduced)):
        hermitian_error = abs(matrix - matrix.conj().T).max() / abs(matrix).max()
        assert hermitian_error < 1e-12, (name, hermitian_error)
    assert np.isclose(mass_reduced[-1, -1], RESONATOR_MASS, rtol=1e-12)
    slepc_frequencies, _ = solve_bloch_evp(stiffness_reduced, mass_reduced, number_of_bands=8)
    dense_frequencies = np.sqrt(np.clip(eigh(stiffness_reduced.toarray(), mass_reduced.toarray(),
                                             eigvals_only=True)[:8], 0, None)) / (2 * np.pi)
    assert np.allclose(slepc_frequencies, dense_frequencies, rtol=1e-8, atol=1e-6), (slepc_frequencies, dense_frequencies)
    print("structure: K_r, M_r Hermitian, resonator mass on the diagonal, SLEPc matches dense eigh")

    # 2. limits
    bare_frequencies = sweep()
    previous_deviation = None
    for mass_fraction in (1e-3, 1e-5):
        tiny_mass = mass_fraction * CELL_PLATE_MASS        # same spring: resonator frequency far above 5 kHz
        deviation = band_difference(sweep(resonator_list(0.05, 0.05, mass=tiny_mass)), bare_frequencies)
        print(f"mass -> 0: m = {mass_fraction:.0e} x rho h a^2, max deviation from bare {deviation:.2e}")
        if previous_deviation is not None:
            assert deviation < 0.05 * previous_deviation, "deviation does not shrink with the mass"
        previous_deviation = deviation
    assert deviation < 1e-4                                 # linear scaling above is the real check

    previous_deviation = None
    for decoupled_frequency_hz in (1.0, 0.1):
        tiny_stiffness = RESONATOR_MASS * (2 * np.pi * decoupled_frequency_hz) ** 2
        frequencies = sweep(resonator_list(0.05, 0.05, stiffness=tiny_stiffness))
        resonator_band, plate_bands = frequencies[:, :1], frequencies[:, 1:]
        assert np.all(resonator_band < 2 * decoupled_frequency_hz), "flat resonator band missing"
        # Gamma: the bare zero mode becomes resonator band 0 and the rigid translation at f_r sqrt(1 + mu)
        plate_bands_without_gamma = plate_bands[1:-1]
        bare_without_gamma = bare_frequencies[1:-1, :plate_bands.shape[1]]
        deviation = band_difference(plate_bands_without_gamma, bare_without_gamma)
        print(f"spring -> 0: f_r = {decoupled_frequency_hz} Hz, flat band below {resonator_band.max():.3f} Hz, "
              f"max deviation of the other bands from bare {deviation:.2e}")
        if previous_deviation is not None:
            assert deviation < 0.05 * previous_deviation, "deviation does not shrink with the stiffness"
        previous_deviation = deviation
    assert deviation < 1e-4                                 # linear scaling above is the real check
    print("limits: mass -> 0 and spring -> 0 recover the bare bands")

    # 3. position invariance (reference: cell centre)
    reference_frequencies = sweep(resonator_list(0.05, 0.05))
    positions = {"corner (0, 0), master DOF": (0.0, 0.0),
                 "corner (a, a), slave DOF": (CELL_LENGTH, CELL_LENGTH),
                 "bottom edge (a/2, 0)": (0.05, 0.0),
                 "right edge (a, a/2), slave DOF": (CELL_LENGTH, 0.05),
                 "interior node (0.03, 0.07)": (0.03, 0.07),
                 "inside an element (0.0337, 0.0712)": (0.0337, 0.0712)}
    for name, (x, y) in positions.items():
        difference = band_difference(sweep(resonator_list(x, y)), reference_frequencies)
        tolerance = INSIDE_ELEMENT_TOLERANCE if "inside" in name else NODE_TOLERANCE
        print(f"position {name:<36} max relative difference to centre {difference:.2e}")
        assert difference < tolerance, (name, difference)
    print("position invariance: bands independent of the resonator position in the cell")

    # 4. plane-wave expansion (bare_cell_relative_error compares sorted bands with a margin at the limit)
    resonator_result = dispersion_sweep(function_space=function_space, problem=problem, plate_config=plate_config,
                                        wavevector_path=path, maximum_frequency_hz=MAXIMUM_FREQUENCY_HZ,
                                        vamm_list=resonator_list(0.05, 0.05))
    for theory in ("reissner_mindlin", "kirchhoff"):
        analytic_frequencies = analytic_resonator_dispersion(path, plate_config, RESONATOR_STIFFNESS, RESONATOR_MASS,
                                                             1.05 * MAXIMUM_FREQUENCY_HZ, theory=theory)
        error = bare_cell_relative_error(resonator_result, analytic_frequencies)
        print(f"plane-wave expansion ({theory}): max relative difference to FE {error:.2e}")
        if theory == "reissner_mindlin":
            assert error < PRODUCTION_TOLERANCE, error
    print("FE resonator cell matches the Reissner-Mindlin plane-wave expansion")

    # 5. attachment resolution: 10x10 vs 20x20
    fine_config = PlateConfig(length=CELL_LENGTH, width=CELL_LENGTH, thickness=THICKNESS, nx=2 * elements_x,
                              ny=2 * elements_y, rho=DENSITY, mu=77e9, lambda_=115e9)
    _, fine_function_space, _, fine_problem = build_plate_problem(fine_config, deg=2, el_type="Lagrange")
    fine_frequencies = dispersion_sweep(function_space=fine_function_space, problem=fine_problem,
                                        plate_config=fine_config, wavevector_path=path,
                                        maximum_frequency_hz=MAXIMUM_FREQUENCY_HZ,
                                        vamm_list=resonator_list(0.05, 0.05)).frequencies
    difference = band_difference(resonator_result.frequencies, fine_frequencies)
    print(f"mesh 10x10 vs 20x20 with resonator: max relative difference {difference:.2e}")
    assert difference < PRODUCTION_TOLERANCE, difference
    print("attachment resolved: 100 elements per metre within the production tolerance")
