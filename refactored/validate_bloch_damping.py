"""Why the Bloch eigenproblem is solved undamped: quantitative check of the damping effect on the band positions.

Run in fenicsx-env-complex (a few minutes). Production cell: 0.1 m, h = 5 mm steel, 10x10 Lagrange Q2, one
resonator at the centre (f_r = 1.2 kHz, mass ratio 0.25), bands below 5 kHz. Hysteretic damping as in the
forced-response pipeline: plate stiffness K_plate (1 + i gamma_plate), resonator spring (1 + i gamma_resonator).

1. Matrix split: the damped augmented stiffness is K_plate (1 + i gamma_plate) + K_spring (1 + i gamma_resonator),
   with K_spring = (undamped augmented K) - (plate K padded with the resonator DOF). Checked against the pipeline's
   augment_stiffness_matrix with a damped resonator.
2. Bare cell, plate damping only (exact limit): K (1 + i gamma) and M have the eigenvalues (1 + i gamma) lambda_0
   with unchanged mode shapes, so Re f / f_0 = Re sqrt(1 + i gamma) = 1 + gamma^2/8 + O(gamma^4) for every mode.
   Also checks the non-Hermitian solve itself.
3. Solver: sparse shift-invert (ARPACK) agrees with the dense non-Hermitian solve at two wavevectors.
4. Sweep over the whole path, five damping cases: largest shift of Re f against the undamped band, where it occurs,
   and the plate share of that mode.
5. Scaling: the shift grows with gamma^2 (ratio 4 for plate gamma 0.04 vs 0.02, ratio 25 for resonator gamma
   0.10 vs 0.02).
6. Explanation of shifts above gamma^2/8: with a resonator, damping only the plate (or only the spring) is no
   longer proportional damping. Second-order perturbation theory with the damping matrix D (H = undamped + i D):
       lambda ~ lambda_0 + i D_nn - sum_{m != n} |D_mn|^2 / (lambda_0n - lambda_0m),
   whose real part is largest where two modes with different damping are close (avoided crossings). The
   prediction must match the solve at the worst wavevector of each case.
"""
import time
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs
from config import PlateConfig
from problem_setup import build_plate_problem
from periodic_bc import build_pairs
from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
from augmented_system import petsc_to_scipy, scipy_to_petsc, augment_stiffness_matrix
from bloch_solver import assemble_unit_cell_matrices, reduce_to_bloch, solve_bloch_evp
from dispersion import build_wavevector_path, unit_cell_element_counts

CELL_LENGTH, THICKNESS, DENSITY = 0.1, 0.005, 7850
MAXIMUM_FREQUENCY_HZ = 5000.0
RESONATOR_FREQUENCY_HZ = 1200.0
MASS_RATIO = 0.25
CELL_PLATE_MASS = DENSITY * THICKNESS * CELL_LENGTH ** 2
RESONATOR_MASS = MASS_RATIO * CELL_PLATE_MASS
RESONATOR_STIFFNESS = RESONATOR_MASS * (2 * np.pi * RESONATOR_FREQUENCY_HZ) ** 2
DAMPING_CASES = [(0.02, 0.0), (0.04, 0.0), (0.0, 0.02), (0.0, 0.10), (0.04, 0.02)]   # (plate gamma, resonator gamma)
NUMBER_OF_MODES = 20                     # more than the bands below 5 kHz at every wavevector
SHIFT_TARGET = -1.0                      # slightly negative omega^2 target, as in solve_bloch_evp
SPLIT_TOLERANCE = 1e-12
BARE_EXACTNESS_TOLERANCE = 1e-8
SOLVER_AGREEMENT_TOLERANCE = 1e-8
SCALING_TOLERANCE = 0.02                 # relative, on the median ratio
PERTURBATION_TOLERANCE = 0.05            # relative to the measured worst shift
PRODUCTION_TOLERANCE = 1e-2              # mesh tolerance of the band positions


def relative_frequency_shift(eigenvalues_damped, eigenvalues_undamped):
    """Signed Re f / f_0 - 1 for damped eigenvalues (complex omega^2) against undamped ones, matched by order."""
    frequencies_damped = np.sqrt(eigenvalues_damped.astype(complex))
    frequencies_damped = np.where(frequencies_damped.real < 0, -frequencies_damped, frequencies_damped)
    return frequencies_damped.real / np.sqrt(eigenvalues_undamped) - 1


def sorted_by_real_frequency(eigenvalues):
    frequencies = np.sqrt(eigenvalues.astype(complex))
    return eigenvalues[np.argsort(np.abs(frequencies.real))]


def damped_eigenvalues_sparse(stiffness_reduced, mass_reduced, number_of_modes=NUMBER_OF_MODES):
    """Non-Hermitian generalized problem by ARPACK shift-invert near omega^2 = 0, sorted by Re f."""
    eigenvalues = eigs(stiffness_reduced.tocsc(), k=number_of_modes, M=mass_reduced.tocsc(), sigma=SHIFT_TARGET,
                       which="LM", return_eigenvectors=False)
    return sorted_by_real_frequency(eigenvalues)


def damped_eigenvalues_dense(stiffness_reduced, mass_reduced):
    eigenvalues = eig(stiffness_reduced.toarray(), mass_reduced.toarray(), right=False)
    return sorted_by_real_frequency(eigenvalues[np.isfinite(eigenvalues)])


def plate_share(eigenvector, mass_reduced):
    return 1 - RESONATOR_MASS * abs(eigenvector[-1]) ** 2 / np.vdot(mass_reduced @ eigenvector, eigenvector).real


def pad_with_resonator_dof(plate_matrix):
    """Plate matrix with one extra zero row and column for the resonator DOF (appended last)."""
    return sparse.block_diag([plate_matrix, sparse.csr_matrix((1, 1))], format="csr")


def damped_stiffness(plate_stiffness_padded, spring_stiffness, plate_gamma, resonator_gamma):
    return (plate_stiffness_padded * (1 + 1j * plate_gamma) + spring_stiffness * (1 + 1j * resonator_gamma)).tocsr()


def perturbation_shift(eigenvalues_undamped, eigenvectors_undamped, damping_matrix_reduced, band_indices):
    """Second-order prediction of Re f / f_0 - 1 for the undamped problem plus i * damping matrix.
    Eigenvectors are M_r-normalized columns of the full undamped spectrum."""
    coupling = eigenvectors_undamped.conj().T @ (damping_matrix_reduced @ eigenvectors_undamped)
    predictions = []
    for n in band_indices:
        others = np.abs(eigenvalues_undamped - eigenvalues_undamped[n]) > 1e-8 * eigenvalues_undamped[n]
        second_order = -np.sum(np.abs(coupling[others, n]) ** 2 / (eigenvalues_undamped[n] - eigenvalues_undamped[others]))
        eigenvalue = eigenvalues_undamped[n] + 1j * coupling[n, n].real + second_order
        predictions.append(np.sqrt(eigenvalue).real / np.sqrt(eigenvalues_undamped[n]) - 1)
    return np.array(predictions)


def wavevector_label(wavenumber_x, wavenumber_y):
    return f"({wavenumber_x * CELL_LENGTH / np.pi:.3f}, {wavenumber_y * CELL_LENGTH / np.pi:.3f}) pi/a"


if __name__ == "__main__":
    start_time = time.time()
    elements_x, elements_y = unit_cell_element_counts(CELL_LENGTH, CELL_LENGTH)
    plate_config = PlateConfig(length=CELL_LENGTH, width=CELL_LENGTH, thickness=THICKNESS, nx=elements_x,
                               ny=elements_y, rho=DENSITY, mu=77e9, lambda_=115e9)
    _, function_space, _, problem = build_plate_problem(plate_config, deg=2, el_type="Lagrange")
    pairs = build_pairs(function_space, CELL_LENGTH, CELL_LENGTH)
    path = build_wavevector_path(CELL_LENGTH, CELL_LENGTH, points_per_segment=20)

    # --- 1. Matrix split into plate part and spring part
    resonators = create_vamm_list_and_assign_indices([(CELL_LENGTH / 2, CELL_LENGTH / 2, RESONATOR_STIFFNESS, RESONATOR_MASS)])
    plate_stiffness, plate_mass = assemble_unit_cell_matrices(function_space, problem)
    augmented_stiffness, augmented_mass = assemble_unit_cell_matrices(function_space, problem, vamm_list=resonators)
    plate_stiffness_padded = pad_with_resonator_dof(plate_stiffness)
    spring_stiffness = (augmented_stiffness - plate_stiffness_padded).tocsr()
    assert np.isclose(spring_stiffness[-1, -1], RESONATOR_STIFFNESS, rtol=1e-12)

    test_gamma = 0.02
    damped_resonators = create_vamm_list_and_assign_indices(
        [(CELL_LENGTH / 2, CELL_LENGTH / 2, RESONATOR_STIFFNESS, RESONATOR_MASS, test_gamma)])
    phi_list, dofs_list, _ = compute_phi_and_dofs_for_vamm_list(domain=function_space.mesh, function_space=function_space,
                                                                vamm_list=damped_resonators)
    pipeline_stiffness_petsc, _ = augment_stiffness_matrix(K=scipy_to_petsc(plate_stiffness), vamm_list=damped_resonators,
                                                           phi_list=phi_list, global_dofs_parent_list=dofs_list)
    pipeline_stiffness = petsc_to_scipy(pipeline_stiffness_petsc).tocsr()
    split_stiffness = damped_stiffness(plate_stiffness_padded, spring_stiffness, plate_gamma=0.0, resonator_gamma=test_gamma)
    split_error = sparse.linalg.norm(split_stiffness - pipeline_stiffness) / sparse.linalg.norm(pipeline_stiffness)
    print(f"1. split K_plate + K_spring (1 + i gamma) vs pipeline augmentation: relative difference {split_error:.1e}", flush=True)
    assert split_error < SPLIT_TOLERANCE

    # --- 2. Bare cell, plate damping only: exact limit
    bare_gamma = 0.04
    exact_shift = np.sqrt(1 + 1j * bare_gamma).real - 1
    print(f"2. bare cell, plate gamma {bare_gamma}: exact Re sqrt(1 + i gamma) - 1 = {exact_shift:.4e} "
          f"(gamma^2/8 = {bare_gamma ** 2 / 8:.4e})", flush=True)
    check_points = [path.wavevectors[len(path.wavevectors) // 3], path.wavevectors[-len(path.wavevectors) // 4]]
    for wavenumber_x, wavenumber_y in check_points:
        stiffness_reduced, mass_reduced = reduce_to_bloch(plate_stiffness, plate_mass, pairs, wavenumber_x, wavenumber_y,
                                                          CELL_LENGTH, CELL_LENGTH)
        undamped = eigh(stiffness_reduced.toarray(), mass_reduced.toarray(), eigvals_only=True)
        damped = damped_eigenvalues_dense(stiffness_reduced * (1 + 1j * bare_gamma), mass_reduced)
        in_range = (undamped > (2 * np.pi) ** 2) & (undamped < (2 * np.pi * MAXIMUM_FREQUENCY_HZ) ** 2)
        real_error = np.max(np.abs(damped.real[in_range] / undamped[in_range] - 1))
        imaginary_error = np.max(np.abs(damped.imag[in_range] / (bare_gamma * undamped[in_range]) - 1))
        shift_error = np.max(np.abs(relative_frequency_shift(damped[in_range], undamped[in_range]) - exact_shift))
        print(f"   k = {wavevector_label(wavenumber_x, wavenumber_y)}, {in_range.sum()} modes: "
              f"|Re lambda / lambda_0 - 1| <= {real_error:.1e}, |Im lambda / (gamma lambda_0) - 1| <= {imaginary_error:.1e}, "
              f"shift - exact <= {shift_error:.1e}  ({time.time() - start_time:.0f} s)", flush=True)
        assert max(real_error, imaginary_error) < BARE_EXACTNESS_TOLERANCE
        assert shift_error < BARE_EXACTNESS_TOLERANCE

    # --- 3. Sparse shift-invert against the dense non-Hermitian solve
    combined_stiffness = damped_stiffness(plate_stiffness_padded, spring_stiffness, plate_gamma=0.04, resonator_gamma=0.02)
    for wavenumber_x, wavenumber_y in check_points:
        stiffness_reduced, mass_reduced = reduce_to_bloch(combined_stiffness, augmented_mass, pairs, wavenumber_x,
                                                          wavenumber_y, CELL_LENGTH, CELL_LENGTH, number_of_extra_dofs=1)
        sparse_values = damped_eigenvalues_sparse(stiffness_reduced, mass_reduced)
        dense_values = damped_eigenvalues_dense(stiffness_reduced, mass_reduced)[:NUMBER_OF_MODES]
        in_range = np.sqrt(np.abs(dense_values)) < 2 * np.pi * MAXIMUM_FREQUENCY_HZ
        agreement = np.max(np.abs(sparse_values[in_range] - dense_values[in_range]) / np.abs(dense_values[in_range]))
        print(f"3. k = {wavevector_label(wavenumber_x, wavenumber_y)}: sparse vs dense, {in_range.sum()} modes, "
              f"max relative difference {agreement:.1e}  ({time.time() - start_time:.0f} s)", flush=True)
        assert agreement < SOLVER_AGREEMENT_TOLERANCE

    # --- 4. Sweep over the whole path
    damped_matrices = {case: damped_stiffness(plate_stiffness_padded, spring_stiffness, *case) for case in DAMPING_CASES}
    shifts = {case: [] for case in DAMPING_CASES}
    worst = {case: dict(shift=0.0) for case in DAMPING_CASES}
    for point_index, (wavenumber_x, wavenumber_y) in enumerate(path.wavevectors):
        stiffness_reduced, mass_reduced = reduce_to_bloch(augmented_stiffness, augmented_mass, pairs, wavenumber_x,
                                                          wavenumber_y, CELL_LENGTH, CELL_LENGTH, number_of_extra_dofs=1)
        frequencies_undamped, eigenvectors_undamped = solve_bloch_evp(stiffness_reduced, mass_reduced,
                                                                      number_of_bands=NUMBER_OF_MODES)
        band_indices = np.flatnonzero((frequencies_undamped > 1.0) & (frequencies_undamped < MAXIMUM_FREQUENCY_HZ))
        eigenvalues_undamped = (2 * np.pi * frequencies_undamped[band_indices]) ** 2
        for case, stiffness in damped_matrices.items():
            damped_reduced, _ = reduce_to_bloch(stiffness, augmented_mass, pairs, wavenumber_x, wavenumber_y,
                                                CELL_LENGTH, CELL_LENGTH, number_of_extra_dofs=1)
            damped_values = damped_eigenvalues_sparse(damped_reduced, mass_reduced)[band_indices]
            case_shifts = relative_frequency_shift(damped_values, eigenvalues_undamped)
            shifts[case].append(case_shifts)
            largest = int(np.argmax(np.abs(case_shifts)))
            if abs(case_shifts[largest]) > abs(worst[case]["shift"]):
                worst[case] = dict(shift=case_shifts[largest], wavevector=(wavenumber_x, wavenumber_y),
                                   band=int(band_indices[largest]), frequency=frequencies_undamped[band_indices[largest]],
                                   plate_share=plate_share(eigenvectors_undamped[:, band_indices[largest]], mass_reduced))
        if point_index % 10 == 0:
            print(f"   sweep {point_index + 1}/{len(path.wavevectors)}  ({time.time() - start_time:.0f} s)", flush=True)

    print("4. largest shift of Re f over the path (bands below 5 kHz):")
    for case, result in worst.items():
        gamma_scale = max(case) ** 2 / 8
        print(f"   plate gamma {case[0]:.2f}, resonator gamma {case[1]:.2f}: {100 * result['shift']:+.4f} % "
              f"(gamma^2/8 = {100 * gamma_scale:.4f} %), band {result['band'] + 1} at {result['frequency']:.1f} Hz, "
              f"k = {wavevector_label(*result['wavevector'])}, plate share {100 * result['plate_share']:.0f} %")
        assert abs(result["shift"]) < PRODUCTION_TOLERANCE

    # --- 5. gamma^2 scaling, per mode
    for strong, weak, expected in (((0.04, 0.0), (0.02, 0.0), 4.0), ((0.0, 0.10), (0.0, 0.02), 25.0)):
        strong_shifts, weak_shifts = np.concatenate(shifts[strong]), np.concatenate(shifts[weak])
        measurable = np.abs(weak_shifts) > 1e-9
        ratios = strong_shifts[measurable] / weak_shifts[measurable]
        print(f"5. shift ratio gamma {max(strong)} / {max(weak)}: median {np.median(ratios):.3f} (expected {expected}), "
              f"range {ratios.min():.3f} .. {ratios.max():.3f} over {measurable.sum()} modes")
        assert abs(np.median(ratios) / expected - 1) < SCALING_TOLERANCE

    # --- 6. Second-order perturbation theory at the worst wavevector of each case
    print("6. perturbation prediction vs solve at the worst wavevector (worst mode):")
    for case, result in worst.items():
        wavenumber_x, wavenumber_y = result["wavevector"]
        stiffness_reduced, mass_reduced = reduce_to_bloch(augmented_stiffness, augmented_mass, pairs, wavenumber_x,
                                                          wavenumber_y, CELL_LENGTH, CELL_LENGTH, number_of_extra_dofs=1)
        damping_matrix = (case[0] * plate_stiffness_padded + case[1] * spring_stiffness).tocsr()
        damping_reduced, _ = reduce_to_bloch(damping_matrix, augmented_mass, pairs, wavenumber_x, wavenumber_y,
                                             CELL_LENGTH, CELL_LENGTH, number_of_extra_dofs=1)
        all_eigenvalues, all_eigenvectors = eigh(stiffness_reduced.toarray(), mass_reduced.toarray())
        band = result["band"]
        predicted = perturbation_shift(all_eigenvalues, all_eigenvectors, damping_reduced.toarray(), [band])[0]
        coupling = all_eigenvectors.conj().T @ (damping_reduced.toarray() @ all_eigenvectors[:, band])
        contributions = np.abs(coupling) ** 2 / np.where(np.arange(len(all_eigenvalues)) == band, np.inf,
                                                          np.abs(all_eigenvalues[band] - all_eigenvalues))
        partner = int(np.argmax(contributions))
        partner_frequency = np.sqrt(abs(all_eigenvalues[partner])) / (2 * np.pi)
        error = abs(predicted - result["shift"]) / abs(result["shift"])
        print(f"   case {case}: solve {100 * result['shift']:+.5f} %, prediction {100 * predicted:+.5f} %, "
              f"relative difference {error:.1e}; dominant partner mode at {partner_frequency:.1f} Hz "
              f"({abs(partner_frequency - result['frequency']):.1f} Hz away)")
        assert error < PERTURBATION_TOLERANCE

    print(f"All damping checks passed ({time.time() - start_time:.0f} s).")
