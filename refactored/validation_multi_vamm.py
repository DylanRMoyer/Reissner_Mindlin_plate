import numpy as np
from dataclasses import replace
from dolfinx import fem
from config import PlateConfig
from problem_setup import build_plate_problem
from scipy.sparse import csr_matrix, coo_matrix
from slepc4py import SLEPc
from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
from solve import solve_evp, assemble_plate_matrix
from augmented_system import scipy_to_petsc, petsc_to_scipy
from weak_form import define_weak_form

def with_modified_vamm(vamm_list, index, stiffness=None, mass=None):
    """Return a NEW list in which vamm_list[index] has stiffness and/or mass
    overridden. The input list and its VAMM objects are left untouched."""
    original = vamm_list[index]
    modified = replace(
        original,
        stiffness=original.stiffness if stiffness is None else stiffness,
        mass=original.mass if mass is None else mass,
    )
    new_list = list(vamm_list)          # shallow copy of the list itself
    new_list[index] = modified
    return new_list

def isolated_estimate_hz(vamm):
    return np.sqrt(vamm.stiffness / vamm.mass) / (2 * np.pi)

def drop_nearest(freqs, target, max_dist=None):
    """Remove the single frequency in freqs closest to target.
    If max_dist is given and no frequency is within it, return freqs unchanged
    and None (branch not present in this window)."""
    freqs = list(freqs)
    idx = int(np.argmin([abs(f - target) for f in freqs]))
    if max_dist is not None and abs(freqs[idx] - target) > max_dist:
        return freqs, None
    dropped = freqs.pop(idx)
    return freqs, dropped

def build_reference_mass_matrix(M, vamm_list, phi_list, global_dofs_parent_list):
    """Rigid-attachment reference: M_plate + sum(m_i * phi_i phi_i^T), no extra DOFs.
    Represents the k->infinity limit of the augmented system, derived independently
    (static condensation of q_r), not a re-parameterization of the augmented code."""
    assert (len(vamm_list) == len(phi_list) == len(global_dofs_parent_list)), \
        "vamm_list, phi_list, global_dofs_parent_list must have equal length"

    M_sp = petsc_to_scipy(M)
    n_plate = M_sp.shape[0]

    M_point_mass_total = csr_matrix((n_plate, n_plate))

    for vamm, phi, global_dofs_parent in zip(vamm_list, phi_list, global_dofs_parent_list):
        phi_ext = np.zeros(n_plate)
        phi_ext[global_dofs_parent] = phi

        nz = np.nonzero(phi_ext)[0]
        rows = np.repeat(nz, len(nz))
        cols = np.tile(nz, len(nz))
        vals = vamm.mass * np.outer(phi_ext[nz], phi_ext[nz]).flatten()
        M_point_mass_total = M_point_mass_total + coo_matrix((vals, (rows, cols)), shape=(n_plate, n_plate)).tocsr()

    M_ref_sp = M_sp + M_point_mass_total
    return scipy_to_petsc(M_ref_sp)

def solve_evp_validation(
        domain, function_space, problem, bcs, vamm_list,
        phi_list, global_dofs_parent_list, eigenmode_number: int = 6, validate_augmentation=True):
    m, _, a = define_weak_form(function_space, problem)
    K_plate = assemble_plate_matrix(a, bcs, diag_value=1e10)
    M_plate = assemble_plate_matrix(m, bcs)

    M_ref = build_reference_mass_matrix(M_plate, vamm_list, phi_list, global_dofs_parent_list)

    n_plate = K_plate.getSize()[0]

    eps = SLEPc.EPS().create(domain.comm)
    eps.setOperators(K_plate, M_ref)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(0.0)  # target near zero -- lowest frequencies

    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0)
    eps.setDimensions(nev=eigenmode_number, ncv = 2 * eigenmode_number)  # how many eigenpairs to converge
    eps.solve()

    n_converged = eps.getConverged()
    if n_converged < eigenmode_number:
        raise RuntimeError(
            f"SLEPc converged only {n_converged} of {eigenmode_number} requested eigenpairs "
            f"(reason={eps.getConvergedReason()}); increase ncv or check the problem setup."
        )

    # create PETSc vectors matching K's layout (real and imaginary parts)
    vr, vi = K_plate.createVecs()

    eigenfrequencies = []
    eigenmodes = []

    for i in range(eigenmode_number):
        eigval = eps.getEigenpair(i, vr, vi)  # fills vr, vi; returns eigenvalue
        omega_sq = eigval.real
        tolerance = 1.0e-3 # To-Do: set dynamic tolerance!
        if -tolerance < omega_sq < tolerance:
            omega_sq = 0.0
        if omega_sq < 0.0:
            raise ValueError(f"omega_sq={omega_sq} < 0.0")
        freq_hz = np.sqrt(omega_sq) / (2 * np.pi)

        plate_part = vr.getArray()[:n_plate].real  # undamped EVP: eigenvectors are real; explicit cast
        mode_function = fem.Function(function_space)
        mode_function.x.petsc_vec.setArray(plate_part.real)  # Revisit for damped eigenproblems!
        mode_function.x.scatter_forward()  # sync ghost values (matters in parallel)

        eigenfrequencies.append(freq_hz)
        eigenmodes.append(mode_function)

    return eigenfrequencies, eigenmodes

def compute_mode_shift(domain, function_space, bcs, problem,
                       vamm_list, phi_list, global_dofs_parent_list,
                       target_bare_freq, search_window=None):
    bare_plate_frequencies, _ = solve_evp(
        domain=domain, function_space=function_space, bcs=bcs, problem=problem,
        eigenmode_number=12)

    modified_frequencies, _ = solve_evp(
        domain=domain, function_space=function_space, bcs=bcs, problem=problem,
        vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
        eigenmode_number=12)

    bare_remaining, bare_freq = drop_nearest(bare_plate_frequencies, target_bare_freq)
    modified_remaining, modified_freq = drop_nearest(modified_frequencies, target_bare_freq,
                                                     max_dist=search_window)
    shift = bare_freq - modified_freq
    relative_shift = shift / bare_freq

    return shift, relative_shift, bare_freq, modified_freq




if __name__ == "__main__":

    plate_config = PlateConfig(
        length=1.1,
        width=1,
        thickness=0.02,
        nx=60,
        ny=50,
        rho=7850,
        mu=77e9,
        lambda_=115e9,
        #constant_force = 1 # Use if wanting to prescribe a custom constant force across the plate
    )
    domain, function_space, _, plate_problem_constants = build_plate_problem(plate_config)
    # Add degree custom degree (default: deg = 2) or function type (default: el_type = "S") if needed

    bcs = []

    free_plate_freqs, free_plate_eigenmodes = solve_evp(
        domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
        eigenmode_number=11)

    eigenmode_index=7


    mode_to_plot, q_r_to_plot = free_plate_eigenmodes[eigenmode_index]
    w_mode = mode_to_plot.sub(0).collapse()

    V_plot = fem.functionspace(domain, ("Lagrange", 2))
    w_plot = fem.Function(V_plot)
    w_plot.interpolate(w_mode)

    dof_coords = V_plot.tabulate_dof_coordinates()  # shape (n_dofs, 3)
    values = w_plot.x.array

    antinode_idx = np.argmax(np.abs(values))
    antinode_xy = dof_coords[antinode_idx, :2]
    print(f"Antinode at {antinode_xy}, |w| = {abs(values[antinode_idx]):.6e}")

    # for the node: find the DOF with smallest |w| that's still well inside the plate
    # (avoid picking a boundary point, which trivially has w=0 if clamped -- not an issue
    # here since bcs=[], but still avoid points right at the free edge for interpretation)
    node_idx = np.argmin(np.abs(values))
    node_xy = dof_coords[node_idx, :2]
    print(f"Node at {node_xy}, |w| = {abs(values[node_idx]):.6e}")


    vamm_list = create_vamm_list_and_assign_indices(
        [(0.5,0.5,100,1), (0.6, 0.6, 200, 0.5), (0.1, 0.1, 50, 5)], 0.01)

    vamm_list_nodal_line_mode_five = create_vamm_list_and_assign_indices(
        [(node_xy[0], node_xy[1], 1e10, 1)], 0.01)
        # no shift expected, resonance frequency @ 147.4898 Hz
    vamm_list_antinodal_line_mode_five = create_vamm_list_and_assign_indices(
        [(antinode_xy[0], antinode_xy[1], 1e10, 1)], 0.01)
        # large shift from original resonance frequency @ 147.4898 Hz expected

    stiff_limits =\
        [with_modified_vamm(vamm_list=vamm_list, index=index, stiffness=1e12)  for index in range(len(vamm_list))]
    mass_limits =\
        [with_modified_vamm(vamm_list=vamm_list, index=index, mass=1e-7) for index in range(len(vamm_list))]

    phi_list, global_dofs_parent_list, local_to_global_w_list = compute_phi_and_dofs_for_vamm_list(
        domain=domain, function_space=function_space, vamm_list=vamm_list
    )

    phi_list_nodal, global_dofs_parent_list_nodal, local_to_global_w_list_nodal =(
        compute_phi_and_dofs_for_vamm_list(
        domain=domain, function_space=function_space, vamm_list=vamm_list_nodal_line_mode_five
    ))

    phi_list_antinodal, global_dofs_parent_list_antinodal,local_to_global_w_list_antinodal = (
        compute_phi_and_dofs_for_vamm_list(
        domain=domain, function_space=function_space, vamm_list=vamm_list_antinodal_line_mode_five
    ))

    nodal_shift, nodal_relative_shift, bare_plate_freq_nodal, nodal_freq = compute_mode_shift(
        domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
        vamm_list=vamm_list_nodal_line_mode_five, phi_list=phi_list_nodal,
        global_dofs_parent_list=global_dofs_parent_list_nodal,
        target_bare_freq=147.4849, search_window=30
    )

    antinodal_shift, antinodal_relative_shift, bare_plate_freq_antinodal, antinodal_freq = compute_mode_shift(
        domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
        vamm_list=vamm_list_antinodal_line_mode_five, phi_list=phi_list_antinodal,
        global_dofs_parent_list=global_dofs_parent_list_antinodal,
        target_bare_freq=147.4849, search_window=30
    )

    print(f"Antinodal shift: {antinodal_shift} ({100*antinodal_relative_shift}%)\n"
          f"Initial frequency: {bare_plate_freq_antinodal}, modified: {antinodal_freq}")
    print(f"Nodal shift: {nodal_shift} ({100*nodal_relative_shift}%)\n"
          f"Initial frequency: {bare_plate_freq_nodal}, modified: {nodal_freq}")

    assert np.abs(antinodal_shift) > np.abs(nodal_shift), (
        f"Unexpected behavior after VAMM placement:\n"
        f"Frequency shift for nodal placement of VAMM: {nodal_shift}\n"
        f"Frequency shift for antinodal placement of VAMM: {antinodal_shift}\n"
    )


    baseline_freqs, _ = solve_evp(
            domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
        vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
        eigenmode_number=11)

    eigenfrequencies_validate, _ = solve_evp_validation(
        domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
        vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
        eigenmode_number=11)

    all_stiff = vamm_list
    for index in range(len(vamm_list)):
        all_stiff = with_modified_vamm(all_stiff, index=index, stiffness=1e10)

    eigenfrequencies_augmented_limit, _ = solve_evp(
        domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
        vamm_list=all_stiff, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
        eigenmode_number=11)

    assert np.allclose(eigenfrequencies_validate, eigenfrequencies_augmented_limit[:len(eigenfrequencies_validate)],
                       rtol=1e-3), (
        f"M_ref vs augmented k->inf mismatch:\n"
        f"M_ref={eigenfrequencies_validate}\naugmented={eigenfrequencies_augmented_limit}"
    )

    relative_deviation = []
    for i in range(len(eigenfrequencies_validate)):
        if eigenfrequencies_validate[i] > 1e-2:
           relative_deviation.append(
               (eigenfrequencies_validate[i] - eigenfrequencies_augmented_limit[i])/eigenfrequencies_validate[i]
           )

    print(np.max(relative_deviation))
    print("First validation done")





    for i in range(len(vamm_list)):
        target = isolated_estimate_hz(vamm_list[i])
        baseline_subset, _ = drop_nearest(baseline_freqs, target)

        branch_freqs = []
        m_tests = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
        for m_test in m_tests:
            # residual coupling of the "rigid" resonator to the plate falls off ~1/sqrt(k)
            rtol_dynamic = 5e-1 * np.sqrt(m_test * 1e6)  # tune the constant to your k_tests range
            if rtol_dynamic < 5e-2:
                rtol_dynamic = 5e-2
            modified = with_modified_vamm(vamm_list, index=i, mass=m_test)
            freqs, _ = solve_evp(
                domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
                vamm_list=modified, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
                eigenmode_number=11)

            # as k grows, the branch's own frequency rises (omega = sqrt(k/m)) and may
            # leave the visible window entirely rather than merging into the rigid cluster
            current_target = np.sqrt(vamm_list[i].stiffness / m_test) / (2 * np.pi)
            untouched, branch = drop_nearest(freqs, current_target, max_dist=current_target * 0.5)

            if branch is not None:
                branch_freqs.append(branch)
                compare_len = min(len(untouched), len(baseline_subset))
                assert np.allclose(untouched[:compare_len], baseline_subset[:compare_len], rtol=rtol_dynamic), (
                    f"Failed at m_test={m_test}, i={i}, rtol_dynamic={rtol_dynamic}\n"
                    f"untouched={untouched}\nbaseline_subset={baseline_subset}"
                )
            else:
                # branch has left the window; only compare the low, stable modes since
                # the topmost slot is volatile once a mode exits the eigenmode_number cutoff
                n_stable = len(baseline_subset) - 1
                assert np.allclose(freqs[:n_stable], baseline_subset[:n_stable], rtol=rtol_dynamic), (
                    f"Failed at m_test={m_test}, i={i} (branch left window), rtol_dynamic={rtol_dynamic}\n"
                    f"freqs={freqs}\nbaseline_subset={baseline_subset}"
                )


    for i in range(len(vamm_list)):
        target = isolated_estimate_hz(vamm_list[i])
        baseline_subset, _ = drop_nearest(baseline_freqs, target)

        branch_freqs = []
        k_tests = [1e4, 1e6, 1e8, 1e10, 1e12]
        for k_test in k_tests:
            # residual coupling of the "rigid" resonator to the plate falls off ~1/sqrt(k)
            rtol_dynamic = 5e-1 / np.sqrt(k_test / 1e6)  # tune the constant to your k_tests range
            if rtol_dynamic < 5e-2:
                rtol_dynamic = 5e-2
            modified = with_modified_vamm(vamm_list, index=i, stiffness=k_test)
            freqs, _ = solve_evp(
                domain=domain, function_space=function_space, bcs=bcs, problem=plate_problem_constants,
                vamm_list=modified, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
                eigenmode_number=11)

            # as k grows, the branch's own frequency rises (omega = sqrt(k/m)) and may
            # leave the visible window entirely rather than merging into the rigid cluster
            current_target = np.sqrt(k_test / vamm_list[i].mass) / (2 * np.pi)
            untouched, branch = drop_nearest(freqs, current_target, max_dist=current_target * 0.5)

            if branch is not None:
                branch_freqs.append(branch)
                compare_len = min(len(untouched), len(baseline_subset))
                assert np.allclose(untouched[:compare_len], baseline_subset[:compare_len], rtol=rtol_dynamic), (
                    f"Failed at k_test={k_test}, i={i}, rtol_dynamic={rtol_dynamic}\n"
                    f"untouched={untouched}\nbaseline_subset={baseline_subset}"
                )
            else:
                # branch has left the window; only compare the low, stable modes since
                # the topmost slot is volatile once a mode exits the eigenmode_number cutoff
                n_stable = len(baseline_subset) - 1
                assert np.allclose(freqs[:n_stable], baseline_subset[:n_stable], rtol=rtol_dynamic), (
                    f"Failed at k_test={k_test}, i={i} (branch left window), rtol_dynamic={rtol_dynamic}\n"
                    f"freqs={freqs}\nbaseline_subset={baseline_subset}"
                )
