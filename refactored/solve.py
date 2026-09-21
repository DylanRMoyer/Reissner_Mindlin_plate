import numpy as np
from dolfinx import fem
from slepc4py import SLEPc
from weak_form import define_weak_form
from dolfinx.fem.petsc import LinearProblem
from augmented_system import augment_stiffness_matrix, augment_mass_matrix, validate_augmented_system

def assemble_plate_matrix(a, bcs, diag_value=1.0):
    """Assemble a plate system matrix (K or M) with Dirichlet BCs applied.

    diag_value: value placed on the diagonal for constrained DOFs.
    Use the default 1.0 for a normal, well-scaled matrix (e.g. M).
    Use a large value (e.g. 1e10) for K specifically, to push
    constrained-DOF eigenmodes far outside the eigenvalue range of
    interest and avoid spurious eigenvalue clusters in the SLEPc solve.
    """
    assembled_matrix = fem.petsc.assemble_matrix(fem.form(a), bcs=bcs, diag=diag_value)
    assembled_matrix.assemble()

    return assembled_matrix

def solve_evp(
        domain, function_space, problem, bcs, vamm_list=None,
        phi_list=None, global_dofs_parent_list=None, eigenmode_number: int = 6, validate_augmentation=True):
    m, _, a = define_weak_form(function_space, problem)
    K = assemble_plate_matrix(a, bcs, diag_value=1e10)
    M = assemble_plate_matrix(m, bcs)

    if vamm_list is not None:
        if phi_list is None or global_dofs_parent_list is None:
            raise TypeError("phi_list and global_dofs_parent_list are required when vamm_config_list is provided")
        K_aug, n_plate = augment_stiffness_matrix(
            K=K, vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list)
        M_aug = augment_mass_matrix(M=M, vamm_list=vamm_list)

        # Validate augmentation
        if validate_augmentation:
            validate_augmented_system(
                K_aug=K_aug, M_aug=M_aug, K=K, n_plate=n_plate,
                vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list)

    else:
        K_aug, M_aug, n_plate = K, M, K.getSize()[0]

    eps = SLEPc.EPS().create(domain.comm)
    eps.setOperators(K_aug, M_aug)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(0.0)  # target near zero -- lowest frequencies

    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0)
    eps.setDimensions(nev=eigenmode_number)  # how many eigenpairs to converge
    eps.solve()

    # create PETSc vectors matching K's layout (real and imaginary parts)
    vr, vi = K_aug.createVecs()

    eigenfrequencies = []
    eigenmodes = []

    for i in range(eps.getConverged()):
        eigval = eps.getEigenpair(i, vr, vi)  # fills vr, vi; returns eigenvalue
        omega_sq = eigval.real
        freq_hz = np.sqrt(omega_sq) / (2 * np.pi)

        plate_part = vr.getArray()[:n_plate].real  # undamped EVP: eigenvectors are real; explicit cast
        q_r_values = (
            [vr.getArray()[n_plate + vamm.index].real for vamm in vamm_list]
            if vamm_list is not None else None
        )

        mode_function = fem.Function(function_space)
        mode_function.x.petsc_vec.setArray(plate_part.real)  # Revisit for damped eigenproblems!
        mode_function.x.scatter_forward()  # sync ghost values (matters in parallel)

        eigenfrequencies.append(freq_hz)
        eigenmodes.append((mode_function, q_r_values))

    return eigenfrequencies, eigenmodes
