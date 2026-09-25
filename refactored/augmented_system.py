import numpy as np
from petsc4py import PETSc
from scipy.sparse import bmat
from scipy.sparse import csr_matrix, coo_matrix
from point_coupling import assert_vamm_indices_registered


# --- helpers: PETSc <-> scipy round-trip ---

def petsc_to_scipy(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.getSize())

def scipy_to_petsc(A_sp):
    A_sp = A_sp.tocsr()
    A_petsc = PETSc.Mat().createAIJ(size=A_sp.shape,
                                    csr=(A_sp.indptr, A_sp.indices, A_sp.data))
    A_petsc.assemble()
    return A_petsc

# --- Build the augmented (n+1) x (n+1) system ---

def augment_mass_matrix(M, vamm_list): # Remove v2 once done implementing VAMM pipeline!
    """
    Embed point masses into mass matrix. User must ensure that indexing is order-matching.
    Otherwise, augmentation will raise an assertion error. It is therefore recommended to use
    create_vamm_list_and_assign_indices from point_coupling.py in order to create vamm_list,
    which handles correct indexing automatically.
    """
    assert_vamm_indices_registered(vamm_list)
    M_sp = petsc_to_scipy(M)
    additional_masses = csr_matrix(np.diag([vamm.mass for vamm in vamm_list]))
    M_aug_sp = bmat([[M_sp, None],
                     [None, additional_masses]], format="csr")
    return scipy_to_petsc(M_aug_sp)

def augment_stiffness_matrix(
        K, vamm_list, phi_list, global_dofs_parent_list):
    """Add each spring's rank-1 stiffness coupling as one extra DOF in the stiffness matrix."""
    assert (len(vamm_list) == len(phi_list) and len(vamm_list) == len(global_dofs_parent_list)), \
        (f"vamm_list (length {len(vamm_list)} must have the same length as phi_list (length {len(phi_list)} and"
         f" global_dofs_parent_list (length {len(global_dofs_parent_list)}))")
    K_sp = petsc_to_scipy(K)
    n_plate = K_sp.shape[0]
    N = len(vamm_list)
    n_total = n_plate + N

    K_spring_total = csr_matrix((n_total, n_total))

    for vamm, phi, global_dofs_parent in zip(vamm_list, phi_list, global_dofs_parent_list):
        phi_ext = np.zeros(n_total)
        phi_ext[global_dofs_parent] = phi
        phi_ext[n_plate + vamm.index] = -1.0

        nz = np.nonzero(phi_ext)[0]
        rows = np.repeat(nz, len(nz))
        cols = np.tile(nz, len(nz))
        vals = vamm.stiffness * (1 + 1j*vamm.gamma) * np.outer(phi_ext[nz], phi_ext[nz]).flatten()
        K_spring_total = K_spring_total + coo_matrix((vals, (rows, cols)), shape=(n_total, n_total)).tocsr()

    K_embedded = bmat([[K_sp, None],
                       [None, csr_matrix((N, N))]], format="csr")
    K_aug_sp = K_embedded + K_spring_total

    return scipy_to_petsc(K_aug_sp), n_plate

# --- Add stiffness contribution directly onto plate's existing DOFs

def add_condensed_spring_stiffness(A, phi, global_dofs_parent, stiffness):
    """Add the spring's rank-1 stiffness contribution directly onto the
    plate's existing DOFs (static condensation of the spring's own
    coordinate — used for static solves where the spring's own
    displacement isn't needed as an independent unknown).

    Modifies A in place via PETSc ADD_VALUES; caller must still call
    A.assemblyBegin()/A.assemblyEnd() (or A.assemble()) afterward.
    """
    for i, dof_i in enumerate(global_dofs_parent):
        for j, dof_j in enumerate(global_dofs_parent):
            A.setValue(dof_i, dof_j, stiffness * phi[i] * phi[j],
                       addv=PETSc.InsertMode.ADD_VALUES)

    return A

def validate_augmented_system(
        K_aug, M_aug, K, n_plate,
        vamm_list, phi_list, global_dofs_parent_list):

    K_sp = petsc_to_scipy(K)
    M_aug_sp = petsc_to_scipy(M_aug)

    # 1. q_r's own diagonal should be exactly k_r (mass m_r), nothing else touches it
    for vamm in vamm_list:
        current_index = n_plate + vamm.index
        assert(np.isclose(K_aug[current_index, current_index], vamm.stiffness*(1+1j*vamm.gamma), rtol=1e-9)), \
            f"K_aug[n,n] = {K_aug[current_index, current_index]}, expected: {vamm.stiffness*(1+1j*vamm.gamma)}"
        assert(np.isclose(M_aug[current_index, current_index], vamm.mass, rtol=1e-9)), \
            f"M_aug[n,n] = {M_aug[current_index, current_index]} expected: {vamm.mass}"

    # 2. coupling entries should be -k_r * phi_i, symmetric
    for vamm, phi, global_dofs_parent in zip(vamm_list, phi_list, global_dofs_parent_list):
        current_index = n_plate + vamm.index
        for k, dof_i in enumerate(global_dofs_parent):
            expected_coupling = -vamm.stiffness *(1+1j*vamm.gamma)* phi[k]
            assert(np.isclose(K_aug[dof_i, current_index], expected_coupling, rtol = 1e-9)), \
                f"K_aug[i0, n] = {K_aug[dof_i, current_index]} expected: {expected_coupling}"
            assert(np.isclose(K_aug[current_index, dof_i], expected_coupling, rtol = 1e-9)), \
                f"K_aug[n, i0] = {K_aug[current_index, dof_i]} expected: {expected_coupling}" # symmetry requirement

    # 3. plate-plate block should equal original K plus the SUM of all VAMMs'
    #    outer-product contributions at each shared plate DOF (a DOF can be
    #    touched by more than one VAMM if their basis-function supports overlap)
    expected_delta = {}
    for vamm, phi, global_dofs_parent in zip(vamm_list, phi_list, global_dofs_parent_list):
        for k, dof_i in enumerate(global_dofs_parent):
            expected_delta[dof_i] = expected_delta.get(dof_i, 0.0) + vamm.stiffness *(1+1j*vamm.gamma)* phi[k]**2

    for dof_i, delta in expected_delta.items():
        expected_value = K_sp[dof_i, dof_i] + delta
        assert np.isclose(K_aug[dof_i, dof_i], expected_value, rtol=1e-9), \
            f"K_aug[{dof_i},{dof_i}] = {K_aug[dof_i, dof_i]} expected: {expected_value}"

    # 4. mass matrix has zero coupling anywhere in the last row/column except the diagonal
    for vamm in vamm_list:
        current_index = n_plate + vamm.index
        row_n = M_aug_sp[current_index, :].toarray().flatten()
        off_diag = np.delete(row_n, current_index)
        assert np.allclose(off_diag, 0.0, rtol = 1e-9), f"M_aug row n has nonzero off-diagonal entries: {off_diag}"
        assert np.isclose(row_n[current_index], vamm.mass, rtol = 1e-9), \
            f"M_aug[n, n] = {M_aug[current_index, current_index]}, expected: {vamm.mass}"
