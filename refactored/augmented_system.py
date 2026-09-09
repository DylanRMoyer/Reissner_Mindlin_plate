import numpy as np
from petsc4py import PETSc
from scipy.sparse import bmat
from scipy.sparse import csr_matrix, coo_matrix


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

def augment_mass_matrix(M, point_mass):
    """Embed a point mass as one extra diagonal DOF in the mass matrix."""
    M_sp = petsc_to_scipy(M)
    M_aug_sp = bmat([[M_sp, None],
                      [None, csr_matrix([[point_mass]])]], format="csr")

    return scipy_to_petsc(M_aug_sp)

def augment_stiffness_matrix(K, point_stiffness, phi, global_dofs_parent):
    """Add the spring's rank-1 stiffness coupling as one extra DOF in the stiffness matrix."""
    K_sp = petsc_to_scipy(K)
    n = K_sp.shape[0]

    phi_ext = np.zeros(n + 1)
    phi_ext[global_dofs_parent] = phi
    phi_ext[n] = -1.0

    nz = np.nonzero(phi_ext)[0]
    rows = np.repeat(nz, len(nz))
    cols = np.tile(nz, len(nz))
    vals = point_stiffness * np.outer(phi_ext[nz], phi_ext[nz]).flatten()
    K_spring = coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()

    K_embedded = bmat([[K_sp, None],
                        [None, csr_matrix([[0.0]])]], format="csr")
    K_aug_sp = K_embedded + K_spring

    return scipy_to_petsc(K_aug_sp), n

# --- Add stiffness contribution directly onto plate's existing DOFs

def add_condensed_spring_stiffness(A, phi, global_dofs_parent, point_stiffness):
    """Add the spring's rank-1 stiffness contribution directly onto the
    plate's existing DOFs (static condensation of the spring's own
    coordinate — used for static solves where the spring's own
    displacement isn't needed as an independent unknown).

    Modifies A in place via PETSc ADD_VALUES; caller must still call
    A.assemblyBegin()/A.assemblyEnd() (or A.assemble()) afterward.
    """
    for i, dof_i in enumerate(global_dofs_parent):
        for j, dof_j in enumerate(global_dofs_parent):
            A.setValue(dof_i, dof_j, point_stiffness * phi[i] * phi[j],
                       addv=PETSc.InsertMode.ADD_VALUES)

    return A


def validate_augmented_system(
        K_aug, M_aug, K, n, vamm_config,
        phi, global_dofs_parent):

    K_sp = petsc_to_scipy(K)
    M_aug_sp = petsc_to_scipy(M_aug)

    # 1. q_r's own diagonal should be exactly k_r (mass m_r), nothing else touches it
    assert(np.isclose(K_aug[n, n], vamm_config.point_stiffness, rtol=1e-9)), \
           f"K_aug[n,n] = {K_aug[n, n]}, expected: {vamm_config.point_stiffness}"
    assert(np.isclose(M_aug[n, n], vamm_config.point_mass, rtol=1e-9)), \
           f"M_aug[n,n] = {M_aug[n, n]} expected: {vamm_config.point_mass}"

    # 2. coupling entries should be -k_r * phi_i, symmetric
    for k, dof_i in enumerate(global_dofs_parent):
        expected_coupling = -vamm_config.point_stiffness * phi[k]
        assert(np.isclose(K_aug[dof_i, n], expected_coupling, rtol = 1e-9)), \
               f"K_aug[i0, n] = {K_aug[dof_i, n]} expected: {expected_coupling}"
        assert(np.isclose(K_aug[n, dof_i], expected_coupling, rtol = 1e-9)), \
               f"K_aug[n, i0] = {K_aug[n, dof_i]} expected: {expected_coupling}" # symmetry requirement

    # 3. plate-plate block should equal original K plus the outer product, e.g. at (i0, i0)
    for k, dof_i in enumerate(global_dofs_parent):
        assert(np.isclose(K_aug[dof_i, dof_i], K_sp[dof_i, dof_i] + vamm_config.point_stiffness * phi[k]**2, rtol = 1e-9)), \
            f"K_aug[i0, i0] = {K_aug[dof_i, dof_i]} expected: {K_sp[dof_i, dof_i] + vamm_config.point_stiffness * phi[k]**2}"

    # 4. mass matrix has zero coupling anywhere in the last row/column except the diagonal
    row_n = M_aug_sp[n, :].toarray().flatten()
    off_diag = np.delete(row_n, n)
    assert np.allclose(off_diag, 0.0, rtol = 1e-9), f"M_aug row n has nonzero off-diagonal entries: {off_diag}"
    assert np.isclose(row_n[n], vamm_config.point_mass, rtol = 1e-9),\
        f"M_aug[n, n] = {M_aug[n, n]}, expected: {vamm_config.point_mass}"
