from dolfinx import fem

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
