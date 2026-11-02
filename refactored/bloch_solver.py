import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from weak_form import define_weak_form
from solve import assemble_plate_matrix
from augmented_system import petsc_to_scipy
from periodic_bc import build_T


def assemble_unit_cell_matrices(function_space, problem, shear_quadrature_degree=None):
    """Stiffness and mass of the free unit cell (no Dirichlet BCs) as scipy CSR matrices.
    The periodic coupling is applied afterwards through the reduction matrix T(k).

    shear_quadrature_degree: None means full integration (2 * degree, i.e. 3x3 Gauss for degree 2).
    The reduced 2x2 rule of the forced-response pipeline gives Lagrange Q2 a zero-energy hourglass mode
    that repeats every element, fits any periodic cell and shows up as a spurious branch out of Gamma."""
    if shear_quadrature_degree is None:
        shear_quadrature_degree = 2 * function_space.ufl_element().degree
    mass_form, _, stiffness_form = define_weak_form(function_space, problem,
                                                    shear_quadrature_degree=shear_quadrature_degree)
    stiffness = petsc_to_scipy(assemble_plate_matrix(stiffness_form, [])).tocsr()
    mass = petsc_to_scipy(assemble_plate_matrix(mass_form, [])).tocsr()
    return stiffness, mass


def reduce_to_bloch(stiffness, mass, pairs, wavenumber_x, wavenumber_y, length_x, length_y):
    """K_r = T^H K T and M_r = T^H M T for the Bloch wavevector (wavenumber_x, wavenumber_y)."""
    reduction_matrix = build_T(pairs, wavenumber_x, wavenumber_y, length_x, length_y)
    reduction_matrix_hermitian = reduction_matrix.conj().T
    stiffness_reduced = (reduction_matrix_hermitian @ stiffness @ reduction_matrix).tocsr()
    mass_reduced = (reduction_matrix_hermitian @ mass @ reduction_matrix).tocsr()
    return stiffness_reduced, mass_reduced


def scipy_to_petsc_complex(matrix, comm):
    matrix = matrix.tocsr()
    petsc_matrix = PETSc.Mat().createAIJ(size=matrix.shape, comm=comm,
                                         csr=(matrix.indptr, matrix.indices, matrix.data.astype(PETSc.ScalarType)))
    petsc_matrix.assemble()
    return petsc_matrix


def solve_bloch_evp(stiffness_reduced, mass_reduced, number_of_bands: int = 10,
                    target_omega_squared: float = -1.0, ncv_factor: int = 4,
                    guard_eigenpairs: int = 4, factorization_package: str = "petsc", comm=MPI.COMM_SELF):
    """Lowest Bloch bands of the reduced (complex Hermitian) problem K_r q = omega^2 M_r q.

    Needs the complex PETSc build (fenicsx-env-complex). Returns frequencies in Hz (sorted ascending)
    and the matching reduced eigenvectors (columns). The undamped problem is used.

    target_omega_squared: slightly negative so that shift-invert stays nonsingular at Gamma,
        where K_r has a rigid-body null space.
    guard_eigenpairs: extra pairs requested beyond number_of_bands, so a degenerate partner of the
        last returned band is not cut off. Only number_of_bands are returned.
    factorization_package: sparse LU used by shift-invert, "petsc" (built-in LU) or "mumps".
        PETSc LU is faster up to about 16x16 elements (production mesh); MUMPS wins from about 32x32
        (timings in the commit history).
    """
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError("solve_bloch_evp needs complex PETSc; run in fenicsx-env-complex")

    number_requested = number_of_bands + guard_eigenpairs
    stiffness_petsc = scipy_to_petsc_complex(stiffness_reduced, comm)
    mass_petsc = scipy_to_petsc_complex(mass_reduced, comm)

    eigensolver = SLEPc.EPS().create(comm)
    eigensolver.setOperators(stiffness_petsc, mass_petsc)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)   # complex Hermitian K_r, Hermitian positive definite M_r
    spectral_transform = eigensolver.getST()
    spectral_transform.setType(SLEPc.ST.Type.SINVERT)
    linear_solver = spectral_transform.getKSP()
    linear_solver.setType(PETSc.KSP.Type.PREONLY)          # one direct solve per application, no iterations
    preconditioner = linear_solver.getPC()
    preconditioner.setType(PETSc.PC.Type.LU)
    preconditioner.setFactorSolverType(factorization_package)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eigensolver.setTarget(target_omega_squared)
    eigensolver.setDimensions(nev=number_requested, ncv=ncv_factor * number_requested)
    eigensolver.solve()

    number_converged = eigensolver.getConverged()
    if number_converged < number_requested:
        raise RuntimeError(f"SLEPc converged only {number_converged} of {number_requested} eigenpairs "
                           f"(reason={eigensolver.getConvergedReason()}); increase ncv_factor")

    eigenvector_petsc, _ = stiffness_petsc.createVecs()
    omega_squared = np.empty(number_converged)
    eigenvectors = np.empty((stiffness_reduced.shape[0], number_converged), dtype=complex)
    for i in range(number_converged):
        eigenvalue = eigensolver.getEigenpair(i, eigenvector_petsc)
        omega_squared[i] = eigenvalue.real            # Hermitian problem: imaginary part is round-off
        eigenvectors[:, i] = eigenvector_petsc.getArray()

    for petsc_object in (eigensolver, stiffness_petsc, mass_petsc, eigenvector_petsc):
        petsc_object.destroy()

    # SLEPc orders by distance to the target, not by frequency: sort values and vectors together
    order = np.argsort(omega_squared)[:number_of_bands]
    omega_squared, eigenvectors = omega_squared[order], eigenvectors[:, order]

    # round-off can make the rigid-body mode at Gamma slightly negative
    scale = max(abs(omega_squared).max(), 1.0)
    if (omega_squared < -1e-8 * scale).any():
        raise ValueError(f"clearly negative omega^2 found: {omega_squared[omega_squared < 0]}")
    frequencies_hz = np.sqrt(np.clip(omega_squared, 0.0, None)) / (2 * np.pi)
    return frequencies_hz, eigenvectors
