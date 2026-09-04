from dolfinx import fem
from petsc4py import PETSc
from augmented_system import add_condensed_spring_stiffness

# --- Solve static problem ---


def assemble_condensed_stiffness_matrix(a, bcs, phi, global_dofs_parent, vamm_config):
    """Assemble the plate stiffness matrix with the spring's stiffness
    statically condensed onto the plate's existing DOFs."""
    bilinear_form = fem.form(a)
    A = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    A = add_condensed_spring_stiffness(A, phi, global_dofs_parent, vamm_config.point_stiffness)
    A.assemble()
    return A, bilinear_form


def assemble_spring_load_vector(A, bilinear_form, bcs, phi, global_dofs_parent,
                                vamm_config, spring_end_displacement):
    """Build the RHS vector representing an imposed displacement at the
    spring's free end, condensed onto the plate's existing DOFs."""
    b = A.createVecRight()
    b.zeroEntries()

    for phi_i, parent_dof in zip(phi, global_dofs_parent):
        b.setValue(parent_dof, phi_i * vamm_config.point_stiffness * spring_end_displacement,
                   addv=PETSc.InsertMode.ADD_VALUES)
    b.assemblyBegin()
    b.assemblyEnd()

    fem.petsc.apply_lifting(b, [bilinear_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)
    return b


def solve_point_load(domain, function_space, A, b):

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")

    u_point = fem.Function(function_space)
    solver.solve(b, u_point.x.petsc_vec)
    u_point.x.scatter_forward()

    w_point = u_point.sub(0).collapse()
    #print(f"Point-load deflection at target: {max(abs(w_point.x.array)):.6e}")

    return w_point
