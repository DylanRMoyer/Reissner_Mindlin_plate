from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from weak_form import define_weak_form
from boundary_conditions import dirichlet_boundary_conditions

def assemble_dynamic_stiffness(a, m, Omega, bcs):
    form = fem.form(a - Omega**2 * m)
    A = fem.petsc.assemble_matrix(form, bcs=bcs)
    A.assemble()
    return A, form

def assemble_shaker_rhs(A, form, bcs):
    b = A.createVecRight()
    b.zeroEntries()                                   # no body force — pure homogeneous eqn
    fem.petsc.apply_lifting(b, [form], bcs=[bcs])      # computes -(K12 - Ω²M12) q2 internally
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)                           # writes q2 = u2*phi_0 into boundary rows
    return b

def solve_linear_system(domain, function_space, A, b):

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")

    u_point = fem.Function(function_space)
    solver.solve(b, u_point.x.petsc_vec)
    u_point.x.scatter_forward()

    w_point = u_point.sub(0).collapse()

    return u_point, w_point

def conduct_frequency_response(domain, function_space, problem, Omega, plate_config, phi_0 = 1):

    m, _, a = define_weak_form(function_space, problem)
    bcs = dirichlet_boundary_conditions(
        domain, function_space, plate_config.length, plate_config.width, phi_0)

    A, form = assemble_dynamic_stiffness(a, m, Omega, bcs)

    b = assemble_shaker_rhs(A, form, bcs)

    _, w_point = solve_linear_system(domain, function_space, A, b)

    return w_point