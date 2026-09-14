import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from weak_form import define_weak_form, PlateProblemConstants
from boundary_conditions import dirichlet_boundary_conditions, make_border_marker
from weak_form import collapse_subspace
from petsc4py.PETSc import ScalarType

def assemble_dynamic_stiffness(a, m, domain, bcs):
    """Build the (a - Omega^2 m) form ONCE, with Omega as a mutable Constant.
    Returns the compiled form and the Constant so the caller can change
    Omega cheaply later without triggering a JIT recompile."""
    Omega_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    form = fem.form(a - Omega_const**2 * m)
    return form, Omega_const

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
    solver.solve(b, u_point.x.petsc_vec) # currently solves real problem only (no damping)
    u_point.x.scatter_forward()

    w_point = u_point.sub(0).collapse()

    return u_point, w_point

def conduct_single_frequency_response(domain, function_space, form, bcs, Omega_const, Omega):
    """Now only does the Omega-dependent work: set Omega, assemble, solve."""
    Omega_const.value = Omega

    A = fem.petsc.assemble_matrix(form, bcs=bcs)
    A.assemble()

    b = assemble_shaker_rhs(A, form, bcs)

    _, w_point = solve_linear_system(domain, function_space, A, b)

    return w_point

def locate_boundary_w_dofs_of_plate(domain, function_space, border_function):

    topological_dimension_mesh = domain.topology.dim
    facet_dim = topological_dimension_mesh - 1

    clamped_facets = mesh.locate_entities_boundary(domain, facet_dim, border_function)
    function_space_w, _ = collapse_subspace(function_space, 0)
    dofs_w = fem.locate_dofs_topological((function_space.sub(0), function_space_w), facet_dim, clamped_facets)
    dofs_w_collapsed = dofs_w[1]

    return dofs_w_collapsed

def frequency_sweep_plate(
        domain, function_space, problem, plate_config, border,
        f_start: float, f_end: float, Omega_size: int = 100, phi_0: float = 1.0):

    Omega_start, Omega_end = 2 * np.pi * f_start, 2 * np.pi * f_end
    Omega_values = np.linspace(Omega_start, Omega_end, Omega_size)
    f_values = Omega_values / (2 * np.pi)

    # --- everything Omega-INdependent: build ONCE, outside the loop ---
    m, _, a = define_weak_form(function_space, problem)
    bcs = dirichlet_boundary_conditions(
        domain, function_space, plate_config.length, plate_config.width, phi_0)
    form, Omega_const = assemble_dynamic_stiffness(a=a, m=m, domain=domain, bcs=bcs)
    dofs_w_collapsed = locate_boundary_w_dofs_of_plate(domain=domain, function_space=function_space,
                                                       border_function=border)

    w_max_plate = []
    for Omega in Omega_values:
        w_point = conduct_single_frequency_response(domain=domain, function_space=function_space,
                                                    form=form, bcs=bcs, Omega_const=Omega_const, Omega=Omega)

        w_point.x.array[dofs_w_collapsed] = 0
        w_max_plate.append(max(w_point.x.array, key=abs))

    return f_values, w_max_plate
