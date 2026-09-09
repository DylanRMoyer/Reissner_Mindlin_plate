from dolfinx import fem
from petsc4py import PETSc
from augmented_system import add_condensed_spring_stiffness
from weak_form import define_weak_form
from point_coupling import locate_target_cell_degrees_of_freedom, VAMMConfig, locate_cell_and_reference_coords
from problem_setup import build_plate_problem
from config import PlateConfig
from dolfinx.fem.petsc import assemble_matrix

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


if __name__ == "__main__":

    mesh_sizes = [(5,4), (10, 8), (15, 12), (20, 17), (40, 33), (60, 50), (80, 67), (160, 134)]
    deflections = []

    for nx, ny in mesh_sizes:
        plate_config = PlateConfig(
            length=1.1,
            width=1,
            thickness=0.02,
            nx=nx,
            ny=ny,
            rho=7850,
            mu=77e9,
            lambda_=115e9,
            #constant_force = 1 # Use if wanting to prescribe a custom constant force across the plate
        )

        domain, function_space, bcs, plate_problem_constants = build_plate_problem(plate_config)
        # Add degree custom degree (default: deg = 2) or function type (default: el_type = "S") if needed

        vamm_config = VAMMConfig(
            target_x=0.137,
            target_y=0.912,
            point_stiffness=100,
            point_mass=1
        )

        cell, x_ref = locate_cell_and_reference_coords(domain, vamm_config.target_x, vamm_config.target_y)

        _, _, a = define_weak_form(function_space, plate_problem_constants)

        phi, global_dofs_parent, local_to_global_w \
        = locate_target_cell_degrees_of_freedom(domain, function_space, vamm_config)

        spring_end_displacement = 1

        A, bilinear_form = assemble_condensed_stiffness_matrix(a, bcs, phi, global_dofs_parent, vamm_config)
        b = assemble_spring_load_vector(A, bilinear_form, bcs, phi, global_dofs_parent, vamm_config, spring_end_displacement)

        w_point = solve_point_load(domain, function_space, A, b)

        deflections.append(max(abs(w_point.x.array)))
        print(f"nx={nx}, ny={ny}: deflection = {deflections[-1]:.6e}, cell = {cell}, x_ref = {x_ref}, phi = {phi}")
        if len(deflections) > 1:
            print(f"relative increase = {100*(deflections[-1] - deflections[-2])/deflections[-2]:.3g} %")
