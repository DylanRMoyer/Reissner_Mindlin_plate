"""
This file needs to be adapted to plate's geometry and boundary conditions should one wish to compare the results
to the analytical results of the Kirchhoff-Love plate theory.
One must look up the maximum deflection, see e. g. http://www.ltas-cm3.ulg.ac.be/MECA0028-1/StructAeroPlatesPart2.pdf

In this file, a fully clamped unit square plate is assumed. If one wanted to run this validation test with another
geometry, one must first look up the appropriate value to divide the plate bending rigidity D by.
"""

import ufl
from math import ceil
from dolfinx import fem
from config import PlateConfig
from problem_setup import build_plate_problem
from weak_form import define_weak_form
from dolfinx.fem.petsc import LinearProblem

# We then solve for the solution and print the deflection normalized with respect to the Love-Kirchhoff thin plate
# analytical solution:

def create_load_vector(function_space, plate_problem_constants, norm_to_one = True):
    u_ = ufl.TestFunction(function_space)
    dx = ufl.Measure("dx")

    if norm_to_one:
        f = -plate_problem_constants.D / 1.265319087e-3
        # with this we have w_Love-Kirchhoff = 1.0 FOR UNIT SQUARE PLATE ONLY!
        # ADAPT FOR DIFFERENT GEOMETRIES/ BOUNDARY CONDITIONS!
    else:
        f = plate_problem_constants # Constant force

    L = f * u_[0] * dx
    return L


def love_kirchhoff_check(function_space, a, L, bcs):
    u = fem.Function(function_space)
    problem = fem.petsc.LinearProblem(

    a, L, u=u, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="Reissner-Mindlin")
    problem.solve()
    w = u.sub(0).collapse()
    w.name = "Deflection"
    return w


def validate_love_kirchhoff(plate_config):

    domain, function_space, bcs, plate_problem_constants = build_plate_problem(plate_config)
    # Add degree custom degree (default: deg = 2) or function type (default: el_type = "S") if needed

    _, _, a = define_weak_form(function_space, plate_problem_constants)

    L = create_load_vector(function_space, plate_problem_constants)

    w = love_kirchhoff_check(function_space, a, L, bcs)

    return w



if __name__ == "__main__":

    initial_thickness = 0.02
    refinement_steps = 3 # Going any larger will dramatically increase the computational cost!
    thickness_and_cell_sizes = [(initial_thickness/n, ceil(n/initial_thickness)) for n in range(1, refinement_steps + 1)]

    w_max = []

    for thickness, n in thickness_and_cell_sizes:

        plate_config = PlateConfig(
            length=1,
            width=1,
            thickness=thickness,
            nx=n,
            ny=n,
            rho=7850,
            mu=77e9,
            lambda_=115e9,
            #constant_force = 1 # Use if wanting to prescribe a custom constant force across the plate
        )

        w = validate_love_kirchhoff(plate_config)

        # Extract the scalar peak deflection float value
        max_deflection = float(max(abs(w.x.array)))
        w_max.append(max_deflection)

        print(f"Reissner-Mindlin FE deflection: {max(abs(w.x.array)):.5f}")
        if len(w_max) > 1:
            rel_defl = 100 * (w_max[-1] - w_max[-2]) / w_max[-2]
            print(f"  relative increase: deflection = {rel_defl:.3g} %")

    gaps = [abs(w - 1) for w in w_max]
    assert gaps[-1] < gaps[0], f"Deflection is not converging toward 1: gaps = {gaps}"
    assert gaps[-1] < 0.01, f"Final deflection not close enough to 1: |w_max[-1]-1| = {gaps[-1]:.2%}"