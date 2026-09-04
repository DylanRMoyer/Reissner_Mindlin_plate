"""
This file needs to be adapted to plate's geometry and boundary conditions should one wish to compare the results
to the analytical results of the Kirchhoff-Love plate theory.
One must look up the maximum deflection, see e. g. http://www.ltas-cm3.ulg.ac.be/MECA0028-1/StructAeroPlatesPart2.pdf

In this file, a fully clamped unit square plate is assumed. If one wanted to run this validation test with another
geometry, one must first look up the appropriate value to divide the plate bending rigidity D by.
"""

import ufl
from dolfinx import fem

# We then solve for the solution and print the deflection normalized with respect to the Love-Kirchhoff thin plate
# analytical solution:

def create_load_vector(domain, function_space, plate_problem_constants, norm_to_one = True):
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
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="Reissner-Mendlin")
    problem.solve()
    w = u.sub(0).collapse()
    w.name = "Deflection"
    #print(f"Reissner-Mindlin FE deflection: {max(abs(w.x.array)):.5f}")
    return w

