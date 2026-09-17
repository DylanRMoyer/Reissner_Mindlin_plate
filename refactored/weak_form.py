from dolfinx import fem, default_scalar_type
import basix
import ufl
import numpy as np
from dataclasses import dataclass


# --- Define constants needed for RM plate theory ---

@dataclass
class PlateProblemConstants:
    domain: object
    thick: fem.Constant
    rho_const: fem.Constant
    nu: fem.Constant
    E: fem.Constant
    D: ufl.core.expr.Expr
    F: ufl.core.expr.Expr
    f: fem.Constant

def reissner_mindlin_constants(domain,
                               thickness: float,
                               rho: float,
                               mu: float,
                               lambda_: float,
                               constant_force: float = 0.0
                               ):
    thickness = float(thickness)
    rho = float(rho)
    mu = float(mu)
    lambda_ = float(lambda_)
    constant_force = float(constant_force)
    thick = fem.Constant(domain, default_scalar_type(thickness))
    rho_const = fem.Constant(domain, default_scalar_type(rho))
    nu = fem.Constant(domain, default_scalar_type(lambda_ / (2 * (lambda_ + mu))))
    E = fem.Constant(domain, default_scalar_type(mu * (3 * lambda_ + 2 * mu) / (mu + lambda_)))

    D = E * thick ** 3 / (1 - nu ** 2) / 12.0  # Plate bending rigidity
    F = E / 2 / (1 + nu) * thick * 5.0 / 6.0  # Shear stiffness
    f = fem.Constant(domain, constant_force)

    return PlateProblemConstants(domain = domain, thick = thick, rho_const = rho_const, nu = nu, E = E, D = D, F = F, f = f)


# --- Create function spaces and get necessary indices ---

def create_function_spaces(domain, deg:int = 2, el_type:str = "S"):
    We = basix.ufl.element(el_type, domain.basix_cell(), deg)
    Te = basix.ufl.element(el_type, domain.basix_cell(), deg, shape=(2,))

    function_space = fem.functionspace(domain, basix.ufl.mixed_element([We, Te]))

    return function_space

def collapse_subspace(function_space, subspace_index:int):
    subspace, to_parent = function_space.sub(subspace_index).collapse()
    return subspace, np.asarray(to_parent).reshape(-1)

def collapse_w_subspace(function_space):
    return collapse_subspace(function_space, subspace_index=0)


# --- Preliminary equations for the weak form ---

def strain2voigt(eps):
    return ufl.as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

def voigt2stress(S):
    return ufl.as_tensor([[S[0], S[2]], [S[2], S[1]]])

def curv(u):
    (w, theta) = ufl.split(u)
    return ufl.sym(ufl.grad(theta))

def extract_theta(u):
    (w, theta) = ufl.split(u)
    return theta

def shear_strain(u):
    (w, theta) = ufl.split(u)
    return ufl.grad(w) - theta

def bending_moment(u, D, nu):
    DD = ufl.as_tensor([[D, nu * D, 0], [nu * D, D, 0], [0, 0, D * (1 - nu) / 2.0]])
    return voigt2stress(ufl.dot(DD, strain2voigt(curv(u))))

def shear_force(u, F):
    return F * shear_strain(u)


# --- Define weak form ---

def define_weak_form(function_space, problem: PlateProblemConstants):

    u_ = ufl.TestFunction(function_space)
    du = ufl.TrialFunction(function_space)

    deg = function_space.ufl_element().degree

    dx = ufl.Measure("dx")
    dx_shear = ufl.Measure("dx", metadata={"quadrature_degree": 2 * deg - 2})

    m = (problem.rho_const * problem.thick * ufl.inner(du[0], u_[0]) * dx
         + problem.rho_const * problem.thick**3/12 * ufl.inner(extract_theta(du), extract_theta(u_)) * dx)

    L = problem.f * ufl.conj(u_[0]) * dx
    a = (
        ufl.inner(bending_moment(du, problem.D, problem.nu), curv(u_)) * dx
        + ufl.inner(shear_force(du, problem.F), shear_strain(u_)) * dx_shear
    )

    return m, L, a
