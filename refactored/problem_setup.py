from geometry import create_rectangular_plate_mesh
from weak_form import create_function_spaces, reissner_mindlin_constants
from boundary_conditions import dirichlet_boundary_conditions

def build_plate_problem(plate_config, deg=2, el_type="S"):
    """Assemble the shared plate problem setup (mesh, function space,
    BCs, and material/plate constants) from a PlateConfig.

    Does not build weak forms (m, L, a) — callers do that themselves via
    weak_form.define_weak_form(...), since the load L in particular can
    vary by use case (main pipeline vs. validation scripts).
    """
    domain = create_rectangular_plate_mesh(plate_config.length, plate_config.width,
                                            plate_config.nx, plate_config.ny)
    function_space = create_function_spaces(domain, deg=deg, el_type=el_type)
    bcs = dirichlet_boundary_conditions(domain, function_space,
                                         plate_config.length, plate_config.width)
    plate_problem_constants = reissner_mindlin_constants(
        domain, plate_config.thickness, plate_config.rho,
        plate_config.mu, plate_config.lambda_, plate_config.constant_force
    )
    return domain, function_space, bcs, plate_problem_constants