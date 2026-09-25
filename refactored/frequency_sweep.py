import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from petsc4py import PETSc
from weak_form import define_weak_form
from augmented_system import augment_stiffness_matrix, augment_mass_matrix
from boundary_conditions import dirichlet_boundary_conditions, make_border_marker
from weak_form import collapse_subspace
from shaker_force import make_force_excitation_rhs_builder

# --- Prescribed motion machinery ---
# --- NOT implemented in the current version: Force excitation only ---
# --- Implementation for prescribed motion to be implemented ---

def assemble_dynamic_stiffness(a, m, domain, gamma: float = 0.0):
    """Build the (damping_factor * a - Omega² m) form ONCE, with Omega as a mutable Constant.
    Reduces to (a - Omega² m), i. e., the undampened case if gamma is not specified.
    Returns the compiled form and the Constant so the caller can change
    Omega cheaply later without triggering a JIT recompile."""
    Omega_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    damping_factor = fem.Constant(domain, PETSc.ScalarType(1.0 + 1j * gamma))
    form = fem.form(damping_factor * a - Omega_const**2 * m)
    return form, Omega_const

def assemble_shaker_rhs(K_plate, form, bcs=None):
    """K_plate must be a PLATE-SIZED matrix, never the augmented operator —
    used only for createVecRight() layout. Returns a plate-sized vector."""
    b = K_plate.createVecRight()
    b.zeroEntries()                                   # no body force — pure homogeneous eqn
    if bcs:
        fem.petsc.apply_lifting(b, [form], bcs=[bcs])      # computes -(K12 - Ω²M12) q2 internally
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    if bcs:
        fem.petsc.set_bc(b, bcs)                           # writes q2 = u2*phi_0 into boundary rows
    return b

# --- Decide on shaker implementation ---

def build_rhs_builder(excitation, domain, function_space, bcs, shaker_config=None):
    if excitation == "force":
        if shaker_config is None:
            raise ValueError("force excitation requires a shaker_config")
        return make_force_excitation_rhs_builder(domain, function_space, shaker_config)
    elif excitation == "motion":
        if not bcs:
            raise ValueError("motion excitation requires a clamped plate (free_plate=False)")
        return assemble_shaker_rhs  # already matches the (A, form, bcs) signature
    else:
        raise ValueError(f"unknown excitation kind: {excitation!r}")

def pad_to_augmented_size(b_plate, n_plate, n_total, comm):
    """Zero-pad a plate-sized RHS vector to the augmented (n_plate+N) size.
    No-op when there are no VAMMs (n_total == n_plate). Zero-padding is exact, since
    neither a point force nor a plate boundary BC has any
    direct action on a VAMM's own free coordinate, so those extra rows are
    always zero, not merely negligible."""
    if n_total == n_plate:
        return b_plate
    b_full = PETSc.Vec().create(comm=comm)
    b_full.setSizes(n_total)
    b_full.setUp()
    b_full.zeroEntries()
    idx = np.arange(n_plate, dtype=PETSc.IntType)
    b_full.setValues(idx, b_plate.getArray())
    b_full.assemblyBegin()
    b_full.assemblyEnd()
    return b_full

def locate_boundary_w_dofs_of_plate(domain, function_space, border_function):

    topological_dimension_mesh = domain.topology.dim
    facet_dim = topological_dimension_mesh - 1

    clamped_facets = mesh.locate_entities_boundary(domain, facet_dim, border_function)
    function_space_w, _ = collapse_subspace(function_space, 0)
    dofs_w = fem.locate_dofs_topological((function_space.sub(0), function_space_w), facet_dim, clamped_facets)
    dofs_w_collapsed = dofs_w[1]

    return dofs_w_collapsed

# --- Force excitation machinery ---

def assemble_dynamic_operators(a, m, bcs, domain, gamma: float = 0.0):
    """Assemble K_plate and M_plate ONCE, gamma baked into K_plate as complex
    hysteretic damping (1+i*gamma). No Omega dependence -- callers combine
    K_aug - Omega^2*M_aug via cheap PETSc arithmetic per frequency step,
    not FEM reassembly."""
    damping_factor = fem.Constant(domain, PETSc.ScalarType(1.0 + 1j * gamma))
    K_plate = fem.petsc.assemble_matrix(fem.form(damping_factor * a), bcs=bcs)
    K_plate.assemble()
    M_plate = fem.petsc.assemble_matrix(fem.form(m), bcs=bcs)
    M_plate.assemble()
    return K_plate, M_plate

# --- Frequency sweep machinery, independent of shaker implementation ---

def compute_rms_velocity(u_point, Omega_const, domain):
    """Spatial RMS of the plate velocity field at a single frequency Omega.
    v = i*Omega*w, so |v|^2 = Omega^2 * |w|^2 -- the i drops out under magnitude.
    Returns a real float; the |.|^2 field is mesh-integrated (not a discrete
    point-average), so this is smooth in Omega and immune to the argmax-DOF
    discontinuity seen with the pointwise peak-deflection metric.

    Omega_const must be the same fem.Constant already updated by the caller
    for this frequency step (not a raw Python float) -- passing a bare 0.0
    causes UFL to constant-fold the whole integrand to Zero, which loses
    the integration domain and raises "missing an integration domain".
    """
    w, theta = ufl.split(u_point)  # w is the deflection sub-field of the mixed function

    area_form = fem.form(fem.Constant(domain, PETSc.ScalarType(1.0)) * ufl.dx)
    area = domain.comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)

    mean_sq_form = fem.form(Omega_const**2 * ufl.inner(w, w) * ufl.dx)
    mean_sq_velocity = domain.comm.allreduce(fem.assemble_scalar(mean_sq_form), op=MPI.SUM) / area

    return np.sqrt(mean_sq_velocity.real)

def solve_linear_system(domain, function_space, A, b, n_plate):

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")

    x = A.createVecRight()
    solver.solve(b, x)

    plate_part = x.getArray()[:n_plate]
    u_point = fem.Function(function_space)
    u_point.x.petsc_vec.setArray(plate_part)
    u_point.x.scatter_forward()

    w_point = u_point.sub(0).collapse()

    return u_point, w_point

def conduct_single_frequency_response(domain, function_space, K_plate, K_aug, M_aug,
                                       Omega, rhs_builder, n_plate, bcs=None, form=None):
    """ Does NOT support prescribed motion excitation yet """
    A = K_aug.copy()
    A.axpy(-Omega**2, M_aug)   # A = K_aug - Omega^2 * M_aug
    A.assemble()
    n_total = A.getSize()[0]

    if bcs and form is None:
        raise ValueError(f"Nonzero bcs requires form to be passed as well")

    b_plate = rhs_builder(K_plate=K_plate, form=form, bcs=bcs)
    # always plate-sized (n_plate), regardless of excitation kind
    b = pad_to_augmented_size(b_plate, n_plate, n_total, domain.comm)

    u_point, w_point = solve_linear_system(domain, function_space, A, b, n_plate)
    return u_point, w_point

def frequency_sweep_plate(
        domain, function_space, problem, plate_config,
        f_start: float, f_end: float,
        vamm_list = None, phi_list=None, global_dofs_parent_list=None,
        Omega_size: int = 100, phi_0: float = 1.0, gamma: float = 0.0,
        compute_max_metric: bool = False,
        free_plate = True,
        excitation: str = "force",
        shaker_config = None):

    """
    Builds the damped/undamped dynamic-stiffness form (see
    assemble_dynamic_stiffness) once, then solves the shaker-driven
    linear system at each frequency in [f_start, f_end]. Always returns
    the RMS surface velocity (a smooth, spatially-integrated metric that
    is robust across the whole sweep). Optionally also returns the
    signed peak plate deflection (a pointwise metric, useful near an
    isolated resonance for its phase-flip sign, but not reliable as a
    smooth function of frequency between resonances -- see notes in
    compute_rms_velocity).
    """

    Omega_start, Omega_end = 2 * np.pi * f_start, 2 * np.pi * f_end
    Omega_values = np.linspace(Omega_start, Omega_end, Omega_size)
    f_values = Omega_values / (2 * np.pi)

    # --- everything Omega-INdependent: build ONCE, outside the loop ---
    m, _, a = define_weak_form(function_space, problem)

    # plate-only, unconstrained, Omega-mutable form -- ONLY used for apply_lifting/set_bc
    form, Omega_const = assemble_dynamic_stiffness(a=a, m=m, domain=domain, gamma=gamma)

    if free_plate:
        bcs = []
        border = None
    else:
        bcs = dirichlet_boundary_conditions(
            domain, function_space, plate_config.length, plate_config.width, phi_0)
        border = make_border_marker(plate_config.length, plate_config.width)

    K_plate, M_plate = assemble_dynamic_operators(a=a, m=m, bcs=bcs, domain=domain, gamma=gamma)

    if vamm_list is not None:
        K_aug, n_plate = augment_stiffness_matrix(K=K_plate, vamm_list=vamm_list,
                                                  phi_list=phi_list,
                                                  global_dofs_parent_list=global_dofs_parent_list)
        M_aug = augment_mass_matrix(M=M_plate, vamm_list=vamm_list)
    else:
        K_aug, M_aug, n_plate = K_plate, M_plate, K_plate.getSize()[0]

    rhs_builder = build_rhs_builder(excitation, domain, function_space, bcs, shaker_config)

    if compute_max_metric:
        if free_plate:
            dofs_w_collapsed = None # no boundary to zero out — see loop below
        else:
            dofs_w_collapsed = locate_boundary_w_dofs_of_plate(domain=domain, function_space=function_space,
                                                       border_function=border)
        w_max_plate = []

    rms_velocity = []
    for Omega in Omega_values:
        Omega_const.value = Omega
        u_point, w_point = conduct_single_frequency_response(
            domain=domain, function_space=function_space, K_plate=K_plate, K_aug=K_aug, M_aug=M_aug, Omega=Omega,
            rhs_builder=rhs_builder, n_plate=n_plate, bcs=bcs, form=form)

        if compute_max_metric:
            if dofs_w_collapsed is not None:
                w_point.x.array[dofs_w_collapsed] = 0
            w_max_plate.append(max(w_point.x.array, key=abs).real)

        rms_velocity.append(compute_rms_velocity(u_point=u_point, Omega_const=Omega_const, domain=domain))

    if compute_max_metric:
        return f_values, np.array(w_max_plate), np.array(rms_velocity)
    else:
        return f_values, None, np.array(rms_velocity)
