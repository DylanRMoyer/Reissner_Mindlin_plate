import numpy as np
import dolfinx
from probe import Probe
from weak_form import collapse_w_subspace
from vamm_grid import cell_center_positions   # after the small refactor

rho_air, c_air = 1.2, 343.0

def build_radiation_points(domain, function_space, plate_config):
    """One point per mesh cell. Returns probes, cell area dS, and pairwise distances r."""
    coords = cell_center_positions(plate_config.length, plate_config.width,
                                   plate_config.nx, plate_config.ny)

    pts = np.array([[x, y, 0.0] for x, y in coords])

    # once: locate all points in one batch
    tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, pts)
    colliding = dolfinx.geometry.compute_colliding_cells(domain, candidates, pts)

    # once: the collapsed w-space (this is what creates communicators)
    V_w, _ = collapse_w_subspace(function_space)
    element = V_w.element.basix_element
    cmap = domain.geometry.cmaps[0]

    probes = []
    for i, (x, y) in enumerate(coords):
        cell = colliding.links(i)[0]
        geom_dofs = np.array(domain.geometry.dofmaps[0][cell])
        x_ref = cmap.pull_back(pts[i:i + 1, :domain.geometry.dim],
                               np.array(domain.geometry.x[geom_dofs]))
        phi = element.tabulate(0, x_ref)[0, 0, :, 0]
        probes.append(Probe(x=x, y=y, phi=phi,
                            local_to_global_w=V_w.dofmap.cell_dofs(cell)))

    dS = (plate_config.length / plate_config.nx) * (plate_config.width / plate_config.ny)
    # area of one plate "piston"
    xy = np.asarray(coords)
    r = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    return probes, dS, r

def radiation_resistance_matrix(r, dS, omega):
    k = omega / c_air
    return (omega**2 * rho_air * dS**2 / (4 * np.pi * c_air)) * np.sinc(k * r / np.pi)
    # np.sinc(x) = sin(pi x)/(pi x), hence the division by pi; handles r = 0 automatically

def radiated_power(Y_grid, r, dS, f_values, force_amplitude=1.0):
    """Y_grid: probe_mobility of the radiation points, shape (n_points, n_freq)."""
    if Y_grid.shape[0] != r.shape[0]:
        raise ValueError(f"Y_grid has {Y_grid.shape[0]} points but r is {r.shape[0]}x{r.shape[0]}: "
                         "pass the mobility evaluated at the radiation points, not at the plot probes.")
    P = np.empty(len(f_values))
    for n, f in enumerate(f_values):
        v = Y_grid[:, n] * force_amplitude
        R = radiation_resistance_matrix(r, dS, 2 * np.pi * f)
        P[n] = np.real(np.vdot(v, R @ v))      # vdot conjugates its first argument: v^T R v
    return P

def radiation_efficiency(P, Y_grid, S, force_amplitude=1.0):
    v_sq_mean = np.mean(np.abs(Y_grid * force_amplitude)**2, axis=0)   # amplitude convention
    return P / (rho_air * c_air * S * v_sq_mean / 2)

def power_level_db(P, P_ref=1e-12):
    return 10 * np.log10(np.maximum(P, np.finfo(float).tiny) / P_ref)