"""Open checks on the finite 1x1 m plate (forced-response pipeline: Lagrange Q2, reduced 2x2 shear rule, free edges).

(a) Fourth zero-frequency mode. Candidate: the hourglass pattern w = (xi^2 - 1/3)(eta^2 - 1/3) in every element,
    theta = 0 (nodal values 4/9 at element corners, -2/9 at edge midpoints, 1/9 at element centres; continuous).
    Its shear strain grad w vanishes at the 2x2 Gauss points, so it costs no energy under the reduced rule.
    Checks: Rayleigh quotient with 2x2 vs 3x3 shear; zero-mode count; share of the hourglass vector in the zero-mode
    subspace. Effects: driving-point mobility, radiated power of the bare plate (radiation points sit at element
    centres, where the pattern is a uniform 1/9, so the radiation model sees it as a piston although its true mean
    is zero), and coupling to the resonators (they sit on element corners, value 4/9).
(c) Size of the effect on the insertion loss: IL of the 1x1 m plate with 10x10 resonators, 900-1500 Hz, with the
    pipeline's 2x2 shear rule (cached main.py sweeps) vs full 3x3 (bare and resonator plate swept in that window,
    ~25 min, cached). Run with the argument "insertion_loss"; without it, (a) and (b) run.
(b) Modes of the plate with the 10x10 resonators (f_r 1.2 kHz, mass ratio 0.25) in the 1240-1290 Hz cluster, where
    the insertion loss has its -14 dB dip. Hypothesis: these are small-k (Gamma-like) modes of the flat resonator
    branch, whose plate motion is supersonic and radiates efficiently.

Run in fenicsx-env-complex (a few minutes)."""
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import eigsh

from config import PlateConfig
from problem_setup import build_plate_problem
from weak_form import define_weak_form, collapse_w_subspace
from solve import assemble_plate_matrix
from augmented_system import petsc_to_scipy, augment_stiffness_matrix, augment_mass_matrix
from point_coupling import compute_phi_and_dofs_for_vamm_list
from probe import make_probe
from radiation import build_radiation_points, radiated_power, rho_air, c_air
from shaker_force import ShakerParameters
from vamm_grid import GridConfig, state_to_vamm_list
from frequency_sweep import build_frequency_grid
from parameter_study import FinitePlateSetup, cached_bare_radiated_power, windowed_insertion_loss

FULL_SHEAR_DEGREE = 4      # 3x3 Gauss for the shear term (Bloch-cell default); None = reduced 2x2 (pipeline)


def production_plate():
    """Same plate, shaker and resonator grid as main.py (cache keys must match)."""
    plate_config = PlateConfig(length=1.0, width=1.0, thickness=0.005, nx=100, ny=100, rho=7850, mu=77e9,
                               lambda_=115e9)
    domain, function_space, _, problem = build_plate_problem(plate_config, deg=2, el_type="Lagrange")
    shaker_config = ShakerParameters(x=plate_config.length / 3, y=plate_config.width / 3, force_amplitude=1.0, phase=0.0)
    grid_cell_length = 0.1
    grid_resonator_mass = 0.25 * plate_config.rho * plate_config.thickness * grid_cell_length ** 2
    grid_config = GridConfig(n_x=10, n_y=10, stiffness=grid_resonator_mass * (2 * np.pi * 1200.0) ** 2,
                             mass=grid_resonator_mass, gamma=0.02)
    vamm_list = state_to_vamm_list(np.repeat(1, 100), grid_config, plate_config.length, plate_config.width)
    return plate_config, domain, function_space, problem, shaker_config, vamm_list


def plate_matrices(function_space, problem, shear_quadrature_degree=None):
    """Free-plate K and M as PETSc (for augmentation) and scipy (real) matrices."""
    mass_form, _, stiffness_form = define_weak_form(function_space, problem,
                                                    shear_quadrature_degree=shear_quadrature_degree)
    stiffness_petsc = assemble_plate_matrix(stiffness_form, [])
    mass_petsc = assemble_plate_matrix(mass_form, [])
    return stiffness_petsc, mass_petsc, petsc_to_scipy(stiffness_petsc).real.tocsc(), petsc_to_scipy(mass_petsc).real.tocsc()


def hourglass_vector(function_space, plate_config, mass):
    """Hourglass pattern on the w-DOFs (theta = 0), mass-normalised. Element corner nodes sit on multiples of the
    element size h, midside/centre nodes half-way: factor 2/3 or -1/3 per direction."""
    w_space, w_to_parent = collapse_w_subspace(function_space)
    coordinates = w_space.tabulate_dof_coordinates()
    element_size_x, element_size_y = plate_config.length / plate_config.nx, plate_config.width / plate_config.ny

    def factor(coordinate, size):
        on_corner = np.abs(coordinate / size - np.round(coordinate / size)) < 1e-6
        return np.where(on_corner, 2.0 / 3.0, -1.0 / 3.0)
    vector = np.zeros(mass.shape[0])
    vector[w_to_parent] = factor(coordinates[:, 0], element_size_x) * factor(coordinates[:, 1], element_size_y)
    return vector / np.sqrt(vector @ (mass @ vector)), w_to_parent


def translation_vector(function_space, mass):
    """Rigid translation w = 1, theta = 0, mass-normalised."""
    _, w_to_parent = collapse_w_subspace(function_space)
    vector = np.zeros(mass.shape[0])
    vector[w_to_parent] = 1.0
    return vector / np.sqrt(vector @ (mass @ vector))


def values_at_probes(vector, w_to_parent, probes):
    """w of a full plate vector at probes (probe indices refer to the collapsed w-space)."""
    w_values = vector[w_to_parent]
    return np.array([probe.phi @ w_values[probe.local_to_global_w] for probe in probes])


def lowest_modes(stiffness, mass, count, target_hz=0.0, shift=-10.0):
    """count eigenpairs nearest target_hz (shift-invert; slightly negative shift at 0 for the singular free plate).
    Eigenvectors are mass-normalised, frequencies sorted."""
    sigma = (2 * np.pi * target_hz) ** 2 if target_hz > 0 else shift
    omega_squared, vectors = eigsh(stiffness, k=count, M=mass, sigma=sigma, which="LM")
    order = np.argsort(omega_squared)
    return np.sqrt(np.clip(omega_squared[order], 0, None)) / (2 * np.pi), omega_squared[order], vectors[:, order]


def augmented_matrices(stiffness_petsc, mass_petsc, vamm_list, domain, function_space):
    phi_list, global_dofs_parent_list, _ = compute_phi_and_dofs_for_vamm_list(
        domain=domain, function_space=function_space, vamm_list=vamm_list)
    stiffness_aug, plate_dof_count = augment_stiffness_matrix(K=stiffness_petsc, vamm_list=vamm_list, phi_list=phi_list,
                                                              global_dofs_parent_list=global_dofs_parent_list)
    mass_aug = augment_mass_matrix(M=mass_petsc, vamm_list=vamm_list)
    return petsc_to_scipy(stiffness_aug).real.tocsc(), petsc_to_scipy(mass_aug).real.tocsc(), plate_dof_count


# --- (a) ---

def check_hourglass(setup_objects):
    plate_config, domain, function_space, problem, shaker_config, vamm_list = setup_objects
    print("(a) fourth zero mode of the free plate", flush=True)
    stiffness_reduced_petsc, mass_petsc, stiffness_reduced, mass = plate_matrices(function_space, problem)
    _, _, stiffness_full, _ = plate_matrices(function_space, problem, shear_quadrature_degree=FULL_SHEAR_DEGREE)
    hourglass, w_to_parent = hourglass_vector(function_space, plate_config, mass)
    translation = translation_vector(function_space, mass)
    for name, stiffness in [("2x2 shear", stiffness_reduced), ("3x3 shear", stiffness_full)]:
        rayleigh = hourglass @ (stiffness @ hourglass)
        print(f"   hourglass Rayleigh quotient, {name}: omega^2 = {rayleigh:.3e} -> "
              f"{np.sqrt(max(rayleigh, 0)) / (2 * np.pi):.1f} Hz")
    for name, stiffness in [("2x2 shear", stiffness_reduced), ("3x3 shear", stiffness_full)]:
        frequencies, _, vectors = lowest_modes(stiffness, mass, count=8)
        zero_count = int(np.sum(frequencies < 1.0))
        message = f"   lowest 8 modes, {name}: {np.round(frequencies, 3)} Hz -> {zero_count} zero modes"
        if name == "2x2 shear":
            zero_space = vectors[:, :zero_count]
            captured = np.sum((zero_space.T @ (mass @ hourglass)) ** 2)
            message += f"; hourglass vector lies in the zero-mode subspace to {captured:.6f}"
        print(message, flush=True)

    # Effect 1: driving point. Modal mobility of a zero mode with unit modal mass: |Y| = phi_s^2 / omega.
    shaker_probe = make_probe(domain, function_space, shaker_config.x, shaker_config.y)
    hourglass_at_shaker = values_at_probes(hourglass, w_to_parent, [shaker_probe])[0]
    translation_at_shaker = values_at_probes(translation, w_to_parent, [shaker_probe])[0]
    print(f"   driving point: hourglass / rigid-translation mobility = "
          f"{hourglass_at_shaker ** 2 / translation_at_shaker ** 2:.4f} (frequency independent, both mass-like)")

    # Effect 2: radiated power of the bare plate, hourglass part alone vs the cached total.
    radiation_probes, cell_area, distances = build_radiation_points(domain, function_space, plate_config)
    hourglass_at_centres = values_at_probes(hourglass, w_to_parent, radiation_probes)
    print(f"   hourglass at the radiation points (element centres): mean {hourglass_at_centres.mean():.4e}, "
          f"std {hourglass_at_centres.std():.1e} (a uniform piston); true area mean "
          f"{(translation @ (mass @ hourglass)):.1e} (M-inner product with translation)")
    setup = FinitePlateSetup(domain=domain, function_space=function_space, problem=problem, plate_config=plate_config,
                             shaker_config=shaker_config, frequency_grid=build_frequency_grid(f_start=500, f_end=5000,
                                                                                               gamma=0.02),
                             gamma_plate=0.04, free_plate=True, excitation="force", deg=2, el_type="Lagrange")
    bare_power = cached_bare_radiated_power(setup)
    subset = np.arange(0, len(setup.frequency_grid), 12)
    subset_frequencies = setup.frequency_grid[subset]
    angular = 2 * np.pi * subset_frequencies
    hourglass_mobility = (1j * angular)[None, :] * hourglass_at_centres[:, None] * hourglass_at_shaker \
        / (-angular[None, :] ** 2)
    hourglass_power = radiated_power(hourglass_mobility, distances, cell_area, subset_frequencies)
    share_db = 10 * np.log10(hourglass_power / bare_power[subset])
    print(f"   bare plate: hourglass-only radiated power / total (cached) on {len(subset)} frequencies: "
          f"median {np.median(share_db):.1f} dB, max {share_db.max():.1f} dB at {subset_frequencies[np.argmax(share_db)]:.0f} Hz",
          flush=True)

    # Effect 3: coupling to the resonators.
    stiffness_aug_reduced, mass_aug, plate_dof_count = augmented_matrices(
        stiffness_reduced_petsc, mass_petsc, vamm_list, domain, function_space)
    stiffness_full_petsc, _, _, _ = plate_matrices(function_space, problem, shear_quadrature_degree=FULL_SHEAR_DEGREE)
    stiffness_aug_full, _, _ = augmented_matrices(stiffness_full_petsc, mass_petsc, vamm_list, domain, function_space)
    resonator_probes = [make_probe(domain, function_space, vamm.x, vamm.y) for vamm in vamm_list]
    hourglass_at_resonators = values_at_probes(hourglass, w_to_parent, resonator_probes)
    translation_at_resonators = values_at_probes(translation, w_to_parent, resonator_probes)
    resonator_masses = np.array([vamm.mass for vamm in vamm_list])
    resonator_frequency_hz = np.sqrt(vamm_list[0].stiffness / vamm_list[0].mass) / (2 * np.pi)
    predicted_hz = resonator_frequency_hz * np.sqrt(1 + np.sum(resonator_masses * hourglass_at_resonators ** 2)
                                                    + np.sum(resonator_masses * translation_at_resonators ** 2))
    print(f"   resonators: hourglass value at the attachments {hourglass_at_resonators.mean():.4f} "
          f"(translation {translation_at_resonators.mean():.4f}); three-mass estimate of the spurious mode "
          f"(hourglass + translation + in-phase resonators): {predicted_hz:.0f} Hz", flush=True)
    hourglass_aug = np.concatenate([hourglass, np.zeros(len(vamm_list))])
    spurious = {}
    for name, stiffness_aug in [("2x2 shear", stiffness_aug_reduced), ("3x3 shear", stiffness_aug_full)]:
        frequencies, _, vectors = lowest_modes(stiffness_aug, mass_aug, count=30, target_hz=predicted_hz)
        participation = (vectors.T @ (mass_aug @ hourglass_aug)) ** 2
        best = int(np.argmax(participation))
        spurious[name] = frequencies[best]
        print(f"   {name}: modes {frequencies[0]:.0f}-{frequencies[-1]:.0f} Hz, largest hourglass participation "
              f"{participation[best]:.3f} at {frequencies[best]:.1f} Hz (sum over the window {participation.sum():.3f})",
              flush=True)

    # Insertion loss of the main.py run near the spurious mode (cached full sweep with resonators).
    frequencies_il, insertion_loss_db = windowed_insertion_loss(setup, vamm_list, (0.0, np.inf), bare_power,
                                                                require_cached=True)
    near = np.abs(frequencies_il - spurious["2x2 shear"]) < 150
    print(f"   cached IL within +-150 Hz of {spurious['2x2 shear']:.0f} Hz: min {insertion_loss_db[near].min():+.1f} dB at "
          f"{frequencies_il[near][np.argmin(insertion_loss_db[near])]:.0f} Hz, max {insertion_loss_db[near].max():+.1f} dB")
    return dict(stiffness_aug_reduced=stiffness_aug_reduced, mass_aug=mass_aug, plate_dof_count=plate_dof_count,
                stiffness_reduced=stiffness_reduced, mass=mass, hourglass=hourglass, translation=translation,
                w_to_parent=w_to_parent, radiation=(radiation_probes, cell_area, distances))


# --- (b) ---

def describe_modes(frequencies, vectors, matrices, resonator_count, label):
    """Per mode: plate share of the modal mass, rigid-translation content of the plate part (true net volume
    velocity), resonator phase coherence |sum u_r| / sum |u_r| (1 = all in phase, Gamma-like), hourglass content,
    and the radiation efficiency with the pipeline's element-centre sampling, without and with the hourglass part."""
    mass, hourglass, translation, w_to_parent = (matrices[key] for key in ("mass", "hourglass", "translation",
                                                                           "w_to_parent"))
    radiation_probes, cell_area, distances = matrices["radiation"]
    plate_area = 1.0
    print(f"   {label}:")
    print("      f [Hz]   plate share  translation  resonator coherence  hourglass   sigma (pipeline)  sigma (no hourglass)")
    for frequency, vector in zip(frequencies, vectors.T):
        plate_part = vector[:mass.shape[0]]
        plate_norm = plate_part @ (mass @ plate_part)
        translation_content = (translation @ (mass @ plate_part)) ** 2 / plate_norm
        hourglass_amplitude = hourglass @ (mass @ plate_part)
        hourglass_content = hourglass_amplitude ** 2 / plate_norm
        resonator_part = vector[mass.shape[0]:]
        coherence = abs(resonator_part.sum()) / np.abs(resonator_part).sum() if resonator_count else np.nan
        efficiencies = []
        for field in (plate_part, plate_part - hourglass_amplitude * hourglass):
            velocity = 1j * 2 * np.pi * frequency * values_at_probes(field, w_to_parent, radiation_probes)
            power = radiated_power(velocity[:, None], distances, cell_area, [frequency])[0]
            efficiencies.append(power / (rho_air * c_air * plate_area * np.mean(np.abs(velocity) ** 2)))
        print(f"      {frequency:7.1f}   {plate_norm:9.3f}   {translation_content:10.4f}   {coherence:12.3f}        "
              f"{hourglass_content:8.4f}   {efficiencies[0]:12.4f}      {efficiencies[1]:12.4f}", flush=True)


def check_cluster(setup_objects, matrices, centre_hz=1265.0, count=40):
    vamm_list = setup_objects[5]
    print(f"(b) modes near {centre_hz:.0f} Hz with resonators (2x2 shear, as in the pipeline)", flush=True)
    frequencies, _, vectors = lowest_modes(matrices["stiffness_aug_reduced"], matrices["mass_aug"], count=count,
                                           target_hz=centre_hz)
    keep = (frequencies > 1235) & (frequencies < 1295)
    describe_modes(frequencies[keep], vectors[:, keep], matrices, len(vamm_list), "plate with 10x10 resonators")
    frequencies_bare, _, vectors_bare = lowest_modes(matrices["stiffness_reduced"], matrices["mass"], count=8,
                                                     target_hz=centre_hz)
    describe_modes(frequencies_bare, vectors_bare, matrices, 0, "bare plate (reference)")


# --- (c) ---

def compare_shear_rules_insertion_loss(setup_objects, frequency_window_hz=(900.0, 1500.0)):
    from parameter_study import windowed_radiated_power
    from insertion_loss import global_insertion_loss_radiated_db
    from postprocessing_dispersion import plot_insertion_loss_vs_gaps
    import dataclasses
    plate_config, domain, function_space, problem, shaker_config, vamm_list = setup_objects
    setup_reduced = FinitePlateSetup(domain=domain, function_space=function_space, problem=problem,
                                     plate_config=plate_config, shaker_config=shaker_config,
                                     frequency_grid=build_frequency_grid(f_start=500, f_end=5000, gamma=0.02),
                                     gamma_plate=0.04, free_plate=True, excitation="force", deg=2, el_type="Lagrange")
    setup_full = dataclasses.replace(setup_reduced, shear_quadrature_degree=FULL_SHEAR_DEGREE)
    full_grid = np.asarray(setup_reduced.frequency_grid)
    in_window = (full_grid >= frequency_window_hz[0]) & (full_grid <= frequency_window_hz[1])

    # 2x2: the cached full-range sweeps of main.py, sliced to the window
    frequencies_all, insertion_loss_reduced_all = windowed_insertion_loss(
        setup_reduced, vamm_list, (0.0, np.inf), cached_bare_radiated_power(setup_reduced), require_cached=True)
    frequencies = frequencies_all[in_window]
    insertion_loss_reduced = insertion_loss_reduced_all[in_window]
    # 3x3: bare and resonator plate swept in the window only
    _, bare_power_full = windowed_radiated_power(setup_full, None, frequency_window_hz)
    _, vamm_power_full = windowed_radiated_power(setup_full, vamm_list, frequency_window_hz)
    insertion_loss_full = global_insertion_loss_radiated_db(bare_power_full, vamm_power_full)

    difference = insertion_loss_full - insertion_loss_reduced
    print(f"(c) IL 3x3 minus 2x2 shear, {frequency_window_hz[0]:.0f}-{frequency_window_hz[1]:.0f} Hz: "
          f"mean {difference.mean():+.2f} dB, max |difference| {np.abs(difference).max():.1f} dB at "
          f"{frequencies[np.argmax(np.abs(difference))]:.0f} Hz")
    for name, values in [("2x2", insertion_loss_reduced), ("3x3", insertion_loss_full)]:
        print(f"   {name}: IL max {values.max():+.1f} dB at {frequencies[np.argmax(values)]:.0f} Hz, "
              f"min {values.min():+.1f} dB at {frequencies[np.argmin(values)]:.0f} Hz")
    plot_insertion_loss_vs_gaps(
        [dict(label="$f_r$ = 1200 Hz, mass ratio 0.25", f_values=frequencies, insertion_loss_db=insertion_loss_reduced,
              curve_label="$IL_{rad}$, 2x2 shear (pipeline)", gaps=[(1031.4, 1195.3)], resonator_frequency_hz=1200.0,
              extra_curves=[("$IL_{rad}$, 3x3 shear (no hourglass)", frequencies, insertion_loss_full)])],
        title="1x1 m plate, 10x10 resonators: insertion loss with reduced vs full shear integration",
        save_path="results/insertion_loss_shear_rule_1x1m_100resonators_1200hz_900to1500hz.pdf")


if __name__ == "__main__":
    import sys
    setup_objects = production_plate()
    if "insertion_loss" in sys.argv[1:]:
        compare_shear_rules_insertion_loss(setup_objects)
    else:
        matrices = check_hourglass(setup_objects)
        check_cluster(setup_objects, matrices)
