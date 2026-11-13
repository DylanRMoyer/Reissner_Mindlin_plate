"""Stage 7 parameter study: complete band gaps of a square cell with one centred resonator, as a function of the
tuning f_r / f_X and the mass ratio mu, plus the insertion loss of the finite plate at a few points of that map.

Dimensionless form: f_X = lowest bare Reissner-Mindlin band at X (bending wavelength = 2 a, the first Bragg fold),
so cells of different size, thickness or material can be overlaid (validate_parameter_study.py checks the
plane-wave model used here against FE). Gaps come from dispersion.find_complete_gaps on the sorted bands, path
Gamma-X-M-Gamma; for this cell type the zone interior adds no extrema (checked near f_r = f_X in the validation).

Run everything in fenicsx-env-complex."""
import dataclasses
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dispersion import (build_wavevector_path, plate_constants, plane_wave_omega_squared,
                        analytic_resonator_dispersion, find_complete_gaps, dispersion_sweep)

FREQUENCY_RATIOS = np.round(np.arange(0.2, 2.5 + 1e-9, 0.05), 4)   # f_r / f_X
MASS_RATIOS = (0.05, 0.1, 0.25, 0.5, 1.0)                          # resonator mass / plate mass per cell
LIMIT_OVER_BRAGG = 4.2      # bands up to 4.2 f_X: above f_r sqrt(1 + mu) = 3.54 f_X for f_r / f_X = 2.5, mu = 1
MAXIMUM_GAP_COUNT = 8       # columns of the NaN-padded gap table
POINTS_PER_SEGMENT = 20     # wavevectors on Gamma-X (61 on the whole path)


def bragg_frequency_x(plate_config):
    """Lowest bare Reissner-Mindlin band at X (k = pi / Lx), in Hz: the first Bragg fold, used to normalise f_r.
    Analytic; the production FE mesh reproduces it within 0.1 %."""
    omega_squared = plane_wave_omega_squared(np.pi / plate_config.length, plate_constants(plate_config))
    return np.sqrt(omega_squared.min()) / (2 * np.pi)


def resonator_parameters(plate_config, resonator_frequency_hz, mass_ratio):
    """(stiffness, mass) of one resonator per cell: mass = mu rho h Lx Ly, stiffness = mass (2 pi f_r)^2."""
    mass = mass_ratio * plate_config.rho * plate_config.thickness * plate_config.length * plate_config.width
    return mass * (2 * np.pi * resonator_frequency_hz) ** 2, mass


def cell_band_gaps(plate_config, wavevector_path, resonator_frequency_hz, mass_ratio, maximum_frequency_hz,
                   method="plane_wave", function_space=None, problem=None, minimum_relative_width=0.005):
    """Complete gaps (list of dispersion.BandGap) of the cell with one resonator at its center.
    method "plane_wave": analytic_resonator_dispersion (no mesh needed); "fe": dispersion_sweep on the given
    function_space / problem (Lagrange, the cell of plate_config)."""
    stiffness, mass = resonator_parameters(plate_config, resonator_frequency_hz, mass_ratio)
    if method == "plane_wave":
        frequencies = analytic_resonator_dispersion(wavevector_path, plate_config, stiffness, mass,
                                                    maximum_frequency_hz=maximum_frequency_hz)
    elif method == "fe":
        from point_coupling import create_vamm_list_and_assign_indices
        vamm_list = create_vamm_list_and_assign_indices(
            [(plate_config.length / 2, plate_config.width / 2, stiffness, mass)])
        frequencies = dispersion_sweep(function_space, problem, plate_config, wavevector_path,
                                       maximum_frequency_hz=maximum_frequency_hz, vamm_list=vamm_list).frequencies
    else:
        raise ValueError(f"unknown method: {method!r}")
    return find_complete_gaps(frequencies, wavevector_path, minimum_relative_width=minimum_relative_width)


def gap_edges_row(gaps):
    """Gap list -> (MAXIMUM_GAP_COUNT, 2) array of (lower, upper) edges in Hz, ascending, NaN-padded."""
    edges = np.full((MAXIMUM_GAP_COUNT, 2), np.nan)
    if len(gaps) > MAXIMUM_GAP_COUNT:
        raise ValueError(f"{len(gaps)} gaps found; raise MAXIMUM_GAP_COUNT")
    for gap_index, gap in enumerate(gaps):
        edges[gap_index] = gap.lower_edge_hz, gap.upper_edge_hz
    return edges


@dataclasses.dataclass
class GapStudy:
    cell_length: float
    thickness: float
    bragg_frequency_hz: float          # f_X of the bare cell
    frequency_ratios: np.ndarray       # f_r / f_X, shape (n_f,)
    mass_ratios: np.ndarray            # shape (n_mu,)
    gap_edges_hz: np.ndarray           # shape (n_mu, n_f, MAXIMUM_GAP_COUNT, 2), NaN-padded

    def widest_relative_width(self):
        """Relative width (width / centre) of the widest complete gap below the computed limit, 0 if none;
        shape (n_mu, n_f)."""
        lower, upper = self.gap_edges_hz[..., 0], self.gap_edges_hz[..., 1]
        relative_width = np.nan_to_num((upper - lower) / (0.5 * (lower + upper)), nan=0.0)
        return relative_width.max(axis=-1)


def _plane_wave_gap_task(task):
    """Worker for gap_parameter_sweep (module level so a spawned process can import it)."""
    plate_config, resonator_frequency_hz, mass_ratio, maximum_frequency_hz = task
    path = build_wavevector_path(plate_config.length, plate_config.width, points_per_segment=POINTS_PER_SEGMENT)
    return gap_edges_row(cell_band_gaps(plate_config, path, resonator_frequency_hz, mass_ratio, maximum_frequency_hz))


def gap_parameter_sweep(plate_config, frequency_ratios=FREQUENCY_RATIOS, mass_ratios=MASS_RATIOS,
                        number_of_workers=8):
    """Plane-wave gap table over (mass ratio, f_r / f_X), one diagram per pair (~7 s each), spread over worker
    processes (spawned, so no forked PETSc/MPI state)."""
    bragg_frequency_hz = bragg_frequency_x(plate_config)
    maximum_frequency_hz = LIMIT_OVER_BRAGG * bragg_frequency_hz
    tasks = [(plate_config, ratio * bragg_frequency_hz, mass_ratio, maximum_frequency_hz)
             for mass_ratio in mass_ratios for ratio in frequency_ratios]
    print(f"Gap study: {len(tasks)} plane-wave diagrams of the {plate_config.length} m cell "
          f"(f_X = {bragg_frequency_hz:.1f} Hz) on {number_of_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=number_of_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(_plane_wave_gap_task, tasks))
    return GapStudy(cell_length=plate_config.length, thickness=plate_config.thickness,
                    bragg_frequency_hz=bragg_frequency_hz, frequency_ratios=np.asarray(frequency_ratios, dtype=float),
                    mass_ratios=np.asarray(mass_ratios, dtype=float),
                    gap_edges_hz=np.array(rows).reshape(len(mass_ratios), len(frequency_ratios), MAXIMUM_GAP_COUNT, 2))


def get_or_compute_gap_study(path, compute, settings_blob=""):
    """Load the GapStudy from 'path' if it exists, otherwise call compute() and save it (same pattern as the
    sweep cache; the key should contain the cell config, ratios, limit and code version)."""
    path = Path(path)
    if path.exists():
        print(f"Loading cached gap study: {path.name}")
        with np.load(path) as data:
            return GapStudy(**{field.name: data[field.name] for field in dataclasses.fields(GapStudy)})
    print(f"No cache at {path.name}; computing gap study ...")
    study = compute()
    np.savez(path, settings_blob=settings_blob, **dataclasses.asdict(study))
    return study


# --- Finite plate: insertion loss in a frequency window around the gap ---

@dataclasses.dataclass
class FinitePlateSetup:
    """Everything the forced-response sweep of main.py needs besides the resonators. frequency_grid is the full
    grid of the cached bare sweep; windowed runs use a slice of it, so the bare radiated power can be sliced too."""
    domain: object
    function_space: object
    problem: object
    plate_config: object
    shaker_config: object
    frequency_grid: np.ndarray
    gamma_plate: float
    free_plate: bool
    excitation: str
    deg: int
    el_type: str
    shear_quadrature_degree: int | None = None   # None = reduced 2x2 (validated pipeline); 4 = full 3x3


def sweep_cache_settings(setup, frequency_grid):
    """Same settings dict as main.py's forced-response block, so the bare-plate cache keys match. The shear rule
    enters the key only when it differs from the default, so all existing cache files stay valid."""
    settings = dict(plate_config=setup.plate_config, shaker_config=setup.shaker_config,
                    grid=np.asarray(frequency_grid), gamma_plate=setup.gamma_plate,
                    free_plate=setup.free_plate, excitation=setup.excitation,
                    deg=setup.deg, el_type=setup.el_type, code_version="v2")
    if setup.shear_quadrature_degree is not None:
        settings["shear_quadrature_degree"] = setup.shear_quadrature_degree
    return settings


def radiation_cache_settings(setup):
    """Same radiation settings as main.py."""
    from radiation import rho_air, c_air
    return dict(rho_air=rho_air, c_air=c_air, force_amplitude=setup.shaker_config.force_amplitude,
                code_version="v1")


def cached_bare_radiated_power(setup):
    """Bare-plate radiated power on the full grid, from the cache main.py writes (never recomputed here: the
    full bare sweep takes ~30 min; run main.py with do_frequency_sweep = True once if this fails)."""
    from sweep_cache import CACHE_DIR, make_cache_key, get_or_compute_radiated_power
    key_bare, _ = make_cache_key(vamm_list=None, **sweep_cache_settings(setup, setup.frequency_grid))
    key_radiation, _ = make_cache_key(sweep_key=key_bare, **radiation_cache_settings(setup))

    def missing():
        raise FileNotFoundError("bare radiated power not cached; run main.py with do_frequency_sweep = True")
    return get_or_compute_radiated_power(CACHE_DIR / f"radiation_bare_{key_radiation}.npz",
                                         f_values=np.asarray(setup.frequency_grid), compute=missing)


def windowed_insertion_loss(setup, vamm_list, frequency_window_hz, bare_radiated_power, require_cached=False):
    """Equal-force radiated-power insertion loss 10 log10(P_bare / P_vamm) on the part of the full grid inside
    frequency_window_hz = (lower, upper). Only the resonator plate is swept (MUMPS, ~3 s + ~2.4 s radiation per
    frequency at 100 x 100); both the sweep and its radiated power are cached. Returns (f_values, IL dB).
    require_cached: raise instead of sweeping if the sweep is not cached (guards a case meant to reuse main.py's
    full sweep against a silent ~50 min recomputation when its key does not match).
    bare_radiated_power: on the full grid (sliced here to the window)."""
    full_grid = np.asarray(setup.frequency_grid)
    in_window = (full_grid >= frequency_window_hz[0]) & (full_grid <= frequency_window_hz[1])
    window_grid, vamm_power = windowed_radiated_power(setup, vamm_list, frequency_window_hz,
                                                      require_cached=require_cached)
    from insertion_loss import global_insertion_loss_radiated_db
    return window_grid, global_insertion_loss_radiated_db(bare_radiated_power[in_window], vamm_power)


def windowed_radiated_power(setup, vamm_list, frequency_window_hz, require_cached=False):
    """Radiated power of the plate (vamm_list None = bare) on the part of the full grid inside frequency_window_hz,
    sweep and power both cached. Returns (f_values, P)."""
    from point_coupling import compute_phi_and_dofs_for_vamm_list
    from frequency_sweep import frequency_sweep_plate
    from probe import mobility_at_probes
    from radiation import radiated_power, build_radiation_points
    from sweep_cache import CACHE_DIR, make_cache_key, get_or_compute_sweep, get_or_compute_radiated_power

    full_grid = np.asarray(setup.frequency_grid)
    in_window = (full_grid >= frequency_window_hz[0]) & (full_grid <= frequency_window_hz[1])
    window_grid = full_grid[in_window]
    phi_list, global_dofs_parent_list = (None, None) if vamm_list is None else \
        compute_phi_and_dofs_for_vamm_list(domain=setup.domain, function_space=setup.function_space,
                                           vamm_list=vamm_list)[:2]
    key_sweep, blob_sweep = make_cache_key(vamm_list=vamm_list, **sweep_cache_settings(setup, window_grid))
    kind = "bare" if vamm_list is None else "vamm"
    sweep_path = CACHE_DIR / f"sweep_{kind}_{key_sweep}.npz"
    if require_cached and not sweep_path.exists():
        raise FileNotFoundError(f"{sweep_path.name} not cached (require_cached=True)")
    sweep = get_or_compute_sweep(
        sweep_path,
        compute=lambda: frequency_sweep_plate(
            domain=setup.domain, function_space=setup.function_space, problem=setup.problem,
            plate_config=setup.plate_config, f_start=window_grid[0], f_end=window_grid[-1],
            vamm_list=vamm_list, phi_list=phi_list, global_dofs_parent_list=global_dofs_parent_list,
            frequency_grid=window_grid, gamma=setup.gamma_plate, free_plate=setup.free_plate,
            excitation=setup.excitation, shaker_config=setup.shaker_config, factorization_package="mumps",
            shear_quadrature_degree=setup.shear_quadrature_degree),
        settings_blob=blob_sweep)
    key_radiation, blob_radiation = make_cache_key(sweep_key=key_sweep, **radiation_cache_settings(setup))

    def compute_radiated_power():
        radiation_probes, cell_area, distances = build_radiation_points(
            domain=setup.domain, function_space=setup.function_space, plate_config=setup.plate_config)
        mobility = mobility_at_probes(sweep=sweep, probes=radiation_probes,
                                      force_amplitude=setup.shaker_config.force_amplitude)
        return radiated_power(Y_grid=mobility, r=distances, dS=cell_area, f_values=sweep.f_values,
                              force_amplitude=setup.shaker_config.force_amplitude)
    power = get_or_compute_radiated_power(CACHE_DIR / f"radiation_{kind}_{key_radiation}.npz",
                                          f_values=sweep.f_values, compute=compute_radiated_power,
                                          settings_blob=blob_radiation)
    return window_grid, power
