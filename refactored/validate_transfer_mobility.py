"""
Validation of the transfer-mobility pipeline.

Run in fenicsx-env-complex. The sweep cache now stores the physical result (w_field) only,
so it depends on shaker position, VAMMs, grid and plate settings, but NOT on the probes.
Per label (bare / vamm) three sweeps are needed: shaker at s, and shaker at each of the
two chosen probes r (for reciprocity). The first run needs 6 sweeps on a coarse 25-point
grid (a few minutes); afterwards everything comes from the cache, and changing probe
positions costs nothing.
Reciprocity is an algebraic identity of the discrete system, so a coarse grid is
sufficient: it holds at every frequency, resonant or not.
"""
import numpy as np
from dataclasses import dataclass

from config import PlateConfig
from problem_setup import build_plate_problem
from point_coupling import create_vamm_list_and_assign_indices, compute_phi_and_dofs_for_vamm_list
from shaker_force import ShakerParameters
from probe import make_probe, build_probe_coordinates, mobility_at_probes
from frequency_sweep import frequency_sweep_plate
from sweep_cache import CACHE_DIR, make_cache_key, get_or_compute_sweep

GAMMA_PLATE = 0.04
FREE_PLATE = True
FORCE_AMPLITUDE = 1.0
DEG, EL_TYPE = 2, "S"
F_START, F_END, N_FREQ = 100.0, 300.0, 25
RECIPROCITY_TOL = 1e-6      # expected actual values are far smaller (round-off level)
CONTROL_MIN_DIFF = 1e-2     # the negative control must differ by at least this much
CONTROL_OFFSET_ELEMENTS = 5


@dataclass
class SweepContext:
    domain: object
    function_space: object
    problem: object
    plate_config: object
    phi_list: list
    global_dofs_parent_list: list
    grid: np.ndarray


def sweep_with_shaker_at(ctx, shaker_xy, vamm_list):
    shaker_config = ShakerParameters(
        x=shaker_xy[0], y=shaker_xy[1], force_amplitude=FORCE_AMPLITUDE, phase=0.0)
    return frequency_sweep_plate(
        domain=ctx.domain, function_space=ctx.function_space, problem=ctx.problem,
        plate_config=ctx.plate_config, f_start=float(ctx.grid[0]), f_end=float(ctx.grid[-1]),
        vamm_list=vamm_list, phi_list=ctx.phi_list,
        global_dofs_parent_list=ctx.global_dofs_parent_list,
        frequency_grid=ctx.grid, gamma=GAMMA_PLATE,
        free_plate=FREE_PLATE, excitation="force", shaker_config=shaker_config)


def cached_sweep(ctx, name, shaker_xy, vamm_list):
    # No probe information in the key: the cached w_field is probe-independent.
    key, blob = make_cache_key(
        plate_config=ctx.plate_config, shaker_xy=shaker_xy, force_amplitude=FORCE_AMPLITUDE,
        vamm_list=vamm_list, grid=ctx.grid, gamma_plate=GAMMA_PLATE,
        free_plate=FREE_PLATE, excitation="force", deg=DEG, el_type=EL_TYPE,
        code_version="v2")
    return get_or_compute_sweep(
        CACHE_DIR / f"val_{name}_{key}.npz",
        compute=lambda: sweep_with_shaker_at(ctx, shaker_xy, vamm_list),
        settings_blob=blob)


def relative_difference(Y_a, Y_b):
    """Pointwise |Y_a - Y_b| / max(|Y_a|, |Y_b|). No conjugation: the system is
    complex-symmetric (not Hermitian), so reciprocity compares the values directly."""
    return np.abs(Y_a - Y_b) / np.maximum(np.abs(Y_a), np.abs(Y_b))


def report(name, err, f_values, Y_ref):
    k = int(np.argmax(err))
    print(f"  {name}: max rel. diff {err[k]:.2e} at {f_values[k]:.1f} Hz (|Y| there: {abs(Y_ref[k]):.2e})")


def check_driving_point_passivity(name, Y_dp):
    ok = np.all(Y_dp.real >= -1e-9 * np.abs(Y_dp))
    max_phase = np.degrees(np.max(np.abs(np.angle(Y_dp))))
    print(f"  {name}: driving-point passivity {'OK' if ok else 'VIOLATED'}, max |phase| = {max_phase:.1f} deg")
    assert ok, f"{name}: Re(Y_dp) < 0 somewhere, max |phase| = {max_phase:.1f} deg"


def run_checks(ctx, label, vamm_list, all_probes, pair_indices, control_probe, check_order=False):
    """all_probes[0] is at the shaker location (driving point), all_probes[1:] the transfer probes."""
    print(f"\n=== {label} ===")
    s_probe = all_probes[0]
    s_xy = (s_probe.x, s_probe.y)
    base = cached_sweep(ctx, f"{label}_base", s_xy, vamm_list)
    f = base.f_values
    Y_base = mobility_at_probes(base, all_probes, FORCE_AMPLITUDE)

    # B2: driving point of the base sweep
    check_driving_point_passivity("base (shaker at s)", Y_base[0])

    # B1: row i of the mobility must belong to probe i, whatever the list order.
    # Pure postprocessing now: same sweep, probes in reversed order, no extra solve.
    if check_order:
        Y_rev = mobility_at_probes(base, all_probes[::-1], FORCE_AMPLITUDE)
        err = relative_difference(Y_rev[::-1], Y_base)
        print(f"  B1 order invariance: max rel. diff {err.max():.2e}")
        assert err.max() < 1e-9, "probe rows are not independent of probe-list order"

    # C: reciprocity for the chosen pairs, plus negative control
    for i in pair_indices:
        r_probe = all_probes[i]
        r_xy = (r_probe.x, r_probe.y)
        swapped = cached_sweep(ctx, f"{label}_swap{i}", r_xy, vamm_list)
        Y_swapped = mobility_at_probes(swapped, [r_probe, s_probe, control_probe], FORCE_AMPLITUDE)
        check_driving_point_passivity(f"swapped (shaker at probe {i})", Y_swapped[0])

        Y_rs = Y_base[i]        # force at s, response at r
        Y_sr = Y_swapped[1]     # force at r, response at s
        Y_ctrl = Y_swapped[2]   # force at r, response displaced from s

        err = relative_difference(Y_rs, Y_sr)
        err_ctrl = relative_difference(Y_rs, Y_ctrl)
        print(f"  C pair (s, probe {i}):")
        report("reciprocity       ", err, f, Y_rs)
        report("negative control  ", err_ctrl, f, Y_rs)
        assert err.max() < RECIPROCITY_TOL, (
            f"reciprocity violated for probe {i}: {err.max():.2e} at {f[np.argmax(err)]:.1f} Hz")
        assert err_ctrl.max() > CONTROL_MIN_DIFF, (
            f"negative control did not differ (max {err_ctrl.max():.2e}); the test has no teeth")


def main():
    plate_config = PlateConfig(length=1.1, width=1, thickness=0.02, nx=60, ny=50,
                               rho=7850, mu=77e9, lambda_=115e9)
    domain, function_space, _, problem = build_plate_problem(
        plate_config, deg=DEG, el_type=EL_TYPE)

    vamm_list = create_vamm_list_and_assign_indices([
        (1.1, 0.0, 85878.35366, 1.0, 0.02),
        (0.55, 0.5, 10042816.24, 10.0, 0.05),
        (0.1, 0.1, 500, 5, 0.01)])
    phi_list, global_dofs_parent_list, _ = compute_phi_and_dofs_for_vamm_list(
        domain=domain, function_space=function_space, vamm_list=vamm_list)

    grid = np.geomspace(F_START, F_END, N_FREQ)
    ctx = SweepContext(domain=domain, function_space=function_space, problem=problem,
                       plate_config=plate_config, phi_list=phi_list,
                       global_dofs_parent_list=global_dofs_parent_list, grid=grid)

    shaker_xy = (plate_config.length / 3, plate_config.width / 3)
    shaker_config = ShakerParameters(x=shaker_xy[0], y=shaker_xy[1],
                                     force_amplitude=FORCE_AMPLITUDE, phase=0.0)
    probe_xy = build_probe_coordinates(plate_config=plate_config, shaker_config=shaker_config,
                                       amount=5, vamm_list=vamm_list)
    all_xy = [shaker_xy] + probe_xy                      # index 0 = driving point, as in main.py
    all_probes = [make_probe(domain, function_space, x, y) for x, y in all_xy]
    pair_indices = [1, len(all_probes) - 1]              # nearest and farthest transfer probe

    h_x = plate_config.length / plate_config.nx
    offset_xy = (min(shaker_xy[0] + CONTROL_OFFSET_ELEMENTS * h_x, plate_config.length), shaker_xy[1])
    control_probe = make_probe(domain, function_space, *offset_xy)

    run_checks(ctx, "bare", None, all_probes, pair_indices, control_probe, check_order=True)
    run_checks(ctx, "vamm", vamm_list, all_probes, pair_indices, control_probe)
    print("\nAll transfer-mobility validations passed.")


if __name__ == "__main__":
    main()