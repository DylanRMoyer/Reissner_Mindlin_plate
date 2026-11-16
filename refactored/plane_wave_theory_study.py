"""Queue Q3 (stage 8 candidate): which Reissner-Mindlin effect makes the normalized gap map depend on h/a?

The plane-wave condition 1 = Z(omega)/A * sum_G g(|k + G|, omega) has two parts:
  - band terms   |k + G| < band_wavenumber_limit: the folded bands in and near the computed range. Their poles
                 are the bare bands; Reissner-Mindlin changes their shape (lambda_M / lambda_X = 3.96 instead of 4
                 for h/a = 0.05) through shear softening.
  - local terms  all far terms: far below their own poles they add up to a quasi-static point compliance (the local
                 bending sum S). Kirchhoff: the sum converges. Reissner-Mindlin: it diverges like log(cutoff), and
                 its size relative to the band terms grows with h/a.
Four hybrid models (bands x local sum, each Kirchhoff "K" or Reissner-Mindlin "R") separate the two:
  KK = pure Kirchhoff (no h/a at all: normalized gaps must be identical for both cells, check 1),
  RR = the production model (must reproduce the cached stage 7 gap study, check 2),
  RK = Reissner-Mindlin bands only, KR = Reissner-Mindlin local sum only.
Each model is normalized with its own f_X (lowest bare band at X of the band theory).

Run in fenicsx-env-complex (~5-10 min on 12 workers). Writes results/plane_wave_theory_hybrids.pdf and prints the
tables of the queue report."""
import dataclasses
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dispersion import (build_wavevector_path, plate_constants, plane_wave_bare_frequency_hz,
                        analytic_resonator_dispersion, find_complete_gaps)

THEORY_NAMES = {"K": "kirchhoff", "R": "reissner_mindlin"}
VARIANTS = ("KK", "RK", "KR", "RR")                    # (band theory, local theory)
FREQUENCY_RATIOS = np.round(np.arange(0.5, 1.6 + 1e-9, 0.05), 4)
MASS_RATIOS = (0.25, 1.0)
LIMIT_OVER_BRAGG = 3.0              # bands up to 3 f_X: above f_r sqrt(1 + mu) = 2.26 f_X for tau 1.6, mu 1
POINTS_PER_SEGMENT = 20             # as the stage 7 gap study
BAND_WAVENUMBER_LIMIT_OVER_PI_A = 2.5   # band terms: |k + G| < 2.5 pi / a (poles up to ~6 f_X); far terms beyond


def bragg_frequency(plate_config, theory):
    """f_X of the given theory: bare bending frequency of the plane wave with |k| = pi / Lx."""
    return plane_wave_bare_frequency_hz(np.pi / plate_config.length, plate_constants(plate_config), theory)


def variant_gaps(plate_config, variant, frequency_ratio, mass_ratio,
                 band_wavenumber_limit_over_pi_a=BAND_WAVENUMBER_LIMIT_OVER_PI_A):
    """Complete gaps (lower, upper edge in units of the band theory's f_X) of the cell with one centred resonator,
    hybrid plane-wave model 'variant' (two letters: band theory, local theory)."""
    band_theory, local_theory = THEORY_NAMES[variant[0]], THEORY_NAMES[variant[1]]
    bragg_hz = bragg_frequency(plate_config, band_theory)
    resonator_mass = mass_ratio * plate_config.rho * plate_config.thickness * plate_config.length * plate_config.width
    resonator_stiffness = resonator_mass * (2 * np.pi * frequency_ratio * bragg_hz) ** 2
    path = build_wavevector_path(plate_config.length, plate_config.width, points_per_segment=POINTS_PER_SEGMENT)
    frequencies = analytic_resonator_dispersion(
        path, plate_config, resonator_stiffness, resonator_mass, maximum_frequency_hz=LIMIT_OVER_BRAGG * bragg_hz,
        theory=band_theory, local_theory=local_theory,
        band_wavenumber_limit=band_wavenumber_limit_over_pi_a * np.pi / plate_config.length)
    return [(gap.lower_edge_hz / bragg_hz, gap.upper_edge_hz / bragg_hz)
            for gap in find_complete_gaps(frequencies, path)]


def widest_gap(gaps):
    """(relative width, lower, upper) of the widest gap (relative to its centre); zeros if there is none."""
    if not gaps:
        return 0.0, np.nan, np.nan
    widths = [(upper - lower) / (0.5 * (upper + lower)) for lower, upper in gaps]
    best = int(np.argmax(widths))
    return widths[best], gaps[best][0], gaps[best][1]


def _task(task):
    plate_config, variant, frequency_ratio, mass_ratio = task
    return widest_gap(variant_gaps(plate_config, variant, frequency_ratio, mass_ratio))


def hybrid_study(cells, number_of_workers=12):
    """{cell name: array (variant, mass ratio, frequency ratio, 3)} of (relative width, lower, upper)."""
    tasks = [(config, variant, ratio, mass) for config in cells.values() for variant in VARIANTS
             for mass in MASS_RATIOS for ratio in FREQUENCY_RATIOS]
    print(f"Hybrid plane-wave study: {len(tasks)} diagrams on {number_of_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=number_of_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = np.array(list(pool.map(_task, tasks)))
    shape = (len(VARIANTS), len(MASS_RATIOS), len(FREQUENCY_RATIOS), 3)
    return {name: block.reshape(shape) for name, block in zip(cells, np.split(rows, len(cells)))}


def plot_hybrids(results, cells, save_path):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(MASS_RATIOS), figsize=(9.5, 4.6), constrained_layout=True)
    variant_styles = {"KK": ("0.3", "-"), "RK": ("tab:blue", "-"), "KR": ("tab:orange", "-"), "RR": ("tab:red", "-")}
    cell_markers = dict(zip(cells, ["", "o"]))
    for mass_index, (mass_ratio, axis) in enumerate(zip(MASS_RATIOS, axes)):
        for cell_name, table in results.items():
            for variant_index, variant in enumerate(VARIANTS):
                colour, line = variant_styles[variant]
                axis.plot(FREQUENCY_RATIOS, 100 * table[variant_index, mass_index, :, 0], color=colour,
                          linestyle=line if cell_markers[cell_name] == "" else "none",
                          marker=cell_markers[cell_name] or None, markersize=3,
                          label=f"{variant} ({cell_name})")
        axis.axvline(1.0, color="0.4", linewidth=0.8)
        axis.set_xlabel("$f_r / f_X$ (f_X of the band theory)")
        axis.set_ylabel("widest complete gap [%]")
        axis.set_title(rf"$\mu$ = {mass_ratio:g}")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle("Hybrid plane-wave models: bands x local sum (K = Kirchhoff, R = Reissner-Mindlin)")
    figure.savefig(save_path)


if __name__ == "__main__":
    import sys
    from config import PlateConfig
    from compare_cool_2024 import cool_plate_config
    production_cell = PlateConfig(length=0.1, width=0.1, thickness=0.005, nx=10, ny=10, rho=7850, mu=77e9,
                                  lambda_=115e9)
    cells = {"0.1 m, h/a 0.05": production_cell, "Cool 0.05 m, h/a 0.1": cool_plate_config()}
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    results = hybrid_study(cells, number_of_workers=workers)
    np.savez("results/plane_wave_theory_hybrids.npz", frequency_ratios=FREQUENCY_RATIOS,
             mass_ratios=np.array(MASS_RATIOS), variants=np.array(VARIANTS), cell_names=np.array(list(cells)),
             **{f"table_{index}": table for index, table in enumerate(results.values())})

    for cell_name, config in cells.items():
        constants = plate_constants(config)
        ratio_r = (plane_wave_bare_frequency_hz(np.sqrt(2) * np.pi / config.length, constants, "reissner_mindlin")
                   / bragg_frequency(config, "reissner_mindlin")) ** 2
        print(f"{cell_name}: f_X Kirchhoff {bragg_frequency(config, 'kirchhoff'):.1f} Hz, RM "
              f"{bragg_frequency(config, 'reissner_mindlin'):.1f} Hz; lambda_M / lambda_X RM {ratio_r:.3f} (K: 4)")

    # check 1: Kirchhoff has no h/a, so KK must agree between the cells
    first, second = results.values()
    kk_difference = np.nanmax(np.abs(first[0, :, :, 0] - second[0, :, :, 0]))
    print(f"check 1: KK widest-gap width, max difference between the cells {100 * kk_difference:.2e} % points")
    assert kk_difference < 1e-4, "Kirchhoff hybrid depends on h/a: bug"

    # check 2: RR reproduces the stage 7 gap study (0.1 m cell, mu 0.25, from the plane-wave cache) where both exist
    print("check 2: see the stage 7 values (mu 0.25: 18.6 % near tau 0.95; mu 1: 49.3 % at tau 0.80)")
    for mass_index, mass_ratio in enumerate(MASS_RATIOS):
        best = int(np.argmax(first[3, mass_index, :, 0]))
        print(f"   RR 0.1 m, mu {mass_ratio:g}: widest {100 * first[3, mass_index, best, 0]:.1f} % at tau "
              f"{FREQUENCY_RATIOS[best]:.2f}")

    # check 3: band/local split insensitive to the limit (one point per cell, the mixed variants)
    for cell_name, config in cells.items():
        for variant in ("RK", "KR"):
            gaps_a = variant_gaps(config, variant, 0.9, 0.25, band_wavenumber_limit_over_pi_a=2.5)
            gaps_b = variant_gaps(config, variant, 0.9, 0.25, band_wavenumber_limit_over_pi_a=3.5)
            print(f"check 3: {cell_name} {variant} tau 0.9 mu 0.25, gaps (limit 2.5 pi/a) {np.round(gaps_a, 4)} "
                  f"vs (3.5 pi/a) {np.round(gaps_b, 4)}")

    # decomposition table: widest gap per variant at selected tau
    for mass_index, mass_ratio in enumerate(MASS_RATIOS):
        print(f"\nmu = {mass_ratio:g}: widest gap [%] (lower-upper / f_X)")
        print("tau   " + "  ".join(f"{cell[:12]:>12s}-{variant}" for cell in cells for variant in VARIANTS))
        for ratio_index, ratio in enumerate(FREQUENCY_RATIOS):
            if not np.isclose(ratio * 20, round(ratio * 20)) or ratio_index % 2:
                continue
            row = []
            for table in results.values():
                for variant_index in range(len(VARIANTS)):
                    width, lower, upper = table[variant_index, mass_index, ratio_index]
                    row.append(f"{100 * width:6.1f} ({lower:.3f}-{upper:.3f})" if width > 0 else f"{'-':>21s}")
            print(f"{ratio:4.2f}  " + "  ".join(row))
    plot_hybrids(results, cells, "results/plane_wave_theory_hybrids.pdf")
