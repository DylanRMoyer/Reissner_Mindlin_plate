"""Queue Q4 (stage 8 candidate): triangular vs square lattice, plane-wave model (lattice_plane_wave.py).

1. Rigid-mass threshold: the smallest mass ratio mu = m / (rho h A) for which a rigid point mass per cell opens a
   complete gap between bands 1 and 2. Few-wave estimates (theory notes, Chapter 3):
     square:     band 1 at M (4 waves) below band 2 at X:  mu > (lambda_M / lambda_X - 1) / 4  (Kirchhoff 0.75)
     triangular: band 1 at K (3 waves) below band 2 at M:  mu > (lambda_K / lambda_M - 1) / 3  (Kirchhoff 0.26)
   These are truncations, so the exact (all-wave) threshold should lie at or below them (Section 3.4).
   Found by bisection on the path, then checked on a zone-interior grid.
2. Resonator gaps: widest complete gap vs tau = f_r / f_fold (f_fold = first Bragg fold: X for square, M for
   triangular) for several mu. Test of the closing rule tau_close = 1 / sqrt(1 - mu / mu_threshold).
Both cells have the same area A = 0.01 m^2 (same resonator count per plate area), 5 mm steel, Reissner-Mindlin.

Run in fenicsx-env-complex (~10 min on 12 workers)."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from config import PlateConfig
from dispersion import find_complete_gaps, plate_constants, plane_wave_bare_frequency_hz
from lattice_plane_wave import (square_lattice, triangular_lattice, lattice_path, lattice_attachment_dispersion,
                                irreducible_wedge_grid)

CELL_AREA = 0.01
SQUARE_SPACING = np.sqrt(CELL_AREA)
TRIANGULAR_SPACING = np.sqrt(2 * CELL_AREA / np.sqrt(3))       # A = sqrt(3) / 2 a^2
PLATE = PlateConfig(length=SQUARE_SPACING, width=SQUARE_SPACING, thickness=0.005, nx=10, ny=10, rho=7850,
                    mu=77e9, lambda_=115e9)     # length/width unused by the lattice model except via the lattice
POINTS_PER_SEGMENT = 20
FREQUENCY_RATIOS = np.round(np.arange(0.4, 2.0 + 1e-9, 0.05), 4)
MASS_RATIOS = (0.1, 0.25, 0.5, 1.0)
LIMIT_OVER_FOLD = 4.2


def lattices():
    return {"square": square_lattice(SQUARE_SPACING), "triangular": triangular_lattice(TRIANGULAR_SPACING)}


def fold_wavenumbers(lattice):
    """(|first fold point|, |band-1 maximum point|): (X, M) for square, (M, K) for triangular."""
    return np.linalg.norm(lattice.symmetry_points[1]), np.linalg.norm(lattice.symmetry_points[2])


def fold_frequency(lattice, theory="reissner_mindlin"):
    return plane_wave_bare_frequency_hz(fold_wavenumbers(lattice)[0], plate_constants(PLATE), theory)


def few_wave_threshold(lattice, theory="reissner_mindlin"):
    """(lambda_max / lambda_fold - 1) / d, d = number of degenerate waves at the band-1 maximum point
    (4 at M of the square, 3 at K of the triangle)."""
    first, second = fold_wavenumbers(lattice)
    constants = plate_constants(PLATE)
    ratio = (plane_wave_bare_frequency_hz(second, constants, theory)
             / plane_wave_bare_frequency_hz(first, constants, theory)) ** 2
    degenerate_waves = 4 if lattice.name == "square" else 3
    return (ratio - 1) / degenerate_waves


def attachment_mass(mass_ratio, lattice):
    return mass_ratio * PLATE.rho * PLATE.thickness * lattice.cell_area


def rigid_mass_bands(lattice, mass_ratio, wavevector_path, theory):
    limit = 2.3 * fold_frequency(lattice, theory)
    return lattice_attachment_dispersion(wavevector_path.wavevectors, lattice, PLATE,
                                         attachment_mass(mass_ratio, lattice), limit, theory=theory)


def rigid_mass_gap(lattice, mass_ratio, wavevector_path, theory):
    """Gap between the first and second band (sorted indices 0 and 1) of the rigid-mass lattice, or None."""
    frequencies = rigid_mass_bands(lattice, mass_ratio, wavevector_path, theory)
    for gap in find_complete_gaps(frequencies, wavevector_path, minimum_relative_width=0.0):
        if gap.lower_band_index == 0:
            return gap
    return None


def rigid_mass_threshold(lattice, theory, tolerance=1e-3):
    """Bisection for the smallest mu with a band 1-2 gap on the path."""
    path = lattice_path(lattice, POINTS_PER_SEGMENT)
    low, high = 0.01, 1.5
    assert rigid_mass_gap(lattice, high, path, theory) is not None and rigid_mass_gap(lattice, low, path, theory) is None
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if rigid_mass_gap(lattice, middle, path, theory) is None:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _threshold_task(task):
    lattice_name, theory = task
    return rigid_mass_threshold(lattices()[lattice_name], theory)


def _resonator_task(task):
    lattice_name, frequency_ratio, mass_ratio = task
    lattice = lattices()[lattice_name]
    fold_hz = fold_frequency(lattice)
    mass = attachment_mass(mass_ratio, lattice)
    stiffness = mass * (2 * np.pi * frequency_ratio * fold_hz) ** 2
    path = lattice_path(lattice, POINTS_PER_SEGMENT)
    frequencies = lattice_attachment_dispersion(path.wavevectors, lattice, PLATE, mass, LIMIT_OVER_FOLD * fold_hz,
                                                attachment_stiffness=stiffness)
    gaps = find_complete_gaps(frequencies, path)
    if not gaps:
        return 0.0, np.nan, np.nan
    widths = [(gap.upper_edge_hz - gap.lower_edge_hz) / (0.5 * (gap.upper_edge_hz + gap.lower_edge_hz)) for gap in gaps]
    best = gaps[int(np.argmax(widths))]
    return max(widths), best.lower_edge_hz / fold_hz, best.upper_edge_hz / fold_hz


if __name__ == "__main__":
    spawn = multiprocessing.get_context("spawn")
    lattice_set = lattices()
    for name, lattice in lattice_set.items():
        print(f"{name}: spacing {np.linalg.norm(lattice.lattice_vectors[0]):.4f} m, area {lattice.cell_area:.5f} m^2, "
              f"f_fold RM {fold_frequency(lattice):.1f} Hz (Kirchhoff {fold_frequency(lattice, 'kirchhoff'):.1f}), "
              f"few-wave threshold RM {few_wave_threshold(lattice):.3f}, Kirchhoff "
              f"{few_wave_threshold(lattice, 'kirchhoff'):.3f}", flush=True)

    # 1. rigid-mass thresholds
    threshold_tasks = [(name, theory) for name in lattice_set for theory in ("kirchhoff", "reissner_mindlin")]
    with ProcessPoolExecutor(max_workers=4, mp_context=spawn) as pool:
        thresholds = dict(zip(threshold_tasks, pool.map(_threshold_task, threshold_tasks)))
    for (name, theory), value in thresholds.items():
        print(f"rigid-mass threshold {name:10s} {theory:16s}: all-wave {value:.3f}  vs few-wave "
              f"{few_wave_threshold(lattice_set[name], theory):.3f}", flush=True)

    # zone-interior check just above the RM threshold: band 1 max / band 2 min must not move off the path values
    for name, lattice in lattice_set.items():
        mass_ratio = thresholds[(name, "reissner_mindlin")] + 0.02
        path = lattice_path(lattice, POINTS_PER_SEGMENT)
        wedge = irreducible_wedge_grid(lattice, points_per_side=12)
        on_path = rigid_mass_bands(lattice, mass_ratio, path, "reissner_mindlin")
        inside = rigid_mass_bands(lattice, mass_ratio, wedge, "reissner_mindlin")
        print(f"zone interior, {name}, mu {mass_ratio:.3f}: band 1 max path {np.nanmax(on_path[:, 0]):.2f} / wedge "
              f"{np.nanmax(inside[:, 0]):.2f} Hz; band 2 min path {np.nanmin(on_path[:, 1]):.2f} / wedge "
              f"{np.nanmin(inside[:, 1]):.2f} Hz", flush=True)

    # 2. resonator gap maps
    resonator_tasks = [(name, ratio, mass) for name in lattice_set for mass in MASS_RATIOS for ratio in FREQUENCY_RATIOS]
    print(f"resonator gap maps: {len(resonator_tasks)} diagrams", flush=True)
    import sys
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 12
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
        rows = np.array(list(pool.map(_resonator_task, resonator_tasks)))
    table = rows.reshape(len(lattice_set), len(MASS_RATIOS), len(FREQUENCY_RATIOS), 3)
    np.savez("results/triangular_vs_square_gap_maps.npz", table=table, frequency_ratios=FREQUENCY_RATIOS,
             mass_ratios=np.array(MASS_RATIOS), lattices=np.array(list(lattice_set)),
             thresholds=np.array([[thresholds[(name, theory)] for theory in ("kirchhoff", "reissner_mindlin")]
                                  for name in lattice_set]))
    for lattice_index, name in enumerate(lattice_set):
        threshold = thresholds[(name, "reissner_mindlin")]
        for mass_index, mass_ratio in enumerate(MASS_RATIOS):
            widths = table[lattice_index, mass_index, :, 0]
            best = int(np.argmax(widths))
            open_above = FREQUENCY_RATIOS[(FREQUENCY_RATIOS > 1.0) & (widths > 0)]
            last_open = open_above.max() if open_above.size else np.nan
            predicted_close = 1 / np.sqrt(1 - mass_ratio / threshold) if mass_ratio < threshold else np.inf
            print(f"{name:10s} mu {mass_ratio:4.2f}: widest {100 * widths[best]:5.1f} % at tau "
                  f"{FREQUENCY_RATIOS[best]:.2f} (edges {table[lattice_index, mass_index, best, 1]:.3f}-"
                  f"{table[lattice_index, mass_index, best, 2]:.3f} f_fold); last open tau above 1: {last_open}; "
                  f"predicted tau_close {predicted_close:.2f}", flush=True)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(MASS_RATIOS)))
    for mass_index, (mass_ratio, colour) in enumerate(zip(MASS_RATIOS, colours)):
        axis.plot(FREQUENCY_RATIOS, 100 * table[0, mass_index, :, 0], color=colour, label=rf"square, $\mu$ = {mass_ratio:g}")
        axis.plot(FREQUENCY_RATIOS, 100 * table[1, mass_index, :, 0], "o--", color=colour, markersize=3,
                  label=rf"triangular, $\mu$ = {mass_ratio:g}")
    axis.axvline(1.0, color="0.4", linewidth=0.8)
    axis.set_xlabel(r"$f_r / f_\mathrm{fold}$ (fold: X for square, M for triangular)")
    axis.set_ylabel("widest complete gap [%]")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    axis.set_title("Square vs triangular lattice, one resonator per cell (plane-wave model, equal cell area)")
    figure.savefig("results/triangular_vs_square_gap_maps.pdf")
