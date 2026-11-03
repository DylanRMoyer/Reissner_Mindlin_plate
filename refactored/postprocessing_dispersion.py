import numpy as np
import matplotlib.pyplot as plt

# --- Plotting dispersion diagrams (unit cell, Bloch-Floquet); finite-plate plots live in postprocessing.py ---


def plot_dispersion(result, analytic_frequencies=None, frequency_limit=None,
                    title="Dispersion diagram", save_path=None, analytic_label="analytic (folded plate)",
                    bare_frequencies=None, resonator_frequencies_hz=None):
    """Band diagram along the wavevector path (Gamma-X-M-Gamma or Gamma-X-M-Y-Gamma).

    result: DispersionResult from dispersion.dispersion_sweep.
    analytic_frequencies: optional, shape (n_k, n_bands) on the same path (dispersion.analytic_dispersion),
        drawn as thin grey lines underneath as a check for the bare cell. Compute it slightly beyond the
        y-limit so the lines reach the top edge.
    analytic_label: legend label of the analytic overlay (e.g. the plane-wave expansion for a resonator cell).
    bare_frequencies: optional bare-plate bands on the same path, drawn as light dashed lines, to show what
        the resonators change.
    resonator_frequencies_hz: optional list of resonator frequencies sqrt(k/m)/2pi, drawn as horizontal lines.
    frequency_limit: optional upper y-limit in Hz; defaults to the sweep's maximum_frequency_hz, if set.
    save_path: if given, saves the figure instead of showing it.

    The bands are sorted by frequency at each k (no branch tracking yet), so they are drawn as markers:
    lines would show artificial kinks wherever two branches cross.
    """
    path = result.path
    fig, ax = plt.subplots(figsize=(8, 6))

    if analytic_frequencies is not None:
        for band_index in range(analytic_frequencies.shape[1]):
            ax.plot(path.path_coordinate, analytic_frequencies[:, band_index], color="0.6", linewidth=0.8,
                    zorder=1, label=analytic_label if band_index == 0 else None)

    if bare_frequencies is not None:
        for band_index in range(bare_frequencies.shape[1]):
            ax.plot(path.path_coordinate, bare_frequencies[:, band_index], color="0.75", linewidth=0.8,
                    linestyle="--", zorder=0, label="bare plate" if band_index == 0 else None)

    for resonator_index, resonator_frequency in enumerate(resonator_frequencies_hz or []):
        ax.axhline(resonator_frequency, color="C3", linewidth=0.8, linestyle=":", zorder=0,
                   label="resonator frequency $f_r$" if resonator_index == 0 else None)

    for band_index in range(result.frequencies.shape[1]):
        ax.plot(path.path_coordinate, result.frequencies[:, band_index].real, linestyle="none", marker="o",
                markersize=3, color="C0", zorder=2, label="FE (Bloch)" if band_index == 0 else None)

    for position in path.symmetry_point_positions:
        ax.axvline(position, color="0.3", linewidth=0.8)
    ax.set_xticks(path.symmetry_point_positions)
    ax.set_xticklabels(path.symmetry_point_labels)
    ax.set_xlim(path.path_coordinate[0], path.path_coordinate[-1])
    if frequency_limit is None:
        frequency_limit = result.maximum_frequency_hz
    ax.set_ylim(0, frequency_limit)

    ax.set_xlabel("Wavevector k")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
