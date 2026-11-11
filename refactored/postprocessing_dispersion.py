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


def plate_share_colormap():
    """Diverging map for the plate share 0..1: red = resonator motion, grey = fully hybridized (50 %),
    blue = plate motion. The grey centre stays visible on a white background."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("plate_share", ["#b2182b", "#8c8c8c", "#2166ac"])


def plot_tracked_dispersion(result, colour_by="plate_share", frequency_limit=None, title="Dispersion diagram",
                            save_path=None, resonator_frequencies_hz=None, analytic_frequencies=None,
                            analytic_label="analytic (folded plate)"):
    """Band diagram with tracked branches (dispersion_sweep(..., track_bands=True)) drawn as continuous lines.

    The line itself carries the branch identity: at a true crossing two lines pass through each other, at an
    avoided crossing they bend apart. The colour carries
      colour_by="plate_share": plate part of the kinetic energy along the branch (needs resonators to be useful);
      colour_by="branch": one colour per branch (check plot for the tracker: a wrong swap shows as a colour jump).
    Steps where the tracker fell back to plain sorting or broke a branch are marked by short ticks at the top.
    analytic_frequencies (optional, sorted per k) are drawn as thin grey lines underneath."""
    from matplotlib.collections import LineCollection
    tracking = result.tracking
    if tracking is None:
        raise ValueError("result has no tracking; run dispersion_sweep(..., track_bands=True)")
    if colour_by not in ("plate_share", "branch"):
        raise ValueError(f"colour_by must be 'plate_share' or 'branch', got {colour_by!r}")
    path = result.path
    fig, ax = plt.subplots(figsize=(8.5, 6))

    if analytic_frequencies is not None:
        for band_index in range(analytic_frequencies.shape[1]):
            ax.plot(path.path_coordinate, analytic_frequencies[:, band_index], color="0.75", linewidth=0.7,
                    zorder=1, label=analytic_label if band_index == 0 else None)
    for resonator_index, resonator_frequency in enumerate(resonator_frequencies_hz or []):
        ax.axhline(resonator_frequency, color="0.3", linewidth=0.8, linestyle=":", zorder=0,
                   label="resonator frequency $f_r$" if resonator_index == 0 else None)

    colormap, branch_colours = plate_share_colormap(), plt.get_cmap("tab20")
    number_of_branches = tracking.frequencies.shape[1]
    for label in range(tracking.frequencies.shape[1]):
        frequencies = tracking.frequencies[:, label]
        both_present = ~np.isnan(frequencies[:-1]) & ~np.isnan(frequencies[1:])
        step_indices = np.flatnonzero(both_present)
        if len(step_indices) == 0:
            continue
        segments = np.stack([np.column_stack([path.path_coordinate[step_indices], frequencies[step_indices]]),
                             np.column_stack([path.path_coordinate[step_indices + 1],
                                              frequencies[step_indices + 1]])], axis=1)
        if colour_by == "plate_share":
            share = tracking.plate_share[:, label]
            segment_share = 0.5 * (share[step_indices] + share[step_indices + 1])
            lines = LineCollection(segments, cmap=colormap, linewidth=2.0, zorder=2)
            lines.set_array(segment_share)
            lines.set_clim(0.0, 1.0)
        else:
            # 20 colours; later branches thinner, so a degenerate partner drawn on top leaves the earlier one visible
            linewidth = 3.0 - 2.0 * label / max(number_of_branches - 1, 1)
            lines = LineCollection(segments, colors=[branch_colours(label % 20)], linewidth=linewidth, zorder=2)
        ax.add_collection(lines)

    flagged_steps = [step + 1 for step, status in enumerate(tracking.step_status) if status != "tracked"]
    if frequency_limit is None:
        frequency_limit = result.maximum_frequency_hz
    for step in flagged_steps:
        ax.plot(path.path_coordinate[step], 0.985 * frequency_limit, marker="v", color="k", markersize=5,
                zorder=3, label="fallback / broken step" if step == flagged_steps[0] else None)

    if colour_by == "plate_share":
        colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=colormap), ax=ax, pad=0.02)
        colourbar.set_label("plate share of kinetic energy")
        colourbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        colourbar.set_ticklabels(["0 (resonator)", "0.25", "0.5", "0.75", "1 (plate)"])

    for position in path.symmetry_point_positions:
        ax.axvline(position, color="0.3", linewidth=0.8)
    ax.set_xticks(path.symmetry_point_positions)
    ax.set_xticklabels(path.symmetry_point_labels)
    ax.set_xlim(path.path_coordinate[0], path.path_coordinate[-1])
    ax.set_ylim(0, frequency_limit)
    ax.set_xlabel("Wavevector k")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
