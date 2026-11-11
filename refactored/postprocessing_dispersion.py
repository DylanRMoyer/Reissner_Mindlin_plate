import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# --- Plotting dispersion diagrams (unit cell, Bloch-Floquet); finite-plate plots live in postprocessing.py ---
# Style follows postprocessing.plot_radiation_bode: 9.5 in wide, constrained layout, full grid (alpha 0.3),
# legend font size 8, figure saved as given by save_path (PDF for reports).

LEGEND_FONTSIZE = 8
GAP_COLOUR = "#f0e442"          # light yellow: stays visible under the grey/red/blue band colours


def dispersion_figure(insertion_loss_curves=None):
    """Figure and axes for a band diagram. With insertion_loss_curves, a narrow panel on the right shares the
    frequency axis, so band gaps and the finite-plate insertion loss can be compared at the same height."""
    if not insertion_loss_curves:
        fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
        return fig, ax, None
    fig, (ax, ax_il) = plt.subplots(1, 2, figsize=(12.5, 6.5), sharey=True, constrained_layout=True,
                                    gridspec_kw={"width_ratios": [3, 1]})
    return fig, ax, ax_il


def draw_band_gaps(ax, band_gaps, frequency_limit, ax_il=None):
    """Shade each complete gap (dispersion.find_complete_gaps) across the band diagram, and across the
    insertion-loss panel if given, with the edge frequencies and relative width written inside the gap."""
    text_position = blended_transform_factory(ax.transAxes, ax.transData)
    visible_gaps = [gap for gap in band_gaps or [] if gap.lower_edge_hz < frequency_limit]
    for gap_index, gap in enumerate(visible_gaps):
        upper_edge_hz = min(gap.upper_edge_hz, frequency_limit)
        for axis in (ax, ax_il):
            if axis is not None:
                axis.axhspan(gap.lower_edge_hz, upper_edge_hz, color=GAP_COLOUR, alpha=0.45, linewidth=0, zorder=0,
                             label="complete band gap" if gap_index == 0 and axis is ax else None)
        ax.text(0.015, 0.5 * (gap.lower_edge_hz + upper_edge_hz),
                f"{gap.lower_edge_hz:.0f}–{gap.upper_edge_hz:.0f} Hz ({gap.relative_width:.1%})",
                transform=text_position, va="center", ha="left", fontsize=LEGEND_FONTSIZE, zorder=4)


def draw_insertion_loss_panel(ax_il, insertion_loss_curves):
    """Insertion loss of the finite plate with frequency on the vertical axis (shared with the band diagram).
    insertion_loss_curves: list of (label, frequencies_hz, insertion_loss_db); line styles as in
    plot_radiation_bode (first solid black, second dashed black, third thin grey)."""
    line_styles = [dict(color="k", linewidth=1.4), dict(color="k", linewidth=1.2, linestyle="--"),
                   dict(color="0.5", linewidth=0.9)]
    for curve_index, (label, frequencies_hz, insertion_loss_db) in enumerate(insertion_loss_curves):
        ax_il.plot(np.asarray(insertion_loss_db).real, np.asarray(frequencies_hz).real, label=label,
                   zorder=2, **line_styles[curve_index % len(line_styles)])
    ax_il.axvline(0, color="gray", linewidth=0.8)
    ax_il.set_xlabel("IL [dB]")
    ax_il.set_title("Insertion loss (finite plate)")
    ax_il.grid(True, alpha=0.3)
    ax_il.legend(loc="lower left", fontsize=LEGEND_FONTSIZE, framealpha=0.9)  # empty: IL starts at 500 Hz


def finish_dispersion_figure(fig, ax, ax_il, path, frequency_limit, title, save_path, legend_location="best"):
    """Symmetry-point ticks and lines, limits, labels, grid, legend; then save or show."""
    for position in path.symmetry_point_positions:
        ax.axvline(position, color="0.3", linewidth=0.8)
    ax.set_xticks(path.symmetry_point_positions)
    ax.set_xticklabels(path.symmetry_point_labels)
    ax.set_xlim(path.path_coordinate[0], path.path_coordinate[-1])
    ax.set_ylim(0, frequency_limit)
    ax.set_xlabel("Wavevector k")
    ax.set_ylabel("Frequency [Hz]")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_location, fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    if ax_il is None:
        ax.set_title(title)
    else:
        ax.set_title("Band structure (unit cell)")
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_dispersion(result, analytic_frequencies=None, frequency_limit=None,
                    title="Dispersion diagram", save_path=None, analytic_label="analytic (folded plate)",
                    bare_frequencies=None, resonator_frequencies_hz=None, band_gaps=None,
                    insertion_loss_curves=None):
    """Band diagram along the wavevector path (Gamma-X-M-Gamma or Gamma-X-M-Y-Gamma).

    result: DispersionResult from dispersion.dispersion_sweep.
    analytic_frequencies: optional, shape (n_k, n_bands) on the same path (dispersion.analytic_dispersion),
        drawn as thin grey lines underneath as a check for the bare cell. Compute it slightly beyond the
        y-limit so the lines reach the top edge.
    analytic_label: legend label of the analytic overlay (e.g. the plane-wave expansion for a resonator cell).
    bare_frequencies: optional bare-plate bands on the same path, drawn as light dashed lines, to show what
        the resonators change.
    resonator_frequencies_hz: optional list of resonator frequencies sqrt(k/m)/2pi, drawn as horizontal lines.
    band_gaps: optional list of dispersion.BandGap (find_complete_gaps), shaded and labelled.
    insertion_loss_curves: optional list of (label, frequencies_hz, insertion_loss_db) of the finite plate,
        drawn in a side panel that shares the frequency axis.
    frequency_limit: optional upper y-limit in Hz; defaults to the sweep's maximum_frequency_hz, if set.
    save_path: if given, saves the figure instead of showing it.

    The bands are sorted by frequency at each k (no branch tracking), so they are drawn as markers:
    lines would show artificial kinks wherever two branches cross.
    """
    path = result.path
    if frequency_limit is None:
        frequency_limit = result.maximum_frequency_hz
    fig, ax, ax_il = dispersion_figure(insertion_loss_curves)

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

    draw_band_gaps(ax, band_gaps, frequency_limit, ax_il=ax_il)
    if ax_il is not None:
        draw_insertion_loss_panel(ax_il, insertion_loss_curves)
    finish_dispersion_figure(fig, ax, ax_il, path, frequency_limit, title, save_path)


def plate_share_colormap():
    """Diverging map for the plate share 0..1: red = resonator motion, grey = fully hybridized (50 %),
    blue = plate motion. The grey centre stays visible on a white background."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("plate_share", ["#b2182b", "#8c8c8c", "#2166ac"])


def plot_tracked_dispersion(result, colour_by="plate_share", frequency_limit=None, title="Dispersion diagram",
                            save_path=None, resonator_frequencies_hz=None, analytic_frequencies=None,
                            analytic_label="analytic (folded plate)", band_gaps=None, insertion_loss_curves=None):
    """Band diagram with tracked branches (dispersion_sweep(..., track_bands=True)) drawn as continuous lines.

    The line itself carries the branch identity: at a true crossing two lines pass through each other, at an
    avoided crossing they bend apart. The colour carries
      colour_by="plate_share": plate part of the kinetic energy along the branch (needs resonators to be useful);
      colour_by="branch": one colour per branch (check plot for the tracker: a wrong swap shows as a colour jump).
    Steps where the tracker fell back to plain sorting or broke a branch are marked by short ticks at the top.
    analytic_frequencies (optional, sorted per k) are drawn as thin grey lines underneath.
    band_gaps and insertion_loss_curves as in plot_dispersion (gaps come from the sorted frequencies)."""
    from matplotlib.collections import LineCollection
    tracking = result.tracking
    if tracking is None:
        raise ValueError("result has no tracking; run dispersion_sweep(..., track_bands=True)")
    if colour_by not in ("plate_share", "branch"):
        raise ValueError(f"colour_by must be 'plate_share' or 'branch', got {colour_by!r}")
    path = result.path
    if frequency_limit is None:
        frequency_limit = result.maximum_frequency_hz
    fig, ax, ax_il = dispersion_figure(insertion_loss_curves)

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
    for step in flagged_steps:
        ax.plot(path.path_coordinate[step], 0.985 * frequency_limit, marker="v", color="k", markersize=5,
                zorder=3, label="fallback / broken step" if step == flagged_steps[0] else None)

    if colour_by == "plate_share":
        colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=colormap),
                                 ax=ax if ax_il is None else [ax, ax_il], pad=0.02)
        colourbar.set_label("plate share of kinetic energy")
        colourbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        colourbar.set_ticklabels(["0 (resonator)", "0.25", "0.5", "0.75", "1 (plate)"])

    draw_band_gaps(ax, band_gaps, frequency_limit, ax_il=ax_il)
    if ax_il is not None:
        draw_insertion_loss_panel(ax_il, insertion_loss_curves)
    finish_dispersion_figure(fig, ax, ax_il, path, frequency_limit, title, save_path, legend_location="lower right")
