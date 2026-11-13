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


# --- Parameter study (stage 7): gap maps over the tuning f_r / f_X and the mass ratio ---

def plot_gap_map(studies, labels, title="Complete band gaps vs tuning", save_path=None, frequency_ratio_limit=4.0,
                 reference_designs=None):
    """One panel per mass ratio: complete gaps (vertical axis f / f_X) over the tuning f_r / f_X.
    studies: list of parameter_study.GapStudy with the same ratio grids; the first is drawn as shaded bars,
    the others as edge markers on top (e.g. a second cell as a scaling check). Reference lines: f = f_r,
    effective-medium upper edge f = f_r sqrt(1 + mu), first Bragg fold f = f_X.
    reference_designs: optional list of dicts (label, mass_ratio, frequency_ratio, lower_ratio, upper_ratio), e.g. an
    FE gap of a design computed elsewhere, drawn as a red bar in the panel whose mass ratio matches (else skipped)."""
    reference_study = studies[0]
    mass_ratios = reference_study.mass_ratios
    fig, axes = plt.subplots(1, len(mass_ratios), figsize=(9.5, 4.8), sharey=True, constrained_layout=True)
    marker_styles = ["o", "s", "^"]
    for mass_index, (ax, mass_ratio) in enumerate(zip(np.atleast_1d(axes), mass_ratios)):
        ratios = reference_study.frequency_ratios
        normalised_edges = reference_study.gap_edges_hz[mass_index] / reference_study.bragg_frequency_hz
        for gap_column in range(normalised_edges.shape[1]):
            lower, upper = normalised_edges[:, gap_column, 0], normalised_edges[:, gap_column, 1]
            ax.fill_between(ratios, lower, upper, where=np.isfinite(lower), step="mid", color=GAP_COLOUR,
                            alpha=0.8, linewidth=0, label=labels[0] if gap_column == 0 else None)
        for study_index, study in enumerate(studies[1:]):
            edges = study.gap_edges_hz[mass_index] / study.bragg_frequency_hz
            ax.plot(np.repeat(study.frequency_ratios, edges.shape[1]), edges[..., 0].ravel(),
                    marker_styles[study_index % 3], color="k", markersize=2.2, linestyle="none",
                    label=labels[study_index + 1])
            ax.plot(np.repeat(study.frequency_ratios, edges.shape[1]), edges[..., 1].ravel(),
                    marker_styles[study_index % 3], color="k", markersize=2.2, linestyle="none")
        ax.plot(ratios, ratios, color="tab:red", linewidth=0.9, linestyle=":", label="$f = f_r$")
        ax.plot(ratios, ratios * np.sqrt(1 + mass_ratio), color="tab:blue", linewidth=0.9, linestyle="--",
                label=r"$f = f_r\sqrt{1+\mu}$")
        ax.axhline(1.0, color="0.4", linewidth=0.8, label="$f = f_X$")
        ax.axvline(1.0, color="0.4", linewidth=0.8)
        for design in reference_designs or []:
            if np.isclose(design["mass_ratio"], mass_ratio):
                ax.plot([design["frequency_ratio"]] * 2, [design["lower_ratio"], design["upper_ratio"]],
                        color="tab:red", linewidth=2.0, marker="_", markersize=9, label=design["label"])
        ax.set_xlim(ratios[0], ratios[-1])
        ax.set_ylim(0, frequency_ratio_limit)
        ax.set_title(rf"$\mu$ = {mass_ratio:g}", fontsize=10)
        ax.set_xlabel("$f_r / f_X$")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("$f / f_X$")
    legend_entries = {}
    for ax in np.atleast_1d(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_entries.setdefault(label, handle)
    fig.legend(legend_entries.values(), legend_entries.keys(), loc="outside lower center", ncol=3,
               fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    fig.suptitle(title)
    _save_or_show(fig, save_path)


def plot_gap_width(studies, labels, title="Widest complete gap vs tuning", save_path=None):
    """Relative width (width / centre) of the widest complete gap over f_r / f_X, one colour per mass ratio.
    First study as lines, the others as markers; dashed: effective-medium width of [f_r, f_r sqrt(1 + mu)],
    2 (s - 1) / (s + 1) with s = sqrt(1 + mu), which ignores the plate's dispersion and the Bragg fold."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(studies[0].mass_ratios)))
    marker_styles = ["o", "s", "^"]
    for mass_index, (mass_ratio, colour) in enumerate(zip(studies[0].mass_ratios, colours)):
        ax.plot(studies[0].frequency_ratios, 100 * studies[0].widest_relative_width()[mass_index], color=colour,
                linewidth=1.4, label=rf"$\mu$ = {mass_ratio:g}")
        for study_index, study in enumerate(studies[1:]):
            ax.plot(study.frequency_ratios, 100 * study.widest_relative_width()[mass_index],
                    marker_styles[study_index % 3], color=colour, markersize=3, linestyle="none")
        scale = np.sqrt(1 + mass_ratio)
        ax.axhline(100 * 2 * (scale - 1) / (scale + 1), color=colour, linewidth=0.8, linestyle="--")
    ax.axvline(1.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("$f_r / f_X$")
    ax.set_ylabel("relative width of the widest complete gap [%]")
    ax.set_xlim(studies[0].frequency_ratios[0], studies[0].frequency_ratios[-1])
    ax.set_ylim(0, 1.45 * 100 * max(study.widest_relative_width().max() for study in studies))  # room for the legends
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    colour_legend = ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.9, title="mass ratio",
                              title_fontsize=LEGEND_FONTSIZE)
    ax.add_artist(colour_legend)
    style_handles = [Line2D([], [], color="k", linewidth=1.4)] + \
        [Line2D([], [], color="k", marker=marker_styles[index % 3], markersize=3, linestyle="none")
         for index in range(len(studies) - 1)] + [Line2D([], [], color="k", linewidth=0.8, linestyle="--")]
    ax.legend(style_handles, list(labels) + ["effective medium $f_r \\ldots f_r\\sqrt{1+\\mu}$"],
              loc="upper left", fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    ax.set_title(title)
    _save_or_show(fig, save_path)


def plot_insertion_loss_vs_gaps(cases, title="Finite-plate insertion loss vs Bloch gaps", save_path=None):
    """One panel per case: equal-force radiated-power IL of the finite plate over frequency, with the complete
    gaps of the matching Bloch cell shaded and f_r marked.
    cases: list of dicts with label, f_values, insertion_loss_db, gaps (list of (lower_hz, upper_hz)),
    resonator_frequency_hz; optional curve_label (default "$IL_{rad}$ (equal force)") and extra_curves, a list of
    (label, f_values, insertion_loss_db) drawn dashed on top (e.g. the same IL with another model setting)."""
    fig, axes = plt.subplots(1, len(cases), figsize=(9.5, 4.0), sharey=True, constrained_layout=True)
    for ax, case in zip(np.atleast_1d(axes), cases):
        f_values = np.asarray(case["f_values"]).real
        for gap_index, (lower_hz, upper_hz) in enumerate(case["gaps"]):
            ax.axvspan(lower_hz, upper_hz, color=GAP_COLOUR, alpha=0.6, linewidth=0, zorder=0,
                       label="complete gap (Bloch)" if gap_index == 0 else None)
        ax.plot(f_values, np.asarray(case["insertion_loss_db"]).real, color="k", linewidth=1.1,
                label=case.get("curve_label", "$IL_{rad}$ (equal force)"))
        for extra_label, extra_f_values, extra_insertion_loss_db in case.get("extra_curves", []):
            ax.plot(np.asarray(extra_f_values).real, np.asarray(extra_insertion_loss_db).real, color="tab:blue",
                    linewidth=1.1, linestyle="--", label=extra_label)
        ax.axvline(case["resonator_frequency_hz"], color="tab:red", linewidth=0.9, linestyle=":", label="$f_r$")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlim(f_values[0], f_values[-1])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_title(case["label"], fontsize=10)
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("IL [dB]")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    fig.suptitle(title)
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
