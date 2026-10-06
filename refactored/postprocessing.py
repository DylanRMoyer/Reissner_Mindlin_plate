import pyvista
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.patches import Rectangle
from dolfinx import fem, plot

# --- Plotting solutions to Eigenvalue problem ---

def create_w_plot(domain, function_space, eigenmodes, eigenmode_index, length):

    deg = function_space.ufl_element().degree

    if eigenmode_index > len(eigenmodes) - 1:
        raise ValueError(f"Eigenmode {eigenmode_index} out of range, "
                         f"highest possible index is {len(eigenmodes) - 1}.")
    mode_to_plot, q_r_to_plot = eigenmodes[eigenmode_index]
    w_mode = mode_to_plot.sub(0).collapse()   # scalar w-part of this eigenmode, still on Serendipity space

    V_plot = fem.functionspace(domain, ("Lagrange", deg))
    w_plot = fem.Function(V_plot)
    w_plot.interpolate(w_mode)

    topology, cell_types, geometry = plot.vtk_mesh(V_plot)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["w"] = w_plot.x.array.real

    max_w = np.max(np.abs(w_plot.x.array))
    target_visual_amplitude = 0.05 * length
    factor_scale = target_visual_amplitude / max_w

    warped = grid.warp_by_scalar("w", factor=factor_scale)

    return warped, mode_to_plot, q_r_to_plot, w_mode, factor_scale, deg


# --- Rotation field (theta) overlay, on top of the warped w-surface ---

def add_theta_plot(function_space, mode_to_plot, warped, factor_scale, deg):

    theta_mode = mode_to_plot.sub(1).collapse()   # vector theta-part of this eigenmode

    V_plot_vec = fem.functionspace(function_space.mesh, ("Lagrange", deg, (2,)))
    theta_plot = fem.Function(V_plot_vec)
    theta_plot.interpolate(theta_mode)

    # theta is a 2-component in-plane field; pad with a zero z-component so
    # PyVista (which wants 3D vectors) can glyph it, and so we can lay the
    # arrows flat/tangent at each warped surface point.
    theta_vals = theta_plot.x.array.reshape((-1, 2)).real
    theta_3d = np.zeros((theta_vals.shape[0], 3))
    theta_3d[:, 0:2] = theta_vals

    # attach to the *warped* grid so arrows sit at the already-deformed points
    warped["theta"] = theta_3d

    glyphs = warped.glyph(orient="theta", scale="theta", factor=factor_scale * 0.1)
    # factor here is a separate visual scale from the w-warp factor —
    # adjust the *.1 multiplier to taste if arrows are too small/large

    return glyphs


def add_point_mass_to_plot(vamm, phi, w_mode, q_r_to_plot, local_to_global_w, factor_scale, deg):

    attachment_xy = np.array([vamm.x, vamm.y])
    plate_w_at_target = np.dot(phi, w_mode.x.array[local_to_global_w])  # w_h at target, this mode

    mass_marker = pyvista.PolyData(np.array([[
        attachment_xy[0],
        attachment_xy[1],
        factor_scale * q_r_to_plot
    ]]))

    # a line from the plate surface to the mass, i.e. the spring itself
    spring_line = pyvista.Line(
        pointa=[attachment_xy[0], attachment_xy[1], factor_scale * plate_w_at_target.real],
        pointb=[attachment_xy[0], attachment_xy[1], factor_scale * q_r_to_plot.real]
    )

    return mass_marker, spring_line


def build_plot(warped, glyphs=None, mass_markers=None, spring_lines=None,
                view_vector=(2, 2, -1), save_path=None):
    """Assemble the full eigenmode plot from its sub-parts.

        glyphs: theta-rotation glyphs (from add_theta_plot). Omit to skip.
        mass_markers, spring_lines: lists of point-mass visualizations, one
            per VAMM (from repeated add_point_mass_to_plot calls). Omit
            either/both to skip. Must be same-length lists if both given.
        view_vector: camera view direction. Defaults to (2, 2, -1), matching
            the original script's fixed viewing angle.
        save_path: if given, saves the plot to this path (as PDF) instead of
            showing it interactively.
        """
    p = pyvista.Plotter()

    p.add_mesh(warped, scalars="w", show_edges=True)

    if glyphs is not None:
        p.add_mesh(glyphs, color="red")

    if mass_markers is not None:
        for mass_marker in mass_markers:
            p.add_mesh(mass_marker, color="blue", point_size=15, render_points_as_spheres=True)

    if spring_lines is not None:
        for spring_line in spring_lines:
            p.add_mesh(spring_line, color="blue", line_width=3)

    p.show_axes()
    p.view_vector(view_vector)

    if save_path is not None:
        p.save_graphic(save_path)
    else:
        p.show(auto_close=False)

    p.close()

def plot_eigenmode(domain, function_space, eigenmodes, eigenmode_index, length,
                    vamm_list=None, phi_list=None, local_to_global_w_list=None,
                    include_theta=True, include_mass=True,
                    view_vector=(2, 2, -1), save_path=None):
    """Build and display/save the full eigenmode plot for one mode.
    By default, all overlays (theta, mass) are included.

    Set include_theta/include_mass=False to skip those overlays.
    vamm_list/phi_list/local_to_global_w_list are required only if
    include_mass=True, which is the default setting. Every VAMM in
    vamm_list gets its own marker and spring line.
    If the system is to be solved without VAMMs to begin with, that step is skipped regardless.
    """
    warped, mode_to_plot, q_r_values_to_plot, w_mode, factor_scale, deg = create_w_plot(
        domain, function_space, eigenmodes, eigenmode_index, length
    )

    glyphs = None
    if include_theta:
        glyphs = add_theta_plot(function_space, mode_to_plot, warped, factor_scale, deg)

    mass_markers, spring_lines = None, None

    if q_r_values_to_plot is not None:
        if include_mass:
            mass_markers = []
            spring_lines = []
            for vamm, phi, local_to_global_w in zip(vamm_list, phi_list, local_to_global_w_list):
                q_r = q_r_values_to_plot[vamm.index]
                marker, spring_line = add_point_mass_to_plot(
                    vamm, phi, w_mode, q_r, local_to_global_w, factor_scale, deg
                )
                mass_markers.append(marker)
                spring_lines.append(spring_line)

    build_plot(warped, glyphs=glyphs, mass_markers=mass_markers, spring_lines=spring_lines,
               view_vector=view_vector, save_path=save_path)

# --- Plotting routine for frequency responses ---

def plot_frequency_response(f_values, rms_velocity, w_max_plate=None,
                             use_max_metric=False, eigenfrequencies=None, save_path=None):
    """Plot the plate's frequency response.

    By default plots the spatial RMS surface velocity (smooth, robust
    across the whole sweep -- see frequency_sweep_plate). If
    use_max_metric=True, plots the signed peak plate deflection instead
    (a pointwise metric; only meaningful near isolated resonances -- see
    frequency_sweep_plate's docstring for why it can be discontinuous
    between resonances). w_max_plate must be provided (not None) when
    use_max_metric=True, i.e. the sweep must have been run with
    compute_max_metric=True.

    f_values: frequencies in Hz (not Omega/rad-s) — same length as
        whichever metric array is plotted.
    rms_velocity: spatial RMS surface velocity values from
        frequency_sweep_plate. Always required.
    w_max_plate: signed peak |w|-with-sign values from
        frequency_sweep_plate, or None if that metric wasn't computed.
        Only used when use_max_metric=True.
    use_max_metric: if True, plot w_max_plate instead of rms_velocity.
    eigenfrequencies: optional list of Hz values to overlay as vertical
        lines (e.g. from solve_evp), for later use — omit for now.
    save_path: if given, saves to this path instead of showing interactively.
    """
    if use_max_metric and w_max_plate is None:
        raise ValueError(
            "use_max_metric=True requires w_max_plate (run frequency_sweep_plate "
            "with compute_max_metric=True first)"
        )

    if use_max_metric:
        y_values = w_max_plate
        y_label = "Peak plate deflection $w_{max}$ [m] (signed)"
        title = "Frequency response: Peak plate deflection under shaker base excitation"
    else:
        y_values = rms_velocity
        y_label = "RMS surface velocity [m/s]"
        title = "Frequency response: RMS surface velocity under shaker base excitation"

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(f_values, y_values, color="C0", linewidth=1.2)

    if use_max_metric:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    if eigenfrequencies is not None:
        labeled = False
        for f_n in eigenfrequencies:
            if f_values[0] <= f_n <= f_values[-1]:
                ax.axvline(f_n, color="red", linewidth=0.8, linestyle=":",
                           label="eigenfrequencies" if not labeled else None)
                labeled = True
        if labeled:
            ax.legend()

    ax.set_xlabel("Excitation frequency [Hz]")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)

# --- Plotting routine for mobility Bode plots ---

from dataclasses import dataclass

@dataclass
class PlateLayout:
    plate_config: object
    shaker_config: object
    probes: list | None = None      # Probe objects (need .x and .y)
    vamm_list: list | None = None
    connect: bool = True            # dashed shaker->probe line; False for hand-placed probes

def draw_plate_layout(ax, plate_config, shaker_config, probes=None,
                      vamm_list=None, connect=True, fontsize=7, highlight=None):
    """Sketch of the plate: outline, shaker (red star), VAMMs (blue squares),
    numbered probes (black dots). connect=True draws the shaker->probe line,
    meaningful for ray layouts; set False for hand-placed probes.
    Draws on whatever Axes it is given (inset or dedicated panel)."""
    L, W = plate_config.length, plate_config.width
    ax.add_patch(Rectangle((0, 0), L, W, fill=False, edgecolor="gray", linewidth=1.2))

    if probes and connect:
        xs = [shaker_config.x] + [p.x for p in probes]
        ys = [shaker_config.y] + [p.y for p in probes]
        ax.plot(xs, ys, linestyle="--", color="gray", linewidth=0.8, zorder=1)

    for v in (vamm_list or []):
        ax.plot(v.x, v.y, marker="s", color="C0", markersize=4, linestyle="none", zorder=3)

    for i, p in enumerate(probes or []):
        is_hl = (i == highlight)
        ax.plot(p.x, p.y, marker="o", color="C1" if is_hl else "k",
                markersize=6 if is_hl else 3.5, linestyle="none", zorder=4)
        ax.annotate(str(i), (p.x, p.y), textcoords="offset points",
                    xytext=(3, 3), fontsize=fontsize)

    ax.plot(shaker_config.x, shaker_config.y, marker="*", color="red",
            markersize=8, linestyle="none", zorder=5)

    pad = 0.05 * max(L, W)
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-pad, W + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def add_layout_inset(ax, layout: PlateLayout, bounds=(0.76, 0.58, 0.23, 0.38), highlight=None):
    inset = ax.inset_axes(list(bounds))   # bounds: [x0, y0, width, height] in axes fractions
    draw_plate_layout(inset, layout.plate_config, layout.shaker_config,
                      probes=layout.probes, vamm_list=layout.vamm_list,
                      connect=layout.connect, highlight=highlight)
    return inset

Y_REF = 1.0  # mobility reference [m/(N s)] for dB levels; keep one value for all figures

def _phase_for_plot(f_values, Y, mode):
    """Return (f, phase_deg) ready for plotting. 'wrapped' inserts NaN at every jump
    across +-180 deg, so the line breaks there instead of drawing a vertical streak."""
    if mode == "unwrapped":
        return f_values, np.degrees(np.unwrap(np.angle(Y)))
    if mode != "wrapped":
        raise ValueError(f"phase must be 'wrapped' or 'unwrapped', got {mode!r}")
    phi = np.degrees(np.angle(Y))
    jumps = np.where(np.abs(np.diff(phi)) > 180)[0] + 1
    return np.insert(f_values, jumps, np.nan), np.insert(phi, jumps, np.nan)


def plot_mobility_bode(
        f_values, mobilities, labels, eigenfrequencies=None, y_infinite=None, title="Mobility",
        save_path=None, layout=None, highlight_probe=None, phase="wrapped"):
    """Bode plot (magnitude in dB, phase in degrees) of one or more complex
    mobilities Y(f), plotted on shared frequency axes.

    f_values: frequencies in Hz, shape (n_freq,).
    mobilities: list of complex arrays, each shape (n_freq,), e.g.
        [sweep_bare.probe_mobility[0], sweep_vamm.probe_mobility[0]].
        A single 1D array is also accepted.
    labels: list of legend labels, same length as mobilities.
    eigenfrequencies: optional. Either a flat list of Hz values (one red set of
        vertical dotted lines), or a list of lists, one set per entry of
        mobilities and in the same order, drawn in the matching curve's color,
        e.g. [eigenfrequencies_bare, eigenfrequencies_vamm].
    y_infinite: optional infinite-plate driving-point mobility
        1/(8*sqrt(D*rho*h)) [m/(N s)], drawn as a horizontal guide line
        in the magnitude panel.
    save_path: if given, saves the figure instead of showing it.
    layout: optional PlateLayout; if given, a plate map (shaker, VAMMs, numbered
        probes) is drawn in its own panel to the right, with the legend below it,
        so nothing covers the curves.
    highlight_probe: index of the probe whose mobility is plotted, marked on the
        map (None for none, e.g. a driving-point plot).
    phase: 'wrapped' (default; values in (-180, 180], line broken at the jumps)
        or 'unwrapped'. Unwrapping is only trustworthy if every resonance and
        antiresonance is resolved, and for transfer mobilities the absolute
        offset (multiples of 360 deg) is not physically meaningful.

    Level convention: 20*log10(|Y| / Y_REF), i.e. dB re 1 m/(N s).
    """
    if isinstance(mobilities, np.ndarray) and mobilities.ndim == 1:
        mobilities = [mobilities]
    if len(mobilities) != len(labels):
        raise ValueError(f"got {len(mobilities)} mobilities but {len(labels)} labels")

    per_curve_eigs = (eigenfrequencies is not None and len(eigenfrequencies) > 0
                      and np.ndim(eigenfrequencies[0]) > 0)
    if per_curve_eigs and len(eigenfrequencies) != len(mobilities):
        raise ValueError(f"got {len(eigenfrequencies)} eigenfrequency sets "
                         f"but {len(mobilities)} mobilities")

    if layout is not None:
        fig = plt.figure(figsize=(11.5, 7), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.3], height_ratios=[2, 1])
        ax_mag = fig.add_subplot(gs[0, 0])
        ax_phase = fig.add_subplot(gs[1, 0], sharex=ax_mag)
        ax_layout = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, 1])
        ax_legend.axis("off")
    else:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    curve_colors = []
    for Y, label in zip(mobilities, labels):
        magnitude_db = 20 * np.log10(np.abs(Y) / Y_REF)
        line, = ax_mag.plot(f_values, magnitude_db, linewidth=1.2, label=label)
        curve_colors.append(line.get_color())
        f_phase, phase_deg = _phase_for_plot(f_values, Y, phase)
        ax_phase.plot(f_phase, phase_deg, linewidth=1.2, color=line.get_color(), label=label)

    if y_infinite is not None:
        ax_mag.axhline(20 * np.log10(y_infinite / Y_REF), color="gray",
                       linewidth=0.9, linestyle="--", label="infinite plate")

    if eigenfrequencies is not None:
        if per_curve_eigs:
            eig_sets = [(f_set, color, f"{label} eigenfrequencies")
                        for f_set, color, label in zip(eigenfrequencies, curve_colors, labels)]
        else:
            eig_sets = [(eigenfrequencies, "red", "eigenfrequencies")]
        for f_set, color, set_label in eig_sets:
            labeled = False
            for f_n in f_set:
                if f_values[0] <= f_n <= f_values[-1]:
                    for ax in (ax_mag, ax_phase):
                        ax.axvline(f_n, color=color, linewidth=0.8, linestyle=":", alpha=0.7,
                                   label=set_label if (not labeled and ax is ax_mag) else None)
                    labeled = True

    ax_mag.set_ylabel("Mobility level [dB re 1 m/(N s)]")
    ax_mag.set_title(title)
    ax_mag.grid(True, alpha=0.3)

    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.grid(True, alpha=0.3)
    if phase == "wrapped":
        ax_phase.set_yticks(np.arange(-180, 181, 90))
        ax_phase.set_ylim(-190, 190)

    if layout is not None:
        draw_plate_layout(ax_layout, layout.plate_config, layout.shaker_config,
                          probes=layout.probes, vamm_list=layout.vamm_list,
                          connect=layout.connect, highlight=highlight_probe)
        handles, legend_labels = ax_mag.get_legend_handles_labels()
        ax_legend.legend(handles, legend_labels, loc="upper left")
    else:
        ax_mag.legend()
        plt.tight_layout()      # not used with constrained layout

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)


# --- Helpers for plot_insertion_loss_bode ---

def _break_at_wrap(f_values, phase_deg, jump_deg=180.0):
    """Insert NaNs where a wrapped phase jumps by more than jump_deg between
    neighbouring samples, so matplotlib does not draw a line across the whole
    panel at every +-180 deg wrap. Returns (f_with_nans, phase_with_nans).
    (Swap in your existing helper from plot_mobility_bode if you prefer.)"""
    f = np.asarray(f_values, dtype=float)
    p = np.asarray(phase_deg, dtype=float)
    jumps = np.where(np.abs(np.diff(p)) > jump_deg)[0] + 1
    return np.insert(f, jumps, np.nan), np.insert(p, jumps, np.nan)


def _default_probe_colors(n_probes):
    """One distinct colour per probe, for any number of probes."""
    if n_probes <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n_probes)]
    cmap = plt.get_cmap("turbo")
    return [cmap(i / (n_probes - 1)) for i in range(n_probes)]


def _draw_eigenfrequency_lines(axes, f_values, frequencies, label, label_axis, **style):
    """Vertical lines at the eigenfrequencies inside the plotted band, in every
    axis of `axes`; only the first line in `label_axis` carries a legend label."""
    if frequencies is None:
        return
    in_band = [fn for fn in frequencies if f_values[0] <= fn <= f_values[-1]]
    for ax in axes:
        for k, fn in enumerate(in_band):
            ax.axvline(fn, zorder=0, label=label if (ax is label_axis and k == 0) else None, **style)


# --- Insertion-loss Bode plot ---

def plot_insertion_loss_bode(
        f_values, il_pointwise, dphi_deg, il_force, il_power=None,
        eigenfrequencies_vamm=None, eigenfrequencies_bare=None,
        probe_labels=None, probe_colors=None, probes_to_show=None,
        layout=None, draw_layout=None,
        il_ylim=None, log_x=False,
        title="Insertion loss", save_path=None):
    """Three stacked panels sharing the frequency axis, plus an optional plate-layout panel:
        top:    global insertion loss, IL^F (solid) and, if given, IL^P (dashed)
        middle: pointwise insertion loss per probe [dB]
        bottom: phase shift per probe, wrapped to (-180, 180] deg
    Positive IL = attenuation by the VAMMs, a horizontal line marks 0 dB.

    Works for any number of probes: n_probes is read from the shape of il_pointwise.

    f_values: frequencies in Hz, shape (n_freq,)
    il_pointwise, dphi_deg: shape (n_probes, n_freq), from insertion_loss.py.
        Row 0 is treated as the driving point (shaker location) in the default labels.
    il_force, il_power: shape (n_freq,); il_power may be None.
    eigenfrequencies_vamm / eigenfrequencies_bare: optional lists in Hz. The VAMM-system
        ones are drawn darker and dashed, the bare ones lighter and dotted.
    probe_labels: optional list of n_probes strings (default: "probe i").
    probe_colors: optional list of n_probes colours, e.g. the ones used in the mobility
        plots; indexed by absolute probe index, so colours stay stable when
        probes_to_show selects a subset. Default: tab10 (turbo for > 10 probes).
    probes_to_show: optional iterable of probe indices to plot (default: all).
    layout, draw_layout: if both are given, a plate-layout panel is added on the right and
        draw_layout(ax, layout) is called to fill it (adapt to draw_plate_layout's signature).
    il_ylim: optional (low, high) for the pointwise panel; IL spikes at
        antiresonances can otherwise dominate the scale.
    log_x: logarithmic frequency axis.
    save_path: if given, saves the figure instead of showing it interactively.
    """
    f = np.asarray(f_values, dtype=float)
    il_pointwise = np.atleast_2d(np.asarray(il_pointwise, dtype=float))
    dphi_deg = np.atleast_2d(np.asarray(dphi_deg, dtype=float))
    il_force = np.asarray(il_force, dtype=float)
    n_probes, n_freq = il_pointwise.shape

    if dphi_deg.shape != il_pointwise.shape:
        raise ValueError(f"dphi_deg shape {dphi_deg.shape} != il_pointwise shape {il_pointwise.shape}")
    if len(f) != n_freq or il_force.shape != (n_freq,):
        raise ValueError(f"f_values (len {len(f)}), il_force {il_force.shape} and il_pointwise "
                         f"({n_probes}, {n_freq}) must share the same number of frequencies")
    if il_power is not None:
        il_power = np.asarray(il_power, dtype=float)
        if il_power.shape != (n_freq,):
            raise ValueError(f"il_power shape {il_power.shape} != ({n_freq},)")

    indices = list(range(n_probes)) if probes_to_show is None else list(probes_to_show)
    if not indices or any(i < 0 or i >= n_probes for i in indices):
        raise ValueError(f"probes_to_show must be non-empty indices in [0, {n_probes - 1}], got {indices}")

    if probe_labels is None:
        probe_labels = ["probe 0 (driving point)"] + [f"probe {i}" for i in range(1, n_probes)]
    if probe_colors is None:
        probe_colors = _default_probe_colors(n_probes)
    if len(probe_labels) != n_probes or len(probe_colors) != n_probes:
        raise ValueError(f"probe_labels/probe_colors need {n_probes} entries, got "
                         f"{len(probe_labels)}/{len(probe_colors)}")

    with_layout = layout is not None and draw_layout is not None
    fig = plt.figure(figsize=(12.5 if with_layout else 9.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2 if with_layout else 1,
                          width_ratios=[3, 1] if with_layout else None,
                          height_ratios=[1, 1.3, 1])
    ax_global = fig.add_subplot(gs[0, 0])
    ax_point = fig.add_subplot(gs[1, 0], sharex=ax_global)
    ax_phase = fig.add_subplot(gs[2, 0], sharex=ax_global)
    axes = (ax_global, ax_point, ax_phase)

    # Eigenfrequency lines first, so they sit behind the curves.
    _draw_eigenfrequency_lines(axes, f, eigenfrequencies_bare, "bare eigenfrequencies", ax_global,
                               color="0.7", linestyle=":", linewidth=0.9)
    _draw_eigenfrequency_lines(axes, f, eigenfrequencies_vamm, "VAMM-system eigenfrequencies", ax_global,
                               color="0.3", linestyle="--", linewidth=0.9)

    # Top: global insertion loss
    ax_global.plot(f, il_force, color="k", linewidth=1.4, label=r"$IL^F$ (equal force)")
    if il_power is not None:
        ax_global.plot(f, il_power, color="k", linewidth=1.2, linestyle="--", label=r"$IL^P$ (equal input power)")
    ax_global.axhline(0, color="gray", linewidth=0.8)
    ax_global.set_ylabel("global IL [dB]")
    ax_global.set_title("Global insertion loss")
    ax_global.legend(loc="upper right", fontsize=8)

    # Middle: pointwise insertion loss
    for i in indices:
        ax_point.plot(f, il_pointwise[i], color=probe_colors[i], linewidth=1.1, label=probe_labels[i])
    ax_point.axhline(0, color="gray", linewidth=0.8)
    if il_ylim is not None:
        ax_point.set_ylim(*il_ylim)
    ax_point.set_ylabel("pointwise IL [dB]")
    ax_point.set_title("Pointwise insertion loss")
    ax_point.legend(loc="upper right", fontsize=8, ncol=min(len(indices), 3))

    # Bottom: phase shift, wrapped
    for i in indices:
        f_plot, phase_plot = _break_at_wrap(f, dphi_deg[i])
        ax_phase.plot(f_plot, phase_plot, color=probe_colors[i], linewidth=1.1)
    ax_phase.axhline(0, color="gray", linewidth=0.8)
    ax_phase.set_ylim(-190, 190)
    ax_phase.set_yticks([-180, -90, 0, 90, 180])
    ax_phase.set_ylabel(r"$\Delta\varphi$ [deg]")
    ax_phase.set_title(r"Phase shift $\arg(Y_{vamm}/Y_{bare})$")
    ax_phase.set_xlabel("Frequency [Hz]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    ax_global.tick_params(labelbottom=False)
    ax_point.tick_params(labelbottom=False)
    if log_x:
        ax_global.set_xscale("log")
    ax_global.set_xlim(f[0], f[-1])

    if with_layout:
        ax_layout = fig.add_subplot(gs[:, 1])
        draw_layout(ax_layout, layout)
        ax_layout.set_title("Probe layout")

    fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)