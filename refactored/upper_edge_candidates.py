"""Section 5.11, Rule 2: the upper edge of the local-resonance gap is the lower of two candidates,
  (a) band 2 at Gamma (the resonator band):  f_Gamma = f_r sqrt(1 + mu (1 - S(f_Gamma))),
  (b) the sin pattern at X (pure Bragg):     f_X.
This script plots both candidates against tau = f_r / f_X for each mass ratio and marks the switch.

Plane-wave (Reissner-Mindlin) model, same sum as dispersion.resonator_dispersion_function. At Gamma the G = 0 term
of 1 = mu_eff * rho h omega^2 * sum_G g(|G|, omega) is exactly -1 (g = -1 / (rho h omega^2) for |G| = 0), so
    1 = mu_eff (-1 + S),   S(f) = rho h omega^2 * sum_{G != 0} g(|G|, omega),   mu_eff = mu / (1 - (f_r / f)^2)
is exact and S depends only on the plate and the frequency, not on f_r or mu. Two consequences:
  - candidate (a) can be drawn parametrically in f, without root finding: tau(f) = (f / f_X) / sqrt(1 + mu (1 - S(f)));
  - at the switch f_Gamma = f_X, so tau_switch = 1 / sqrt(1 + mu (1 - S(f_X))) is exact, with S taken at f_X.
Checked against analytic_resonator_frequencies at k = 0 (band 2 at Gamma) and shown next to the true upper edges
of the cached stage 7 gap study (band 2 minimum over the path).

Run in fenicsx-env-complex (~10 s). Writes results/upper_edge_candidates.pdf."""
import dataclasses
from pathlib import Path

import numpy as np

from dispersion import plate_constants, plane_wave_point_compliance, analytic_resonator_frequencies
from parameter_study import bragg_frequency_x, resonator_parameters

RESULTS_DIR = Path(__file__).parent / "results"
HARMONIC_CUTOFF = 12        # same as analytic_resonator_frequencies (the RM sum diverges like log(cutoff))


def gamma_spring_sum(plate_config, frequency_hz, harmonic_cutoff=HARMONIC_CUTOFF):
    """S(f) = rho h omega^2 * sum over all reciprocal vectors G != 0 of the RM point compliance g(|G|, omega):
    the spring-like part of the plane-wave sum at Gamma (the G = 0 term, exactly -1, is left out)."""
    constants = plate_constants(plate_config)
    harmonics = np.arange(-harmonic_cutoff, harmonic_cutoff + 1)
    reciprocal_x = 2 * np.pi * harmonics[:, None] / plate_config.length
    reciprocal_y = 2 * np.pi * harmonics[None, :] / plate_config.width
    magnitudes = np.hypot(reciprocal_x, reciprocal_y).ravel()
    magnitudes = magnitudes[magnitudes > 0]
    angular_frequency = 2 * np.pi * np.atleast_1d(frequency_hz)[:, None]
    compliance = plane_wave_point_compliance(magnitudes[None, :], angular_frequency, constants, "reissner_mindlin")
    spring_sum = constants["mass_per_area"] * angular_frequency[:, 0] ** 2 * compliance.sum(axis=1)
    return spring_sum if np.ndim(frequency_hz) else spring_sum[0]


def resonator_candidate_curve(plate_config, mass_ratio, frequency_over_bragg):
    """Candidate (a) as a parametric curve: for each band-2 frequency f (given as f / f_X) the tuning
    tau = f_r / f_X that puts band 2 at Gamma exactly there. Returns (tau, f / f_X, S)."""
    bragg_frequency_hz = bragg_frequency_x(plate_config)
    spring_sum = gamma_spring_sum(plate_config, frequency_over_bragg * bragg_frequency_hz)
    frequency_factor_squared = 1 + mass_ratio * (1 - spring_sum)          # beta = (f / f_r)^2
    with np.errstate(invalid="ignore"):     # beta <= 0 (S > 1 + 1/mu, only near the next poles): no tuning, NaN
        tau = frequency_over_bragg / np.sqrt(frequency_factor_squared)
    return tau, frequency_over_bragg, spring_sum


def switch_ratio(plate_config, mass_ratio):
    """tau_switch = 1 / sqrt(1 + mu (1 - S(f_X))): the tuning at which band 2 at Gamma equals f_X (exact in the
    plane-wave model, since S depends on frequency only)."""
    spring_sum_at_bragg = gamma_spring_sum(plate_config, bragg_frequency_x(plate_config))
    return 1 / np.sqrt(1 + mass_ratio * (1 - spring_sum_at_bragg))


def band_two_at_gamma(plate_config, frequency_ratio, mass_ratio):
    """Band 2 at Gamma (Hz) from the full root finder (independent check of the parametric curve)."""
    bragg_frequency_hz = bragg_frequency_x(plate_config)
    stiffness, mass = resonator_parameters(plate_config, frequency_ratio * bragg_frequency_hz, mass_ratio)
    frequencies = analytic_resonator_frequencies(0.0, 0.0, plate_config, stiffness, mass,
                                                 maximum_frequency_hz=3.0 * bragg_frequency_hz)
    return frequencies[1]          # frequencies[0] is the rigid zero mode


def load_cached_gap_study(name_prefix):
    """Upper and lower edge of the lowest complete gap from the cached stage 7 gap study (in units of f_X)."""
    path = sorted(RESULTS_DIR.glob(f"gap_study_{name_prefix}_*.npz"))[0]
    with np.load(path) as data:
        edges = data["gap_edges_hz"][:, :, 0, :] / float(data["bragg_frequency_hz"])
        return data["frequency_ratios"], data["mass_ratios"], edges


def plot_upper_edge_candidates(cells, mass_ratios, save_path):
    """One panel per mass ratio: candidate (a) vs tau (solid: 0.1 m cell, dotted: Cool cell), candidate (b) = 1,
    the S = 0 estimate f_r sqrt(1 + mu) (dashed), the true gap edges of the cached study (markers) and the switch.
    Last panel: tau_switch vs mu for both cells and for S = 0."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(9.5, 7.0), layout="constrained")
    frequency_grid = np.linspace(0.2, 2.6, 2000)
    tau_limits = (0.5, 2.0)
    for axis, mass_ratio in zip(axes.flat, mass_ratios):
        for cell_name, (cell_config, study_prefix, line_style) in cells.items():
            tau, frequency_over_bragg, _ = resonator_candidate_curve(cell_config, mass_ratio, frequency_grid)
            axis.plot(tau, frequency_over_bragg, line_style, color="tab:blue", lw=1.4,
                      label=f"(a) band 2 at Γ, {cell_name}")
            if study_prefix == "cell0.1m":
                tau_switch = switch_ratio(cell_config, mass_ratio)
                axis.axvline(tau_switch, color="tab:red", lw=0.9, ls="-.")
                axis.annotate(f"τ_switch = {tau_switch:.3f}", (tau_switch, 1.5), xytext=(4, 0),
                              textcoords="offset points", fontsize=8, color="tab:red")
                ratios, study_mass_ratios, edges = load_cached_gap_study(study_prefix)
                mass_index = int(np.argmin(abs(study_mass_ratios - mass_ratio)))
                axis.plot(ratios, edges[mass_index, :, 1], "o", ms=3.5, color="k",
                          label="true upper edge (gap study)")
                axis.plot(ratios, edges[mass_index, :, 0], "o", ms=3, mfc="none", color="0.55",
                          label="true lower edge (gap study)")
        tau_line = np.linspace(*tau_limits, 50)
        axis.plot(tau_line, tau_line * np.sqrt(1 + mass_ratio), "--", color="0.5", lw=1,
                  label="S = 0: f_r √(1+μ)")
        axis.axhline(1.0, color="tab:orange", lw=1.4, label="(b) sin pattern at X: f_X")
        axis.plot(tau_line, tau_line, ":", color="0.7", lw=0.8, label="f = f_r")
        axis.set(xlim=tau_limits, ylim=(0.6, 1.6), title=f"μ = {mass_ratio:g}",
                 xlabel="τ = f_r / f_X", ylabel="f / f_X")
        axis.grid(True, alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)

    summary_axis = axes.flat[len(mass_ratios)]
    mass_ratio_grid = np.linspace(0.01, 1.5, 200)
    for cell_name, (cell_config, _, line_style) in cells.items():
        summary_axis.plot(mass_ratio_grid, switch_ratio(cell_config, mass_ratio_grid), line_style, color="tab:red",
                          label=f"{cell_name}")
    summary_axis.plot(mass_ratio_grid, 1 / np.sqrt(1 + mass_ratio_grid), "--", color="0.5", label="S = 0")
    summary_axis.set(xlabel="μ", ylabel="τ_switch", title="Switch of the upper edge")
    summary_axis.grid(True, alpha=0.3)
    summary_axis.legend(fontsize=8)
    figure.savefig(save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    from config import PlateConfig
    from compare_cool_2024 import cool_plate_config

    production_cell = PlateConfig(length=0.1, width=0.1, thickness=0.005, nx=10, ny=10,
                                  rho=7850, mu=77e9, lambda_=115e9)
    cool_cell = cool_plate_config()
    mass_ratios = (0.05, 0.1, 0.25, 0.5, 1.0)

    # Check 1: parametric curve == full root finder (band 2 at Gamma) at a few tunings
    worst_relative_error = 0.0
    for mass_ratio in (0.1, 0.25, 1.0):
        for frequency_ratio in (0.6, 0.9, 1.0, 1.3):
            band_two_over_bragg = band_two_at_gamma(production_cell, frequency_ratio, mass_ratio) \
                / bragg_frequency_x(production_cell)
            tau_from_curve, _, _ = resonator_candidate_curve(production_cell, mass_ratio,
                                                             np.array([band_two_over_bragg]))
            worst_relative_error = max(worst_relative_error, abs(tau_from_curve[0] / frequency_ratio - 1))
    print(f"Check 1, parametric curve vs root finder: worst relative tau error {worst_relative_error:.1e}")
    assert worst_relative_error < 1e-6

    # Check 2: production numbers of Section 5.8 (mu 0.25, f_r 1200 Hz: band 2 at Gamma 1273.5 Hz, S ~ 0.50)
    bragg_hz = bragg_frequency_x(production_cell)
    band_two_hz = band_two_at_gamma(production_cell, 1200 / bragg_hz, 0.25)
    print(f"Check 2, f_r = 1200 Hz, mu = 0.25: band 2 at Gamma {band_two_hz:.1f} Hz, "
          f"S = {gamma_spring_sum(production_cell, band_two_hz):.3f}")

    for cell_name, cell_config in (("0.1 m cell (h/a 0.05)", production_cell), ("Cool cell (h/a 0.1)", cool_cell)):
        spring_sum_at_bragg = gamma_spring_sum(cell_config, bragg_frequency_x(cell_config))
        switches = ", ".join(f"{switch_ratio(cell_config, mass_ratio):.3f}" for mass_ratio in mass_ratios)
        print(f"{cell_name}: f_X = {bragg_frequency_x(cell_config):.1f} Hz, S(f_X) = {spring_sum_at_bragg:.3f}, "
              f"tau_switch for mu {mass_ratios} = {switches}")

    cells = {"0.1 m cell": (production_cell, "cell0.1m", "-"),
             "Cool cell (h/a 0.1)": (cool_cell, "cool_cell0.05m", ":")}
    plot_upper_edge_candidates(cells, mass_ratios, RESULTS_DIR / "upper_edge_candidates.pdf")
