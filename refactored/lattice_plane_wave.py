"""Queue Q4 (stage 8 candidate): plane-wave model of a plate with one point attachment per cell on a general 2D
Bravais lattice (square or triangular), plane-wave model only (no FE cell for the triangular lattice).

Same condition as dispersion.resonator_dispersion_function, with general reciprocal vectors:
    1 = Z(omega) / A * sum_G g(|k + G|, omega),   G = m b1 + n b2,
A = cell area, g = plane-wave point compliance (Kirchhoff or Reissner-Mindlin), and
    Z = k m omega^2 / (k - m omega^2)   (resonator),   Z = m omega^2   (rigid mass, the limit k -> infinity).
The G sum is cut off on a circle |G| <= G_max (the square-lattice code uses a square |m|, |n| <= N), so lattices
of different shape keep the same attachment size in the log-divergent Reissner-Mindlin sum; for Kirchhoff the sum
converges and the cutoff does not matter.

Run in fenicsx-env-complex (dispersion imports the FE modules)."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from dispersion import (WavevectorPath, plate_constants, plane_wave_point_compliance, plane_wave_bare_frequency_hz,
                        resonator_force_factor, pad_with_nan)


@dataclass
class Lattice:
    name: str
    lattice_vectors: np.ndarray        # rows a1, a2 (m)
    reciprocal_vectors: np.ndarray     # rows b1, b2 (rad/m), a_i . b_j = 2 pi delta_ij
    symmetry_points: list              # wavevectors of the irreducible path corners, rad/m
    symmetry_point_labels: list

    @property
    def cell_area(self):
        return abs(np.linalg.det(self.lattice_vectors))


def make_lattice(name, lattice_vectors, symmetry_points, labels):
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    reciprocal_vectors = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    return Lattice(name=name, lattice_vectors=lattice_vectors, reciprocal_vectors=reciprocal_vectors,
                   symmetry_points=[np.asarray(point, dtype=float) for point in symmetry_points],
                   symmetry_point_labels=labels)


def square_lattice(spacing):
    """Square lattice, path Gamma-X-M-Gamma (as dispersion.build_wavevector_path)."""
    return make_lattice("square", [[spacing, 0.0], [0.0, spacing]],
                        [[0, 0], [np.pi / spacing, 0], [np.pi / spacing, np.pi / spacing], [0, 0]],
                        ["Γ", "X", "M", "Γ"])


def triangular_lattice(spacing):
    """Triangular (hexagonal) lattice a1 = a (1, 0), a2 = a (1/2, sqrt(3)/2); path Gamma-M-K-Gamma.
    M = b2 / 2 (edge midpoint, |M| = 2 pi / (sqrt(3) a), 2 waves degenerate),
    K = (2 pi / (3 a), 2 pi / (sqrt(3) a)) (zone corner, |K| = 4 pi / (3 a), 3 waves degenerate)."""
    return make_lattice("triangular", [[spacing, 0.0], [0.5 * spacing, np.sqrt(3) / 2 * spacing]],
                        [[0, 0], [0, 2 * np.pi / (np.sqrt(3) * spacing)],
                         [2 * np.pi / (3 * spacing), 2 * np.pi / (np.sqrt(3) * spacing)], [0, 0]],
                        ["Γ", "M", "K", "Γ"])


def lattice_path(lattice, points_per_segment=20):
    """WavevectorPath through the lattice's symmetry points; points_per_segment intervals on the first segment,
    the others with the same k spacing (as build_wavevector_path)."""
    corners = lattice.symmetry_points
    spacing = np.linalg.norm(corners[1] - corners[0]) / points_per_segment
    wavevectors, positions = [], [0.0]
    for start, end in zip(corners[:-1], corners[1:]):
        segment_length = np.linalg.norm(end - start)
        intervals = max(1, round(segment_length / spacing))
        wavevectors.extend(start + fraction * (end - start) for fraction in np.arange(intervals) / intervals)
        positions.append(positions[-1] + segment_length)
    wavevectors.append(corners[-1])
    wavevectors = np.array(wavevectors)
    path_coordinate = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wavevectors, axis=0), axis=1))])
    return WavevectorPath(wavevectors=wavevectors, path_coordinate=path_coordinate,
                          symmetry_point_positions=np.array(positions),
                          symmetry_point_labels=list(lattice.symmetry_point_labels))


def reciprocal_lattice_points(lattice, maximum_magnitude):
    """All G = m b1 + n b2 with |G| <= maximum_magnitude, shape (n_G, 2)."""
    shortest = min(np.linalg.norm(lattice.reciprocal_vectors, axis=1))
    # |m|, |n| bound: generous (the reciprocal cell can be sheared), the circle test does the selection
    bound = int(np.ceil(2 * maximum_magnitude / shortest)) + 1
    indices = np.arange(-bound, bound + 1)
    points = (indices[:, None, None] * lattice.reciprocal_vectors[0]
              + indices[None, :, None] * lattice.reciprocal_vectors[1]).reshape(-1, 2)
    return points[np.linalg.norm(points, axis=1) <= maximum_magnitude]


def attachment_force_factor(angular_frequency, attachment_mass, attachment_stiffness):
    """Z(omega): resonator (dispersion.resonator_force_factor) or, with attachment_stiffness None, a rigid mass."""
    if attachment_stiffness is None:
        return attachment_mass * angular_frequency ** 2
    return resonator_force_factor(attachment_stiffness, attachment_mass, angular_frequency)


def lattice_attachment_frequencies(wavevector, lattice, plate_config, attachment_mass, maximum_frequency_hz,
                                   attachment_stiffness=None, theory="reissner_mindlin",
                                   reciprocal_cutoff=None, points_per_interval=200):
    """Bloch frequencies (Hz, sorted, <= maximum_frequency_hz) of the plate with one point attachment per cell.
    Same root search as dispersion.analytic_resonator_frequencies (scan between the poles, brentq, degenerate
    bare waves added (multiplicity - 1) times, the zero mode at Gamma added by hand).
    reciprocal_cutoff: G_max in rad/m (default: 24.5 * 2 pi / a * sqrt(4 / pi), the circle with the same area
    as the 49 x 49 square |m|, |n| <= 24 of the production square-lattice model, a = |a1|)."""
    constants = plate_constants(plate_config)
    if reciprocal_cutoff is None:
        spacing = np.linalg.norm(lattice.lattice_vectors[0])
        reciprocal_cutoff = 24.5 * 2 * np.pi / spacing * np.sqrt(4 / np.pi)
    folded = np.linalg.norm(np.asarray(wavevector)[None, :]
                            + reciprocal_lattice_points(lattice, reciprocal_cutoff), axis=1)
    cell_area = lattice.cell_area
    bare_frequencies = np.sort([plane_wave_bare_frequency_hz(magnitude, constants, theory) for magnitude in folded])
    bare_frequencies = bare_frequencies[bare_frequencies < 1.2 * maximum_frequency_hz]

    distinct_poles, multiplicities = [], []
    for frequency in bare_frequencies:
        if distinct_poles and np.isclose(frequency, distinct_poles[-1], rtol=1e-9, atol=1e-9):
            multiplicities[-1] += 1
        else:
            distinct_poles.append(frequency)
            multiplicities.append(1)
    poles = list(distinct_poles)
    if attachment_stiffness is not None:
        poles.append(np.sqrt(attachment_stiffness / attachment_mass) / (2 * np.pi))
    poles = np.sort(poles)

    def dispersion_function(frequency_hz):
        angular_frequency = 2 * np.pi * frequency_hz
        with np.errstate(divide="ignore"):
            compliance_sum = plane_wave_point_compliance(folded, angular_frequency, constants, theory).sum()
        return 1.0 - attachment_force_factor(angular_frequency, attachment_mass, attachment_stiffness) \
            / cell_area * compliance_sum

    roots = []
    interval_edges = np.concatenate([[0.0], poles, [1.2 * maximum_frequency_hz]])
    for lower, upper in zip(interval_edges[:-1], interval_edges[1:]):
        if upper - lower < 1e-9:
            continue
        span = upper - lower
        grid = np.linspace(lower + 1e-9 * span, upper - 1e-9 * span, points_per_interval)
        values = np.array([dispersion_function(frequency) for frequency in grid])
        for index in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
            root = brentq(dispersion_function, grid[index], grid[index + 1], xtol=1e-10)
            if abs(dispersion_function(root)) < 1e-6:
                roots.append(root)
    unchanged = [pole for pole, multiplicity in zip(distinct_poles, multiplicities) for _ in range(multiplicity - 1)]
    if distinct_poles[0] < 1e-9:
        unchanged.append(0.0)        # Gamma: plate and attachment translating together
    frequencies = np.sort(np.concatenate([roots, unchanged]))
    return frequencies[frequencies <= maximum_frequency_hz]


def lattice_attachment_dispersion(wavevectors, lattice, plate_config, attachment_mass, maximum_frequency_hz,
                                  **options):
    """lattice_attachment_frequencies for every wavevector, NaN-padded, shape (n_k, n_bands)."""
    rows = [lattice_attachment_frequencies(wavevector, lattice, plate_config, attachment_mass, maximum_frequency_hz,
                                           **options) for wavevector in wavevectors]
    return pad_with_nan(rows, max(len(row) for row in rows))


def irreducible_wedge_grid(lattice, points_per_side=12):
    """Points filling the triangle spanned by the first three path corners (Gamma, M, K or Gamma, X, M),
    as a WavevectorPath without symmetry points (for the zone-interior check)."""
    gamma, first, second = lattice.symmetry_points[:3]
    points = [gamma + (i / points_per_side) * (first - gamma) + (j / points_per_side) * (second - first)
              for i in range(points_per_side + 1) for j in range(i + 1)]
    points = np.array(points)
    return WavevectorPath(wavevectors=points, path_coordinate=np.arange(len(points), dtype=float),
                          symmetry_point_positions=np.array([]), symmetry_point_labels=[])
