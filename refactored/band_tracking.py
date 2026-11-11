import numpy as np
from dataclasses import dataclass
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import linear_sum_assignment

# --- Branch tracking of Bloch bands by mass-weighted eigenvector overlap (MAC) between neighbouring wavevectors ---
#
# Plain sorting labels the bands by frequency at every k. That relabels two branches wherever they truly cross.
# Tracking instead matches the bands at each k to those at the previous k by eigenvector similarity:
#   MAC(a, b) = |a^H M b|^2 / ((a^H M a) (b^H M b)),  M = reduced mass matrix incl. the resonator DOFs.
# Degenerate groups (equal frequencies, arbitrary basis) are compared as subspaces. A crossing that falls exactly on
# a k-point, or several degenerate groups meeting at a symmetry point, is a degenerate group there; it is
# bridged by comparing the k-points on either side directly.


@dataclass
class BandOverlap:
    """Mass-weighted inner products of the bands at two wavevectors (left = earlier, right = later on the path),
    all taken with the reduced mass matrix of the later wavevector."""
    cross: np.ndarray           # (n_left, n_right): left^H M right
    left_gram: np.ndarray       # (n_left, n_left):  left^H M left
    right_gram: np.ndarray      # (n_right, n_right): right^H M right


@dataclass
class TrackedBands:
    labels: np.ndarray          # (n_k, n_bands) int: branch label of each frequency-sorted band, -1 for NaN padding
    frequencies: np.ndarray     # (n_k, n_labels) Hz, one column per branch, NaN where the branch is absent
    plate_share: np.ndarray     # (n_k, n_labels) plate part of the kinetic energy, 1 for a bare cell, NaN if absent
    match_quality: np.ndarray   # (n_k, n_labels) quality of the match that continued the branch at this k (NaN: new)
    step_status: list           # per step k[j-1] -> k[j]: "tracked", "fallback" (plain sorting) or "broken"


def band_overlap(left_vectors, right_vectors, mass_reduced):
    """Inner products for the MAC between the bands of two wavevectors (vectors as columns)."""
    mass_times_right = mass_reduced @ right_vectors
    mass_times_left = mass_reduced @ left_vectors
    return BandOverlap(cross=left_vectors.conj().T @ mass_times_right,
                       left_gram=left_vectors.conj().T @ mass_times_left,
                       right_gram=right_vectors.conj().T @ mass_times_right)


def mac_matrix(overlap):
    """Plain band-by-band MAC, shape (n_left, n_right), values 0..1."""
    left_norms = np.real(np.diag(overlap.left_gram))
    right_norms = np.real(np.diag(overlap.right_gram))
    return np.abs(overlap.cross) ** 2 / np.outer(left_norms, right_norms)


def resonator_energy_share(vectors, mass_reduced, number_of_resonators):
    """Resonator part of the kinetic energy of each band (columns of vectors): sum m |u|^2 / (psi^H M psi).
    The resonator DOFs are the last number_of_resonators entries; their mass entries are diagonal."""
    if number_of_resonators == 0:
        return np.zeros(vectors.shape[1])
    resonator_masses = np.real(mass_reduced.diagonal()[-number_of_resonators:])
    resonator_energy = (resonator_masses[:, None] * np.abs(vectors[-number_of_resonators:]) ** 2).sum(axis=0)
    total_energy = np.real(np.sum(vectors.conj() * (mass_reduced @ vectors), axis=0))
    return resonator_energy / total_energy


def degenerate_groups(frequencies, relative_tolerance=1e-6, absolute_tolerance_hz=1e-3):
    """Split ascending frequencies into groups of (numerically) equal values. Returns a list of index lists;
    a non-degenerate band is a group of one."""
    groups = [[0]] if len(frequencies) else []
    for band_index in range(1, len(frequencies)):
        gap = frequencies[band_index] - frequencies[band_index - 1]
        if gap <= max(absolute_tolerance_hz, relative_tolerance * abs(frequencies[band_index])):
            groups[-1].append(band_index)
        else:
            groups.append([band_index])
    return groups


def subspace_quality(overlap, left_bands, right_bands):
    """Overlap of the subspaces spanned by two sets of bands, normalised to 0..1.

    Each set is orthonormalised in the mass inner product (Cholesky of its Gram block); the squared Frobenius norm
    of the orthonormal overlap is the sum of the MACs over the group (between 0 and the smaller set size) and is
    divided by the smaller set size. For two single bands this is the plain MAC. For a degenerate group against
    one band it is the share of that band lying in the group's subspace, independent of the arbitrary basis."""
    cross = overlap.cross[np.ix_(left_bands, right_bands)]
    lower_left = cholesky(overlap.left_gram[np.ix_(left_bands, left_bands)], lower=True)
    lower_right = cholesky(overlap.right_gram[np.ix_(right_bands, right_bands)], lower=True)
    orthonormal_cross = solve_triangular(lower_left, cross, lower=True)
    orthonormal_cross = solve_triangular(lower_right, orthonormal_cross.conj().T, lower=True).conj().T
    return float(np.sum(np.abs(orthonormal_cross) ** 2)) / min(len(left_bands), len(right_bands))


def quality_matrix(overlap, left_groups, right_groups):
    """Band-by-band match quality, shape (n_left, n_right); every member pair of two groups gets the group value."""
    number_left = sum(len(group) for group in left_groups)
    number_right = sum(len(group) for group in right_groups)
    quality = np.zeros((number_left, number_right))
    for left_group in left_groups:
        for right_group in right_groups:
            quality[np.ix_(left_group, right_group)] = subspace_quality(overlap, left_group, right_group)
    return quality


def subspace_overlap(overlap, left_bands, right_bands):
    """Number of shared dimensions of two band subspaces (0 .. smaller set size): the squared Frobenius norm of
    their overlap after orthonormalisation, i.e. subspace_quality without the normalisation."""
    return subspace_quality(overlap, left_bands, right_bands) * min(len(left_bands), len(right_bands))


def assign_bands(quality):
    """Optimal one-to-one matching (maximum total quality). Exactly equal scores only occur inside degenerate
    groups, where the basis is arbitrary and any order is equally right; a tiny penalty on the distance between
    sorted positions then picks the sorted order, so tracking adds no label changes where it has no information.
    The penalty (1e-9 per position) is far below any real quality difference."""
    left_positions, right_positions = np.indices(quality.shape)
    return linear_sum_assignment(quality - 1e-9 * np.abs(left_positions - right_positions), maximize=True)


def bridge_degenerate_groups(quality, labels, step, groups, bridge_overlap, minimum_quality):
    """Carry labels through a degenerate point at k[step-1] by comparing k[step-2] directly with k[step].

    Used for a group at k[step-1] whose bands were not one and the same group at k[step-2] (a crossing on a
    k-point, or several degenerate groups meeting at a symmetry point). The bands at k[step-2] that feed it are
    taken as source subspaces S (their own degenerate groups). A source subspace S and a group h at k[step] share
    overlap(S, h) dimensions; the assignment must send exactly that many labels of S into h. Returns
    (rows, columns, recorded quality per row) or None if there is nothing to bridge or the result is inconsistent.
    """
    previous_groups, groups_two_back, current_groups = groups[step - 1], groups[step - 2], groups[step]
    source_band_of_row = {}
    for row in range(len(labels[step - 1])):
        if labels[step - 1, row] < 0:
            continue
        source = np.flatnonzero(labels[step - 2] == labels[step - 1, row])
        if len(source) == 1:
            source_band_of_row[row] = int(source[0])
    group_two_back_of_band = {band: tuple(group) for group in groups_two_back for band in group}

    involved_sources = set()
    for group in previous_groups:
        if len(group) == 1:
            continue
        if any(row not in source_band_of_row for row in group):
            continue                                    # a branch entered here: nothing to bridge from
        sources = {group_two_back_of_band[source_band_of_row[row]] for row in group}
        if len(sources) == 1:
            continue                                    # all from one group (persisting or splitting off): no crossing
        involved_sources |= sources
    if not involved_sources:
        return None

    rows_of_source = {source: [row for row, band in source_band_of_row.items()
                               if group_two_back_of_band[band] == source] for source in involved_sources}
    if any(len(rows) != len(source) for source, rows in rows_of_source.items()):
        return None                                     # part of a source subspace left the range
    shared = {(source, tuple(group)): subspace_overlap(bridge_overlap, list(source), group)
              for source in involved_sources for group in current_groups}
    bridged_quality = quality.copy()
    for source, rows in rows_of_source.items():
        for row in rows:
            for group in current_groups:
                bridged_quality[row, group] = shared[(source, tuple(group))] / len(group)
    rows, columns = assign_bands(bridged_quality)
    assigned_column = dict(zip(rows, columns))
    group_of_column = {column: tuple(group) for group in current_groups for column in group}

    recorded_quality = {}
    for source, source_rows in rows_of_source.items():
        if any(row not in assigned_column for row in source_rows):
            return None
        target_groups = [group_of_column[assigned_column[row]] for row in source_rows]
        for group in current_groups:
            if abs(target_groups.count(tuple(group)) - shared[(source, tuple(group))]) >= 0.5:
                return None                             # labels do not follow the shared dimensions
        source_quality = sum(shared[(source, group)] for group in set(target_groups)) / len(source)
        if source_quality < minimum_quality:
            return None
        for row in source_rows:
            recorded_quality[row] = source_quality
    recorded = np.array([recorded_quality.get(row, quality[row, column]) for row, column in zip(rows, columns)])
    return rows, columns, recorded


def track_bands(frequencies, overlaps, bridge_overlaps=None, resonator_share=None, minimum_quality=0.5,
                relative_tolerance=1e-6, absolute_tolerance_hz=1e-3):
    """Follow each branch along the path by eigenvector similarity.

    frequencies: (n_k, n_bands) Hz, ascending per k, NaN-padded (DispersionResult.frequencies).
    overlaps: list of n_k - 1 BandOverlap, overlaps[j - 1] between k[j - 1] and k[j].
    bridge_overlaps: optional list of n_k - 2 BandOverlap, bridge_overlaps[j - 2] between k[j - 2] and k[j];
        used to carry labels through a crossing that falls exactly on a k-point.
    resonator_share: optional (n_k, n_bands) resonator energy share of the sorted bands (0 for a bare cell).
    minimum_quality: a match below this counts as unreliable.

    Per step the bands are matched by maximum total quality (assignment problem, not greedy). If any match is
    unreliable: with equal band counts the step falls back to plain sorting ("fallback"); otherwise the reliable
    matches are kept and the unreliable ones end their branch and start a new one ("broken"). Bands entering at
    the top of the frequency range start new branches, bands leaving end theirs."""
    number_of_wavevectors, width = frequencies.shape
    counts = [int(np.count_nonzero(~np.isnan(row))) for row in frequencies]
    labels = np.full((number_of_wavevectors, width), -1, dtype=int)
    quality_of_band = np.full((number_of_wavevectors, width), np.nan)
    labels[0, :counts[0]] = np.arange(counts[0])
    next_label = counts[0]
    step_status = []
    groups = [degenerate_groups(row[:count], relative_tolerance, absolute_tolerance_hz)
              for row, count in zip(frequencies, counts)]

    for step in range(1, number_of_wavevectors):
        previous_count, current_count = counts[step - 1], counts[step]
        quality = quality_matrix(overlaps[step - 1], groups[step - 1], groups[step])
        rows, columns = assign_bands(quality)

        matched_quality = quality[rows, columns]
        if bridge_overlaps is not None and step >= 2:
            bridged = bridge_degenerate_groups(quality, labels, step, groups, bridge_overlaps[step - 2],
                                               minimum_quality)
            if bridged is not None:
                rows, columns, matched_quality = bridged

        unreliable = matched_quality < minimum_quality
        if not unreliable.any():
            status = "tracked"
        elif previous_count == current_count:
            status = "fallback"
            rows = columns = np.arange(current_count)
            matched_quality = quality[rows, columns]
            unreliable = np.zeros(current_count, dtype=bool)
        else:
            status = "broken"
        step_status.append(status)

        for row_index, column_index, match_quality, is_unreliable in zip(rows, columns, matched_quality, unreliable):
            if is_unreliable:
                continue                                # left branch ends; right band gets a new label below
            labels[step, column_index] = labels[step - 1, row_index]
            quality_of_band[step, column_index] = match_quality
        for column_index in range(current_count):
            if labels[step, column_index] < 0:
                labels[step, column_index] = next_label
                next_label += 1

    tracked_frequencies = np.full((number_of_wavevectors, next_label), np.nan)
    plate_share = np.full((number_of_wavevectors, next_label), np.nan)
    match_quality = np.full((number_of_wavevectors, next_label), np.nan)
    for step in range(number_of_wavevectors):
        present = labels[step, :counts[step]]
        tracked_frequencies[step, present] = frequencies[step, :counts[step]]
        plate_share[step, present] = 1.0 - (resonator_share[step, :counts[step]] if resonator_share is not None
                                            else 0.0)
        match_quality[step, present] = quality_of_band[step, :counts[step]]
    return TrackedBands(labels=labels, frequencies=tracked_frequencies, plate_share=plate_share,
                        match_quality=match_quality, step_status=step_status)


def relabelled_bands(tracked):
    """Number of (step, sorted band) pairs where tracking assigns a different branch than at the same sorted
    position one step earlier, ignoring bands that only entered or left. Zero means tracking and plain
    sorting give the same curves."""
    count = 0
    for step in range(1, tracked.labels.shape[0]):
        previous, current = tracked.labels[step - 1], tracked.labels[step]
        both_present = (previous >= 0) & (current >= 0)
        existed_before = np.isin(current, previous[previous >= 0])
        count += int(np.count_nonzero(both_present & existed_before & (previous != current)))
    return count
