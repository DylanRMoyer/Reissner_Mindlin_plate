# Validation: branch tracking of the Bloch bands (stage 5)

Script: `validate_band_tracking.py` (run in `fenicsx-env-complex` from `refactored/`, about 2.5 min). Numbers below
are from that run (2026-11-11). Code: `band_tracking.py`, `dispersion_sweep(..., track_bands=True)`,
`plot_tracked_dispersion`.

## Question
Plain sorting labels the bands by frequency at every k and relabels two branches wherever they truly cross. Does the
tracker follow each physical branch instead, also through avoided crossings, degenerate groups and crossings that
fall exactly on a k-point, and does it flag the steps where it cannot decide?

## Method in short
- Similarity between the bands at neighboring k: MAC with the full eigenvector (plate and resonator DOFs), weighted
  with the reduced mass matrix of the later k (theory notes §13f).
- Degenerate groups (equal frequency within 1e-6 relative or 1 mHz) are compared as subspaces: each group is
  orthonormalised in the mass inner product (Cholesky of its Gram block); the summed MAC between the orthonormal
  bases counts the shared dimensions and does not depend on the arbitrary basis the solver returns.
- Per step the bands are matched by the maximum total quality (assignment problem). Exactly equal scores occur only
  inside degenerate groups; a tie-breaker then keeps the sorted order.
- Where several groups merge at one k-point (a crossing on a k-point, two degenerate pairs meeting at M), the
  k-points on either side are compared directly ("bridge"); each source subspace must send as many labels into each
  later group as they share dimensions.
- A match below quality 0.5 makes the step unreliable: "fallback" to plain sorting for that step if the band count
  is unchanged, otherwise "broken" (unreliable branches end, new ones start). Flagged steps are marked in the plot.

## Checks and results

| Check | Criterion | Result |
|---|---|---|
| A1. Synthetic true crossing between k-points and exactly on a k-point | both branches followed | pass (2 label changes against sorting each) |
| A2. Synthetic avoided crossing (two-level model) | labels stay sorted | pass, lowest quality 0.985 |
| A3. Synthetic crossing through a 3-fold degenerate point | all three branches followed (bridge) | pass |
| A4. Unrelated eigenvectors | step flagged "fallback" | pass |
| B. Bare cell, 20x20 elements, bands up to 9 kHz: each tracked branch follows one analytic plane wave k + G over every window of 5 k-points (0.5 % tolerance) | tracked 0 violations, plain sorting > 0 | tracked 0, plain sorting 12; all 137 steps tracked, lowest quality 0.999 |
| C. Resonator cell (f_r = 1.2 kHz, mass ratio 0.25), production mesh: all steps tracked | no fallback | all 137 steps tracked, lowest quality 0.769 (next to Γ) |
| C. Avoided crossing: band 1 keeps its label along the whole path | label unchanged | pass; plate share 80 % at Γ, 5 % at M (theory session: 80 %, 5 %) |
| C. Degenerate groups at the symmetry points matched | quality above 0.5 next to them | Γ: 3x 4730 Hz, 0.769; X: none, 0.979; M: 3x 2379 Hz, 0.970 |
| C. Label changes against plain sorting below 5 kHz | 0 | 0 |
| D. Coarse path (3 points per segment, 11 wavevectors) | at least one flagged step | 3 of 10 steps "broken" |

Path: Γ–X–M–Γ, 40 points per segment (138 wavevectors) for B and C. Tracking runs 10 % above the plotted range
(5.5 kHz for C, 9.5 kHz for B), so branches enter and leave outside the plot.

## Findings
- Below 5 kHz the folded bands of the bare 0.1 m cell meet only at Γ, X and M: the bisector between two plane waves
  k + G and k + G' meets the path only at the symmetry points for these small G. The first true crossing between
  symmetry points lies at |k + G| = 2.5 π/a (about 7.5 kHz). Check B therefore runs to 9 kHz, on a finer cell so
  the FE bands stay within the tolerance. The true crossings on Γ–X near 7.3 kHz and on M–Γ near 6.5 kHz (the bands
  5/6 noted in the theory session) are followed.
- The resonator cell has no true crossing below 5 kHz either: tracked and sorted curves coincide. Tracking adds the
  continuous line drawing and the plate-share coloring there; it matters for higher frequencies, the bare-cell
  comparison and the parameter study.
- A single-step comparison is not a sharp test of branch identity: next to a crossing both analytic branches are
  within tolerance on both sides of one step. The window of 5 k-points is what makes check B fail for plain sorting.
- Corrections during development: (1) at M two degenerate pairs merge into a 4-fold group; without the bridge the
  labels inside it came out arbitrary (3 violations in B). (2) A first version of the bridge also acted where a group
  only splits off from a larger one (start of the path at Γ) and broke the labels there (57 violations); the bridge
  now acts only where several groups merge. (3) The plain sum of MACs over a group (theory notes §13f, 0.97–1.00) is
  exact only for an orthonormal basis; the Cholesky step makes it exact for any basis the solver returns.

## Limitations
- Inside a group that is degenerate along a whole segment (e.g. pairs on X–M of the bare cell), which of its labels
  continues where is arbitrary; the lines lie on top of each other, so this is not visible.
- Gap detection should keep using the sorted frequencies (`result.frequencies`), not the tracked branches.
- The check plot cycles through 20 colours, in light/dark pairs of one hue; two branches can look alike.
