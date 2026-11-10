# Validation: damping in the Bloch eigenproblem

Script: `validate_bloch_damping.py` (run in `fenicsx-env-complex`, about 5 min). Numbers below are from that run
(2026-11-10). It replaces the earlier dense-only check (`results/damping_check.py`, ~50 min, not in the repo).

## Question
The Bloch eigenproblem is solved undamped (`assemble_unit_cell_matrices` sets the resonator gamma to 0). How much
does hysteretic damping, as used in the forced-response pipeline, move the band frequencies?

Setup: production cell (0.1 m, h = 5 mm steel, 10x10 Lagrange Q2, full 3x3 shear integration), one resonator at the
centre, f_r = 1.2 kHz, mass ratio 0.25, bands below 5 kHz, path Γ–X–M–Γ with 20 points per segment (69 wavevectors).
Damping: plate stiffness K_plate (1 + i γ_plate), resonator spring k (1 + i γ_resonator).

## Theory
Plate damping without a resonator is proportional: K(1 + iγ) has the same mode shapes as K, and every eigenvalue
becomes λ = λ₀(1 + iγ). With √(1 + x) = 1 + x/2 − x²/8 + … and x = iγ (x² = −γ²):

  Re f / f₀ = Re √(1 + iγ) = 1 + γ²/8 + O(γ⁴).

With a resonator, damping only the plate or only the spring is no longer proportional. Writing the damped
stiffness as K + iD (D = γ_plate K_plate + γ_resonator K_spring) and using the undamped modes ψ_n (M_r-normalized),
second-order perturbation theory gives

  λ_n ≈ λ₀ₙ + i D_nn − Σ_{m≠n} |D_mn|² / (λ₀ₙ − λ₀ₘ).

The first-order term is purely imaginary (attenuation); the real shift is of order γ². The second-order sum is
large when a mode is a mixture of plate and resonator motion, because then D couples it strongly to its hybridized
partner. For the lower partner (λ₀ₙ < λ₀ₘ) the term is positive, so the frequency rises.

## Checks and results

| Check | Criterion | Result |
|---|---|---|
| 1. Split K_plate + K_spring (1 + iγ) vs the pipeline's `augment_stiffness_matrix` | < 1e-12 | 3.4e-20 |
| 2. Bare cell, γ_plate = 0.04: Re λ/λ₀ − 1, Im λ/(γλ₀) − 1, shift − (Re √(1+iγ) − 1) | < 1e-8 | ≤ 8.4e-11, 1.5e-9, 4.3e-11 |
| 3. Sparse shift-invert (ARPACK) vs dense non-Hermitian solve, two wavevectors | < 1e-8 | 5.3e-11, 3.8e-11 |
| 4. Largest shift over the path, every case | < 1 % (mesh tolerance) | ≤ 0.205 % |
| 5. γ² scaling: median per-mode ratio (expected 4 and 25) | within 2 % | 3.999 and 24.987 |
| 6. Second-order perturbation theory vs solve at each worst wavevector | within 5 % | ≤ 0.18 % |

Exact value for γ = 0.04: Re √(1 + 0.04i) − 1 = 1.9990e-4 against γ²/8 = 2.0000e-4.

Largest shift of Re f (all positive):

| γ_plate | γ_resonator | Largest shift | γ²/8 | Where | Plate share of that mode |
|---|---|---|---|---|---|
| 0.02 | 0 | 0.0087 % | 0.0050 % | band 1, 910 Hz, k = (0.714, 0.714) π/a | 29 % |
| 0.04 | 0 | 0.0349 % | 0.0200 % | band 1, 910 Hz, k = (0.714, 0.714) π/a | 29 % |
| 0 | 0.02 | 0.0082 % | 0.0050 % | band 1, 949 Hz, k = (0.750, 0.750) π/a | 21 % |
| 0 | 0.10 | 0.2047 % | 0.1250 % | band 1, 949 Hz, k = (0.750, 0.750) π/a | 21 % |
| 0.04 | 0.02 | 0.0207 % | 0.0200 % | band 1, 801 Hz, k = (0.643, 0.643) π/a | 48 % |

At each worst point the dominant term of the second-order sum comes from band 2 at the same wavevector (550–590 Hz
higher), i.e. the hybridized partner of the plate–resonator avoided crossing on M–Γ. That is why the shift exceeds
γ²/8 there: the extra term comes from the non-proportional damping of two mixed modes, not from a numerical effect.
With plate and resonator damped at once, the coupling partly cancels: since ψ_mᴴ K ψ_n = 0 for m ≠ n, the
off-diagonal entries are D_mn = (γ_plate − γ_resonator) ψ_mᴴ K_plate ψ_n, so only the difference of the two loss
factors couples the modes (equal loss factors would be proportional damping, with shift exactly γ²/8). For 0.04 and
0.02 the worst shift stays at γ²/8 of the larger loss factor.

The per-mode ratio for resonator γ 0.10 / 0.02 ranges from 0.88 to 28.2 around the median 25; the outliers are
modes whose shift is close to zero (the second-order term and γ²s²/8 nearly cancel), where higher orders dominate
the ratio. Plate γ 0.04 / 0.02 ranges 3.99–4.12.

## Conclusion
For the damping used in the pipeline (γ_plate = 0.04, γ_resonator = 0.02) the band frequencies move by at most
0.021 %, fifty times below the 1 % mesh tolerance; even γ_resonator = 0.10 gives 0.20 %. The shift scales with γ²,
is reproduced by second-order perturbation theory, and is largest on hybridized plate–resonator modes. Undamped Bloch
bands are therefore sufficient for band positions. Not covered: the imaginary parts (attenuation, depth of a gap
with damping), which need the k(ω) formulation.
