# Theory notes: dispersion diagrams of a plate with point resonators

Reference for the dispersion branch (`dispersion_diagram`). Collected from the explanations in sessions 2–3
(Nov 2026) and extended by the theory session of 2026-11-09/10 (§13; full tutorial-style derivations in
`theory_notes_tex/chapter_01`–`08`). Numbers refer to the production configuration unless stated otherwise:
steel, h = 5 mm, unit cell a = 0.1 m (square), bands below 5 kHz, Reissner–Mindlin (RM) plate, Lagrange Q2,
100 elements per metre. Plate data: ρh ≈ 39.3 kg/m², D ≈ 2.3 kN·m, plate mass per cell ρh·a² ≈ 0.39 kg.
Statements marked *(estimate)* were not measured; everything else was computed in the validation scripts.

---

## 1. Bloch's theorem: why one cell is enough

A periodic structure repeats along lattice vectors **a₁** = (Lx, 0), **a₂** = (0, Ly). Every free wave has the form

  u(**x** + **R**) = e^{i **k**·**R**} u(**x**),  **R** = m·**a₁** + n·**a₂**.

Moving one cell over does not change the wave's shape, only multiplies it by a phase. So one cell with
"right edge = left edge · e^{i kx Lx}" and "top edge = bottom edge · e^{i ky Ly}" describes the infinite
structure. For each wavevector **k** this gives an eigenproblem; the curves ω(**k**) are the **bands**
(dispersion curves). A frequency range without any band for any **k** is a **complete band gap**.

Equivalent form: u(**x**) = e^{i**k**·**x**} p(**x**) with p periodic in the cell.

## 2. Brillouin zone and the points Γ, X, M, Y

A phase is only defined modulo 2π, so **k** and **k + G** describe the same wave, with the
**reciprocal lattice vectors** **G** = 2π(m/Lx, n/Ly). All distinct wavevectors lie in the
**first Brillouin zone** −π/Lx ≤ kx ≤ π/Lx, −π/Ly ≤ ky ≤ π/Ly. Reciprocity (ω(k) = ω(−k)) and the
cell's mirror symmetries shrink this to the **irreducible Brillouin zone**; band diagrams follow its edge.

| Point | k | Phase per cell | Picture |
|---|---|---|---|
| Γ | (0, 0) | 1, 1 | all cells move identically (infinite wavelength) |
| X | (π/Lx, 0) | −1 in x | neighbours in x move oppositely (λ = 2Lx along x) |
| M | (π/Lx, π/Ly) | −1, −1 | checkerboard |
| Y | (0, π/Ly) | −1 in y | only needed for rectangular cells |

Path: Γ–X–M–Γ for a square cell, Γ–X–M–Y–Γ for a rectangle (`build_wavevector_path` chooses automatically).

## 3. Band folding

**Uniform plate, no cells:** plane waves e^{iκx}, one frequency per physical wavenumber κ. Thin-plate limit:
f = κ²·√(D/ρh)/2π. For the 5 mm plate f ≈ 1.215·κ² Hz (κ in rad/m).

**Same plate described with cells of size a:** the Bloch wavenumber only sees κ modulo 2π/a. All waves
κ = k + 2πm/a have the same phase step per cell, so at one k the cell has a whole list of frequencies
f_m(k) = f(|k + 2πm/a|). Sorted, these are band 1, 2, 3, … Picture: cut the parabola f(κ) into strips of width
2π/a along the **k-axis** and slide each strip **sideways** (frequency unchanged) into the first zone. Strips from
the positive and negative κ side overlap, so bands meet in pairs at Γ and at the zone edge.

Nothing physical changes: folding is a relabelling. The band index m counts extra wiggles of the wave inside
one cell (p(x) = e^{i2πmx/a}).

**1D numbers (a = 1 m, 20 mm plate, sessions 2):** Γ: 0, 192 Hz (×2), 768 Hz (×2); X: 48 Hz (×2), 432 Hz (×2).

**2D:** at fixed k, only the discrete grid of points k + G counts. Infinitely many directions have |κ| = 2π/a in
the continuous plate, but at fixed k only the grid points exactly on that circle appear (four at Γ:
(±2π/a, 0), (0, ±2π/a) → 4-fold degeneracy). The other directions appear at other k (each plane wave of the
infinite plate appears exactly once in the diagram, at the k it folds to). Square 1 m cell, 20 mm: Γ 192 Hz ×4,
X 48 Hz ×2, M 96 Hz ×4 (FE reproduced all).

**Bare 0.1 m cell, 5 mm:** first meeting at X ≈ 1.19 kHz (λ = 2a), at M ≈ 2.38 kHz, 4-fold group at Γ
≈ 4.73 kHz. Only 2–5 bands below 5 kHz.

**Band count below f_max:** number of G inside the circle |k + G| ≤ κ_max:
  N ≈ π κ_max² / ((2π)²/(Lx Ly)) = κ_max² · Lx Ly / (4π).
Checked: 1.1 × 1.0 m cell, 160 Hz → 11.5 predicted, 10 observed. At 5 kHz: 1.1 m cell ≈ 360 bands,
0.1 m cell ≈ 3–4 (observed 2–5).

**Why kinks appear at Γ, X, M** (three overlapping effects):
1. The path turns a corner there; the band surface f(kx, ky) is smooth, its cut along a bent path is not.
2. Symmetry forces bands to meet at the zone edges: at X, k and k − G = (−π/a, 0) have equal |κ|. In 1D all
   crossings sit at Γ or the zone edge (|k + 2πm/a| = |k + 2πn/a| only for k = 0 or ±π/a).
3. In 2D, crossings also occur where |k + G₁| = |k + G₂| (perpendicular bisectors in k-space). For low-order G
   these pass through symmetry points; for higher G they cut segments at ordinary points (seen on the 1.1 m
   cell near 67 Hz on Γ–X).

Plain sorting by frequency (stage 3) draws kinks wherever branches cross; branch tracking (stage 5) fixes that.

## 4. FE implementation of Bloch's theorem (stages 1–2)

- DOFs categorised: interior, edges L/R/B/T, corners LL/LR/UL/UR. Slaves (R, T, LR, UL, UR) are tied to masters
  one lattice vector away: R ← L · e^{i kx Lx}, T ← B · e^{i ky Ly}, LR/UL/UR ← LL with e^{i kx Lx},
  e^{i ky Ly}, e^{i(kx Lx + ky Ly)}.
- Reduction matrix T(k): q_full = T q_indep (identity on independent DOFs, one phase per slave row).
- K q = ω² M q, multiplied by Tᴴ: K_r(k) = Tᴴ K T, M_r(k) = Tᴴ M T, complex Hermitian → real ω².
  Needs complex PETSc (`fenicsx-env-complex`).
- Checks: Hermiticity, one zero mode at Γ (rigid translation), none at generic k, ω(k) = ω(−k).

## 5. Mesh resolution and computational cost

**What sets the error** (not the cell size):
- elements per bending wavelength at f_max (resolution of the wave shape), and
- element size / plate thickness (shear locking under full integration).
"Elements per metre" is just 1/element size; it only transfers between configurations with the same thickness
and f_max.

| Case | λ at top band | elements/λ | element/h | error |
|---|---|---|---|---|
| 20 mm, 16/m, ~960 Hz | 0.45 m | 7.2 | 3.1 | 0.96 % |
| 5 mm, 16/m, ~160 Hz | 0.55 m | 8.7 | 12.5 | 1.87 % |
| 5 mm, 100/m, 5 kHz | ~0.10 m | ~10 | 2.0 | 0.33 % |

Q2 frequency error ∝ (element size · κ)⁴ asymptotically; with locking the observed order was ~3.5.

**Why a smaller cell is cheaper:**
1. nx = cell length / required element size, and the required element size is fixed by physics (wavelength,
   thickness), not by the cell. The 0.1 m cell holds ~1 wavelength at 5 kHz (10 elements per side),
   the 1.1 m cell ~11 (≈ 110 per side). DOFs N ∝ density² · cell area.
2. The number of bands below f_max is ∝ cell area (formula in §3): ~4 vs ~360. Eigensolver cost grows with the
   number of bands (back-solves ∝ ncv, orthogonalization ∝ N·ncv², projected problem ∝ ncv³).
A smaller cell does not resolve better, it resolves equally well with fewer elements.

Other cost factors: f_max (density ∝ √f_max, N ∝ f_max, band count ∝ f_max), thickness (locking, κ ∝ h^{-1/2}),
accuracy (halving the error needs only 2^{1/4} ≈ 1.19× finer elements), number of k-points (linear),
sparse LU ~N^1.5 (fixed overhead ~0.1–0.2 s per k at small N), PETSc LU vs MUMPS (MUMPS only wins from ~12k DOFs),
k-points are independent (parallelisable).

**Measured (0.1 m cell, 5 mm, < 5 kHz):** 80/m 0.72 %, 100/m 0.33 %, 120/m 0.17 %; 0.2 s per k at 100/m.

## 6. The hourglass mode (why the Bloch cell uses 3×3 shear integration)

Reduced 2×2 integration of the shear energy avoids shear locking but lets patterns whose shear strain vanishes
at the four Gauss points cost no energy. For Q2: w = (ξ² − 1/3)(η² − 1/3) per element, θ = 0, continuous across
elements, repeating every element → fits any periodic cell → spurious zero mode at Γ. Away from Γ its energy
grows ∝ k², so it forms a branch rising linearly (~7148 Hz · |k|L/π, mesh-independent) through the spectrum.
In folding terms: a very short wave that the quadrature wrongly moved from very high frequency to 0 Hz.
Full 3×3 integration removes it (costs mild locking, absorbed by the 100/m mesh).
The forced-response pipeline still uses 2×2 (Serendipity); whether it has a similar mode is an open question.

## 7. Band gap mechanisms

### 7a. Bragg gaps — reflections add up
*Everyday example:* clapping in front of a long picket fence gives a chirped echo; soap-bubble colours.
Each resonator reflects a little of a passing wave. If 2a = λ, all reflections return in phase and add up; after
a few dozen cells the wave is fully reflected. Off this condition the phases spread and cancel.
For the 0.1 m lattice: λ = 0.2 m at ≈ 1.2 kHz, exactly the meeting point at X.

Standing-wave picture at X: the right- and left-going waves form
- pattern A: antinodes at the resonators, w ∝ cos(πx/a) — feels the resonators fully, frequency shifts;
- pattern B: nodes at the resonators, w ∝ sin(πx/a) — does not see them, frequency unchanged.
The degenerate pair splits; between them no wave exists. Coupling only matters at crossings: the resonators
couple waves differing by a G; away from a crossing those waves have very different frequencies and a weak
coupling barely shifts them (degenerate perturbation theory, avoided crossing).

*(estimate, 1D, rigid point mass m):* pattern A frequency × 1/√(1 + 2m/(ρh a²)); m = 39 g → 1190 → ~1090 Hz,
gap ~9 %. Limitation in 2D: the Bragg condition is at X ≈ 1.2 kHz but at M ≈ 2.4 kHz, so directional gaps
rarely overlap into a complete gap with modest masses.

### 7b. Local-resonance gaps — every resonator is a vibration absorber
*Everyday example:* the tuned mass damper in Taipei 101; Den Hartog's dynamic vibration absorber: a tuned
mass-spring keeps its attachment point almost still at f_r. One every 10 cm "nails down" the plate near f_r,
independent of the wavelength.

**Effective mass.** Resonator (m, k, ω_r² = k/m) on a base moving as w:
  −m ω² u = k (w − u) ⇒ u = w ω_r²/(ω_r² − ω²);
force on the base k(u − w) = m ω² u, the same as a rigid mass
  **m_eff(ω) = m ω_r² / (ω_r² − ω²)**
(≈ m far below f_r, → +∞ just below, large negative just above: antiphase, pulls the plate back).

**Smeared over the plate** (wavelength ≫ a): ρh → ρh + m_eff/a² = ρh [1 + μ ω_r²/(ω_r² − ω²)],
mass ratio **μ = m/(ρh a²)** (resonator mass per cell over plate mass per cell; with one resonator per cell
= total resonator mass / total plate mass). Bending wave:
  D κ⁴ = ω² ρh · [ω_r²(1 + μ) − ω²] / (ω_r² − ω²).
κ⁴ < 0 (no propagating wave) for **f_r < f < f_r √(1 + μ)**.
- Upper edge = Γ (κ = 0): one cell moves as a rigid mass ρh a² against the resonator; two-mass system
  ω² = k(1/m + 1/M) = ω_r²(1 + μ).
- Lower edge: the lowest branch flattens towards f_r for short waves.
Gap width grows with μ (μ = 0.1 → ~5 %, 0.5 → ~22 %, 1 → ~41 % for f_r = 2 kHz) *(effective-medium values)*.
In the band diagram: flat resonator band at f_r, avoided crossing with the plate band, gap for every k.

**Validity:** this is the plane-wave model keeping only G = 0. Holds only well below the first Bragg frequency.

### 7c. Comparison

| | Bragg | Local resonance |
|---|---|---|
| Cause | reflections from the lattice add in phase | resonators pin the plate (negative effective mass) |
| Location | set by spacing (λ ≈ 2a) | set by tuning (f_r … f_r√(1+μ), only for f_r well below the Bragg frequency, see §13d) |
| Repelling branches | two plate branches at a fold | plate branch and flat resonator branch |
| Direction | different per direction | same in all directions (effective-medium limit only; near a Bragg frequency it is direction dependent, §13g) |
| Wavelength | must match spacing | works for λ ≫ a (subwavelength) |

Correction made in session 3: "every crossing is a potential gap" only holds for Bragg gaps.
Tuning f_r near a Bragg frequency couples the mechanisms (candidate for the stage 7 parameter study).

## 8. Resonators in the FE Bloch problem (stage 4)

- One extra DOF q per resonator; spring couples q to w(x₀) = Σ φᵢ qᵢ (FE basis weights at the attachment).
  K_aug = plate K + k (φ, −1)(φ, −1)ᵀ, M_aug = plate M + m on q's diagonal (existing `augment_*` functions).
- Resonator DOFs are inside the cell, always independent: T_aug(k) = blockdiag(T(k), I).
- **Augment first vs reduce first:** equivalent only if the coupling vector is transformed with T
  (reduced coupling φᵀT carries the phases when the resonator touches boundary DOFs). Naively adding the spring
  at the same DOF numbers after reduction is wrong on edges/corners. Augment-first gets the phases from T for free.
- **Position invariance:** moving the resonator in the cell only shifts the lattice origin → identical bands
  (nodes incl. slave corner/edge DOFs agree to ≤ 1.8e-9; a point inside an element differs by 0.25 %).
- **Limits:** mass → 0 (f_r → ∞, the massless DOF follows the plate) and spring → 0 (resonator decouples,
  flat band at f_r) recover the bare bands, deviation ∝ the small parameter. Mass limit deviation ≈ 2μ, matching
  the antinode estimate of §7a.

## 9. Plane-wave expansion (PWE): analytic reference with resonators

1. **Resonator → force.** Force on the plate per unit plate deflection at the attachment:
   Z(ω) = k m ω²/(k − m ω²) = m_eff ω².
2. **Plate response to a lattice of point forces with Bloch phase:** only plane waves k + G (folding again).
   Amplitude of each: W_G = (f/A)·g(|k + G|, ω), with the plate's compliance per plane wave
   g = 1/(D κ⁴ − ρh ω²) (Kirchhoff) or the w-entry of (K(κ) − ω² M)⁻¹ of the RM 2×2 system.
   Division by the cell area A: one point force per cell as a Fourier series.
3. **Self-consistency:** w₀ = Σ_G W_G ⇒ **1 = (Z(ω)/A) · Σ_G g(|k + G|, ω)**
   ("the resonator's force produces exactly the deflection that produces that force").

Built-in checks: G = 0 alone gives ω = ω_r√(1 + μ) at Γ (effective-medium upper edge); Z → 0 gives the bare
bands; the resonator position drops out (|e^{i(k+G)·r₀}|² = 1).

Root finding: h(ω) = 1 − (Z/A)Σg has poles at all bare frequencies and at f_r; between neighbouring poles it
swings from ±∞ to ∓∞, so typically one root per interval → **new bands interlace with the old ones**
(mathematical form of the avoided crossings). Degenerate bare frequencies with multiplicity d keep d − 1
unchanged modes (nodes at the resonator, pattern B of §7a); the Γ zero mode (plate and resonator translating
together) is added by hand because the derivation divided by ω².

**Kirchhoff vs RM:** Kirchhoff g ~ 1/(Dκ⁴) → the sum converges. RM g ~ 1/(Fκ²) for short waves → in 2D the
sum grows like ln(κ_max): a point force on a shear-deformable plate gives a logarithmically singular dent.
`harmonic_cutoff` represents the attachment size (12 harmonics ≈ a/12 ≈ 8 mm); the FE mesh regularises at
element size, hence a weak mesh drift of resonator bands (1273.5 / 1271.7 / 1270.5 Hz for 10/20/40 elements).
Local plate compliance ~1e-9 m/N vs spring compliance ~1.8e-7 m/N *(estimate)* → sub-percent effect.
Measured: FE vs RM-PWE 0.33 % (plate discretisation), FE vs Kirchhoff-PWE 1.6 % (missing shear and rotary
inertia near 5 kHz). RM is the right reference.

## 10. Damping (why the Bloch problem is undamped)

Hysteretic convention K(1 + iγ).
- **Plate damping alone (proportional):** same mode shapes, λ = ω₀²(1 + iγ) exactly, ω = ω₀√(1 + iγ),
  Re ω ≈ ω₀(1 + γ²/8). (Not ω₀√(1 + γ²): that is the shift of |λ|.) γ = 0.04 → 0.02 %.
  For hysteretic damping the forced-response peak sits exactly at the undamped ω₀.
- **Resonator damping (non-proportional):** first-order change of λ purely imaginary (undamped problem is
  Hermitian), real shift O(γ²); can be larger near avoided crossings (second-order term divided by the frequency
  difference).
- **Measured** (dense non-Hermitian solve, 0.1 m cell, f_r 1.2 kHz, μ 0.25, whole path, bands < 5 kHz):

| plate γ | resonator γ | max shift Re f | γ²/8 |
|---|---|---|---|
| 0.04 | 0 | 0.035 % | 0.020 % |
| 0 | 0.02 | 0.008 % | 0.005 % |
| 0.04 | 0.02 | 0.021 % | 0.020 % |
| 0 | 0.10 | 0.20 % | 0.13 % |

  γ² scaling confirmed (5× γ → 25× shift); worst case at a hybridised band (925 Hz on M–Γ).
- Not covered: the imaginary parts (attenuation). With damping, a gap becomes a region of strong but finite
  attenuation; quantifying gap depth needs the k(ω) formulation (ω given, complex k solved for). Optional, later.
  The undamped in-gap decay was computed with the Kirchhoff plane-wave model in the theory session (§13g);
  the FE version with damping is still open.

## 11. First result: why the 1.2 kHz gap sits below f_r

f_r = 1.2 kHz, μ = 0.25 (m ≈ 98 g, k ≈ 5.6·10⁶ N/m): complete gap **1031–1195 Hz (14.7 %)**.
Lower edge = maximum of band 1 at M, upper edge = minimum of band 2 at X. The lowest plate branch flattens
towards f_r (836 Hz at X instead of 1190, 1031 Hz at M instead of 2379); the resonator branch is nearly flat
at ~1270 Hz near Γ. The effective-medium estimate (1200–1342 Hz) fails because f_r coincides with the Bragg
fold at X (~1.19 kHz): G ≠ 0 terms matter and pull the edges down (both mechanisms interact).
Diagram: `refactored/results/dispersion_vamm_1200.png`.

## 12. Topics to deepen in the theory session

All covered on 2026-11-09/10 (results in §13, derivations in `theory_notes_tex/`):
- Why the lower gap edge lies at M and the upper at X; detuning below and above 1.2 kHz → §13d, §13e (ch. 5, 6).
- Bragg–local-resonance coupling, widening vs narrowing → §13d (ch. 5).
- Complete vs directional gaps in 2D → §13b, §13g (ch. 3, 8). Local-resonance gaps are omnidirectional only in
  the effective-medium limit.
- Degenerate perturbation theory / two-level model → §13a (ch. 2, 4, 7).
- Eigenvectors and resonator participation, preparation for MAC tracking → §13f (ch. 7).
- k(ω), evanescent waves, link to insertion loss → §13g (ch. 8).
- Infinite lattice vs the finite 1.1 × 1.0 m plate → §13g, qualitative only (ch. 8). Quantitative comparison
  with the forced-response pipeline still open (needs resonators on the 0.1 m lattice).

## 13. Theory session results (2026-11-09/10)

Numbers from the FE production cell or the plane-wave model (marked PWE; RM unless "Kirchhoff"). The scratch
scripts behind them were not kept in the repo; the numbers are reproduced in the tex chapters.

### 13a. Two-level model and avoided crossings (ch. 2, 4, 7)
- Two modes, H = [[H₁₁, c], [c*, H₂₂]]: λ = H̄ ± R, H̄ = (H₁₁+H₂₂)/2, Δ = (H₁₁−H₂₂)/2, R = √(Δ² + |c|²).
  At a crossing (Δ = 0) the splitting is ±|c| (first order); far away the shift is |c|²/(λ₁−λ₂) (second order).
- Rigid point mass couples two plate waves: H_nn = λ_n⁰ − μλ, c = −μλ (coupling ∝ μ). Resonator as its own DOF
  (v = √m u): H = [[λ_p + μλ_r, −√μ λ_r], [−√μ λ_r, λ_r]] (coupling ∝ √μ). Γ: eigenvalues 0 and λ_r(1+μ).
- Eigenvectors depend only on c/Δ: Δ = R cos 2χ, c = R sin 2χ; upper band (cos χ, sin χ), lower (−sin χ, cos χ);
  plate share ½(1 ± Δ/R). Half the 90° turn happens within |Δ| < |c|, i.e. a window 4√μ λ_r wide in λ_p
  (lighter resonator → narrower crossing). At Γ (λ_p = 0): Δ/R = −0.6 → band 1 starts at 80 % plate, not 100 %.
- True crossings: c = 0 by symmetry (mode with a node at the resonator, e.g. sin pattern at X).

### 13b. Bragg gaps in 2D (ch. 3)
- A d-fold degenerate group of plane waves has one coupled combination with |φ(x₀)|² = d/(ρhA)
  (d = 2 at X, 4 at M); the other d − 1 modes keep the bare frequency (3 modes at 2379 Hz at M, 4730 Hz at Γ,
  5881 Hz at X in FE).
- d-wave model: (λ_p − λ)(λ_r − λ) = dμ λ_r λ; roots sum λ_p + λ_r(1 + dμ), product λ_p λ_r.
- Plate share at X: model 34 / 100 / 66 %, FE 31 / 100 / 69 % (836 / 1195 / 1653 Hz).

### 13c. Effective mass and the plane-wave condition (ch. 5, 6)
- Resonator on a base moving w₀: force = m_eff λ w₀, m_eff = m/(1−β), β = λ/λ_r = (f/f_r)², μ_eff = μ/(1−β).
- Mass-normalized plane waves φ_n = e^{i(k+G_n)·x}/√(ρhA): **1 = μ_eff Σ_n λ/(λ_n⁰ − λ)** (force × compliance = 1).
- Picture: resonator line Y(λ) = (1 − β)/μ against plate curve P(λ) = Σ λ/(λ_n⁰ − λ). One root per interval
  between distinct poles; d − 1 stuck modes per degenerate group; P > 0 ⇔ root below f_r. Raising f_r raises
  all coupled roots; raising μ turns Y about λ_r and pushes roots away from f_r.
- At Γ: β = 1 + μ − μS with S the sum over G ≠ 0; the resonator band lies above f_r only while S < 1
  (up to f_r ≈ 1765 Hz).
- RM: g ~ 1/(D_s κ²) for short waves → the sum diverges like ln(cutoff) (point force singularity); Kirchhoff
  converges.

### 13d. Why the gap is 1031–1195 Hz and not 1200–1342 Hz (ch. 5)
- Upper edge = sin pattern at X (node at the resonator, does not see it, fixed at f_X = 1194 Hz).
  Lower edge = band 1 at M, pulled down by μ_eff (four waves couple). Band 1 always stays below f_r and below the
  bare band (rule 1). A "Bragg gap with resonance-boosted mass".
- f_r sweep (PWE, μ = 0.25): upper edge = min(resonator band near Γ, f_X); switch at f_r ≈ 1120 Hz.
  f_r < f_X: gap contains f_r (500: 487–554, 1000: 901–1076, widest 1100: 969–1170 = 201 Hz);
  f_X < f_r < ≈ 1510 Hz: gap entirely below f_r (1200: 1032–1194, 1400: 1142–1194);
  above ≈ 1510 Hz: no complete gap (band 1 at M passes f_X). A gap above f_r (bands 2–3) never opened for
  300–2000 Hz.

### 13e. Detuning limits and several resonators (ch. 6)
- f_r → ∞: resonator becomes a rigid mass; band 1 at M saturates at 1648 Hz (all waves; 4-wave formula 1682 Hz).
- Two resonators per cell (PWE, f_r 1200, μ 0.25 each; gap between bands 2 and 3):
  (0,0)+(a/2,0): 1137–1194 Hz (antiphase resonator mode at Y, 1137 Hz = max of band 2, lands in the old gap);
  (0,0)+(a/2,a/2): 1114–1429 Hz (rotated lattice a/√2, Bragg frequency moves to 2378 Hz, gap contains f_r).
  One resonator with μ = 0.5: 926–1194 Hz.

### 13f. Eigenvectors, plate share and MAC (ch. 7) — input for stage 5
- Reduced eigenvector ψ: plate DOFs at independent nodes + resonator displacement u (last entry). Scale and
  phase arbitrary; different λ ⇒ M_r-orthogonal; degenerate group ⇒ arbitrary basis.
- Plate share = 1 − m|u|²/(ψᴴM_rψ). Exact sum rule: resonator shares of **all** bands at one k add to 100 %
  (Ψ Ψᴴ = M_r⁻¹). Plate shares of two bands alone only approximately (up to 104 % near M).
- M–Γ (FE): band 1 80 % plate at Γ → 5 % at M (50 % near k ≈ 0.63 π/a), band 2 the reverse. One-wave model
  matches to a few % up to k ≈ 0.6, fails near M (four waves).
- MAC = |⟨ψ₁,ψ₂⟩|²/(⟨ψ₁,ψ₁⟩⟨ψ₂,ψ₂⟩), ⟨·,·⟩ M_r-weighted. 1 − MAC ∝ δk²: band 1 worst step 0.9622 / 0.9898 /
  0.9974 for 20 / 40 / 80 points per segment (69 / 138 / 274 wavevectors; M–Γ gets √2× more points, 57 steps at
  40). On M–Γ: FE min 0.99736, two-level model 0.99778. Baseline 1 − MAC ≈ 0.0004 from the e^{ik·x} pattern
  itself (≈ 0.8·(δk a)²/6).
- Variants (band 1, 40 points): full vector M_r-weighted 0.990, plate entries only 0.969, unweighted 0.619 →
  use full vector incl. u, M_r-weighted.
- Three situations: smooth (MAC → 1 with refinement); degeneracy (bands 2/3/4 next to M/Γ/X: 0.23 / 0.2 / 0.06,
  **not** improved by refinement; row sums of the MAC matrix against the group ≈ 1 (0.97–1.00) → compare
  groups/subspaces, compute all members); true crossing (sorted MAC ≈ 0, bands 5/6 near k ≈ 0.33 on M–Γ at
  ~6.5 kHz; symmetry not checked).
- End-to-end MAC is meaningless (band 1 Γ vs M 0.35 although every step ≥ 0.99).
- Tracking (identity) ≠ gap finding (sorted frequencies per k). Tracked bands give wrong gap edges when bands
  cross → keep plain sorting for gaps.

### 13g. Inside a gap: complex k and the finite plate (ch. 8)
- ψ(x + a) = e^{ika}ψ(x), k = k_re + iσ: amplitude × e^{−σa} per cell, 8.686 σ dB/m. Roots in sets
  {k, −k, k*, −k*}. The bending near field (k = iκ_B, κ_B a ≈ 3 at 1.1 kHz) exists without resonators; a gap is
  where the propagating pair disappears.
- Two-level model with k = π/a + iσ: Δ ≈ sδ, s = 4λ_X a/π ⇒ **σ = √(c² − (λ − H̄)²)/s**; σ = 0 at the edges and
  grows like √(distance from the edge); with resonator (c = −μ_eff λ, H̄ = λ_X − |c|):
  σa = (π/4)√(x(2|c| − x))/λ_X, x = λ_X − λ. Good to ~10 % mid-gap, worse near f_r (many waves).
- Kirchhoff PWE (complete gap 1036.7–1199.8 Hz; FE 1031–1195): along x the decaying root has k_re = π/a
  (antiphase), σa up to 0.407 at 1050 Hz = 35 dB/m; √-check at the lower edge σa/√(f − 839.2) = 0.038.
  Along the diagonal: 25–133 dB/m (complex pair with k_re off the zone edge from ~1075 Hz), gap up to ~1270 Hz.
  The weaker direction (x, y) governs plate-level measures; intermediate angles not computed.
- Finite plate (~10 cells, 1 m): ≥ 20 dB transmission loss over 1 m along x only for ≈ 880–1180 Hz (narrower
  than the Bloch gap). Evanescent decay dissipates nothing (energy stays near the source / is reflected).
  Damping η = 0.04: σ = ηκ_B/4 ≈ 2.6 dB/m at 1.1 kHz (0.65 for η = 0.01) — small against the gap decay, rounds
  the edges, dominates outside the gap. Drive-point mobility drops much less than transfer mobility.
  Applies only with resonators on the 0.1 m lattice.
- FE k(ω): with z = e^{ik_x a}, (A₋/z + A₀ + A₊z)ψ = 0 is a quadratic eigenproblem at fixed ω (roots z, 1/z);
  for complex k use T(−k)ᵀ instead of Tᴴ. Optional later step.

### 13h. Corrections made during the session
- "On plate DOFs alone a resonator-dominated mode has almost nothing left to compare" was overstated
  (0.969 vs 0.990); the real failure is omitting M_r (0.619). Fixed in ch. 4.
- "The resonator band always lies above f_r" holds only while S < 1 (f_r ≲ 1765 Hz).
- "Plate shares of bands 1 and 2 add to 100 %" is approximate; the exact rule is the resonator-share sum rule.
- Local-resonance gaps are omnidirectional and located at f_r … f_r√(1+μ) only in the effective-medium limit.

### 13i. Open after the session
- FE k(ω) with damping (gap depth), intermediate propagation angles.
- Symmetry check of the true crossing of bands 5/6 on M–Γ.
- Comparison with the forced-response insertion loss on a plate with resonators on the 0.1 m lattice.
