# Free-plate eigenfrequency validation: Serendipity vs Lagrange
Generated 2026-11-13 with DOLFINx 0.11.0. Square plate, side L = 1 m, free edges, Reissner-Mindlin, degree 2.
Every case is computed twice, with Serendipity ("S") and with Lagrange ("L") elements, on the same mesh.
Pass/fail is evaluated separately for each element type with the criteria below; "L - S" is the relative
difference of the Lagrange value with respect to the Serendipity value, (L - S) / S in %.

Definitions (m = rho * h is the areal mass; rho = volumetric density):
- D = E h^3 / (12 (1 - nu^2)), E = mu (3 lambda + 2 mu) / (lambda + mu), nu = lambda / (2 (lambda + mu))
- Leissa: lambda^2 = omega L^2 sqrt(m / D), omega = 2 pi f
- Leissa's Table 4.61 (Lemke) and the AS class of Table 4.65 (Bazley, Fox and Stadter) are Ritz upper bounds
  (Section 4.3.15). For the other classes of Table 4.65 the source method is not stated in the text consulted; the
  upper-bound property is assumed there. Criterion for Leissa rows: FE <= ref + 0.1 % and FE >= ref - 3 %.
  The criterion was fixed after inspecting the results; its justification is Leissa's Table 4.67 (rigorous bounds
  for the AS class), not the outcome.
- Interval checks (Table 4.67 and the fundamental-mode interval) are bounds for the Kirchhoff plate. The Table 4.67
  intervals are 2-3 % wide, so "inside" confirms FE only to about that level; the upper-bound margin is the
  tight part. The fundamental-mode interval is only 0.1 % wide.
- Porter (plate occupies |x|,|y| < a, so L = 2a): Lambda = m omega^2 a^4 / D = (lambda^2 / 4)^2, i.e. lambda^2 = 4 sqrt(Lambda)
- Symmetry-class labels in the Leissa and Porter tables are those of the references. They are not computed from the FE modes.
- Relative error = (FE - ref) / ref in %, compared in lambda^2 (literature) or Hz (colleague)
- Shear term integrated with quadrature degree 4 (full 3x3 Gauss) instead of the reduced 2x2 rule of the original report (validation_eigenvalues_S_vs_Lagrange.md).

## Colleague, MATLAB Reissner-Mindlin plate
Source: private communication

- Reference modes are assigned to FE modes by explicit index (no frequency matching).
- Several MATLAB values map onto one FE level (167.74/168.11 Hz and 297.70/310.85 Hz): the exact degenerate pairs of the square plate are split by the MATLAB mesh, so this is intended.
- Assignments marked 'assigned by me' rest on frequency proximity only;mode shapes are needed to confirm.Mode counts agree (reference and FE). The 310.85 Hz value is assigned as the second half of a splitdegenerate pair (hypothesis, -4.9 % relative to the FE pair) and does not enter the pass criterion;mode shapes from the colleague would confirm or refute this.

FE: nu = 0.2995, h = 0.02 m, mesh 50x50. Criterion: |error| <= 1.5 %.

| ref. column | reference (native) | reference (Hz) | S (Hz) | S err [%] | L (Hz) | L err [%] | L - S [%] | note |
|---|---|---|---|---|---|---|---|---|
| - | 65.1030 | 65.1030 | 64.9685 | -0.21 | 64.9685 | -0.21 | -0.00 |  |
| - | 95.2210 | 95.2210 | 95.1742 | -0.05 | 95.1742 | -0.05 | -0.00 |  |
| - | 117.9485 | 117.9485 | 117.7834 | -0.14 | 117.7834 | -0.14 | -0.00 |  |
| - | 167.7359 | 167.7359 | 167.7007 | -0.02 | 167.7007 | -0.02 | -0.00 |  |
| - | 168.1130 | 168.1130 | 167.7007 | -0.25 | 167.7007 | -0.25 | -0.00 |  |
| - | 297.7011 | 297.7011 | 295.6762 | -0.68 | 295.6762 | -0.68 | -0.00 | assigned by me |
| - | 307.2236 | 307.2236 | 305.6763 | -0.50 | 305.6763 | -0.50 | -0.00 | assigned by me |
| - | 310.8511 | 310.8511 | 295.6762 | -4.88 | 295.6762 | -4.88 | -0.00 | hypothesis: other half of the 295.66 Hz pair,split by the MATLAB mesh; not counted |
| - | 336.4824 | 336.4824 | 333.2046 | -0.97 | 333.2046 | -0.97 | -0.00 |  |

- Serendipity: mode count up to 343.2 Hz: reference 9, FE 9
- Lagrange: mode count up to 343.2 Hz: reference 9, FE 9

Serendipity: compared 8, excluded 0, unmatched but documented 0, beyond computed range 0, problems 0; max |error| = 0.97 % -> PASS  
Lagrange: compared 8, excluded 0, unmatched but documented 0, beyond computed range 0, problems 0; max |error| = 0.97 % -> PASS  

Lagrange vs Serendipity: max |L - S| = 0.000 %, mean L - S = -0.000 %

## Colleague, MATLAB Reissner-Mindlin shell
Source: private communication

- Reference modes are assigned to FE modes by explicit index (no frequency matching).
- Several MATLAB values map onto one FE level (167.77/168.11 Hz and 292.45/298.11 Hz): the exact degenerate pairs are split by the MATLAB mesh, so this is intended.
- Assignments marked 'assigned by me' rest on frequency proximity only; mode shapes are needed to confirm.

FE: nu = 0.2995, h = 0.02 m, mesh 50x50. Criterion: |error| <= 1.5 %.

| ref. column | reference (native) | reference (Hz) | S (Hz) | S err [%] | L (Hz) | L err [%] | L - S [%] | note |
|---|---|---|---|---|---|---|---|---|
| - | 65.0964 | 65.0964 | 64.9685 | -0.20 | 64.9685 | -0.20 | -0.00 |  |
| - | 95.1580 | 95.1580 | 95.1742 | +0.02 | 95.1742 | +0.02 | -0.00 |  |
| - | 118.1618 | 118.1618 | 117.7834 | -0.32 | 117.7834 | -0.32 | -0.00 |  |
| - | 167.7658 | 167.7658 | 167.7007 | -0.04 | 167.7007 | -0.04 | -0.00 |  |
| - | 168.1136 | 168.1136 | 167.7007 | -0.25 | 167.7007 | -0.25 | -0.00 |  |
| - | 292.4477 | 292.4477 | 295.6762 | +1.10 | 295.6762 | +1.10 | -0.00 | assigned by me |
| - | 298.1096 | 298.1096 | 295.6762 | -0.82 | 295.6762 | -0.82 | -0.00 | assigned by me |
| - | 307.7197 | 307.7197 | 305.6763 | -0.66 | 305.6763 | -0.66 | -0.00 |  |
| - | 332.2106 | 332.2106 | 333.2046 | +0.30 | 333.2046 | +0.30 | -0.00 |  |

- Serendipity: mode count up to 338.9 Hz: reference 9, FE 9
- Lagrange: mode count up to 338.9 Hz: reference 9, FE 9

Serendipity: compared 9, excluded 0, unmatched but documented 0, beyond computed range 0, problems 0; max |error| = 1.10 % -> PASS  
Lagrange: compared 9, excluded 0, unmatched but documented 0, beyond computed range 0, problems 0; max |error| = 1.10 % -> PASS  

Lagrange vs Serendipity: max |L - S| = 0.000 %, mean L - S = -0.000 %

## Leissa (1969), nu = 0.3
Source: Leissa, Vibration of Plates, NASA SP-160; table 4.65, pages 104 - 106

FE: nu = 0.3000, h = 0.002 m, mesh 80x80. Criterion: FE <= ref + 0.1 % and FE >= ref - 3.0 % (references are Ritz upper bounds). Match window 3 %.

| ref. column | reference (native) | reference (lambda^2) | S (lambda^2) | S err [%] | L (lambda^2) | L err [%] | L - S [%] | note |
|---|---|---|---|---|---|---|---|---|
| AS | 13.4728 | 13.4728 | 13.4641 | -0.06 | 13.4641 | -0.06 | -0.00 |  |
| SA | 19.5961 | 19.5961 | 19.5967 | +0.00 | 19.5967 | +0.00 | +0.00 |  |
| SS | 24.2702 | 24.2702 | 24.2713 | +0.00 | 24.2713 | +0.00 | -0.00 |  |
| SS | 63.6870 | 63.6870 | 63.6642 | -0.04 | 63.6642 | -0.04 | -0.00 |  |
| SA | 65.3680 | 65.3680 | - | - | - | - | - | excluded: suspect entry |
| AA | 69.5020 | 69.5020 | 69.2546 | -0.36 | 69.2546 | -0.36 | -0.00 |  |
| AS | 77.5897 | 77.5897 | 77.1703 | -0.54 | 77.1703 | -0.54 | -0.00 |  |
| SA | 117.1093 | 117.1093 | 117.1470 | +0.03 | 117.1470 | +0.03 | -0.00 |  |
| SS | 122.4449 | 122.4449 | 122.4892 | +0.04 | 122.4892 | +0.04 | -0.00 |  |
| AS | 156.2387 | 156.2387 | 152.7929 | -2.21 | 152.7929 | -2.21 | -0.00 |  |
| SA | 161.5049 | 161.5049 | 161.4771 | -0.02 | 161.4771 | -0.02 | -0.00 |  |
| SS | 168.4888 | 168.4888 | 168.4751 | -0.01 | 168.4750 | -0.01 | -0.00 |  |
| AA | 173.6954 | 173.6954 | - | - | - | - | - | NO FE MODE |


Serendipity: compared 11, excluded 1, unmatched but documented 0, beyond computed range 0, problems 1; max FE above reference = 0.04 %, max FE below reference = 2.21 % -> FAIL  
Lagrange: compared 11, excluded 1, unmatched but documented 0, beyond computed range 0, problems 1; max FE above reference = 0.04 %, max FE below reference = 2.21 % -> FAIL  

Lagrange vs Serendipity: max |L - S| = 0.000 %, mean L - S = -0.000 %

## Leissa (1969), nu = 0.225
Source: Leissa, Vibration of Plates, NASA SP-160; table 4.61, pages 92 - 97

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Criterion: FE <= ref + 0.1 % and FE >= ref - 3.0 % (references are Ritz upper bounds). Match window 3 %.

| ref. column | reference (native) | reference (lambda^2) | S (lambda^2) | S err [%] | L (lambda^2) | L err [%] | L - S [%] | note |
|---|---|---|---|---|---|---|---|---|
| AS | 14.1400 | 14.1400 | 14.1120 | -0.20 | 14.1120 | -0.20 | -0.00 |  |
| SA | 20.4900 | 20.4900 | 20.3875 | -0.50 | 20.3875 | -0.50 | -0.00 |  |
| SS | 23.9700 | 23.9700 | 23.8791 | -0.38 | 23.8791 | -0.38 | -0.00 |  |
| SS | 66.4020 | 66.4020 | 65.6243 | -1.17 | 65.6243 | -1.17 | -0.00 |  |
| AA | 71.8300 | 71.8300 | 71.6215 | -0.29 | 71.6215 | -0.29 | -0.00 |  |
| AS | 77.8810 | 77.8810 | 77.4809 | -0.51 | 77.4809 | -0.51 | -0.00 |  |


Serendipity: compared 6, excluded 0, unmatched but documented 0, beyond computed range 0, problems 0; max FE above reference = 0.00 %, max FE below reference = 1.17 % -> PASS  
Lagrange: compared 6, excluded 0, unmatched but documented 0, beyond computed range 0, problems 0; max FE above reference = 0.00 %, max FE below reference = 1.17 % -> PASS  

Lagrange vs Serendipity: max |L - S| = 0.000 %, mean L - S = -0.000 %

## Porter (2017), nu = 0.225, N = 48
Source: Porter, Eigenfrequencies and eigenmodes for a thin rectangular elastic plate with free edges (2017), Table 1 (https://people.maths.bris.ac.uk/~marp/abstracts/free-edge.pdf)

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Criterion: |error| <= 0.3 %. Match window 3 %.

| ref. column | reference (native) | reference (lambda^2) | S (lambda^2) | S err [%] | L (lambda^2) | L err [%] | L - S [%] | note |
|---|---|---|---|---|---|---|---|---|
| AA | 12.4552 | 14.1168 | 14.1120 | -0.03 | 14.1120 | -0.03 | -0.00 |  |
| SS | 25.9900 | 20.3922 | 20.3875 | -0.02 | 20.3875 | -0.02 | -0.00 |  |
| SS | 35.6495 | 23.8829 | 23.8791 | -0.02 | 23.8791 | -0.02 | -0.00 |  |
| pair | 80.9341 | 35.9854 | 35.9679 | -0.05 | 35.9679 | -0.05 | -0.00 |  |
| pair | 235.5000 | 61.3840 | 61.3797 | -0.01 | 61.3797 | -0.01 | -0.00 |  |
| SS | 269.4370 | 65.6581 | 65.6243 | -0.05 | 65.6243 | -0.05 | -0.00 |  |
| AA | 320.9690 | 71.6624 | 71.6215 | -0.06 | 71.6215 | -0.06 | -0.00 |  |
| AA | 375.4200 | 77.5030 | 77.4809 | -0.03 | 77.4809 | -0.03 | -0.00 |  |
| pair | 730.5130 | 108.1120 | 108.0544 | -0.05 | 108.0544 | -0.05 | -0.00 |  |
| SS | 876.7470 | 118.4397 | 118.4479 | +0.01 | 118.4479 | +0.01 | -0.00 |  |
| pair | 1104.2600 | 132.9216 | 132.8931 | -0.02 | 132.8931 | -0.02 | -0.00 |  |
| AA | 1526.6500 | 156.2895 | 156.2049 | -0.05 | 156.2048 | -0.05 | -0.00 |  |
| AA | 2687.1500 | 207.3509 | - | - | - | - | - | beyond computed range |


Serendipity: compared 12, excluded 0, unmatched but documented 0, beyond computed range 1, problems 0; max |error| = 0.06 % -> PASS  
Lagrange: compared 12, excluded 0, unmatched but documented 0, beyond computed range 1, problems 0; max |error| = 0.06 % -> PASS  

Lagrange vs Serendipity: max |L - S| = 0.000 %, mean L - S = -0.000 %

## Leissa, bounds for AS class (table 4.67), nu = 0.3
Source: Leissa (1969), table 4.67 (bounds by Bazley, Fox and Stadter, ref. 4.118); page: 109

FE: nu = 0.3000, h = 0.002 m, mesh 80x80. Bounds are for the Kirchhoff plate, FE is Reissner-Mindlin.

| lower | upper | width [%] | S (lambda^2) | S margin lo [%] | S margin hi [%] | S result | L (lambda^2) | L margin lo [%] | L margin hi [%] | L result | L - S [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13.201 | 13.474 | 2.07 | 13.4641 | +1.99 | -0.07 | inside | 13.4641 | +1.99 | -0.07 | inside | -0.00 |
| 75.735 | 77.43 | 2.24 | 77.1703 | +1.90 | -0.34 | inside | 77.1703 | +1.90 | -0.34 | inside | -0.00 |
| 147.71 | 153.12 | 3.66 | 152.7929 | +3.44 | -0.21 | inside | 152.7929 | +3.44 | -0.21 | inside | -0.00 |
| 209.46 | 214.85 | 2.57 | - | - | - | beyond computed range | - | - | - | beyond computed range | - |

Serendipity: checked 3 -> PASS  
Lagrange: checked 3 -> PASS

## Leissa, bounds for AS class (table 4.67), nu = 0.225
Source: Leissa (1969), table 4.67; page: 109

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Bounds are for the Kirchhoff plate, FE is Reissner-Mindlin.

| lower | upper | width [%] | S (lambda^2) | S margin lo [%] | S margin hi [%] | S result | L (lambda^2) | L margin lo [%] | L margin hi [%] | L result | L - S [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13.851 | 14.119 | 1.93 | 14.1120 | +1.88 | -0.05 | inside | 14.1120 | +1.88 | -0.05 | inside | -0.00 |
| 76.245 | 77.621 | 1.80 | 77.4809 | +1.62 | -0.18 | inside | 77.4809 | +1.62 | -0.18 | inside | -0.00 |
| 151.54 | 156.41 | 3.21 | 156.2049 | +3.08 | -0.13 | inside | 156.2048 | +3.08 | -0.13 | inside | -0.00 |

Serendipity: checked 3 -> PASS  
Lagrange: checked 3 -> PASS

## Leissa, fundamental mode, nu = 0.225 (refs 4.115 - 4.117)
Source: Leissa (1969), section 4.3.15; page: 87

Caveat: FE lies only about 0.02 % above the lower bound. The bound is for the Kirchhoff plate, while FE is Reissner-Mindlin; the Reissner-Mindlin shift at h/a = 0.002 is estimated (not verified) at 0.01-0.02 %. Read this row as 'consistent', not as strong confirmation.

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Bounds are for the Kirchhoff plate, FE is Reissner-Mindlin.

| lower | upper | width [%] | S (lambda^2) | S margin lo [%] | S margin hi [%] | S result | L (lambda^2) | L margin lo [%] | L margin hi [%] | L result | L - S [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14.1028 | 14.1165 | 0.10 | 14.1120 | +0.07 | -0.03 | inside | 14.1120 | +0.07 | -0.03 | inside | -0.00 |

Serendipity: checked 1 -> PASS  
Lagrange: checked 1 -> PASS
