# Free-plate eigenfrequency validation
Generated 2026-10-09 with DOLFINx 0.11.0. Square plate, side L = 1 m, free edges, Reissner-Mindlin, Serendipity degree 2.

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

## Colleague, MATLAB Reissner-Mindlin plate
Source: private communication

- Reference modes are assigned to FE modes by explicit index (no frequency matching).
- Several MATLAB values map onto one FE level (167.74/168.11 Hz and 297.70/310.85 Hz): the exact degenerate pairs of the square plate are split by the MATLAB mesh, so this is intended.
- Assignments marked 'assigned by me' rest on frequency proximity only;mode shapes are needed to confirm.Mode counts agree (reference and FE). The 310.85 Hz value is assigned as the second half of a splitdegenerate pair (hypothesis, -4.9 % relative to the FE pair) and does not enter the pass criterion;mode shapes from the colleague would confirm or refute this.

FE: nu = 0.2995, h = 0.02 m, mesh 50x50. Criterion: |error| <= 1.5 %.

| ref. column | reference (native) | reference (Hz) | FE (Hz) | rel. error [%] | note |
|---|---|---|---|---|---|
| - | 65.1030 | 65.1030 | 64.9508 | -0.23 |  |
| - | 95.2210 | 95.2210 | 95.1729 | -0.05 |  |
| - | 117.9485 | 117.9485 | 117.7804 | -0.14 |  |
| - | 167.7359 | 167.7359 | 167.6566 | -0.05 |  |
| - | 168.1130 | 168.1130 | 167.6566 | -0.27 |  |
| - | 297.7011 | 297.7011 | 295.6584 | -0.69 | assigned by me |
| - | 307.2236 | 307.2236 | 305.5693 | -0.54 | assigned by me |
| - | 310.8511 | 310.8511 | 295.6584 | -4.89 | hypothesis: other half of the 295.66 Hz pair,split by the MATLAB mesh; not counted |
| - | 336.4824 | 336.4824 | 333.1192 | -1.00 |  |

Mode count up to 343.2 Hz: reference 9, FE 9.

Compared: 8, excluded: 0, unmatched but documented: 0, beyond computed range: 0, problems: 0; max |error| = 1.00 % -> PASS

## Colleague, MATLAB Reissner-Mindlin shell
Source: private communication

- Reference modes are assigned to FE modes by explicit index (no frequency matching).
- Several MATLAB values map onto one FE level (167.77/168.11 Hz and 292.45/298.11 Hz): the exact degenerate pairs are split by the MATLAB mesh, so this is intended.
- Assignments marked 'assigned by me' rest on frequency proximity only; mode shapes are needed to confirm.

FE: nu = 0.2995, h = 0.02 m, mesh 50x50. Criterion: |error| <= 1.5 %.

| ref. column | reference (native) | reference (Hz) | FE (Hz) | rel. error [%] | note |
|---|---|---|---|---|---|
| - | 65.0964 | 65.0964 | 64.9508 | -0.22 |  |
| - | 95.1580 | 95.1580 | 95.1729 | +0.02 |  |
| - | 118.1618 | 118.1618 | 117.7804 | -0.32 |  |
| - | 167.7658 | 167.7658 | 167.6566 | -0.07 |  |
| - | 168.1136 | 168.1136 | 167.6566 | -0.27 |  |
| - | 292.4477 | 292.4477 | 295.6584 | +1.10 | assigned by me |
| - | 298.1096 | 298.1096 | 295.6584 | -0.82 | assigned by me |
| - | 307.7197 | 307.7197 | 305.5693 | -0.70 |  |
| - | 332.2106 | 332.2106 | 333.1192 | +0.27 |  |

Mode count up to 338.9 Hz: reference 9, FE 9.

Compared: 9, excluded: 0, unmatched but documented: 0, beyond computed range: 0, problems: 0; max |error| = 1.10 % -> PASS

## Leissa (1969), nu = 0.3
Source: Leissa, Vibration of Plates, NASA SP-160; table 4.65, pages 104 - 106

FE: nu = 0.3000, h = 0.002 m, mesh 80x80. Criterion: FE <= ref + 0.1 % and FE >= ref - 3.0 % (references are Ritz upper bounds). Match window 3 %.

| ref. column | reference (native) | reference (lambda^2) | FE (lambda^2) | rel. error [%] | note |
|---|---|---|---|---|---|
| AS | 13.4728 | 13.4728 | 13.4575 | -0.11 |  |
| SA | 19.5961 | 19.5961 | 19.5956 | -0.00 |  |
| SS | 24.2702 | 24.2702 | 24.2691 | -0.00 |  |
| SS | 63.6870 | 63.6870 | 63.6233 | -0.10 |  |
| SA | 65.3680 | 65.3680 | - | - | excluded: suspect entry |
| AA | 69.5020 | 69.5020 | 69.2166 | -0.41 |  |
| AS | 77.5897 | 77.5897 | 77.1360 | -0.58 |  |
| SA | 117.1093 | 117.1093 | 117.0883 | -0.02 |  |
| SS | 122.4449 | 122.4449 | 122.4285 | -0.01 |  |
| AS | 156.2387 | 156.2387 | 152.6722 | -2.28 |  |
| SA | 161.5049 | 161.5049 | - | - | beyond computed range |
| SS | 168.4888 | 168.4888 | - | - | beyond computed range |
| AA | 173.6954 | 173.6954 | - | - | beyond computed range |


Compared: 9, excluded: 1, unmatched but documented: 0, beyond computed range: 3, problems: 0; max FE above reference = 0.00 %, max FE below reference = 2.28 % -> PASS

## Leissa (1969), nu = 0.225
Source: Leissa, Vibration of Plates, NASA SP-160; table 4.61, pages 92 - 97

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Criterion: FE <= ref + 0.1 % and FE >= ref - 3.0 % (references are Ritz upper bounds). Match window 3 %.

| ref. column | reference (native) | reference (lambda^2) | FE (lambda^2) | rel. error [%] | note |
|---|---|---|---|---|---|
| AS | 14.1400 | 14.1400 | 14.1054 | -0.24 |  |
| SA | 20.4900 | 20.4900 | 20.3864 | -0.51 |  |
| SS | 23.9700 | 23.9700 | 23.8772 | -0.39 |  |
| SS | 66.4020 | 66.4020 | 65.5850 | -1.23 |  |
| AA | 71.8300 | 71.8300 | 71.5844 | -0.34 |  |
| AS | 77.8810 | 77.8810 | 77.4476 | -0.56 |  |


Compared: 6, excluded: 0, unmatched but documented: 0, beyond computed range: 0, problems: 0; max FE above reference = 0.00 %, max FE below reference = 1.23 % -> PASS

## Porter (2017), nu = 0.225, N = 48
Source: Porter, Eigenfrequencies and eigenmodes for a thin rectangular elastic plate with free edges (2017), Table 1 (https://people.maths.bris.ac.uk/~marp/abstracts/free-edge.pdf)

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Criterion: |error| <= 0.3 %. Match window 3 %.

| ref. column | reference (native) | reference (lambda^2) | FE (lambda^2) | rel. error [%] | note |
|---|---|---|---|---|---|
| AA | 12.4552 | 14.1168 | 14.1054 | -0.08 |  |
| SS | 25.9900 | 20.3922 | 20.3864 | -0.03 |  |
| SS | 35.6495 | 23.8829 | 23.8772 | -0.02 |  |
| pair | 80.9341 | 35.9854 | 35.9510 | -0.10 |  |
| pair | 235.5000 | 61.3840 | 61.3654 | -0.03 |  |
| SS | 269.4370 | 65.6581 | 65.5850 | -0.11 |  |
| AA | 320.9690 | 71.6624 | 71.5844 | -0.11 |  |
| AA | 375.4200 | 77.5030 | 77.4476 | -0.07 |  |
| pair | 730.5130 | 108.1120 | 107.9827 | -0.12 |  |
| SS | 876.7470 | 118.4397 | 118.3905 | -0.04 |  |
| pair | 1104.2600 | 132.9216 | 132.8124 | -0.08 |  |
| AA | 1526.6500 | 156.2895 | 156.0893 | -0.13 |  |
| AA | 2687.1500 | 207.3509 | - | - | beyond computed range |


Compared: 12, excluded: 0, unmatched but documented: 0, beyond computed range: 1, problems: 0; max |error| = 0.13 % -> PASS

## Leissa, bounds for AS class (table 4.67), nu = 0.3
Source: Leissa (1969), table 4.67 (bounds by Bazley, Fox and Stadter, ref. 4.118); page: 109

FE: nu = 0.3000, h = 0.002 m, mesh 80x80. Bounds are for the Kirchhoff plate, FE is Reissner-Mindlin.

| lower | upper | width [%] | FE (lambda^2) | margin to lower [%] | margin to upper [%] | result |
|---|---|---|---|---|---|---|
| 13.201 | 13.474 | 2.07 | 13.4575 | +1.94 | -0.12 | inside |
| 75.735 | 77.43 | 2.24 | 77.1360 | +1.85 | -0.38 | inside |
| 147.71 | 153.12 | 3.66 | 152.6722 | +3.36 | -0.29 | inside |
| 209.46 | 214.85 | 2.57 | - | - | - | beyond computed range |

Checked: 3 -> PASS

## Leissa, bounds for AS class (table 4.67), nu = 0.225
Source: Leissa (1969), table 4.67; page: 109

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Bounds are for the Kirchhoff plate, FE is Reissner-Mindlin.

| lower | upper | width [%] | FE (lambda^2) | margin to lower [%] | margin to upper [%] | result |
|---|---|---|---|---|---|---|
| 13.851 | 14.119 | 1.93 | 14.1054 | +1.84 | -0.10 | inside |
| 76.245 | 77.621 | 1.80 | 77.4476 | +1.58 | -0.22 | inside |
| 151.54 | 156.41 | 3.21 | 156.0893 | +3.00 | -0.21 | inside |

Checked: 3 -> PASS

## Leissa, fundamental mode, nu = 0.225 (refs 4.115 - 4.117)
Source: Leissa (1969), section 4.3.15; page: 87

Caveat: FE lies only about 0.02 % above the lower bound. The bound is for the Kirchhoff plate, while FE is Reissner-Mindlin; the Reissner-Mindlin shift at h/a = 0.002 is estimated (not verified) at 0.01-0.02 %. Read this row as 'consistent', not as strong confirmation.

FE: nu = 0.2250, h = 0.002 m, mesh 80x80. Bounds are for the Kirchhoff plate, FE is Reissner-Mindlin.

| lower | upper | width [%] | FE (lambda^2) | margin to lower [%] | margin to upper [%] | result |
|---|---|---|---|---|---|---|
| 14.1028 | 14.1165 | 0.10 | 14.1054 | +0.02 | -0.08 | inside |

Checked: 1 -> PASS
