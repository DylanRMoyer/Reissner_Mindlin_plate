Relevant code inside refactored. Other files are preliminary versions or test files.

In-depth explanation in progress.

Free-plate validation. The FE pipeline (Reissner–Mindlin, Serendipity deg. 2) reproduces the free-edge
eigenfrequencies of a square steel plate at h/a = 0.002 to within 0.13 % of Porter (2017, ν = 0.225, 12 modes,
identical mode counts and pair multiplicities) and lies inside all rigorous Kirchhoff bounds in
Leissa (1969, Tables 4.67 and §4.3.15). Leissa’s point values (Ritz upper bounds) lie 0.0–2.3 % above FE.
At h/a = 0.02, two independent MATLAB models agree within 1.1 % with matching mode counts.
Open points: one Leissa entry (ν = 0.3, λ² = 65.368) has no counterpart in FE or Porter, and the assignment
of a colleague’s 310.85 Hz value to a split degenerate pair is unconfirmed. Not covered: rectangular plates,
mode shapes, higher modes. Details, formulas and tolerances: validation_eigenvalues.md; regenerate with
python -m validation.validate_eigenvalues.