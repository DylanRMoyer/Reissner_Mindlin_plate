import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from periodic_bc import categorize, phase_factors, build_pairs, build_T

# rectangular, non-square cell with nx != ny so that any x/y mix-up shows up
LX, LY, NX, NY = 1.3, 0.7, 5, 3


# 1. categorize on hand-placed points (no mesh needed)
pts = np.array([[0.5, 0.3],                        # interior
                [0.0, 0.3], [LX, 0.3],             # L, R
                [0.5, 0.0], [0.5, LY],             # B, T
                [0.0, 0.0], [LX, 0.0],             # LL, LR
                [0.0, LY], [LX, LY],               # UL, UR
                [LX * (1 - 1e-12), LY + 1e-12]])   # UR within tolerance
expected = ["interior", "L", "R", "B", "T", "LL", "LR", "UL", "UR", "UR"]
assert list(categorize(pts, LX, LY)) == expected, categorize(pts, LX, LY)
print("categorize checks passed")

# 2. phase_factors
kx, ky = 0.3 * np.pi / LX, 0.8 * np.pi / LY
ph = phase_factors(np.array(["x", "y", "xy"]), kx, ky, LX, LY)
assert np.allclose(ph, [np.exp(0.3j * np.pi), np.exp(0.8j * np.pi), np.exp(1.1j * np.pi)])
assert np.allclose(phase_factors(np.array(["x", "y", "xy"]), 0.0, 0.0, LX, LY), 1.0)
print("phase_factors checks passed")

# 3. build_pairs on a Lagrange deg-2 mesh
cfg = PlateConfig(length=LX, width=LY, thickness=0.02, nx=NX, ny=NY, rho=7850, mu=77e9, lambda_=115e9)
domain, V, _, _ = build_plate_problem(cfg, deg=2, el_type="Lagrange")
P = build_pairs(V, LX, LY)
parent, xy, comp, cat = P["table"]
n_total, ind = len(parent), P["independent"]

# Q2 on an NX x NY grid: (2NX+1)(2NY+1) nodes per component
assert n_total == 3 * (2 * NX + 1) * (2 * NY + 1), n_total
for c in range(3):
    counts = {k: int(np.sum((comp == c) & (cat == k))) for k in ("L", "R", "B", "T", "LL", "LR", "UL", "UR")}
    assert counts["L"] == counts["R"] == 2 * NY - 1, (c, counts)
    assert counts["B"] == counts["T"] == 2 * NX - 1, (c, counts)
    assert all(counts[k] == 1 for k in ("LL", "LR", "UL", "UR")), (c, counts)

# slaves: R + T + 3 corners per component; independents: the 2NX x 2NY periodic node grid
assert len(P["slave"]) == 3 * (2 * NX + 2 * NY + 1), len(P["slave"])
assert len(ind) == 3 * (2 * NX) * (2 * NY), len(ind)
assert len(np.unique(P["slave"])) == len(P["slave"])
assert not np.intersect1d(P["slave"], P["master"]).size, "a master is itself a slave"
assert np.array_equal(np.sort(np.concatenate([ind, P["slave"]])), np.arange(n_total)), "slaves + independents != all DOFs"

# every pair: same component, slave = master + lattice vector of its kind
pos = {p: (x, y, c) for p, (x, y), c in zip(parent, xy, comp)}
shift = {"x": (LX, 0), "y": (0, LY), "xy": (LX, LY)}
for s, m, k in zip(P["slave"], P["master"], P["kind"]):
    (xs, ys, cs), (xm, ym, cm) = pos[s], pos[m]
    assert cs == cm and np.allclose([xs - xm, ys - ym], shift[k], atol=1e-12), (s, m, k)

# all three slave corners map directly onto the LL corner of the same component
for c in range(3):
    ll = parent[(comp == c) & (cat == "LL")][0]
    corners = parent[(comp == c) & np.isin(cat, ["LR", "UL", "UR"])]
    master_of = dict(zip(P["slave"], P["master"]))
    assert all(master_of[s] == ll for s in corners), c
print("build_pairs checks passed")

# 4. build_T structure
T = build_T(P, kx, ky, LX, LY)
assert T.shape == (n_total, len(ind))
assert T.nnz == n_total and (np.diff(T.indptr) == 1).all(), "every row must have exactly one entry"
assert np.allclose(np.abs(T.data), 1.0), "all entries must have modulus 1"
G = (T.conj().T @ T).toarray()
assert np.allclose(G, np.diag(1 + np.bincount(np.searchsorted(ind, P["master"]), minlength=len(ind)))), \
    "T^dagger T should be diagonal with 1 + (number of slaves) per master"

# 5. a Bloch plane wave exp(i k.x) is reproduced exactly from its independent values
for k in [(kx, ky), (-kx, 2 * ky), (0.0, 0.0)]:
    Tk = build_T(P, *k, LX, LY)
    q = np.zeros(n_total, dtype=complex)
    q[parent] = np.exp(1j * (k[0] * xy[:, 0] + k[1] * xy[:, 1])) * (1 + comp)   # distinct amplitude per component
    assert np.allclose(Tk @ q[ind], q), f"plane wave not reproduced at k={k}"

# a non-periodic field must not be in the range of T(0)
q_lin = np.zeros(n_total)
q_lin[parent] = xy[:, 0]
assert not np.allclose(build_T(P, 0.0, 0.0, LX, LY) @ q_lin[ind], q_lin)
print("build_T checks passed")
