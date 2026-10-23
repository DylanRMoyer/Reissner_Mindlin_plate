import numpy as np
from weak_form import collapse_subspace
import scipy.sparse as sp

# slave category -> (shift to master x, shift to master y, phase kind)
RULES = {"R":  (-1, 0, "x"),  "T":  (0, -1, "y"),
         "LR": (-1, 0, "x"),  "UL": (0, -1, "y"),  "UR": (-1, -1, "xy")}


def dof_table(function_space):
    """Per-DOF arrays for the mixed (w, theta) space: parent index, xy, component (0=w, 1=thx, 2=thy).
    Needs nodal elements (Lagrange) and a serial run."""
    assert function_space.mesh.comm.size == 1, "serial only for now"
    V_w, w_par = collapse_subspace(function_space, 0)
    V_t, t_par = collapse_subspace(function_space, 1)
    xw = V_w.tabulate_dof_coordinates()[:, :2]
    xt = V_t.tabulate_dof_coordinates()[:, :2]
    assert len(w_par) == len(xw) and len(t_par) == 2 * len(xt), (
        f"unexpected to_parent layout: w {len(w_par)} vs {len(xw)} nodes, theta {len(t_par)} vs {len(xt)} nodes")
    t_par = t_par.reshape(-1, 2)
    parent = np.concatenate([w_par, t_par[:, 0], t_par[:, 1]])
    xy = np.vstack([xw, xt, xt])
    comp = np.concatenate([np.zeros(len(xw), int), np.ones(len(xt), int), 2 * np.ones(len(xt), int)])
    n_total = function_space.dofmap.index_map.size_global * function_space.dofmap.index_map_bs
    assert len(parent) == n_total and len(np.unique(parent)) == n_total, "DOF table does not cover the space"
    return parent, xy, comp


def categorize(xy, Lx, Ly):
    tol = 1e-8 * max(Lx, Ly)
    left, right = np.abs(xy[:, 0]) < tol, np.abs(xy[:, 0] - Lx) < tol
    bottom, top = np.abs(xy[:, 1]) < tol, np.abs(xy[:, 1] - Ly) < tol
    cat = np.full(len(xy), "interior", dtype="<U8")
    cat[left & ~bottom & ~top] = "L";   cat[right & ~bottom & ~top] = "R"
    cat[bottom & ~left & ~right] = "B"; cat[top & ~left & ~right] = "T"
    cat[left & bottom] = "LL";  cat[right & bottom] = "LR"
    cat[left & top] = "UL";     cat[right & top] = "UR"
    return cat


def build_pairs(function_space, Lx, Ly):
    """Returns dict with slave/master parent-index arrays, phase kind per pair, the independent DOF set,
    and the raw table (for tests)."""
    parent, xy, comp = dof_table(function_space)
    cat = categorize(xy, Lx, Ly)
    key = lambda c, x, y: (int(c), round(float(x) / Lx, 9) + 0.0, round(float(y) / Ly, 9) + 0.0)
    lookup = {key(c, x, y): p for p, (x, y), c in zip(parent, xy, comp)}

    slaves, masters, kinds = [], [], []
    for i in np.flatnonzero(np.isin(cat, list(RULES))):
        sx, sy, kind = RULES[cat[i]]
        x, y = xy[i]
        k = key(comp[i], x + sx * Lx, y + sy * Ly)
        if k not in lookup:
            raise KeyError(f"no master for slave dof {parent[i]} (cat {cat[i]}, comp {comp[i]}) at {xy[i]}")
        slaves.append(parent[i]); masters.append(lookup[k]); kinds.append(kind)

    slaves, masters, kinds = np.array(slaves), np.array(masters), np.array(kinds)
    independent = np.setdiff1d(parent, slaves)
    return dict(slave=slaves, master=masters, kind=kinds, independent=independent,
                table=(parent, xy, comp, cat))

def phase_factors(kind, kx, ky, Lx, Ly):
    px, py = np.exp(1j * kx * Lx), np.exp(1j * ky * Ly)
    lut = {"x": px, "y": py, "xy": px * py}
    return np.array([lut[k] for k in kind])

def build_T(pairs, kx, ky, Lx, Ly):
    n_total = len(pairs["table"][0])
    independent = pairs["independent"]
    n_ind = len(independent)

    col_of = np.full(n_total, -1, dtype=int)
    col_of[independent] = np.arange(n_ind)

    slave_cols = col_of[pairs["master"]]
    assert (slave_cols >= 0).all(), "a master is not an independent DOF"

    rows = np.concatenate([independent, pairs["slave"]])
    cols = np.concatenate([np.arange(n_ind), slave_cols])
    vals = np.concatenate([np.ones(n_ind, dtype=complex),
                           phase_factors(pairs["kind"], kx, ky, Lx, Ly)])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_total, n_ind)).tocsr()



if __name__ == "__main__":
    from config import PlateConfig
    from problem_setup import build_plate_problem

    NX = NY = 4
    LX, LY = 1.0, 1.0
    cfg = PlateConfig(length=LX, width=LY, thickness=0.02, nx=NX, ny=NY, rho=7850, mu=77e9, lambda_=115e9)
    domain, V, _, _ = build_plate_problem(cfg, deg=2, el_type="Lagrange")
    P = build_pairs(V, LX, LY)
    parent, xy, comp, cat = P["table"]

    # counts per component and category
    for c, name in enumerate(("w", "theta_x", "theta_y")):
        counts = {k: int(np.sum((comp == c) & (cat == k))) for k in
                  ("interior", "L", "R", "B", "T", "LL", "LR", "UL", "UR")}
        print(name, counts)
        assert counts["R"] == counts["L"] == 2 * NY - 1 and counts["T"] == counts["B"] == 2 * NX - 1
        assert all(counts[k] == 1 for k in ("LL", "LR", "UL", "UR"))

    n_slave_per_comp = 2 * NX + 2 * NY + 1
    assert len(P["slave"]) == 3 * n_slave_per_comp, len(P["slave"])
    assert len(P["independent"]) == 3 * 4 * NX * NY, len(P["independent"])
    assert len(np.unique(P["slave"])) == len(P["slave"]) and not np.intersect1d(P["slave"], P["master"]).size

    # translation consistency + same component on both ends
    pos = {p: (x, y, c) for p, (x, y), c in zip(parent, xy, comp)}
    shift = {"x": (LX, 0), "y": (0, LY), "xy": (LX, LY)}
    for s, m, k in zip(P["slave"], P["master"], P["kind"]):
        (xs, ys, cs), (xm, ym, cm) = pos[s], pos[m]
        assert cs == cm and np.allclose([xs - xm, ys - ym], shift[k], atol=1e-12), (s, m, k)
    print("pair counts, independence and translation checks passed")

    # ---- tests for build_T (add inside the existing __main__ block, after the pair checks) ----
    ind = P["independent"]
    n_total, n_ind = len(parent), len(ind)
    kx, ky = np.pi / 2 / LX, np.pi / LY            # phase advances (pi/2, pi)
    T = build_T(P, kx, ky, LX, LY)

    assert T.shape == (n_total, n_ind)
    assert T.nnz == n_total and (np.diff(T.indptr) == 1).all(), "every row must have exactly one entry"
    assert np.allclose(np.abs(T.data), 1.0), "all entries must have modulus 1"

    # columns are orthogonal, so T^dagger T is diagonal: 1 + number of slaves attached to that master
    G = (T.conj().T @ T).toarray()
    expected = 1 + np.bincount(np.searchsorted(ind, P["master"]), minlength=n_ind)
    assert np.allclose(G, np.diag(expected)), "T^dagger T should be diagonal"

    # Bloch relation on a random independent vector
    rng = np.random.default_rng(0)
    q_i = rng.normal(size=n_ind) + 1j * rng.normal(size=n_ind)
    q = T @ q_i
    ph = {"x": np.exp(1j * kx * LX), "y": np.exp(1j * ky * LY), "xy": np.exp(1j * (kx * LX + ky * LY))}
    for s, m, k in zip(P["slave"], P["master"], P["kind"]):
        assert np.isclose(q[s], ph[k] * q[m])
    assert np.allclose(q[ind], q_i)               # independent DOFs are passed through unchanged

    # k = 0: a uniform deflection (w = 1, theta = 0) must be in the range of T
    T0 = build_T(P, 0.0, 0.0, LX, LY)
    q_w = np.zeros(n_total, dtype=complex)
    q_w[parent[comp == 0]] = 1.0
    assert np.allclose(T0 @ q_w[ind], q_w)
    print("build_T checks passed")