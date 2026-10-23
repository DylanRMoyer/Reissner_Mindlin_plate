# probe_dof_coordinates.py
import numpy as np
from config import PlateConfig
from problem_setup import build_plate_problem
from weak_form import collapse_subspace

NX = NY = 4
L = W = 1.0

def inspect(el_type):
    cfg = PlateConfig(length=L, width=W, thickness=0.02, nx=NX, ny=NY,
                      rho=7850, mu=77e9, lambda_=115e9)
    domain, V, _, _ = build_plate_problem(cfg, deg=2, el_type=el_type)
    hx, hy = L / NX, W / NY
    n_vertex_edge_pts = (2 * NX + 1) * (2 * NY + 1)
    expected_pts = n_vertex_edge_pts - NX * NY if el_type == "S" else n_vertex_edge_pts

    print(f"\n===== el_type = {el_type} =====")
    for name, idx in (("w", 0), ("theta", 1)):
        Vs, _ = collapse_subspace(V, idx)
        bs = Vs.dofmap.index_map_bs
        print(f"[{name}] nodes={Vs.dofmap.index_map.size_global}, block size={bs}")

        try:
            be = Vs.element.basix_element
            per_dim = [[len(d) for d in dim] for dim in be.entity_dofs]
            print("  entity_dofs (vertices / edges / interior):", per_dim)
        except Exception as e:
            print("  entity_dofs unavailable:", type(e).__name__, e)

        try:
            X = Vs.tabulate_dof_coordinates()
        except Exception as e:
            print(f"  tabulate_dof_coordinates FAILED -> {type(e).__name__}: {e}")
            continue

        xy = X[:, :2]
        n_unique = len(np.unique(np.round(xy, 10), axis=0))
        on_half_grid = np.allclose(xy[:, 0] / (hx / 2), np.round(xy[:, 0] / (hx / 2)), atol=1e-8) and \
                       np.allclose(xy[:, 1] / (hy / 2), np.round(xy[:, 1] / (hy / 2)), atol=1e-8)
        print(f"  tabulate_dof_coordinates OK: shape {X.shape}, unique points {n_unique} "
              f"(expected {expected_pts}), on half-grid: {on_half_grid}")

for t in ("S", "Lagrange"):
    inspect(t)