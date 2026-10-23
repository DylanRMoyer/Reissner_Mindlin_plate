import numpy as np
from scipy.linalg import eigh
from config import PlateConfig
from problem_setup import build_plate_problem
from weak_form import define_weak_form
from solve import assemble_plate_matrix
from augmented_system import petsc_to_scipy
from periodic_bc import build_pairs, build_T

LX, LY, NX, NY = 1.0, 1.0, 4, 4

cfg = PlateConfig(length=LX, width=LY, thickness=0.02, nx=NX, ny=NY, rho=7850, mu=77e9, lambda_=115e9)
domain, V, _, prob = build_plate_problem(cfg, deg=2, el_type="Lagrange")
m, _, a = define_weak_form(V, prob)

# free plate: no Dirichlet BCs, so the diag argument is irrelevant
K = petsc_to_scipy(assemble_plate_matrix(a, [])).tocsr()
M = petsc_to_scipy(assemble_plate_matrix(m, [])).tocsr()

P = build_pairs(V, LX, LY)
assert K.shape[0] == len(P["table"][0]), "K and the DOF table disagree on the number of DOFs"


def reduced(kx, ky):
    T = build_T(P, kx, ky, LX, LY)
    Th = T.conj().T
    return Th @ K @ T, Th @ M @ T


def herm_err(A):
    return abs(A - A.conj().T).max() / abs(A).max()


def spectrum(kx, ky):
    Kr, Mr = reduced(kx, ky)
    return eigh(Kr.toarray(), Mr.toarray(), eigvals_only=True)   # complex Hermitian, M_r positive definite


# 1. Hermiticity at a generic k
kg = (0.7 * np.pi / LX, 0.4 * np.pi / LY)
Kr, Mr = reduced(*kg)
print(f"Hermiticity error K_r: {herm_err(Kr):.2e}, M_r: {herm_err(Mr):.2e}")
assert herm_err(Kr) < 1e-10 and herm_err(Mr) < 1e-10

# 2. Gamma: two zero modes expected: rigid translation w = const and the Q2 hourglass mode
Kr0, Mr0 = reduced(0.0, 0.0)
ev0, V0 = eigh(Kr0.toarray(), Mr0.toarray())
print("lowest omega^2 at Gamma:", ev0[:4])
assert abs(ev0[0]) < 1e-6 * ev0[2] and abs(ev0[1]) < 1e-6 * ev0[2], "expected two near-zero modes at Gamma"
assert ev0[2] > 0, "third eigenvalue should be clearly positive"

# both null vectors must carry no rotation (theta = 0), only w
parent, xy, comp, cat = P["table"]
q0 = build_T(P, 0.0, 0.0, LX, LY) @ V0[:, :2]
w_norm = np.linalg.norm(q0[parent[comp == 0]], axis=0)
th_norm = np.linalg.norm(q0[parent[comp != 0]], axis=0)
print("w norm / theta norm of the two null vectors:", w_norm, th_norm)
assert (th_norm < 1e-6 * w_norm).all(), "null vectors have rotation: not the rigid + hourglass pair"

# 3. generic k: no zero mode
ev_k = spectrum(*kg)
print("lowest omega^2 at generic k:", ev_k[:4])
assert ev_k[0] > 0

# 4. reciprocity: omega(k) = omega(-k)
ev_mk = spectrum(-kg[0], -kg[1])
assert np.allclose(ev_k, ev_mk, rtol=1e-8), "spectrum at k and -k differ: check the phase sign convention"
print("all reduction checks passed")