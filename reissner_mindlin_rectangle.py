import pyvista
from dolfinx import mesh, fem, plot
from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
import basix
from slepc4py import SLEPc
from petsc4py import PETSc
from scipy.sparse import csr_matrix, coo_matrix


length = 1.1
width = 1
thickness = 0.02
rho = 7850
mu = 77e9
lambda_ = 115e9

norm_to_one = True


domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([length,width])], (60,50), mesh.CellType.quadrilateral)



# The uniform loading $f$ is scaled by the Love-Kirchhoff solution so that the deflection converges to a
# constant value of 1 in the thin plate. This thin plate limit will be used to check the sensitivity of the element to shear locking.
# Formulations which exhibit shear locking are expected to provide overly stiff results (low values of the deflection) for a given mesh in the limit of
# small thickness $h \to 0$.


# Material parameters for isotropic linear elastic behavior are first defined:

nu = fem.Constant(domain, lambda_/(2*(lambda_ + mu)))
E = fem.Constant(domain, mu*(3*lambda_ + 2*mu)/(mu + lambda_))

# Plate bending stiffness $\textsf{D}=\dfrac{Eh^3}{12(1-\nu^2)}$ and shear stiffness $\textsf{F} = \kappa G h$
# with a shear correction factor $\kappa = 5/6$ for a homogeneous plate of thickness $h$:

thick = fem.Constant(domain, thickness)
D = E * thick**3 / (1 - nu**2) / 12.0
F = E / 2 / (1 + nu) * thick * 5.0 / 6.0

f = fem.Constant(domain, 1.0)

deg = 2
el_type = "S"  # or "Q"
We = basix.ufl.element(el_type, domain.basix_cell(), deg)
Te = basix.ufl.element(el_type, domain.basix_cell(), deg, shape=(2,))

#function_space_w = fem.functionspace(domain, el_type, deg)

function_space = fem.functionspace(domain, basix.ufl.mixed_element([We, Te]))

# degree = print(function_space.ufl_element().degree)  # gives you the degree of the function space


function_space_w, w_to_parent = function_space.sub(0).collapse()   # w-subspace, with its own dof map
w_to_parent = np.asarray(w_to_parent).reshape(-1)
function_space_theta, _ = function_space.sub(1).collapse()   # theta-subspace, with its own dof map

#print(w_to_parent)

# --- Implementing point mass ---

# Point mass parameters

target_point_mass_x = 0.137
target_point_mass_y = 0.912
target_point_vector = np.array([[target_point_mass_x, target_point_mass_y, 0.0]])

point_stiffness = 100 # Newton per meter, keeping it SI
point_mass = 1 # kilogram, also keeping it SI

point_eigenfrequency = np.sqrt(point_stiffness/point_mass)/(2*np.pi)
#print(f"Eigenfrequency of point mass is: {point_eigenfrequency:.2e} Hz")

# --- Find cell containing the chosen point

bounding_box_tree = dolfinx.geometry.bb_tree(domain,2)
possible_boxes = dolfinx.geometry.compute_collisions_points(bounding_box_tree, target_point_vector)
#print(possible_boxes)
target_cells = dolfinx.geometry.compute_colliding_cells(domain, possible_boxes, target_point_vector)
#print(target_cells)

# Step A — take one definite cell
cell = target_cells.links(0)[0]
#print("Found cell number:", cell)

# Step B — pull the physical target point back into this cell's reference coordinates
cmap = domain.geometry.cmaps[0]
#print(domain.geometry.dofmaps)
geom_dofs = np.array(domain.geometry.dofmaps[0][cell])
#print(geom_dofs)
cell_geom = np.array(domain.geometry.x[geom_dofs])
x_ref = cmap.pull_back(target_point_vector[:, :domain.geometry.dim], cell_geom)
print("x_ref =", x_ref)

# Step C — evaluate the w-subspace's local basis functions at that reference point
basix_element_w = function_space_w.element.basix_element
tab = basix_element_w.tabulate(0, x_ref)      # 0th derivative = values only
phi = tab[0, 0, :, 0]                         # one weight per local dof of this cell

# Step D — local dofs of this cell (in w-space numbering) -> parent (mixed-space) numbering
local_to_global_w = function_space_w.dofmap.cell_dofs(cell)
#print(local_to_global_w)
global_dofs_parent = w_to_parent[local_to_global_w]

print("basis weights at target point:", phi)
print("parent dofs receiving the point load:", global_dofs_parent)


# Clamped boundary conditions on the lateral boundary are defined as:

# +
# Boundary of the plate
def border(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], length)),
        np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], width)),
    )


facet_dim = 1
clamped_facets = mesh.locate_entities_boundary(domain, facet_dim, border)
clamped_dofs = fem.locate_dofs_topological(function_space, facet_dim, clamped_facets)
#print(clamped_facets)

u0_w = fem.Function(function_space_w)   # zero by default — clamped displacement
u0_t = fem.Function(function_space_theta)   # zero by default — clamped rotation

dofs_w = fem.locate_dofs_topological((function_space.sub(0), function_space_w), facet_dim, clamped_facets)
dofs_t = fem.locate_dofs_topological((function_space.sub(1), function_space_theta), facet_dim, clamped_facets)

#print(len(dofs_w[0]))
#print(len(dofs_t))

bc_w = fem.dirichletbc(u0_w, dofs_w, function_space.sub(0))
bc_t = fem.dirichletbc(u0_t, dofs_t, function_space.sub(1))

bcs = [bc_w, bc_t]



# Some useful functions for implementing generalized constitutive relations are now
# defined:

# +
def strain2voigt(eps):
    return ufl.as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])


def voigt2stress(S):
    return ufl.as_tensor([[S[0], S[2]], [S[2], S[1]]])


def curv(u):
    (w, theta) = ufl.split(u)
    return ufl.sym(ufl.grad(theta))

def extract_theta(u):
    (w, theta) = ufl.split(u)
    return theta


def shear_strain(u):
    (w, theta) = ufl.split(u)
    return ufl.grad(w) - theta


def bending_moment(u):
    DD = ufl.as_tensor([[D, nu * D, 0], [nu * D, D, 0], [0, 0, D * (1 - nu) / 2.0]])
    return voigt2stress(ufl.dot(DD, strain2voigt(curv(u))))


def shear_force(u):
    return F * shear_strain(u)


# -

# The contribution of shear forces to the total energy is under-integrated using
# a custom quadrature rule of degree $2d-2$ i.e. for linear ($d=1$)
# quadrilaterals, the shear energy is integrated as if it were constant (1 Gauss point instead of 2x2)
# and for quadratic ($d=2$) quadrilaterals, as if it were quadratic (2x2 Gauss points instead of 3x3).
#
# ```{see also}
# See the [](/tips/quadrature_schemes/quadrature_schemes.md) tour for more details on the choice of quadrature points.
# ```

# +
u = fem.Function(function_space)
u_ = ufl.TestFunction(function_space)
du = ufl.TrialFunction(function_space)


dx = ufl.Measure("dx")
dx_shear = ufl.Measure("dx", metadata={"quadrature_degree": 2 * deg - 2})

m = rho * thick * ufl.inner(u_[0], du[0]) * dx + rho * thick**3/12 * ufl.inner(extract_theta(u_), extract_theta(du)) * dx

L = f * u_[0] * dx
a = (
        ufl.inner(bending_moment(u_), curv(du)) * dx
        + ufl.dot(shear_force(u_), shear_strain(du)) * dx_shear
)

# Static problem for point load

spring_end_displacement = 1.0e3   # starting displacement of the point mass

bilinear_form = fem.form(a)
A = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
A.assemble()

# --- 9.5: add the spring's rank-1 stiffness contribution ---
for i, dof_i in enumerate(global_dofs_parent):
    for j, dof_j in enumerate(global_dofs_parent):
        A.setValue(dof_i, dof_j, point_stiffness * phi[i] * phi[j],
                   addv=PETSc.InsertMode.ADD_VALUES)
A.assemblyBegin()
A.assemblyEnd()

# start from a zero RHS (or fem.form(L) if you want the uniform pressure too — your call)
b = A.createVecRight()
b.zeroEntries()

for phi_i, parent_dof in zip(phi, global_dofs_parent):
    b.setValue(parent_dof, phi_i * point_stiffness * spring_end_displacement, addv=PETSc.InsertMode.ADD_VALUES)
b.assemblyBegin()
b.assemblyEnd()

fem.petsc.apply_lifting(b, [bilinear_form], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")

u_point = fem.Function(function_space)
solver.solve(b, u_point.x.petsc_vec)
u_point.x.scatter_forward()

w_point = u_point.sub(0).collapse()
print(f"Point-load deflection at target: {max(abs(w_point.x.array)):.6e}")


# We then solve for the solution and print the deflection normalized with respect to the Love-Kirchhoff thin plate
# analytical solution:

# +
problem = fem.petsc.LinearProblem(
    a, L, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="Reissner-Mindlin"
)
problem.solve()


w = u.sub(0).collapse()
w.name = "Deflection"

print(f"Reissner-Mindlin FE deflection: {max(abs(w.x.array)):.5f}")


# --- Solve EVP ---

K = fem.petsc.assemble_matrix(fem.form(a), bcs = bcs, diag=1e10)
K.assemble()

M = fem.petsc.assemble_matrix(fem.form(m), bcs = bcs)
M.assemble()


# --- helpers: PETSc <-> scipy round-trip ---

def petsc_to_scipy(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.getSize())

def scipy_to_petsc(A_sp):
    A_sp = A_sp.tocsr()
    A_petsc = PETSc.Mat().createAIJ(size=A_sp.shape,
                                     csr=(A_sp.indptr, A_sp.indices, A_sp.data))
    A_petsc.assemble()
    return A_petsc

# --- Build the augmented (n+1) x (n+1) system ---

K_sp = petsc_to_scipy(K)      # your existing Day-8 K (plate only, diag=1e10 fix included)
M_sp = petsc_to_scipy(M)      # your existing Day-8 M (plate only)
n = K_sp.shape[0]

# extended coefficient vector, length n+1, mostly zero
phi_ext = np.zeros(n + 1)
phi_ext[global_dofs_parent] = phi
phi_ext[n] = -1.0             # q_r's own coordinate, index n (0-indexed -> the (n+1)-th dof)

# nonzero indices only, for a sparse outer product (phi_ext is almost entirely zero)
nz = np.nonzero(phi_ext)[0]
rows = np.repeat(nz, len(nz))
cols = np.tile(nz, len(nz))
vals = point_stiffness * np.outer(phi_ext[nz], phi_ext[nz]).flatten()

K_spring = coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()

# embed original K, M into (n+1)x(n+1), then add the spring stiffness and point mass
from scipy.sparse import bmat

K_aug = bmat([[K_sp, None],
              [None, csr_matrix([[0.0]])]], format="csr") + K_spring

M_aug = bmat([[M_sp, None],
              [None, csr_matrix([[point_mass]])]], format="csr")

"""""
# 1. q_r's own diagonal should be exactly k_r (mass m_r), nothing else touches it
print("K_aug[n,n] =", K_aug[n, n], " expected:", point_stiffness)
print("M_aug[n,n] =", M_aug[n, n], " expected:", point_mass)

# 2. coupling entries should be -k_r * phi_i, symmetric
i0 = global_dofs_parent[0]
print("K_aug[i0, n] =", K_aug[i0, n], " expected:", -point_stiffness * phi[0])
print("K_aug[n, i0] =", K_aug[n, i0], " should match the line above")

# 3. plate-plate block should equal original K plus the outer product, e.g. at (i0, i0)
print("K_aug[i0, i0] =", K_aug[i0, i0], " expected:", K_sp[i0, i0] + point_stiffness * phi[0]**2)

# 4. mass matrix has zero coupling anywhere in the last row/column except the diagonal
print("M_aug row n (should be all zero except last entry):",
      M_aug[n, :].toarray())
"""

K = scipy_to_petsc(K_aug)
M = scipy_to_petsc(M_aug)


eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(K, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.0)  # target near zero -- lowest frequencies

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(0.0)
eps.setDimensions(nev=6)  # how many eigenpairs to converge
eps.solve()

# create PETSc vectors matching K's layout (real and imaginary parts)
vr, vi = K.createVecs()

eigenmodes = []

for i in range(eps.getConverged()):
    eigval = eps.getEigenpair(i, vr, vi)  # fills vr, vi; returns eigenvalue
    omega_sq = eigval.real
    freq_hz = np.sqrt(omega_sq) / (2*np.pi)
    print(f"mode {i}: {freq_hz:.4f} Hz")

    plate_part = vr.getArray()[:n]
    q_r_value = vr.getArray()[n]

    mode_function = fem.Function(function_space)
    mode_function.x.petsc_vec.setArray(plate_part)
    mode_function.x.scatter_forward()  # sync ghost values (matters in parallel)

    eigenmodes.append((mode_function, q_r_value))

mode_to_plot, q_r_to_plot = eigenmodes[3]
w_mode = mode_to_plot.sub(0).collapse()   # scalar w-part of this eigenmode, still on Serendipity space

V_plot = fem.functionspace(domain, ("Lagrange", deg))
w_plot = fem.Function(V_plot)
w_plot.interpolate(w_mode)

topology, cell_types, geometry = plot.vtk_mesh(V_plot)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["w"] = w_plot.x.array

max_w = np.max(np.abs(w_plot.x.array))
target_visual_amplitude = 0.05 * length
factor_scale = target_visual_amplitude / max_w
print(f"suggested warp factor: {factor_scale}")

warped = grid.warp_by_scalar("w", factor=factor_scale)

# --- Rotation field (theta) overlay, on top of the warped w-surface ---

theta_mode = mode_to_plot.sub(1).collapse()   # vector theta-part of this eigenmode

V_plot_vec = fem.functionspace(function_space.mesh, ("Lagrange", deg, (2,)))
theta_plot = fem.Function(V_plot_vec)
theta_plot.interpolate(theta_mode)

# theta is a 2-component in-plane field; pad with a zero z-component so
# PyVista (which wants 3D vectors) can glyph it, and so we can lay the
# arrows flat/tangent at each warped surface point.
theta_vals = theta_plot.x.array.reshape((-1, 2))
theta_3d = np.zeros((theta_vals.shape[0], 3))
theta_3d[:, 0:2] = theta_vals

# attach to the *warped* grid so arrows sit at the already-deformed points
warped["theta"] = theta_3d

glyphs = warped.glyph(orient="theta", scale="theta", factor=factor_scale * 0.1)
# factor here is a separate visual scale from the w-warp factor —
# adjust the *5 multiplier to taste if arrows are too small/large

p = pyvista.Plotter()

attachment_xy = np.array([target_point_mass_x, target_point_mass_y])
plate_w_at_target = np.dot(phi, w_mode.x.array[local_to_global_w])  # w_h at target, this mode

mass_marker = pyvista.PolyData(np.array([[
    attachment_xy[0],
    attachment_xy[1],
    factor_scale * q_r_to_plot
]]))

p.add_mesh(mass_marker, color="blue", point_size=15, render_points_as_spheres=True)

# a line from the plate surface to the mass, i.e. the spring itself
spring_line = pyvista.Line(
    pointa=[attachment_xy[0], attachment_xy[1], factor_scale * plate_w_at_target],
    pointb=[attachment_xy[0], attachment_xy[1], factor_scale * q_r_to_plot]
)
p.add_mesh(spring_line, color="blue", line_width=3)

p.add_mesh(warped, scalars="w", show_edges=True)
p.add_mesh(glyphs, color="red")
p.show_axes()
import os
#p.view_isometric()
p.view_vector((2,2,-1))
p.show(auto_close=False)
#save_path = os.path.expanduser("~/PycharmProjects/Plots/rm_plate_onemass_bottom_view.pdf")
#p.save_graphic(save_path)
p.close()
