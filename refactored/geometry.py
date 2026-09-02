from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np



def create_rectangular_plate_mesh(length, width, nx, ny):
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([length, width])], (nx, ny),
                                   mesh.CellType.quadrilateral)
    return domain



