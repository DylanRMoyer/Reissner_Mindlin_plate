from dataclasses import dataclass
from petsc4py import PETSc
from dolfinx import fem
from point_coupling import locate_target_cell_degrees_of_freedom

# Point force shaker parameters

@dataclass
class ShakerParameters:
    x: float
    y: float
    force_amplitude: float
    phase: float = 0

def assemble_load_vector_through_force(A,
                                       form,
                                       phi,
                                       global_dofs_parent,
                                       force_amplitude,
                                       bcs = None):
    """Build the RHS vector representing a force displacement."""
    b = A.createVecRight()
    b.zeroEntries()

    for phi_i, parent_dof in zip(phi, global_dofs_parent):
        b.setValue(parent_dof, phi_i * force_amplitude,
                   addv=PETSc.InsertMode.ADD_VALUES)
    b.assemblyBegin()
    b.assemblyEnd()

    if bcs:
        fem.petsc.apply_lifting(b, [form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    if bcs:
        fem.petsc.set_bc(b, bcs)
    return b

def make_force_excitation_rhs_builder(domain, function_space, shaker_parameters):
    phi, global_dofs_parent, _ = locate_target_cell_degrees_of_freedom(
        domain, function_space, shaker_parameters
    )
    def force_excitation_rhs_builder(A, form, bcs=None):
        return assemble_load_vector_through_force(
        A=A, form=form, phi=phi, global_dofs_parent=global_dofs_parent,
            force_amplitude=shaker_parameters.force_amplitude, bcs=bcs
        )
    return force_excitation_rhs_builder
