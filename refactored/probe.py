import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from point_coupling import locate_target_cell_degrees_of_freedom


@dataclass
class Probe:
    x: float
    y: float
    phi: np.ndarray
    local_to_global_w: np.ndarray

def make_probe(domain, function_space, x, y):
    phi, _, local_to_global_w = locate_target_cell_degrees_of_freedom(domain, function_space, SimpleNamespace(x=x, y=y))
    return Probe(x=x, y=y, phi=phi, local_to_global_w=local_to_global_w)

def evaluate_w_at_probe(w_point, probe):
    """
    Note that w_point must be the collapsed w-subfunction, as local_to_global_w uses w-space numbering. Also,
    probes are tied to mesh and function space. When changing either, recompute the probe(s).
    Result is deliberately complex.
    """
    w_at_target = np.dot(probe.phi, w_point.x.array[probe.local_to_global_w])
    return w_at_target

def compute_mobility(w_value, Omega, force_amplitude):
    mobility = 1j * Omega * w_value / force_amplitude
    return mobility