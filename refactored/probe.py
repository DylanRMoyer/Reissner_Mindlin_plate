import numpy as np
import warnings
from types import SimpleNamespace
from dataclasses import dataclass
from shaker_force import ShakerParameters
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

def dense_at_end_open(start, stop, num, ratio: float=4):
    """
    Generates 'num' points between 'start' and 'stop' (exclusive),
    concentrated densely towards 'stop'.
    Ratio == 1 returns an evenly spaced linspace.
    Ratios between 0 and 1 concentrate densely towards 'start'.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if np.isclose(ratio, 1.0):
        return np.linspace(start, stop, num + 2)[1:-1]
    g = np.geomspace(ratio, 1, num + 2)
    g_norm = (ratio - g) / (ratio - 1)
    return (start + g_norm * (stop - start))[1:-1]

def distance_to_plate_edge(length, width, x0, y0, dx, dy):
    """Distance along the unit direction (dx, dy) from (x0, y0) to the plate boundary."""
    candidates = []
    if dx > 1e-12:  candidates.append((length - x0) / dx)
    if dx < -1e-12: candidates.append((0.0 - x0) / dx)
    if dy > 1e-12:  candidates.append((width - y0) / dy)
    if dy < -1e-12: candidates.append((0.0 - y0) / dy)
    return min(candidates)

def ray_probe_coordinates(length, width, x0, y0, angle, amount, ratio):
    """Probes on a straight ray from (x0, y0) at 'angle', ending before the plate edge.
    Returns (coords, s_max); coords is None if the ray has no length inside the plate."""
    dx, dy = np.cos(angle), np.sin(angle)
    s_max = distance_to_plate_edge(length, width, x0, y0, dx, dy)
    if s_max <= 1e-12:
        return None, 0.0
    s_values = dense_at_end_open(0.0, s_max, amount, ratio)
    coords = [(float(x0 + s * dx), float(y0 + s * dy)) for s in s_values]
    return coords, float(s_max)


def probe_clearances(coords, plate_config, shaker_config, vamm_list=None):
    """All (distance_in_elements, description) pairs a probe layout should keep large:
    distance to the symmetry lines, plate edges, the shaker, VAMMs and other probes."""
    L, W = plate_config.length, plate_config.width
    hx, hy = L / plate_config.nx, W / plate_config.ny

    def dist(p, q):
        return float(np.hypot((p[0] - q[0]) / hx, (p[1] - q[1]) / hy))

    out = []
    for i, (x, y) in enumerate(coords):
        tag = f"probe {i} ({x:.3f}, {y:.3f})"
        out.append((abs(x - L / 2) / hx, f"{tag} is close to the symmetry line x = L/2"))
        out.append((abs(y - W / 2) / hy, f"{tag} is close to the symmetry line y = W/2"))
        out.append((min(x, L - x) / hx, f"{tag} is close to a left/right plate edge"))
        out.append((min(y, W - y) / hy, f"{tag} is close to a top/bottom plate edge"))
        out.append((dist((x, y), (shaker_config.x, shaker_config.y)),
                    f"{tag} is close to the shaker"))
        for v in (vamm_list or []):
            out.append((dist((x, y), (v.x, v.y)),
                        f"{tag} is close to the VAMM at ({v.x:.3f}, {v.y:.3f})"))
        for j in range(i + 1, len(coords)):
            out.append((dist((x, y), coords[j]), f"{tag} is close to probe {j}"))
    return out


def warn_if_probes_poorly_placed(coords, plate_config, shaker_config,
                                 vamm_list=None, min_clearance: float = 2.0):
    """Warn (never raise) about every placement closer than min_clearance element sizes."""
    for d, msg in probe_clearances(coords, plate_config, shaker_config, vamm_list):
        if d < min_clearance:
            warnings.warn(f"{msg} ({d:.1f} elements, threshold {min_clearance:g}).",
                          stacklevel=3)


def build_probe_coordinates(plate_config, shaker_config, amount: int = 5,
                            probe_coordinates=None, vamm_list=None,
                            min_clearance: float = 2.0, target_clearance: float = 3.0,
                            angles_deg=range(0, 360, 5), ratios=(1.0, 2.0, 4.0)):
    """Probe coordinates for the transfer-mobility sweep.

    Default: search straight rays from the shaker (all angles in angles_deg, point
    spacings in ratios) and keep the layout with the largest clearance (capped at
    target_clearance element sizes) from symmetry lines, edges, shaker, VAMMs and
    other probes; ties go to the longer ray, i.e. the larger span in distance.
    Override: pass probe_coordinates=[(x, y), ...] to use your own points ('amount'
    is then ignored). Poor placements only warn; points outside the plate raise.
    """
    if probe_coordinates is not None:
        coords = [(float(x), float(y)) for x, y in probe_coordinates]
        for x, y in coords:
            if not (0 <= x <= plate_config.length and 0 <= y <= plate_config.width):
                raise ValueError(f"probe ({x}, {y}) lies outside the plate")
        warn_if_probes_poorly_placed(coords, plate_config, shaker_config,
                                     vamm_list, min_clearance)
        return coords

    if amount < 1:
        raise ValueError(f"amount must be at least 1, got {amount}")

    best = None  # (score, coords, angle, ratio, clearance)
    for angle_deg in angles_deg:
        for ratio in ratios:
            coords, s_max = ray_probe_coordinates(
                plate_config.length, plate_config.width, shaker_config.x, shaker_config.y,
                np.deg2rad(angle_deg), amount, ratio)
            if coords is None:
                continue
            clearance = min(d for d, _ in probe_clearances(
                coords, plate_config, shaker_config, vamm_list))
            score = (min(clearance, target_clearance), s_max)
            if best is None or score > best[0]:
                best = (score, coords, angle_deg, ratio, clearance)

    if best is None:
        raise ValueError("no valid ray from the shaker found; pass probe_coordinates manually")

    _, coords, angle_deg, ratio, clearance = best
    print(f"Probe layout: {amount} probes on a ray at {angle_deg} deg, ratio {ratio:g}, "
          f"min clearance {clearance:.1f} elements")
    warn_if_probes_poorly_placed(coords, plate_config, shaker_config, vamm_list, min_clearance)
    return coords