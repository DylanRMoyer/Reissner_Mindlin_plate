import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "results"


def make_cache_key(**settings):
    """Short hash of everything that determines a sweep's result. Dataclasses are
    expanded field by field and arrays are hashed by content, so changing any
    setting gives a different key (= a different cache file)."""
    def default(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, np.ndarray):
            return {"__array_sha1__": hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest()}
        return str(obj)
    blob = json.dumps(settings, sort_keys=True, default=default)
    return hashlib.sha1(blob.encode()).hexdigest()[:10], blob


def save_sweep(path, sweep, settings_blob=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(f_values=sweep.f_values, rms_velocity=sweep.rms_velocity,
                  w_field=sweep.w_field, settings=np.array(settings_blob))
    if getattr(sweep, "w_max_plate", None) is not None:
        arrays["w_max_plate"] = sweep.w_max_plate
    np.savez(path, **arrays)


def load_sweep(path):
    with np.load(path, allow_pickle=False) as data:
        return SimpleNamespace(
            f_values=data["f_values"], rms_velocity=data["rms_velocity"],
            w_field=data["w_field"],
            w_max_plate=data["w_max_plate"] if "w_max_plate" in data.files else None)


def get_or_compute_sweep(path, compute, settings_blob=""):
    """Load the sweep from 'path' if it exists, otherwise call compute() and save it."""
    path = Path(path)
    if path.exists():
        print(f"Loading cached sweep: {path.name}")
        return load_sweep(path)
    print(f"No cache at {path.name}; running sweep ...")
    sweep = compute()
    save_sweep(path, sweep, settings_blob)
    return sweep

# --- Radiated power cache (Rayleigh integral over the radiation points, ~2.4 s per frequency on 100x100) ---

def save_radiated_power(path, f_values, radiated_power, settings_blob=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, f_values=f_values, radiated_power=radiated_power, settings=np.array(settings_blob))


def load_radiated_power(path):
    with np.load(path, allow_pickle=False) as data:
        return data["f_values"], data["radiated_power"]


def get_or_compute_radiated_power(path, f_values, compute, settings_blob=""):
    """Load the radiated power P(f) from 'path' if it exists, otherwise call compute() and save it.
    Same pattern as get_or_compute_sweep. The key should contain the key of the sweep it was computed from
    (that fixes plate, mesh, radiation points, grid, excitation and resonators) plus the air constants;
    f_values is stored and checked on loading as a guard against a mismatched file."""
    path = Path(path)
    if path.exists():
        print(f"Loading cached radiated power: {path.name}")
        cached_f_values, radiated_power = load_radiated_power(path)
        if not np.array_equal(cached_f_values, np.asarray(f_values)):
            raise ValueError(f"{path.name}: cached frequencies differ from the sweep; delete the file")
        return radiated_power
    print(f"No cache at {path.name}; computing radiated power ...")
    radiated_power = compute()
    save_radiated_power(path, np.asarray(f_values), radiated_power, settings_blob)
    return radiated_power


# --- Eigenfrequency cache (frequencies only; 500 modes on 100x100 take ~10-20 min, the mode shapes ~1 GB) ---

def get_or_compute_eigenfrequencies(path, compute, settings_blob=""):
    """Load the eigenfrequencies (Hz) from 'path' if it exists, otherwise call compute() and save them.
    Same pattern as get_or_compute_sweep; compute() must return the frequencies only."""
    path = Path(path)
    if path.exists():
        print(f"Loading cached eigenfrequencies: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            return data["eigenfrequencies"]
    print(f"No cache at {path.name}; solving eigenproblem ...")
    eigenfrequencies = np.asarray(compute())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, eigenfrequencies=eigenfrequencies, settings=np.array(settings_blob))
    return eigenfrequencies


# --- Dispersion cache (Bloch bands along a k-path, ~25 s per diagram; eigenvectors are not stored) ---

def save_dispersion(path, result, settings_blob=""):
    """Store a DispersionResult: path, sorted frequencies, resonator share and (if present) the tracked tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        wavevectors=result.path.wavevectors, path_coordinate=result.path.path_coordinate,
        symmetry_point_positions=np.asarray(result.path.symmetry_point_positions, dtype=float),
        symmetry_point_labels=np.array(result.path.symmetry_point_labels, dtype=str),
        frequencies=result.frequencies,
        maximum_frequency_hz=np.array(np.nan if result.maximum_frequency_hz is None else result.maximum_frequency_hz),
        settings=np.array(settings_blob))
    if result.resonator_share is not None:
        arrays["resonator_share"] = result.resonator_share
    if result.tracking is not None:
        arrays.update(tracking_labels=result.tracking.labels, tracking_frequencies=result.tracking.frequencies,
                      tracking_plate_share=result.tracking.plate_share,
                      tracking_match_quality=result.tracking.match_quality,
                      tracking_step_status=np.array(result.tracking.step_status, dtype=str))
    np.savez(path, **arrays)


def load_dispersion(path):
    """Inverse of save_dispersion; eigenvectors come back as None. Imports inside, so this module stays usable
    without the complex FEniCSx environment."""
    from dispersion import DispersionResult, WavevectorPath
    from band_tracking import TrackedBands
    with np.load(path, allow_pickle=False) as data:
        wavevector_path = WavevectorPath(
            wavevectors=data["wavevectors"], path_coordinate=data["path_coordinate"],
            symmetry_point_positions=data["symmetry_point_positions"],
            symmetry_point_labels=[str(label) for label in data["symmetry_point_labels"]])
        maximum_frequency_hz = float(data["maximum_frequency_hz"])
        tracking = None
        if "tracking_labels" in data.files:
            tracking = TrackedBands(
                labels=data["tracking_labels"], frequencies=data["tracking_frequencies"],
                plate_share=data["tracking_plate_share"], match_quality=data["tracking_match_quality"],
                step_status=[str(status) for status in data["tracking_step_status"]])
        return DispersionResult(
            path=wavevector_path, frequencies=data["frequencies"], eigenvectors=None,
            maximum_frequency_hz=None if np.isnan(maximum_frequency_hz) else maximum_frequency_hz,
            resonator_share=data["resonator_share"] if "resonator_share" in data.files else None,
            tracking=tracking)


def get_or_compute_dispersion(path, compute, settings_blob=""):
    """Load the DispersionResult from 'path' if it exists, otherwise call compute() and save it. Same pattern as
    get_or_compute_sweep. The key should contain the cell config (material, size, mesh), element type and degree,
    the wavevectors, the frequency limit (or band count), the resonator list, track_bands and the tracking
    threshold, plus a code version. A run that needs eigenvectors must bypass the cache."""
    path = Path(path)
    if path.exists():
        print(f"Loading cached dispersion: {path.name}")
        return load_dispersion(path)
    print(f"No cache at {path.name}; running dispersion sweep ...")
    result = compute()
    save_dispersion(path, result, settings_blob)
    return result
