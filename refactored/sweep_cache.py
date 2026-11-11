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
