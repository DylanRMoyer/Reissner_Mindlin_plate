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
                  probe_mobility=sweep.probe_mobility, settings=np.array(settings_blob))
    if getattr(sweep, "w_max_plate", None) is not None:
        arrays["w_max_plate"] = sweep.w_max_plate
    np.savez(path, **arrays)


def load_sweep(path):
    with np.load(path, allow_pickle=False) as data:
        return SimpleNamespace(
            f_values=data["f_values"], rms_velocity=data["rms_velocity"],
            probe_mobility=data["probe_mobility"],
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