from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime
import json
import platform
import sys
import time
import numpy as np


def make_run_dir(*, root: str | Path, run_name: str | None = None) -> Path:
    """
    Create a run directory under `root`.

    If run_name is None -> timestamped folder name like:
      2026-02-11_16-34-20
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # fail if exists
    return run_dir


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """
    Save arrays into a compressed NPZ.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _jsonify(obj):
    """
    Convert common scientific objects into JSON-serializable form.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _jsonify(asdict(obj))
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    return str(obj)


def save_manifest(
    *,
    path: str | Path,
    config: dict,
    outputs: dict,
    timings: dict,
    extra: dict | None = None,
) -> Path:
    """
    Save a manifest.json with enough metadata to reproduce a run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "config": _jsonify(config),
        "outputs": _jsonify(outputs),
        "timings": _jsonify(timings),
        "extra": _jsonify(extra or {}),
    }

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


class Timer:
    """
    Simple timing context.
    Usage:
        with Timer() as t:
            ...
        elapsed = t.elapsed_s
    """
    def __enter__(self):
        self._t0 = time.perf_counter()
        self.elapsed_s = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_s = time.perf_counter() - self._t0
        return False  # don't suppress exceptions
