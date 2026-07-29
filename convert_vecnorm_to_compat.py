"""Re-save VecNormalize .pkl files in a numpy-version-agnostic format.

Root cause: host (numpy 2.x) pickled VecNormalize contains BitGenerator refs
that Docker (numpy<2.0, required for ROS2) cannot deserialize. We only need
the running-mean/std stats at inference, not the training random state.

This tool reads each source .pkl with the host's numpy, strips problematic
numpy.random.Generator fields, and re-saves with pickle protocol 4 (broadly
compatible). The rewritten file loads cleanly with numpy<2.0.

Usage:
    python3 convert_vecnorm_to_compat.py \
        lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_vecnormalize_180000_steps.pkl

    # Or sweep all MS600 seeds:
    python3 convert_vecnorm_to_compat.py --all-ms600
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent / "lightweight_train"

MS600_RUNS = {
    42:  "results_3d_shape_delta_800k_maxsteps600",
    100: "results_3d_shape_delta_ms600_s100",
    200: "results_3d_shape_delta_ms600_s200",
}
CKPT_STEPS = 180_000


_seen: set[int] = set()


def strip_incompatible_random_state(obj, path="root"):
    """Walk object attributes and replace numpy.random.Generator instances with None.
    SB3 VecNormalize doesn't use the stored RNG at inference (eval-only), so None is safe.
    Uses object.__dict__ directly to bypass VecNormalize's __getattr__ recursion trap.
    """
    if isinstance(obj, (np.random.Generator, np.random.BitGenerator)):
        return None
    if id(obj) in _seen:
        return obj
    if isinstance(obj, dict):
        return {k: strip_incompatible_random_state(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_incompatible_random_state(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(strip_incompatible_random_state(v, f"{path}[{i}]") for i, v in enumerate(obj))
    if isinstance(obj, (int, float, str, bool, bytes, type(None), np.ndarray)):
        return obj
    try:
        d = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return obj
    _seen.add(id(obj))
    for k, v in list(d.items()):
        new_v = strip_incompatible_random_state(v, f"{path}.{k}")
        if new_v is not v:
            d[k] = new_v
    return obj


def convert(src: Path, dst: Path | None = None) -> Path:
    if dst is None:
        dst = src.with_suffix(".compat.pkl")
    print(f"[convert] {src}")
    with open(src, "rb") as f:
        obj = pickle.load(f)
    _seen.clear()
    obj = strip_incompatible_random_state(obj)
    # SB3 VecNormalize.__getstate__ deletes venv/class_attributes/returns, but
    # __setstate__ only restores some of them. Re-seed the keys so the re-pickle
    # doesn't KeyError, then __getstate__ will strip them again on dump.
    try:
        d = object.__getattribute__(obj, "__dict__")
        if "venv" in d:
            d["venv"] = None
        if "_venv" in d:
            d["_venv"] = None
        if "class_attributes" not in d:
            d["class_attributes"] = {}
        if "returns" not in d:
            d["returns"] = np.zeros(1)
    except AttributeError:
        pass
    with open(dst, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"[convert]   -> {dst}  ({dst.stat().st_size} bytes)")
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="VecNormalize .pkl paths to convert")
    p.add_argument("--all-ms600", action="store_true",
                   help="Convert the selected MS600 vecnormalize files for all 3 seeds")
    args = p.parse_args()

    paths: list[Path] = []
    if args.all_ms600:
        for seed, rundir in MS600_RUNS.items():
            paths.append(LIGHTWEIGHT_DIR / rundir / "models" / f"slam3d_vecnormalize_{CKPT_STEPS}_steps.pkl")
    paths.extend(Path(p).resolve() for p in args.paths)

    if not paths:
        sys.exit("No paths given. Pass explicit .pkl paths or --all-ms600.")

    for src in paths:
        if not src.exists():
            print(f"[skip] {src} does not exist")
            continue
        convert(src)

    print("\nDone. In the Docker container, load with the .compat.pkl suffix.")


if __name__ == "__main__":
    main()
