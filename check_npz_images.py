#!/usr/bin/env python3
"""Inspect a .npz file and report keys that look like image / camera arrays."""

import argparse
import re
from pathlib import Path

import numpy as np

IMAGE_NAME_HINTS = re.compile(
    r"rgb|bgr|image|camera|cam|depth|frame|pixel|photo|wrist|base|img|vision",
    re.I,
)


def _is_likely_image_array(name: str, arr: np.ndarray) -> tuple[bool, str]:
    """Return (is_likely, reason)."""
    if not isinstance(arr, np.ndarray):
        return False, "not an ndarray"

    if arr.dtype == object and arr.size > 0:
        elem = arr.flat[0]
        if isinstance(elem, np.ndarray):
            ok, sub = _is_likely_image_array(name, elem)
            if ok:
                return True, f"object array; first cell {sub}"
        sample = [arr.flat[i] for i in range(min(64, arr.size))]
        if all(np.isscalar(x) and not isinstance(x, (str, bytes)) for x in sample):
            return False, "object dtype with scalar cells (tabular — not image rasters)"
        if sample and all(isinstance(x, np.ndarray) for x in sample[: min(8, len(sample))]):
            first = sample[0]
            ok, sub = _is_likely_image_array(name, first)
            if ok:
                return True, f"object array of ndarrays: {sub}"
            return False, f"object array of ndarrays, first {first.shape} {first.dtype}"

    name_hit = bool(IMAGE_NAME_HINTS.search(name))

    if arr.ndim == 2:
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h >= 8 and w >= 8 and max(h, w) <= 16384:
            if arr.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                return True, f"{arr.dtype} {arr.shape} (grayscale / 2D map)"
    if arr.ndim == 3:
        h, w, c = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        if h >= 8 and w >= 8 and c in (1, 3, 4) and max(h, w) <= 16384:
            if arr.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                return (
                    True,
                    f"{arr.dtype} {arr.shape} (HxWxC image)",
                )

    if name_hit and arr.ndim >= 2:
        return True, f"name hint + shape {arr.shape} {arr.dtype}"

    return False, f"shape {arr.shape} {arr.dtype}"


def main():
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("npz_path", type=Path, help="Path to .npz file")
    args = p.parse_args()

    path = args.npz_path.expanduser()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    data = np.load(path, allow_pickle=True)
    print(f"File: {path.resolve()}")
    print(f"Keys ({len(data.files)}): {', '.join(data.files)}\n")

    likely = []
    for key in data.files:
        arr = data[key]
        ok, reason = _is_likely_image_array(key, arr)
        line = f"  {key}: {reason}"
        print(line)
        if ok:
            likely.append(key)

    print()
    if likely:
        print("Image-like keys found:", ", ".join(likely))
    else:
        print("No image-like keys detected (no HxW or HxWx{1,3,4} arrays with typical dtypes).")
    print("\nTip: if images are nested in object arrays, load one row with pickle and inspect keys.")


if __name__ == "__main__":
    main()
