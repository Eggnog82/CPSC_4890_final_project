# #!/usr/bin/env python3
# """Inspect a .npz file and report keys that look like image / camera arrays."""

# import argparse
# import re
# from pathlib import Path

# import numpy as np

# IMAGE_NAME_HINTS = re.compile(
#     r"rgb|bgr|image|camera|cam|depth|frame|pixel|photo|wrist|base|img|vision",
#     re.I,
# )


# def _is_likely_image_array(name: str, arr: np.ndarray) -> tuple[bool, str]:
#     """Return (is_likely, reason)."""
#     if not isinstance(arr, np.ndarray):
#         return False, "not an ndarray"

#     if arr.dtype == object and arr.size > 0:
#         elem = arr.flat[0]
#         if isinstance(elem, np.ndarray):
#             ok, sub = _is_likely_image_array(name, elem)
#             if ok:
#                 return True, f"object array; first cell {sub}"
#         sample = [arr.flat[i] for i in range(min(64, arr.size))]
#         if all(np.isscalar(x) and not isinstance(x, (str, bytes)) for x in sample):
#             return False, "object dtype with scalar cells (tabular — not image rasters)"
#         if sample and all(isinstance(x, np.ndarray) for x in sample[: min(8, len(sample))]):
#             first = sample[0]
#             ok, sub = _is_likely_image_array(name, first)
#             if ok:
#                 return True, f"object array of ndarrays: {sub}"
#             return False, f"object array of ndarrays, first {first.shape} {first.dtype}"

#     name_hit = bool(IMAGE_NAME_HINTS.search(name))

#     if arr.ndim == 2:
#         h, w = int(arr.shape[0]), int(arr.shape[1])
#         if h >= 8 and w >= 8 and max(h, w) <= 16384:
#             if arr.dtype in (np.uint8, np.uint16, np.float32, np.float64):
#                 return True, f"{arr.dtype} {arr.shape} (grayscale / 2D map)"
#     if arr.ndim == 3:
#         h, w, c = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
#         if h >= 8 and w >= 8 and c in (1, 3, 4) and max(h, w) <= 16384:
#             if arr.dtype in (np.uint8, np.uint16, np.float32, np.float64):
#                 return (
#                     True,
#                     f"{arr.dtype} {arr.shape} (HxWxC image)",
#                 )

#     if name_hit and arr.ndim >= 2:
#         return True, f"name hint + shape {arr.shape} {arr.dtype}"

#     return False, f"shape {arr.shape} {arr.dtype}"


# def main():
#     p = argparse.ArgumentParser(description=__doc__.strip())
#     p.add_argument("npz_path", type=Path, help="Path to .npz file")
#     args = p.parse_args()

#     path = args.npz_path.expanduser()
#     if not path.is_file():
#         raise SystemExit(f"Not a file: {path}")

#     data = np.load(path, allow_pickle=True)
#     print(f"File: {path.resolve()}")
#     print(f"Keys ({len(data.files)}): {', '.join(data.files)}\n")

#     likely = []
#     for key in data.files:
#         arr = data[key]
#         ok, reason = _is_likely_image_array(key, arr)
#         line = f"  {key}: {reason}"
#         print(line)
#         if ok:
#             likely.append(key)

#     print()
#     if likely:
#         print("Image-like keys found:", ", ".join(likely))
#     else:
#         print("No image-like keys detected (no HxW or HxWx{1,3,4} arrays with typical dtypes).")
#     print("\nTip: if images are nested in object arrays, load one row with pickle and inspect keys.")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""Inspect a .npy file and report entries that look like image / camera arrays."""

import argparse
import re
from pathlib import Path

import numpy as np

IMAGE_NAME_HINTS = re.compile(
    r"rgb|bgr|image|camera|cam|depth|frame|pixel|photo|wrist|base|img|vision",
    re.I,
)


def _is_likely_image_array(name: str, arr) -> tuple[bool, str]:
    """Return (is_likely, reason)."""
    if not isinstance(arr, np.ndarray):
        return False, f"not an ndarray: {type(arr)}"

    if arr.dtype == object and arr.size > 0:
        elem = arr.flat[0]

        if isinstance(elem, dict):
            hits = []
            for k, v in elem.items():
                ok, sub = _is_likely_image_array(str(k), v)
                if ok:
                    hits.append(f"{k}: {sub}")
            if hits:
                return True, "object array; first cell dict contains " + "; ".join(hits)

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
                return True, f"{arr.dtype} {arr.shape} (HxWxC image)"

    if name_hit and arr.ndim >= 2:
        return True, f"name hint + shape {arr.shape} {arr.dtype}"

    return False, f"shape {arr.shape} {arr.dtype}"


def inspect_item(name: str, item, indent: str = "  ") -> list[str]:
    """Print info recursively and return image-like names."""
    likely = []

    if isinstance(item, dict):
        print(f"{indent}{name}: dict with keys {list(item.keys())}")
        for k, v in item.items():
            likely.extend(inspect_item(str(k), v, indent + "  "))
        return likely

    if isinstance(item, np.ndarray):
        ok, reason = _is_likely_image_array(name, item)
        print(f"{indent}{name}: {reason}")
        if ok:
            likely.append(name)

        if item.dtype == object and item.size > 0 and isinstance(item.flat[0], dict):
            print(f"{indent}  inspecting first object-array element:")
            likely.extend(inspect_item(f"{name}[0]", item.flat[0], indent + "    "))

        return likely

    print(f"{indent}{name}: {type(item)}")
    return likely


def main():
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("npy_path", type=Path, help="Path to .npy file")
    args = p.parse_args()

    path = args.npy_path.expanduser()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    raw = np.load(path, allow_pickle=True)

    print(f"File: {path.resolve()}")
    print(f"Loaded object: {type(raw)}")

    # Common case: np.save("demo.npy", dict_obj)
    if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        obj = raw.item()
        likely = inspect_item("root", obj)
    else:
        likely = inspect_item("root", raw)

    print()
    if likely:
        print("Image-like entries found:", ", ".join(likely))
    else:
        print("No image-like entries detected.")
    print("\nTip: if the .npy stores a list/object array of demos, this script inspects the first nested object.")


if __name__ == "__main__":
    main()
