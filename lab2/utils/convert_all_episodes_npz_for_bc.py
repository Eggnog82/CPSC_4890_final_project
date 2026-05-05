"""
Convert ULAS / GELLO demo bundles into the format expected by scripts/bc.py.

**Input (--in)** — either:

1. **`all_episodes.npz`** from `demo_to_npy_from_ulas.py` (key `episodes`), or
2. A **directory** containing **`episode_*.npy`** (one file per demo — same as `_npy/`
   before consolidation). Use this to **skip** the slow `all_episodes.npz` step.

**Output** — single `.npz` for `bc.py`:

  - states, actions, images, optional wrist_images, image_size, …

Δq = control[:7] - q (matches lab bc.py inference).

Each timestep dict must have keys `state`, `action`, `image`, and usually
`wrist_image`.
"""

import argparse
import re
from pathlib import Path

import numpy as np

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _resize_uint8_rgb(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize HxWx3 uint8 image to (target_h, target_w, 3)."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unexpected image shape {img.shape}")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    if _HAS_PIL:
        return np.asarray(
            Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR)
        )
    ys = (np.linspace(0, img.shape[0] - 1, target_h)).astype(np.int64)
    xs = (np.linspace(0, img.shape[1] - 1, target_w)).astype(np.int64)
    return img[ys][:, xs]


def _episode_to_arrays(episode_obj: np.ndarray, target_h: int, target_w: int):
    """episode_obj: 1D object array of dicts (one per timestep)."""
    states_out = []
    actions_out = []
    images_out = []
    wrist_out = []
    has_wrist = True

    for t in range(len(episode_obj)):
        row = episode_obj[t]
        if row is None or not isinstance(row, dict):
            continue
        s = np.asarray(row["state"], dtype=np.float64).ravel()
        ctrl = np.asarray(row["action"], dtype=np.float64).ravel()
        if s.size < 7 or ctrl.size < 8:
            raise ValueError(
                f"Need state dim>=7 and action dim>=8; got {s.size}, {ctrl.size}"
            )
        q = s[:7].copy()
        g = float(s[7]) if s.size >= 8 else 0.0
        dq = (ctrl[:7] - q).astype(np.float32)
        g_cmd = float(ctrl[7])

        img = row.get("image")
        if img is None:
            raise ValueError("Timestep missing 'image' key.")
        img_resized = _resize_uint8_rgb(img, target_h, target_w)

        wrist = row.get("wrist_image")
        if wrist is None:
            has_wrist = False
            wrist_resized = None
        else:
            wrist_resized = _resize_uint8_rgb(wrist, target_h, target_w)

        states_out.append(np.concatenate([q, [g]]).astype(np.float32))
        actions_out.append(np.concatenate([dq, [g_cmd]]).astype(np.float32))
        images_out.append(img_resized)
        if has_wrist:
            wrist_out.append(wrist_resized)

    if not states_out:
        return None

    return (
        np.stack(states_out),
        np.stack(actions_out),
        np.stack(images_out),
        np.stack(wrist_out) if has_wrist and wrist_out else None,
    )


def _episode_paths_from_dir(dir_path: Path) -> list[Path]:
    paths = sorted(
        Path(dir_path).glob("episode_*.npy"),
        key=lambda p: int(m.group(1))
        if (m := re.match(r"episode_(\d+)\.npy$", p.name))
        else p.name,
    )
    if not paths:
        raise FileNotFoundError(
            f"No episode_*.npy files in {dir_path}. "
            "Use the parent `_npy` folder from demo_to_npy_from_ulas."
        )
    return paths


def _load_episodes_list(path_in: Path) -> list[np.ndarray]:
    if path_in.is_dir():
        return [
            np.load(str(p), allow_pickle=True)
            for p in _episode_paths_from_dir(path_in)
        ]
    data = np.load(path_in, allow_pickle=True)
    if "episodes" not in data.files:
        raise KeyError(
            f"Expected key 'episodes' in {path_in} or pass a directory of "
            f"episode_*.npy; found: {list(data.files)}"
        )
    ep = data["episodes"]
    return [ep[i] for i in range(len(ep))]


def convert(
    path_in: Path,
    path_out: Path,
    image_size: int = 84,
) -> None:
    episodes = _load_episodes_list(path_in)

    states_list = []
    actions_list = []
    images_list = []
    wrist_list = []
    any_wrist = True

    for ei in range(len(episodes)):
        out = _episode_to_arrays(episodes[ei], image_size, image_size)
        if out is None:
            print(f"Skipping empty episode index {ei}")
            continue
        s_arr, a_arr, img_arr, wrist_arr = out
        states_list.append(s_arr)
        actions_list.append(a_arr)
        images_list.append(img_arr)
        if wrist_arr is None:
            any_wrist = False
        else:
            wrist_list.append(wrist_arr)

    if not states_list:
        raise ValueError("No valid episodes after conversion.")

    if not any_wrist and wrist_list:
        print("Some episodes missing wrist images; dropping wrist for the whole dataset.")
        wrist_list = []

    path_out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "states": np.array(states_list, dtype=object),
        "actions": np.array(actions_list, dtype=object),
        "images": np.array(images_list, dtype=object),
        "image_size": np.array(f"{image_size}x{image_size}"),
        "action_type": np.array("delta_joint_angles_joint_abs_gripper"),
        "unit": np.array("radians"),
    }
    if any_wrist and wrist_list:
        arrays["wrist_images"] = np.array(wrist_list, dtype=object)
    np.savez_compressed(path_out, **arrays)
    print(
        f"Wrote {path_out}: {len(states_list)} episodes, "
        f"images {image_size}x{image_size}, "
        f"wrist={'yes' if (any_wrist and wrist_list) else 'no'}."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--in", dest="path_in", type=Path, required=True)
    ap.add_argument(
        "--out", dest="path_out", type=Path, default=Path("asset/bc_train.npz")
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=84,
        help="Square resize side for both cameras (default 84).",
    )
    args = ap.parse_args()
    convert(args.path_in.expanduser(), args.path_out.expanduser(), args.image_size)


if __name__ == "__main__":
    main()
