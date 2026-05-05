"""
BC training from a folder of `episode_*.npy` files (ULAS / demo_to_npy output).

Each `.npy` is a 1D object array of dicts with keys:
  image, wrist_image, state, action, ...

No intermediate `bc_train.npz` required. Same model and checkpoints as bc_image.py.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.convert_all_episodes_npz_for_bc import (
    _episode_paths_from_dir,
    _episode_to_arrays,
)

# Load bc_image from same directory (avoid `scripts` package requirement).
_bc_spec = importlib.util.spec_from_file_location(
    "_bc_image", Path(__file__).resolve().parent / "bc_image.py"
)
_bc = importlib.util.module_from_spec(_bc_spec)
_bc_spec.loader.exec_module(_bc)

BCImagePolicy = _bc.BCImagePolicy
_DictDataset = _bc._DictDataset
_move_batch = _bc._move_batch
compute_norm_stats = _bc.compute_norm_stats
evaluate = _bc.evaluate
load_data_from_stacked_episodes = _bc.load_data_from_stacked_episodes
normalize = _bc.normalize


def folder_to_stacked_episodes(npy_dir: Path, image_size: int):
    """
    Load every episode_*.npy under npy_dir and return (states, actions, images, wrists)
    as object arrays matching bc_train.npz layout.
    """
    paths = _episode_paths_from_dir(npy_dir)
    states_list, actions_list, images_list, wrist_list = [], [], [], []
    any_wrist = True

    for p in paths:
        ep = np.load(str(p), allow_pickle=True)
        out = _episode_to_arrays(ep, image_size, image_size)
        if out is None:
            print(f"Skipping empty episode {p.name}")
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
        raise ValueError(f"No usable episodes in {npy_dir}")
    if not any_wrist:
        wrist_list = []

    images_obj = np.array(images_list, dtype=object)
    wrists_obj = np.array(wrist_list, dtype=object) if wrist_list else None
    return (
        np.array(states_list, dtype=object),
        np.array(actions_list, dtype=object),
        images_obj,
        wrists_obj,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--npy-dir",
        type=Path,
        required=True,
        help="Folder containing episode_0.npy, episode_1.npy, ...",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=84,
        help="Square resize for cameras (must match converter / bc_image).",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-horizon", type=int, default=1)
    parser.add_argument("--policy-path", default="asset/bc_image_policy.pt")
    parser.add_argument("--norm-path", default="asset/bc_image_norm.npz")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npy_dir = args.npy_dir.expanduser().resolve()
    states, actions, images, wrists = folder_to_stacked_episodes(
        npy_dir, args.image_size
    )

    train, test = load_data_from_stacked_episodes(
        states,
        actions,
        images,
        wrists,
        H=args.obs_horizon,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    if len(train["state"]) == 0:
        raise RuntimeError("No training samples after flattening.")
    if "image" not in train:
        raise ValueError("Expected images in episodes; check NPZ conversion / demos.")

    S_mean, S_std = compute_norm_stats(train["state"])
    A_mean, A_std = compute_norm_stats(train["action"])
    train["state"] = normalize(train["state"], S_mean, S_std).astype(np.float32)
    train["action"] = normalize(train["action"], A_mean, A_std).astype(np.float32)
    if len(test["state"]) > 0:
        test["state"] = normalize(test["state"], S_mean, S_std).astype(np.float32)
        test["action"] = normalize(test["action"], A_mean, A_std).astype(np.float32)

    state_dim = train["state"].shape[1]
    act_dim = train["action"].shape[1]
    img_channels = train["image"].shape[1]
    wrist_channels = train["wrist"].shape[1] if "wrist" in train else 0

    print(
        f"Dir: {npy_dir} | Episodes: {len(states)} | "
        f"Train samples: {len(train['state'])} | Test: {len(test['state'])} | "
        f"img_ch={img_channels} wrist_ch={wrist_channels}"
    )

    train_loader = DataLoader(
        _DictDataset(train), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        _DictDataset(test), batch_size=args.batch_size, shuffle=False
    )

    model = BCImagePolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        img_channels=img_channels,
        wrist_channels=wrist_channels,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            opt.zero_grad()
            loss = loss_fn(
                model(batch["state"], batch.get("image"), batch.get("wrist")),
                batch["action"],
            )
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == 1:
            tr = evaluate(model, train_loader, device)
            te = evaluate(model, test_loader, device)
            te_s = f"{te:.6f}" if not np.isnan(te) else "n/a"
            print(f"Epoch {ep:03d} | Train MSE: {tr:.6f} | Test: {te_s}")

    os.makedirs(os.path.dirname(args.policy_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.policy_path)
    np.savez(
        args.norm_path,
        X_mean=S_mean,
        X_std=S_std,
        Y_mean=A_mean,
        Y_std=A_std,
        obs_horizon=args.obs_horizon,
        obs_dim_single=state_dim // args.obs_horizon,
        act_dim=act_dim,
        img_channels=img_channels,
        wrist_channels=wrist_channels,
    )
    print(f"Saved {args.policy_path} and {args.norm_path}")


if __name__ == "__main__":
    main()
