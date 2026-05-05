"""
Behavior Cloning with base + wrist RGB (and proprio).

Expects .npz from lab2/utils/convert_all_episodes_npz_for_bc.py:

  - states, actions, images[, wrist_images], image_size, ...

For proprio-only BC use scripts/bc.py instead.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load_data_from_stacked_episodes(
    states,
    actions,
    images,
    wrists,
    H,
    test_frac=0.2,
    seed=0,
):
    """
    Build train/test dicts from per-episode stacked arrays (same layout as bc_train.npz).

    Args:
        states:  (E,) object, each (T, state_dim)
        actions: (E,) object, each (T, act_dim)
        images:  (E,) object, each (T, h, w, 3) uint8, or None
        wrists:  (E,) object, each (T, h, w, 3) uint8, or None
    """
    assert len(states) == len(actions)
    has_image = images is not None
    has_wrist = wrists is not None
    E = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)

    n_test = int(test_frac * E)
    if E > 1 and n_test == 0:
        n_test = 1
    test_eps = perm[:n_test]
    train_eps = perm[n_test:]

    def _stack_history(seq, H, t):
        window = seq[t - H + 1 : t + 1]
        if window.ndim == 4:
            return np.concatenate([window[i] for i in range(H)], axis=-1)
        return window.reshape(-1)

    def flatten(eps, H):
        S, A, I, W = [], [], [], []
        for ei in eps:
            s = np.asarray(states[ei], dtype=np.float32)
            a = np.asarray(actions[ei], dtype=np.float32)
            T = s.shape[0]
            if T < H:
                continue
            for t in range(H - 1, T):
                S.append(_stack_history(s, H, t))
                A.append(a[t])
                if has_image:
                    img_seq = np.asarray(images[ei])
                    I.append(_stack_history(img_seq, H, t))
                if has_wrist:
                    wrist_seq = np.asarray(wrists[ei])
                    W.append(_stack_history(wrist_seq, H, t))
        out = {
            "state": np.asarray(S, dtype=np.float32),
            "action": np.asarray(A, dtype=np.float32),
        }
        if has_image and I:
            arr = np.asarray(I, dtype=np.uint8)
            out["image"] = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32) / 255.0
        if has_wrist and W:
            arr = np.asarray(W, dtype=np.uint8)
            out["wrist"] = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32) / 255.0
        return out

    return flatten(train_eps, H), flatten(test_eps, H)


def load_data_by_episode(path, H, test_frac=0.2, seed=0):
    """
    train, test = each dict with keys state, action, and optionally image, wrist.
    state: (N, H*state_dim); images: (N, C, H, W) float [0,1].
    """
    data = np.load(path, allow_pickle=True)
    states = data["states"]
    actions = data["actions"]
    images = data["images"] if "images" in data.files else None
    wrists = data["wrist_images"] if "wrist_images" in data.files else None

    return load_data_from_stacked_episodes(
        states, actions, images, wrists, H, test_frac=test_frac, seed=seed
    )


def compute_norm_stats(X, eps=1e-8):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.maximum(std, eps)
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


class _DictDataset(Dataset):
    def __init__(self, payload):
        self.payload = payload
        self.n = len(payload["state"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {
            "state": torch.from_numpy(self.payload["state"][i]),
            "action": torch.from_numpy(self.payload["action"][i]),
        }
        if "image" in self.payload:
            out["image"] = torch.from_numpy(self.payload["image"][i])
        if "wrist" in self.payload:
            out["wrist"] = torch.from_numpy(self.payload["wrist"][i])
        return out


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BCImagePolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        img_channels: int,
        wrist_channels: int,
        hidden=(256, 256),
        feat_dim: int = 128,
    ):
        super().__init__()
        self.has_image = img_channels > 0
        self.has_wrist = wrist_channels > 0
        self.base_enc = _ImageEncoder(img_channels, feat_dim) if self.has_image else None
        self.wrist_enc = (
            _ImageEncoder(wrist_channels, feat_dim) if self.has_wrist else None
        )
        d_in = state_dim + (feat_dim if self.has_image else 0) + (
            feat_dim if self.has_wrist else 0
        )
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, act_dim)]
        self.head = nn.Sequential(*layers)

    def forward(self, state, image=None, wrist=None):
        feats = [state]
        if self.has_image:
            feats.append(self.base_enc(image))
        if self.has_wrist:
            feats.append(self.wrist_enc(wrist))
        return self.head(torch.cat(feats, dim=-1))


def _move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    if len(loader.dataset) == 0:
        return float("nan")
    mse, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model(
                batch["state"],
                image=batch.get("image"),
                wrist=batch.get("wrist"),
            )
            mse += (pred - batch["action"]).pow(2).sum().item()
            n += batch["action"].numel()
    return mse / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument(
        "--data",
        default="asset/bc_train.npz",
        help=".npz with states, actions, images[, wrist_images] (converter output).",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip", default=None)
    parser.add_argument("--out", default="asset/inf_image.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--inf_steps", type=int, default=10)
    parser.add_argument("--policy-path", default="asset/bc_image_policy.pt")
    parser.add_argument("--norm-path", default="asset/bc_image_norm.npz")
    args = parser.parse_args()

    if args.mode == "inference" and not args.ip:
        parser.error("--ip is required for inference mode")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train, test = load_data_by_episode(
            args.data,
            H=args.obs_horizon,
            test_frac=args.test_frac,
            seed=args.seed,
        )

        if len(train["state"]) == 0:
            raise RuntimeError("No training samples.")
        if "image" not in train:
            raise ValueError(
                "Dataset has no images; use bc.py for proprio-only or re-run "
                "convert_all_episodes_npz_for_bc.py."
            )

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
            f"Train: {len(train['state'])} | Test: {len(test['state'])} | "
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

    elif args.mode == "inference":
        raise NotImplementedError(
            "Image BC inference is not implemented here: add live camera capture "
            "(same preprocessing as training) and run BCImagePolicy forward. "
            "Proprio-only robot rollout: use bc.py --mode inference."
        )


if __name__ == "__main__":
    main()
