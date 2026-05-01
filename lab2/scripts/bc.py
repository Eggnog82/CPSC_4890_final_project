"""
Behavior Cloning — lab implementation with history stacking, normalization, MLP, train/infer.
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xarm_lab.arm_utils import (
    connect_arm,
    disconnect_arm,
    ArmConfig,
    get_joint_angles,
    get_tcp_pose,
    get_gripper_position,
)
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose
from utils.plot import plot_3d_positions


def load_data_by_episode(path, H, test_frac=0.2, seed=0):
    """
    Loads an episode-structured dataset (.npz) and flattens it into supervised samples.

    Expected .npz format:
      - states:  (E,) object array, each item is (T, obs_dim)
      - actions: (E,) object array, each item is (T, act_dim)

    Training pairs:
      x_t = concat([s_{t-H+1}, ..., s_t])   -> shape (H*obs_dim,)
      y_t = a_t                            -> shape (act_dim,)
    """
    data = np.load(path, allow_pickle=True)

    states = data["states"]
    actions = data["actions"]

    assert len(states) == len(actions)

    E = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)

    n_test = int(test_frac * E)
    if E > 1 and n_test == 0:
        n_test = 1
    test_eps = perm[:n_test]
    train_eps = perm[n_test:]

    def flatten(episode_indices, H):
        X, Y = [], []
        for ei in episode_indices:
            s = np.asarray(states[ei], dtype=np.float64)
            a = np.asarray(actions[ei], dtype=np.float64)
            t_len = s.shape[0]
            if t_len < H:
                continue
            for t in range(H - 1, t_len):
                hist = s[t - H + 1 : t + 1].reshape(-1)
                X.append(hist)
                Y.append(a[t])
        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
        )

    X_train, Y_train = flatten(train_eps, H)
    X_test, Y_test = flatten(test_eps, H)

    return X_train, Y_train, X_test, Y_test


def compute_norm_stats(X, eps=1e-8):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.maximum(std, eps)
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True)])
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    if len(loader.dataset) == 0:
        return float("nan")
    mse, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse += (pred - y).pow(2).sum().item()
            n += y.numel()
    return mse / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--data", default="asset/demo.npz")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip", default=None, help="xArm controller IP (required for inference)")
    parser.add_argument("--out", default="asset/inf.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--inf_steps", type=int, default=10)
    parser.add_argument("--policy-path", default="asset/bc_policy.pt")
    parser.add_argument("--norm-path", default="asset/bc_norm.npz")
    args = parser.parse_args()

    if args.mode == "inference" and not args.ip:
        parser.error("--ip is required for inference mode")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        Xtr, Ytr, Xte, Yte = load_data_by_episode(
            args.data,
            H=args.obs_horizon,
            test_frac=args.test_frac,
            seed=args.seed,
        )

        if len(Xtr) == 0:
            raise RuntimeError(
                "No training samples — check --data, --obs_horizon, and that episodes are long enough."
            )

        X_mean, X_std = compute_norm_stats(Xtr)
        Y_mean, Y_std = compute_norm_stats(Ytr)

        Xtr_n = normalize(Xtr, X_mean, X_std)
        Xte_n = normalize(Xte, X_mean, X_std) if len(Xte) > 0 else Xte
        Ytr_n = normalize(Ytr, Y_mean, Y_std)
        Yte_n = normalize(Yte, Y_mean, Y_std) if len(Yte) > 0 else Yte

        obs_dim = Xtr.shape[1]
        act_dim = Ytr.shape[1]

        print(f"Train samples: {len(Xtr)} | Test samples: {len(Xte)}")

        train_ds = TensorDataset(
            torch.from_numpy(Xtr_n),
            torch.from_numpy(Ytr_n),
        )
        test_ds = TensorDataset(
            torch.from_numpy(Xte_n),
            torch.from_numpy(Yte_n),
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        model = BCPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

            if ep % 5 == 0 or ep == 1:
                train_mse = evaluate(model, train_loader, device)
                test_mse = evaluate(model, test_loader, device)
                ts = f"{test_mse:.6f}" if not np.isnan(test_mse) else "n/a"
                print(
                    f"Epoch {ep:03d} | "
                    f"Train MSE (norm): {train_mse:.6f} | "
                    f"Test MSE (norm): {ts}"
                )

        os.makedirs(os.path.dirname(args.policy_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.policy_path)
        np.savez(
            args.norm_path,
            X_mean=X_mean,
            X_std=X_std,
            Y_mean=Y_mean,
            Y_std=Y_std,
            obs_horizon=args.obs_horizon,
            obs_dim_single=Xtr.shape[1] // args.obs_horizon,
            act_dim=act_dim,
        )

        print(f"Saved policy to {args.policy_path}, norms to {args.norm_path}")

    elif args.mode == "inference":

        norms = np.load(args.norm_path)
        X_mean = norms["X_mean"]
        X_std = norms["X_std"]
        Y_mean = norms["Y_mean"]
        Y_std = norms["Y_std"]
        obs_dim = X_mean.shape[0]
        act_dim = Y_mean.shape[0]
        saved_h = int(norms["obs_horizon"])
        if saved_h != args.obs_horizon:
            print(
                f"Warning: --obs_horizon={args.obs_horizon} != saved {saved_h}; using saved value."
            )
            args.obs_horizon = saved_h

        model = BCPolicy(obs_dim=obs_dim, act_dim=act_dim).to(device)
        model.load_state_dict(torch.load(args.policy_path, map_location=device))
        model.eval()

        arm = connect_arm(ArmConfig(ip=args.ip))

        ep_states_list = []
        ep_actions_list = []

        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(5000)

            print("\n=== Pick-and-Place (BC Inference) ===")

            for ep in range(args.episodes):

                code, initial_joints = arm.get_initial_point()
                arm.set_servo_angle(
                    angle=initial_joints,
                    speed=20.0,
                    wait=True,
                    is_radian=False,
                )
                pose = get_tcp_pose(arm)
                pose[:3] += np.random.uniform(-5, 5, size=3)
                joint_angles = ik_from_pose(arm, pose)
                arm.set_servo_angle(
                    angle=joint_angles,
                    speed=20.0,
                    wait=True,
                    is_radian=True,
                )
                arm.set_gripper_position(600, wait=True, speed=0.1)

                print(f"Episode {ep+1}: robot homed to {np.asarray(pose, dtype=float)}")

                states = []
                actions = []
                eefs = []

                obs_buffer = deque(maxlen=args.obs_horizon)

                for t in range(args.inf_steps):

                    q = get_joint_angles(arm)
                    g = float(get_gripper_position(arm))
                    state = np.concatenate([q, [g]]).astype(np.float32)
                    eef_state = get_tcp_pose(arm)

                    obs_buffer.append(state)
                    if len(obs_buffer) < args.obs_horizon:
                        continue

                    obs_stack = np.concatenate(list(obs_buffer), axis=0).astype(np.float32)
                    x = (obs_stack - X_mean) / X_std
                    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        a_norm = model(x_t).cpu().numpy().reshape(-1)

                    action = a_norm * Y_std + Y_mean
                    dq = action[:7]
                    grip_cmd = float(action[7])

                    arm.set_servo_angle(
                        angle=(q + dq).tolist(),
                        speed=0.5,
                        wait=False,
                        is_radian=True,
                    )
                    arm.set_gripper_position(grip_cmd, wait=False, speed=0.1)

                    states.append(state)
                    actions.append(action.copy())
                    eefs.append(eef_state)

                ep_states_list.append(np.asarray(states, dtype=np.float32))
                ep_actions_list.append(np.asarray(actions, dtype=np.float32))

                plot_3d_positions(np.array(eefs)[:, :3])

            np.savez(
                args.out,
                states=np.array(ep_states_list, dtype=object),
                actions=np.array(ep_actions_list, dtype=object),
                action_type="delta_joint_angles",
                unit="radians",
            )

            print(f"\nDataset saved to {args.out}")

        finally:
            disconnect_arm(arm)


if __name__ == "__main__":
    main()
