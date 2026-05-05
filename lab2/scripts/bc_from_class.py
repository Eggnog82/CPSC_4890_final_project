"""
Behavior Cloning (Student Version)

Students will fill in:
- dataset flattening (history stacking)
- normalization utilities
- model forward pass
- training step (loss/optimizer)
- inference: read state, build history, normalize input, predict, unnormalize, execute
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

from xarm_lab.arm_utils import (
    connect_arm, disconnect_arm, ArmConfig,
    get_joint_angles, get_tcp_pose, get_gripper_position
)
from utils.plot import plot_3d_positions
from xarm_lab.safety import enable_basic_safety, clear_faults
from xarm_lab.kinematics import ik_from_pose
# from utils.plot import plot_3d_positions

# -----------------------------
# Load + flatten dataset
# -----------------------------

def load_data_by_episode(path, H, test_frac=0.2, seed=0):
    """
    Loads an episode-structured dataset (.npz) and flattens it into supervised samples.

    Expected .npz format:
      - states:  (E,) object array, each item is (T, obs_dim)
      - actions: (E,) object array, each item is (T, act_dim)

    We create training pairs:
      x_t = concat([s_{t-H+1}, ..., s_t])   -> shape (H*obs_dim,)
      y_t = a_t                            -> shape (act_dim,)
    """
    data = np.load(path, allow_pickle=True)

    states = data["states"]    # (E,) object array
    actions = data["actions"]  # (E,) object arrayute normalization stats
# =====================================

    assert len(states) == len(actions)

    E = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)

    n_test = int(test_frac * E)
    test_eps = perm[:n_test]
    train_eps = perm[n_test:]

    def flatten(episode_indices, H):
        """
        TODO:
        - iterate over selected episodes
        - for each episode, for each time t >= H-1:
            X.append( s[t-H+1 : t+1].reshape(-1) )
            Y.append( a[t] )
        - skip episodes with T < H
        - return X, Y as float32 numpy arrays
        """
        X, Y = [], []

        # TODO: implement episode flattening with horizon H
        for i in episode_indices:
            s = states[i]
            a = actions[i]
            for t in range(H-1, len(s)):
                X.append( s[t-H+1 : t+1].reshape(-1) )
                Y.append( a[t] )
            np.asarray(Y, dtype=np.float32)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
        )

    X_train, Y_train = flatten(train_eps, H)
    X_test, Y_test   = flatten(test_eps, H)

    return X_train, Y_train, X_test, Y_test
    
# ============================================================
# TODO: Compute normalization stats
# ============================================================
def compute_norm_stats(X, eps=1e-8):
    """
    TODO:
    - compute mean and std over axis 0
    - clamp std with eps to avoid divide-by-zero
    - return (mean, std)
    """
    # TODO
    #raise NotImplementedError
    std = np.std(X,axis=0)
    #std = np.max(np.concatenate([std,eps*np.ones_like(std)]))
    return np.mean(X,axis=0),np.std(X,axis=0) + eps


def normalize(X, mean, std):
    """
    TODO:
    - return normalized X
    """
    # TODO
    #raise NotImplementedError
    return (X-mean)/std

# -----------------------------
# BC policy
# -----------------------------

class BCPolicy(nn.Module):
    """
    Student TODO:
    Implement a simple MLP policy mapping observations to actions
    Suggested:
        - Input layer: obs_dim
        - Hidden layers: 1-2 layers with ReLU
        - Output layer: act_dim
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # -------------------------------
        # TODO: define network layers
        self.net = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.ReLU(), 
            nn.Linear(obs_dim, obs_dim), 
            nn.ReLU(),
            nn.Linear(obs_dim, act_dim)
        )
        # -------------------------------

    def forward(self, x):
        """
        TODO (optional):
        - return self.net(x)
        """
        # TODO
        #raise NotImplementedError
        return self.net(x)

# -----------------------------
# Train / eval
# -----------------------------

def evaluate(model, loader, device):
    """
    Compute mean squared error (MSE) over a dataset.

    Student TODO:
    Fill in the loss calculation only. The forward pass is provided.
    """
    model.eval()
    mse, n = 0.0, 0

    with torch.no_grad():

        loss = nn.MSELoss()

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # TODO: forward pass
            pred = model(x)

            # TODO: accumulate sum of squared error
            mse += loss(pred,y)
            n += 1
            #raise NotImplementedError

    return mse / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--data", default="asset/demo.npz")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip", default = "192.168.1.205")
    parser.add_argument("--out", default="asset/inf.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--inf_steps", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    Xtr, Ytr, Xte, Yte = load_data_by_episode(
        args.data,
        H=args.obs_horizon,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # Normalize using TRAIN statistics only
    # TODO: compute (X_mean, X_std) from Xtr
    # TODO: compute (Y_mean, Y_std) from Ytr
    X_mean, X_std = compute_norm_stats(Xtr)# TODO
    Y_mean, Y_std = compute_norm_stats(Ytr) # TODO

    # TODO: normalize Xtr and Xte using X_mean/X_std
    # TODO: normalize Ytr and Yte using Y_mean/Y_std
    Xtr = normalize(Xtr, X_mean, X_std)
    Ytr = normalize(Ytr, Y_mean, Y_std)
    Xte = normalize(Xte, X_mean, X_std)
    Yte = normalize(Yte, Y_mean, Y_std)
    #raise NotImplementedError

    if args.mode == "train":

        print(f"Train samples: {len(Xtr)} | Test samples:  {len(Xte)}")

        train_ds = TensorDataset(
            torch.from_numpy(Xtr), torch.from_numpy(Ytr)
        )
        test_ds = TensorDataset(
            torch.from_numpy(Xte), torch.from_numpy(Yte)
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        # Model
        model = BCPolicy(obs_dim=Xtr.shape[1], act_dim=Ytr.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        # Train
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # TODO:
                pred = model(x)
                loss = loss_fn(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #raise NotImplementedError


            if ep % 5 == 0 or ep == 1:
                train_mse = evaluate(model, train_loader, device)
                test_mse = evaluate(model, test_loader, device)
                print(
                    f"Epoch {ep:03d} | "
                    f"Train MSE: {train_mse:.6f} | "
                    f"Test MSE: {test_mse:.6f}"
                )

        # Save model
        # TODO: save model weights to asset/bc_policy.pt
        #raise NotImplementedError
        PATH = "asset/bc_policy.pt" 
        torch.save(model.state_dict(), PATH)

        # Save normalization stats
        # TODO: save X_mean, X_std, Y_mean, Y_std to asset/bc_norm.npz
        #raise NotImplementedError
        np.savez('asset/bc_norm.npz', a=X_mean, b=X_std, c=Y_mean, d=Y_std)

        print("Model and normalization saved.")

    elif args.mode == "inference":

        # Load model
        model = BCPolicy(obs_dim=Xtr.shape[1], act_dim=Ytr.shape[1]).to(device)

        # TODO: load weights from asset/bc_policy.pt
        #raise NotImplementedError
        checkpoint = torch.load('asset/bc_policy.pt', weights_only=True)
        model.load_state_dict(checkpoint)

        model.eval()

        # Load normalization
        # TODO: load bc_norm.npz and set X_mean, X_std, Y_mean, Y_std
        #raise NotImplementedError
        data = np.load('asset/bc_norm.npz')
        X_mean = data['a']
        X_std = data['b']
        Y_mean = data['c']
        Y_std = data['d']

        arm = connect_arm(ArmConfig(ip=args.ip))

        ep_states_list = []
        ep_actions_list = []

        try:
            clear_faults(arm)
            enable_basic_safety(arm)

            # Gripper setup (kept as-is for safety/consistency)
            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(5000)

            print("\n=== Pick-and-Place (BC Inference) ===")

            for ep in range(args.episodes):

                # Home / randomize start (kept as-is)
                code, initial_joints = arm.get_initial_point()
                arm.set_servo_angle(
                    angle=initial_joints,
                    speed=20.0,
                    wait=True,
                    is_radian=False
                )
                pose = get_tcp_pose(arm)
                pose[:3] += np.random.uniform(-50, 50, size=3)
                joint_angles = ik_from_pose(arm, pose)
                arm.set_servo_angle(
                    angle=joint_angles,
                    speed=20.0,
                    wait=True,
                    is_radian=True
                )
                arm.set_gripper_position(600, wait=True, speed=0.1)

                print(f"Episode {ep+1}: robot homed to {np.asarray(pose, dtype=float)}")

                states = []
                actions = []
                eefs = []

                obs_buffer = deque(maxlen=args.obs_horizon)
                for t in range(args.inf_steps):  # fixed horizon (safety)

                    # ---- Read state ----
                    # TODO:
                    # - q = get_joint_angles(arm)
                    # - g = get_gripper_position(arm)
                    # - state = np.concatenate([q, [g]])  (or whatever your obs definition is)
                    # - eef_state = get_tcp_pose(arm)
                    q = get_joint_angles(arm)          # TODO
                    g = get_gripper_position(arm)          # TODO
                    state = np.concatenate([q, [g]])      # TODO
                    eef_state = get_tcp_pose(arm)  # TODO
                    # TODO: obs_buffer.append(state)
                    # TODO: if len(obs_buffer) < obs_horizon: continue
                    #raise NotImplementedError
                    obs_buffer.append(state)
                    if len(obs_buffer) < args.obs_horizon: continue

                    # ---- Stack + normalize ----
                    # TODO:
                    # - obs_stack = np.concatenate(list(obs_buffer), axis=0)
                    # - x = (obs_stack - X_mean) / X_std
                    # - x = torch.tensor(x, dtype=torch.float32).to(device)
                    #raise NotImplementedError
                    obs_stack = np.concatenate(list(obs_buffer), axis=0)
                    x = (obs_stack - X_mean) / X_std
                    x = torch.tensor(x, dtype=torch.float32).to(device)

                    # ---- Predict ----
                    # TODO:
                    # with torch.no_grad():
                    #   a_norm = model(x).cpu().numpy()
                    #raise NotImplementedError
                    with torch.no_grad():
                        a_norm = model(x).cpu().numpy()

                    # ---- Unnormalize ----
                    # TODO:
                    # action = a_norm * Y_std + Y_mean
                    # dq = action[:7]
                    # dg = ...
                    #raise NotImplementedErrorpython scripts/bc.py --mode train --ip <robot_ip> --epochs <epochs> --batch_size <batch_size> --lr <lr>
                    action = a_norm * Y_std + Y_mean
                    dq = action[:7]
                    dg = action[-1]

                    # ---- Execute ----
                    # TODO:
                    arm.set_servo_angle(angle=(q + dq).tolist(), speed=0.5, wait=True, is_radian=True)
                    arm.set_gripper_position(dg, wait=True, speed=0.1)

                    states.append(state)
                    actions.append(action)
                    eefs.append(eef_state)

                ep_states_list.append(np.asarray(states, dtype=np.float32))
                ep_actions_list.append(np.asarray(actions, dtype=np.float32))

                plot_3d_positions(np.array(eefs)[:, :3])

            # Save inference rollouts
            np.savez(
                args.out,
                states=np.array(ep_states_list, dtype=object),
                actions=np.array(ep_actions_list, dtype=object),
                action_type="delta_joint_angles",
                unit="radians"
            )

            print(f"\nDataset saved to {args.out}")

        finally:
            disconnect_arm(arm)


if __name__ == "__main__":

    main()


