"""
Convert a flat GELLO-style dataset.npz into episode .npz format for lab2/scripts/bc.py.

Input (typical): joint_positions, gripper_position, control — shapes (N, ...)
Output: states (E,) object, actions (E,) object — each episode (T, 8).

State  = [q_0..q_6, gripper_reading]
Action = [Δq_0..Δq_6, gripper_command_abs]
where Δq = control[:7] - q (expert absolute command minus current joint).
"""

import argparse
from pathlib import Path

import numpy as np


def _row(arr, i: int) -> np.ndarray:
    return np.asarray(arr[i], dtype=np.float64).ravel()


def convert(
    path_in: Path,
    path_out: Path,
    chunk_size: int = 280,
) -> None:
    data = np.load(path_in, allow_pickle=True)
    required = ("joint_positions", "gripper_position", "control")
    for k in required:
        if k not in data.files:
            raise KeyError(f"Missing key {k} in {path_in}; valid: {data.files}")

    jp = data["joint_positions"]
    gp = data["gripper_position"]
    ctrl = data["control"]
    n = len(jp)
    if n == 0:
        raise ValueError("Empty dataset")

    q = np.stack([_row(jp, i)[:7] for i in range(n)])
    g = np.empty(n, dtype=np.float64)
    for i in range(n):
        ri = _row(gp, i).ravel()
        g[i] = float(ri[0]) if ri.size else 0.0
    states_mat = np.column_stack([q, g]).astype(np.float32)

    ctrl_mat = np.stack([_row(ctrl, i) for i in range(n)])
    if ctrl_mat.shape[1] < 8:
        raise ValueError(f"control must have at least 8 columns, got {ctrl_mat.shape}")
    dq = (ctrl_mat[:, :7] - q).astype(np.float32)
    g_cmd = ctrl_mat[:, 7].astype(np.float32)
    actions_mat = np.column_stack([dq, g_cmd]).astype(np.float32)

    episodes_states = []
    episodes_actions = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        episodes_states.append(states_mat[start:end])
        episodes_actions.append(actions_mat[start:end])

    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path_out,
        states=np.array(episodes_states, dtype=object),
        actions=np.array(episodes_actions, dtype=object),
        action_type="delta_joint_angles_joint_abs_gripper",
        unit="radians",
    )
    print(
        f"Wrote {path_out} with {len(episodes_states)} episodes, "
        f"{n} total steps, chunk_size={chunk_size}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="path_in",
        type=Path,
        required=True,
        help="Input dataset.npz (GELLO / flat layout)",
    )
    ap.add_argument(
        "--out",
        dest="path_out",
        type=Path,
        default=Path("asset/bc_dataset.npz"),
        help="Output path for bc.py",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=280,
        help="Timesteps per pseudo-episode (for train/test split by episode)",
    )
    args = ap.parse_args()
    convert(args.path_in, args.path_out, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
