"""
Convert `all_episodes.npz` from demo_to_npy_from_ulas.py into the format expected by
scripts/bc.py:

  - states:  (E,) object, each (T, 8)  -> [q_0..q_6, gripper_reading]
  - actions: (E,) object, each (T, 8)  -> [Δq_0..Δq_6, gripper_command_abs]
            Δq = control[:7] - q at each step (matches lab bc.py inference).

Input .npz must contain key `episodes`: (E,) object, each element is a length-T
vector of dicts with keys `state` (joint_positions) and `action` (control).
"""

import argparse
from pathlib import Path

import numpy as np


def _episode_to_state_action_arrays(episode_obj: np.ndarray):
    """episode_obj: 1D object array of dicts (one per timestep)."""
    states_out = []
    actions_out = []
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
        if s.size >= 8:
            g = float(s[7])
        else:
            g = 0.0
        dq = (ctrl[:7] - q).astype(np.float32)
        g_cmd = float(ctrl[7])
        states_out.append(np.concatenate([q, [g]]).astype(np.float32))
        actions_out.append(np.concatenate([dq, [g_cmd]]).astype(np.float32))
    if not states_out:
        return None, None
    return np.stack(states_out), np.stack(actions_out)


def convert(path_in: Path, path_out: Path) -> None:
    data = np.load(path_in, allow_pickle=True)
    if "episodes" not in data.files:
        raise KeyError(
            f"Expected key 'episodes' in {path_in}; found: {list(data.files)}"
        )
    episodes = data["episodes"]
    states_list = []
    actions_list = []
    for ei in range(len(episodes)):
        ep = episodes[ei]
        s_arr, a_arr = _episode_to_state_action_arrays(ep)
        if s_arr is None:
            print(f"Skipping empty episode index {ei}")
            continue
        states_list.append(s_arr)
        actions_list.append(a_arr)

    if not states_list:
        raise ValueError("No valid episodes after conversion.")

    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path_out,
        states=np.array(states_list, dtype=object),
        actions=np.array(actions_list, dtype=object),
        action_type="delta_joint_angles_joint_abs_gripper",
        unit="radians",
    )
    print(f"Wrote {path_out} ({len(states_list)} episodes).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--in", dest="path_in", type=Path, required=True)
    ap.add_argument(
        "--out",
        dest="path_out",
        type=Path,
        default=Path("asset/bc_train.npz"),
    )
    args = ap.parse_args()
    convert(args.path_in.expanduser(), args.path_out.expanduser())


if __name__ == "__main__":
    main()
