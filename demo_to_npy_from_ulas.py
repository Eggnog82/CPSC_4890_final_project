import glob
import os
import pickle
import sys

import numpy as np
from natsort import natsorted
from tqdm import tqdm

NPZ_NAME = "all_episodes.npz"


def convert_single_demo(source_dir, output_dir, episode_idx):
    """
    Converts a single demonstration consisting of multiple `.pkl` timesteps into a `.npy` file.
    """

    pkls = natsorted(glob.glob(os.path.join(source_dir, "*.pkl")))

    if len(pkls) <= 30:
        print(f"Skipping {source_dir} (only {len(pkls)} frames)")
        return 0  # Skip short episodes

    pkls = pkls[:-5]  # Ignore the last 5 frames

    # Ask the user for a language instruction for this demonstration
    instruction = input(f"Enter language instruction for episode {episode_idx}: ")

    episode = []

    for pkl in pkls:
        try:
            with open(pkl, "rb") as f:
                demo = pickle.load(f)
        except Exception as e:
            print(f"Skipping {pkl} (corrupted): {e}")
            continue

        # Extract required data
        img = demo.get("base_rgb", None)  # Base camera RGB
        w_img = demo.get("wrist_rgb", None)  # Wrist camera RGB
        state = demo.get("joint_positions", None)  # Joint positions
        action = demo.get("control", None)  # Joint space actions

        if img is None or w_img is None or state is None or action is None:
            print(f"Skipping {pkl} due to missing data.")
            print(f"img = {img}, w_img = {w_img}")
            continue

        # Store timestep in dictionary format
        episode.append(
            {
                "image": img,
                "wrist_image": w_img,
                "state": state,
                "action": action,
                "language_instruction": instruction,
            }
        )

    if len(episode) == 0:
        print(f"Skipping {source_dir} (no valid timesteps)")
        return 0

    # Convert episode to a NumPy array and save it as `episode_{i}.npy`
    episode_array = np.array(episode, dtype=object)
    save_path = os.path.join(output_dir, f"episode_{episode_idx}.npy")
    np.save(save_path, episode_array)

    print(f"Saved: {save_path}")
    return len(episode)


def consolidate_npy_to_npz(output_dir: str) -> None:
    """Write `<output_dir>/all_episodes.npz` from every `episode_*.npy` in that folder."""
    paths = natsorted(glob.glob(os.path.join(output_dir, "episode_*.npy")))
    if not paths:
        print(f"No episode_*.npy files in {output_dir}; skipping .npz.")
        return

    episodes = [np.load(p, allow_pickle=True) for p in paths]
    out_path = os.path.join(output_dir, NPZ_NAME)
    np.savez_compressed(
        out_path,
        episodes=np.array(episodes, dtype=object),
        episode_source_paths=np.array(paths, dtype=object),
        num_episodes=len(episodes),
    )
    print(f"Consolidated {len(episodes)} episodes -> {out_path}")


def main(source_dir):
    """
    Processes multiple demonstrations from a given folder, saves `.npy` per episode,
    then writes `all_episodes.npz` in the same `_npy` folder.
    """

    subdirs = natsorted(glob.glob(os.path.join(source_dir, "*/")))

    if len(subdirs) == 0:
        print("No demonstration folders found.")
        return

    output_dir = os.path.join(source_dir, "_npy")
    os.makedirs(output_dir, exist_ok=True)

    total_episodes = 0
    starting_eps = int(input("Starting episode number: "))

    for i, subdir in enumerate(tqdm(subdirs, desc="Processing Episodes")):
        num_frames = convert_single_demo(subdir, output_dir, i + starting_eps)
        if num_frames > 0:
            total_episodes += 1

    print(f"Finished! Total saved episodes: {total_episodes}")
    consolidate_npy_to_npz(output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <source_directory>")
        sys.exit(1)

    main(sys.argv[1])
