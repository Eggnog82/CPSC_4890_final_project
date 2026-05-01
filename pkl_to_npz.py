import os
import pickle
import numpy as np
import argparse


def convert_all_to_npz(input_dir, output_file):
    all_demos = []

    # Walk through all subfolders
    for root, dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if fname.endswith(".pkl"):
                path = os.path.join(root, fname)
                print(f"Loading {path}...")
                with open(path, "rb") as f:
                    demo = pickle.load(f)
                    all_demos.append(demo)

    if len(all_demos) == 0:
        raise ValueError("No .pkl files found.")

    print(f"\nLoaded {len(all_demos)} total demos.")

    # Collect all possible keys across demos
    keys = set().union(*(demo.keys() for demo in all_demos))

    dataset = {}
    for key in keys:
        dataset[key] = np.array(
            [demo.get(key, None) for demo in all_demos],
            dtype=object  # preserve variable-length trajectories
        )

    print(f"Saving combined dataset to {output_file}...")
    np.savez(output_file, **dataset)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="dataset.npz")
    args = parser.parse_args()

    convert_all_to_npz(args.input_dir, args.output_file)