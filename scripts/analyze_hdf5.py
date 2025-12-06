"""
Script to analyze HDF5 demonstration files.

This script loads an HDF5 file and prints information about:
- Action space (dimensionality, bounds)
- State/observation space (keys and shapes)
- Robot metadata
- Environment metadata

Args:
    dataset (str): path to hdf5 dataset

Example usage:
    python analyze_hdf5.py --dataset datasets/robocasa_datasets/two_arm_can_sort_random.hdf5
"""

import h5py
import json
import argparse
import numpy as np


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_hdf5(dataset_path):
    """Analyze an HDF5 dataset and print key information."""

    # Open the HDF5 file
    f = h5py.File(dataset_path, "r")

    # Get list of demonstrations
    demos = sorted(list(f["data"].keys()))
    print(f"\nFound {len(demos)} demonstrations in the dataset")

    # Use the first demo to extract information
    first_demo = demos[0]

    # ========== ACTION SPACE ==========
    print_section("ACTION SPACE")
    actions = f[f"data/{first_demo}/actions"][()]
    print(f"Action dimensionality: {actions.shape[1]}")
    print(f"Number of timesteps in first demo: {actions.shape[0]}")

    # Compute action statistics across all demos
    action_mins = []
    action_maxs = []
    action_means = []

    for demo in demos:
        demo_actions = f[f"data/{demo}/actions"][()]
        action_mins.append(np.min(demo_actions, axis=0))
        action_maxs.append(np.max(demo_actions, axis=0))
        action_means.append(np.mean(demo_actions, axis=0))

    action_min = np.min(action_mins, axis=0)
    action_max = np.max(action_maxs, axis=0)
    action_mean = np.mean(action_means, axis=0)

    print(f"\nAction bounds across all demos:")
    print(f"  Min: {action_min}")
    print(f"  Max: {action_max}")
    print(f"  Mean: {action_mean}")

    # ========== STATE/OBSERVATION SPACE ==========
    print_section("STATE/OBSERVATION SPACE")

    # Check if observations exist
    if "obs" in f[f"data/{first_demo}"]:
        print("Observation keys and shapes:")
        obs_group = f[f"data/{first_demo}/obs"]
        for obs_key in sorted(obs_group.keys()):
            obs_shape = obs_group[obs_key].shape
            obs_dtype = obs_group[obs_key].dtype
            print(f"  {obs_key:40s} shape: {str(obs_shape):20s} dtype: {obs_dtype}")
    else:
        print("No observation data found in dataset")

    # Check for state data
    if "states" in f[f"data/{first_demo}"]:
        states = f[f"data/{first_demo}/states"]
        print(f"\nState dimensionality: {states.shape[1]}")
        print(f"State shape: {states.shape}")

    # ========== ROBOT METADATA ==========
    print_section("ROBOT METADATA")

    try:
        env_args = json.loads(f["data"].attrs["env_args"])

        # Extract robot-specific information
        if "env_kwargs" in env_args:
            env_kwargs = env_args["env_kwargs"]

            if "robots" in env_kwargs:
                print(f"Robot(s): {env_kwargs['robots']}")

            if "robot_configs" in env_kwargs:
                print(f"\nRobot configurations:")
                for config in env_kwargs["robot_configs"]:
                    print(f"  {config}")

            if "controller_configs" in env_kwargs:
                print(f"\nController configuration:")
                for key, value in env_kwargs["controller_configs"].items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")

        # Print robot info if available in first demo metadata
        if f"data/{first_demo}".encode() in f["data"] or first_demo in f["data"]:
            if "ep_meta" in f[f"data/{first_demo}"].attrs:
                ep_meta = json.loads(f[f"data/{first_demo}"].attrs["ep_meta"])
                if "robot_info" in ep_meta:
                    print(f"\nRobot info from episode metadata:")
                    print(json.dumps(ep_meta["robot_info"], indent=2))

    except Exception as e:
        print(f"Error reading robot metadata: {e}")

    # ========== ENVIRONMENT METADATA ==========
    print_section("ENVIRONMENT METADATA")

    try:
        env_args = json.loads(f["data"].attrs["env_args"])

        print("Environment name:", env_args.get("env_name", "N/A"))

        if "env_kwargs" in env_args:
            env_kwargs = env_args["env_kwargs"]

            # Print key environment settings
            important_keys = [
                "has_renderer",
                "has_offscreen_renderer",
                "use_camera_obs",
                "control_freq",
                "horizon",
                "reward_shaping",
                "camera_names",
                "camera_heights",
                "camera_widths",
            ]

            print("\nEnvironment settings:")
            for key in important_keys:
                if key in env_kwargs:
                    print(f"  {key}: {env_kwargs[key]}")

        # Print episode-specific metadata from first demo
        if "ep_meta" in f[f"data/{first_demo}"].attrs:
            ep_meta = json.loads(f[f"data/{first_demo}"].attrs["ep_meta"])
            print(f"\nEpisode metadata (from {first_demo}):")

            # Print relevant episode metadata fields
            metadata_keys = [
                "lang",
                "task_id",
                "layout_id",
                "style_id",
                "obj_registries",
                "object_cfgs",
            ]

            for key in metadata_keys:
                if key in ep_meta:
                    value = ep_meta[key]
                    if isinstance(value, (dict, list)) and len(str(value)) > 100:
                        print(f"  {key}: <complex structure>")
                    else:
                        print(f"  {key}: {value}")

        print("\nFull environment arguments:")
        print(json.dumps(env_args, indent=2))

    except Exception as e:
        print(f"Error reading environment metadata: {e}")

    # ========== SUMMARY ==========
    print_section("DATASET SUMMARY")
    print(f"Total demonstrations: {len(demos)}")
    print(f"Action dimension: {actions.shape[1]}")

    traj_lengths = [f[f"data/{demo}/actions"].shape[0] for demo in demos]
    print(f"Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
    print(f"Total transitions: {sum(traj_lengths)}")

    f.close()
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HDF5 demonstration dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to HDF5 dataset file",
    )
    args = parser.parse_args()

    analyze_hdf5(args.dataset)
