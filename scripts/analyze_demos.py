#!/usr/bin/env python3
"""Analyze collected demonstration files.

Usage:
    python scripts/analyze_demos.py --demo-dir demos/reach_two_hands --file-idx 0
    python scripts/analyze_demos.py --demo-dir demos/reach_two_hands --file-idx 0 --step 10
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def load_demo(filepath):
    """Load a demo file (pkl or npz)."""
    filepath = Path(filepath)
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        return {k: data[k] for k in data.keys()}
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def analyze_obs(obs, step_idx):
    """Analyze observation structure."""
    print(f"\n{'='*60}")
    print(f"STEP {step_idx}")
    print(f"{'='*60}")
    
    if isinstance(obs, dict):
        print(f"Observation type: dict")
        print(f"Keys: {list(obs.keys())}")
        print()
        for key, val in obs.items():
            if hasattr(val, 'shape'):
                print(f"  {key}:")
                print(f"    Shape: {val.shape}")
                print(f"    Dtype: {val.dtype}")
                if val.size <= 10:
                    print(f"    Value: {val}")
                else:
                    print(f"    Min/Max: [{val.min():.3f}, {val.max():.3f}]")
                    if 'image' in key.lower():
                        print(f"    (Image data)")
            else:
                print(f"  {key}: {val}")
    elif hasattr(obs, 'shape'):
        print(f"Observation type: array")
        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        if obs.size <= 10:
            print(f"  Value: {obs}")
        else:
            print(f"  Min/Max: [{obs.min():.3f}, {obs.max():.3f}]")
    else:
        print(f"Observation: {obs}")


def main():
    parser = argparse.ArgumentParser(description="Analyze demonstration files")
    parser.add_argument(
        "--demo-dir",
        type=str,
        required=True,
        help="Directory containing demo files",
    )
    parser.add_argument(
        "--file-idx",
        type=int,
        default=0,
        help="Index of file to analyze (default: 0)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific step to analyze (default: show first, middle, last)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary statistics, no per-step details",
    )

    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    if not demo_dir.exists():
        print(f"Error: Directory not found: {demo_dir}")
        return

    # Find all demo files
    demo_files = sorted(list(demo_dir.glob("*.pkl")) + list(demo_dir.glob("*.npz")))
    if not demo_files:
        print(f"Error: No demo files found in {demo_dir}")
        return

    if args.file_idx >= len(demo_files):
        print(f"Error: File index {args.file_idx} out of range (found {len(demo_files)} files)")
        return

    filepath = demo_files[args.file_idx]
    print(f"\nAnalyzing: {filepath.name}")
    print(f"Total files in directory: {len(demo_files)}")

    # Load data
    data = load_demo(filepath)
    
    obs = data['obs']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    
    traj_len = len(rewards)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    print(f"Length: {traj_len} steps")
    print(f"Total reward: {rewards.sum():.2f}")
    print(f"Mean reward: {rewards.mean():.3f}")
    print(f"Terminated early: {any(dones[:-1])}")
    if any(dones):
        first_done = np.where(dones)[0][0]
        print(f"First done at step: {first_done}")
    
    print(f"\nActions:")
    print(f"  Shape: {actions.shape}")
    print(f"  Dtype: {actions.dtype}")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std: {actions.std(axis=0)}")
    
    print(f"\nObservations:")
    print(f"  Type: {type(obs)}")
    print(f"  Array shape: {obs.shape}")
    if len(obs) > 0:
        sample_obs = obs[0]
        print(f"  Single obs type: {type(sample_obs)}")
        if isinstance(sample_obs, dict):
            print(f"  Keys: {list(sample_obs.keys())}")
            for key, val in sample_obs.items():
                if hasattr(val, 'shape'):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}")

    if args.summary_only:
        return

    # Detailed step analysis
    if args.step is not None:
        steps_to_show = [args.step] if args.step < traj_len else []
    else:
        # Show first, middle, and last steps
        steps_to_show = [0]
        if traj_len > 2:
            steps_to_show.append(traj_len // 2)
        if traj_len > 1:
            steps_to_show.append(traj_len - 1)
    
    for step_idx in steps_to_show:
        if step_idx >= traj_len:
            continue
            
        analyze_obs(obs[step_idx], step_idx)
        
        print(f"\nAction at step {step_idx}:")
        print(f"  {actions[step_idx]}")
        
        print(f"\nReward: {rewards[step_idx]:.3f}")
        print(f"Done: {dones[step_idx]}")


if __name__ == "__main__":
    main()
