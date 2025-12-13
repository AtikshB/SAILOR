#!/usr/bin/env python3
"""Collect humanoid-bench demonstrations into a single HDF5 using bundled or custom policies.

Usage:
    python scripts/collect_humanoid_demo.py --task h1hand_hand_reach --outdir demos/reach --num-episodes 20
    python scripts/collect_humanoid_demo.py --task h1hand_push --outdir demos/push --num-episodes 50 --policy-type reach_single
"""

# MUST BE FIRST: Enable osmesa backend for headless rendering (CPU-based)
import os
os.environ["MUJOCO_GL"] = "osmesa"

import sys
import argparse
from types import SimpleNamespace
from pathlib import Path

# Add repo root to path so we can import environments module
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.humanoid_bench.utils import collect_with_bundled_policy


def main():
    parser = argparse.ArgumentParser(description="Collect humanoid-bench demonstrations into a single HDF5")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task in format 'robot_control_task' (e.g., h1hand_hand_reach, h1hand_push, g1_walk)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="demos/humanoid",
        help="Output directory for the aggregated HDF5 (default: demos/humanoid)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to collect (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=84,
        help="Image size for env (default: 84)",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Action repeat for env wrapper (default: 1)",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default=None,
        help="Policy type for hierarchical control (e.g., reach_single, reach_double_relative, None for flat)",
    )

    args = parser.parse_args()

    # Build config
    config = SimpleNamespace(
        task=f"humanoid-bench__{args.task}",
        image_size=args.image_size,
        action_repeat=args.action_repeat,
    )

    print(f"Collecting demonstrations for task: {args.task}")
    print(f"Output directory: {args.outdir}")
    print(f"Episodes: {args.num_episodes}, Max steps per episode: {args.max_steps}")
    if args.policy_type:
        print(f"Policy type: {args.policy_type}")

    # Collect
    files = collect_with_bundled_policy(
        config=config,
        outdir=args.outdir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        policy_type=args.policy_type,
    )

    # Summary
    print(f"\n✓ Collection complete!")
    print(f"  Collected {args.num_episodes} episodes")
    if files:
        h5_path = Path(files[0])
        fsize = h5_path.stat().st_size / 1024 / 1024
        print(f"  Saved HDF5: {h5_path} ({fsize:.1f} MB)")


if __name__ == "__main__":
    main()
