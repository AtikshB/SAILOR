#!/usr/bin/env python3
"""Collect demonstrations using a trained PPO model.

Usage:
    python scripts/collect_ppo_demos.py \
        --task h1hand_pos_insert_normal \
        --ppo-model ppo.zip \
        --outdir demos/insert_ppo \
        --num-episodes 50
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import sys
import argparse
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.humanoid_bench.utils import collect_with_ppo_policy


def main():
    parser = argparse.ArgumentParser(description="Collect demonstrations using trained PPO policy")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task in format 'robot_control_task' (e.g., h1hand_pos_insert_normal)",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        required=True,
        help="Path to trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="demos/ppo",
        help="Output directory for HDF5 file (default: demos/ppo)",
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
        default=64,
        help="Image size for env (default: 64)",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Action repeat for env wrapper (default: 1)",
    )

    args = parser.parse_args()

    # Build config
    config = SimpleNamespace(
        task=f"humanoid-bench__{args.task}",
        image_size=args.image_size,
        action_repeat=args.action_repeat,
    )

    print(f"Collecting demonstrations for task: {args.task}")
    print(f"Using PPO model: {args.ppo_model}")
    print(f"Output directory: {args.outdir}")
    print(f"Episodes: {args.num_episodes}, Max steps per episode: {args.max_steps}")

    # Collect
    files = collect_with_ppo_policy(
        config=config,
        outdir=args.outdir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        ppo_model_path=args.ppo_model,
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
