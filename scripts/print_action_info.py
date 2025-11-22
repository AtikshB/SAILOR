"""
Script to print action info for any robot.

This script creates an environment with a specified robot and prints
detailed information about the action space, including dimensions and indices.
"""

import argparse
import robosuite as suite

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Print action info for a robot")
    parser.add_argument(
        "--robot",
        type=str,
        default="GR1ArmsOnly",
        help="Robot name (e.g., GR1FloatingBody, Panda, Sawyer)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Door",
        help="Environment name (default: Door)",
    )
    args = parser.parse_args()

    # Create environment with specified robot
    env = suite.make(
        env_name=args.env,
        robots=args.robot,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    # Reset environment to initialize the robot
    env.reset()

    # Print action info for each robot (in this case, just one)
    for robot in env.robots:
        print("\n" + "="*80)
        print(f"Robot: {robot.name} (Type: {robot.__class__.__name__})")
        print("="*80)

        # Print detailed action info as a dictionary
        robot.print_action_info_dict()

        # Also print basic action info
        robot.print_action_info()

        # Print action limits (min/max values)
        low, high = robot.action_limits
        print("\n" + "-"*80)
        print("Action Limits:")
        print("-"*80)
        print(f"Action dimension: {low.shape[0]}")
        print(f"Lower bounds: {low}")
        print(f"Upper bounds: {high}")
        print("-"*80 + "\n")
