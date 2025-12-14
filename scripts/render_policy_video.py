"""Render a video of a policy running in humanoid-bench environment.

This script supports rendering videos from:
1. Trained SAILOR policies (diffusion policies)
2. Trained PPO policies (stable-baselines3)
3. Bundled reach policies from humanoid-bench

Usage:
    # Render PPO policy
    python scripts/render_policy_video.py \
        --task h1hand_pos_insert_normal \
        --policy-type ppo \
        --policy-path models/ppo_insert.zip \
        --output-video videos/ppo_insert.mp4 \
        --num-episodes 5 \
        --fps 30 \
        --width 640 \
        --height 480

    # Render SAILOR policy (TODO: implement)
    python scripts/render_policy_video.py \
        --task h1hand_pos_insert_normal \
        --policy-type sailor \
        --policy-path checkpoints/sailor_model.pt \
        --output-video videos/sailor_insert.mp4

    # Render bundled reach policy
    python scripts/render_policy_video.py \
        --task h1hand_pos_reach \
        --policy-type reach \
        --output-video videos/reach.mp4
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from termcolor import cprint

# Add humanoid-bench to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "humanoid-bench"))

# Set up headless rendering - try osmesa first for better compatibility
if 'MUJOCO_GL' not in os.environ:
    # Try osmesa for better offscreen rendering support
    os.environ['MUJOCO_GL'] = 'osmesa'
    cprint("Using MUJOCO_GL=osmesa for rendering", "cyan")


def parse_task_string(task_str: str):
    """Parse task string of form 'robot_control_task' (e.g., 'h1hand_pos_reach')."""
    parts = task_str.split("_")
    if len(parts) < 3:
        raise ValueError("humanoid-bench tasks must be of form 'robot_control_task'")
    robot, control, *rest = parts
    taskname = "_".join(rest)
    return robot, control, taskname


def render_ppo_policy(args):
    """Render video using a trained PPO policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    
    from humanoid_bench.env import HumanoidEnv
    from humanoid_bench.mjx.visualization_utils import save_numpy_as_video
    import mujoco
    
    robot, control, taskname = parse_task_string(args.task)
    
    cprint(f"Loading PPO policy from {args.policy_path}", "yellow")
    model = PPO.load(args.policy_path)
    
    # Create environment with standard resolution first to get the MuJoCo model
    cprint(f"Creating environment: {robot}_{control}_{taskname}", "yellow")
    env = HumanoidEnv(
        robot=robot,
        control=control,
        task=taskname,
        render_mode="rgb_array",
        width=64,  # Very small to ensure compatibility
        height=64,
        obs_wrapper="False",  # PPO expects flat privileged state
    )
    
    # Now create a separate high-res renderer for video recording
    cprint(f"Setting up high-resolution renderer ({args.width}x{args.height})...", "yellow")
    use_custom_renderer = False
    renderer = None
    if not args.no_custom_renderer:
        try:
            renderer = mujoco.Renderer(env.unwrapped.model, args.height, args.width)
            # Set camera if specified
            if args.camera:
                renderer.enable_depth_rendering = False
            use_custom_renderer = True
            cprint(f"✓ Custom high-res renderer initialized ({args.width}x{args.height})", "green")
        except Exception as e:
            cprint(f"Warning: Could not create high-res renderer: {e}", "yellow")
            cprint(f"Falling back to default resolution (64x64)", "yellow")
    else:
        cprint("Using default environment resolution (64x64)", "yellow")
    
    all_frames = []
    
    for ep in range(args.num_episodes):
        cprint(f"Recording episode {ep+1}/{args.num_episodes}...", "cyan")
        frames = []
        
        raw = env.reset()
        obs = raw[0] if isinstance(raw, tuple) else raw
        
        for t in range(args.max_steps):
            # Render frame using custom high-res renderer or default
            if use_custom_renderer:
                renderer.update_scene(env.unwrapped.data)
                frame = renderer.render()
            else:
                frame = env.render()
            frames.append(frame)
            
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                obs, rew, term, trunc, info = out
                done = term or trunc
            elif isinstance(out, tuple) and len(out) == 4:
                obs, rew, done, info = out
            else:
                obs = out[0] if isinstance(out, tuple) else out
                done = False
            
            if done:
                break
        
        cprint(f"  Episode {ep+1} recorded: {len(frames)} frames", "green")
        all_frames.append(np.array(frames))
    
    env.close()
    if use_custom_renderer:
        renderer.close()
    
    # Concatenate all episodes
    if args.concat_episodes:
        video_array = np.concatenate(all_frames, axis=0)
        cprint(f"Saving concatenated video: {video_array.shape[0]} frames to {args.output_video}", "yellow")
    else:
        # Just use first episode
        video_array = all_frames[0]
        cprint(f"Saving video: {video_array.shape[0]} frames to {args.output_video}", "yellow")
    
    # Save video
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_as_video(video_array, str(output_path), fps=args.fps)
    
    cprint(f"✓ Video saved to {args.output_video}", "green")


def render_reach_policy(args):
    """Render video using bundled reach policy."""
    from humanoid_bench.env import HumanoidEnv
    from humanoid_bench.mjx.visualization_utils import save_numpy_as_video
    import mujoco
    
    robot, control, taskname = parse_task_string(args.task)
    
    # Find bundled policy
    data_root = REPO_ROOT / "humanoid-bench" / "data"
    policy_dir = None
    
    for d in data_root.iterdir():
        if d.is_dir() and taskname in d.name and (d / "torch_model.pt").exists() and "two" in d.name:
            policy_dir = d
            break
    
    if policy_dir is None:
        raise ValueError(f"No bundled policy found for task {taskname}")
    cprint(f"Using bundled policy from {policy_dir}", "yellow")
    
    # Determine policy type based on directory name
    dir_name = policy_dir.name.lower()
    # Default to double absolute (works for reach_two_hands)
    policy_type = "reach_double_absolute"
    
    cprint(f"Using policy type: {policy_type}", "cyan")
    
    # Create environment with hierarchical wrapper at very small resolution
    # to avoid framebuffer errors
    cprint(f"Creating environment: {robot}_{control}_{taskname}", "yellow")
    env = HumanoidEnv(
        robot=robot,
        control=control,
        task=taskname,
        render_mode="rgb_array",
        width=64,  # Very small to ensure compatibility
        height=64,
        obs_wrapper="gym_dict",
        policy_path=str(policy_dir / "torch_model.pt"),
        mean_path=str(policy_dir / "mean.npy"),
        var_path=str(policy_dir / "var.npy"),
        policy_type=policy_type,
    )
    
    # Create separate high-res renderer
    cprint(f"Setting up high-resolution renderer ({args.width}x{args.height})...", "yellow")
    use_custom_renderer = False
    if not args.no_custom_renderer:
        try:
            renderer = mujoco.Renderer(env.unwrapped.model, args.height, args.width)
            use_custom_renderer = True
            cprint("✓ Custom high-res renderer initialized", "green")
        except Exception as e:
            cprint(f"Warning: Could not create high-res renderer: {e}", "yellow")
            cprint(f"Falling back to default resolution (64x64)", "yellow")
    else:
        cprint("Using default environment resolution (64x64)", "yellow")
    
    all_frames = []
    
    for ep in range(args.num_episodes):
        cprint(f"Recording episode {ep+1}/{args.num_episodes}...", "cyan")
        frames = []
        
        raw = env.reset()
        obs = raw[0] if isinstance(raw, tuple) else raw
        
        for t in range(args.max_steps):
            # Render frame using custom high-res renderer or default
            if use_custom_renderer:
                renderer.update_scene(env.unwrapped.data)
                frame = renderer.render()
            else:
                frame = env.render()
            frames.append(frame)
            
            # For hierarchical policy, just pass zero actions (policy is in the wrapper)
            action = np.zeros(env.action_space.shape)
            
            # Step environment
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                obs, rew, term, trunc, info = out
                done = term or trunc
            elif isinstance(out, tuple) and len(out) == 4:
                obs, rew, done, info = out
            else:
                done = False
            
            if done:
                break
        
        cprint(f"  Episode {ep+1} recorded: {len(frames)} frames", "green")
        all_frames.append(np.array(frames))
    
    env.close()
    if use_custom_renderer:
        renderer.close()
    
    # Save video
    if args.concat_episodes:
        video_array = np.concatenate(all_frames, axis=0)
    else:
        video_array = all_frames[0]
    
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_as_video(video_array, str(output_path), fps=args.fps)
    
    cprint(f"✓ Video saved to {args.output_video}", "green")


def main():
    parser = argparse.ArgumentParser(description="Render policy video in humanoid-bench")
    parser.add_argument("--task", type=str, required=True,
                       help="Task name (e.g., h1hand_pos_insert_normal)")
    parser.add_argument("--policy-type", type=str, required=True,
                       choices=["ppo", "reach", "sailor"],
                       help="Type of policy to render")
    parser.add_argument("--policy-path", type=str, default=None,
                       help="Path to policy checkpoint (required for ppo/sailor)")
    parser.add_argument("--output-video", type=str, default="output.mp4",
                       help="Output video path")
    parser.add_argument("--num-episodes", type=int, default=1,
                       help="Number of episodes to record")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--concat-episodes", action="store_true",
                       help="Concatenate all episodes into one video")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for output video")
    parser.add_argument("--width", type=int, default=640,
                       help="Render width (may fall back to smaller if framebuffer fails)")
    parser.add_argument("--height", type=int, default=480,
                       help="Render height (may fall back to smaller if framebuffer fails)")
    parser.add_argument("--no-custom-renderer", action="store_true",
                       help="Don't use custom high-res renderer, just use env default")
    parser.add_argument("--camera", type=str, default=None,
                       help="Camera name to render from (e.g., 'cam_default', 'cam_hand_visible')")
    
    args = parser.parse_args()
    
    if args.policy_type in ["ppo", "sailor"] and args.policy_path is None:
        parser.error(f"--policy-path is required for policy-type={args.policy_type}")
    
    if args.policy_type == "ppo":
        render_ppo_policy(args)
    elif args.policy_type == "reach":
        render_reach_policy(args)
    elif args.policy_type == "sailor":
        cprint("SAILOR policy rendering not yet implemented", "red")
        sys.exit(1)
    else:
        parser.error(f"Unknown policy type: {args.policy_type}")


if __name__ == "__main__":
    main()
