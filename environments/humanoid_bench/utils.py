"""Helpers to create humanoid-bench environments and collect demo data.

This module provides:
- `make_env_humanoid(config, suite, task, **kwargs)` — instantiate and
    wrap a `HumanoidEnv` for use in SAILOR.
- `collect_with_bundled_policy(...)` — helper that will run episodes using a
    bundled low-level policy (if available) and save all rollouts into a single
    HDF5 file.
- `get_train_val_datasets(config)` — infers dims and optionally hooks into a
    local demo directory. If `config.collect_demo=True` and a bundled policy is
    available, the function can collect episodes automatically into `demos/`.

The collector leverages the `HumanoidEnv` wrappers which accept
`policy_path`, `mean_path`, `var_path` and `policy_type` kwargs (see
`humanoid_bench.env`). This keeps loader logic minimal and defers policy
instantiation to the benchmark code.
"""

import collections
from pathlib import Path
import os
import time
import numpy as np
from termcolor import cprint

from environments.humanoid_bench.humanoid_wrapper import HumanoidBenchWrapper
from environments.humanoid_bench.constants import IMAGE_OBS_KEYS
from sailor.classes.rollout_utils import get_act_stacked, get_obs_stacked
from sailor.dreamer.tools import add_to_cache


def _parse_task_string(task_str: str):
    """Parse task string of form 'robot_control_task' (e.g., 'h1hand_hand_reach')."""
    parts = task_str.split("_")
    if len(parts) < 3:
        raise ValueError("humanoid-bench tasks must be of form 'robot_control_task'")
    robot, control, *rest = parts
    taskname = "_".join(rest)
    return robot, control, taskname


def make_env_humanoid(config, suite, task, **env_kwargs):
    """Create a `HumanoidEnv` and wrap it for SAILOR.

    Extra keyword arguments are forwarded to the `HumanoidEnv` constructor and
    can include `policy_path`, `mean_path`, `var_path`, and `policy_type`.
    """
    try:
        from humanoid_bench.env import HumanoidEnv
    except Exception as e:
        raise ImportError("Could not import humanoid_bench. Ensure it's installed or on PYTHONPATH.") from e

    robot, control, taskname = _parse_task_string(task)

    kwargs = dict(
        robot=robot,
        control=control,
        task=taskname,
        render_mode="rgb_array",
        width=int(getattr(config, "image_size", 84)),
        height=int(getattr(config, "image_size", 84)),
        obs_wrapper="True",
        sensors="image,proprio",
    )
    # user-supplied env kwargs override defaults
    kwargs.update(env_kwargs)

    env = HumanoidEnv(**kwargs)
    wrapped = HumanoidBenchWrapper(env, config, keys=IMAGE_OBS_KEYS, action_repeat=getattr(config, "action_repeat", 1))
    cprint(f"Initialized humanoid-bench env {robot}_{control}_{taskname}", "yellow")
    return wrapped


def _find_bundled_policy_dir(taskname: str):
    """Find a matching bundled policy directory under `humanoid-bench/data`.

    The upstream repo ships directories like `reach_one_hand` / `reach_two_hands`.
    We attempt to pick a directory that contains a `torch_model.pt` and whose
    name matches or contains the taskname.
    """
    # Start from this file's location and go up to find the repo root
    # environments/humanoid_bench/utils.py -> parents[2] is repo root
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "humanoid-bench" / "data"
    if not data_root.exists():
        return None

    # prefer exact matches, otherwise substring matches
    for d in data_root.iterdir():
        if not d.is_dir():
            continue
        if d.name == taskname and (d / "torch_model.pt").exists():
            return d

    for d in data_root.iterdir():
        if not d.is_dir():
            continue
        if taskname in d.name and (d / "torch_model.pt").exists():
            return d

    # if none matched, but there is any dir with a model, return the first
    for d in data_root.iterdir():
        if not d.is_dir():
            continue
        if (d / "torch_model.pt").exists():
            return d

    return None


def collect_with_bundled_policy(config, outdir: str, num_episodes: int = 10, max_steps: int = 1000, policy_type: str = None):
    """Collect rollouts using a bundled low-level policy (if available) into a single HDF5.

    This function will:
    - look under `humanoid-bench/data` for a matching policy for the task
    - instantiate a `HumanoidEnv` with `policy_path/mean_path/var_path` and
      `policy_type` forwarded to the env so the benchmark's wrappers load the
      low-level policy
    - run `num_episodes` episodes and save everything into one HDF5 file under `outdir`
    """
    try:
        suite, task = config.task.split("__", 1)
    except Exception:
        raise ValueError("config.task must be 'humanoid-bench__<robot>_<control>_<task>'")

    robot, control, taskname = _parse_task_string(task)
    
    # Map policy type to bundled policy directory
    if policy_type and "double" in policy_type:
        policy_dir_name = "reach_two_hands"
    elif policy_type and "single" in policy_type:
        policy_dir_name = "reach_one_hand"
    else:
        # Fallback to task-based lookup
        policy_dir = _find_bundled_policy_dir(taskname)
        if policy_dir is None:
            cprint(f"No bundled policy found for task '{taskname}' under humanoid-bench/data", "red")
            return []
        policy_dir_name = policy_dir.name
    
    policy_dir = Path(__file__).parent.parent.parent / "humanoid-bench" / "data" / policy_dir_name
    if not policy_dir.exists():
        cprint(f"Policy directory not found: {policy_dir}", "red")
        return []

    policy_path = str(policy_dir / "torch_model.pt")
    mean_path = str(policy_dir / "mean.npy") if (policy_dir / "mean.npy").exists() else None
    var_path = str(policy_dir / "var.npy") if (policy_dir / "var.npy").exists() else None

    cprint(f"Using bundled policy at {policy_path}", "yellow")

    env = make_env_humanoid(config, suite, task, policy_path=policy_path, mean_path=mean_path, var_path=var_path, policy_type=policy_type)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    traj_lengths = []

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise ImportError("h5py is required for humanoid-bench data collection. Install with `pip install h5py`." ) from e

    import pickle

    h5_path = outdir / f"humanoid_{taskname}.hdf5"
    h5file = h5py.File(h5_path, "w")

    for ep in range(int(num_episodes)):
        obs_out = []
        actions = []
        rewards = []
        dones = []

        raw = env.reset()
        raw_obs = raw[0] if isinstance(raw, tuple) and len(raw) >= 1 else raw

        for t in range(int(max_steps)):
            # sample from env.action_space — when a hierarchical wrapper is active
            # this action space will correspond to the high-level action.
            action = env.action_space.sample()
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                raw_obs, rew, term, trunc, info = out
                done = bool(term or trunc)
            elif isinstance(out, tuple) and len(out) == 4:
                raw_obs, rew, done, info = out
            else:
                raw_obs = out[0] if isinstance(out, tuple) else out
                rew = 0.0
                done = False

            # Gymnasium max_episode_steps is not enforced when we instantiate
            # HumanoidEnv directly, so mark done when we hit the caller's cap.
            if (t + 1) >= int(max_steps):
                done = True

            obs_out.append(raw_obs)
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append(float(rew))
            dones.append(bool(done))

            if done:
                break

        grp = h5file.create_group(f"ep_{ep:05d}")
        grp.create_dataset("actions", data=np.array(actions, dtype=np.float32), compression="gzip")
        grp.create_dataset("rewards", data=np.array(rewards, dtype=np.float32), compression="gzip")
        grp.create_dataset("dones", data=np.array(dones, dtype=np.uint8), compression="gzip")

        # Store observations as pickled bytes to preserve dict/array structure
        obs_bytes = np.array([np.void(pickle.dumps(o, protocol=4)) for o in obs_out], dtype=np.void)
        grp.create_dataset("obs", data=obs_bytes, compression="gzip")

        traj_lengths.append(len(rewards))
        cprint(f"Saved rollout ep_{ep:05d} to HDF5 (len={len(rewards)})", "cyan")
        # brief pause to let mujoco release resources between resets if needed
        time.sleep(0.1)

    try:
        env.close()
    except Exception:
        pass

    if traj_lengths:
        lengths = np.array(traj_lengths)
        cprint(
            f"Traj lengths — mean: {lengths.mean():.1f}, min: {lengths.min()}, max: {lengths.max()}",
            "yellow",
        )

    h5file.flush()
    h5file.close()
    cprint(f"Saved aggregated HDF5: {h5_path}", "yellow")
    return [str(h5_path)]


def add_traj_to_cache(traj_id, traj_data, cache, config, norm_dict=None):
    """Add a trajectory to the cache with proper stacking for obs_horizon and pred_horizon.
    
    Similar to ManiSkill's add_traj_to_cache, this creates individual transitions
    with stacked observations and actions.
    """
    from collections import defaultdict
    
    # Extract state if it exists
    concat_state = []
    if "state" in traj_data:
        for t in range(len(traj_data["state"])):
            curr_obs_state_vec = np.array(traj_data["state"][t], dtype=np.float32)
            concat_state.append(curr_obs_state_vec)
            
            # Update norm_dict for the environment
            if norm_dict is not None:
                norm_dict["ob_max"] = np.maximum(norm_dict["ob_max"], curr_obs_state_vec)
                norm_dict["ob_min"] = np.minimum(norm_dict["ob_min"], curr_obs_state_vec)
    
    # Stack Observations for State and Pixel Keys
    stacked_obs = {}
    if concat_state:
        stacked_obs["state"] = get_obs_stacked(concat_state, config.obs_horizon)
    
    # Stack Actions
    stacked_acts = get_act_stacked(traj_data["actions"], config.pred_horizon)
    
    # Stack Images
    stacked_images = {}
    if "agentview_image" in traj_data:
        stacked_images["agentview_image"] = get_obs_stacked(
            traj_data["agentview_image"], config.obs_horizon
        )
    if "robot0_eye_in_hand_image" in traj_data:
        stacked_images["robot0_eye_in_hand_image"] = get_obs_stacked(
            traj_data["robot0_eye_in_hand_image"], config.obs_horizon
        )
    
    # Update norm_dict for actions
    if norm_dict is not None:
        acts_np_array = np.array(traj_data["actions"])
        norm_dict["ac_max"] = np.maximum(
            norm_dict["ac_max"], np.max(acts_np_array, axis=0)
        )
        norm_dict["ac_min"] = np.minimum(
            norm_dict["ac_min"], np.min(acts_np_array, axis=0)
        )
    
    # Fill all the transitions in the cache
    traj_len = len(traj_data["actions"])
    for t in range(traj_len):
        transition = defaultdict(np.array)
        
        if "state" in stacked_obs:
            transition["state"] = stacked_obs["state"][t]
        
        for key, value in stacked_images.items():
            transition[key] = value[t]
        
        rewards_t = traj_data["rewards"][t] if "rewards" in traj_data else 0.0
        
        # Set done to 1 at last transition
        done_t = 0 if t < len(stacked_acts) - 1 else 1
        
        transition["reward"] = np.array(rewards_t, dtype=np.float32)
        transition["is_first"] = np.array(t == 0, dtype=np.bool_)
        transition["action"] = stacked_acts[t]
        transition["is_last"] = np.array(done_t, dtype=np.bool_)
        transition["is_terminal"] = np.array(done_t, dtype=np.bool_)
        
        add_to_cache(cache, f"exp_traj_{traj_id}", transition)


def get_train_val_datasets(config):
    """Return train/val datasets with proper stacking and normalization.

    If `config.collect_demo` is truthy and a bundled policy exists the
    function will collect demos into `demos/humanoid/<taskname>` and return
    that path in the metadata so callers can find the data.
    """
    try:
        suite, task = config.task.split("__", 1)
    except Exception:
        raise ValueError("config.task must be of form 'humanoid-bench__<robot>_<control>_<task>'")

    # create a sample env to infer dims
    env = make_env_humanoid(config=config, suite=suite, task=task)

    state_dim = 0
    if hasattr(env, "observation_space") and getattr(env.observation_space, "spaces", None) and "state" in env.observation_space.spaces:
        state_dim = int(np.prod(env.observation_space.spaces["state"].shape))
    else:
        try:
            sample = env.reset()
            if isinstance(sample, dict) and "state" in sample:
                state_dim = int(np.prod(sample["state"].shape))
        except Exception:
            state_dim = 0

    action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else 0

    # If user asked to collect demos and bundled policy exists, collect them
    collect_flag = bool(getattr(config, "collect_demo", False))
    if collect_flag:
        # where to write demos
        outdir = getattr(config, "demo_outdir", "demos/humanoid")
        policy_type = getattr(config, "policy_type", None)
        saved = collect_with_bundled_policy(config=config, outdir=outdir, num_episodes=getattr(config, "num_demo_episodes", 10), policy_type=policy_type)

    # If demos already exist on disk, load them into train_eps
    # Resolve demo directory preference: explicit demo_dir > demo_outdir > datadir > default
    candidate_dir = getattr(config, "demo_dir", None)
    if candidate_dir is None:
        candidate_dir = getattr(config, "demo_outdir", None)
    if candidate_dir is None:
        candidate_dir = getattr(config, "datadir", None)
    if candidate_dir is None:
        candidate_dir = "demos/humanoid"

    demo_dir = Path(candidate_dir)

    if not demo_dir.exists():
        raise ValueError(
            f"No humanoid-bench demos found under {demo_dir}. Provide an HDF5 file via --demo_dir or run with --collect_demo True to generate one."
        )

    import pickle
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise ImportError("h5py is required to load humanoid-bench demos. Install with `pip install h5py`." ) from e

    # Load all trajectories from HDF5 files
    all_trajs = []
    demo_files = sorted(list(demo_dir.glob("*.h5")) + list(demo_dir.glob("*.hdf5")))
    
    for fpath in demo_files:
        with h5py.File(fpath, "r") as hf:
            for ep_key in sorted(hf.keys()):
                grp = hf[ep_key]
                actions = np.array(grp["actions"], dtype=np.float32)
                action_dim = actions.shape[1] if len(actions.shape) > 1 else 1
                rewards = np.array(grp["rewards"], dtype=np.float32)
                dones = np.array(grp["dones"], dtype=np.bool_)
                obs_list = [pickle.loads(bytes(item)) for item in grp["obs"]]
                
                # Flatten obs dicts into per-key arrays to avoid object dtype
                obs_keys = list(obs_list[0].keys()) if obs_list else []
                obs_arrays = {}
                for k in obs_keys:
                    obs_arrays[k] = np.stack([o[k] for o in obs_list]) if len(obs_list) > 0 else np.array([])

                # Normalize image keys
                traj_dict = {}
                if "agentview_image" in obs_arrays:
                    traj_dict["agentview_image"] = obs_arrays["agentview_image"]
                elif "image_left_eye" in obs_arrays:
                    traj_dict["agentview_image"] = obs_arrays["image_left_eye"]
                elif "cam0" in obs_arrays:
                    traj_dict["agentview_image"] = obs_arrays["cam0"]

                if "robot0_eye_in_hand_image" in obs_arrays:
                    traj_dict["robot0_eye_in_hand_image"] = obs_arrays["robot0_eye_in_hand_image"]
                elif "image_right_eye" in obs_arrays:
                    traj_dict["robot0_eye_in_hand_image"] = obs_arrays["image_right_eye"]
                elif "cam1" in obs_arrays:
                    traj_dict["robot0_eye_in_hand_image"] = obs_arrays["cam1"]

                # Keep state (flatten all non-image keys into state)
                state_keys = [k for k in obs_keys if k not in ("agentview_image", "image_left_eye", "cam0", "robot0_eye_in_hand_image", "image_right_eye", "cam1")]
                if state_keys:
                    state_list = [obs_arrays[k] for k in state_keys]
                    # Flatten each to 2D (timesteps, features) and concatenate
                    state_list_flat = []
                    for s in state_list:
                        if s.ndim == 1:
                            s = s[:, None]
                        elif s.ndim > 2:
                            s = s.reshape(s.shape[0], -1)
                        state_list_flat.append(s)
                    traj_dict["state"] = np.concatenate(state_list_flat, axis=-1)
                    state_dim = traj_dict["state"].shape[-1]

                traj_dict["actions"] = actions
                traj_dict["rewards"] = rewards
                traj_dict["dones"] = dones
                
                all_trajs.append(traj_dict)

    if not all_trajs:
        raise ValueError(f"No trajectories loaded from {demo_dir}")
    
    num_exp_trajs = getattr(config, "num_exp_trajs", len(all_trajs))
    num_val_trajs = getattr(config, "num_exp_val_trajs", 0)
    
    cprint(
        f"Loading {num_exp_trajs} train and {num_val_trajs} val trajs for humanoid-bench",
        "green",
    )
    
    # Print trajectory statistics
    traj_lengths = [len(t["actions"]) for t in all_trajs]
    cprint(f"Average trajectory length: {np.mean(traj_lengths):.1f}", "green")
    cprint(f"Min trajectory length: {np.min(traj_lengths)}", "green")
    cprint(f"Max trajectory length: {np.max(traj_lengths)}", "green")
    
    # Assert if we have enough data
    assert len(all_trajs) >= num_exp_trajs + num_val_trajs, (
        f"Not enough data! Found {len(all_trajs)} expert trajectories, "
        f"but config specifies {num_exp_trajs} training and "
        f"{num_val_trajs} validation episodes."
    )
    
    # Initialize normalization dict
    norm_dict = {
        "ob_max": -np.inf * np.ones(state_dim, dtype=np.float32),
        "ob_min": np.inf * np.ones(state_dim, dtype=np.float32),
        "ac_max": -np.inf * np.ones(action_dim, dtype=np.float32),
        "ac_min": np.inf * np.ones(action_dim, dtype=np.float32),
    }
    
    train_expert_eps = collections.OrderedDict()
    val_expert_eps = collections.OrderedDict()
    
    # Add training trajectories to cache
    for i in range(num_exp_trajs):
        add_traj_to_cache(
            traj_id=i,
            traj_data=all_trajs[i],
            cache=train_expert_eps,
            config=config,
            norm_dict=norm_dict,
        )
    cprint(f"Loaded {len(train_expert_eps.keys())} training episodes", "green")
    
    # Add validation trajectories to cache
    for i in range(num_val_trajs):
        add_traj_to_cache(
            traj_id=i + num_exp_trajs,
            traj_data=all_trajs[i + num_exp_trajs],
            cache=val_expert_eps,
            config=config,
        )
    cprint(f"Loaded {len(val_expert_eps.keys())} validation episodes", "green")
    

    return train_expert_eps, val_expert_eps, norm_dict, state_dim, action_dim
