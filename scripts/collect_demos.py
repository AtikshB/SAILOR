import argparse
from copy import deepcopy
import datetime
import json
import os
import time
from glob import glob

import h5py
import imageio
import mujoco
import numpy as np
import robosuite

# from robosuite import load_controller_config
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from termcolor import colored

import robocasa
import robocasa.macros as macros
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format


def is_empty_input_spacemouse(action_dict):
    if not np.all(action_dict["right_delta"] == 0):
        return False
    if "base_mode" in action_dict and action_dict["base_mode"] != -1:
        return False
    if "base" in action_dict and not np.all(action_dict["base"] == 0):
        return False

    return True


def collect_human_trajectory(
    env,
    device,
    arm,
    env_configuration,
    mirror_actions,
    render=True,
    max_fr=None,
    print_info=True,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    ep_meta = env.get_ep_meta()
    # print(json.dumps(ep_meta, indent=4))
    lang = ep_meta.get("lang", None)
    if print_info and lang is not None:
        print(colored(f"Instruction: {lang}", "green"))

    # degugging: code block here to quickly test and close env
    # env.close()
    # return None, True

    if render:
        # ID = 2 always corresponds to agentview
        env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    nonzero_ac_seen = False

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    zero_action = np.zeros(env.action_dim)
    for _ in range(1):
        # do a dummy step thru base env to initalize things, but don't record the step
        if isinstance(env, DataCollectionWrapper):
            env.env.step(zero_action)
        else:
            env.step(zero_action)

    discard_traj = False

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]
        active_arm = device.active_arm

        # Get the newest action
        input_ac_dict = device.input2action(mirror_actions=mirror_actions)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # set arm actions
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        if is_empty_input_spacemouse(action_dict):
            if not nonzero_ac_seen:
                if render:
                    env.render()
                continue
        else:
            nonzero_ac_seen = True

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        # Run environment step
        obs, _, _, _ = env.step(env_action)
        if render:
            env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # with open("/home/soroushn/tmp/model.xml", "w") as f:
        #     f.write(env.model.get_xml())
        # exit()

    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory
    else:
        ep_directory = None

    # cleanup for end of data collection episodes
    env.close()

    return ep_directory, discard_traj


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file. Updates existing file if present.
    
    To avoid duplication when appending, this function now tracks which
    episodes have already been processed.
    """
    
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    file_exists = os.path.exists(hdf5_path)
    
    # Keep track of processed episodes to avoid duplication
    processed_episodes = set()
    
    # Open in append mode if file exists, otherwise create new
    mode = "a" if file_exists else "w"
    print(f"{'Updating' if file_exists else 'Creating new'} hdf5 at {hdf5_path}")
    
    f = h5py.File(hdf5_path, mode)
    
    # Get or create the data group
    if "data" in f:
        grp = f["data"]
        # Find the current number of demos
        existing_demos = [k for k in grp.keys() if k.startswith("demo_")]
        next_demo_num = max([int(d.split("_")[1]) for d in existing_demos]) + 1 if existing_demos else 1
        
        # If the file exists, look for a processed_episodes attribute
        if "processed_episodes" in grp.attrs:
            try:
                processed_episodes = set(grp.attrs["processed_episodes"].split(","))
            except:
                # If there's any issue parsing, just start fresh
                processed_episodes = set()
    else:
        grp = f.create_group("data")
        next_demo_num = 1
    
    num_new_eps = 0
    env_name = None  # will get populated at some point
    
    for ep_directory in os.listdir(directory):
        # Skip if this episode is excluded or already processed
        if ((excluded_episodes is not None) and (ep_directory in excluded_episodes)) or (ep_directory in processed_episodes):
            continue
            
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        
        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
        
        if len(states) == 0:
            continue
            
        # Delete the last state
        del states[-1]
        assert len(states) == len(actions)
        
        num_new_eps += 1
        ep_data_grp = grp.create_group(f"demo_{next_demo_num}")
        next_demo_num += 1
        
        # Mark this episode as processed
        processed_episodes.add(ep_directory)
        
        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as xml_file:
            xml_str = xml_file.read()
        ep_data_grp.attrs["model_file"] = xml_str
        
        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as meta_file:
                ep_meta = meta_file.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta
            
        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))
    
    # Save the list of processed episodes to avoid duplication in future runs
    if processed_episodes:
        grp.attrs["processed_episodes"] = ",".join(processed_episodes)
    
    # Get final demo count after adding new ones
    final_demos = [k for k in grp.keys() if k.startswith("demo_")]
    final_demo_count = len(final_demos)
    
    print(f"Added {num_new_eps} new demos. Total demos: {final_demo_count}")
    
    if not file_exists and num_new_eps > 0:
        # Only write metadata if creating a new file
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["robocasa_version"] = robocasa.__version__
        grp.attrs["robosuite_version"] = robosuite.__version__
        grp.attrs["mujoco_version"] = mujoco.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info
    
    f.close()
    
    return hdf5_path

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="spacemouse",
        choices=["keyboard", "keyboardmobile", "spacemouse", "dummy", "dualsense"],
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale rotation user inputs",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"]
    )
    parser.add_argument(
        "--max_fr", default=30, type=int, help="If specified, limit the frame rate"
    )

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument(
        "--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    )
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--store_in_old", action="store_true")
    parser.add_argument(
        "--time_dir",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    args = parser.parse_args()

    # Get controller config
    # controller_config = load_controller_config(default_controller=args.controller)
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import (
            WholeBodyMinkIK,
        )

    env_name = args.environment

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if args.generative_textures is True:
        config["generative_textures"] = "100p"

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print(
                "Specifying 'obj_groups' in non-kitchen environment does not have an effect."
            )
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = True
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        if args.camera is None:
            args.camera = "robot0_frontview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
        args.camera=None
        config["translucent_robot"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    # Create environment
    if "layout_ids" in config:
        del config["layout_ids"]
    if "style_ids" in config:
        del config["style_ids"]
    if "obj_groups" in config:
        del config["obj_groups"]
    del config["translucent_robot"]
    del config["obj_instance_split"]

    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            vendor_id=0x054c,
            product_id=0x0ce6,
            reverse_xy=True
        )
    else:
        raise ValueError

    # make a new timestamped directory
    if args.store_in_old:
        new_dir = os.path.join(args.directory, args.time_dir)
    else:
        new_dir = os.path.join(args.directory, time_str)
        os.makedirs(new_dir)
    excluded_eps = []

    # collect demonstrations
    while True:
        print()
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            args.arm,
            args.config,
            mirror_actions,
            render=(args.renderer != "mjviewer"),
            max_fr=args.max_fr,
        )

        print("Keep traj?", not discard_traj)

        if not args.debug:
            if discard_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split("/")[-1])
            hdf5_path = gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, excluded_episodes=excluded_eps
            )
            convert_to_robomimic_format(hdf5_path)
