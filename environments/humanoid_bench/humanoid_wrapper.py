"""Adapter to expose `humanoid_bench` environments to SAILOR.

This wrapper normalizes/unnormalizes actions to [-1, 1], exposes camera
observations as `agentview_image` and `robot0_eye_in_hand_image`, and
provides the optional flattened `state` key when available.

It is intentionally small: higher-level wrappers (e.g. `wrappers.SelectAction`)
are applied by `train_sailor` to present a consistent API to the training
stack.
"""

from collections import OrderedDict
import numpy as np
from gym import spaces


class HumanoidBenchWrapper:
    def __init__(self, env, config, keys=None, action_repeat=1, n_succ_before_term=5):
        self.env = env
        self.config = config
        self.action_repeat = int(action_repeat)
        self.n_succ_before_term = int(n_succ_before_term)

        if keys is None:
            keys = ["image_left_eye", "image_right_eye"]

        # Build a normalized action space in [-1, 1]
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "low"):
            low = -np.ones_like(self.env.action_space.low, dtype=np.float32)
            high = np.ones_like(self.env.action_space.high, dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, shape=self.env.action_space.shape, dtype=np.float32)
        else:
            # Fallback: unknown action space
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(0,), dtype=np.float32)

        # Observation space: images plus state
        img_shape = (int(config.image_size), int(config.image_size), 3)
        observation_space = spaces.Dict()
        observation_space["agentview_image"] = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        observation_space["robot0_eye_in_hand_image"] = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        
        # Try to detect state dimension from a sample observation
        try:
            sample = self._safe_reset_sample()
            if isinstance(sample, np.ndarray):
                state_dim = int(np.prod(sample.shape))
                observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        except Exception:
            # Default to 190D for H1 humanoid if detection fails
            observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(190,), dtype=np.float32)

        self.observation_space = observation_space

        # Internal counters
        self.step_count = 0
        self.success_count = 0
        self.prev_success = False

    def _safe_reset_sample(self):
        out = self.env.reset()
        # Support Gymnasium-style (obs, info) return or older gym (obs,)
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        return out

    def _process_raw_obs(self, raw_obs):
        """Process raw observation from environment.
        
        When obs_wrapper="False", raw_obs is a flat numpy array (privileged state).
        We need to manually render images and build the observation dict.
        """
        obs = OrderedDict()
        
        # Manually render images from cameras
        img_left = self.env.unwrapped.mujoco_renderer.render(
            render_mode="rgb_array", camera_name="left_eye_camera"
        )
        img_right = self.env.unwrapped.mujoco_renderer.render(
            render_mode="rgb_array", camera_name="right_eye_camera"
        )
        
        obs["agentview_image"] = np.asarray(img_left, dtype=np.uint8)
        obs["robot0_eye_in_hand_image"] = np.asarray(img_right, dtype=np.uint8)
        
        # Include state (raw_obs is the privileged state array)
        obs["state"] = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        
        return obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        raw_obs = out[0] if isinstance(out, tuple) and len(out) >= 1 else out
        obs = self._process_raw_obs(raw_obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self.step_count = 0
        self.success_count = 0
        self.prev_success = False
        return obs

    def step(self, action):
        # Expect `action` as a numpy array in [-1,1]; map back to env action space
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "low"):
            low = self.env.action_space.low.astype(np.float32)
            high = self.env.action_space.high.astype(np.float32)
            real_action = (action + 1.0) / 2.0 * (high - low) + low
        else:
            real_action = action

        # Repeat for `action_repeat` steps
        raw_obs = None
        rew = 0.0
        done = False
        info = {}
        for _ in range(self.action_repeat):
            out = self.env.step(real_action)
            # gymnasium-style: (obs, reward, terminated, truncated, info)
            if isinstance(out, tuple) and len(out) == 5:
                raw_obs, rew, terminated, truncated, info = out
                done = bool(terminated or truncated)
            elif isinstance(out, tuple) and len(out) == 4:
                raw_obs, rew, done, info = out
            else:
                # Unexpected format; try to unpack defensively
                try:
                    raw_obs = out[0]
                except Exception:
                    raw_obs = None

        self.step_count += 1

        success = bool(info.get("success", False))
        if rew == 1 or success:
            if self.prev_success:
                self.success_count += 1
            else:
                self.success_count = 1
            self.prev_success = True
        else:
            self.success_count = 0
            self.prev_success = False

        if self.success_count >= self.n_succ_before_term:
            done = True
            info["success"] = True
        else:
            info["success"] = False

        obs = self._process_raw_obs(raw_obs)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = done
        info["orig_reward"] = rew

        # Reward shifting for compatibility with dataset conventions
        reward = float(rew) - 1.0
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        try:
            return self.env.render(*args, **kwargs)
        except Exception:
            # Some backends use `mujoco_renderer.render`
            try:
                return self.env.mujoco_renderer.render(*args, **kwargs)
            except Exception:
                raise

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

