from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import registry, register

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from src.env import TBlockToBinEnv


GYM_ID = "gym_myvla/TBlockToBin-v0"


class TBlockToBinGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        cfg_path="configs/t_block_to_bin.json",
        image_size=256,
        episode_length=600,
        sim_substeps=8,
        gripper_mode="binary",
        action_type="delta_eef_pose",
        proprio_type="eef_pose",
        env_state_type="eef_pose",
        object_pad=10,
        render_mode="rgb_array",
        terminate_on_success=True,
    ):
        self.render_mode = render_mode
        self.image_size = int(image_size)
        self._max_episode_steps = int(episode_length)
        self.sim_substeps = int(sim_substeps)
        self.gripper_mode = gripper_mode
        self.proprio_type = proprio_type
        self.env_state_type = env_state_type
        self.object_pad = int(object_pad)
        self.terminate_on_success = bool(terminate_on_success)
        self._elapsed_steps = 0
        self._env = TBlockToBinEnv(cfg_path=cfg_path, action_type=action_type)
        self._task_description = self._env.cfg.get(
            "task",
            "pick up the hollow cylinder and place it into the trash bin",
        )
        self.task = self._task_description

        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        env_state_dim = self._env_state_dim()
        env_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env_state_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "agentview": image_space,
                        "egocentric": image_space,
                    }
                ),
                "agent_pos": state_space,
                "environment_state": env_state_space,
            }
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def task_description(self):
        return self._task_description

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        self._env.reset(seed=seed, leader_pose=False)
        return self._observation(), {"is_success": self._env.check_success()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(7)
        self._env.step(action, gripper_mode=self.gripper_mode)
        self._env.step_env(n_substeps=self.sim_substeps)
        self._elapsed_steps += 1

        success = self._env.check_success()
        terminated = bool(success and self.terminate_on_success)
        truncated = self._elapsed_steps >= self._max_episode_steps
        reward = np.float32(1.0 if success else 0.0)
        info = {"is_success": bool(success)}
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        return self._env.get_camera_rgb(
            "agentview",
            width=self.image_size,
            height=self.image_size,
        )

    def close(self):
        self._env.close()

    def _observation(self):
        obs = self._env.get_observation()
        agent_image, wrist_image = self._env.grab_image(
            return_side=False,
            width=self.image_size,
            height=self.image_size,
        )
        return {
            "pixels": {
                "agentview": agent_image.astype(np.uint8),
                "egocentric": wrist_image.astype(np.uint8),
            },
            "agent_pos": self._agent_pos(obs),
            "environment_state": self._environment_state(obs),
        }

    def _agent_pos(self, obs):
        if self.proprio_type == "joint_pos":
            return obs["joint_pos"].astype(np.float32)
        if self.proprio_type == "eef_pose":
            return obs["eef_pose"].astype(np.float32)
        raise ValueError(f"Unknown proprio_type={self.proprio_type!r}.")

    def _env_state_dim(self):
        if self.env_state_type == "eef_pose":
            return 7
        if self.env_state_type == "compact":
            return 13
        if self.env_state_type == "object_pose":
            return self.object_pad * 7
        raise ValueError(f"Unknown env_state_type={self.env_state_type!r}.")

    def _environment_state(self, obs):
        if self.env_state_type == "eef_pose":
            return obs["eef_pose"].astype(np.float32)
        if self.env_state_type == "compact":
            return np.concatenate(
                [obs["eef_pose"], obs["target_pos"], obs["bin_pos"]],
                dtype=np.float32,
            )
        if self.env_state_type == "object_pose":
            obj_states, obj_q_states = self._env.get_object_pose(pad=self.object_pad)
            return self._sorted_object_state(obj_states, obj_q_states)
        raise ValueError(f"Unknown env_state_type={self.env_state_type!r}.")

    def _sorted_object_state(self, obj_states, obj_q_states):
        obj_slots = np.zeros((self.object_pad, 6), dtype=np.float32)
        out_idx = 0
        for src_idx in np.argsort(obj_states["names"]):
            name = obj_states["names"][int(src_idx)]
            if name.startswith("pad_"):
                continue
            if out_idx >= self.object_pad:
                break
            obj_slots[out_idx] = obj_states["poses"][int(src_idx)]
            out_idx += 1

        q_slots = np.zeros((self.object_pad,), dtype=np.float32)
        out_idx = 0
        for src_idx in np.argsort(obj_q_states["names"]):
            name = obj_q_states["names"][int(src_idx)]
            if name.startswith("pad_"):
                continue
            if out_idx >= self.object_pad:
                break
            q_slots[out_idx] = obj_q_states["poses"][int(src_idx)]
            out_idx += 1
        return np.concatenate([obj_slots.reshape(-1), q_slots], dtype=np.float32)


@EnvConfig.register_subclass("myvla")
@dataclass
class MyVLAEnvConfig(EnvConfig):
    task: str | None = "TBlockToBin-v0"
    fps: int = 20
    episode_length: int = 600
    image_size: int = 256
    sim_substeps: int = 8
    gripper_mode: str = "binary"
    action_type: str = "delta_eef_pose"
    proprio_type: str = "eef_pose"
    env_state_type: str = "eef_pose"
    object_pad: int = 10
    cfg_path: str = "configs/t_block_to_bin.json"
    render_mode: str = "rgb_array"
    terminate_on_success: bool = True

    def __post_init__(self):
        image_shape = (self.image_size, self.image_size, 3)
        env_state_shape = (self._env_state_dim(),)
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "pixels.agentview": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
            "pixels.egocentric": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "environment_state": PolicyFeature(type=FeatureType.ENV, shape=env_state_shape),
        }
        self.features_map = {
            ACTION: ACTION,
            "pixels.agentview": f"{OBS_IMAGES}.agentview",
            "pixels.egocentric": f"{OBS_IMAGES}.egocentric",
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
        }

    @property
    def gym_kwargs(self):
        return {
            "cfg_path": self.cfg_path,
            "image_size": self.image_size,
            "episode_length": self.episode_length,
            "sim_substeps": self.sim_substeps,
            "gripper_mode": self.gripper_mode,
            "action_type": self.action_type,
            "proprio_type": self.proprio_type,
            "env_state_type": self.env_state_type,
            "object_pad": self.object_pad,
            "render_mode": self.render_mode,
            "terminate_on_success": self.terminate_on_success,
        }

    def _env_state_dim(self):
        if self.env_state_type == "eef_pose":
            return 7
        if self.env_state_type == "compact":
            return 13
        if self.env_state_type == "object_pose":
            return self.object_pad * 7
        raise ValueError(f"Unknown env_state_type={self.env_state_type!r}.")


if GYM_ID not in registry:
    register(
        id=GYM_ID,
        entry_point="src.lerobot_myvla:TBlockToBinGymEnv",
        max_episode_steps=None,
        order_enforce=False,
        disable_env_checker=True,
    )
