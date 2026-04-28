import json
from pathlib import Path

import mujoco
import numpy as np


def _axis_angle_from_rot(rot):
    trace = np.trace(rot)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return angle * axis / axis_norm


def _rpy_to_rot(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    c_roll, s_roll = np.cos(roll), np.sin(roll)
    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [
                c_yaw * c_pitch,
                -s_yaw * c_roll + c_yaw * s_pitch * s_roll,
                s_yaw * s_roll + c_yaw * s_pitch * c_roll,
            ],
            [
                s_yaw * c_pitch,
                c_yaw * c_roll + s_yaw * s_pitch * s_roll,
                -c_yaw * s_roll + s_yaw * s_pitch * c_roll,
            ],
            [-s_pitch, c_pitch * s_roll, c_pitch * c_roll],
        ],
        dtype=np.float64,
    )


def _rot_to_rpy(rot):
    rot = np.asarray(rot, dtype=np.float64)
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _rot_to_quat(rot):
    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / scale,
                (rot[2, 1] - rot[1, 2]) * scale,
                (rot[0, 2] - rot[2, 0]) * scale,
                (rot[1, 0] - rot[0, 1]) * scale,
            ],
            dtype=np.float64,
        )

    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    if rot[1, 1] > rot[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        return np.array(
            [
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ],
            dtype=np.float64,
        )

    scale = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
    return np.array(
        [
            (rot[1, 0] - rot[0, 1]) / scale,
            (rot[0, 2] + rot[2, 0]) / scale,
            (rot[1, 2] + rot[2, 1]) / scale,
            0.25 * scale,
        ],
        dtype=np.float64,
    )


class TBlockToBinEnv:
    """MuJoCo OMY task env, following the tutorial's delta_eef_pose control style."""

    def __init__(self, cfg_path="configs/t_block_to_bin.json", seed=None, action_type=None):
        self.root = Path(__file__).resolve().parents[2]
        self.cfg_path = self.root / cfg_path
        with self.cfg_path.open("r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.model = mujoco.MjModel.from_xml_path(str(self.root / self.cfg["xml_file"]))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self._renderers = {}

        self.action_type = action_type or self.cfg.get("control_mode", "delta_eef_pose")
        self.robot_joint_names = self.cfg["robot_joint_names"]
        self.arm_actuator_names = self.cfg.get("arm_actuator_names", self.robot_joint_names)
        self.gripper_joint_names = self.cfg.get("gripper_joint_names", [])
        self.gripper_actuator_names = self.cfg.get(
            "gripper_actuator_names",
            self.gripper_joint_names,
        )
        self.objects = self.cfg["objects"]

        endpoint_cfg = self.cfg["end_effector"]
        self.end_effector_type = endpoint_cfg["type"]
        self.end_effector_name = endpoint_cfg["name"]
        self.end_effector_id = mujoco.mj_name2id(
            self.model,
            self._endpoint_obj_type,
            self.end_effector_name,
        )
        if self.end_effector_id < 0:
            raise ValueError(f"Missing end-effector {self.end_effector_name!r}.")

        self.robot_qpos_addrs = [
            self.model.joint(name).qposadr[0] for name in self.robot_joint_names
        ]
        self.robot_dof_addrs = [
            self.model.joint(name).dofadr[0] for name in self.robot_joint_names
        ]
        self.gripper_qpos_addrs = [
            self.model.joint(name).qposadr[0] for name in self.gripper_joint_names
        ]
        self.robot_joint_ranges = np.array(
            [self.model.joint(name).range for name in self.robot_joint_names],
            dtype=np.float64,
        )
        self.arm_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.arm_actuator_names
        ]
        self.gripper_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.gripper_actuator_names
        ]
        missing = [
            name
            for name, actuator_id in zip(
                self.arm_actuator_names + self.gripper_actuator_names,
                self.arm_actuator_ids + self.gripper_actuator_ids,
            )
            if actuator_id < 0
        ]
        if missing:
            raise ValueError(f"Missing actuator(s): {missing}")

        self.free_joint_addrs = {
            name: self.model.joint(f"{name}_free").qposadr[0] for name in self.objects
        }
        self.free_joint_dof_addrs = {
            name: self.model.joint(f"{name}_free").dofadr[0] for name in self.objects
        }
        self.target_marker_name = self.cfg.get("target_marker_body", "teleop_target_marker")
        self.target_marker_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.target_marker_name,
        )
        self.right_gripper_geom_ids = self._geom_ids_for_body_prefix("rh_p12_rn_r")
        self.left_gripper_geom_ids = self._geom_ids_for_body_prefix("rh_p12_rn_l")
        self.gripper_geom_ids = self.right_gripper_geom_ids | self.left_gripper_geom_ids
        self.object_geom_ids = self._geom_ids_for_bodies(self.objects)
        self.object_geom_to_name = self._geom_id_to_body_name(self.objects)
        self.right_grip_pad_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "rh_r2_grip_pad",
        )
        self.left_grip_pad_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "rh_l2_grip_pad",
        )
        self.table_top_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "table_top",
        )

        self.home_qpos = np.asarray(self.cfg["home_qpos"], dtype=np.float64)
        self.leader_home_qpos = np.asarray(
            self.cfg.get("leader_home_qpos", self.cfg["home_qpos"]),
            dtype=np.float64,
        )
        self.action_dim = len(self.robot_joint_names) + 1
        if self.home_qpos.shape != (self.action_dim,):
            raise ValueError(f"Expected home_qpos length {self.action_dim}.")

        self.q = self.home_qpos.copy()
        self.p0 = np.zeros(3, dtype=np.float64)
        self.R0 = np.eye(3, dtype=np.float64)
        self.last_action = np.zeros(7, dtype=np.float64)
        self.gripper_contact_hold = False
        self.held_gripper_target = self.cfg["gripper_closed"]
        self.grasped_object = None
        self.grasp_rel_pos = None
        self.grasp_rel_rot = None
        self.reset(seed=seed)

    @property
    def _endpoint_obj_type(self):
        if self.end_effector_type == "site":
            return mujoco.mjtObj.mjOBJ_SITE
        if self.end_effector_type == "body":
            return mujoco.mjtObj.mjOBJ_BODY
        raise ValueError(f"Unsupported endpoint type: {self.end_effector_type}")

    def reset(self, seed=None, leader_pose=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if leader_pose is None:
            leader_pose = self.cfg.get("reset_leader_pose", False)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.gripper_contact_hold = False
        self.held_gripper_target = self.cfg["gripper_closed"]
        self._release_grasped_object()

        if leader_pose:
            qpos = self.leader_home_qpos.copy()
        else:
            qpos = self._get_keyboard_home_qpos()

        self._set_robot_qpos(qpos)
        self._apply_action_to_ctrl(qpos)
        self.q = qpos.copy()
        self._randomize_objects()
        mujoco.mj_forward(self.model, self.data)
        self.p0, self.R0 = self.get_end_effector_pose()
        self.set_target_marker(self.p0, self.R0)
        return self.get_observation()

    def step(self, action, gripper_mode="binary", n_substeps=0):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected tutorial action shape (7,), got {action.shape}.")

        if self.action_type == "delta_eef_pose":
            self.p0 = self._clamp_xyz(self.p0 + action[:3])
            self.R0 = self.R0 @ _rpy_to_rot(action[3:6])
            q_arm = self.solve_ik_pose(
                target_pos=self.p0,
                target_rot=self.R0,
                q_init=self.get_robot_qpos(),
                max_iters=100,
                step_scale=0.9,
                max_dq=0.08,
            )
        elif self.action_type == "eef_pose":
            self.p0 = self._clamp_xyz(action[:3])
            self.R0 = _rpy_to_rot(action[3:6])
            q_arm = self.solve_ik_pose(self.p0, self.R0, q_init=self.get_robot_qpos())
        elif self.action_type == "joint":
            q_arm = action[: len(self.robot_joint_names)]
        elif self.action_type == "delta_joint":
            q_arm = self.get_robot_qpos() + action[: len(self.robot_joint_names)]
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        q_arm = np.clip(q_arm, self.robot_joint_ranges[:, 0], self.robot_joint_ranges[:, 1])
        close_requested = action[-1] > 0.5
        if self.action_type in {"eef_pose", "delta_eef_pose"} and gripper_mode == "binary":
            if close_requested:
                gripper = (
                    self.held_gripper_target
                    if self.gripper_contact_hold
                    else self.cfg["gripper_closed"]
                )
            else:
                self.gripper_contact_hold = False
                self.held_gripper_target = self.cfg["gripper_closed"]
                self._release_grasped_object()
                gripper = self.cfg["gripper_open"]
        else:
            gripper = np.clip(action[-1], self.cfg["gripper_open"], self.cfg["gripper_closed"])

        self.q = np.concatenate([q_arm, [gripper]], dtype=np.float64)
        self.last_action = action.copy()
        self._apply_action_to_ctrl(self.q)
        if n_substeps > 0:
            self.step_env(n_substeps=n_substeps)
        return self.get_observation()

    def step_env(self, n_substeps=1):
        self._apply_action_to_ctrl(self.q)
        if self.cfg.get("teleop", {}).get("gravity_compensation", True):
            for _ in range(n_substeps):
                mujoco.mj_forward(self.model, self.data)
                self.data.qfrc_applied[:] = 0.0
                self.data.qfrc_applied[self.robot_dof_addrs] = self.data.qfrc_bias[
                    self.robot_dof_addrs
                ]
                mujoco.mj_step(self.model, self.data, nstep=1)
                self._maybe_hold_gripper_after_contact()
                self._maybe_attach_grasped_object()
                self._update_grasped_object_pose()
            self.data.qfrc_applied[:] = 0.0
        else:
            for _ in range(n_substeps):
                mujoco.mj_step(self.model, self.data, nstep=1)
                self._maybe_hold_gripper_after_contact()
                self._maybe_attach_grasped_object()
                self._update_grasped_object_pose()
        return self.get_observation()

    def get_observation(self):
        arm_qpos = self.get_robot_qpos()
        gripper_qpos = self.get_gripper_qpos()
        target_pos = self.get_body_pos(self.cfg["target_object"]).astype(np.float32)
        bin_pos = self.get_body_pos(self.cfg["trash_bin"]["body_name"]).astype(np.float32)
        eef_pos, eef_rot = self.get_end_effector_pose()
        gripper_state = float(gripper_qpos > 0.5)
        return {
            "joint_pos": np.concatenate([arm_qpos, [gripper_qpos]], dtype=np.float32),
            "eef_pos": eef_pos.astype(np.float32),
            "eef_pose": np.concatenate(
                [eef_pos, _rot_to_rpy(eef_rot), [gripper_state]],
                dtype=np.float32,
            ),
            "target_pos": target_pos,
            "bin_pos": bin_pos,
            "success": np.array([self.check_success()], dtype=np.float32),
        }

    def grab_image(self, return_side=False, width=640, height=480):
        agent = self.get_camera_rgb("agentview", width=width, height=height)
        wrist = self.get_camera_rgb("egocentric", width=width, height=height)
        if return_side:
            side = self.get_camera_rgb("sideview", width=width, height=height)
            return agent, wrist, side
        return agent, wrist

    def get_camera_rgb(self, camera_name="agentview", width=640, height=480):
        key = (int(width), int(height))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render().copy()

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def get_body_pos(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_body_rot(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xmat[body_id].reshape(3, 3).copy()

    def get_site_pos(self, site_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id].copy()

    def get_end_effector_pose(self):
        if self.end_effector_type == "site":
            pos = self.data.site_xpos[self.end_effector_id].copy()
            rot = self.data.site_xmat[self.end_effector_id].reshape(3, 3).copy()
        else:
            pos = self.data.xpos[self.end_effector_id].copy()
            rot = self.data.xmat[self.end_effector_id].reshape(3, 3).copy()
        return pos, rot

    def get_end_effector_pos(self):
        pos, _ = self.get_end_effector_pose()
        return pos

    def get_gripper_tip_positions(self):
        if self.left_grip_pad_geom_id < 0 or self.right_grip_pad_geom_id < 0:
            eef_pos, eef_rot = self.get_end_effector_pose()
            closing_axis = eef_rot[:, 0]
            return {
                "left": eef_pos - 0.055 * closing_axis,
                "right": eef_pos + 0.055 * closing_axis,
            }
        return {
            "left": self.data.geom_xpos[self.left_grip_pad_geom_id].copy(),
            "right": self.data.geom_xpos[self.right_grip_pad_geom_id].copy(),
        }

    def get_projection_plane_z(self):
        if self.table_top_geom_id >= 0:
            return float(
                self.data.geom_xpos[self.table_top_geom_id][2]
                + self.model.geom_size[self.table_top_geom_id][2]
            )
        return float(self.cfg["workspace"]["z"])

    def get_robot_qpos(self):
        return np.array([self.data.qpos[addr] for addr in self.robot_qpos_addrs])

    def get_gripper_qpos(self):
        if not self.gripper_qpos_addrs:
            return 0.0
        values = np.array([self.data.qpos[addr] for addr in self.gripper_qpos_addrs])
        return float(values.mean())

    def get_ee_pose(self):
        pos, rot = self.get_end_effector_pose()
        return np.concatenate([pos, _rot_to_rpy(rot)], dtype=np.float32)

    def get_control_target_joint_pos(self):
        return self.q.astype(np.float32).copy()

    def get_control_target_eef_pose(self):
        gripper_state = float(self.q[-1] > 0.5)
        return np.concatenate(
            [self.p0, _rot_to_rpy(self.R0), [gripper_state]],
            dtype=np.float32,
        )

    def get_object_pose(self, pad=10):
        poses = np.zeros((pad, 6), dtype=np.float32)
        names = []
        for idx, object_name in enumerate(self.objects[:pad]):
            pos = self.get_body_pos(object_name)
            rot = self.get_body_rot(object_name)
            poses[idx] = np.concatenate([pos, _rot_to_rpy(rot)]).astype(np.float32)
            names.append(object_name)

        for idx in range(len(names), pad):
            names.append(f"pad_{idx}")

        receptacle_q = {
            "poses": np.zeros((pad,), dtype=np.float32),
            "names": [f"pad_{idx}" for idx in range(pad)],
        }
        return {"poses": poses, "names": names}, receptacle_q

    def set_object_pose(self, obj_pose, obj_names, obj_q_states=None, obj_q_names=None):
        if isinstance(obj_names, str):
            obj_names = [name for name in obj_names.split(",") if name]
        obj_pose = np.asarray(obj_pose, dtype=np.float64)
        for name, pose in zip(obj_names, obj_pose):
            if not name or name.startswith("pad_") or name not in self.free_joint_addrs:
                continue
            pos = pose[:3]
            quat = _rot_to_quat(_rpy_to_rot(pose[3:6]))
            qpos_addr = self.free_joint_addrs[name]
            self.data.qpos[qpos_addr : qpos_addr + 3] = pos
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat

        if obj_q_states is not None and obj_q_names is not None:
            if isinstance(obj_q_names, str):
                obj_q_names = [name for name in obj_q_names.split(",") if name]
            obj_q_states = np.asarray(obj_q_states, dtype=np.float64)
            for name, value in zip(obj_q_names, obj_q_states):
                if not name or name.startswith("pad_"):
                    continue
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    continue
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

        mujoco.mj_forward(self.model, self.data)

    def check_success(self):
        target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.cfg["target_site"],
        )
        bin_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.cfg["trash_bin"]["success_site"],
        )
        target_pos = self.data.site_xpos[target_site_id]
        bin_center = self.data.site_xpos[bin_site_id]
        bin_half_size = self.model.site_size[bin_site_id]
        lower = bin_center - bin_half_size
        upper = bin_center + bin_half_size
        return bool(np.all(target_pos >= lower) and np.all(target_pos <= upper))

    def set_target_marker(self, pos, rot=None):
        if self.target_marker_id < 0:
            return
        self.model.body_pos[self.target_marker_id] = np.asarray(pos, dtype=np.float64)
        if rot is not None:
            self.model.body_quat[self.target_marker_id] = _rot_to_quat(rot)
        mujoco.mj_forward(self.model, self.data)

    def downward_gripper_rotation(self):
        return _rpy_to_rot(np.deg2rad([90.0, 0.0, 90.0]))

    def solve_ik_pose(
        self,
        target_pos,
        target_rot,
        q_init=None,
        max_iters=120,
        damping=1e-4,
        step_scale=0.8,
        max_dq=0.08,
        pos_tol=1e-3,
        rot_tol=np.radians(1.0),
        rot_weight=0.3,
    ):
        return self._solve_ik(
            target_pos=target_pos,
            target_rot=target_rot,
            q_init=q_init,
            max_iters=max_iters,
            damping=damping,
            step_scale=step_scale,
            max_dq=max_dq,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            rot_weight=rot_weight,
        )

    def solve_ik_position(
        self,
        target_pos,
        q_init=None,
        max_iters=100,
        damping=1e-4,
        step_scale=0.8,
        max_dq=0.08,
        tol=1e-3,
    ):
        return self._solve_ik(
            target_pos=target_pos,
            target_rot=None,
            q_init=q_init,
            max_iters=max_iters,
            damping=damping,
            step_scale=step_scale,
            max_dq=max_dq,
            pos_tol=tol,
        )

    def _solve_ik(
        self,
        target_pos,
        target_rot=None,
        q_init=None,
        max_iters=100,
        damping=1e-4,
        step_scale=0.8,
        max_dq=0.08,
        pos_tol=1e-3,
        rot_tol=np.radians(1.0),
        rot_weight=0.3,
    ):
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_rot = None if target_rot is None else np.asarray(target_rot, dtype=np.float64)
        q_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        q = self.get_robot_qpos() if q_init is None else np.asarray(q_init, dtype=np.float64).copy()
        q = np.clip(q, self.robot_joint_ranges[:, 0], self.robot_joint_ranges[:, 1])
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        try:
            for _ in range(max_iters):
                self.data.qpos[self.robot_qpos_addrs] = q
                mujoco.mj_forward(self.model, self.data)

                current_pos, current_rot = self.get_end_effector_pose()
                pos_error = target_pos - current_pos

                jacp.fill(0.0)
                jacr.fill(0.0)
                if self.end_effector_type == "site":
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.end_effector_id)
                else:
                    mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.end_effector_id)
                jac_pos = jacp[:, self.robot_dof_addrs]

                if target_rot is None:
                    err = pos_error
                    jac = jac_pos
                    done = np.linalg.norm(pos_error) < pos_tol
                else:
                    rot_error = current_rot @ _axis_angle_from_rot(current_rot.T @ target_rot)
                    err = np.concatenate([pos_error, rot_weight * rot_error])
                    jac_rot = jacr[:, self.robot_dof_addrs]
                    jac = np.vstack([jac_pos, rot_weight * jac_rot])
                    done = (
                        np.linalg.norm(pos_error) < pos_tol
                        and np.linalg.norm(rot_error) < rot_tol
                    )
                if done:
                    break

                lhs = jac @ jac.T + damping * np.eye(jac.shape[0])
                dq = jac.T @ np.linalg.solve(lhs, err)
                dq = np.clip(step_scale * dq, -max_dq, max_dq)
                q = np.clip(q + dq, self.robot_joint_ranges[:, 0], self.robot_joint_ranges[:, 1])
        finally:
            self.data.qpos[:] = q_backup
            self.data.qvel[:] = qvel_backup
            mujoco.mj_forward(self.model, self.data)

        return q

    def _get_keyboard_home_qpos(self):
        home_pose = self.cfg.get("keyboard_home_pose", {})
        if not home_pose and "keyboard_home_qpos" in self.cfg:
            return np.asarray(self.cfg["keyboard_home_qpos"], dtype=np.float64).copy()

        target_pos = np.asarray(home_pose.get("pos", [0.3, -0.1, 1.0]), dtype=np.float64)
        target_rot = _rpy_to_rot(np.deg2rad(home_pose.get("rpy_deg", [90.0, 0.0, 90.0])))
        q_init = np.asarray(
            self.cfg.get("keyboard_home_qpos", np.zeros(self.action_dim)),
            dtype=np.float64,
        )[: len(self.robot_joint_names)]
        q_arm = self.solve_ik_pose(
            target_pos=target_pos,
            target_rot=target_rot,
            q_init=q_init,
            max_iters=500,
            step_scale=0.9,
            max_dq=0.08,
            rot_weight=0.25,
        )
        return np.concatenate([q_arm, [self.cfg["gripper_open"]]], dtype=np.float64)

    def _apply_action_to_ctrl(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        for value, actuator_id in zip(qpos[: len(self.robot_joint_names)], self.arm_actuator_ids):
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(value, low, high)

        gripper_value = qpos[-1]
        for actuator_id in self.gripper_actuator_ids:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(gripper_value, low, high)

    def _maybe_hold_gripper_after_contact(self):
        teleop_cfg = self.cfg.get("teleop", {})
        if not teleop_cfg.get("stop_gripper_on_contact", True):
            return
        if self.q[-1] <= (self.cfg["gripper_open"] + self.cfg["gripper_closed"]) * 0.5:
            return
        if self.gripper_contact_hold:
            self.q[-1] = self.held_gripper_target
            self._apply_action_to_ctrl(self.q)
            return
        contact_object = self._graspable_contact_object()
        if contact_object is None:
            return

        margin = teleop_cfg.get("gripper_hold_margin", 0.06)
        current = self.get_gripper_qpos()
        self.held_gripper_target = np.clip(
            current + margin,
            self.cfg["gripper_open"],
            self.cfg["gripper_closed"],
        )
        self.gripper_contact_hold = True
        self.q[-1] = self.held_gripper_target
        self._apply_action_to_ctrl(self.q)

    def _maybe_attach_grasped_object(self):
        teleop_cfg = self.cfg.get("teleop", {})
        if not teleop_cfg.get("grasp_assist", True):
            return
        if self.grasped_object is not None:
            return
        if not self.gripper_contact_hold:
            return
        if self.get_gripper_qpos() < teleop_cfg.get("grasp_attach_min_q", 0.45):
            return
        contact_object = self._graspable_contact_object()
        if contact_object is None:
            return
        self._attach_grasped_object(contact_object)

    def _attach_grasped_object(self, object_name):
        eef_pos, eef_rot = self.get_end_effector_pose()
        obj_pos = self.get_body_pos(object_name)
        obj_rot = self.get_body_rot(object_name)
        self.grasped_object = object_name
        self.grasp_rel_pos = eef_rot.T @ (obj_pos - eef_pos)
        self.grasp_rel_rot = eef_rot.T @ obj_rot

    def _update_grasped_object_pose(self):
        if self.grasped_object is None:
            return
        if self.get_gripper_qpos() < self.cfg.get("teleop", {}).get("grasp_release_q", 0.35):
            self._release_grasped_object()
            return

        eef_pos, eef_rot = self.get_end_effector_pose()
        obj_pos = eef_pos + eef_rot @ self.grasp_rel_pos
        obj_rot = eef_rot @ self.grasp_rel_rot
        qpos_addr = self.free_joint_addrs[self.grasped_object]
        dof_addr = self.free_joint_dof_addrs[self.grasped_object]
        self.data.qpos[qpos_addr : qpos_addr + 3] = obj_pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = _rot_to_quat(obj_rot)
        self.data.qvel[dof_addr : dof_addr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _release_grasped_object(self):
        self.grasped_object = None
        self.grasp_rel_pos = None
        self.grasp_rel_rot = None

    def _graspable_contact_object(self):
        contacts_by_object = self._gripper_object_contact_sides_by_object()
        for object_name in self.objects:
            sides = contacts_by_object.get(object_name)
            if sides and sides["left"] and sides["right"]:
                return object_name
        return None

    def _gripper_object_contact_sides(self):
        sides = {"left": False, "right": False}
        for object_sides in self._gripper_object_contact_sides_by_object().values():
            sides["left"] = sides["left"] or object_sides["left"]
            sides["right"] = sides["right"] or object_sides["right"]
        return sides

    def _gripper_object_contact_sides_by_object(self):
        contacts_by_object = {
            object_name: {"left": False, "right": False} for object_name in self.objects
        }
        if not self.gripper_geom_ids or not self.object_geom_ids:
            return contacts_by_object
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 in self.object_geom_ids:
                object_geom = geom1
                gripper_geom = geom2
            elif geom2 in self.object_geom_ids:
                object_geom = geom2
                gripper_geom = geom1
            else:
                continue

            object_name = self.object_geom_to_name.get(object_geom)
            if object_name is None:
                continue
            if gripper_geom in self.left_gripper_geom_ids:
                contacts_by_object[object_name]["left"] = True
            if gripper_geom in self.right_gripper_geom_ids:
                contacts_by_object[object_name]["right"] = True
        return contacts_by_object

    def _geom_ids_for_body_prefix(self, prefix):
        geom_ids = set()
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and body_name.startswith(prefix):
                geom_ids.add(geom_id)
        return geom_ids

    def _geom_ids_for_bodies(self, body_names):
        body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        }
        return {
            geom_id
            for geom_id in range(self.model.ngeom)
            if self.model.geom_bodyid[geom_id] in body_ids
        }

    def _geom_id_to_body_name(self, body_names):
        body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name): name
            for name in body_names
        }
        return {
            geom_id: body_ids[self.model.geom_bodyid[geom_id]]
            for geom_id in range(self.model.ngeom)
            if self.model.geom_bodyid[geom_id] in body_ids
        }

    def _set_robot_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        self.data.qpos[self.robot_qpos_addrs] = qpos[: len(self.robot_joint_names)]
        for addr in self.gripper_qpos_addrs:
            self.data.qpos[addr] = qpos[-1]

    def _randomize_objects(self):
        placements = self._sample_object_placements(len(self.objects))
        object_z = self.cfg.get("object_z", {})
        for obj_name, (x, y, yaw) in zip(self.objects, placements):
            z = object_z.get(obj_name, self.cfg["workspace"]["z"])
            self._set_free_body_pose(obj_name, np.array([x, y, z], dtype=np.float64), yaw)

    def _sample_object_placements(self, n_objects):
        workspace = self.cfg["workspace"]
        x_range = workspace["x_range"]
        y_range = workspace["y_range"]
        min_dist = workspace["min_xy_dist"]
        bin_pos = self.model.body(self.cfg["trash_bin"]["body_name"]).pos[:2]
        bin_avoid_radius = self.cfg["trash_bin"].get("avoid_radius", 0.13)

        placements = []
        for _ in range(n_objects):
            for _attempt in range(1000):
                x = self.rng.uniform(*x_range)
                y = self.rng.uniform(*y_range)
                xy = np.array([x, y], dtype=np.float64)
                yaw = self.rng.uniform(-np.pi, np.pi)
                far_from_objects = all(np.linalg.norm(xy - p[:2]) > min_dist for p in placements)
                far_from_bin = np.linalg.norm(xy - bin_pos) > bin_avoid_radius
                if far_from_objects and far_from_bin:
                    placements.append(np.array([x, y, yaw], dtype=np.float64))
                    break
            else:
                raise RuntimeError("Could not sample non-overlapping object placements.")
        return placements

    def _set_free_body_pose(self, object_name, pos, yaw):
        qpos_addr = self.free_joint_addrs[object_name]
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
        self.data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat

    def _clamp_xyz(self, xyz):
        bounds = self.cfg["teleop"]["xyz_bounds"]
        xyz = np.asarray(xyz, dtype=np.float64).copy()
        xyz[0] = np.clip(xyz[0], *bounds["x"])
        xyz[1] = np.clip(xyz[1], *bounds["y"])
        xyz[2] = np.clip(xyz[2], *bounds["z"])
        return xyz
