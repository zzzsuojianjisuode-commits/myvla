import sys
import time
from pathlib import Path

import glfw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers import load_controller
from src.env import TBlockToBinEnv
from src.viewer import KeyboardTeleopViewer


CONTROLS = """Keyboard teleop
W/S: delta x -/+
A/D: delta y -/+
R/F: z up/down
Q/E, arrows: rotate end effector
Space: toggle gripper
Z: reset episode
H: reset home
Mouse drag/scroll: adjust view
Esc: quit
"""


def main():
    env = TBlockToBinEnv(seed=0, action_type="delta_eef_pose")
    env.reset(leader_pose=False)
    controller = load_controller("keyboard", env.cfg)
    controller.reset(env)

    viewer = KeyboardTeleopViewer(env.model, env.data, title="OMY keyboard teleop")
    viewer_cfg = env.cfg.get("viewer", {}).get("keyboard", {})
    viewer.set_camera(
        lookat=viewer_cfg.get("lookat", [0.4, 0.0, 0.6]),
        distance=viewer_cfg.get("distance", 2.0),
        azimuth=viewer_cfg.get("azimuth", 180),
        elevation=viewer_cfg.get("elevation", -40),
    )
    viewer.set_camera_previews(env.cfg.get("image_cameras", []))

    teleop_cfg = env.cfg.get("teleop", {})
    control_period = 1.0 / teleop_cfg.get("control_hz", 20)
    sim_substeps = teleop_cfg.get("sim_substeps_per_frame", 5)
    next_control_time = env.data.time
    wall_start = time.time()
    sim_start = env.data.time
    obs = env.get_observation()

    print(CONTROLS)
    try:
        while viewer.is_alive():
            if viewer.consume_key(glfw.KEY_Z) or viewer.consume_key(glfw.KEY_H):
                env.reset(leader_pose=False)
                controller.reset(env)
                next_control_time = env.data.time
                wall_start = time.time()
                sim_start = env.data.time

            if env.data.time >= next_control_time:
                action = controller.get_action(viewer)
                env.step(action)
                next_control_time += control_period

            obs = env.step_env(n_substeps=sim_substeps)
            eef_pos, eef_rot = env.get_end_effector_pose()
            env.set_target_marker(eef_pos, eef_rot)

            eef_target_err = np.linalg.norm(obs["eef_pos"] - env.p0)
            viewer.set_text(
                CONTROLS,
                (
                    f"target xyz: {env.p0.round(3)}\n"
                    f"eef xyz: {obs['eef_pos'].round(3)}\n"
                    f"eef-target err: {eef_target_err:.3f}\n"
                    f"gripper target/q: {env.q[-1]:.2f}/{obs['joint_pos'][-1]:.2f}\n"
                    f"gripper hold: {env.gripper_contact_hold}\n"
                    f"grasped: {env.grasped_object or '-'}\n"
                    f"sim time: {env.data.time:.2f}s\n"
                    f"success: {bool(obs['success'][0])}"
                ),
            )
            viewer.render()
            target_wall_time = wall_start + (env.data.time - sim_start)
            sleep_time = target_wall_time - time.time()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.02))
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
