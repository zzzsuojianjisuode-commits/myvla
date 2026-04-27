import argparse
import shutil
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers import load_controller
from src.dataset import (
    build_teleoperation_frame,
    make_teleoperation_dataset,
    write_episode_images,
)
from src.env import TBlockToBinEnv
from src.viewer import KeyboardTeleopViewer


CONTROLS = """Keyboard teleop
W/S: delta x -/+
A/D: delta y -/+
R/F: z up/down
Q/E, arrows: rotate end effector
Space: toggle gripper
C: start/pause recording
Enter: save episode
X: discard episode
Z: reset episode
H: reset home
Mouse drag/scroll: adjust view
Esc: quit
"""


PROJECTION_MARKER_TABLE_PENETRATION = 0.15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(record=True)
    parser.add_argument("--record", action="store_true", help="Record LeRobot episodes.")
    parser.add_argument("--no-record", action="store_false", dest="record")
    parser.add_argument("--dataset-root", default="dataset/teleoperation_dataset")
    parser.add_argument("--repo-id", default="t_block_to_bin")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--object-pad", type=int, default=10)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--no-auto-save-on-success", action="store_true")
    return parser.parse_args()


def resolve_dataset_root(path):
    dataset_root = Path(path).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    return dataset_root


def open_recording_dataset(args, fps):
    if not args.record:
        return None, 0
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")

    dataset_root = resolve_dataset_root(args.dataset_root)
    if dataset_root.exists():
        has_info = (dataset_root / "meta" / "info.json").exists()
        has_tasks = (dataset_root / "meta" / "tasks.parquet").exists()
        if args.overwrite:
            shutil.rmtree(dataset_root)
        elif has_info and has_tasks:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(args.repo_id, root=dataset_root)
            return dataset, dataset.num_episodes
        elif has_info and not has_tasks:
            shutil.rmtree(dataset_root)
        elif not any(dataset_root.iterdir()):
            shutil.rmtree(dataset_root)
        else:
            raise FileExistsError(
                f"{dataset_root} already exists but is not a LeRobot dataset. "
                "Use --overwrite or choose another --dataset-root."
            )

    dataset = make_teleoperation_dataset(
        dataset_root,
        repo_id=args.repo_id,
        fps=fps,
        action_dim=7,
        state_dim=7,
        object_pad=args.object_pad,
        image_size=args.image_size,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    return dataset, 0


def add_gripper_markers(viewer, eef_pos, tip_positions, projection_z):
    viewer.plot_sphere(
        pos=eef_pos,
        radius=0.02,
        rgba=(0.95, 0.05, 0.05, 0.55),
    )
    for tip_pos in tip_positions.values():
        top = np.asarray(tip_pos, dtype=np.float64)
        bottom = top.copy()
        bottom[2] = projection_z - PROJECTION_MARKER_TABLE_PENETRATION
        height = top[2] - bottom[2]
        if height <= 0.005:
            continue
        viewer.plot_cylinder_between_points(
            start=top,
            end=bottom,
            radius=0.006,
            rgba=(0.05, 0.95, 0.05, 0.55),
        )


def main():
    args = parse_args()
    env = TBlockToBinEnv(seed=0, action_type="delta_eef_pose")
    env.reset(leader_pose=False)
    controller = load_controller("keyboard", env.cfg)
    controller.reset(env)

    teleop_cfg = env.cfg.get("teleop", {})
    control_hz = teleop_cfg.get("control_hz", 20)
    control_period = 1.0 / control_hz
    sim_substeps = teleop_cfg.get("sim_substeps_per_frame", 5)
    task = env.cfg.get("task", "")
    config_file_name = str(env.cfg_path.relative_to(ROOT))
    dataset_root = resolve_dataset_root(args.dataset_root)
    dataset_display = (
        dataset_root.relative_to(ROOT) if dataset_root.is_relative_to(ROOT) else dataset_root
    )
    dataset = None
    episode_id = 0
    recording = False
    episode_frame_count = 0
    episode_images = []
    last_action = np.zeros(7, dtype=np.float32)

    viewer = KeyboardTeleopViewer(env.model, env.data, title="OMY keyboard teleop")
    viewer_cfg = env.cfg.get("viewer", {}).get("keyboard", {})
    viewer.set_camera(
        lookat=viewer_cfg.get("lookat", [0.4, 0.0, 0.6]),
        distance=viewer_cfg.get("distance", 2.0),
        azimuth=viewer_cfg.get("azimuth", 180),
        elevation=viewer_cfg.get("elevation", -40),
    )
    viewer.set_camera_previews(env.cfg.get("image_cameras", []))

    next_control_time = env.data.time
    wall_start = time.time()
    sim_start = env.data.time
    obs = env.get_observation()

    def reset_sim():
        nonlocal next_control_time, wall_start, sim_start, obs
        obs = env.reset(leader_pose=False)
        controller.reset(env)
        next_control_time = env.data.time
        wall_start = time.time()
        sim_start = env.data.time

    def discard_episode(reason="discard"):
        nonlocal recording, episode_frame_count
        if dataset is not None and episode_frame_count > 0:
            dataset.clear_episode_buffer()
            print(f"Discard episode ({reason}), frames={episode_frame_count}")
        recording = False
        episode_frame_count = 0
        episode_images.clear()
        reset_sim()

    def save_episode(reason="manual"):
        nonlocal recording, episode_frame_count, episode_id
        if dataset is None:
            return
        if episode_frame_count <= 0:
            print("No recorded frames to save.")
            recording = False
            reset_sim()
            return
        saved_episode_id = episode_id
        dataset.save_episode()
        saved_images = write_episode_images(dataset_root, saved_episode_id, episode_images)
        episode_id += 1
        print(
            f"Saved episode {episode_id} ({reason}), "
            f"frames={episode_frame_count}, images={saved_images}, root={dataset_root}"
        )
        recording = False
        episode_frame_count = 0
        episode_images.clear()
        reset_sim()

    try:
        dataset, episode_id = open_recording_dataset(args, fps=control_hz)
        recording = False
        print(CONTROLS)
        if dataset is not None:
            print(f"Recording dataset: {dataset_root}")
            print("Press C to start/pause recording.")
            print("Press Enter to save, X to discard.")

        while viewer.is_alive() and (dataset is None or episode_id < args.num_episodes):
            if viewer.consume_key(glfw.KEY_Z) or viewer.consume_key(glfw.KEY_H):
                discard_episode(reason="reset")
                continue

            if dataset is not None and viewer.consume_key(glfw.KEY_C):
                recording = not recording
                print("Start recording." if recording else "Pause recording.")

            save_pressed = viewer.consume_key(glfw.KEY_ENTER) or viewer.consume_key(
                glfw.KEY_KP_ENTER
            )
            if dataset is not None and save_pressed:
                save_episode(reason="manual")
                continue

            if dataset is not None and viewer.consume_key(glfw.KEY_X):
                discard_episode(reason="manual")
                continue

            should_record_frame = False
            if env.data.time >= next_control_time:
                action = controller.get_action(viewer)
                last_action = action.astype(np.float32)
                env.step(action)
                next_control_time += control_period
                should_record_frame = dataset is not None and recording

            obs = env.step_env(n_substeps=sim_substeps)
            eef_pos, eef_rot = env.get_end_effector_pose()
            env.set_target_marker(eef_pos, eef_rot)
            add_gripper_markers(
                viewer,
                eef_pos,
                env.get_gripper_tip_positions(),
                env.get_projection_plane_z(),
            )

            if should_record_frame:
                agent_image = viewer.capture_fixed_camera_rgb(
                    "agentview",
                    width=args.image_size,
                    height=args.image_size,
                )
                wrist_image = viewer.capture_fixed_camera_rgb(
                    "egocentric",
                    width=args.image_size,
                    height=args.image_size,
                )
                frame = build_teleoperation_frame(
                    env=env,
                    action=last_action,
                    task=task,
                    config_file_name=config_file_name,
                    image_size=args.image_size,
                    object_pad=args.object_pad,
                    agent_image=agent_image,
                    wrist_image=wrist_image,
                )
                dataset.add_frame(frame)
                episode_images.append(
                    {
                        "observation.image": agent_image.copy(),
                        "observation.wrist_image": wrist_image.copy(),
                    }
                )
                episode_frame_count += 1
                if (
                    not args.no_auto_save_on_success
                    and episode_frame_count > 0
                    and bool(obs["success"][0])
                ):
                    save_episode(reason="success")
                    continue

            eef_target_err = np.linalg.norm(obs["eef_pos"] - env.p0)
            record_text = ""
            if dataset is not None:
                state = "ON" if recording else "OFF - press C"
                record_text = (
                    f"RECORDING: {state}\n"
                    f"episode: {episode_id}/{args.num_episodes}\n"
                    f"episode frames: {episode_frame_count}\n"
                    f"dataset: {dataset_display}\n"
                )
            viewer.set_text(
                CONTROLS,
                (
                    f"task: {task}\n"
                    f"{record_text}"
                    f"target xyz: {env.p0.round(3)}\n"
                    f"eef xyz: {obs['eef_pos'].round(3)}\n"
                    f"eef-target err: {eef_target_err:.3f}\n"
                    f"gripper target/q: {env.q[-1]:.2f}/{obs['joint_pos'][-1]:.2f}\n"
                    f"gripper hold: {env.gripper_contact_hold}\n"
                    f"grasped: {env.grasped_object or '-'}\n"
                    f"sim time: {env.data.time:.2f}s\n"
                    f"success: {bool(obs['success'][0])}"
                ),
                gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            )
            viewer.render()
            target_wall_time = wall_start + (env.data.time - sim_start)
            sleep_time = target_wall_time - time.time()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.02))
    finally:
        if dataset is not None and episode_frame_count > 0:
            dataset.clear_episode_buffer()
            episode_images.clear()
            print(f"Cleared unsaved episode buffer, frames={episode_frame_count}")
        if dataset is not None:
            dataset.finalize()
        viewer.close()


if __name__ == "__main__":
    main()
