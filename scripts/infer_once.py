import argparse
import csv
import json
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from src.env import TBlockToBinEnv
from src.viewer import KeyboardTeleopViewer


CONTROLS = """BC policy live inference
Esc: quit
R: reset episode
P: pause/resume
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one LeRobot behavior cloning policy rollout with realtime MuJoCo display."
    )
    parser.add_argument("--policy-path", default="ckpt/act_joint")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode seed. Omit for a fresh random scene on each run; pass an int to reproduce.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--gripper-mode", choices=["binary", "continuous"], default="binary")
    parser.add_argument(
        "--action-type",
        choices=["delta_eef_pose", "eef_pose", "joint", "delta_joint"],
        default="joint",
    )
    parser.add_argument(
        "--proprio-type",
        choices=["eef_pose", "joint_pos"],
        default="joint_pos",
    )
    parser.add_argument(
        "--env-state-type",
        choices=["eef_pose", "compact", "object_pose"],
        default="eef_pose",
    )
    parser.add_argument("--object-pad", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Runtime denoising steps for Diffusion Policy. Pass 100 to restore the checkpoint's full DDPM cost.",
    )
    parser.add_argument(
        "--default-diffusion-num-inference-steps",
        type=int,
        default=16,
        help="Used when a Diffusion checkpoint leaves num_inference_steps unset.",
    )
    parser.add_argument("--show-camera-previews", action="store_true")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--trace-path", default="outputs/infer_once_trace.csv")
    parser.add_argument("--no-trace", action="store_true")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_lerobot_policy_dir(path):
    path = resolve_path(path)
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_dirs = sorted(
            [candidate for candidate in checkpoints_dir.iterdir() if candidate.is_dir()],
            key=lambda candidate: int(candidate.name) if candidate.name.isdigit() else -1,
            reverse=True,
        )
        candidates.extend(candidate / "pretrained_model" for candidate in checkpoint_dirs)
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a LeRobot pretrained_model folder under {path}. "
        "Expected config.json and model.safetensors."
    )


def load_processors(policy_path, device=None):
    overrides = {}
    if device is not None:
        overrides["device_processor"] = {"device": device}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_input_features(policy_path):
    config_path = policy_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return set(config.get("input_features", {}))


def load_policy_type(policy_path):
    config_path = policy_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("type")


def load_policy(policy_path, device):
    policy_type = load_policy_type(policy_path)
    if not policy_type:
        raise ValueError(f"{policy_path / 'config.json'} does not contain a policy type.")
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(policy_path, local_files_only=True)
    if device is not None:
        policy.config.device = device
        policy.to(device)
    policy.eval()
    return policy, policy_type


def configure_policy_runtime(policy, policy_type, args):
    if policy_type != "diffusion":
        return

    requested = args.num_inference_steps
    if requested is None and getattr(policy.config, "num_inference_steps", None) is None:
        if args.default_diffusion_num_inference_steps <= 0:
            raise ValueError("--default-diffusion-num-inference-steps must be a positive integer.")
        requested = args.default_diffusion_num_inference_steps
    if requested is None:
        return
    if requested <= 0:
        raise ValueError("--num-inference-steps must be a positive integer.")

    policy.config.num_inference_steps = requested
    if hasattr(policy, "diffusion") and hasattr(policy.diffusion, "num_inference_steps"):
        policy.diffusion.num_inference_steps = requested
    print(
        "Diffusion runtime: "
        f"num_inference_steps={requested}, "
        f"n_action_steps={policy.config.n_action_steps}"
    )


def make_viewer(env, args):
    viewer = KeyboardTeleopViewer(env.model, env.data, title="BC policy live inference")
    viewer_cfg = env.cfg.get("viewer", {}).get("keyboard", {})
    viewer.set_camera(
        lookat=viewer_cfg.get("lookat", [0.4, 0.0, 0.6]),
        distance=viewer_cfg.get("distance", 2.0),
        azimuth=viewer_cfg.get("azimuth", 270),
        elevation=viewer_cfg.get("elevation", -40),
    )
    if args.show_camera_previews:
        viewer.set_camera_previews(env.cfg.get("image_cameras", []))
    return viewer


def sorted_object_environment_state(env, object_pad):
    obj_states, obj_q_states = env.get_object_pose(pad=object_pad)
    obj_slots = np.zeros((object_pad, 6), dtype=np.float32)
    out_idx = 0
    for src_idx in np.argsort(obj_states["names"]):
        name = obj_states["names"][int(src_idx)]
        if name.startswith("pad_"):
            continue
        if out_idx >= object_pad:
            break
        obj_slots[out_idx] = obj_states["poses"][int(src_idx)]
        out_idx += 1

    q_slots = np.zeros((object_pad,), dtype=np.float32)
    out_idx = 0
    for src_idx in np.argsort(obj_q_states["names"]):
        name = obj_q_states["names"][int(src_idx)]
        if name.startswith("pad_"):
            continue
        if out_idx >= object_pad:
            break
        q_slots[out_idx] = obj_q_states["poses"][int(src_idx)]
        out_idx += 1
    return np.concatenate([obj_slots.reshape(-1), q_slots], dtype=np.float32)


def environment_state(env, obs, args):
    if args.env_state_type == "eef_pose":
        return obs["eef_pose"].astype(np.float32)
    if args.env_state_type == "compact":
        return np.concatenate(
            [obs["eef_pose"], obs["target_pos"], obs["bin_pos"]],
            dtype=np.float32,
        )
    if args.env_state_type == "object_pose":
        return sorted_object_environment_state(env, args.object_pad)
    raise ValueError(f"Unknown env_state_type={args.env_state_type!r}.")


def image_feature_camera_name(feature_key):
    if feature_key == "observation.image":
        return "agentview"
    if feature_key == "observation.wrist_image":
        return "egocentric"
    if feature_key == "observation.images.agentview":
        return "agentview"
    if feature_key == "observation.images.egocentric":
        return "egocentric"
    if feature_key.startswith("observation.images."):
        return feature_key.rsplit(".", 1)[-1]
    return None


def build_policy_input(env, viewer, args, input_features):
    obs = env.get_observation()
    frame = {}
    for feature_key in input_features:
        camera_name = image_feature_camera_name(feature_key)
        if camera_name is None:
            continue
        image = viewer.capture_fixed_camera_rgb(
            camera_name,
            width=args.image_size,
            height=args.image_size,
        )
        image = np.ascontiguousarray(image)
        frame[feature_key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    if "observation.state" in input_features:
        state = obs["joint_pos"] if args.proprio_type == "joint_pos" else obs["eef_pose"]
        frame["observation.state"] = torch.tensor(state, dtype=torch.float32)
    if "observation.eef_pose" in input_features:
        frame["observation.eef_pose"] = torch.tensor(obs["eef_pose"], dtype=torch.float32)
    if "observation.environment_state" in input_features:
        frame["observation.environment_state"] = torch.tensor(
            environment_state(env, obs, args),
            dtype=torch.float32,
        )
    frame["task"] = args.task or env.cfg.get("task", "")
    return frame


def ensure_visual_batch_dims(batch, policy):
    for feature_key in policy.config.image_features:
        value = batch.get(feature_key)
        if isinstance(value, torch.Tensor) and value.dim() == 3:
            batch[feature_key] = value.unsqueeze(0)
    return batch


def select_action(policy, preprocessor, postprocessor, frame):
    frame = preprocessor(frame)
    frame = ensure_visual_batch_dims(frame, policy)
    with torch.inference_mode():
        action = policy.select_action(frame)
        action = postprocessor(action)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def maybe_reset_episode(env, policy, seed):
    obs = env.reset(seed=seed, leader_pose=False)
    policy.reset()
    return obs


def format_vec(value, precision=3):
    return np.array2string(np.asarray(value), precision=precision, suppress_small=True)


def append_trace_row(rows, step, action, obs, env, infer_s, loop_s):
    target_pos = obs["target_pos"]
    bin_pos = obs["bin_pos"]
    eef_pos = obs["eef_pos"]
    eef_target_delta = eef_pos - target_pos
    rows.append(
        {
            "step": step,
            "time": env.data.time,
            "success": bool(obs["success"][0]),
            "infer_s": infer_s,
            "loop_s": loop_s,
            "eef_x": float(eef_pos[0]),
            "eef_y": float(eef_pos[1]),
            "eef_z": float(eef_pos[2]),
            "target_x": float(target_pos[0]),
            "target_y": float(target_pos[1]),
            "target_z": float(target_pos[2]),
            "eef_target_dx": float(eef_target_delta[0]),
            "eef_target_dy": float(eef_target_delta[1]),
            "eef_target_dz": float(eef_target_delta[2]),
            "eef_target_dist": float(np.linalg.norm(eef_target_delta)),
            "bin_x": float(bin_pos[0]),
            "bin_y": float(bin_pos[1]),
            "bin_z": float(bin_pos[2]),
            "action_dx": float(action[0]),
            "action_dy": float(action[1]),
            "action_dz": float(action[2]),
            "action_droll": float(action[3]),
            "action_dpitch": float(action[4]),
            "action_dyaw": float(action[5]),
            "action_gripper": float(action[6]),
            "grasped": env.grasped_object or "",
        }
    )


def save_trace(rows, trace_path):
    if not rows or trace_path is None:
        return None
    trace_path = resolve_path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return trace_path


def add_debug_markers(viewer, obs):
    viewer.plot_sphere(obs["target_pos"], radius=0.018, rgba=(1.0, 0.0, 1.0, 0.5), label="target")
    viewer.plot_sphere(obs["bin_pos"], radius=0.025, rgba=(0.0, 1.0, 0.0, 0.35), label="bin")
    viewer.plot_sphere(obs["eef_pos"], radius=0.012, rgba=(1.0, 0.1, 0.1, 0.6), label="eef")


def main():
    args = parse_args()
    policy_path = resolve_lerobot_policy_dir(args.policy_path)
    print(f"Loading policy: {policy_path}")
    print(f"Episode seed: {'random' if args.seed is None else args.seed}")

    policy, policy_type = load_policy(policy_path, args.device)
    print(f"Policy type: {policy_type}")
    configure_policy_runtime(policy, policy_type, args)
    preprocessor, postprocessor = load_processors(policy_path, args.device)
    input_features = load_input_features(policy_path)

    env = TBlockToBinEnv(seed=args.seed, action_type=args.action_type)
    viewer = make_viewer(env, args)
    sim_substeps = args.sim_substeps
    if sim_substeps is None:
        sim_substeps = env.cfg.get("teleop", {}).get("sim_substeps_per_frame", 8)

    obs = maybe_reset_episode(env, policy, args.seed)
    paused = False
    trace_rows = []
    next_wall_time = time.time()
    last_action = np.zeros(7, dtype=np.float32)

    try:
        step = 0
        while viewer.is_alive() and step < args.max_steps:
            if viewer.consume_key(glfw.KEY_R):
                obs = maybe_reset_episode(env, policy, args.seed)
                trace_rows.clear()
                step = 0
                paused = False
                next_wall_time = time.time()
            if viewer.consume_key(glfw.KEY_P):
                paused = not paused

            loop_t0 = time.time()
            success = bool(obs["success"][0])
            if success:
                paused = True

            if not paused:
                infer_t0 = time.time()
                frame = build_policy_input(env, viewer, args, input_features)
                last_action = select_action(policy, preprocessor, postprocessor, frame)
                infer_s = time.time() - infer_t0

                env.step(last_action, gripper_mode=args.gripper_mode)
                obs = env.step_env(n_substeps=sim_substeps)
                loop_s = time.time() - loop_t0
                append_trace_row(trace_rows, step, last_action, obs, env, infer_s, loop_s)

                if args.print_freq > 0 and step % args.print_freq == 0:
                    eef_target_dist = np.linalg.norm(obs["eef_pos"] - obs["target_pos"])
                    print(
                        f"step={step:04d} success={bool(obs['success'][0])} "
                        f"eef={format_vec(obs['eef_pos'])} "
                        f"target={format_vec(obs['target_pos'])} "
                        f"eef_target_dist={eef_target_dist:.3f} "
                        f"action={format_vec(last_action)} "
                        f"infer_s={infer_s:.3f} loop_s={loop_s:.3f}"
                    )
                step += 1
            else:
                infer_s = 0.0
                loop_s = time.time() - loop_t0

            add_debug_markers(viewer, obs)
            viewer.set_text(
                CONTROLS,
                (
                    f"policy: {policy_path}\n"
                    f"type: {policy_type}\n"
                    f"step: {step}/{args.max_steps}\n"
                    f"paused: {paused}\n"
                    f"success: {bool(obs['success'][0])}\n"
                    f"eef xyz: {format_vec(obs['eef_pos'])}\n"
                    f"target xyz: {format_vec(obs['target_pos'])}\n"
                    f"eef-target: {format_vec(obs['eef_pos'] - obs['target_pos'])} "
                    f"dist={np.linalg.norm(obs['eef_pos'] - obs['target_pos']):.3f}\n"
                    f"bin xyz: {format_vec(obs['bin_pos'])}\n"
                    f"action: {format_vec(last_action)}\n"
                    f"gripper target/q: {env.q[-1]:.2f}/{obs['joint_pos'][-1]:.2f}\n"
                    f"grasped: {env.grasped_object or '-'}\n"
                    f"infer/loop: {infer_s:.3f}s / {loop_s:.3f}s"
                ),
                gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            )
            viewer.render()

            next_wall_time += 1.0 / args.control_hz
            sleep_s = next_wall_time - time.time()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.02))
            else:
                next_wall_time = time.time()
    finally:
        trace_path = None if args.no_trace else args.trace_path
        saved_path = save_trace(trace_rows, trace_path)
        if saved_path is not None:
            print(f"Saved trace: {saved_path}")
        viewer.close()


if __name__ == "__main__":
    main()
