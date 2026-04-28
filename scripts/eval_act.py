import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RENAME_MAP = {
    "observation.images.agentview": "observation.image",
    "observation.images.egocentric": "observation.wrist_image",
}


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


def lerobot_eval_executable():
    candidate = Path(sys.executable).resolve().parent / "lerobot-eval"
    if candidate.exists():
        return str(candidate)
    return "lerobot-eval"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ACT with the official lerobot-eval entrypoint."
    )
    parser.add_argument(
        "--backend",
        choices=["fast", "official"],
        default="fast",
        help="fast skips LeRobot video rendering; official uses lerobot-eval and records videos.",
    )
    parser.add_argument("--policy-path", default="ckpt/act_joint")
    parser.add_argument("--env-type", default="myvla")
    parser.add_argument("--env-task", default=None)
    parser.add_argument("--plugin", default="src.lerobot_myvla")
    parser.add_argument("--output-dir", default="outputs/eval/act_joint")
    parser.add_argument("--job-name", default="act_joint_eval")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sim-substeps", type=int, default=8)
    parser.add_argument("--gripper-mode", choices=["binary", "continuous"], default="binary")
    parser.add_argument(
        "--env-action-type",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def load_act_runtime(policy_path, device):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
    if device is not None:
        policy.config.device = device
        policy.to(device)
    policy.eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def load_input_features(policy_path):
    config_path = policy_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return set(config.get("input_features", {}))


def image_to_tensor(rgb):
    rgb = np.ascontiguousarray(rgb)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


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


def build_policy_input(env, args, input_features):
    obs = env.get_observation()
    frame = {}

    need_agent = "observation.image" in input_features
    need_wrist = "observation.wrist_image" in input_features
    if need_agent and need_wrist:
        agent_image, wrist_image = env.grab_image(
            return_side=False,
            width=args.image_size,
            height=args.image_size,
        )
        frame["observation.image"] = image_to_tensor(agent_image)
        frame["observation.wrist_image"] = image_to_tensor(wrist_image)
    elif need_agent:
        frame["observation.image"] = image_to_tensor(
            env.get_camera_rgb("agentview", width=args.image_size, height=args.image_size)
        )
    elif need_wrist:
        frame["observation.wrist_image"] = image_to_tensor(
            env.get_camera_rgb("egocentric", width=args.image_size, height=args.image_size)
        )

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
    return frame


def select_action(policy, preprocessor, postprocessor, frame):
    frame = preprocessor(frame)
    with torch.inference_mode():
        action = policy.select_action(frame)
        action = postprocessor(action)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def run_fast_eval(args):
    from src.env import TBlockToBinEnv

    policy_path = resolve_lerobot_policy_dir(args.policy_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fast eval policy: {policy_path}")
    policy, preprocessor, postprocessor = load_act_runtime(policy_path, args.device)
    input_features = load_input_features(policy_path)
    env = TBlockToBinEnv(seed=args.seed, action_type=args.env_action_type)

    start = time.time()
    episode_metrics = []
    try:
        for episode in range(args.episodes):
            obs = env.reset(seed=args.seed + episode, leader_pose=False)
            policy.reset()
            episode_start = time.time()
            success = bool(obs["success"][0])
            max_reward = float(success)
            sum_reward = float(success)
            steps = 0

            for step in range(args.max_steps):
                if success:
                    break
                frame = build_policy_input(env, args, input_features)
                action = select_action(policy, preprocessor, postprocessor, frame)
                env.step(action, gripper_mode=args.gripper_mode)
                obs = env.step_env(n_substeps=args.sim_substeps)
                success = bool(obs["success"][0])
                reward = 1.0 if success else 0.0
                sum_reward += reward
                max_reward = max(max_reward, reward)
                steps = step + 1

            episode_metrics.append(
                {
                    "episode": episode,
                    "success": success,
                    "steps": steps,
                    "sum_reward": sum_reward,
                    "max_reward": max_reward,
                    "episode_s": time.time() - episode_start,
                }
            )
            print(
                f"episode={episode:03d} success={success} steps={steps} "
                f"episode_s={episode_metrics[-1]['episode_s']:.2f}"
            )
    finally:
        env.close()

    eval_s = time.time() - start
    successes = [item["success"] for item in episode_metrics]
    success_count = int(np.sum(successes))
    success_rate = float(100.0 * success_count / len(successes)) if successes else 0.0
    failed_episodes = [
        item["episode"] for item in episode_metrics if not item["success"]
    ]
    info = {
        "backend": "fast",
        "policy_path": str(policy_path),
        "n_episodes": len(episode_metrics),
        "success_count": success_count,
        "avg_sum_reward": float(np.mean([item["sum_reward"] for item in episode_metrics])),
        "avg_max_reward": float(np.mean([item["max_reward"] for item in episode_metrics])),
        "pc_success": success_rate,
        "eval_s": eval_s,
        "eval_ep_s": eval_s / max(1, len(episode_metrics)),
        "failed_episodes": failed_episodes,
        "episodes": episode_metrics,
    }
    info_path = output_dir / "eval_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(
        f"Fast eval summary: success_rate={success_rate:.1f}% "
        f"({success_count}/{len(episode_metrics)}) "
        f"avg_max_reward={info['avg_max_reward']:.3f} "
        f"avg_sum_reward={info['avg_sum_reward']:.3f} "
        f"eval_s={eval_s:.2f}"
    )
    if failed_episodes:
        print(f"Failed episodes: {failed_episodes}")
    print(f"Saved: {info_path}")


def build_rename_map(policy_path):
    rename_map = dict(DEFAULT_RENAME_MAP)
    config_path = policy_path / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        input_features = config.get("input_features", {})
        if "observation.eef_pose" in input_features:
            rename_map["observation.environment_state"] = "observation.eef_pose"
    return rename_map


def build_command(args, extra_args):
    policy_path = resolve_lerobot_policy_dir(args.policy_path)
    output_dir = resolve_path(args.output_dir)
    rename_map = build_rename_map(policy_path)
    command = [
        lerobot_eval_executable(),
        f"--env.discover_packages_path={args.plugin}",
        f"--policy.path={policy_path}",
        f"--env.type={args.env_type}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--env.episode_length={args.max_steps}",
        f"--env.image_size={args.image_size}",
        f"--env.sim_substeps={args.sim_substeps}",
        f"--env.gripper_mode={args.gripper_mode}",
        f"--env.action_type={args.env_action_type}",
        f"--env.proprio_type={args.proprio_type}",
        f"--env.env_state_type={args.env_state_type}",
        f"--env.object_pad={args.object_pad}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
        f"--seed={args.seed}",
        f"--trust_remote_code={str(args.trust_remote_code).lower()}",
        f"--rename_map={json.dumps(rename_map)}",
    ]
    if args.env_task:
        command.append(f"--env.task={args.env_task}")
    command.extend(extra_args)
    return command


def main():
    args, extra_args = parse_args()
    if args.backend == "fast":
        run_fast_eval(args)
        return

    command = build_command(args, extra_args)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    print("Running:")
    print(" ".join(command))
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()


r'''
The previous hand-written MuJoCo rollout evaluator is intentionally disabled.
It is kept below only as a reference. The active code above uses the official
lerobot-eval entrypoint, which requires a registered LeRobot/Gym environment.

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from src.env import TBlockToBinEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an official lerobot-train ACT checkpoint in TBlockToBinEnv."
    )
    parser.add_argument("--policy-path", default="ckpt/act_delta_eef")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sim-substeps", type=int, default=None)
    parser.add_argument("--gripper-mode", choices=["binary", "continuous"], default="binary")
    parser.add_argument("--device", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
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
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a LeRobot pretrained_model folder under {path}. "
        "Expected config.json and model.safetensors."
    )


def rgb_to_tensor(rgb, image_size):
    rgb = np.asarray(rgb)
    if rgb.shape[:2] != (image_size, image_size):
        rgb = np.asarray(Image.fromarray(rgb).resize((image_size, image_size)))
    rgb = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def build_policy_input(env, image_size):
    obs = env.get_observation()
    agent_image, wrist_image = env.grab_image(
        return_side=False,
        width=image_size,
        height=image_size,
    )
    return {
        "observation.state": torch.tensor(obs["eef_pose"], dtype=torch.float32),
        "observation.image": rgb_to_tensor(agent_image, image_size),
        "observation.wrist_image": rgb_to_tensor(wrist_image, image_size),
    }


def load_processors(policy_path):
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def maybe_make_viewer(env, enabled):
    if not enabled:
        return None
    from src.viewer import KeyboardTeleopViewer

    viewer = KeyboardTeleopViewer(env.model, env.data, title="ACT evaluation")
    viewer_cfg = env.cfg.get("viewer", {}).get("keyboard", {})
    viewer.set_camera(
        lookat=viewer_cfg.get("lookat", [0.4, 0.0, 0.6]),
        distance=viewer_cfg.get("distance", 2.0),
        azimuth=viewer_cfg.get("azimuth", 270),
        elevation=viewer_cfg.get("elevation", -40),
    )
    viewer.set_camera_previews(env.cfg.get("image_cameras", []))
    return viewer


def render_eval_frame(viewer, env, episode, step, success):
    if viewer is None:
        return True
    viewer.set_text(
        "ACT evaluation",
        (
            f"episode: {episode}\n"
            f"step: {step}\n"
            f"eef xyz: {env.get_observation()['eef_pos'].round(3)}\n"
            f"grasped: {env.grasped_object or '-'}\n"
            f"success: {success}"
        ),
    )
    viewer.render()
    return viewer.is_alive()


def run_episode(policy, preprocessor, postprocessor, env, args, episode_index, viewer=None):
    env.reset(seed=args.seed + episode_index, leader_pose=False)
    policy.reset()
    sim_substeps = args.sim_substeps
    if sim_substeps is None:
        sim_substeps = env.cfg.get("teleop", {}).get("sim_substeps_per_frame", 8)

    for step in range(args.max_steps):
        success = env.check_success()
        if success:
            return True, step

        frame = build_policy_input(env, args.image_size)
        frame = preprocessor(frame)
        with torch.no_grad():
            action = policy.select_action(frame)
            action = postprocessor(action)
        action = action.squeeze(0).detach().cpu().numpy()

        env.step(action, gripper_mode=args.gripper_mode)
        env.step_env(n_substeps=sim_substeps)

        success = env.check_success()
        if not render_eval_frame(viewer, env, episode_index, step, success):
            return success, step + 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    return env.check_success(), args.max_steps


def main():
    args = parse_args()
    policy_path = resolve_lerobot_policy_dir(args.policy_path)
    print(f"Loading policy: {policy_path}")

    policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
    if args.device is not None:
        policy.config.device = args.device
        policy.to(args.device)
    policy.eval()

    preprocessor, postprocessor = load_processors(policy_path)
    env = TBlockToBinEnv(seed=args.seed, action_type="delta_eef_pose")
    viewer = maybe_make_viewer(env, args.render)

    successes = []
    try:
        for episode in range(args.episodes):
            success, steps = run_episode(
                policy,
                preprocessor,
                postprocessor,
                env,
                args,
                episode_index=episode,
                viewer=viewer,
            )
            successes.append(bool(success))
            print(
                f"episode={episode + 1:03d}/{args.episodes} "
                f"success={bool(success)} steps={steps}"
            )
    finally:
        if viewer is not None:
            viewer.close()

    rate = float(np.mean(successes)) if successes else 0.0
    print(f"success_rate={rate * 100:.1f}% ({sum(successes)}/{len(successes)})")


if __name__ == "__main__":
    main()
'''
