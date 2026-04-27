import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def lerobot_eval_executable():
    candidate = Path(sys.executable).resolve().parent / "lerobot-eval"
    if candidate.exists():
        return str(candidate)
    return "lerobot-eval"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ACT with the official lerobot-eval entrypoint."
    )
    parser.add_argument("--policy-path", default="ckpt/act_delta_eef")
    parser.add_argument(
        "--env-type",
        required=True,
        help=(
            "LeRobot/Gym environment type, for example pusht. "
            "The myvla MuJoCo environment must be registered before it can be used here."
        ),
    )
    parser.add_argument("--env-task", default=None)
    parser.add_argument("--output-dir", default="outputs/eval/act_delta_eef")
    parser.add_argument("--job-name", default="act_delta_eef_eval")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def build_command(args, extra_args):
    policy_path = resolve_lerobot_policy_dir(args.policy_path)
    output_dir = resolve_path(args.output_dir)
    command = [
        lerobot_eval_executable(),
        f"--policy.path={policy_path}",
        f"--env.type={args.env_type}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
        f"--seed={args.seed}",
        f"--trust_remote_code={str(args.trust_remote_code).lower()}",
    ]
    if args.env_task:
        command.append(f"--env.task={args.env_task}")
    command.extend(extra_args)
    return command


def main():
    args, extra_args = parse_args()
    command = build_command(args, extra_args)
    print("Running:")
    print(" ".join(command))
    raise SystemExit(subprocess.call(command, cwd=ROOT))


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
