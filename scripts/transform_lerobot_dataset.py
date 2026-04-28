import argparse
from argparse import Namespace
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


RAW_ALIGNED_MARKERS = ("raw.joint_pos_before", "raw.eef_pose_before")
IMAGE_KEYS = ("observation.image", "observation.wrist_image")
PRESETS = {
    "image_joint": {
        "action_type": "joint",
        "proprio_type": "joint_pos",
        "observation_type": "image",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "image_delta_joint": {
        "action_type": "delta_joint",
        "proprio_type": "joint_pos",
        "observation_type": "image",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "image_eef": {
        "action_type": "eef_pose",
        "proprio_type": "eef_pose",
        "observation_type": "image",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "image_delta_eef": {
        "action_type": "delta_eef_pose",
        "proprio_type": "eef_pose",
        "observation_type": "image",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "object_joint": {
        "action_type": "joint",
        "proprio_type": "joint_pos",
        "observation_type": "object_pose",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "object_delta_joint": {
        "action_type": "delta_joint",
        "proprio_type": "joint_pos",
        "observation_type": "object_pose",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "object_eef": {
        "action_type": "eef_pose",
        "proprio_type": "eef_pose",
        "observation_type": "object_pose",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
    "object_delta_eef": {
        "action_type": "delta_eef_pose",
        "proprio_type": "eef_pose",
        "observation_type": "object_pose",
        "include_env_state": False,
        "env_state_source": "object_pose",
    },
}
MODEL_PRESET_ALIASES = {
    "act": "image_joint",
    "diffusion": "image_joint",
    "smolvla": "image_joint",
}
ALL_PRESETS = tuple(PRESETS)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transform raw myvla teleoperation data into a clean LeRobot policy dataset. "
            "This mirrors the tutorial transform step: keep raw collection broad, then choose "
            "the proprio/action/observation representation for training."
        )
    )
    parser.add_argument("--source-root", default="dataset/teleoperation_dataset")
    parser.add_argument("--source-repo-id", default="t_block_to_bin")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preset",
        choices=["custom", "all", *MODEL_PRESET_ALIASES.keys(), *PRESETS.keys()],
        default="act",
        help=(
            "Named transform recipe. Defaults to act, which writes the dataset used by "
            "scripts/train_act.py. act/diffusion/smolvla currently map to image_joint. "
            "Use all to generate every canonical dataset under --output-root."
        ),
    )
    parser.add_argument(
        "--action-type",
        choices=["auto", "joint", "delta_joint", "eef_pose", "delta_eef_pose"],
        default="auto",
        help="Training action representation. auto uses joint targets when raw joint targets exist, otherwise eef_pose.",
    )
    parser.add_argument(
        "--proprio-type",
        choices=["auto", "joint_pos", "eef_pose"],
        default="auto",
        help="Training observation.state representation.",
    )
    parser.add_argument(
        "--observation-type",
        choices=["image", "object_pose"],
        default="image",
        help="Use camera images or object-pose environment_state as the main observation.",
    )
    parser.add_argument(
        "--include-env-state",
        action="store_true",
        help="Also include observation.environment_state when --observation-type=image.",
    )
    parser.add_argument(
        "--env-state-source",
        choices=["object_pose", "compact"],
        default="object_pose",
        help="object_pose flattens env.obj_pose/env.obj_q_states; compact uses eef/target/bin positions.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--object-pad", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def default_output_root(args):
    if args.output_root is not None:
        return args.output_root
    if args.preset == "custom":
        return "dataset/transformed_act_dataset"
    if args.preset == "all":
        return "dataset/transforms"
    preset_name = MODEL_PRESET_ALIASES.get(args.preset, args.preset)
    return f"dataset/transforms/{preset_name}"


def default_output_repo_id(args, preset_name=None):
    if args.output_repo_id is not None and args.preset != "all":
        return args.output_repo_id
    if args.preset == "custom" and preset_name is None:
        return "t_block_to_bin_transformed"
    preset_name = preset_name or MODEL_PRESET_ALIASES.get(args.preset, args.preset)
    return f"t_block_to_bin_{preset_name}"


def apply_preset(args, preset_name):
    preset_name = MODEL_PRESET_ALIASES.get(preset_name, preset_name)
    if preset_name not in PRESETS:
        return args
    values = vars(args).copy()
    values.update(PRESETS[preset_name])
    values["preset"] = preset_name
    values["output_root"] = str(resolve_path(default_output_root(Namespace(**values))))
    values["output_repo_id"] = default_output_repo_id(Namespace(**values), preset_name)
    return Namespace(**values)


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Image.Image):
        return np.asarray(value)
    return np.asarray(value)


def to_float32(value, shape=None):
    array = to_numpy(value).astype(np.float32)
    if shape is not None:
        array = array.reshape(shape)
    return array


def to_scalar_int(value):
    return int(to_numpy(value).reshape(()))


def to_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def image_to_hwc_uint8(value, image_size):
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}.")
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.max(initial=0.0) <= 1.0 else 1.0
        image = image * scale
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[:2] != (image_size, image_size):
        image = np.asarray(Image.fromarray(image).resize((image_size, image_size)))
    return np.ascontiguousarray(image)


def feature_names(dataset):
    return set(dataset.meta.features.keys())


def is_aligned_raw(dataset):
    names = feature_names(dataset)
    return any(key in names for key in RAW_ALIGNED_MARKERS)


def state_feature_is_joint(dataset):
    ft = dataset.meta.features.get("observation.state", {})
    names = ft.get("names") or []
    return "joint" in " ".join(str(name) for name in names).lower()


def resolve_action_type(requested, dataset):
    if requested != "auto":
        return requested
    if "raw.target_joint_pos" in feature_names(dataset):
        return "joint"
    return "eef_pose"


def resolve_proprio_type(requested, dataset):
    if requested != "auto":
        return requested
    if "raw.joint_pos_before" in feature_names(dataset) or state_feature_is_joint(dataset):
        return "joint_pos"
    return "eef_pose"


def infer_object_pad(dataset, requested):
    if requested is not None:
        return requested
    ft = dataset.meta.features.get("env.obj_pose")
    if ft is not None:
        return int(ft["shape"][0])
    return 10


def get_joint_state(frame, dataset, before=True):
    raw_key = "raw.joint_pos_before" if before else "raw.joint_pos_after"
    if raw_key in frame:
        return to_float32(frame[raw_key], (7,))
    if state_feature_is_joint(dataset) and "observation.state" in frame:
        return to_float32(frame["observation.state"], (7,))
    raise KeyError(
        "Joint proprio/action transform needs raw.joint_pos_before/after or an "
        "observation.state recorded as joint_pos. Re-collect with the updated teleop script."
    )


def get_eef_state(frame, before=True):
    raw_key = "raw.eef_pose_before" if before else "raw.eef_pose_after"
    if raw_key in frame:
        return to_float32(frame[raw_key], (7,))
    if "observation.eef_pose" in frame:
        return to_float32(frame["observation.eef_pose"], (7,))
    if "observation.state" in frame:
        return to_float32(frame["observation.state"], (7,))
    raise KeyError("Could not find an eef pose in this source dataset.")


def get_state(frame, dataset, proprio_type):
    if proprio_type == "joint_pos":
        return get_joint_state(frame, dataset, before=True)
    if proprio_type == "eef_pose":
        return get_eef_state(frame, before=True)
    raise ValueError(f"Unknown proprio_type={proprio_type!r}.")


def get_delta_eef_action(action_frame):
    if "raw.delta_eef_action" in action_frame:
        return to_float32(action_frame["raw.delta_eef_action"], (7,))
    return to_float32(action_frame["action"], (7,))


def get_target_eef_action(action_frame):
    if "raw.target_eef_pose" in action_frame:
        return to_float32(action_frame["raw.target_eef_pose"], (7,))
    if "raw.eef_pose_after" in action_frame:
        target = to_float32(action_frame["raw.eef_pose_after"], (7,))
    else:
        target = get_eef_state(action_frame, before=False)
    if "action" in action_frame:
        target[-1] = float(to_float32(action_frame["action"], (7,))[-1] > 0.5)
    return target.astype(np.float32)


def get_target_joint_action(action_frame, dataset):
    if "raw.target_joint_pos" in action_frame:
        return to_float32(action_frame["raw.target_joint_pos"], (7,))
    return get_joint_state(action_frame, dataset, before=False)


def get_action(obs_frame, action_frame, dataset, action_type):
    if action_type == "delta_eef_pose":
        return get_delta_eef_action(action_frame)
    if action_type == "eef_pose":
        return get_target_eef_action(action_frame)
    if action_type == "joint":
        return get_target_joint_action(action_frame, dataset)
    if action_type == "delta_joint":
        target = get_target_joint_action(action_frame, dataset)
        current = get_joint_state(obs_frame, dataset, before=True)
        return np.concatenate([target[:6] - current[:6], [target[-1]]]).astype(np.float32)
    raise ValueError(f"Unknown action_type={action_type!r}.")


def sorted_object_environment_state(frame, object_pad):
    obj_pose = to_float32(frame["env.obj_pose"]).reshape(-1, 6)
    obj_names = [name for name in to_text(frame["env.obj_names"]).split(",") if name]
    obj_slots = np.zeros((object_pad, 6), dtype=np.float32)
    out_idx = 0
    for src_idx in np.argsort(obj_names):
        name = obj_names[int(src_idx)]
        if name.startswith("pad_"):
            continue
        if out_idx >= object_pad or src_idx >= len(obj_pose):
            break
        obj_slots[out_idx] = obj_pose[int(src_idx)]
        out_idx += 1

    q_slots = np.zeros((object_pad,), dtype=np.float32)
    if "env.obj_q_states" in frame and "env.obj_q_names" in frame:
        q_states = to_float32(frame["env.obj_q_states"]).reshape(-1)
        q_names = [name for name in to_text(frame["env.obj_q_names"]).split(",") if name]
        out_idx = 0
        for src_idx in np.argsort(q_names):
            name = q_names[int(src_idx)]
            if name.startswith("pad_"):
                continue
            if out_idx >= object_pad or src_idx >= len(q_states):
                break
            q_slots[out_idx] = q_states[int(src_idx)]
            out_idx += 1

    return np.concatenate([obj_slots.reshape(-1), q_slots], dtype=np.float32)


def compact_environment_state(frame):
    eef_pose = get_eef_state(frame, before=True)
    if "env.target_pos" in frame:
        target_pos = to_float32(frame["env.target_pos"], (3,))
    else:
        target_pos = np.zeros((3,), dtype=np.float32)
    if "env.bin_pos" in frame:
        bin_pos = to_float32(frame["env.bin_pos"], (3,))
    else:
        bin_pos = np.zeros((3,), dtype=np.float32)
    return np.concatenate([eef_pose, target_pos, bin_pos], dtype=np.float32)


def environment_state(frame, args, object_pad):
    if args.env_state_source == "compact":
        return compact_environment_state(frame)
    return sorted_object_environment_state(frame, object_pad)


def get_episode_bounds(dataset, episode_index):
    episodes = dataset.meta.episodes
    return (
        int(episodes["dataset_from_index"][episode_index]),
        int(episodes["dataset_to_index"][episode_index]),
    )


def task_from_episode(dataset, start_index, fallback):
    if fallback is not None:
        return fallback
    frame = dataset.hf_dataset[start_index]
    task_index = to_scalar_int(frame["task_index"])
    tasks = dataset.meta.tasks
    matches = tasks[tasks["task_index"] == task_index]
    if len(matches) > 0:
        return str(matches.index[0])
    return "pick up the hollow cylinder and place it into the trash bin"


def episode_has_success(dataset, start_index, end_index):
    names = feature_names(dataset)
    if "raw.success" not in names:
        return True
    for idx in range(start_index, end_index):
        if float(to_float32(dataset.hf_dataset[idx]["raw.success"], (1,))[0]) > 0.5:
            return True
    return False


def output_features(args, proprio_type, action_type, object_pad, env_state_dim):
    state_name = "joint_pos" if proprio_type == "joint_pos" else "eef_pose"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [state_name],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [action_type],
        },
    }
    if args.observation_type == "image":
        features["observation.image"] = {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        }
        features["observation.wrist_image"] = {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        }

    if args.observation_type == "object_pose" or args.include_env_state:
        names = ["compact_env"] if args.env_state_source == "compact" else ["object_pose"]
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (env_state_dim,),
            "names": names,
        }

    return features


def create_output_dataset(args, source_dataset, proprio_type, action_type, object_pad, env_state_dim):
    output_root = resolve_path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    return LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=output_root,
        robot_type="omy",
        fps=source_dataset.fps,
        features=output_features(args, proprio_type, action_type, object_pad, env_state_dim),
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def add_transformed_frame(output_dataset, obs_frame, action_frame, dataset, args, task, object_pad, action_type, proprio_type):
    frame = {
        "observation.state": get_state(obs_frame, dataset, proprio_type),
        "action": get_action(obs_frame, action_frame, dataset, action_type),
        "task": task,
    }
    if args.observation_type == "image":
        for key in IMAGE_KEYS:
            frame[key] = image_to_hwc_uint8(obs_frame[key], args.image_size)
    if args.observation_type == "object_pose" or args.include_env_state:
        frame["observation.environment_state"] = environment_state(obs_frame, args, object_pad)
    output_dataset.add_frame(frame)


def save_transform_metadata(args, output_root, action_type, proprio_type, aligned_source):
    metadata = {
        "source_root": str(resolve_path(args.source_root)),
        "source_repo_id": args.source_repo_id,
        "output_repo_id": args.output_repo_id,
        "preset": args.preset,
        "action_type": action_type,
        "proprio_type": proprio_type,
        "observation_type": args.observation_type,
        "include_env_state": args.include_env_state,
        "env_state_source": args.env_state_source,
        "aligned_source": aligned_source,
    }
    path = output_root / "transform_args.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def normalize_custom_args(args):
    values = vars(args).copy()
    values["output_root"] = str(resolve_path(default_output_root(args)))
    values["output_repo_id"] = default_output_repo_id(args)
    return Namespace(**values)


def transform_dataset(args, source_dataset):
    aligned_source = is_aligned_raw(source_dataset)
    action_type = resolve_action_type(args.action_type, source_dataset)
    proprio_type = resolve_proprio_type(args.proprio_type, source_dataset)
    object_pad = infer_object_pad(source_dataset, args.object_pad)
    env_state_dim = 13 if args.env_state_source == "compact" else object_pad * 7
    output_dataset = create_output_dataset(
        args,
        source_dataset,
        proprio_type,
        action_type,
        object_pad,
        env_state_dim,
    )

    print(
        "Transform config: "
        f"aligned_source={aligned_source}, proprio_type={proprio_type}, "
        f"action_type={action_type}, observation_type={args.observation_type}, "
        f"env_state_dim={env_state_dim}"
    )

    episodes = source_dataset.num_episodes
    if args.max_episodes is not None:
        episodes = min(episodes, args.max_episodes)

    saved_episodes = 0
    total_frames = 0
    try:
        for episode_index in range(episodes):
            start, end = get_episode_bounds(source_dataset, episode_index)
            if args.success_only and not episode_has_success(source_dataset, start, end):
                print(f"Skip episode {episode_index}: no success marker.")
                continue

            task = task_from_episode(source_dataset, start, args.task)
            frame_count = 0
            if aligned_source:
                frame_indices = range(start, end)
                for idx in frame_indices:
                    source_frame = source_dataset.hf_dataset[idx]
                    add_transformed_frame(
                        output_dataset,
                        source_frame,
                        source_frame,
                        source_dataset,
                        args,
                        task,
                        object_pad,
                        action_type,
                        proprio_type,
                    )
                    frame_count += 1
            else:
                frame_indices = range(start + 1, end)
                for idx in frame_indices:
                    obs_frame = source_dataset.hf_dataset[idx - 1]
                    action_frame = source_dataset.hf_dataset[idx]
                    add_transformed_frame(
                        output_dataset,
                        obs_frame,
                        action_frame,
                        source_dataset,
                        args,
                        task,
                        object_pad,
                        action_type,
                        proprio_type,
                    )
                    frame_count += 1

            if frame_count > 0:
                output_dataset.save_episode()
                saved_episodes += 1
                total_frames += frame_count
                print(f"Saved episode {episode_index}: frames={frame_count}")
            else:
                output_dataset.clear_episode_buffer()
                print(f"Skip episode {episode_index}: no transformable frames.")
    finally:
        output_dataset.finalize()

    output_root = resolve_path(args.output_root)
    save_transform_metadata(args, output_root, action_type, proprio_type, aligned_source)
    print(
        f"Done. saved_episodes={saved_episodes}, frames={total_frames}, root={output_root}"
    )


def main():
    args = parse_args()
    source_root = resolve_path(args.source_root)
    source_dataset = LeRobotDataset(args.source_repo_id, root=source_root)

    if args.preset == "all":
        base_root = resolve_path(default_output_root(args))
        print(f"Generating presets under: {base_root}")
        for preset_name in ALL_PRESETS:
            values = vars(args).copy()
            values["preset"] = preset_name
            values["output_root"] = str(base_root / preset_name)
            values["output_repo_id"] = default_output_repo_id(
                Namespace(**values),
                preset_name,
            )
            preset_args = apply_preset(Namespace(**values), preset_name)
            print(f"\n=== Transform preset: {preset_name} ===")
            transform_dataset(preset_args, source_dataset)
        return

    if args.preset == "custom":
        args = normalize_custom_args(args)
    else:
        args = apply_preset(args, args.preset)
    transform_dataset(args, source_dataset)


if __name__ == "__main__":
    main()
