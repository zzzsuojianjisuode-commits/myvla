from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_IMAGE_KEYS = ("observation.image", "observation.wrist_image")


def make_teleoperation_dataset(
    root,
    repo_id="t_block_to_bin",
    fps=20,
    action_dim=7,
    state_dim=7,
    object_pad=10,
    image_size=256,
    image_writer_threads=10,
    image_writer_processes=5,
):
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
        "observation.eef_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["eef_pose"],
        },
        "env.obj_pose": {
            "dtype": "float32",
            "shape": (object_pad, 6),
            "names": ["obj_pose"],
        },
        "env.obj_names": {
            "dtype": "string",
            "shape": (1,),
            "names": ["obj_names"],
        },
        "env.obj_q_names": {
            "dtype": "string",
            "shape": (1,),
            "names": ["obj_q_names"],
        },
        "env.obj_q_states": {
            "dtype": "float32",
            "shape": (object_pad,),
            "names": ["obj_q_states"],
        },
        "env.config_file_name": {
            "dtype": "string",
            "shape": (1,),
            "names": ["file_name"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=Path(root),
        robot_type="omy",
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def build_teleoperation_frame(
    env,
    action,
    task,
    config_file_name,
    image_size=256,
    object_pad=10,
    agent_image=None,
    wrist_image=None,
):
    obs = env.get_observation()
    if agent_image is None or wrist_image is None:
        agent_image, wrist_image = env.grab_image(return_side=False)
    agent_image = _resize_rgb(agent_image, image_size)
    wrist_image = _resize_rgb(wrist_image, image_size)
    obj_states, obj_q_states = env.get_object_pose(pad=object_pad)

    return {
        "observation.image": agent_image,
        "observation.wrist_image": wrist_image,
        "observation.state": obs["eef_pose"].astype(np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "observation.eef_pose": obs["eef_pose"].astype(np.float32),
        "env.obj_pose": np.asarray(obj_states["poses"], dtype=np.float32),
        "env.obj_names": ",".join(obj_states["names"]),
        "env.obj_q_names": ",".join(obj_q_states["names"]),
        "env.obj_q_states": np.asarray(obj_q_states["poses"], dtype=np.float32),
        "env.config_file_name": str(config_file_name),
        "task": task,
    }


def _resize_rgb(rgb, image_size):
    rgb = np.asarray(rgb)
    if rgb.shape[:2] == (image_size, image_size):
        return rgb.copy()
    return np.asarray(Image.fromarray(rgb).resize((image_size, image_size)))


def materialize_episode_images(root, episode_index, image_keys=DEFAULT_IMAGE_KEYS):
    """Write embedded parquet image bytes back to the tutorial-style PNG tree."""
    root = Path(root)
    data_dir = root / "data"
    if not data_dir.exists():
        return 0

    saved = 0
    columns = ["episode_index", "frame_index", *image_keys]
    for parquet_path in sorted(data_dir.glob("**/*.parquet")):
        episode_df = _read_episode_parquet(parquet_path, episode_index, columns)
        if episode_df is None or episode_df.empty:
            continue

        for _, row in episode_df.iterrows():
            frame_index = int(row["frame_index"])
            for image_key in image_keys:
                if image_key not in row:
                    continue
                image_bytes = _image_bytes(row[image_key])
                if image_bytes is None:
                    continue

                image_path = _episode_image_path(root, episode_index, image_key, frame_index)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)
                saved += 1
    return saved


def write_episode_images(root, episode_index, image_frames, image_keys=DEFAULT_IMAGE_KEYS):
    saved = 0
    for frame_index, frame_images in enumerate(image_frames):
        for image_key in image_keys:
            if image_key not in frame_images:
                continue
            image = _as_uint8_rgb(frame_images[image_key])
            image_path = _episode_image_path(root, episode_index, image_key, frame_index)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(image_path, compress_level=6)
            saved += 1
    return saved


def _episode_image_path(root, episode_index, image_key, frame_index):
    return (
        Path(root)
        / "images"
        / image_key
        / f"episode-{episode_index:06d}"
        / f"frame-{frame_index:06d}.png"
    )


def _read_episode_parquet(parquet_path, episode_index, columns):
    import pandas as pd

    try:
        return pd.read_parquet(
            parquet_path,
            columns=columns,
            filters=[("episode_index", "==", episode_index)],
        )
    except (KeyError, TypeError, ValueError):
        df = pd.read_parquet(parquet_path)
        if "episode_index" not in df.columns or "frame_index" not in df.columns:
            return None
        return df[df["episode_index"] == episode_index]


def _image_bytes(value):
    if isinstance(value, dict):
        value = value.get("bytes")
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return None


def _as_uint8_rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image
