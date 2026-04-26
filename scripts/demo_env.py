import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import TBlockToBinEnv


def main():
    env = TBlockToBinEnv(seed=0, action_type="delta_eef_pose")
    obs = env.reset(leader_pose=False)
    print("eef_pose:", np.round(obs["eef_pose"], 4))
    print("target_pos:", np.round(obs["target_pos"], 4))
    print("bin_pos:", np.round(obs["bin_pos"], 4))
    for cam_name in env.cfg["image_cameras"]:
        rgb = env.get_camera_rgb(cam_name, width=160, height=120)
        print(cam_name, rgb.shape, int(rgb.mean()), int(rgb.std()))


if __name__ == "__main__":
    main()
