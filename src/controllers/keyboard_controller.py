import glfw
import numpy as np


def _rotation_matrix(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _rot_to_rpy(rot):
    rot = np.asarray(rot, dtype=np.float64)
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


class KeyboardDeltaController:
    """Keyboard controller mirroring the tutorial's delta_eef_pose mapping."""

    def __init__(self, pose_gain=0.01, rot_gain=0.1):
        self.pose_gain = pose_gain
        self.rot_gain = rot_gain
        self.gripper_state = False

    def reset(self, env=None):
        self.gripper_state = False
        self.env = env

    def get_action(self, viewer):
        if viewer.consume_key(glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state

        dpos = np.zeros(3, dtype=np.float64)
        drot = np.eye(3, dtype=np.float64)

        if viewer.is_key_down(glfw.KEY_W):
            dpos[0] -= self.pose_gain
        if viewer.is_key_down(glfw.KEY_S):
            dpos[0] += self.pose_gain
        if viewer.is_key_down(glfw.KEY_A):
            dpos[1] -= self.pose_gain
        if viewer.is_key_down(glfw.KEY_D):
            dpos[1] += self.pose_gain
        if viewer.is_key_down(glfw.KEY_R):
            dpos[2] += self.pose_gain
        if viewer.is_key_down(glfw.KEY_F):
            dpos[2] -= self.pose_gain

        if viewer.is_key_down(glfw.KEY_Q):
            drot = _rotation_matrix(self.rot_gain, [0.0, 0.0, 1.0])
        if viewer.is_key_down(glfw.KEY_E):
            drot = _rotation_matrix(-self.rot_gain, [0.0, 0.0, 1.0])
        if viewer.is_key_down(glfw.KEY_DOWN):
            drot = _rotation_matrix(self.rot_gain, [1.0, 0.0, 0.0])
        if viewer.is_key_down(glfw.KEY_UP):
            drot = _rotation_matrix(-self.rot_gain, [1.0, 0.0, 0.0])
        if viewer.is_key_down(glfw.KEY_LEFT):
            drot = _rotation_matrix(self.rot_gain, [0.0, -1.0, 0.0])
        if viewer.is_key_down(glfw.KEY_RIGHT):
            drot = _rotation_matrix(-self.rot_gain, [0.0, -1.0, 0.0])

        return np.concatenate(
            [dpos, _rot_to_rpy(drot), [float(self.gripper_state)]],
            dtype=np.float64,
        )
