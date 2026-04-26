from .keyboard_controller import KeyboardDeltaController


def load_controller(controller_type, cfg):
    if controller_type != "keyboard":
        raise ValueError(f"Unknown controller type: {controller_type}")

    teleop_cfg = cfg.get("teleop", {})
    return KeyboardDeltaController(
        pose_gain=teleop_cfg.get("position_step", 0.01),
        rot_gain=teleop_cfg.get("rotation_step", 0.1),
    )


__all__ = ["KeyboardDeltaController", "load_controller"]
