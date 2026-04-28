import json
import logging
import math
import shutil
from pathlib import Path

from lerobot.scripts import lerobot_train


_ORIGINAL_UPDATE_POLICY = lerobot_train.update_policy
_ORIGINAL_SAVE_CHECKPOINT = lerobot_train.save_checkpoint

_last_loss = None
_best_loss = None
_best_step = None


def update_policy_and_track_loss(*args, **kwargs):
    global _last_loss
    train_metrics, output_dict = _ORIGINAL_UPDATE_POLICY(*args, **kwargs)
    try:
        _last_loss = float(train_metrics.loss)
    except (TypeError, ValueError):
        _last_loss = None
    return train_metrics, output_dict


def best_checkpoint_dir(output_dir, total_steps, step):
    return Path(output_dir) / "checkpoints" / "best"


def should_save_best(step):
    global _best_loss, _best_step
    if _best_step is None:
        return True
    if _last_loss is None or not math.isfinite(_last_loss):
        return False
    if _best_loss is None or not math.isfinite(_best_loss):
        return True
    return _last_loss < _best_loss


def write_best_metadata(checkpoint_dir, step):
    metadata = {
        "selection_metric": "train_loss",
        "best_step": step,
        "best_loss": _best_loss,
        "note": "Best among checkpoint save candidates, controlled by --save-freq.",
    }
    with (checkpoint_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def replace_dir(source, target):
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    source.rename(target)


def save_best_checkpoint(checkpoint_dir, step, *args, **kwargs):
    global _best_loss, _best_step
    checkpoint_dir = Path(checkpoint_dir)
    if not should_save_best(step):
        logging.info(
            "Skip checkpoint at step %s; train_loss=%s, best_loss=%s at step %s",
            step,
            _last_loss,
            _best_loss,
            _best_step,
        )
        return

    tmp_dir = checkpoint_dir.parent / f".best_tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    _ORIGINAL_SAVE_CHECKPOINT(tmp_dir, step, *args, **kwargs)
    replace_dir(tmp_dir, checkpoint_dir)

    _best_loss = _last_loss
    _best_step = step
    write_best_metadata(checkpoint_dir, step)
    logging.info("Saved new best checkpoint at step %s with train_loss=%s", step, _best_loss)


def main():
    lerobot_train.update_policy = update_policy_and_track_loss
    lerobot_train.get_step_checkpoint_dir = best_checkpoint_dir
    lerobot_train.save_checkpoint = save_best_checkpoint
    lerobot_train.main()


if __name__ == "__main__":
    main()
