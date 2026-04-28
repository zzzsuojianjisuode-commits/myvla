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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ACT with the official lerobot-train entrypoint."
    )
    parser.add_argument("--dataset-root", default="dataset/transforms/image_joint")
    parser.add_argument("--repo-id", default="t_block_to_bin_image_joint")
    parser.add_argument("--output-dir", default="ckpt/act_joint")
    parser.add_argument("--job-name", default="act_joint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=-1)
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-project", default="myvla")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def lerobot_train_executable():
    candidate = Path(sys.executable).resolve().parent / "lerobot-train"
    if candidate.exists():
        return str(candidate)
    return "lerobot-train"


def build_command(args, extra_args):
    output_dir = resolve_path(args.output_dir)
    dataset_root = resolve_path(args.dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not info_path.exists() or not tasks_path.exists():
        raise FileNotFoundError(
            f"{dataset_root} is not a local LeRobot dataset. "
            "Run `python scripts/transform_lerobot_dataset.py --preset act --overwrite` "
            "first, or pass the matching --dataset-root/--repo-id for an existing dataset."
        )
    command = [
        lerobot_train_executable(),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=act",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--wandb.enable={str(args.wandb_enable).lower()}",
        f"--wandb.project={args.wandb_project}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--eval_freq={args.eval_freq}",
    ]
    if args.resume:
        command.append("--resume=true")
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
The previous standalone ACT training implementation is intentionally disabled.
It is kept below only as a reference now that the official lerobot-train path works.

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def patch_lerobot_policy_imports():
    """Avoid importing every LeRobot policy when we only need ACT."""
    if "lerobot.policies" in sys.modules:
        return
    import lerobot

    policies_dir = Path(lerobot.__file__).resolve().parent / "policies"
    module = types.ModuleType("lerobot.policies")
    module.__path__ = [str(policies_dir)]
    module.__package__ = "lerobot"
    sys.modules["lerobot.policies"] = module


patch_lerobot_policy_imports()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT on the local myvla LeRobot dataset.")
    parser.add_argument("--dataset-root", default="dataset/teleoperation_dataset")
    parser.add_argument("--repo-id", default="t_block_to_bin")
    parser.add_argument("--output-dir", default="ckpt/act_delta_eef")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=1)
    parser.add_argument("--no-vae", action="store_true")
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=["observation.image", "observation.wrist_image"],
        help="Image observation keys to use. Use an empty value with --no-images.",
    )
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def make_delta_timestamps(fps, chunk_size):
    return {ACTION: [idx / fps for idx in range(chunk_size)]}


def build_policy_config(dataset, args):
    features = dataset_to_policy_features(dataset.meta.features)
    if OBS_STATE not in features:
        raise KeyError(f"Dataset is missing required key {OBS_STATE!r}.")
    if ACTION not in features:
        raise KeyError(f"Dataset is missing required key {ACTION!r}.")

    input_features = {OBS_STATE: features[OBS_STATE]}
    if not args.no_images:
        for key in args.image_keys:
            if key not in features:
                raise KeyError(f"Dataset is missing image key {key!r}.")
            input_features[key] = features[key]

    return ACTConfig(
        input_features=input_features,
        output_features={ACTION: features[ACTION]},
        device=args.device,
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
        optimizer_weight_decay=args.weight_decay,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_vae=not args.no_vae,
        kl_weight=args.kl_weight,
    )


def save_policy_bundle(output_dir, policy, preprocessor, postprocessor, args, step=None):
    save_dir = output_dir if step is None else output_dir / "checkpoints" / f"step-{step:06d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(save_dir, config_filename="policy_postprocessor.json")
    with (save_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    return save_dir


def cycle_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First load with action deltas so ACT receives [chunk_size, action_dim] targets.
    probe_dataset = LeRobotDataset(args.repo_id, root=dataset_root)
    fps = probe_dataset.fps
    del probe_dataset

    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        delta_timestamps=make_delta_timestamps(fps, args.chunk_size),
    )
    cfg = build_policy_config(dataset, args)
    device = torch.device(cfg.device)

    preprocessor, postprocessor = make_act_pre_post_processors(
        config=cfg,
        dataset_stats=dataset.meta.stats,
    )
    policy = ACTPolicy(cfg).to(device)
    optimizer_cfg = cfg.get_optimizer_preset()
    optimizer_cfg.grad_clip_norm = args.grad_clip_norm
    optimizer = optimizer_cfg.build(policy.get_optim_params())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    batches = cycle_dataloader(dataloader)

    print(f"Dataset: {dataset_root}")
    print(f"Episodes: {dataset.num_episodes}, frames: {dataset.num_frames}, fps: {fps}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}, batch_size: {args.batch_size}, chunk_size: {args.chunk_size}")

    policy.train()
    start = time.time()
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        batch = next(batches)
        batch = preprocessor(batch)

        loss, loss_dict = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            args.grad_clip_norm,
            error_if_nonfinite=False,
        )
        optimizer.step()

        running_loss += float(loss.item())
        if args.log_freq > 0 and step % args.log_freq == 0:
            elapsed = time.time() - start
            avg_loss = running_loss / args.log_freq
            running_loss = 0.0
            details = ""
            if loss_dict:
                details = " " + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            print(
                f"step={step:06d}/{args.steps} "
                f"loss={avg_loss:.5f} grad={float(grad_norm):.3f} "
                f"elapsed={elapsed:.1f}s{details}"
            )

        if args.save_freq > 0 and step % args.save_freq == 0:
            save_dir = save_policy_bundle(output_dir, policy, preprocessor, postprocessor, args, step=step)
            print(f"Saved checkpoint: {save_dir}")

    save_dir = save_policy_bundle(output_dir, policy, preprocessor, postprocessor, args)
    print(f"Saved final policy: {save_dir}")


if __name__ == "__main__":
    main()
'''
