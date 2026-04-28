import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRANSFORM_PRESETS = {
    "image_joint",
    "image_delta_joint",
    "image_eef",
    "image_delta_eef",
    "object_joint",
    "object_delta_joint",
    "object_eef",
    "object_delta_eef",
}
POLICY_TRANSFORM_PRESETS = {
    "act": "image_joint",
    "diffusion": "image_joint",
    "smolvla": "image_joint",
}
POLICY_DEFAULTS = {
    "act": {
        "output_dir": "ckpt/act_joint",
        "job_name": "act_joint",
        "steps": 25000,
        "batch_size": 8,
        "chunk_size": 20,
        "n_action_steps": 10,
    },
    "diffusion": {
        "output_dir": "ckpt/diffusion_joint",
        "job_name": "diffusion_joint",
        "steps": 50000,
        "batch_size": 8,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
    },
    "smolvla": {
        "output_dir": "ckpt/smolvla_joint",
        "job_name": "smolvla_joint",
        "steps": 20000,
        "batch_size": 1,
        "chunk_size": 50,
        "n_action_steps": 50,
        "pretrained_policy_path": "pretrained/smolvla_base",
        "vlm_model_name": "pretrained/SmolVLM2-500M-Video-Instruct",
        "hf_vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "localized_policy_path": "pretrained/smolvla_base_local",
    },
}


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LeRobot behavior cloning policies: ACT, Diffusion Policy, or SmolVLA."
    )
    parser.add_argument(
        "--policy-type",
        choices=["act", "diffusion", "smolvla"],
        default="act",
        help="Policy family to train. ACT and diffusion train from scratch by default; SmolVLA fine-tunes a pretrained model.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local transformed dataset root. Defaults to dataset/transforms/<policy transform preset>.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot dataset repo_id. Defaults to t_block_to_bin_<policy transform preset>.",
    )
    parser.add_argument("--source-root", default="dataset/teleoperation_dataset")
    parser.add_argument("--source-repo-id", default="t_block_to_bin")
    parser.add_argument("--transform-script", default="scripts/transform_lerobot_dataset.py")
    parser.add_argument(
        "--transform-preset",
        choices=["act", "diffusion", "smolvla", *sorted(TRANSFORM_PRESETS)],
        default=None,
        help="Transform preset to generate before training. Defaults from --policy-type.",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Only check that --dataset-root already exists; do not auto-generate it.",
    )
    parser.add_argument(
        "--force-transform",
        action="store_true",
        help="Regenerate the transformed dataset before launching training.",
    )
    parser.add_argument("--max-transform-episodes", type=int, default=None)
    parser.add_argument("--transform-image-size", type=int, default=256)
    parser.add_argument("--transform-success-only", action="store_true")
    parser.add_argument("--transform-task", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-obs-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--vision-backbone", default=None)
    parser.add_argument("--pretrained-backbone-weights", default=None)
    parser.add_argument(
        "--pretrained-policy-path",
        default=None,
        help="Fine-tune an existing policy checkpoint/model. For SmolVLA this defaults to pretrained/smolvla_base.",
    )
    parser.add_argument(
        "--smolvla-vlm-model-path",
        default=None,
        help="Local SmolVLA VLM backbone path. Defaults to pretrained/SmolVLM2-500M-Video-Instruct.",
    )
    parser.add_argument(
        "--smolvla-localized-policy-path",
        default=None,
        help="Generated local SmolVLA policy dir with config/tokenizer paths patched to the local VLM.",
    )
    parser.add_argument(
        "--smolvla-allow-vlm-download",
        action="store_true",
        help="Allow SmolVLA to use the HuggingFace VLM id if the local VLM backbone is missing.",
    )
    parser.add_argument(
        "--smolvla-from-scratch",
        action="store_true",
        help="Use --policy.type=smolvla instead of the recommended pretrained SmolVLA fine-tuning path.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-mode",
        choices=["best", "all"],
        default="best",
        help="Use best to keep only ckpt/<policy>/checkpoints/best selected by training loss; use all for LeRobot's normal checkpoint history.",
    )
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=-1)
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-project", default="myvla")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def lerobot_train_command(args):
    if args.checkpoint_mode == "best":
        return [sys.executable, str(resolve_path("scripts/lerobot_train_best.py"))]
    candidate = Path(sys.executable).resolve().parent / "lerobot-train"
    if candidate.exists():
        return [str(candidate)]
    return ["lerobot-train"]


def defaulted(args, name):
    value = getattr(args, name)
    if value is not None:
        return value
    return POLICY_DEFAULTS[args.policy_type].get(name)


def effective_transform_preset(args):
    requested = args.transform_preset or args.policy_type
    return POLICY_TRANSFORM_PRESETS.get(requested, requested)


def default_dataset_root(args):
    if args.dataset_root is not None:
        return resolve_path(args.dataset_root)
    return resolve_path(f"dataset/transforms/{effective_transform_preset(args)}")


def default_repo_id(args):
    if args.repo_id is not None:
        return args.repo_id
    return f"t_block_to_bin_{effective_transform_preset(args)}"


def is_lerobot_dataset(root):
    return (root / "meta" / "info.json").exists() and (root / "meta" / "tasks.parquet").exists()


def build_transform_command(args, dataset_root, repo_id):
    command = [
        sys.executable,
        str(resolve_path(args.transform_script)),
        "--preset",
        args.transform_preset or args.policy_type,
        "--source-root",
        str(resolve_path(args.source_root)),
        "--source-repo-id",
        args.source_repo_id,
        "--output-root",
        str(dataset_root),
        "--output-repo-id",
        repo_id,
        "--image-size",
        str(args.transform_image_size),
        "--overwrite",
    ]
    if args.max_transform_episodes is not None:
        command.extend(["--max-episodes", str(args.max_transform_episodes)])
    if args.transform_success_only:
        command.append("--success-only")
    if args.transform_task is not None:
        command.extend(["--task", args.transform_task])
    return command


def ensure_training_dataset(args):
    dataset_root = default_dataset_root(args)
    repo_id = default_repo_id(args)
    dataset_ready = is_lerobot_dataset(dataset_root)

    if dataset_ready and not args.force_transform:
        print(f"Using existing transformed dataset: {dataset_root}")
        return dataset_root, repo_id

    if args.skip_transform:
        raise FileNotFoundError(
            f"{dataset_root} is not a ready local LeRobot dataset. "
            "Remove --skip-transform or pass an existing --dataset-root/--repo-id."
        )

    source_root = resolve_path(args.source_root)
    if dataset_root.resolve() == source_root.resolve():
        raise ValueError(
            "--dataset-root/output root must be different from --source-root; "
            "auto-transform would overwrite the raw dataset."
        )
    if not is_lerobot_dataset(source_root):
        raise FileNotFoundError(
            f"{source_root} is not a local raw LeRobot dataset. "
            "Collect data with scripts/keyboard_teleop.py first, or pass --source-root/--source-repo-id."
        )

    action = "Regenerating" if args.force_transform else "Generating"
    print(
        f"{action} transformed dataset for policy={args.policy_type}, "
        f"preset={args.transform_preset or args.policy_type} -> {dataset_root}"
    )
    transform_command = build_transform_command(args, dataset_root, repo_id)
    print("Running transform:")
    print(" ".join(transform_command))
    subprocess.check_call(transform_command, cwd=ROOT)
    return dataset_root, repo_id


def resolve_existing_or_literal(path):
    resolved = resolve_path(path)
    if resolved.exists():
        return str(resolved)
    return path


def smolvla_vlm_model_name(args):
    defaults = POLICY_DEFAULTS["smolvla"]
    requested = args.smolvla_vlm_model_path or defaults["vlm_model_name"]
    resolved = resolve_path(requested)
    if resolved.exists():
        return str(resolved)
    if args.smolvla_allow_vlm_download:
        return defaults["hf_vlm_model_name"]

    raise FileNotFoundError(
        "SmolVLA needs both the policy weights and the VLM backbone. "
        f"Found policy path default, but missing local VLM backbone: {resolved}\n"
        "Download it first, for example:\n"
        "  HF_ENDPOINT=https://hf-mirror.com "
        "/home/zjx/miniconda3/envs/vla/bin/huggingface-cli download "
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct "
        "--local-dir pretrained/SmolVLM2-500M-Video-Instruct\n"
        "Or pass --smolvla-allow-vlm-download to let transformers download it online."
    )


def link_or_copy_file(source, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def policy_feature_from_dataset_feature(name, feature):
    shape = list(feature["shape"])
    if feature.get("dtype") == "image":
        names = feature.get("names") or []
        if names == ["height", "width", "channels"] or (len(shape) == 3 and shape[-1] in {1, 3, 4}):
            shape = [shape[2], shape[0], shape[1]]
        return {"type": "VISUAL", "shape": shape}
    if name == "action":
        return {"type": "ACTION", "shape": shape}
    if name == "observation.state":
        return {"type": "STATE", "shape": shape}
    raise ValueError(f"Unsupported SmolVLA dataset feature: {name} ({feature})")


def smolvla_features_from_dataset(dataset_root):
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        features = json.load(f)["features"]

    required = ["observation.state", "action"]
    missing = [key for key in required if key not in features]
    if missing:
        raise KeyError(f"SmolVLA dataset is missing required feature(s): {missing}")

    input_features = {
        "observation.state": policy_feature_from_dataset_feature(
            "observation.state", features["observation.state"]
        )
    }
    for key, feature in features.items():
        if key.startswith("observation.") and feature.get("dtype") == "image":
            input_features[key] = policy_feature_from_dataset_feature(key, feature)

    if not any(feature["type"] == "VISUAL" for feature in input_features.values()):
        raise KeyError(f"SmolVLA needs at least one image feature in {info_path}")

    output_features = {"action": policy_feature_from_dataset_feature("action", features["action"])}
    return input_features, output_features


def localize_smolvla_policy_path(policy_path, vlm_model_name, args, dataset_root):
    source_dir = resolve_path(policy_path)
    if not source_dir.exists():
        return policy_path

    localized = resolve_path(
        args.smolvla_localized_policy_path
        or POLICY_DEFAULTS["smolvla"]["localized_policy_path"]
    )
    localized.mkdir(parents=True, exist_ok=True)

    for source in source_dir.iterdir():
        if source.is_file() or source.is_symlink():
            link_or_copy_file(source, localized / source.name)

    config_path = localized / "config.json"
    with (source_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    input_features, output_features = smolvla_features_from_dataset(dataset_root)
    config["vlm_model_name"] = vlm_model_name
    config["input_features"] = input_features
    config["output_features"] = output_features
    if config_path.exists() or config_path.is_symlink():
        config_path.unlink()
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    preprocessor_path = localized / "policy_preprocessor.json"
    source_preprocessor_path = source_dir / "policy_preprocessor.json"
    if source_preprocessor_path.exists():
        with source_preprocessor_path.open("r", encoding="utf-8") as f:
            preprocessor = json.load(f)
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                step.setdefault("config", {})["tokenizer_name"] = vlm_model_name
            if step.get("registry_name") == "normalizer_processor":
                step.setdefault("config", {})["features"] = {**input_features, **output_features}
        if preprocessor_path.exists() or preprocessor_path.is_symlink():
            preprocessor_path.unlink()
        with preprocessor_path.open("w", encoding="utf-8") as f:
            json.dump(preprocessor, f, indent=2)

    return str(localized)


def pretrained_policy_path(args, dataset_root):
    if args.pretrained_policy_path:
        path = resolve_existing_or_literal(args.pretrained_policy_path)
        if args.policy_type == "smolvla" and not args.smolvla_from_scratch:
            vlm_model_name = smolvla_vlm_model_name(args)
            return localize_smolvla_policy_path(path, vlm_model_name, args, dataset_root)
        return path
    if args.policy_type == "smolvla" and not args.smolvla_from_scratch:
        path = resolve_existing_or_literal(POLICY_DEFAULTS["smolvla"]["pretrained_policy_path"])
        vlm_model_name = smolvla_vlm_model_name(args)
        return localize_smolvla_policy_path(path, vlm_model_name, args, dataset_root)
    return None


def append_policy_hparams(command, args):
    policy_type = args.policy_type
    n_action_steps = defaulted(args, "n_action_steps")
    if n_action_steps is not None:
        command.append(f"--policy.n_action_steps={n_action_steps}")

    if policy_type in {"act", "smolvla"}:
        chunk_size = defaulted(args, "chunk_size")
        if chunk_size is not None:
            command.append(f"--policy.chunk_size={chunk_size}")

    if policy_type == "diffusion":
        horizon = defaulted(args, "horizon")
        n_obs_steps = defaulted(args, "n_obs_steps")
        if horizon is not None:
            command.append(f"--policy.horizon={horizon}")
        if n_obs_steps is not None:
            command.append(f"--policy.n_obs_steps={n_obs_steps}")
        if args.num_inference_steps is not None:
            command.append(f"--policy.num_inference_steps={args.num_inference_steps}")

    if policy_type in {"act", "diffusion"}:
        if args.vision_backbone is not None:
            command.append(f"--policy.vision_backbone={args.vision_backbone}")
        if args.pretrained_backbone_weights is not None:
            command.append(f"--policy.pretrained_backbone_weights={args.pretrained_backbone_weights}")


def build_command(args, extra_args):
    output_dir = resolve_path(defaulted(args, "output_dir"))
    dataset_root, repo_id = ensure_training_dataset(args)
    policy_path = pretrained_policy_path(args, dataset_root)
    command = [
        *lerobot_train_command(args),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={output_dir}",
        f"--job_name={defaulted(args, 'job_name')}",
        f"--policy.device={args.device}",
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--wandb.enable={str(args.wandb_enable).lower()}",
        f"--wandb.project={args.wandb_project}",
        f"--steps={defaulted(args, 'steps')}",
        f"--batch_size={defaulted(args, 'batch_size')}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--eval_freq={args.eval_freq}",
    ]
    if policy_path is not None:
        command.append(f"--policy.path={policy_path}")
    else:
        command.append(f"--policy.type={args.policy_type}")
    append_policy_hparams(command, args)
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
