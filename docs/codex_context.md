# Codex 项目上下文

本文档是 `/home/zjx/myvla` 的跨会话交接摘要。下一轮对话应先读本文，再读 `docs/project_workflow.md`。

维护原则：本文档的目标是让下一轮对话更快理解项目，而不是复盘全部调试过程。更新时应保留当前核心状态、关键决策、稳定结论、风险和下一步；重复的调试流水、已被取代的尝试、临时命令输出和细枝末节应省略或删去，只留下对继续推进项目有帮助的语义。

## 1. 项目目标

项目目标是搭建一个 MuJoCo + LeRobot + 模仿学习/VLA 的机器人学习流程。当前任务：

```text
pick up the hollow cylinder and place it into the trash bin
```

也就是 OMY 机械臂抓取 `hollow_cylinder` 并放入垃圾桶。任务配置：

```text
configs/t_block_to_bin.json
task_t_block_to_bin.xml
```

当前场景已简化为两个可移动物体：

```text
t_block
hollow_cylinder
```

如果 checkpoint 或数据来自旧的五物体 clutter 场景，它与当前简化环境的视觉分布不严格匹配，应重新采集、transform、训练。

## 2. 用户长期要求

1. 优先参考 `/home/zjx/Lerobot-MujoCo-VLA-Tutorial`，不确定时先看 tutorial。
2. 优先使用 LeRobot 官方训练、评估和推理接口，少手写模型训练逻辑。
3. 显存有限，优先小模型：ACT、Diffusion Policy、SmolVLA 或轻量自定义模型。
4. 代码入口要能在 IDE/终端直接运行，默认路径应指向当前正确数据和 checkpoint。
5. 解释时重点讲清数据格式、transform、策略输入输出、训练/评估流程。

## 3. 当前代码结构

```text
configs/t_block_to_bin.json               任务、机器人、相机、遥操作参数
task_t_block_to_bin.xml                   MuJoCo 场景入口
src/env/t_block_to_bin_env.py             MuJoCo 环境和控制逻辑
src/controllers/keyboard_controller.py    键盘 delta_eef_pose 控制器
src/viewer/keyboard_viewer.py             GLFW 可视化和固定相机截图
src/dataset/utils.py                      LeRobot 数据集 schema 和帧构造
src/lerobot_myvla/__init__.py             LeRobot/Gym eval 插件
scripts/keyboard_teleop.py                键盘遥操作采集 raw 数据
scripts/transform_lerobot_dataset.py      raw -> 策略专用 LeRobot 数据集
scripts/train.py                          ACT/Diffusion/SmolVLA 训练入口
scripts/infer_once.py                     单次实时推理入口
scripts/eval.py                           fast/official 评估入口
```

旧文件名 `train_act.py`、`infer_act_once.py`、`eval_act.py` 已重命名为上面的三个统一入口。

## 4. 当前数据状态

raw 遥操作数据：

```text
root: dataset/teleoperation_dataset
repo_id: t_block_to_bin
episodes: 25
frames: 2517
fps: 20
```

raw 数据是信息尽量完整的中间数据，主要字段：

```text
observation.image              agentview 图像
observation.wrist_image        egocentric/wrist 图像
observation.state              joint_pos
action                         delta_eef_action
observation.eef_pose
env.obj_pose / env.obj_names
env.target_pos / env.bin_pos
raw.joint_pos_before / raw.joint_pos_after
raw.eef_pose_before / raw.eef_pose_after
raw.delta_eef_action
raw.target_joint_pos
raw.target_eef_pose
raw.success
task
```

当前默认训练数据：

```text
root: dataset/transforms/image_joint
repo_id: t_block_to_bin_image_joint
episodes: 25
frames: 2517
fps: 20
```

`image_joint` schema：

```text
observation.state        joint_pos, shape=(7,)
action                   joint target, shape=(7,)
observation.image        agentview RGB, shape=(256, 256, 3)
observation.wrist_image  egocentric RGB, shape=(256, 256, 3)
task
```

## 5. Transform 状态

`scripts/transform_lerobot_dataset.py` 支持这些 preset：

```text
act -> image_joint
diffusion -> image_joint
smolvla -> image_joint
image_joint
image_delta_joint
image_eef
image_delta_eef
object_joint
object_delta_joint
object_eef
object_delta_eef
all
```

当前 ACT、Diffusion Policy、SmolVLA 都先共用 `image_joint`。这不是永久限制，而是当前最稳妥的 baseline。后续如果某个模型需要不同输入，只需新增/调整 transform preset，并修改 `scripts/train.py` 中的 policy->preset 映射。

训练脚本现在会自动检查数据；如果目标 transformed dataset 不存在，会按 `--policy-type` 自动调用 transform。`--force-transform` 会无条件重建 transformed dataset，只应在新增 raw 数据、修改 transform 逻辑或修改相机/schema 后使用：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --force-transform
```

## 6. 训练、推理、评估入口

训练 ACT：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py
```

训练 Diffusion Policy：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --policy-type diffusion
```

训练 SmolVLA：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --policy-type smolvla
```

SmolVLA 默认从本地加载预训练模型再微调，不建议从零训练。它需要两个本地目录：

```text
pretrained/smolvla_base
pretrained/SmolVLM2-500M-Video-Instruct
```

只下载 `pretrained/smolvla_base` 不够，因为 SmolVLA 内部还会加载 VLM backbone/tokenizer。`scripts/train.py` 会生成 `pretrained/smolvla_base_local`，把 config 和 tokenizer 路径本地化后传给 `lerobot-train`。生成本地化配置时会读取当前 transformed dataset 的 `meta/info.json`，把预训练 SmolVLA 默认的三相机/6 维 schema 改成项目当前的两路图像 `observation.image`、`observation.wrist_image` 和 7 维 state/action。ACT 和 Diffusion Policy 默认从本地演示数据从零训练。`pretrained/` 已在 `.gitignore` 中忽略，不应提交权重文件。

训练默认只保留最优 checkpoint：`scripts/train.py` 会通过 `scripts/lerobot_train_best.py` 调用 LeRobot 训练流程，把 checkpoint 固定保存到 `<output_dir>/checkpoints/best/pretrained_model`，并让 `last -> best`。best 按 checkpoint 保存候选时刻的训练 loss 选择，候选频率由 `--save-freq` 控制。需要恢复 LeRobot 原始的完整历史保存时，加 `--checkpoint-mode all`。

单次实时推理：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py
```

快速评估：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py
```

`eval.py` 默认 `--backend fast`，会直接打印成功率并保存 `eval_info.json`。`--backend official` 会调用 `lerobot-eval` 并录视频，明显更慢。

Diffusion Policy 的当前 checkpoint 没有显式保存 `num_inference_steps`，LeRobot 默认会用 100 个 DDPM 去噪步，导致每隔一个 action chunk 明显卡顿。`infer_once.py` 和 `eval.py` 已默认在这种情况下使用 16 个去噪步；需要恢复完整成本时传 `--num-inference-steps 100`。

## 7. 当前 checkpoint 状态

当前保留的 checkpoint：

```text
ckpt/act_joint/checkpoints/last/pretrained_model
ckpt/diffusion_joint/checkpoints/last/pretrained_model
ckpt/smolvla_joint/checkpoints/last/pretrained_model
```

`infer_once.py` 和 `eval.py` 会自动解析：

1. `ckpt/act_joint`
2. `ckpt/act_joint/checkpoints/last/pretrained_model`
3. `ckpt/act_joint/checkpoints/best/pretrained_model`
4. 最新数字 checkpoint 下的 `pretrained_model`

注意：如果刚改过环境、相机、数据 schema 或重新采集 raw 数据，不要默认信任旧 checkpoint；应重新 `--force-transform` 并训练。

## 8. 当前图像输入结论

目前 ACT、Diffusion Policy、SmolVLA 默认吃同一组图像：

```text
observation.image        -> MuJoCo camera: agentview
observation.wrist_image  -> MuJoCo camera: egocentric
```

这是合理的第一版 baseline：

1. `agentview` 提供桌面、物体、垃圾桶和机械臂的全局关系。
2. `egocentric` 提供夹爪附近的局部抓取/投放视角。
3. 当前数据量只有 25 episodes，先不要盲目增加更多视觉输入，避免复杂度和过拟合一起上升。

更好的下一步视觉实验是做 camera ablation：

```text
A. agentview + egocentric              当前 baseline
B. topview + egocentric                值得优先尝试
C. agentview + topview + egocentric    数据量更多后再试
D. sideview                            可辅助观察，不建议单独主用
```

重要限制：当前 raw 数据只保存了 `agentview` 和 `egocentric`。`topview` 和 `sideview` 虽可在 MuJoCo 中渲染，但没有写入当前 dataset。若要训练这些相机，需要先改采集 schema/transform，并重新采集数据。

实现细节：LeRobot 的 `to_batch_processor` 会自动给 `observation.image` 和 `observation.images.*` 加 batch 维，但不会自动处理自定义 key `observation.wrist_image`。`scripts/infer_once.py` 和 `scripts/eval.py` 已在调用 `policy.select_action` 前统一补齐所有视觉输入的 batch 维，避免 Diffusion Policy 多相机 `torch.stack` 时出现 `[1,3,H,W]` 和 `[3,H,W]` 混用。

## 9. 关键结论

1. raw 数据必须尽量完整保存；训练数据通过 transform 变成“策略专用干净数据集”。
2. 不要把所有 `observation.*` 都塞进训练集让 LeRobot 自动选择，容易造成训练/推理特征不匹配。
3. 当前主线使用 `image_joint`：
   ```text
   joint_pos + agentview + egocentric -> joint target
   ```
4. 推理动作异常时，优先检查：
   ```text
   训练数据 action 类型
   checkpoint
   推理/eval action_type 和 proprio_type
   ```
5. 当前 `ckpt/act_joint` 在 20 个 eval seed 上曾约为 2/20 成功。主要失败模式是未稳定抓住 hollow cylinder、抓错物体或抓住后未投放，更像数据量/数据质量不足，而不是成功率计算错误。
6. 25 episodes 仍主要适合验证 pipeline。视觉策略稳定训练建议至少 50 个成功 episode，最好更多。

## 10. 下一步建议

推荐顺序：

1. 继续采集高质量成功演示，优先把成功 episode 提到 50+：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/keyboard_teleop.py \
     --dataset-root dataset/teleoperation_dataset \
     --repo-id t_block_to_bin \
     --resume
   ```

2. 新增 raw 数据或修改 schema 后，训练前强制重新 transform：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --force-transform
   ```

   如果只是切换 ACT/Diffusion/SmolVLA 训练，当前三者共用 `image_joint`，不要加 `--force-transform`。

3. 单次推理观察行为：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py
   ```

4. 批量评估成功率：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py
   ```

5. ACT baseline 稳定后，再比较 Diffusion Policy；SmolVLA 需要本地预训练权重、依赖和显存，默认是微调路线。

## 11. 注意事项

1. 不要随意删除 `.gitignore`、`dataset/`、`ckpt/`、`outputs/`。
2. `dataset/`、`ckpt/`、`outputs/` 通常被 `.gitignore` 忽略。
3. 如果训练报 HuggingFace `Repository Not Found`，优先检查本地 `dataset.root` 是否存在 `meta/info.json` 和 `meta/tasks.parquet`。
4. 单次推理不传 `--seed` 会随机重置场景；传固定 seed 可复现实验。
5. official eval 慢是正常的；调试成功率优先用默认 fast backend。
