# Codex 项目上下文

## 0. 文档作用与维护方式

本文档是 myvla 项目的跨会话交接文档，作用是让下一轮对话里的 Codex 能快速接上当前项目目标、用户要求、代码状态、数据状态、模型状态和后续工作方向。每次重要开发或排障结束后，都应该根据本节说明更新本文档，然后把更新后的版本留给下下一轮对话使用。

本文档应持续包含以下内容：

1. 项目总体目标：用户正在搭建什么机器人学习/VLA项目，当前任务是什么。
2. 参考项目作用：`/home/zjx/Lerobot-MujoCo-VLA-Tutorial` 在本项目中提供哪些设计依据。
3. 用户长期要求：优先参考 tutorial、优先用 LeRobot 官方训练/推理、显存限制、先 ACT 再 SmolVLA/Diffusion Policy 等。
4. 当前代码结构：关键脚本、环境、数据集工具、LeRobot/Gym 插件分别负责什么。
5. 当前数据状态：raw 数据在哪里、repo_id 是什么、episode/frame 数量、schema 里有哪些关键字段。
6. 当前 transform 状态：有哪些 preset，ACT 当前使用哪份 transformed 数据，输入输出是什么。
7. 当前模型状态：checkpoint 在哪里，最新可用模型是哪一个，训练数据和 action/proprio 类型是什么。
8. 已完成的重要修改：本轮或历史上已经完成的关键代码改动和原因。
9. 当前关键结论：已经验证或排除的问题，例如 eval 为什么慢、推理差是否因为 action_type 不匹配等。
10. 下一步建议：下一轮最应该先做什么，应该运行哪些命令，哪些风险要优先检查。
11. 注意事项：不要误删的数据、常见报错含义、训练/推理必须匹配的字段等。

每次更新时建议按下面顺序检查：

1. 如果新增或改动了脚本，更新“当前代码结构”和“已完成的重要修改”。
2. 如果重新采集或 transform 了数据，更新 episode/frame 数量、路径、repo_id 和 schema。
3. 如果训练了新模型，更新 checkpoint 路径、训练数据、训练步数、默认推理参数。
4. 如果定位了新的问题或结论，更新“当前关键结论”和“注意事项”。
5. 如果下一轮工作重点变化，更新“下一步建议”。

本文档用于下一轮对话快速理解当前项目目标、代码状态、用户要求和后续工作思路。请优先阅读本文，再阅读 `docs/project_workflow.md`。

## 1. 项目目标

当前项目是 `/home/zjx/myvla`，目标是从 0 搭建一个基于 MuJoCo + LeRobot + VLA/模仿学习策略的机器人学习项目。用户希望按照下面的学习路线推进：

1. 搭建 MuJoCo 仿真环境。
2. 用键盘遥操作采集演示数据。
3. 将采集数据转换成适合不同策略的 LeRobot 数据集。
4. 先训练和评估 ACT。
5. 后续继续尝试 SmolVLA 和 Diffusion Policy。

当前任务是桌面上的 OMY 机械臂执行：

```text
pick up the hollow cylinder and place it into the trash bin
```

也就是抓取 `hollow_cylinder` 并放入垃圾桶。环境配置在 `configs/t_block_to_bin.json`，MuJoCo XML 入口是 `task_t_block_to_bin.xml`。

当前环境已经简化为只保留两个可移动物体：

```text
t_block
hollow_cylinder
```

原先的 `cylinder_small`、`cylinder_tall`、`cylinder_wide` 已从 XML 和配置里的 `objects` / `object_z` 删除。旧的 15 条 raw 数据和 `ckpt/act_joint` 仍来自五物体 clutter 场景，后续应按简化场景重新采集、transform、训练。

## 2. 参考项目的作用

参考教程项目是：

```text
/home/zjx/Lerobot-MujoCo-VLA-Tutorial
```

它不是直接复制粘贴的目标，而是本项目的重要范例。它主要提供了这些参考价值：

1. 数据采集范式：先保存包含关节、末端、对象状态、图像和动作的原始 LeRobot 数据集。
2. transform 思路：不要把所有策略需要的数据混在一个训练集里，而是从 raw 数据转换出策略专用的数据集。
3. 训练/评估流程：LeRobot 官方模型优先走 `lerobot-train` / `lerobot-eval`，自定义策略可参考 tutorial 的 `train_custom.py`、`transform.py`、eval notebook。
4. 动作空间选择：raw 数据可以记录 delta eef action，但训练 ACT/Diffusion/SmolVLA 时更推荐生成 joint target 或 eef target 等清晰动作表示。

用户明确要求：不会或不确定的地方要优先参考 tutorial 代码，而不是凭空设计。

## 3. 用户对项目协作方式的要求

用户希望我作为“机器人学习 / VLA / LeRobot / MuJoCo 项目导师”长期协助：

1. 先理解 tutorial，再在 myvla 中实现。
2. 讲清楚为什么这么做，尤其是数据格式、transform、策略输入输出、训练评估流程。
3. 尽量使用 LeRobot 官方训练和推理能力，少手写模型训练逻辑。
4. 模型部署受显存限制，tutorial 里的大 VLA 模型显存不够，优先小模型：ACT、SmolVLA、Diffusion Policy 或自定义轻量模型。
5. 代码要能在 IDE 里直接运行；默认参数要尽量指向当前正确路径，避免隐式使用旧模型或旧数据。

## 4. 当前代码结构

核心文件：

```text
configs/t_block_to_bin.json               任务、机器人、相机、遥操作参数
task_t_block_to_bin.xml                   MuJoCo 场景入口
src/env/t_block_to_bin_env.py             MuJoCo 环境和控制逻辑
src/controllers/keyboard_controller.py    键盘 delta_eef_pose 控制器
src/viewer/keyboard_viewer.py             GLFW 可视化和固定相机截图
src/dataset/utils.py                      LeRobot 数据集 schema 和帧构造
src/lerobot_myvla/__init__.py             LeRobot/Gym eval 插件
scripts/keyboard_teleop.py                键盘遥操作采集 raw 数据
scripts/transform_lerobot_dataset.py      raw -> 多种策略专用数据集
scripts/train_act.py                      LeRobot 官方 ACT 训练封装
scripts/eval_act.py                       ACT 快速评估/官方评估封装
scripts/infer_act_once.py                 ACT 单次实时推理展示
```

## 5. 当前数据状态

raw 遥操作数据：

```text
root: dataset/teleoperation_dataset
repo_id: t_block_to_bin
episodes: 15
frames: 1569
fps: 20
```

raw 数据现在是新 schema，包含：

```text
observation.image
observation.wrist_image
observation.state              joint_pos
action                         delta_eef_action
observation.eef_pose
env.obj_pose / env.obj_names
env.target_pos / env.bin_pos
raw.joint_pos_before
raw.joint_pos_after
raw.eef_pose_before
raw.eef_pose_after
raw.delta_eef_action
raw.target_joint_pos
raw.target_eef_pose
raw.success
task
```

ACT 默认训练数据：

```text
root: dataset/transforms/image_joint
repo_id: t_block_to_bin_image_joint
episodes: 15
frames: 1569
fps: 20
```

ACT 数据格式：

```text
observation.state        joint_pos, shape=(7,)
action                   joint target, shape=(7,)
observation.image        agentview RGB, shape=(256, 256, 3)
observation.wrist_image  egocentric RGB, shape=(256, 256, 3)
```

这份数据由下面命令生成：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset act \
  --overwrite
```

`scripts/transform_lerobot_dataset.py` 现在默认 `--preset act`，所以不传 `--preset` 时也会输出到 `dataset/transforms/image_joint`，与 `scripts/train_act.py` 默认训练路径一致。

注意：LeRobot 不会自动“聪明地选择 joint 字段”。我们通过 transform 生成策略专用数据集，再让训练脚本指向正确数据集。

## 6. 当前模型状态

当前 ACT 模型输出目录：

```text
ckpt/act_joint
```

已经存在数字 checkpoint：

```text
ckpt/act_joint/checkpoints/001000 ... 025000
```

同时有：

```text
ckpt/act_joint/checkpoints/last/pretrained_model
```

`infer_act_once.py` 和 `eval_act.py` 已经支持自动解析：

1. 直接传 `ckpt/act_joint`
2. 传 `ckpt/act_joint/checkpoints/last/pretrained_model`
3. 没有 `last` 时自动选择最新数字 checkpoint 的 `pretrained_model`

## 7. 已完成的重要修改

### 7.1 数据采集改造

`scripts/keyboard_teleop.py` 已改成记录动作前观测和动作后状态，避免旧数据里的“执行动作后才保存 observation”的错位问题。

新采集数据会保存更丰富 raw 字段，方便以后生成 ACT、SmolVLA、Diffusion Policy 或自定义策略数据集。

### 7.2 transform 脚本

新增 `scripts/transform_lerobot_dataset.py`。

支持 preset：

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

其中 `act` 默认输出：

```text
dataset/transforms/image_joint
repo_id=t_block_to_bin_image_joint
```

transform 脚本默认 preset 已统一为 `act`，避免无参数 transform 写到旧的 `dataset/transformed_act_dataset` 后训练脚本找不到数据。

### 7.3 ACT 训练脚本

`scripts/train_act.py` 当前默认：

```text
--dataset-root dataset/transforms/image_joint
--repo-id t_block_to_bin_image_joint
--output-dir ckpt/act_joint
--job-name act_joint
--steps 30000
--batch-size 4
```

它调用官方：

```text
lerobot-train --policy.type=act
```

并且增加了本地数据集存在性检查。如果 transform 没跑，会直接提示先运行 transform，避免 LeRobot 因找不到本地数据而去 HuggingFace Hub 查 repo，导致误导性的 `Repository Not Found`。

### 7.4 ACT 单次实时推理

`scripts/infer_act_once.py` 默认：

```text
--policy-path ckpt/act_joint
--action-type joint
--proprio-type joint_pos
--seed omitted -> random scene each run
```

如果需要复现完全相同的场景和动作，可以显式传 `--seed 0` 等固定整数。ACT 在 eval 模式下是确定性策略，同一 checkpoint + 同一场景 + 同一观测会输出相同动作。

它使用 GLFW viewer 实时显示推理效果，适合人工确认模型行为。

### 7.5 ACT 评估脚本

`scripts/eval_act.py` 现在有两个 backend：

```text
--backend fast      默认，不录视频，速度接近单次推理
--backend official  调用官方 lerobot-eval，会录视频，速度更慢
```

fast backend 会在终端明确打印 `success_rate=...% (成功数/总数)`，同时保存到 `outputs/eval/act_joint/eval_info.json` 的 `pc_success` / `success_count` 字段。

官方 eval 慢的原因已经定位：

1. LeRobot eval 固定 `max_episodes_rendered=10`，会为前 10 个 episode 录视频。
2. 每个 step 已经要渲染 agentview 和 egocentric 作为策略输入。
3. official eval 又额外调用 `env.render()` 渲染 agentview 作为视频帧。
4. 之前 `TBlockToBinEnv.get_camera_rgb()` 每次新建 `mujoco.Renderer`，现已改为按分辨率缓存 renderer。

## 8. 当前关键结论

1. raw 数据必须尽量多保存，不要只保存某一个模型需要的字段。
2. transform 后应该生成多个策略专用数据集，不要把所有字段都塞进同一个 LeRobot 训练集。
3. ACT 当前应使用 `image_joint` 数据：
   ```text
   joint_pos + 双相机图像 -> joint target
   ```
4. 之前推理差，主要曾经是因为使用了旧的 `act_delta_eef` 模型或 action_type 不匹配，不应把问题直接归因于任务太复杂。
5. 15 个 episode 只能做 pipeline sanity check。要让视觉策略稳定，建议继续采集到至少 50 个成功 episode。

## 9. 下一步建议

优先顺序：

1. 确认 `ckpt/act_joint` 的单次推理效果：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/infer_act_once.py
   ```
2. 快速评估：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/eval_act.py
   ```
3. 如果效果仍不稳定，继续采集更多成功 episode：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/keyboard_teleop.py \
     --dataset-root dataset/teleoperation_dataset \
     --repo-id t_block_to_bin \
     --resume
   ```
4. 每次新增 raw 数据后重新 transform：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
     --preset act \
     --overwrite
   ```
5. 重新训练 ACT：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/train_act.py
   ```
6. ACT 跑通后，再补 SmolVLA 和 Diffusion Policy 的训练/评估脚本，优先复用 transform 输出的数据集。

## 10. 注意事项

1. 不要随意删除 `.gitignore`、现有 checkpoint、dataset。
2. `dataset/`、`ckpt/`、`outputs/` 通常被 `.gitignore` 忽略。
3. 如果训练报 HuggingFace `Repository Not Found`，优先检查本地 `--dataset.root` 是否存在 `meta/info.json` 和 `meta/tasks.parquet`。
4. 如果推理效果异常，优先检查三者是否匹配：
   ```text
   训练数据 action 类型
   checkpoint
   推理 env action_type
   ```
5. official eval 慢是正常的；调试时默认用 `--backend fast`。
6. 单次推理如果不传 `--seed`，每次运行会随机重置场景；如果传固定 seed，则场景和确定性策略动作都会复现。
7. 当前 `ckpt/act_joint` 在 20 个 eval seed 上约为 2/20 成功。诊断显示主要失败模式是未稳定抓住 hollow cylinder、少数抓错物体或抓住后未投放；这更像 15 条视觉演示数据量不足导致的泛化问题，而不是 eval 成功率计算错误。
