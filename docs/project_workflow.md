# myvla 项目流程与原理说明

本文档说明整个项目从 MuJoCo 环境、遥操作采集、数据转换，到行为克隆策略训练/评估/推理的完整工作流。ACT、Diffusion Policy 和 SmolVLA 都按同一套 raw 数据和 transform 思路扩展。

## 0. 当前命令速查

训练 ACT 默认模型：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py
```

新增 raw 数据后，训练前强制重新 transform：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --force-transform
```

训练 Diffusion Policy / SmolVLA：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --policy-type diffusion
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --policy-type smolvla
```

`--force-transform` 只在新增 raw 数据、修改 transform 逻辑、修改相机/schema 后使用。当前 ACT、Diffusion Policy、SmolVLA 都共用 `dataset/transforms/image_joint`，所以正常切换模型训练时不需要重复 transform。

SmolVLA 默认从本地预训练目录读取：

```text
pretrained/smolvla_base                       SmolVLA policy 权重
pretrained/SmolVLM2-500M-Video-Instruct       VLM backbone/tokenizer
```

注意：只下载 `pretrained/smolvla_base` 还不够。SmolVLA 的 config 内部还会加载 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，如果本地没有第二个目录，训练会继续卡在 VLM 的 `model.safetensors` 下载。`scripts/train.py` 会在缺少本地 VLM 时提前报错并给出下载命令，而不是默默联网卡住。

`scripts/train.py` 还会在生成 `pretrained/smolvla_base_local` 时读取当前 transformed dataset 的 `meta/info.json`，把预训练 SmolVLA 默认的 `observation.images.camera1/2/3` 三相机、6 维 state/action schema 自动改成项目当前的 `observation.image`、`observation.wrist_image` 两相机和 7 维 state/action。

训练默认只保留一个最优 checkpoint：

```text
<output_dir>/checkpoints/best/pretrained_model
<output_dir>/checkpoints/last -> best
```

这个 best 按 checkpoint 保存候选时刻的训练 loss 选择，候选频率由 `--save-freq` 控制。若要恢复 LeRobot 原本保存所有历史 checkpoint 的行为，可加 `--checkpoint-mode all`。

单次实时推理：

```bash
# ACT 默认
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py

# Diffusion / SmolVLA
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py --policy-path ckpt/diffusion_joint
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py --policy-path ckpt/smolvla_joint
```

批量评估成功率：

```bash
# ACT 默认
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py

# Diffusion / SmolVLA
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py --policy-path ckpt/diffusion_joint
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py --policy-path ckpt/smolvla_joint
```

Diffusion checkpoint 如果没有显式保存 `num_inference_steps`，`infer_once.py` 和 `eval.py` 会默认用 16 个去噪步，避免每隔一个 action chunk 卡顿数秒。若想恢复完整 DDPM 成本，可加：

```bash
--num-inference-steps 100
```

## 1. 总体流程

项目推荐流程是：

```text
MuJoCo 环境
  -> 键盘遥操作采集 raw 数据
  -> transform 生成策略专用 LeRobot 数据集
  -> 使用 lerobot-train 训练策略
  -> 使用单次实时推理观察行为
  -> 使用快速 eval 或官方 eval 统计成功率
```

核心原则：

```text
raw 数据尽量完整保存
训练数据按策略单独 transform
训练和推理必须使用一致的 observation/action 表示
```

不要把所有字段都放进同一个训练集，希望 LeRobot 自动选择。LeRobot 会把符合 `observation.*` 的字段当作候选输入特征。我们应该明确生成每个策略要用的数据集。

## 2. MuJoCo 环境

环境实现：

```text
src/env/t_block_to_bin_env.py
```

任务配置：

```text
configs/t_block_to_bin.json
```

任务目标：

```text
pick up the hollow cylinder and place it into the trash bin
```

当前简化场景中只保留两个可移动物体：

```text
t_block
hollow_cylinder
```

如果修改过物体集合，已有数据集和 checkpoint 不再严格匹配当前视觉分布，应该重新采集、transform、训练。

机器人是 OMY 机械臂，动作维度为 7：

```text
[6 个 arm joint 或 6 维 eef pose/delta, gripper]
```

环境支持的 action type：

```text
delta_eef_pose
eef_pose
joint
delta_joint
```

各自含义：

```text
delta_eef_pose  末端位姿增量控制，前三维是 dx/dy/dz，后三维是 rpy 增量，最后一维夹爪
eef_pose        绝对末端位姿控制，前三维 xyz，后三维 rpy，最后一维夹爪
joint           绝对关节目标，前六维 arm joint，最后一维夹爪
delta_joint     关节增量控制，前六维相对当前关节增量，最后一维夹爪
```

键盘遥操作实际使用 `delta_eef_pose`。环境内部会通过 IK 将末端目标转为关节控制。

## 3. 遥操作采集 raw 数据

采集脚本：

```text
scripts/keyboard_teleop.py
```

推荐命令：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/keyboard_teleop.py \
  --dataset-root dataset/teleoperation_dataset \
  --repo-id t_block_to_bin \
  --resume
```

如果想从头重采：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/keyboard_teleop.py \
  --dataset-root dataset/teleoperation_dataset \
  --repo-id t_block_to_bin \
  --overwrite
```

当前 raw 数据位置：

```text
dataset/teleoperation_dataset
repo_id=t_block_to_bin
```

当前 raw 数据规模：

```text
25 episodes
2517 frames
20 fps
```

## 4. raw 数据保存了什么

raw 数据不是直接给某个策略训练用的最终数据，而是“信息尽量完整”的中间数据。当前 raw schema 主要包括：

```text
observation.image              agentview 图像
observation.wrist_image        egocentric/wrist 图像
observation.state              joint_pos
action                         delta_eef_action
observation.eef_pose           动作前 eef pose
env.obj_pose                   对象位姿
env.obj_names                  对象名称
env.target_pos                 目标物位置
env.bin_pos                    垃圾桶位置
raw.joint_pos_before           动作前关节状态
raw.joint_pos_after            动作后关节状态
raw.eef_pose_before            动作前末端位姿
raw.eef_pose_after             动作后末端位姿
raw.delta_eef_action           键盘实际输入的 delta eef action
raw.target_joint_pos           env.step 后控制器目标关节
raw.target_eef_pose            env.step 后控制器目标末端位姿
raw.success                    当前帧后是否成功
task                           语言任务
```

这里最关键的是动作前/动作后分开保存。模仿学习样本应该是：

```text
当前观测 -> 当前应该执行的动作
```

而不是：

```text
动作执行后的观测 -> 刚刚执行过的动作
```

后者会造成一帧错位，训练出的策略容易方向错误或振荡。

## 5. 为什么需要 transform

不同策略需要的输入输出形式不同。

raw 数据中既有：

```text
joint_pos
eef_pose
delta_eef_action
target_joint_pos
target_eef_pose
object_pose
images
```

但训练某个模型时，应该只给它需要的字段。例如 ACT 当前最好使用：

```text
observation.state = joint_pos
action = joint target
observation.image + observation.wrist_image
```

如果把 `observation.eef_pose`、`observation.environment_state`、对象状态等全部塞进同一个训练集，LeRobot 可能会把它们都当作输入。这会让训练和推理字段复杂化，也更容易发生不匹配。

所以 transform 的作用是：

```text
从完整 raw 数据中，生成某个策略专用的干净 LeRobot 数据集
```

## 6. 当前图像输入与相机选择

当前 ACT、Diffusion Policy、SmolVLA 默认使用同一组图像输入：

```text
observation.image        -> MuJoCo camera: agentview
observation.wrist_image  -> MuJoCo camera: egocentric
```

这是当前最合理的 baseline：

```text
agentview   提供桌面、物体、垃圾桶、机械臂的全局关系
egocentric  提供夹爪附近的抓取/投放局部视角
```

当前 raw dataset 只保存了这两路图像。因此 ACT、Diffusion Policy 和 SmolVLA 现在都先共用 `image_joint`，即：

```text
joint_pos + agentview + egocentric -> joint target
```

如果后续要改善视觉输入，推荐做 camera ablation，而不是一次性加很多相机：

```text
A. agentview + egocentric              当前 baseline
B. topview + egocentric                值得优先尝试
C. agentview + topview + egocentric    数据量更多后再试
D. sideview                            可辅助观察，不建议单独主用
```

注意：`topview` 和 `sideview` 可以在 MuJoCo 中渲染，但当前没有写入 raw dataset。要训练这些相机，需要先修改采集 schema、transform 和训练 preset，并重新采集数据。

实现上要注意一点：`observation.wrist_image` 是本项目的自定义图像 key，LeRobot 的默认 batch processor 不会像处理 `observation.image` / `observation.images.*` 那样自动给它补 batch 维。`infer_once.py` 和 `eval.py` 已经在 `policy.select_action` 前统一补齐所有视觉输入的 batch 维，以兼容 Diffusion Policy 的多相机 stack。

## 7. transform 脚本

脚本：

```text
scripts/transform_lerobot_dataset.py
```

ACT 默认 transform：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset act \
  --overwrite
```

`--preset act` 也是脚本默认值，因此下面的命令等价：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --overwrite
```

输出：

```text
dataset/transforms/image_joint
repo_id=t_block_to_bin_image_joint
```

输出 schema：

```text
observation.state        joint_pos
action                   joint target
observation.image        agentview image
observation.wrist_image  egocentric image
```

所有可用 preset：

```text
act                 等价于 image_joint
diffusion           当前也先等价于 image_joint
smolvla             当前也先等价于 image_joint
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

一次生成全部格式：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset all \
  --overwrite
```

## 8. 行为克隆策略训练

训练脚本：

```text
scripts/train.py
```

这个脚本现在是 ACT / Diffusion Policy / SmolVLA 共用的 LeRobot 训练封装。默认会通过 `scripts/lerobot_train_best.py` 调用官方训练流程，并只保留训练 loss 最低的 checkpoint：

```text
<output_dir>/checkpoints/best/pretrained_model
<output_dir>/checkpoints/last -> best
```

如果需要 LeRobot 原本的完整 checkpoint 历史，训练时加 `--checkpoint-mode all`，此时会直接调用 `lerobot-train`。

训练前它会先检查对应 transformed 数据是否存在；如果缺失，会根据 `--policy-type` 自动调用 `scripts/transform_lerobot_dataset.py` 生成数据。当前映射是：

```text
act       -> image_joint
diffusion -> image_joint
smolvla   -> image_joint
```

所以这三种行为克隆方法暂时共用 `dataset/transforms/image_joint`。新增 raw 数据后，如果想强制重新生成训练数据，给训练命令加 `--force-transform`。

默认仍然训练 ACT：

```text
policy_type:  act
dataset root: dataset/transforms/image_joint
repo_id:      t_block_to_bin_image_joint
output_dir:   ckpt/act_joint
steps:        25000
batch_size:   8
chunk_size:   20
n_action_steps: 10
device:       cuda
```

直接训练：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py
```

训练 Diffusion Policy：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py \
  --policy-type diffusion
```

SmolVLA 适合合并到这个训练脚本里，但它不是从零训练的小 BC 模型。默认会从本地目录 `pretrained/smolvla_base` 加载策略预训练权重，从 `pretrained/SmolVLM2-500M-Video-Instruct` 加载 VLM backbone/tokenizer，再生成 `pretrained/smolvla_base_local` 来适配当前数据集的图像 key 和 state/action 维度，然后在本地数据上微调：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py \
  --policy-type smolvla
```

显存不够时可以降低 batch size：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py \
  --batch-size 2
```

默认训练结果：

```text
ACT:       ckpt/act_joint
Diffusion: ckpt/diffusion_joint
SmolVLA:   ckpt/smolvla_joint
```

默认 best-only 模式只会保存：

```text
<output_dir>/checkpoints/best/pretrained_model
<output_dir>/checkpoints/last/pretrained_model
```

`best` 按 checkpoint 保存候选时刻的训练 loss 选择，候选频率由 `--save-freq` 控制。当前默认 `--save-freq 1000`，所以 best 是每 1000 step 和最终 step 这些候选里的最低训练 loss。如果想更细地选 best，可以降低 `--save-freq`，但模型保存会更频繁。

## 9. ACT 的算法直觉

ACT 是 Action Chunking with Transformers。

普通行为克隆往往是：

```text
当前观测 -> 下一步动作
```

ACT 学的是：

```text
当前观测 -> 未来一段动作序列
```

也就是 action chunk。当前设置：

```text
chunk_size = 20
n_action_steps = 10
```

含义是模型预测 20 步动作块，实际每次使用其中 10 步。这样可以减少高频控制下逐帧预测造成的抖动，也能让模型学到短时动作计划。

当前 ACT 输入：

```text
joint_pos + agentview 图像 + wrist 图像
```

当前 ACT 输出：

```text
joint target
```

为什么用 joint target：

1. LeRobot 官方 ACT 常见输入是图像 + robot state，输出 robot action。
2. 对机械臂来说，关节目标比 tiny delta eef action 更稳定，尺度也更适合训练。
3. 之前 delta eef 模型容易学成小幅平均动作，导致朝错误方向缓慢漂移。

## 10. 单次实时推理

脚本：

```text
scripts/infer_once.py
```

这个脚本会从 checkpoint 的 `config.json` 自动识别 `act`、`diffusion` 或 `smolvla`。默认参数：

```text
policy-path: ckpt/act_joint
action-type: joint
proprio-type: joint_pos
seed: omitted, each run samples a fresh random scene
```

如果希望完全复现实验，可以显式传固定 seed：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py --seed 0
```

ACT / Diffusion / SmolVLA 推理在 eval 模式下是确定性的，同一 checkpoint、同一 seed、同一观测会得到同样的动作。

运行：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py
```

查看 Diffusion Policy：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py \
  --policy-path ckpt/diffusion_joint
```

它会打开实时 MuJoCo viewer，显示：

```text
目标位置
eef 位置
action
success
grasped object
推理耗时
```

它还会保存 trace：

```text
outputs/infer_once_trace.csv
```

这个脚本用于人工观察策略行为，比 batch eval 更适合调试“机械臂为什么往错误地方走”。

## 11. 行为克隆策略评估

脚本：

```text
scripts/eval.py
```

默认使用 fast backend：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py
```

评估 Diffusion Policy：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py \
  --policy-path ckpt/diffusion_joint
```

fast backend：

```text
不录视频
直接 rollout
输出 success、steps、reward、耗时、成功率
速度接近单次推理
```

评估结束会打印类似：

```text
Fast eval summary: success_rate=10.0% (2/20) ...
```

并把完整结果保存到：

```text
outputs/eval/act_joint/eval_info.json
```

官方 backend：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py \
  --backend official
```

official backend 会调用：

```text
lerobot-eval
```

它会录制前 10 个 episode 的视频，所以明显更慢。每一步至少需要：

```text
agentview 图像   用于策略输入
egocentric 图像  用于策略输入
agentview render 用于视频
```

因此调试时推荐 fast backend，需要视频时再用 official backend。

## 12. 当前常见问题与排查

### 12.1 Repository Not Found

如果训练时报：

```text
Repository Not Found for url: https://huggingface.co/api/datasets/...
```

通常不是账号问题，而是本地 `dataset.root` 不存在或不是 LeRobot 数据集。先检查：

```text
dataset/transforms/image_joint/meta/info.json
dataset/transforms/image_joint/meta/tasks.parquet
```

正常情况下 `scripts/train.py` 会自动 transform。如果显式用了 `--skip-transform`，或想手动确认 transform，可运行：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset act \
  --overwrite
```

更推荐的训练入口是：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --force-transform
```

### 12.2 推理动作尺度不对

如果 joint 模型推理时 action 很小，例如：

```text
dx/dy/dz ~= 0.003
```

说明可能还在用旧 delta_eef 模型，或 `action-type` 不匹配。joint 模型输出应该是关节角，范围通常接近：

```text
[-0.6, 2.6] 等关节角范围
```

检查三者是否一致：

```text
训练数据 action 类型
checkpoint
推理 env action_type
```

当前 ACT 应该是：

```text
训练数据: dataset/transforms/image_joint
checkpoint: ckpt/act_joint
推理 action_type: joint
推理 proprio_type: joint_pos
```

### 12.3 eval 比 infer 慢

原因：

```text
official lerobot-eval 会录视频
每步额外 env.render()
视频编码也需要时间
```

解决：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py --backend fast
```

## 13. SmolVLA 和 Diffusion Policy

后续不应该重新设计采集逻辑，而是继续复用 raw 数据：

```text
dataset/teleoperation_dataset
```

然后按模型训练。当前数据格式一致，已有 `dataset/transforms/image_joint` 时不要加 `--force-transform`：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --policy-type diffusion
/home/zjx/miniconda3/envs/vla/bin/python scripts/train.py --policy-type smolvla
```

当前 `smolvla` 和 `diffusion` preset 暂时都映射到 `image_joint`，所以 ACT、Diffusion Policy、SmolVLA 可以先共用 `dataset/transforms/image_joint`。后续如果发现官方配置更适合其他字段，可以单独拆分成：

```text
dataset/transforms/smolvla_image_joint
dataset/transforms/diffusion_image_joint
```

Diffusion Policy 是从本地演示数据从零训练的行为克隆策略。SmolVLA 也是用 LeRobot 数据做监督微调，但它依赖预训练 VLM 权重和语言 task 字段。默认会读取 `pretrained/smolvla_base` 和 `pretrained/SmolVLM2-500M-Video-Instruct`，并生成 `pretrained/smolvla_base_local` 作为本地化配置目录。`pretrained/` 已加入 `.gitignore`，不应提交权重文件。

## 14. 推荐当前执行顺序

如果继续推进 ACT：

1. 确认 raw 数据：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset('t_block_to_bin', root='dataset/teleoperation_dataset'); print(ds.num_episodes, ds.num_frames)"
   ```

2. train：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/train.py
   ```

   如果刚新增了 raw 数据，希望训练前重新 transform：

   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/train.py \
     --force-transform
   ```

3. 单次推理观察：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/infer_once.py
   ```

4. 快速评估：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/eval.py
   ```

5. 如果效果不稳定，继续采集到至少 50 个成功 episode，再用 `--force-transform` 重新训练。

## 15. 当前阶段判断

当前 pipeline 已经基本打通：

```text
环境可运行
遥操作可采集
raw 数据 schema 已扩展
transform 可生成 ACT 数据集
ACT 可用 LeRobot 官方训练
单次实时推理可展示
快速 eval 可避免 official eval 录视频慢的问题
```

当前主要短板是数据量：

```text
25 episodes 仍主要适合验证流程
视觉策略稳定训练建议至少 50 个成功 episode
```

后续应优先围绕数据质量、数据量、动作表示一致性、模型输入输出匹配继续迭代。
