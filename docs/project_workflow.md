# myvla 项目流程与原理说明

本文档说明整个项目从 MuJoCo 环境、遥操作采集、数据转换，到 ACT 训练/评估/推理的完整工作流。后续加入 SmolVLA 和 Diffusion Policy 时，也按同一套 raw 数据和 transform 思路扩展。

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
15 episodes
1569 frames
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

## 6. transform 脚本

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

## 7. ACT 训练

训练脚本：

```text
scripts/train_act.py
```

它不是手写训练循环，而是封装官方：

```text
lerobot-train --policy.type=act
```

当前默认参数：

```text
dataset root: dataset/transforms/image_joint
repo_id:      t_block_to_bin_image_joint
output_dir:   ckpt/act_joint
steps:        30000
batch_size:   4
chunk_size:   20
n_action_steps: 10
device:       cuda
```

直接训练：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train_act.py
```

显存不够时：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/train_act.py \
  --batch-size 2
```

训练结果：

```text
ckpt/act_joint
```

LeRobot 通常会保存：

```text
ckpt/act_joint/checkpoints/001000/pretrained_model
...
ckpt/act_joint/checkpoints/last/pretrained_model
```

## 8. ACT 的算法直觉

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

## 9. 单次实时推理

脚本：

```text
scripts/infer_act_once.py
```

默认参数：

```text
policy-path: ckpt/act_joint
action-type: joint
proprio-type: joint_pos
seed: omitted, each run samples a fresh random scene
```

如果希望完全复现实验，可以显式传固定 seed：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_act_once.py --seed 0
```

ACT 推理在 eval 模式下是确定性的，同一 checkpoint、同一 seed、同一观测会得到同样的动作。

运行：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/infer_act_once.py
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
outputs/infer_act_once_trace.csv
```

这个脚本用于人工观察策略行为，比 batch eval 更适合调试“机械臂为什么往错误地方走”。

## 10. ACT 评估

脚本：

```text
scripts/eval_act.py
```

默认使用 fast backend：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval_act.py
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
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval_act.py \
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

## 11. 当前常见问题与排查

### 11.1 Repository Not Found

如果训练时报：

```text
Repository Not Found for url: https://huggingface.co/api/datasets/...
```

通常不是账号问题，而是本地 `dataset.root` 不存在或不是 LeRobot 数据集。先检查：

```text
dataset/transforms/image_joint/meta/info.json
dataset/transforms/image_joint/meta/tasks.parquet
```

如果不存在，先运行：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset act \
  --overwrite
```

### 11.2 推理动作尺度不对

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

### 11.3 eval 比 infer 慢

原因：

```text
official lerobot-eval 会录视频
每步额外 env.render()
视频编码也需要时间
```

解决：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/eval_act.py --backend fast
```

## 12. 后续扩展到 SmolVLA 和 Diffusion Policy

后续不应该重新设计采集逻辑，而是继续复用 raw 数据：

```text
dataset/teleoperation_dataset
```

然后按模型生成数据集：

```bash
/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset smolvla \
  --overwrite

/home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
  --preset diffusion \
  --overwrite
```

当前 `smolvla` 和 `diffusion` preset 暂时都映射到 `image_joint`，后续如果发现官方配置更适合其他字段，可以单独拆分成：

```text
dataset/transforms/smolvla_image_joint
dataset/transforms/diffusion_image_joint
```

SmolVLA 还会更依赖语言 task 字段；Diffusion Policy 通常也可以使用多相机图像 + robot state + action sequence。

## 13. 推荐当前执行顺序

如果继续推进 ACT：

1. 确认 raw 数据：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset('t_block_to_bin', root='dataset/teleoperation_dataset'); print(ds.num_episodes, ds.num_frames)"
   ```

2. transform：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/transform_lerobot_dataset.py \
     --preset act \
     --overwrite
   ```

3. train：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/train_act.py
   ```

4. 单次推理观察：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/infer_act_once.py
   ```

5. 快速评估：
   ```bash
   /home/zjx/miniconda3/envs/vla/bin/python scripts/eval_act.py
   ```

6. 如果效果不稳定，继续采集到至少 50 个成功 episode，再重复 transform 和 train。

## 14. 当前阶段判断

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
15 episodes 只适合验证流程
视觉策略稳定训练建议至少 50 个成功 episode
```

后续应优先围绕数据质量、数据量、动作表示一致性、模型输入输出匹配继续迭代。
