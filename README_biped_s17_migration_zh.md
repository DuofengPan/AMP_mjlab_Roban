# biped_s17 迁移指南（G1 → AMP_mjlab）

本文档记录将 AMP_mjlab 从 Unitree G1 迁移到 **biped_s17（Kuavo S17）** 的完整流程，便于复现与后续维护。

主项目说明见 [README_zh.md](README_zh.md)。G1 训练流程仍按原 README 操作；本文仅描述 **新机器人** 相关差异与步骤。

---

## 1. 迁移目标

在 AMP_mjlab 中实现与 G1 相同的能力栈：

- 单一 policy 同时学习 **locomotion（走/跑）** 与 **recovery（跌倒恢复）**
- AMP 判别器约束动作风格
- 训练 → ONNX 导出 → [wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) 部署

当前进度：


| 模块                                        | 状态          | 路径/说明                                         |
| ----------------------------------------- | ----------- | --------------------------------------------- |
| 机器人 MJCF + mesh                           | ✅ 已就绪       | `src/assets/robots/biped_s17/`                |
| AMP 格式 motion NPZ（loco）                   | ✅ 已就绪（16 条） | `src/assets/motions/s17/amp/loco/`            |
| AMP 格式 motion NPZ（FlatWalk / amp_gait）    | ✅ 已就绪（30 条） | `src/assets/motions/s17/amp/FlatWalk/`        |
| mimic → AMP 转换脚本                          | ✅ 已就绪       | `scripts/mimic_npz_to_amp_npz.py`             |
| retarget gait → AMP 转换脚本                  | ✅ 已就绪       | `scripts/retarget_npz_to_amp_npz.py`          |
| motion 校验 / 可视化工具                         | ✅ 已就绪       | `scripts/utils/validate_amp_motions.py` 等     |
| MJCF 传感器 / foot site                      | ✅ 已就绪       | `biped_s17.xml`（§5.2）                         |
| `s17_constants.py` + `kuavo_actuators.py` | ✅ 已就绪       | §5.1、§5.5（**21 DOF action，头部 fixed**）         |
| AMP 任务注册与配置                               | ✅ 已就绪       | `src/tasks/amp_loco/config/biped_s17/`        |
| Flat / FlatWalk 训练验证                      | 🔄 进行中      | `Biped-S17-AMP-Flat`、`Biped-S17-AMP-FlatWalk` |
| wbc_fsm 部署映射                              | ⏳ 待完成       | obs/action 维度与关节顺序                            |


---

## 2. G1 与 biped_s17 关键差异


| 项目                    | G1                         | biped_s17                                     |
| --------------------- | -------------------------- | --------------------------------------------- |
| 可控 DOF（policy action） | 29                         | **21**（头 2 DOF 在 MJCF 中 **fixed**，不参与控制）      |
| 仿真 hinge DOF          | 29                         | **21**（`get_spec()` 删除 `zhead_*` hinge）       |
| NPZ `joint_pos` 列数    | 29                         | **23**（legacy，末 2 列为 head=0；加载时自动 strip 为 21） |
| 腰                     | yaw / roll / pitch         | **仅 waist_yaw**                               |
| 臂                     | 7×2（含腕）                    | **4×2**                                       |
| Root link             | `pelvis`                   | `base_link`（free joint）                       |
| AMP anchor            | `torso_link`               | `torso`                                       |
| IMU site              | pelvis，`imu_ang_vel`       | torso 上 `imu`（已对齐 mjlab 命名）                   |
| Foot site             | `left_foot` / `right_foot` | `leg_l6/r6_link` 上已添加                         |
| 执行器模型                 | `UnitreeActuatorCfg`       | `KuavoActuatorCfg`_*（kuavo.json）              |
| 任务 ID（示例）             | `Unitree-G1-AMP-Flat`      | `Biped-S17-AMP-Flat` / `FlatWalk` / `Rough`   |
| NPZ bodies            | 30                         | **24**                                        |


**不能直接复用 G1 的 NPZ、policy 或任务配置**，必须按 s17 维度和命名重新配置。

---

## 3. 整体流程概览

涉及两个仓库：

```text
leju_soma_retarget                          AMP_mjlab
────────────────────                        ────────────────────────────────
LAFAN1 SOMA BVH (长片段)
    │ cut_bvh_clips.py
    ▼
短 BVH clips (8–15 s)
    │ bvh_to_npz_converter (retarget → biped_s17)
    ▼
mimic NPZ  (keys: data, fps)                  scripts/mimic_npz_to_amp_npz.py
    │ ───────────────────────────────────────►
    ▼                                          AMP NPZ (joint/body + fps)
                                               src/assets/motions/s17/amp/
                                                 loco/ 或 FlatWalk/
                                                    │
                                                    ▼
                                               s17_constants.py + kuavo_actuators.py
                                               config/biped_s17/（env + rl）
                                                    │
                                                    ▼
                                               train.py（Flat / Rough / FlatWalk）
                                                    │
                                                    ▼
                                               ONNX → wbc_fsm 部署
```

### 3.1 两种 NPZ 格式（重要）


| 来源                                              | Keys                                      | 能否直接给 AMP 训练      |
| ----------------------------------------------- | ----------------------------------------- | ----------------- |
| SOMA `bvh_to_npz_converter.py`                  | `data`, `fps`                             | **否**（mimic 中间格式） |
| AMP `csv_to_npz.py` / `mimic_npz_to_amp_npz.py` | `fps`, `joint_pos`, `joint_vel`, `body_`* | **是**             |


SOMA 的 `data` 布局：`(T, nq)` = `[root_pos(3), root_quat(xyzw)(4), joint_pos(ndof)]`。

AMP 训练加载器见 `src/tasks/amp_loco/ampmotion_loader.py` 与 `rsl_rl/utils/motion_loader.py`，要求完整 body FK 字段。

---

## 4. 运动数据流水线（已完成部分）

### 4.1 切割 LAFAN BVH（leju_soma_retarget）

完整 LAFAN SOMA BVH 单条约 100–260 s，需先切成 8–15 s 再 retarget。

```bash
cd /path/to/leju_soma_retarget

# 预览
python app/utilities/cut_bvh_clips.py --dry-run

# 导出 clips
python app/utilities/cut_bvh_clips.py
```

- 配置：`app/utilities/cut_bvh_clips.py` 顶部 `CLIP_JOBS`
- 说明：`app/utilities/README_cut_bvh_clips.md`
- 默认输出：`assets/motions/lafan1/lafan1_soma_clips/{WalkandRun,Recovery}/`

帧编号为 **1-based 闭区间**（与 `csv_to_npz.py --line-range` 一致）。

### 4.2 BVH → mimic NPZ（SOMA retarget）

```bash
cd /path/to/leju_soma_retarget

# 使用 conda env: soma（需 warp + newton + CUDA）
bash run/retargeting/retarget_lafan_soma_clips_to_npz.sh
```

- 配置：`assets/lafan1_soma_clips_bvh_config.json`
- retarget 配置：`soma_retargeter/configs/biped_s17/extra_cfg/lafan/`
- 默认输出：`outputs/biped_s17/lafan1_soma_clips_mimic_npz/`

环境变量（可选）：

```bash
RETARGET_BATCH_SIZE=16          # OOM 时改小
RETARGET_OVERWRITE_EXISTING=1     # 覆盖已有 npz
```

### 4.3 mimic NPZ → AMP NPZ（AMP_mjlab）

```bash
cd /path/to/AMP_mjlab

/home/lzl/miniforge3/envs/soma/bin/python scripts/mimic_npz_to_amp_npz.py \
  --input-dir  /path/to/leju_soma_retarget/outputs/biped_s17/lafan1_soma_clips_mimic_npz \
  --output-dir src/assets/motions/s17/amp/loco \
  --input-fps 30 \
  --output-fps 50 \
  --mjcf src/assets/robots/biped_s17/xml/biped_s17.xml
```

输出目录结构（**loco 与 FlatWalk 分离，避免判别器混用**）：

```text
src/assets/motions/s17/
  paths.py                    # 各任务 motion 根路径常量
  amp/
    loco/                     # Rough + Flat 任务共用
      WalkandRun/             # 12 条（LAFAN SOMA）
      Recovery/               # 4 条
    FlatWalk/                 # FlatWalk 任务专用（amp_gait retarget，30 条）
```

路径常量见 `src/assets/motions/s17/paths.py`：


| 常量                      | 用途                                           |
| ----------------------- | -------------------------------------------- |
| `LOCO_MOTION_ROOT`      | Rough / Flat：env reset + AMP 判别器（递归 `loco/`） |
| `LOCO_WALK_AND_RUN_DIR` | Rough / Flat：reset 采样 WalkandRun             |
| `LOCO_RECOVERY_DIR`     | Rough / Flat：recovery 子集                     |
| `FLATWALK_MOTION_ROOT`  | FlatWalk：env reset + AMP 判别器（仅 `FlatWalk/`）  |


### 4.3.1 amp_gait retarget → AMP NPZ（FlatWalk 专用）

若 motion 来自 Roban gait retarget（`*_retarget.npz`，含 `dof_names` / `dof_positions` 等），使用：

```bash
cd /path/to/AMP_mjlab
conda activate amp   # 或 soma

python scripts/retarget_npz_to_amp_npz.py \
  --input-dir  src/assets/motions/s17/amp_gait-original \
  --output-dir src/assets/motions/s17/amp/FlatWalk \
  --mjcf src/assets/robots/biped_s17/xml/biped_s17.xml
```

转换时 head 关节写入 0；输出 schema 与 §4.3 相同（`joint_pos` 仍为 **23 列**，末 2 列为 head）。

批量校验：

```bash
python scripts/utils/validate_amp_motions.py \
  --robot s17 \
  --motion-root src/assets/motions/s17/amp/FlatWalk \
  --fail-on-issues

python scripts/utils/validate_amp_motions.py \
  --robot s17 \
  --motion-root src/assets/motions/s17/amp/loco \
  --fail-on-issues
```

单条 motion 可视化（MuJoCo mesh 回放）：

```bash
python scripts/utils/visualize_amp_npz.py \
  --robot s17 \
  --npz src/assets/motions/s17/amp/FlatWalk/直行_低速_小摆手_Skeleton_segment_000_f0-1271_retarget.npz \
  --mujoco-robot --loop --root-frame center
```

当前 AMP NPZ 规格：

- `fps`: 50
- `joint_pos` / `joint_vel`: `(T, 23)`
- `body_pos_w` / `body_quat_w` / `body_lin_vel_w` / `body_ang_vel_w`: `(T, 24, *)`
- `body_quat_w`: **wxyz**

### 4.4 新增/扩充 motion 的检查清单

1. 确认目标子目录：`amp/loco/`（Flat/Rough）或 `amp/FlatWalk/`（FlatWalk）
2. LAFAN 链路：在 `cut_bvh_clips.py` 增加 `CLIP_JOBS` → §4.2 → `mimic_npz_to_amp_npz.py --output-dir .../amp/loco`
3. amp_gait 链路：`retarget_npz_to_amp_npz.py --output-dir .../amp/FlatWalk`
4. 运行 `validate_amp_motions.py --fail-on-issues`
5. 目视：`visualize_amp_npz.py --mujoco-robot` 或训练 smoke test

---

## 5. AMP_mjlab 机器人接入（已完成）

motion 与 XML 就绪后，已完成机器人 Entity、任务配置与注册。整体策略：**AMP 算法框架与任务 MDP 不改**，仅在 G1 的「机器人适配层」上替换为 S17。

```text
共享（未改）                         S17 替换/新增
──────────────────────────────      ──────────────────────────────
make_amp_env_cfg()                   s17_constants.py + kuavo_actuators.py
amp_env_cfg: reward/obs/event        biped_s17.xml（传感器、foot site）
AMPPPO + Discriminator               config/biped_s17/env_cfgs.py
AMPOnPolicyRunner / AMPLoader        config/biped_s17/rl_cfg.py
Delayed Termination / Recovery       motions/s17/amp/*.npz
```

### 5.1 机器人 constants（`s17_constants.py`）

路径：`src/assets/robots/biped_s17/s17_constants.py`


| 项                  | 实现                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------- |
| `get_spec()`       | 加载 `biped_s17.xml` + mesh；删除 XML 原生 `<motor>`；**删除 head hinge（fixed 刚性连接）**；自动命名碰撞 geom |
| 初始姿态               | `KNEES_BENT_KEYFRAME`，z=0.95（参考 G1 弯膝，关节名映射到 s17）                                       |
| 执行器                | `KuavoActuatorCfg_`*（见 §5.5），8 组共 **21 DOF**（不含 head）                                   |
| Policy action      | **21 维**（`S17_NUM_ACTIONS`）；actor/critic 的 joint obs 亦为 21 维                            |
| 头部                 | MJCF 中 `zhead_1/2_link` 保留为 **fixed** 子连杆；legacy NPZ 末 2 列 head 在加载时 strip              |
| PD 增益              | 与 G1 相同公式：`stiffness = armature × (10 Hz × 2π)²`，`armature=0.003`                       |
| `S17_ACTION_SCALE` | `0.25 × effort / stiffness`，写入 `joint_pos` action                                       |
| 导出                 | `src/assets/robots/__init__.py` → `get_biped_s17_robot_cfg` / `S17_ACTION_SCALE`        |


### 5.2 MJCF 修改（`biped_s17.xml`）

`amp_env_cfg.py` 硬编码读取以下传感器（挂载在 torso 的 `imu` site）：

```python
"robot/imu_ang_vel"
"robot/imu_lin_vel"
```

已在 `<sensor>` 中增加 mjlab 命名，并保留原 Kuavo 传感器名（`BodyGyro` 等）以兼容其他工具链：

```xml
<gyro name="imu_ang_vel" site="imu"/>
<velocimeter name="imu_lin_vel" site="imu"/>
```

Foot 相关 reward / 摩擦随机化需要 foot site，已在 `leg_l6_link` / `leg_r6_link` 添加：

```xml
<site name="left_foot" pos="0.04 0 -0.035" .../>
<site name="right_foot" pos="0.04 0 -0.035" .../>
```

### 5.3 AMP 任务配置（`config/biped_s17/`）


| 文件            | 作用                                                                |
| ------------- | ----------------------------------------------------------------- |
| `env_cfgs.py` | 机器人 entity、body/foot/contact 映射、**分任务 motion 路径**、21-DOF action   |
| `rl_cfg.py`   | AMP 判别器 body 映射、`min_normalized_std`（**21 维**）、`amp_motion_files` |
| `__init__.py` | 注册 `Biped-S17-AMP-Flat` / `Rough` / `**FlatWalk`**                |


**已注册任务一览：**


| 任务 ID                    | 地形       | Motion 来源                          | `experiment_name`    | 说明                                     |
| ------------------------ | -------- | ---------------------------------- | -------------------- | -------------------------------------- |
| `Biped-S17-AMP-Rough`    | 随机 rough | `amp/loco/`（WalkandRun + Recovery） | `s17_amp_locomotion` | 与 G1 Rough 对应                          |
| `Biped-S17-AMP-Flat`     | 平面       | 同上                                 | `s17_amp_locomotion` | LAFAN loco + recovery                  |
| `Biped-S17-AMP-FlatWalk` | 平面       | `**amp/FlatWalk/` 仅**              | `s17_amp_flatwalk`   | amp_gait 平地步态；无 recovery、无 delay reset |


**FlatWalk 相对 Flat 的主要差异**（其余 reward / AMP 网络 / 21-DOF 设定相同）：

- Motion：30 条 amp_gait retarget，**不加载** `loco/Recovery`
- AMP 判别器：`amp_motion_files` 指向 `FlatWalk/`，不与 loco 混训
- 速度指令范围更窄（训练 `lin_vel_x∈[-0.6,1.2]` 等；见 `env_cfgs.py`）
- `delay_reset_env_ratio=0`（Flat 在 play 时仍可能启用 motion delay）

**AMP body 映射（11 个，与 G1 思路一致）：**

```text
base_link
leg_l2_link, leg_l4_link, leg_l6_link
leg_r2_link, leg_r4_link, leg_r6_link
zarm_l2_link, zarm_l4_link
zarm_r2_link, zarm_r4_link
anchor: torso
root:   base_link
```

### 5.4 环境侧适配明细（`env_cfgs.py` vs G1）

在 `make_amp_env_cfg()` 基础上覆写，reward 权重、push 扰动、delayed reset 等与 G1 **相同**。


| 用途                     | G1                          | S17                                          |
| ---------------------- | --------------------------- | -------------------------------------------- |
| 机器人 Entity             | `get_g1_robot_cfg()`        | `get_biped_s17_robot_cfg()`                  |
| 根 body（角速度惩罚等）         | `pelvis`                    | `base_link`                                  |
| Anchor（速度跟踪、AMP 相对坐标）  | `torso_link`                | `torso`                                      |
| 地形扫描 frame             | `pelvis`                    | `base_link`                                  |
| 足地接触 subtree           | `*_ankle_roll_link`         | `leg_l6_link` / `leg_r6_link`                |
| 自碰撞 subtree            | `pelvis`                    | `base_link`                                  |
| 足摩擦 geom               | `left_foot1~7_collision`    | `left_foot1~9_collision`                     |
| COM 随机化 body           | `torso_link`                | `torso`                                      |
| WalkandRun motion      | `motions/g1/amp/WalkandRun` | `motions/s17/amp/loco/WalkandRun`            |
| Recovery motion        | `motions/g1/amp/Recovery`   | `motions/s17/amp/loco/Recovery`              |
| FlatWalk motion        | —                           | `motions/s17/amp/FlatWalk/`（仅 FlatWalk 任务）   |
| AMP 判别器 motion 根       | `motions/g1/amp`            | loco 任务：`amp/loco/`；FlatWalk：`amp/FlatWalk/` |
| Action 维 / joint obs 维 | 29                          | **21**                                       |


Critic / AMP 观测组的 `body_pos_b`、`body_ori_b`、`body_lin_vel_b`、`body_ang_vel_b` 均使用上表 anchor + 11 body 列表。

### 5.5 执行器模型（`kuavo_actuators.py`）

路径：`src/assets/robots/biped_s17/kuavo_actuators.py`

参考 G1 的 `unitree_actuators.py`，实现 T-N 曲线 + 摩擦模型的 `KuavoActuator`，参数来自：

- `kuavo-ros-control` → `kuavo_v17/kuavo.json`（`MOTORS_TYPE`、`joint_peak_torque_limits`、`joint_peak_velocity_limits`）
- `biped_s17.xml`（`actuatorfrcrange` / motor `ctrlrange` 作为 `effort_limit`）


| 电机型号                 | 关节                                          |
| -------------------- | ------------------------------------------- |
| `PA81_25`            | leg_l1/l2/l4, leg_r1/r2/r4                  |
| `PA76_25`            | leg_l3/r3                                   |
| `PA76_25_WAIST`      | waist_yaw                                   |
| `PA4315_36`          | leg_l5/l6, leg_r5/r6                        |
| `ruiwoPA60_16`       | zarm_l1/r1                                  |
| `ruiwoPA4315_36`     | zarm_l2~~l4, zarm_r2~~r4                    |
| ~~`ruiwoPA4310_25`~~ | ~~zhead_1/2~~（head 已 fixed，无 head actuator） |


**注意：** 当前 `X1`（满扭拐点速度）未标定，默认为 `1e9`，T-N 降扭实际上未生效；等价于恒定扭矩上限 `min(Y1, effort_limit)`。后续可从电机手册补 `X1` 或采用 `X1 ≈ 0.7×X2` 启发式。

### 5.6 算法侧适配明细（`rl_cfg.py` vs G1）

PPO / AMP 网络结构与超参与 G1 **完全一致**（512-256-128、lr=1e-3、`amp_reward_coef=0.1` 等），仅改机器人相关字段：


| 参数                   | G1                  | S17（Flat / Rough）      | S17（FlatWalk）              |
| -------------------- | ------------------- | ---------------------- | -------------------------- |
| `experiment_name`    | `g1_amp_locomotion` | `s17_amp_locomotion`   | `s17_amp_flatwalk`         |
| `amp_motion_files`   | `motions/g1/amp`    | `motions/s17/amp/loco` | `motions/s17/amp/FlatWalk` |
| `min_normalized_std` | 29 × 0.05           | **21 × 0.05**          | **21 × 0.05**              |
| `amp_anchor_name`    | `torso_link`        | `torso`                | `torso`                    |
| `amp_body_names`     | G1 连杆名              | s17 连杆名（§5.3）          | 同左                         |


`AMPLoader` 按 `amp_motion_files` 递归加载 NPZ；`num_actions` 从 env 读取（**21**）。legacy NPZ 的 23 列 `joint_pos` 在 `ampmotion_loader.py` 中自动去掉 head 列。**未修改** `amp_ppo.py`、discriminator 结构或 `amp_task_reward_lerp`。

### 5.7 验证任务注册

**安装（两个包都必须 editable 安装）：**

```bash
conda activate amp    # 或 mjlab 环境
cd ~/Blueprint/AMP_mjlab
pip install -e .
pip install -e rsl_rl/
```

```bash
python scripts/list_envs.py --keyword S17
# 预期：Biped-S17-AMP-Flat、Biped-S17-AMP-Rough、Biped-S17-AMP-FlatWalk
```

常见失败：`ModuleNotFoundError: rsl_rl.runners.amp_on_policy_runner` → 未安装本仓库 fork 的 `rsl_rl/`（见 §9）。

---

## 6. 训练与评估

### 6.1 Smoke 测试（建议正式训练前）

```bash
conda activate amp
cd /path/to/AMP_mjlab

# 可选：单 env reset 验证
python -c "
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
cfg = load_env_cfg('Biped-S17-AMP-FlatWalk')
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
obs, _ = env.reset()
print('reset OK, num_actions=', env.action_manager.total_action_dim)
env.close()
"

# Smoke 训练：少量 env + 少量 iter（FlatWalk 示例）
python scripts/train.py Biped-S17-AMP-FlatWalk \
  --env.scene.num-envs=64 \
  --agent.max-iterations=50 \
  --agent.run-name=smoke_flatwalk
```

通过标准：能加载对应 motion 目录、无 body 名报错、checkpoint 写入 `logs/rsl_rl/<experiment_name>/`。

### 6.2 正式训练

```bash
# LAFAN loco + recovery（平面）
python scripts/train.py Biped-S17-AMP-Flat --env.scene.num-envs=4096

# 同上 motion，rough 地形
python scripts/train.py Biped-S17-AMP-Rough --env.scene.num-envs=4096

# amp_gait 平地步态库（FlatWalk 专用 motion，不与 loco 混用）
python scripts/train.py Biped-S17-AMP-FlatWalk --env.scene.num-envs=4096

# from FlatWalk model resume.
python scripts/train.py Biped-S17-AMP-Flat \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --agent.load_run=from_flatwalk \
  --agent.load_checkpoint=model_40600.pt
```



TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/s17_amp_flatwalk
tensorboard --logdir logs/rsl_rl/s17_amp_locomotion
```

### 6.3 策略回放（play）

`scripts/play.py` 支持三种 viewer：


| `--viewer` | 说明                                                               |
| ---------- | ---------------------------------------------------------------- |
| `auto`（默认） | 有 `DISPLAY`/`WAYLAND` 时用 Native MuJoCo 窗口，否则 Viser               |
| `native`   | 本地 MuJoCo GUI（**无**键盘速度控制，twist 仍由 env 随机采样）                     |
| `viser`    | 浏览器 3D 界面（`http://localhost:8080`），带仿真控制与 debug 面板，**推荐远程/无头环境** |


**Viser 回放（推荐）：**

```bash
python scripts/play.py Biped-S17-AMP-FlatWalk \
  --checkpoint-file logs/rsl_rl/s17_amp_flatwalk/<run-dir>/model_3300.pt \
  --viewer viser \
  --num-envs 1
```

浏览器打开终端提示的 URL（通常 `http://127.0.0.1:8080`）。在 Viser 侧边栏可暂停/单步、切换 env、查看 contact 等 overlay；速度指令仍由环境的 `twist` command 采样（FlatWalk play 配置见 `env_cfgs.py`）。

**Native 窗口回放：**

```bash
python scripts/play.py Biped-S17-AMP-Flat \
  --checkpoint-file logs/rsl_rl/s17_amp_locomotion/<run-dir>/model_<iter>.pt \
  --viewer native
```

**Flat / Rough loco 示例：**

```bash
python scripts/play.py Biped-S17-AMP-Flat \
  --checkpoint-file logs/rsl_rl/s17_amp_locomotion/<run-dir>/model_<iter>.pt \
  --viewer viser
```

**常用 play 参数：**

```bash
# 指定 GPU / 单环境 / 关闭 termination（便于长时间观察）
python scripts/play.py Biped-S17-AMP-FlatWalk \
  --checkpoint-file path/to/model.pt \
  --viewer viser \
  --device cuda:0 \
  --num-envs 1 \
  --no-terminations

# 录制视频（需 checkpoint，输出到 run 目录 videos/play/）
python scripts/play.py Biped-S17-AMP-FlatWalk \
  --checkpoint-file path/to/model.pt \
  --video --video-length 500

# 导出 ONNX（默认开启，输出到 logs/.../export/）
python scripts/play.py Biped-S17-AMP-FlatWalk \
  --checkpoint-file path/to/model.pt \
  --export-onnx
```

**说明：**

- Checkpoint **只存 21 维 policy 权重**；修改机器人模型（如 head fixed）后，**无需重新训练**即可直接 play 查看新外观。
- 任务 ID 必须与训练时一致（FlatWalk checkpoint 用 `Biped-S17-AMP-FlatWalk`）。
- 日志目录由 `rl_cfg.py` 的 `experiment_name` 决定：`s17_amp_locomotion`（Flat/Rough）或 `s17_amp_flatwalk`。

训练现象参考 G1：约 20k iter 附近可能出现 recovery 能力阶跃，属正常现象（见主 README）。

---

## 7. 部署（wbc_fsm）

训练导出的 ONNX 含 obs normalizer，但部署端仍需对齐：


| 项目             | 说明                                                               |
| -------------- | ---------------------------------------------------------------- |
| obs 维度         | 由 **21** 关节 + IMU + command 等决定，与 G1（29 关节）不同                    |
| action 维度      | **21**（head 已 fixed，不在 policy 中）                                 |
| 关节顺序           | 与仿真 **21 hinge** 顺序一致（见 `S17_SIM_JOINT_NAMES`）；sim ↔ 真机需 mapping |
| IMU 帧          | s17 IMU 在 **torso**（G1 在 pelvis），部署语义不同                          |
| `ACTION_SCALE` | 来自 `s17_constants.py` 的 `S17_ACTION_SCALE`                       |
| AMP style body | 11 个 link，anchor=`torso`（见 §5.3）                                 |


部署代码仓库：[ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm)（`MJAmp State`）。

---

## 8. 目录速查

### AMP_mjlab

```text
src/assets/robots/biped_s17/
  xml/biped_s17.xml                    # MJCF 源文件（IMU、foot site；head hinge 在 get_spec 中删除）
  s17_constants.py                     # Entity + PD + 21-DOF action + head fixed
  kuavo_actuators.py                   # Kuavo 电机 T-N 模型
src/assets/motions/s17/
  paths.py                             # loco / FlatWalk motion 路径
  amp/loco/WalkandRun/ Recovery/       # Flat + Rough 任务
  amp/FlatWalk/                        # FlatWalk 任务（30 条）
scripts/mimic_npz_to_amp_npz.py        # LAFAN mimic → AMP
scripts/retarget_npz_to_amp_npz.py     # amp_gait retarget → AMP
scripts/utils/validate_amp_motions.py  # 批量校验 NPZ
scripts/utils/visualize_amp_npz.py     # 单条 motion 可视化
scripts/play.py                        # play（--viewer native|viser|auto）
src/tasks/amp_loco/config/g1/          # G1 参考配置
src/tasks/amp_loco/config/biped_s17/   # S17 任务配置（env_cfgs / rl_cfg / __init__）
rsl_rl/                                # 带 AMP 扩展的 rsl_rl（需 pip install -e rsl_rl/）
logs/rsl_rl/s17_amp_locomotion/        # Flat / Rough 训练日志
logs/rsl_rl/s17_amp_flatwalk/          # FlatWalk 训练日志
```

### leju_soma_retarget

```text
assets/motions/lafan1/lafan1_soma/              # 原始长 BVH
assets/motions/lafan1/lafan1_soma_clips/        # 切割后 BVH
assets/lafan1_soma_clips_bvh_config.json        # retarget 批处理配置
app/utilities/cut_bvh_clips.py                  # BVH 切割
app/retargeting/bvh_to_npz_converter.py         # BVH → mimic NPZ
run/retargeting/retarget_lafan_soma_clips_to_npz.sh
outputs/biped_s17/lafan1_soma_clips_mimic_npz/  # mimic NPZ 输出
soma_retargeter/configs/biped_s17/extra_cfg/lafan/  # IK / scaler 配置
```

---

## 9. 常见问题

**Q: `list_envs.py --keyword S17` 找不到任务？**  
A: 确认在 **Blueprint 目录** 下执行，且 `pip install -e .` 指向当前仓库（`pip show wbc_mjlab` 的 Editable location 应为当前路径）。若曾从其他路径安装，需重新 `pip install -e .`。

**Q: 报错 `No module named rsl_rl.runners.amp_on_policy_runner`？**  
A: 需安装本仓库 fork 的 rsl_rl：`pip install -e rsl_rl/`。mjlab 依赖的官方 `rsl-rl-lib` 不含 AMP runner。

**Q: SOMA 脚本产出的 NPZ 能直接训练吗？**  
A: 不能。必须先经 `mimic_npz_to_amp_npz.py` 转为 AMP schema。

**Q: 能否用 G1 的 CSV + csv_to_npz？**  
A: 可以，但需改 `csv_to_npz.py` 的机器人 env 与关节列表；当前推荐 BVH → mimic → AMP 链路，已与 SOMA retarget 对齐。

**Q: Recovery 片段要多长？**  
A: 建议单次 fall+getup **6–10 s**；当前 4 条 Recovery 约 6.7–8.1 s（@30fps 源），转 50fps 后约 6.7–8.1 s 有效时长。

**Q: 头关节要不要进 policy？**  
A: **不进**。`get_spec()` 将 `zhead_1/2_joint` 删 hinge（等价 URDF fixed），policy 为 **21 维**；legacy NPZ 仍含 23 列（head=0），加载时自动 strip。已有 21 维 checkpoint 可直接 play。

**Q: FlatWalk 和 Flat 有什么区别？**  
A: 地形同为平面，但 **motion 库与 AMP 判别器数据分离**：Flat/Rough 用 `amp/loco/`（LAFAN + recovery），FlatWalk 仅用 `amp/FlatWalk/`（amp_gait）。FlatWalk 无 recovery、速度指令范围更窄、`experiment_name=s17_amp_flatwalk`。详见 §5.3。

**Q: play 时机器人不走 / 原地转？**  
A: Native viewer 无键盘控速，twist 由 env 随机采样（含一定比例的 standing / heading）。建议 `--viewer viser --num-envs 1`；FlatWalk 训练 checkpoint 需用任务 ID `Biped-S17-AMP-FlatWalk`。

**Q: Viser 打不开？**  
A: 确认 `--viewer viser`，查看终端输出的 URL；远程机器需 SSH 端口转发（如 `ssh -L 8080:localhost:8080`）。无 DISPLAY 时 `auto` 也会选 Viser。

**Q: Rough 和 Flat 会混用 FlatWalk motion 吗？**  
A: **不会**。`paths.py` + `env_cfgs.py` / `rl_cfg.py` 已按任务分开；Rough/Flat 的 `amp_motion_files` 为 `amp/loco/`，FlatWalk 为 `amp/FlatWalk/`。

**Q: Kuavo 执行器的 T-N 曲线生效了吗？**  
A: 当前 `X1` 未标定（默认 1e9），高速降扭未启用；实际为恒定扭矩上限。见 §5.5。

**Q: retarget OOM / Killed？**  
A: 降低 `RETARGET_BATCH_SIZE`（如 4 或 8）。

---

## 10. 推荐实施顺序

```text
[✓] motion NPZ + 机器人 XML
[✓] loco motion 迁至 amp/loco/；FlatWalk motion（30 条）+ retarget 脚本
[✓] 修改 biped_s17.xml（IMU 名、foot site）；get_spec head fixed + 21 DOF
[✓] s17_constants.py + kuavo_actuators.py + robots/__init__.py
[✓] config/biped_s17/ 三套任务 + FlatWalk 注册
[✓] pip install -e . + pip install -e rsl_rl/
[✓] list_envs → smoke 训练
[→] FlatWalk / Flat 全量训练 → play（Viser）可视化
[ ] Rough 地形训练
[ ] ONNX 导出 → wbc_fsm 部署映射
```

FlatWalk 与 Flat 可并行实验；评估时务必匹配任务 ID 与 checkpoint 的 `experiment_name`。