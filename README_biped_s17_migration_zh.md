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


| 模块                                        | 状态          | 路径/说明                                  |
| ----------------------------------------- | ----------- | -------------------------------------- |
| 机器人 MJCF + mesh                           | ✅ 已就绪       | `src/assets/robots/biped_s17/`         |
| AMP 格式 motion NPZ                         | ✅ 已就绪（16 条） | `src/assets/motions/s17/amp/`          |
| mimic → AMP 转换脚本                          | ✅ 已就绪       | `scripts/mimic_npz_to_amp_npz.py`      |
| MJCF 传感器 / foot site                      | ✅ 已就绪       | `biped_s17.xml`（§5.2）                  |
| `s17_constants.py` + `kuavo_actuators.py` | ✅ 已就绪       | §5.1、§5.5                              |
| AMP 任务注册与配置                               | ✅ 已就绪       | `src/tasks/amp_loco/config/biped_s17/` |
| Flat 训练验证                                 | 🔄 进行中      | `Biped-S17-AMP-Flat`                   |
| wbc_fsm 部署映射                              | ⏳ 待完成       | obs/action 维度与关节顺序                     |


---

## 2. G1 与 biped_s17 关键差异


| 项目          | G1                         | biped_s17                        |
| ----------- | -------------------------- | -------------------------------- |
| 可控 DOF      | 29                         | **23**（不含头则 21，当前 XML 含头 2 DOF）  |
| 腰           | yaw / roll / pitch         | **仅 waist_yaw**                  |
| 臂           | 7×2（含腕）                    | **4×2**                          |
| Root link   | `pelvis`                   | `base_link`（free joint）          |
| AMP anchor  | `torso_link`               | `torso`                          |
| IMU site    | pelvis，`imu_ang_vel`       | torso 上 `imu`（已对齐 mjlab 命名）      |
| Foot site   | `left_foot` / `right_foot` | `leg_l6/r6_link` 上已添加            |
| 执行器模型       | `UnitreeActuatorCfg`       | `KuavoActuatorCfg_`*（kuavo.json） |
| 任务 ID       | `Unitree-G1-AMP-Flat`      | `Biped-S17-AMP-Flat`             |
| NPZ bodies  | 30                         | **24**                           |
| NPZ joint 维 | 29                         | **23**                           |


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
                                                    │
                                                    ▼
                                               s17_constants.py + kuavo_actuators.py
                                               config/biped_s17/（env + rl）
                                                    │
                                                    ▼
                                               train.py（Biped-S17-AMP-Flat）
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
  --output-dir src/assets/motions/s17/amp \
  --input-fps 30 \
  --output-fps 50 \
  --mjcf src/assets/robots/biped_s17/xml/biped_s17.xml

/home/lzl/miniforge3/envs/soma/bin/python scripts/mimic_npz_to_amp_npz.py   --input-dir /home/lzl/Projects/leju_soma_retarget/outputs/roban/lafan1/amp   --output-dir src/assets/motions/s17/amp/   --input-fps 30   --output-fps 50   --mjcf src/assets/robots/biped_s17/xml/biped_s17.xml
```

输出目录结构（与 G1 一致）：

```text
src/assets/motions/s17/amp/
  WalkandRun/    # 12 条
  Recovery/      # 4 条
```

当前 AMP NPZ 规格：

- `fps`: 50
- `joint_pos` / `joint_vel`: `(T, 23)`
- `body_pos_w` / `body_quat_w` / `body_lin_vel_w` / `body_ang_vel_w`: `(T, 24, *)`
- `body_quat_w`: **wxyz**

### 4.4 新增/扩充 motion 的检查清单

1. 在 `cut_bvh_clips.py` 增加 `CLIP_JOBS` 条目
2. 重新跑 §4.2、§4.3
3. 目视或脚本检查：根高度、关节范围、无明显穿透
4. 确认 `WalkandRun` 与 `Recovery` 子目录分类正确

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


| 项                  | 实现                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------- |
| `get_spec()`       | 加载 `biped_s17.xml` + mesh；删除 XML 原生 `<motor>`；自动命名碰撞 geom（`left_foot*_collision` 等） |
| 初始姿态               | `KNEES_BENT_KEYFRAME`，z=0.95（参考 G1 弯膝，关节名映射到 s17）                                   |
| 执行器                | `KuavoActuatorCfg_*`（见 §5.5），8 组共 **23 DOF**                                        |
| PD 增益              | 与 G1 相同公式：`stiffness = armature × (10 Hz × 2π)²`，`armature=0.003`                   |
| `S17_ACTION_SCALE` | `0.25 × effort / stiffness`，写入 `joint_pos` action                                   |
| 导出                 | `src/assets/robots/__init__.py` → `get_biped_s17_robot_cfg` / `S17_ACTION_SCALE`    |


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


| 文件            | 作用                                                     |
| ------------- | ------------------------------------------------------ |
| `env_cfgs.py` | 机器人 entity、body/foot/contact 映射、motion 路径、action scale |
| `rl_cfg.py`   | AMP 判别器 body 映射、`min_normalized_std`（23 维）、motion 目录   |
| `__init__.py` | 注册 `Biped-S17-AMP-Flat` / `Biped-S17-AMP-Rough`        |


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


| 用途                     | G1                          | S17                           |
| ---------------------- | --------------------------- | ----------------------------- |
| 机器人 Entity             | `get_g1_robot_cfg()`        | `get_biped_s17_robot_cfg()`   |
| 根 body（角速度惩罚等）         | `pelvis`                    | `base_link`                   |
| Anchor（速度跟踪、AMP 相对坐标）  | `torso_link`                | `torso`                       |
| 地形扫描 frame             | `pelvis`                    | `base_link`                   |
| 足地接触 subtree           | `*_ankle_roll_link`         | `leg_l6_link` / `leg_r6_link` |
| 自碰撞 subtree            | `pelvis`                    | `base_link`                   |
| 足摩擦 geom               | `left_foot1~7_collision`    | `left_foot1~9_collision`      |
| COM 随机化 body           | `torso_link`                | `torso`                       |
| WalkandRun motion      | `motions/g1/amp/WalkandRun` | `motions/s17/amp/WalkandRun`  |
| Recovery motion        | `motions/g1/amp/Recovery`   | `motions/s17/amp/Recovery`    |
| Action 维 / joint obs 维 | 29                          | **23**（由 entity 自动推断）         |


Critic / AMP 观测组的 `body_pos_b`、`body_ori_b`、`body_lin_vel_b`、`body_ang_vel_b` 均使用上表 anchor + 11 body 列表。

### 5.5 执行器模型（`kuavo_actuators.py`）

路径：`src/assets/robots/biped_s17/kuavo_actuators.py`

参考 G1 的 `unitree_actuators.py`，实现 T-N 曲线 + 摩擦模型的 `KuavoActuator`，参数来自：

- `kuavo-ros-control` → `kuavo_v17/kuavo.json`（`MOTORS_TYPE`、`joint_peak_torque_limits`、`joint_peak_velocity_limits`）
- `biped_s17.xml`（`actuatorfrcrange` / motor `ctrlrange` 作为 `effort_limit`）


| 电机型号             | 关节                                        |
| ---------------- | ----------------------------------------- |
| `PA81_25`        | leg_l1/l2/l4, leg_r1/r2/r4                |
| `PA76_25`        | leg_l3/r3                                 |
| `PA76_25_WAIST`  | waist_yaw                                 |
| `PA4315_36`      | leg_l5/l6, leg_r5/r6                      |
| `ruiwoPA60_16`   | zarm_l1/r1                                |
| `ruiwoPA4315_36` | zarm_l2~~l4, zarm_r2~~r4                  |
| `ruiwoPA4310_25` | zhead_1/2（Y1 取自 joint `actuatorfrcrange`） |


**注意：** 当前 `X1`（满扭拐点速度）未标定，默认为 `1e9`，T-N 降扭实际上未生效；等价于恒定扭矩上限 `min(Y1, effort_limit)`。后续可从电机手册补 `X1` 或采用 `X1 ≈ 0.7×X2` 启发式。

### 5.6 算法侧适配明细（`rl_cfg.py` vs G1）

PPO / AMP 网络结构与超参与 G1 **完全一致**（512-256-128、lr=1e-3、`amp_reward_coef=0.1` 等），仅改机器人相关字段：


| 参数                   | G1                  | S17                  |
| -------------------- | ------------------- | -------------------- |
| `experiment_name`    | `g1_amp_locomotion` | `s17_amp_locomotion` |
| `amp_motion_files`   | `motions/g1/amp`    | `motions/s17/amp`    |
| `min_normalized_std` | 29 × 0.05           | **23 × 0.05**        |
| `amp_anchor_name`    | `torso_link`        | `torso`              |
| `amp_body_names`     | G1 连杆名              | s17 连杆名（§5.3）        |


`AMPLoader` 启动时递归加载 `s17/amp/` 下全部 NPZ，按 `amp_body_names` + `amp_anchor_name` 计算 style 特征；`num_actions` 从 env 读取（23），自动对齐策略维度。**未修改** `amp_ppo.py`、discriminator 结构或 `amp_task_reward_lerp`。

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
# 预期：Biped-S17-AMP-Flat、Biped-S17-AMP-Rough
```

常见失败：`ModuleNotFoundError: rsl_rl.runners.amp_on_policy_runner` → 未安装本仓库 fork 的 `rsl_rl/`（见 §9）。

---

## 6. 训练与评估

### 6.1 Smoke 测试（建议正式训练前）

```bash
# 可选：单 env reset 验证
python -c "
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
cfg = load_env_cfg('Biped-S17-AMP-Flat')
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
obs, _ = env.reset()
print('reset OK')
env.close()
"

# Smoke 训练：少量 env + 少量 iter
python scripts/train.py Biped-S17-AMP-Flat \
  --env.scene.num-envs=64 \
  --agent.max-iterations=50 \
  --agent.run-name=smoke_s17
```

通过标准：能加载 16 条 motion、无 body 名报错、写出 checkpoint 到 `logs/rsl_rl/s17_amp_locomotion/`。

### 6.2 正式训练

```bash
python scripts/train.py Biped-S17-AMP-Flat --env.scene.num-envs=4096
```

### 6.3 评估

```bash
python scripts/play.py Biped-S17-AMP-Flat \
  --checkpoint-file logs/rsl_rl/s17_amp_locomotion/<run>/model_<iter>.pt
```

日志目录由 `rl_cfg.py` 中 `experiment_name` 决定（`s17_amp_locomotion`）。

训练现象参考 G1：约 20k iter 附近可能出现 recovery 能力阶跃，属正常现象（见主 README）。

---

## 7. 部署（wbc_fsm）

训练导出的 ONNX 含 obs normalizer，但部署端仍需对齐：


| 项目             | 说明                                                        |
| -------------- | --------------------------------------------------------- |
| obs 维度         | 由 23 关节 + IMU + command 等决定，与 G1（29 关节）不同                 |
| action 维度      | **23**（当前含头 2 DOF；若锁头需同步改 constants 与 motion）             |
| 关节顺序           | 与 `biped_s17.xml` 中 actuated joint 顺序一致，sim ↔ 真机需 mapping |
| IMU 帧          | s17 IMU 在 **torso**（G1 在 pelvis），部署语义不同                   |
| `ACTION_SCALE` | 来自 `s17_constants.py` 的 `S17_ACTION_SCALE`                |
| AMP style body | 11 个 link，anchor=`torso`（见 §5.3）                          |


部署代码仓库：[ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm)（`MJAmp State`）。

---

## 8. 目录速查

### AMP_mjlab

```text
src/assets/robots/biped_s17/
  xml/biped_s17.xml                    # MJCF（IMU、foot site 已改）
  s17_constants.py                     # Entity + PD + action scale
  kuavo_actuators.py                   # Kuavo 电机 T-N 模型
src/assets/motions/s17/amp/           # AMP 训练用 NPZ
scripts/mimic_npz_to_amp_npz.py       # mimic → AMP 转换
src/tasks/amp_loco/config/g1/         # G1 参考配置
src/tasks/amp_loco/config/biped_s17/  # S17 任务配置（env_cfgs / rl_cfg / __init__）
rsl_rl/                               # 带 AMP 扩展的 rsl_rl（需 pip install -e rsl_rl/）
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
A: locomotion 建议锁头或固定默认角，减少 action 维与 motion 复杂度；当前 XML 保留 2 DOF（23 维 action），motion 与训练已一致包含头关节。

**Q: Kuavo 执行器的 T-N 曲线生效了吗？**  
A: 当前 `X1` 未标定（默认 1e9），高速降扭未启用；实际为恒定扭矩上限。见 §5.5。

**Q: retarget OOM / Killed？**  
A: 降低 `RETARGET_BATCH_SIZE`（如 4 或 8）。

---

## 10. 推荐实施顺序

```text
[✓] motion NPZ + 机器人 XML
[✓] 修改 biped_s17.xml（IMU 名、foot site）
[✓] s17_constants.py + kuavo_actuators.py + robots/__init__.py
[✓] src/tasks/amp_loco/config/biped_s17/ 三套配置 + register
[✓] pip install -e . + pip install -e rsl_rl/
[✓] list_envs → smoke 训练
[→] Flat 全量训练（4096 env）→ play 可视化
[ ] Rough 地形训练
[ ] ONNX 导出 → wbc_fsm 部署映射
```

训练已在 `Biped-S17-AMP-Flat` 上启动后，下一步为监控 TensorBoard、play 可视化，再视需要上 Rough 与部署。