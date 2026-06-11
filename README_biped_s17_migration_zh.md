# biped_s17 迁移指南（G1 → AMP_mjlab）

本文档记录将 AMP_mjlab 从 Unitree G1 迁移到 **biped_s17（Kuavo S17）** 的完整流程，便于复现与后续维护。

主项目说明见 [README_zh.md](README_zh.md)。G1 训练流程仍按原 README 操作；本文仅描述 **新机器人** 相关差异与步骤。

---

## 1. 迁移目标

在 AMP_mjlab 中实现与 G1 相同的能力栈：

- 单一 policy 同时学习 **locomotion（走/跑）** 与 **recovery（跌倒恢复）**
- AMP 判别器约束动作风格
- 训练 → ONNX 导出 → [wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) 部署

当前进度（截至文档编写时）：

| 模块 | 状态 | 路径/说明 |
|------|------|-----------|
| 机器人 MJCF + mesh | 已就绪 | `src/assets/robots/biped_s17/` |
| AMP 格式 motion NPZ | 已就绪（16 条） | `src/assets/motions/s17/amp/` |
| mimic → AMP 转换脚本 | 已就绪 | `scripts/mimic_npz_to_amp_npz.py` |
| `s17_constants.py` + mjlab Entity | **待完成** | 参考 `unitree_g1/g1_constants.py` |
| AMP 任务注册与配置 | **待完成** | 参考 `src/tasks/amp_loco/config/g1/` |
| XML 传感器名对齐 mjlab | **待完成** | 见下文 §4 |
| wbc_fsm 部署映射 | **待完成** | obs/action 维度与关节顺序 |

---

## 2. G1 与 biped_s17 关键差异

| 项目 | G1 | biped_s17 |
|------|-----|-----------|
| 可控 DOF | 29 | **23**（不含头则 21，当前 XML 含头 2 DOF） |
| 腰 | yaw / roll / pitch | **仅 waist_yaw** |
| 臂 | 7×2（含腕） | **4×2** |
| Root link | `pelvis` | `base_link`（free joint） |
| AMP anchor | `torso_link` | `torso` |
| IMU site | pelvis，`imu_ang_vel` | torso 上 `imu`，需改名为 mjlab 约定 |
| Foot site | `left_foot` / `right_foot` | 需在 `leg_l6/r6_link` 上添加 |
| NPZ bodies | 30 | **24** |
| NPZ joint 维 | 29 | **23** |

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
                                               s17 task config + 训练
                                                    │
                                                    ▼
                                               ONNX → wbc_fsm 部署
```

### 3.1 两种 NPZ 格式（重要）

| 来源 | Keys | 能否直接给 AMP 训练 |
|------|------|---------------------|
| SOMA `bvh_to_npz_converter.py` | `data`, `fps` | **否**（mimic 中间格式） |
| AMP `csv_to_npz.py` / `mimic_npz_to_amp_npz.py` | `fps`, `joint_pos`, `joint_vel`, `body_*` | **是** |

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

## 5. AMP_mjlab 内待完成的机器人接入

motion 与 XML 已就绪后，还需完成以下模块才能 `train.py`：

### 5.1 机器人 constants（`src/assets/robots/biped_s17/`）

参考 G1 的 `g1_constants.py`，新建例如 `s17_constants.py`：

- `get_spec()`：加载 `xml/biped_s17.xml` 与 mesh
- `EntityCfg`：初始姿态（可用 `scene.xml` 里 `home` keyframe）
- `EntityArticulationInfoCfg`：各关节 PD / effort（XML 内已有 motor 力矩范围可作参考）
- `S17_ACTION_SCALE`：与 G1 相同公式 `0.25 * effort / stiffness`
- 在 `src/assets/robots/__init__.py` 导出

### 5.2 修改 MJCF 以匹配 mjlab 任务

G1 AMP 任务硬编码使用以下传感器名（见 `src/tasks/amp_loco/amp_env_cfg.py`）：

```python
"robot/imu_ang_vel"
"robot/imu_lin_vel"
```

建议在 `biped_s17.xml` 的 `<sensor>` 中增加（或重命名）：

```xml
<gyro name="imu_ang_vel" site="imu"/>
<velocimeter name="imu_lin_vel" site="imu"/>
```

Foot 相关奖励需要 foot site，建议在 `leg_l6_link` / `leg_r6_link` 添加：

```xml
<site name="left_foot" .../>
<site name="right_foot" .../>
```

### 5.3 AMP 任务配置（复制 G1 并改映射）

新建目录：`src/tasks/amp_loco/config/biped_s17/`

| 文件 | 作用 |
|------|------|
| `env_cfgs.py` | 机器人 entity、body 名、foot/contact、motion 路径 |
| `rl_cfg.py` | `amp_body_names`、`amp_anchor_name`、`min_normalized_std`（长度 23）、motion 目录 |
| `__init__.py` | `register_mjlab_task`，如 `Biped-S17-AMP-Flat` |

**建议 AMP body 映射（11 个，与 G1 思路一致）：**

```text
base_link
leg_l2_link, leg_l4_link, leg_l6_link
leg_r2_link, leg_r4_link, leg_r6_link
zarm_l2_link, zarm_l4_link
zarm_r2_link, zarm_r4_link
anchor: torso
root:   base_link
```

**env_cfgs.py 中还需修改：**

- `terrain_scan.frame.name` → `base_link`
- 接触传感器 subtree → `leg_l6_link` / `leg_r6_link`
- `motion_dir` / `recovery_dir` → `src/assets/motions/s17/amp/...`
- `min_normalized_std` → 23 维

### 5.4 验证任务注册

```bash
conda activate mjlab   # 或 amp 环境，需能 import mjlab
cd AMP_mjlab
python scripts/list_envs.py --keyword S17
```

---

## 6. 训练与评估（配置完成后）

```bash
# Flat 起步
python scripts/train.py Biped-S17-AMP-Flat --env.scene.num-envs=4096

# 评估
python scripts/play.py Biped-S17-AMP-Flat \
  --checkpoint-file logs/rsl_rl/s17_amp_locomotion/<run>/model_<iter>.pt
```

日志目录由 `rl_cfg.py` 中 `experiment_name` 决定（建议设为 `s17_amp_locomotion`）。

训练现象参考 G1：约 20k iter 附近可能出现 recovery 能力阶跃，属正常现象（见主 README）。

---

## 7. 部署（wbc_fsm）

训练导出的 ONNX 含 obs normalizer，但部署端仍需对齐：

| 项目 | 说明 |
|------|------|
| obs 维度 | 由 s17 关节数、AMP body 数、height scan 等决定，与 G1 不同 |
| action 维度 | 23（或锁定头关节后的维数） |
| 关节顺序 | sim ↔ 真机 mapping，需与训练一致 |
| IMU 帧 | s17 IMU 在 torso，与 G1 pelvis 语义不同 |
| `ACTION_SCALE` | 来自 `s17_constants.py`，部署需一致 |

部署代码仓库：[ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm)（`MJAmp State`）。

---

## 8. 目录速查

### AMP_mjlab

```text
src/assets/robots/biped_s17/          # MJCF、mesh、URDF
src/assets/motions/s17/amp/           # AMP 训练用 NPZ
scripts/mimic_npz_to_amp_npz.py       # mimic → AMP 转换
src/tasks/amp_loco/config/g1/         # G1 参考配置
src/tasks/amp_loco/config/biped_s17/  # 【待建】s17 配置
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

**Q: SOMA 脚本产出的 NPZ 能直接训练吗？**  
A: 不能。必须先经 `mimic_npz_to_amp_npz.py` 转为 AMP schema。

**Q: 能否用 G1 的 CSV + csv_to_npz？**  
A: 可以，但需改 `csv_to_npz.py` 的机器人 env 与关节列表；当前推荐 BVH → mimic → AMP 链路，已与 SOMA retarget 对齐。

**Q: Recovery 片段要多长？**  
A: 建议单次 fall+getup **6–10 s**；当前 4 条 Recovery 约 6.7–8.1 s（@30fps 源），转 50fps 后约 6.7–8.1 s 有效时长。

**Q: 头关节要不要进 policy？**  
A: locomotion 建议锁头或固定默认角，减少 action 维与 motion 复杂度；若 XML 保留 2 DOF，需在 constants 与 motion 中一致处理。

**Q: retarget OOM / Killed？**  
A: 降低 `RETARGET_BATCH_SIZE`（如 4 或 8）。

---

## 10. 推荐实施顺序

```text
[已完成] motion NPZ + 机器人 XML
    ↓
[ ] s17_constants.py + robots/__init__.py
    ↓
[ ] 修改 biped_s17.xml（IMU 名、foot site）
    ↓
[ ] src/tasks/amp_loco/config/biped_s17/ 三套配置 + register
    ↓
[ ] list_envs → 单 env reset  smoke test
    ↓
[ ] Flat 训练 → play 可视化 → Rough → ONNX → wbc_fsm
```

完成 §5 后即可在本仓库内启动 s17 AMP 训练；部署阶段在 wbc_fsm 中补齐接口映射。
