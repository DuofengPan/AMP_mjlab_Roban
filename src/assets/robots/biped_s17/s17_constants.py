"""Kuavo biped_s17 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from src.assets.robots.biped_s17.kuavo_actuators import (
    KuavoActuatorCfg_PA4315_36,
    KuavoActuatorCfg_PA76_25,
    KuavoActuatorCfg_PA76_25_WAIST,
    KuavoActuatorCfg_PA81_25,
    KuavoActuatorCfg_ruiwoPA4315_36,
    KuavoActuatorCfg_ruiwoPA60_16,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

##
# MJCF and assets.
##

S17_XML: Path = SRC_PATH / "assets" / "robots" / "biped_s17" / "xml" / "biped_s17.xml"
assert S17_XML.exists()

S17_HEAD_JOINT_NAMES: tuple[str, ...] = ("zhead_1_joint", "zhead_2_joint")

# Full MJCF hinge order used by legacy AMP NPZ (head columns are always last).
S17_MOTION_JOINT_NAMES: tuple[str, ...] = (
  "leg_l1_joint",
  "leg_l2_joint",
  "leg_l3_joint",
  "leg_l4_joint",
  "leg_l5_joint",
  "leg_l6_joint",
  "leg_r1_joint",
  "leg_r2_joint",
  "leg_r3_joint",
  "leg_r4_joint",
  "leg_r5_joint",
  "leg_r6_joint",
  "waist_yaw_joint",
  "zarm_l1_joint",
  "zarm_l2_joint",
  "zarm_l3_joint",
  "zarm_l4_joint",
  "zarm_r1_joint",
  "zarm_r2_joint",
  "zarm_r3_joint",
  "zarm_r4_joint",
  *S17_HEAD_JOINT_NAMES,
)

S17_SIM_JOINT_NAMES: tuple[str, ...] = tuple(
  name for name in S17_MOTION_JOINT_NAMES if name not in S17_HEAD_JOINT_NAMES
)
assert len(S17_SIM_JOINT_NAMES) == 21


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, S17_XML.parent / "meshes", meshdir)
  return assets


def _clear_native_actuators(spec: mujoco.MjSpec) -> None:
  """Remove XML <motor> actuators; mjlab adds position actuators instead."""
  for actuator in list(spec.actuators):
    spec.delete(actuator)


def _fix_head_joints_as_fixed(spec: mujoco.MjSpec) -> None:
  """Weld head links to torso (MuJoCo fixed joint = no hinge DOF).

  Removes passive head hinges and their sensors so the head stays at the MJCF
  zero pose. Legacy AMP NPZ still stores 23 joint columns; loaders drop the
  trailing head columns to match the 21-DOF sim model.
  """
  head_joints = set(S17_HEAD_JOINT_NAMES)
  for sensor in list(spec.sensors):
    if getattr(sensor, "objname", None) in head_joints:
      spec.delete(sensor)
  for joint in list(spec.joints):
    if joint.name in head_joints:
      spec.delete(joint)


def strip_head_from_motion_dof(array):
  """Drop trailing head columns from legacy 23-dim motion joint arrays."""
  import numpy as np

  arr = np.asarray(array)
  ndof = int(arr.shape[-1])
  sim_ndof = len(S17_SIM_JOINT_NAMES)
  motion_ndof = len(S17_MOTION_JOINT_NAMES)
  if ndof == sim_ndof:
    return arr
  if ndof == motion_ndof:
    return arr[..., :sim_ndof]
  raise ValueError(
    f"Expected motion joint dim {sim_ndof} or legacy {motion_ndof}, got {ndof}"
  )


def _assign_collision_geom_names(spec: mujoco.MjSpec) -> None:
  """Name collision geoms so mjlab CollisionCfg regexes can match them."""
  counters: dict[str, int] = {}
  for geom in spec.geoms:
    if geom.contype == 0 and geom.conaffinity == 0:
      continue
    parent = geom.parent
    body_name = getattr(parent, "name", None) if parent is not None else None
    if not body_name:
      continue
    if body_name == "leg_l6_link":
      prefix = "left_foot"
    elif body_name == "leg_r6_link":
      prefix = "right_foot"
    else:
      prefix = body_name
    idx = counters.get(prefix, 0) + 1
    counters[prefix] = idx
    if geom.name:
      continue
    if prefix in ("left_foot", "right_foot"):
      geom.name = f"{prefix}{idx}_collision"
    elif idx == 1:
      geom.name = f"{prefix}_collision"
    else:
      geom.name = f"{prefix}_collision_{idx}"


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(S17_XML))
  spec.assets = get_assets(spec.meshdir)
  _clear_native_actuators(spec)
  _fix_head_joints_as_fixed(spec)
  _assign_collision_geom_names(spec)
  return spec


##
# Actuator config.
##

# Nominal reflected inertia from MJCF joint armature (0.003 on all joints).
ARMATURE = 0.003

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

# Effort limits from biped_s17.xml motor ctrlrange (simulation hard cap).
# Waist: motor peak is 80 N·m (Roban 2.2) but MJCF ctrlrange is ±50 — keep 50.
# Head yaw: MJCF ctrlrange ±1.5 N·m (conservative); pitch ±12 N·m.
EFFORT_LEG_HIGH = 150.0
EFFORT_LEG_MID = 70.0
EFFORT_ANKLE = 74.0
EFFORT_WAIST = 50.0
EFFORT_ARM = 37.0
EFFORT_ARM_SHOULDER = 14.1

S17_ACTUATOR_LEG_HIGH = KuavoActuatorCfg_PA81_25(
  target_names_expr=(
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l4_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r4_joint",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LEG_HIGH,
  armature=ARMATURE,
)
S17_ACTUATOR_LEG_MID = KuavoActuatorCfg_PA76_25(
  target_names_expr=("leg_l3_joint", "leg_r3_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LEG_MID,
  armature=ARMATURE,
)
S17_ACTUATOR_ANKLE = KuavoActuatorCfg_PA4315_36(
  target_names_expr=(
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r5_joint",
    "leg_r6_joint",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_ANKLE,
  armature=ARMATURE,
)
S17_ACTUATOR_WAIST = KuavoActuatorCfg_PA76_25_WAIST(
  target_names_expr=("waist_yaw_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_WAIST,
  armature=ARMATURE,
)
S17_ACTUATOR_ARM = KuavoActuatorCfg_ruiwoPA4315_36(
  target_names_expr=(
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
  ),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_ARM,
  armature=ARMATURE,
)
S17_ACTUATOR_ARM_SHOULDER = KuavoActuatorCfg_ruiwoPA60_16(
  target_names_expr=("zarm_l1_joint", "zarm_r1_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_ARM_SHOULDER,
  armature=ARMATURE,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.98),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.95),
  joint_pos={
    "leg_l1_joint": -0.312,
    "leg_r1_joint": -0.312,
    "leg_l4_joint": 0.669,
    "leg_r4_joint": 0.669,
    "leg_l5_joint": -0.363,
    "leg_r5_joint": -0.363,
    "zarm_l2_joint": 0.2,
    "zarm_r2_joint": -0.2,
    "zarm_l4_joint": 0.6,
    "zarm_r4_joint": 0.6,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_REGEX = r"^(left|right)_foot[0-9]+_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: (0.6,)},
)

##
# Final config.
##

S17_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    S17_ACTUATOR_LEG_HIGH,
    S17_ACTUATOR_LEG_MID,
    S17_ACTUATOR_ANKLE,
    S17_ACTUATOR_WAIST,
    S17_ACTUATOR_ARM,
    S17_ACTUATOR_ARM_SHOULDER,
  ),
  soft_joint_pos_limit_factor=0.9,
)

S17_NUM_ACTIONS = 21
S17_ACTUATED_JOINT_NAMES: tuple[str, ...] = tuple(
  name
  for actuator in S17_ARTICULATION.actuators
  for name in actuator.target_names_expr
)
assert len(S17_ACTUATED_JOINT_NAMES) == S17_NUM_ACTIONS


def get_biped_s17_robot_cfg() -> EntityCfg:
  """Get a fresh biped_s17 robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=S17_ARTICULATION,
  )


# Cap hip/knee action scale to G1-like authority (raw 0.25*effort/stiffness can exceed 3.0).
S17_LEG_HIP_KNEE_JOINT_NAMES: tuple[str, ...] = (
  "leg_l1_joint",
  "leg_l2_joint",
  "leg_l3_joint",
  "leg_l4_joint",
  "leg_r1_joint",
  "leg_r2_joint",
  "leg_r3_joint",
  "leg_r4_joint",
)
S17_LEG_HIP_KNEE_ACTION_SCALE_CAP: float = 0.55

S17_ACTION_SCALE: dict[str, float] = {}
for actuator in S17_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort is not None
  for name in actuator.target_names_expr:
    S17_ACTION_SCALE[name] = 0.25 * effort / stiffness

for _joint_name in S17_LEG_HIP_KNEE_JOINT_NAMES:
  S17_ACTION_SCALE[_joint_name] = min(
    S17_ACTION_SCALE[_joint_name],
    S17_LEG_HIP_KNEE_ACTION_SCALE_CAP,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_biped_s17_robot_cfg())
  viewer.launch(robot.spec.compile())
