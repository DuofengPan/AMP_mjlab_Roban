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
    KuavoActuatorCfg_ruiwoPA4310_25,
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


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, S17_XML.parent / "meshes", meshdir)
  return assets


def _clear_native_actuators(spec: mujoco.MjSpec) -> None:
  """Remove XML <motor> actuators; mjlab adds position actuators instead."""
  for actuator in list(spec.actuators):
    spec.delete(actuator)


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
EFFORT_HEAD_YAW = 1.5
EFFORT_HEAD_PITCH = 12.0

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
S17_ACTUATOR_HEAD_YAW = KuavoActuatorCfg_ruiwoPA4310_25(
  target_names_expr=("zhead_1_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_HEAD_YAW,
  Y1=EFFORT_HEAD_YAW,
  armature=ARMATURE,
)
S17_ACTUATOR_HEAD_PITCH = KuavoActuatorCfg_ruiwoPA4310_25(
  target_names_expr=("zhead_2_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_HEAD_PITCH,
  Y1=EFFORT_HEAD_PITCH,
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
    S17_ACTUATOR_HEAD_YAW,
    S17_ACTUATOR_HEAD_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_biped_s17_robot_cfg() -> EntityCfg:
  """Get a fresh biped_s17 robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=S17_ARTICULATION,
  )


S17_ACTION_SCALE: dict[str, float] = {}
for actuator in S17_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort is not None
  for name in actuator.target_names_expr:
    S17_ACTION_SCALE[name] = 0.25 * effort / stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_biped_s17_robot_cfg())
  viewer.launch(robot.spec.compile())
