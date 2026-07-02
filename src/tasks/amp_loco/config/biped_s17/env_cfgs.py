"""Kuavo biped_s17 AMP locomotion environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.assets.motions.s17.paths import (
  FLATWALK_MOTION_ROOT_STR,
  LOCO_RECOVERY_DIR_STR,
  LOCO_WALK_AND_RUN_DIR_STR,
)
from src.assets.robots.biped_s17.s17_constants import (
  S17_ACTION_SCALE,
  S17_ACTUATED_JOINT_NAMES,
  get_biped_s17_robot_cfg,
)
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg


def _apply_s17_no_head_joint_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use actuated joints only (21 DOF) in actor/critic joint observations."""
  joint_cfg = SceneEntityCfg("robot", joint_names=S17_ACTUATED_JOINT_NAMES)
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.terms["joint_pos"].params["asset_cfg"] = joint_cfg
    group.terms["joint_vel"].params["asset_cfg"] = joint_cfg


def biped_s17_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create biped_s17 rough terrain AMP configuration."""
  cfg = make_amp_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_biped_s17_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 10)
  )
  body_names = (
    "base_link",
    "leg_l2_link",
    "leg_l4_link",
    "leg_l6_link",
    "leg_r2_link",
    "leg_r4_link",
    "leg_r6_link",
    "zarm_l2_link",
    "zarm_l4_link",
    "zarm_r2_link",
    "zarm_r4_link",
  )
  anchor_name = "torso"
  root_name = "base_link"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_l6_link|leg_r6_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = S17_ACTUATED_JOINT_NAMES
  joint_pos_action.scale = S17_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.25

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  cfg.events["init_motion_loader"].params["motion_dir"] = LOCO_WALK_AND_RUN_DIR_STR
  cfg.events["init_motion_loader"].params["recovery_dir"] = LOCO_RECOVERY_DIR_STR
  cfg.events["reset_from_motion"].params["motion_dir"] = LOCO_WALK_AND_RUN_DIR_STR

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = (
    body_names
  )

  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = (
    body_names
  )

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = (
    body_names
  )

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (
    anchor_name,
  )
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = (
    body_names
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

  _apply_s17_no_head_joint_obs(cfg)

  return cfg


def biped_s17_amp_flatwalk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create biped_s17 flat terrain AMP configuration using amp_gait motion library."""
  cfg = biped_s17_amp_flat_env_cfg(play=play)

  cfg.events["init_motion_loader"].params["motion_dir"] = FLATWALK_MOTION_ROOT_STR
  cfg.events["init_motion_loader"].params["recovery_dir"] = None
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0
  cfg.events["reset_from_motion"].params["motion_dir"] = FLATWALK_MOTION_ROOT_STR

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.6, 1.2)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  if play:
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)
    twist_cmd.ranges.lin_vel_y = (-0.25, 0.25)
    twist_cmd.ranges.ang_vel_z = (-2.5, 2.5)
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0

  return cfg


def biped_s17_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create biped_s17 flat terrain AMP configuration."""
  cfg = biped_s17_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg
