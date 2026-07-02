"""RL configuration for biped_s17 AMP locomotion task."""

from dataclasses import dataclass, field
from typing import List

from src.assets.motions.s17.paths import FLATWALK_MOTION_ROOT_STR, LOCO_MOTION_ROOT_STR
from src.assets.robots.biped_s17.s17_constants import S17_NUM_ACTIONS
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Extended runner config with AMP-specific parameters."""
  amp_reward_coef: float = 0.1
  amp_motion_files: str = ""
  amp_num_preload_transitions: int = 200000
  amp_task_reward_lerp: float = 0.75
  amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
  min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * S17_NUM_ACTIONS)
  amp_body_names: tuple = ()
  amp_anchor_name: str = ""


def biped_s17_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create RL runner configuration for biped_s17 AMP locomotion task."""
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
    ),
    experiment_name="s17_amp_locomotion",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100001,
    amp_reward_coef=0.1,
    amp_motion_files=LOCO_MOTION_ROOT_STR,
    amp_num_preload_transitions=200000,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * S17_NUM_ACTIONS,
    amp_body_names=(
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
    ),
    amp_anchor_name="torso",
  )


def biped_s17_amp_flatwalk_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create RL runner configuration for biped_s17 AMP flat-walk (amp_gait) task."""
  cfg = biped_s17_amp_ppo_runner_cfg()
  cfg.experiment_name = "s17_amp_flatwalk"
  cfg.amp_motion_files = FLATWALK_MOTION_ROOT_STR
  return cfg
