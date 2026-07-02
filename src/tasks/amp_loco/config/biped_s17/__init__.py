from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  biped_s17_amp_flat_env_cfg,
  biped_s17_amp_flatwalk_env_cfg,
  biped_s17_amp_rough_env_cfg,
)
from .rl_cfg import biped_s17_amp_flatwalk_ppo_runner_cfg, biped_s17_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Biped-S17-AMP-Rough",
  env_cfg=biped_s17_amp_rough_env_cfg(),
  play_env_cfg=biped_s17_amp_rough_env_cfg(play=True),
  rl_cfg=biped_s17_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Biped-S17-AMP-Flat",
  env_cfg=biped_s17_amp_flat_env_cfg(),
  play_env_cfg=biped_s17_amp_flat_env_cfg(play=True),
  rl_cfg=biped_s17_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Biped-S17-AMP-FlatWalk",
  env_cfg=biped_s17_amp_flatwalk_env_cfg(),
  play_env_cfg=biped_s17_amp_flatwalk_env_cfg(play=True),
  rl_cfg=biped_s17_amp_flatwalk_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
