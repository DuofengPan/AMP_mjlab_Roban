"""S17 AMP motion directory layout.

Task-specific roots (no cross-mixing between loco and flatwalk):

  amp/loco/          Rough + Flat: WalkandRun + Recovery (env reset + discriminator)
    WalkandRun/      amp_gait retarget clips (same library as FlatWalk/)
    Recovery/        AMP NPZ only (no mimic intermediates under amp/loco/)
  amp/FlatWalk/      FlatWalk task only (env reset + discriminator)
  mimic/             SOMA mimic NPZ backups (data+fps); not loaded by training
"""

from __future__ import annotations

import os
from pathlib import Path

_S17_MOTIONS_ROOT = Path(__file__).resolve().parent

AMP_ROOT = _S17_MOTIONS_ROOT / "amp"
LOCO_MOTION_ROOT = AMP_ROOT / "loco"
LOCO_WALK_AND_RUN_DIR = LOCO_MOTION_ROOT / "WalkandRun"
LOCO_RECOVERY_DIR = LOCO_MOTION_ROOT / "Recovery"
FLATWALK_MOTION_ROOT = AMP_ROOT / "FlatWalk"


def as_repo_path(path: Path) -> str:
  """Return normalized absolute path string."""
  return os.path.normpath(str(path.resolve()))

LOCO_MOTION_ROOT_STR = as_repo_path(LOCO_MOTION_ROOT)
LOCO_WALK_AND_RUN_DIR_STR = as_repo_path(LOCO_WALK_AND_RUN_DIR)
LOCO_RECOVERY_DIR_STR = as_repo_path(LOCO_RECOVERY_DIR)
FLATWALK_MOTION_ROOT_STR = as_repo_path(FLATWALK_MOTION_ROOT)
