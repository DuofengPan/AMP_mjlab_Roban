"""Kuavo (biped_s17) actuator model with optional torque-speed limiting.

Motor parameters are sourced only from confirmed project data:
  - kuavo_v17/kuavo.json  (joint_peak_torque_limits, joint_peak_velocity_limits, MOTORS_TYPE)
  - biped_s17.xml         (actuatorfrcrange / effort_limit, armature, frictionloss)

Parameters left at defaults when not available in those sources:
  - X1 (T-N knee speed): unknown — default disables speed-dependent derating
  - Y2 (reverse peak torque): unknown — defaults to Y1
  - Fs, Fd (friction): MJCF frictionloss=0.0 — defaults to 0
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco

from mjlab.actuator import Actuator, ActuatorCmd, BuiltinPositionActuatorCfg
from mjlab.utils.spec import create_position_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


class KuavoActuator(Actuator):
    """Actuator with PD control, torque-speed (T-N) clipping, and friction model."""

    cfg: KuavoActuatorCfg

    _joint_vel: torch.Tensor
    _effort_y1: torch.Tensor
    _effort_y2: torch.Tensor
    _velocity_x1: torch.Tensor
    _velocity_x2: torch.Tensor
    _friction_static: torch.Tensor
    _friction_dynamic: torch.Tensor
    _activation_vel: torch.Tensor

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            actuator = create_position_actuator(
                spec,
                target_name,
                stiffness=self.cfg.stiffness,
                damping=self.cfg.damping,
                effort_limit=self.cfg.effort_limit,
                armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,
                transmission_type=self.cfg.transmission_type,
            )
            self._mjs_actuators.append(actuator)

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)

        num_envs = data.nworld
        num_joints = len(self.target_names)
        shape = (num_envs, num_joints)

        y1 = self.cfg.Y1 if self.cfg.Y1 > 0.0 else self.cfg.effort_limit or 0.0

        self._joint_vel = torch.zeros(shape, dtype=torch.float, device=device)
        self._effort_y1 = torch.full(shape, y1, dtype=torch.float, device=device)
        self._effort_y2 = torch.full(
            shape,
            y1 if self.cfg.Y2 is None else self.cfg.Y2,
            dtype=torch.float,
            device=device,
        )
        self._velocity_x1 = torch.full(shape, self.cfg.X1, dtype=torch.float, device=device)
        self._velocity_x2 = torch.full(shape, self.cfg.X2, dtype=torch.float, device=device)
        self._friction_static = torch.full(shape, self.cfg.Fs, dtype=torch.float, device=device)
        self._friction_dynamic = torch.full(shape, self.cfg.Fd, dtype=torch.float, device=device)
        self._activation_vel = torch.full(shape, self.cfg.Va, dtype=torch.float, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        self._joint_vel[:] = cmd.vel

        effort = self.cfg.stiffness * (cmd.position_target - cmd.pos)
        effort += self.cfg.damping * (cmd.velocity_target - cmd.vel)
        effort += cmd.effort_target
        effort = self._clip_effort(effort)

        effort -= (
            self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
            + self._friction_dynamic * cmd.vel
        )

        kp = torch.as_tensor(self.cfg.stiffness, dtype=cmd.pos.dtype, device=cmd.pos.device)
        kd = torch.as_tensor(self.cfg.damping, dtype=cmd.pos.dtype, device=cmd.pos.device)
        kp = torch.clamp(kp, min=1e-6)
        return cmd.pos + (effort + kd * cmd.vel) / kp

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)

        if self.cfg.effort_limit is not None:
            limit = torch.as_tensor(
                self.cfg.effort_limit,
                dtype=max_effort.dtype,
                device=max_effort.device,
            )
            max_effort = torch.minimum(max_effort, limit)

        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        denom = torch.clamp(self._velocity_x2 - self._velocity_x1, min=1e-6)
        k = -max_effort / denom
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@dataclass(kw_only=True)
class KuavoActuatorCfg(BuiltinPositionActuatorCfg):
    """Base configuration for Kuavo biped_s17 actuators."""

    X1: float = 1e9
    """T-N knee speed (rad/s). Default disables speed-dependent derating."""

    X2: float = 1e9
    """No-load speed (rad/s)."""

    Y1: float = 0.0
    """Peak torque, same direction as velocity (N·m). 0 falls back to effort_limit."""

    Y2: float | None = None
    """Peak torque, opposite direction (N·m). None uses Y1."""

    Fs: float = 0.0
    """Static friction (N·m). biped_s17.xml frictionloss=0."""

    Fd: float = 0.0
    """Dynamic friction (N·m·s/rad)."""

    Va: float = 0.01
    """Friction activation velocity scale (rad/s)."""

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> KuavoActuator:
        return KuavoActuator(self, entity, target_ids, target_names)


# --- Per-motor configs (kuavo_v17 MOTORS_TYPE + joint_peak_* limits) ---


@dataclass(kw_only=True)
class KuavoActuatorCfg_PA81_25(KuavoActuatorCfg):
    """PA81_25 — leg_l1/l2/l4, leg_r1/r2/r4."""

    Y1: float = 150.9
    X2: float = 14.62
    armature: float = 0.003


@dataclass(kw_only=True)
class KuavoActuatorCfg_PA76_25(KuavoActuatorCfg):
    """PA76_25 — leg_l3, leg_r3 (leg yaw)."""

    Y1: float = 70.4
    X2: float = 12.93
    armature: float = 0.003


@dataclass(kw_only=True)
class KuavoActuatorCfg_PA76_25_WAIST(KuavoActuatorCfg):
    """PA76_25 — waist_yaw_joint."""

    Y1: float = 80.9
    X2: float = 12.46
    armature: float = 0.003


@dataclass(kw_only=True)
class KuavoActuatorCfg_PA4315_36(KuavoActuatorCfg):
    """PA4315_36 — leg_l5/l6, leg_r5/r6 (ankle pitch/roll)."""

    Y1: float = 74.9
    X2: float = 17.46
    armature: float = 0.003


@dataclass(kw_only=True)
class KuavoActuatorCfg_ruiwoPA60_16(KuavoActuatorCfg):
    """ruiwoPA60_16 — zarm_l1, zarm_r1."""

    Y1: float = 14.67
    X2: float = 10.8
    armature: float = 0.003


@dataclass(kw_only=True)
class KuavoActuatorCfg_ruiwoPA4315_36(KuavoActuatorCfg):
    """ruiwoPA4315_36 — zarm_l2/l3/l4, zarm_r2/r3/r4."""

    Y1: float = 37.9
    X2: float = 15.0
    armature: float = 0.003


@dataclass(kw_only=True)
class KuavoActuatorCfg_ruiwoPA4310_25(KuavoActuatorCfg):
    """ruiwoPA4310_25 — head joints.

    No reliable joint_peak_velocity in kuavo_v17 (placeholder 1800); X2 left at default.
    Set Y1 and effort_limit from biped_s17.xml ctrlrange per joint.
    """

    armature: float = 0.003
