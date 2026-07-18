#!/usr/bin/env python3
"""Clip a long Recovery AMP NPZ into reviewable stand->fall->stand segments.

Each output clip:
  - starts in an upright standing pose
  - contains at least one fallen phase in between
  - ends at a settled upright pose (not the leaning frames before the next fall)
  - may overlap neighbouring clips when that keeps transitions smoother
  - optionally repairs single-frame arm joint spikes and refreshes body FK

Example:
  python scripts/utils/clip_recovery_motion.py \\
    --input src/assets/motions/s17/amp/loco/Recovery/fallAndGetUp1_subject1.npz \\
    --output-dir src/assets/motions/s17/amp/loco/Recovery/_clip_review
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_opening

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.amp_npz_io import (
  S17_LEGACY_MOTION_JOINT_COUNT,
  align_amp_motion_to_ground,
  amp_npz_to_qpos,
  as_fps_scalar,
  audit_motion,
  collect_body_names,
  collect_joint_info,
  load_amp_npz,
  load_mujoco_model,
  resolve_path,
  resolve_robot_paths,
)

# Legacy 23-DOF AMP NPZ arm columns (head is last 2 cols; arms are 14-20).
DEFAULT_ARM_JOINT_INDICES = tuple(range(14, 21))


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Apply inverse quaternion rotation (q: wxyz) to vectors v."""
  q_w = q[:, 0]
  q_vec = q[:, 1:4]
  a = v * (2.0 * q_w**2 - 1.0)[:, None]
  b = np.cross(q_vec, v) * (2.0 * q_w)[:, None]
  c = q_vec * (2.0 * np.sum(q_vec * v, axis=1))[:, None]
  return a - b + c


def standing_mask(
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  *,
  base_index: int = 0,
  height_min: float = 0.88,
  upright_xy_max: float = 0.12,
  morph_frames: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Return (standing_smoothed, standing_raw, z, upright_xy)."""
  z = body_pos_w[:, base_index, 2]
  gravity_w = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (z.shape[0], 1))
  gravity_b = quat_apply_inverse(body_quat_w[:, base_index, :], gravity_w)
  upright_xy = gravity_b[:, 0] ** 2 + gravity_b[:, 1] ** 2
  stand_raw = (z > height_min) & (upright_xy < upright_xy_max)
  structure = np.ones(max(1, morph_frames), dtype=bool)
  stand_smooth = binary_opening(binary_closing(stand_raw, structure), structure)
  return stand_smooth, stand_raw, z, upright_xy


def fallen_mask(
  z: np.ndarray,
  upright_xy: np.ndarray,
  *,
  height_max: float = 0.65,
  upright_xy_min: float = 0.35,
) -> np.ndarray:
  return (z < height_max) | (upright_xy > upright_xy_min)


def find_standing_runs(
  stand_smooth: np.ndarray,
  *,
  min_frames: int,
) -> list[tuple[int, int]]:
  """Return half-open intervals [start, end) where the robot is standing."""
  changes = np.diff(stand_smooth.astype(np.int8))
  starts = list(np.where(changes == 1)[0] + 1)
  ends = list(np.where(changes == -1)[0] + 1)
  if stand_smooth[0]:
    starts = [0, *starts]
  if stand_smooth[-1]:
    ends = [*ends, stand_smooth.shape[0]]
  return [(s, e) for s, e in zip(starts, ends) if e - s >= min_frames]


def pick_settled_end_frame(
  run: tuple[int, int],
  z: np.ndarray,
  upright_xy: np.ndarray,
  *,
  fps: float,
  settle_s: float = 0.4,
  strict_height: float = 0.92,
  strict_upright_xy: float = 0.06,
) -> int:
  """Pick an end frame inside the final standing run, before pre-fall leaning."""
  start, end = run
  settle_frames = max(2, int(round(settle_s * fps)))
  for frame in range(end - 1, start + settle_frames - 2, -1):
    window = slice(frame - settle_frames + 1, frame + 1)
    if (
      np.all(z[window] > strict_height)
      and np.all(upright_xy[window] < strict_upright_xy)
    ):
      return frame + 1

  # Fallback: best settled score in the inner 80% of the standing run.
  inner_start = start + max(1, int(0.1 * (end - start)))
  inner_end = end - max(1, int(0.1 * (end - start)))
  if inner_end <= inner_start:
    inner_start, inner_end = start, end
  candidates = np.arange(inner_start, inner_end)
  scores = z[candidates] - 0.5 * upright_xy[candidates]
  return int(candidates[int(np.argmax(scores))] + 1)


def find_recovery_clips(
  runs: list[tuple[int, int]],
  fall: np.ndarray,
  stand_raw: np.ndarray,
  z: np.ndarray,
  upright_xy: np.ndarray,
  *,
  fps: float,
) -> list[tuple[int, int]]:
  """Build stand->fall->stand clips; consecutive runs may overlap."""
  clips: list[tuple[int, int]] = []
  for idx in range(len(runs) - 1):
    start = runs[idx][0]
    mid_start = runs[idx][1]
    mid_end = runs[idx + 1][0]
    if mid_end <= mid_start or not fall[mid_start:mid_end].any():
      continue

    end = pick_settled_end_frame(runs[idx + 1], z, upright_xy, fps=fps)
    if not (stand_raw[start] and stand_raw[end - 1] and end > start + 2):
      continue
    clips.append((start, end))
  return clips


def repair_joint_spikes(
  joint_pos: np.ndarray,
  *,
  threshold: float = 0.12,
  pad: int = 1,
  max_passes: int = 8,
  joint_indices: tuple[int, ...] = DEFAULT_ARM_JOINT_INDICES,
) -> tuple[np.ndarray, list[int]]:
  """Interpolate arm joint glitches inside short spike segments only.

  Unlike per-frame blending, this keeps clean frames untouched and anchors each
  repaired segment to stable poses immediately before/after the glitch.
  """
  repaired = np.asarray(joint_pos, dtype=np.float64).copy()
  num_frames, num_joints = repaired.shape
  arm_indices = [int(j) for j in joint_indices if 0 <= j < num_joints]
  if not arm_indices:
    return repaired.astype(np.float32), []

  repaired_frames: set[int] = set()
  for _ in range(max_passes):
    dq = np.abs(np.diff(repaired, axis=0))
    per_joint_bad = dq[:, arm_indices] > threshold
    if not per_joint_bad.any():
      break

    frame_bad = per_joint_bad.any(axis=1)
    diff_index = 0
    while diff_index < frame_bad.shape[0]:
      if not frame_bad[diff_index]:
        diff_index += 1
        continue

      run_start = diff_index
      while diff_index < frame_bad.shape[0] and frame_bad[diff_index]:
        diff_index += 1
      run_end = diff_index - 1

      seg_start = max(0, run_start - pad)
      seg_end = min(num_frames - 1, run_end + 1 + pad)
      anchor_left = seg_start - 1
      anchor_right = seg_end + 1
      if anchor_left < 0 or anchor_right >= num_frames:
        continue

      spiking_joints: set[int] = set()
      for bad_diff in range(run_start, run_end + 1):
        for local_idx, joint_idx in enumerate(arm_indices):
          if per_joint_bad[bad_diff, local_idx]:
            spiking_joints.add(joint_idx)

      for frame_idx in range(seg_start, seg_end + 1):
        alpha = (frame_idx - anchor_left) / float(anchor_right - anchor_left)
        for joint_idx in spiking_joints:
          repaired[frame_idx, joint_idx] = (
            (1.0 - alpha) * repaired[anchor_left, joint_idx]
            + alpha * repaired[anchor_right, joint_idx]
          )
          repaired_frames.add(frame_idx)

  return repaired.astype(np.float32), sorted(repaired_frames)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  q = np.asarray(quat, dtype=np.float64)
  norm = np.linalg.norm(q, axis=-1, keepdims=True)
  norm = np.maximum(norm, 1e-12)
  return q / norm


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
  out = np.asarray(quat, dtype=np.float64).copy()
  out[..., 1:] *= -1.0
  return out


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = left.T
  rw, rx, ry, rz = right.T
  return np.stack(
    [
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ],
    axis=-1,
  )


def _so3_derivative(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  """Central-difference angular velocity from wxyz quaternions. Returns (T, 3)."""
  q = _normalize_quat_wxyz(quat_wxyz)
  if q.shape[0] == 1:
    return np.zeros((1, 3), dtype=np.float64)
  q_prev, q_next = q[:-2], q[2:]
  q_rel = _quat_mul_wxyz(q_next, _quat_conj_wxyz(q_prev))
  q_rel = _normalize_quat_wxyz(q_rel)
  w = np.clip(q_rel[:, 0], -1.0, 1.0)
  angle = 2.0 * np.arccos(w)
  s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
  axis = np.zeros((q_rel.shape[0], 3), dtype=np.float64)
  mask = s > 1e-8
  axis[mask] = q_rel[mask, 1:] / s[mask][:, None]
  omega = axis * (angle[:, None] / (2.0 * float(dt)))
  return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def refresh_body_kinematics_from_fk(
  data: dict[str, np.ndarray],
  model,
) -> dict[str, np.ndarray]:
  """Recompute body_* fields from repaired joint_pos via MuJoCo FK."""
  import mujoco

  qpos, fps = amp_npz_to_qpos(data, model.nq)
  num_frames = qpos.shape[0]
  body_count = int(data["body_pos_w"].shape[1])
  body_ids = list(range(1, min(model.nbody, body_count + 1)))

  body_pos = np.zeros((num_frames, body_count, 3), dtype=np.float64)
  body_quat = np.zeros((num_frames, body_count, 4), dtype=np.float64)
  sim = mujoco.MjData(model)
  for frame_idx in range(num_frames):
    sim.qpos[:] = qpos[frame_idx]
    mujoco.mj_forward(model, sim)
    fk_pos = np.asarray(sim.xpos[body_ids], dtype=np.float64)
    fk_quat = np.asarray(sim.xquat[body_ids], dtype=np.float64)
    body_pos[frame_idx, : fk_pos.shape[0]] = fk_pos
    body_quat[frame_idx, : fk_quat.shape[0]] = fk_quat

  dt = 1.0 / fps
  out = dict(data)
  out["body_pos_w"] = body_pos.astype(np.float32)
  out["body_quat_w"] = body_quat.astype(np.float32)
  out["body_lin_vel_w"] = np.gradient(body_pos, dt, axis=0).astype(np.float32)
  body_ang_vel = np.zeros((num_frames, body_count, 3), dtype=np.float32)
  for body_idx in range(body_count):
    body_ang_vel[:, body_idx, :] = _so3_derivative(body_quat[:, body_idx, :], dt).astype(
      np.float32
    )
  out["body_ang_vel_w"] = body_ang_vel
  return out


def slice_motion(data: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
  """Slice AMP motion arrays on [start, end)."""
  sl = slice(start, end)
  out = {}
  for key, value in data.items():
    if key == "fps":
      out[key] = np.asarray(value).copy()
      continue
    out[key] = np.asarray(value)[sl].copy()
  fps = as_fps_scalar(data["fps"])
  dt = 1.0 / fps
  joint_pos = np.asarray(out["joint_pos"], dtype=np.float64)
  body_pos_w = np.asarray(out["body_pos_w"], dtype=np.float64)
  out["joint_vel"] = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
  out["body_lin_vel_w"] = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
  return out


@dataclass
class ClipInfo:
  index: int
  start_frame: int
  end_frame: int
  num_frames: int
  duration_s: float
  ground_lift_m: float
  min_body_z_before: float
  min_body_z_after: float
  end_base_z: float
  end_upright_xy: float
  repaired_joint_frames: int
  output_file: str
  audit_ok: bool
  audit_issues: list[str]
  fk_body_rmse_mm: float | None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Clip Recovery AMP NPZ into review segments.")
  parser.add_argument("--input", type=Path, required=True, help="Source Recovery NPZ.")
  parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
  parser.add_argument("--min-stand-s", type=float, default=1.0, help="Min standing duration.")
  parser.add_argument("--ground-clearance", type=float, default=0.01, help="Min body z after align.")
  parser.add_argument("--settle-s", type=float, default=0.4, help="Settled-tail duration before cut.")
  parser.add_argument(
    "--spike-threshold",
    type=float,
    default=0.12,
    help="Per-frame arm joint jump threshold for spike repair (rad).",
  )
  parser.add_argument(
    "--no-repair-joint-spikes",
    action="store_true",
    help="Disable arm joint spike repair + FK refresh.",
  )
  parser.add_argument("--robot", choices=("s17",), default="s17")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  input_path = resolve_path(args.input)
  output_dir = resolve_path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  data = load_amp_npz(input_path)
  fps = as_fps_scalar(data["fps"])
  min_stand_frames = max(1, int(round(args.min_stand_s * fps)))

  stand_smooth, stand_raw, z, upright_xy = standing_mask(
    np.asarray(data["body_pos_w"]),
    np.asarray(data["body_quat_w"]),
  )
  fall = fallen_mask(z, upright_xy)
  runs = find_standing_runs(stand_smooth, min_frames=min_stand_frames)
  clips = find_recovery_clips(
    runs,
    fall,
    stand_raw,
    z,
    upright_xy,
    fps=fps,
  )

  if not clips:
    raise RuntimeError("No recovery clips found. Try relaxing thresholds.")

  profile, _, mjcf_path, _ = resolve_robot_paths(args.robot)
  model = load_mujoco_model(mjcf_path)
  joint_info = collect_joint_info(model)
  body_names = collect_body_names(model)

  stem = input_path.stem
  manifest: list[dict] = []

  print(f"[clip_recovery_motion] source={input_path}")
  print(f"[clip_recovery_motion] standing runs (>={args.min_stand_s:.1f}s): {len(runs)}")
  print(f"[clip_recovery_motion] clips (overlap allowed): {len(clips)}")

  for index, (start, end) in enumerate(clips):
    sliced = slice_motion(data, start, end)
    repaired_frames: list[int] = []
    if not args.no_repair_joint_spikes:
      joint_pos = np.asarray(sliced["joint_pos"], dtype=np.float64)
      if joint_pos.shape[1] == S17_LEGACY_MOTION_JOINT_COUNT:
        repaired_pos, repaired_frames = repair_joint_spikes(
          joint_pos,
          threshold=args.spike_threshold,
        )
        sliced["joint_pos"] = repaired_pos
        fps_local = as_fps_scalar(sliced["fps"])
        sliced["joint_vel"] = np.gradient(repaired_pos, 1.0 / fps_local, axis=0).astype(
          np.float32
        )
        sliced = refresh_body_kinematics_from_fk(sliced, model)

    min_before = float(np.min(sliced["body_pos_w"][..., 2]))
    aligned, lift = align_amp_motion_to_ground(
      sliced,
      ground_clearance=args.ground_clearance,
    )
    min_after = float(np.min(aligned["body_pos_w"][..., 2]))
    end_base_z = float(aligned["body_pos_w"][-1, 0, 2])
    end_gravity_b = quat_apply_inverse(
      aligned["body_quat_w"][-1:, 0, :],
      np.array([[0.0, 0.0, -1.0]]),
    )[0]
    end_upright_xy = float(end_gravity_b[0] ** 2 + end_gravity_b[1] ** 2)

    out_name = f"{stem}__clip{index:02d}__f{start:05d}-{end:05d}.npz"
    out_path = output_dir / out_name
    np.savez_compressed(out_path, **aligned)

    audit = audit_motion(
      out_path,
      aligned,
      model=model,
      joint_info=joint_info,
      body_names=body_names,
      robot_profile=profile,
    )
    info = ClipInfo(
      index=index,
      start_frame=int(start),
      end_frame=int(end),
      num_frames=int(end - start),
      duration_s=float((end - start) / fps),
      ground_lift_m=float(lift),
      min_body_z_before=float(min_before),
      min_body_z_after=float(min_after),
      end_base_z=end_base_z,
      end_upright_xy=end_upright_xy,
      repaired_joint_frames=len(repaired_frames),
      output_file=str(out_path),
      audit_ok=audit.ok,
      audit_issues=list(audit.issues),
      fk_body_rmse_mm=(audit.fk_body_rmse * 1000.0 if audit.fk_body_rmse is not None else None),
    )
    manifest.append(asdict(info))

    status = "OK" if audit.ok else "WARN"
    print(
      f"  [{status}] clip{index:02d}: frames {start}-{end} "
      f"({info.duration_s:.2f}s) end_z={end_base_z:.3f} end_upright={end_upright_xy:.3f} "
      f"repair_frames={len(repaired_frames)} -> {out_name}"
    )
    if audit.issues:
      for issue in audit.issues:
        print(f"         issue: {issue}")

  manifest_path = output_dir / f"{stem}__clip_manifest.json"
  manifest_path.write_text(
    json.dumps(
      {
        "source": str(input_path),
        "fps": fps,
        "num_clips": len(clips),
        "overlap_allowed": True,
        "settle_s": args.settle_s,
        "clips": manifest,
      },
      indent=2,
      ensure_ascii=False,
    ),
    encoding="utf-8",
  )
  print(f"[clip_recovery_motion] manifest: {manifest_path}")
  print(
    "[clip_recovery_motion] review playback:\n"
    f"  python scripts/utils/visualize_amp_npz_sequence.py "
    f"--robot s17 --motion-root {output_dir} --audit --root-frame center"
  )


if __name__ == "__main__":
  main()
