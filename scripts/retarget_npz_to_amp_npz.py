#!/usr/bin/env python3
"""Convert retarget gait NPZ to AMP_mjlab motion NPZ format.

Retarget layout (e.g. ``amp_gait/*_retarget.npz``):
  - ``fps``, ``dof_names``, ``body_names``
  - ``dof_positions``, ``dof_velocities``
  - ``body_positions``, ``body_rotations`` (wxyz)
  - ``body_linear_velocities``, ``body_angular_velocities``

AMP_mjlab training expects:
  - ``fps``, ``joint_pos``, ``joint_vel``
  - ``body_pos_w``, ``body_quat_w`` (wxyz), ``body_lin_vel_w``, ``body_ang_vel_w``

Head joints (``zhead_1_joint``, ``zhead_2_joint``) are locked to zero in the output.

Example:
  python scripts/retarget_npz_to_amp_npz.py \\
    --input-dir  src/assets/motions/s17/amp_gait \\
    --output-dir src/assets/motions/s17/amp/FlatWalk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from mimic_npz_to_amp_npz import (
    _collect_body_ids,
    _collect_joint_names,
    _collect_joint_qpos_indices,
    _finite_diff,
    _load_mujoco_model,
    _so3_derivative,
)

REQUIRED_RETARGET_KEYS = (
    "fps",
    "dof_names",
    "dof_positions",
    "dof_velocities",
    "body_positions",
    "body_rotations",
)


def _as_fps_scalar(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(()))
    if arr.ndim == 1 and arr.shape[0] == 1:
        return float(arr[0])
    raise ValueError(f"fps must be scalar-ish, got shape {arr.shape}")


def _validate_retarget(data: dict[str, np.ndarray]) -> tuple[int, int, int, float]:
    missing = [key for key in REQUIRED_RETARGET_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing keys: {missing}. Present: {sorted(data.keys())}")

    fps = _as_fps_scalar(data["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid fps={fps}")

    dof_pos = np.asarray(data["dof_positions"])
    dof_vel = np.asarray(data["dof_velocities"])
    body_pos = np.asarray(data["body_positions"])
    body_rot = np.asarray(data["body_rotations"])

    if dof_pos.ndim != 2:
        raise ValueError(f"dof_positions must be (T, ndof), got {dof_pos.shape}")
    if dof_vel.shape != dof_pos.shape:
        raise ValueError(f"dof_velocities shape {dof_vel.shape} != dof_positions {dof_pos.shape}")
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"body_positions must be (T, nb, 3), got {body_pos.shape}")
    if body_rot.shape[:2] != body_pos.shape[:2] or body_rot.shape[-1] != 4:
        raise ValueError(f"body_rotations shape mismatch: {body_rot.shape} vs {body_pos.shape}")

    num_frames = int(dof_pos.shape[0])
    if num_frames <= 0:
        raise ValueError("Empty motion (T=0)")

    for name, arr in (
        ("dof_positions", dof_pos),
        ("dof_velocities", dof_vel),
        ("body_positions", body_pos),
        ("body_rotations", body_rot),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")

    return num_frames, int(dof_pos.shape[1]), int(body_pos.shape[1]), fps


def convert_one(
    npz_in: Path,
    npz_out: Path,
    *,
    model,
    data_sim,
    body_ids: list[int],
    joint_qpos_indices: list[int],
    target_joint_names: list[str],
    lock_joint_values: dict[str, float],
    output_fps: float | None,
    skip_existing: bool,
) -> None:
    if skip_existing and npz_out.exists():
        return

    pack = np.load(npz_in, allow_pickle=True)
    data = {key: pack[key] for key in pack.files}
    num_frames, _ndof, _nbodies, fps_in = _validate_retarget(data)

    fps_out = float(output_fps) if output_fps is not None else fps_in
    if fps_out <= 0:
        raise ValueError(f"Invalid output fps={fps_out}")

    import mujoco

    dof_names = [str(x) for x in np.asarray(data["dof_names"]).reshape(-1)]
    body_names = [str(x) for x in np.asarray(data["body_names"]).reshape(-1)]
    if "base_link" not in body_names:
        raise ValueError(f"{npz_in}: base_link not found in body_names")

    base_idx = body_names.index("base_link")
    unknown_joints = [name for name in dof_names if name not in target_joint_names]
    if unknown_joints:
        raise ValueError(f"{npz_in}: unknown joints for target MJCF: {unknown_joints}")

    nq = int(model.nq)
    qpos = np.zeros((num_frames, nq), dtype=np.float64)
    qpos[:, 0:3] = np.asarray(data["body_positions"][:, base_idx], dtype=np.float64)
    qpos[:, 3:7] = np.asarray(data["body_rotations"][:, base_idx], dtype=np.float64)

    for ji, joint_name in enumerate(dof_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"{npz_in}: joint {joint_name!r} not found in MJCF")
        adr = int(model.jnt_qposadr[jid])
        qpos[:, adr] = np.asarray(data["dof_positions"][:, ji], dtype=np.float64)

    for joint_name, value in lock_joint_values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Lock joint {joint_name!r} not found in MJCF")
        adr = int(model.jnt_qposadr[jid])
        qpos[:, adr] = float(value)

    if abs(fps_out - fps_in) > 1e-6:
        raise ValueError(
            f"{npz_in}: resampling not implemented (input fps={fps_in}, output fps={fps_out})"
        )

    dt = 1.0 / fps_out
    joint_pos = qpos[:, joint_qpos_indices].astype(np.float32)
    joint_vel = _finite_diff(joint_pos, dt).astype(np.float32)

    nb = len(body_ids)
    body_pos_w = np.zeros((num_frames, nb, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, nb, 4), dtype=np.float32)

    for t in range(num_frames):
        data_sim.qpos[:] = qpos[t]
        data_sim.qvel[:] = 0.0
        mujoco.mj_forward(model, data_sim)
        body_pos_w[t] = np.asarray(data_sim.xpos[body_ids], dtype=np.float32)
        body_quat_w[t] = np.asarray(data_sim.xquat[body_ids], dtype=np.float32)

    body_lin_vel_w = _finite_diff(body_pos_w, dt).astype(np.float32)
    body_ang_vel_w = np.zeros((num_frames, nb, 3), dtype=np.float32)
    for b in range(nb):
        body_ang_vel_w[:, b, :] = _so3_derivative(body_quat_w[:, b, :], dt).astype(np.float32)

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(npz_out),
        fps=np.array([fps_out], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


def _iter_npz_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {input_dir}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert retarget gait NPZ to AMP_mjlab format.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("src/assets/robots/biped_s17/xml/biped_s17.xml"),
        help="Target MuJoCo model used for FK (default: biped_s17.xml).",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Output fps (default: keep input fps). Resampling is not supported yet.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-files", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mjcf_path = args.mjcf if args.mjcf.is_absolute() else (repo_root / args.mjcf).resolve()

    if not mjcf_path.is_file():
        raise SystemExit(f"[ERROR] MJCF not found: {mjcf_path}")

    files = _iter_npz_files(input_dir)
    if args.max_files > 0:
        files = files[: args.max_files]

    import mujoco

    model = _load_mujoco_model(mjcf_path)
    data_sim = mujoco.MjData(model)
    body_ids = _collect_body_ids(model)
    joint_qpos_indices = _collect_joint_qpos_indices(model)
    target_joint_names = _collect_joint_names(model)
    lock_joint_values = {"zhead_1_joint": 0.0, "zhead_2_joint": 0.0}

    print(f"[INFO] Input : {input_dir} ({len(files)} files)")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] MJCF  : {mjcf_path}")
    print(f"[INFO] Bodies: {len(body_ids)}, actuated joints: {len(joint_qpos_indices)}")
    print(f"[INFO] Locked joints: {lock_joint_values}")

    for npz_in in tqdm(files, desc="Convert retarget -> AMP"):
        npz_out = output_dir / npz_in.name
        convert_one(
            npz_in,
            npz_out,
            model=model,
            data_sim=data_sim,
            body_ids=body_ids,
            joint_qpos_indices=joint_qpos_indices,
            target_joint_names=target_joint_names,
            lock_joint_values=lock_joint_values,
            output_fps=args.output_fps,
            skip_existing=args.skip_existing,
        )

    print(f"[OK] Wrote {len(files)} AMP motion files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
