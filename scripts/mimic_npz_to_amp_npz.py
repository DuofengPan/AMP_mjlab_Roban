#!/usr/bin/env python3
"""Convert SOMA mimic NPZ (data + fps) to AMP_mjlab motion NPZ format.

SOMA ``bvh_to_npz_converter.py`` output (mimic / tracking-unified layout):
  - ``data``: float32 (T, nq) — [root_pos(3), root_quat(xyzw)(4), joint_pos(ndof)]
  - ``fps``: scalar

AMP_mjlab ``csv_to_npz.py`` / training loader expects:
  - ``fps``, ``joint_pos``, ``joint_vel``
  - ``body_pos_w``, ``body_quat_w`` (wxyz), ``body_lin_vel_w``, ``body_ang_vel_w``

This script resamples to the target fps, runs MuJoCo FK, and writes the AMP layout.

Example:
  python scripts/mimic_npz_to_amp_npz.py \\
    --input-dir  /path/to/mimic_npz/WalkandRun \\
    --output-dir src/assets/motions/s17/amp/loco/WalkandRun \\
    --input-fps 30 --output-fps 50
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils.amp_npz_io import align_amp_motion_to_ground


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, eps, None)


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    dot = np.sum(q0 * q1, axis=-1)
    flip = dot < 0.0
    if np.any(flip):
        q1 = q1.copy()
        q1[flip] *= -1.0
        dot = np.sum(q0 * q1, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    out = np.zeros_like(q0)
    small = sin_omega < 1e-6
    if np.any(small):
        out[small] = _normalize_quat_wxyz(
            (1.0 - t[small])[:, None] * q0[small] + t[small][:, None] * q1[small]
        )
    if np.any(~small):
        so = sin_omega[~small]
        a = np.sin((1.0 - t[~small]) * omega[~small]) / so
        b = np.sin(t[~small] * omega[~small]) / so
        out[~small] = _normalize_quat_wxyz(a[:, None] * q0[~small] + b[:, None] * q1[~small])
    return out


def _resample_motion(data: np.ndarray, input_fps: float, output_fps: float) -> tuple[np.ndarray, float]:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Expected data shape (T, >=7), got {data.shape}")
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError(f"Invalid fps: input={input_fps}, output={output_fps}")

    input_dt = 1.0 / float(input_fps)
    out_dt = 1.0 / float(output_fps)
    in_frames = int(data.shape[0])
    duration = (in_frames - 1) * input_dt
    if duration <= 0:
        return data[:1].astype(np.float32), out_dt

    times = np.arange(0.0, duration + 0.5 * out_dt, out_dt, dtype=np.float64)
    if times[-1] > duration:
        times = times[times <= duration + 1e-9]
    if times.size == 0:
        times = np.array([0.0], dtype=np.float64)

    phase = times / max(duration, 1e-12)
    idx0 = np.floor(phase * (in_frames - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, in_frames - 1)
    blend = phase * (in_frames - 1) - idx0.astype(np.float64)

    root_pos = (1.0 - blend)[:, None] * data[idx0, 0:3] + blend[:, None] * data[idx1, 0:3]

    q0_xyzw = data[idx0, 3:7]
    q1_xyzw = data[idx1, 3:7]
    q0_wxyz = q0_xyzw[:, [3, 0, 1, 2]]
    q1_wxyz = q1_xyzw[:, [3, 0, 1, 2]]
    q_wxyz = _slerp_wxyz(q0_wxyz, q1_wxyz, blend)
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]

    joint = (1.0 - blend)[:, None] * data[idx0, 7:] + blend[:, None] * data[idx1, 7:]
    out = np.concatenate([root_pos, q_xyzw, joint], axis=-1).astype(np.float32)
    return out, out_dt


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


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape[0] == 1:
        return np.zeros_like(x)
    return np.gradient(x, float(dt), axis=0)


def _load_mujoco_model(mjcf_path: Path):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(mjcf_path))


def _collect_body_ids(model) -> list[int]:
    import mujoco

    body_ids: list[int] = []
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name is None:
            raise ValueError(f"Unnamed body id={bid}")
        body_ids.append(bid)
    return body_ids


def _collect_joint_qpos_indices(model) -> list[int]:
    """Actuated joint qpos indices (excludes free-joint root)."""
    import mujoco

    indices: list[int] = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        adr = int(model.jnt_qposadr[jid])
        indices.append(adr)
    return indices


def _collect_joint_names(model) -> list[str]:
    """Actuated joint names in MuJoCo qpos order (excludes free joint)."""
    import mujoco

    names: list[str] = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            raise ValueError(f"Unnamed joint id={jid}")
        names.append(name)
    return names


def _build_mimic_joint_remap(source_names: list[str], target_names: list[str]) -> np.ndarray:
    """Map mimic NPZ joint columns (source order) to target qpos[7:] order."""
    if len(source_names) != len(target_names):
        raise ValueError(
            f"Joint count mismatch: source={len(source_names)} target={len(target_names)}"
        )
    remap = np.zeros(len(target_names), dtype=np.int64)
    for target_idx, name in enumerate(target_names):
        try:
            source_idx = source_names.index(name)
        except ValueError as exc:
            raise ValueError(f"Target joint {name!r} not found in source MJCF") from exc
        remap[target_idx] = source_idx
    return remap


def _reorder_mimic_joints(joint_cols: np.ndarray, remap: np.ndarray) -> np.ndarray:
    """Reorder mimic joint columns from source MJCF order to target MJCF order."""
    return np.asarray(joint_cols, dtype=np.float64)[:, remap]


def convert_one(
    npz_in: Path,
    npz_out: Path,
    *,
    model,
    data_sim,
    body_ids: list[int],
    joint_qpos_indices: list[int],
    joint_remap: np.ndarray,
    input_fps: float | None,
    output_fps: float,
    skip_existing: bool,
) -> None:
    if skip_existing and npz_out.exists():
        return

    pack = np.load(npz_in, allow_pickle=False)
    if "data" not in pack.files:
        raise ValueError(f"Expected key 'data' in {npz_in}, got {pack.files}")

    data_in = np.asarray(pack["data"], dtype=np.float32)
    fps_in = float(np.asarray(pack["fps"]).reshape(-1)[0]) if "fps" in pack.files else None
    if input_fps is None:
        if fps_in is None:
            raise ValueError(f"No fps in {npz_in}; pass --input-fps")
        input_fps = fps_in

    data_rs, dt = _resample_motion(data_in, input_fps=input_fps, output_fps=output_fps)

    import mujoco

    nq = int(model.nq)
    if data_rs.shape[1] != nq:
        raise ValueError(
            f"{npz_in}: data has {data_rs.shape[1]} columns but model nq={nq}"
        )

    num_frames = int(data_rs.shape[0])
    qpos = np.zeros((num_frames, nq), dtype=np.float64)
    qpos[:, 0:3] = data_rs[:, 0:3]
    q_xyzw = data_rs[:, 3:7]
    qpos[:, 3:7] = q_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = _reorder_mimic_joints(data_rs[:, 7:], joint_remap)

    joint_pos = qpos[:, joint_qpos_indices].astype(np.float32)
    joint_vel = _finite_diff(joint_pos, dt).astype(np.float32)

    nb = len(body_ids)
    body_pos_w = np.zeros((num_frames, nb, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, nb, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((num_frames, nb, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, nb, 3), dtype=np.float32)

    for t in range(num_frames):
        data_sim.qpos[:] = qpos[t]
        data_sim.qvel[:] = 0.0
        mujoco.mj_forward(model, data_sim)
        body_pos_w[t] = np.asarray(data_sim.xpos[body_ids], dtype=np.float32)
        body_quat_w[t] = np.asarray(data_sim.xquat[body_ids], dtype=np.float32)

    body_lin_vel_w = _finite_diff(body_pos_w, dt).astype(np.float32)
    for b in range(nb):
        body_ang_vel_w[:, b, :] = _so3_derivative(body_quat_w[:, b, :], dt).astype(np.float32)

    motion_pack = {
        "fps": np.array([float(output_fps)], dtype=np.float64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    motion_pack, lift = align_amp_motion_to_ground(motion_pack, ground_clearance=0.01)
    if lift > 0.0:
        qpos[:, 2] += lift

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(npz_out), **motion_pack)


def _iter_npz_files(input_dir: Path, exclude_substrings: tuple[str, ...] = ()) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.rglob("*.npz"))
    if exclude_substrings:
        files = [
            path
            for path in files
            if not any(substr in path.name for substr in exclude_substrings)
        ]
    if not files:
        raise FileNotFoundError(f"No .npz files under {input_dir}")
    return files


def _mirror_relative_path(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    return output_root / rel


def _default_source_mjcf(repo_root: Path) -> Path | None:
    candidates = [
        Path.home() / "Projects/leju_soma_retarget/soma_retargeter/configs/biped_s17/xml/biped_s17_mjcf.xml",
        repo_root / "external/leju_soma_retarget/soma_retargeter/configs/biped_s17/xml/biped_s17_mjcf.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SOMA mimic NPZ to AMP_mjlab motion NPZ.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("src/assets/robots/biped_s17/xml/biped_s17.xml"),
        help="Target MuJoCo model used for FK (default: AMP_mjlab biped_s17.xml).",
    )
    parser.add_argument(
        "--source-mjcf",
        type=Path,
        default=None,
        help="Source MuJoCo model describing mimic NPZ joint column order (default: SOMA biped_s17_mjcf.xml).",
    )
    parser.add_argument("--input-fps", type=float, default=None, help="Override input fps.")
    parser.add_argument("--output-fps", type=float, default=50.0, help="Output fps (default: 50).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip files whose name contains this substring (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mjcf_path = args.mjcf if args.mjcf.is_absolute() else (repo_root / args.mjcf).resolve()
    source_mjcf_path = args.source_mjcf
    if source_mjcf_path is None:
        source_mjcf_path = _default_source_mjcf(repo_root)
    elif not source_mjcf_path.is_absolute():
        source_mjcf_path = (repo_root / source_mjcf_path).resolve()
    else:
        source_mjcf_path = source_mjcf_path.resolve()

    if not mjcf_path.is_file():
        raise SystemExit(f"[ERROR] Target MJCF not found: {mjcf_path}")
    if source_mjcf_path is None or not source_mjcf_path.is_file():
        raise SystemExit(
            "[ERROR] Source MJCF not found. Mimic NPZ joint columns follow SOMA "
            "biped_s17_mjcf.xml order; pass --source-mjcf explicitly."
        )

    exclude = tuple(args.exclude)
    files = _iter_npz_files(input_dir, exclude_substrings=exclude)
    if args.max_files > 0:
        files = files[: args.max_files]

    import mujoco

    model = _load_mujoco_model(mjcf_path)
    source_model = _load_mujoco_model(source_mjcf_path)
    data_sim = mujoco.MjData(model)
    body_ids = _collect_body_ids(model)
    joint_qpos_indices = _collect_joint_qpos_indices(model)
    target_joint_names = _collect_joint_names(model)
    source_joint_names = _collect_joint_names(source_model)
    joint_remap = _build_mimic_joint_remap(source_joint_names, target_joint_names)

    print(f"[INFO] Input : {input_dir} ({len(files)} files)")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Target MJCF : {mjcf_path}")
    print(f"[INFO] Source MJCF: {source_mjcf_path}")
    print(f"[INFO] Bodies: {len(body_ids)}, actuated joints: {len(joint_qpos_indices)}")
    if not np.array_equal(joint_remap, np.arange(len(joint_remap))):
        moved = [
            f"{source_joint_names[int(joint_remap[i])]} -> {target_joint_names[i]}"
            for i in range(len(joint_remap))
            if joint_remap[i] != i
        ]
        print(f"[INFO] Joint remap: {len(moved)} columns reordered (mimic/SOMA -> AMP_mjlab)")
    else:
        print("[INFO] Joint order: source matches target (no remap needed)")

    for src in tqdm(files, desc="Converting mimic NPZ -> AMP NPZ"):
        dst = _mirror_relative_path(src, input_dir, output_dir)
        convert_one(
            src,
            dst,
            model=model,
            data_sim=data_sim,
            body_ids=body_ids,
            joint_qpos_indices=joint_qpos_indices,
            joint_remap=joint_remap,
            input_fps=args.input_fps,
            output_fps=float(args.output_fps),
            skip_existing=bool(args.skip_existing),
        )

    print(f"[INFO] Done. Wrote AMP NPZ under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
