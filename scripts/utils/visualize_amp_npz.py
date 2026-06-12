#!/usr/bin/env python3
"""Visualize and inspect one AMP motion NPZ file.

Modes (combine as needed):
  --report          Print format/audit summary (default if no viewer flags)
  --plot            Matplotlib time-series smoothness plots
  --mujoco-robot    Replay robot mesh via qpos reconstructed from AMP NPZ
  --mujoco-bodies   Replay body_pos_w as MuJoCo mocap markers

Examples:
  python scripts/utils/visualize_amp_npz.py --robot s17 --npz src/assets/motions/s17/amp/WalkandRun/walk1_subject1__walk_forward_loop_001__f02641-02940.npz --report
  python scripts/utils/visualize_amp_npz.py --robot g1 --npz-dir src/assets/motions/g1/amp/WalkandRun --seed 0 --mujoco-robot --loop
  python scripts/utils/visualize_amp_npz.py --npz ... --plot --plot-body-components --body-index 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.amp_npz_io import (
    amp_npz_to_qpos,
    apply_root_frame_for_vis,
    as_fps_scalar,
    audit_motion,
    collect_body_names,
    collect_joint_info,
    load_amp_npz,
    load_mujoco_model,
    mujoco_play_qpos_trajectory,
    pick_npz,
    print_audit,
    print_root_trajectory_stats,
    resolve_robot_paths,
    smoothness_stats,
    validate_format,
)


def plot_timeseries(
    data: dict[str, np.ndarray],
    *,
    max_frames: int,
    stride: int,
    body_index: int,
    plot_body_components: bool,
) -> None:
    import matplotlib.pyplot as plt

    fps = as_fps_scalar(data["fps"])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

    num_frames = joint_pos.shape[0]
    limit = min(num_frames, max_frames) if max_frames > 0 else num_frames
    idx = np.arange(0, limit, max(1, stride))
    times = idx / fps

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(times, np.linalg.norm(joint_pos[idx], axis=1))
    axes[0].set_ylabel("||joint_pos||")
    axes[1].plot(times, np.linalg.norm(joint_vel[idx], axis=1))
    axes[1].set_ylabel("||joint_vel||")
    axes[2].plot(times, np.linalg.norm(body_lin_vel_w[idx], axis=2).mean(axis=1))
    axes[2].set_ylabel("mean ||body_lin_vel||")
    axes[3].plot(times, np.linalg.norm(body_ang_vel_w[idx], axis=2).mean(axis=1))
    axes[3].set_ylabel("mean ||body_ang_vel||")
    axes[4].plot(times, body_pos_w[idx, :, 2].mean(axis=1))
    axes[4].set_ylabel("mean body z")
    axes[4].set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()

    if plot_body_components:
        lin_vel = body_lin_vel_w[idx, body_index]
        ang_vel = body_ang_vel_w[idx, body_index]
        fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for axis_id, label in enumerate("xyz"):
            axes2[0].plot(times, lin_vel[:, axis_id], label=label)
        axes2[0].plot(times, np.linalg.norm(lin_vel, axis=1), label="|v|", linewidth=2.0, alpha=0.7)
        axes2[0].legend(loc="upper right")
        axes2[0].set_ylabel(f"body_lin_vel (body {body_index})")
        for axis_id, label in enumerate("xyz"):
            axes2[1].plot(times, ang_vel[:, axis_id], label=label)
        axes2[1].plot(times, np.linalg.norm(ang_vel, axis=1), label="|w|", linewidth=2.0, alpha=0.7)
        axes2[1].legend(loc="upper right")
        axes2[1].set_ylabel(f"body_ang_vel (body {body_index})")
        axes2[1].set_xlabel("time (s)")
        fig2.tight_layout()
        plt.show()


def mujoco_visualize_bodies(
    data: dict[str, np.ndarray],
    *,
    max_frames: int,
    stride: int,
    loop: bool,
    speed: float,
) -> None:
    import mujoco
    from mujoco import viewer

    fps = as_fps_scalar(data["fps"])
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
    num_frames, num_bodies, _ = body_pos_w.shape

    limit = min(num_frames, max_frames) if max_frames > 0 else num_frames
    stride = max(1, stride)
    frame_ids = list(range(0, limit, stride))
    if not frame_ids:
        raise SystemExit("[ERROR] no frames to visualize")

    bodies_xml = []
    for body_idx in range(num_bodies):
        bodies_xml.append(
            "\n".join(
                [
                    f'    <body name="b{body_idx}" mocap="true">',
                    '      <geom type="sphere" size="0.02" rgba="0.2 0.8 1.0 0.9" contype="0" conaffinity="0"/>',
                    "    </body>",
                ]
            )
        )
    xml = "\n".join(
        [
            '<mujoco model="amp_body_markers">',
            '  <option gravity="0 0 -9.81"/>',
            "  <worldbody>",
            '    <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>',
            '    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>',
            "  " + "\n  ".join(bodies_xml),
            "  </worldbody>",
            "</mujoco>",
        ]
    )

    model = mujoco.MjModel.from_xml_string(xml)
    sim = mujoco.MjData(model)
    mocap_ids = np.zeros((num_bodies,), dtype=np.int32)
    for body_idx in range(num_bodies):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"b{body_idx}")
        mocap_ids[body_idx] = int(model.body_mocapid[body_id])

    dt = (1.0 / fps) / max(1e-6, speed)
    with viewer.launch_passive(model, sim) as v:
        while v.is_running():
            for frame_idx in frame_ids:
                for body_idx in range(num_bodies):
                    mocap_id = mocap_ids[body_idx]
                    sim.mocap_pos[mocap_id] = body_pos_w[frame_idx, body_idx]
                    sim.mocap_quat[mocap_id] = body_quat_w[frame_idx, body_idx]
                mujoco.mj_forward(model, sim)
                v.sync()
                time.sleep(dt)
                if not v.is_running():
                    break
            if not loop:
                break


def mujoco_visualize_robot(
    data: dict[str, np.ndarray],
    mjcf_path: Path,
    *,
    loop: bool,
    speed: float,
    stride: int,
    root_frame: str,
) -> None:
    model = load_mujoco_model(mjcf_path)
    qpos_traj, fps = amp_npz_to_qpos(data, model.nq)
    print(f"[INFO] MJCF: {mjcf_path}")
    print(f"[INFO] qpos trajectory: {qpos_traj.shape[0]} frames @ {fps:.2f} fps")
    print_root_trajectory_stats(qpos_traj)
    qpos_traj = apply_root_frame_for_vis(qpos_traj, root_frame)
    if root_frame != "full":
        print(f"[INFO] root frame mode: {root_frame} (visualization only)")

    mujoco_play_qpos_trajectory(
        model,
        qpos_traj,
        fps,
        stride=stride,
        speed=speed,
        loop=loop,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one AMP motion NPZ.")
    parser.add_argument(
        "--robot",
        choices=("s17", "g1"),
        default="s17",
        help="Robot preset for MJCF, body names, and default motion root.",
    )
    parser.add_argument("--npz", type=Path, default=None, help="Path to one AMP NPZ.")
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Directory to pick from when --npz is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic file pick index.")
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="MJCF for --mujoco-robot (default: robot scene with floor).",
    )
    parser.add_argument(
        "--root-frame",
        choices=("center", "inplace", "full"),
        default="center",
        help="Reframe root XY for viewing. Motions are often far from world origin (default: center).",
    )
    parser.add_argument("--max-frames", type=int, default=2000, help="Limit frames for plot/viewer (-1=all).")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--report", action="store_true", help="Print audit report.")
    parser.add_argument("--plot", action="store_true", help="Plot smoothness time series.")
    parser.add_argument("--plot-body-components", action="store_true", help="Also plot per-body velocity components.")
    parser.add_argument("--body-index", type=int, default=0, help="Body index for component plots.")
    parser.add_argument("--mujoco-bodies", action="store_true", help="Visualize body_pos_w as mocap markers.")
    parser.add_argument("--mujoco-robot", action="store_true", help="Replay robot mesh from reconstructed qpos.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile, motion_root, audit_mjcf_path, vis_mjcf_path = resolve_robot_paths(
        args.robot,
        motion_root=args.npz_dir,
        vis_mjcf=args.mjcf,
    )
    path = pick_npz(args.npz, motion_root, seed=int(args.seed))
    data = load_amp_npz(path)
    num_frames, ndof, nbodies, fps = validate_format(data)

    audit_model = load_mujoco_model(audit_mjcf_path)
    joint_info = collect_joint_info(audit_model)
    body_names = collect_body_names(audit_model)

    print(f"[INFO] robot: {profile.name}")
    print(f"[INFO] file: {path}")
    print(f"[INFO] fps={fps:.2f} T={num_frames} ndof={ndof} nbodies={nbodies}")
    print("[INFO] smoothness:")
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        stats = smoothness_stats(np.asarray(data[key]))
        print(
            f"  - {key}: d1_rms={stats['d1_rms']:.4g} d2_rms={stats['d2_rms']:.4g} "
            f"d1_max={stats['d1_max']:.4g} d2_max={stats['d2_max']:.4g}"
        )

    do_report = args.report or not (
        args.plot or args.mujoco_bodies or args.mujoco_robot
    )
    if do_report:
        audit = audit_motion(
            path,
            data,
            model=audit_model,
            joint_info=joint_info,
            body_names=body_names,
            robot_profile=profile,
        )
        print()
        print_audit(audit)

    if args.plot:
        plot_timeseries(
            data,
            max_frames=int(args.max_frames),
            stride=int(args.stride),
            body_index=int(args.body_index),
            plot_body_components=bool(args.plot_body_components),
        )
    if args.mujoco_bodies:
        mujoco_visualize_bodies(
            data,
            max_frames=int(args.max_frames),
            stride=int(args.stride),
            loop=bool(args.loop),
            speed=float(args.speed),
        )
    if args.mujoco_robot:
        mujoco_visualize_robot(
            data,
            vis_mjcf_path,
            loop=bool(args.loop),
            speed=float(args.speed),
            stride=int(args.stride),
            root_frame=str(args.root_frame),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
