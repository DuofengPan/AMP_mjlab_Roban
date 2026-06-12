#!/usr/bin/env python3
"""Sequentially visualize AMP NPZ clips in MuJoCo (WalkandRun + Recovery).

Uses scene.xml (floor + lighting) by default and reframes root XY so clips are
visible near the origin. Motions from retargeting often have world coordinates
many meters from (0,0); the default camera targets the origin.

Example:
  python scripts/utils/visualize_amp_npz_sequence.py
  python scripts/utils/visualize_amp_npz_sequence.py --robot s17 --motion-root src/assets/motions/s17/amp/WalkandRun
  python scripts/utils/visualize_amp_npz_sequence.py --robot g1 --motion-root src/assets/motions/g1/amp/WalkandRun
  python scripts/utils/visualize_amp_npz_sequence.py --root-frame inplace --loop-files
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.amp_npz_io import (
    amp_npz_to_qpos,
    apply_root_frame_for_vis,
    as_fps_scalar,
    audit_motion,
    collect_body_names,
    collect_joint_info,
    iter_amp_npz_files,
    load_amp_npz,
    load_mujoco_model,
    mujoco_play_qpos_trajectory,
    print_audit,
    print_root_trajectory_stats,
    resolve_robot_paths,
)
from scripts.utils.visualize_amp_npz import mujoco_visualize_bodies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequentially visualize AMP NPZ clips.")
    parser.add_argument(
        "--robot",
        choices=("s17", "g1"),
        default="s17",
        help="Robot preset for MJCF, body names, and default motion root.",
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=None,
        help="Directory containing NPZ files (searched recursively).",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="MuJoCo scene for robot playback (default: robot scene.xml with floor).",
    )
    parser.add_argument(
        "--mode",
        choices=("robot", "bodies"),
        default="robot",
        help="robot: replay mesh from qpos; bodies: mocap markers from body_pos_w.",
    )
    parser.add_argument(
        "--root-frame",
        choices=("center", "inplace", "full"),
        default="center",
        help="Reframe root XY for viewing (default: center first frame at origin).",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start from this sorted file index.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = all).")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride during playback.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop-files", action="store_true", help="Loop over the file list.")
    parser.add_argument("--loop-motion", action="store_true", help="Loop each clip until advancing manually.")
    parser.add_argument("--audit", action="store_true", help="Print audit summary before each clip.")
    parser.add_argument("--pause-between", type=float, default=0.5, help="Pause between clips (seconds).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile, motion_root, audit_mjcf_path, vis_mjcf_path = resolve_robot_paths(
        args.robot,
        motion_root=args.motion_root,
        vis_mjcf=args.mjcf,
    )

    files = iter_amp_npz_files(motion_root)
    start = max(0, int(args.start_index))
    files = files[start:]
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]

    audit_model = load_mujoco_model(audit_mjcf_path)
    joint_info = collect_joint_info(audit_model)
    body_names = collect_body_names(audit_model)

    print(f"[INFO] Robot: {profile.name}")
    print(f"[INFO] MJCF: {vis_mjcf_path}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Root frame: {args.root_frame}")
    print(f"[INFO] Files: {len(files)} under {motion_root}")

    if args.mode == "bodies":
        for index, path in enumerate(files):
            data = load_amp_npz(path)
            fps = as_fps_scalar(data["fps"])
            num_frames = int(data["joint_pos"].shape[0])
            print(f"\n[INFO] ({index + 1}/{len(files)}) {path.name} ({num_frames} frames @ {fps:.1f} fps)")
            if args.audit:
                audit = audit_motion(
                    path,
                    data,
                    model=audit_model,
                    joint_info=joint_info,
                    body_names=body_names,
                    robot_profile=profile,
                )
                print_audit(audit)
            mujoco_visualize_bodies(
                data,
                max_frames=-1,
                stride=int(args.stride),
                loop=bool(args.loop_motion),
                speed=float(args.speed),
            )
            if args.pause_between > 0:
                time.sleep(float(args.pause_between))
        return 0

    import mujoco
    from mujoco import viewer

    vis_model = load_mujoco_model(vis_mjcf_path)
    sim = mujoco.MjData(vis_model)

    with viewer.launch_passive(vis_model, sim) as v:
        while v.is_running():
            for index, path in enumerate(files):
                if not v.is_running():
                    break

                data = load_amp_npz(path)
                fps = as_fps_scalar(data["fps"])
                num_frames = int(data["joint_pos"].shape[0])
                print(f"\n[INFO] ({index + 1}/{len(files)}) {path.name} ({num_frames} frames @ {fps:.1f} fps)")

                if args.audit:
                    audit = audit_motion(
                        path,
                        data,
                        model=audit_model,
                        joint_info=joint_info,
                        body_names=body_names,
                        robot_profile=profile,
                    )
                    print_audit(audit)

                qpos_traj, _ = amp_npz_to_qpos(data, vis_model.nq)
                print_root_trajectory_stats(qpos_traj)
                qpos_traj = apply_root_frame_for_vis(qpos_traj, str(args.root_frame))

                while v.is_running():
                    mujoco_play_qpos_trajectory(
                        vis_model,
                        qpos_traj,
                        fps,
                        stride=int(args.stride),
                        speed=float(args.speed),
                        loop=bool(args.loop_motion),
                        viewer=v,
                        sim=sim,
                    )
                    break

                if args.pause_between > 0:
                    time.sleep(float(args.pause_between))

            if not args.loop_files:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
