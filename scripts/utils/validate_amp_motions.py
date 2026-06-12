#!/usr/bin/env python3
"""Batch-validate AMP motion NPZ files for format, FK consistency, and joint limits.

Example:
  python scripts/utils/validate_amp_motions.py
  python scripts/utils/validate_amp_motions.py --robot s17
  python scripts/utils/validate_amp_motions.py --robot g1
  python scripts/utils/validate_amp_motions.py --fail-on-issues
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.amp_npz_io import (
    audit_motion,
    collect_body_names,
    collect_joint_info,
    iter_amp_npz_files,
    load_amp_npz,
    load_mujoco_model,
    print_audit,
    print_batch_summary,
    resolve_robot_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AMP motion NPZ files.")
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
        help="Root directory containing WalkandRun/ and Recovery/ NPZ files.",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="MuJoCo MJCF used for joint limits and FK checks.",
    )
    parser.add_argument(
        "--foot-penetration-tol",
        type=float,
        default=0.02,
        help="Max allowed foot z below ground for WalkandRun clips (m).",
    )
    parser.add_argument(
        "--max-joint-violation-ratio",
        type=float,
        default=0.01,
        help="Fail if joint limit violations exceed this fraction of frame×dof.",
    )
    parser.add_argument(
        "--no-fk-check",
        action="store_true",
        help="Skip FK consistency check between qpos and body_pos_w.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with code 1 if any motion has issues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile, motion_root, mjcf_path, _ = resolve_robot_paths(
        args.robot,
        motion_root=args.motion_root,
        mjcf=args.mjcf,
    )

    model = load_mujoco_model(mjcf_path)
    joint_info = collect_joint_info(model)
    body_names = collect_body_names(model)

    files = iter_amp_npz_files(motion_root)
    print(f"[INFO] Robot: {profile.name}")
    print(f"[INFO] MJCF: {mjcf_path}")
    print(f"[INFO] Motion root: {motion_root} ({len(files)} files)")
    print(f"[INFO] Bodies: {len(body_names)}, joints: {len(joint_info.names)}")

    audits = []
    for path in files:
        data = load_amp_npz(path)
        audit = audit_motion(
            path,
            data,
            model=model,
            joint_info=joint_info,
            body_names=body_names,
            robot_profile=profile,
            foot_penetration_tol=float(args.foot_penetration_tol),
            max_joint_violation_ratio=float(args.max_joint_violation_ratio),
            check_fk=not args.no_fk_check,
        )
        audits.append(audit)
        print_audit(audit)
        print()

    print_batch_summary(audits, robot=profile.name)
    num_fail = sum(1 for audit in audits if not audit.ok)
    if args.fail_on_issues and num_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
