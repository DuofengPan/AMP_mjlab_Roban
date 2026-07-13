#!/usr/bin/env python3
"""Lift existing AMP NPZ clips in +Z to remove ground penetration.

Example:
  python scripts/utils/align_amp_motion_ground.py \\
    --robot s17 \\
    --motion-root src/assets/motions/s17/amp/loco/Recovery \\
    --in-place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.amp_npz_io import (
    align_amp_motion_to_ground,
    compute_ground_lift,
    iter_amp_npz_files,
    load_amp_npz,
    resolve_robot_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align AMP motion NPZ clips to ground plane z=0.")
    parser.add_argument("--robot", choices=("s17", "g1"), default="s17")
    parser.add_argument("--motion-root", type=Path, default=None)
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.01,
        help="Minimum body z above ground after alignment (m).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite NPZ files in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report lifts only; do not write files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile, motion_root, _, _ = resolve_robot_paths(args.robot, motion_root=args.motion_root)
    files = iter_amp_npz_files(motion_root)

    print(f"[INFO] Robot: {profile.name}")
    print(f"[INFO] Motion root: {motion_root} ({len(files)} files)")
    print(f"[INFO] Ground clearance: {args.ground_clearance:.3f} m")

    changed = 0
    for path in files:
        data = load_amp_npz(path)
        lift_before = compute_ground_lift(
            data["body_pos_w"],
            ground_clearance=args.ground_clearance,
        )
        if lift_before <= 0.0:
            print(f"[SKIP] {path.relative_to(motion_root)} (already clear)")
            continue

        aligned, lift = align_amp_motion_to_ground(
            data,
            ground_clearance=args.ground_clearance,
        )
        rel = path.relative_to(motion_root)
        print(f"[LIFT] {rel}: +{lift:.4f} m")
        if args.dry_run:
            changed += 1
            continue

        out_path = path if args.in_place else path.with_name(path.stem + "_ground_aligned.npz")
        np.savez(str(out_path), **aligned)
        changed += 1

    print(f"[DONE] aligned {changed}/{len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
