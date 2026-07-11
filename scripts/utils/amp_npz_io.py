"""Shared helpers for AMP_mjlab motion NPZ I/O, qpos reconstruction, and audits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

DEFAULT_MJCF = Path("src/assets/robots/biped_s17/xml/biped_s17.xml")
DEFAULT_VIS_MJCF = Path("src/assets/robots/biped_s17/xml/scene.xml")
DEFAULT_MOTION_ROOT = Path("src/assets/motions/s17/amp/loco")
S17_XML_DIR = Path("src/assets/robots/biped_s17/xml")
S17_ROBOT_XML = S17_XML_DIR / "biped_s17.xml"
S17_SCENE_XML = S17_XML_DIR / "scene.xml"
S17_SIM_JOINT_COUNT = 21
S17_LEGACY_MOTION_JOINT_COUNT = 23


@dataclass(frozen=True)
class RobotProfile:
    """Robot-specific paths and body names for AMP motion audit/visualization."""

    name: str
    mjcf: Path
    vis_mjcf: Path
    motion_root: Path
    base_body: str
    foot_left_body: str
    foot_right_body: str
    walk_base_z_min: float = 0.45
    foot_contact_band: float = 0.03
    foot_slip_p95_tol: float = 0.20


ROBOT_PROFILES: dict[str, RobotProfile] = {
    "s17": RobotProfile(
        name="s17",
        mjcf=Path("src/assets/robots/biped_s17/xml/biped_s17.xml"),
        vis_mjcf=Path("src/assets/robots/biped_s17/xml/scene.xml"),
        motion_root=Path("src/assets/motions/s17/amp/loco"),
        base_body="base_link",
        foot_left_body="leg_l6_link",
        foot_right_body="leg_r6_link",
        walk_base_z_min=0.45,
    ),
    "g1": RobotProfile(
        name="g1",
        mjcf=Path("src/assets/robots/unitree_g1/xmls/g1.xml"),
        vis_mjcf=Path("src/assets/robots/unitree_g1/xmls/scene_g1.xml"),
        motion_root=Path("src/assets/motions/g1/amp"),
        base_body="pelvis",
        foot_left_body="left_ankle_roll_link",
        foot_right_body="right_ankle_roll_link",
        walk_base_z_min=0.55,
    ),
}


def get_robot_profile(name: str) -> RobotProfile:
    key = name.lower()
    if key not in ROBOT_PROFILES:
        choices = ", ".join(sorted(ROBOT_PROFILES))
        raise ValueError(f"Unknown robot {name!r}. Choose from: {choices}")
    return ROBOT_PROFILES[key]


def resolve_robot_paths(
    robot: str,
    *,
    motion_root: Path | None = None,
    mjcf: Path | None = None,
    vis_mjcf: Path | None = None,
) -> tuple[RobotProfile, Path, Path, Path]:
    profile = get_robot_profile(robot)
    resolved_motion_root = resolve_path(motion_root or profile.motion_root)
    resolved_mjcf = resolve_path(mjcf or profile.mjcf)
    resolved_vis_mjcf = resolve_path(vis_mjcf or profile.vis_mjcf)
    return profile, resolved_motion_root, resolved_mjcf, resolved_vis_mjcf


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def iter_amp_npz_files(root: Path) -> list[Path]:
    root = resolve_path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {root}")
    return files


def pick_npz(npz: Path | None, npz_dir: Path, seed: int = 0) -> Path:
    if npz is not None:
        path = resolve_path(npz)
        if not path.is_file():
            raise FileNotFoundError(f"NPZ not found: {path}")
        return path
    files = iter_amp_npz_files(npz_dir)
    return files[int(seed) % len(files)]


def as_fps_scalar(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(()))
    if arr.ndim == 1 and arr.shape[0] == 1:
        return float(arr[0])
    raise ValueError(f"fps must be scalar-ish, got shape {arr.shape}")


def load_amp_npz(path: Path) -> dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=False)
    return {key: pack[key] for key in pack.files}


def validate_format(data: dict[str, np.ndarray]) -> tuple[int, int, int, float]:
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing keys: {missing}. Present: {sorted(data.keys())}")

    fps = as_fps_scalar(data["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid fps={fps}")

    joint_pos = np.asarray(data["joint_pos"])
    joint_vel = np.asarray(data["joint_vel"])
    body_pos_w = np.asarray(data["body_pos_w"])
    body_quat_w = np.asarray(data["body_quat_w"])
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"])

    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be (T, ndof), got {joint_pos.shape}")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"joint_vel shape {joint_vel.shape} != joint_pos {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must be (T, nb, 3), got {body_pos_w.shape}")
    if body_quat_w.shape[:2] != body_pos_w.shape[:2] or body_quat_w.shape[-1] != 4:
        raise ValueError(f"body_quat_w shape mismatch: {body_quat_w.shape} vs {body_pos_w.shape}")
    if body_lin_vel_w.shape != body_pos_w.shape:
        raise ValueError(f"body_lin_vel_w shape mismatch: {body_lin_vel_w.shape}")
    if body_ang_vel_w.shape != body_pos_w.shape:
        raise ValueError(f"body_ang_vel_w shape mismatch: {body_ang_vel_w.shape}")

    num_frames = int(joint_pos.shape[0])
    if num_frames <= 0:
        raise ValueError("Empty motion (T=0)")

    for name, arr in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("body_pos_w", body_pos_w),
        ("body_quat_w", body_quat_w),
        ("body_lin_vel_w", body_lin_vel_w),
        ("body_ang_vel_w", body_ang_vel_w),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")

    return num_frames, int(joint_pos.shape[1]), int(body_pos_w.shape[1]), fps


def mjcf_xml_without_mesh_files(mjcf_path: Path) -> str:
    text = mjcf_path.read_text(encoding="utf-8")
    text = re.sub(r'^\s*<joint\b[^>]*\btype="fixed"[^>]*/>\s*$', "", text, flags=re.MULTILINE)
    text = re.sub(r'^\s*<mesh name="[^"]+" file="[^"]+"/>\s*$', "", text, flags=re.MULTILINE)
    text = re.sub(r'\btype="mesh"', 'type="sphere" size="0.01"', text)
    text = re.sub(r'\s+mesh="[^"]+"', "", text)
    return text


def _is_s17_mjcf(mjcf_path: Path) -> bool:
    resolved = resolve_path(mjcf_path)
    s17_root = resolve_path(S17_XML_DIR)
    try:
        return resolved.is_relative_to(s17_root)
    except AttributeError:
        return str(resolved).startswith(str(s17_root))


def compile_s17_mujoco_model(*, with_scene: bool = False):
    """Compile the 21-DOF S17 model used in training (head hinges removed)."""
    import mujoco

    from src.assets.robots.biped_s17.s17_constants import get_spec

    spec = get_spec()
    if with_scene:
        has_floor = any(getattr(geom, "name", None) == "floor" for geom in spec.worldbody.geoms)
        if not has_floor:
            spec.worldbody.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
            spec.worldbody.add_geom(
                name="floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05],
                rgba=[0.2, 0.3, 0.4, 1],
            )
    return spec.compile()


def load_mujoco_model(mjcf_path: Path):
    import mujoco

    mjcf_path = resolve_path(mjcf_path)
    if _is_s17_mjcf(mjcf_path):
        with_scene = mjcf_path.name == S17_SCENE_XML.name
        model = compile_s17_mujoco_model(with_scene=with_scene)
        return model
    try:
        return mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception:
        return mujoco.MjModel.from_xml_string(mjcf_xml_without_mesh_files(mjcf_path))


@dataclass
class JointInfo:
    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    qpos_indices: list[int]


def collect_joint_info(model) -> JointInfo:
    import mujoco

    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    qpos_indices: list[int] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        lo, hi = model.jnt_range[joint_id]
        lower.append(float(lo))
        upper.append(float(hi))
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return JointInfo(
        names=names,
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        qpos_indices=qpos_indices,
    )


def collect_body_names(model) -> list[str]:
    import mujoco

    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            raise ValueError(f"Unnamed body id={body_id}")
        names.append(name)
    return names


def suggest_robot_for_ndof(ndof: int) -> str | None:
    if int(ndof) == S17_SIM_JOINT_COUNT:
        return "s17"
    for name, profile in ROBOT_PROFILES.items():
        if name == "s17":
            continue
        mjcf = resolve_path(profile.mjcf)
        try:
            model = load_mujoco_model(mjcf)
            if int(model.nq) - 7 == int(ndof):
                return name
        except Exception:
            continue
    return None


def align_joint_pos_to_sim(joint_pos: np.ndarray, sim_joint_count: int) -> np.ndarray:
    """Map NPZ joint_pos columns to the MuJoCo sim hinge count.

    S17 legacy AMP NPZ stores 23 hinge columns (incl. fixed head); training sim uses 21.
    G1 and other robots use joint_pos width matching their sim model directly.
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    file_ndof = int(joint_pos.shape[-1])
    if file_ndof == int(sim_joint_count):
        return joint_pos
    if file_ndof == S17_LEGACY_MOTION_JOINT_COUNT and int(sim_joint_count) == S17_SIM_JOINT_COUNT:
        from src.assets.robots.biped_s17.s17_constants import strip_head_from_motion_dof

        return strip_head_from_motion_dof(joint_pos)
    return joint_pos


def amp_npz_to_qpos(data: dict[str, np.ndarray], model_nq: int) -> tuple[np.ndarray, float]:
    """Build MuJoCo qpos from AMP NPZ (root from body 0 + joint_pos)."""
    fps = as_fps_scalar(data["fps"])
    root_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0, :]
    root_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0, :]
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    sim_joint_count = int(model_nq) - 7
    joint_pos = align_joint_pos_to_sim(joint_pos, sim_joint_count)

    num_frames = joint_pos.shape[0]
    if root_pos.shape[0] != num_frames or root_quat.shape[0] != num_frames:
        raise ValueError("Frame count mismatch between root pose and joint_pos")
    if 7 + joint_pos.shape[1] != int(model_nq):
        hint = suggest_robot_for_ndof(int(joint_pos.shape[1]))
        msg = (
            f"joint_pos ndof={joint_pos.shape[1]} incompatible with model.nq={model_nq} "
            f"(expected {int(model_nq) - 7} joints). "
            "Motion data and --robot/MJCF do not match."
        )
        if hint is not None:
            msg += f" Try --robot {hint} for this NPZ."
        if int(joint_pos.shape[1]) == S17_SIM_JOINT_COUNT and int(model_nq) - 7 == S17_LEGACY_MOTION_JOINT_COUNT:
            msg += (
                " S17 training uses a 21-DOF sim model (head fixed); use default --robot s17 "
                "so scene.xml loads via get_spec(), not the raw 23-DOF MJCF."
            )
        raise ValueError(msg)

    qpos = np.zeros((num_frames, int(model_nq)), dtype=np.float64)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat
    qpos[:, 7:] = joint_pos
    return qpos, fps


def apply_root_frame_for_vis(qpos_traj: np.ndarray, mode: str) -> np.ndarray:
    """Reframe root XY for MuJoCo viewing.

    Motions retain world-frame coordinates from retargeting (often meters away from
    the origin). Without reframing, the default camera looks at (0,0) and the robot
    appears to flail in empty space.

    Modes:
      - full: keep original world XY
      - center: subtract first-frame XY so the clip starts above the origin
      - inplace: force XY=0 each frame (gait in place, SOMA-style)
    """
    qpos = np.asarray(qpos_traj, dtype=np.float64).copy()
    if mode == "full":
        return qpos
    if mode == "center":
        qpos[:, 0] -= qpos[0, 0]
        qpos[:, 1] -= qpos[0, 1]
        return qpos
    if mode == "inplace":
        qpos[:, 0] = 0.0
        qpos[:, 1] = 0.0
        return qpos
    raise ValueError(f"Unknown root frame mode: {mode!r}. Use full|center|inplace.")


def print_root_trajectory_stats(qpos_traj: np.ndarray) -> None:
    xy = qpos_traj[:, 0:2]
    z = qpos_traj[:, 2]
    print(
        f"[INFO] root world traj: x=[{xy[:, 0].min():.2f}, {xy[:, 0].max():.2f}] "
        f"y=[{xy[:, 1].min():.2f}, {xy[:, 1].max():.2f}] z=[{z.min():.2f}, {z.max():.2f}]"
    )


def mujoco_play_qpos_trajectory(
    model,
    qpos_traj: np.ndarray,
    fps: float,
    *,
    stride: int = 1,
    speed: float = 1.0,
    loop: bool = False,
    viewer=None,
    sim=None,
) -> None:
    """Play qpos frames in an existing or new passive MuJoCo viewer."""
    import mujoco
    import time

    from mujoco import viewer as mj_viewer

    if sim is None:
        sim = mujoco.MjData(model)
    dt = (1.0 / max(1e-6, float(fps))) / max(1e-6, float(speed))
    stride = max(1, int(stride))
    frame_ids = range(0, int(qpos_traj.shape[0]), stride)

    def _play_once(v) -> None:
        while v.is_running():
            for frame_idx in frame_ids:
                sim.qpos[:] = qpos_traj[frame_idx]
                mujoco.mj_forward(model, sim)
                v.sync()
                time.sleep(dt)
                if not v.is_running():
                    return
            if not loop:
                return

    if viewer is None:
        with mj_viewer.launch_passive(model, sim) as v:
            _play_once(v)
    else:
        _play_once(viewer)


def smoothness_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 3:
        return {"d1_rms": float("nan"), "d2_rms": float("nan"), "d1_max": float("nan"), "d2_max": float("nan")}
    d1 = np.diff(arr, axis=0)
    d2 = np.diff(d1, axis=0)
    return {
        "d1_rms": float(np.sqrt(np.mean(d1 * d1))),
        "d2_rms": float(np.sqrt(np.mean(d2 * d2))),
        "d1_max": float(np.max(np.abs(d1))),
        "d2_max": float(np.max(np.abs(d2))),
    }


@dataclass
class JointLimitReport:
    name: str
    below: int
    above: int
    data_min: float
    data_max: float
    limit_lo: float
    limit_hi: float

    @property
    def total(self) -> int:
        return self.below + self.above


@dataclass
class FootSlipReport:
    contact_frames: int
    contact_ratio: float
    slip_mean: float
    slip_max: float
    slip_p95: float
    penetration_max: float


@dataclass
class MotionAudit:
    path: Path
    category: str
    robot: str
    num_frames: int
    ndof: int
    nbodies: int
    fps: float
    base_z_min: float
    base_z_max: float
    foot_left_z_min: float
    foot_left_z_max: float
    foot_right_z_min: float
    foot_right_z_max: float
    foot_left_slip: FootSlipReport
    foot_right_slip: FootSlipReport
    joint_limit_violations: int
    worst_joints: list[JointLimitReport] = field(default_factory=list)
    smoothness: dict[str, dict[str, float]] = field(default_factory=dict)
    fk_body_rmse: float | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


def _body_index(body_names: Sequence[str], name: str) -> int:
    try:
        return body_names.index(name)
    except ValueError as exc:
        raise KeyError(f"Body {name!r} not in model. Available: {body_names}") from exc


def foot_slip_stats(
    foot_pos: np.ndarray,
    foot_vel: np.ndarray,
    other_foot_z: np.ndarray,
    *,
    contact_band: float = 0.03,
) -> FootSlipReport:
    """Estimate foot XY slip speed during stance (near-ground and lower than other foot)."""
    foot_pos = np.asarray(foot_pos, dtype=np.float64)
    foot_vel = np.asarray(foot_vel, dtype=np.float64)
    other_foot_z = np.asarray(other_foot_z, dtype=np.float64)
    z = foot_pos[:, 2]
    xy_speed = np.linalg.norm(foot_vel[:, :2], axis=1)
    near_own_min = z <= (float(z.min()) + float(contact_band))
    is_lower = z <= (other_foot_z + 0.005)
    in_contact = near_own_min & is_lower
    penetration_max = float(max(0.0, -z.min()))

    if not np.any(in_contact):
        return FootSlipReport(
            contact_frames=0,
            contact_ratio=0.0,
            slip_mean=float("nan"),
            slip_max=float("nan"),
            slip_p95=float("nan"),
            penetration_max=penetration_max,
        )

    contact_speeds = xy_speed[in_contact]
    return FootSlipReport(
        contact_frames=int(in_contact.sum()),
        contact_ratio=float(in_contact.mean()),
        slip_mean=float(contact_speeds.mean()),
        slip_max=float(contact_speeds.max()),
        slip_p95=float(np.percentile(contact_speeds, 95)),
        penetration_max=penetration_max,
    )


def audit_motion(
    path: Path,
    data: dict[str, np.ndarray],
    *,
    model,
    joint_info: JointInfo,
    body_names: Sequence[str],
    robot_profile: RobotProfile | None = None,
    foot_penetration_tol: float = 0.02,
    walk_base_z_min: float | None = None,
    max_joint_violation_ratio: float = 0.01,
    foot_slip_p95_tol: float | None = None,
    check_fk: bool = True,
) -> MotionAudit:
    import mujoco

    profile = robot_profile or get_robot_profile("s17")
    walk_base_z_min = profile.walk_base_z_min if walk_base_z_min is None else walk_base_z_min
    foot_slip_p95_tol = profile.foot_slip_p95_tol if foot_slip_p95_tol is None else foot_slip_p95_tol

    num_frames, ndof, nbodies, fps = validate_format(data)
    category = path.parent.name

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_pos = align_joint_pos_to_sim(joint_pos, len(joint_info.names))
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)

    foot_l = _body_index(body_names, profile.foot_left_body)
    foot_r = _body_index(body_names, profile.foot_right_body)
    base_i = _body_index(body_names, profile.base_body)

    base_z = body_pos_w[:, base_i, 2]
    foot_l_z = body_pos_w[:, foot_l, 2]
    foot_r_z = body_pos_w[:, foot_r, 2]
    foot_left_slip = foot_slip_stats(
        body_pos_w[:, foot_l],
        body_lin_vel_w[:, foot_l],
        foot_r_z,
        contact_band=profile.foot_contact_band,
    )
    foot_right_slip = foot_slip_stats(
        body_pos_w[:, foot_r],
        body_lin_vel_w[:, foot_r],
        foot_l_z,
        contact_band=profile.foot_contact_band,
    )

    worst_joints: list[JointLimitReport] = []
    total_violations = 0
    for joint_idx, name in enumerate(joint_info.names):
        below = int(np.sum(joint_pos[:, joint_idx] < joint_info.lower[joint_idx] - 1e-3))
        above = int(np.sum(joint_pos[:, joint_idx] > joint_info.upper[joint_idx] + 1e-3))
        count = below + above
        if count > 0:
            report = JointLimitReport(
                name=name,
                below=below,
                above=above,
                data_min=float(joint_pos[:, joint_idx].min()),
                data_max=float(joint_pos[:, joint_idx].max()),
                limit_lo=float(joint_info.lower[joint_idx]),
                limit_hi=float(joint_info.upper[joint_idx]),
            )
            worst_joints.append(report)
            total_violations += count
    worst_joints.sort(key=lambda item: item.total, reverse=True)

    smoothness = {
        name: smoothness_stats(np.asarray(data[name]))
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w")
    }

    fk_body_rmse = None
    if check_fk:
        qpos, _ = amp_npz_to_qpos(data, model.nq)
        body_ids = list(range(1, model.nbody))
        sim = mujoco.MjData(model)
        errors: list[float] = []
        for frame_idx in range(num_frames):
            sim.qpos[:] = qpos[frame_idx]
            sim.qvel[:] = 0.0
            mujoco.mj_forward(model, sim)
            fk = np.asarray(sim.xpos[body_ids], dtype=np.float64)
            ref = body_pos_w[frame_idx]
            errors.append(float(np.sqrt(np.mean((fk - ref) ** 2))))
        fk_body_rmse = float(np.mean(errors))

    issues: list[str] = []
    if fk_body_rmse is not None and fk_body_rmse > 1e-2:
        issues.append(f"FK/body_pos_w RMSE={fk_body_rmse:.4f} m (>1cm)")

    violation_ratio = total_violations / max(1, num_frames * joint_pos.shape[1])
    if violation_ratio > max_joint_violation_ratio:
        top = worst_joints[0].name if worst_joints else "?"
        issues.append(
            f"joint limit violations={total_violations} ({100 * violation_ratio:.1f}% of frame×dof), worst={top}"
        )

    if category == "WalkandRun":
        if float(foot_l_z.min()) < -foot_penetration_tol or float(foot_r_z.min()) < -foot_penetration_tol:
            issues.append(
                f"foot penetration: left_min={foot_l_z.min():.3f}, right_min={foot_r_z.min():.3f}"
            )
        if float(base_z.min()) < walk_base_z_min:
            issues.append(f"base height too low: min_z={base_z.min():.3f}")
        if category == "WalkandRun":
            for side, slip in (("left", foot_left_slip), ("right", foot_right_slip)):
                if np.isfinite(slip.slip_p95) and slip.slip_p95 > foot_slip_p95_tol:
                    issues.append(
                        f"foot slip ({side}): p95={slip.slip_p95:.3f} m/s "
                        f"(>{foot_slip_p95_tol:.2f}, contact_ratio={100 * slip.contact_ratio:.0f}%)"
                    )

    audit = MotionAudit(
        path=path,
        category=category,
        robot=profile.name,
        num_frames=num_frames,
        ndof=ndof,
        nbodies=nbodies,
        fps=fps,
        base_z_min=float(base_z.min()),
        base_z_max=float(base_z.max()),
        foot_left_z_min=float(foot_l_z.min()),
        foot_left_z_max=float(foot_l_z.max()),
        foot_right_z_min=float(foot_r_z.min()),
        foot_right_z_max=float(foot_r_z.max()),
        foot_left_slip=foot_left_slip,
        foot_right_slip=foot_right_slip,
        joint_limit_violations=total_violations,
        worst_joints=worst_joints[:6],
        smoothness=smoothness,
        fk_body_rmse=fk_body_rmse,
        issues=issues,
    )
    return audit


def _format_slip(slip: FootSlipReport) -> str:
    if not np.isfinite(slip.slip_p95):
        return "n/a"
    return (
        f"p95={slip.slip_p95:.3f} mean={slip.slip_mean:.3f} max={slip.slip_max:.3f} "
        f"contact={100 * slip.contact_ratio:.0f}% pen={slip.penetration_max:.3f}"
    )


def print_audit(audit: MotionAudit) -> None:
    status = "OK" if audit.ok else "FAIL"
    rel = audit.path.name
    print(f"[{status}] {audit.category}/{rel}")
    print(
        f"  T={audit.num_frames} ndof={audit.ndof} fps={audit.fps:.1f} "
        f"base_z=[{audit.base_z_min:.3f},{audit.base_z_max:.3f}] "
        f"footL_z=[{audit.foot_left_z_min:.3f},{audit.foot_left_z_max:.3f}] "
        f"footR_z=[{audit.foot_right_z_min:.3f},{audit.foot_right_z_max:.3f}]"
    )
    print(f"  foot slip L: {_format_slip(audit.foot_left_slip)}")
    print(f"  foot slip R: {_format_slip(audit.foot_right_slip)}")
    if audit.fk_body_rmse is not None:
        print(f"  FK RMSE vs body_pos_w: {audit.fk_body_rmse * 1000:.2f} mm")
    print(f"  joint limit violations: {audit.joint_limit_violations}")
    for joint in audit.worst_joints[:3]:
        print(
            f"    - {joint.name}: viol={joint.total} "
            f"data=[{joint.data_min:.3f},{joint.data_max:.3f}] "
            f"lim=[{joint.limit_lo:.3f},{joint.limit_hi:.3f}]"
        )
    if audit.issues:
        for issue in audit.issues:
            print(f"  ISSUE: {issue}")


def print_batch_summary(audits: Sequence[MotionAudit], *, robot: str) -> None:
    if not audits:
        print("[SUMMARY] no motions")
        return

    num_fail = sum(1 for audit in audits if not audit.ok)
    num_ok = len(audits) - num_fail
    walk = [audit for audit in audits if audit.category == "WalkandRun"]
    recovery = [audit for audit in audits if audit.category == "Recovery"]

    def _collect(values: list[float]) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.min()), float(np.median(arr)), float(arr.max())

    def _slip_p95(audit: MotionAudit) -> list[float]:
        out: list[float] = []
        for slip in (audit.foot_left_slip, audit.foot_right_slip):
            if np.isfinite(slip.slip_p95):
                out.append(slip.slip_p95)
        return out

    print(f"[SUMMARY] robot={robot} {num_ok} OK, {num_fail} FAIL / {len(audits)} total")
    print(f"  categories: WalkandRun={len(walk)} Recovery={len(recovery)}")

    foot_l_min = [audit.foot_left_z_min for audit in audits]
    foot_r_min = [audit.foot_right_z_min for audit in audits]
    pen_l = [audit.foot_left_slip.penetration_max for audit in audits]
    pen_r = [audit.foot_right_slip.penetration_max for audit in audits]
    slip_all = [v for audit in audits for v in _slip_p95(audit)]

    print(
        f"  foot z min (L): min/med/max = {_collect(foot_l_min)[0]:.3f} / "
        f"{_collect(foot_l_min)[1]:.3f} / {_collect(foot_l_min)[2]:.3f}"
    )
    print(
        f"  foot z min (R): min/med/max = {_collect(foot_r_min)[0]:.3f} / "
        f"{_collect(foot_r_min)[1]:.3f} / {_collect(foot_r_min)[2]:.3f}"
    )
    print(
        f"  penetration depth (L/R): L {_collect(pen_l)[2]:.3f} max, "
        f"R {_collect(pen_r)[2]:.3f} max"
    )
    if slip_all:
        lo, med, hi = _collect(slip_all)
        print(f"  foot slip p95 (contact): min/med/max = {lo:.3f} / {med:.3f} / {hi:.3f} m/s")

    issue_types: dict[str, int] = {}
    for audit in audits:
        for issue in audit.issues:
            key = issue.split(":", 1)[0]
            issue_types[key] = issue_types.get(key, 0) + 1
    if issue_types:
        print("  issue breakdown:")
        for key, count in sorted(issue_types.items(), key=lambda item: (-item[1], item[0])):
            print(f"    - {key}: {count}")
