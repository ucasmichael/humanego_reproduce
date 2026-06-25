# -- coding: UTF-8
"""
Convert a RealSense raw capture session into a HumanEgo RobotPreprocess session.

Input session layout is produced by read_frames.py:
    raw_session/
        metadata.json
        frames.jsonl
        rgb/000000.png ...          (optional if rgb.mp4 exists)
        depth/000000.png ...        (aligned to RGB)
        rgb.mp4

Output layout expected by HumanEgo preprocess.RobotPreprocess:
    <humanego>/data/<task>/<source>/teaching_<task>_<idx>/
        preprocess/session_meta.json
        preprocess/all_data/00000/rgb.png
        preprocess/all_data/00000/depth.png
        preprocess/all_data/00000/robot_state.json
        ...

Pose logs are optional for smoke tests, but required for meaningful training.
Supported hand pose JSONL fields per line:
    host_time_s | timestamp | ts
    side: "right" | "left"                         (optional, default --side)
    T_hand_to_world | T_hand_to_camera | T_ee_in_cam (4x4 list)
    grasp: 0..1 OR gripper_q: 0..1                  (optional)

Supported camera pose JSONL fields per line:
    host_time_s | timestamp | ts
    T_camera_to_world | T_cam_to_world | c2w         (4x4 list)

If a hand pose is camera-relative and a camera pose is available, the script writes
T_camera_to_world @ T_hand_to_camera into robot_state.json. HumanEgo's current
RobotDatasetGen interprets robot_state["arms"][side]["T_ee_in_cam"] as the world-frame
hand pose, despite the historical field name.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_frames_manifest(raw_session: Path) -> List[Dict[str, Any]]:
    frames_path = raw_session / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"Missing frame manifest: {frames_path}")
    frames = read_jsonl(frames_path)
    frames.sort(key=lambda x: int(x.get("frame_index", 0)))
    return frames


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def time_value(row: Dict[str, Any], preferred_key: str = "host_time_s") -> Optional[float]:
    value = row.get(preferred_key)
    if value is None:
        value = first_present(row, ["host_time_s", "timestamp", "ts", "time_s"])
    if value is None:
        return None
    return float(value)


class TimeIndex:
    def __init__(self, rows: List[Dict[str, Any]], time_key: str):
        pairs = []
        for row in rows:
            t = time_value(row, time_key)
            if t is not None and math.isfinite(t):
                pairs.append((t, row))
        pairs.sort(key=lambda x: x[0])
        self.times = [p[0] for p in pairs]
        self.rows = [p[1] for p in pairs]

    def nearest(self, t: float, max_dt: float) -> Optional[Tuple[Dict[str, Any], float]]:
        if not self.times:
            return None
        pos = bisect.bisect_left(self.times, t)
        candidates = []
        if pos < len(self.times):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best = min(candidates, key=lambda i: abs(self.times[i] - t))
        dt = abs(self.times[best] - t)
        if dt > max_dt:
            return None
        return self.rows[best], dt


def as_matrix(value: Any, field_name: str) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"{field_name} must contain 16 numbers, got shape {arr.shape}")
    return arr.reshape(4, 4)


def get_camera_pose(row: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if row is None:
        return None
    value = first_present(row, ["T_camera_to_world", "T_cam_to_world", "c2w", "camera_to_world"])
    if value is None:
        return None
    return as_matrix(value, "camera pose")


def get_hand_pose(row: Optional[Dict[str, Any]], side: str) -> Tuple[Optional[np.ndarray], str]:
    if row is None:
        return None, "missing"

    arms = row.get("arms")
    if isinstance(arms, dict) and side in arms:
        arm = arms[side]
        if "T_hand_to_world" in arm:
            return as_matrix(arm["T_hand_to_world"], f"arms.{side}.T_hand_to_world"), "world"
        if "T_hand_to_camera" in arm:
            return as_matrix(arm["T_hand_to_camera"], f"arms.{side}.T_hand_to_camera"), "camera"
        if "T_ee_in_cam" in arm:
            return as_matrix(arm["T_ee_in_cam"], f"arms.{side}.T_ee_in_cam"), "world"

    value = first_present(row, ["T_hand_to_world", "T_world_hand", "T_hand_world"])
    if value is not None:
        return as_matrix(value, "T_hand_to_world"), "world"

    value = first_present(row, ["T_hand_to_camera", "T_camera_hand", "T_ee_in_cam"])
    if value is not None:
        return as_matrix(value, "T_hand_to_camera"), "camera"

    return None, "missing"


def get_gripper_q(row: Optional[Dict[str, Any]], side: str, default_gripper_q: float) -> float:
    if row is None:
        return float(default_gripper_q)

    arms = row.get("arms")
    if isinstance(arms, dict) and side in arms:
        arm = arms[side]
        if "gripper_q" in arm:
            return float(np.clip(float(arm["gripper_q"]), 0.0, 1.0))
        if "grasp" in arm:
            return float(np.clip(1.0 - float(arm["grasp"]), 0.0, 1.0))

    if "gripper_q" in row:
        return float(np.clip(float(row["gripper_q"]), 0.0, 1.0))
    if "grasp" in row:
        return float(np.clip(1.0 - float(row["grasp"]), 0.0, 1.0))
    return float(default_gripper_q)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def save_depth_mm(src: Path, dst: Path, depth_scale: Optional[float], mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    scale = 0.001 if depth_scale is None else float(depth_scale)
    if abs(scale - 0.001) < 1e-9 and mode == "symlink":
        copy_or_link(src, dst, mode)
        return
    if abs(scale - 0.001) < 1e-9 and mode == "copy":
        copy_or_link(src, dst, mode)
        return

    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Could not read depth image: {src}")
    depth_mm = np.rint(raw.astype(np.float64) * scale * 1000.0)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    ok = cv2.imwrite(str(dst), depth_mm)
    if not ok:
        raise RuntimeError(f"Could not write depth image: {dst}")


def extract_rgb_from_video(raw_session: Path, frame_index: int, capture: cv2.VideoCapture) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not extract frame {frame_index} from {raw_session / 'rgb.mp4'}")
    return frame


def build_output_session(args: argparse.Namespace) -> Path:
    session_name = f"{args.source_type}_{args.task}_{int(args.index):03d}"
    return Path(args.humanego_root) / "data" / args.task / args.source_type / session_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-session", required=True, help="Session directory produced by read_frames.py")
    parser.add_argument("--humanego-root", default="/data/lyx/HumanEgo", help="HumanEgo repository root")
    parser.add_argument("--task", required=True, help="Task name, e.g. serve_bread")
    parser.add_argument("--index", type=int, default=0, help="Session index. 000 is usually eval; 001+ train")
    parser.add_argument("--source-type", default="teaching", choices=["teaching", "teleop", "aria"], help="Data source folder used by HumanEgo training")
    parser.add_argument("--side", default="right", choices=["right", "left"], help="Dominant hand side")
    parser.add_argument("--hand-poses", default=None, help="Optional hand pose JSONL aligned by timestamp")
    parser.add_argument("--camera-poses", default=None, help="Optional camera pose JSONL aligned by timestamp")
    parser.add_argument("--raw-time-key", default="host_time_s", help="Time field in frames.jsonl")
    parser.add_argument("--pose-time-key", default="host_time_s", help="Time field in pose JSONL files")
    parser.add_argument("--max-pose-dt-s", type=float, default=0.05, help="Maximum nearest-neighbor pose time gap")
    parser.add_argument("--default-gripper-q", type=float, default=1.0, help="Used when pose log has no grasp/gripper. 1=open, 0=closed")
    parser.add_argument("--require-hand-pose", action="store_true", help="Fail if no matching hand pose exists for any frame")
    parser.add_argument("--dual-arm", action="store_true", help="Mark session as dual-arm in session_meta.json")
    parser.add_argument("--finished-tail-frames", type=int, default=30, help="Mark last N frames is_finished=true")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="Copy or symlink RGB/depth files when possible")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing converted session")
    args = parser.parse_args()

    raw_session = Path(args.raw_session).resolve()
    metadata_path = raw_session / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")

    metadata = read_json(metadata_path)
    frames = read_frames_manifest(raw_session)
    if not frames:
        raise RuntimeError(f"No frames found in {raw_session / 'frames.jsonl'}")

    hand_index = TimeIndex(read_jsonl(Path(args.hand_poses)), args.pose_time_key) if args.hand_poses else None
    camera_index = TimeIndex(read_jsonl(Path(args.camera_poses)), args.pose_time_key) if args.camera_poses else None

    out_session = build_output_session(args)
    preprocess_dir = out_session / "preprocess"
    all_data_dir = preprocess_dir / "all_data"
    if out_session.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {out_session}. Use --force to overwrite.")
        shutil.rmtree(out_session)
    all_data_dir.mkdir(parents=True, exist_ok=True)

    color_intr = metadata.get("color_intrinsics") or {}
    K = color_intr.get("k")
    if K is None:
        raise KeyError("metadata.json has no color_intrinsics.k. Re-record with updated read_frames.py.")

    depth_scale = metadata.get("depth_scale_meters_per_unit")
    video_capture = None
    if any(frame.get("rgb_path") is None for frame in frames):
        video_capture = cv2.VideoCapture(str(raw_session / "rgb.mp4"))
        if not video_capture.isOpened():
            raise RuntimeError(f"Could not open RGB video: {raw_session / 'rgb.mp4'}")

    matched_hand = 0
    missing_hand = 0
    matched_camera = 0
    missing_depth = 0

    total = len(frames)
    for out_idx, frame in enumerate(frames):
        frame_idx = int(frame.get("frame_index", out_idx))
        frame_dir = all_data_dir / f"{out_idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        rgb_rel = frame.get("rgb_path")
        if rgb_rel:
            copy_or_link(raw_session / rgb_rel, frame_dir / "rgb.png", args.mode)
        else:
            rgb = extract_rgb_from_video(raw_session, frame_idx, video_capture)
            cv2.imwrite(str(frame_dir / "rgb.png"), rgb)

        depth_rel = frame.get("depth_path")
        if depth_rel:
            save_depth_mm(raw_session / depth_rel, frame_dir / "depth.png", depth_scale, args.mode)
        else:
            missing_depth += 1

        t = time_value(frame, args.raw_time_key)
        if t is None:
            raise KeyError(f"Frame {frame_idx} has no usable timestamp field")

        hand_row = None
        hand_dt = None
        if hand_index is not None:
            found = hand_index.nearest(t, args.max_pose_dt_s)
            if found is not None:
                hand_row, hand_dt = found

        camera_row = None
        camera_dt = None
        T_cam_to_world = np.eye(4, dtype=np.float64)
        if camera_index is not None:
            found = camera_index.nearest(t, args.max_pose_dt_s)
            if found is not None:
                camera_row, camera_dt = found
                pose = get_camera_pose(camera_row)
                if pose is not None:
                    T_cam_to_world = pose
                    matched_camera += 1

        T_hand, pose_frame = get_hand_pose(hand_row, args.side)
        if T_hand is None:
            if args.require_hand_pose:
                raise RuntimeError(f"No hand pose matched frame {frame_idx} at t={t:.6f}")
            T_hand_world = np.eye(4, dtype=np.float64)
            missing_hand += 1
        elif pose_frame == "camera":
            T_hand_world = T_cam_to_world @ T_hand
            matched_hand += 1
        else:
            T_hand_world = T_hand
            matched_hand += 1

        gripper_q = get_gripper_q(hand_row, args.side, args.default_gripper_q)
        is_finished = out_idx >= max(0, total - int(args.finished_tail_frames))

        robot_state = {
            "ts": t,
            "frame_index": out_idx,
            "raw_frame_index": frame_idx,
            "is_finished": bool(is_finished),
            "arms": {
                args.side: {
                    "T_ee_in_cam": T_hand_world.tolist(),
                    "gripper_q": float(gripper_q),
                    "pose_convention": "humanego_gripper_y_approach",
                    "pose_source": "hand_pose_jsonl",
                }
            },
            "camera": {
                "T_camera_to_world": T_cam_to_world.tolist(),
            },
            "sync": {
                "hand_dt_s": hand_dt,
                "camera_dt_s": camera_dt,
                "raw_time_key": args.raw_time_key,
                "pose_time_key": args.pose_time_key,
            },
        }
        with (frame_dir / "robot_state.json").open("w", encoding="utf-8") as f:
            json.dump(robot_state, f, indent=2)

    if video_capture is not None:
        video_capture.release()

    session_meta = {
        "fps": float(metadata.get("fps", 30)),
        "w": int(color_intr.get("width", metadata.get("width", 640))),
        "h": int(color_intr.get("height", metadata.get("height", 480))),
        "k": K,
        "n_frames": total,
        "source_type": args.source_type,
        "dual_arm": bool(args.dual_arm),
        "cameras": {
            "cam0": {
                "k": K,
                "w": int(color_intr.get("width", metadata.get("width", 640))),
                "h": int(color_intr.get("height", metadata.get("height", 480))),
            }
        },
        "raw_session": str(raw_session),
        "raw_metadata": metadata,
        "depth_unit": "millimetres",
        "depth_aligned_to_color": bool(metadata.get("depth_aligned_to_color", True)),
        "hand_pose_source": args.hand_poses,
        "camera_pose_source": args.camera_poses,
        "conversion_notes": [
            "RobotDatasetGen treats robot_state arms T_ee_in_cam as the world-frame hand pose.",
            "This converted session is directly usable by preprocess.RobotPreprocess.",
            "For moving ego cameras, current HumanEgo RobotPreprocess still assumes static cam0 for object lifting; use this path as a practical RGB-D approximation unless you add a moving-camera depth/triangulation stage.",
        ],
    }
    with (preprocess_dir / "session_meta.json").open("w", encoding="utf-8") as f:
        json.dump(session_meta, f, indent=2)

    summary = {
        "output_session": str(out_session),
        "frames": total,
        "matched_hand_frames": matched_hand,
        "missing_hand_frames": missing_hand,
        "matched_camera_frames": matched_camera,
        "missing_depth_frames": missing_depth,
    }
    with (preprocess_dir / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nNext:")
    print(f"  cd {Path(args.humanego_root).resolve()}")
    print(f"  python -m preprocess.RobotPreprocess --session_path {out_session} --task {args.task} --start_from auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
