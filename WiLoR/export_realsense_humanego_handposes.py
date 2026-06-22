from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full


ARIA_REQUIRED = {
    "thumb_tip": 0,
    "index_tip": 1,
    "wrist": 5,
    "thumb_base": 6,
    "index_base": 8,
    "middle_mcp": 11,
}

# WiLoR/MANO wrapper outputs OpenPose-style hand ordering:
#   0=Wrist, 1=ThumbCMC, 2=ThumbMCP, 3=ThumbIP, 4=ThumbTip,
#   5=IndexMCP, 6=IndexPIP, 7=IndexDIP, 8=IndexTip, ...
# HumanEgo/Aria uses:
#   0=ThumbTip, 1=IndexTip, 5=Wrist, 6=ThumbMCP, 8=IndexMCP, ...
# WILOR_TO_ARIA[aria_idx] = wilor_idx. Aria 20 PalmCenter is computed.
WILOR_TO_ARIA = [
    4,   # Aria 0  = ThumbTip
    8,   # Aria 1  = IndexTip
    12,  # Aria 2  = MiddleTip
    16,  # Aria 3  = RingTip
    20,  # Aria 4  = PinkyTip
    0,   # Aria 5  = Wrist
    2,   # Aria 6  = ThumbMCP
    3,   # Aria 7  = ThumbIP
    5,   # Aria 8  = IndexMCP
    6,   # Aria 9  = IndexPIP
    7,   # Aria 10 = IndexDIP
    9,   # Aria 11 = MiddleMCP
    10,  # Aria 12 = MiddlePIP
    11,  # Aria 13 = MiddleDIP
    13,  # Aria 14 = RingMCP
    14,  # Aria 15 = RingPIP
    15,  # Aria 16 = RingDIP
    17,  # Aria 17 = PinkyMCP
    18,  # Aria 18 = PinkyPIP
    19,  # Aria 19 = PinkyDIP
    -1,  # Aria 20 = PalmCenter
]

CORE_VIS_KEYPOINTS = [
    ARIA_REQUIRED["thumb_tip"],
    ARIA_REQUIRED["index_tip"],
    ARIA_REQUIRED["wrist"],
    ARIA_REQUIRED["thumb_base"],
    ARIA_REQUIRED["index_base"],
]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def remap_wilor_to_aria(kpts_wilor_21: np.ndarray) -> np.ndarray:
    kpts_wilor_21 = np.asarray(kpts_wilor_21)
    kpts_aria = np.full((21, kpts_wilor_21.shape[1]), np.nan, dtype=kpts_wilor_21.dtype)
    for aria_idx in range(20):
        kpts_aria[aria_idx] = kpts_wilor_21[WILOR_TO_ARIA[aria_idx]]
    kpts_aria[20] = (kpts_wilor_21[0] + kpts_wilor_21[5] + kpts_wilor_21[9]) / 3.0
    return kpts_aria


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def iter_frame_rows(session_dir: Path) -> List[Dict[str, Any]]:
    frames_path = session_dir / "frames.jsonl"
    if frames_path.exists():
        rows = read_jsonl(frames_path)
        rows.sort(key=lambda x: int(x.get("frame_index", 0)))
        return rows

    rgb_dir = session_dir / "rgb"
    rows = []
    for idx, path in enumerate(sorted(rgb_dir.glob("*.png"))):
        rows.append(
            {
                "frame_index": idx,
                "host_time_s": float(idx),
                "rgb_path": str(path.relative_to(session_dir)),
                "depth_path": str((session_dir / "depth" / path.name).relative_to(session_dir)),
            }
        )
    return rows


def project_points(points_cam: np.ndarray, focal_length: float, img_size_wh: np.ndarray) -> np.ndarray:
    cx = float(img_size_wh[0]) / 2.0
    cy = float(img_size_wh[1]) / 2.0
    z = np.maximum(points_cam[:, 2:3], 1e-6)
    return np.column_stack(
        [
            focal_length * points_cam[:, 0] / z[:, 0] + cx,
            focal_length * points_cam[:, 1] / z[:, 0] + cy,
        ]
    )


def project_with_intrinsics(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = points_cam[:, 2]
    out = np.full((len(points_cam), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1e-6)
    if np.any(valid):
        homo = (K @ points_cam[valid].T).T
        out[valid, 0] = homo[:, 0] / homo[:, 2]
        out[valid, 1] = homo[:, 1] / homo[:, 2]
    return out


def backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def sample_depth_m(
    depth_raw: np.ndarray,
    u: float,
    v: float,
    depth_scale: float,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
    trim_quantile: float,
) -> Tuple[float, int]:
    h, w = depth_raw.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return math.nan, 0

    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    vals = depth_raw[y0:y1, x0:x1].astype(np.float64) * depth_scale
    vals = vals[np.isfinite(vals) & (vals >= min_depth_m) & (vals <= max_depth_m)]
    if vals.size == 0:
        return math.nan, 0

    if vals.size >= 5 and trim_quantile > 0:
        lo = np.quantile(vals, trim_quantile)
        hi = np.quantile(vals, 1.0 - trim_quantile)
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        if trimmed.size > 0:
            vals = trimmed
    return float(np.median(vals)), int(vals.size)


def depth_lift_keypoints(
    keypoints_2d: np.ndarray,
    depth_raw: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
    trim_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoints_3d = np.full((len(keypoints_2d), 3), np.nan, dtype=np.float64)
    depth_samples = np.full(len(keypoints_2d), np.nan, dtype=np.float64)
    sample_counts = np.zeros(len(keypoints_2d), dtype=np.int32)

    for idx, (u, v) in enumerate(keypoints_2d):
        z, count = sample_depth_m(
            depth_raw,
            float(u),
            float(v),
            depth_scale,
            radius,
            min_depth_m,
            max_depth_m,
            trim_quantile,
        )
        depth_samples[idx] = z
        sample_counts[idx] = count
        if math.isfinite(z):
            keypoints_3d[idx] = backproject(float(u), float(v), z, K)

    return keypoints_3d, depth_samples, sample_counts


def safe_normalize(vec: np.ndarray, eps: float = 1e-6) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < eps or not math.isfinite(norm):
        return None
    return vec / norm


def build_midpoint_pose(keypoints_3d: np.ndarray, prev_R: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Build a HumanEgo gripper-like midpoint pose from Aria-ordered keypoints."""
    required = [ARIA_REQUIRED[k] for k in ("thumb_tip", "index_tip", "wrist", "thumb_base", "index_base")]
    if not np.all(np.isfinite(keypoints_3d[required])):
        return None

    thumb = keypoints_3d[ARIA_REQUIRED["thumb_tip"]]
    index = keypoints_3d[ARIA_REQUIRED["index_tip"]]
    wrist = keypoints_3d[ARIA_REQUIRED["wrist"]]
    thumb_base = keypoints_3d[ARIA_REQUIRED["thumb_base"]]
    index_base = keypoints_3d[ARIA_REQUIRED["index_base"]]

    midpoint = 0.5 * (thumb + index)
    x_axis = safe_normalize(index_base - thumb_base)
    if x_axis is None:
        return None

    base_midpoint = 0.5 * (thumb_base + index_base)
    y_raw = base_midpoint - wrist
    y_proj = y_raw - float(np.dot(y_raw, x_axis)) * x_axis
    y_axis = safe_normalize(y_proj)
    if y_axis is None:
        return None

    z_axis = safe_normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = safe_normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    if prev_R is not None and float(np.dot(prev_R[:, 0], x_axis)) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
        z_axis = np.cross(x_axis, y_axis)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = midpoint
    return T


def draw_hand_visualization(
    img_bgr: np.ndarray,
    K: np.ndarray,
    keypoints_3d: np.ndarray,
    T_hand_to_camera: Optional[np.ndarray],
    bbox_xyxy: Optional[Iterable[float]],
    frame_index: int,
    axis_length_m: float,
) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)

    keypoints_2d = project_with_intrinsics(keypoints_3d, K)
    for idx in CORE_VIS_KEYPOINTS:
        u, v = keypoints_2d[idx]
        if np.isfinite([u, v]).all() and 0 <= u < w and 0 <= v < h:
            cv2.circle(vis, (int(round(u)), int(round(v))), 5, (0, 0, 255), -1, cv2.LINE_AA)

    if T_hand_to_camera is not None:
        origin = T_hand_to_camera[:3, 3]
        axes = T_hand_to_camera[:3, :3]
        points = np.vstack(
            [
                origin,
                origin + axes[:, 0] * axis_length_m,
                origin + axes[:, 1] * axis_length_m,
                origin + axes[:, 2] * axis_length_m,
            ]
        )
        projected = project_with_intrinsics(points, K)
        p0 = projected[0]
        if np.isfinite(p0).all():
            p0_i = (int(round(p0[0])), int(round(p0[1])))
            axis_specs = [
                (1, (0, 0, 255), "X"),
                (2, (0, 255, 0), "Y"),
                (3, (255, 0, 0), "Z"),
            ]
            for point_idx, color, label in axis_specs:
                p1 = projected[point_idx]
                if np.isfinite(p1).all():
                    p1_i = (int(round(p1[0])), int(round(p1[1])))
                    cv2.arrowedLine(vis, p0_i, p1_i, color, 3, cv2.LINE_AA, tipLength=0.25)
                    cv2.putText(vis, label, p1_i, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.putText(
        vis,
        f"frame {frame_index}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def estimate_grasp(keypoints_3d: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Estimate grasp from Aria-ordered keypoints."""
    thumb = keypoints_3d[ARIA_REQUIRED["thumb_tip"]]
    index = keypoints_3d[ARIA_REQUIRED["index_tip"]]
    wrist = keypoints_3d[ARIA_REQUIRED["wrist"]]
    middle = keypoints_3d[ARIA_REQUIRED["middle_mcp"]]
    if not np.all(np.isfinite([thumb, index, wrist, middle])):
        return 0.0, math.nan

    tip_distance = float(np.linalg.norm(thumb - index))
    palm_size = float(np.linalg.norm(middle - wrist))
    if palm_size < 1e-6:
        return 0.0, math.nan
    ratio = tip_distance / palm_size
    return (1.0 if ratio < threshold else 0.0), ratio


def smooth_binary(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values
    half = window // 2
    out = []
    arr = np.asarray(values, dtype=np.float64)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out.append(float(np.mean(arr[lo:hi]) >= 0.5))
    return out


def choose_detection(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    is_right: np.ndarray,
    side: str,
    allow_other_side: bool,
) -> Optional[int]:
    desired = 1 if side == "right" else 0
    candidates = [i for i, val in enumerate(is_right.astype(int).tolist()) if val == desired]
    if not candidates and allow_other_side:
        candidates = list(range(len(boxes_xyxy)))
    if not candidates:
        return None
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clip(min=0) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clip(min=0)
    return max(candidates, key=lambda i: float(areas[i]) * float(scores[i]))


def load_models(args: argparse.Namespace, device: torch.device):
    model, model_cfg = load_wilor(
        checkpoint_path=str(args.wilor_checkpoint),
        cfg_path=str(args.wilor_config),
    )
    model = model.to(device)
    model.eval()
    detector = YOLO(str(args.detector_checkpoint)).to(device)
    return model, model_cfg, detector


def run_wilor_on_image(
    img_bgr: np.ndarray,
    model,
    model_cfg,
    detector,
    device: torch.device,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    detections = detector(img_bgr, conf=args.det_conf, verbose=False)[0]
    boxes = []
    scores = []
    is_right = []
    for det in detections:
        data = det.boxes.data.detach().cpu().reshape(-1).numpy()
        if data.size < 6:
            continue
        boxes.append(data[:4].astype(np.float64))
        scores.append(float(data[4]))
        is_right.append(int(round(float(data[5]))))
    if not boxes:
        return None

    boxes_np = np.stack(boxes)
    scores_np = np.asarray(scores, dtype=np.float64)
    right_np = np.asarray(is_right, dtype=np.float32)
    chosen_det = choose_detection(boxes_np, scores_np, right_np, args.side, args.allow_other_side)
    if chosen_det is None:
        return None

    dataset = ViTDetDataset(
        model_cfg,
        img_bgr,
        boxes_np[[chosen_det]],
        right_np[[chosen_det]],
        rescale_factor=args.rescale_factor,
        fp16=False,
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    pred_cam = out["pred_cam"].clone()
    multiplier = 2 * batch["right"] - 1
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal).detach().cpu().numpy()[0]

    joints = out["pred_keypoints_3d"][0].detach().cpu().numpy().astype(np.float64)
    hand_is_right = float(batch["right"][0].detach().cpu().item())
    joints[:, 0] = (2 * hand_is_right - 1) * joints[:, 0]
    joints_cam = joints + cam_t.reshape(1, 3)
    keypoints_2d = project_points(joints_cam, float(focal.detach().cpu().item()), img_size[0].detach().cpu().numpy())

    return {
        "detection_index": int(chosen_det),
        "detected_side": "right" if int(right_np[chosen_det]) == 1 else "left",
        "score": float(scores_np[chosen_det]),
        "bbox_xyxy": boxes_np[chosen_det].tolist(),
        "wilor_keypoints_camera": joints_cam,
        "keypoints_2d": keypoints_2d,
        "wilor_cam_t": cam_t.tolist(),
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export RealSense + WiLoR hand poses for HumanEgo.")
    parser.add_argument("--session", required=True, help="RealSense session directory with metadata.json and frames.jsonl")
    parser.add_argument("--out", default=None, help="Output JSONL path. Default: <session>/hand_poses_wilor.jsonl")
    parser.add_argument("--side", default="right", choices=["right", "left"], help="Hand side to export")
    parser.add_argument("--allow-other-side", action="store_true", help="Use any detected hand if requested side is absent")
    parser.add_argument("--det-conf", type=float, default=0.3, help="YOLO hand detector confidence")
    parser.add_argument("--rescale-factor", type=float, default=2.0, help="WiLoR bbox padding factor")
    parser.add_argument("--depth-radius", type=int, default=3, help="Depth median window radius in pixels")
    parser.add_argument("--depth-trim-quantile", type=float, default=0.1, help="Trim low/high depth quantiles in the local window")
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--min-valid-keypoints", type=int, default=12)
    parser.add_argument("--grasp-threshold", type=float, default=1.0, help="thumb-index distance / palm size threshold")
    parser.add_argument("--grasp-smooth-window", type=int, default=3)
    parser.add_argument("--fallback-wilor-depth", action="store_true", help="Fill invalid depth keypoints with WiLoR camera estimates")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip frames before this manifest position")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames for debugging")
    parser.add_argument("--visualize", action="store_true", help="Write an RGB overlay MP4 with five core keypoints and pose axes")
    parser.add_argument("--vis-out", default=None, help="Visualization MP4 path. Default: <out stem>_vis.mp4")
    parser.add_argument("--axis-length-m", type=float, default=0.06, help="Length of visualized pose axes in meters")
    parser.add_argument("--wilor-checkpoint", type=Path, default=Path("pretrained_models/wilor_final.ckpt"))
    parser.add_argument("--wilor-config", type=Path, default=Path("pretrained_models/model_config.yaml"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("pretrained_models/detector.pt"))
    args = parser.parse_args()

    for attr in ("wilor_checkpoint", "wilor_config", "detector_checkpoint"):
        path = getattr(args, attr)
        if not path.is_absolute():
            setattr(args, attr, script_dir / path)

    session_dir = Path(args.session).resolve()
    out_path = Path(args.out).resolve() if args.out else session_dir / "hand_poses_wilor.jsonl"
    vis_path = Path(args.vis_out).resolve() if args.vis_out else out_path.with_name(f"{out_path.stem}_vis.mp4")
    metadata = read_json(session_dir / "metadata.json")
    color_intr = metadata.get("color_intrinsics") or {}
    if "k" not in color_intr:
        raise KeyError("metadata.json is missing color_intrinsics.k")
    K = np.asarray(color_intr["k"], dtype=np.float64).reshape(3, 3)
    depth_scale = float(metadata.get("depth_scale_meters_per_unit", 0.001))

    frames = iter_frame_rows(session_dir)
    frames = frames[max(0, args.start_frame) :]
    if args.frame_stride > 1:
        frames = frames[:: args.frame_stride]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg, detector = load_models(args, device)

    rows: List[Dict[str, Any]] = []
    prev_R: Optional[np.ndarray] = None
    stats = {"frames": len(frames), "written": 0, "no_detection": 0, "bad_depth": 0, "bad_pose": 0}
    vis_writer = None
    vis_fps = float(metadata.get("fps", 15))

    def write_vis_frame(frame_bgr: np.ndarray) -> None:
        nonlocal vis_writer
        if not args.visualize:
            return
        if vis_writer is None:
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vis_writer = cv2.VideoWriter(str(vis_path), fourcc, vis_fps, (width, height))
            if not vis_writer.isOpened():
                raise RuntimeError(f"Could not open visualization video for writing: {vis_path}")
        vis_writer.write(frame_bgr)

    for frame in frames:
        frame_index = int(frame.get("frame_index", len(rows)))
        rgb_path = session_dir / frame["rgb_path"]
        depth_path = session_dir / frame["depth_path"]
        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read RGB image: {rgb_path}")
        if depth is None:
            raise RuntimeError(f"Could not read depth image: {depth_path}")

        pred = run_wilor_on_image(img, model, model_cfg, detector, device, args)
        if pred is None:
            stats["no_detection"] += 1
            write_vis_frame(img)
            continue

        kpts_depth, depth_samples, sample_counts = depth_lift_keypoints(
            pred["keypoints_2d"],
            depth,
            K,
            depth_scale,
            args.depth_radius,
            args.min_depth_m,
            args.max_depth_m,
            args.depth_trim_quantile,
        )
        valid_depth = np.isfinite(kpts_depth[:, 2])
        if args.fallback_wilor_depth and not np.all(valid_depth):
            wilor_kpts = pred["wilor_keypoints_camera"]
            wilor_z_ok = (wilor_kpts[:, 2] >= args.min_depth_m) & (wilor_kpts[:, 2] <= args.max_depth_m)
            fill_mask = (~valid_depth) & wilor_z_ok
            kpts_depth[fill_mask] = wilor_kpts[fill_mask]

        valid_count = int(np.isfinite(kpts_depth[:, 2]).sum())
        if valid_count < args.min_valid_keypoints:
            stats["bad_depth"] += 1
            write_vis_frame(img)
            continue

        kpts_depth_aria = remap_wilor_to_aria(kpts_depth)
        kpts_2d_aria = remap_wilor_to_aria(np.column_stack([pred["keypoints_2d"], np.zeros(21)]))[:, :2]

        T = build_midpoint_pose(kpts_depth_aria, prev_R)
        if T is None:
            stats["bad_pose"] += 1
            write_vis_frame(img)
            continue
        prev_R = T[:3, :3].copy()

        grasp, grasp_ratio = estimate_grasp(kpts_depth_aria, args.grasp_threshold)
        row = {
            "host_time_s": float(frame.get("host_time_s", frame.get("timestamp", frame_index))),
            "frame_index": frame_index,
            "side": args.side,
            "T_hand_to_camera": T,
            "grasp": grasp,
            "confidence": float(pred["score"]),
            "valid_keypoints": valid_count,
            "grasp_ratio": grasp_ratio,
            "detected_side": pred["detected_side"],
            "bbox_xyxy": pred["bbox_xyxy"],
            "keypoints_2d": kpts_2d_aria,
            "keypoints_2d_aria": kpts_2d_aria,
            "keypoints_2d_wilor": pred["keypoints_2d"],
            "keypoints_3d_camera": kpts_depth_aria,
            "keypoints_3d_camera_aria": kpts_depth_aria,
            "keypoints_3d_camera_wilor": kpts_depth,
            "depth_samples_m": depth_samples,
            "depth_sample_counts": sample_counts,
            "wilor_keypoints_camera": pred["wilor_keypoints_camera"],
        }
        rows.append(row)
        stats["written"] += 1

        if args.visualize:
            vis = draw_hand_visualization(
                img,
                K,
                kpts_depth_aria,
                T,
                pred["bbox_xyxy"],
                frame_index,
                args.axis_length_m,
            )
            write_vis_frame(vis)

    smoothed = smooth_binary([float(r["grasp"]) for r in rows], args.grasp_smooth_window)
    for row, grasp in zip(rows, smoothed):
        row["grasp"] = grasp

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), separators=(",", ":"), allow_nan=False) + "\n")

    if vis_writer is not None:
        vis_writer.release()

    stats["output"] = str(out_path)
    if args.visualize:
        stats["visualization"] = str(vis_path)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
