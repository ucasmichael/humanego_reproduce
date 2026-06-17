# -- coding: UTF-8
"""
Read, preview, and manually record frames from a locally connected Intel RealSense camera.

Usage examples:
    python read_frames.py
    python read_frames.py --serial 123456789012
    python read_frames.py --width 1280 --height 720 --fps 30
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not installed. Install it with: pip install pyrealsense2")
    sys.exit(1)


def list_realsense_devices():
    """Return connected RealSense devices with useful info."""
    context = rs.context()
    devices = []

    for device in context.query_devices():
        def get_info(info_type, default="Unknown"):
            if device.supports(info_type):
                return device.get_info(info_type)
            return default

        devices.append(
            {
                "name": get_info(rs.camera_info.name),
                "serial": get_info(rs.camera_info.serial_number),
                "firmware": get_info(rs.camera_info.firmware_version),
                "usb": get_info(rs.camera_info.usb_type_descriptor),
            }
        )

    return devices


def choose_serial(devices, requested_serial=None):
    """Validate a requested serial or choose the only connected device."""
    if not devices:
        raise RuntimeError("No RealSense device found. Please check USB/power connection.")

    print("Detected RealSense device(s):")
    for index, device in enumerate(devices):
        print(
            f"  [{index}] name={device['name']}  serial={device['serial']}  "
            f"firmware={device['firmware']}  usb={device['usb']}"
        )

    serials = [device["serial"] for device in devices]
    if requested_serial:
        if requested_serial not in serials:
            raise RuntimeError(
                f"Requested serial {requested_serial} was not found. "
                f"Available serials: {', '.join(serials)}"
            )
        return requested_serial

    if len(devices) == 1:
        return devices[0]["serial"]

    print("\nMultiple cameras found. Use --serial to select one explicitly.")
    return None


def start_pipeline(serial, width, height, fps, enable_depth):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if enable_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # Let auto exposure settle for a moment before showing frames.
    time.sleep(0.5)
    return pipeline, profile


def get_depth_scale(profile):
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()
    except RuntimeError:
        return None


def rs_intrinsics_to_dict(intrinsics):
    """Convert pyrealsense2 intrinsics into a JSON-serializable dict."""
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(x) for x in intrinsics.coeffs],
        "k": [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
    }


def get_stream_intrinsics(profile, stream):
    try:
        return rs_intrinsics_to_dict(profile.get_stream(stream).as_video_stream_profile().get_intrinsics())
    except RuntimeError:
        return None


def sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "camera"


def make_depth_preview(depth_image):
    depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def draw_status(preview, recording, recorded_frames, elapsed):
    if recording:
        status = f"REC {recorded_frames} frames  {elapsed:.1f}s  SPACE: stop/save  ESC: quit"
        color = (0, 0, 255)
    else:
        status = "READY  ENTER: start recording  ESC/q: quit"
        color = (0, 255, 0)

    cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        preview,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return preview


class RecordingSession:
    def __init__(
        self,
        serial,
        width,
        height,
        fps,
        enable_depth,
        depth_scale,
        data_dir,
        color_intrinsics,
        depth_intrinsics,
        save_frame_images=True,
    ):
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        serial_name = sanitize_name(serial)

        self.session_id = f"{session_id}_{serial_name}"
        self.session_dir = Path(data_dir) / self.session_id
        self.rgb_frame_dir = self.session_dir / "rgb"
        self.depth_frame_dir = self.session_dir / "depth"
        self.rgb_path = self.session_dir / "rgb.mp4"
        self.metadata_path = self.session_dir / "metadata.json"
        self.frames_path = self.session_dir / "frames.jsonl"
        self.enable_depth = enable_depth
        self.save_frame_images = save_frame_images
        self.color_intrinsics = color_intrinsics
        self.depth_intrinsics = depth_intrinsics

        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.save_frame_images:
            self.rgb_frame_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_depth:
            self.depth_frame_dir.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.rgb_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open RGB video writer: {self.rgb_path}")

        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale = depth_scale
        self.started_at = datetime.now()
        self.started_time = time.time()
        self.frame_count = 0
        self.depth_frame_count = 0
        self.missing_depth_count = 0
        self.closed = False
        self.frames_file = self.frames_path.open("w", encoding="utf-8")

        self._write_metadata(finished=False)

    def write(self, color_image, depth_image, color_frame=None, depth_frame=None):
        frame_index = self.frame_count
        self.writer.write(color_image)
        self.frame_count += 1

        rgb_rel = None
        if self.save_frame_images:
            rgb_path = self.rgb_frame_dir / f"{frame_index:06d}.png"
            ok = cv2.imwrite(str(rgb_path), color_image)
            if not ok:
                raise RuntimeError(f"Could not write RGB frame: {rgb_path}")
            rgb_rel = rgb_path.relative_to(self.session_dir).as_posix()

        depth_rel = None
        has_depth = False

        if self.enable_depth and depth_image is None:
            self.missing_depth_count += 1
        elif self.enable_depth:
            # Store raw RealSense z16 depth. Convert to millimetres later with
            # raw_value * depth_scale_meters_per_unit * 1000.
            depth_path = self.depth_frame_dir / f"{frame_index:06d}.png"
            ok = cv2.imwrite(str(depth_path), depth_image)
            if not ok:
                raise RuntimeError(f"Could not write depth frame: {depth_path}")
            depth_rel = depth_path.relative_to(self.session_dir).as_posix()
            self.depth_frame_count += 1
            has_depth = True

        frame_record = {
            "frame_index": frame_index,
            "host_time_s": time.time(),
            "rgb_path": rgb_rel,
            "depth_path": depth_rel,
            "has_depth": has_depth,
        }
        if color_frame is not None:
            frame_record.update(
                {
                    "color_timestamp_ms": float(color_frame.get_timestamp()),
                    "color_frame_number": int(color_frame.get_frame_number()),
                }
            )
        if depth_frame is not None:
            frame_record.update(
                {
                    "depth_timestamp_ms": float(depth_frame.get_timestamp()),
                    "depth_frame_number": int(depth_frame.get_frame_number()),
                }
            )

        self.frames_file.write(json.dumps(frame_record) + "\n")

    def close(self):
        if self.closed:
            return self.summary()

        self.writer.release()
        self.frames_file.close()
        self.closed = True
        self._write_metadata(finished=True)
        return self.summary()

    def summary(self):
        elapsed = time.time() - self.started_time
        return {
            "session_dir": self.session_dir.resolve(),
            "rgb_path": self.rgb_path.resolve(),
            "rgb_frame_dir": self.rgb_frame_dir.resolve() if self.save_frame_images else None,
            "depth_dir": self.depth_frame_dir.resolve() if self.enable_depth else None,
            "metadata_path": self.metadata_path.resolve(),
            "frames_path": self.frames_path.resolve(),
            "depth_enabled": self.enable_depth,
            "rgb_frames": self.frame_count,
            "depth_frames": self.depth_frame_count,
            "missing_depth_frames": self.missing_depth_count,
            "duration_seconds": elapsed,
            "average_fps": self.frame_count / elapsed if elapsed > 0 else 0.0,
        }

    def _write_metadata(self, finished):
        metadata = {
            "session_id": self.session_id,
            "serial": self.serial,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "rgb_video": self.rgb_path.as_posix(),
            "rgb_frame_dir": self.rgb_frame_dir.as_posix() if self.save_frame_images else None,
            "depth_enabled": self.enable_depth,
            "depth_dir": self.depth_frame_dir.as_posix() if self.enable_depth else None,
            "depth_format": "16-bit PNG, raw RealSense z16 units" if self.enable_depth else None,
            "depth_scale_meters_per_unit": self.depth_scale,
            "depth_aligned_to_color": self.enable_depth,
            "color_intrinsics": self.color_intrinsics,
            "depth_intrinsics": self.depth_intrinsics,
            "humanego_note": "Use color_intrinsics.k with aligned depth. Convert depth PNG to millimetres before HumanEgo DepthLifter if depth_scale != 0.001.",
            "frames_jsonl": self.frames_path.as_posix(),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished": finished,
            "rgb_frames": self.frame_count,
            "depth_frames": self.depth_frame_count,
            "missing_depth_frames": self.missing_depth_count,
        }
        if finished:
            metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
            metadata["duration_seconds"] = time.time() - self.started_time

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def print_recording_summary(summary):
    print("\nRecording saved:")
    print(f"  Session dir:    {summary['session_dir']}")
    print(f"  RGB video:      {summary['rgb_path']}")
    if summary["rgb_frame_dir"] is not None:
        print(f"  RGB frames:     {summary['rgb_frame_dir']}")
    if summary["depth_dir"] is not None:
        print(f"  Depth frames:   {summary['depth_dir']}")
    else:
        print("  Depth frames:   disabled")
    print(f"  Metadata:       {summary['metadata_path']}")
    print(f"  Frame manifest: {summary['frames_path']}")
    print(f"  RGB frames:     {summary['rgb_frames']}")
    if summary["depth_enabled"]:
        print(f"  Depth frames:   {summary['depth_frames']}")
        print(f"  Missing depth:  {summary['missing_depth_frames']}")
    print(f"  Duration:       {summary['duration_seconds']:.2f} s")
    print(f"  Average FPS:    {summary['average_fps']:.2f}")


def preview_and_record(pipeline, profile, args, serial):
    align = rs.align(rs.stream.color)
    window_name = "RealSense Preview - ENTER start, SPACE stop/save"
    session = None
    enable_depth = not args.no_depth
    depth_scale = get_depth_scale(profile) if enable_depth else None
    color_intrinsics = get_stream_intrinsics(profile, rs.stream.color)
    depth_intrinsics = get_stream_intrinsics(profile, rs.stream.depth) if enable_depth else None
    segment_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if enable_depth:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            if enable_depth:
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            else:
                depth_image = None

            if session is not None:
                session.write(color_image, depth_image, color_frame=color_frame, depth_frame=depth_frame if enable_depth else None)

            if depth_image is not None:
                depth_preview = make_depth_preview(depth_image)
                preview = np.hstack((color_image, depth_preview))
            else:
                preview = color_image.copy()

            elapsed = time.time() - session.started_time if session is not None else 0.0
            recorded_frames = session.frame_count if session is not None else 0
            preview = draw_status(preview.copy(), session is not None, recorded_frames, elapsed)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (10, 13):
                if session is None:
                    session = RecordingSession(
                        serial=serial,
                        width=args.width,
                        height=args.height,
                        fps=args.fps,
                        enable_depth=enable_depth,
                        depth_scale=depth_scale,
                        data_dir=args.data_dir,
                        color_intrinsics=color_intrinsics,
                        depth_intrinsics=depth_intrinsics,
                        save_frame_images=not args.no_frame_images,
                    )
                    segment_count += 1
                    print("\nRecording started. Press SPACE in the preview window to stop and save.")
                else:
                    print("Recording is already running. Press SPACE to stop and save.")

            elif key == 32:
                if session is None:
                    print("Recording has not started yet. Press ENTER first.")
                else:
                    summary = session.close()
                    print_recording_summary(summary)
                    session = None
                    print("\nReady for next recording. Press ENTER to start another segment, or ESC/q to quit.")

            elif key in (27, ord("q")):
                if session is not None:
                    summary = session.close()
                    print("\nRecording stopped by exit key; current segment was saved.")
                    print_recording_summary(summary)
                    session = None
                break
    finally:
        if session is not None and not session.closed:
            summary = session.close()
            print("\nRecording interrupted; partial data was saved.")
            print_recording_summary(summary)

        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nExited recording loop. Saved segments: {segment_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview and manually record frames from a local RealSense camera.")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number to use.")
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS.")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory.")
    parser.add_argument("--no-depth", action="store_true", help="Record RGB only; do not stream or save depth frames.")
    parser.add_argument("--no-frame-images", action="store_true", help="Do not save per-frame RGB PNGs; keeps only rgb.mp4 plus depth/metadata.")
    parser.add_argument("--depth", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    devices = list_realsense_devices()
    serial = choose_serial(devices, args.serial)
    if serial is None:
        return 2

    print(f"\nUsing RealSense serial: {serial}")
    print("Opening camera preview.")
    print("Press ENTER in the preview window to start recording.")
    print("Press SPACE in the preview window to stop and save the current segment.")
    print("Press ESC or q in the preview window to exit the recording loop.")
    if args.no_depth:
        print("Depth recording is disabled; RGB only mode is active.")

    pipeline, profile = start_pipeline(serial, args.width, args.height, args.fps, not args.no_depth)
    preview_and_record(pipeline, profile, args, serial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
