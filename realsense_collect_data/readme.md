# RealSense 数据采集与 HumanEgo 转换

这个目录提供两步脚本：

1. `read_frames.py`：采集 RealSense aligned RGB-D 原始 session。
2. `prepare_humanego_session.py`：把原始 session 加外部手/相机位姿 JSONL 转成 HumanEgo `RobotPreprocess` 可读格式。

## 环境安装

```bash
cd /data/lyx/humanego_reproduce/realsense_collect_data
pip install -r requirements.txt
```

## Step 1. 采集 RealSense RGB-D

```bash
python read_frames.py --width 640 --height 480 --fps 30 --data-dir ./data
```

交互方式：

- `ENTER`：开始单条 demonstration
- `SPACE`：结束并保存当前 demonstration
- `ESC/q`：退出采集

常用参数：

- `--serial`：指定 RealSense 序列号；不指定时会列出设备。
- `--width` / `--height` / `--fps`：采集分辨率和帧率。
- `--no-depth`：只保存 RGB，不建议用于 HumanEgo。
- `--no-frame-images`：不保存逐帧 RGB PNG，只保存 `rgb.mp4`；不建议用于 HumanEgo。

每段录制输出：

```text
data/<timestamp>_<serial>/
├── metadata.json
├── frames.jsonl
├── rgb.mp4
├── rgb/000000.png ...
└── depth/000000.png ...
```

`metadata.json` 包含 RealSense 内参 `color_intrinsics.k`、depth scale、分辨率和对齐信息。`frames.jsonl` 包含逐帧时间戳和 RGB/depth 路径。

## Step 2. 准备外部位姿 JSONL

正式训练还需要每帧手位姿和抓取状态。推荐从 WoVR 或其他手部跟踪系统导出：

```jsonl
{"host_time_s": 1790000000.123, "side": "right", "T_hand_to_camera": [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]], "grasp": 0.0}
{"host_time_s": 1790000000.156, "side": "right", "T_hand_to_camera": [[...]], "grasp": 1.0}
```

如果相机移动，还应导出相机位姿：

```jsonl
{"host_time_s": 1790000000.123, "T_camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

时间戳最好和 `frames.jsonl` 的 `host_time_s` 同源。否则需要先做时间偏移校正。

## Step 3. 转成 HumanEgo session

```bash
python prepare_humanego_session.py \
  --raw-session ./data/<timestamp>_<serial> \
  --humanego-root /data/lyx/HumanEgo \
  --task serve_bread_rs \
  --index 0 \
  --source-type teaching \
  --side right \
  --hand-poses ./poses/session_000_hands.jsonl \
  --camera-poses ./poses/session_000_camera.jsonl \
  --require-hand-pose
```

输出目录：

```text
/data/lyx/HumanEgo/data/<task>/teaching/teaching_<task>_<idx>/
└── preprocess/
    ├── session_meta.json
    ├── conversion_summary.json
    └── all_data/00000/rgb.png depth.png robot_state.json ...
```

然后进入 HumanEgo 主仓库继续：

```bash
cd /data/lyx/HumanEgo
python -m preprocess.RobotPreprocess --session_path ./data/<task>/teaching --task <task> --start_from auto
```

完整训练说明见：

```text
/data/lyx/HumanEgo/train_with_own_data.md
```
