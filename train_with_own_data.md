# 用 RealSense RGB-D + 手/相机位姿训练 HumanEgo

这份文档面向当前工作区：

- HumanEgo 主仓库：`/data/lyx/HumanEgo`
- 自采数据脚本仓库：`/data/lyx/humanego_reproduce`
- 采集脚本：`/data/lyx/humanego_reproduce/realsense_collect_data/read_frames.py`
- 转换脚本：`/data/lyx/humanego_reproduce/realsense_collect_data/prepare_humanego_session.py`

核心目标不是复刻 `sample.vrs` 文件，而是生成 HumanEgo 训练真正读取的最终格式：

```text
<session>/preprocess/all_data/<idx>/training_data.json
```

原版 Aria 路线是：

```text
sample.vrs + MPS SLAM + MPS hand tracking
-> preprocess.Preprocess
-> RGB / camera pose / hand pose / object pose / masks
-> training_data.json
-> training.FlowMatchingTrainer
```

RealSense 路线应走 `RobotPreprocess`：

```text
aligned RGB-D + camera intrinsics + hand pose/grasp (+ camera pose)
-> HumanEgo RobotPreprocess session
-> DINO/SAM + CoTracker + DepthLifter + LaMa + VisualKpts + RobotDatasetGen
-> training_data.json
-> training.FlowMatchingTrainer
```

## 0. 还需要采集什么

你现在脚本原来只采了 `RGB mp4 + depth png`。这不够训练 HumanEgo，因为训练监督需要手部 6DoF 轨迹、抓取状态和物体 3D 位姿。物体 3D 位姿可以由 RGB-D 后处理估计，手部位姿必须额外提供。

每条 demonstration 最低需要：

| 数据 | 必须性 | 用途 | 当前处理方式 |
|---|---:|---|---|
| aligned `rgb.png` | 必须 | 视觉输入、DINO/SAM、CoTracker | `read_frames.py` 已保存 |
| aligned `depth.png` | 必须 | DepthLifter 将 2D object keypoints 反投影到 3D | `read_frames.py` 已保存，转换时变成毫米单位 |
| `camera intrinsics K` | 必须 | 反投影 `u,v,depth -> X,Y,Z` | 从 RealSense SDK 自动写入 `metadata.json` |
| per-frame timestamp | 必须 | 对齐 RGB-D、手位姿、相机位姿 | `frames.jsonl` |
| `T_hand_to_camera` 或 `T_hand_to_world` | 必须 | 训练目标中的手部动作轨迹 | 由 WoVR/外部手部跟踪导出 JSONL |
| `grasp` 或 `gripper_q` | 强烈建议 | 决定是否抓取和物体 latch | 外部估计；没有就只能用默认值跑通 |
| `T_camera_to_world` | 移动相机必须 | 接近 Aria 的 ego camera trajectory | 由 WoVR/SLAM 导出 JSONL |
| task object prompts | 必须 | 指定要分割/跟踪的物体 | 写到 `cfg/preprocess/tasks/<task>.yaml` |

如果你想尽可能贴近 Aria：

- Aria 的 `sample.vrs` 自带多传感器同步、相机标定、IMU、SLAM camera、RGB；MPS 输出 camera/device trajectory 和 hand tracking。
- RealSense 方案要补齐等价信息：RGB-D 时间同步、相机内参、相机轨迹、手部轨迹、抓取状态。
- 当前 HumanEgo `RobotPreprocess` 的 `DepthLifter` 更适合固定 RealSense 相机，因为它默认 `cam0` 是世界坐标。若 RealSense 真的是头戴/手持移动 ego 相机，应仍然采集 `T_camera_to_world`，但当前不改 HumanEgo 主代码时，物体 3D lifting 还不能完全等价 Aria 的 moving-camera triangulation。这是主要差距。

实操建议：第一版先用固定 RealSense 或运动很小的 RealSense，把流程跑通；然后再针对 moving ego camera 增加“使用每帧 camera pose 的 RGB-D object lifting/triangulation”阶段。

## 1. 数据采集脚本修改后会输出什么

`read_frames.py` 已更新为每段录制保存一个 raw session 目录：

```text
/data/lyx/humanego_reproduce/realsense_collect_data/data/
└── 20260617_153000_<serial>/
    ├── metadata.json
    ├── frames.jsonl
    ├── rgb.mp4
    ├── rgb/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── depth/
        ├── 000000.png
        ├── 000001.png
        └── ...
```

`metadata.json` 包含：

- RealSense serial、width、height、fps
- `color_intrinsics.k`
- `depth_intrinsics`
- `depth_scale_meters_per_unit`
- `depth_aligned_to_color: true`
- RGB/depth frame 目录

`frames.jsonl` 每行一帧，包含：

- `frame_index`
- `host_time_s`
- `color_timestamp_ms`
- `depth_timestamp_ms`
- `rgb_path`
- `depth_path`

这些字段用于后续把 WoVR 的手/相机位姿按时间戳对齐到每帧 RGB-D。

## 2. Step-by-step 采集 pipeline

### Step 1. 安装采集依赖

```bash
cd /data/lyx/humanego_reproduce/realsense_collect_data
pip install -r requirements.txt
```

### Step 2. 采集 RGB-D 原始 session

```bash
cd /data/lyx/humanego_reproduce/realsense_collect_data

python read_frames.py \
  --width 640 \
  --height 480 \
  --fps 30 \
  --data-dir ./data
```

交互方式：

- `ENTER`：开始一段 demonstration
- `SPACE`：结束并保存当前 demonstration
- `ESC/q`：退出

采集要求：

1. `rgb.png` 和 `depth.png` 必须来自 `rs.align(rs.stream.color)` 后的 aligned frames。
2. 不要在采集后单独 resize RGB 或 depth；如果必须 resize，`K` 要同步缩放。
3. 每条 demonstration 尽量包含完整任务，从初始状态到完成状态。
4. 同一个 task 至少采 2 条 session：`000` 做 eval，`001+` 做 train。

### Step 3. 同步导出手/相机位姿

你需要从 WoVR 或其他系统导出 JSONL。时间戳要尽可能和 `frames.jsonl` 里的 `host_time_s` 同源；如果不同源，必须先做时间偏移校正。

推荐手部位姿格式：

```jsonl
{"host_time_s": 1790000000.123, "side": "right", "T_hand_to_camera": [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]], "grasp": 0.0}
{"host_time_s": 1790000000.156, "side": "right", "T_hand_to_camera": [[...]], "grasp": 1.0}
```

也支持直接给世界系手位姿：

```jsonl
{"host_time_s": 1790000000.123, "side": "right", "T_hand_to_world": [[...]], "gripper_q": 0.0}
```

推荐相机位姿格式：

```jsonl
{"host_time_s": 1790000000.123, "T_camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
{"host_time_s": 1790000000.156, "T_camera_to_world": [[...]]}
```

字段含义：

- `T_hand_to_camera`：手/手腕/末端在当前 RGB camera 坐标系下的 4x4 pose。
- `T_hand_to_world`：手/手腕/末端在统一 world 坐标系下的 4x4 pose。
- `T_camera_to_world`：当前 RGB camera 到 world 的 4x4 pose。
- `grasp`：1 表示抓取，0 表示未抓取。
- `gripper_q`：HumanEgo robot 约定，0 表示闭合/抓取，1 表示张开。脚本会把 `grasp` 转成 `gripper_q = 1 - grasp`。

如果没有 `grasp/gripper_q`，转换脚本会使用 `--default-gripper-q`。这只能跑通流程，不建议用于正式训练。

## 3. Step-by-step 转换为 HumanEgo session

### Step 4. 转换 raw session

示例：把第一条 session 转成 HumanEgo eval session `teaching_<task>_000`。

```bash
cd /data/lyx/humanego_reproduce/realsense_collect_data

python prepare_humanego_session.py \
  --raw-session ./data/20260617_153000_<serial> \
  --humanego-root /data/lyx/HumanEgo \
  --task serve_bread_rs \
  --index 0 \
  --source-type teaching \
  --side right \
  --hand-poses ./poses/serve_bread_rs_000_hands.jsonl \
  --camera-poses ./poses/serve_bread_rs_000_camera.jsonl \
  --max-pose-dt-s 0.05 \
  --require-hand-pose
```

第二条训练 session：

```bash
python prepare_humanego_session.py \
  --raw-session ./data/20260617_154200_<serial> \
  --humanego-root /data/lyx/HumanEgo \
  --task serve_bread_rs \
  --index 1 \
  --source-type teaching \
  --side right \
  --hand-poses ./poses/serve_bread_rs_001_hands.jsonl \
  --camera-poses ./poses/serve_bread_rs_001_camera.jsonl \
  --max-pose-dt-s 0.05 \
  --require-hand-pose
```

输出目录：

```text
/data/lyx/HumanEgo/data/serve_bread_rs/teaching/
├── teaching_serve_bread_rs_000/
│   └── preprocess/
│       ├── session_meta.json
│       ├── conversion_summary.json
│       └── all_data/
│           ├── 00000/
│           │   ├── rgb.png
│           │   ├── depth.png
│           │   └── robot_state.json
│           └── ...
└── teaching_serve_bread_rs_001/
    └── preprocess/all_data/...
```

转换脚本做了什么：

1. 读取 raw session 的 `metadata.json` 和 `frames.jsonl`。
2. 把 aligned RGB/depth 拷贝到 HumanEgo `preprocess/all_data/<idx>/`。
3. 将 RealSense 原始 depth 按 `depth_scale_meters_per_unit` 转成 HumanEgo `DepthLifter` 期望的毫米 `uint16 depth.png`。
4. 按 `host_time_s` 最近邻匹配 hand pose 和 camera pose。
5. 如果 hand pose 是 `T_hand_to_camera` 且有 `T_camera_to_world`，计算：

   ```text
   T_hand_to_world = T_camera_to_world @ T_hand_to_camera
   ```

6. 写每帧 `robot_state.json`。注意字段名仍叫 `T_ee_in_cam`，但 HumanEgo 的 `RobotDatasetGen` 会把它当成 `T_hand_to_world` 使用。
7. 写 `session_meta.json`，包含 `K`、fps、宽高、帧数、source type。

### Step 5. 写 task 配置

创建：

```text
/data/lyx/HumanEgo/cfg/preprocess/tasks/serve_bread_rs.yaml
```

示例：

```yaml
DINOSAM:
  dinosam_prompt:
    obj1: "piece of bread ."
    obj2: "a plate ."
    arm: "human arms . human hands ."

DepthLifter:
  pose_method:
    obj1: "pca1"
    obj2: "pca1"
    default: "pca1"

RobotDatasetGen:
  finished_tail_frames: 30
  gripper_grasp_threshold: 0.5
```

调试时先确认 `dinosam_mask_obj*.png` 真的分到了正确物体。prompt 写错会导致后续 object pose 全错。

### Step 6. 跑 HumanEgo RobotPreprocess

单条 session：

```bash
cd /data/lyx/HumanEgo

python -m preprocess.RobotPreprocess \
  --session_path ./data/serve_bread_rs/teaching/teaching_serve_bread_rs_000 \
  --task serve_bread_rs \
  --start_from auto
```

批处理全部 teaching sessions：

```bash
python -m preprocess.RobotPreprocess \
  --session_path ./data/serve_bread_rs/teaching \
  --task serve_bread_rs \
  --start_from auto
```

各阶段做什么：

1. `init`：读取 `session_meta.json`，枚举 `all_data/<idx>/rgb.png`。
2. `dinosam`：用 Grounding DINO + SAM2 分割物体和手/臂 mask。
3. `kptsselector`：在参考帧物体 mask 上选 object keypoints。
4. `cotracker`：跨帧跟踪 object keypoints。
5. `depthlifter`：用 `depth.png + K` 将 2D keypoints 反投影到 3D，估计 `T_obj_to_world`。
6. `lama`：生成 `rgb_WoArm.png`。
7. `visualkpts`：生成 `rgb_WArmObjKpts.png` 和 `rgb_WoArm_WArmObjKpts.png`。
8. `datasetgen`：读取 `robot_state.json + depthlifter_results.json`，生成 `training_data.json`。

### Step 7. 检查预处理结果

每条 session 跑完后应有：

```text
preprocess/
├── depthlifter_results.json
├── cotracker_results.json
├── kptsselector_results.json
├── object_centric.ply
├── object_centric.png
└── all_data/
    ├── 00000/
    │   ├── rgb.png
    │   ├── depth.png
    │   ├── robot_state.json
    │   ├── mask_obj1.png
    │   ├── mask_obj2.png
    │   ├── mask_arm.png
    │   ├── rgb_WoArm.png
    │   ├── rgb_WArmObjKpts.png
    │   ├── rgb_WoArm_WArmObjKpts.png
    │   └── training_data.json
    └── ...
```

重点检查：

- `conversion_summary.json` 中 `missing_hand_frames` 必须接近 0。
- `mask_obj*.png` 是否分割正确。
- `depthlifter_results.json` 是否存在，object 3D 尺度是否合理。
- `object_centric.png` / `.ply` 里的手轨迹、物体位置是否在米级尺度。
- 随机打开一个 `training_data.json`，确认：

  ```text
  entities.hands.right.T_hand_to_world
  entities.hands.right.grasp
  entities.objects.obj1.T_obj_to_world
  entities.objects.obj2.T_obj_to_world
  metadata.k
  obs.rgb_WoArm_WArmObjKpts_path
  ```

## 4. 训练

### Step 8. 写训练配置

创建：

```text
/data/lyx/HumanEgo/cfg/training/serve_bread_rs/HumanEgo.yaml
```

可以从 `cfg/training/serve_bread/HumanEgo.yaml` 复制，然后改数据源：

```yaml
single_hand: True
single_hand_side: "right"
hand_tracking_method: "aria_mps"

data_sources:
  teaching: 10
eval_source: "teaching"
data_root: "./data"

img_name: "rgb_WoArm_WArmObjKpts.png"
image_size: [240, 320]
pred_horizon: 50
```

这里 `hand_tracking_method: "aria_mps"` 不是说使用 Aria，而是训练代码会读取 `training_data.json` 里的 `entities.hands` 字段。`RobotDatasetGen` 正好写这个字段。

### Step 9. 启动训练

先 smoke test：

```bash
cd /data/lyx/HumanEgo

python -m training.FlowMatchingTrainer \
  --task serve_bread_rs \
  --use_cfg \
  --job HumanEgo \
  --epochs 5 \
  --data_num 1
```

正式训练：

```bash
python -m training.FlowMatchingTrainer \
  --task serve_bread_rs \
  --use_cfg \
  --job HumanEgo
```

训练器会从：

```text
data/serve_bread_rs/teaching/
```

自动发现 sessions；`eval_source: teaching` 时，第一个 session 做 eval，其余做 train。

## 5. 数据质量和 Aria 接近程度

和 Aria/MPS 对齐的等价关系：

| Aria/MPS | RealSense/WoVR 替代 |
|---|---|
| `sample.vrs` RGB stream | aligned `rgb/*.png` + `rgb.mp4` |
| VRS camera calibration | RealSense `color_intrinsics.k` |
| MPS SLAM `closed_loop_trajectory.csv` | WoVR/SLAM `T_camera_to_world` JSONL |
| MPS hand tracking | WoVR `T_hand_to_camera/world` JSONL |
| MPS timestamps | `frames.jsonl` + pose JSONL timestamps |
| Aria multi-view triangulation | RealSense `depth.png + K` via `DepthLifter` |
| `training_data.json` | `RobotDatasetGen` 输出 |

主要风险：

1. 只有 RGB-D，没有手 pose：不能训练 HumanEgo 手部动作策略。
2. 有手 pose，但没有 grasp：物体 latch 状态不可靠，动态物体轨迹会差。
3. RealSense 移动较大：当前 `RobotPreprocess` 的 `DepthLifter` 静态相机假设会削弱物体 3D 一致性。先用固定 RealSense 跑通，后续再扩展 moving-camera lifting。
4. RGB/depth 未对齐：`DepthLifter` 会在错误像素取深度，物体 3D pose 会错。
5. `K` 不匹配最终图像尺寸：3D 尺度和位置会错。
6. DINO/SAM prompt 不稳定：mask 错了，后面全部错。

## 6. 最小可运行命令总览

采集：

```bash
cd /data/lyx/humanego_reproduce/realsense_collect_data
python read_frames.py --width 640 --height 480 --fps 30 --data-dir ./data
```

转换：

```bash
python prepare_humanego_session.py \
  --raw-session ./data/<raw_session_id> \
  --humanego-root /data/lyx/HumanEgo \
  --task serve_bread_rs \
  --index 0 \
  --source-type teaching \
  --side right \
  --hand-poses ./poses/session_000_hands.jsonl \
  --camera-poses ./poses/session_000_camera.jsonl \
  --require-hand-pose
```

预处理：

```bash
cd /data/lyx/HumanEgo
python -m preprocess.RobotPreprocess --session_path ./data/serve_bread_rs/teaching --task serve_bread_rs --start_from auto
```

训练：

```bash
python -m training.FlowMatchingTrainer --task serve_bread_rs --use_cfg --job HumanEgo
```
