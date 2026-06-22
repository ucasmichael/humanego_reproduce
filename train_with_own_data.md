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

## 当前状态与 TODO

当前策略：**第一阶段默认 RealSense 固定不动**，用 HumanEgo 已有 `RobotPreprocess + DepthLifter` 打通数据链路；运动 RealSense 作为第二阶段扩展，不直接改 HumanEgo 主项目代码。

已具备：

- `read_frames.py` 已能采集 aligned RGB-D、RealSense 内参 `K`、depth scale 和逐帧时间戳。
- `prepare_humanego_session.py` 已能把 raw RealSense session 转成 HumanEgo `RobotPreprocess` 需要的 `session_meta.json + all_data/<idx>/rgb.png/depth.png/robot_state.json`。
- HumanEgo 已有 `RobotPreprocess`，可以从 RGB-D 估计物体 `T_obj_to_world`，再生成 `training_data.json`。

**TODO - 当前必须补齐：**

- **TODO 1：HaWoR/WaVoR 输出每帧 21 个 3D hand keypoints。** 需要有时间戳，最好和 RealSense `frames.jsonl` 的 `host_time_s` 同源。
- **TODO 2：确认 21 keypoints 顺序，并 remap 到 HumanEgo/Aria 约定。** HumanEgo 关键索引假设包括 `0=thumb tip`、`1=index tip`、`5=wrist`、`6=thumb base/MCP`、`8=index MCP`、`11=middle MCP`、`20=palm center`。这个映射错了，末端 pose 和 grasp 都会错。
- **TODO 3：新增/补充一个 keypoints-to-gripper 脚本。** 输入 HaWoR/WaVoR 21 点，输出每帧 `T_hand_to_camera` 或 `T_hand_to_world`，以及 `grasp` 或 `gripper_q`。
- **TODO 4：验证时间同步。** 检查 hand pose 与 RGB-D 帧的最近邻时间差，`missing_hand_frames` 应接近 0，典型阈值先用 `--max-pose-dt-s 0.05`。
- **TODO 5：写 task YAML。** 明确 `obj1/obj2/...` 的 DINO/SAM prompt，否则物体 mask 错了后面全部错。

**TODO - 运动 RealSense 第二阶段：**

- **TODO 6：若 RealSense 是头戴/手持运动相机，必须额外记录 `T_camera_to_world`。** 这一步只是记录还不够，因为当前 HumanEgo `DepthLifter` 默认静态相机。
- **TODO 7：新增 moving-camera object lifting 脚本。** 不改 HumanEgo 主代码的前提下，可以在 `humanego_reproduce` 新增脚本，读取 `cotracker_results.json + depth.png + K + 每帧 T_camera_to_world`，生成与 `depthlifter_results.json` 等价的结果，再从 `RobotPreprocess --start_from lama` 或 `--start_from datasetgen` 继续。
- **TODO 8：或者新增 Aria-like 适配脚本。** 把每帧 `K/c2w/rgb` 写成 `aria_cam_rgb.json` 风格，再复用 `CamTriangulator` 的 SLAM 多视角三角化思路。这个更接近论文，但适配量更大。

## 0. 还需要采集什么

你现在脚本原来只采了 `RGB mp4 + depth png`。这不够训练 HumanEgo，因为训练监督需要手部 6DoF 轨迹、抓取状态和物体 3D 位姿。物体 3D 位姿可以由 RGB-D 后处理估计，手部位姿必须额外提供。

每条 demonstration 最低需要：

| 数据 | 必须性 | 用途 | 当前处理方式 |
|---|---:|---|---|
| aligned `rgb.png` | 必须 | 视觉输入、DINO/SAM、CoTracker | `read_frames.py` 已保存 |
| aligned `depth.png` | 必须 | DepthLifter 将 2D object keypoints 反投影到 3D | `read_frames.py` 已保存，转换时变成毫米单位 |
| `camera intrinsics K` | 必须 | 反投影 `u,v,depth -> X,Y,Z` | 从 RealSense SDK 自动写入 `metadata.json` |
| per-frame timestamp | 必须 | 对齐 RGB-D、手位姿、相机位姿 | `frames.jsonl` |
| `T_hand_to_camera` 或 `T_hand_to_world` | 必须 | 训练目标中的手部动作轨迹 | 由 HaWoR/外部手部跟踪导出 JSONL |
| `grasp` 或 `gripper_q` | 强烈建议 | 决定是否抓取和物体 latch | 外部估计；没有就只能用默认值跑通 |
| `T_camera_to_world` | 移动相机必须 | 接近 Aria 的 ego camera trajectory | 由 HaWoR/SLAM 导出 JSONL |
| task object prompts | 必须 | 指定要分割/跟踪的物体 | 写到 `cfg/preprocess/tasks/<task>.yaml` |

手部 21 点到末端控制量的计划：

```text
HaWoR/WaVoR 21 keypoints
-> remap 到 HumanEgo/Aria keypoint order
-> thumb/index tips 算 midpoint
-> thumb/index MCP + wrist 构造 gripper-like orientation
-> thumb-index distance / palm size 算 grasp
-> 输出 hand pose JSONL，供 prepare_humanego_session.py 对齐
```

推荐输出格式：

```jsonl
{"host_time_s": 1790000000.123, "side": "right", "T_hand_to_camera": [[...]], "grasp": 0.0}
{"host_time_s": 1790000000.156, "side": "right", "T_hand_to_camera": [[...]], "grasp": 1.0}
```

**TODO：** 目前还需要新增这个 “21 keypoints -> `T_hand_to_camera/world` + `grasp`” 脚本；现有转换脚本假设这一步已经完成。

如果你想尽可能贴近 Aria：

- Aria 的 `sample.vrs` 自带多传感器同步、相机标定、IMU、SLAM camera、RGB；MPS 输出 camera/device trajectory 和 hand tracking。
- RealSense 方案要补齐等价信息：RGB-D 时间同步、相机内参、相机轨迹、手部轨迹、抓取状态。
- 当前 HumanEgo `RobotPreprocess` 的 `DepthLifter` 更适合固定 RealSense 相机，因为它默认 `cam0` 是世界坐标。若 RealSense 真的是头戴/手持移动 ego 相机，应仍然采集 `T_camera_to_world`，但当前不改 HumanEgo 主代码时，物体 3D lifting 还不能完全等价 Aria 的 moving-camera triangulation。这是主要差距。

实操建议：第一版先用固定 RealSense，把 `RobotPreprocess` 原样跑通；然后再针对 moving ego camera 增加“使用每帧 camera pose 的 RGB-D object lifting/triangulation”阶段。

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

这些字段用于后续把 HaWoR 的手/相机位姿按时间戳对齐到每帧 RGB-D。

当前固定 RealSense 约定：

```text
world = RealSense RGB camera frame
T_camera_to_world = Identity
```

在这个约定下，如果 HaWoR 输出的是 `T_hand_to_camera`，它也可以直接作为训练用的 `T_hand_to_world` 写入 `robot_state.json`。转换脚本仍支持传入 `--camera-poses`，但固定相机第一阶段可以不提供，默认用 identity。

运动 RealSense 约定：

```text
world = 第一帧相机坐标系或 SLAM/world 坐标系
每帧必须有 T_camera_to_world
T_hand_to_world = T_camera_to_world @ T_hand_to_camera
```

**TODO：** 运动 RealSense 时，手部 pose 可以通过转换脚本变到 world；但物体 `T_obj_to_world` 仍需要新增 moving-camera object lifting，不能完全依赖当前静态 `DepthLifter`。

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

你需要从 HaWoR 或其他系统导出 JSONL。时间戳要尽可能和 `frames.jsonl` 里的 `host_time_s` 同源；如果不同源，必须先做时间偏移校正。

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

### Step 3.1. 将 21 keypoints 转成末端位姿和 gripper

这是当前最关键的待补脚本，不属于 HumanEgo 主项目，建议新增在：

```text
/data/lyx/humanego_reproduce/realsense_collect_data/hand_keypoints_to_gripper.py
```

输入建议：

```jsonl
{"host_time_s": 1790000000.123, "side": "right", "keypoints_3d_camera": [[x,y,z], ... 21 points ...], "confidence": 0.9}
```

输出建议：

```jsonl
{"host_time_s": 1790000000.123, "side": "right", "T_hand_to_camera": [[...]], "grasp": 0.0, "confidence": 0.9}
```

算法计划：

1. `remap_keypoints()`：把 HaWoR/WaVoR keypoint order 转到 HumanEgo/Aria order。
2. `build_gripper_pose()`：用 `thumb_tip/index_tip/wrist/thumb_base/index_base` 构造 midpoint frame。
3. `estimate_grasp()`：用 `||thumb_tip - index_tip|| / ||middle_mcp - wrist||` 判断 grasp，初始阈值用 `1.0`，后续根据数据调。
4. `smooth_pose_and_grasp()`：可选，先做简单滑动窗口/中值滤波，避免 grasp 抖动。
5. 写 hand pose JSONL 给 `prepare_humanego_session.py`。

**TODO：** 需要用一小段真实 HaWoR/WaVoR 输出确认 keypoint order 和坐标单位。如果输出是 root-relative hand frame，而不是 camera/world frame，还需要额外把它恢复到 camera/world 坐标，否则不能直接训练。

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

固定 RealSense 情况：

- 可以不传 `--camera-poses`，转换脚本默认 `T_camera_to_world = I`。
- 若 hand pose JSONL 里是 `T_hand_to_camera`，最终会等价写成 `T_hand_to_world`，因为 `world = camera frame`。

运动 RealSense 情况：

- 必须传 `--camera-poses`，否则手 pose 无法稳定对齐到 world。
- 转换脚本能处理手部：`T_hand_to_world = T_camera_to_world @ T_hand_to_camera`。
- **TODO：** 当前转换脚本不会改变 HumanEgo `DepthLifter` 的静态相机假设；运动相机时还需要新增 object lifting 结果生成脚本，见 Step 6.1。

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

固定 RealSense：

- 当前 `RobotPreprocess` 原样支持。
- `DepthLifter` 使用 `depth.png + K` 反投影物体 2D keypoints，默认 `cam0_c2w = Identity`。

运动 RealSense：

- 当前 `RobotPreprocess` 可以跑，但 `DepthLifter` 的物体世界坐标会隐含静态相机假设，不等价 Aria SLAM。
- **TODO：** 不修改 HumanEgo 项目代码时，建议新增外部脚本生成 `preprocess/depthlifter_results.json`，然后让 `RobotPreprocess` 从后续阶段继续。

### Step 6.1. 运动 RealSense 需要新增的 object lifting 步骤

仅当 RealSense 运动时需要。固定 RealSense 跳过本节。

当前 `DepthLifter` 做的是：

```text
2D object keypoint (u, v)
+ depth.png[v, u]
+ K
-> p_cam
-> p_world = p_cam   # 静态相机假设
```

运动 RealSense 应改成：

```text
2D object keypoint (u, v)
+ depth.png[v, u]
+ K
-> p_cam_t
-> p_world = T_camera_t_to_world @ p_cam_t
```

建议新增脚本，不改 HumanEgo 主代码：

```text
/data/lyx/humanego_reproduce/realsense_collect_data/moving_depthlifter_results.py
```

输入：

```text
HumanEgo session/preprocess/cotracker_results.json
HumanEgo session/preprocess/all_data/<idx>/depth.png
HumanEgo session/preprocess/all_data/<idx>/robot_state.json  # 里面已有 camera.T_camera_to_world
HumanEgo session/preprocess/session_meta.json                # K
```

输出：

```text
HumanEgo session/preprocess/depthlifter_results.json
```

然后从后续阶段继续：

```bash
cd /data/lyx/HumanEgo
python -m preprocess.RobotPreprocess \
  --session_path ./data/serve_bread_rs/teaching/teaching_serve_bread_rs_000 \
  --task serve_bread_rs \
  --start_from lama
```

**TODO：** 这个 moving depthlifter 还未实现；当前文档第一阶段以固定 RealSense 为准。

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

| Aria/MPS | RealSense/HaWoR 替代 |
|---|---|
| `sample.vrs` RGB stream | aligned `rgb/*.png` + `rgb.mp4` |
| VRS camera calibration | RealSense `color_intrinsics.k` |
| MPS SLAM `closed_loop_trajectory.csv` | HaWoR/SLAM `T_camera_to_world` JSONL |
| MPS hand tracking | HaWoR `T_hand_to_camera/world` JSONL |
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

当前阶段的明确缺口：

- **TODO：HaWoR/WaVoR 21 keypoints -> gripper pose/grasp 脚本未实现。** 没有这个，`prepare_humanego_session.py` 缺少正式 hand pose 输入。
- **TODO：keypoint remapping 未验证。** 必须用真实 21 点样例确认 thumb/index/wrist/MCP/palm center 的索引。
- **TODO：grasp 阈值未标定。** 初始可以用相对阈值 `thumb-index distance / palm_size < 1.0`，但需要看视频/曲线调。
- **TODO：运动 RealSense 的 object lifting 未实现。** 固定 RealSense 可用现有 `DepthLifter`；运动相机要新增外部脚本生成 `depthlifter_results.json`。

## 6. 最小可运行命令总览

下面以 `pick_banana_into_the_pot` 为例，当前完整链路是：

```text
RealSense RGB-D raw session
-> WiLoR + aligned depth 生成 hand_poses_wilor.jsonl
-> prepare_humanego_session.py 转成 HumanEgo RobotPreprocess session
-> RobotPreprocess 生成 training_data.json
-> FlowMatchingTrainer 训练
```

### 6.1 采集 RealSense RGB-D

```bash
cd /data/lyx/humanego_reproduce/realsense_collect_data

python read_frames.py \
  --width 640 \
  --height 480 \
  --fps 15 \
  --data-dir /data/lyx/datasets/egodata/data0617_depth_ir
```

采集结束后得到类似：

```text
/data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934/
├── metadata.json
├── frames.jsonl
├── rgb.mp4
├── rgb/
└── depth/
```

### 6.2 生成手部 pose JSONL

使用 WiLoR 检测 21 点，并用 RealSense aligned depth 反投影到相机坐标系，输出 HumanEgo 转换脚本需要的：

```jsonl
{"host_time_s": ..., "side": "right", "T_hand_to_camera": [[...]], "grasp": ...}
```

```bash
cd /data/lyx/WiLoR

/data/lyx/miniconda3/envs/wilor/bin/python export_realsense_humanego_handposes.py \
  --session /data/lyx/datasets/egodata/data0617_depth_ir/<raw_session_id> \
  --out /data/lyx/datasets/egodata/data0617_depth_ir/<raw_session_id>/hand_poses_wilor.jsonl \
  --side right \
  --visualize
```

调试时可以先只跑几帧：

```bash
/data/lyx/miniconda3/envs/wilor/bin/python export_realsense_humanego_handposes.py \
  --session /data/lyx/datasets/egodata/data0617_depth_ir/<raw_session_id> \
  --out /tmp/hand_poses_wilor_debug.jsonl \
  --vis-out /tmp/hand_poses_wilor_debug.mp4 \
  --start-frame 30 \
  --max-frames 30 \
  --side right \
  --visualize
```

检查生成的 `<raw_session_id>/hand_poses_wilor_vis.mp4`：红点应落在 `wrist / thumb MCP / thumb tip / index MCP / index tip`，RGB 箭头表示导出的 `T_hand_to_camera`。

### 6.3 转换为 HumanEgo dataset session

```bash
/data/lyx/miniconda3/envs/wilor/bin/python /data/lyx/humanego_reproduce/realsense_collect_data/prepare_humanego_session.py \
  --raw-session /data/lyx/datasets/egodata/data0617_depth_ir/<raw_session_id> \
  --humanego-root /data/lyx/HumanEgo \
  --task pick_banana_into_the_pot \
  --index <session_index> \
  --source-type teaching \
  --side right \
  --hand-poses /data/lyx/datasets/egodata/data0617_depth_ir/<raw_session_id>/hand_poses_wilor.jsonl \
  --force
```

当前已跑过的两个例子：

```bash
/data/lyx/miniconda3/envs/wilor/bin/python /data/lyx/humanego_reproduce/realsense_collect_data/prepare_humanego_session.py \
  --raw-session /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934 \
  --humanego-root /data/lyx/HumanEgo \
  --task pick_banana_into_the_pot \
  --index 1 \
  --source-type teaching \
  --side right \
  --hand-poses /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155420_318122300934/hand_poses_wilor.jsonl \
  --force

/data/lyx/miniconda3/envs/wilor/bin/python /data/lyx/humanego_reproduce/realsense_collect_data/prepare_humanego_session.py \
  --raw-session /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155442_318122300934 \
  --humanego-root /data/lyx/HumanEgo \
  --task pick_banana_into_the_pot \
  --index 2 \
  --source-type teaching \
  --side right \
  --hand-poses /data/lyx/datasets/egodata/data0617_depth_ir/20260617_155442_318122300934/hand_poses_wilor.jsonl \
  --force
```

输出目录：

```text
/data/lyx/HumanEgo/data/pick_banana_into_the_pot/teaching/teaching_pick_banana_into_the_pot_001/
/data/lyx/HumanEgo/data/pick_banana_into_the_pot/teaching/teaching_pick_banana_into_the_pot_002/
```

### 6.4 写/检查 task 预处理配置

`RobotPreprocess --task pick_banana_into_the_pot` 会读取：

```text
/data/lyx/HumanEgo/cfg/preprocess/tasks/pick_banana_into_the_pot.yaml
```

当前应为：

```yaml
DINOSAM:
  dinosam_prompt:
    obj1: "a pot ."
    obj2: "a banana ."
    arm: "robot arms . robot hands ."
```

这里建议 `obj1=pot`，`obj2=banana`。HumanEgo 会把第一个 object 当作 `virtual_static_anchor`，锅更适合作为静态容器/锚点，香蕉更适合作为动态物体。

### 6.5 跑 HumanEgo RobotPreprocess

```bash
cd /data/lyx/HumanEgo

python -m preprocess.RobotPreprocess \
  --session_path /data/lyx/HumanEgo/data/pick_banana_into_the_pot/teaching/teaching_pick_banana_into_the_pot_001 \
  --task pick_banana_into_the_pot \
  --start_from init
```

批量处理 `teaching/` 下所有 session：

```bash
python -m preprocess.RobotPreprocess \
  --session_path /data/lyx/HumanEgo/data/pick_banana_into_the_pot/teaching \
  --task pick_banana_into_the_pot \
  --start_from auto
```

如果只改了 DINO/SAM prompt，且 `preprocess/cfg/` 已经存在，可以从 `dinosam` 重新跑：

```bash
python -m preprocess.RobotPreprocess \
  --session_path /data/lyx/HumanEgo/data/pick_banana_into_the_pot/teaching/teaching_pick_banana_into_the_pot_001 \
  --task pick_banana_into_the_pot \
  --start_from dinosam
```

跑完重点检查：

```text
preprocess/dinosam_mask_obj1.png       # pot
preprocess/dinosam_mask_obj2.png       # banana
preprocess/kptsselector_results.json   # objects 里应有 obj1 和 obj2
preprocess/depthlifter_results.json    # objects 里应有 obj1 和 obj2
preprocess/all_data/00000/training_data.json
```

### 6.6 写训练配置并训练

如果还没有训练配置，先创建：

```bash
mkdir -p /data/lyx/HumanEgo/cfg/training/pick_banana_into_the_pot
cp /data/lyx/HumanEgo/cfg/training/serve_bread/HumanEgo.yaml \
   /data/lyx/HumanEgo/cfg/training/pick_banana_into_the_pot/HumanEgo.yaml
```

然后打开：

```text
/data/lyx/HumanEgo/cfg/training/pick_banana_into_the_pot/HumanEgo.yaml
```

至少确认：

```yaml
single_hand: True
single_hand_side: "right"
hand_tracking_method: "aria_mps"
data_sources:
  teaching: 10
eval_source: "teaching"
data_root: "./data"
```

启动 smoke test：

```bash
cd /data/lyx/HumanEgo

python -m training.FlowMatchingTrainer \
  --task pick_banana_into_the_pot \
  --use_cfg \
  --job HumanEgo \
  --epochs 5 \
  --data_num 1
```

正式训练：

```bash
python -m training.FlowMatchingTrainer \
  --task pick_banana_into_the_pot \
  --use_cfg \
  --job HumanEgo
```
