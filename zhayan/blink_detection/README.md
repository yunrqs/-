# 眨眼识别模块（Blink Detection）

基于 **MediaPipe Face Mesh + EAR（Eye Aspect Ratio）** 的实时眨眼检测模块，用于从连续视频帧中识别完整的 `Open → Closing → Closed → Open` 过程，并在一次完整闭眼结束后输出一次 `blink_detected=True`。

本模块严格遵循 `../眨眼识别模块——论文复现与代码生成规格说明.md` 实现。

---

## 1. 核心链路

```text
输入视频帧
    ↓
MediaPipe Face Mesh
    ↓
人脸 468 点关键点
    ↓
提取左右眼 6 个关键点
    ↓
分别计算 Left EAR / Right EAR
    ↓
计算双眼 EAR
    ↓
自适应 EAR 基线与动态阈值
    ↓
时序状态机
    ↓
输出 blink event
```

---

## 2. 目录结构

```text
blink_detection/
├── main.py                 # 入口：摄像头 / 视频文件
├── blink_detector.py       # BlinkDetector + BlinkStateMachine
├── face_mesh.py            # MediaPipe Face Mesh 封装
├── ear.py                  # EAR 计算（landmark -> EAR）
├── adaptive_threshold.py   # baseline 初始化 / 更新 / threshold 计算
├── blink_types.py          # EyeState / BlinkResult
├── visualization.py        # 调试可视化（overlay）
├── csv_logger.py           # CSV 记录
├── config.py               # 集中配置 BlinkConfig
├── requirements.txt
├── test_blink_detector.py  # 单元测试
└── README.md
```

> 说明：规格说明第 20 节建议的文件名为 `types.py`，但 `types` 与 Python 标准库
> 模块同名，直接命名会遮蔽标准库并破坏 `typing`/`dataclasses` 的导入。这里改为
> `blink_types.py`，仅调整文件名，职责与接口不变。

---

## 3. 安装

```bash
pip install -r requirements.txt
```

本模块**不依赖** PyTorch / TensorFlow / dlib，仅需：

- `mediapipe`（Face Mesh）
- `opencv-python`（图像 I/O 与显示）
- `numpy`（几何计算）

### MediaPipe 后端说明

`face_mesh.py` 兼容两种 MediaPipe API：

1. **legacy 后端**：优先使用规格说明第 4 节要求的
   `mp.solutions.face_mesh.FaceMesh(...)`（旧版 mediapipe，无需模型文件）；
2. **tasks 后端**：当 `mp.solutions` 不可用时（mediapipe >= 0.10.35 已移除旧
   API），自动回退到 `mediapipe.tasks.python.vision.FaceLandmarker`，并在首次
   运行时自动下载官方 `face_landmarker.task` 模型（约 3.6 MB）到
   `~/.cache/blink_detection/`。

如需使用本地模型，可通过 `BlinkConfig(model_asset_path="path/to/face_landmarker.task")`
指定。

---

## 4. 快速开始

### 4.1 摄像头

```bash
cd blink_detection
python main.py
```

### 4.2 摄像头 + 调试 overlay + 保存 CSV

```bash
python main.py --debug --record
```

### 4.3 视频文件

```bash
python main.py --source demo.mp4 --debug
```

### 4.4 robust 模式

```bash
python main.py --mode robust --debug
```

运行时按 `q` 退出。

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--source` | `0` | 摄像头索引（`0`）或视频文件路径 |
| `--mode` | `paper` | `paper`=论文复现；`robust`=工程增强 |
| `--debug` | 关闭 | 显示 EAR/Baseline/Threshold/State/Blinks 及眼部 landmark |
| `--record` | 关闭 | 保存 `blink_log.csv` |
| `--csv` | `blink_log.csv` | CSV 输出路径 |
| `--calibration-frames` | `60` | 校准所需帧数 |
| `--consec-frames` | `3` | 连续低 EAR 帧数（与 FPS 相关） |
| `--adaptation-alpha` | `0.01` | baseline 缓慢更新的 EMA 系数 |
| `--threshold-ratio` | `0.7` | 动态阈值比例（工程默认，非论文公式） |

---

## 5. 算法说明

### 5.1 EAR（规格说明第 5~7 节）

对每只眼睛取 6 个关键点：

- 左眼 `[33, 160, 158, 133, 153, 144]`
- 右眼 `[362, 385, 387, 263, 373, 380]`

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

水平距离接近 0 时返回 `None`（而非 0 或抛除零异常）。左右眼分别计算后再合并：

- 双眼都无效 → `None`
- 单眼无效 → 使用另一只眼
- 都有效 → 平均

### 5.2 自适应 baseline（规格说明第 9~13 节）

系统启动后先进入 `CALIBRATING` 状态，收集 `calibration_frames` 个有效 EAR 样本，
取**中位数**作为初始 baseline（避免眨眼帧和异常 landmark 的影响）。

之后仅在**稳定睁眼（`OPEN`）**状态下用 EMA 缓慢更新 baseline：

```python
baseline = (1 - alpha) * baseline + alpha * current_ear   # alpha = 0.01
```

闭眼期间**冻结** baseline，避免 baseline 跟随眨眼快速下降。

### 5.3 动态 threshold（规格说明第 12 节）

```python
threshold = baseline * threshold_ratio    # threshold_ratio = 0.7
```

> 注意：`baseline * ratio` 只是默认工程参数，**不是论文公式**。若论文正文给出
> 明确的 Modified EAR / Adaptive EAR 公式，应替换 `adaptive_threshold.calculate_threshold`。

### 5.4 时序状态机（规格说明第 15~17 节）

```text
UNKNOWN → OPEN → (EAR < threshold) CLOSING
        → (连续低 EAR ≥ consec_frames) CLOSED
        → (EAR ≥ threshold) OPEN + 一次 blink event
```

- 单帧异常低 EAR 不会产生 blink；
- 一次持续闭眼只在恢复睁眼时产生**一次** blink event。

### 5.5 无人脸处理（规格说明第 22 节）

无人脸时不产生 blink；连续丢失超过 `max_no_face_frames` 帧后重置状态机为
`UNKNOWN`，但**保留 baseline**，用户短暂离开画面后无需重新校准。

---

## 6. paper 与 robust 模式（规格说明第 30 节）

| 特性 | `paper` | `robust` |
| --- | --- | --- |
| MediaPipe + EAR + 状态机 | ✅ | ✅ |
| 中位数 baseline | ✅ | ✅ |
| 缓慢自适应 baseline（EMA） | ✅ | ✅ |
| EAR EMA 平滑 | ❌ | ✅ |
| 校准阶段 MAD 离群点剔除 | ❌ | ✅ |
| 调试可视化 / CSV 记录 | 可选 | 可选 |

两种模式共用同一核心（EAR、状态机、自适应 baseline），工程增强项以显式开关区分，
便于实验时对比论文原始算法与工程增强后的性能。

---

## 7. 输出

### 7.1 调试 overlay（`--debug`）

画面叠加：

```text
EAR:       0.31
Baseline:  0.34
Threshold: 0.24
State:     OPEN
Blinks:    12
Face:      detected
```

并绘制左右眼各 6 个 landmark。

### 7.2 CSV（`--record`）

`blink_log.csv` 列：

```text
frame_id,timestamp,left_ear,right_ear,ear,baseline,threshold,state,blink
```

可直接画出 `EAR vs Frame` 曲线，观察 EAR、Threshold 与 blink event 的关系。

---

## 8. 单元测试

```bash
cd blink_detection
python -m unittest test_blink_detector -v
```

覆盖规格说明第 32 节的 6 个用例：正常睁眼、单帧异常低 EAR、完整眨眼、长时间闭眼、
无人脸、异常 landmark（`p1 == p4` → `EAR = None`，不抛 `ZeroDivisionError`）。

---

## 9. 上层调用示例

```python
from blink_detector import BlinkDetector
from config import BlinkConfig

detector = BlinkDetector(BlinkConfig(mode="paper"))

# 每帧调用一次（frame 为 BGR 图像）
result = detector.process_frame(frame)

if result.blink_detected:
    handle_blink()

# 调试 / 论文实验
print(result.ear, result.left_ear, result.right_ear)
print(result.baseline, result.threshold, result.state)
print(result.face_detected)
```

模块对外只暴露 `视频帧 → BlinkDetector.process_frame() → BlinkResult → blink_detected`，
后续用于手势控制、人机交互、liveness detection、论文实验、blink frequency analysis
时，无需修改核心 EAR 与状态机代码。

---

## 10. 参考实现

- [Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV](https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV)
- [shakirsadiq6/Blink_Detection_Python](https://github.com/shakirsadiq6/Blink_Detection_Python)
- [QuangPham2404/Blink-Detection-Using-Adaptive-Eye-Aspect-Ratio](https://github.com/QuangPham2404/Blink-Detection-Using-Adaptive-Eye-Aspect-Ratio)
- [YogamruthReddy/face-mesh-verification-system](https://github.com/YogamruthReddy/face-mesh-verification-system)
