# 眨眼识别模块——论文复现与代码生成规格说明

## 0. 模块定位

本模块用于从连续视频帧中检测用户的眨眼事件。

核心处理链路：

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

本模块的目标是从连续视频中识别一个完整的：

```text
Open → Closing → Closed → Open
```

过程，并在一次完整闭眼事件结束后输出一次 `blink_detected=True`。

---

# 1. 实现目标

实现一个可以直接运行的 Python 眨眼检测模块，满足以下要求：

1. 输入：
   - 摄像头视频流或视频文件；
   - 每次处理一帧 BGR/RGB 图像。

2. 人脸检测：
   - 使用 MediaPipe Face Mesh；
   - 默认只处理一张人脸；
   - 优先选择稳定的人脸跟踪结果。

3. 眼睛特征：
   - 使用 MediaPipe Face Mesh 的指定眼部 landmark；
   - 左右眼分别计算 EAR；
   - 再计算双眼平均 EAR。

4. 眨眼判断：
   - 不直接使用固定 EAR 阈值作为最终判定；
   - 建立针对当前用户的 EAR baseline；
   - 根据 baseline 计算动态阈值；
   - 使用连续帧和状态机判断一次完整眨眼。

5. 鲁棒性：
   - 无人脸时不能崩溃；
   - 人脸短暂丢失时正确重置或保持状态；
   - EAR 分母接近 0 时不能产生异常；
   - 不因为单帧 landmark 抖动产生 blink；
   - 一次持续闭眼只能产生一次 blink event。

6. 可调试：
   - 支持输出当前 EAR；
   - 支持输出 baseline；
   - 支持输出动态阈值；
   - 支持输出当前眼睛状态；
   - 支持实时 blink counter；
   - 支持绘制眼部 landmark。

---

# 2. 技术选型

## 2.1 人脸关键点检测

使用：

```python
MediaPipe Face Mesh
```

Face Mesh 为人脸提供 468 个基础 facial landmarks。MediaPipe 官方相关实现也采用 Face Landmark 模块作为面部关键点基础。若启用 iris/refined landmarks，则眼部区域还可以获得更细致的关键点，但本模块的基础 EAR 算法不依赖额外 iris 点。

第一版复现代码优先使用与论文/原始大纲一致的 468 点 Face Mesh，而不是自行更换检测模型。

---

# 3. Python 依赖

# requirements_emotion.txt
torch == 2.12.0
torchvision == 0.27.0
opencv-pytho == 4.9.0.80
numpy == 1.26.4
scikit-learn == 1.7.2
matplotlib == 3.10.9
tqdm == 4.67.3
pillow == 12.2.0
pandas == 2.3.3
streamlit == 1.58.0
ploty == 6.8.0

不要在第一版实现中引入 PyTorch、TensorFlow、dlib 等额外深度学习框架。

---

# 4. MediaPipe 配置

初始化 Face Mesh 时：

```python
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

要求：

- `static_image_mode=False`，因为输入是连续视频；
- `max_num_faces=1`，因为当前模块只需要检测用户本人；
- 使用 tracking 模式降低连续视频中的重复检测开销；
- detection/tracking confidence 必须做成可配置参数。

如果后续实验发现眼部关键点精度不足，可以增加：

```python
refine_landmarks=True
```

但不要在第一版代码中无理由改变论文复现设置。

---

# 5. Eye Landmark 定义

使用以下 landmark index。

## 左眼

```python
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
```

## 右眼

```python
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
```

这组点与常见的 MediaPipe + EAR 实现一致，GitHub 上已有实时眨眼检测项目直接采用这一组 landmark。

对于每只眼睛，统一定义：

```text
p1 = eye_points[0]
p2 = eye_points[1]
p3 = eye_points[2]
p4 = eye_points[3]
p5 = eye_points[4]
p6 = eye_points[5]
```

代码中不要依赖“左眼/右眼”的语义来计算 EAR，而应该保证两组 landmark 均按照同样的几何顺序传入 `calculate_ear()`。

---

# 6. EAR 计算

对于每只眼睛计算：

\[
EAR =
\frac{
\|p_2-p_6\|+\|p_3-p_5\|
}{
2\|p_1-p_4\|
}
\]

其中：

- `p1-p4`：眼睛水平方向距离；
- `p2-p6`：第一组垂直距离；
- `p3-p5`：第二组垂直距离；
- `||a-b||`：两点之间的欧氏距离。

对应 Python：

```python
def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal < EPSILON:
        return None

    return (vertical_1 + vertical_2) / (2.0 * horizontal)
```

其中：

```python
EPSILON = 1e-6
```

防止异常 landmark 导致除零。

---

# 7. 双眼 EAR

每一帧分别计算：

```python
left_ear
right_ear
```

然后：

```python
ear_avg = (left_ear + right_ear) / 2.0
```

但不要简单忽略异常值。

推荐实现：

```python
if left_ear is None and right_ear is None:
    return None

if left_ear is None:
    return right_ear

if right_ear is None:
    return left_ear

return (left_ear + right_ear) / 2.0
```

这样单眼 landmark 暂时异常时不会直接导致整个检测器失效。

GitHub 中成熟的 MediaPipe EAR 实现也普遍采用左右眼 EAR 后再进行联合判断或平均。

---

# 8. 不使用固定 0.25 作为最终阈值

原始实现中可以存在：

```python
INITIAL_THRESHOLD = 0.25
```

但必须明确：

> `0.25` 只能作为调试/初始化参考值，不能作为论文复现版本的最终自适应阈值。

原因是不同人的：

- 眼睛大小；
- 眼裂高度；
- 摄像头距离；
- 摄像头角度；
- 面部结构；
- 眼睛自然睁开程度

都会导致 EAR baseline 不同。

因此正式检测必须使用：

```text
个人 EAR baseline
        ↓
动态 threshold
        ↓
blink detection
```

而不是：

```text
EAR < 0.25 → blink
```

---

# 9. Adaptive EAR Baseline

这是本模块最重要的部分之一。

必须把“baseline”和“threshold”分开。

定义：

```python
ear_baseline
ear_threshold
```

其中：

- `ear_baseline`：当前用户正常睁眼状态下的 EAR；
- `ear_threshold`：根据 baseline 计算出的闭眼判断阈值。

---

# 10. Baseline 初始化

系统启动后首先进入：

```text
CALIBRATING
```

状态。

在用户正常睁眼时收集 EAR 样本：

```python
calibration_samples = []
```

例如：

```python
CALIBRATION_FRAMES = 60
```

连续收集有效 EAR。

注意：

**校准阶段不能把明显闭眼帧加入 baseline。**

因此第一版实现可以采用：

```python
if ear_avg is not None:
    calibration_samples.append(ear_avg)
```

但更严格的实现应该过滤明显异常值。

推荐使用中位数或截尾平均：

```python
ear_baseline = np.median(calibration_samples)
```

这样比简单平均值更不容易受到眨眼和异常 landmark 的影响。

---

# 11. Adaptive Threshold

论文要求使用 per-participant slowly adaptive baseline。

因此代码必须将：

```python
update_baseline()
```

设计成独立函数，而不是把公式写死在 `detect_blink()` 内部。

推荐接口：

```python
def update_baseline(
    current_ear,
    current_state,
    baseline,
    alpha,
):
    ...
```

核心原则：

> 只有在“确认眼睛处于稳定睁眼状态”时，才允许缓慢更新 baseline。

例如：

```python
if state == EyeState.OPEN:
    baseline = (1 - alpha) * baseline + alpha * current_ear
```

其中：

```python
0 < alpha << 1
```

例如：

```python
alpha = 0.01
```

表示 baseline 缓慢适应当前用户。

**禁止：**

```python
baseline = current_ear
```

因为这会导致 baseline 对单帧噪声高度敏感。

---

# 12. Threshold 计算

由于当前提供的大纲没有给出论文中 adaptive threshold 的精确数学公式，因此不要擅自声称某个公式就是论文公式。

代码设计成：

```python
threshold = calculate_threshold(baseline)
```

独立函数。

例如工程默认可以提供：

```python
def calculate_threshold(baseline):
    return baseline * THRESHOLD_RATIO
```

其中：

```python
THRESHOLD_RATIO = 0.7
```

只是默认工程参数，不应标注为“论文公式”。

如果论文正文明确给出了 Modified EAR / Adaptive EAR 的公式，则必须优先替换这里，并保持论文公式原样实现。

GitHub 上已有专门针对 Adaptive EAR 的项目，并实现了 Modified EAR threshold 以及 3D EAR 等不同版本，因此这里建议把 adaptive strategy 设计成可插拔组件，而不是把一种公式硬编码进系统。

---

# 13. 为什么 baseline 必须慢速更新

baseline 的目标是适应：

- 摄像头距离缓慢变化；
- 用户姿态缓慢变化；
- 光照变化；
- 用户自然睁眼程度变化；
- landmark 模型的小幅系统误差。

但 baseline 不能跟随一次眨眼快速下降。

因此：

```text
正常睁眼
   ↓
slow update baseline

眨眼
   ↓
冻结 baseline

眼睛恢复睁开
   ↓
重新允许 baseline 缓慢更新
```

这是本模块“slowly adaptive”的关键。

---

# 14. 眨眼检测不能只使用一个 if

不要实现成：

```python
if ear < threshold:
    blink = True
```

这种实现会导致：

```text
闭眼持续 10 帧
→ 可能产生 10 次 blink
```

正确实现应该使用一个简单状态机。

---

# 15. Blink State Machine

定义：

```python
class EyeState(Enum):
    UNKNOWN = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3
```

推荐状态：

```text
UNKNOWN
   ↓
OPEN
   ↓ EAR < threshold
CLOSING
   ↓ 连续低 EAR
CLOSED
   ↓ EAR >= threshold
OPEN + blink_event
```

也可以简化为两个内部状态：

```text
OPEN
CLOSED
```

但推荐保留 `CLOSING`，方便后续调试和论文实验。

---

# 16. 连续帧判定

定义：

```python
CONSEC_FRAMES = 2
```

但这个值不能被写死。

必须理解：

```text
连续帧数
```

与：

```text
视频 FPS
```

直接相关。

例如：

```text
30 FPS + 2 frames ≈ 67 ms
30 FPS + 3 frames ≈ 100 ms
30 FPS + 5 frames ≈ 167 ms
```

因此应该将其作为配置项：

```python
consec_frames
```

而不是在代码中到处出现数字 `2`。

GitHub 中典型的 MediaPipe EAR 实现也是通过 `ear_threshold + consec_frames` 的组合来减少单帧噪声误报。

---

# 17. Blink Event 的严格定义

一次 blink 必须满足：

```text
1. 之前处于 OPEN
2. EAR 下降到 threshold 以下
3. 连续低于 threshold 至少 N 帧
4. 随后 EAR 恢复到 threshold 以上
5. 在恢复时产生一次 blink event
```

例如：

```text
EAR:

0.31
0.32
0.30
0.21
0.18
0.16
0.19
0.27
0.31
0.32

状态：

OPEN
OPEN
OPEN
CLOSING
CLOSED
CLOSED
CLOSED
OPEN
OPEN
OPEN

输出：

blink_detected = False
False
False
False
False
False
False
True
False
False
```

注意：

> `blink_detected=True` 只允许出现一次。

---

# 18. 推荐 BlinkDetector 类接口

不要把所有变量写成 global。

禁止使用：

```python
global frame_counter
global EAR_THRESHOLD
```

推荐封装成：

```python
class BlinkDetector:
    def __init__(
        self,
        consec_frames=3,
        calibration_frames=60,
        adaptation_alpha=0.01,
        threshold_ratio=0.7,
    ):
        ...
```

核心接口：

```python
def process_frame(self, frame):
    """
    输入一帧图像。

    Returns:
        BlinkResult
    """
```

---

# 19. BlinkResult

推荐定义：

```python
@dataclass
class BlinkResult:
    blink_detected: bool
    face_detected: bool
    left_ear: Optional[float]
    right_ear: Optional[float]
    ear: Optional[float]
    baseline: Optional[float]
    threshold: Optional[float]
    state: str
```

这样上层程序不仅能够获得：

```python
blink_detected
```

还可以获得调试信息。

例如：

```python
result = detector.process_frame(frame)

if result.blink_detected:
    print("Blink detected")
```

同时：

```python
print(result.ear)
print(result.threshold)
```

可以用于论文实验和参数调试。

---

# 20. 推荐代码模块结构

项目建议组织成：

```text
blink_detection/
│
├── main.py
├── blink_detector.py
├── face_mesh.py
├── ear.py
├── adaptive_threshold.py
├── types.py
├── visualization.py
├── config.py
├── requirements.txt
└── README.md
```

职责如下。

### `face_mesh.py`

负责：

```text
MediaPipe 初始化
输入图像
输出 landmarks
```

### `ear.py`

只负责：

```text
landmark → EAR
```

不要在这里处理 blink state。

### `adaptive_threshold.py`

负责：

```text
baseline 初始化
baseline 更新
threshold 计算
```

### `blink_detector.py`

负责：

```text
EAR
→ threshold
→ state machine
→ blink event
```

### `types.py`

定义：

```text
EyeState
BlinkResult
```

### `visualization.py`

负责：

```text
绘制眼部 landmark
EAR
threshold
state
blink count
```

### `main.py`

负责：

```text
读取摄像头
读取视频
调用 BlinkDetector
显示结果
```

---

# 21. 单帧完整处理逻辑

`process_frame()` 必须按照以下顺序执行：

```text
Step 1:
输入 frame

Step 2:
转换颜色空间

Step 3:
MediaPipe Face Mesh

Step 4:
检查是否检测到人脸

Step 5:
获取第一张脸 landmarks

Step 6:
提取左右眼 landmarks

Step 7:
计算 left EAR

Step 8:
计算 right EAR

Step 9:
计算 average EAR

Step 10:
更新 calibration / baseline

Step 11:
计算 adaptive threshold

Step 12:
更新 blink state

Step 13:
产生 blink event

Step 14:
返回 BlinkResult
```

不要把这些步骤打乱。

---

# 22. 无人脸处理

如果：

```python
results.multi_face_landmarks is None
```

则：

```python
face_detected = False
blink_detected = False
```

不要产生 blink。

对于连续无人脸帧，推荐：

```python
no_face_frames += 1
```

当超过：

```python
MAX_NO_FACE_FRAMES
```

时重置：

```python
state = UNKNOWN
```

但不要立刻删除 baseline。

这样用户短暂离开画面再回来时不需要完全重新校准。

---

# 23. Landmark 异常处理

必须检查：

```python
horizontal_distance > EPSILON
```

如果 EAR 无法计算：

```python
ear = None
```

当前帧：

```python
blink_detected = False
```

不要使用：

```python
0
```

作为异常 EAR。

因为：

```python
EAR = 0
```

会被错误解释成“眼睛完全闭合”。

---

# 24. 多人脸处理

第一版：

```python
max_num_faces=1
```

只处理第一张脸。

不要为了“通用性”在第一版中实现复杂的多人脸跟踪。

如果未来需要多人脸，再扩展为：

```python
face_id → BlinkDetector instance
```

即每个人拥有独立的：

```text
baseline
threshold
state
frame_counter
blink_count
```

---

# 25. 参数配置

所有参数集中放在：

```python
@dataclass
class BlinkConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    calibration_frames: int = 60

    adaptation_alpha: float = 0.01

    threshold_ratio: float = 0.7

    consec_frames: int = 3

    max_no_face_frames: int = 30
```

禁止在业务代码中出现大量 magic numbers。

例如禁止：

```python
if counter > 3:
```

应该：

```python
if counter >= config.consec_frames:
```

---

# 26. 可视化调试模式

提供：

```python
debug=True
```

打开后，在视频画面显示：

```text
EAR:       0.31
Baseline:  0.34
Threshold: 0.24
State:     OPEN
Blinks:    12
Face:      detected
```

并绘制：

```text
左眼 6 个 landmark
右眼 6 个 landmark
```

这样可以直接观察论文算法的实际行为。

成熟 GitHub 实现通常也会把 EAR、眼部 landmarks 和 blink count 叠加到视频中，这对于调参和验证非常有用。

---

# 27. EAR 曲线记录

必须支持实验模式：

```python
record=True
```

记录：

```text
timestamp
frame_id
left_ear
right_ear
ear
baseline
threshold
state
blink_detected
```

保存为：

```text
blink_log.csv
```

格式：

```text
frame_id,timestamp,left_ear,right_ear,ear,baseline,threshold,state,blink
```

这样可以直接画：

```text
EAR vs Frame
```

并观察：

```text
EAR 曲线
Threshold
Blink event
```

这对于论文复现非常重要。

---

# 28. 核心伪代码

最终代码逻辑应该接近下面结构：

```python
class BlinkDetector:

    def process_frame(self, frame):

        # --------------------------------
        # 1. Face landmarks
        # --------------------------------
        landmarks = self.face_mesh.process(frame)

        if landmarks is None:
            self.handle_no_face()
            return BlinkResult(
                blink_detected=False,
                face_detected=False,
                ...
            )

        # --------------------------------
        # 2. Extract eyes
        # --------------------------------
        left_points = extract_landmarks(
            landmarks,
            LEFT_EYE_INDICES
        )

        right_points = extract_landmarks(
            landmarks,
            RIGHT_EYE_INDICES
        )

        # --------------------------------
        # 3. Calculate EAR
        # --------------------------------
        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)

        ear = calculate_average_ear(
            left_ear,
            right_ear
        )

        if ear is None:
            return invalid_result()

        # --------------------------------
        # 4. Calibration
        # --------------------------------
        if not self.calibrator.is_ready():

            self.calibrator.add_sample(ear)

            if self.calibrator.is_ready():
                self.baseline = \
                    self.calibrator.get_baseline()

            return result(
                blink_detected=False,
                state="CALIBRATING"
            )

        # --------------------------------
        # 5. Adaptive threshold
        # --------------------------------
        threshold = self.threshold_manager.calculate(
            self.baseline
        )

        # --------------------------------
        # 6. Blink state machine
        # --------------------------------
        blink_detected = self.state_machine.update(
            ear=ear,
            threshold=threshold
        )

        # --------------------------------
        # 7. Slowly update baseline
        # --------------------------------
        if self.state_machine.is_stable_open():

            self.baseline = self.baseline_updater.update(
                baseline=self.baseline,
                current_ear=ear
            )

        # --------------------------------
        # 8. Return result
        # --------------------------------
        return BlinkResult(
            blink_detected=blink_detected,
            face_detected=True,
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            baseline=self.baseline,
            threshold=threshold,
            state=self.state_machine.state,
        )
```

---

# 29. 必须避免的错误实现

## 错误 1：每一帧直接判断 blink

错误：

```python
blink = ear < threshold
```

这只能表示：

```text
眼睛当前是否闭合
```

不能表示：

```text
是否发生了一次眨眼
```

---

## 错误 2：baseline 每帧直接赋值

错误：

```python
baseline = ear
```

会导致 adaptive threshold 跟随 blink 一起下降。

---

## 错误 3：闭眼期间更新 baseline

错误：

```python
if ear < threshold:
    baseline = update(baseline, ear)
```

必须冻结 baseline。

---

## 错误 4：使用固定 0.25 代替 adaptive threshold

错误：

```python
if ear < 0.25:
    blink = True
```

这只能作为 baseline implementation / ablation baseline。

---

## 错误 5：连续低 EAR 每帧都输出 blink

错误：

```text
EAR < threshold
EAR < threshold
EAR < threshold
EAR < threshold

→ 4 次 blink
```

正确：

```text
低 EAR 持续
→ 一个 CLOSED 状态
→ 恢复 OPEN
→ 只产生一个 blink event
```

---

# 30. 论文复现与工程增强必须分开

代码中建议保留两种模式：

```python
mode="paper"
```

和：

```python
mode="robust"
```

## `paper` 模式

严格按照论文描述：

```text
MediaPipe
+
指定 landmark
+
EAR
+
论文 adaptive threshold
+
论文 temporal criterion
```

不加入未经论文验证的额外算法。

## `robust` 模式

可以增加：

```text
EAR smoothing
异常帧过滤
baseline EMA
短时无人脸容错
debug visualization
CSV logging
```

这样实验时可以清楚区分：

```text
论文原始算法性能
```

和：

```text
工程增强后的性能
```

---

# 31. 关于 3D EAR

当前基础实现使用 2D/归一化 landmark 坐标计算 EAR。

如果论文明确提出使用 3D EAR，则另行实现：

```python
calculate_3d_ear()
```

不能直接把 2D EAR 改名成 3D EAR。

已有 GitHub Adaptive EAR 项目专门比较了普通 EAR、Modified EAR 和 3D EAR，并指出 3D landmark 可以用于缓解不同头部姿态带来的 EAR 变化。

因此：

```text
普通 EAR
```

和：

```text
3D EAR
```

应作为两个独立实验分支。

---

# 32. 测试要求

至少实现以下测试。

## Test 1：正常睁眼

输入：

```text
EAR > threshold
```

预期：

```text
state = OPEN
blink_detected = False
```

---

## Test 2：单帧异常低 EAR

输入：

```text
OPEN
OPEN
LOW
OPEN
OPEN
```

预期：

```text
0 blink
```

---

## Test 3：完整眨眼

输入：

```text
OPEN
LOW
LOW
LOW
OPEN
```

预期：

```text
1 blink
```

---

## Test 4：长时间闭眼

输入：

```text
OPEN
LOW
LOW
LOW
LOW
LOW
LOW
OPEN
```

预期：

```text
1 blink
```

而不是：

```text
6 blinks
```

---

## Test 5：无人脸

输入：

```text
face = None
```

预期：

```text
blink_detected = False
```

且程序不能 crash。

---

## Test 6：异常 landmark

使：

```text
p1 == p4
```

预期：

```text
EAR = None
```

程序不能产生：

```text
ZeroDivisionError
```

---

# 33. 最终代码生成要求

请生成一个**完整、可直接运行的 Python 项目**，而不是只给出代码片段。

必须包含：

```text
1. requirements.txt
2. blink_detector.py
3. main.py
4. 配置类
5. EAR 计算
6. MediaPipe Face Mesh
7. Adaptive baseline
8. Adaptive threshold
9. Blink state machine
10. BlinkResult
11. Debug visualization
12. CSV logging
13. 单元测试
14. README.md
```

代码要求：

- Python 3.10+；
- 使用类型注解；
- 不使用 global 状态；
- 每个核心模块使用独立 class/function；
- 所有参数集中配置；
- 函数具有清晰 docstring；
- 对异常输入进行处理；
- 可以直接运行摄像头；
- 可以切换到视频文件；
- 支持实时显示 EAR、threshold、state、blink count；
- 支持保存 CSV；
- 不允许省略关键函数并写 `pass`；
- 不允许用伪代码代替实际实现。

---

# 34. 代码生成时的优先级

如果不同要求发生冲突，按照以下优先级：

```text
Priority 1
论文中明确给出的算法公式和参数

Priority 2
本规格书明确规定的 landmark / EAR / 状态机

Priority 3
成熟 GitHub 实现中的工程结构和异常处理

Priority 4
通用工程优化
```

尤其注意：

> 不要因为 GitHub 项目使用了固定 `EAR_THRESHOLD=0.25`，就把本项目的 adaptive threshold 改成固定阈值。

GitHub 项目主要用于参考**代码工程实现方式**，而论文复现参数和算法定义仍然优先以论文为准。

---

# 35. 推荐参考实现

以下 GitHub 项目可作为代码实现参考：

1. `Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV`

   适合参考：
   - MediaPipe Face Mesh；
   - EAR 计算；
   - consecutive frames；
   - 实时 EAR visualization；
   - blink counter。

   [GitHub 项目](https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV?utm_source=chatgpt.com)

2. `shakirsadiq6/Blink_Detection_Python`

   适合参考：
   - 简单的 MediaPipe + OpenCV 实时 blink detector；
   - 摄像头输入；
   - landmark 提取；
   - blink counter。

   [GitHub 项目](https://github.com/shakirsadiq6/Blink_Detection_Python?utm_source=chatgpt.com)

3. `QuangPham2404/Blink-Detection-Using-Adaptive-Eye-Aspect-Ratio`

   **本项目最值得重点参考。**

   适合参考：
   - Adaptive EAR；
   - Modified EAR；
   - per-person threshold；
   - 3D EAR；
   - 不同 blink detection 方法之间的比较。

   [GitHub 项目](https://github.com/QuangPham2404/Blink-Detection-Using-Adaptive-Eye-Aspect-Ratio?utm_source=chatgpt.com)

4. `YogamruthReddy/face-mesh-verification-system`

   适合参考：
   - MediaPipe 468 landmarks；
   - EAR 模块化；
   - liveness/blink gate；
   - 工程化的 landmark feature extraction。

   [GitHub 项目](https://github.com/YogamruthReddy/face-mesh-verification-system?utm_source=chatgpt.com)

---

# 36. 最终模块接口

上层程序最终只需要调用：

```python
detector = BlinkDetector(config)

result = detector.process_frame(frame)
```

然后：

```python
if result.blink_detected:
    # 产生一次眨眼事件
    handle_blink()
```

其他信息用于：

```python
result.ear
result.left_ear
result.right_ear
result.baseline
result.threshold
result.state
result.face_detected
```

整个模块对外暴露的是：

```text
视频帧
   ↓
BlinkDetector.process_frame()
   ↓
BlinkResult
   ↓
blink_detected
```

因此后续无论将眨眼用于：

- 手势控制；
- 人机交互；
- liveness detection；
- 论文实验；
- blink frequency analysis；

都不需要修改核心 EAR 和状态机代码。