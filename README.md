# 基于眨眼与情绪的走神预测复现

本项目复现论文 **Detecting Attention and Two Different Mind Wandering States Using
Blink Frequency and Emotion Analysis** 的特征与经典机器学习流程。当前使用可直接公开下载的
MWDET 数据训练 `focused / not_focused` 二分类器，并保留 PAFE 视频入口及未来
`focused / deliberate_mw / spontaneous_mw` 三分类接口。

> MWDET 包含真实走神探针与 WebGazer/Tobii 眼动数据，但不包含原始人脸视频。因此当前
> 真实模型使用眼动特征，不会伪造情绪特征。PAFE 视频仍需向原作者申请。

## 1. 项目结构

```text
.
├── E109.D_2025AHP0006.pdf       # 复现目标论文
├── emo/
│   ├── emotion/                 # FER-2013 情绪识别代码
│   └── models/                  # 已训练的情绪模型
├── zhayan/blink_detection/      # MediaPipe + EAR 眨眼检测模块
└── mind_wandering/
    ├── config.py                # 标签、情绪效价和默认路径
    ├── dataset/
    │   ├── download_mwdet.py     # 官方 GitHub 数据下载与哈希校验
    │   ├── prepare_mwdet.py      # MWDET 探针配对和眼动特征生成
    │   ├── prepare_manifest.py  # 探针清单校验及标准化
    │   └── manifest.example.csv # 输入格式示例
    ├── features/
    │   ├── aggregation.py       # 5 秒窗口统计特征
    │   └── extract_features.py  # 从视频提取眨眼和情绪特征
    ├── training/train.py        # 七种经典模型及分组交叉验证
    ├── inference/
    │   └── predict_features.py  # 对特征 CSV 推理
    ├── tests/                   # 不需要 PAFE 的基础测试
    └── requirements.txt
```

## 2. 方法与论文对应关系

每个注意力探针对应探针前一个时间窗口，默认长度为 5 秒。系统从窗口中提取：

- EAR 均值、标准差、四分位数、中位数和 MAD；
- 眨眼次数及每分钟眨眼频率；
- angry、disgust、fear、happy、sad、surprise、neutral 概率统计；
- 正面、负面和中性效价；
- 人脸有效帧比例，用于发现遮挡或检测失败。

训练脚本比较 Random Forest、Decision Tree、Gradient Boosting、SVM、KNN、
Logistic Regression 和 Gaussian Naive Bayes。验证使用 `subject_id` 分组，确保同一人的
片段不会同时出现在训练折和验证折中。主要比较 balanced accuracy、macro-F1、
weighted-F1、逐类指标和混淆矩阵。

论文报告的 99% accuracy 不能直接视为本项目预期结果。PAFE 的任务、参与者和标签定义
不同，而且走神类别不平衡；按受试者分组的结果通常也会低于随机按片段切分的结果。

## 3. 环境安装

推荐 Python 3.10 或 3.11。MediaPipe、PyTorch 和 torchvision 在较新的 Python 版本上
可能没有兼容轮子。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r mind_wandering\requirements.txt
```

已有眨眼模块也带有独立依赖说明：

```powershell
python -m pip install -r zhayan\blink_detection\requirements.txt
```

所有以下命令均在项目根目录运行。

## 4. 单页摄像头监测界面

项目提供原生 HTML、CSS、JavaScript 编写的一页式界面，所有功能按钮位于左侧栏。页面
展示实时摄像头、七类情绪概率、EAR、动态阈值、眼部状态、眨眼次数、眨眼频率和趋势图。
当前不会把摄像头结果送入 MWDET 走神模型。

```powershell
python -m pip install -r web\requirements.txt
python web\server.py
```

打开 `http://127.0.0.1:5000`，点击左侧“开启摄像头”，授权后点击“开始识别”。所有画面只
发送到本机 `127.0.0.1` 服务处理，服务不保存视频；“保存截图”仅在用户点击时由浏览器下载
当前帧。详细说明见 `web/README.md`。

## 5. MWDET：可直接复现的真实数据实验

MWDET 来源于 Zhao、Lofi 和 Hauff 2017 年的实验：13 名参与者分别观看两段 MOOC 视频，
系统每隔 30–60 秒响铃；如果参与者在过去 30 秒发生走神，就在响铃后按键，否则继续观看。
公开数据包含 200 个探针区间，其中 58 个走神、142 个专注。

### 5.1 下载

```powershell
python -m mind_wandering.dataset.download_mwdet --output-dir data\raw\mwdet
```

下载器使用 GitHub 官方文件并校验 SHA-256：
`b12aa3f0b64e76c3fad5cbc3ad8c8ed469f97096121eb6b18397949d6b018e64`。

### 5.2 生成 WebGazer 特征

```powershell
python -m mind_wandering.dataset.prepare_mwdet `
  --root data\raw\mwdet\Data_Publish\Data_Publish `
  --output data\processed\mwdet_webgazer_features.csv
```

每个样本使用相邻两次响铃之间的眼动数据，与原论文一致。特征包括 gaze X/Y 分布、眼动轨迹
长度、相邻采样点位移、注视次数与时长、扫视次数与角度、有效采样率等。阶段、探针序号和
视频时间只作为审计信息保存，训练时自动排除，防止时间位置泄漏。

### 5.3 留一受试者训练

```powershell
python -m mind_wandering.training.train `
  --features data\processed\mwdet_webgazer_features.csv `
  --output-dir outputs\mwdet_webgazer_lopo `
  --task binary `
  --folds 13 `
  --seed 42
```

13 折对应 leave-one-participant-out：每折完整留出一名参与者。这比随机按样本切分更能反映
模型面对新用户时的泛化能力。

### 5.4 当前复现结果

在排除阶段、探针序号和视频时间等元数据后，本仓库当前运行结果如下：

| 模型 | Balanced accuracy | Macro-F1 | 走神 Precision | 走神 Recall | 走神 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.548 | **0.540** | 0.342 | 0.448 | 0.388 |
| Naive Bayes | 0.508 | 0.410 | 0.295 | **0.741** | **0.422** |
| Gradient Boosting | 0.521 | 0.517 | 0.333 | 0.224 | 0.268 |
| KNN | 0.533 | 0.527 | 0.375 | 0.207 | 0.267 |
| Random Forest | 0.538 | 0.504 | 0.600 | 0.103 | 0.176 |

程序按 macro-F1 选择并保存 SVM，因为它在两个类别间更均衡；如果应用更重视“不漏掉走神”，
可以使用 Naive Bayes，但需要接受更多误报。完整七模型指标保存在
`outputs/mwdet_webgazer_lopo/metrics.json`，折外预测保存在同目录的
`cross_validated_predictions.csv`。原 MWDET 项目报告 WebGazer + Naive Bayes 的走神 F1
约为 0.405；本实现的 0.422 与其处于相近范围，但特征实现和模型选择协议并非完全相同。

## 6. PAFE 数据申请与清单

PAFE 论文为：T. Lee et al., *Predicting Mind-Wandering With Facial Videos in Online
Lectures*, CVPR Workshops 2022。建议使用学校邮箱联系 `taekyung@kaist.ac.kr`，说明研究
用途并申请以下内容：

- 640p、30 FPS facial videos；
- probe timestamps；
- Focused / Not-Focused / Skip labels；
- subject IDs；
- 数据许可和引用要求。

不要把原始人脸视频提交到 Git，也不要向第三方分发。

获得数据后，参照 `mind_wandering/dataset/manifest.example.csv` 建立原始清单：

```csv
subject_id,video_path,probe_time_sec,label
P001,data/raw/pafe/P001/session.mp4,40,focused
P001,data/raw/pafe/P001/session.mp4,80,not_focused
```

标准化并检查视频路径：

```powershell
python -m mind_wandering.dataset.prepare_manifest `
  --input data\raw\pafe_annotations.csv `
  --output data\processed\manifest.csv `
  --task binary
```

视频尚未到位时，可只验证 CSV 格式：

```powershell
python -m mind_wandering.dataset.prepare_manifest `
  --input mind_wandering\dataset\manifest.example.csv `
  --output data\processed\manifest.example.normalized.csv `
  --task binary `
  --no-check-files
```

## 7. PAFE 视频特征提取

```powershell
python -m mind_wandering.features.extract_features `
  --manifest data\processed\manifest.csv `
  --output data\processed\features.csv `
  --window-sec 5 `
  --sample-fps 10
```

`--sample-fps 10` 用于降低情绪网络推理开销；眨眼检测对采样率敏感。如果设备性能允许且
需要更接近原始 30 FPS，应改成 `--sample-fps 30`。不同采样率的结果不能直接混合训练。

默认情绪权重为 `emo/models/fer2013_best_model.pth`，也可通过 `--emotion-model` 指定。

## 8. 通用训练与评估

```powershell
python -m mind_wandering.training.train `
  --features data\processed\features.csv `
  --output-dir outputs\pafe_binary `
  --task binary `
  --folds 5 `
  --seed 42
```

输出包括：

```text
outputs/pafe_binary/
├── mind_wandering_model.joblib       # 在全部数据上重训的最佳模型
├── metrics.json                      # 各模型分组交叉验证指标
└── cross_validated_predictions.csv  # 每个探针的折外预测
```

训练要求至少两个不同 `subject_id`。如果受试者少于五人，程序会自动把折数降低到受试者数。

## 9. 批量推理

对已经提取的特征进行预测：

```powershell
python -m mind_wandering.inference.predict_features `
  --model outputs\pafe_binary\mind_wandering_model.joblib `
  --features data\processed\features.csv `
  --output outputs\predictions.csv
```

输出包含 `predicted_label`、`predicted_target`，以及模型支持时的各类别概率。

## 10. 三分类扩展

将来获得带 intentionality 的数据后，清单标签写为：

```text
focused
deliberate_mw
spontaneous_mw
```

然后在清单准备和训练命令中使用 `--task three_class`。特征列、受试者分组验证和推理文件
无需修改。PAFE 的 `not_focused` 不能自动拆成两个真实类别；项目不会用情绪或规则生成的
伪标签冒充主动/无意走神真值。

## 11. 测试

无需 PAFE 数据即可运行：

```powershell
python -m unittest discover -s mind_wandering\tests -v
python -m mind_wandering.dataset.download_mwdet --help
python -m mind_wandering.dataset.prepare_mwdet --help
python -m mind_wandering.dataset.prepare_manifest --help
python -m mind_wandering.features.extract_features --help
python -m mind_wandering.training.train --help
python -m mind_wandering.inference.predict_features --help
```

眨眼模块测试：

```powershell
Push-Location zhayan\blink_detection
python -m unittest test_blink_detector -v
Pop-Location
```

## 12. 已知限制

- PAFE 只提供专注/非专注真值，本阶段不是论文三分类的完全复现。
- MWDET 没有人脸视频，真实数据实验不能使用 FER 情绪模块或 EAR 眨眼模块。
- MWDET 的 WebGazer 数据约为 5 Hz，样本仅来自 13 名计算机背景参与者，泛化能力有限。
- FER-2013 在真实网课光照、姿态和人群上的领域偏移可能降低情绪特征质量。
- 面部行为只能提供走神概率，不能证明一个人的主观心理状态。
- 眼镜、遮挡、摄像头帧率、光线和人脸检测失败会影响 EAR 与眨眼率。
- 模型不能用于考试监控、惩罚或高风险人员评估。
- 发布实验结果时必须报告数据划分方式、类别分布和每类指标，不能只报告 accuracy。

## 13. 参考资料

- Mamun et al., *Detecting Attention and Two Different Mind Wandering States Using
  Blink Frequency and Emotion Analysis*, IEICE Transactions, 2026.
- Lee et al., [Predicting Mind-Wandering With Facial Videos in Online Lectures](https://openaccess.thecvf.com/content/CVPR2022W/CVPM/html/Lee_Predicting_Mind-Wandering_With_Facial_Videos_in_Online_Lectures_CVPRW_2022_paper.html), CVPRW 2022.
- Zhao, Lofi and Hauff, [Scalable Mind-Wandering Detection for MOOCs: A Webcam-Based Approach](https://github.com/Yue-ZHAO/MWDET_Project), EC-TEL 2017.
- Goodfellow et al., FER-2013.
- Soukupova and Cech, *Real-Time Eye Blink Detection using Facial Landmarks*, 2016.
