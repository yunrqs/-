📁 项目结构总览
text

emo/emotion/
├── train_with_fer2013.py # 使用 FER-2013 CSV 数据集训练模型
├── train_model.py # 使用图片文件夹数据集训练模型
├── use_model.py # 实时摄像头情绪识别（基础版）
├── emotion_predictor.py # 高精度实时情绪识别系统（高级版）
├── predict_image.py # 单张图片情绪识别
├── predict_video.py # 视频文件情绪识别
├── pretrained_emotion.py # 基于预训练 ResNet 的情绪识别
├── download_pretrained_model.py # 下载 FER2013 预训练模型
└── FER-2013/ # FER-2013 数据集目录
├── fer2013.csv # 原始 CSV 数据文件
└── test/ # 测试图片（按类别分文件夹）
📄 各文件功能详解

1. train_with_fer2013.py — FER-2013 CSV 数据集训练
   功能：使用 FER-2013 数据集的 CSV 格式（train.csv, val.csv, test.csv）训练改进的 CNN 模型
   模型架构：自定义 4 层卷积网络 + 全局平均池化 + Dropout 全连接层
   特点：
   处理 CSV 中的像素字符串，转换为 48×48 灰度图
   使用类别权重处理数据不平衡问题
   支持数据增强（随机翻转、旋转）
   自动保存最佳模型到 ../models/fer2013_best_model.pth
   输入数据格式：CSV 文件，包含 emotion（标签）和 pixels（像素字符串）列
2. train_model.py — 图片文件夹数据集训练
   功能：使用按类别分文件夹的图片数据集训练模型
   模型架构：两种可选
   EmotionCNN：基于预训练 ResNet18 的迁移学习模型
   ImprovedEmotionCNN：自定义深层 CNN（与 train_with_fer2013.py 类似，但输入为 3 通道）
   输入数据格式：
   text

dataset/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/
输出：保存最佳模型到 models/best_model.pth 3. use_model.py — 实时摄像头情绪识别（基础版）
功能：调用摄像头进行实时人脸情绪识别
依赖模型：../models/fer2013_best_model.pth（由 train_with_fer2013.py 生成）
特点：
使用 OpenCV Haar 级联分类器检测人脸
历史结果平滑（减少抖动）
实时显示 FPS 和情绪统计
支持按键操作：q 退出、s 截图、d 显示详细概率 4. emotion_predictor.py — 高精度实时情绪识别系统（高级版）
功能：功能更完善的实时/离线情绪识别系统
主要类：
EmotionPredictor：单张图片预测，含预处理（灰度化、均衡化、归一化）
RealTimeEmotionRecognizer：实时摄像头识别，带完整统计和交互功能
特点：
更丰富的图像预处理（直方图均衡化）
概率条可视化
支持命令行参数：--model、--image、--camera
额外按键：r 重置统计、p 打印详细概率 5. predict_image.py — 单张图片识别
功能：对单张图片进行人脸检测和情绪识别
使用方式：直接修改代码中的图片路径 "test_face.jpg"
输出：打印识别结果、置信度、各类别概率，并显示带标注的图像 6. predict_video.py — 视频文件识别
功能：对视频文件进行逐帧人脸情绪识别
使用方式：直接修改代码中的视频路径 "test_video.mp4"
输出：实时显示识别结果的视频窗口，按 q 退出 7. pretrained_emotion.py — 预训练 ResNet 情绪识别
功能：基于 ImageNet 预训练的 ResNet50 进行情绪识别
特点：
使用 torchvision.models.resnet50(pretrained=True)
修改最后一层全连接层输出 7 类情绪
使用 ImageNet 的标准归一化参数
适用场景：快速测试，无需自己训练模型 8. download_pretrained_model.py — 下载预训练模型
功能：从网络下载 FER2013 预训练的 ResNet50 模型
下载地址：https://github.com/justusschock/torch-emotion/...
输出：保存到 models/fer2013_resnet50.pth
🚀 使用说明
环境准备
确保安装以下依赖：

bash

pip install torch torchvision pandas numpy opencv-python matplotlib tqdm scikit-learn Pillow
使用流程
方案一：使用 FER-2013 CSV 数据集训练（推荐）
准备数据：将 FER-2013 数据集拆分为 train.csv、val.csv、test.csv，放入 FER-2013/ 目录
训练模型：
bash

cd emo/emotion
python train_with_fer2013.py
实时识别：
bash

python use_model.py
方案二：使用图片文件夹训练
准备数据：按类别将图片放入 dataset/ 下的各个文件夹
训练模型：
bash

python train_model.py
实时识别：
bash

python emotion_predictor.py
方案三：单张图片/视频识别
bash

# 单张图片（需修改代码中的路径）

python predict_image.py

# 视频文件（需修改代码中的路径）

python predict_video.py
方案四：使用预训练模型（无需训练）
bash

# 下载预训练模型

python download_pretrained_model.py

# 或在代码中直接使用 pretrained_emotion.py

python pretrained_emotion.py
⚠️ 注意事项
问题 解决方案
模型路径不匹配 检查 use_model.py 中的 model_path 是否指向正确位置
找不到 CSV 文件 确保 FER-2013/ 目录下有 train.csv、val.csv
摄像头无法打开 检查摄像头权限，或修改 cv2.VideoCapture(0) 中的设备 ID
CUDA 不可用 代码已自动回退到 CPU，无需修改
🎯 情绪类别说明
所有模型识别 7 种情绪：

英文 中文
angry 愤怒
disgust 厌恶
fear 恐惧
happy 快乐
sad 悲伤
surprise 惊讶
neutral 平静
