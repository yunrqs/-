# use_model.py - 实时识别脚本
import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
import time
import os


# 模型定义（必须与训练时完全一致）
class ImprovedEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ImprovedEmotionCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EmotionRecognizer:
    def __init__(self, model_path="../models/fer2013_best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # 加载模型
        self.model = ImprovedEmotionCNN(num_classes=7).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ 模型加载成功: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
            return

        self.model.eval()
        self.history = deque(maxlen=10)  # 平滑结果

        # 颜色映射
        self.colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 255, 255),
            'fear': (255, 0, 255),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (255, 255, 0),
            'neutral': (200, 200, 200)
        }

    def preprocess_face(self, face_img):
        """预处理面部图像"""
        # 转换为灰度图
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # 调整大小到48x48
        resized = cv2.resize(gray, (48, 48))

        # 归一化
        normalized = resized.astype(np.float32) / 255.0

        # 转换为tensor并添加维度
        input_tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)

        return input_tensor.to(self.device)

    def predict(self, face_img):
        """预测情绪"""
        if face_img is None or face_img.size == 0:
            return None, None, None

        # 预处理
        input_tensor = self.preprocess_face(face_img)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()

        emotion = self.emotions[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()

        # 获取所有情绪的概率
        all_probs = {self.emotions[i]: probabilities[0][i].item()
                     for i in range(len(self.emotions))}

        # 历史平滑
        self.history.append(emotion)
        from collections import Counter
        smooth_emotion = Counter(self.history).most_common(1)[0][0]

        return smooth_emotion, confidence, all_probs

    def draw_results(self, frame, face_rect, emotion, confidence):
        """在图像上绘制结果"""
        x, y, w, h = face_rect
        color = self.colors.get(emotion, (255, 255, 255))

        # 绘制人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 显示情绪和置信度
        text = f"{emotion.upper()} ({confidence * 100:.1f}%)"
        label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # 绘制背景框
        cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame


def main():
    print("=" * 60)
    print("FER2013情绪识别系统")
    print("=" * 60)

    # 初始化识别器
    recognizer = EmotionRecognizer()

    # 初始化人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n✓ 摄像头启动成功")
    print("\n情绪类别: angry, disgust, fear, happy, sad, surprise, neutral")
    print("\n按键控制:")
    print("  q - 退出")
    print("  s - 截图")
    print("  d - 显示详细概率")
    print("=" * 60)

    # 统计
    stats = {emotion: 0 for emotion in recognizer.emotions}
    frame_count = 0
    last_time = time.time()
    fps_counter = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # 对每个人脸进行情绪识别
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            emotion, confidence, all_probs = recognizer.predict(face_roi)

            if emotion:
                # 更新统计
                stats[emotion] += 1

                # 绘制结果
                frame = recognizer.draw_results(frame, (x, y, w, h), emotion, confidence)

        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if current_time != last_time else 0
        last_time = current_time
        fps_counter.append(fps)
        avg_fps = sum(fps_counter) / len(fps_counter)

        # 显示FPS和统计
        frame_count += 1
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 显示实时统计
        y_offset = frame.shape[0] - 80
        cv2.putText(frame, "Statistics:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        top_emotions = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (emotion, count) in enumerate(top_emotions):
            color = recognizer.colors.get(emotion, (255, 255, 255))
            cv2.putText(frame, f"{emotion}: {count}", (10, y_offset + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 显示图像
        cv2.imshow('Emotion Recognition System', frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"✓ 截图保存: {filename}")
        elif key == ord('d') and len(faces) > 0:
            print(f"\n详细概率 (帧 {frame_count}):")
            for emotion, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 50)
                print(f"  {emotion:10}: {prob * 100:5.1f}% {bar}")

    # 清理
    cap.release()
    cv2.destroyAllWindows()

    # 最终统计
    print("\n" + "=" * 60)
    print("最终统计")
    print("=" * 60)
    total = sum(stats.values())
    for emotion, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {emotion:10}: {count:5} ({percentage:5.1f}%) {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()