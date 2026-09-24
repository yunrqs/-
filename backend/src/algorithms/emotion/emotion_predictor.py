from backend.src.config.paths import EMOTION_MODELS, FER2013_ROOT, OUTPUTS_ROOT
# emotion_predictor.py
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from collections import deque
import time


class EmotionPredictor:
    """高精度情绪识别器"""

    def __init__(self, model_path=str(EMOTION_MODELS / 'best_model.pth')):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 历史结果平滑
        self.history = deque(maxlen=10)

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

    def _load_model(self, model_path):
        """加载模型"""
        from train_model import ImprovedEmotionCNN

        model = ImprovedEmotionCNN(num_classes=7).to(self.device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ 模型加载成功: {model_path}")
        except FileNotFoundError:
            print(f"⚠️ 模型文件不存在: {model_path}")
            print("将使用随机初始化的模型")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")

        return model

    def preprocess_face(self, face_img):
        """预处理面部图像"""
        # 转换为灰度图
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # 调整大小
        resized = cv2.resize(gray, (48, 48))

        # 直方图均衡化（增强对比度）
        equalized = cv2.equalizeHist(resized)

        # 归一化
        normalized = equalized / 255.0

        # 转换为3通道
        normalized = np.stack([normalized, normalized, normalized], axis=2)

        return normalized

    def predict(self, face_img):
        """预测情绪"""
        if face_img is None or face_img.size == 0:
            return None, None

        # 预处理
        processed = self.preprocess_face(face_img)

        # 转换为tensor
        input_tensor = self.transform(processed).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()

        # 获取结果
        emotion = self.emotions[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()

        # 获取所有情绪的概率
        all_probs = {self.emotions[i]: probabilities[0][i].item()
                     for i in range(len(self.emotions))}

        # 历史平滑
        self.history.append(emotion)
        if len(self.history) > 0:
            from collections import Counter
            smooth_emotion = Counter(self.history).most_common(1)[0][0]
        else:
            smooth_emotion = emotion

        return smooth_emotion, confidence, all_probs

    def draw_results(self, frame, face_rect, emotion, confidence, all_probs=None):
        """在图像上绘制结果"""
        x, y, w, h = face_rect

        # 选择颜色
        color = self.colors.get(emotion, (255, 255, 255))

        # 绘制人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 显示情绪和置信度
        text = f"{emotion.upper()} ({confidence * 100:.1f}%)"
        label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # 绘制背景框
        cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 显示概率条（可选）
        if all_probs:
            bar_x = x
            bar_y = y + h + 5
            bar_width = w
            bar_height = 20

            # 绘制背景
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

            # 绘制概率条
            prob_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + prob_width, bar_y + bar_height), color, -1)

        return frame


class RealTimeEmotionRecognizer:
    """实时情绪识别系统"""

    def __init__(self, model_path=str(EMOTION_MODELS / 'best_model.pth')):
        self.predictor = EmotionPredictor(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 性能统计
        self.fps_counter = deque(maxlen=30)
        self.stats = {emotion: 0 for emotion in self.predictor.emotions}

    def run(self, camera_id=0):
        """运行实时识别"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 60)
        print("高精度情绪识别系统")
        print("=" * 60)
        print(f"设备: {self.predictor.device}")
        print(f"情绪类别: {', '.join(self.predictor.emotions)}")
        print("\n按键控制:")
        print("  q - 退出")
        print("  s - 截图")
        print("  r - 重置统计")
        print("  p - 打印详细概率")
        print("=" * 60)

        last_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            # 对每个人脸进行情绪识别
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                emotion, confidence, all_probs = self.predictor.predict(face_roi)

                if emotion:
                    # 更新统计
                    self.stats[emotion] = self.stats.get(emotion, 0) + 1

                    # 绘制结果
                    frame = self.predictor.draw_results(frame, (x, y, w, h),
                                                        emotion, confidence, all_probs)

            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if current_time != last_time else 0
            last_time = current_time
            self.fps_counter.append(fps)
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)

            # 显示FPS和统计
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_count += 1
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 显示实时统计
            y_offset = frame.shape[0] - 100
            cv2.putText(frame, "Live Statistics:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i, (emotion, count) in enumerate(sorted(self.stats.items(),
                                                        key=lambda x: x[1], reverse=True)[:3]):
                color = self.predictor.colors.get(emotion, (255, 255, 255))
                cv2.putText(frame, f"{emotion}: {count}", (10, y_offset + 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 显示图像
            cv2.imshow('High-Precision Emotion Recognition', frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_screenshot_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"✓ 截图保存: {filename}")
            elif key == ord('r'):
                self.stats = {emotion: 0 for emotion in self.predictor.emotions}
                print("✓ 统计重置")
            elif key == ord('p') and 'faces' in locals() and len(faces) > 0:
                print(f"\n详细概率 (帧 {frame_count}):")
                for emotion, prob in all_probs.items():
                    bar = "█" * int(prob * 50)
                    print(f"  {emotion:10}: {prob * 100:5.1f}% {bar}")

        cap.release()
        cv2.destroyAllWindows()

        # 打印最终统计
        print("\n" + "=" * 60)
        print("最终统计")
        print("=" * 60)
        total = sum(self.stats.values())
        for emotion, count in self.stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  {emotion:10}: {count:5} ({percentage:5.1f}%) {bar}")
        print("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='高精度情绪识别系统')
    parser.add_argument('--model', type=str, default=str(EMOTION_MODELS / 'best_model.pth'),
                        help='模型文件路径')
    parser.add_argument('--image', type=str, help='单张图片识别')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')

    args = parser.parse_args()

    if args.image:
        # 单张图片识别
        predictor = EmotionPredictor(args.model)
        image = cv2.imread(args.image)

        if image is not None:
            # 检测人脸
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_roi = image[y:y + h, x:x + w]
                emotion, confidence, all_probs = predictor.predict(face_roi)

                if emotion:
                    image = predictor.draw_results(image, (x, y, w, h), emotion, confidence, all_probs)

                    print(f"\n识别结果:")
                    print(f"  情绪: {emotion}")
                    print(f"  置信度: {confidence * 100:.1f}%")
                    print(f"\n详细概率:")
                    for e, p in all_probs.items():
                        print(f"  {e:10}: {p * 100:5.1f}%")

            cv2.imshow('Emotion Recognition', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"无法读取图片: {args.image}")

    else:
        # 实时视频识别
        recognizer = RealTimeEmotionRecognizer(args.model)
        recognizer.run(args.camera)


if __name__ == "__main__":
    main()