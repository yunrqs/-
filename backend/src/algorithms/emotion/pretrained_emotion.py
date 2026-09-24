from backend.src.config.paths import EMOTION_MODELS, FER2013_ROOT, OUTPUTS_ROOT
# pretrained_emotion.py - 使用预训练模型
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


class PretrainedEmotionRecognizer:
    """使用预训练模型的情绪识别器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # 使用在ImageNet上预训练的ResNet
        self.model = models.resnet50(pretrained=True)

        # 修改最后一层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 7)

        # 加载预训练的情绪识别权重（如果有）
        # self.model.load_state_dict(torch.load('resnet50_emotion.pth'))

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, face_img):
        """预测情绪"""
        # 预处理
        input_tensor = self.transform(face_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()

        emotion = self.emotions[predicted_idx]
        confidence = probabilities[0][predicted_idx].item()

        return emotion, confidence


# 快速测试
if __name__ == "__main__":
    recognizer = PretrainedEmotionRecognizer()
    print("预训练模型加载完成")