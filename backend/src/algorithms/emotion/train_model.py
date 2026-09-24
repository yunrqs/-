from backend.src.config.paths import EMOTION_MODELS, FER2013_ROOT, OUTPUTS_ROOT
# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


# 配置
class Config:
    # 数据集路径
    data_path = "dataset/"
    model_save_path = str(EMOTION_MODELS / 'best_model.pth')

    # 训练参数
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    num_classes = 7  # 7种情绪

    # 图像参数
    img_size = 48  # FER2013使用48x48
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 情绪标签
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class EmotionDataset(Dataset):
    """情绪数据集类"""

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历数据集
        for label_idx, emotion in enumerate(Config.emotions):
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_file)
                        self.images.append(img_path)
                        self.labels.append(label_idx)

        print(f"加载了 {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        image = cv2.resize(image, (Config.img_size, Config.img_size))

        # 转换为3通道（适配预训练模型）
        image = np.stack([image, image, image], axis=2)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class EmotionCNN(nn.Module):
    """改进的CNN模型，准确率更高"""

    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # 使用预训练的ResNet18作为backbone
        self.backbone = models.resnet18(pretrained=True)

        # 修改第一层以接受灰度图（但我们已经转为3通道）
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ImprovedEmotionCNN(nn.Module):
    """自定义的高精度CNN模型"""

    def __init__(self, num_classes=7):
        super(ImprovedEmotionCNN, self).__init__()

        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 卷积块3
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

        # 卷积块4
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

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
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


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """训练模型"""

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in pbar:
            images = images.to(Config.device)
            labels = labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})

        train_acc = 100. * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.device)
                labels = labels.to(Config.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)

        print(
            f'\nEpoch {epoch + 1}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_path)
            print(f'✓ 保存最佳模型，验证准确率: {val_acc:.2f}%')

    return train_losses, val_losses, train_accs, val_accs


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')

    plt.tight_layout()
    (OUTPUTS_ROOT / 'emotion').mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUTS_ROOT / 'emotion' / 'training_history.png')
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("情绪识别模型训练")
    print("=" * 60)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    print("加载训练数据...")
    full_dataset = EmotionDataset(Config.data_path, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 创建模型
    model = ImprovedEmotionCNN(num_classes=Config.num_classes).to(Config.device)
    # 或者使用预训练模型
    # model = EmotionCNN(num_classes=Config.num_classes).to(Config.device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, Config.num_epochs
    )

    # 绘制训练曲线
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    print("\n" + "=" * 60)
    print(f"训练完成！最佳模型保存在: {Config.model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
