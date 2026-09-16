#  - 修复train_with_fer2013.py数据格式问题
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter


# 配置参数
class Config:
    num_classes = 7
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # 数据集路径
    data_dir = "FER-2013"
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")


# 改进的CNN模型
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


# FER2013数据集类 - 修复格式问题
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 确定列名
        if 'pixels' in self.data.columns:
            self.pixels_col = 'pixels'
        elif 'feature' in self.data.columns:
            self.pixels_col = 'feature'
        else:
            print(f"CSV列名: {self.data.columns.tolist()}")
            raise ValueError("找不到像素列")

        if 'emotion' in self.data.columns:
            self.label_col = 'emotion'
        elif 'label' in self.data.columns:
            self.label_col = 'label'
        else:
            raise ValueError("找不到标签列")

        print(f"加载 {os.path.basename(csv_file)}: {len(self.data)} 张图像")

        # 打印类别分布
        emotion_counts = Counter(self.data[self.label_col])
        for emotion_idx, count in emotion_counts.items():
            if emotion_idx < len(Config.emotions):
                print(f"  {Config.emotions[emotion_idx]}: {count} ({count / len(self.data) * 100:.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 解析像素字符串
        pixels_str = row[self.pixels_col]
        pixels = np.array([int(p) for p in pixels_str.split()], dtype=np.uint8)

        # 重塑为48x48灰度图
        image = pixels.reshape(48, 48)

        # 转换为PIL Image（transforms需要）
        image = Image.fromarray(image, mode='L')

        label = row[self.label_col]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """训练模型"""

    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(Config.device), labels.to(Config.device)

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

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.device), labels.to(Config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        # 记录
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'\nEpoch {epoch + 1}:')
        print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

        # 学习率调整
        if scheduler:
            scheduler.step(val_loss_avg)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/fer2013_best_model.pth')
            print(f'  ✓ 保存最佳模型，验证准确率: {val_acc:.2f}%')

        # 早停（可选）
        if val_acc > 75:
            print(f"\n达到目标准确率，提前停止训练")
            break

    return train_losses, val_losses, train_accs, val_accs, best_val_acc


def plot_results(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("使用FER2013数据集训练情绪识别模型")
    print("=" * 60)

    # 切换到FER-2013目录
    if not os.path.exists(Config.data_dir):
        print(f"\n错误：找不到 {Config.data_dir} 文件夹！")
        print(f"当前目录: {os.getcwd()}")
        return

    if not os.path.exists(Config.train_csv):
        print(f"\n错误：找不到训练文件 {Config.train_csv}")
        return

    print(f"\n✓ 找到数据集目录: {Config.data_dir}")
    print(f"✓ 训练集: {Config.train_csv}")
    print(f"✓ 验证集: {Config.val_csv}")
    print(f"✓ 使用设备: {Config.device}")
    print(f"✓ 情绪类别: {Config.emotions}")

    # 数据增强（使用PIL图像）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载数据集
    print("\n加载训练集...")
    train_dataset = FER2013Dataset(Config.train_csv, transform=train_transform)

    print("\n加载验证集...")
    val_dataset = FER2013Dataset(Config.val_csv, transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)

    print(f"\n训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 创建模型
    model = ImprovedEmotionCNN(num_classes=Config.num_classes).to(Config.device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 对于不平衡数据集，使用类别权重
    class_counts = [3995, 436, 4097, 7215, 4830, 3171, 4965]  # 从训练集统计
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(Config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练
    print("\n开始训练...")
    print("=" * 60)

    train_losses, val_losses, train_accs, val_accs, best_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, Config.num_epochs
    )

    # 绘制结果
    plot_results(train_losses, val_losses, train_accs, val_accs)

    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存位置: ../models/fer2013_best_model.pth")
    print("=" * 60)

    # 在测试集上评估
    if os.path.exists(Config.test_csv):
        print("\n加载测试集...")
        test_dataset = FER2013Dataset(Config.test_csv, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="测试中"):
                images, labels = images.to(Config.device), labels.to(Config.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        print(f"测试集准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()