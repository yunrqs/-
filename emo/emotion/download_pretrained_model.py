# download_pretrained_model.py
import torch
import urllib.request
import os


def download_fer2013_model():
    """下载FER2013预训练模型"""
    # FER2013预训练模型下载链接
    model_url = "https://github.com/justusschock/torch-emotion/raw/master/emotion/models/fer2013_resnet50.pth"
    model_path = "models/fer2013_resnet50.pth"

    os.makedirs("models", exist_ok=True)

    print("下载预训练模型...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✓ 模型下载完成: {model_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载或使用自己的数据集训练")


if __name__ == "__main__":
    download_fer2013_model()