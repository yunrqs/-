from backend.src.config.paths import EMOTION_MODELS, FER2013_ROOT, OUTPUTS_ROOT
# predict_image.py - 识别单张图片
from backend.src.algorithms.emotion.use_model import EmotionRecognizer
import cv2

# 初始化
recognizer = EmotionRecognizer()

# 读取图片
image = cv2.imread("test_face.jpg")

# 检测人脸
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    face_roi = image[y:y + h, x:x + w]
    emotion, confidence, all_probs = recognizer.predict(face_roi)

    print(f"识别结果: {emotion}")
    print(f"置信度: {confidence * 100:.1f}%")
    print("\n详细概率:")
    for e, p in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {e}: {p * 100:.1f}%")

    # 绘制结果
    image = recognizer.draw_results(image, (x, y, w, h), emotion, confidence)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()