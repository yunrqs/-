from backend.src.config.paths import EMOTION_MODELS, FER2013_ROOT, OUTPUTS_ROOT
# predict_video.py - 识别视频文件
from backend.src.algorithms.emotion.use_model import EmotionRecognizer
import cv2

recognizer = EmotionRecognizer()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开视频文件
cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        emotion, confidence, _ = recognizer.predict(face_roi)

        if emotion:
            frame = recognizer.draw_results(frame, (x, y, w, h), emotion, confidence)

    cv2.imshow('Video Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()