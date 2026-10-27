#!/usr/bin/env python3
#
# 基于 dlib 68 点关键点的疲劳驾驶检测示例。
#
# 原 OpenFace 项目的人脸识别功能已移除，本脚本只保留
# 关键点检测能力，并在此基础上实现：
#   1. 打哈欠检测 (MAR, Mouth Aspect Ratio)
#   2. 头部姿态角计算 (pitch / yaw / roll)
#
# 依赖：dlib、opencv-python、numpy
# 运行前请确保 models/dlib/shape_predictor_68_face_landmarks.dat 存在。

import argparse
import os
import time

import cv2
import dlib
import numpy as np


# dlib 68 点关键点中与嘴部相关的索引（0 起始）
MOUTH_INNER = list(range(60, 68))          # 内唇 60-67
MOUTH_LEFT = 60                            # 内唇左角
MOUTH_RIGHT = 64                           # 内唇右角

# 头部姿态解算所用的 2D 关键点索引
POSE_IDX = {
    'nose': 30,
    'chin': 8,
    'left_eye': 36,
    'right_eye': 45,
    'left_mouth': 48,
    'right_mouth': 54,
}

# 对应的人脸 3D 模型点（毫米，通用人脸平均模型）
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # 鼻尖
    (0.0, -63.6, -62.5),         # 下巴
    (-43.3, 32.7, -26.0),        # 左眼左角
    (43.3, 32.7, -26.0),         # 右眼右角
    (-28.9, -28.9, -24.1),       # 左嘴角
    (28.9, -28.9, -24.1),        # 右嘴角
])


def distance(a, b):
    """计算两个 (x, y) 点之间的欧氏距离。"""
    return np.linalg.norm(np.array(a) - np.array(b))


def mouth_aspect_ratio(landmarks):
    """计算嘴部纵横比 (MAR)。

    MAR 越大说明嘴巴张得越开。打哈欠时嘴巴会持续张开，
    因此 MAR 会明显升高。
    """
    p = landmarks
    # 内唇上下距离取三处平均值
    a = distance(p[61], p[67])
    b = distance(p[62], p[66])
    c = distance(p[63], p[65])
    # 内唇左右宽度
    d = distance(p[MOUTH_LEFT], p[MOUTH_RIGHT])
    if d == 0:
        return 0.0
    return (a + b + c) / (3.0 * d)


def head_pose(landmarks, frame_size):
    """使用 solvePnP 计算头部姿态角。

    返回 (pitch, yaw, roll)，单位：度。
    """
    h, w = frame_size[:2]

    image_points = np.array([
        landmarks[POSE_IDX['nose']],
        landmarks[POSE_IDX['chin']],
        landmarks[POSE_IDX['left_eye']],
        landmarks[POSE_IDX['right_eye']],
        landmarks[POSE_IDX['left_mouth']],
        landmarks[POSE_IDX['right_mouth']],
    ], dtype=np.float64)

    # 相机内参：焦距近似取图像宽度，主点在图像中心
    focal = w
    camera_matrix = np.array([
        [focal, 0, w / 2.0],
        [0, focal, h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # pitch / yaw / roll
    return euler_angles.flatten()


def draw_status(frame, mar, is_yawning, yawn_duration, pose):
    """在画面上叠加状态文字。"""
    h, w = frame.shape[:2]

    # MAR 数值
    cv2.putText(frame, "MAR: {:.2f}".format(mar),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 头部姿态角
    if pose is not None:
        pitch, yaw, roll = pose
        cv2.putText(
            frame,
            "pitch:{:+.1f} yaw:{:+.1f} roll:{:+.1f}".format(pitch, yaw, roll),
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "pose: -", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 打哈欠状态
    if is_yawning:
        cv2.putText(frame, "YAWNING {:.1f}s".format(yawn_duration),
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "normal", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(
        description="基于 dlib 68 点关键点的打哈欠疲劳驾驶检测")
    parser.add_argument(
        '--predictor',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', 'models', 'dlib', 'shape_predictor_68_face_landmarks.dat'),
        help="dlib 68 点关键点模型路径")
    parser.add_argument(
        '--mar-threshold', type=float, default=0.6,
        help="判定为张嘴（打哈欠）的 MAR 阈值")
    parser.add_argument(
        '--yawn-seconds', type=float, default=3.0,
        help="连续张嘴超过该秒数即告警")
    parser.add_argument('--camera', type=int, default=0,
                        help="摄像头设备号，0 为默认摄像头")
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    if not os.path.isfile(args.predictor):
        raise SystemExit(
            "未找到关键点模型：{}\n"
            "请先下载 dlib 68 点模型到 models/dlib/ 目录。".format(
                args.predictor))

    # dlib 人脸检测器 + 关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("无法打开摄像头 {}".format(args.camera))

    yawn_start_time = None
    is_yawning = False

    print("按 'q' 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        mar = 0.0
        pose = None
        current_is_yawning = False

        for rect in faces:
            shape = predictor(gray, rect)
            landmarks = [(p.x, p.y) for p in shape.parts()]

            mar = mouth_aspect_ratio(landmarks)
            pose = head_pose(landmarks, frame.shape)

            # 打哈欠判定：MAR 持续超过阈值
            if mar > args.mar_threshold:
                current_is_yawning = True
                if yawn_start_time is None:
                    yawn_start_time = time.time()
            else:
                yawn_start_time = None

            # 画人脸框和嘴部关键点
            cv2.rectangle(frame,
                          (rect.left(), rect.top()),
                          (rect.right(), rect.bottom()),
                          (0, 255, 0), 2)
            for idx in MOUTH_INNER:
                cv2.circle(frame, landmarks[idx], 2, (0, 0, 255), -1)

        # 打哈欠持续时间
        if current_is_yawning:
            is_yawning = True
            yawn_duration = time.time() - yawn_start_time
        else:
            is_yawning = False
            yawn_duration = 0.0

        # 连续张嘴超过阈值 => 疲劳告警
        if is_yawning and yawn_duration >= args.yawn_seconds:
            h = frame.shape[0]
            cv2.putText(frame, "DROWSY ALERT!", (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        draw_status(frame, mar, is_yawning, yawn_duration, pose)
        cv2.imshow("Fatigue Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()