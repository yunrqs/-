"""dlib 68 点关键点的嘴部与头部姿态几何计算（移植自 openface 疲劳检测示例）。"""

import cv2
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

    MAR 越大说明嘴巴张得越开。打哈欠时嘴巴会持续张开，因此 MAR 会明显升高。
    """
    p = landmarks
    a = distance(p[61], p[67])
    b = distance(p[62], p[66])
    c = distance(p[63], p[65])
    d = distance(p[MOUTH_LEFT], p[MOUTH_RIGHT])
    if d == 0:
        return 0.0
    return (a + b + c) / (3.0 * d)


def head_pose(landmarks, frame_size):
    """使用 solvePnP 计算头部姿态角，返回 (pitch, yaw, roll)，单位：度。

    无人脸或解算失败时返回 None。
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