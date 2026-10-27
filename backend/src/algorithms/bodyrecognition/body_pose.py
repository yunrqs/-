"""基于 PyTorch OpenPose 的上半身肢体动作检测器。

移植自 ``pytorch-openpose``（CMU OpenPose 的 PyTorch 复现）的身体姿态估计，
只保留身体（Body）模型，用于补充 dlib 68 点无法覆盖的上半身动作：

- 肩 / 肘 / 腕 / 颈 / 鼻 / 髋 关键点；
- 左右手肘弯曲角、左右手腕相对肩部高度（归一化）；
- 身体侧倾角（颈-髋连线与竖直方向夹角）、肩部倾斜角。

OpenPose 在 CPU 上推理较慢，因此这里内置隔帧策略：每隔 ``interval`` 帧做一次
完整推理，其余帧复用上一次结果，避免拖慢眨眼 / 情绪 / 打哈欠等其他模块。
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from backend.src.algorithms.bodyrecognition.types import BodyPoseResult

# OpenPose 身体模型 18 个关键点（COCO 顺序，0 起始）。
# 0=nose  1=neck  2=R-shoulder  3=R-elbow  4=R-wrist
# 5=L-shoulder  6=L-elbow  7=L-wrist  8=R-hip  9=R-knee  10=R-ankle
# 11=L-hip  12=L-knee  13=L-ankle  14=R-eye  15=L-eye  16=R-ear  17=L-ear
KEYPOINT = {
    "nose": 0,
    "neck": 1,
    "right_shoulder": 2,
    "right_elbow": 3,
    "right_wrist": 4,
    "left_shoulder": 5,
    "left_elbow": 6,
    "left_wrist": 7,
    "right_hip": 8,
    "left_hip": 11,
}


def _make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


class _BodyPoseModel(nn.Module):
    """CMU OpenPose 身体姿态网络（caffemodel -> pytorch 权重结构一致）。"""

    def __init__(self):
        super().__init__()
        no_relu_layers = ["conv5_5_CPM_L1", "conv5_5_CPM_L2", "Mconv7_stage2_L1",
                          "Mconv7_stage2_L2", "Mconv7_stage3_L1", "Mconv7_stage3_L2",
                          "Mconv7_stage4_L1", "Mconv7_stage4_L2", "Mconv7_stage5_L1",
                          "Mconv7_stage5_L2", "Mconv7_stage6_L1", "Mconv7_stage6_L1"]
        block0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]),
            ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]),
            ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]),
            ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]),
            ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3_CPM", [512, 256, 3, 1, 1]),
            ("conv4_4_CPM", [256, 128, 3, 1, 1]),
        ])
        block1_1 = OrderedDict([
            ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
            ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
        ])
        block1_2 = OrderedDict([
            ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
            ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
        ])
        blocks = {}
        blocks["block1_1"] = block1_1
        blocks["block1_2"] = block1_2
        self.model0 = _make_layers(block0, no_relu_layers)

        for i in range(2, 7):
            blocks["block%d_1" % i] = OrderedDict([
                ("Mconv1_stage%d_L1" % i, [185, 128, 7, 1, 3]),
                ("Mconv2_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv4_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv6_stage%d_L1" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d_L1" % i, [128, 38, 1, 1, 0]),
            ])
            blocks["block%d_2" % i] = OrderedDict([
                ("Mconv1_stage%d_L2" % i, [185, 128, 7, 1, 3]),
                ("Mconv2_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv4_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv6_stage%d_L2" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d_L2" % i, [128, 19, 1, 1, 0]),
            ])

        for k in blocks.keys():
            blocks[k] = _make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks["block1_1"]
        self.model2_1 = blocks["block2_1"]
        self.model3_1 = blocks["block3_1"]
        self.model4_1 = blocks["block4_1"]
        self.model5_1 = blocks["block5_1"]
        self.model6_1 = blocks["block6_1"]
        self.model1_2 = blocks["block1_2"]
        self.model2_2 = blocks["block2_2"]
        self.model3_2 = blocks["block3_2"]
        self.model4_2 = blocks["block4_2"]
        self.model5_2 = blocks["block5_2"]
        self.model6_2 = blocks["block6_2"]

    def forward(self, x):
        out1 = self.model0(x)
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        return out6_1, out6_2


def _transfer(model, model_weights):
    """把 caffemodel2pytorch 权重映射到当前 ``state_dict`` 的 key。"""
    transferred = {}
    for weights_name in model.state_dict().keys():
        converted = ".".join(weights_name.split(".")[1:])
        transferred[weights_name] = model_weights[converted]
    return transferred


def _pad_right_down_corner(img, stride, pad_value):
    h, w = img.shape[:2]
    pad = [0, 0, 0 if (h % stride == 0) else stride - (h % stride),
           0 if (w % stride == 0) else stride - (w % stride)]
    padded = np.pad(img, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
                    mode="constant", constant_values=pad_value)
    return padded, pad


def _angle_deg(a, b, c) -> Optional[float]:
    """三点夹角，顶点为 ``b``，单位：度。"""
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


class BodyPoseDetector:
    """OpenPose 上半身肢体动作检测器，带有隔帧缓存。"""

    def __init__(self, model_path: str, interval: int = 5, device: str = "auto"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"OpenPose 身体模型不存在：{model_path}")
        self._model_path = model_path
        if interval <= 0:
            raise ValueError("interval 必须 > 0")
        self._interval = interval
        self._frame_index = 0
        self._last: Optional[BodyPoseResult] = None
        self._device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else
            ("cuda" if device == "cuda" else "cpu"))
        self._model = self._load_model()

    def _load_model(self):
        model = _BodyPoseModel()
        state = torch.load(self._model_path, map_location="cpu")
        model.load_state_dict(_transfer(model, state))
        model.to(self._device)
        model.eval()
        return model

    def process_frame(self, frame) -> BodyPoseResult:
        """输入一帧 BGR 图像，返回 :class:`BodyPoseResult`。

        每 ``interval`` 帧做一次完整推理，其余帧复用上一次结果。
        """
        self._frame_index += 1
        if self._last is not None and self._frame_index % self._interval != 0:
            return self._last

        candidate, subset = self._estimate(frame)
        result = self._to_result(candidate, subset)
        self._last = result
        return result

    def reset(self) -> None:
        """清空隔帧缓存（供会话重置时调用）。"""
        self._frame_index = 0
        self._last = None

    def _estimate(self, ori_img):
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        pad_value = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / ori_img.shape[0] for x in scale_search]

        heatmap_avg = np.zeros((ori_img.shape[0], ori_img.shape[1], 19))
        paf_avg = np.zeros((ori_img.shape[0], ori_img.shape[1], 38))

        for m in multiplier:
            image_to_test = cv2.resize(ori_img, (0, 0), fx=m, fy=m,
                                       interpolation=cv2.INTER_CUBIC)
            image_padded, pad = _pad_right_down_corner(image_to_test, stride, pad_value)
            im = np.transpose(np.float32(image_padded[:, :, :, np.newaxis]),
                              (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)
            data = torch.from_numpy(im).float().to(self._device)
            with torch.no_grad():
                paf_tensor, heatmap_tensor = self._model(data)

            heatmap = np.transpose(np.squeeze(heatmap_tensor.cpu().numpy()), (1, 2, 0))
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride,
                                 interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:image_padded.shape[0] - pad[2],
                              :image_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (ori_img.shape[1], ori_img.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)

            paf = np.transpose(np.squeeze(paf_tensor.cpu().numpy()), (1, 2, 0))
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride,
                             interpolation=cv2.INTER_CUBIC)
            paf = paf[:image_padded.shape[0] - pad[2],
                      :image_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (ori_img.shape[1], ori_img.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)
            paf_avg += paf / len(multiplier)

        all_peaks = []
        peak_counter = 0
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = cv2.GaussianBlur(map_ori, (0, 0), 3)
            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right,
                 one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_id = [peaks_with_score[i] + (peak_id[i],)
                             for i in range(len(peak_id))]
            all_peaks.append(peaks_with_id)
            peak_counter += len(peaks)

        limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                    [1, 16], [16, 18], [3, 17], [6, 18]]
        map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20],
                   [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50],
                   [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(map_idx)):
            score_mid = paf_avg[:, :, [x - 19 for x in map_idx[k]]]
            cand_a = all_peaks[limb_seq[k][0] - 1]
            cand_b = all_peaks[limb_seq[k][1] - 1]
            n_a = len(cand_a)
            n_b = len(cand_b)
            if n_a != 0 and n_b != 0:
                connection_candidate = []
                for i in range(n_a):
                    for j in range(n_b):
                        vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(zip(
                            np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                            np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))
                        vec_x = np.array([score_mid[int(round(startend[I][1])),
                                                   int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])),
                                                   int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * ori_img.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior,
                                 score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2],
                                              reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack(
                            [connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                        if len(connection) >= min(n_a, n_b):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(map_idx)):
            if k not in special_k:
                part_as = connection_all[k][:, 0]
                part_bs = connection_all[k][:, 1]
                index_a, index_b = np.array(limb_seq[k]) - 1
                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_b] != part_bs[i]:
                            subset[j][index_b] = part_bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_bs[i].astype(int), 2] + \
                                connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) +
                                      (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][index_b] = part_bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + \
                                connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[index_a] = part_as[i]
                        row[index_b] = part_bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                            connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        delete_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                delete_idx.append(i)
        if delete_idx:
            subset = np.delete(subset, delete_idx, axis=0)

        return candidate, subset

    def _to_result(self, candidate, subset) -> BodyPoseResult:
        if subset is None or len(subset) == 0:
            return BodyPoseResult(person_detected=False)

        # 驾驶场景通常画面中只有一个人，取关键点数量最多者。
        person = subset[int(np.argmax(subset[:, -1]))]

        def point(key):
            cid = int(person[KEYPOINT[key]])
            if cid < 0 or cid >= len(candidate):
                return None
            return [float(candidate[cid][0]), float(candidate[cid][1])]

        nose = point("nose")
        neck = point("neck")
        left_shoulder = point("left_shoulder")
        right_shoulder = point("right_shoulder")
        left_elbow = point("left_elbow")
        right_elbow = point("right_elbow")
        left_wrist = point("left_wrist")
        right_wrist = point("right_wrist")
        left_hip = point("left_hip")
        right_hip = point("right_hip")

        # 肩宽用于归一化手腕高度，缺失时退回 1 避免除零。
        shoulder_width = 1.0
        if left_shoulder and right_shoulder:
            shoulder_width = max(
                math.hypot(left_shoulder[0] - right_shoulder[0],
                           left_shoulder[1] - right_shoulder[1]), 1.0)

        left_elbow_angle = right_elbow_angle = None
        if left_shoulder and left_elbow and left_wrist:
            left_elbow_angle = _angle_deg(left_shoulder, left_elbow, left_wrist)
        if right_shoulder and right_elbow and right_wrist:
            right_elbow_angle = _angle_deg(right_shoulder, right_elbow, right_wrist)

        # 手腕相对肩部高度（正值 = 手腕高于肩，即举手 / 抬臂）。
        left_wrist_raise = right_wrist_raise = None
        if left_shoulder and left_wrist:
            left_wrist_raise = (left_shoulder[1] - left_wrist[1]) / shoulder_width
        if right_shoulder and right_wrist:
            right_wrist_raise = (right_shoulder[1] - right_wrist[1]) / shoulder_width

        lean_angle = None
        if neck and left_hip and right_hip:
            hip_mid = [(left_hip[0] + right_hip[0]) / 2.0,
                       (left_hip[1] + right_hip[1]) / 2.0]
            # 颈-髋连线相对竖直向下的夹角（图像坐标 y 向下）。
            lean_angle = float(np.degrees(
                np.arctan2(hip_mid[0] - neck[0], hip_mid[1] - neck[1])))

        shoulder_tilt = None
        if left_shoulder and right_shoulder:
            dx = left_shoulder[0] - right_shoulder[0]
            dy = left_shoulder[1] - right_shoulder[1]
            shoulder_tilt = float(np.degrees(np.arctan2(dy, dx)))

        return BodyPoseResult(
            person_detected=True,
            nose=nose,
            neck=neck,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
            left_elbow=left_elbow,
            right_elbow=right_elbow,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            left_hip=left_hip,
            right_hip=right_hip,
            left_elbow_angle=left_elbow_angle,
            right_elbow_angle=right_elbow_angle,
            left_wrist_raise=left_wrist_raise,
            right_wrist_raise=right_wrist_raise,
            lean_angle=lean_angle,
            shoulder_tilt=shoulder_tilt,
        )

    def close(self) -> None:
        """释放模型显存 / 内存。"""
        self._model = None
        self._last = None
        self._frame_index = 0