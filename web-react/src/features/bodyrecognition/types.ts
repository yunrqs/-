// 疲劳驾驶识别的前端展示类型，字段与后端 bodyrecognition 模块返回保持一致。

// OpenPose 上半身肢体动作结果，字段与后端 BodyPoseResult 的 JSON 序列化一致。
export interface BodyPoseStatus {
  person_detected: boolean;
  nose: [number, number] | null;
  neck: [number, number] | null;
  left_shoulder: [number, number] | null;
  right_shoulder: [number, number] | null;
  left_elbow: [number, number] | null;
  right_elbow: [number, number] | null;
  left_wrist: [number, number] | null;
  right_wrist: [number, number] | null;
  left_hip: [number, number] | null;
  right_hip: [number, number] | null;
  left_elbow_angle: number | null;
  right_elbow_angle: number | null;
  left_wrist_raise: number | null;
  right_wrist_raise: number | null;
  lean_angle: number | null;
  shoulder_tilt: number | null;
}

export interface FatigueStatus {
  mar: number | null;
  pitch: number | null;
  yaw: number | null;
  roll: number | null;
  is_yawning: boolean;
  yawn_duration: number;
  is_drowsy: boolean;
  pose: BodyPoseStatus | null;
}