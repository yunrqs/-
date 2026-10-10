// 疲劳驾驶识别的前端展示类型，字段与后端 bodyrecognition 模块返回保持一致。
export interface FatigueStatus {
  mar: number | null;
  pitch: number | null;
  yaw: number | null;
  roll: number | null;
  is_yawning: boolean;
  yawn_duration: number;
  is_drowsy: boolean;
}