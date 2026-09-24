// 字段名与 FastAPI 返回的 JSON 保持一致，便于对照前后端代码。
export type Emotion = 'angry' | 'disgust' | 'fear' | 'happy' | 'sad' | 'surprise' | 'neutral';

export interface HealthResponse {
  ok: boolean;
  ready: boolean;
}

export interface AnalysisResult {
  frame_width: number;
  frame_height: number;
  face_detected: boolean;
  face_box: [number, number, number, number] | null;
  ear: number | null;
  threshold: number | null;
  eye_state: string;
  blink_detected: boolean;
  blink_count: number;
  blink_rate_per_min: number;
  valid_face_ratio: number;
  emotion: Emotion | null;
  emotion_confidence: number | null;
  emotion_probabilities: Partial<Record<Emotion, number>>;
}

export interface ChartPoint {
  ear: number | null;
  threshold: number | null;
}
