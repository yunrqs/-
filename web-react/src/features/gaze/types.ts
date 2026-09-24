export interface GazePoint {
  x: number; y: number; normalizedX: number; normalizedY: number;
  viewportWidth: number; viewportHeight: number; timestamp: number; inViewport: boolean;
}
export interface GazeFrame { point: GazePoint | null; faceDetected: boolean; fps: number }
export interface ValidationResult { meanError: number | null; validRatio: number }
