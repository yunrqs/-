export class GazeRegression {
  reset(): void;
  features(eyes: { left: EyePatch; right: EyePatch }): number[];
  add(features: number[], x: number, y: number): void;
  predict(features: number[]): { x: number; y: number } | null;
  clearFilter(): void;
}
export interface EyePatch { patch: ImageData; width: number; height: number }
