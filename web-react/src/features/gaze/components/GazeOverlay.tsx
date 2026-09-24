import type { GazePoint } from '../types';
export default function GazeOverlay({ point }: { point: GazePoint | null }) {
  if (!point?.inViewport) return null;
  return <div className="gaze-dot" aria-hidden="true" style={{ left: point.x, top: point.y }} />;
}
