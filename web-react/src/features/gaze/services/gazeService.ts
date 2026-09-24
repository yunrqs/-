import { GazeRegression, type EyePatch } from '../core/regression.mjs';
import type { GazeFrame, GazePoint } from '../types';

interface Landmark { x: number; y: number; z: number }
interface FaceMesh {
  setOptions(options: Record<string, number | boolean>): void;
  onResults(callback: (results: { multiFaceLandmarks: Landmark[][] }) => void): void;
  send(input: { image: HTMLCanvasElement }): Promise<void>;
  initialize(): Promise<void>;
  close(): Promise<void>;
}
declare global { interface Window {
  FaceMesh?: new (options: { locateFile: (file: string) => string }) => FaceMesh;
} }
let scriptPromise: Promise<void> | null = null;
function loadRuntime() {
  if (window.FaceMesh) return Promise.resolve();
  if (!scriptPromise) scriptPromise = new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/mediapipe/face_mesh/face_mesh.js';
    script.onload = () => resolve();
    script.onerror = () => { script.remove(); reject(new Error('视线模型加载失败，请检查本地模型资源')); };
    document.head.appendChild(script);
  }).catch(error => { scriptPromise = null; throw error; });
  return scriptPromise;
}

const EYES = [[466,388,387,386,385,384,398,263,249,390,373,374,380,381,382,362],
  [246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133]];

export class GazeService {
  private model: FaceMesh | null = null;
  private regression = new GazeRegression();
  private canvas = document.createElement('canvas');
  private features: number[] | null = null;
  private featureTime = 0;
  private sampledTime = -1;
  private ready = false;
  private generation = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private pending: Promise<void> = Promise.resolve();
  private disposed = false;
  private initializing: Promise<void> | null = null;

  private initialize() {
    if (!this.initializing) this.initializing = (async () => {
      await loadRuntime();
      if (this.disposed) return;
      if (!window.FaceMesh) throw new Error('MediaPipe 初始化失败');
      this.model = new window.FaceMesh({ locateFile: file => `/mediapipe/face_mesh/${file}` });
      this.model.setOptions({ maxNumFaces: 1, refineLandmarks: false,
        selfieMode: false, minDetectionConfidence: .5, minTrackingConfidence: .5 });
      this.model.onResults(results => {
        this.features = null;
        const landmarks = results.multiFaceLandmarks?.[0];
        if (!landmarks) return;
        const context = this.canvas.getContext('2d', { willReadFrequently: true })!;
        const patches = EYES.map(indices => {
          const points = indices.map(i => landmarks[i]);
          if (points.some(p => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) return null;
          const x = Math.max(0, Math.floor(Math.min(...points.map(p => p.x)) * this.canvas.width));
          const y = Math.max(0, Math.floor(Math.min(...points.map(p => p.y)) * this.canvas.height));
          const right = Math.min(this.canvas.width, Math.ceil(Math.max(...points.map(p => p.x)) * this.canvas.width));
          const bottom = Math.min(this.canvas.height, Math.ceil(Math.max(...points.map(p => p.y)) * this.canvas.height));
          const width = right - x, height = bottom - y;
          return width > 1 && height > 1 ? { patch: context.getImageData(x, y, width, height), width, height } : null;
        });
        if (patches.some(p => !p)) return;
        this.features = this.regression.features({ left: patches[0] as EyePatch, right: patches[1] as EyePatch });
        this.featureTime = performance.now();
      });
      await this.model.initialize();
    })();
    return this.initializing;
  }

  async start(video: HTMLVideoElement, onFrame: (frame: GazeFrame) => void, onError: (error: unknown) => void) {
    this.pause();
    const generation = this.generation;
    await this.pending;
    await this.initialize();
    if (generation !== this.generation || this.disposed) return;
    let lastFrame = performance.now();
    const tick = async () => {
      if (generation !== this.generation || this.disposed) return;
      try {
        if (!video.videoWidth || video.readyState < 2) throw new Error('摄像头画面尚未就绪');
        this.canvas.width = 640;
        this.canvas.height = Math.round(640 * video.videoHeight / video.videoWidth);
        // Raw unmirrored frame for both landmarks and eye crops; CSS preview alone is mirrored.
        this.canvas.getContext('2d', { willReadFrequently: true })!.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
        this.features = null;
        await this.model!.send({ image: this.canvas });
        if (generation !== this.generation || this.disposed) return;
        const now = performance.now();
        const prediction = this.features && this.ready ? this.regression.predict(this.features) : null;
        if (!this.features) this.regression.clearFilter();
        let point: GazePoint | null = null;
        if (prediction) {
          const { x, y } = prediction, width = window.innerWidth, height = window.innerHeight;
          point = { x, y, normalizedX: x / width, normalizedY: y / height,
            viewportWidth: width, viewportHeight: height, timestamp: Date.now(),
            inViewport: x >= 0 && y >= 0 && x < width && y < height };
        }
        onFrame({ point, faceDetected: !!this.features, fps: 1000 / Math.max(1, now - lastFrame) });
        lastFrame = now;
        this.timer = setTimeout(() => { this.pending = tick(); }, 50);
      } catch (error) {
        if (generation === this.generation && !this.disposed) { this.pause(); onError(error); }
      }
    };
    this.pending = tick();
  }
  pause() { this.generation++; clearTimeout(this.timer); this.features = null; this.regression.clearFilter(); }
  reset() { this.ready = false; this.regression.reset(); this.sampledTime = -1; }
  calibrate(x: number, y: number) {
    if (!this.features || performance.now() - this.featureTime > 300 || this.sampledTime === this.featureTime) return false;
    this.regression.add(this.features, x, y); this.sampledTime = this.featureTime; return true;
  }
  finishCalibration() { this.ready = true; this.regression.clearFilter(); }
  async dispose() {
    this.disposed = true; this.pause();
    await this.initializing?.catch(() => {});
    await this.pending;
    await this.model?.close();
    this.model = null;
  }
}
