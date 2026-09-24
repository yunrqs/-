import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import type { GazeFrame, ValidationResult } from '../types';
import type { GazeService } from '../services/gazeService';

const emptyFrame: GazeFrame = { point: null, faceDetected: false, fps: 0 };
export function useGazePrediction(video: RefObject<HTMLVideoElement | null>, cameraOn: boolean) {
  const service = useRef<GazeService | null>(null);
  const epoch = useRef(0);
  const [status, setStatus] = useState<'idle' | 'loading' | 'running' | 'paused' | 'error'>('idle');
  const [frame, setFrame] = useState<GazeFrame>(emptyFrame);
  const [calibrated, setCalibrated] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [validating, setValidating] = useState(false);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [showPoint, setShowPoint] = useState(true);
  const [message, setMessage] = useState('开启摄像头后，可独立启动视线预测。');

  const pause = useCallback(() => {
    epoch.current++; service.current?.pause();
    setStatus('paused'); setFrame(emptyFrame); setCalibrating(false); setValidating(false);
    setMessage('视线预测已关闭。');
  }, []);
  const invalidate = useCallback(() => {
    service.current?.reset(); setCalibrated(false); setCalibrating(false); setValidating(false);
    setValidation(null); setFrame(emptyFrame);
  }, []);

  useEffect(() => {
    if (!cameraOn) { pause(); invalidate(); setMessage('开启摄像头后，可独立启动视线预测。'); }
  }, [cameraOn, pause, invalidate]);
  useEffect(() => {
    const resized = () => { invalidate(); setMessage('显示区域已改变，请重新校准。'); };
    window.addEventListener('resize', resized);
    document.addEventListener('fullscreenchange', resized);
    return () => {
      epoch.current++; void service.current?.dispose().catch(() => {}); service.current = null;
      window.removeEventListener('resize', resized);
      document.removeEventListener('fullscreenchange', resized);
    };
  }, [invalidate]);

  async function start() {
    if (!cameraOn || !video.current || status === 'loading' || status === 'running') return;
    const run = ++epoch.current;
    setStatus('loading'); setMessage('正在加载本地视线模型…');
    try {
      if (!service.current) {
        const { GazeService: Service } = await import('../services/gazeService');
        if (run !== epoch.current) return;
        service.current = new Service();
      }
      await service.current.start(video.current, next => {
        if (run === epoch.current) setFrame(next);
      }, error => {
        if (run !== epoch.current) return;
        epoch.current++; setStatus('error'); setFrame(emptyFrame);
        setCalibrating(false); setValidating(false);
        setMessage(error instanceof Error ? error.message : '视线预测失败');
        const failed = service.current; service.current = null;
        void failed?.dispose().catch(() => {}); setCalibrated(false); setValidation(null);
      });
      if (run !== epoch.current) return;
      setStatus('running'); setMessage(calibrated ? '视线预测已恢复。' : '模型已就绪，请完成 16 点校准。');
    } catch (error) {
      if (run !== epoch.current) return;
      setStatus('error'); setMessage(error instanceof Error ? error.message : '视线模型加载失败');
      const failed = service.current; service.current = null;
      void failed?.dispose().catch(() => {});
      invalidate();
    }
  }
  function beginCalibration() {
    if (status !== 'running') return;
    invalidate(); setCalibrating(true); setMessage('注视目标中心，每点点击 5 次；无有效眼部图像时不计数。');
  }
  function sample(x: number, y: number) {
    const accepted = service.current?.calibrate(x, y) ?? false;
    if (!accepted) setMessage('未取得新的有效眼部图像，请保持头部稳定，稍候再点击。');
    else setMessage('样本已记录，请继续注视目标。');
    return accepted;
  }
  function finishCalibration() {
    service.current?.finishCalibration(); setCalibrated(true); setCalibrating(false);
    setValidating(true); setMessage('校准完成，正在使用独立目标验证误差。');
  }
  function finishValidation(result: ValidationResult) {
    setValidation(result); setValidating(false); setMessage(result.meanError === null
      ? '有效验证样本不足，请重新校准。' : '验证完成；落点仅作显示，不控制鼠标。');
  }
  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch { setMessage('无法切换全屏，可在当前网页区域校准。'); }
  }
  return { status, frame, calibrated, calibrating, validating, validation, message,
    showPoint, setShowPoint, toggleFullscreen,
    start, pause, beginCalibration, sample, finishCalibration, finishValidation,
    cancelCalibration: () => { invalidate(); setMessage('校准已取消，请重新校准。'); },
    cancelValidation: () => { setValidating(false); setMessage('已跳过误差验证，当前校准尚未验证。'); },
    setMessage };
}

export type GazePrediction = ReturnType<typeof useGazePrediction>;
