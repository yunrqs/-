import { useEffect, useRef, useState } from 'react';
import * as api from '../../features/monitoring/api';
import { captureVideo, playCamera, toJpeg } from '../../features/monitoring/camera';
import Sidebar from '../../features/monitoring/components/Sidebar';
import ResultPanels from '../../features/monitoring/components/ResultPanels';
import Icon from '../../components/Icon';
import type { AnalysisResult, ChartPoint } from '../../features/monitoring/types';
import GazePanel from '../../features/gaze/components/GazePanel';
import { useGazePrediction } from '../../features/gaze/hooks/useGazePrediction';

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '操作失败，请重试';
}

export default function App() {
  // useState 保存需要显示到页面的数据；useRef 保存视频、媒体流和请求等对象。
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [service, setService] = useState({ online: false, text: '服务检查中' });
  const [clock, setClock] = useState(new Date().toLocaleTimeString('zh-CN', { hour12: false }));
  const [resolution, setResolution] = useState('—');
  const [videoAspectRatio, setVideoAspectRatio] = useState(16 / 9);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [fps, setFps] = useState('0.0');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<ChartPoint[]>([]);
  const [toast, setToast] = useState({ message: '', error: false });

  const videoRef = useRef<HTMLVideoElement>(null);
  const gaze = useGazePrediction(videoRef, cameraOn);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mountedRef = useRef(true);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const pendingFrameRef = useRef<Promise<void> | null>(null);
  // 每次停止时更新编号，旧请求返回后发现编号不同，就不再更新页面。
  const runIdRef = useRef(0);

  function notify(message: string, error = false) {
    if (!mountedRef.current) return;
    clearTimeout(toastTimerRef.current);
    setToast({ message, error });
    toastTimerRef.current = setTimeout(() => setToast({ message: '', error: false }), 3200);
  }

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;
    async function refreshHealth() {
      try {
        const health = await api.checkHealth();
        if (!cancelled) setService({ online: true, text: health.ready ? '识别服务就绪' : '服务在线 · 模型待加载' });
      } catch {
        if (!cancelled) setService({ online: false, text: '识别服务离线' });
      }
    }
    void refreshHealth();
    const healthTimer = setInterval(refreshHealth, 10000);
    const clockTimer = setInterval(() => setClock(new Date().toLocaleTimeString('zh-CN', { hour12: false })), 1000);
    return () => {
      cancelled = true;
      mountedRef.current = false;
      runIdRef.current += 1;
      clearInterval(healthTimer);
      clearInterval(clockTimer);
      clearTimeout(toastTimerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  function clearOverlay() {
    const canvas = overlayRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  }

  function pauseAnalysis() {
    runIdRef.current += 1;
    setAnalyzing(false);
    setFps('0.0');
    clearOverlay();
  }

  async function toggleCamera() {
    if (streamRef.current) {
      pauseAnalysis();
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraOn(false);
      setResolution('—');
      return;
    }
    setCameraLoading(true);
    try {
      if (!navigator.mediaDevices) throw new Error('请通过 localhost、127.0.0.1 或 HTTPS 打开页面');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
      });
      const video = videoRef.current;
      if (!mountedRef.current || !video) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      video.srcObject = stream;
      await playCamera(video);
      if (!mountedRef.current) return;
      setResolution(`${video.videoWidth}×${video.videoHeight}`);
      setCameraOn(true);
      notify('摄像头已开启，画面只在本机处理。');
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      notify(`无法开启摄像头：${errorMessage(error)}`, true);
    } finally {
      if (mountedRef.current) setCameraLoading(false);
    }
  }

  async function startAnalysis() {
    if (!streamRef.current || analyzing || modelLoading || resetting) return;
    const runId = runIdRef.current;
    setModelLoading(true);
    try {
      // 先等暂停前的请求结束，确保服务器不会同时处理两帧。
      await pendingFrameRef.current;
      await api.startSession();
      if (!mountedRef.current || runId !== runIdRef.current || !streamRef.current) return;
      setService({ online: true, text: '识别服务就绪' });
      setAnalyzing(true);
      notify('情绪识别与眨眼检测已启动。');
    } catch (error) {
      if (runId === runIdRef.current) notify(errorMessage(error), true);
    } finally {
      if (mountedRef.current) setModelLoading(false);
    }
  }

  useEffect(() => {
    if (!analyzing) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const runId = runIdRef.current;
    const frameTimes: number[] = [];
    let lastCompleted = performance.now();

    async function sendFrame() {
      const video = videoRef.current;
      if (!video || cancelled || runId !== runIdRef.current) return;
      const started = performance.now();
      try {
        const blob = await toJpeg(captureVideo(video, 640));
        if (cancelled || runId !== runIdRef.current) return;
        const nextResult = await api.analyzeFrame(blob);
        if (cancelled || runId !== runIdRef.current) return;
        setResult(nextResult);
        setHistory((previous) => [...previous, { ear: nextResult.ear, threshold: nextResult.threshold }].slice(-80));
        const completed = performance.now();
        frameTimes.push(completed - lastCompleted);
        lastCompleted = completed;
        if (frameTimes.length > 20) frameTimes.shift();
        const average = frameTimes.reduce((sum, value) => sum + value, 0) / frameTimes.length;
        setFps((1000 / average).toFixed(1));
        // 请求完成后再安排下一帧，慢速模型不会造成请求堆积。
        timer = setTimeout(tick, Math.max(0, 220 - (completed - started)));
      } catch (error) {
        if (!cancelled && runId === runIdRef.current) {
          pauseAnalysis();
          notify(errorMessage(error), true);
        }
      }
    }

    function tick() { pendingFrameRef.current = sendFrame(); }
    tick();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [analyzing]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!analyzing || !cameraOn || !result?.face_box) return;
    canvas.width = result.frame_width;
    canvas.height = result.frame_height;
    const [x, y, width, height] = result.face_box;
    const color = result.blink_detected ? '#ffbd68' : '#42e8d1';
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.setLineDash([12, 7]);
    context.strokeRect(x, y, width, height);
    context.setLineDash([]);
    context.fillStyle = 'rgba(7,16,29,.82)';
    context.fillRect(x, Math.max(0, y - 22), 105, 19);
    context.fillStyle = color;
    context.font = '10px ui-monospace, monospace';
    context.fillText(result.blink_detected ? 'BLINK DETECTED' : 'FACE TRACKED', x + 7, Math.max(13, y - 9));
  }, [result, analyzing, cameraOn]);

  async function resetStats() {
    if (resetting || modelLoading) return;
    setResetting(true);
    pauseAnalysis();
    try {
      // 旧帧完成后再重置后端，避免旧帧把计数重新加回来。
      await pendingFrameRef.current;
      await api.resetSession();
      if (!mountedRef.current) return;
      setResult(null);
      setHistory([]);
      notify('统计已重置，点击“开始识别”继续。');
    } catch (error) {
      notify(`重置失败：${errorMessage(error)}`, true);
    } finally {
      if (mountedRef.current) setResetting(false);
    }
  }

  function saveCapture() {
    if (!streamRef.current || !videoRef.current) return;
    try {
      const video = videoRef.current;
      const canvas = captureVideo(video, video.videoWidth);
      const link = document.createElement('a');
      link.download = `focuslens-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
      link.href = canvas.toDataURL('image/jpeg', 0.92);
      link.click();
      notify('当前画面已保存。');
    } catch (error) { notify(errorMessage(error), true); }
  }

  function updateVideoSize() {
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight) {
      setVideoAspectRatio(video.videoWidth / video.videoHeight);
      setResolution(`${video.videoWidth}×${video.videoHeight}`);
    }
  }

  return <div className={`app-shell${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
    <Sidebar cameraOn={cameraOn} cameraLoading={cameraLoading} modelLoading={modelLoading}
      gaze={gaze} collapsed={sidebarCollapsed} onToggleCollapsed={() => setSidebarCollapsed(value => !value)}
      serviceOnline={service.online} serviceText={service.text}
      analyzing={analyzing} resetting={resetting} onCamera={toggleCamera} onStart={startAnalysis}
      onPause={pauseAnalysis} onReset={resetStats} onCapture={saveCapture} />
    <main className="workspace">
      <header className="topbar">
        <div><p className="eyebrow">FOCUSLENS / 视觉状态实验台</p><h1>情绪与眨眼监测</h1></div>
        <div className="top-status">
          <span className={`status-pill ${service.online ? 'online' : 'offline'}`}><i />{service.text}</span>
          <time id="clock">{clock}</time>
        </div>
      </header>
      <section className="dashboard">
        <article className="panel camera-panel">
          <div className="panel-heading">
            <h2>实时画面</h2>
            <div className={`live-badge${analyzing ? ' live' : ''}`}><i /><span>{analyzing ? '实时分析' : cameraOn ? '已暂停' : '待机'}</span></div>
          </div>
          <div className="video-stage" style={{ aspectRatio: videoAspectRatio }}>
            <video ref={videoRef} id="camera" playsInline muted onLoadedMetadata={updateVideoSize} onResize={updateVideoSize} />
            <canvas ref={overlayRef} id="overlay" aria-hidden="true" />
            <div className={`video-placeholder${cameraOn ? ' hidden' : ''}`}>
              <div className="scan-symbol"><Icon name="camera" /></div><strong>摄像头尚未开启</strong>
              <p>点击左侧「开启摄像头」开始采集</p>
            </div>
            <div className="frame-corners" aria-hidden="true"><i /><i /><i /><i /></div>
          </div>
          <div className="camera-footer">
            <span>分辨率 <b>{resolution}</b></span><span>分析 FPS <b>{fps}</b></span>
            <span>人脸状态 <b style={{ color: analyzing && result?.face_detected ? 'var(--cyan)' : undefined }}>{!analyzing ? '未检测' : result?.face_detected ? '已锁定' : '未检测'}</b></span>
          </div>
        </article>
        <ResultPanels result={result} history={history} />
        <GazePanel gaze={gaze} />
      </section>
      <p className={`toast${toast.message ? ' show' : ''}${toast.error ? ' error' : ''}`} role="status">{toast.message}</p>
    </main>
  </div>;
}
