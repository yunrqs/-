import { createPortal } from 'react-dom';
import type { GazePrediction } from '../hooks/useGazePrediction';
import { Calibration, Validation } from './Calibration';
import GazeOverlay from './GazeOverlay';

export default function GazePanel({ gaze }: { gaze: GazePrediction }) {
  const running = gaze.status === 'running';
  const point = gaze.frame.point;
  return <article className="panel gaze-panel">
    <div className="panel-heading"><h2>视线预测结果</h2><span className="model-tag">本地处理</span></div>
    <p className="gaze-description">先校准，再预测；全屏或窗口尺寸改变后请重新校准。</p>
    <div className="gaze-kpis">
      <div><span>跟踪状态</span><strong>{!running ? (gaze.status === 'error' ? '运行失败' : gaze.status === 'loading' ? '模型加载中' : '未运行') : !gaze.frame.faceDetected ? '跟踪丢失' : gaze.calibrated ? '正在预测' : '等待校准'}</strong></div>
      <div><span>落点坐标（CSS 像素）</span><strong>{point ? `${point.x.toFixed(0)}, ${point.y.toFixed(0)}${point.inViewport ? '' : ' · 区域外'}` : '—'}</strong></div>
      <div><span>归一化坐标</span><strong>{point ? `${point.normalizedX.toFixed(3)}, ${point.normalizedY.toFixed(3)}` : '—'}</strong></div>
      <div><span>预测 FPS</span><strong>{gaze.frame.fps.toFixed(1)}</strong></div>
      <div><span>验证平均误差</span><strong>{gaze.validation?.meanError != null ? `${gaze.validation.meanError.toFixed(0)} px` : '未验证 / 样本不足'}</strong></div>
      <div><span>验证有效采样率</span><strong>{gaze.validation ? `${(gaze.validation.validRatio * 100).toFixed(0)}%` : '—'}</strong></div>
    </div>
    <p className="gaze-message" role="status">{gaze.message}</p>
    {createPortal(<>
      {gaze.showPoint && running && !gaze.calibrating && !gaze.validating && <GazeOverlay point={point} />}
      {gaze.calibrating && <Calibration sample={gaze.sample} finish={gaze.finishCalibration} cancel={gaze.cancelCalibration} message={gaze.message} />}
      {gaze.validating && <Validation frame={gaze.frame} finish={gaze.finishValidation} cancel={gaze.cancelValidation} />}
    </>, document.body)}
  </article>;
}
