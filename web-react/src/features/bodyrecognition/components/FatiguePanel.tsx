import type { FatigueStatus } from '../types';

function number(value: number | null | undefined, digits: number, fallback = '—'): string {
  return value != null && Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

interface Props {
  status: FatigueStatus | null;
}

export default function FatiguePanel({ status }: Props) {
  return (
    <article className="panel fatigue-panel">
      <div className="panel-heading">
        <div><h2>疲劳驾驶</h2></div>
        <span className="model-tag">dlib 68 点</span>
      </div>
      <div className="blink-kpis">
        <div className="kpi">
          <span>张嘴 MAR</span>
          <strong><b>{number(status?.mar, 3)}</b></strong>
        </div>
        <div className="kpi">
          <span>打哈欠</span>
          <strong><b>{status?.is_yawning ? '是' : '否'}</b></strong>
        </div>
        <div className="kpi">
          <span>持续张嘴</span>
          <strong><b>{number(status?.yawn_duration, 1, '0.0')}</b><small>秒</small></strong>
        </div>
        <div className="kpi">
          <span>pitch</span>
          <strong><b>{number(status?.pitch, 1)}</b><small>°</small></strong>
        </div>
        <div className="kpi">
          <span>yaw</span>
          <strong><b>{number(status?.yaw, 1)}</b><small>°</small></strong>
        </div>
        <div className="kpi">
          <span>roll</span>
          <strong><b>{number(status?.roll, 1)}</b><small>°</small></strong>
        </div>
      </div>
      <div className={`notice-card${status?.is_drowsy ? ' drowsy' : ''}`}>
        <span className="notice-mark">{status?.is_drowsy ? '!' : 'i'}</span>
        <p>
          <strong>{status?.is_drowsy ? '疲劳驾驶预警' : '状态正常'}</strong>
          {status?.is_drowsy
            ? '连续张嘴超过阈值，疑似疲劳驾驶，请注意休息。'
            : '未检测到持续打哈欠行为。'}
        </p>
      </div>
    </article>
  );
}