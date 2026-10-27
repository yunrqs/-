import type { FatigueStatus } from '../types';

function number(value: number | null | undefined, digits: number, fallback = '—'): string {
  return value != null && Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

interface Props {
  status: FatigueStatus | null;
}

export default function FatiguePanel({ status }: Props) {
  const pose = status?.pose ?? null;

  return (
    <article className="panel fatigue-panel">
      <div className="panel-heading">
        <div><h2>疲劳驾驶</h2></div>
        <span className="model-tag">dlib 68 点 + OpenPose</span>
      </div>

      {/* 打哈欠与头部姿态：dlib 68 点。 */}
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

      {/* 上半身肢体动作：OpenPose。 */}
      <div className="panel-heading pose-heading">
        <div><h2>上半身肢体动作</h2></div>
        <span className="model-tag">OpenPose</span>
      </div>
      {pose && pose.person_detected ? (
        <div className="blink-kpis">
          <div className="kpi">
            <span>左肘弯曲角</span>
            <strong><b>{number(pose.left_elbow_angle, 0)}</b><small>°</small></strong>
          </div>
          <div className="kpi">
            <span>右肘弯曲角</span>
            <strong><b>{number(pose.right_elbow_angle, 0)}</b><small>°</small></strong>
          </div>
          <div className="kpi">
            <span>左腕相对肩高</span>
            <strong><b>{number(pose.left_wrist_raise, 2)}</b></strong>
          </div>
          <div className="kpi">
            <span>右腕相对肩高</span>
            <strong><b>{number(pose.right_wrist_raise, 2)}</b></strong>
          </div>
          <div className="kpi">
            <span>身体侧倾</span>
            <strong><b>{number(pose.lean_angle, 1)}</b><small>°</small></strong>
          </div>
          <div className="kpi">
            <span>肩部倾斜</span>
            <strong><b>{number(pose.shoulder_tilt, 1)}</b><small>°</small></strong>
          </div>
        </div>
      ) : (
        <p className="pose-empty">未检测到上半身身体，请调整坐姿让肩部与手臂入镜。</p>
      )}
    </article>
  );
}