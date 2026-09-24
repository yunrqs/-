import { useEffect, useRef, useState } from 'react';
import type { GazeFrame, ValidationResult } from '../types';

const calibrationTargets = [.12, .37, .63, .88].flatMap(y => [.1, .37, .63, .9].map(x => [x, y]));
const validationTargets = [[.5,.5], [.25,.25], [.75,.25], [.25,.75], [.75,.75]];

export function Calibration({ sample, finish, cancel, message }: {
  sample: (x: number, y: number) => boolean; finish: () => void; cancel: () => void; message: string;
}) {
  const [count, setCount] = useState(0);
  const target = calibrationTargets[Math.floor(count / 5)];
  const moved = useRef(performance.now());
  return <div className="gaze-calibration" role="dialog" aria-modal="true" aria-label="视线校准">
    <div className="gaze-instructions"><strong>16 点校准 · 第 {Math.floor(count / 5) + 1} 点 · {count % 5}/5</strong>
      <p>保持头部稳定，注视圆点中心后点击。{message}</p><button onClick={cancel}>取消校准</button></div>
    <button className="gaze-target" aria-label="记录校准点" style={{ left: `${target[0]*100}%`, top: `${target[1]*100}%` }}
      onClick={event => {
        if (performance.now() - moved.current < 500) return;
        const rect = event.currentTarget.getBoundingClientRect();
        if (!sample(rect.left + rect.width / 2, rect.top + rect.height / 2)) return;
        if (count === 79) { finish(); return; }
        if ((count + 1) % 5 === 0) moved.current = performance.now();
        setCount(count + 1);
      }}>{count % 5 + 1}</button>
  </div>;
}

export function Validation({ frame, finish, cancel }: {
  frame: GazeFrame; finish: (result: ValidationResult) => void; cancel: () => void;
}) {
  const [index, setIndex] = useState(0);
  const started = useRef(performance.now());
  const targetRef = useRef<HTMLDivElement>(null);
  const stats = useRef({ total: 0, errors: [] as number[], perTarget: [0,0,0,0,0] });
  const finishRef = useRef(finish); finishRef.current = finish;
  useEffect(() => {
    started.current = performance.now();
    const timer = setTimeout(() => {
      if (index < validationTargets.length - 1) setIndex(index + 1);
      else {
        const { total, errors, perTarget } = stats.current;
        finishRef.current({ meanError: perTarget.every(n => n >= 3)
          ? errors.reduce((a, b) => a + b, 0) / errors.length : null,
          validRatio: total ? errors.length / total : 0 });
      }
    }, 3000);
    return () => clearTimeout(timer);
  }, [index]);
  useEffect(() => {
    if (performance.now() - started.current < 800) return;
    stats.current.total++;
    const point = frame.point, rect = targetRef.current?.getBoundingClientRect();
    if (point && rect) {
      stats.current.errors.push(Math.hypot(point.x - rect.left - rect.width / 2, point.y - rect.top - rect.height / 2));
      stats.current.perTarget[index]++;
    }
  }, [frame, index]);
  const target = validationTargets[index];
  return <div className="gaze-calibration gaze-validation" role="dialog" aria-modal="true" aria-label="视线验证">
    <div className="gaze-instructions"><strong>误差验证 · {index + 1}/5</strong>
      <p>只需注视圆点，每点 3 秒，无需点击。此阶段不训练模型。</p><button onClick={cancel}>跳过验证</button></div>
    <div ref={targetRef} className="gaze-target validation-target" style={{ left: `${target[0]*100}%`, top: `${target[1]*100}%` }}>+</div>
  </div>;
}
