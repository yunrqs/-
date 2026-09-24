import { useEffect, useRef } from 'react';
import type { ChartPoint } from '../types';

interface Props { history: ChartPoint[] }

export default function EarChart({ history }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function draw() {
      if (!canvas) return;
      const context = canvas.getContext('2d');
      if (!context) return;
      const { width, height } = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.scale(ratio, ratio);
      context.strokeStyle = 'rgba(151,175,204,.09)';
      context.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        context.beginPath();
        context.moveTo(0, i * height / 4);
        context.lineTo(width, i * height / 4);
        context.stroke();
      }

      function drawLine(field: 'ear' | 'threshold', color: string, lineWidth: number) {
        if (!context) return;
        context.beginPath();
        context.strokeStyle = color;
        context.lineWidth = lineWidth;
        let connected = false;
        history.forEach((point, index) => {
          const value = point[field];
          // 丢失人脸时断开曲线，避免画出并不存在的测量值。
          if (value === null) { connected = false; return; }
          const x = index * width / Math.max(1, history.length - 1);
          const y = height - Math.max(0, Math.min(0.5, value)) / 0.5 * height;
          if (connected) context.lineTo(x, y);
          else context.moveTo(x, y);
          connected = true;
        });
        context.stroke();
      }

      drawLine('threshold', '#ffbd68', 1);
      drawLine('ear', '#42e8d1', 2);
    }

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [history]);

  return <canvas ref={canvasRef} id="earChart" role="img" aria-label="最近 80 帧 EAR 与阈值时间序列图" />;
}
