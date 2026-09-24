import type { AnalysisResult, HealthResponse } from './types';

// 所有 HTTP 请求集中在这里；组件只关心结果，不必重复处理状态码。
async function request(path: string, options?: RequestInit): Promise<Response> {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
      else if (typeof body.error === 'string') message = body.error;
    } catch {
      // 代理错误可能返回 HTML，此时保留上面的 HTTP 状态提示。
    }
    throw new Error(message);
  }
  return response;
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await request('/api/health');
  return response.json();
}

export async function startSession(): Promise<void> {
  await request('/api/session/start', { method: 'POST' });
}

export async function resetSession(): Promise<void> {
  await request('/api/session/reset', { method: 'POST' });
}

export async function analyzeFrame(frame: Blob): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('frame', frame, 'frame.jpg');
  // FormData 的 Content-Type 由浏览器设置，不能手动省略 boundary。
  const response = await request('/api/analyze', { method: 'POST', body: form });
  return response.json();
}
