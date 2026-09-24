import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 浏览器只请求 /api，由 Vite 转发到本机 FastAPI，避免开发时的跨域问题。
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8001' },
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8001' },
  },
});
