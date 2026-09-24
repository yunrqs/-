import { defineConfig } from '@playwright/test';

// 本机测试请求不经过系统代理，否则某些 Windows 代理会返回 502。
process.env.NO_PROXY = [process.env.NO_PROXY, '127.0.0.1', 'localhost'].filter(Boolean).join(',');

export default defineConfig({
  testDir: './tests/e2e',
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:5173',
    // Windows 自带 Edge；测试使用虚拟摄像头，不会打开真实摄像头。
    channel: 'msedge',
    launchOptions: {
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    },
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: false,
  },
});
