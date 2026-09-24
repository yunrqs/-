import { expect, test } from './fixtures';

test('本地 FaceMesh 运行文件可加载，空白摄像头不会产生虚假落点', async ({ page }) => {
  test.setTimeout(60000);
  const remoteRequests: string[] = [];
  page.on('request', request => {
    if (!request.url().startsWith('http://127.0.0.1:5173') && !request.url().startsWith('data:')) remoteRequests.push(request.url());
  });
  await page.route('**/api/**', route => route.fulfill({ status: 503, json: { detail: 'offline' } }));
  await page.goto('/');
  await page.getByRole('button', { name: '开启摄像头' }).click();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.getByRole('button', { name: '开始视线校准' })).toBeEnabled({ timeout: 45000 });
  await expect.poll(async () => Number(await page.locator('.gaze-kpis > div').nth(3).locator('strong').textContent())).toBeGreaterThan(0);
  await expect(page.locator('.gaze-panel')).toContainText('跟踪丢失');
  await expect(page.locator('.gaze-dot')).toHaveCount(0);
  await page.getByRole('button', { name: '关闭摄像头' }).click();
  await expect(page.locator('.gaze-panel')).toContainText('未运行');
  expect(remoteRequests).toEqual([]);
});

test('独立校准、验证、暂停、重启及视口失效，且只申请一次摄像头', async ({ page }) => {
  test.setTimeout(90000);
  await page.addInitScript(() => {
    const state = window as typeof window & { cameraRequests: number; gazeFace: boolean };
    state.gazeFace = true;
    // Only replace the detector. Real extraction, regression, calibration and UI still run.
    window.FaceMesh = class {
      callback = (_: { multiFaceLandmarks: { x: number; y: number; z: number }[][] }) => {};
      setOptions() {}
      onResults(callback: typeof this.callback) { this.callback = callback; }
      async initialize() {}
      async close() {}
      async send() {
        this.callback({ multiFaceLandmarks: state.gazeFace ? [Array.from({ length: 468 }, (_, i) => ({
          x: .35 + (i % 10) * .01, y: .35 + (i % 7) * .01, z: 0,
        }))] : [] });
      }
    };
  });
  await page.route('**/api/**', route => route.fulfill({ status: 503, json: { detail: 'offline' } }));
  await page.goto('/');
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeDisabled();
  await page.getByRole('button', { name: '开启摄像头' }).click();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await page.getByRole('button', { name: '开始视线校准' }).click();
  await page.getByRole('button', { name: '取消校准' }).click();
  await expect(page.locator('.gaze-dot')).toHaveCount(0);
  await page.getByRole('button', { name: '开始视线校准' }).click();
  const target = page.getByRole('button', { name: '记录校准点' });
  for (let i = 0; i < 16; i++) {
    await page.waitForTimeout(550); // Target settling is intentional in the calibration contract.
    for (let j = 0; j < 5; j++) { await target.click(); await page.waitForTimeout(160); }
  }
  await expect(page.getByRole('dialog', { name: '视线验证' })).toBeVisible();
  await expect(page.getByRole('dialog', { name: '视线验证' })).toHaveCount(0, { timeout: 20000 });
  await expect(page.locator('.gaze-panel')).toContainText('验证完成');
  await expect(page.locator('.gaze-panel')).toContainText('正在预测');
  await page.getByRole('button', { name: '视线预测', exact: true }).click();
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeHidden();
  await expect(page.locator('.gaze-panel')).toContainText('正在预测');
  await page.getByRole('button', { name: '视线预测', exact: true }).click();
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeChecked();
  expect(await page.evaluate(() => (window as typeof window & { cameraRequests: number }).cameraRequests)).toBe(1);
  await expect(page.locator('.gaze-dot')).toHaveCSS('pointer-events', 'none');
  await page.evaluate(() => { (window as typeof window & { gazeFace: boolean }).gazeFace = false; });
  await expect(page.locator('.gaze-panel')).toContainText('跟踪丢失');
  await expect(page.locator('.gaze-dot')).toHaveCount(0);
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.getByRole('button', { name: '保存截图' })).toBeEnabled();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.getByRole('button', { name: '重新校准视线' })).toBeEnabled();
  await page.setViewportSize({ width: 1000, height: 700 });
  await expect(page.locator('.gaze-panel')).toContainText('显示区域已改变');
  await expect(page.getByRole('button', { name: '开始视线校准' })).toBeEnabled();
  await page.getByRole('button', { name: '关闭摄像头' }).click();
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeDisabled();
});

test('模型加载失败可重试且不影响其他分析', async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ json: { ok: true, ready: true } }));
  await page.route('**/mediapipe/face_mesh/face_mesh.js', route => route.abort());
  await page.goto('/');
  await page.getByRole('button', { name: '开启摄像头' }).click();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.locator('.gaze-panel')).toContainText('视线模型加载失败');
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeEnabled();
  await expect(page.getByRole('button', { name: '开始识别', exact: true })).toBeEnabled();
  await expect(page.locator('.gaze-dot')).toHaveCount(0);
});

test('加载中关闭摄像头后，不接受迟到的初始化和预测结果', async ({ page }) => {
  await page.addInitScript(() => {
    window.FaceMesh = class {
      setOptions() {}
      onResults() {}
      async initialize() { await new Promise(resolve => setTimeout(resolve, 1500)); }
      async close() {}
      async send() {}
    };
  });
  await page.route('**/api/**', route => route.fulfill({ json: { ok: true, ready: true } }));
  await page.goto('/');
  await page.getByRole('button', { name: '开启摄像头' }).click();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.getByRole('switch', { name: '视线预测' })).toContainText('加载中');
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.getByRole('switch', { name: '视线预测' })).not.toBeChecked();
  await page.waitForTimeout(1800);
  await expect(page.locator('.gaze-panel')).toContainText('未运行');
  await page.getByRole('switch', { name: '视线预测' }).click();
  await expect(page.getByRole('button', { name: '开始视线校准' })).toBeEnabled();
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeChecked();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await page.getByRole('switch', { name: '视线预测' }).click();
  await page.getByRole('button', { name: '关闭摄像头' }).click();
  await page.waitForTimeout(1800);
  await expect(page.locator('.gaze-panel')).toContainText('未运行');
  await expect(page.getByRole('button', { name: '开始视线校准' })).toBeDisabled();
  await expect(page.locator('.gaze-dot')).toHaveCount(0);
});
