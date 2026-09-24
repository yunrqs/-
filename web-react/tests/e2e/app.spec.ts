import { expect, test } from './fixtures';

// 浏览器测试返回固定识别结果，单独检查页面操作，不依赖模型速度。
const result = {
  frame_width: 640, frame_height: 480, face_detected: true,
  face_box: [120, 80, 160, 200], ear: 0.3, threshold: 0.2, eye_state: 'OPEN',
  blink_detected: false, blink_count: 3, blink_rate_per_min: 12,
  valid_face_ratio: 1, emotion: 'happy', emotion_confidence: 0.9,
  emotion_probabilities: { happy: 0.9, neutral: 0.1 },
};

test('摄像头、识别、暂停、截图和重置', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('**/api/**', async (route) => {
    if (route.request().url().endsWith('/analyze')) {
      expect(route.request().headers()['content-type']).toContain('multipart/form-data');
      await route.fulfill({ json: result });
    } else await route.fulfill({ json: { ok: true, ready: true } });
  });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '情绪与眨眼监测' })).toBeVisible();
  await expect(page.getByRole('button', { name: '开始识别' })).toBeDisabled();
  await page.getByRole('button', { name: '开启摄像头' }).click();
  await expect(page.getByRole('button', { name: '开始识别' })).toBeEnabled();
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: 'happy' })).toBeVisible();
  await expect(page.getByText('置信度 90.0%')).toBeVisible();
  await page.screenshot({ path: 'test-results/desktop-running.png', fullPage: true, animations: 'disabled' });
  await page.getByRole('button', { name: '暂停分析' }).click();
  await expect(page.getByRole('button', { name: '暂停分析' })).toBeDisabled();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: '保存截图' }).click();
  expect((await download).suggestedFilename()).toMatch(/^focuslens-.*\.jpg$/);
  await page.getByRole('button', { name: '重置统计' }).click();
  await expect(page.getByRole('heading', { name: '等待分析' })).toBeVisible();
  await page.getByRole('button', { name: '关闭摄像头' }).click();
  await expect(page.getByRole('button', { name: '保存截图' })).toBeDisabled();
  await expect(page.locator('.toast')).not.toHaveClass(/show/);
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true, animations: 'disabled' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: '开启摄像头' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true, animations: 'disabled' });
  expect(errors).toEqual([]);
});

test('模块独立收起、侧栏展开和实际视频比例适配', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.route('**/api/**', route => route.fulfill({ json: { ok: true, ready: true } }));
  await page.goto('/');
  const camera = page.getByRole('button', { name: '开启摄像头' });
  await camera.click();
  await expect(page.getByRole('button', { name: '关闭摄像头' })).toBeVisible();
  const checkRatio = async () => {
    const dimensions = await page.locator('#camera').evaluate((element: HTMLVideoElement) => ({
      actual: element.videoWidth / element.videoHeight,
      display: element.getBoundingClientRect().width / element.getBoundingClientRect().height,
    }));
    expect(Math.abs(dimensions.actual - dimensions.display)).toBeLessThan(.01);
  };
  await checkRatio();
  const cameraBox = await page.locator('.camera-panel').boundingBox();
  const emotionBox = await page.locator('.state-panel').boundingBox();
  expect(cameraBox!.y).toBe(emotionBox!.y);
  expect(cameraBox!.height).toBeCloseTo(emotionBox!.height, 0);
  expect(emotionBox!.x).toBeGreaterThan(cameraBox!.x);
  await page.evaluate(() => (window as typeof window & { resizeCamera: (width: number, height: number) => void }).resizeCamera(1280, 720));
  await expect(page.locator('.camera-footer')).toContainText('1280×720');
  await checkRatio();
  await page.getByRole('button', { name: '采集控制', exact: true }).click();
  await expect(page.getByRole('button', { name: '关闭摄像头' })).toBeHidden();
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeEnabled();
  await page.getByRole('button', { name: '采集控制', exact: true }).click();
  await page.getByRole('button', { name: '视线预测', exact: true }).click();
  await expect(page.getByRole('switch', { name: '视线预测' })).toBeHidden();
  await expect(page.getByRole('button', { name: '保存截图' })).toBeEnabled();
  await page.getByRole('button', { name: '视线预测', exact: true }).click();
  await page.getByRole('button', { name: '工具', exact: true }).click();
  await expect(page.getByRole('button', { name: '保存截图' })).toBeHidden();
  await page.getByRole('button', { name: '工具', exact: true }).click();
  await page.getByRole('button', { name: '收起侧边栏' }).click();
  await expect(page.getByRole('button', { name: '展开侧边栏' })).toBeVisible();
  await checkRatio();
  await page.screenshot({ path: 'test-results/sidebar-collapsed.png', fullPage: true });
  await page.getByRole('button', { name: '展开侧边栏' }).click();
  const sidebar = await page.locator('.sidebar').boundingBox();
  const tools = await page.locator('.tools-group').boundingBox();
  expect(sidebar!.y + sidebar!.height - tools!.y - tools!.height).toBeLessThan(30);
  await page.setViewportSize({ width: 1440, height: 640 });
  const shortTools = await page.locator('.tools-group').boundingBox();
  expect(shortTools!.y + shortTools!.height).toBeLessThanOrEqual(640);
  await expect(page.getByRole('button', { name: '保存截图' })).toBeInViewport();
  await page.setViewportSize({ width: 390, height: 844 });
  await checkRatio();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: '收起侧边栏' }).click();
  await checkRatio();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/mobile-collapsed.png', fullPage: true });
});

test('暂停后忽略迟到的结果，重置失败时保留已有统计', async ({ page }) => {
  let releaseFrame: () => void = () => {};
  const waitingFrame = new Promise<void>((resolve) => { releaseFrame = resolve; });
  let frameRequested: () => void = () => {};
  const frameStarted = new Promise<void>((resolve) => { frameRequested = resolve; });
  await page.route('**/api/**', async (route) => {
    const url = route.request().url();
    if (url.endsWith('/analyze')) {
      frameRequested();
      await waitingFrame;
      await route.fulfill({ json: result });
    } else if (url.endsWith('/reset')) {
      await route.fulfill({ status: 500, json: { detail: '后端重置失败' } });
    } else await route.fulfill({ json: { ok: true, ready: true } });
  });
  await page.goto('/');
  await page.getByRole('button', { name: '开启摄像头' }).click();
  await page.getByRole('button', { name: '开始识别' }).click();
  await frameStarted;
  await page.getByRole('button', { name: '暂停分析' }).click();
  const response = page.waitForResponse('**/api/analyze');
  releaseFrame();
  await response;
  await expect(page.getByRole('heading', { name: '等待分析' })).toBeVisible();
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: 'happy' })).toBeVisible();
  await page.getByRole('button', { name: '重置统计' }).click();
  await expect(page.locator('.toast')).toContainText('重置失败');
  await expect(page.getByRole('heading', { name: 'happy' })).toBeVisible();
});
