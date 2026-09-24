import { test as base, expect } from '@playwright/test';

// Use a deterministic MediaStream instead of the host's intermittently stalled
// fake camera driver. Video playback, canvas capture and track cleanup stay real.
export const test = base.extend({
  page: async ({ page }, use) => {
    await page.addInitScript(() => {
      const state = window as typeof window & { cameraRequests: number; resizeCamera: (width: number, height: number) => void };
      state.cameraRequests = 0;
      navigator.mediaDevices.getUserMedia = async () => {
        state.cameraRequests++;
        const canvas = document.createElement('canvas');
        canvas.width = 640; canvas.height = 480;
        state.resizeCamera = (width, height) => { canvas.width = width; canvas.height = height; };
        const context = canvas.getContext('2d')!;
        const draw = () => {
          const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
          gradient.addColorStop(0, '#293852'); gradient.addColorStop(1, '#adc5e6');
          context.fillStyle = gradient; context.fillRect(0, 0, canvas.width, canvas.height);
        };
        draw();
        const stream = canvas.captureStream(20);
        const timer = setInterval(draw, 50);
        const track = stream.getVideoTracks()[0], stop = track.stop.bind(track);
        track.stop = () => { clearInterval(timer); stop(); };
        return stream;
      };
    });
    await use(page);
  },
});
export { expect };
