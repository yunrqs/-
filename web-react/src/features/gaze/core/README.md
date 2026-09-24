# Source and adaptations

Source: the supplied `视线预测/src` WebGazer 3.5.3 project, Copyright
2016–2026 Brown WebGazer Team. See LICENSE.md for the supplied license notice.
Embedded notices in util.mjs and mat.mjs are retained.

- util.mjs: eye resizing, grayscale and histogram normalization;
  old debug UI and viewport clipping removed.
- util_regression.mjs / mat.mjs: ridge solver and matrix operations. Solver
  retries are bounded and non-finite solutions rejected.
- regression.mjs: weighted ridge adapted from ridgeWeightedReg.mjs. Coefficients
  are cached until samples change. Uses one exponential smoothing path (0.45)
  instead of the original Kalman plus display-only smoothing/amplification.
- ../services/gazeService.ts: MediaPipe FaceMesh directly uses the supplied local runtime;
  eye contours follow facemesh.mjs, with clamped bounds and invalid crop checks.

No original index.mjs, global mouse listeners, click/move auto-training,
localforage persistence, dwell selection, game logic or generated WebGazer
bundle is imported. Calibration explicitly adds samples. MediaPipe resources
are copied from the supplied www/mediapipe/face_mesh directory with its metadata.
