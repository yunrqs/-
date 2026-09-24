// Adapted from WebGazer weighted ridge regression (see LICENSE.md).
import util from './util.mjs';
import regression from './util_regression.mjs';

export class GazeRegression {
  constructor() { this.reset(); }
  reset() { this.samples = []; this.coefficients = null; this.filtered = null; }
  features(eyes) { return util.getEyeFeats(eyes); }
  add(features, x, y) {
    if (!features.length || !features.every(Number.isFinite)) throw new Error('眼部特征无效');
    this.samples.push({ features: [...features], x, y });
    this.samples = this.samples.slice(-500);
    this.coefficients = null;
  }
  predict(features) {
    if (!this.samples.length) return null;
    if (!this.coefficients) {
      const n = this.samples.length;
      const weights = this.samples.map((_, i) => Math.sqrt(1 / (n - i)));
      const X = this.samples.map((s, i) => s.features.map(v => v * weights[i]));
      this.coefficients = ['x', 'y'].map(axis => regression.ridge(
        this.samples.map((s, i) => [s[axis] * weights[i]]), X, 1e-6));
    }
    const point = this.coefficients.map(c => c.reduce((sum, v, i) => sum + v * features[i], 0));
    if (!point.every(Number.isFinite)) return null;
    // One smoothing path for both published coordinates and visible marker.
    this.filtered = this.filtered ? point.map((v, i) => .45 * v + .55 * this.filtered[i]) : point;
    return { x: this.filtered[0], y: this.filtered[1] };
  }
  clearFilter() { this.filtered = null; }
}
