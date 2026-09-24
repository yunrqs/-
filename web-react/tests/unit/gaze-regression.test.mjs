import test from 'node:test';
import assert from 'node:assert/strict';
import { GazeRegression } from '../../src/features/gaze/core/regression.mjs';

test('weighted ridge learns known coordinates and invalidates its cached fit', () => {
  const model = new GazeRegression();
  assert.equal(model.predict([1, 1]), null);
  for (let x = 1; x <= 10; x++) model.add([x, 1], 10 * x + 5, 20 * x + 3);
  let result = model.predict([4, 1]);
  assert.ok(Math.abs(result.x - 45) < .01);
  assert.ok(Math.abs(result.y - 83) < .01);
  model.reset();
  assert.equal(model.predict([4, 1]), null);
  for (let x = 1; x <= 10; x++) model.add([x, 1], 30 * x, 40 * x);
  result = model.predict([4, 1]);
  assert.ok(Math.abs(result.x - 120) < .01);
  assert.ok(Math.abs(result.y - 160) < .01);
  assert.throws(() => model.add([NaN, 1], 0, 0), /无效/);
});
