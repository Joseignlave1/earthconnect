import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findAreas, scoreDistribution, thresholdCode } from '../src/picks.ts';
import { scorePixels } from '../src/scores.ts';
const patch = (siteKey, width, values) => ({
  grid: { siteKey, width, height: values.length / width },
  scores: Uint16Array.from(values),
});
test('histogram accounts for missing, physical zero, and the 100% endpoint', () => {
  const d = scoreDistribution([patch('a', 2, [0, 65535, 65534, 32767])]);
  assert.equal(d.valid, 3);
  assert.equal(d.missing, 1);
  assert.equal(d.zero, 1);
  assert.equal(d.bins[0], 1);
  assert.equal(d.bins[25], 1);
  assert.equal(d.bins[49], 1);
});
test('threshold is inclusive in packed score space and hides low scores without mutation', () => {
  const values = new Uint16Array([0, 32767, 65534, 65535]);
  const p = scorePixels(values, thresholdCode(50));
  assert.equal(p[3], 0);
  assert.ok(p[7] > 0);
  assert.ok(p[11] > p[7]);
  assert.equal(p[15], 0);
  assert.deepEqual([...values], [0, 32767, 65534, 65535]);
  assert.throws(() => thresholdCode(NaN));
});
test('connected areas use four neighbors without wrapping rows or joining patches', () => {
  const r = findAreas([patch('a', 3, [0, 0, 60000, 60000, 0, 0]), patch('b', 1, [60000])], 50000);
  assert.equal(r.count, 3);
  assert.equal(r.areaCount, 3);
  assert.deepEqual(
    r.top.map((a) => a.count),
    [1, 1, 1],
  );
});
test('connected pixels collapse to one pick with correct bounds, peak and mean', () => {
  const r = findAreas([patch('a', 3, [60000, 61000, 0, 0, 62000, 0])], 60000);
  assert.equal(r.areaCount, 1);
  assert.deepEqual(r.top[0], {
    siteKey: 'a',
    count: 3,
    peak: 62000,
    peakIndex: 4,
    minX: 0,
    maxX: 1,
    minY: 0,
    maxY: 1,
    mean: 61000,
  });
});
test('top areas are stable, bounded, and ignore all-zero and unknown results', () => {
  const r = findAreas([patch('z', 3, [65000, 0, 65000]), patch('a', 2, [65534, 65535])], 0, 2);
  assert.equal(r.areaCount, 3);
  assert.equal(r.top.length, 2);
  assert.equal(r.top[0].siteKey, 'a');
  assert.equal(r.top[1].peakIndex, 0);
  assert.equal(findAreas([patch('a', 2, [0, 65535])], 0).areaCount, 0);
});
test('pilot-size distribution and area search finish within an interactive budget', () => {
  const scores = new Uint16Array(270000);
  for (let i = 0; i < scores.length; i++) scores[i] = i % 3 ? 65534 : 0;
  const p = { grid: { siteKey: 'stress', width: 1000, height: 270 }, scores };
  const start = performance.now();
  const result = findAreas([p], 50000);
  assert.equal(result.count, 180000);
  assert.ok(performance.now() - start < 1000);
});
