import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  decodeScores,
  scorePixels,
  scoreSummary,
  prepareScores,
  gridExtent,
} from '../src/scores.ts';
const grid = { siteKey: 'patch', width: 2, height: 2, transform: [1000, 0, -1000, 0, -1000, 1000] };
test('packed uint16 scores decode little-endian from unaligned protobuf bytes', () => {
  const underlying = new Uint8Array([99, 0, 0, 255, 255, 254, 255, 0, 1, 99]);
  const scores = decodeScores(underlying.subarray(1, 9), grid);
  assert.deepEqual([...scores], [0, 65535, 65534, 256]);
  assert.throws(() => decodeScores(underlying, grid));
  assert.deepEqual(gridExtent(grid), [-1000, -1000, 1000, 1000]);
  const pixels = scorePixels(scores);
  assert.equal(pixels[3], 0);
  assert.equal(pixels[7], 0);
  assert.equal(pixels[8], 168);
  assert.equal(pixels[9], 85);
  assert.equal(pixels[11], 235);
  const s = scoreSummary([{ grid, scores }]);
  assert.equal(s.missing, 1);
  assert.equal(s.rejected, 1);
  assert.equal(s.best.index, 2);
});
test('result addressing rejects unknown or duplicate patches and sparse matrices', () => {
  const scores = new Uint8Array(8);
  const datasource = { grids: [grid] };
  assert.equal(prepareScores([{ site_key: 'patch', scores }], datasource).length, 1);
  assert.throws(() => prepareScores([{ site_key: 'elsewhere', scores }], datasource));
  assert.throws(() =>
    prepareScores(
      [
        { site_key: 'patch', scores },
        { site_key: 'patch', scores },
      ],
      datasource,
    ),
  );
  assert.throws(() =>
    prepareScores([{ site_key: 'patch', scores: new Uint8Array(2) }], datasource),
  );
  assert.throws(() => gridExtent({ ...grid, transform: [1000, 10, 0, 0, -1000, 0] }));
});

test('all-zero or missing results never invent a best location', () => {
  const summary = scoreSummary([{ grid, scores: new Uint16Array([0, 65535, 0, 65535]) }]);
  assert.equal(summary.best, null);
  assert.equal(summary.rejected, 2);
  assert.equal(summary.missing, 2);
});
