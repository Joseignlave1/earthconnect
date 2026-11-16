import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scorePyramid, overviewLevel } from '../src/score-overviews.ts';

test('an isolated peak survives every overview without changing the native matrix', () => {
  const cells = new Uint16Array(16);
  cells[3] = 60000;
  cells[12] = 42000;
  const before = cells.slice();
  const levels = scorePyramid(cells, 4, 4);
  assert.deepEqual(
    levels.map((l) => [l.width, l.height]),
    [
      [4, 4],
      [2, 2],
      [1, 1],
    ],
  );
  for (const level of levels) assert.equal(Math.max(...level.scores), 60000);
  assert.deepEqual([...levels[1].scores], [0, 60000, 42000, 0]);
  assert.deepEqual(cells, before);
  assert.equal(levels[0].scores, cells);
});

test('overviews keep zero separate from missing and preserve odd-grid edge peaks', () => {
  assert.equal(
    scorePyramid(new Uint16Array([65535, 65535, 65535, 65535]), 2, 2)[1].scores[0],
    65535,
  );
  assert.equal(scorePyramid(new Uint16Array([65535, 0, 65535, 0]), 2, 2)[1].scores[0], 0);
  const cells = new Uint16Array(15);
  cells[0] = 65535;
  cells[14] = 12345;
  const levels = scorePyramid(cells, 5, 3);
  assert.deepEqual(
    levels.map((l) => [l.width, l.height]),
    [
      [5, 3],
      [3, 2],
      [2, 1],
      [1, 1],
    ],
  );
  for (const level of levels) assert.ok(level.scores.includes(12345));
});

test('zoom selects screen-sized max texels and returns native cells for close inspection', () => {
  const levels = scorePyramid(new Uint16Array(10000), 100, 100);
  assert.equal(overviewLevel(levels, [1, 1], 0.001), 0);
  assert.equal(overviewLevel(levels, [1, 1], 0.019), 1);
  assert.equal(levels[overviewLevel(levels, [1, 1], 1)].width, 1);
  assert.equal(levels[overviewLevel(levels, [1, 1], 2)].width, 1);
});
