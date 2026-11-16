import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scorePixels, MAX_SCORE } from '../src/scores.ts';
import { scoreIntensity, strongestAreas, glowOpacity } from '../src/score-visuals.ts';

const patch = (siteKey: string, values: number[]) => ({
  grid: { siteKey, width: values.length, height: 1 },
  scores: new Uint16Array(values),
});

test('purple opacity uses one result-wide ceiling, with zero, unknown and filtered cells transparent', () => {
  const maximum = 20000;
  const codes = new Uint16Array([0, 65535, 5000, 10000, maximum]);
  const pixels = scorePixels(codes, 0, maximum);
  assert.deepEqual([pixels[3], pixels[7]], [0, 0]);
  assert.ok(pixels[11] < pixels[15] && pixels[15] < pixels[19]);
  assert.equal(pixels[19], 235); // A modest absolute best still has full visual strength.
  assert.equal(scorePixels(new Uint16Array([10000]), 0, maximum)[3], pixels[15]);
  assert.equal(scorePixels(codes, 10001, maximum)[15], 0);
  assert.equal(scorePixels(codes, 10001, maximum)[19], pixels[19]); // Cutoff does not rescale.
  assert.equal(scoreIntensity(MAX_SCORE, 0), 0);
  assert.equal(scoreIntensity(0, maximum), 0);
});

test('distant halos mark bounded actual area peaks relative to the global best, excluding weak patches', () => {
  const result = strongestAreas(
    [patch('best', [0, 20000, 19000]), patch('weak', [9000, 0, 10000])],
    20000,
  );
  assert.equal(result.length, 1);
  assert.deepEqual(
    { site: result[0].siteKey, index: result[0].peakIndex, peak: result[0].peak },
    { site: 'best', index: 1, peak: 20000 },
  );
  assert.deepEqual(strongestAreas([patch('none', [0, 65535])], 0), []);
  const islands = Array.from({ length: 300 }, (_, i) => (i % 2 ? 0 : 20000));
  assert.equal(strongestAreas([patch('many', islands)], 20000).length, 100);
});

test('highlights fade once an area is four pixels across', () => {
  assert.equal(glowOpacity(1, 0.01, 1), 1);
  assert.equal(glowOpacity(0.002, 0.01, 1), 0);
  assert.ok(glowOpacity(0.00375, 0.01, 1) > 0 && glowOpacity(0.00375, 0.01, 1) < 1);
  assert.equal(glowOpacity(0.6, 0.9, 1), 1); // 100 km patch at world scale.
  assert.equal(glowOpacity(0.1, 0.9, 1), 0); // Shape is now visible; no beacon.
  assert.ok(glowOpacity(1, 0.01, 0.9) < glowOpacity(1, 0.01, 1));
});
