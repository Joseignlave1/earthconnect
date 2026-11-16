import { test } from 'node:test';
import assert from 'node:assert/strict';
import proj4 from 'proj4';
import { scoreAtlas, sampleAtlas, atlasLevel } from '../src/score-atlas.ts';
import type { ScorePatch } from '../src/types.ts';

function patch(key: string, x: number, scores: number[]): ScorePatch {
  return {
    grid: {
      siteKey: key,
      srid: 999002,
      crsWkt: '+proj=longlat +datum=WGS84 +no_defs',
      width: 3,
      height: 2,
      transform: [1, 0, x, 0, -1, 2],
    },
    scores: new Uint16Array(scores),
  };
}
test('one atlas addresses multiple regions, gaps, zero and missing without changing native values', () => {
  const a = patch('a', 0, [0, 10000, 65535, 30000, 40000, 50000]);
  const b = patch('b', 6, [65534, 0, 65535, 1, 2, 3]);
  const atlas = scoreAtlas([a, b]);
  assert.deepEqual([...atlas.lookup], [0, 65535, 1]);
  for (const p of [a, b])
    for (let y = 0; y < 2; y++)
      for (let x = 0; x < 3; x++)
        assert.equal(
          sampleAtlas(atlas, p.grid.transform[2] + x + 0.5, 1.5 - y),
          p.scores[y * 3 + x],
        );
  assert.equal(sampleAtlas(atlas, 4, 1), 65535);
  assert.equal(sampleAtlas(atlas, -1, 1), 65535);
  assert.equal(sampleAtlas(atlas, 9, 1), 65535);
  assert.equal(sampleAtlas(atlas, 181, 1), 65535);
  assert.equal(atlas.patches[0].levels[0].scores, a.scores);
});
test('odd rectangular overviews keep their exact footprint and isolated peaks', () => {
  const atlas = scoreAtlas([patch('a', 0, [0, 0, 65535, 0, 0, 60000])]);
  const last = atlas.patches[0].levels.length - 1;
  assert.equal(sampleAtlas(atlas, 0.1, 1.9, last), 60000);
  assert.equal(sampleAtlas(atlas, 2.9, 0.1, last), 60000);
  assert.equal(sampleAtlas(atlas, 2.9, -0.1, last), 65535);
  assert.equal(atlasLevel(atlas, 0.01), 0);
  assert.equal(atlasLevel(atlas, 10), last);
});
test('sinusoidal lookup agrees with native cell centers at the dateline and high latitudes', () => {
  const wkt = '+proj=sinu +R=6371007.181 +units=m +no_defs';
  for (const [longitude, latitude] of [
    [179, 70],
    [-179, -70],
    [0, 0],
  ]) {
    const [x, y] = proj4('EPSG:4326', wkt, [longitude, latitude]);
    const p: ScorePatch = {
      grid: {
        siteKey: 's',
        srid: 999003,
        crsWkt: wkt,
        width: 1,
        height: 1,
        transform: [4000, 0, x - 2000, 0, -4000, y + 2000],
      },
      scores: new Uint16Array([12345]),
    };
    assert.equal(sampleAtlas(scoreAtlas([p]), longitude, latitude), 12345);
  }
});
test('misaligned, overlapping and incompatible regions fail rather than drawing misleading data', () => {
  const a = patch('a', 0, [0, 0, 0, 0, 0, 0]);
  assert.throws(() => scoreAtlas([a, patch('bad', 3.1, [0, 0, 0, 0, 0, 0])]), /Misaligned/);
  assert.throws(() => scoreAtlas([a, patch('bad', 0, [0, 0, 0, 0, 0, 0])]), /Overlapping/);
  const bad = patch('bad', 3, [0, 0, 0, 0, 0, 0]);
  bad.grid.transform[0] = 2;
  assert.throws(() => scoreAtlas([a, bad]), /regular grid/);
});
