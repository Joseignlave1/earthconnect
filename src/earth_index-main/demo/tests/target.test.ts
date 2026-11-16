import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { targetAt, tileUrl } from '../src/target.ts';

const configs = JSON.parse(readFileSync(new URL('../src/maps.json', import.meta.url)));

test('target coordinates wrap longitude, preserve poles, and reject nonplanet clicks', () => {
  assert.deepEqual(targetAt('moon', [181, 90]), { body: 'moon', longitude: -179, latitude: 90 });
  assert.deepEqual(targetAt('mars', [-181, -90]), { body: 'mars', longitude: 179, latitude: -90 });
  assert.deepEqual(targetAt('moon', [-0.00000001, -0.00000001]), {
    body: 'moon',
    longitude: 0,
    latitude: 0,
  });
  for (const [body, point] of [
    ['earth', [0, 0]],
    ['moon', [0, 91]],
    ['mars', [NaN, 0]],
    ['moon', [0, Infinity]],
  ])
    assert.equal(targetAt(body, point), null);
});

test('planetary grids cover two hemispheres with a north-west origin', () => {
  for (const config of configs.filter((c) => c.id !== 'earth')) {
    assert.equal(config.resolutions[0], 180 / 256);
    assert.deepEqual(config.sizes[0], [2, 1]);
    assert.equal(tileUrl(config, [0, 0, 0]), `/demo/overview/${config.id}-0.jpg`);
    assert.equal(tileUrl(config, [0, 1, 0]), `/demo/overview/${config.id}-1.jpg`);
    assert.match(tileUrl(config, [2, 4, 1]), /\/2\/1\/4\.jpg$/);
    assert.equal(tileUrl(config, [0, 2, 0]), undefined);
    assert.equal(tileUrl(config, [0, 0, 1]), undefined);
  }
});

test('Earth uses the GIBS grid rather than the planetary or Mercator grid', () => {
  const earth = configs.find((c) => c.id === 'earth');
  assert.ok(Math.abs(earth.resolutions[0] - 0.5625) < 1e-10);
  assert.deepEqual(earth.sizes[1], [3, 2]);
  assert.equal(tileUrl(earth, [0, 0, 0]), '/demo/overview/earth-0.jpg');
  assert.match(tileUrl(earth, [1, 2, 1]), /\/500m\/1\/1\/2\.jpeg$/);
});
