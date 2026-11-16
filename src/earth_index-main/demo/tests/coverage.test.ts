import { test } from 'node:test';
import assert from 'node:assert/strict';
import WKB from 'ol/format/WKB.js';
import Polygon from 'ol/geom/Polygon.js';
import { coverageFeatures } from '../src/coverage.ts';

test('binary API WKB preserves real polygon vertices, holes and SRID validation', () => {
  const coordinates = [
    [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
      [0, 0],
    ],
    [
      [0.5, 0.5],
      [0.5, 1],
      [1, 1],
      [1, 0.5],
      [0.5, 0.5],
    ],
  ];
  const bytes = new Uint8Array(new WKB({ hex: false }).writeGeometry(new Polygon(coordinates)));
  const features = coverageFeatures({ coverage: [{ siteKey: 'patch', srid: 4326, wkb: bytes }] });
  assert.equal(features[0].get('siteKey'), 'patch');
  assert.deepEqual(features[0].getGeometry().getCoordinates(), coordinates);
  assert.throws(() => coverageFeatures({ coverage: [{ srid: 3857, wkb: bytes }] }));
});
