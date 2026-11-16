import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fieldsFor, buildQuery } from '../src/query.ts';
import { encodeParams, decodeValue, resultRows } from '../src/values.ts';
const fields = [
  { key: 'slope_mean_deg', label: 'Slope' },
  { key: 'bare_fraction', label: 'Bare' },
  { key: 'dominant_lithology_code', label: 'Rock', categorical: true },
];
const criterion = {
  key: 'slope_mean_deg',
  enabled: true,
  target: 0,
  tolerance: 10,
  weight: 1,
  min: null,
  max: 30,
};
test('builder binds named values, scores without sorting cells and preserves the full matrix', () => {
  const q = buildQuery(fields, [criterion]);
  assert.equal(q.params.target_0, 0);
  assert.equal(q.params.max_0, 30);
  assert.match(q.query, /IS NULL.*THEN NULL/);
  assert.match(q.query, /WHEN NOT.*THEN 0\.0/);
  assert.match(q.query, /index := pixel/);
  assert.match(q.query, /pack_u16_at/);
  assert.match(q.query, /round_even/);
  assert.match(q.query, /FROM measurements/);
  assert.match(q.query, /GROUP BY region_id/);
  assert.doesNotMatch(q.query, /WHERE|ORDER BY score/);
  assert.match(q.query, /exp\(-0\.5/);
  assert.match(q.query, /65534/);
});
test('metadata limits builder fields and categories use exact equality', () => {
  assert.deepEqual(
    fieldsFor({
      relations: [
        {
          name: 'cells',
          columns: [
            { name: 'geom', unit: '' },
            { name: 'terrain_valid_fraction', unit: '1' },
            { name: 'slope_mean_deg', unit: 'degree' },
          ],
        },
      ],
    }).map((f) => f.key),
    ['slope_mean_deg'],
  );
  const q = buildQuery(fields, [
    { key: 'dominant_lithology_code', enabled: true, target: 5, weight: 2 },
  ]);
  assert.match(q.query, /CASE WHEN "dominant_lithology_code" = :target_0/);
  assert.doesNotMatch(q.query, /power/);
  assert.throws(() => buildQuery(fields, [{ ...criterion, key: 'not_a_column' }]));
  assert.throws(() => buildQuery(fields, [{ ...criterion, tolerance: 0 }]));
  assert.throws(() => buildQuery(fields, [{ ...criterion, min: 50 }]));
});
test('recursive params and results retain null, zero, bytes and exact int64', () => {
  const params = encodeParams({
    zero: 0,
    flag: false,
    list: [null, 1.5],
    nested: { name: 'rock' },
  });
  assert.equal(decodeValue(params.zero), 0);
  assert.deepEqual(decodeValue(params.list), [null, 1.5]);
  const bytes = new Uint8Array([0, 0, 255, 255]);
  assert.equal(decodeValue({ kind: { case: 'bytesValue', value: bytes } }), bytes);
  assert.equal(
    decodeValue({ kind: { case: 'integerValue', value: 9007199254740993n } }),
    '9007199254740993',
  );
  assert.throws(() => encodeParams({ big: 9007199254740992 }));
  assert.throws(() => encodeParams([]));
  assert.throws(() => resultRows({ columns: [{ name: 'same' }, { name: 'same' }], rows: [] }));
});
