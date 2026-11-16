import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadCharacteristics, numericMeasurement, targetLoader } from '../src/characteristics.ts';
import { MeasurementStatus, PlanetaryBody } from '../src/gen/earth_index_pb.ts';
const measured = (variableId, unit, number, extras = {}) => ({
  variableId,
  unit,
  status: MeasurementStatus.AVAILABLE,
  value: { kind: { case: 'numberValue', value: number } },
  qualityFlags: [],
  ...extras,
});

test('map target uses one backend RPC; zero and raw-scale units remain distinct', async () => {
  let calls = 0;
  const target = { body: 'mars', longitude: 23.47, latitude: 0.67 };
  const signal = new AbortController().signal;
  const result = await loadCharacteristics(target, signal, {
    async samplePoint(request, options) {
      calls++;
      assert.deepEqual(request, { body: PlanetaryBody.MARS, longitude: 23.47, latitude: 0.67 });
      assert.equal(options.signal, signal);
      return {
        partial: true,
        measurements: [
          measured('mars.local_slope', 'degree', 0),
          measured('mars.raw', 'raw', 3, { qualityFlags: ['semantic_review_required'] }),
          measured('mars.bad', 'degree', 99, {
            status: MeasurementStatus.NO_DATA,
            value: undefined,
          }),
        ],
      };
    },
  });
  assert.equal(calls, 1);
  assert.equal(result.failed, true);
  assert.equal(numericMeasurement(result.measurements, 'mars.local_slope', 'degree'), 0);
  assert.equal(numericMeasurement(result.measurements, 'mars.local_slope', 'm'), null);
  assert.equal(numericMeasurement(result.measurements, 'mars.raw', 'raw'), null);
  assert.equal(numericMeasurement(result.measurements, 'mars.bad', 'degree'), null);
});

test('rapid target changes and clearing suppress obsolete RPC responses', async () => {
  const pending = [],
    results = [];
  const load = targetLoader(
    (result, state) => results.push({ result, state }),
    (target, signal) => new Promise((resolve) => pending.push({ target, signal, resolve })),
  );
  const a = load({ body: 'moon' }),
    b = load({ body: 'mars' });
  assert.equal(pending[0].signal.aborted, true);
  pending[1].resolve({ body: 'mars' });
  await b;
  pending[0].resolve({ body: 'moon' });
  await a;
  assert.equal(results.at(-1).result.body, 'mars');
  const c = load({ body: 'moon' });
  await load(null);
  pending[2].resolve({ body: 'moon' });
  await c;
  assert.deepEqual(results.at(-1), { result: null, state: 'empty' });
});
