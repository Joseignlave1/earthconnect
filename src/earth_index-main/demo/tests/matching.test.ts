import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fieldsFor, buildQuery } from '../src/query.ts';
import { applyMeasurements } from '../src/matching.ts';
import { MeasurementStatus } from '../src/gen/earth_index_pb.ts';

import { fields, newCriteria, measurement, sample, target } from '../test-support/matching.ts';

test('defaults enable every compatible terrain measurement and label the bare preference', () => {
  const c = newCriteria();
  assert.deepEqual(applyMeasurements(c, target, sample), {
    measured: 5,
    preferences: 1,
  });
  assert.equal(c.filter((x) => x.enabled).length, 6);
  assert.equal(c.find((x) => x.key === 'elevation_median_m').enabled, false);
  assert.equal(c.find((x) => x.key === 'coldest_month_air_c').target, null);
  assert.equal(c.find((x) => x.key === 'slope_mean_deg').tolerance, 2);
  assert.equal(c.find((x) => x.key === 'relief_21km_m').tolerance, 175);
  assert.match(c.find((x) => x.key === 'bare_fraction').origin, /Not a planetary measurement/);
  const q = buildQuery(fields, c);
  assert.equal(q.params.minimum_coverage, 0.9);
  assert.match(q.query, /"landcover_valid_fraction" IS NULL/);
  assert.match(q.query, /"terrain_valid_fraction" < :minimum_coverage/);
});

test('missing, flagged and wrong-unit values cannot enable automatic comparisons', () => {
  const c = newCriteria();
  applyMeasurements(c, target, {
    measurements: [
      measurement('slope_1km', 0, 'degree', {
        status: MeasurementStatus.NO_DATA,
      }),
      measurement('relief_5km', 1, 'raw'),
      measurement('tpi_5km', 2, 'm', {
        qualityFlags: ['semantic_review_required'],
      }),
    ],
  });
  assert.equal(c.filter((x) => x.enabled).length, 0);
  assert.throws(() => buildQuery(fields, c), /Select at least one/);
  applyMeasurements(c, null, null);
  assert.equal(c.find((x) => x.key === 'bare_fraction').target, null);
});

test('user selections and tolerances survive new targets, missing evidence disables stale targets', () => {
  const c = newCriteria();
  applyMeasurements(c, target, sample);
  const slope = c.find((x) => x.key === 'slope_mean_deg');
  Object.assign(slope, {
    enabled: false,
    selectionTouched: true,
    tolerance: 0.5,
    toleranceTouched: true,
  });
  const bare = c.find((x) => x.key === 'bare_fraction');
  Object.assign(bare, { target: 0.8, targetTouched: true });
  applyMeasurements(c, target, {
    measurements: [measurement('slope_1km', 0, 'degree')],
  });
  assert.equal(slope.target, 0);
  assert.equal(slope.enabled, false);
  assert.equal(slope.tolerance, 0.5);
  assert.equal(bare.target, 0.8);
  assert.equal(c.find((x) => x.key === 'relief_5km_m').enabled, false);
});

test('invalid empty and extreme inputs fail in the builder', () => {
  const c = newCriteria();
  assert.throws(() => buildQuery(fields, c), /Select at least one/);
  applyMeasurements(c, target, sample);
  for (const change of [
    { target: Infinity },
    { target: 1e100 },
    { tolerance: 1e-200 },
    { weight: 1e100 },
    { weight: NaN },
  ]) {
    assert.throws(() => buildQuery(fields, [{ ...c[0], ...change }]));
  }
});

test('a temporary missing measurement cannot erase the user selection on recovery', () => {
  const c = newCriteria();
  applyMeasurements(c, target, sample);
  const relief = c.find((x) => x.key === 'relief_5km_m');
  Object.assign(relief, { selectionTouched: true, selectedByUser: true });
  applyMeasurements(c, target, { measurements: [] });
  assert.equal(relief.enabled, false);
  applyMeasurements(c, target, sample);
  assert.equal(relief.enabled, true);
});

test('thermal targets convert lunar Kelvin, distinguish approximate evidence, and preserve user weights', () => {
  const thermalFields = [
    'surface_day_c',
    'surface_night_c',
    'surface_mean_c',
    'surface_low_c',
    'surface_high_c',
    'surface_range_c',
  ];
  const c = thermalFields.map((key) => ({ key, enabled: false, target: null, weight: 1 }));
  const lunar = {
    measurements: [
      {
        variableId: 'moon.surface_min',
        unit: 'K',
        decoded: 100,
        status: MeasurementStatus.AVAILABLE,
        qualityFlags: [],
      },
      {
        variableId: 'moon.surface_max',
        unit: 'K',
        decoded: 400,
        status: MeasurementStatus.AVAILABLE,
        qualityFlags: [],
      },
    ],
  };
  applyMeasurements(c, { body: 'moon' }, lunar);
  assert.equal(c.find((x) => x.key === 'surface_low_c').target, -173.14999999999998);
  assert.equal(c.find((x) => x.key === 'surface_high_c').target, 126.85000000000002);
  assert.equal(c.find((x) => x.key === 'surface_mean_c').enabled, false);
  assert.equal(c.filter((x) => x.enabled).length, 2);
  assert.ok(c.filter((x) => x.enabled).every((x) => x.kind === 'approximate' && x.weight === 0.5));
  const low = c.find((x) => x.key === 'surface_low_c');
  low.weight = 0.1;
  low.weightTouched = true;
  applyMeasurements(c, { body: 'moon' }, lunar);
  assert.equal(low.weight, 0.1);
  applyMeasurements(c, { body: 'mars' }, { measurements: [] });
  assert.equal(c.filter((x) => x.enabled).length, 0);
});
