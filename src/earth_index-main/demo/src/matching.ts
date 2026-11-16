import type { Criterion, Target, Sample } from './types.ts';
import { numericMeasurement } from './characteristics.ts';

// Only quantities with an Earth counterpart are mapped. Bare cover is an
// editable Earth analogue preference, explicitly not a sampled planetary class.
const mappings: [string, string, string, number, number][] = [
  ['slope_mean_deg', 'slope_1km', 'degree', 2, 0.35],
  ['relief_5km_m', 'relief_5km', 'm', 50, 0.35],
  ['relief_21km_m', 'relief_21km', 'm', 150, 0.35],
  ['tpi_5km_m', 'tpi_5km', 'm', 25, 0.5],
  ['tpi_21km_m', 'tpi_21km', 'm', 75, 0.5],
];
function temperature(sample: Sample | null, body: string, id: string) {
  if (!sample) return null;
  for (const unit of ['degree_C', 'degree_Celsius', 'degC', 'K']) {
    const value = numericMeasurement(sample.measurements, `${body}.${id}`, unit);
    if (value != null) return unit === 'K' ? value - 273.15 : value;
  }
  return null;
}
export function applyMeasurements(
  criteria: Criterion[],
  target: Pick<Target, 'body'> | null,
  sample: Sample | null,
) {
  let measured = 0;
  for (const [key, id, unit, floor, fraction] of mappings) {
    const c = criteria.find((c) => c.key === key);
    if (!c) continue;
    c.target =
      target && sample
        ? numericMeasurement(sample.measurements, `${target.body}.${id}`, unit)
        : null;
    c.origin = 'Measured terrain; 1 km sample spacing. Earth uses aggregated cells.';
    c.kind = 'measured';
    if (c.selectionTouched && c.selectedByUser === undefined) c.selectedByUser = c.enabled;
    c.enabled = c.target != null && (!c.selectionTouched || c.selectedByUser === true);
    if (!c.toleranceTouched) c.tolerance = Math.max(floor, Math.abs(c.target ?? 0) * fraction);
    if (c.enabled) measured++;
  }
  const elevation = criteria.find((c) => c.key === 'elevation_median_m');
  if (elevation) {
    elevation.target =
      target && sample
        ? numericMeasurement(sample.measurements, `${target.body}.elevation`, 'm')
        : null;
    elevation.origin = 'Different planetary elevation datums; manual opt-in only.';
    if (elevation.target == null) elevation.enabled = false;
  }
  const bare = criteria.find((c) => c.key === 'bare_fraction');
  if (bare) {
    if (!bare.targetTouched) bare.target = target && sample ? 1 : null;
    if (!bare.toleranceTouched) bare.tolerance = 0.2;
    if (bare.selectionTouched && bare.selectedByUser === undefined)
      bare.selectedByUser = bare.enabled;
    bare.enabled =
      target && sample
        ? bare.selectionTouched
          ? bare.selectedByUser === true
          : measured > 0
        : false;
    bare.origin =
      'Analogue preference: favour Earth bare land (IGBP class 16). Not a planetary measurement or a land mask.';
    bare.kind = 'preference';
  }
  let approximate = 0;
  const thermal = [
    ['surface_low_c', target?.body === 'moon' ? 'surface_min' : 'modeled_surface_temperature_min'],
    ['surface_high_c', target?.body === 'moon' ? 'surface_max' : 'modeled_surface_temperature_max'],
    ['surface_mean_c', 'modeled_surface_temperature_mean'],
    ['surface_day_c', 'surface_noon'],
    ['surface_night_c', 'surface_midnight'],
  ];
  for (const [key, id] of thermal) {
    const c = criteria.find((c) => c.key === key);
    if (!c) continue;
    c.target = target ? temperature(sample, target.body, id) : null;
    // Noon/midnight are approximate counterparts only for the lunar products;
    // Mars's annual mean is separate from its sampled extrema.
    if (key === 'surface_mean_c' && target?.body !== 'mars') c.target = null;
    if (c.selectionTouched && c.selectedByUser === undefined) c.selectedByUser = c.enabled;
    c.enabled = c.target != null && (!c.selectionTouched || c.selectedByUser === true);
    if (!c.toleranceTouched) c.tolerance = 30;
    if (!c.weightTouched) c.weight = 0.5;
    c.kind = 'approximate';
    c.origin =
      'Approximate thermal comparison: MODIS clear-sky 2023 composite summaries versus planetary model/composite temperatures. Overpass times and temporal definitions differ; all thermal variables share one weight family.';
    if (c.enabled) approximate++;
  }
  return { measured, preferences: bare?.enabled ? 1 : 0, ...(approximate ? { approximate } : {}) };
}
