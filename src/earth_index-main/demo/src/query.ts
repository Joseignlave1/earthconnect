import type { FieldsMetadata, Field, Criterion } from './types.ts';
// Form -> SQL is deterministic. Every variable identifier comes from Metadata;
// all user-entered numbers are bound parameters, never interpolated SQL.
const defaults: Record<string, [string, number]> = {
  slope_mean_deg: ['Mean slope', 2],
  elevation_median_m: ['Elevation', 1000],
  relief_5km_m: ['5 km relief', 500],
  relief_21km_m: ['21 km relief', 1500],
  tpi_5km_m: ['5 km terrain position', 300],
  tpi_21km_m: ['21 km terrain position', 1000],
  coldest_month_air_c: ['Coldest-month air temperature', 20],
  warmest_month_air_c: ['Warmest-month air temperature', 20],
  annual_precip_mm: ['Annual precipitation', 500],
  bare_fraction: ['Bare ground', 0.25],
  permanent_snow_ice_fraction: ['Permanent snow / ice', 0.25],
  permafrost_fraction: ['Permafrost', 0.25],
  dominant_lithology_code: ['Lithology code', 1],
  surface_day_c: ['Mean daytime ground temperature', 30],
  surface_night_c: ['Mean nighttime ground temperature', 30],
  surface_mean_c: ['Mean paired ground temperature', 30],
  surface_low_c: ['Coldest observed nighttime composite', 30],
  surface_high_c: ['Hottest observed daytime composite', 30],
  surface_range_c: ['Observed ground-temperature range', 40],
};
export function fieldsFor(datasource: FieldsMetadata | null): Field[] {
  const columns = datasource?.relations.find((r) => r.name === 'cells')?.columns ?? [];
  const names = new Set(columns.map((c) => c.name));
  return columns
    .filter((c) => c.unit && !c.name.endsWith('_valid_fraction'))
    .map((c) => ({
      key: c.name,
      label: defaults[c.name]?.[0] ?? c.name.replaceAll('_', ' '),
      unit: c.unit,
      description: c.description,
      categorical: c.unit === 'category_code',
      tolerance: defaults[c.name]?.[1] ?? 1,
      group: c.name.startsWith('surface_')
        ? 'thermal'
        : c.name.startsWith('relief_')
          ? 'relief'
          : c.name.startsWith('tpi_')
            ? 'position'
            : c.name,
      qualityKey: names.has(qualityColumn(c.name)) ? qualityColumn(c.name) : null,
    }));
}
function qualityColumn(key: string): string {
  if (/^(slope_|elevation_|relief_|tpi_)/.test(key)) return 'terrain_valid_fraction';
  if (['bare_fraction', 'permanent_snow_ice_fraction'].includes(key))
    return 'landcover_valid_fraction';
  if (key === 'permafrost_fraction') return 'permafrost_valid_fraction';
  return '';
}
export function buildQuery(fields: Field[], criteria: Criterion[]) {
  const names = new Set(fields.map((field) => field.key));
  const params: Record<string, number> = {},
    terms: { weight: string; loss: string; group: string; rawWeight: number }[] = [],
    required = new Set<string>(),
    filters: string[] = [],
    quality = new Set<string>();
  let i = 0;
  for (const criterion of criteria) {
    if (!names.has(criterion.key)) throw new Error('Unknown query variable');
    const field = fields.find((f) => f.key === criterion.key)!;
    const column = `"${criterion.key.replaceAll('"', '""')}"`;
    const n = i++;
    const add = (name: string, value: number | null) => {
      if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1e12)
        throw new Error(`${field.label}: enter a finite number within ±1e12`);
      params[`${name}_${n}`] = value;
      return `:${name}_${n}`;
    };
    if ((criterion.enabled || criterion.min != null || criterion.max != null) && field.qualityKey)
      quality.add(`"${field.qualityKey}"`);
    if (criterion.min !== null && criterion.min !== undefined) {
      filters.push(`${column} >= ${add('min', criterion.min)}`);
      required.add(column);
    }
    if (criterion.max !== null && criterion.max !== undefined) {
      filters.push(`${column} <= ${add('max', criterion.max)}`);
      required.add(column);
    }
    if (criterion.min != null && criterion.max != null && criterion.min > criterion.max)
      throw new Error(`${field.label}: minimum exceeds maximum`);
    if (!criterion.enabled) continue;
    if (!(criterion.weight > 0) || criterion.weight > 1e6)
      throw new Error(`${field.label}: weight must be positive and at most 1e6`);
    if (!field.categorical && !(criterion.tolerance >= 1e-6))
      throw new Error(`${field.label}: tolerance must be at least 0.000001`);
    required.add(column);
    const target = add('target', criterion.target),
      weight = `(${add('weight', criterion.weight)}::double precision)`;
    const loss = field.categorical
      ? `CASE WHEN ${column} = ${target} THEN 0.0 ELSE 25.0 END`
      : `power((${column} - ${target}) / ${add('tolerance', criterion.tolerance)}, 2)`;
    terms.push({
      weight,
      loss,
      group: field.group ?? field.key,
      rawWeight: criterion.weight,
    });
  }
  if (!terms.length) throw new Error('Select at least one score variable with a target');
  // Each family's total weight is its strongest member, so adding a correlated
  // scale never buys that family extra votes. Use a geometric mean of Gaussian
  // similarities (exp of average loss) to prevent compensation by perfect terms.
  const groups = [...new Set(terms.map((t) => t.group))].map((group) => {
    const members = terms.filter((t) => t.group === group);
    const weight = members.reduce((best, t) => (t.rawWeight > best.rawWeight ? t : best)).weight;
    return {
      weight,
      loss: `(${members.map((t) => `${t.weight} * (${t.loss})`).join(' + ')}) / (${members.map((t) => t.weight).join(' + ')})`,
    };
  });
  // exp(-500) already rounds to zero in uint16; the clamp also avoids a
  // floating-point underflow for very distant candidates.
  const score = `exp(-0.5 * least(1000.0, (${groups.map((g) => `${g.weight} * (${g.loss})`).join(' +\n           ')}) / (${groups.map((g) => g.weight).join(' + ')})))`;
  const cases = [];
  if (quality.size) {
    params.minimum_coverage = 0.9;
    cases.push(
      `WHEN ${[...quality].map((column) => `${column} IS NULL OR ${column} < :minimum_coverage`).join(' OR ')} THEN NULL`,
    );
  }
  if (required.size)
    cases.push(`WHEN ${[...required].map((column) => `${column} IS NULL`).join(' OR ')} THEN NULL`);
  if (filters.length) cases.push(`WHEN NOT (${filters.join(' AND ')}) THEN 0.0`);
  // Keep rejected cells in the matrix as zero; NULL stays missing (65535).
  const expression = cases.length
    ? `CASE ${cases.join('\n         ')}\n         ELSE ${score} END`
    : score;
  return {
    params,
    query: `WITH scored AS (\n  SELECT region_id, pixel,\n         ${expression} AS score\n  FROM measurements\n), matrices AS (\n  SELECT region_id, list(struct_pack(\n    index := pixel,\n    code := CASE WHEN score IS NULL THEN NULL\n                 ELSE round_even(65534 * greatest(0.0, least(1.0, score)), 0)::INTEGER END\n  )) AS codes\n  FROM scored\n  GROUP BY region_id\n)\nSELECT site_key, pack_u16_at(codes, width * height) AS scores\nFROM matrices JOIN regions USING (region_id)\nORDER BY site_key;`,
  };
}
