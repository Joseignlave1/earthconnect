import type { MessageInitShape } from '@bufbuild/protobuf'
import { createClient } from '@connectrpc/connect'
import { createGrpcWebTransport } from '@connectrpc/connect-web'
import proj4 from 'proj4'
import type { BodyId, Criterion, EarthMatch } from './explorerData'
import {
  EarthIndex,
  MeasurementStatus,
  PlanetaryBody,
  type Datasource,
  type PointMeasurement,
  type QueryResponse,
  type Value,
  type ValueSchema,
} from './earth_index-main/demo/src/gen/earth_index_pb.ts'

export const API_BASE_URL = import.meta.env.VITE_EARTH_INDEX_API_URL ?? 'https://earth-index-crisrpia.fly.dev'

const api = createClient(EarthIndex, createGrpcWebTransport({ baseUrl: API_BASE_URL }))

type DecodedValue = null | boolean | number | string | Uint8Array | DecodedValue[] | { [key: string]: DecodedValue }
type AreaSearchResult = { count: number; areaCount: number; top: Area[] }
export type SampleMeasurement = PointMeasurement & { decoded: DecodedValue }
export type EarthDatasource = Datasource
export type SampleResult = { measurements: SampleMeasurement[]; partial: boolean; cached: boolean }
export type QueryTable = { columns: string[]; rows: Record<string, DecodedValue>[] }
export type ScorePatch = { grid: Datasource['grids'][number]; scores: Uint16Array }
export type ScoreDistribution = { bins: number[]; valid: number; missing: number; zero: number; best: number }
export type ScoreDataset = { patches: ScorePatch[]; datasource: Datasource; distribution: ScoreDistribution }
export type ScoreRanking = { matches: EarthMatch[]; cellCount: number; areaCount: number }
export type QueryResult = { kind: 'matches'; matches: EarthMatch[]; summary: string; dataset: ScoreDataset; threshold: number; cellCount: number; areaCount: number } | { kind: 'table'; table: QueryTable; summary: string }

const FIELD_DEFAULTS: Record<string, [string, number]> = {
  slope_mean_deg: ['Mean slope', 2],
  elevation_median_m: ['Elevation', 1000],
  relief_5km_m: ['5 km relief', 500],
  relief_21km_m: ['21 km relief', 1500],
  tpi_5km_m: ['5 km terrain position', 300],
  tpi_21km_m: ['21 km terrain position', 1000],
  coldest_month_air_c: ['Coldest-month air temperature', 20],
  warmest_month_air_c: ['Warmest-month air temperature', 20],
  annual_precip_mm: ['Annual precipitation', 500],
  bare_fraction: ['Bare ground', .25],
  permanent_snow_ice_fraction: ['Permanent snow / ice', .25],
  permafrost_fraction: ['Permafrost', .25],
  dominant_lithology_code: ['Lithology code', 1],
  surface_day_c: ['Mean daytime ground temperature', 30],
  surface_night_c: ['Mean nighttime ground temperature', 30],
  surface_mean_c: ['Mean paired ground temperature', 30],
  surface_low_c: ['Coldest observed nighttime composite', 30],
  surface_high_c: ['Hottest observed daytime composite', 30],
  surface_range_c: ['Observed ground-temperature range', 40],
}

function errorText(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}

export function friendlyApiError(error: unknown) {
  const message = errorText(error)
  if (/deadline|timeout/i.test(message)) return 'Earth Index timed out. Try again.'
  if (/fetch|network|unavailable/i.test(message)) return 'Earth Index is offline. Try again shortly.'
  return message
}

export async function loadDatasource(signal: AbortSignal) {
  const response = await api.metadata({}, { signal, timeoutMs: 120000 })
  const datasources = response.datasources
    .filter((item) => item.body.toLowerCase() === 'earth')
    .sort((a, b) => b.coverage.length - a.coverage.length || cellColumnCount(b) - cellColumnCount(a) || a.id.localeCompare(b.id))
  if (!datasources.length) throw new Error('The API did not return an Earth datasource.')
  return datasources[0]
}

function cellColumnCount(datasource: Datasource) {
  return datasource.relations.find((relation) => relation.name === 'cells')?.columns.length ?? 0
}

function qualityColumn(key: string) {
  if (/^(slope_|elevation_|relief_|tpi_)/.test(key)) return 'terrain_valid_fraction'
  if (['bare_fraction', 'permanent_snow_ice_fraction'].includes(key)) return 'landcover_valid_fraction'
  if (key === 'permafrost_fraction') return 'permafrost_valid_fraction'
  return ''
}

export function criteriaFor(datasource: Datasource, previous: Criterion[] = []): Criterion[] {
  const columns = datasource.relations.find((relation) => relation.name === 'cells')?.columns ?? []
  const names = new Set(columns.map((column) => column.name))
  return columns
    .filter((column) => column.unit && !column.name.endsWith('_valid_fraction'))
    .map((column) => {
      const prior = previous.find((item) => item.key === column.name)
      const quality = qualityColumn(column.name)
      return {
        ...prior,
        key: column.name,
        label: FIELD_DEFAULTS[column.name]?.[0] ?? column.name.replaceAll('_', ' '),
        unit: column.unit,
        description: column.description,
        group: column.name.startsWith('surface_') ? 'thermal' : column.name.startsWith('relief_') ? 'relief' : column.name.startsWith('tpi_') ? 'position' : column.name,
        enabled: prior?.enabled ?? false,
        target: prior?.target ?? null,
        tolerance: prior?.tolerance ?? FIELD_DEFAULTS[column.name]?.[1] ?? 1,
        weight: prior?.weight ?? 1,
        min: prior?.min ?? null,
        max: prior?.max ?? null,
        categorical: column.unit === 'category_code',
        qualityKey: quality && names.has(quality) ? quality : null,
      }
    })
}

export async function samplePoint(body: BodyId, latitude: number, longitude: number, signal: AbortSignal): Promise<SampleResult> {
  const response = await api.samplePoint({
    body: body === 'moon' ? PlanetaryBody.MOON : PlanetaryBody.MARS,
    latitude,
    longitude,
  }, { signal, timeoutMs: 60000 })
  return {
    measurements: response.measurements.map((measurement) => ({ ...measurement, decoded: decodeValue(measurement.value) })),
    partial: response.partial,
    cached: response.cached,
  }
}

function numericMeasurement(measurements: SampleMeasurement[], id: string, unit: string) {
  const measurement = measurements.find((item) => item.variableId === id && item.status === MeasurementStatus.AVAILABLE && item.unit === unit && !item.qualityFlags.includes('semantic_review_required') && typeof item.decoded === 'number' && Number.isFinite(item.decoded))
  return typeof measurement?.decoded === 'number' ? measurement.decoded : null
}

function temperature(measurements: SampleMeasurement[], body: BodyId, id: string) {
  for (const unit of ['degree_C', 'degree_Celsius', 'degC', 'K']) {
    const value = numericMeasurement(measurements, `${body}.${id}`, unit)
    if (value != null) return unit === 'K' ? value - 273.15 : value
  }
  return null
}

export function applySample(criteria: Criterion[], body: BodyId, sample: SampleResult): Criterion[] {
  const next = criteria.map((item) => ({ ...item }))
  let measured = 0
  const mappings: [string, string, string, number, number][] = [
    ['slope_mean_deg', 'slope_1km', 'degree', 2, .35],
    ['relief_5km_m', 'relief_5km', 'm', 50, .35],
    ['relief_21km_m', 'relief_21km', 'm', 150, .35],
    ['tpi_5km_m', 'tpi_5km', 'm', 25, .5],
    ['tpi_21km_m', 'tpi_21km', 'm', 75, .5],
  ]
  for (const [key, id, unit, floor, fraction] of mappings) {
    const criterion = next.find((item) => item.key === key)
    if (!criterion) continue
    criterion.target = numericMeasurement(sample.measurements, `${body}.${id}`, unit)
    criterion.kind = 'measured'
    criterion.origin = 'Planet data sampled at roughly 1 km.'
    criterion.enabled = criterion.target != null && (!criterion.selectionTouched || criterion.selectedByUser === true)
    if (!criterion.toleranceTouched) criterion.tolerance = Math.max(floor, Math.abs(criterion.target ?? 0) * fraction)
    if (criterion.enabled) measured++
  }
  const elevation = next.find((item) => item.key === 'elevation_median_m')
  if (elevation) {
    elevation.target = numericMeasurement(sample.measurements, `${body}.elevation`, 'm')
    elevation.enabled = elevation.selectionTouched === true && elevation.selectedByUser === true && elevation.target != null
    elevation.kind = 'manual'
    elevation.origin = 'Moon, Mars and Earth use different elevation datums. Enable this field manually.'
  }
  const bare = next.find((item) => item.key === 'bare_fraction')
  if (bare) {
    if (!bare.targetTouched) bare.target = 1
    if (!bare.toleranceTouched) bare.tolerance = .2
    bare.enabled = bare.selectionTouched ? bare.selectedByUser === true : measured > 0
    bare.kind = 'preference'
    bare.origin = 'Earth-only preference for bare land.'
  }
  const thermal: [string, string][] = [
    ['surface_low_c', body === 'moon' ? 'surface_min' : 'modeled_surface_temperature_min'],
    ['surface_high_c', body === 'moon' ? 'surface_max' : 'modeled_surface_temperature_max'],
    ['surface_mean_c', 'modeled_surface_temperature_mean'],
    ['surface_day_c', 'surface_noon'],
    ['surface_night_c', 'surface_midnight'],
  ]
  for (const [key, id] of thermal) {
    const criterion = next.find((item) => item.key === key)
    if (!criterion) continue
    criterion.target = temperature(sample.measurements, body, id)
    if (key === 'surface_mean_c' && body !== 'mars') criterion.target = null
    criterion.enabled = criterion.target != null && (!criterion.selectionTouched || criterion.selectedByUser === true)
    if (!criterion.toleranceTouched) criterion.tolerance = 30
    if (!criterion.weightTouched) criterion.weight = .5
    criterion.kind = 'approximate'
    criterion.origin = 'Approximate surface-temperature comparison.'
  }
  return next
}

export function buildMatchQuery(criteria: Criterion[]) {
  const params: Record<string, number> = {}
  const terms: { weight: string; loss: string; group: string; rawWeight: number }[] = []
  const required = new Set<string>()
  const filters: string[] = []
  const quality = new Set<string>()
  let index = 0
  for (const criterion of criteria) {
    const column = `"${criterion.key.replaceAll('"', '""')}"`
    const current = index++
    const add = (name: string, value: number | null) => {
      if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1e12) throw new Error(`${criterion.label}: enter a finite number within ±1e12`)
      params[`${name}_${current}`] = value
      return `:${name}_${current}`
    }
    if ((criterion.enabled || criterion.min != null || criterion.max != null) && criterion.qualityKey) quality.add(`"${criterion.qualityKey}"`)
    if (criterion.min != null) { filters.push(`${column} >= ${add('min', criterion.min)}`); required.add(column) }
    if (criterion.max != null) { filters.push(`${column} <= ${add('max', criterion.max)}`); required.add(column) }
    if (criterion.min != null && criterion.max != null && criterion.min > criterion.max) throw new Error(`${criterion.label}: minimum exceeds maximum`)
    if (!criterion.enabled) continue
    if (!(criterion.weight > 0) || criterion.weight > 1e6) throw new Error(`${criterion.label}: weight must be positive and at most 1e6`)
    if (!criterion.categorical && criterion.tolerance < 1e-6) throw new Error(`${criterion.label}: tolerance must be at least 0.000001`)
    required.add(column)
    const target = add('target', criterion.target)
    const weight = `(${add('weight', criterion.weight)}::double precision)`
    const loss = criterion.categorical ? `CASE WHEN ${column} = ${target} THEN 0.0 ELSE 25.0 END` : `power((${column} - ${target}) / ${add('tolerance', criterion.tolerance)}, 2)`
    terms.push({ weight, loss, group: criterion.group, rawWeight: criterion.weight })
  }
  if (!terms.length) throw new Error('Select at least one variable with a sampled target.')
  const groups = [...new Set(terms.map((term) => term.group))].map((group) => {
    const members = terms.filter((term) => term.group === group)
    const weight = members.reduce((best, term) => term.rawWeight > best.rawWeight ? term : best).weight
    return { weight, loss: `(${members.map((term) => `${term.weight} * (${term.loss})`).join(' + ')}) / (${members.map((term) => term.weight).join(' + ')})` }
  })
  const score = `exp(-0.5 * least(1000.0, (${groups.map((group) => `${group.weight} * (${group.loss})`).join(' +\n           ')}) / (${groups.map((group) => group.weight).join(' + ')})))`
  const cases: string[] = []
  if (quality.size) { params.minimum_coverage = .9; cases.push(`WHEN ${[...quality].map((column) => `${column} IS NULL OR ${column} < :minimum_coverage`).join(' OR ')} THEN NULL`) }
  if (required.size) cases.push(`WHEN ${[...required].map((column) => `${column} IS NULL`).join(' OR ')} THEN NULL`)
  if (filters.length) cases.push(`WHEN NOT (${filters.join(' AND ')}) THEN 0.0`)
  const expression = cases.length ? `CASE ${cases.join('\n         ')}\n         ELSE ${score} END` : score
  return {
    params,
    query: `WITH scored AS (\n  SELECT region_id, pixel,\n         ${expression} AS score\n  FROM measurements\n), matrices AS (\n  SELECT region_id, list(struct_pack(\n    index := pixel,\n    code := CASE WHEN score IS NULL THEN NULL\n                 ELSE round_even(65534 * greatest(0.0, least(1.0, score)), 0)::INTEGER END\n  )) AS codes\n  FROM scored\n  GROUP BY region_id\n)\nSELECT site_key, pack_u16_at(codes, width * height) AS scores\nFROM matrices JOIN regions USING (region_id)\nORDER BY site_key;`,
  }
}

export async function runQuery(datasource: Datasource, query: string, params: unknown, signal: AbortSignal, criteria: Criterion[] = []): Promise<QueryResult> {
  const response = await api.query({ datasourceId: datasource.id, query, params: encodeParams(params) }, { signal, timeoutMs: 120000 })
  const rows = resultRows(response)
  const names = response.columns.map((column) => column.name)
  if (!names.includes('site_key') || !names.includes('scores')) return { kind: 'table', table: { columns: names, rows }, summary: `${rows.length.toLocaleString()} ${rows.length === 1 ? 'row' : 'rows'} returned` }
  const dataset = prepareDataset(rows, datasource)
  const threshold = Math.min(75, Math.floor((dataset.distribution.best / SCORE_MAX) * 900) / 10)
  const ranking = await rankScoreDataset(dataset, criteria, threshold, signal)
  return { kind: 'matches', ...ranking, summary: `${ranking.areaCount.toLocaleString()} areas`, dataset, threshold }
}

function encodeValue(value: unknown): MessageInitShape<typeof ValueSchema> {
  let kind: MessageInitShape<typeof ValueSchema>['kind']
  if (value === null) kind = { case: 'nullValue', value: 0 }
  else if (typeof value === 'boolean') kind = { case: 'boolValue', value }
  else if (typeof value === 'string') kind = { case: 'stringValue', value }
  else if (typeof value === 'number') {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) throw new Error('Query parameters contain an unsafe number.')
    kind = Number.isInteger(value) ? { case: 'integerValue', value: BigInt(value) } : { case: 'numberValue', value }
  } else if (Array.isArray(value)) kind = { case: 'listValue', value: { values: value.map(encodeValue) } }
  else if (value && typeof value === 'object') kind = { case: 'objectValue', value: { fields: encodeParams(value) } }
  else throw new Error('Unsupported query parameter value.')
  return { kind }
}

function encodeParams(params: unknown): Record<string, MessageInitShape<typeof ValueSchema>> {
  if (!params || Array.isArray(params) || typeof params !== 'object') throw new Error('Query parameters must be a JSON object.')
  return Object.fromEntries(Object.entries(params).map(([key, value]) => [key, encodeValue(value)]))
}

function decodeValue(value: Value | undefined): DecodedValue {
  const kind = value?.kind
  if (!kind) return null
  switch (kind.case) {
    case 'nullValue': return null
    case 'integerValue': return kind.value >= BigInt(Number.MIN_SAFE_INTEGER) && kind.value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(kind.value) : kind.value.toString()
    case 'numberValue': case 'boolValue': case 'stringValue': case 'bytesValue': return kind.value
    case 'listValue': return kind.value.values.map(decodeValue)
    case 'objectValue': return Object.fromEntries(Object.entries(kind.value.fields).map(([key, item]) => [key, decodeValue(item)]))
    default: return null
  }
}

function resultRows(response: QueryResponse) {
  const names = response.columns.map((column) => column.name)
  if (new Set(names).size !== names.length) throw new Error('Query result column names must be unique.')
  return response.rows.map((row) => Object.fromEntries(names.map((name, index) => [name, decodeValue(row.values[index])])) as Record<string, DecodedValue>)
}

type Area = { siteKey: string; count: number; peak: number; peakIndex: number; mean: number; minX: number; maxX: number; minY: number; maxY: number }
type PreparedMatch = { match: EarthMatch; area: Area }
const SCORE_MAX = 65534
const SCORE_MISSING = 65535

function prepareDataset(rows: Record<string, DecodedValue>[], datasource: Datasource): ScoreDataset {
  const grids = new Map(datasource.grids.map((grid) => [grid.siteKey, grid]))
  const seen = new Set<string>()
  const patches: ScorePatch[] = rows.map((row) => {
    if (typeof row.site_key !== 'string' || !(row.scores instanceof Uint8Array)) throw new Error('The score query returned an unexpected shape.')
    const grid = grids.get(row.site_key)
    if (!grid) throw new Error(`Metadata is missing grid ${row.site_key}.`)
    if (seen.has(grid.siteKey)) throw new Error(`The query returned grid ${grid.siteKey} more than once.`)
    seen.add(grid.siteKey)
    if (row.scores.byteLength !== grid.width * grid.height * 2) throw new Error(`Score grid ${grid.siteKey} has an unexpected size.`)
    const view = new DataView(row.scores.buffer, row.scores.byteOffset, row.scores.byteLength)
    const scores = new Uint16Array(grid.width * grid.height)
    for (let index = 0; index < scores.length; index++) scores[index] = view.getUint16(index * 2, true)
    return { grid, scores }
  })
  return { patches, datasource, distribution: scoreDistribution(patches) }
}

function scoreDistribution(patches: ScorePatch[], bands = 50): ScoreDistribution {
  const bins = new Array<number>(bands).fill(0)
  let valid = 0
  let missing = 0
  let zero = 0
  let best = 0
  for (const patch of patches) for (const score of patch.scores) {
    if (score === SCORE_MISSING) { missing++; continue }
    valid++
    if (score === 0) zero++
    best = Math.max(best, score)
    bins[Math.min(bands - 1, Math.floor((score / SCORE_MAX) * bands))]++
  }
  return { bins, valid, missing, zero, best }
}

export async function rankScoreDataset(dataset: ScoreDataset, criteria: Criterion[], threshold: number, signal: AbortSignal) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 100) throw new Error('Minimum score must be between 0 and 100.')
  const grids = new Map(dataset.datasource.grids.map((grid) => [grid.siteKey, grid]))
  const cutoff = Math.ceil((threshold * SCORE_MAX) / 100)
  const result = findAreas(dataset.patches, cutoff, 50)
  const prepared = result.top.map((area, index) => ({ match: areaToMatch(area, grids.get(area.siteKey)!, index), area }))
  const matches = criteria.some((criterion) => criterion.enabled && criterion.target != null)
    ? await enrichMatches(dataset.datasource, prepared, criteria, signal)
    : prepared.map((item) => item.match)
  return { matches, cellCount: result.count, areaCount: result.areaCount }
}

async function enrichMatches(datasource: Datasource, prepared: PreparedMatch[], criteria: Criterion[], signal: AbortSignal) {
  if (!prepared.length) return []
  const available = new Set(datasource.relations.find((relation) => relation.name === 'cells')?.columns.map((column) => column.name) ?? [])
  const active = criteria.filter((criterion) => criterion.enabled && criterion.target != null && available.has(criterion.key))
  if (!active.length) return prepared.map((item) => item.match)
  const params: Record<string, string | number> = {}
  const locations = prepared.map((item, index) => {
    params[`site_${index}`] = item.area.siteKey
    params[`pixel_${index}`] = item.area.peakIndex
    return `(r.site_key = :site_${index} AND m.pixel = :pixel_${index})`
  })
  const columns = active.map((criterion) => `m."${criterion.key.replaceAll('"', '""')}"`).join(',\n       ')
  const query = `SELECT r.site_key, m.pixel,\n       ${columns}\nFROM measurements AS m\nJOIN regions AS r USING (region_id)\nWHERE ${locations.join('\n   OR ')}\nORDER BY r.site_key, m.pixel;`
  try {
    const response = await api.query({ datasourceId: datasource.id, query, params: encodeParams(params) }, { signal, timeoutMs: 60000 })
    const rows = resultRows(response)
    const values = new Map(rows.map((row) => [`${String(row.site_key)}:${String(row.pixel)}`, row]))
    return prepared.map(({ match, area }) => explainMatch(match, area, active, values.get(`${area.siteKey}:${area.peakIndex}`)))
  } catch (error) {
    if (signal.aborted) throw error
    return prepared.map((item) => item.match)
  }
}

function explainMatch(match: EarthMatch, area: Area, criteria: Criterion[], row: Record<string, DecodedValue> | undefined): EarthMatch {
  if (!row) return match
  const reasons = criteria.flatMap((criterion) => {
    const earth = row[criterion.key]
    if (typeof earth !== 'number' || criterion.target == null || !Number.isFinite(earth)) return []
    const similarity = criterion.categorical
      ? earth === criterion.target ? 100 : 0
      : 100 * Math.exp(-.5 * Math.pow((earth - criterion.target) / criterion.tolerance, 2))
    const digits = Math.max(Math.abs(earth), Math.abs(criterion.target)) < 10 ? 2 : 1
    const unit = criterion.unit === 'category_code' ? '' : ` ${criterion.unit}`
    return [{
      label: criterion.label,
      comparison: `Earth ${earth.toFixed(digits)}${unit} · target ${criterion.target.toFixed(digits)}${unit}`,
      similarity: Number(similarity.toFixed(1)),
    }]
  }).sort((a, b) => b.similarity - a.similarity)
  const strongest = reasons.slice(0, 3)
  if (!strongest.length) return match
  return {
    ...match,
    description: `Closest fields: ${strongest.map((reason) => reason.label.toLowerCase()).join(', ')}. ${area.count.toLocaleString()} adjacent ${area.count === 1 ? 'cell clears' : 'cells clear'} the cutoff.`,
    tags: match.tags,
    reasons: strongest,
    metrics: [
      ...strongest.map((reason) => ({ label: reason.label, value: reason.comparison.replace('Earth ', '') })),
      { label: 'Peak similarity', value: `${match.score.toFixed(2)}%` },
    ],
  }
}

function findAreas(patches: ScorePatch[], cutoff: number, limit: number): AreaSearchResult {
  let count = 0
  let areaCount = 0
  const top: Area[] = []
  const compare = (a: Area, b: Area) => b.peak - a.peak || b.mean - a.mean || b.count - a.count || a.siteKey.localeCompare(b.siteKey) || a.peakIndex - b.peakIndex
  for (const { grid, scores } of patches) {
    const visited = new Uint8Array(scores.length)
    const stack = new Int32Array(scores.length)
    const matches = (index: number) => scores[index] !== SCORE_MISSING && scores[index] > 0 && scores[index] >= cutoff
    for (let seed = 0; seed < scores.length; seed++) {
      if (visited[seed] || !matches(seed)) continue
      let size = 1
      let sum = 0
      stack[0] = seed
      visited[seed] = 1
      const area: Area = { siteKey: grid.siteKey, count: 0, peak: 0, peakIndex: seed, mean: 0, minX: grid.width, maxX: 0, minY: grid.height, maxY: 0 }
      while (size) {
        const item = stack[--size]
        const x = item % grid.width
        const y = Math.floor(item / grid.width)
        const score = scores[item]
        area.count++
        sum += score
        if (score > area.peak || (score === area.peak && item < area.peakIndex)) { area.peak = score; area.peakIndex = item }
        area.minX = Math.min(area.minX, x); area.maxX = Math.max(area.maxX, x); area.minY = Math.min(area.minY, y); area.maxY = Math.max(area.maxY, y)
        for (const neighbor of [x > 0 ? item - 1 : -1, x + 1 < grid.width ? item + 1 : -1, y > 0 ? item - grid.width : -1, y + 1 < grid.height ? item + grid.width : -1]) {
          if (neighbor >= 0 && !visited[neighbor] && matches(neighbor)) { visited[neighbor] = 1; stack[size++] = neighbor }
        }
      }
      area.mean = sum / area.count
      count += area.count
      areaCount++
      if (top.length < limit || compare(area, top[top.length - 1]) < 0) {
        let low = 0
        let high = top.length
        while (low < high) {
          const middle = (low + high) >>> 1
          if (compare(area, top[middle]) < 0) high = middle
          else low = middle + 1
        }
        top.splice(low, 0, area)
        if (top.length > limit) top.pop()
      }
    }
  }
  return { count, areaCount, top }
}

function areaToMatch(area: Area, grid: Datasource['grids'][number], rank: number): EarthMatch {
  const x = area.peakIndex % grid.width
  const y = Math.floor(area.peakIndex / grid.width)
  const [a, b, c, d, e, f] = grid.transform
  const projectedX = c + a * (x + .5) + b * (y + .5)
  const projectedY = f + d * (x + .5) + e * (y + .5)
  const [lon, lat] = proj4(grid.crsWkt || `EPSG:${grid.srid}`, 'EPSG:4326', [projectedX, projectedY])
  const score = area.peak / SCORE_MAX * 100
  const mean = area.mean / SCORE_MAX * 100
  const cellWidth = Math.hypot(a, d)
  const cellHeight = Math.hypot(b, e)
  const areaWidth = Math.max(8000, (area.maxX - area.minX + 1) * cellWidth)
  const areaHeight = Math.max(8000, (area.maxY - area.minY + 1) * cellHeight)
  const site = grid.siteKey.replaceAll('_', ' ').replaceAll('-', ' ')
  return {
    id: `${grid.siteKey}-${area.peakIndex}`,
    title: `${site.replace(/\b\w/g, (letter) => letter.toUpperCase())} · #${rank + 1}`,
    region: grid.siteKey,
    country: '4 km grid',
    description: `${area.count.toLocaleString()} adjacent ${area.count === 1 ? 'cell is' : 'cells are'} above the cutoff. The marker shows the highest score.`,
    lat, lon,
    score: Number(score.toFixed(1)),
    areaSemiMajor: Math.max(areaWidth, areaHeight) / 2,
    areaSemiMinor: Math.min(areaWidth, areaHeight) / 2,
    tags: [`${area.count.toLocaleString()} cells`, `${mean.toFixed(1)}% area mean`],
    metrics: [
      { label: 'Peak similarity', value: `${score.toFixed(2)}%` },
      { label: 'Area mean', value: `${mean.toFixed(2)}%` },
      { label: 'Grid cell', value: `${x + 1}, ${y + 1}` },
      { label: 'Connected cells', value: area.count.toLocaleString() },
    ],
  }
}
