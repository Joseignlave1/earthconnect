import { createClient } from '@connectrpc/connect'
import { createGrpcWebTransport } from '@connectrpc/connect-web'
import type { BodyId, Criterion } from './explorerData'
import {
  EarthIndex,
  MeasurementStatus,
  PlanetaryBody,
  type Datasource,
  type PointMeasurement,
  type Value,
} from '../vendor/earth-index/earth_index_pb.ts'

const API_BASE_URL = import.meta.env.VITE_EARTH_INDEX_API_URL ?? 'https://earth-index-crisrpia.fly.dev'

const api = createClient(EarthIndex, createGrpcWebTransport({ baseUrl: API_BASE_URL }))

type DecodedValue = null | boolean | number | string | Uint8Array | DecodedValue[] | { [key: string]: DecodedValue }
export type SampleMeasurement = PointMeasurement & { decoded: DecodedValue }
export type SampleResult = { measurements: SampleMeasurement[]; partial: boolean; cached: boolean }

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
    .filter(item => item.body.toLowerCase() === 'earth')
    .sort(
      (a, b) =>
        b.coverage.length - a.coverage.length || cellColumnCount(b) - cellColumnCount(a) || a.id.localeCompare(b.id),
    )
  if (!datasources.length) throw new Error('The API did not return an Earth datasource.')
  return datasources[0]
}

function cellColumnCount(datasource: Datasource) {
  return datasource.relations.find(relation => relation.name === 'cells')?.columns.length ?? 0
}

function qualityColumn(key: string) {
  if (/^(slope_|elevation_|relief_|tpi_)/.test(key)) return 'terrain_valid_fraction'
  if (['bare_fraction', 'permanent_snow_ice_fraction'].includes(key)) return 'landcover_valid_fraction'
  if (key === 'permafrost_fraction') return 'permafrost_valid_fraction'
  return ''
}

export function criteriaFor(datasource: Datasource, previous: Criterion[] = []): Criterion[] {
  return criteriaForColumns(datasource.relations.find(relation => relation.name === 'cells')?.columns ?? [], previous)
}

type CriteriaColumn = Pick<Datasource['relations'][number]['columns'][number], 'name' | 'unit' | 'description'>

function criteriaForColumns(columns: readonly CriteriaColumn[], previous: Criterion[] = []): Criterion[] {
  const names = new Set(columns.map(column => column.name))
  return columns
    .filter(column => column.unit && !column.name.endsWith('_valid_fraction'))
    .map(column => {
      const prior = previous.find(item => item.key === column.name)
      const quality = qualityColumn(column.name)
      return {
        ...prior,
        key: column.name,
        label: FIELD_DEFAULTS[column.name]?.[0] ?? column.name.replaceAll('_', ' '),
        unit: column.unit,
        description: column.description,
        group: column.name.startsWith('surface_')
          ? 'thermal'
          : column.name.startsWith('relief_')
            ? 'relief'
            : column.name.startsWith('tpi_')
              ? 'position'
              : column.name,
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

export async function samplePoint(
  body: BodyId,
  latitude: number,
  longitude: number,
  signal: AbortSignal,
): Promise<SampleResult> {
  const response = await api.samplePoint(
    {
      body: body === 'moon' ? PlanetaryBody.MOON : PlanetaryBody.MARS,
      latitude,
      longitude,
    },
    { signal, timeoutMs: 60000 },
  )
  return {
    measurements: response.measurements.map(measurement => ({
      ...measurement,
      decoded: decodeValue(measurement.value),
    })),
    partial: response.partial,
    cached: response.cached,
  }
}

function numericMeasurement(measurements: SampleMeasurement[], id: string, unit: string) {
  const measurement = measurements.find(
    item =>
      item.variableId === id &&
      item.status === MeasurementStatus.AVAILABLE &&
      item.unit === unit &&
      !item.qualityFlags.includes('semantic_review_required') &&
      typeof item.decoded === 'number' &&
      Number.isFinite(item.decoded),
  )
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
  const next = criteria.map(item => ({ ...item }))
  let measured = 0
  const mappings: [string, string, string, number, number][] = [
    ['slope_mean_deg', 'slope_1km', 'degree', 2, 0.35],
    ['relief_5km_m', 'relief_5km', 'm', 50, 0.35],
    ['relief_21km_m', 'relief_21km', 'm', 150, 0.35],
    ['tpi_5km_m', 'tpi_5km', 'm', 25, 0.5],
    ['tpi_21km_m', 'tpi_21km', 'm', 75, 0.5],
  ]
  for (const [key, id, unit, floor, fraction] of mappings) {
    const criterion = next.find(item => item.key === key)
    if (!criterion) continue
    criterion.target = numericMeasurement(sample.measurements, `${body}.${id}`, unit)
    criterion.kind = 'measured'
    criterion.origin = 'Planet data sampled at roughly 1 km.'
    criterion.enabled = criterion.target != null && (!criterion.selectionTouched || criterion.selectedByUser === true)
    if (!criterion.toleranceTouched) criterion.tolerance = Math.max(floor, Math.abs(criterion.target ?? 0) * fraction)
    if (criterion.enabled) measured++
  }
  const elevation = next.find(item => item.key === 'elevation_median_m')
  if (elevation) {
    elevation.target = numericMeasurement(sample.measurements, `${body}.elevation`, 'm')
    elevation.enabled =
      elevation.selectionTouched === true && elevation.selectedByUser === true && elevation.target != null
    elevation.kind = 'manual'
    elevation.origin = 'Moon, Mars and Earth use different elevation datums. Enable this field manually.'
  }
  const bare = next.find(item => item.key === 'bare_fraction')
  if (bare) {
    if (!bare.targetTouched) bare.target = 1
    if (!bare.toleranceTouched) bare.tolerance = 0.2
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
    const criterion = next.find(item => item.key === key)
    if (!criterion) continue
    criterion.target = temperature(sample.measurements, body, id)
    if (key === 'surface_mean_c' && body !== 'mars') criterion.target = null
    criterion.enabled = criterion.target != null && (!criterion.selectionTouched || criterion.selectedByUser === true)
    if (!criterion.toleranceTouched) criterion.tolerance = 30
    if (!criterion.weightTouched) criterion.weight = 0.5
    criterion.kind = 'approximate'
    criterion.origin = 'Approximate surface-temperature comparison.'
  }
  return next
}

function decodeValue(value: Value | undefined): DecodedValue {
  const kind = value?.kind
  if (!kind) return null
  switch (kind.case) {
    case 'nullValue':
      return null
    case 'integerValue':
      return kind.value >= BigInt(Number.MIN_SAFE_INTEGER) && kind.value <= BigInt(Number.MAX_SAFE_INTEGER)
        ? Number(kind.value)
        : kind.value.toString()
    case 'numberValue':
    case 'boolValue':
    case 'stringValue':
    case 'bytesValue':
      return kind.value
    case 'listValue':
      return kind.value.values.map(decodeValue)
    case 'objectValue':
      return Object.fromEntries(Object.entries(kind.value.fields).map(([key, item]) => [key, decodeValue(item)]))
    default:
      return null
  }
}
