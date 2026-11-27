import type { BodyId, Criterion } from './explorerData'
import { earthIndexClient, type SampleMeasurement, type SampleResult } from './earthIndexApi'
import { MeasurementStatus, type Datasource } from '../vendor/earth-index/earth_index_pb.ts'

type FieldDefault = Readonly<{ label: string; tolerance: number }>

const FIELD_DEFAULTS: Record<string, FieldDefault> = {
  slope_mean_deg: { label: 'Mean slope', tolerance: 2 },
  elevation_median_m: { label: 'Elevation', tolerance: 1000 },
  relief_5km_m: { label: '5 km relief', tolerance: 500 },
  relief_21km_m: { label: '21 km relief', tolerance: 1500 },
  tpi_5km_m: { label: '5 km terrain position', tolerance: 300 },
  tpi_21km_m: { label: '21 km terrain position', tolerance: 1000 },
  coldest_month_air_c: { label: 'Coldest-month air temperature', tolerance: 20 },
  warmest_month_air_c: { label: 'Warmest-month air temperature', tolerance: 20 },
  annual_precip_mm: { label: 'Annual precipitation', tolerance: 500 },
  bare_fraction: { label: 'Bare ground', tolerance: 0.25 },
  permanent_snow_ice_fraction: { label: 'Permanent snow / ice', tolerance: 0.25 },
  permafrost_fraction: { label: 'Permafrost', tolerance: 0.25 },
  dominant_lithology_code: { label: 'Lithology code', tolerance: 1 },
  surface_day_c: { label: 'Mean daytime ground temperature', tolerance: 30 },
  surface_night_c: { label: 'Mean nighttime ground temperature', tolerance: 30 },
  surface_mean_c: { label: 'Mean paired ground temperature', tolerance: 30 },
  surface_low_c: { label: 'Coldest observed nighttime composite', tolerance: 30 },
  surface_high_c: { label: 'Hottest observed daytime composite', tolerance: 30 },
  surface_range_c: { label: 'Observed ground-temperature range', tolerance: 40 },
}

type MeasuredFieldMapping = Readonly<{
  key: string
  variableId: string
  unit: string
  toleranceFloor: number
  toleranceFraction: number
}>

const MEASURED_FIELD_MAPPINGS: readonly MeasuredFieldMapping[] = [
  { key: 'slope_mean_deg', variableId: 'slope_1km', unit: 'degree', toleranceFloor: 2, toleranceFraction: 0.35 },
  { key: 'relief_5km_m', variableId: 'relief_5km', unit: 'm', toleranceFloor: 50, toleranceFraction: 0.35 },
  { key: 'relief_21km_m', variableId: 'relief_21km', unit: 'm', toleranceFloor: 150, toleranceFraction: 0.35 },
  { key: 'tpi_5km_m', variableId: 'tpi_5km', unit: 'm', toleranceFloor: 25, toleranceFraction: 0.5 },
  { key: 'tpi_21km_m', variableId: 'tpi_21km', unit: 'm', toleranceFloor: 75, toleranceFraction: 0.5 },
]

export async function loadDatasource(signal: AbortSignal) {
  const response = await earthIndexClient.metadata({}, { signal, timeoutMs: 120000 })
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
      const defaults = FIELD_DEFAULTS[column.name]
      const quality = qualityColumn(column.name)
      return {
        ...prior,
        key: column.name,
        label: defaults?.label ?? column.name.replaceAll('_', ' '),
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
        tolerance: prior?.tolerance ?? defaults?.tolerance ?? 1,
        weight: prior?.weight ?? 1,
        min: prior?.min ?? null,
        max: prior?.max ?? null,
        categorical: column.unit === 'category_code',
        qualityKey: quality && names.has(quality) ? quality : null,
      }
    })
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
  for (const { key, variableId, unit, toleranceFloor, toleranceFraction } of MEASURED_FIELD_MAPPINGS) {
    const criterion = next.find(item => item.key === key)
    if (!criterion) continue
    criterion.target = numericMeasurement(sample.measurements, `${body}.${variableId}`, unit)
    criterion.kind = 'measured'
    criterion.origin = 'Planet data sampled at roughly 1 km.'
    criterion.enabled = criterion.target != null && (!criterion.selectionTouched || criterion.selectedByUser === true)
    if (!criterion.toleranceTouched) {
      criterion.tolerance = Math.max(toleranceFloor, Math.abs(criterion.target ?? 0) * toleranceFraction)
    }
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
  const thermalMappings: readonly Readonly<{ key: string; variableId: string }>[] = [
    { key: 'surface_low_c', variableId: body === 'moon' ? 'surface_min' : 'modeled_surface_temperature_min' },
    { key: 'surface_high_c', variableId: body === 'moon' ? 'surface_max' : 'modeled_surface_temperature_max' },
    { key: 'surface_mean_c', variableId: 'modeled_surface_temperature_mean' },
    { key: 'surface_day_c', variableId: 'surface_noon' },
    { key: 'surface_night_c', variableId: 'surface_midnight' },
  ]
  for (const { key, variableId } of thermalMappings) {
    const criterion = next.find(item => item.key === key)
    if (!criterion) continue
    criterion.target = temperature(sample.measurements, body, variableId)
    if (key === 'surface_mean_c' && body !== 'mars') criterion.target = null
    criterion.enabled = criterion.target != null && (!criterion.selectionTouched || criterion.selectedByUser === true)
    if (!criterion.toleranceTouched) criterion.tolerance = 30
    if (!criterion.weightTouched) criterion.weight = 0.5
    criterion.kind = 'approximate'
    criterion.origin = 'Approximate surface-temperature comparison.'
  }
  return next
}
