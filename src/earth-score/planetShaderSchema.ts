// Numeric projection of the planetary SamplePoint catalogue. Keep this
// structural type independent of the transport so the standalone scorer can
// receive the same decoded measurements as the workspace.
export type PlanetaryShaderMeasurement = Readonly<{
  variableId: string
  unit: string
  status: number
  qualityFlags: readonly string[]
  decoded: unknown
}>

type Body = 'moon' | 'mars'

export type PlanetaryShaderSample = Readonly<{
  body: Body
  latitude: number
  longitude: number
  measurements: readonly PlanetaryShaderMeasurement[]
  partial?: boolean
}>

export const PLANET_NO_DATA = -1e20
const missing = Math.fround(PLANET_NO_DATA)
// MeasurementStatus.AVAILABLE in earth_index.v1.
const available = 1

type Field = { name: string; id: string; unit: string; description: string; units: readonly string[] }
type Group = { name: string; type: string; fields: readonly Field[] }

export type PlanetaryShaderCall = Readonly<{ body: Body; slot: number; offset: number; length: number }>
export type PlanetaryShaderData = Readonly<{
  calls: readonly PlanetaryShaderCall[]
  samples: readonly PlanetaryShaderSample[]
}>

function field(name: string, id: string, unit: string, description: string, units = [unit]): Field {
  return { name, id, unit, description, units }
}

const terrain = { name: 'terrain', type: 'PointTerrain', fields: [
  field('elevation_m', 'elevation', 'm', 'Elevation relative to the body’s reference datum'),
  field('slope_1km_deg', 'slope_1km', 'degrees', 'DEM slope on a 1 km sample grid', ['degree']),
  field('relief_5km_m', 'relief_5km', 'm', 'Elevation range in a 5 km window'),
  field('relief_21km_m', 'relief_21km', 'm', 'Elevation range in a 21 km window'),
  field('tpi_5km_m', 'tpi_5km', 'm', 'Elevation above the 5 km neighborhood mean'),
  field('tpi_21km_m', 'tpi_21km', 'm', 'Elevation above the 21 km neighborhood mean'),
] } satisfies Group

const moon = { type: 'MoonPoint', groups: [
  terrain,
  { name: 'temperature', type: 'MoonTemperature', fields: [
    field('noon_k', 'surface_noon', 'K', 'Diviner local-noon bolometric surface temperature'),
    field('midnight_k', 'surface_midnight', 'K', 'Diviner local-midnight bolometric surface temperature'),
    field('min_k', 'surface_min', 'K', 'Diviner minimum bolometric surface temperature'),
    field('max_k', 'surface_max', 'K', 'Diviner maximum bolometric surface temperature'),
    field('range_k', 'surface_temperature_range', 'K difference', 'Diviner maximum minus minimum', ['K_difference']),
  ] },
  { name: 'visibility', type: 'MoonVisibility', fields: [
    field('solar_illumination_fraction', 'solar_illumination', '0–1', 'Long-term modeled sunlight visibility in covered polar areas', ['fraction']),
    field('earth_visibility_fraction', 'earth_visibility', '0–1', 'Long-term modeled Earth visibility in covered polar areas', ['fraction']),
  ] },
  { name: 'composition', type: 'MoonComposition', fields: [
    field('feo_wt_percent', 'feo', 'wt% FeO', 'Kaguya modeled iron oxide abundance', ['wt_percent_FeO']),
    field('optical_maturity', 'optical_maturity', 'unitless', 'Kaguya optical maturity index', ['1']),
    field('plagioclase_grain_size_um', 'plagioclase_grain_size', 'µm', 'Kaguya spectral-model plagioclase grain size', ['micrometre']),
    field('thorium_ppm', 'thorium', 'ppm', 'Lunar Prospector thorium abundance'),
    field('titanium_wt_percent', 'titanium', 'wt% Ti', 'Lunar Prospector elemental titanium abundance', ['wt_percent_Ti']),
    field('christiansen_feature_um', 'christiansen_feature', 'µm', 'Diviner Christiansen-feature wavelength', ['micrometre']),
  ] },
  { name: 'radar', type: 'MoonRadar', fields: [
    field('cpr', 'radar_cpr', 'ratio', 'Mini-RF circular polarization ratio'),
    field('malapert_backscatter_db', 'radar_backscatter_malapert', 'dB', 'Regional incidence-normalized Malapert backscatter'),
  ] },
  { name: 'regional', type: 'MoonRegionalTerrain', fields: [
    field('rock_count_100m', 'regional_rock_count', 'count per 100 m cell', 'LRO NAC detected rocks in the covered site', ['count_per_100m_by_100m_cell']),
    field('slope_deg', 'regional_published_slope', 'degrees', 'Published slope in covered regions', ['degree']),
  ] },
] } satisfies { type: string; groups: Group[] }

const celsius = ['degree_C', 'degree_Celsius', 'degC']
const inertia = ['J_m-2_K-1_s-0.5']
const tesMinerals = ['olivine', 'feldspar', 'clinopyroxene', 'plagioclase', 'carbonate', 'hematite', 'amphibole', 'quartz', 'sulfate', 'dust'] as const

const mars = { type: 'MarsPoint', groups: [
  terrain,
  { name: 'temperature', type: 'MarsTemperature', fields: [
    field('mean_c', 'modeled_surface_temperature_mean', '°C', 'KRC modeled annual mean surface temperature', celsius),
    field('min_c', 'modeled_surface_temperature_min', '°C', 'KRC modeled annual minimum surface temperature', celsius),
    field('max_c', 'modeled_surface_temperature_max', '°C', 'KRC modeled annual maximum surface temperature', celsius),
    field('range_k', 'surface_temperature_range', 'K difference', 'KRC maximum minus minimum', ['K_difference']),
  ] },
  { name: 'surface', type: 'MarsSurface', fields: [
    field('albedo', 'albedo', '0–1', 'TES Lambert albedo', ['1']),
    field('dust_index', 'surface_dust_index', 'unitless', 'TES surface dust-cover index', ['1']),
    field('water_equivalent_hydrogen_wt_percent', 'water_equivalent_hydrogen', 'wt% WEH', 'Mars Odyssey neutron-derived hydrogen proxy', ['wt_percent_WEH']),
    field('thermal_inertia_tes', 'thermal_inertia_tes', 'J m⁻² K⁻¹ s⁻½', 'TES apparent thermal inertia', inertia),
    field('thermal_inertia_tes_night', 'thermal_inertia_tes_putzig_night', 'J m⁻² K⁻¹ s⁻½', 'TES Putzig nighttime apparent thermal inertia', inertia),
    field('thermal_inertia_themis', 'thermal_inertia_themis', 'J m⁻² K⁻¹ s⁻½', 'THEMIS apparent thermal inertia', inertia),
  ] },
  { name: 'minerals', type: 'MarsMinerals', fields: [
    ...tesMinerals.map(name => field(`${name}_fraction`, `tes_${name}`, '0–1', `TES spectral-model ${name} areal fraction`, ['areal_fraction'])),
    field('fit_error', 'tes_mineral_fit_error', 'source RMS error', 'TES spectral-model fit error', ['raw_fit_error']),
  ] },
] } satisfies { type: string; groups: Group[] }

const bodies = { moon, mars }
const types = [terrain, ...moon.groups.slice(1), ...mars.groups.slice(1)]
export const PLANET_SHADER_RESERVED_NAMES = new Set([
  ...Object.keys(bodies).map(body => `${body}_point`),
  ...Object.values(bodies).map(body => body.type),
  ...types.map(group => group.type),
])
const measurementStart = 3 // latitude, longitude, partial
const stride = measurementStart + Math.max(...Object.values(bodies).map(body => body.groups.reduce((n, group) => n + group.fields.length, 0)))

export function planetShaderBytes(sampleCount: number) {
  return Math.max(1, sampleCount * stride) * Float32Array.BYTES_PER_ELEMENT
}

export const PLANET_SHADER_TYPES = [
  '// Point terrain uses the sampled planetary DEM and its native datum.',
  ...types.map(({ type, fields }) => `struct ${type} {
${fields.map(({ name, unit, description }) => `  // ${description} (${unit}).
  ${name}: f32,`).join('\n')}
}`),
  ...Object.values(bodies).map(({ type, groups }) => `struct ${type} {
  latitude_deg: f32,
  longitude_deg: f32,
  // A sampled response may omit individual measurements. Use has_data.
  sampled: bool,
  partial: bool,
${groups.map(group => `  ${group.name}: ${group.type},`).join('\n')}
}`),
].join('\n\n') + '\n'

function wgslGroups(body: Body, fieldValue: (field: Field, offset: number) => string) {
  let offset = 0
  return bodies[body].groups.map(group =>
    `${group.type}(${group.fields.map(field => fieldValue(field, offset++)).join(', ')})`)
}

export const PLANET_SHADER_DECLARATIONS = (Object.keys(bodies) as Body[]).map(body => `
// Samples a fixed latitude and east-positive longitude in degrees.
fn ${body}_point(latitude: f32, longitude: f32) -> ${bodies[body].type} {
  return ${bodies[body].type}(latitude, longitude, false, false, ${wgslGroups(body, () => 'NO_DATA').join(', ')});
}
`).join('\n')

type PlanetarySlot = { body: Body; alias: string; slot: number }

export function lowerPlanetaryCalls(source: string, data?: PlanetaryShaderData) {
  const slots = new Map<number, PlanetarySlot>()
  if (!data?.calls.length) return { source, slots }
  const parts = []
  const calls = [...data.calls].sort((a, b) => a.offset - b.offset)
  let cursor = 0, sequence = 0
  for (const call of calls) {
    const name = `${call.body}_point`
    const end = call.offset + call.length
    if (!Number.isInteger(call.offset) || !Number.isInteger(call.length) ||
      call.offset < cursor || call.length <= name.length || end > source.length ||
      source.slice(call.offset, call.offset + name.length) !== name ||
      source[end - 1] !== ')' || data.samples[call.slot]?.body !== call.body) {
      throw new Error('Planetary point calls are out of sync with the WGSL source or sample data.')
    }
    let slot = slots.get(call.slot)
    if (!slot) {
      let alias: string
      do { alias = `p${(sequence++).toString(36).padStart(9, '0')}` } while (source.includes(alias))
      slot = { body: call.body, alias, slot: call.slot }
      slots.set(call.slot, slot)
    }

    // Replace only the callee using exactly ten ASCII code units. Arguments,
    // Unicode, and line breaks retain their original compiler positions.
    parts.push(source.slice(cursor, call.offset), slot.alias)
    cursor = call.offset + name.length
  }
  parts.push(source.slice(cursor))
  return { source: parts.join(''), slots }
}

export function planetShaderFunctions(internal: string, slots: ReadonlyMap<number, PlanetarySlot>) {
  if (!slots.size) return ''
  const buffer = `${internal}points`
  return /* wgsl */ `
@group(0) @binding(3) var<storage, read> ${buffer}: array<f32>;

${[...slots.values()].map(({ body, alias, slot }) => `fn ${alias}(latitude: f32, longitude: f32) -> ${bodies[body].type} {
  let row = ${slot * stride}u;
  return ${bodies[body].type}(
    ${buffer}[row], ${buffer}[row + 1u], true, ${buffer}[row + 2u] != 0.0,
    ${wgslGroups(body, (_, index) => `${buffer}[row + ${measurementStart + index}u]`).join(',\n    ')}
  );
}`).join('\n\n')}
`
}

function normalizedPoint(sample: PlanetaryShaderSample) {
  if (!Object.hasOwn(bodies, sample.body)) throw new Error('Unknown body for a WGSL planetary sample.')
  if (!(Math.abs(sample.latitude) <= 90 && Math.abs(sample.longitude) <= 180)) {
    throw new Error('Planetary coordinates must be finite and within ±90° latitude and ±180° longitude.')
  }
  const latitude = Math.fround(sample.latitude) || 0
  const lon = Math.fround(sample.longitude) || 0
  const longitude = Math.abs(latitude) === 90 ? 0 : lon === 180 ? -180 : lon
  const schema = bodies[sample.body]
  const values = schema.groups.flatMap(group => group.fields.map(({ id, units }) => {
    const measurement = sample.measurements.find(value =>
      value.variableId === `${sample.body}.${id}` &&
      value.status === available && units.includes(value.unit) &&
      !value.qualityFlags.includes('semantic_review_required') &&
      typeof value.decoded === 'number' && Number.isFinite(Math.fround(value.decoded)) && value.decoded > missing)
    return measurement ? Math.fround(measurement.decoded as number) : missing
  }))
  return { schema, latitude, longitude, partial: sample.partial === true, values }
}

export function packPlanetaryShaderPoints(samples: readonly PlanetaryShaderSample[]) {
  const packed = new Float32Array(Math.max(1, samples.length * stride))
  for (const [slot, sample] of samples.entries()) {
    const { latitude, longitude, partial, values } = normalizedPoint(sample)
    const row = slot * stride
    packed.set([latitude, longitude, Number(partial)], row)
    packed.set(values, row + measurementStart)
  }
  return packed
}

/** Create a typed, self-contained WGSL expression that captures a live point. */
export function freezePlanetaryShaderPoint(sample: PlanetaryShaderSample) {
  const point = normalizedPoint(sample)
  const literal = (value: number) => {
    if (value === missing) return 'NO_DATA'
    const text = value.toString()
    return /[.e]/.test(text) ? text : `${text}.0`
  }
  let index = 0
  const groups = point.schema.groups.map(group => `  ${group.type}(
${group.fields.map(field => `    ${literal(point.values[index++])}, // ${field.name}`).join('\n')}
  )`)
  return `${point.schema.type}(
  ${literal(point.latitude)}, ${literal(point.longitude)}, true, ${point.partial},
${groups.join(',\n')}
)`
}
