import { GRID_WIDTH, PIXELS_PER_SITE } from './layout.ts'
import {
  PLANET_NO_DATA,
  PLANET_SHADER_DECLARATIONS,
  PLANET_SHADER_TYPES,
  lowerPlanetaryCalls,
  planetShaderFunctions,
  type PlanetaryShaderData,
} from './planetShaderSchema.ts'
import { TILE_SIZE, WORKGROUP_SIZE, type ScoreManifest } from './manifest.ts'
import { SCORE_MAX, SCORE_MISSING, scoreLookupLayout, sourceCodeWGSL } from './scoring.ts'

export type ScoreShaderFieldDescriptor = Readonly<{
  name: string
  type: 'f32'
  kind: 'measurement' | 'coverage'
  unit: string
  description: string
}>

export type EarthCellFieldDescriptor = Readonly<{
  name: string
  type: 'f32' | 'u32'
  kind: 'measurement' | 'coverage' | 'coordinate'
  unit: string
  description: string
}>

export const SCORE_SHADER_FIELDS = [
  { name: 'elevation_median_m', type: 'f32', kind: 'measurement', unit: 'm', description: 'Median terrain elevation' },
  { name: 'slope_mean_deg', type: 'f32', kind: 'measurement', unit: '°', description: 'Mean terrain slope' },
  {
    name: 'relief_5km_m',
    type: 'f32',
    kind: 'measurement',
    unit: 'm',
    description: 'Elevation range across a 5 km neighborhood',
  },
  {
    name: 'relief_21km_m',
    type: 'f32',
    kind: 'measurement',
    unit: 'm',
    description: 'Elevation range across a 21 km neighborhood',
  },
  {
    name: 'tpi_5km_m',
    type: 'f32',
    kind: 'measurement',
    unit: 'm',
    description: 'Elevation relative to the surrounding 5 km terrain',
  },
  {
    name: 'tpi_21km_m',
    type: 'f32',
    kind: 'measurement',
    unit: 'm',
    description: 'Elevation relative to the surrounding 21 km terrain',
  },
  {
    name: 'annual_precip_mm',
    type: 'f32',
    kind: 'measurement',
    unit: 'mm/year',
    description: 'Annual total precipitation',
  },
  {
    name: 'coldest_month_air_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Mean air temperature in the coldest month',
  },
  {
    name: 'warmest_month_air_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Mean air temperature in the warmest month',
  },
  {
    name: 'bare_fraction',
    type: 'f32',
    kind: 'measurement',
    unit: '0–1',
    description: 'Fraction of the cell with bare ground',
  },
  {
    name: 'permanent_snow_ice_fraction',
    type: 'f32',
    kind: 'measurement',
    unit: '0–1',
    description: 'Fraction with permanent snow or ice',
  },
  {
    name: 'permafrost_fraction',
    type: 'f32',
    kind: 'measurement',
    unit: '0–1',
    description: 'Fraction with mapped permafrost',
  },
  {
    name: 'dominant_lithology_code',
    type: 'f32',
    kind: 'measurement',
    unit: 'category code',
    description: 'Dominant lithology class. Compare codes for equality.',
  },
  {
    name: 'surface_day_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Daytime land surface temperature',
  },
  {
    name: 'surface_night_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Nighttime land surface temperature',
  },
  {
    name: 'surface_mean_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Mean land surface temperature',
  },
  {
    name: 'surface_low_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Coldest land surface temperature composite',
  },
  {
    name: 'surface_high_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Hottest land surface temperature composite',
  },
  {
    name: 'surface_range_c',
    type: 'f32',
    kind: 'measurement',
    unit: '°C',
    description: 'Land surface temperature range',
  },
  {
    name: 'landcover_valid_fraction',
    type: 'f32',
    kind: 'coverage',
    unit: '0–1',
    description: 'Valid land-cover observations',
  },
  {
    name: 'terrain_valid_fraction',
    type: 'f32',
    kind: 'coverage',
    unit: '0–1',
    description: 'Valid terrain observations',
  },
  {
    name: 'permafrost_valid_fraction',
    type: 'f32',
    kind: 'coverage',
    unit: '0–1',
    description: 'Valid permafrost observations',
  },
  {
    name: 'surface_day_valid_fraction',
    type: 'f32',
    kind: 'coverage',
    unit: '0–1',
    description: 'Valid daytime surface temperature observations',
  },
  {
    name: 'surface_night_valid_fraction',
    type: 'f32',
    kind: 'coverage',
    unit: '0–1',
    description: 'Valid nighttime surface temperature observations',
  },
  {
    name: 'surface_paired_valid_fraction',
    type: 'f32',
    kind: 'coverage',
    unit: '0–1',
    description: 'Valid paired daytime and nighttime observations',
  },
] as const satisfies readonly ScoreShaderFieldDescriptor[]

export const SCORE_MEASUREMENT_FIELDS: ReadonlySet<string> = new Set(
  SCORE_SHADER_FIELDS.filter(field => field.kind === 'measurement').map(field => field.name),
)

export const SCORE_COVERAGE_FIELDS: ReadonlySet<string> = new Set(
  SCORE_SHADER_FIELDS.filter(field => field.kind === 'coverage').map(field => field.name),
)

const coordinateFields = [
  {
    name: 'latitude_deg',
    type: 'f32',
    kind: 'coordinate',
    unit: '°',
    description: 'Cell-center latitude, positive north. NO_DATA outside the Earth footprint.',
  },
  {
    name: 'longitude_deg',
    type: 'f32',
    kind: 'coordinate',
    unit: '°',
    description: 'Cell-center longitude, positive east. NO_DATA outside the Earth footprint.',
  },
  {
    name: 'grid_x',
    type: 'u32',
    kind: 'coordinate',
    unit: 'column',
    description: 'Zero-based column of the full sinusoidal raster, increasing rightward.',
  },
  {
    name: 'grid_y',
    type: 'u32',
    kind: 'coordinate',
    unit: 'row',
    description: 'Zero-based row of the full sinusoidal raster, increasing southward.',
  },
] as const satisfies readonly EarthCellFieldDescriptor[]

export const EARTH_CELL_FIELDS: readonly EarthCellFieldDescriptor[] = [...SCORE_SHADER_FIELDS, ...coordinateFields]

const earthScorePrelude = /* wgsl */ `struct EarthCell {
${EARTH_CELL_FIELDS.map(({ name, type, unit, description }) => `  // ${description} (${unit})\n  ${name}: ${type},`).join('\n')}
}
const NO_DATA = ${PLANET_NO_DATA.toExponential()};
const NO_SCORE = -1.0;
fn has_data(value: f32) -> bool {
  return value > NO_DATA;
}
`

const scorePrelude = earthScorePrelude + '\n' + PLANET_SHADER_TYPES

// The editor resolves these signatures without binding a GPU buffer.
export const SCORE_SHADER_PRELUDE = scorePrelude + PLANET_SHADER_DECLARATIONS

export const DEFAULT_SCORE_SHADER = /* wgsl */ `// High, dry mountain basins
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 ||
      !has_data(cell.elevation_median_m) ||
      !has_data(cell.slope_mean_deg) ||
      !has_data(cell.relief_21km_m) ||
      !has_data(cell.tpi_21km_m) ||
      !has_data(cell.annual_precip_mm) ||
      !has_data(cell.coldest_month_air_c) ||
      !has_data(cell.warmest_month_air_c)) {
    return NO_SCORE;
  }

  let terrain = vec4f(
    (cell.elevation_median_m - 3000.0) / 1200.0,
    max(0.0, cell.slope_mean_deg - 4.0) / 3.0,
    (cell.relief_21km_m - 1500.0) / 700.0,
    (cell.tpi_21km_m + 200.0) / 300.0,
  );
  let rain = max(0.0, cell.annual_precip_mm - 250.0) / 250.0;
  let air = vec2f(
    (cell.coldest_month_air_c + 10.0) / 12.0,
    (cell.warmest_month_air_c - 15.0) / 10.0,
  );
  let loss = (dot(terrain, terrain) + rain * rain + 0.5 * dot(air, air)) / 6.0;
  return exp(-0.5 * min(1000.0, loss));
}`

export const SCORE_SHADER_EXAMPLES = [
  { name: 'High, dry mountain basins', source: DEFAULT_SCORE_SHADER },
  {
    name: 'Flat, bare terrain',
    source: /* wgsl */ `fn score(cell: EarthCell) -> f32 {
  if (!has_data(cell.bare_fraction) || !has_data(cell.slope_mean_deg)) {
    return NO_SCORE;
  }
  return cell.bare_fraction * exp(-cell.slope_mean_deg / 8.0);
}`,
  },
  {
    name: 'Cold, rocky uplands',
    source: /* wgsl */ `fn score(cell: EarthCell) -> f32 {
  if (!has_data(cell.bare_fraction) || !has_data(cell.elevation_median_m) ||
      !has_data(cell.coldest_month_air_c)) {
    return NO_SCORE;
  }
  let cold = 1.0 - smoothstep(-25.0, 5.0, cell.coldest_month_air_c);
  let highland = smoothstep(500.0, 2500.0, cell.elevation_median_m);
  return cold * highland * cell.bare_fraction;
}`,
  },
] as const

export type ScoreShaderDiagnostic = Readonly<{
  line: number
  column: number
  length: number
  message: string
  severity: 'error' | 'warning' | 'info'
}>

export class ScoreShaderError extends Error {
  readonly diagnostics: readonly ScoreShaderDiagnostic[]

  constructor(diagnostics: readonly ScoreShaderDiagnostic[]) {
    super(
      diagnostics.find(diagnostic => diagnostic.severity === 'error')?.message ?? 'Could not compile the score shader.',
    )
    this.name = 'ScoreShaderError'
    this.diagnostics = diagnostics
  }
}

function sourceLocation(source: string, offset: number, length: number) {
  const preceding = source.slice(0, offset)
  return { line: preceding.split('\n').length, column: preceding.length - preceding.lastIndexOf('\n'), length }
}

function inspectSource(source: string) {
  const lex = /\s+|\/\/[^\n]*|\/\*|[\p{ID_Start}_][\p{ID_Continue}_]*|[^]/gu
  let previous = ''
  let scoreLocation: ReturnType<typeof sourceLocation> | undefined
  const identifiers = new Set<string>()
  for (let match = lex.exec(source); match; match = lex.exec(source)) {
    const text = match[0]
    if (text.startsWith('//') || /^\s/u.test(text)) continue
    if (text === '/*') {
      const comments = /\/\*|\*\//g
      comments.lastIndex = lex.lastIndex
      let depth = 1
      for (let next = comments.exec(source); next; next = comments.exec(source)) {
        depth += next[0] === '/*' ? 1 : -1
        if (!depth) break
      }
      if (depth) break // Report the syntax error through the WGSL compiler.
      lex.lastIndex = comments.lastIndex
      continue
    }
    if (previous === '@' && (text === 'group' || text === 'binding')) {
      throw new ScoreShaderError([
        {
          ...sourceLocation(source, match.index, text.length),
          message:
            'GPU resource bindings are supplied by the score engine. Use the EarthCell argument to access the data.',
          severity: 'error',
        },
      ])
    }
    if (previous === 'fn' && text === 'score') scoreLocation = sourceLocation(source, match.index, text.length)
    if (text.length > 1 || /[\p{ID_Start}_]/u.test(text)) identifiers.add(text)
    previous = text
  }
  return { scoreLocation, identifiers }
}

// Let the WGSL compiler validate the editable module. The engine adds the
// one entry point it invokes and chooses private identifiers absent from the
// submitted code. User functions, constants, and locals can use any name.
export function prepareScoreShader(source: string, manifest: ScoreManifest, planetary?: PlanetaryShaderData) {
  const lowered = lowerPlanetaryCalls(source, planetary)
  source = lowered.source.replace(/\r\n?/g, '\n')
  const { scoreLocation, identifiers } = inspectSource(source)
  const taken = new Set(Array.from(source.matchAll(/earth_score_internal_\d+_/g), ([match]) => match))
  let index = 0
  while (taken.has(`earth_score_internal_${index}_`)) index++
  const internal = `earth_score_internal_${index}_`
  let readsColumns = false
  const fields = SCORE_SHADER_FIELDS.map(({ name }) => {
    const column = manifest.columns.indexOf(name)
    if (column < 0 || !identifiers.has(name)) return 'NO_DATA'
    readsColumns = true
    const { origin, denominator } = manifest.encodings[column]
    return `${internal}value(rows, ${column}u, row, ${origin}i, ${denominator}.0)`
  })
  const needsGeo = identifiers.has('latitude_deg') || identifiers.has('longitude_deg')
  const needsGrid = needsGeo || identifiers.has('grid_x') || identifiers.has('grid_y')
  const engineCode = readsColumns
    ? sourceCodeWGSL
        .replaceAll('sourceCode', `${internal}source_code`)
        .replaceAll('packedInput', `${internal}packed_input`) +
      `
fn ${internal}value(rows: u32, field: u32, row: u32, origin: i32, denominator: f32) -> f32 {
  let code = ${internal}source_code(rows, field, row);
  if (code == 0u) { return NO_DATA; }
  return f32(i32(code) - origin) / denominator;
}`
    : ''
  const lookupLayout = scoreLookupLayout(manifest)
  const gridCode = needsGrid
    ? `
  let site = ${internal}first_site + row / ${PIXELS_PER_SITE}u;
  let pixel = row % ${PIXELS_PER_SITE}u;
  let origin = ${internal}lookup[${lookupLayout.siteOriginsOffset}u + site];
  let grid_x = origin % ${GRID_WIDTH}u + pixel % ${TILE_SIZE}u;
  let grid_y = origin / ${GRID_WIDTH}u + pixel / ${TILE_SIZE}u;`
    : ''
  const geoCode = needsGeo
    ? `
  let location = ${lookupLayout.rowGeometryOffset}u + grid_y * ${lookupLayout.rowStride}u;
  let x_offset = f32(grid_x) + 0.5 - ${GRID_WIDTH / 2}.0;
  let inside = abs(x_offset) < f32(${internal}lookup[location + ${lookupLayout.validHalfWidthOffset}u]);
  let latitude = select(NO_DATA, bitcast<f32>(${internal}lookup[location + ${lookupLayout.latitudeOffset}u]), inside);
  let longitude = select(NO_DATA, x_offset * bitcast<f32>(${internal}lookup[location + ${lookupLayout.lonScaleOffset}u]), inside);`
    : ''
  const bindings = [
    readsColumns && `@group(0) @binding(0) var<storage, read> ${internal}packed_input: array<u32>;`,
    `@group(0) @binding(1) var<storage, read_write> ${internal}scores: array<u32>;`,
    needsGrid && `@group(0) @binding(2) var<storage, read> ${internal}lookup: array<u32>;`,
    needsGrid && `@group(0) @binding(4) var<uniform> ${internal}first_site: u32;`,
  ]
    .filter(Boolean)
    .join('\n')
  const suffix = /* wgsl */ `
${bindings}
${engineCode}
${planetShaderFunctions(internal, lowered.slots)}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn ${internal}dispatch(@builtin(global_invocation_id) id: vec3u) {
  let rows = arrayLength(&${internal}scores);
  let row = id.x;
  if (row >= rows) { return; }${gridCode}${geoCode}
  let result: f32 = score(EarthCell(
    ${fields.join(',\n    ')},
    ${needsGeo ? 'latitude, longitude' : 'NO_DATA, NO_DATA'}, ${needsGrid ? 'grid_x, grid_y' : '0u, 0u'}
  ));
  let exponent = bitcast<u32>(result) & 0x7f800000u;
  if (exponent == 0x7f800000u || result < 0.0) {
    ${internal}scores[row] = ${SCORE_MISSING}u;
  } else {
    let scaled = clamp(result, 0.0, 1.0) * ${SCORE_MAX}.0;
    let whole = u32(floor(scaled));
    let fraction = scaled - f32(whole);
    let up = fraction > 0.5 || (fraction == 0.5 && (whole & 1u) != 0u);
    ${internal}scores[row] = whole + select(0u, 1u, up);
  }
}
`
  return {
    code: source + '\n' + scorePrelude + suffix,
    entryPoint: `${internal}dispatch`,
    sourceLines: source.split('\n').length,
    scoreLocation,
    planetaryAliases: [...lowered.slots.values()].map(({ alias, body }) => [alias, `${body}_point`] as const),
    callLine: (source + '\n' + scorePrelude + suffix.slice(0, suffix.indexOf('score(EarthCell('))).split('\n').length,
  }
}

export async function makeScorePipeline(
  device: GPUDevice,
  layout: GPUPipelineLayout,
  shader: ReturnType<typeof prepareScoreShader>,
) {
  device.pushErrorScope('validation')
  let module: GPUShaderModule
  let validation: Promise<GPUError | null>
  try {
    module = device.createShaderModule({ label: 'Custom Earth score function', code: shader.code })
  } finally {
    validation = device.popErrorScope()
    void validation.catch(() => {})
  }
  const info = module.getCompilationInfo()
  const pipeline = device
    .createComputePipelineAsync({
      label: 'Custom Earth score function',
      layout,
      compute: { module, entryPoint: shader.entryPoint },
    })
    .then(
      value => ({ ok: true as const, value }),
      error => ({ ok: false as const, error }),
    )
  const [compilation, fault, built] = await Promise.all([info, validation, pipeline])
  const diagnostics = compilation.messages.map((value): ScoreShaderDiagnostic => {
    const line = value.lineNum
    const signature = line === shader.callLine ? shader.scoreLocation : undefined
    const location =
      line >= 1 && line <= shader.sourceLines
        ? { line, column: value.linePos, length: value.length }
        : (signature ?? { line: 0, column: 0, length: 0 })
    const message = shader.planetaryAliases.reduce((text, [alias, name]) => text.replaceAll(alias, name), value.message)
    return { ...location, message, severity: value.type }
  })
  if (diagnostics.some(diagnostic => diagnostic.severity === 'error')) throw new ScoreShaderError(diagnostics)
  if (fault) throw new ScoreShaderError([{ line: 0, column: 0, length: 0, message: fault.message, severity: 'error' }])
  if (!built.ok) {
    throw new ScoreShaderError([
      {
        line: 0,
        column: 0,
        length: 0,
        message: built.error instanceof Error ? built.error.message : String(built.error),
        severity: 'error',
      },
    ])
  }
  return built.value
}
