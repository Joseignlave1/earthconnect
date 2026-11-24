import type { FieldEncoding, ScoreManifest } from './manifest.ts'

const scoreFields = new Set([
  'annual_precip_mm',
  'bare_fraction',
  'coldest_month_air_c',
  'dominant_lithology_code',
  'elevation_median_m',
  'permafrost_fraction',
  'permanent_snow_ice_fraction',
  'relief_21km_m',
  'relief_5km_m',
  'slope_mean_deg',
  'surface_day_c',
  'surface_high_c',
  'surface_low_c',
  'surface_mean_c',
  'surface_night_c',
  'surface_range_c',
  'tpi_21km_m',
  'tpi_5km_m',
  'warmest_month_air_c',
])
const coverageFields = new Set([
  'landcover_valid_fraction',
  'permafrost_valid_fraction',
  'terrain_valid_fraction',
  'surface_day_valid_fraction',
  'surface_night_valid_fraction',
  'surface_paired_valid_fraction',
])
const minimumCoverage = 0.9
const firstCode = 1
const lastCode = 65535
const numericTerm = lastCode + 1

/** The scoring properties of the explorer's Criterion. Sample metadata can also be present. */
export type EarthQueryCriterion = Readonly<{
  key: string
  label: string
  group: string
  enabled: boolean
  target: number | null
  tolerance: number
  weight: number
  min: number | null
  max: number | null
  categorical?: boolean
  qualityKey?: string | null
}>

declare const earthQueryBrand: unique symbol
export type EarthQuery = Readonly<{ [earthQueryBrand]: true }>

type SoftTerm = { target: number; tolerance: number; weight: number; categorical: boolean; group: string }
type QueryTerm = { field: string; min: number | null; max: number | null; soft?: SoftTerm }
type Family = { sum: number; max: number }
type QueryPlan = { coverage: Set<string>; terms: QueryTerm[]; families: Map<string, Family>; totalWeight: number }
const plans = new WeakMap<EarthQuery, QueryPlan>()

/** Snapshot a set of sampled or manually entered criteria for dispatchQuery. */
export function buildEarthQuery(criteria: readonly EarthQueryCriterion[]): EarthQuery {
  const coverage = new Set<string>()
  const terms: QueryTerm[] = []
  const families = new Map<string, Family>()
  for (const criterion of criteria) {
    const { enabled, key, label, qualityKey } = criterion
    if (!enabled && criterion.min == null && criterion.max == null) continue
    if (!scoreFields.has(key)) throw new Error(`Unsupported Earth Score field: ${key}`)
    if (qualityKey != null && qualityKey !== '') {
      if (!coverageFields.has(qualityKey)) throw new Error(`Unsupported Earth Score coverage field: ${qualityKey}`)
      coverage.add(qualityKey)
    }
    const number = (value: number | null) => {
      if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1e12) {
        throw new Error(`${label}: enter a finite number within ±1e12`)
      }
      return value
    }
    const min = criterion.min == null ? null : number(criterion.min)
    const max = criterion.max == null ? null : number(criterion.max)
    if (min != null && max != null && min > max) throw new Error(`${label}: minimum exceeds maximum`)
    const term: QueryTerm = { field: key, min, max }
    terms.push(term)
    if (!enabled) continue

    const { group, weight } = criterion
    if (typeof weight !== 'number' || !(weight > 0) || weight > 1e6) {
      throw new Error(`${label}: weight must be positive and at most 1e6`)
    }
    const categorical = criterion.categorical === true
    if (!categorical && criterion.tolerance < 1e-6) {
      throw new Error(`${label}: tolerance must be at least 0.000001`)
    }
    term.soft = {
      target: number(criterion.target),
      tolerance: categorical ? 1 : number(criterion.tolerance),
      weight,
      categorical,
      group,
    }
    const family = families.get(group) ?? { sum: 0, max: 0 }
    family.sum += weight
    family.max = Math.max(family.max, weight)
    families.set(group, family)
  }
  if (!families.size) throw new Error('Select at least one variable with a sampled target.')
  const totalWeight = [...families.values()].reduce((total, family) => total + family.max, 0)
  const query = Object.freeze({}) as EarthQuery
  plans.set(query, { coverage, terms, families, totalWeight })
  return query
}

// Decode in JS double precision when resolving cutoffs. Searching the result of
// the division keeps comparisons inclusive even when value * denominator
// would round across an integer.
function codeBound(encoding: FieldEncoding, value: number, exclusive = false) {
  let lo = firstCode
  let hi = lastCode + 1
  while (lo < hi) {
    const middle = (lo + hi) >>> 1
    const physical = (middle - encoding.origin) / encoding.denominator
    if (physical < value || (exclusive && physical === value)) lo = middle + 1
    else hi = middle
  }
  return lo
}

const headerWords = 4 + 4 * coverageFields.size
const termWords = 8

export function createQueryParameters(manifest: ScoreManifest, query: EarthQuery, maxBytes: number) {
  const plan = plans.get(query)
  if (!plan) throw new Error('Expected a query from buildEarthQuery')
  const { terms, coverage, families, totalWeight } = plan
  const size = headerWords + termWords * terms.length
  if (size * 4 > maxBytes) throw new Error('Earth Score query exceeds the device storage limit')
  const words = new Uint32Array(size)
  const floats = new Float32Array(words.buffer)
  const fieldIndex = (key: string) => {
    const index = manifest.columns.indexOf(key)
    if (index === -1) throw new Error(`Earth Score data does not contain ${key}`)
    return index
  }
  words[0] = terms.length
  words[1] = coverage.size
  for (const [i, key] of [...coverage].entries()) {
    const field = fieldIndex(key)
    words.set([field, codeBound(manifest.encodings[field], minimumCoverage)], 4 + 4 * i)
  }
  for (const [i, { field, min, max, soft }] of terms.entries()) {
    const index = fieldIndex(field)
    const encoding = manifest.encodings[index]
    const start = headerWords + termWords * i
    words.set(
      [
        index,
        min == null ? firstCode : codeBound(encoding, min),
        max == null ? lastCode : codeBound(encoding, max, true) - 1,
        numericTerm,
      ],
      start,
    )
    if (!soft) continue
    const family = families.get(soft.group)!
    // Apply the square root of the normalized weight before squaring the loss.
    // This retains the effect of small weights on large distances in float32.
    const factor = (Math.sqrt(soft.weight) / Math.sqrt(family.sum)) * (Math.sqrt(family.max) / Math.sqrt(totalWeight))
    const { origin, denominator } = encoding
    if (soft.categorical) {
      const code = codeBound(encoding, soft.target)
      words[start + 3] = code <= lastCode && (code - origin) / denominator === soft.target ? code : 0
      floats[start + 6] = 5 * factor
    } else {
      // Center the calculation near the target code before converting to f32.
      // Subtracting two physical f32 values can lose the entire distance at
      // narrow tolerances, even for samples well inside the encoding's range.
      const pivot = Math.min(lastCode, Math.max(firstCode, Math.round(soft.target * denominator + origin)))
      const offset = ((pivot - origin) / denominator - soft.target) * denominator
      floats.set([pivot, offset, factor / (denominator * soft.tolerance)], start + 4)
    }
  }
  return words
}

/** Lower the same validated, encoded plan used by dispatchQuery to editable WGSL. */
export function createQueryShaderSource(manifest: ScoreManifest, query: EarthQuery, maxBytes: number) {
  const words = createQueryParameters(manifest, query, maxBytes)
  const floats = new Float32Array(words.buffer)
  const plan = plans.get(query)!
  const literal = (value: number) => {
    const text = value.toString()
    return /[.e]/.test(text) ? text : `${text}.0`
  }
  const code = (field: number) => {
    const { origin, denominator } = manifest.encodings[field]
    return `measurement_code(cell.${manifest.columns[field]}, ${literal(origin)}, ${literal(denominator)})`
  }
  const coverage = [...plan.coverage].map((field, i) => {
    const start = 4 + i * 4
    return `  // Require at least 90% ${field.replace('_valid_fraction', '').replaceAll('_', ' ')} coverage.
  if (${code(words[start])} < ${words[start + 1]}u) { return NO_SCORE; }`
  })
  const terms = plan.terms.map((term, i) => {
    const start = headerWords + i * termWords
    return {
      name: `code_${i + 1}`,
      term,
      decode: code(words[start]),
      min: words[start + 1],
      max: words[start + 2],
      category: words[start + 3],
      pivot: floats[start + 4],
      offset: floats[start + 5],
      factor: floats[start + 6],
    }
  })
  const required = terms.map(
    ({ name, term, decode }) =>
      `  // ${term.field}
  let ${name} = ${decode};
  if (${name} == 0u) { return NO_SCORE; }`,
  )
  const limits = terms.flatMap(({ name, term, min, max }) => {
    if (term.min == null && term.max == null) return []
    const tests = [term.min != null && `${name} < ${min}u`, term.max != null && `${name} > ${max}u`].filter(Boolean)
    const bounds = [term.min != null && `minimum ${term.min}`, term.max != null && `maximum ${term.max}`].filter(
      Boolean,
    )
    return [
      `  // ${term.field}: ${bounds.join(', ')} in physical units.
  if (${tests.join(' || ')}) { return 0.0; }`,
    ]
  })
  const termsCode = terms.flatMap(({ name, term, category, pivot, offset, factor }, i) => {
    const { soft, field } = term
    if (!soft || factor === 0) return []
    const distance = `distance_${i + 1}`
    if (category !== numericTerm)
      return [
        `  // ${field}: class ${soft.target}, weight ${soft.weight}.
  let ${distance} = select(${literal(factor)}, 0.0, ${name} == ${category}u);
  loss += ${distance} * ${distance};`,
      ]
    return [
      `  // ${field}: target ${soft.target}, tolerance ${soft.tolerance}, weight ${soft.weight}.
  let ${distance} = (f32(i32(${name}) - ${pivot}i) + ${literal(offset)}) * ${literal(factor)};
  loss += ${distance} * ${distance};`,
    ]
  })
  return `// Sampled Earth analogue
// Work in measurement codes so cutoffs and narrow tolerances retain source precision.
fn measurement_code(value: f32, origin: f32, denominator: f32) -> u32 {
  if (!has_data(value)) { return 0u; }
  return u32(round(value * denominator + origin));
}

fn score(cell: EarthCell) -> f32 {
${coverage.concat(required).join('\n')}
${limits.length ? `\n${limits.join('\n')}\n` : ''}
  // The distance multipliers include the normalized weights of the criteria families.
  var loss = 0.0;
${termsCode.join('\n')}
  return exp(-0.5 * min(1000.0, loss));
}`
}
