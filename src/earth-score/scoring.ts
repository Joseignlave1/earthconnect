import { createScoreGeometry, GRID_HEIGHT } from './layout.ts'
import { BLOCK_SIZE, COVERAGE_FIELDS, FEATURE_COUNT, GROUP_COUNT, WORKGROUP_SIZE, type ScoreManifest } from './manifest.ts'

export const SOURCE_MISSING = 0
export const SCORE_MISSING = 65535
export const SCORE_MAX = 65534
export const PARAMETER_BYTES = 16 + 16 * FEATURE_COUNT

export type ScoreOverrides = Readonly<Partial<Record<string, Readonly<{
  target?: number
  tolerance?: number
  weight?: number
}>>>>

// Can also be used by callers binding a chunk's packedInput as array<u32>.
export const sourceCodeWGSL = /* wgsl */ `
fn sourceCode(rows: u32, field: u32, row: u32) -> u32 {
  let blocks = (rows + ${BLOCK_SIZE - 1}u) / ${BLOCK_SIZE}u;
  let descriptor = 4u + 2u * (field * blocks + row / ${BLOCK_SIZE}u);
  let offset = packedInput[descriptor];
  let config = packedInput[descriptor + 1u];
  let base = config & 65535u;
  let bits = (config >> 16u) & 31u;
  if (bits == 0u) { return base; }
  let bit = (row % ${BLOCK_SIZE}u) * bits;
  let shift = bit % 32u;
  let word = offset + bit / 32u;
  var value = packedInput[word] >> shift;
  if (shift + bits > 32u) { value |= packedInput[word + 1u] << (32u - shift); }
  value &= (1u << bits) - 1u;
  if ((config & (1u << 21u)) != 0u) {
    if (value == 0u) { return 0u; }
    return base + value - 1u;
  }
  return base + value;
}
`

export function scoreShader(manifest: ScoreManifest) {
  const fieldIndex = (field: string) => manifest.columns.indexOf(field)
  const required = manifest.features.map(({ field }) => `sourceCode(p.rows, ${fieldIndex(field)}u, row) == 0u`)
  const coverage = COVERAGE_FIELDS.map(field => {
    const index = fieldIndex(field)
    return `sourceCode(p.rows, ${index}u, row) == 0u || physicalValue(${index}u, row) < 0.9`
  })
  const terms = manifest.features.map(({ field }, i) => `
    let d${i} = (physicalValue(${fieldIndex(field)}u, row) - p.terms[${i}].x) * p.terms[${i}].y;
    loss += p.terms[${i}].z * d${i} * d${i};`)

  return /* wgsl */ `
struct Parameters {
  rows: u32,
  terms: array<vec4f, ${FEATURE_COUNT}>,
}
@group(0) @binding(0) var<storage, read> packedInput: array<u32>;
@group(0) @binding(1) var<storage, read_write> scores: array<u32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read> lookup: array<u32>;
${sourceCodeWGSL}
fn physicalValue(field: u32, row: u32) -> f32 {
  return bitcast<f32>(lookup[field * 65536u + sourceCode(p.rows, field, row)]);
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn score(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  if (row >= p.rows) { return; }
  if (${required.concat(coverage).join(' ||\n      ')}) {
    scores[row] = ${SCORE_MISSING}u;
    return;
  }
  var loss = 0.0;
  ${terms.join('')}
  scores[row] = u32(round(clamp(exp(-0.5 * min(1000.0, loss)), 0.0, 1.0) * ${SCORE_MAX}.0));
}
`
}

export function lookupBytes(manifest: ScoreManifest) {
  return (manifest.columns.length * 65536 + manifest.sites.length + GRID_HEIGHT * 3) * 4
}

export function createLookup(manifest: ScoreManifest) {
  const values = new Float32Array(lookupBytes(manifest) / 4)
  const words = new Uint32Array(values.buffer)
  for (const [i, { origin, denominator }] of manifest.encodings.entries()) {
    const start = i * 65536
    words[start] = 0x7fc00000
    for (let code = 1; code < 65536; code++) values[start + code] = (code - origin) / denominator
  }
  words.set(createScoreGeometry(manifest), manifest.columns.length * 65536)
  return words
}

export function createParameters(manifest: ScoreManifest, overrides: ScoreOverrides = {}) {
  const f32 = Math.fround
  for (const [name, value] of Object.entries(overrides)) {
    if (!manifest.features.some(({ field }) => field === name)) throw new Error(`Unknown score feature: ${name}`)
    if (!value || typeof value !== 'object' || Object.entries(value).some(([key, number]) =>
      !['target', 'tolerance', 'weight'].includes(key) || !Number.isFinite(number))) {
      throw new Error(`Invalid override for ${name}`)
    }
  }
  const groupSum = Array<number>(GROUP_COUNT).fill(0)
  const groupMax = Array<number>(GROUP_COUNT).fill(0)
  const features = manifest.features.map(base => {
    const f = { ...base, ...overrides[base.field] }
    const target = f32(f.target), tolerance = f32(f.tolerance), weight = f32(f.weight)
    const inverseTolerance = f32(1 / tolerance)
    if (!Number.isFinite(target) || !Number.isFinite(tolerance) || tolerance <= 0 ||
        !Number.isFinite(inverseTolerance) || !Number.isFinite(weight) || weight <= 0) {
      throw new Error(`Invalid score parameters for ${base.field}`)
    }
    groupSum[f.group] = f32(groupSum[f.group] + weight)
    groupMax[f.group] = Math.max(groupMax[f.group], weight)
    return { target, inverseTolerance, weight, group: f.group }
  })
  const total = groupMax.reduce((sum, weight) => f32(sum + weight), 0)
  const floats = new Float32Array(PARAMETER_BYTES / 4)
  for (const [i, { target, inverseTolerance, weight, group }] of features.entries()) {
    // Match the spike's float32 operation order and family normalization.
    const normalized = f32(f32(f32(weight * groupMax[group]) / groupSum[group]) / total)
    if (!Number.isFinite(normalized) || normalized <= 0) throw new Error('Score weights exceed the supported float32 range')
    floats.set([target, inverseTolerance, normalized, 0], 4 + 4 * i)
  }
  return new Uint32Array(floats.buffer)
}
