import {
  createScoreGeometry,
  GRID_HEIGHT,
  SCORE_GEOMETRY_LATITUDE_OFFSET,
  SCORE_GEOMETRY_LON_SCALE_OFFSET,
  SCORE_GEOMETRY_ROW_STRIDE,
  SCORE_GEOMETRY_VALID_HALF_WIDTH_OFFSET,
} from './layout.ts'
import { BLOCK_SIZE, type ScoreManifest } from './manifest.ts'

export const SOURCE_MISSING = 0
export const SCORE_MISSING = 65535
export const SCORE_MAX = 65534
export const LOOKUP_CODES_PER_COLUMN = 65536

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

export function scoreLookupLayout(manifest: ScoreManifest) {
  const siteOriginsOffset = manifest.columns.length * LOOKUP_CODES_PER_COLUMN
  const rowGeometryOffset = siteOriginsOffset + manifest.sites.length
  const totalWords = rowGeometryOffset + GRID_HEIGHT * SCORE_GEOMETRY_ROW_STRIDE
  return {
    codesPerColumn: LOOKUP_CODES_PER_COLUMN,
    siteOriginsOffset,
    rowGeometryOffset,
    rowStride: SCORE_GEOMETRY_ROW_STRIDE,
    latitudeOffset: SCORE_GEOMETRY_LATITUDE_OFFSET,
    lonScaleOffset: SCORE_GEOMETRY_LON_SCALE_OFFSET,
    validHalfWidthOffset: SCORE_GEOMETRY_VALID_HALF_WIDTH_OFFSET,
    totalWords,
    totalBytes: totalWords * Uint32Array.BYTES_PER_ELEMENT,
  }
}

export function lookupBytes(manifest: ScoreManifest) {
  return scoreLookupLayout(manifest).totalBytes
}

export function createLookup(manifest: ScoreManifest) {
  const layout = scoreLookupLayout(manifest)
  const values = new Float32Array(layout.totalWords)
  const words = new Uint32Array(values.buffer)
  for (const [i, { origin, denominator }] of manifest.encodings.entries()) {
    const start = i * layout.codesPerColumn
    words[start] = 0x7fc00000
    for (let code = 1; code < layout.codesPerColumn; code++) values[start + code] = (code - origin) / denominator
  }
  words.set(createScoreGeometry(manifest), layout.siteOriginsOffset)
  return words
}
