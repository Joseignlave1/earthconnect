export const BLOCK_SIZE = 128
export const WORKGROUP_SIZE = 256
export const TILE_SIZE = 278
export const COVERAGE_FIELDS = ['landcover_valid_fraction', 'terrain_valid_fraction'] as const
export const FEATURE_COUNT = 9
export const GROUP_COUNT = 5
const maxChunkBytes = 128 * 1024 ** 2

export type FieldEncoding = Readonly<{ origin: number; denominator: number }>
export type ScoreFeature = Readonly<{
  field: string
  target: number
  tolerance: number
  weight: number
  group: number
}>

export type ChunkAsset = Readonly<{
  file: string
  rows: number
  packedBytes: number
  compressedBytes: number
}>

export type ScoreManifest = Readonly<{
  format: 'earth-score-v1'
  rows: number
  tileSize: typeof TILE_SIZE
  sites: readonly string[]
  columns: readonly string[]
  encodings: readonly FieldEncoding[]
  features: readonly ScoreFeature[]
  chunks: readonly ChunkAsset[]
}>

function integer(value: number, min: number, max: number) {
  return Number.isSafeInteger(value) && value >= min && value <= max
}

export function parseManifest(data: unknown): ScoreManifest {
  const m = data as ScoreManifest
  if (
    m?.format !== 'earth-score-v1' ||
    m.tileSize !== TILE_SIZE ||
    !Array.isArray(m.sites) ||
    !m.sites.length ||
    m.sites.length > 648 ||
    m.rows !== m.sites.length * TILE_SIZE ** 2
  ) {
    throw new Error('Invalid Earth Score manifest or tile dimensions')
  }
  if (
    m.sites.some(site => typeof site !== 'string' || !/^h(?:[0-2]\d|3[0-5])v(?:0\d|1[0-7])$/.test(site)) ||
    new Set(m.sites).size !== m.sites.length
  ) {
    throw new Error('Invalid or duplicate Earth Score site keys')
  }
  if (
    !Array.isArray(m.columns) ||
    !Array.isArray(m.encodings) ||
    m.columns.length !== m.encodings.length ||
    m.columns.length > 25 ||
    m.columns.some(field => typeof field !== 'string' || !/^[a-z][a-z0-9_]*$/.test(field)) ||
    new Set(m.columns).size !== m.columns.length ||
    m.encodings.some(e => !e || !integer(e.origin, -1_000_000, 1_000_000) || !integer(e.denominator, 1, 1_000_000))
  ) {
    throw new Error('Invalid Earth Score field encodings')
  }
  if (
    !Array.isArray(m.features) ||
    m.features.length !== FEATURE_COUNT ||
    m.features.some(
      f =>
        !f ||
        !m.columns.includes(f.field) ||
        !Number.isFinite(f.target) ||
        !Number.isFinite(f.tolerance) ||
        f.tolerance <= 0 ||
        !Number.isFinite(f.weight) ||
        f.weight <= 0 ||
        !integer(f.group, 0, GROUP_COUNT - 1),
    ) ||
    new Set(m.features.map(f => f.field)).size !== FEATURE_COUNT ||
    new Set(m.features.map(f => f.group)).size !== GROUP_COUNT ||
    COVERAGE_FIELDS.some(field => !m.columns.includes(field))
  ) {
    throw new Error('Invalid Earth Score baseline features')
  }
  if (
    !Array.isArray(m.chunks) ||
    !m.chunks.length ||
    m.chunks.length > m.sites.length ||
    m.chunks.some(
      c =>
        !c ||
        typeof c.file !== 'string' ||
        !/^[a-zA-Z0-9_-]+\.zst$/.test(c.file) ||
        !integer(c.rows, TILE_SIZE ** 2, m.rows) ||
        c.rows % TILE_SIZE ** 2 !== 0 ||
        !integer(c.packedBytes, 16 + 8 * m.columns.length * Math.ceil(c.rows / BLOCK_SIZE), maxChunkBytes) ||
        c.packedBytes % 4 !== 0 ||
        !integer(c.compressedBytes, 1, maxChunkBytes),
    ) ||
    new Set(m.chunks.map(c => c.file)).size !== m.chunks.length ||
    m.chunks.reduce((rows, c) => rows + c.rows, 0) !== m.rows
  ) {
    throw new Error('Invalid Earth Score chunk layout')
  }
  m.encodings.forEach(Object.freeze)
  m.features.forEach(Object.freeze)
  m.chunks.forEach(Object.freeze)
  for (const list of [m.sites, m.columns, m.encodings, m.features, m.chunks]) Object.freeze(list)
  return Object.freeze(m)
}
