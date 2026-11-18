import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { zstdDecompressSync } from 'node:zlib'
import { buildEarthQuery, freezePlanetaryShaderPoint, type PlanetaryShaderSample } from '../../src/earth-score/index.ts'
import { lowerPlanetaryCalls, packPlanetaryShaderPoints, planetShaderBytes } from '../../src/earth-score/planetShaderSchema.ts'
import { prepareScoreShader, SCORE_SHADER_EXAMPLES, SCORE_SHADER_FIELDS, SCORE_SHADER_PRELUDE, ScoreShaderError } from '../../src/earth-score/custom-shader.ts'
import { createLayout, createScoreGeometry, GRID_WIDTH } from '../../src/earth-score/layout.ts'
import { parseManifest } from '../../src/earth-score/manifest.ts'
import { createQueryParameters, type EarthQueryCriterion } from '../../src/earth-score/query.ts'
import { createLookup, createParameters } from '../../src/earth-score/scoring.ts'
import type { Criterion } from '../../src/explorerData.ts'

const world = new URL('../../compute-data/world/', import.meta.url)
const data = JSON.parse(await readFile(new URL('manifest.json', world), 'utf8'))
const manifest = parseManifest(data)

test('world sites and chunk ranges retain exact native addresses, including padding', () => {
  const layout = createLayout(manifest)
  assert.equal(layout.rows, 35550640)
  assert.equal(layout.sites.length, 460)
  assert.equal(layout.chunks.length, 36)
  assert.equal(layout.locate(0)?.siteKey, 'h00v07')
  assert.equal(layout.locate(0)?.center, null)
  assert.equal(layout.locate(1004691)?.chunk, 0)
  assert.equal(layout.locate(1004692)?.row, 0)
  assert.equal(layout.locate(layout.rows - 1)?.siteKey, 'h35v10')
  assert.equal(layout.locate(layout.rows), null)
  assert.equal(layout.addressOf('h00v00', 0), null)
  assert.equal(layout.addressOf('h00v07', 278 ** 2), null)
  assert.equal(layout.addressAt(Infinity, 0), null)
  let physicalSites = 0
  for (const [i, site] of layout.sites.entries()) {
    const middle = layout.addressOf(site, 139 * 278 + 139)!
    assert.equal(middle.address, i * 278 ** 2 + middle.pixel)
    if (middle.center) {
      const { longitude, latitude } = middle.center
      assert.equal(layout.addressAt(longitude, latitude)?.address, middle.address)
      physicalSites++
    }
  }
  assert(physicalSites > 0 && physicalSites < layout.sites.length)
})

test('shader spatial lookup retains absolute tile origins and exact Earth padding boundaries', () => {
  const words = createScoreGeometry(manifest)
  const floats = new Float32Array(words.buffer)
  const before = manifest.chunks[0].rows / 278 ** 2 - 1
  const after = before + 1
  assert.equal(manifest.sites[before], 'h02v07')
  assert.equal(manifest.sites[after], 'h02v08')
  assert.equal(words[before], 7 * 278 * GRID_WIDTH + 2 * 278)
  assert.equal(words[after], 8 * 278 * GRID_WIDTH + 2 * 278)

  const y = 1739
  const row = manifest.sites.length + y * 3
  const latitude = 90 - (y + 0.5) * 180 / 5004
  assert.equal(floats[row], Math.fround(latitude))
  assert.equal(floats[row + 1], Math.fround(180 / (5004 * Math.cos(latitude * Math.PI / 180))))
  assert.equal(words[row + 2], 4441)
  const layout = createLayout(manifest)
  assert.equal(layout.addressOf('h02v06', 71 * 278 + 6)?.center, null)
  const adjacent = layout.addressOf('h02v06', 71 * 278 + 7)!
  assert(Math.abs(adjacent.center!.longitude + 179.959474707) < 0.000001)
  assert.equal(layout.addressAt(adjacent.center!.longitude, adjacent.center!.latitude)?.address, adjacent.address)
})

test('rejects corrupt manifest layout before allocating GPU resources', () => {
  const duplicateSite = structuredClone(data)
  duplicateSite.sites[1] = duplicateSite.sites[0]
  assert.throws(() => parseManifest(duplicateSite), /site keys/)
  const escapedPath = structuredClone(data)
  escapedPath.chunks[0].file = '../chunk.zst'
  assert.throws(() => parseManifest(escapedPath), /chunk layout/)
  const unalignedChunk = structuredClone(data)
  unalignedChunk.chunks[0].rows--
  assert.throws(() => parseManifest(unalignedChunk), /chunk layout/)
})

test('exact float lookup preserves valid physical zero and encodes missing separately', () => {
  const lookup = createLookup(manifest)
  const floats = new Float32Array(lookup.buffer)
  const slope = manifest.columns.indexOf('slope_mean_deg') * 65536
  assert.equal(lookup[slope], 0x7fc00000)
  assert.equal(floats[slope + 1], 0)
  assert.equal(floats[slope + 2], Math.fround(0.1))
  const temperature = manifest.columns.indexOf('surface_low_c') * 65536
  assert.equal(floats[temperature + 1], -150)
  assert.equal(floats[temperature + 7501], 0)
  assert.equal(floats[temperature + 65535], Math.fround((65535 - 7501) / 50))
})

test('target, tolerance and weight overrides keep the baseline reusable', () => {
  const floats = (overrides?: Parameters<typeof createParameters>[1]) =>
    new Float32Array(createParameters(manifest, overrides).buffer)
  const initial = floats()
  const moved = floats({ slope_mean_deg: { target: 1.25, tolerance: 4, weight: 2 } })
  assert.equal(moved[16], 1.25)
  assert.equal(moved[17], 0.25)
  assert.equal(moved[18], Math.fround(2 / 5.5))
  assert.equal(initial[18], Math.fround(1 / 4.5))
  assert.deepEqual(floats(), initial)
  assert.throws(() => floats({ no_field: { target: 0 } }), /Unknown score feature/)
  assert.throws(() => floats({ slope_mean_deg: { tolerance: 0 } }), /Invalid score parameters/)
  assert.throws(() => floats({ slope_mean_deg: { weight: 1e39 } }), /Invalid score parameters/)
})

function criterion(key: string, settings: Partial<EarthQueryCriterion> = {}): Criterion {
  return {
    key, label: key, group: key, unit: 'test unit', enabled: true, target: 0,
    tolerance: 1, weight: 1, min: null, max: null, ...settings,
  }
}

test('criteria queries accept the full non-coverage field set and snapshot sampled subsets', () => {
  const scoreFields = manifest.columns.filter(key => !key.endsWith('_valid_fraction'))
  const pack = (criteria: Criterion[]) => createQueryParameters(manifest, buildEarthQuery(criteria), Infinity)
  assert.equal(pack(scoreFields.map(key => criterion(key)))[0], 19)

  const criteria = [
    criterion('surface_night_c', { target: -45, tolerance: 20 }),
    criterion('slope_mean_deg', { enabled: false, target: null, qualityKey: 'terrain_valid_fraction' }),
    criterion('unsupported_inactive_field', { enabled: false }),
  ]
  const query = buildEarthQuery(criteria)
  const words = pack(criteria)
  criteria[0].target = 65
  assert.deepEqual(createQueryParameters(manifest, query, Infinity), words)
  assert.deepEqual(Array.from(words.slice(0, 2)), [1, 0])
  assert.equal(words[words.length - 8], manifest.columns.indexOf('surface_night_c'))
  assert.throws(() => createQueryParameters(manifest, query, 32), /device storage limit/)
  const reduced = { ...manifest, columns: manifest.columns.filter(key => key !== 'surface_night_c') }
  assert.throws(() => createQueryParameters(reduced, query, Infinity), /data does not contain surface_night_c/)
})

test('criteria queries validate active fields, coverage and numeric parameters', () => {
  const slope = criterion('slope_mean_deg')
  assert.throws(() => buildEarthQuery([criterion('terrain_valid_fraction')]), /Unsupported Earth Score field/)
  assert.throws(() => buildEarthQuery([{ ...slope, qualityKey: 'bare_fraction' }]), /Unsupported Earth Score coverage/)
  assert.throws(() => buildEarthQuery([{ ...slope, target: null }]), /enter a finite number/)
  assert.throws(() => buildEarthQuery([{ ...slope, tolerance: NaN }]), /enter a finite number/)
  assert.throws(() => buildEarthQuery([{ ...slope, tolerance: 1e-7 }]), /tolerance must be at least/)
  assert.throws(() => buildEarthQuery([{ ...slope, weight: Infinity }]), /weight must be positive/)
  assert.throws(() => buildEarthQuery([{ ...slope, max: 1e13 }]), /enter a finite number/)
  assert.throws(() => buildEarthQuery([{ ...slope, min: 12, max: 3 }]), /minimum exceeds maximum/)
  assert.throws(() => buildEarthQuery([{ ...slope, enabled: false, min: 3 }]), /at least one variable/)
  assert.doesNotThrow(() => buildEarthQuery([criterion('dominant_lithology_code', { categorical: true, tolerance: NaN })]))
})

test('criteria query hard bounds distinguish a stored decimal from its neighbors', () => {
  const makeQuery = (min: number, max: number) => buildEarthQuery([
    criterion('bare_fraction', { min, max, qualityKey: 'landcover_valid_fraction' }),
  ])
  const inclusive = createQueryParameters(manifest, makeQuery(.9, .9), Infinity)
  const above = createQueryParameters(manifest, makeQuery(.9000000000000001, 1), Infinity)
  const below = createQueryParameters(manifest, makeQuery(0, .8999999999999999), Infinity)
  assert.equal(inclusive[5], 901) // minimum coverage
  assert.deepEqual(Array.from(inclusive.slice(-7, -5)), [901, 901])
  assert.equal(above.at(-7), 902)
  assert.equal(below.at(-6), 900)
})

test('the packaged world matches all 36 frozen Zstd frames and packed hashes', async () => {
  const reference = JSON.parse(await readFile(new URL('./reference/manifest.json', import.meta.url), 'utf8'))
  const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex')
  let compressedBytes = 0
  let packedBytes = 0
  for (const [i, shard] of manifest.chunks.entries()) {
    const frame = await readFile(new URL(shard.file, world))
    assert.equal(frame.length, shard.compressedBytes)
    assert.equal(hash(frame), reference.chunks[i].compressedSha256, shard.file)
    compressedBytes += frame.length
    const packed = zstdDecompressSync(frame)
    assert.equal(packed.length, shard.packedBytes)
    assert.equal(hash(packed), reference.chunks[i].packedSha256, shard.file)
    packedBytes += packed.length
  }
  assert.equal(compressedBytes, 183395621)
  assert.equal(packedBytes, 384789304)
})

test('editable WGSL retains code and supplies all physical fields independently of access syntax', () => {
  assert.deepEqual([...SCORE_SHADER_FIELDS.map(field => field.name)].sort(), [...manifest.columns].sort())
  for (const example of SCORE_SHADER_EXAMPLES) {
    const result = prepareScoreShader(example.source, manifest)
    assert.ok(result.code.startsWith(example.source + '\nstruct EarthCell'))
    assert.match(result.code, /struct MoonPoint/)
    assert.match(SCORE_SHADER_PRELUDE, /fn mars_point\(latitude: f32, longitude: f32\) -> MarsPoint/)
    assert.equal(result.sourceLines, example.source.split('\n').length)
  }
  assert.match(SCORE_SHADER_PRELUDE, /latitude_deg: f32/)
  assert.match(SCORE_SHADER_PRELUDE, /longitude_deg: f32/)
  assert.match(SCORE_SHADER_PRELUDE, /grid_x: u32/)
  assert.match(SCORE_SHADER_PRELUDE, /grid_y: u32/)
  const program = `/* @group(0) /* @binding(1) */ */\r
const earth_score_internal_0_example = 0.2;
fn score(pixel: EarthCell) -> f32 {
  let result = array<f32, 1>(pixel.bare_fraction);
  for (var i = 0; i < 1; i++) {}
  return result[0];
}`
  const compiled = prepareScoreShader(program, manifest)
  assert.match(compiled.entryPoint, /earth_score_internal_1_/)
  assert.equal(compiled.sourceLines, 7)
  for (const field of SCORE_SHADER_FIELDS) {
    assert.match(compiled.code, new RegExp(`_value\\(rows, ${manifest.columns.indexOf(field.name)}u, row\\)`))
  }
})

test('the score wrapper owns GPU resource bindings', () => {
  const source = 'fn score(pixel: EarthCell) -> f32 { return 1.0; }\n@/*comment*/group(0) @binding(1) var<storage, read> data: array<f32>;'
  assert.throws(() => prepareScoreShader(source, manifest), (error: unknown) => {
    assert.ok(error instanceof ScoreShaderError)
    const [diagnostic] = error.diagnostics
    assert.equal(diagnostic.line, 2)
    assert.equal(diagnostic.column, 13)
    assert.match(diagnostic.message, /GPU resource bindings/)
    return true
  })
})

test('float32 physical fields retain the exact encoded measurement for generated queries', () => {
  const limits = [
    { origin: -1_000_000, denominator: 1 },
    { origin: 1_000_000, denominator: 1 },
    { origin: -1_000_000, denominator: 999_983 },
    { origin: 1_000_000, denominator: 999_983 },
  ]
  for (const { origin, denominator } of [...manifest.encodings, ...limits]) {
    for (let code = 1; code <= 65535; code++) {
      const physical = Math.fround((code - origin) / denominator)
      const restored = Math.round(Math.fround(Math.fround(physical * denominator) + origin))
      assert.equal(restored, code)
    }
  }
})

test('planetary sample layout and frozen values enforce backend status and units', () => {
  const sample: PlanetaryShaderSample = {
    body: 'moon', latitude: 90, longitude: 180, partial: true,
    measurements: [
      { variableId: 'moon.relief_5km', unit: 'm', status: 2, qualityFlags: [], decoded: 4000 },
      { variableId: 'moon.relief_5km', unit: 'km', status: 1, qualityFlags: [], decoded: 4 },
      { variableId: 'moon.relief_5km', unit: 'm', status: 1, qualityFlags: ['semantic_review_required'], decoded: 4000 },
      { variableId: 'moon.relief_5km', unit: 'm', status: 1, qualityFlags: [], decoded: 42.25 },
      { variableId: 'moon.surface_noon', unit: 'K', status: 1, qualityFlags: [], decoded: 265.5 },
      { variableId: 'mars.elevation', unit: 'm', status: 1, qualityFlags: [], decoded: 313 },
      { variableId: 'moon.elevation', unit: 'm', status: 1, qualityFlags: [], decoded: ['invalid'] },
    ],
  }
  const packed = packPlanetaryShaderPoints([sample, { ...sample, longitude: -34 }])
  assert.deepEqual([...packed.slice(0, 3)], [90, 0, 1])
  assert.equal(packed[5], 42.25)
  assert.equal(packed[9], 265.5)
  const frozen = freezePlanetaryShaderPoint(sample)
  assert.match(frozen, /MoonPoint\(\s+90.0, 0.0, true, true/)
  assert.match(frozen, /42.25, \/\/ relief_5km_m/)
  assert.match(frozen, /265.5, \/\/ noon_k/)
  assert.match(frozen, /NO_DATA, \/\/ elevation_m/)
  assert.throws(() => freezePlanetaryShaderPoint({ ...sample, latitude: NaN }), /coordinates/)
  const many = Array.from({ length: 40 }, (_, longitude) => ({ ...sample, latitude: 0, longitude }))
  assert.equal(packPlanetaryShaderPoints(many).byteLength, planetShaderBytes(many.length))
  const call = 'moon_point(\n  90., 180.,\n)'
  const source = '/* 🌓 café */ ' + call
  const offset = source.indexOf(call)
  const lowered = lowerPlanetaryCalls(source, {
    samples: many,
    calls: [{ body: 'moon', offset, length: call.length, slot: 38 }],
  })
  assert.equal(lowered.source.length, source.length)
  assert.equal(lowered.source.slice(0, offset), source.slice(0, offset))
  assert.equal(lowered.source.slice(offset + 10), source.slice(offset + 10))
  assert.equal(lowered.slots.get(38)?.body, 'moon')
})
