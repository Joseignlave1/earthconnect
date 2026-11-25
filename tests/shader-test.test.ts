import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFileSync } from 'node:fs'
import {
  cameraBasis,
  cameraMatrix,
  createMesh,
  EARTH_AXIS,
  EARTH_FLATTENING,
  EQUAL_EARTH_HALF_HEIGHT,
  EQUAL_EARTH_HALF_WIDTH,
  equalEarthInverse,
  equalEarthProject,
  FOV,
  grabCenter,
  grabRotation,
  orbitRotation,
  pickSurface,
  SurfacePicker,
  surfacePoint,
  type Camera,
  type Rotation,
  type Vec3,
} from '../src/shader-test/geometry.ts'
import {
  decodeTerrain,
  decodeTerrainEct2Payload,
  elevationAt,
  elevationPyramid,
  packRg8Levels,
  poleElevations,
} from '../src/shader-test/terrain.ts'
import { solarPosition } from '../src/shader-test/sun.ts'
import { contourDetail } from '../src/shader-test/contours.ts'
import { CELL, MISSING, RADIUS, SIZE, cellAddress, tileRow, tileX, worldTiles } from '../src/shader-test/world-grid.ts'
import {
  AUTO_POSITIONS,
  normalizedScore,
  palettePosition,
  resolveScoreStyle,
  ScoreDistribution,
  scoreAtPalette,
  scorePalette,
} from '../src/shader-test/score-style.ts'
import { computedScoreLayout } from '../src/shader-test/computed-score-layout.ts'
import { createLayout } from '../src/earth-score/layout.ts'
import { parseManifest } from '../src/earth-score/manifest.ts'
import {
  paintTilePageTable,
  parseTilePackHeader,
  parseTilePackIndex,
  planetTileKey,
  tileKey,
} from '../src/shader-test/virtual-tiles.ts'

const camera: Camera = { yaw: 0, pitch: 0, distance: 6, aspect: 1.5 }
const mesh = createMesh()
const close = (actual: number, expected: number, tolerance = 1e-5) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} ≠ ${expected}`)

test('GPU score tile batches honor manifest address order across texture banks', () => {
  const manifest = parseManifest(
    JSON.parse(readFileSync(new URL('../compute-data/world/manifest.json', import.meta.url), 'utf8')),
  )
  const layout = createLayout(manifest)
  const cells = SIZE ** 2
  const chunks = layout.chunks.map(chunk => ({
    ...chunk,
    scores: { size: chunk.rows * 4 } as GPUBuffer,
    packedInput: {} as GPUBuffer,
  }))
  const { lookup, batches, bytes } = computedScoreLayout({ layout, chunks }, 256, 2)
  assert.equal(
    bytes,
    layout.sites.length * [278, 139, 69, 34, 17, 8, 4, 2, 1].reduce((sum, size) => sum + size ** 2 * 2, 0),
  )
  assert.equal(
    batches.reduce((sum, batch) => sum + batch.layers, 0),
    layout.sites.length,
  )
  assert.ok(batches.some(batch => batch.bank === 1 && batch.layer === 0 && batch.chunkTile > 0))
  for (const { bank, layer, layers, chunk, chunkTile, firstSite } of batches) {
    assert.equal(firstSite, bank * 256 + layer)
    assert.equal(firstSite, chunk.firstAddress / cells + chunkTile)
    assert.ok(layer + layers <= 256 && layers <= 16 && chunkTile + layers <= chunk.rows / cells)
  }
  for (const [i, key] of layout.sites.entries()) {
    assert.equal(lookup[Number(key.slice(4, 6)) * 36 + Number(key.slice(1, 3))], i)
  }
  assert.equal(lookup[0], MISSING)
  assert.throws(() => computedScoreLayout({ layout, chunks: chunks.slice(1) }, 256, 2), /invalid.*chunk layout/i)
})

test('top-score colors reserve the ramp for strong matches continuously without changing global normalization', () => {
  close(normalizedScore(0.7, 0.4, 1), 0.5)
  assert.equal(normalizedScore(0.2, 0.4, 1), 0)
  assert.equal(normalizedScore(1, 0.4, 0.9), 1)
  assert.equal(normalizedScore(0.9, 0.95, 0.9), 0)
  close(palettePosition(0.9, 0.1, 1), 0.225)
  // Widening the top band leaves low-score context unchanged.
  close(palettePosition(0.2, 0.1, 1), palettePosition(0.2, 0.5, 8))
  for (const band of [0.01, 0.1, 0.5, 1])
    for (const contrast of [0.1, 1, 8]) {
      close(palettePosition(0, band, contrast), 0)
      close(palettePosition(1, band, contrast), 1)
      const knee = 1 - band
      close(palettePosition(knee, band, contrast), 0.25 * knee)
      let previous = 0
      for (let i = 0; i <= 1000; i++) {
        const value = palettePosition(i / 1000, band, contrast)
        assert.ok(Number.isFinite(value) && value >= previous && value <= 1)
        previous = value
      }
    }
})

function distribution(entries: [number, number][]) {
  const counts = new Uint32Array(MISSING)
  for (const [score, count] of entries) counts[Math.round(score * 65534)] += count
  return new ScoreDistribution(counts)
}
const autoControls = { colorMode: 'auto' as const, base: 0.4, focus: 0.1, topRange: 0.1, emphasis: 1 }

test('automatic contrast follows the upper tail of eligible native cells, preserving zeros and ties', () => {
  const low = distribution([
    [0, 1000],
    [0.4, 200],
    [0.5, 900],
    [0.7, 90],
    [0.9, 9],
    [1, 1],
  ])
  const high = distribution([
    [0, 1000],
    [0.4, 200],
    [0.8, 900],
    [0.94, 90],
    [0.98, 9],
    [1, 1],
  ])
  assert.equal(low.valid, 2200)
  // Native code rounding determines strict cutoff membership.
  const cutoff = Math.round(0.4 * 65534) / 65534
  assert.equal(low.candidates(cutoff), 1000)
  close(low.quantile(cutoff, 0.9), 0.5, 1 / 65534)
  close(low.quantile(cutoff, 0.99), 0.7, 1 / 65534)
  close(low.quantile(cutoff, 0.999), 0.9, 1 / 65534)
  const a = resolveScoreStyle(low, { ...autoControls, base: cutoff }),
    b = resolveScoreStyle(high, { ...autoControls, base: cutoff })
  assert.equal(a.best, b.best)
  a.anchors.slice(0, 3).forEach((score, i) => close(score, [0.5, 0.7, 0.9][i], 1 / 65534))
  b.anchors.slice(0, 3).forEach((score, i) => close(score, [0.8, 0.94, 0.98][i], 1 / 65534))
  // The same upper-tail ranks get the same colors in these different queries.
  close(scorePalette(0.7, a), scorePalette(0.94, b), 0.0001)
  assert.equal(scorePalette(0, a), 0)
  assert.equal(scorePalette(cutoff, a), 0)
})

test('minimum-score filtering is inclusive, preserves valid zeros, and does not alter contrast statistics', () => {
  const data = distribution([
    [0, 10],
    [0.4, 20],
    [0.7, 30],
    [1, 5],
  ])
  const threshold = Math.round(0.4 * 65534) / 65534
  assert.equal(data.atLeast(0), 65)
  assert.equal(data.atLeast(threshold), 55)
  assert.equal(data.candidates(threshold), 35)
  assert.equal(data.atLeast(1), 5)
  assert.equal(
    distribution([
      [0, 10],
      [0.7, 20],
    ]).atLeast(1),
    0,
  )
  assert.equal(distribution([]).atLeast(0), 0)
  const before = resolveScoreStyle(data, autoControls)
  for (const minimum of [0, 0.2, 0.7, 1]) data.atLeast(minimum)
  assert.deepEqual(resolveScoreStyle(data, autoControls), before)
})

test('automatic curves remain continuous and bounded through tied or almost identical scores', () => {
  const cases = [
    distribution([[0.6, 10000]]),
    distribution([
      [0.9, 10000],
      [0.90002, 100],
      [0.90004, 1],
    ]),
    distribution([[0.40001, 1]]),
  ]
  for (const data of cases)
    for (const focus of [0.01, 0.1, 1]) {
      const style = resolveScoreStyle(data, { ...autoControls, focus })
      const scores = [style.cutoff, ...style.anchors]
      assert.equal(scorePalette(style.best, style), 1)
      for (let i = 1; i < scores.length; i++) {
        assert.ok(scores[i] > scores[i - 1])
        close(scorePalette(scores[i], style), AUTO_POSITIONS[i])
        const slope = (AUTO_POSITIONS[i] - AUTO_POSITIONS[i - 1]) / (scores[i] - scores[i - 1])
        assert.ok(slope <= 1 / Math.min(0.02, style.best - style.cutoff) + 0.00001)
        close(scorePalette(scores[i] - 1e-10, style), scorePalette(scores[i] + 1e-10, style), 0.001)
      }
      for (const position of [0, 0.01, 0.22, 0.25, 0.5, 0.55, 0.75, 0.82, 0.99, 1])
        close(scorePalette(scoreAtPalette(position, style), style), position)
      let previous = 0
      for (let code = 0; code <= 65534; code++) {
        const value = scorePalette(code / 65534, style)
        assert.ok(Number.isFinite(value) && value >= previous && value <= 1)
        previous = value
      }
    }
})

test('locking preserves numeric colors, cutoff, and glow across queries; unlocking recalibrates', () => {
  const a = distribution([
    [0.5, 900],
    [0.7, 90],
    [0.9, 9],
    [1, 1],
  ])
  const b = distribution([
    [0.8, 900],
    [0.94, 90],
    [0.98, 9],
    [1, 1],
  ])
  const original = resolveScoreStyle(a, autoControls)
  const locked = resolveScoreStyle(b, { ...autoControls, colorMode: 'locked', base: 0.8, focus: 0.5 }, original)
  assert.equal(locked, original)
  assert.notDeepEqual(resolveScoreStyle(b, autoControls, locked).anchors, original.anchors)
  const manual = resolveScoreStyle(a, { ...autoControls, colorMode: 'manual' })
  for (const score of [0.4, 0.6, 0.94, 0.99, 1])
    close(scorePalette(score, manual), palettePosition(normalizedScore(score, 0.4, 1), 0.1, 1))
  for (const position of [0, 0.1, 0.5, 0.9, 1]) close(scorePalette(scoreAtPalette(position, manual), manual), position)
})

test('empty candidate sets stay empty and automatic glow does not wash out tied plateaus', () => {
  const empty = resolveScoreStyle(
    distribution([
      [0, 1000],
      [0.2, 10],
    ]),
    autoControls,
  )
  assert.equal(empty.eligible, 0)
  assert.equal(scorePalette(0.2, empty), 0)
  assert.equal(scoreAtPalette(0.5, empty), 0.4)
  const plateau = distribution([
    [0.9, 10000],
    [1, 10000],
  ])
  const style = resolveScoreStyle(plateau, autoControls)
  let emission = 0
  for (let code = 0; code < plateau.counts.length; code++)
    emission += plateau.counts[code] * scorePalette(code / 65534, style) ** 12 * style.emissionScale
  assert.ok(emission / style.eligible <= 0.00500001)
  assert.ok(style.emissionScale < 0.02)
  const manual = resolveScoreStyle(plateau, { ...autoControls, colorMode: 'manual' })
  assert.equal(manual.emissionScale, 1)
})

test('contour families reveal continuously with scale and finer families follow the major ones', () => {
  assert.deepEqual(contourDetail(20000), [0, 0])
  assert.deepEqual(contourDetail(4000), [1, 0])
  assert.deepEqual(contourDetail(500), [1, 1])
  let previous = [0, 0]
  for (let scale = 20000; scale >= 0; scale -= 10) {
    const weights = contourDetail(scale)
    weights.forEach((value, index) => {
      assert.ok(value >= previous[index] && value - previous[index] < 0.006)
      assert.ok(value >= 0 && value <= 1)
    })
    if (weights[1] > 0) assert.equal(weights[0], 1)
    previous = weights
  }
})

function project(position: Vec3, view: Camera) {
  const matrix = cameraMatrix(view)
  const vector = [...position, 1]
  const result = Array.from({ length: 4 }, (_, row) =>
    vector.reduce((sum, value, col) => sum + matrix[col * 4 + row] * value, 0),
  )
  return [result[0] / result[3], result[1] / result[3]]
}

test('keyboard orbit follows screen directions through poles, upside down and after mixed rotations', () => {
  const identity: Rotation = [0, 0, 0, 1]
  const rotations = [-4 * Math.PI, -Math.PI, -Math.PI / 2, -1.6, 0, 1.6, Math.PI / 2, Math.PI, 4 * Math.PI].map(pitch =>
    orbitRotation(identity, 0, pitch),
  )
  rotations.push(orbitRotation(orbitRotation(orbitRotation(identity, 0.8, 2), -0.6, 0.4), 1.2, -0.3))
  const smallMesh = createMesh(48, 24)
  for (const rotation of rotations) {
    const view = { ...camera, rotation }
    const { back } = cameraBasis(view)
    for (const [dx, dy] of [
      [0.01, 0],
      [-0.01, 0],
      [0, 0.01],
      [0, -0.01],
    ]) {
      const moved = { ...view, rotation: orbitRotation(rotation, dx, dy) }
      const [x, y] = project(back, moved)
      if (dx) {
        assert.ok(x * dx > 0)
        close(y, 0)
      }
      if (dy) {
        assert.ok(y * dy < 0)
        close(x, 0)
      }
      const hit = pickSurface(0, 0, moved, 1, smallMesh)
      assert.ok(hit)
      const projected = project(hit.position, moved)
      projected.forEach(value => close(value, 0))
    }
  }
  let rotation = identity
  for (let i = 0; i < 10000; i++) rotation = orbitRotation(rotation, 0.006, 0.004)
  close(Math.hypot(...rotation), 1, 1e-12)
  const { right, up, back } = cameraBasis({ ...camera, rotation })
  for (const axis of [right, up, back]) close(Math.hypot(...axis), 1, 1e-12)
  for (const [a, b] of [
    [right, up],
    [up, back],
    [back, right],
  ])
    close(
      a.reduce((sum, value, i) => sum + value * b[i], 0),
      0,
      1e-12,
    )
})

test('arcball keeps a grabbed terrain point attached and restores its starting orientation after a loop', () => {
  const smallMesh = createMesh(96, 48)
  for (const distance of [1.035, 1.3, 3.8, 9])
    for (const pitch of [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, 1.8, Math.PI]) {
      const start = { ...camera, distance, rotation: orbitRotation(orbitRotation([0, 0, 0, 1], 0.7, 0), 0, pitch) }
      const extent = Math.min(0.7, 0.75 / (Math.tan(FOV / 2) * Math.sqrt(distance * distance - 1)))
      const terrain = { heightAt: () => 1300, exaggeration: 20 }
      const anchor = pickSurface((extent * 0.2) / start.aspect, extent * 0.1, start, 1, smallMesh, terrain)
      assert.ok(anchor)
      for (const [x, y] of [
        [0.3, -0.4],
        [-0.6, 0.2],
        [0.6, 0.3],
        [0.2, 0.1],
      ]) {
        const target = [(extent * x) / start.aspect, extent * y]
        const view = { ...start, rotation: grabRotation(start, anchor.position, target[0], target[1]) }
        const projected = project(anchor.position, view)
        projected.forEach((value, i) => close(value, target[i], 2e-5))
        if (x === 0.2 && y === 0.1) view.rotation.forEach((value, i) => close(value, start.rotation[i], 1e-10))
      }
    }
})

test('cached picking reuses surface geometry across camera moves and refreshes changed terrain settings', () => {
  const smallMesh = createMesh(48, 24),
    picker = new SurfacePicker(smallMesh)
  let calls = 0
  const heightAt = () => {
    calls++
    return 1200
  }
  const terrain = { heightAt, exaggeration: 20 }
  picker.setSurface(1, terrain)
  const preparedCalls = calls
  assert.equal(preparedCalls, smallMesh.uv.length / 2)
  for (const pitch of [-1.8, 0, 1.8, Math.PI]) {
    const view = { ...camera, rotation: orbitRotation([0, 0, 0, 1], 0.3, pitch) }
    assert.ok(picker.pick(0.05, 0.03, view))
  }
  picker.setSurface(1, { ...terrain })
  assert.equal(calls, preparedCalls, 'camera and pointer updates must not rebuild vertices or resample terrain')
  for (const [morph, exaggeration] of [
    [0, 20],
    [0.5, 20],
    [1, 40],
  ]) {
    const settings = { heightAt, exaggeration }
    picker.setSurface(morph, settings)
    const cached = picker.pick(0.05, 0.03, camera),
      direct = pickSurface(0.05, 0.03, camera, morph, smallMesh, settings)
    assert.ok(cached && direct)
    cached.position.forEach((value, i) => close(value, direct.position[i], 1e-12))
    close(cached.longitude, direct.longitude)
    close(cached.latitude, direct.latitude)
  }
})

test('plane and transitional views pan the exact grabbed point under the cursor', () => {
  for (const morph of [0, 0.25, 0.75])
    for (const distance of [1.5, 6, 12]) {
      let view: Camera = {
        ...camera,
        distance,
        rotation: orbitRotation([0, 0, 0, 1], 0.2, 0.15),
        center: [0.1, -0.05, 0],
      }
      const anchor = pickSurface(0.03, 0.02, view, morph, createMesh(48, 24))
      assert.ok(anchor)
      for (const target of [
        [0.2, -0.15],
        [-0.3, 0.4],
        [0.03, 0.02],
      ]) {
        view = { ...view, center: grabCenter(view, anchor.position, target[0], target[1]) }
        project(anchor.position, view).forEach((value, i) => close(value, target[i], 2e-5))
      }
    }
})

test('orbiting across the longitude seam picks the near surface including displaced terrain', () => {
  for (const pitch of [1.6, 1.68, 3.12, -1.6, -1.68, -3.12]) {
    const view = { ...camera, rotation: orbitRotation([0, 0, 0, 1], 0, pitch) }
    const hit = pickSurface(0, 0, view, 1, mesh, {
      heightAt: (_longitude, latitude) => (latitude < 0 ? 2357 : 0),
      exaggeration: 20,
    })
    assert.ok(hit)
    const { back } = cameraBasis(view)
    assert.ok(hit.position.reduce((sum, value, i) => sum + value * back[i], 0) > 0.99)
  }
})

test('the flat map covers the full world and the sphere closes its longitude seam', () => {
  surfacePoint(Math.PI, 0, 0).forEach((value, index) => close(value, [EQUAL_EARTH_HALF_WIDTH, 0, 0][index]))
  surfacePoint(-Math.PI, 0, 0).forEach((value, index) => close(value, [-EQUAL_EARTH_HALF_WIDTH, 0, 0][index]))
  close(surfacePoint(0, Math.PI / 2, 0)[1], EQUAL_EARTH_HALF_HEIGHT)
  close(surfacePoint(0, -Math.PI / 2, 0)[1], -EQUAL_EARTH_HALF_HEIGHT)
  for (const [lon, lat] of [
    [-2.9, -1.4],
    [-1.2, 0.7],
    [0, 0],
    [2.4, 1.35],
    [Math.PI, -0.5],
  ]) {
    const [x, y] = equalEarthProject(lon, lat)
    const inv = equalEarthInverse(x, y)
    assert.ok(inv)
    close(inv.lon, lon, 1e-9)
    close(inv.lat, lat, 1e-9)
  }
  assert.equal(equalEarthInverse(EQUAL_EARTH_HALF_WIDTH + 0.1, 0), null)
  assert.equal(equalEarthInverse(0, EQUAL_EARTH_HALF_HEIGHT + 0.01), null)
  for (const latitude of [-1.2, 0, 1.2]) {
    const left = surfacePoint(-Math.PI, latitude, 1),
      right = surfacePoint(Math.PI, latitude, 1)
    left.forEach((value, index) => close(value, right[index]))
    close(left[0] ** 2 + left[2] ** 2 + (left[1] / (1 - EARTH_FLATTENING)) ** 2, 1)
  }
})

test('numeric terrain decoding preserves exact bytes and rejects malformed payloads', () => {
  const buffer = new ArrayBuffer(28)
  const view = new DataView(buffer)
  view.setUint32(0, 0x45435431)
  view.setUint32(4, 2, true)
  view.setUint32(8, 2, true)
  const pixels = new Uint8ClampedArray(buffer, 12)
  pixels.set([0, 255, 128, 255, 255, 0, 0, 255, 127, 255, 255, 255, 128, 0, 128, 255])
  const terrain = decodeTerrain(buffer)
  assert.deepEqual([terrain.width, terrain.height], [2, 2])
  assert.deepEqual(terrain.pixels, pixels)
  close(elevationAt(terrain, -Math.PI / 2, Math.PI / 4), -32513)
  close(elevationAt(terrain, Math.PI / 2, Math.PI / 4), 32512)
  assert.throws(() => decodeTerrain(buffer.slice(0, 27)), /dimensions or length/)
  assert.throws(() => decodeTerrain(buffer.slice(0, 4)), /header/)
  view.setUint32(4, 0, true)
  assert.throws(() => decodeTerrain(buffer), /dimensions or length/)
  view.setUint32(0, 0)
  assert.throws(() => decodeTerrain(buffer), /header/)
})

test('elevation decoding preserves negative heights, bilinear interpolation and wrapping', () => {
  const pixels = new Uint8ClampedArray(16)
  ;[-1000, 2000, 100, 1000].forEach((height, index) => {
    const code = height + 32768
    pixels[index * 4] = code >> 8
    pixels[index * 4 + 1] = code & 255
    pixels[index * 4 + 3] = 255
    // Coast membership is independent of height (including negative inland basins).
    pixels[index * 4 + 2] = 128 + [8, -8, 8, 8][index] * 4
  })
  const terrain = { pixels, width: 2, height: 2 }
  close(elevationAt(terrain, -Math.PI / 2, Math.PI / 4), -1000)
  close(elevationAt(terrain, 0, Math.PI / 4), 500)
  close(elevationAt(terrain, -Math.PI, Math.PI / 4), 500)
  close(elevationAt(terrain, Math.PI, Math.PI / 4), 500)
  close(elevationAt(terrain, -Math.PI / 2, -Math.PI / 2), 100)
  const levels = elevationPyramid(terrain)
  close(levels.at(-1)!.values[0], 525)
  assert.deepEqual([...levels[0].coast], [0, 0, 8, 8])
  close(levels.at(-1)!.coast[0], 4)
  assert.deepEqual(poleElevations(terrain), [550, 500])
})

test('polar picking recovers geographic coordinates from the surface rather than the UV fan', () => {
  for (const sign of [-1, 1]) {
    const view = { ...camera, pitch: sign * 1.45 }
    for (const longitude of [-179.9, -100, -30, 30, 100, 179.9])
      for (const latitude of [85.3, 89.5, 89.9]) {
        const position = surfacePoint((longitude * Math.PI) / 180, (sign * latitude * Math.PI) / 180, 1)
        const [x, y] = project(position, view)
        const hit = pickSurface(x, y, view, 1, mesh)
        assert.ok(hit)
        const [hx, hy, hz] = hit.position
        close(hit.longitude, (Math.atan2(hx, hz) * 180) / Math.PI, 0.0001)
        const e2 = EARTH_FLATTENING * (2 - EARTH_FLATTENING)
        close(hit.latitude, (Math.atan2(hy / (1 - e2), Math.hypot(hx, hz)) * 180) / Math.PI, 0.0001)
      }
  }
})

test('raycasting follows displaced terrain and keeps negative-height oceans at sea level', () => {
  const hit = pickSurface(0, 0, camera, 1, createMesh(48, 24), { heightAt: () => 1000, exaggeration: 20 })
  assert.ok(hit)
  close(hit.position[2], 1 + 20000 / EARTH_AXIS)
  close(surfacePoint(0, 0, 1, -4000, 20)[2], 1)
})

test('display elevation levels converge to one height at each pole without changing interior samples', () => {
  const pixels = new Uint8ClampedArray(4 * 4 * 4)
  for (let i = 0; i < 16; i++) {
    const code = 32768 + i * 100
    pixels[i * 4] = code >> 8
    pixels[i * 4 + 1] = code & 255
    pixels[i * 4 + 2] = 128 + i
    pixels[i * 4 + 3] = 255
  }
  const levels = elevationPyramid({ pixels, width: 4, height: 4 })
  assert.equal(levels[0].values[5], 500)
  assert.equal(levels[0].coast[5], 1.25)
  for (const level of levels)
    for (const row of [0, level.height - 1]) {
      const heights = level.values.slice(row * level.width, (row + 1) * level.width)
      const coasts = level.coast.slice(row * level.width, (row + 1) * level.width)
      assert.equal(new Set(heights).size, 1)
      assert.equal(new Set(coasts).size, 1)
    }
})

test('isotropic polar elevation pyramids damp longitudinal spikes near poles without altering the equator', () => {
  const width = 64,
    height = 32
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const heightMetres = x % 2 === 0 ? 500 : -500
      const code = 32768 + heightMetres
      const i = (y * width + x) * 4
      pixels[i] = code >> 8
      pixels[i + 1] = code & 255
      pixels[i + 2] = 128
      pixels[i + 3] = 255
    }
  const levels = elevationPyramid({ pixels, width, height }, true)
  const eqRow = Math.floor(height / 2)
  assert.equal(levels[0].values[eqRow * width], 500)
  assert.equal(levels[0].values[eqRow * width + 1], -500)
  let polarPeak = 0
  for (let x = 0; x < width; x++) polarPeak = Math.max(polarPeak, Math.abs(levels[0].values[width + x]))
  assert.ok(polarPeak < 25, `expected high-frequency polar oscillation to be damped, got ${polarPeak}`)
})

test('ECT2 delta-shuffled payload round-trips signed elevations and packs rg8 levels', () => {
  const width = 4,
    height = 3
  const elevations = [-9000, -1200, 0, 5400, -300, 120, 8848, 21000, -50, 42, -1, 1000]
  const codes = elevations.map(h => h + 32768)
  const payload = new Uint8Array(width * height * 2)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const pred =
        y === 0 ? (x === 0 ? 32768 : codes[i - 1]) : x === 0 ? codes[i - width] : (codes[i - 1] + codes[i - width]) >> 1
      const delta = (codes[i] - pred) & 0xffff
      const signed = delta >= 32768 ? delta - 65536 : delta
      const zz = ((signed << 1) ^ (signed >> 31)) & 0xffff
      payload[i] = zz >> 8
      payload[width * height + i] = zz & 0xff
    }
  const terrain = decodeTerrainEct2Payload(width, height, 3, payload)
  assert.equal(terrain.stride, 2)
  assert.equal(terrain.polarSmoothed, true)
  for (let i = 0; i < elevations.length; i++) {
    const decoded = terrain.pixels[i * 2] * 256 + terrain.pixels[i * 2 + 1] - 32768
    assert.equal(decoded, elevations[i])
  }
  const packed = packRg8Levels(elevationPyramid(terrain, true))
  assert.equal(packed[0].data.byteLength, width * height * 2)
})

test('ECT2 delta-shuffled payload with coast channel round-trips elevation and coastline distance', () => {
  const width = 4,
    height = 3
  const elevations = [-4200, -100, 12, 850, -3500, -5, 40, 1200, -2800, 0, 180, 4000]
  const coasts = [4, 120, 132, 240, 8, 126, 136, 250, 12, 128, 144, 255]
  const codes = elevations.map(h => h + 32768)
  const count = width * height
  const payload = new Uint8Array(count * 3)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const pred =
        y === 0 ? (x === 0 ? 32768 : codes[i - 1]) : x === 0 ? codes[i - width] : (codes[i - 1] + codes[i - width]) >> 1
      const delta = (codes[i] - pred) & 0xffff
      const signed = delta >= 32768 ? delta - 65536 : delta
      const zz = ((signed << 1) ^ (signed >> 31)) & 0xffff
      payload[i] = zz >> 8
      payload[count + i] = zz & 0xff
      const cpred =
        y === 0
          ? x === 0
            ? 128
            : coasts[i - 1]
          : x === 0
            ? coasts[i - width]
            : (coasts[i - 1] + coasts[i - width]) >> 1
      const cdelta = (coasts[i] - cpred) & 0xff
      const csigned = cdelta >= 128 ? cdelta - 256 : cdelta
      payload[count * 2 + i] = ((csigned << 1) ^ (csigned >> 31)) & 0xff
    }
  const terrain = decodeTerrainEct2Payload(width, height, 5, payload)
  assert.equal(terrain.stride, 4)
  for (let i = 0; i < count; i++) {
    assert.equal(terrain.pixels[i * 4] * 256 + terrain.pixels[i * 4 + 1] - 32768, elevations[i])
    assert.equal(terrain.pixels[i * 4 + 2], coasts[i])
    assert.equal(terrain.pixels[i * 4 + 3], 255)
  }
})

test('raycasting selects the visible surface, misses the background and follows the morph', () => {
  for (const morph of [0, 0.25, 0.5, 0.75, 1]) {
    const hit = pickSurface(0, 0, camera, morph, mesh)
    assert.ok(hit)
    close(hit.latitude, 0)
    close(hit.longitude, 0)
    close(hit.position[2], morph)
    assert.equal(pickSurface(3, 3, camera, morph, mesh), null)
  }
})

test('rotated camera projection and picking agree at geographic mesh vertices', () => {
  for (const morph of [0, 0.4, 1]) {
    const view = { ...camera, yaw: 0.3, pitch: 0.2 }
    // These coordinates lie on actual mesh vertices, avoiding curved-surface approximation.
    const longitude = 30,
      latitude = 15
    const position = surfacePoint((longitude * Math.PI) / 180, (latitude * Math.PI) / 180, morph)
    const ndc = project(position, view)
    const hit = pickSurface(ndc[0], ndc[1], view, morph, mesh)
    assert.ok(hit)
    close(hit.longitude, longitude, 0.002)
    close(hit.latitude, latitude, 0.002)
  }
})

test('the production 4 km MODIS sinusoidal grid has exact tile and packed byte counts', () => {
  const tiles = worldTiles()
  assert.equal(tiles.length, 460)
  assert.equal(tiles.length * SIZE * SIZE, 35_550_640)
  assert.equal(tiles.length * SIZE * SIZE * 2, 71_101_280)
  close(CELL, 3999.822013548645, 0.000001)
  let valid = 0
  for (const { h, v } of tiles)
    for (let y = 0; y < SIZE; y++) {
      const limit = Math.PI * RADIUS * tileRow(v, y).cos
      for (let x = 0; x < SIZE; x++) if (Math.abs(tileX(h, x)) <= limit) valid++
    }
  assert.equal(valid, 31_881_940)
})

test('inspection addresses agree with native sinusoidal cell centers across latitudes', () => {
  for (const { h, v } of worldTiles()) {
    const y = 137,
      x = 139,
      { latitude, cos } = tileRow(v, y)
    const longitude = tileX(h, x) / (RADIUS * cos)
    if (Math.abs(longitude) > Math.PI) continue
    assert.deepEqual(cellAddress((longitude * 180) / Math.PI, (latitude * 180) / Math.PI), { h, v, x, y })
  }
  assert.equal(cellAddress(181, 0), null)
  assert.equal(cellAddress(0, 91), null)
  assert.equal(cellAddress(NaN, 0), null)
})

test('frozen solar coordinates agree with independent USNO celestial-navigation ephemerides', () => {
  // USNO celnav almanac_data: latitude = dec, east longitude = -gha.
  // https://aa.usno.navy.mil/api/celnav?date=2024-06-20&time=12:00:00&coords=0,0
  // The midnight case uses coords=0,180 so the service includes the Sun.
  const references: [string, number, number][] = [
    ['2000-01-01T12:00:00Z', -23.032432, 0.821285],
    ['2024-03-20T12:00:00Z', 0.146548, 1.827311],
    ['2024-06-20T12:00:00Z', 23.437767, 0.426478],
    ['2024-09-22T12:00:00Z', 0.01191, -1.868454],
    ['2024-12-21T12:00:00Z', -23.438352, -0.427155],
    ['2025-01-01T00:00:00Z', -22.998199, -179.139378],
  ]
  for (const [timestamp, latitude, longitude] of references) {
    const sun = solarPosition(new Date(timestamp))
    close(sun.latitude, latitude, 0.02)
    close(sun.longitude, longitude, 0.02)
    close(Math.hypot(...sun.direction), 1, 1e-12)
    close((Math.asin(sun.direction[1]) * 180) / Math.PI, latitude, 0.02)
    close((Math.atan2(sun.direction[0], sun.direction[2]) * 180) / Math.PI, longitude, 0.02)
  }
})

test('solar time uses the UTC instant, moves westward with time and rejects invalid dates', () => {
  assert.deepEqual(
    solarPosition(new Date('2024-06-20T12:00:00Z')),
    solarPosition(new Date('2024-06-20T09:00:00-03:00')),
  )
  const noon = solarPosition(new Date('2024-06-20T12:00:00Z'))
  const afternoon = solarPosition(new Date('2024-06-20T13:00:00Z'))
  close(afternoon.longitude - noon.longitude, -15, 0.02)
  assert.throws(() => solarPosition(new Date(NaN)), /Invalid solar timestamp/)
})

test('ECVT1 virtual tile pack headers and indices parse cleanly and reject invalid payloads', () => {
  const buf = new ArrayBuffer(16 + 2 * 12)
  const view = new DataView(buf)
  view.setUint32(0, 0x54564345, true) // 'ECVT'
  view.setUint32(4, 1, true)
  view.setUint32(8, 3, true)
  view.setUint32(12, 2, true)
  const k0 = tileKey(0, 20, 10)
  const k1 = tileKey(2, 140, 58)
  view.setUint32(16, k0, true)
  view.setUint32(20, 40, true)
  view.setUint32(24, 12345, true)
  view.setUint32(28, k1, true)
  view.setUint32(32, 12385, true)
  view.setUint32(36, 23456, true)

  const header = parseTilePackHeader(buf.slice(0, 16))
  assert.deepEqual(header, { packId: 3, count: 2 })
  const idx = parseTilePackIndex(buf.slice(16), header!.packId, header!.count)
  assert.deepEqual(idx.get(k0), { packId: 3, offset: 40, length: 12345 })
  assert.deepEqual(idx.get(k1), { packId: 3, offset: 12385, length: 23456 })
  assert.notEqual(planetTileKey(0, 1, 5, 3), planetTileKey(1, 1, 5, 3))

  const page = new Uint8Array(4 * 2 * 4)
  paintTilePageTable(page, 4, 2, [
    { level: 0, tx: 0, ty: 0 },
    { level: 1, tx: 1, ty: 0 },
  ])
  assert.deepEqual([...page.slice(0, 4)], [0, 0, 255, 255])
  assert.deepEqual([...page.slice(4, 8)], [1, 1, 0, 0])

  view.setUint32(0, 0xdeadbeef, true)
  assert.equal(parseTilePackHeader(buf.slice(0, 16)), null)
  assert.equal(parseTilePackHeader(new ArrayBuffer(8)), null)
})
