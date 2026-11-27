import assert from 'node:assert/strict'
import test from 'node:test'
import { cameraBasis, cameraMatrix } from '../src/shader-test/geometry.ts'
import {
  createPatchMesh,
  CUBE_INSTANCE_FLOATS,
  EARTH_INSTANCE_FLOATS,
  EarthQuadtree,
  MAX_PATCH_INSTANCES,
  PATCH_SEGMENTS,
  PlanetCubeQuadtree,
} from '../src/shader-test/quadtreeLod.ts'
import type { Terrain } from '../src/shader-test/terrain.ts'

function makeSyntheticTerrain(width: number, height: number, elevFn: (lon: number, lat: number) => number): Terrain {
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y++) {
    const lat = (0.5 - (y + 0.5) / height) * Math.PI
    for (let x = 0; x < width; x++) {
      const lon = ((x + 0.5) / width - 0.5) * 2 * Math.PI
      const encoded = Math.max(0, Math.min(65535, Math.round(elevFn(lon, lat) + 32768)))
      const o = (y * width + x) * 4
      pixels[o] = (encoded >> 8) & 0xff
      pixels[o + 1] = encoded & 0xff
      pixels[o + 2] = 0
      pixels[o + 3] = 255
    }
  }
  return { pixels, width, height, stride: 4 }
}

test('patch mesh has valid index bounds, inner grid + 4 skirts, and CCW outward winding', () => {
  const mesh = createPatchMesh(PATCH_SEGMENTS)
  assert.equal(mesh.vertexCount, (PATCH_SEGMENTS + 1) ** 2 + (PATCH_SEGMENTS + 1) * 4)
  assert.equal(mesh.indexCount, (PATCH_SEGMENTS * PATCH_SEGMENTS * 2 + PATCH_SEGMENTS * 4 * 2) * 3)
  for (let i = 0; i < mesh.indexCount; i++) {
    assert.ok(mesh.indices[i] < mesh.vertexCount, `index ${mesh.indices[i]} out of range`)
  }
  // Every inner triangle (first 512 tris) must have positive +Z winding in (s, t) space.
  const innerTris = PATCH_SEGMENTS * PATCH_SEGMENTS * 2
  for (let t = 0; t < innerTris; t++) {
    const a = mesh.indices[t * 3] * 3
    const b = mesh.indices[t * 3 + 1] * 3
    const c = mesh.indices[t * 3 + 2] * 3
    const ux = mesh.vertices[b] - mesh.vertices[a]
    const uy = mesh.vertices[b + 1] - mesh.vertices[a + 1]
    const vx = mesh.vertices[c] - mesh.vertices[a]
    const vy = mesh.vertices[c + 1] - mesh.vertices[a + 1]
    assert.ok(ux * vy - uy * vx > 0, `inner triangle ${t} is not CCW`)
  }
})

test('Earth quadtree culls back hemisphere and refines mountain regions more than flat ocean', () => {
  const flatOcean = makeSyntheticTerrain(128, 64, () => -2000)
  // High-frequency rugged alpine range across the visible hemisphere
  const ruggedLand = makeSyntheticTerrain(128, 64, (lon, lat) =>
    Math.abs(lon) < 1.0 && Math.abs(lat) < 0.8
      ? Math.max(0, 6500 * Math.abs(Math.sin(lon * 24) * Math.cos(lat * 20)))
      : -2000,
  )
  const oceanTree = EarthQuadtree.fromTerrain(flatOcean)
  const mountainTree = EarthQuadtree.fromTerrain(ruggedLand)
  const out = new Float32Array(MAX_PATCH_INSTANCES * EARTH_INSTANCE_FLOATS)

  const cam = { yaw: 0, pitch: 0, distance: 3.0, aspect: 1.6 }
  const mat = cameraMatrix(cam)
  const eye = cameraBasis(cam).eye
  const oceanCount = oceanTree.selectPatches(out, mat, eye, 1440, 900, 1, 22)
  const mountainCount = mountainTree.selectPatches(out, mat, eye, 1440, 900, 1, 22)

  assert.ok(oceanCount > 0 && oceanCount < 64, `expected coarse ocean patches, got ${oceanCount}`)
  assert.ok(
    mountainCount > oceanCount,
    `expected mountains (${mountainCount}) to subdivide more than flat ocean (${oceanCount})`,
  )
})

test('Cube-sphere quadtree culls far side and subdivides closer views', () => {
  const terrain = makeSyntheticTerrain(128, 64, (lon, lat) => {
    const d2 = lon * lon + lat * lat
    return d2 < 0.25 ? 12000 * Math.exp(-d2 * 30) : 0
  })
  const radiusKm = 3389.5
  const tree = PlanetCubeQuadtree.fromTerrain(terrain, radiusKm)
  const out = new Float32Array(MAX_PATCH_INSTANCES * CUBE_INSTANCE_FLOATS)

  const camFar = { yaw: 0, pitch: 0, distance: radiusKm * 6, aspect: 1.6, near: 100 }
  const farCount = tree.selectPatches(out, cameraMatrix(camFar), cameraBasis(camFar).eye, 1440, 900, 3, 1)
  const camNear = { yaw: 0, pitch: 0, distance: radiusKm * 1.6, aspect: 1.6, near: 10 }
  const nearCount = tree.selectPatches(out, cameraMatrix(camNear), cameraBasis(camNear).eye, 1440, 900, 3, 1)

  assert.ok(farCount > 0 && farCount <= 32, `expected few patches far away, got ${farCount}`)
  assert.ok(nearCount > farCount, `expected near view (${nearCount}) to refine more than far view (${farCount})`)
})

test('Earth quadtree covers 100% of UV area even when hitting MAX_PATCH_INSTANCES', () => {
  const rough = makeSyntheticTerrain(256, 128, (lon, lat) => 8000 * Math.abs(Math.sin(lon * 37) * Math.cos(lat * 29)))
  const tree = EarthQuadtree.fromTerrain(rough)
  const out = new Float32Array(MAX_PATCH_INSTANCES * EARTH_INSTANCE_FLOATS)
  const cam = { yaw: 0, pitch: 0, distance: 3.6, aspect: 2.0 }
  const count = tree.selectPatches(out, cameraMatrix(cam), cameraBasis(cam).eye, 8192, 4096, 0, 14)

  assert.ok(
    count >= MAX_PATCH_INSTANCES - 4 && count <= MAX_PATCH_INSTANCES,
    `expected cap-bounded count, got ${count}`,
  )
  let uvArea = 0
  for (let i = 0; i < count; i++) {
    uvArea += out[i * EARTH_INSTANCE_FLOATS + 2] * out[i * EARTH_INSTANCE_FLOATS + 3]
  }
  assert.ok(Math.abs(uvArea - 1.0) < 1e-6, `expected full UV coverage (1.0), got ${uvArea}`)
})
