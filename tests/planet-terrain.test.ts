import { strict as assert } from 'node:assert'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { gunzipSync } from 'node:zlib'
import { decodeTerrain } from '../src/shader-test/terrain.ts'
import { buildPlanetTerrain, type TerrainBody } from '../src/workspace/planetTerrain.ts'
import { bodyRadius } from '../src/workspace/solarSystem.ts'

const radians = Math.PI / 180

function sceneTerrain(body: TerrainBody) {
  const url = new URL(`../public/textures/${body}-terrain.bin`, import.meta.url)
  const raw = Uint8Array.from(gunzipSync(readFileSync(url))).buffer
  return buildPlanetTerrain(body, decodeTerrain(raw))
}

test('the MOLA overview registers Olympus Mons in the west', () => {
  const mars = sceneTerrain('mars')
  const olympus = mars.elevationAt(-133.8 * radians, 18.65 * radians)
  assert.ok(olympus > 18000 && olympus < 22000)
  assert.ok(olympus - mars.elevationAt(133.8 * radians, 18.65 * radians) > 15000)
})

test('the LOLA overview joins the antimeridian and closes the poles', () => {
  const moon = sceneTerrain('moon')
  assert.equal(moon.elevationAt(-Math.PI, .23), moon.elevationAt(Math.PI, .23))
  assert.equal(moon.elevationAt(0, Math.PI / 2), moon.elevationAt(2, Math.PI / 2))
  assert.deepEqual(moon.positionAt(0, Math.PI / 2), moon.positionAt(2, Math.PI / 2))
})

test('the mesh and coordinates use the solar scene radius and displayed relief', () => {
  for (const [body, exaggeration] of [['moon', 3], ['mars', 4]] as const) {
    const mesh = sceneTerrain(body)
    assert.equal(mesh.radius, bodyRadius(body))
    const atPrimeMeridian = mesh.positionAt(0, 0)
    const expectedRadius = mesh.radius + mesh.elevationAt(0, 0) * exaggeration / 1000
    assert.equal(atPrimeMeridian[0], 0)
    assert.equal(atPrimeMeridian[1], 0)
    assert.ok(Math.abs(atPrimeMeridian[2] - expectedRadius) < .001)
  }
})
