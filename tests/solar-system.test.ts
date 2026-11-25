import { strict as assert } from 'node:assert'
import test from 'node:test'
import { bodyToWorld, orbitPolyline, solarSystemAt, worldToBody, type Vec3 } from '../src/workspace/solarSystem.ts'

const subtract = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const radiansToDegrees = 180 / Math.PI
const longitude = ([x, y]: Vec3) => Math.atan2(y, x) * radiansToDegrees
const latitude = ([x, y, z]: Vec3) => Math.atan2(z, Math.hypot(x, y)) * radiansToDegrees

test('heliocentric Mars state agrees with JPL Horizons at J2000', () => {
  // Horizons 499 relative to 500@10, J2000 ecliptic, geometric KM-S.
  // JD 2451545.0 TDB corresponds to 2000-01-01 11:58:55.816 UTC within milliseconds.
  const state = solarSystemAt(new Date('2000-01-01T11:58:55.816Z'))
  const horizons: Vec3 = [208048140.641842, -2007052.628025221, -5156288.959268022]
  assert.ok(Math.hypot(...subtract(state.mars.center, horizons)) < 3500)
  assert.deepEqual(state.sun.center, [0, 0, 0])
  assert.equal(Object.keys(state).length, 10)
})

test('the Sun is on the terrestrial equator and near 135° E at the March 2024 equinox', () => {
  const { sun, earth } = solarSystemAt(new Date('2024-03-20T03:06:00Z'))
  const sunFromEarth = worldToBody(earth.rotation, subtract(sun.center, earth.center))
  assert.ok(Math.abs(latitude(sunFromEarth)) < 0.01)
  // 03:06 UTC puts mean solar noon at 133.5° E. The equation of time
  // was about −7.4 minutes, putting apparent solar noon 1.85° farther east.
  assert.ok(Math.abs(longitude(sunFromEarth) - 135.35) < 0.05)
})

test('the Moon stays synchronously oriented while the sub-Earth point librates', () => {
  const longitudes = [
    '2024-04-08T18:17:00Z',
    '2024-04-15T18:17:00Z',
    '2024-04-22T18:17:00Z',
    '2024-04-29T18:17:00Z',
  ].map(date => {
    const { earth, moon } = solarSystemAt(new Date(date))
    const direction = subtract(earth.center, moon.center)
    const earthFromMoon = worldToBody(moon.rotation, direction)
    assert.ok(Math.abs(latitude(earthFromMoon)) < 7.5)
    assert.ok(Math.hypot(...subtract(bodyToWorld(moon.rotation, earthFromMoon), direction)) < 1e-9)
    return longitude(earthFromMoon)
  })
  assert.ok(longitudes.every(value => Math.abs(value) < 9))
  assert.ok(Math.max(...longitudes) - Math.min(...longitudes) > 8)
})

test('Mars rotates eastward by one turn minus 9.1° per Earth day', () => {
  const earlier = solarSystemAt(new Date('2026-11-17T00:00:00Z')).mars.rotation
  const later = solarSystemAt(new Date('2026-11-18T00:00:00Z')).mars.rotation
  const laterPrimeMeridian = worldToBody(earlier, later[0])
  assert.ok(Math.abs(longitude(laterPrimeMeridian) + 9.108) < 0.001)
  assert.ok(Math.abs(latitude(laterPrimeMeridian)) < 0.001)
})

test('planet orbit samples contain the state at the reference epoch', () => {
  const date = new Date('2024-04-08T18:17:00Z')
  const points = orbitPolyline('earth', date)
  assert.equal(points.length, 181)
  assert.deepEqual(points[90], solarSystemAt(date).earth.center)
  assert.deepEqual(orbitPolyline('moon', date), [])
})
