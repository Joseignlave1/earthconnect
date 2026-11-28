import {
  Body,
  HelioVector,
  MakeTime,
  PlanetOrbitalPeriod,
  Rotation_EQD_EQJ,
  Rotation_EQJ_ECL,
  RotationAxis,
  SiderealTime,
} from 'astronomy-engine'

import { vec3Dot, type ReadonlyVec3 } from '../shader-test/vec3.ts'

export type Vec3 = ReadonlyVec3

/** Body-fixed +X (0° E), +Y (90° E), and +Z (north) as columns in world space. */
export type BodyRotation = readonly [Vec3, Vec3, Vec3]

// Mean radii in km for spherical body rendering (IAU nominal solar radius).
const bodies = {
  sun: [Body.Sun, 695700],
  mercury: [Body.Mercury, 2439.7],
  venus: [Body.Venus, 6051.8],
  earth: [Body.Earth, 6371.0],
  moon: [Body.Moon, 1737.4],
  mars: [Body.Mars, 3389.5],
  jupiter: [Body.Jupiter, 69911],
  saturn: [Body.Saturn, 58232],
  uranus: [Body.Uranus, 25362],
  neptune: [Body.Neptune, 24622],
} as const

export type BodyId = keyof typeof bodies

export function bodyRadius(id: BodyId): number {
  return bodies[id][1]
}

export interface BodyState {
  /** Heliocentric position in km, in the J2000 mean ecliptic and equinox frame. */
  center: Vec3
  radius: number
  rotation: BodyRotation
}

const kilometersPerAu = 149597870.7
const radiansPerDegree = Math.PI / 180
const equatorialToEcliptic = Rotation_EQJ_ECL().rot as unknown as BodyRotation

/** Rotate a body-fixed direction into J2000 ecliptic axes. */
export function bodyToWorld([x, y, z]: BodyRotation, direction: Vec3): Vec3 {
  return [
    x[0] * direction[0] + y[0] * direction[1] + z[0] * direction[2],
    x[1] * direction[0] + y[1] * direction[1] + z[1] * direction[2],
    x[2] * direction[0] + y[2] * direction[1] + z[2] * direction[2],
  ]
}

/** Rotate a J2000 ecliptic direction into the body's geographic axes. */
export function worldToBody([x, y, z]: BodyRotation, direction: Vec3): Vec3 {
  return [vec3Dot(x, direction), vec3Dot(y, direction), vec3Dot(z, direction)]
}

type AstroTime = ReturnType<typeof MakeTime>

function bodyRotation(body: Body, time: AstroTime): BodyRotation {
  if (body === Body.Earth) {
    // Greenwich apparent sidereal time measures the prime meridian from
    // the true equinox of date. Then reverse nutation and precession.
    const greenwich = SiderealTime(time) * 15 * radiansPerDegree
    const cos = Math.cos(greenwich)
    const sin = Math.sin(greenwich)
    const toJ2000 = Rotation_EQD_EQJ(time).rot as unknown as BodyRotation
    const toEcliptic = (direction: Vec3) => bodyToWorld(equatorialToEcliptic, bodyToWorld(toJ2000, direction))
    return [toEcliptic([cos, sin, 0]), toEcliptic([-sin, cos, 0]), toEcliptic([0, 0, 1])]
  }

  // The IAU prime meridian angle W advances eastward from the node where
  // the body's equator crosses northward through the J2000 Earth equator.
  // The Moon's IAU orientation includes periodic physical libration terms.
  const { ra, dec, spin, north } = RotationAxis(body, time)
  const rightAscension = ra * 15 * radiansPerDegree
  const declination = dec * radiansPerDegree
  const angle = (spin % 360) * radiansPerDegree
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  const node: Vec3 = [-Math.sin(rightAscension), Math.cos(rightAscension), 0]
  const quadrant: Vec3 = [
    -Math.sin(declination) * Math.cos(rightAscension),
    -Math.sin(declination) * Math.sin(rightAscension),
    Math.cos(declination),
  ]
  const equator: BodyRotation = [node, quadrant, [north.x, north.y, north.z]]
  return [
    bodyToWorld(equatorialToEcliptic, bodyToWorld(equator, [cos, sin, 0])),
    bodyToWorld(equatorialToEcliptic, bodyToWorld(equator, [-sin, cos, 0])),
    bodyToWorld(equatorialToEcliptic, equator[2]),
  ]
}

function heliocentricPosition(body: Body, time: AstroTime): Vec3 {
  const { x, y, z } = HelioVector(body, time)
  return bodyToWorld(equatorialToEcliptic, [x * kilometersPerAu, y * kilometersPerAu, z * kilometersPerAu])
}

/** Sample one sidereal period, centered on the given date, in heliocentric km. */
export function orbitPolyline(id: BodyId, date: Date): readonly Vec3[] {
  if (id === 'sun' || id === 'moon') return []
  const [body] = bodies[id]
  const time = MakeTime(date)
  const period = PlanetOrbitalPeriod(body)
  const segments = 180
  return Array.from({ length: segments + 1 }, (_, index) =>
    heliocentricPosition(body, time.AddDays(period * (index / segments - 0.5))),
  )
}

/** Simultaneous geometric states at the given UTC instant, before light-time correction. */
export function solarSystemAt(date: Date): Record<BodyId, BodyState> {
  const time = MakeTime(date)
  const states = {} as Record<BodyId, BodyState>
  for (const id of Object.keys(bodies) as BodyId[]) {
    const [body, radius] = bodies[id]
    states[id] = {
      center: heliocentricPosition(body, time),
      radius,
      rotation: bodyRotation(body, time),
    }
  }
  return states
}
