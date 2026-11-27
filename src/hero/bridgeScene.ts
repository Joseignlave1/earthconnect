import { EARTH_AXIS, FOV, sphericalNormal, wgs84UnitPoint } from '../shader-test/geometry.ts'
import {
  add,
  cross,
  dot,
  geographicToWorld,
  normalized,
  scale,
  type SolarBodies,
  type SolarFrame,
} from '../workspace/solarCamera.ts'
import { bodyRadius, type BodyId, type BodyRotation, type BodyState, type Vec3 } from '../workspace/solarSystem.ts'
import type { BridgeEndpoints } from './bridgeOverlay.ts'
import { HERO_BRIDGE_SCORE_SHADER, HERO_MARS_SITE } from './editorialData.ts'

export { HERO_BRIDGE_SCORE_SHADER, HERO_MARS_SITE }

const DEG = Math.PI / 180
const EARTH_EQUATORIAL_KM = EARTH_AXIS / 1000

function geoDirection(latDeg: number, lonDeg: number): Vec3 {
  return sphericalNormal(lonDeg * DEG, latDeg * DEG)
}

function earthSurfaceUnit(latDeg: number, lonDeg: number): Vec3 {
  return wgs84UnitPoint(lonDeg * DEG, latDeg * DEG)
}

/**
 * Orient a planet so geographic `(latDeg, lonDeg)` points along `targetWorldDir`
 * with geographic north aligned toward `upHint`.
 */
function orientBodyPoint(latDeg: number, lonDeg: number, targetWorldDir: Vec3, upHint: Vec3): BodyRotation {
  const gFwd = geoDirection(latDeg, lonDeg)
  const gRight = normalized(cross([0, 1, 0], gFwd))
  const gUp = cross(gFwd, gRight)

  const wFwd = normalized(targetWorldDir)
  const wRight = normalized(cross(upHint, wFwd))
  const wUp = cross(wFwd, wRight)

  const mapGeo = (e: Vec3): Vec3 =>
    add(add(scale(wRight, dot(e, gRight)), scale(wUp, dot(e, gUp))), scale(wFwd, dot(e, gFwd)))

  return [mapGeo([0, 0, 1]), mapGeo([1, 0, 0]), mapGeo([0, 1, 0])]
}

/**
 * Base camera basis oriented toward the Scorpius/Sagittarius Milky Way plane
 * so the 8K NASA Deep Star Map sweeps diagonally behind the bridge.
 */
function baseCameraBasis(): { right: Vec3; up: Vec3; back: Vec3 } {
  // Aim near ecliptic longitude ~278°, latitude ~-8° (rich Milky Way dust lane).
  const forward = normalized([0.14, -0.98, -0.14])
  const back = scale(forward, -1)
  const rawUp: Vec3 = [0.24, 0.16, 0.96]
  const right = normalized(cross(rawUp, back))
  const up = cross(back, right)
  return { right, up, back }
}

const BASE_BASIS = baseCameraBasis()

function camToWorld(cx: number, cy: number, cz: number): Vec3 {
  return add(add(scale(BASE_BASIS.right, cx), scale(BASE_BASIS.up, cy)), scale(BASE_BASIS.back, cz))
}

function camDirToWorld(cx: number, cy: number, cz: number): Vec3 {
  return normalized(camToWorld(cx, cy, cz))
}

/**
 * Build an art-directed, faked-distance SolarBodies staging for "The Bridge":
 * - Mars looms large along the lower-left/bottom horizon with Jezero Crater on the
 *   sunlit-to-twilight crest facing upper-right.
 * - Earth floats across the dark starfield in the upper-right, oriented so the live
 *   GPU #1 peak analogue glows brightly on its twilight/night hemisphere.
 * - Moon sits quietly as a razor-sharp silver crescent further out in the upper-right.
 * - A single unified curated Sun sits off-frame to the right-back, carving grazing
 *   terminators across all three worlds simultaneously.
 */
export function buildBridgeBodies(
  aspect: number,
  earthTarget: { latitude: number; longitude: number } = { latitude: 30, longitude: 88 },
): SolarBodies {
  const narrow = aspect < 0.95
  const clampedAspect = Math.max(0.55, Math.min(2.2, aspect))
  const tanY = Math.tan(FOV / 2)
  const tanX = tanY * clampedAspect

  // Place Mars in the foreground lower-left/bottom.
  const marsDepth = narrow ? 12_600 : 9_900
  const marsNdcX = narrow ? -0.34 : -0.35
  const marsNdcY = narrow ? -0.84 : -0.85
  const marsCenter = camToWorld(marsNdcX * marsDepth * tanX, marsNdcY * marsDepth * tanY, -marsDepth)

  // Orient Mars so Jezero Crater sits cleanly on the upper-right sunlit-twilight shoulder
  // facing toward Earth and the camera.
  const marsSiteWorldDir = camDirToWorld(narrow ? 0.32 : 0.46, narrow ? 0.56 : 0.61, narrow ? 0.76 : 0.64)
  const marsUpHint = camDirToWorld(-0.18, 0.97, 0.12)
  const marsRotation = orientBodyPoint(HERO_MARS_SITE.latitude, HERO_MARS_SITE.longitude, marsSiteWorldDir, marsUpHint)

  // Place Earth in the upper-right mid-background at a faked distance where its
  // 3D WGS84 relief, Rayleigh crescent, and glowing HDR Magma heatmap read vividly.
  const earthDepth = narrow ? 58_000 : 45_500
  const earthNdcX = narrow ? 0.28 : 0.44
  const earthNdcY = narrow ? 0.16 : 0.22
  const earthCenter = camToWorld(earthNdcX * earthDepth * tanX, earthNdcY * earthDepth * tanY, -earthDepth)

  // Orient Earth so the live #1 analogue peak faces the camera slightly toward Mars
  // inside Earth's twilight/night hemisphere so its HDR Magma glow pops dramatically.
  const earthPeakWorldDir = camDirToWorld(-0.22, -0.06, 0.973)
  const earthUpHint = camDirToWorld(-0.14, 0.988, 0.04)
  const earthRotation = orientBodyPoint(earthTarget.latitude, earthTarget.longitude, earthPeakWorldDir, earthUpHint)

  // Place Moon quietly behind/above Earth to complete the Earth-Moon-Mars triad.
  const moonDepth = narrow ? 72_000 : 56_000
  const moonNdcX = narrow ? 0.68 : 0.79
  const moonNdcY = narrow ? 0.54 : 0.62
  const moonCenter = camToWorld(moonNdcX * moonDepth * tanX, moonNdcY * moonDepth * tanY, -moonDepth)
  const moonRotation = orientBodyPoint(-41.9, -11.2, camDirToWorld(-0.15, -0.1, 0.98), camDirToWorld(0.12, 0.99, 0))

  // Curated grazing-terminator Sun far off-frame to the right-back.
  const sunDir = camDirToWorld(0.79, 0.16, -0.59)
  const sunCenter = scale(sunDir, 150_000_000)

  // Park non-hero solar bodies far behind the camera so they stay out of the frame.
  const parked: Vec3 = camToWorld(0, 0, 500_000_000)
  const identity: BodyRotation = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]
  const body = (id: BodyId, center: Vec3 = parked, rotation: BodyRotation = identity): BodyState => ({
    center,
    radius: bodyRadius(id),
    rotation,
  })

  return {
    sun: body('sun', sunCenter),
    mercury: body('mercury'),
    venus: body('venus'),
    earth: body('earth', earthCenter, earthRotation),
    moon: body('moon', moonCenter, moonRotation),
    mars: body('mars', marsCenter, marsRotation),
    jupiter: body('jupiter'),
    saturn: body('saturn'),
    uranus: body('uranus'),
    neptune: body('neptune'),
  }
}

/**
 * Compute the live camera frame with gentle breathing drift and subtle pointer parallax.
 */
export function buildBridgeFrame(
  aspect: number,
  timeSeconds: number,
  pointer: { x: number; y: number },
  reducedMotion: boolean,
  scrollProgress = 0,
): SolarFrame {
  const driftX = reducedMotion ? 0 : Math.sin(timeSeconds * 0.16) * 110
  const driftY = reducedMotion ? 0 : Math.cos(timeSeconds * 0.12) * 55
  const px = reducedMotion ? 0 : pointer.x * 180
  const py = reducedMotion ? 0 : pointer.y * 110
  const scrollEase = scrollProgress * scrollProgress * (3 - 2 * scrollProgress)
  const scrollPitch = scrollEase * 0.24
  const eye = camToWorld(driftX + px, driftY + py - scrollEase * 720, 0)

  // Subtle counter-yaw/pitch so the parallax pivots gently around the mid-field between Mars and Earth,
  // and scrolling pans downward toward the rising black hole horizon.
  const yaw = reducedMotion ? 0 : (driftX + px) * -0.000018
  const pitch = (reducedMotion ? 0 : (driftY + py) * 0.000018) - scrollPitch
  const back = normalized(add(add(BASE_BASIS.back, scale(BASE_BASIS.right, yaw)), scale(BASE_BASIS.up, -pitch)))
  const right = normalized(cross(BASE_BASIS.up, back))
  const up = cross(back, right)

  return {
    eye,
    right,
    up,
    back,
    near: 120,
    aspect,
  }
}

/**
 * Project a rest-frame screen UV `(u, v)` anchored on the infinite celestial sphere
 * (`BASE_BASIS`) through the live camera `frame`, matching `solar-sky.wgsl` exactly.
 */
export function projectSkyUv(u: number, v: number, frame: SolarFrame): { u: number; v: number } | null {
  const tanY = Math.tan(FOV / 2)
  const tanX = tanY * frame.aspect
  const dir = camDirToWorld((u * 2 - 1) * tanX, (1 - v * 2) * tanY, -1)
  const cx = dot(dir, frame.right)
  const cy = dot(dir, frame.up)
  const cz = -dot(dir, frame.back)
  if (cz <= 0.02) return null
  return {
    u: (cx / (cz * tanX) + 1) * 0.5,
    v: (1 - cy / (cz * tanY)) * 0.5,
  }
}

/**
 * Project a rest-frame box (CSS px relative to the hero viewport) anchored on the
 * celestial sphere through the live camera `frame`, returning both the pixel-locked
 * live box and its CSS translation delta from rest.
 */
export function projectSkyBox(
  rest: { left: number; top: number; width: number; height: number },
  viewportW: number,
  viewportH: number,
  frame: SolarFrame,
): { left: number; top: number; width: number; height: number; dx: number; dy: number } | null {
  if (viewportW <= 1 || viewportH <= 1 || rest.width <= 1 || rest.height <= 1) return null
  const u0 = (rest.left + rest.width * 0.5) / viewportW
  const v0 = (rest.top + rest.height * 0.5) / viewportH
  const projected = projectSkyUv(u0, v0, frame)
  if (!projected) return null
  const dx = (projected.u - u0) * viewportW
  const dy = (projected.v - v0) * viewportH
  return {
    left: rest.left + dx,
    top: rest.top + dy,
    width: rest.width,
    height: rest.height,
    dx,
    dy,
  }
}

/**
 * Compute exact 3D world-space endpoints for The Bridge filament from Jezero Crater
 * on Mars to the live peak Earth analogue coordinate.
 */
export function bridgeEndpoints(
  bodies: SolarBodies,
  earthTarget: { latitude: number; longitude: number },
  marsDisplacedLocal?: Vec3 | null,
): BridgeEndpoints {
  const marsUnit = geoDirection(HERO_MARS_SITE.latitude, HERO_MARS_SITE.longitude)
  const marsLocal = marsDisplacedLocal ?? scale(marsUnit, bodies.mars.radius)
  const marsPoint = add(bodies.mars.center, geographicToWorld(bodies.mars, marsLocal))
  const marsNormal = normalized(geographicToWorld(bodies.mars, marsUnit))

  const earthLocal = scale(earthSurfaceUnit(earthTarget.latitude, earthTarget.longitude), EARTH_EQUATORIAL_KM)
  const earthUnit = geoDirection(earthTarget.latitude, earthTarget.longitude)
  const earthPoint = add(bodies.earth.center, geographicToWorld(bodies.earth, earthLocal))
  const earthNormal = normalized(geographicToWorld(bodies.earth, earthUnit))

  return { marsPoint, marsNormal, earthPoint, earthNormal }
}
