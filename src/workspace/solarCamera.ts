import { cameraBasis, FOV, type Camera, type Rotation } from '../shader-test/geometry.ts'
import { bodyToWorld, worldToBody, type BodyId, type BodyState, type Vec3 } from './solarSystem.ts'
import { baselineProgress, flightDuration, smootherStep } from './flightEasing.ts'

export type SolarFocus = 'earth' | 'moon' | 'mars' | 'system'
export type SourceBody = Extract<SolarFocus, 'moon' | 'mars'>
export type SolarBodies = Record<BodyId, BodyState>
type Basis = { right: Vec3; up: Vec3; back: Vec3 }
export type SolarFrame = Basis & { eye: Vec3; near: number; aspect: number }
type MutableVec = [number, number, number]
const coordinateAxes: readonly Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]
const orbitNorth: Vec3 = [0, 0, 1]
// Frame Mars's ~249 million km aphelion inside the shorter axis, leaving room
// for the navigation and readouts over the edges of the viewport.
const innerSystemRadius = 310_000_000

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const subtract = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const scale = (a: Vec3, n: number): Vec3 => [a[0] * n, a[1] * n, a[2] * n]
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export const length = (v: Vec3) => Math.hypot(...v)
export const normalized = (v: Vec3) => scale(v, 1 / Math.max(1e-30, length(v)))
const mix = (a: Vec3, b: Vec3, t: number) => add(scale(a, 1 - t), scale(b, t))

export function geographicToWorld(body: BodyState, [x, y, z]: Vec3): Vec3 {
  return bodyToWorld(body.rotation, [z, x, y])
}

export function worldToGeographic(body: BodyState, direction: Vec3): MutableVec {
  const [z, x, y] = worldToBody(body.rotation, direction)
  return [x, y, z]
}

export function toView(frame: SolarFrame, value: Vec3): Vec3 {
  return [dot(value, frame.right), dot(value, frame.up), dot(value, frame.back)]
}

// Translation is reduced in double precision before going to GPU; even a
// terrain flyby still sees kilometre-local positions after AU-scale travel.
export function relativeToEye(frame: SolarFrame, center: Vec3) {
  return toView(frame, subtract(center, frame.eye))
}

export function sceneRay(frame: SolarFrame, nx: number, ny: number) {
  const tangent = Math.tan(FOV / 2)
  return normalized(
    add(scale(frame.back, -1), add(scale(frame.right, nx * tangent * frame.aspect), scale(frame.up, ny * tangent))),
  )
}

export function bodyMatrix(frame: SolarFrame, body: BodyState, unit: number) {
  const f = 1 / Math.tan(FOV / 2)
  const view = coordinateAxes.map(axis => toView(frame, scale(geographicToWorld(body, axis), unit)))
  view.push(relativeToEye(frame, body.center))
  const matrix = new Float32Array(16)
  for (const [column, [x, y, z]] of view.entries()) {
    const offset = column * 4
    // Infinite reverse-Z with native WebGPU depth.
    matrix[offset] = (x * f) / frame.aspect
    matrix[offset + 1] = y * f
    matrix[offset + 3] = -z
  }
  matrix[14] = frame.near
  return matrix
}

export function raySphere(eye: Vec3, direction: Vec3, center: Vec3, radius: number) {
  const delta = subtract(center, eye)
  const middle = dot(delta, direction)
  if (middle <= 0) return Infinity
  // Closest approach is computed directly to avoid subtracting two very
  // large squared distances when aiming at a small, remote body.
  const offset = subtract(delta, scale(direction, middle))
  const gap = radius ** 2 - dot(offset, offset)
  return gap >= 0 ? middle - Math.sqrt(gap) : Infinity
}

const clearTerrain: Partial<Record<BodyId, number>> = { earth: 188, moon: 45, mars: 100 }
const sunClearance = 3_000_000
const unit = (n: number) => Math.max(0, Math.min(1, n))

function projectOntoPlane(direction: Vec3, normal: Vec3) {
  return subtract(direction, scale(normal, dot(direction, normal)))
}

function rotationBetween(from: Vec3, to: Vec3, hint: Vec3) {
  const tangent = cross(from, to)
  const cosine = Math.max(-1, Math.min(1, dot(from, to)))
  const sine = length(tangent)
  const projected = projectOntoPlane(hint, from)
  const fallback =
    length(projected) > 1e-6 ? projected : cross(from, Math.abs(from[2]) < 0.8 ? orbitNorth : coordinateAxes[0])
  return { axis: normalized(sine > 1e-8 ? tangent : fallback), angle: Math.atan2(sine, cosine) }
}

function rotate(vector: Vec3, axis: Vec3, angle: number): Vec3 {
  const cosine = Math.cos(angle)
  return add(
    add(scale(vector, cosine), scale(cross(axis, vector), Math.sin(angle))),
    scale(axis, dot(axis, vector) * (1 - cosine)),
  )
}

function look(basis: Basis, back: Vec3): Basis {
  const { axis, angle } = rotationBetween(basis.back, back, basis.up)
  // Parallel transport the existing horizon instead of inventing a roll
  // or using a global up axis that becomes singular above a pole.
  const right = normalized(cross(rotate(basis.up, axis, angle), back))
  return { right, up: cross(back, right), back }
}

type PathPiece = { a: Vec3; b: Vec3; bend?: Vec3 }
type PathSample = { piece: PathPiece; at: number; distance: number; travel: number; basis: Basis }
type FlightPlan = {
  samples: PathSample[]
  arrival: SolarFrame
  radius: number
  target?: BodyState
  turn: ReturnType<typeof rotationBetween>
  firstTarget: Vec3
  alignmentDistance: number
}
type Flight = { from: SolarFrame; start: number; instant: boolean; plan?: FlightPlan }

function pointOn(piece: PathPiece, t: number): Vec3 {
  const { a, b, bend } = piece
  return bend ? mix(mix(a, bend, t), mix(bend, b, t), t) : mix(a, b, t)
}

function closestOnLeg(a: Vec3, b: Vec3, center: Vec3) {
  const ray = subtract(b, a)
  const fraction = unit(dot(subtract(center, a), ray) / Math.max(1e-18, dot(ray, ray)))
  return add(a, scale(ray, fraction))
}

function clearance(id: BodyId, body: BodyState) {
  return id === 'sun' ? sunClearance : body.radius + (clearTerrain[id] ?? 0)
}

function legIsClear(a: Vec3, b: Vec3, body: BodyState, radius: number) {
  const allowed = Math.min(radius, length(subtract(a, body.center)), length(subtract(b, body.center)))
  return length(subtract(closestOnLeg(a, b, body.center), body.center)) >= allowed - 0.001
}

function tangentDetour(a: Vec3, b: Vec3, body: BodyState, radius: number) {
  const from = subtract(a, body.center),
    to = subtract(b, body.center)
  const first = normalized(from),
    last = normalized(to)
  const nearest = Math.min(length(from), length(to))
  const shell = Math.min(radius * 1.12, (nearest + radius) * 0.5)
  const { axis, angle } = rotationBetween(first, last, orbitNorth)
  const start = Math.acos(unit(shell / length(from)))
  const end = angle - Math.acos(unit(shell / length(to)))
  const arc = Math.max(0, end - start)
  // Short chords around the tangent arc stay outside the physical shell.
  const maxAngle = Math.min(0.12, 1.5 * Math.acos(unit(radius / shell)))
  const steps = Math.max(1, Math.ceil(arc / Math.max(0.008, maxAngle)))
  return Array.from({ length: steps + 1 }, (_, i) =>
    add(body.center, scale(rotate(first, axis, start + (arc * i) / steps), shell)),
  )
}

function routePastBodies(a: Vec3, b: Vec3, bodies: SolarBodies, focus: SolarFocus) {
  const points = [a, b]
  const obstacles = Object.entries(bodies) as [BodyId, BodyState][]
  // A short tangent arc is needed only for a genuine line-of-travel blocker.
  // Recheck its edges against the other bodies as well.
  for (let i = 1, detours = 0; i < points.length && detours < 12; ) {
    const first = points[i - 1],
      last = points[i]
    const blocker = obstacles.find(([id, body]) => id !== focus && !legIsClear(first, last, body, clearance(id, body)))
    if (blocker) {
      const [id, body] = blocker
      points.splice(i, 0, ...tangentDetour(first, last, body, clearance(id, body)))
      detours++
    } else i++
  }
  return points
}

function roundedPath(points: Vec3[]): PathPiece[] {
  const pieces: PathPiece[] = []
  let prior = points[0]
  for (let i = 1; i < points.length - 1; i++) {
    const point = points[i]
    const left = subtract(points[i - 1], point),
      right = subtract(points[i + 1], point)
    const trim = Math.min(length(left), length(right)) * 0.28
    const entry = add(point, scale(normalized(left), trim))
    const exit = add(point, scale(normalized(right), trim))
    pieces.push({ a: prior, b: entry }, { a: entry, b: exit, bend: point })
    prior = exit
  }
  pieces.push({ a: prior, b: points.at(-1)! })
  return pieces
}

function sightline(eye: Vec3, from: SolarFrame, plan: FlightPlan, distance: number) {
  const target = plan.target ? normalized(subtract(eye, plan.target.center)) : plan.arrival.back
  if (distance >= plan.alignmentDistance) return target
  // Travel and aim share the same progress. Use a fixed initial turn axis
  // so that a target directly behind the viewer cannot flip the turn.
  const amount = smootherStep(unit(distance / plan.alignmentDistance))
  const turned = rotate(from.back, plan.turn.axis, plan.turn.angle * amount)
  const drift = rotationBetween(plan.firstTarget, target, from.up)
  return normalized(rotate(turned, drift.axis, drift.angle * amount))
}

function imageOfDeparture(eye: Vec3, basis: Basis, body: BodyState, aspect: number) {
  const delta = subtract(body.center, eye)
  const depth = -dot(delta, basis.back)
  const horizontal = dot(delta, basis.right)
  const x = Math.atan2(horizontal, depth)
  const y = Math.atan2(dot(delta, basis.up), Math.hypot(horizontal, depth))
  const radius = Math.asin(Math.min(1, body.radius / length(delta)))
  const tan = Math.tan(FOV / 2)
  const edge = Math.max((Math.abs(x) - radius) / Math.atan(tan * aspect), (Math.abs(y) - radius) / (FOV / 2))
  // Fade the influence beyond the frame instead of releasing the camera
  // the instant the last visible point of the source crosses the edge.
  const visibility = Math.exp(-((Math.max(0, edge - 0.8) / 1.15) ** 2))
  return { x, y, radius, visibility }
}

function angularDistance(a: Vec3, b: Vec3) {
  return Math.atan2(length(cross(a, b)), dot(a, b))
}

function paceFlight(samples: PathSample[], motion: number[]) {
  // Spread visual effort slightly in path space. A rapidly moving limb
  // starts slowing the camera before it hits the edge, and stops affecting
  // the pace gradually after it leaves. Both passes use the same distance
  // coordinate, so neither introduces a second clock.
  const spread = 0.1
  const density = new Float64Array(samples.length)
  function blend(value: number, step: number, change: number) {
    return step > 1e-12 ? value + (change / step - value) * -Math.expm1(-step / spread) : value
  }
  let filtered = 0
  for (let i = 1; i < samples.length; i++) {
    filtered = blend(filtered, samples[i].distance - samples[i - 1].distance, motion[i])
    density[i] = filtered * 0.5
  }
  filtered = 0
  for (let i = samples.length - 2; i >= 0; i--) {
    filtered = blend(filtered, samples[i + 1].distance - samples[i].distance, motion[i + 1])
    density[i] += filtered * 0.5
  }
  for (let i = 1; i < samples.length; i++) {
    const step = samples[i].distance - samples[i - 1].distance
    samples[i].travel = samples[i - 1].travel + step * (1 + (density[i - 1] + density[i]) * 0.5)
  }
}

function planFlight(
  bodies: SolarBodies,
  from: SolarFrame,
  focus: SolarFocus,
  to: SolarFrame,
  distance: number,
): FlightPlan {
  const target = focus === 'system' ? undefined : bodies[focus]
  const directionTarget = target?.center ?? to.eye
  const nearby = (Object.entries(bodies) as [BodyId, BodyState][]).find(
    ([id, body]) => id !== focus && id !== 'sun' && length(subtract(from.eye, body.center)) < body.radius * 25,
  )
  // Plan through the destination center, then stop at the saved zoom along
  // the final ray. No saved bearing participates in a body-to-body flight.
  const points = routePastBodies(from.eye, directionTarget, bodies, focus)
  const entrance = normalized(subtract(points.at(-2)!, directionTarget))
  const back = target && length(entrance) > 0 ? entrance : to.back
  const eye = target ? add(target.center, scale(back, distance)) : to.eye
  points[points.length - 1] = eye
  const pieces = roundedPath(points)
  const arrival = { ...look(from, back), eye, near: to.near, aspect: to.aspect }
  const firstTarget = target ? normalized(subtract(from.eye, target.center)) : back
  const turn = rotationBetween(from.back, firstTarget, from.up)
  const plan: FlightPlan = {
    samples: [],
    arrival,
    radius: distance,
    target,
    turn,
    firstTarget,
    alignmentDistance: turn.angle < 1e-5 ? 0 : 0.18 + turn.angle * 0.19,
  }
  const targetScale = target?.radius ?? 25_000_000
  const totalKm = points.slice(1).reduce((sum, point, i) => sum + length(subtract(point, points[i])), 1)
  let optical = 0
  let last: Vec3 = from.eye
  let departedRange = 0
  function store(piece: PathPiece, at: number, next: Vec3) {
    const midpoint = mix(last, next, 0.5)
    const range = length(subtract(midpoint, target?.center ?? eye))
    // Logarithmic distance near the destination makes its growth legible.
    // Quiet deep space contributes less than the nearby departure and approach.
    const destination = 1 / (Math.max(range, targetScale) * Math.sqrt(1 + range / (150 * targetScale)))
    const source = nearby?.[1]
    // A flyby past the departure limb cannot reintroduce a slowdown after
    // the camera has already gained that distance from its starting world.
    if (source) departedRange = Math.max(departedRange, length(subtract(midpoint, source.center)) / source.radius)
    const departureWeight = source
      ? 0.58 / (source.radius * Math.max(1, departedRange) * (1 + (departedRange / 12) ** 0.37))
      : 0
    optical += length(subtract(next, last)) * (destination + departureWeight + 0.25 / totalKm)
    last = next
    plan.samples.push({ piece, at, distance: optical, travel: optical, basis: from })
  }
  function subdivide(piece: PathPiece, a: number, b: number, u: Vec3, v: Vec3, depth: number) {
    const segment = length(subtract(v, u))
    const center = target?.center ?? eye
    const closeTarget = Math.max(targetScale, length(subtract(closestOnLeg(u, v, center), center)))
    const source = nearby?.[1]
    const closeSource = source
      ? Math.max(source.radius, length(subtract(closestOnLeg(u, v, source.center), source.center)))
      : Infinity
    // The first long cruise chord can pass near the origin even with the
    // destination an AU away. Resolve both ends of the optical weight.
    if (depth < 24 && (segment > Math.min(closeTarget, closeSource) * 0.044 || (piece.bend && b - a > 0.03125))) {
      const middle = (a + b) / 2
      const point = pointOn(piece, middle)
      subdivide(piece, a, middle, u, point, depth + 1)
      subdivide(piece, middle, b, point, v, depth + 1)
    } else store(piece, b, v)
  }
  for (const piece of pieces) {
    // A zero-distance knot keeps the lookup on the right curve at a join.
    store(piece, 0, piece.a)
    subdivide(piece, 0, 1, piece.a, piece.b, 0)
  }
  plan.alignmentDistance = Math.min(plan.alignmentDistance, optical * 0.45)
  let basis: Basis = from
  const trackedBody = nearby?.[1]
  let foreground = trackedBody && imageOfDeparture(from.eye, from, trackedBody, from.aspect)
  const motion: number[] = []
  for (const sample of plan.samples) {
    const at = pointOn(sample.piece, sample.at)
    const next = look(basis, sightline(at, from, plan, sample.distance))
    const image = trackedBody && imageOfDeparture(at, next, trackedBody, from.aspect)
    // Share the optical budget with the actual turn and foreground movement.
    // Near the limb a small world-space step can shift the globe across a
    // large part of the frame; its departure should not become a snap.
    const turning = Math.max(angularDistance(basis.back, next.back), angularDistance(basis.up, next.up))
    let slip = 0
    if (image && foreground) {
      const horizontal = Math.abs(image.x - foreground.x)
      const x = Math.min(horizontal, 2 * Math.PI - horizontal)
      slip =
        (Math.hypot(x, image.y - foreground.y, image.radius - foreground.radius) *
          (image.visibility + foreground.visibility)) /
        2
    }
    motion.push(0.8 * turning + 1.5 * slip)
    foreground = image
    basis = next
    sample.basis = basis
  }
  paceFlight(plan.samples, motion)
  plan.arrival = { ...look(basis, back), eye, near: to.near, aspect: to.aspect }
  return plan
}

function fly(from: SolarFrame, plan: FlightPlan, progress: number, aspect: number, distance: number) {
  const samples = plan.samples
  const traveled = samples.at(-1)!.travel * progress
  let low = 0,
    high = samples.length - 1
  while (low + 1 < high) {
    const middle = (low + high) >>> 1
    if (samples[middle].travel <= traveled) low = middle
    else high = middle
  }
  const a = samples[low],
    b = samples[high]
  const fraction = unit((traveled - a.travel) / Math.max(1e-16, b.travel - a.travel))
  const position = pointOn(b.piece, a.piece === b.piece ? a.at + (b.at - a.at) * fraction : b.at * fraction)
  // A viewport resize can change the zoom's physical distance during flight.
  // Carry that difference into the same final ray, with no endpoint jump.
  const adjusted = add(position, scale(plan.arrival.back, (distance - plan.radius) * progress ** 4))
  return {
    ...look(a.basis, sightline(adjusted, from, plan, a.distance + (b.distance - a.distance) * fraction)),
    eye: adjusted,
    aspect,
  }
}

function bodyUnit(id: SourceBody | 'earth', body: BodyState) {
  return id === 'earth' ? 6378.137 : body.radius
}

function localRotation(frame: SolarFrame, body: BodyState): Rotation {
  const [a, d, g] = worldToGeographic(body, frame.right)
  const [b, e, h] = worldToGeographic(body, frame.up)
  const [c, f, i] = worldToGeographic(body, frame.back)
  // Convert the three camera axes (matrix columns) to the orbit quaternion.
  const trace = a + e + i
  if (trace > 0) {
    const n = 2 * Math.sqrt(1 + trace)
    return [(h - f) / n, (c - g) / n, (d - b) / n, n / 4]
  }
  if (a > e && a > i) {
    const n = 2 * Math.sqrt(1 + a - e - i)
    return [n / 4, (b + d) / n, (c + g) / n, (h - f) / n]
  }
  if (e > i) {
    const n = 2 * Math.sqrt(1 + e - a - i)
    return [(b + d) / n, n / 4, (f + h) / n, (c - g) / n]
  }
  const n = 2 * Math.sqrt(1 + i - a - e)
  return [(c + g) / n, (f + h) / n, n / 4, (d - b) / n]
}

function destinationFrame(bodies: SolarBodies, focus: SolarFocus, camera: Camera) {
  const basis = cameraBasis(camera)
  const body = focus === 'system' ? undefined : bodies[focus]
  const multiplier = focus === 'system' ? innerSystemRadius : bodyUnit(focus, bodies[focus])
  const toWorld = (v: Vec3) => (body ? geographicToWorld(body, v) : v)
  const distance = body ? camera.distance * multiplier : 0
  const frame: SolarFrame = {
    right: toWorld(basis.right),
    up: toWorld(basis.up),
    back: toWorld(basis.back),
    eye: add(body?.center ?? [0, 0, 0], scale(toWorld(basis.eye), multiplier)),
    near: body ? Math.max(0.01, (camera.near ?? 0.05) * multiplier) : 1000,
    aspect: camera.aspect,
  }
  return { frame, body, distance }
}

export class SolarCamera {
  private flight: Flight | null = null
  private arrival: Rotation | null = null
  frame: SolarFrame | null = null

  startFlight(reduced: boolean, now = performance.now()) {
    const from = this.frame
    this.flight = from ? { from, start: now, instant: reduced } : null
    this.arrival = null
  }

  get flying() {
    return Boolean(this.flight && !this.flight.instant)
  }
  get transitioning() {
    return Boolean(this.flight)
  }

  takeArrival() {
    const arrival = this.arrival
    this.arrival = null
    return arrival
  }

  update(bodies: SolarBodies, focus: SolarFocus, camera: Camera, now: number) {
    const { frame: destination, body, distance } = destinationFrame(bodies, focus, camera)
    const flight = this.flight
    if (!flight) return (this.frame = destination)
    const elapsed = unit((now - flight.start) / flightDuration)
    if (!flight.instant && elapsed === 0) return (this.frame = { ...flight.from, aspect: camera.aspect })
    const plan = (flight.plan ??= planFlight(bodies, flight.from, focus, destination, distance))
    if (flight.instant || elapsed === 1) {
      this.flight = null
      const arrived = body
        ? {
            ...plan.arrival,
            eye: add(body.center, scale(plan.arrival.back, distance)),
            aspect: camera.aspect,
            near: destination.near,
          }
        : destination
      this.arrival = body ? localRotation(arrived, body) : null
      return (this.frame = arrived)
    }

    // One easing of perceived path length, with no independently timed
    // reveal, link, turn, or approach. Position and heading share the path.
    const progress = baselineProgress(elapsed)
    const frame = fly(flight.from, plan, progress, camera.aspect, distance)
    const gap = Math.min(
      ...(Object.entries(bodies) as [BodyId, BodyState][]).map(
        ([id, object]) =>
          length(subtract(object.center, frame.eye)) - (id === 'sun' ? object.radius : clearance(id, object)),
      ),
    )
    return (this.frame = { ...frame, near: Math.max(0.01, Math.min(gap * 0.04, 200_000)) })
  }
}
