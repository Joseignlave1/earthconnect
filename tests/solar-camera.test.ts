import { strict as assert } from 'node:assert'
import test from 'node:test'
import { EARTH_AXIS, orbitRotation } from '../src/shader-test/geometry.ts'
import {
  add, bodyMatrix, dot, length, normalized, raySphere, relativeToEye, scale, sceneRay, SolarCamera, subtract, worldToGeographic,
} from '../src/workspace/solarCamera.ts'
import { solarSystemAt, type Vec3 } from '../src/workspace/solarSystem.ts'
import { baselineProgress, flightDuration } from '../src/workspace/flightEasing.ts'

function project(matrix: Float32Array, [x, y, z]: Vec3) {
  const clip = [0, 1, 2, 3].map(row => matrix[row] * x + matrix[row + 4] * y + matrix[row + 8] * z + matrix[row + 12])
  return {
    x: clip[0] / clip[3],
    y: clip[1] / clip[3],
    distance: clip[3],
    depth: clip[2] / clip[3],
  }
}

test('a single Earth camera resolves a near surface and the occulted Sun at AU scale', () => {
  const bodies = solarSystemAt(new Date('2024-03-20T03:06:00Z'))
  const { earth, sun } = bodies
  const antiSun = normalized(subtract(earth.center, sun.center))
  const ground = worldToGeographic(earth, antiSun)
  const earthUnit = EARTH_AXIS / 1000
  const distance = 1.00001
  const frame = new SolarCamera().update(bodies, 'earth', {
    distance,
    yaw: Math.atan2(ground[0], ground[2]),
    pitch: Math.asin(ground[1]),
    aspect: 1.8,
    near: 0.000001,
  }, 0)

  const earthInView = relativeToEye(frame, earth.center)
  assert.ok(Math.abs(earthInView[2] + distance * earthUnit) < 1e-6)

  const groundView = project(bodyMatrix(frame, earth, earthUnit), ground)
  const sunView = project(bodyMatrix(frame, sun, sun.radius), [0, 0, 0])
  // Resolve a roughly 64 m camera altitude atop a heliocentric position near 1 AU.
  const groundDistance = (distance - 1) * earthUnit
  const sunDistance = Math.hypot(...subtract(earth.center, sun.center)) + distance * earthUnit
  const near = 0.01
  assert.equal(frame.near, near)
  assert.ok(Math.abs(groundView.distance - groundDistance) < 0.001)
  assert.ok(Math.abs(sunView.distance - sunDistance) < 8)
  for (const point of [groundView, sunView]) {
    assert.ok(Math.abs(point.x) < 0.02 && Math.abs(point.y) < 0.02)
    assert.ok(point.depth > 0 && point.depth < 1)
  }
  assert.ok(Math.abs(groundView.depth - near / groundDistance) < 0.003)
  assert.ok(Math.abs(sunView.depth / (near / sunDistance) - 1) < 1e-7)
  assert.ok(groundView.depth > sunView.depth)

  const ray = sceneRay(frame, 0, 0)
  assert.ok(dot(ray, antiSun) < -1 + 1e-10)
  const earthHit = raySphere(frame.eye, ray, earth.center, earthUnit)
  const sunHit = raySphere(frame.eye, ray, sun.center, sun.radius)
  assert.ok(Math.abs(earthHit - groundDistance) < 1e-6)
  assert.ok(Number.isFinite(sunHit) && earthHit < sunHit)
})

const bodies = solarSystemAt(new Date('2026-11-17T21:00:00Z'))
const defaultOrbit = { yaw: 1.5,pitch: .6,distance: 3.8,near: .05 }
const start = 20_000

function projectedBody(frame: NonNullable<SolarCamera['frame']>,body: 'earth' | 'moon' | 'mars') {
  return project(bodyMatrix(frame,bodies[body],1),[0,0,0])
}

function fits(frame: NonNullable<SolarCamera['frame']>,id: 'earth' | 'moon' | 'mars') {
  const { x,y,depth,distance } = projectedBody(frame,id)
  const angularSize = bodies[id].radius / distance / Math.tan(Math.PI / 8)
  return depth > 0 && depth < 1 && Math.abs(x) + angularSize / frame.aspect < 1 && Math.abs(y) + angularSize < 1
}

function viewRotation(longitude: number,latitude: number) {
  const radians = Math.PI / 180
  return orbitRotation(orbitRotation([0,0,0,1],-longitude * radians,0),0,latitude * radians)
}

function angle(a: Vec3,b: Vec3) {
  return Math.acos(Math.max(-1,Math.min(1,dot(a,b)))) * 180 / Math.PI
}

test('an off-screen target takes a direct line and the camera turns while already moving', () => {
  for (const aspect of [1.65,.46]) {
    const camera = new SolarCamera()
    const source = { ...defaultOrbit,aspect,rotation: viewRotation(95,36) }
    const from = camera.update(bodies,'earth',source,start)
    const target = bodies.mars.center
    const bearing = normalized(subtract(target,from.eye))
    assert.ok(dot(bearing,from.back) > 0,'Mars starts behind the camera')
    camera.startFlight(false,start)
    let priorRange = length(subtract(from.eye,target))
    let priorAngle = angle(from.back,scale(bearing,-1))
    for (let elapsed = 16; elapsed < flightDuration; elapsed += 16) {
      const frame = camera.update(bodies,'mars',source,start + elapsed)
      const offset = subtract(frame.eye,from.eye)
      assert.ok(length(subtract(offset,scale(bearing,dot(offset,bearing)))) < 1e-5)
      assert.ok(length(subtract(frame.eye,target)) < priorRange)
      const turn = angle(frame.back,scale(bearing,-1))
      assert.ok(turn <= priorAngle + 1e-5)
      priorRange = length(subtract(frame.eye,target))
      priorAngle = turn
    }
  }
})

test('a blocked line takes a short safe detour instead of a wide reveal orbit', () => {
  const camera = new SolarCamera()
  const direction = normalized(subtract(bodies.moon.center,bodies.earth.center))
  const [x,y,z] = worldToGeographic(bodies.earth,scale(direction,-1))
  const orbit = { aspect: .52,distance: 1.035,yaw: Math.atan2(x,z),pitch: Math.asin(y) }
  const from = camera.update(bodies,'earth',orbit,start)
  assert.ok(Number.isFinite(raySphere(from.eye,direction,bodies.earth.center,bodies.earth.radius)))
  camera.startFlight(false,start)
  let deviation = 0
  for (let elapsed = 8; elapsed < flightDuration; elapsed += 8) {
    const frame = camera.update(bodies,'moon',{ ...orbit,distance: 5 },start + elapsed)
    const offset = subtract(frame.eye,from.eye)
    deviation = Math.max(deviation,length(subtract(offset,scale(direction,dot(offset,direction)))))
    assert.ok(length(subtract(frame.eye,bodies.earth.center)) >= bodies.earth.radius + 188)
  }
  assert.ok(deviation > bodies.earth.radius && deviation < bodies.earth.radius * 1.2)
})

test('a clear target takes the direct path and the arrived bearing becomes the interactive orbit', () => {
  const route = normalized(subtract(bodies.moon.center,bodies.earth.center))
  const lateral = normalized(subtract([0,0,1],scale(route,route[2])))
  const outward = add(scale(route,-Math.cos(.37)),scale(lateral,Math.sin(.37)))
  const [x,y,z] = worldToGeographic(bodies.earth,outward)
  const source = { aspect: 1.8,distance: 4.6,yaw: Math.atan2(x,z),pitch: Math.asin(y) }
  const saved = { aspect: source.aspect,distance: 2.4,yaw: 2.65,pitch: -.54,near: .04 }
  const camera = new SolarCamera()
  const from = camera.update(bodies,'earth',source,start)
  assert.ok(fits(from,'moon'))
  camera.startFlight(false,start)
  const approach = normalized(subtract(from.eye,bodies.moon.center))
  for (let elapsed = 250; elapsed < flightDuration; elapsed += 67) {
    const frame = camera.update(bodies,'moon',saved,start + elapsed)
    const relative = subtract(frame.eye,bodies.moon.center)
    assert.ok(length(subtract(relative,scale(approach,dot(relative,approach)))) < 1e-5)
  }
  assert.ok(camera.flying)
  const arrived = camera.update(bodies,'moon',saved,start + flightDuration)
  assert.equal(camera.flying,false)
  assert.ok(length(subtract(arrived.eye,add(bodies.moon.center,scale(approach,bodies.moon.radius * saved.distance)))) < 1e-6)
  assert.ok(dot(arrived.back,approach) > 1 - 1e-10)
  assert.ok(dot(arrived.back,new SolarCamera().update(bodies,'moon',saved,start).back) < .95)
  const rotation = camera.takeArrival()
  assert.ok(rotation)
  const interactive = camera.update(bodies,'moon',{ ...saved,rotation },start + flightDuration + 16)
  assert.ok(length(subtract(arrived.eye,interactive.eye)) < 1e-6)
  assert.ok(dot(arrived.up,interactive.up) > 1 - 1e-10)
  assert.equal(camera.takeArrival(),null)
})

test('the shorter flight starts gently, grows the destination, and settles without a turn', () => {
  for (const focus of ['moon','mars'] as const) {
    const camera = new SolarCamera()
    const departure = { ...defaultOrbit,aspect: 1.65,rotation: viewRotation(95,36) }
    const arrival = { ...departure,rotation: focus === 'moon' ? viewRotation(0,0) : viewRotation(-75,14.32) }
    let previous = camera.update(bodies,'earth',departure,start)
    camera.startFlight(false,start)
    assert.deepEqual(camera.update(bodies,focus,arrival,start),previous)
    let previousRange = Infinity
    let quiet = 0,rush = 0,landing = 0
    const radii: number[] = []
    for (let elapsed = 16; elapsed < flightDuration; elapsed += 16) {
      const frame = camera.update(bodies,focus,arrival,start + elapsed)
      const step = length(subtract(frame.eye,previous.eye))
      const time = elapsed / flightDuration
      if (time <= .08) {
        quiet = Math.max(quiet,step)
        assert.ok(angle(previous.back,frame.back) < 2)
      }
      if (time > .08 && time < .86) assert.ok(step > .001,'no stop between retreat and approach')
      if (time > .28 && time < .74) rush = Math.max(rush,step)
      if (elapsed > flightDuration - 500) {
        landing = Math.max(landing,step)
        assert.ok(angle(previous.back,frame.back) < .08)
        assert.ok(angle(previous.up,frame.up) < .08)
      }
      const range = length(subtract(frame.eye,bodies[focus].center))
      if (time > .6) assert.ok(range < previousRange)
      const sampleAt = [.3,.57,.72][radii.length]
      if (time >= sampleAt) radii.push(bodies[focus].radius / range)
      previous = frame
      previousRange = range
    }
    assert.ok(rush > quiet * 12)
    assert.ok(landing < rush * .05)
    assert.ok(radii[1] > radii[0] * 3 && radii[2] > radii[1] * 2)
    assert.ok(camera.flying)
    camera.update(bodies,focus,arrival,start + flightDuration)
    assert.equal(camera.flying,false)
  }
})

test('an AU-scale departure remains resolved while it accelerates into the approach', () => {
  const camera = new SolarCamera()
  const pose = { yaw: 1.66,pitch: .62,aspect: 1.6,distance: 3.84,near: .05 }
  let previous = camera.update(bodies,'earth',pose,start)
  camera.startFlight(false,start)
  let previousStep = 0
  let previousSize = 0
  for (let elapsed = 8; elapsed < flightDuration * .85; elapsed += 8) {
    const frame = camera.update(bodies,'mars',pose,start + elapsed)
    const step = length(subtract(frame.eye,previous.eye))
    const sourceSize = bodies.earth.radius / length(subtract(frame.eye,bodies.earth.center))
    if (elapsed > flightDuration * .12) {
      assert.ok(step / previousStep < 2 && previousStep / step < 2,'no speed jump at the join')
      if (sourceSize > .035) {
        assert.ok(sourceSize / previousSize < 1.16 && previousSize / sourceSize < 1.16)
      }
    }
    previous = frame
    previousStep = step
    previousSize = sourceSize
  }
})

test('the visible Earth leaves the frame without a sudden rush on the stock clock', () => {
  const source = { yaw: 1.66,pitch: .62,aspect: 1.6,distance: 3.86 }
  const landmarks = Array.from({ length: 512 },(_,index): Vec3 => {
    const y = 1 - (2 * index + 1) / 512
    const around = index * Math.PI * (3 - Math.sqrt(5))
    const radius = Math.sqrt(1 - y * y)
    return [radius * Math.cos(around),y,radius * Math.sin(around)]
  })
  function projectEarth(frame: NonNullable<SolarCamera['frame']>) {
    const earth = bodies.earth
    const observer = subtract(frame.eye,earth.center)
    return landmarks.map(point => {
      if (dot(point,observer) < earth.radius) return null
      const [x,y,z] = relativeToEye(frame,add(earth.center,scale(point,earth.radius)))
      const halfHeight = -z * Math.tan(Math.PI / 8)
      return halfHeight > 0 && Math.abs(x) < halfHeight * frame.aspect && Math.abs(y) < halfHeight
        ? [x / halfHeight,y / halfHeight]
        : null
    })
  }
  const interval = 1_000 / 120
  for (const destination of ['moon','mars'] as const) {
    const camera = new SolarCamera()
    let previous = camera.update(bodies,'earth',source,start)
    let visible = projectEarth(previous)
    let peakFlow = 0
    let peakTurn = 0
    let previousTurn = 0
    camera.startFlight(false,start)
    for (let tick = 1; tick * interval < flightDuration; tick++) {
      const frame = camera.update(bodies,destination,source,start + tick * interval)
      const next = projectEarth(frame)
      const travel = next.flatMap((point,index) => {
        const prior = visible[index]
        return point && prior ? [Math.hypot(point[0] - prior[0],point[1] - prior[1])] : []
      }).sort((a,b) => a - b)
      if (travel.length > 8) {
        const viewportHeightsPerSecond = travel[Math.floor(travel.length / 2)] * 500 / interval
        peakFlow = Math.max(peakFlow,viewportHeightsPerSecond)
      }
      const turnSpeed = angle(previous.back,frame.back) * 1000 / interval
      peakTurn = Math.max(peakTurn,turnSpeed)
      assert.ok(Math.abs(turnSpeed - previousTurn) < 95,`${destination}: angular speed jumped as the camera locked on`)
      previous = frame
      previousTurn = turnSpeed
      visible = next
    }
    assert.ok(peakFlow < 6.6,`${destination}: foreground speed was ${peakFlow.toFixed(2)} viewport heights/s`)
    assert.ok(peakTurn < 450,`${destination}: camera turn was ${peakTurn.toFixed(1)}°/s`)
  }
})

test('varying viewpoints and close releases stay clear and carry roll continuously', () => {
  for (const [index,from,to,aspect] of [
    [37,'moon','earth',1.65],
    [90,'earth','moon',.52],
    [21,'mars','earth',1.65],
    [61,'earth','mars',1.65],
    [5,'mars','moon',1.65],
  ] as const) {
    const epoch = solarSystemAt(new Date(Date.UTC(2025 + index % 3,index % 12,1 + index % 25)))
    const source = { aspect,distance: index === 37 ? 1.06 : 3.85 / Math.min(1,aspect),
      pitch: Math.sin(index * 4.1) * 1.4,yaw: index * 1.61 }
    const destination = { ...source,distance: 3.85,pitch: Math.sin(index * 9.3) * 1.4,yaw: index * 2.11 }
    const camera = new SolarCamera()
    let previous = camera.update(epoch,from,source,start)
    camera.startFlight(false,start)
    for (let elapsed = 16; elapsed <= flightDuration; elapsed += 16) {
      const frame = camera.update(epoch,to,destination,start + elapsed)
      assert.ok(length(subtract(frame.eye,epoch[from].center)) >= epoch[from].radius + { earth: 188,moon: 45,mars: 100 }[from])
      assert.ok(length(subtract(frame.eye,epoch.sun.center)) > epoch.sun.radius * 2)
      assert.ok(angle(previous.up,frame.up) < 12)
      previous = frame
    }
  }
})

test('retargeting starts at the displayed pose and reduced motion uses the same derived endpoint', () => {
  for (const fraction of [.1,.48,.88]) {
    const interruptAt = flightDuration * fraction
    const camera = new SolarCamera()
    const close = { ...defaultOrbit,aspect: .55,distance: 1.04,near: .001 }
    camera.update(bodies,'earth',close,start)
    camera.startFlight(false,start)
    const interrupted = camera.update(bodies,'mars',close,start + interruptAt)
    const retargetTime = start + interruptAt + 5
    camera.startFlight(false,retargetTime)
    assert.deepEqual(camera.update(bodies,'moon',close,retargetTime),interrupted)
    const next = camera.update(bodies,'moon',close,retargetTime + .01)
    assert.ok(dot(next.back,interrupted.back) > .9999)
    assert.ok(length(subtract(next.eye,interrupted.eye)) < 1)
    const instant = new SolarCamera()
    instant.frame = interrupted
    instant.startFlight(true,retargetTime)
    assert.equal(instant.flying,false)
    const expected = instant.update(bodies,'moon',close,retargetTime)
    const settled = camera.update(bodies,'moon',close,retargetTime + flightDuration)
    assert.deepEqual(settled,expected)
  }
})

test('a continuous path steers clear of the Sun in conjunction', () => {
  const [x,y,z] = bodies.earth.center
  const aligned = { ...bodies,mars: { ...bodies.mars,center: [-x * 1.5,-y * 1.5,-z * 1.5] as Vec3 } }
  const camera = new SolarCamera()
  const orbit = { ...defaultOrbit,aspect: 1.55 }
  camera.update(aligned,'earth',orbit,start)
  camera.startFlight(false,start)
  let closest = Infinity
  for (let elapsed = 2; elapsed < flightDuration; elapsed += 2) {
    const frame = camera.update(aligned,'mars',orbit,start + elapsed)
    closest = Math.min(closest,length(subtract(frame.eye,aligned.sun.center)))
  }
  assert.ok(closest > aligned.sun.radius * 3)
})

test('the stock clock keeps the chosen real-time anchors and brakes through the final half second', () => {
  assert.equal(flightDuration,2750)
  const progress = (milliseconds: number) => baselineProgress(milliseconds / flightDuration)
  const anchors = [[.1670524919172749 * 5000,.235034],[1550,.741808],[2450,.998]]
  for (const [milliseconds,expected] of anchors) assert.ok(Math.abs(progress(milliseconds) - expected) < 1e-12)
  assert.ok(Math.abs(progress(500) - .06643276471734424) < 1e-12)
  assert.ok(Math.abs(progress(2250) - .9933739645583168) < 1e-12)
  for (const junction of [2250,2450]) {
    const before = (progress(junction) - progress(junction - .01)) / .01
    const after = (progress(junction + .01) - progress(junction)) / .01
    assert.ok(Math.abs(before - after) < 1e-8)
  }
  let priorSpeed = Infinity
  for (let time = 2250; time < flightDuration; time += 5) {
    const speed = progress(time + 5) - progress(time)
    assert.ok(speed >= 0 && speed <= priorSpeed)
    priorSpeed = speed
  }
  assert.ok(priorSpeed < 2e-8)
  assert.equal(progress(flightDuration),1)
})
