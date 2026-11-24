export type Vec3 = [number, number, number]
export type Rotation = [number, number, number, number]
export type Selection = { longitude: number; latitude: number; position: Vec3; elevation?: number }
export type Camera = { yaw: number; pitch: number; rotation?: Rotation; center?: Vec3; distance: number; aspect: number; near?: number }
export const FOV = Math.PI / 4
export const COLUMNS = 512
export const ROWS = 256
export const EARTH_AXIS = 6378137
export const EARTH_FLATTENING = 1 / 298.257223563

export function ease(value: number) { return value * value * (3 - 2 * value) }

// Equal Earth (Šavrič et al., 2018): strictly equal-area pseudocylindrical projection (~2.05:1).
const EE_A1 = 1.340264
const EE_A2 = -0.081106
const EE_A3 = 0.000893
const EE_A4 = 0.003796
const EE_M = Math.sqrt(3) / 2
const EE_X_SCALE = 2 / Math.sqrt(3)
export const EQUAL_EARTH_HALF_WIDTH = Math.PI * EE_X_SCALE / EE_A1
export const EQUAL_EARTH_HALF_HEIGHT = (() => {
  const t = Math.PI / 3, t2 = t * t, t6 = t2 * t2 * t2
  return t * (EE_A1 + EE_A2 * t2 + t6 * (EE_A3 + EE_A4 * t2))
})()

export function equalEarthProject(longitude: number, latitude: number): [number, number] {
  const theta = Math.asin(Math.max(-1, Math.min(1, EE_M * Math.sin(latitude))))
  const t2 = theta * theta, t6 = t2 * t2 * t2
  const d = EE_A1 + 3 * EE_A2 * t2 + t6 * (7 * EE_A3 + 9 * EE_A4 * t2)
  return [
    EE_X_SCALE * longitude * Math.cos(theta) / d,
    theta * (EE_A1 + EE_A2 * t2 + t6 * (EE_A3 + EE_A4 * t2)),
  ]
}

export function equalEarthInverse(x: number, y: number): { lon: number; lat: number } | null {
  if (Math.abs(y) > EQUAL_EARTH_HALF_HEIGHT) return null
  let theta = y / EE_A1
  let d = EE_A1
  for (let i = 0; i < 6; i++) {
    const t2 = theta * theta, t6 = t2 * t2 * t2
    d = EE_A1 + 3 * EE_A2 * t2 + t6 * (7 * EE_A3 + 9 * EE_A4 * t2)
    const f = theta * (EE_A1 + EE_A2 * t2 + t6 * (EE_A3 + EE_A4 * t2)) - y
    theta -= f / d
  }
  const cosT = Math.cos(theta)
  if (cosT < 1e-9) return null
  const lon = x * d * EE_M / cosT
  if (Math.abs(lon) > Math.PI) return null
  const lat = Math.asin(Math.max(-1, Math.min(1, Math.sin(theta) / EE_M)))
  return { lon, lat }
}

// Equal Earth flat map opening at ±180°, morphing smoothly to the WGS84 ellipsoid.
export function surfacePoint(longitude: number, latitude: number, morph: number, elevation = 0, exaggeration = 1): Vec3 {
  const cos = Math.abs(latitude) === Math.PI / 2 ? 0 : Math.cos(latitude), sin = Math.sin(latitude)
  const e2 = EARTH_FLATTENING * (2 - EARTH_FLATTENING)
  const n = 1 / Math.sqrt(1 - e2 * sin * sin)
  const height = Math.max(0, elevation) / EARTH_AXIS * exaggeration
  const [px, py] = equalEarthProject(longitude, latitude)
  return [
    px * (1 - morph) + (n + height) * cos * Math.sin(longitude) * morph,
    py * (1 - morph) + (n * (1 - e2) + height) * sin * morph,
    height * (1 - morph) + (n + height) * cos * Math.cos(longitude) * morph,
  ]
}

// Compose in camera space: dragging always turns around the current screen axes,
// including when the camera is over a pole or upside down. Normalize each step.
export function orbitRotation(rotation: Rotation, horizontal: number, vertical: number): Rotation {
  const angle = Math.hypot(horizontal,vertical)
  if (angle === 0) return rotation
  const scale = -Math.sin(angle / 2) / angle
  const dx = vertical * scale, dy = horizontal * scale, dw = Math.cos(angle / 2)
  return composeRotation(rotation,[dx,dy,0,dw])
}

function composeRotation(a: Rotation, b: Rotation): Rotation {
  const [x,y,z,w] = a, [dx,dy,dz,dw] = b
  const next: Rotation = [x * dw + w * dx + y * dz - z * dy,y * dw + z * dx + w * dy - x * dz,z * dw + x * dy - y * dx + w * dz,w * dw - x * dx - y * dy - z * dz]
  const length = Math.hypot(...next)
  return next.map(value => value / length) as Rotation
}

export function cameraBasis(camera: Camera) {
  const { yaw, pitch, distance } = camera
  let right: Vec3 = [Math.cos(yaw), 0, -Math.sin(yaw)]
  let up: Vec3 = [-Math.sin(yaw) * Math.sin(pitch), Math.cos(pitch), -Math.cos(yaw) * Math.sin(pitch)]
  let back: Vec3 = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)]
  if (camera.rotation) {
    const [x,y,z,w] = camera.rotation
    right = [1 - 2 * (y * y + z * z),2 * (x * y + z * w),2 * (x * z - y * w)]
    up = [2 * (x * y - z * w),1 - 2 * (x * x + z * z),2 * (y * z + x * w)]
    back = [2 * (x * z + y * w),2 * (y * z - x * w),1 - 2 * (x * x + y * y)]
  }
  const eye = back.map((value,i) => value * distance + (camera.center?.[i] ?? 0)) as Vec3
  return { right, up, back, eye }
}

// The grabbed position is fixed in world space. Its distance to the orbit center
// is invariant, so intersect the cursor ray with that radius in camera space,
// then rotate that intersection onto the grab. This includes perspective/zoom
// and preserves the exact picked terrain point rather than an approximate sphere.
// Pass the camera captured at grab time for a conservative arcball: each result
// depends on the current cursor and starting pose, not the path between them.
export function grabRotation(camera: Camera, anchor: Vec3, ndcX: number, ndcY: number): Rotation {
  const rotation = camera.rotation ?? orbitRotation(orbitRotation([0,0,0,1],-camera.yaw,0),0,camera.pitch)
  const relative = sub(anchor,camera.center ?? [0,0,0]), radius = Math.hypot(...relative)
  if (radius < 1e-10) return rotation
  const { right,up,back } = cameraBasis(camera)
  const local: Vec3 = [dot(relative,right),dot(relative,up),dot(relative,back)]
  const tangent = Math.tan(FOV / 2), d = camera.distance
  let x = ndcX * tangent * camera.aspect, y = ndcY * tangent
  const slope = Math.hypot(x,y), horizon = radius / Math.sqrt(Math.max(1e-12,d * d - radius * radius))
  // A point cannot project beyond the globe's silhouette. Follow the nearest
  // horizon position while the cursor is outside, and resume on re-entry.
  if (slope > horizon) { x *= horizon / slope; y *= horizon / slope }
  const a = 1 + x * x + y * y
  const root = Math.sqrt(Math.max(0,d * d - a * (d * d - radius * radius)))
  const near = (d - root) / a, far = (d + root) / a
  // Terrain at the limb can lie on either side of its constant-radius sphere.
  const at = (t: number): Vec3 => [x * t,y * t,d - t]
  const qNear = at(near), qFar = at(far)
  const q = dot(qNear,local) >= dot(qFar,local) ? qNear : qFar
  const from = q.map(value => value / radius) as Vec3, to = local.map(value => value / radius) as Vec3
  let axis = cross(from,to), w = 1 + dot(from,to)
  if (w < 1e-10) { axis = cross(from,Math.abs(from[0]) < .9 ? [1,0,0] : [0,1,0]); w = 0 }
  const length = Math.hypot(...axis,w)
  return composeRotation(rotation,[axis[0] / length,axis[1] / length,axis[2] / length,w / length])
}

export function grabCenter(camera: Camera, anchor: Vec3, ndcX: number, ndcY: number): Vec3 {
  const { right,up,back,eye } = cameraBasis(camera)
  const depth = dot(sub(eye,anchor),back), tangent = Math.tan(FOV / 2)
  return anchor.map((value,i) => (camera.center?.[i] ?? 0) + value - (eye[i] + depth * (-back[i] + right[i] * ndcX * tangent * camera.aspect + up[i] * ndcY * tangent))) as Vec3
}

export function cameraMatrix(camera: Camera): Float32Array {
  const { right: r, up: u, back: b, eye } = cameraBasis(camera)
  const dot = (a: Vec3, v: Vec3) => a[0] * v[0] + a[1] * v[1] + a[2] * v[2]
  const view = [r[0], u[0], b[0], 0, r[1], u[1], b[1], 0, r[2], u[2], b[2], 0, -dot(r, eye), -dot(u, eye), -dot(b, eye), 1]
  const f = 1 / Math.tan(FOV / 2), near = camera.near ?? 0.05, far = 100
  const projection = [f / camera.aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0]
  const result = new Float32Array(16)
  for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) {
    for (let k = 0; k < 4; k++) result[col * 4 + row] += projection[k * 4 + row] * view[col * 4 + k]
  }
  return result
}

export function createMesh(columns = COLUMNS, rows = ROWS) {
  const uv = new Float32Array((columns + 1) * (rows + 1) * 2)
  for (let y = 0; y <= rows; y++) for (let x = 0; x <= columns; x++) {
    const index = (y * (columns + 1) + x) * 2
    uv[index] = x / columns
    uv[index + 1] = y / rows
  }
  const indices = new Uint32Array(columns * rows * 6)
  for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) {
    const a = y * (columns + 1) + x, b = a + 1, c = a + columns + 1, d = c + 1
    indices.set([a, b, c, b, d, c], (y * columns + x) * 6)
  }
  return { uv, indices, columns, rows }
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

type TerrainSurface = { heightAt: (longitude: number, latitude: number) => number; exaggeration: number }
type PickBlock = { x: number; y: number; endX: number; endY: number; min: Vec3; max: Vec3 }
type PickGeometry = { vertices: Vec3[]; blocks: PickBlock[] }

function preparePickGeometry(mesh: ReturnType<typeof createMesh>, morph: number, terrain?: TerrainSurface): PickGeometry {
  const vertices: Vec3[] = []
  for (let i = 0; i < mesh.uv.length; i += 2) {
    const longitude = (mesh.uv[i] - .5) * Math.PI * 2, latitude = (mesh.uv[i + 1] - .5) * Math.PI
    vertices.push(surfacePoint(longitude,latitude,morph,terrain?.heightAt(longitude,latitude),terrain?.exaggeration))
  }
  const blocks: PickBlock[] = []
  for (let y = 0; y < mesh.rows; y += 8) for (let x = 0; x < mesh.columns; x += 8) {
    const endX = Math.min(x + 8,mesh.columns), endY = Math.min(y + 8,mesh.rows)
    const min: Vec3 = [Infinity,Infinity,Infinity], max: Vec3 = [-Infinity,-Infinity,-Infinity]
    for (let row = y; row <= endY; row++) for (let col = x; col <= endX; col++) {
      const vertex = vertices[row * (mesh.columns + 1) + col]
      for (let axis = 0; axis < 3; axis++) { min[axis] = Math.min(min[axis],vertex[axis]); max[axis] = Math.max(max[axis],vertex[axis]) }
    }
    blocks.push({ x,y,endX,endY,min,max })
  }
  return { vertices,blocks }
}

function intersectsBlock(block: PickBlock, eye: Vec3, direction: Vec3, nearest: number) {
  let near = 0, far = nearest
  for (let axis = 0; axis < 3; axis++) {
    const min = block.min[axis] - 1e-9, max = block.max[axis] + 1e-9
    if (Math.abs(direction[axis]) < 1e-15) {
      if (eye[axis] < min || eye[axis] > max) return false
    } else {
      const a = (min - eye[axis]) / direction[axis], b = (max - eye[axis]) / direction[axis]
      near = Math.max(near,Math.min(a,b)); far = Math.min(far,Math.max(a,b))
      if (near > far) return false
    }
  }
  return true
}

// Geometry is invariant during camera motion. Rebuild only when the surface
// changes; reject whole 8x8-cell blocks before testing their exact triangles.
export class SurfacePicker {
  private mesh: ReturnType<typeof createMesh>
  private geometry: PickGeometry
  private morph = 1
  private terrain?: TerrainSurface

  constructor(mesh: ReturnType<typeof createMesh>) {
    this.mesh = mesh
    this.geometry = preparePickGeometry(mesh,1)
  }

  setSurface(morph: number, terrain?: TerrainSurface) {
    if (this.morph === morph && this.terrain?.heightAt === terrain?.heightAt && this.terrain?.exaggeration === terrain?.exaggeration) return
    this.geometry = preparePickGeometry(this.mesh,morph,terrain)
    this.morph = morph; this.terrain = terrain
  }

  pick(ndcX: number, ndcY: number, camera: Camera) {
    return pickSurface(ndcX,ndcY,camera,this.morph,this.mesh,this.terrain,this.geometry)
  }
}

// Match the fragment shader: longitude must be recovered from interpolated
// directions on the globe, rather than interpolated through the polar UV fan.
export function geographicCoordinates(uv: [number, number], direction: Vec3, morph: number) {
  const globeU = Math.atan2(direction[0],direction[2]) / (2 * Math.PI) + .5
  const globeV = Math.atan2(direction[1],Math.hypot(direction[0],direction[2])) / Math.PI + .5
  const delta = globeU - uv[0]
  const u = uv[0] + (delta - Math.floor(delta + .5)) * morph
  return { longitude: (((u % 1) + 1) % 1 - .5) * 360, latitude: (uv[1] + (globeV - uv[1]) * morph - .5) * 180 }
}

// Intersect the same triangles that the vertex shader draws, including mid-morph.
export function pickSurface(ndcX: number, ndcY: number, camera: Camera, morph: number, mesh: ReturnType<typeof createMesh>, terrain?: TerrainSurface, prepared?: PickGeometry): Selection | null {
  const { right, up, back, eye } = cameraBasis(camera)
  const t = Math.tan(FOV / 2)
  const direction = back.map((value, i) => -value + right[i] * ndcX * t * camera.aspect + up[i] * ndcY * t) as Vec3
  const { vertices,blocks } = prepared ?? preparePickGeometry(mesh,morph,terrain)
  let nearest = Infinity
  let result: Selection | null = null
  for (const block of blocks) {
    if (!intersectsBlock(block,eye,direction,nearest)) continue
    for (let row = block.y; row < block.endY; row++) for (let col = block.x; col < block.endX; col++) for (let triangle = 0; triangle < 2; triangle++) {
      const i = (row * mesh.columns + col) * 6 + triangle * 3
      const ia = mesh.indices[i], ib = mesh.indices[i + 1], ic = mesh.indices[i + 2]
      const a = vertices[ia], edge1 = sub(vertices[ib], a), edge2 = sub(vertices[ic], a)
      const h = cross(direction, edge2), det = dot(edge1, h)
      if (Math.abs(det) < 1e-10) continue
      // Include shared edges despite roundoff, especially the longitude seam.
      // Otherwise a center ray can miss the near side and select the far side.
      const s = sub(eye, a), rawU = dot(s, h) / det
      if (rawU < -1e-9 || rawU > 1 + 1e-9) continue
      const q = cross(s, edge1), rawV = dot(direction, q) / det
      if (rawV < -1e-9 || rawU + rawV > 1 + 1e-9) continue
      const u = Math.max(0,Math.min(1,rawU)), v = Math.max(0,Math.min(1 - u,rawV))
      const distance = dot(edge2, q) / det
      if (distance <= 0 || distance >= nearest) continue
      nearest = distance
      const interpolate = (offset: number) => mesh.uv[ia * 2 + offset] * (1 - u - v) + mesh.uv[ib * 2 + offset] * u + mesh.uv[ic * 2 + offset] * v
      const normal = (index: number): Vec3 => {
        const lon = (mesh.uv[index * 2] - .5) * Math.PI * 2, lat = (mesh.uv[index * 2 + 1] - .5) * Math.PI
        const cos = Math.abs(lat) === Math.PI / 2 ? 0 : Math.cos(lat)
        return [cos * Math.sin(lon),Math.sin(lat),cos * Math.cos(lon)]
      }
      const na = normal(ia), nb = normal(ib), nc = normal(ic)
      const geodetic = na.map((value,index) => value * (1 - u - v) + nb[index] * u + nc[index] * v) as Vec3
      result = { ...geographicCoordinates([interpolate(0),interpolate(1)],geodetic,morph), position: eye.map((value, index) => value + direction[index] * distance) as Vec3 }
    }
  }
  return result
}
