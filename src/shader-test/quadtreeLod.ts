import { EARTH_AXIS, EARTH_FLATTENING, equalEarthProject, type Vec3 } from './geometry.ts'
import { elevationAt, type Terrain } from './terrain.ts'

export const PATCH_SEGMENTS = 16
export const MAX_PATCH_INSTANCES = 1024
export const INSTANCE_FLOATS = 8 // 2 × vec4f = 32 bytes per visible patch

export type PatchMesh = {
  /** Interleaved float32x3: (s, t, skirt) where s, t ∈ [0, 1] and skirt ∈ {0, 1}. */
  vertices: Float32Array<ArrayBuffer>
  /** Triangle list indices (uint16) with cache-friendly strips + 4 outward-facing skirts. */
  indices: Uint16Array<ArrayBuffer>
  vertexCount: number
  indexCount: number
}

/**
 * Shared 16×16 terrain patch mesh with 1-cell border skirts along all 4 edges.
 * - Inner grid is emitted in 8-column vertical strips for high post-transform vertex-cache reuse.
 * - Border skirts drop inward along -normal in the vertex shader to seal any LOD T-junction crack.
 */
export function createPatchMesh(segments = PATCH_SEGMENTS): PatchMesh {
  const n = segments
  const side = n + 1
  const innerCount = side * side
  const skirtCount = side * 4
  const vertexCount = innerCount + skirtCount
  const vertices = new Float32Array(vertexCount * 3)

  // Inner grid vertices: (s, t, 0)
  for (let iy = 0; iy <= n; iy++) {
    const t = iy / n
    for (let ix = 0; ix <= n; ix++) {
      const idx = (iy * side + ix) * 3
      vertices[idx] = ix / n
      vertices[idx + 1] = t
      vertices[idx + 2] = 0
    }
  }

  // Border skirt vertices: (s, t, 1)
  const botStart = innerCount
  const topStart = botStart + side
  const leftStart = topStart + side
  const rightStart = leftStart + side
  for (let i = 0; i <= n; i++) {
    const f = i / n
    let o = (botStart + i) * 3
    vertices[o] = f
    vertices[o + 1] = 0
    vertices[o + 2] = 1

    o = (topStart + i) * 3
    vertices[o] = f
    vertices[o + 1] = 1
    vertices[o + 2] = 1

    o = (leftStart + i) * 3
    vertices[o] = 0
    vertices[o + 1] = f
    vertices[o + 2] = 1

    o = (rightStart + i) * 3
    vertices[o] = 1
    vertices[o + 1] = f
    vertices[o + 2] = 1
  }

  const innerTris = n * n * 2
  const skirtTris = n * 4 * 2
  const indices = new Uint16Array((innerTris + skirtTris) * 3)
  let ptr = 0

  // Emit inner grid in 8-column vertical strips for GPU vertex-cache locality.
  const stripWidth = 8
  for (let x0 = 0; x0 < n; x0 += stripWidth) {
    const x1 = Math.min(n, x0 + stripWidth)
    for (let y = 0; y < n; y++) {
      for (let x = x0; x < x1; x++) {
        const a = y * side + x
        const b = a + 1
        const c = a + side
        const d = c + 1
        indices[ptr++] = a
        indices[ptr++] = b
        indices[ptr++] = c
        indices[ptr++] = b
        indices[ptr++] = d
        indices[ptr++] = c
      }
    }
  }

  // Bottom skirt (t = 0): outward face points -V.
  for (let i = 0; i < n; i++) {
    const o0 = i
    const o1 = i + 1
    const s0 = botStart + i
    const s1 = botStart + i + 1
    indices[ptr++] = o0
    indices[ptr++] = s0
    indices[ptr++] = o1
    indices[ptr++] = s0
    indices[ptr++] = s1
    indices[ptr++] = o1
  }

  // Top skirt (t = 1): outward face points +V.
  for (let i = 0; i < n; i++) {
    const o0 = n * side + i
    const o1 = o0 + 1
    const s0 = topStart + i
    const s1 = topStart + i + 1
    indices[ptr++] = o0
    indices[ptr++] = o1
    indices[ptr++] = s0
    indices[ptr++] = o1
    indices[ptr++] = s1
    indices[ptr++] = s0
  }

  // Left skirt (s = 0): outward face points -U.
  for (let i = 0; i < n; i++) {
    const o0 = i * side
    const o1 = (i + 1) * side
    const s0 = leftStart + i
    const s1 = leftStart + i + 1
    indices[ptr++] = o0
    indices[ptr++] = o1
    indices[ptr++] = s0
    indices[ptr++] = o1
    indices[ptr++] = s1
    indices[ptr++] = s0
  }

  // Right skirt (s = 1): outward face points +U.
  for (let i = 0; i < n; i++) {
    const o0 = i * side + n
    const o1 = (i + 1) * side + n
    const s0 = rightStart + i
    const s1 = rightStart + i + 1
    indices[ptr++] = o0
    indices[ptr++] = s0
    indices[ptr++] = o1
    indices[ptr++] = s0
    indices[ptr++] = s1
    indices[ptr++] = o1
  }

  return { vertices, indices, vertexCount, indexCount: indices.length }
}

type Plane = [number, number, number, number]

type CameraView = {
  planes: [Plane, Plane, Plane, Plane, Plane]
  r3x: number
  r3y: number
  r3z: number
  r3w: number
  r3Len: number
  projScale: number
}

function normalizePlane(a: number, b: number, c: number, d: number): Plane {
  const len = Math.hypot(a, b, c)
  if (len < 1e-12) return [0, 0, 0, 1]
  const inv = 1 / len
  return [a * inv, b * inv, c * inv, d * inv]
}

/**
 * Extract 5 normalized clip-space frustum planes (left, right, bottom, top, front w>=0)
 * and exact projected pixels-per-unit scale from any column-major 4×4 projection*view matrix.
 * Works identically for reverse-Z, standard depth, and zoomed optical loupe matrices.
 */
function extractCameraView(m: ArrayLike<number>, width: number, height: number): CameraView {
  const r0x = m[0],
    r0y = m[4],
    r0z = m[8],
    r0w = m[12]
  const r1x = m[1],
    r1y = m[5],
    r1z = m[9],
    r1w = m[13]
  const r3x = m[3],
    r3y = m[7],
    r3z = m[11],
    r3w = m[15]

  const planes: [Plane, Plane, Plane, Plane, Plane] = [
    normalizePlane(r3x + r0x, r3y + r0y, r3z + r0z, r3w + r0w), // left
    normalizePlane(r3x - r0x, r3y - r0y, r3z - r0z, r3w - r0w), // right
    normalizePlane(r3x + r1x, r3y + r1y, r3z + r1z, r3w + r1w), // bottom
    normalizePlane(r3x - r1x, r3y - r1y, r3z - r1z, r3w - r1w), // top
    normalizePlane(r3x, r3y, r3z, r3w), // in front of camera (w >= 0)
  ]

  const r3LenSq = r3x * r3x + r3y * r3y + r3z * r3z
  const r3Len = Math.sqrt(r3LenSq)
  let scaleX: number
  let scaleY: number
  if (r3LenSq > 1e-16) {
    // Project out row3 so off-center zoomed loupe matrices recover exact focal scale.
    const k0 = (r0x * r3x + r0y * r3y + r0z * r3z) / r3LenSq
    const k1 = (r1x * r3x + r1y * r3y + r1z * r3z) / r3LenSq
    scaleX = Math.hypot(r0x - k0 * r3x, r0y - k0 * r3y, r0z - k0 * r3z) * (width * 0.5)
    scaleY = Math.hypot(r1x - k1 * r3x, r1y - k1 * r3y, r1z - k1 * r3z) * (height * 0.5)
  } else {
    scaleX = Math.hypot(r0x, r0y, r0z) * (width * 0.5)
    scaleY = Math.hypot(r1x, r1y, r1z) * (height * 0.5)
  }
  return { planes, r3x, r3y, r3z, r3w, r3Len, projScale: Math.max(scaleX, scaleY) }
}

function sphereOutsideFrustum(view: CameraView, cx: number, cy: number, cz: number, radius: number) {
  const { planes } = view
  for (let i = 0; i < 5; i++) {
    const p = planes[i]
    if (p[0] * cx + p[1] * cy + p[2] * cz + p[3] < -radius) return true
  }
  return false
}

function pixelsPerUnitForSphere(view: CameraView, cx: number, cy: number, cz: number, radius: number) {
  const wCenter = view.r3x * cx + view.r3y * cy + view.r3z * cz + view.r3w
  const wNear = Math.max(1e-4, wCenter - radius * view.r3Len)
  return view.projScale / wNear
}

// ---------------------------------------------------------------------------
// 1. Earth: Latitude-Coarsened UV Error-Metric Quadtree (supports Equal Earth morph)
// ---------------------------------------------------------------------------

const EARTH_PYRAMID_LEVELS = 5 // level 0: 4×2 ... level 4: 64×32
const EARTH_ROOT_COLS = 4
const EARTH_ROOT_ROWS = 2
const EARTH_MAX_DEPTH = 6 // up to 256×128 patches × 16 = 4096×2048 effective vertices
const EARTH_E2 = EARTH_FLATTENING * (2 - EARTH_FLATTENING)

type EarthRoughnessLevel = {
  cols: number
  rows: number
  /** Max elevation deviation from bilinear corners (metres, land clamped >= 0). */
  roughness: Float32Array
  /** Max positive elevation in metres (0 over pure ocean). */
  maxElev: Float32Array
}

export class EarthQuadtree {
  private readonly levels: EarthRoughnessLevel[]
  readonly demWidth: number
  readonly demHeight: number
  readonly maxDemMip: number

  private constructor(levels: EarthRoughnessLevel[], demWidth: number, demHeight: number) {
    this.levels = levels
    this.demWidth = demWidth
    this.demHeight = demHeight
    this.maxDemMip = Math.max(0, Math.floor(Math.log2(Math.max(demWidth, demHeight))))
  }

  static createFlat(): EarthQuadtree {
    const levels: EarthRoughnessLevel[] = []
    for (let l = 0; l < EARTH_PYRAMID_LEVELS; l++) {
      const cols = EARTH_ROOT_COLS << l
      const rows = EARTH_ROOT_ROWS << l
      levels.push({
        cols,
        rows,
        roughness: new Float32Array(cols * rows),
        maxElev: new Float32Array(cols * rows),
      })
    }
    return new EarthQuadtree(levels, 4096, 2048)
  }

  static fromTerrain(terrain: Terrain): EarthQuadtree {
    const leafLevel = EARTH_PYRAMID_LEVELS - 1
    const leafCols = EARTH_ROOT_COLS << leafLevel // 64
    const leafRows = EARTH_ROOT_ROWS << leafLevel // 32
    const leafRough = new Float32Array(leafCols * leafRows)
    const leafMax = new Float32Array(leafCols * leafRows)

    const sampleLand = (u: number, v: number) => {
      const lon = (u - 0.5) * 2 * Math.PI
      const lat = (v - 0.5) * Math.PI
      const h = elevationAt(terrain, lon, lat)
      return h > 0 ? h : 0
    }

    // Compute actual DEM roughness (deviation from bilinear corner plane) on a 5×5 probe per leaf.
    const du = 1 / leafCols
    const dv = 1 / leafRows
    for (let row = 0; row < leafRows; row++) {
      const v0 = row * dv
      const v1 = v0 + dv
      for (let col = 0; col < leafCols; col++) {
        const u0 = col * du
        const u1 = u0 + du
        const h00 = sampleLand(u0, v0)
        const h10 = sampleLand(u1, v0)
        const h01 = sampleLand(u0, v1)
        const h11 = sampleLand(u1, v1)
        let maxH = Math.max(h00, h10, h01, h11)
        let minH = Math.min(h00, h10, h01, h11)
        let maxErr = 0
        for (let sy = 0; sy <= 4; sy++) {
          const fy = sy * 0.25
          const v = v0 + dv * fy
          for (let sx = 0; sx <= 4; sx++) {
            const fx = sx * 0.25
            const h = sampleLand(u0 + du * fx, v)
            if (h > maxH) maxH = h
            if (h < minH) minH = h
            const interp = (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy
            const err = Math.abs(h - interp)
            if (err > maxErr) maxErr = err
          }
        }
        const idx = row * leafCols + col
        // Capture both local peak deviation and peak-to-valley amplitude so narrow chains
        // (Andes, Alps, Himalayas) always refine while flat oceans (maxH == minH == 0) stay 0.
        leafRough[idx] = Math.max(maxErr, (maxH - minH) * 0.65)
        leafMax[idx] = maxH
      }
    }

    const levels: EarthRoughnessLevel[] = new Array(EARTH_PYRAMID_LEVELS)
    levels[leafLevel] = { cols: leafCols, rows: leafRows, roughness: leafRough, maxElev: leafMax }

    // Propagate bottom-up; coarser nodes also capture inter-child elevation range.
    for (let l = leafLevel - 1; l >= 0; l--) {
      const child = levels[l + 1]
      const cols = EARTH_ROOT_COLS << l
      const rows = EARTH_ROOT_ROWS << l
      const roughness = new Float32Array(cols * rows)
      const maxElev = new Float32Array(cols * rows)
      for (let r = 0; r < rows; r++) {
        const cr = r * 2
        for (let c = 0; c < cols; c++) {
          const cc = c * 2
          const i00 = cr * child.cols + cc
          const i10 = i00 + 1
          const i01 = (cr + 1) * child.cols + cc
          const i11 = i01 + 1
          const m00 = child.maxElev[i00]
          const m10 = child.maxElev[i10]
          const m01 = child.maxElev[i01]
          const m11 = child.maxElev[i11]
          const maxH = Math.max(m00, m10, m01, m11)
          const minH = Math.min(m00, m10, m01, m11)
          const childR = Math.max(
            child.roughness[i00],
            child.roughness[i10],
            child.roughness[i01],
            child.roughness[i11],
          )
          const idx = r * cols + c
          maxElev[idx] = maxH
          roughness[idx] = Math.max(childR, (maxH - minH) * 0.5)
        }
      }
      levels[l] = { cols, rows, roughness, maxElev }
    }

    return new EarthQuadtree(levels, terrain.width, terrain.height)
  }

  private lookupMetrics(u0: number, v0: number, du: number, dv: number) {
    // Pick pyramid level whose cell size best matches min(du, dv)
    const span = Math.max(du, dv)
    const idealLevel = Math.round(Math.log2(0.5 / Math.max(1e-6, span)))
    const lvlIdx = Math.max(0, Math.min(EARTH_PYRAMID_LEVELS - 1, idealLevel))
    const lvl = this.levels[lvlIdx]
    const c0 = Math.max(0, Math.min(lvl.cols - 1, Math.floor(u0 * lvl.cols)))
    const c1 = Math.max(c0, Math.min(lvl.cols - 1, Math.ceil((u0 + du) * lvl.cols) - 1))
    const r0 = Math.max(0, Math.min(lvl.rows - 1, Math.floor(v0 * lvl.rows)))
    const r1 = Math.max(r0, Math.min(lvl.rows - 1, Math.ceil((v0 + dv) * lvl.rows) - 1))
    let rough = 0
    let maxH = 0
    for (let r = r0; r <= r1; r++) {
      const rowOff = r * lvl.cols
      for (let c = c0; c <= c1; c++) {
        const idx = rowOff + c
        if (lvl.roughness[idx] > rough) rough = lvl.roughness[idx]
        if (lvl.maxElev[idx] > maxH) maxH = lvl.maxElev[idx]
      }
    }
    // Past stored leaf depth, decay roughness gently with each deeper split.
    if (idealLevel > lvlIdx) {
      rough *= Math.pow(0.55, idealLevel - lvlIdx)
    }
    return { rough, maxH }
  }

  /**
   * Select visible Earth patches into `out` (8 floats per patch: u0, v0, du, dv, demMip, skirtDepth, 0, 0).
   * Returns number of emitted patch instances.
   */
  selectPatches(
    out: Float32Array,
    matrix: ArrayLike<number>,
    eye: readonly [number, number, number],
    width: number,
    height: number,
    morph: number,
    activeRelief: number,
    targetErrorPx = 0.75,
  ): number {
    const view = extractCameraView(matrix, width, height)
    const eyeDist = Math.hypot(eye[0], eye[1], eye[2])
    const invEyeDist = eyeDist > 1e-6 ? 1 / eyeDist : 0
    const eyeDirX = eye[0] * invEyeDist
    const eyeDirY = eye[1] * invEyeDist
    const eyeDirZ = eye[2] * invEyeDist
    const useHorizon = morph > 0.995 && eyeDist > 1.002
    const rMin = 1 - EARTH_FLATTENING // polar radius ~0.99665
    const baseHorizonAngle = useHorizon ? Math.acos(Math.min(1, rMin / eyeDist)) : Math.PI
    const reliefScale = activeRelief / EARTH_AXIS
    const isGlobe = morph > 0.995
    const isFlat = morph < 0.005

    let count = 0
    const emit = (u0: number, v0: number, du: number, dv: number, rough: number, curvatureErr: number) => {
      if (count >= MAX_PATCH_INSTANCES) return
      const texelsPerVert = Math.max(du * this.demWidth, dv * this.demHeight) / PATCH_SEGMENTS
      // Cap DEM mip at 2.0 (matching the original ~40 km vertex footprint) so coarse patches
      // never wash out mountain peaks, while zoomed-in patches sharpen down to mip 0.
      const demMip = Math.max(0, Math.min(Math.min(this.maxDemMip, 2), Math.log2(Math.max(1, texelsPerVert)) - 1.5))
      const roughUnit = rough * reliefScale
      const skirtDepth = Math.max(
        curvatureErr * 4.0 + roughUnit * 1.35,
        isFlat ? 0.002 : 0.0012 * Math.max(du, dv) * Math.PI,
      )
      const o = count * INSTANCE_FLOATS
      out[o] = u0
      out[o + 1] = v0
      out[o + 2] = du
      out[o + 3] = dv
      out[o + 4] = demMip
      out[o + 5] = skirtDepth
      out[o + 6] = 0
      out[o + 7] = 0
      count++
    }

    const evalPoint = (u: number, v: number, hUnit: number): Vec3 => {
      const lon = (u - 0.5) * 2 * Math.PI
      const lat = (v - 0.5) * Math.PI
      if (isFlat) {
        const [px, py] = equalEarthProject(lon, lat)
        return [px, py, hUnit]
      }
      const sinLat = Math.sin(lat)
      const cosLat = Math.abs(lat) >= Math.PI / 2 - 1e-7 ? 0 : Math.cos(lat)
      const n = 1 / Math.sqrt(1 - EARTH_E2 * sinLat * sinLat)
      const nx = cosLat * Math.sin(lon)
      const ny = sinLat
      const nz = cosLat * Math.cos(lon)
      const sx = (n + hUnit) * nx
      const sy = (n * (1 - EARTH_E2) + hUnit) * ny
      const sz = (n + hUnit) * nz
      if (isGlobe) return [sx, sy, sz]
      const [px, py] = equalEarthProject(lon, lat)
      const im = 1 - morph
      return [px * im + sx * morph, py * im + sy * morph, hUnit * im + sz * morph]
    }

    const visit = (u0: number, v0: number, du: number, dv: number, depth: number) => {
      if (count >= MAX_PATCH_INSTANCES) return

      const { rough, maxH } = this.lookupMetrics(u0, v0, du, dv)
      const maxHUnit = maxH * reliefScale
      const uMid = u0 + du * 0.5
      const vMid = v0 + dv * 0.5

      // Bounding sphere from center + 4 corners (plus 4 edge midpoints at coarse depth 0-1).
      const center = evalPoint(uMid, vMid, maxHUnit * 0.5)
      const cx = center[0],
        cy = center[1],
        cz = center[2]
      const p00 = evalPoint(u0, v0, 0)
      const p10 = evalPoint(u0 + du, v0, 0)
      const p01 = evalPoint(u0, v0 + dv, 0)
      const p11 = evalPoint(u0 + du, v0 + dv, 0)
      let r2 = Math.max(
        (p00[0] - cx) ** 2 + (p00[1] - cy) ** 2 + (p00[2] - cz) ** 2,
        (p10[0] - cx) ** 2 + (p10[1] - cy) ** 2 + (p10[2] - cz) ** 2,
        (p01[0] - cx) ** 2 + (p01[1] - cy) ** 2 + (p01[2] - cz) ** 2,
        (p11[0] - cx) ** 2 + (p11[1] - cy) ** 2 + (p11[2] - cz) ** 2,
      )
      if (depth <= 1) {
        const pm0 = evalPoint(uMid, v0, maxHUnit)
        const pm1 = evalPoint(uMid, v0 + dv, maxHUnit)
        const p0m = evalPoint(u0, vMid, maxHUnit)
        const p1m = evalPoint(u0 + du, vMid, maxHUnit)
        r2 = Math.max(
          r2,
          (pm0[0] - cx) ** 2 + (pm0[1] - cy) ** 2 + (pm0[2] - cz) ** 2,
          (pm1[0] - cx) ** 2 + (pm1[1] - cy) ** 2 + (pm1[2] - cz) ** 2,
          (p0m[0] - cx) ** 2 + (p0m[1] - cy) ** 2 + (p0m[2] - cz) ** 2,
          (p1m[0] - cx) ** 2 + (p1m[1] - cy) ** 2 + (p1m[2] - cz) ** 2,
        )
      }
      const radius = Math.sqrt(r2) + maxHUnit + 0.004

      // 1. Frustum cull
      if (sphereOutsideFrustum(view, cx, cy, cz, radius)) return

      // 2. Horizon cone cull (globe mode only)
      const lat0 = (v0 - 0.5) * Math.PI
      const lat1 = (v0 + dv - 0.5) * Math.PI
      const latClosestEq = lat0 <= 0 && lat1 >= 0 ? 0 : Math.min(Math.abs(lat0), Math.abs(lat1))
      const cosLatMax = Math.cos(latClosestEq)
      const arcU = du * 2 * Math.PI * cosLatMax
      const arcV = dv * Math.PI
      const maxArc = Math.max(arcU, arcV)

      if (useHorizon) {
        const cLen = Math.hypot(cx, cy, cz)
        if (cLen > 1e-6) {
          const halfCone = Math.min(Math.PI * 0.5, Math.hypot(arcU, arcV) * 0.55)
          const elevHorizon = Math.acos(Math.min(1, rMin / (1 + maxHUnit + 0.002)))
          const maxVisAngle = Math.min(Math.PI, baseHorizonAngle + elevHorizon + halfCone + 0.05)
          const cosDot = (cx * eyeDirX + cy * eyeDirY + cz * eyeDirZ) / cLen
          if (cosDot < Math.cos(maxVisAngle)) return
        }
      }

      // 3. Projected screen-space geometric error (curvature + full terrain roughness + Equal Earth boundary).
      // Flat oceans have rough === 0 and stay coarse; mountain ridges subdivide until sub-pixel accurate.
      const pxPerUnit = pixelsPerUnitForSphere(view, cx, cy, cz, radius)
      const cellHalfAngle = maxArc / (2 * PATCH_SEGMENTS)
      const curvatureErr = morph * (1 - Math.cos(cellHalfAngle))
      // Scale roughness inversely with depth past the root so each subdivision resolves half the ridge error
      // until vertices are dense enough to sample narrow mountain chains cleanly.
      const reliefErr = rough * reliefScale * (depth < 3 ? 1.0 : 0.55)
      const onDatelineEdge = !isGlobe && (u0 <= 1e-5 || u0 + du >= 1 - 1e-5)
      const neatlineErr = onDatelineEdge ? 0.03 * dv * dv * (1 - morph) : 0
      const screenErrorPx = (curvatureErr + reliefErr + neatlineErr) * pxPerUnit
      // Also cap per-quad screen size so curved lighting / Equal Earth projection stays smooth.
      const cellWorldSpan = isGlobe ? maxArc / PATCH_SEGMENTS : Math.max(du * 5.4, dv * 2.65) / PATCH_SEGMENTS
      const cellPx = cellWorldSpan * pxPerUnit

      const shouldSplit =
        depth < EARTH_MAX_DEPTH &&
        count + 4 < MAX_PATCH_INSTANCES &&
        ((cellPx > 6.5 && screenErrorPx > targetErrorPx) || cellPx > 32)

      if (!shouldSplit) {
        emit(u0, v0, du, dv, rough, curvatureErr)
        return
      }

      // Latitude-coarsened splitting: near the poles in globe mode, split 1×2 in latitude only
      // so triangles stay near-isotropic instead of bunching 512 columns into polar slivers.
      const splitU = !isGlobe || arcU >= arcV * 0.55
      const halfV = dv * 0.5
      if (!splitU) {
        visit(u0, v0, du, halfV, depth + 1)
        visit(u0, v0 + halfV, du, halfV, depth + 1)
      } else {
        const halfU = du * 0.5
        visit(u0, v0, halfU, halfV, depth + 1)
        visit(u0 + halfU, v0, halfU, halfV, depth + 1)
        visit(u0, v0 + halfV, halfU, halfV, depth + 1)
        visit(u0 + halfU, v0 + halfV, halfU, halfV, depth + 1)
      }
    }

    const rootDu = 1 / EARTH_ROOT_COLS
    const rootDv = 1 / EARTH_ROOT_ROWS
    for (let r = 0; r < EARTH_ROOT_ROWS; r++) {
      for (let c = 0; c < EARTH_ROOT_COLS; c++) {
        visit(c * rootDu, r * rootDv, rootDu, rootDv, 0)
      }
    }
    return count
  }
}

// ---------------------------------------------------------------------------
// 2. Mars & Moon: Equiangular Cube-Sphere Error-Metric Quadtree (zero polar bunching)
// ---------------------------------------------------------------------------

const CUBE_PYRAMID_LEVELS = 5 // level 0: 1×1 per face ... level 4: 16×16 per face (256×256 effective)
const CUBE_MAX_DEPTH = 5 // up to 32×32 patches/face × 16 = 512×512 per cube face

// Orthonormal cube faces with uAxis × vAxis = +center (CCW outward winding).
const CUBE_FACES: readonly { center: Vec3; u: Vec3; v: Vec3 }[] = [
  { center: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] }, // 0: +X
  { center: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] }, // 1: -X
  { center: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] }, // 2: +Y
  { center: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] }, // 3: -Y
  { center: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, // 4: +Z
  { center: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] }, // 5: -Z
]

function cubeFaceDir(face: number, fx: number, fy: number): Vec3 {
  const f = CUBE_FACES[face]
  const tx = Math.tan(fx * (Math.PI * 0.25))
  const ty = Math.tan(fy * (Math.PI * 0.25))
  const x = f.center[0] + f.u[0] * tx + f.v[0] * ty
  const y = f.center[1] + f.u[1] * tx + f.v[1] * ty
  const z = f.center[2] + f.u[2] * tx + f.v[2] * ty
  const inv = 1 / Math.hypot(x, y, z)
  return [x * inv, y * inv, z * inv]
}

type CubeRoughnessLevel = {
  dim: number
  /** Max elevation deviation from bilinear corners (km, at relief = 1). */
  roughnessKm: Float32Array
  /** Max elevation (km, at relief = 1). */
  maxElevKm: Float32Array
  /** Min elevation (km, at relief = 1). */
  minElevKm: Float32Array
}

export class PlanetCubeQuadtree {
  // levels[face][level]
  private readonly levels: CubeRoughnessLevel[][]
  readonly radiusKm: number
  readonly demHeight: number
  readonly maxDemMip: number

  private constructor(levels: CubeRoughnessLevel[][], radiusKm: number, demHeight: number, demWidth: number) {
    this.levels = levels
    this.radiusKm = radiusKm
    this.demHeight = demHeight
    this.maxDemMip = Math.max(0, Math.floor(Math.log2(Math.max(demWidth, demHeight))))
  }

  static fromTerrain(terrain: Terrain, radiusKm: number): PlanetCubeQuadtree {
    const leafLevel = CUBE_PYRAMID_LEVELS - 1
    const leafDim = 1 << leafLevel // 16
    const allLevels: CubeRoughnessLevel[][] = []

    const sampleElevKm = (dir: Vec3) => {
      const lon = Math.atan2(dir[0], dir[2])
      const lat = Math.atan2(dir[1], Math.hypot(dir[0], dir[2]))
      return elevationAt(terrain, lon, lat) * 0.001
    }

    const cellSpan = 2 / leafDim
    for (let face = 0; face < 6; face++) {
      const leafRough = new Float32Array(leafDim * leafDim)
      const leafMax = new Float32Array(leafDim * leafDim)
      const leafMin = new Float32Array(leafDim * leafDim)

      for (let ry = 0; ry < leafDim; ry++) {
        const fy0 = -1 + ry * cellSpan
        const fy1 = fy0 + cellSpan
        for (let rx = 0; rx < leafDim; rx++) {
          const fx0 = -1 + rx * cellSpan
          const fx1 = fx0 + cellSpan
          const h00 = sampleElevKm(cubeFaceDir(face, fx0, fy0))
          const h10 = sampleElevKm(cubeFaceDir(face, fx1, fy0))
          const h01 = sampleElevKm(cubeFaceDir(face, fx0, fy1))
          const h11 = sampleElevKm(cubeFaceDir(face, fx1, fy1))
          let maxH = Math.max(h00, h10, h01, h11)
          let minH = Math.min(h00, h10, h01, h11)
          let maxErr = 0
          for (let sy = 0; sy <= 4; sy++) {
            const ty = sy * 0.25
            const fy = fy0 + cellSpan * ty
            for (let sx = 0; sx <= 4; sx++) {
              const tx = sx * 0.25
              const h = sampleElevKm(cubeFaceDir(face, fx0 + cellSpan * tx, fy))
              if (h > maxH) maxH = h
              if (h < minH) minH = h
              const interp = (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty
              const err = Math.abs(h - interp)
              if (err > maxErr) maxErr = err
            }
          }
          const idx = ry * leafDim + rx
          leafRough[idx] = Math.max(maxErr, (maxH - minH) * 0.65)
          leafMax[idx] = maxH
          leafMin[idx] = minH
        }
      }

      const faceLevels: CubeRoughnessLevel[] = new Array(CUBE_PYRAMID_LEVELS)
      faceLevels[leafLevel] = { dim: leafDim, roughnessKm: leafRough, maxElevKm: leafMax, minElevKm: leafMin }

      for (let l = leafLevel - 1; l >= 0; l--) {
        const child = faceLevels[l + 1]
        const dim = 1 << l
        const roughnessKm = new Float32Array(dim * dim)
        const maxElevKm = new Float32Array(dim * dim)
        const minElevKm = new Float32Array(dim * dim)
        for (let r = 0; r < dim; r++) {
          const cr = r * 2
          for (let c = 0; c < dim; c++) {
            const cc = c * 2
            const i00 = cr * child.dim + cc
            const i10 = i00 + 1
            const i01 = (cr + 1) * child.dim + cc
            const i11 = i01 + 1
            const maxH = Math.max(
              child.maxElevKm[i00],
              child.maxElevKm[i10],
              child.maxElevKm[i01],
              child.maxElevKm[i11],
            )
            const minH = Math.min(
              child.minElevKm[i00],
              child.minElevKm[i10],
              child.minElevKm[i01],
              child.minElevKm[i11],
            )
            const childR = Math.max(
              child.roughnessKm[i00],
              child.roughnessKm[i10],
              child.roughnessKm[i01],
              child.roughnessKm[i11],
            )
            const idx = r * dim + c
            maxElevKm[idx] = maxH
            minElevKm[idx] = minH
            roughnessKm[idx] = Math.max(childR, (maxH - minH) * 0.35)
          }
        }
        faceLevels[l] = { dim, roughnessKm, maxElevKm, minElevKm }
      }
      allLevels.push(faceLevels)
    }

    return new PlanetCubeQuadtree(allLevels, radiusKm, terrain.height, terrain.width)
  }

  /**
   * Select visible cube-sphere patches into `out`
   * (8 floats per patch: fx0, fy0, size, size, face, demMip, skirtDepthKm, 0).
   */
  selectPatches(
    out: Float32Array,
    matrix: ArrayLike<number>,
    eyeKm: readonly [number, number, number],
    width: number,
    height: number,
    relief: number,
    reliefFade: number,
    targetErrorPx = 0.75,
  ): number {
    const view = extractCameraView(matrix, width, height)
    const eyeDist = Math.hypot(eyeKm[0], eyeKm[1], eyeKm[2])
    const invEye = eyeDist > 1e-6 ? 1 / eyeDist : 0
    const eyeDirX = eyeKm[0] * invEye
    const eyeDirY = eyeKm[1] * invEye
    const eyeDirZ = eyeKm[2] * invEye
    const effRelief = relief * reliefFade
    const rMinSphere = Math.max(1, this.radiusKm - 15 * Math.max(1, effRelief))
    const useHorizon = eyeDist > this.radiusKm * 1.005
    const baseHorizonAngle = useHorizon ? Math.acos(Math.min(1, rMinSphere / eyeDist)) : Math.PI

    let count = 0
    const emit = (face: number, fx0: number, fy0: number, size: number, roughKm: number, curvatureErrKm: number) => {
      if (count >= MAX_PATCH_INSTANCES) return
      const angularSpan = size * (Math.PI * 0.25)
      const texelsPerVert = ((angularSpan / Math.PI) * this.demHeight) / PATCH_SEGMENTS
      // Keep DEM mip sharp (<= 1.0) so crater rims and ridges retain full 3D exaggeration.
      const demMip = Math.max(0, Math.min(Math.min(this.maxDemMip, 1.0), Math.log2(Math.max(1, texelsPerVert)) - 2.5))
      const skirtDepthKm = Math.max(
        curvatureErrKm * 4.5 + roughKm * effRelief * 1.5,
        this.radiusKm * angularSpan * 0.008,
      )
      const o = count * INSTANCE_FLOATS
      out[o] = fx0
      out[o + 1] = fy0
      out[o + 2] = size
      out[o + 3] = size
      out[o + 4] = face
      out[o + 5] = demMip
      out[o + 6] = skirtDepthKm
      out[o + 7] = 0
      count++
    }

    const visit = (face: number, depth: number, ix: number, iy: number, fx0: number, fy0: number, size: number) => {
      if (count >= MAX_PATCH_INSTANCES) return
      const lvlIdx = Math.min(depth, CUBE_PYRAMID_LEVELS - 1)
      const shift = depth - lvlIdx
      const lvl = this.levels[face][lvlIdx]
      const cellIdx = (iy >> shift) * lvl.dim + (ix >> shift)
      const roughKm = lvl.roughnessKm[cellIdx] * (shift > 0 ? Math.pow(0.55, shift) : 1)
      const maxElevKm = lvl.maxElevKm[cellIdx] * effRelief
      const minElevKm = lvl.minElevKm[cellIdx] * effRelief
      const rMid = this.radiusKm + (maxElevKm + minElevKm) * 0.5
      const rTop = this.radiusKm + Math.max(0, maxElevKm)

      const fxMid = fx0 + size * 0.5
      const fyMid = fy0 + size * 0.5
      const dMid = cubeFaceDir(face, fxMid, fyMid)
      const cx = dMid[0] * rMid
      const cy = dMid[1] * rMid
      const cz = dMid[2] * rMid

      // Compute tight bounding sphere radius from the 4 corners + radial thickness
      const c00 = cubeFaceDir(face, fx0, fy0)
      const c10 = cubeFaceDir(face, fx0 + size, fy0)
      const c01 = cubeFaceDir(face, fx0, fy0 + size)
      const c11 = cubeFaceDir(face, fx0 + size, fy0 + size)
      const minDot = Math.min(
        dMid[0] * c00[0] + dMid[1] * c00[1] + dMid[2] * c00[2],
        dMid[0] * c10[0] + dMid[1] * c10[1] + dMid[2] * c10[2],
        dMid[0] * c01[0] + dMid[1] * c01[1] + dMid[2] * c01[2],
        dMid[0] * c11[0] + dMid[1] * c11[1] + dMid[2] * c11[2],
      )
      const halfCone = Math.acos(Math.max(-1, Math.min(1, minDot)))
      const chordRadius = 2 * rTop * Math.sin(halfCone * 0.5)
      const radialPad = Math.max(Math.abs(maxElevKm - minElevKm) * 0.5, this.radiusKm * (1 - Math.cos(halfCone)))
      const radius = chordRadius + radialPad + 2.0

      // 1. Frustum cull
      if (sphereOutsideFrustum(view, cx, cy, cz, radius)) return

      // 2. Horizon cone cull
      if (useHorizon) {
        const elevHorizon = Math.acos(Math.min(1, rMinSphere / Math.max(rMinSphere, rTop)))
        const maxVisAngle = Math.min(Math.PI, baseHorizonAngle + elevHorizon + halfCone + 0.05)
        const cosDot = dMid[0] * eyeDirX + dMid[1] * eyeDirY + dMid[2] * eyeDirZ
        if (cosDot < Math.cos(maxVisAngle)) return
      }

      // 3. Projected screen-space geometric error (curvature + full terrain roughness).
      const pxPerKm = pixelsPerUnitForSphere(view, cx, cy, cz, radius)
      const angularSpan = size * (Math.PI * 0.25)
      const curvatureErrKm = this.radiusKm * (1 - Math.cos(angularSpan / (2 * PATCH_SEGMENTS)))
      const reliefErrKm = roughKm * effRelief * (depth < 3 ? 1.0 : 0.55)
      const screenErrorPx = (curvatureErrKm + reliefErrKm) * pxPerKm
      const cellPx = ((this.radiusKm * angularSpan) / PATCH_SEGMENTS) * pxPerKm

      const shouldSplit =
        depth < CUBE_MAX_DEPTH &&
        count + 4 < MAX_PATCH_INSTANCES &&
        ((cellPx > 6.5 && screenErrorPx > targetErrorPx) || cellPx > 32)

      if (!shouldSplit) {
        emit(face, fx0, fy0, size, roughKm, curvatureErrKm)
        return
      }

      const half = size * 0.5
      const cx2 = ix * 2
      const cy2 = iy * 2
      visit(face, depth + 1, cx2, cy2, fx0, fy0, half)
      visit(face, depth + 1, cx2 + 1, cy2, fx0 + half, fy0, half)
      visit(face, depth + 1, cx2, cy2 + 1, fx0, fy0 + half, half)
      visit(face, depth + 1, cx2 + 1, cy2 + 1, fx0 + half, fy0 + half, half)
    }

    for (let face = 0; face < 6; face++) {
      visit(face, 0, 0, 0, -1, -1, 2)
    }
    return count
  }
}
