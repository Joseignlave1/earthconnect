import { shaderModule } from '../shader-test/gpu.ts'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import crewAstronautWgsl from './crewAstronaut.wgsl?raw'

export type AstronautPoseId = 'cling' | 'laptop' | 'brace' | 'hang' | 'tumble' | 'accept'

export type CrewAstronautSpec = {
  id: string
  name: string
  pose: AstronautPoseId
  /** Optional portrait image URL placed directly on the visor sticky note. */
  photo?: string
}

type Vec3 = [number, number, number]

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s]
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function norm(a: Vec3): Vec3 {
  const len = Math.hypot(a[0], a[1], a[2])
  return len > 1e-6 ? [a[0] / len, a[1] / len, a[2] / len] : [0, 1, 0]
}

function rotX(p: Vec3, a: number): Vec3 {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]
}

function rotY(p: Vec3, a: number): Vec3 {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

function rotZ(p: Vec3, a: number): Vec3 {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]
}

function basisFromAxis(axisRaw: Vec3): { u: Vec3; v: Vec3; w: Vec3 } {
  const w = norm(axisRaw)
  const ref: Vec3 = Math.abs(w[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0]
  const u = norm(cross(ref, w))
  const v = cross(w, u)
  return { u, v, w }
}

class MeshBuilder {
  readonly verts: number[] = [] // 10 floats/vertex: pos(3), normal(3), uv(2), mat(1), wobble(1)
  readonly indices: number[] = []
  private vertexCount = 0

  pushVertex(p: Vec3, n: Vec3, u: number, v: number, matId: number, wobble: number) {
    this.verts.push(p[0], p[1], p[2], n[0], n[1], n[2], u, v, matId, wobble)
    return this.vertexCount++
  }

  pushTri(a: number, b: number, c: number) {
    this.indices.push(a, b, c)
  }

  /** Smooth sphere/ellipsoid at `center` with per-axis radii `radii`. */
  addEllipsoid(
    center: Vec3,
    radii: Vec3,
    matId: number,
    wobble: number,
    segU = 22,
    segV = 14,
    rot?: [number, number, number],
  ) {
    const start = this.vertexCount
    const xform = (v: Vec3): Vec3 => {
      let q = v
      if (rot) {
        q = rotX(q, rot[0])
        q = rotY(q, rot[1])
        q = rotZ(q, rot[2])
      }
      return q
    }
    for (let iv = 0; iv <= segV; iv++) {
      const v = iv / segV
      const phi = v * Math.PI
      const sp = Math.sin(phi)
      const cp = Math.cos(phi)
      for (let iu = 0; iu <= segU; iu++) {
        const u = iu / segU
        const theta = u * Math.PI * 2
        const ct = Math.cos(theta)
        const st = Math.sin(theta)
        const localPos: Vec3 = [sp * ct * radii[0], cp * radii[1], sp * st * radii[2]]
        const localNorm = norm([
          (sp * ct) / Math.max(0.01, radii[0]),
          cp / Math.max(0.01, radii[1]),
          (sp * st) / Math.max(0.01, radii[2]),
        ])
        this.pushVertex(add(center, xform(localPos)), norm(xform(localNorm)), u, v, matId, wobble)
      }
    }
    const stride = segU + 1
    for (let iv = 0; iv < segV; iv++) {
      for (let iu = 0; iu < segU; iu++) {
        const a = start + iv * stride + iu
        const b = a + 1
        const c = a + stride
        const d = c + 1
        this.pushTri(a, c, b)
        this.pushTri(b, c, d)
      }
    }
  }

  /** Smooth tapered capsule from point A to point B (optional cap flattening to blend pelvis/joints cleanly). */
  addCapsule(
    a: Vec3,
    b: Vec3,
    rA: number,
    rB: number,
    matId: number,
    wobble: number,
    segU = 18,
    segLen = 10,
    capScaleA = 1,
    capScaleB = 1,
  ) {
    const dir = sub(b, a)
    const len = Math.hypot(dir[0], dir[1], dir[2])
    if (len < 1e-4) return
    const { u: bu, v: bv, w: bw } = basisFromAxis(dir)
    const capSteps = 5
    const rings: { pos: Vec3; r: number; nAxial: number; vCoord: number }[] = []

    for (let i = 0; i < capSteps; i++) {
      const t = i / capSteps
      const ang = -Math.PI / 2 + t * (Math.PI / 2)
      rings.push({
        pos: add(a, scale(bw, Math.sin(ang) * rA * capScaleA)),
        r: Math.max(0.002, Math.cos(ang) * rA),
        nAxial: Math.sin(ang),
        vCoord: t * 0.15,
      })
    }
    for (let i = 0; i <= segLen; i++) {
      const t = i / segLen
      rings.push({
        pos: add(a, scale(bw, t * len)),
        r: rA + (rB - rA) * t,
        nAxial: 0,
        vCoord: 0.15 + t * 0.7,
      })
    }
    for (let i = 1; i <= capSteps; i++) {
      const t = i / capSteps
      const ang = t * (Math.PI / 2)
      rings.push({
        pos: add(b, scale(bw, Math.sin(ang) * rB * capScaleB)),
        r: Math.max(0.002, Math.cos(ang) * rB),
        nAxial: Math.sin(ang),
        vCoord: 0.85 + t * 0.15,
      })
    }

    const start = this.vertexCount
    const stride = segU + 1
    for (const ring of rings) {
      const nRadial = Math.sqrt(Math.max(0, 1 - ring.nAxial * ring.nAxial))
      for (let iu = 0; iu <= segU; iu++) {
        const u = iu / segU
        const theta = u * Math.PI * 2
        const ct = Math.cos(theta)
        const st = Math.sin(theta)
        const radialDir = add(scale(bu, ct), scale(bv, st))
        const p = add(ring.pos, scale(radialDir, ring.r))
        const n = norm(add(scale(radialDir, nRadial), scale(bw, ring.nAxial)))
        this.pushVertex(p, n, u, ring.vCoord, matId, wobble)
      }
    }
    for (let ir = 0; ir < rings.length - 1; ir++) {
      for (let iu = 0; iu < segU; iu++) {
        const i0 = start + ir * stride + iu
        const i1 = i0 + 1
        const i2 = i0 + stride
        const i3 = i2 + 1
        this.pushTri(i0, i1, i2)
        this.pushTri(i1, i3, i2)
      }
    }
  }

  /** Torus collar / helmet rim ring centered at `center` with normal `axis`. */
  addTorus(
    center: Vec3,
    axis: Vec3,
    majorR: number,
    minorR: number,
    matId: number,
    wobble: number,
    segU = 26,
    segV = 12,
  ) {
    const { u: bu, v: bv, w: bw } = basisFromAxis(axis)
    const start = this.vertexCount
    const stride = segV + 1
    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU
      const th = u * Math.PI * 2
      const radial = add(scale(bu, Math.cos(th)), scale(bv, Math.sin(th)))
      const ringCenter = add(center, scale(radial, majorR))
      for (let iv = 0; iv <= segV; iv++) {
        const v = iv / segV
        const ph = v * Math.PI * 2
        const n = norm(add(scale(radial, Math.cos(ph)), scale(bw, Math.sin(ph))))
        const p = add(ringCenter, scale(n, minorR))
        this.pushVertex(p, n, u, v, matId, wobble)
      }
    }
    for (let iu = 0; iu < segU; iu++) {
      for (let iv = 0; iv < segV; iv++) {
        const i0 = start + iu * stride + iv
        const i1 = i0 + 1
        const i2 = i0 + stride
        const i3 = i2 + 1
        this.pushTri(i0, i2, i1)
        this.pushTri(i1, i2, i3)
      }
    }
  }

  /** Rounded chunky box (backpack, chest unit, floating laptop). */
  addBox(center: Vec3, half: Vec3, rot: [number, number, number], matId: number, wobble: number) {
    const xform = (v: Vec3): Vec3 => rotZ(rotY(rotX(v, rot[0]), rot[1]), rot[2])
    const faces: { n: Vec3; u: Vec3; v: Vec3 }[] = [
      { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
      { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
      { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
      { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
      { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
      { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    ]
    for (const f of faces) {
      const start = this.vertexCount
      const seg = 4
      for (let iy = 0; iy <= seg; iy++) {
        const tv = iy / seg
        const sv = tv * 2 - 1
        for (let ix = 0; ix <= seg; ix++) {
          const tu = ix / seg
          const su = tu * 2 - 1
          const raw: Vec3 = [
            (f.n[0] + f.u[0] * su + f.v[0] * sv) * half[0],
            (f.n[1] + f.u[1] * su + f.v[1] * sv) * half[1],
            (f.n[2] + f.u[2] * su + f.v[2] * sv) * half[2],
          ]
          this.pushVertex(add(center, xform(raw)), norm(xform(f.n)), tu, 1 - tv, matId, wobble)
        }
      }
      const stride = seg + 1
      for (let iy = 0; iy < seg; iy++) {
        for (let ix = 0; ix < seg; ix++) {
          const i0 = start + iy * stride + ix
          const i1 = i0 + 1
          const i2 = i0 + stride
          const i3 = i2 + 1
          this.pushTri(i0, i1, i2)
          this.pushTri(i1, i3, i2)
        }
      }
    }
  }

  /** Curved sticky note + translucent Scotch tape quad slapped right onto the front visor. */
  addStickyNote(helmetCenter: Vec3, headRot: [number, number, number], noteTilt: number, wobble: number) {
    const xformHead = (v: Vec3): Vec3 => rotZ(rotY(rotX(v, headRot[0]), headRot[1]), headRot[2])
    const seg = 10
    const halfW = 0.255
    const halfH = 0.255
    const baseZ = 0.48
    const start = this.vertexCount

    for (let iy = 0; iy <= seg; iy++) {
      const v = iy / seg
      const ny = 1 - v * 2 // +1 top .. -1 bottom
      for (let ix = 0; ix <= seg; ix++) {
        const u = ix / seg
        const nx = u * 2 - 1 // -1 left .. +1 right
        // Gentle wrap around helmet curvature + charming lifted bottom-right corner curl!
        const cornerCurl = Math.max(0, (nx + 0.1) * (-ny + 0.1)) * 0.055
        const domeCurve = -(nx * nx + ny * ny) * 0.048
        const local = rotZ([nx * halfW, ny * halfH, baseZ + domeCurve + cornerCurl], noteTilt)
        const nLocal = norm(rotZ([nx * 0.26 - cornerCurl * 1.4, ny * 0.26 + cornerCurl * 1.4, 1], noteTilt))
        this.pushVertex(add(helmetCenter, xformHead(local)), norm(xformHead(nLocal)), u, v, 4, wobble)
      }
    }
    const stride = seg + 1
    for (let iy = 0; iy < seg; iy++) {
      for (let ix = 0; ix < seg; ix++) {
        const i0 = start + iy * stride + ix
        const i1 = i0 + 1
        const i2 = i0 + stride
        const i3 = i2 + 1
        this.pushTri(i0, i2, i1)
        this.pushTri(i1, i2, i3)
      }
    }

    // Translucent piece of Scotch tape holding the top of the sticky note to the helmet visor.
    const tStart = this.vertexCount
    const tapeW = 0.095
    const tapeH = 0.048
    const tapeY = halfH + 0.008
    const tapePts: [number, number][] = [
      [-tapeW, tapeY + tapeH],
      [tapeW, tapeY + tapeH],
      [-tapeW, tapeY - tapeH],
      [tapeW, tapeY - tapeH],
    ]
    const tapeUvs: [number, number][] = [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]
    for (let i = 0; i < 4; i++) {
      const [tx, ty] = tapePts[i]
      const dome = -(tx * tx + ty * ty) * 0.55
      const local = rotZ([tx, ty, baseZ + 0.022 + dome], noteTilt - 0.07)
      this.pushVertex(
        add(helmetCenter, xformHead(local)),
        norm(xformHead([0, 0.15, 1])),
        tapeUvs[i][0],
        tapeUvs[i][1],
        5,
        wobble,
      )
    }
    this.pushTri(tStart, tStart + 2, tStart + 1)
    this.pushTri(tStart + 1, tStart + 2, tStart + 3)
  }
}

type PoseSpec = {
  bodyRot: [number, number, number]
  bodyOffset: Vec3
  headRot: [number, number, number]
  noteTilt: number
  lElbow: Vec3
  lHand: Vec3
  rElbow: Vec3
  rHand: Vec3
  lKnee: Vec3
  lFoot: Vec3
  rKnee: Vec3
  rFoot: Vec3
  hasLaptop?: boolean
}

const POSE_SPECS: Record<AstronautPoseId, PoseSpec> = {
  // 1. Reaching straight up-left clawing desperately at empty space while legs noodle downward
  cling: {
    bodyRot: [0.04, 0.16, 0.2],
    bodyOffset: [-0.14, 0.36, 0],
    headRot: [0.04, 0.08, -0.06],
    noteTilt: 0.07,
    lElbow: [-0.62, 0.72, 0.22],
    lHand: [-0.74, 1.32, 0.36],
    rElbow: [0.36, 0.82, 0.28],
    rHand: [0.14, 1.44, 0.42],
    lKnee: [-0.14, -0.78, 0.08],
    lFoot: [0.06, -1.65, -0.06],
    rKnee: [0.34, -0.72, -0.06],
    rFoot: [0.56, -1.58, -0.14],
  },
  // 2. Desperately reaching for a tiny floating laptop drifting just out of grasp!
  laptop: {
    bodyRot: [0.04, -0.18, 0.24],
    bodyOffset: [-0.18, 0.32, 0],
    headRot: [0.04, -0.06, -0.1],
    noteTilt: -0.08,
    lElbow: [-0.7, 0.36, 0.16],
    lHand: [-1.02, 0.72, 0.28],
    rElbow: [0.46, 0.66, 0.34],
    rHand: [0.68, 1.24, 0.46],
    lKnee: [-0.08, -0.82, 0.1],
    lFoot: [0.18, -1.64, -0.04],
    rKnee: [0.38, -0.68, -0.12],
    rFoot: [0.66, -1.48, -0.18],
    hasLaptop: true,
  },
  // 3. Heroically bracing both chunky boots against literally nothing while already stretching
  brace: {
    bodyRot: [-0.06, 0.22, -0.12],
    bodyOffset: [-0.08, 0.34, 0],
    headRot: [0.08, -0.1, 0.08],
    noteTilt: 0.1,
    lElbow: [-0.64, 0.18, 0.4],
    lHand: [-0.28, 0.48, 0.56],
    rElbow: [0.58, 0.28, 0.36],
    rHand: [0.16, 0.52, 0.58],
    lKnee: [-0.26, -0.68, 0.34],
    lFoot: [-0.02, -1.52, 0.14],
    rKnee: [0.34, -0.62, 0.28],
    rFoot: [0.54, -1.48, 0.08],
  },
  // 4. Hanging on by one raised hand clutching thin air, other arm flailing sideways
  hang: {
    bodyRot: [0.04, 0.14, 0.28],
    bodyOffset: [-0.14, 0.36, 0],
    headRot: [0.04, 0.06, -0.14],
    noteTilt: -0.06,
    lElbow: [-0.52, 0.86, 0.12],
    lHand: [-0.46, 1.54, 0.18],
    rElbow: [0.68, 0.24, 0.22],
    rHand: [1.08, 0.42, 0.36],
    lKnee: [-0.04, -0.84, 0.04],
    lFoot: [0.22, -1.7, -0.06],
    rKnee: [0.36, -0.78, -0.1],
    rFoot: [0.64, -1.58, -0.18],
  },
  // 5. Chaotic spin-tumble sideways with windmilling arms
  tumble: {
    bodyRot: [0.12, -0.14, 0.46],
    bodyOffset: [-0.18, 0.32, 0],
    headRot: [0.06, 0.12, -0.26],
    noteTilt: 0.12,
    lElbow: [-0.74, 0.52, 0.28],
    lHand: [-1.06, 0.98, 0.12],
    rElbow: [0.58, 0.62, -0.12],
    rHand: [0.96, 1.02, 0.18],
    lKnee: [0.02, -0.76, 0.24],
    lFoot: [0.36, -1.54, 0.08],
    rKnee: [0.42, -0.62, -0.22],
    rFoot: [0.74, -1.42, -0.26],
  },
  // 6. Dramatic tragic acceptance (arms wide open, head tilted slightly into the abyss)
  accept: {
    bodyRot: [-0.1, 0.1, 0.14],
    bodyOffset: [-0.1, 0.34, 0],
    headRot: [0.02, 0.04, 0.04],
    noteTilt: -0.04,
    lElbow: [-0.66, 0.32, 0.18],
    lHand: [-1.08, 0.54, 0.32],
    rElbow: [0.66, 0.34, 0.18],
    rHand: [1.06, 0.58, 0.34],
    lKnee: [-0.16, -0.78, 0.1],
    lFoot: [0.04, -1.66, -0.04],
    rKnee: [0.24, -0.76, 0.06],
    rFoot: [0.48, -1.62, -0.08],
  },
}

function buildAstronautGeometry(poseId: AstronautPoseId): { vertices: Float32Array; indices: Uint32Array } {
  const pose = POSE_SPECS[poseId]
  const m = new MeshBuilder()

  const xf = (p: Vec3): Vec3 =>
    add(pose.bodyOffset, rotZ(rotY(rotX(p, pose.bodyRot[0]), pose.bodyRot[1]), pose.bodyRot[2]))
  const worldHeadRot: [number, number, number] = [
    pose.bodyRot[0] + pose.headRot[0],
    pose.bodyRot[1] + pose.headRot[1],
    pose.bodyRot[2] + pose.headRot[2],
  ]

  // Chunky torso + collar + chest telemetry box + backpack life-support unit
  const pelvis = xf([0, -0.22, 0])
  const chest = xf([0, 0.3, 0])
  const neck = xf([0, 0.44, 0.02])
  const headCenter = xf([0, 0.78, 0.04])

  // Smoothly tapered chunky torso with flattened bottom cap so thighs flow cleanly with zero protruding middle bulge
  m.addCapsule(pelvis, chest, 0.28, 0.36, 0, 0, 26, 22, 0.24, 1)
  m.addTorus(neck, sub(headCenter, chest), 0.29, 0.068, 1, 0, 26, 12)
  m.addBox(xf([0, 0.08, 0.33]), [0.16, 0.13, 0.075], pose.bodyRot, 2, 0)
  m.addBox(xf([0, 0.04, -0.34]), [0.27, 0.34, 0.16], pose.bodyRot, 2, 0)
  // Twin oxygen cylinders flanking the backpack
  m.addCapsule(xf([-0.14, -0.22, -0.44]), xf([-0.14, 0.26, -0.44]), 0.085, 0.085, 1, 0, 14, 10)
  m.addCapsule(xf([0.14, -0.22, -0.44]), xf([0.14, 0.26, -0.44]), 0.085, 0.085, 1, 0, 14, 10)

  // Chunky round helmet + reflective visor + metallic faceplate rim + taped sticky note!
  m.addEllipsoid(headCenter, [0.41, 0.4, 0.41], 0, 0, 26, 18, worldHeadRot)
  const visorOffset = rotZ(rotY(rotX([0, 0, 0.16], worldHeadRot[0]), worldHeadRot[1]), worldHeadRot[2])
  m.addEllipsoid(add(headCenter, visorOffset), [0.33, 0.31, 0.29], 3, 0, 24, 16, worldHeadRot)
  const rimOffset = rotZ(rotY(rotX([0, 0, 0.29], worldHeadRot[0]), worldHeadRot[1]), worldHeadRot[2])
  const rimAxis = rotZ(rotY(rotX([0, 0, 1], worldHeadRot[0]), worldHeadRot[1]), worldHeadRot[2])
  m.addTorus(add(headCenter, rimOffset), rimAxis, 0.31, 0.042, 1, 0, 26, 10)
  m.addStickyNote(headCenter, worldHeadRot, pose.noteTilt, 0)

  // Left & right chunky articulated arms + wrist cuffs + rounded EVA mittens
  const lShoulder = xf([-0.38, 0.26, 0.02])
  const rShoulder = xf([0.38, 0.26, 0.02])
  const lElbow = add(pose.bodyOffset, pose.lElbow)
  const lHand = add(pose.bodyOffset, pose.lHand)
  const rElbow = add(pose.bodyOffset, pose.rElbow)
  const rHand = add(pose.bodyOffset, pose.rHand)

  m.addCapsule(lShoulder, lElbow, 0.135, 0.125, 0, 0, 16, 12)
  m.addCapsule(lElbow, lHand, 0.12, 0.105, 0, 0, 16, 12)
  m.addTorus(add(lElbow, scale(sub(lHand, lElbow), 0.84)), sub(lHand, lElbow), 0.112, 0.028, 1, 0, 16, 8)
  m.addEllipsoid(lHand, [0.13, 0.13, 0.11], 2, 0, 14, 10)

  m.addCapsule(rShoulder, rElbow, 0.135, 0.125, 0, 0, 16, 12)
  m.addCapsule(rElbow, rHand, 0.12, 0.105, 0, 0, 16, 12)
  m.addTorus(add(rElbow, scale(sub(rHand, rElbow), 0.84)), sub(rHand, rElbow), 0.112, 0.028, 1, 0, 16, 8)
  m.addEllipsoid(rHand, [0.13, 0.13, 0.11], 2, 0, 14, 10)

  // Left & right chunky legs flowing cleanly from the pelvis + dense rings for silky spiral twist & noodle curve
  const lHip = xf([-0.145, -0.2, 0])
  const rHip = xf([0.145, -0.2, 0])
  const lKnee = add(pose.bodyOffset, pose.lKnee)
  const lFoot = add(pose.bodyOffset, pose.lFoot)
  const rKnee = add(pose.bodyOffset, pose.rKnee)
  const rFoot = add(pose.bodyOffset, pose.rFoot)

  m.addCapsule(lHip, lKnee, 0.168, 0.145, 0, 0, 22, 26)
  m.addCapsule(lKnee, lFoot, 0.145, 0.13, 0, 0, 22, 32)
  m.addEllipsoid(add(lFoot, [0.02, -0.06, 0.08]), [0.16, 0.13, 0.22], 2, 0, 14, 10)

  m.addCapsule(rHip, rKnee, 0.168, 0.145, 0, 0, 22, 26)
  m.addCapsule(rKnee, rFoot, 0.145, 0.13, 0, 0, 22, 32)
  m.addEllipsoid(add(rFoot, [0.04, -0.06, 0.08]), [0.16, 0.13, 0.22], 2, 0, 14, 10)

  // Snapped umbilical tether cord drifting gently into the vortex (tetherMask = 1)
  const tetherA = xf([0.28, -0.22, -0.16])
  const tetherB = add(tetherA, [0.36, -0.68, -0.08])
  const tetherC = add(tetherB, [0.42, -0.92, -0.06])
  m.addCapsule(tetherA, tetherB, 0.026, 0.024, 1, 0.75, 10, 20)
  m.addCapsule(tetherB, tetherC, 0.024, 0.02, 1, 1.0, 10, 24)

  // Floating open laptop slipping just out of reach on the 'laptop' pose!
  if (pose.hasLaptop) {
    const lapBase = add(pose.bodyOffset, [0.96, 1.56, 0.36])
    m.addBox(lapBase, [0.22, 0.02, 0.15], [0.28, -0.42, -0.32], 2, 0)
    m.addBox(add(lapBase, [-0.04, 0.15, -0.11]), [0.22, 0.15, 0.016], [-0.24, -0.42, -0.32], 6, 0)
  }

  return { vertices: new Float32Array(m.verts), indices: new Uint32Array(m.indices) }
}

/** Draw a Content-Warning-inspired sticky-note face (or photo when provided) onto a 512x512 canvas. */
async function renderStickyNoteCanvas(spec: CrewAstronautSpec): Promise<HTMLCanvasElement> {
  const size = 512
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!

  // Warm matte Post-it paper base with subtle edge shading
  const grad = ctx.createLinearGradient(0, 0, size, size)
  grad.addColorStop(0, '#f8e99b')
  grad.addColorStop(0.65, '#f2db78')
  grad.addColorStop(1, '#dfc25c')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)

  // Subtle top adhesive band shadow
  ctx.fillStyle = 'rgba(130, 100, 30, 0.09)'
  ctx.fillRect(0, 0, size, 48)

  if (spec.photo) {
    try {
      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const el = new Image()
        el.crossOrigin = 'anonymous'
        el.onload = () => resolve(el)
        el.onerror = reject
        el.src = spec.photo!
      })
      const pad = 34
      const boxW = size - pad * 2
      const boxH = size - pad * 2 - 54
      ctx.save()
      ctx.beginPath()
      ctx.rect(pad, pad, boxW, boxH)
      ctx.clip()
      const scaleCover = Math.max(boxW / img.width, boxH / img.height)
      const dw = img.width * scaleCover
      const dh = img.height * scaleCover
      ctx.drawImage(img, pad + (boxW - dw) / 2, pad + (boxH - dh) / 2, dw, dh)
      ctx.restore()
      ctx.strokeStyle = 'rgba(40, 32, 18, 0.25)'
      ctx.lineWidth = 4
      ctx.strokeRect(pad, pad, boxW, boxH)
    } catch {
      drawPanicDoodle(ctx, spec.pose, size)
    }
  } else {
    drawPanicDoodle(ctx, spec.pose, size)
  }

  // Handwritten sharpie first name along bottom of sticky note
  ctx.fillStyle = '#1b1814'
  ctx.font = '600 40px "DM Sans", system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(spec.name, size / 2, size - 46)

  return canvas
}

function drawPanicDoodle(ctx: CanvasRenderingContext2D, pose: AstronautPoseId, size: number) {
  ctx.strokeStyle = '#171512'
  ctx.fillStyle = '#171512'
  ctx.lineWidth = 18
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  const cx = size / 2
  const cy = size * 0.44

  if (pose === 'cling') {
    // Wide-eyed O_O panic + screaming oval + sweat drop
    ctx.beginPath()
    ctx.arc(cx - 82, cy - 34, 42, 0, Math.PI * 2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx + 82, cy - 34, 42, 0, Math.PI * 2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx - 76, cy - 42, 14, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx + 76, cy - 42, 14, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.ellipse(cx, cy + 68, 38, 52, 0, 0, Math.PI * 2)
    ctx.stroke()
    // Sweat drop
    ctx.fillStyle = '#3b82c4'
    ctx.beginPath()
    ctx.moveTo(cx + 158, cy - 115)
    ctx.quadraticCurveTo(cx + 186, cy - 64, cx + 158, cy - 50)
    ctx.quadraticCurveTo(cx + 130, cy - 64, cx + 158, cy - 115)
    ctx.fill()
  } else if (pose === 'laptop') {
    // Furious/desperate >:0 brows
    ctx.beginPath()
    ctx.moveTo(cx - 126, cy - 82)
    ctx.lineTo(cx - 40, cy - 48)
    ctx.moveTo(cx + 126, cy - 82)
    ctx.lineTo(cx + 40, cy - 48)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx - 76, cy - 12, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx + 76, cy - 12, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.ellipse(cx, cy + 72, 54, 40, 0, 0, Math.PI * 2)
    ctx.stroke()
  } else if (pose === 'brace') {
    // Squeezed-shut >_< panic face + wavy clenched mouth
    ctx.beginPath()
    ctx.moveTo(cx - 118, cy - 64)
    ctx.lineTo(cx - 48, cy - 26)
    ctx.lineTo(cx - 118, cy + 12)
    ctx.moveTo(cx + 118, cy - 64)
    ctx.lineTo(cx + 48, cy - 26)
    ctx.lineTo(cx + 118, cy + 12)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(cx - 92, cy + 76)
    ctx.quadraticCurveTo(cx - 46, cy + 44, cx, cy + 76)
    ctx.quadraticCurveTo(cx + 46, cy + 108, cx + 92, cy + 76)
    ctx.stroke()
  } else if (pose === 'hang') {
    // Crying T_T panic face
    ctx.beginPath()
    ctx.moveTo(cx - 124, cy - 44)
    ctx.lineTo(cx - 34, cy - 44)
    ctx.moveTo(cx - 79, cy - 44)
    ctx.lineTo(cx - 79, cy + 44)
    ctx.moveTo(cx + 34, cy - 44)
    ctx.lineTo(cx + 124, cy - 44)
    ctx.moveTo(cx + 79, cy - 44)
    ctx.lineTo(cx + 79, cy + 44)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx, cy + 92, 34, Math.PI * 1.1, Math.PI * 1.9)
    ctx.stroke()
  } else if (pose === 'tumble') {
    // Dizzy @_@ spirals
    const drawSpiral = (sx: number, sy: number) => {
      ctx.beginPath()
      for (let i = 0; i <= 54; i++) {
        const a = i * 0.34
        const r = i * 0.92
        const px = sx + Math.cos(a) * r
        const py = sy + Math.sin(a) * r
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.stroke()
    }
    ctx.lineWidth = 14
    drawSpiral(cx - 82, cy - 26)
    drawSpiral(cx + 82, cy - 26)
    ctx.lineWidth = 16
    ctx.beginPath()
    ctx.arc(cx + 12, cy + 82, 26, 0, Math.PI * 2)
    ctx.stroke()
  } else {
    // Deadpan ._. tragic acceptance
    ctx.beginPath()
    ctx.arc(cx - 84, cy - 24, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx + 84, cy - 24, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.moveTo(cx - 64, cy + 62)
    ctx.lineTo(cx + 64, cy + 62)
    ctx.stroke()
  }
}

function perspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovRad / 2)
  const nf = 1 / (near - far)
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0])
}

function lookAtMatrix(eye: Vec3, target: Vec3, up: Vec3): Float32Array {
  const z = norm(sub(eye, target))
  const x = norm(cross(up, z))
  const y = cross(z, x)
  return new Float32Array([
    x[0],
    y[0],
    z[0],
    0,
    x[1],
    y[1],
    z[1],
    0,
    x[2],
    y[2],
    z[2],
    0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
    1,
  ])
}

function mulMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] + a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3]
    }
  }
  return out
}

type CrewPipeline = {
  pipeline: GPURenderPipeline
  layout: GPUBindGroupLayout
}

const crewPipelineCache = new WeakMap<GPUDevice, Map<GPUTextureFormat, Promise<CrewPipeline>>>()

function getCrewPipeline(device: GPUDevice, format: GPUTextureFormat): Promise<CrewPipeline> {
  let byFormat = crewPipelineCache.get(device)
  if (!byFormat) {
    byFormat = new Map()
    crewPipelineCache.set(device, byFormat)
  }
  let pending = byFormat.get(format)
  if (!pending) {
    pending = (async () => {
      const module = await shaderModule(device, 'Crew astronaut spaghettification shader', crewAstronautWgsl)
      const layout = device.createBindGroupLayout({
        label: 'Crew astronaut layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      })
      const pipeline = await device.createRenderPipelineAsync({
        label: 'Crew astronaut 3D MSAA',
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: {
          module,
          entryPoint: 'vsMain',
          buffers: [
            {
              arrayStride: 40,
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },
                { shaderLocation: 1, offset: 12, format: 'float32x3' },
                { shaderLocation: 2, offset: 24, format: 'float32x2' },
                { shaderLocation: 3, offset: 32, format: 'float32x2' },
              ],
            },
          ],
        },
        fragment: {
          module,
          entryPoint: 'fsMain',
          targets: [
            {
              format,
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              },
            },
          ],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 4 },
      })
      return { pipeline, layout }
    })()
    byFormat.set(format, pending)
  }
  return pending
}

type PreparedCrewMesh = {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  noteTexture: GPUTexture
  bindGroup: GPUBindGroup
}

export class CrewAstronautStage {
  private readonly uniformData = new Float32Array(28) // mat4 (16) + 3 * vec4f (12)
  private readonly uniformBuffer: GPUBuffer
  private readonly sampler: GPUSampler
  private readonly meshes = new Map<string, PreparedCrewMesh>()
  private msaaColor: GPUTexture | null = null
  private depthTex: GPUTexture | null = null
  private texW = 0
  private texH = 0
  private activeId: string
  private active = true
  private hovered = false
  private hoverSmooth = 0
  private pointerNorm = { x: 0, y: 0 }
  private smoothPointer = { x: 0, y: 0 }
  private raf = 0
  private lastTime = performance.now() * 0.001
  private startedAt = performance.now() * 0.001
  private fadeIn = 1
  private destroyed = false
  private readonly reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  private readonly observer: ResizeObserver

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly context: GPUCanvasContext,
    private readonly device: GPUDevice,
    private readonly lease: { release: () => void },
    private readonly format: GPUTextureFormat,
    private readonly gpu: CrewPipeline,
    specs: readonly CrewAstronautSpec[],
    initialId: string,
  ) {
    this.activeId = initialId
    this.uniformBuffer = device.createBuffer({
      label: 'Crew astronaut uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.sampler = device.createSampler({
      label: 'Crew sticky note sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    for (const spec of specs) {
      const { vertices, indices } = buildAstronautGeometry(spec.pose)
      const vertexBuffer = device.createBuffer({
        label: `Crew astronaut verts ${spec.id}`,
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(vertexBuffer, 0, vertices.buffer, vertices.byteOffset, vertices.byteLength)

      const indexBuffer = device.createBuffer({
        label: `Crew astronaut indices ${spec.id}`,
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(indexBuffer, 0, indices.buffer, indices.byteOffset, indices.byteLength)

      const noteTexture = device.createTexture({
        label: `Crew sticky note ${spec.id}`,
        size: [512, 512, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })

      const bindGroup = device.createBindGroup({
        layout: gpu.layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: noteTexture.createView() },
          { binding: 2, resource: this.sampler },
        ],
      })

      this.meshes.set(spec.id, {
        vertexBuffer,
        indexBuffer,
        indexCount: indices.length,
        noteTexture,
        bindGroup,
      })

      void renderStickyNoteCanvas(spec).then(noteCanvas => {
        if (this.destroyed) return
        device.queue.copyExternalImageToTexture({ source: noteCanvas }, { texture: noteTexture }, [512, 512])
        if (this.reducedMotion.matches) this.drawOnce()
      })
    }

    this.observer = new ResizeObserver(() => {
      if (!this.destroyed && this.reducedMotion.matches) this.drawOnce()
    })
    this.observer.observe(canvas)
    this.schedule()
  }

  static async create(
    canvas: HTMLCanvasElement,
    acquireDevice: ReturnType<typeof createSurfaceDevicePool>,
    specs: readonly CrewAstronautSpec[],
    initialId: string,
    signal?: AbortSignal,
  ) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('WebGPU unavailable for crew stage.')
    const lease = acquireDevice()
    const { device } = await lease.ready
    if (signal?.aborted) {
      lease.release()
      throw new DOMException('Aborted', 'AbortError')
    }
    const format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format, alphaMode: 'premultiplied' })
    const gpu = await getCrewPipeline(device, format)
    if (signal?.aborted) {
      lease.release()
      throw new DOMException('Aborted', 'AbortError')
    }
    return new CrewAstronautStage(canvas, context, device, lease, format, gpu, specs, initialId)
  }

  setMember(id: string) {
    if (this.activeId === id || !this.meshes.has(id)) return
    this.activeId = id
    this.fadeIn = this.reducedMotion.matches ? 1 : 0.12
    if (this.reducedMotion.matches) this.drawOnce()
  }

  setActive(active: boolean) {
    if (this.active === active) return
    this.active = active
    if (active) {
      this.lastTime = performance.now() * 0.001
      this.schedule()
    } else if (this.raf) {
      cancelAnimationFrame(this.raf)
      this.raf = 0
    }
  }

  setPointer(nx: number, ny: number, hovered: boolean) {
    this.hovered = hovered
    this.pointerNorm.x = nx
    this.pointerNorm.y = ny
  }

  private ensureTargets(w: number, h: number) {
    if (this.msaaColor && this.depthTex && this.texW === w && this.texH === h) return
    this.msaaColor?.destroy()
    this.depthTex?.destroy()
    this.msaaColor = this.device.createTexture({
      label: 'Crew astronaut MSAA color',
      size: [w, h, 1],
      format: this.format,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.depthTex = this.device.createTexture({
      label: 'Crew astronaut MSAA depth',
      size: [w, h, 1],
      format: 'depth24plus',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.texW = w
    this.texH = h
  }

  private drawOnce() {
    if (this.destroyed) return
    const rect = this.canvas.getBoundingClientRect()
    if (rect.width < 2 || rect.height < 2) return
    const w = Math.max(2, Math.round(rect.width))
    const h = Math.max(2, Math.round(rect.height))
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    this.ensureTargets(w, h)

    const now = performance.now() * 0.001
    const dt = Math.min(0.05, Math.max(0.001, now - this.lastTime))
    this.lastTime = now
    const time = now - this.startedAt

    const reduced = this.reducedMotion.matches
    const lerp = reduced ? 1 : 1 - Math.exp(-dt * 7)
    this.hoverSmooth += ((this.hovered ? 1 : 0) - this.hoverSmooth) * lerp
    this.smoothPointer.x += ((this.hovered ? this.pointerNorm.x : 0) - this.smoothPointer.x) * lerp
    this.smoothPointer.y += ((this.hovered ? this.pointerNorm.y : 0) - this.smoothPointer.y) * lerp
    this.fadeIn = Math.min(1, this.fadeIn + dt * 4.5)

    // Camera framed generously so raised hands, floating laptop, sticky note, and swirling vortex noodle + dust depth all show cleanly
    const camX = 0.14 + this.smoothPointer.x * 0.52 + (reduced ? 0 : Math.sin(time * 0.45) * 0.08)
    const camY = -0.14 - this.smoothPointer.y * 0.32 + (reduced ? 0 : Math.cos(time * 0.38) * 0.05)
    const proj = perspectiveMatrix((42 * Math.PI) / 180, w / h, 0.1, 40)
    const view = lookAtMatrix([camX, camY, 6.65], [0.24, -0.28, -0.35], [0, 1, 0])
    const viewProj = mulMat4(proj, view)

    // Pull vector aims down-and-right AND plunges away into -Z so perspective shrinking sells the depth & length!
    const pull = norm([0.38, -0.78, -0.49])
    const keyLight = norm([-0.68, 0.72, 0.55])

    this.uniformData.set(viewProj, 0)
    this.uniformData.set([time, 1.0, this.hoverSmooth, reduced ? 1 : 0], 16)
    this.uniformData.set([pull[0], pull[1], pull[2], this.fadeIn], 20)
    this.uniformData.set([keyLight[0], keyLight[1], keyLight[2], 0], 24)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const mesh = this.meshes.get(this.activeId)
    if (!mesh) return

    const encoder = this.device.createCommandEncoder({ label: 'Crew astronaut frame' })
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.msaaColor!.createView(),
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTex!.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })
    pass.setPipeline(this.gpu.pipeline)
    pass.setBindGroup(0, mesh.bindGroup)
    pass.setVertexBuffer(0, mesh.vertexBuffer)
    pass.setIndexBuffer(mesh.indexBuffer, 'uint32')
    pass.drawIndexed(mesh.indexCount)
    pass.end()
    this.device.queue.submit([encoder.finish()])
    this.canvas.dataset.crewReady = 'true'
    this.canvas.dispatchEvent(new Event('crewframe'))
  }

  private schedule() {
    if (this.destroyed || !this.active) return
    if (this.reducedMotion.matches) {
      this.drawOnce()
      return
    }
    this.raf = requestAnimationFrame(() => {
      this.raf = 0
      this.drawOnce()
      this.schedule()
    })
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    if (this.raf) cancelAnimationFrame(this.raf)
    this.observer.disconnect()
    this.msaaColor?.destroy()
    this.depthTex?.destroy()
    this.uniformBuffer.destroy()
    for (const mesh of this.meshes.values()) {
      mesh.vertexBuffer.destroy()
      mesh.indexBuffer.destroy()
      mesh.noteTexture.destroy()
    }
    this.meshes.clear()
    this.context.unconfigure()
    this.lease.release()
  }
}
