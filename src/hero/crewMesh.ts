import { vec3Add, vec3Cross, vec3Scale, vec3Sub, type MutableVec3 } from '../shader-test/vec3.ts'
import type { AstronautPoseId } from './crewAstronaut.ts'

export type Vec3 = MutableVec3

const add = vec3Add
const sub = vec3Sub
const scale = vec3Scale
const cross = vec3Cross

export function normOrUp(a: Vec3): Vec3 {
  const len = Math.hypot(a[0], a[1], a[2])
  return len > 1e-6 ? [a[0] / len, a[1] / len, a[2] / len] : [0, 1, 0]
}

const MAT = {
  suit: 0,
  trim: 1,
  gear: 2,
  visor: 3,
  stickyNote: 4,
  tape: 5,
  laptopScreen: 6,
} as const

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
  const w = normOrUp(axisRaw)
  const ref: Vec3 = Math.abs(w[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0]
  const u = normOrUp(cross(ref, w))
  const v = cross(w, u)
  return { u, v, w }
}

class MeshBuilder {
  readonly verts: number[] = [] // 10 fields/vertex: pos(3), normal(3), uv(2), matId(1), tetherMask(1)
  readonly indices: number[] = []
  private vertexCount = 0

  pushVertex(p: Vec3, n: Vec3, u: number, v: number, matId: number, tetherMask: number) {
    this.verts.push(p[0], p[1], p[2], n[0], n[1], n[2], u, v, matId, tetherMask)
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
    tetherMask: number,
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
        const localNorm = normOrUp([
          (sp * ct) / Math.max(0.01, radii[0]),
          cp / Math.max(0.01, radii[1]),
          (sp * st) / Math.max(0.01, radii[2]),
        ])
        this.pushVertex(add(center, xform(localPos)), normOrUp(xform(localNorm)), u, v, matId, tetherMask)
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
    tetherMask: number,
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
        const n = normOrUp(add(scale(radialDir, nRadial), scale(bw, ring.nAxial)))
        this.pushVertex(p, n, u, ring.vCoord, matId, tetherMask)
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
    tetherMask: number,
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
        const n = normOrUp(add(scale(radial, Math.cos(ph)), scale(bw, Math.sin(ph))))
        const p = add(ringCenter, scale(n, minorR))
        this.pushVertex(p, n, u, v, matId, tetherMask)
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
  addBox(center: Vec3, half: Vec3, rot: [number, number, number], matId: number, tetherMask: number) {
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
          this.pushVertex(add(center, xform(raw)), normOrUp(xform(f.n)), tu, 1 - tv, matId, tetherMask)
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
  addStickyNote(helmetCenter: Vec3, headRot: [number, number, number], noteTilt: number, tetherMask: number) {
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
        const cornerCurl = Math.max(0, (nx + 0.1) * (-ny + 0.1)) * 0.055
        const domeCurve = -(nx * nx + ny * ny) * 0.048
        const local = rotZ([nx * halfW, ny * halfH, baseZ + domeCurve + cornerCurl], noteTilt)
        const nLocal = normOrUp(rotZ([nx * 0.26 - cornerCurl * 1.4, ny * 0.26 + cornerCurl * 1.4, 1], noteTilt))
        this.pushVertex(
          add(helmetCenter, xformHead(local)),
          normOrUp(xformHead(nLocal)),
          u,
          v,
          MAT.stickyNote,
          tetherMask,
        )
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
        normOrUp(xformHead([0, 0.15, 1])),
        tapeUvs[i][0],
        tapeUvs[i][1],
        MAT.tape,
        tetherMask,
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

export function buildAstronautGeometry(poseId: AstronautPoseId): {
  vertexData: ArrayBuffer
  indexData: Uint16Array | Uint32Array
  indexFormat: GPUIndexFormat
  indexCount: number
} {
  const pose = POSE_SPECS[poseId]
  const m = new MeshBuilder()

  const xf = (p: Vec3): Vec3 =>
    add(pose.bodyOffset, rotZ(rotY(rotX(p, pose.bodyRot[0]), pose.bodyRot[1]), pose.bodyRot[2]))
  const worldHeadRot: [number, number, number] = [
    pose.bodyRot[0] + pose.headRot[0],
    pose.bodyRot[1] + pose.headRot[1],
    pose.bodyRot[2] + pose.headRot[2],
  ]

  const pelvis = xf([0, -0.22, 0])
  const chest = xf([0, 0.3, 0])
  const neck = xf([0, 0.44, 0.02])
  const headCenter = xf([0, 0.78, 0.04])

  m.addCapsule(pelvis, chest, 0.28, 0.36, MAT.suit, 0, 26, 22, 0.24, 1)
  m.addTorus(neck, sub(headCenter, chest), 0.29, 0.068, MAT.trim, 0, 26, 12)
  m.addBox(xf([0, 0.08, 0.33]), [0.16, 0.13, 0.075], pose.bodyRot, MAT.gear, 0)
  m.addBox(xf([0, 0.04, -0.34]), [0.27, 0.34, 0.16], pose.bodyRot, MAT.gear, 0)
  m.addCapsule(xf([-0.14, -0.22, -0.44]), xf([-0.14, 0.26, -0.44]), 0.085, 0.085, MAT.trim, 0, 14, 10)
  m.addCapsule(xf([0.14, -0.22, -0.44]), xf([0.14, 0.26, -0.44]), 0.085, 0.085, MAT.trim, 0, 14, 10)

  m.addEllipsoid(headCenter, [0.41, 0.4, 0.41], MAT.suit, 0, 26, 18, worldHeadRot)
  const visorOffset = rotZ(rotY(rotX([0, 0, 0.16], worldHeadRot[0]), worldHeadRot[1]), worldHeadRot[2])
  m.addEllipsoid(add(headCenter, visorOffset), [0.33, 0.31, 0.29], MAT.visor, 0, 24, 16, worldHeadRot)
  const rimOffset = rotZ(rotY(rotX([0, 0, 0.29], worldHeadRot[0]), worldHeadRot[1]), worldHeadRot[2])
  const rimAxis = rotZ(rotY(rotX([0, 0, 1], worldHeadRot[0]), worldHeadRot[1]), worldHeadRot[2])
  m.addTorus(add(headCenter, rimOffset), rimAxis, 0.31, 0.042, MAT.trim, 0, 26, 10)
  m.addStickyNote(headCenter, worldHeadRot, pose.noteTilt, 0)

  const addArm = (shoulder: Vec3, elbowLocal: Vec3, handLocal: Vec3) => {
    const elbow = add(pose.bodyOffset, elbowLocal)
    const hand = add(pose.bodyOffset, handLocal)
    m.addCapsule(shoulder, elbow, 0.135, 0.125, MAT.suit, 0, 16, 12)
    m.addCapsule(elbow, hand, 0.12, 0.105, MAT.suit, 0, 16, 12)
    m.addTorus(add(elbow, scale(sub(hand, elbow), 0.84)), sub(hand, elbow), 0.112, 0.028, MAT.trim, 0, 16, 8)
    m.addEllipsoid(hand, [0.13, 0.13, 0.11], MAT.gear, 0, 14, 10)
  }
  addArm(xf([-0.38, 0.26, 0.02]), pose.lElbow, pose.lHand)
  addArm(xf([0.38, 0.26, 0.02]), pose.rElbow, pose.rHand)

  const addLeg = (hip: Vec3, kneeLocal: Vec3, footLocal: Vec3, bootShiftX: number) => {
    const knee = add(pose.bodyOffset, kneeLocal)
    const foot = add(pose.bodyOffset, footLocal)
    m.addCapsule(hip, knee, 0.168, 0.145, MAT.suit, 0, 22, 26)
    m.addCapsule(knee, foot, 0.145, 0.13, MAT.suit, 0, 22, 32)
    m.addEllipsoid(add(foot, [bootShiftX, -0.06, 0.08]), [0.16, 0.13, 0.22], MAT.gear, 0, 14, 10)
  }
  addLeg(xf([-0.145, -0.2, 0]), pose.lKnee, pose.lFoot, 0.02)
  addLeg(xf([0.145, -0.2, 0]), pose.rKnee, pose.rFoot, 0.04)

  const tetherA = xf([0.28, -0.22, -0.16])
  const tetherB = add(tetherA, [0.36, -0.68, -0.08])
  const tetherC = add(tetherB, [0.42, -0.92, -0.06])
  m.addCapsule(tetherA, tetherB, 0.026, 0.024, MAT.trim, 0.75, 10, 20)
  m.addCapsule(tetherB, tetherC, 0.024, 0.02, MAT.trim, 1.0, 10, 24)

  if (pose.hasLaptop) {
    const lapBase = add(pose.bodyOffset, [0.96, 1.56, 0.36])
    m.addBox(lapBase, [0.22, 0.02, 0.15], [0.28, -0.42, -0.32], MAT.gear, 0)
    m.addBox(add(lapBase, [-0.04, 0.15, -0.11]), [0.22, 0.15, 0.016], [-0.24, -0.42, -0.32], MAT.laptopScreen, 0)
  }

  const vertexCount = m.verts.length / 10
  const vertexData = new ArrayBuffer(vertexCount * 40)
  const f32 = new Float32Array(vertexData)
  const u32 = new Uint32Array(vertexData)
  for (let i = 0; i < vertexCount; i++) {
    const o = i * 10
    for (let k = 0; k < 8; k++) f32[o + k] = m.verts[o + k]
    u32[o + 8] = m.verts[o + 8] >>> 0
    f32[o + 9] = m.verts[o + 9]
  }
  const useUint16 = vertexCount <= 65535
  const indexData = useUint16 ? new Uint16Array(m.indices) : new Uint32Array(m.indices)
  return {
    vertexData,
    indexData,
    indexFormat: useUint16 ? 'uint16' : 'uint32',
    indexCount: m.indices.length,
  }
}
