export type ReadonlyVec3 = readonly [number, number, number]
export type MutableVec3 = [number, number, number]

export function vec3Add(a: ReadonlyVec3, b: ReadonlyVec3): MutableVec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function vec3Sub(a: ReadonlyVec3, b: ReadonlyVec3): MutableVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function vec3Scale(a: ReadonlyVec3, n: number): MutableVec3 {
  return [a[0] * n, a[1] * n, a[2] * n]
}

export function vec3Dot(a: ReadonlyVec3, b: ReadonlyVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function vec3Cross(a: ReadonlyVec3, b: ReadonlyVec3): MutableVec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

export function vec3Length(v: ReadonlyVec3): number {
  return Math.hypot(v[0], v[1], v[2])
}

export function vec3Normalize(v: ReadonlyVec3, minLength = 1e-30): MutableVec3 {
  return vec3Scale(v, 1 / Math.max(minLength, vec3Length(v)))
}
