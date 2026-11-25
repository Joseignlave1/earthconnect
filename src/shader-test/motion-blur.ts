import type { SolarFrame } from '../workspace/solarCamera.ts'
import { FOV } from './geometry.ts'
import { joinWgsl } from './gpu.ts'
import colorWgsl from './shaders/color.wgsl?raw'
import equalEarthWgsl from './shaders/equal-earth.wgsl?raw'
import fullscreenWgsl from './shaders/fullscreen.wgsl?raw'
import motionBlurWgsl from './shaders/motion-blur.wgsl?raw'
import postUniformsWgsl from './shaders/post-uniforms.wgsl?raw'
import reticleWgsl from './shaders/reticle.wgsl?raw'

const tangent = Math.tan(FOV / 2)
const shutterMilliseconds = 14
const maximumHistoryAge = 96

export type MotionBlurUniforms = Float32Array<ArrayBuffer>

export function cameraChanged(previous: SolarFrame, current: SolarFrame) {
  if (current.aspect !== previous.aspect) return true
  const delta = current.eye.map((value, index) => value - previous.eye[index])
  const before = [previous.right, previous.up, previous.back]
  const after = [current.right, current.up, current.back]
  const moved = delta.some(value => Math.abs(value) > Math.max(1e-7, current.near * 1e-8))
  const turned = after.some((axis, column) => axis.some((value, row) => Math.abs(value - before[column][row]) > 1e-9))
  return moved || turned
}

export function motionBlurUniforms(
  previous: SolarFrame,
  current: SolarFrame,
  elapsed: number,
  pixelRatio: number,
  inputDriven = false,
): MotionBlurUniforms | undefined {
  if (elapsed < 1 || current.aspect !== previous.aspect) return
  if (elapsed > maximumHistoryAge) {
    // With input-driven rendering, the previous pose is still on screen after
    // an idle interval. The interval before the input is not time spent moving.
    if (!inputDriven) return
    elapsed = 1000 / 60
  }
  if (!cameraChanged(previous, current)) return
  const delta = current.eye.map((value, index) => value - previous.eye[index])
  const before = [previous.right, previous.up, previous.back]
  const after = [current.right, current.up, current.back]

  // Subtract heliocentric eye positions in double precision before upload.
  // Transform current view-space coordinates directly into the previous view.
  const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const packed = new Float32Array(24)
  for (const [column, axis] of after.entries()) {
    packed.set(
      before.map(basis => dot(basis, axis)),
      column * 4,
    )
  }
  packed.set(
    before.map(basis => dot(basis, delta)),
    12,
  )
  packed[15] = current.near
  packed.set([tangent * current.aspect, tangent, tangent * previous.aspect, tangent], 16)
  packed.set([Math.min(3.5, shutterMilliseconds / elapsed), 20 * pixelRatio, 0.7 * pixelRatio, 0], 20)
  return packed
}

export const motionBlurShader = joinWgsl(
  fullscreenWgsl,
  colorWgsl,
  equalEarthWgsl,
  postUniformsWgsl,
  reticleWgsl,
  motionBlurWgsl,
)
