import type { SolarFrame } from '../workspace/solarCamera.ts'
import { FOV } from './geometry.ts'
import { fullscreenVertex } from './gpu.ts'

const tangent = Math.tan(FOV / 2)
const shutterMilliseconds = 14
const maximumHistoryAge = 96

export type MotionBlurUniforms = Float32Array<ArrayBuffer>

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
  const delta = current.eye.map((value, index) => value - previous.eye[index])
  const before = [previous.right, previous.up, previous.back]
  const after = [current.right, current.up, current.back]
  const moved = delta.some(value => Math.abs(value) > Math.max(1e-7, current.near * 1e-8))
  const turned = after.some((axis, column) => axis.some((value, row) => Math.abs(value - before[column][row]) > 1e-9))
  if (!moved && !turned) return

  // Subtract heliocentric eye positions in double precision before upload.
  // Transform current view-space coordinates directly into the previous view.
  const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const packed = new Float32Array(24)
  for (const [column, axis] of after.entries()) {
    packed.set(before.map(basis => dot(basis, axis)), column * 4)
  }
  packed.set(before.map(basis => dot(basis, delta)), 12)
  packed[15] = current.near
  packed.set([tangent * current.aspect, tangent, tangent * previous.aspect, tangent], 16)
  packed.set([Math.min(3.5, shutterMilliseconds / elapsed), 20 * pixelRatio, .7 * pixelRatio, 0], 20)
  return packed
}

export const motionBlurShader = `${fullscreenVertex}
struct Motion {
  currentToPrevious: mat3x3f,
  eyeDeltaAndNear: vec4f,
  projection: vec4f,
  shutter: vec4f, // fraction of frame interval, maximum trail span, subpixel cutoff, reserved
}
@group(0) @binding(0) var<uniform> motion: Motion;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_multisampled_2d;

struct Depth {
  closest: f32,
  farthest: f32,
  coverage: u32,
}

fn readDepth(pixel: vec2i) -> Depth {
  var result = Depth(0., 1., 0u);
  for (var sample = 0; sample < 4; sample++) {
    let depth = textureLoad(sceneDepth, pixel, sample);
    result.closest = max(result.closest, depth);
    result.farthest = min(result.farthest, depth);
    result.coverage += select(0u, 1u, depth > 0.);
  }
  return result;
}

fn mixedDepth(depth: Depth) -> bool {
  return depth.coverage != 0u && depth.closest - depth.farthest > .035 * depth.closest;
}

@fragment fn blurFragment(input: ScreenVertex) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(scene));
  let pixel = vec2i(input.position.xy);
  let color = textureLoad(scene, pixel, 0);
  let center = readDepth(pixel);
  // A resolved silhouette may blend sky with terrain or two different bodies.
  // Keep its 4x coverage intact and never use mixed colors to shade either layer.
  if (mixedDepth(center)) { return color; }

  let ndc = input.uv * vec2f(2., -2.) + vec2f(-1., 1.);
  let ray = vec3f(ndc * motion.projection.xy, -1.);
  // Reverse-Z stores near / viewDistance. Divide the transformed position by
  // that distance before projecting. Clear sky is at infinity and only rotates.
  let previous = motion.currentToPrevious * ray +
    motion.eyeDeltaAndNear.xyz * (center.closest / motion.eyeDeltaAndNear.w);
  if (previous.z >= -0.0001) { return color; }
  let previousNdc = previous.xy / (-previous.z * motion.projection.zw);
  var velocity = (previousNdc - ndc) * vec2f(.5, -.5) * size * motion.shutter.x;
  let speed = length(velocity);
  if (speed < motion.shutter.z) { return color; }
  velocity *= min(1., motion.shutter.y / speed);

  var sum = color;
  var weights = 1.;
  for (var tap = -4; tap <= 4; tap++) {
    if (tap == 0) { continue; }
    let offset = f32(tap) / 8.;
    let q = vec2i(floor(input.position.xy + velocity * offset));
    if (any(q < vec2i(0)) || any(q >= vec2i(size))) { continue; }
    let depth = readDepth(q);
    if (depth.coverage != center.coverage || mixedDepth(depth)) { continue; }
    var weight = 1. - abs(offset);
    if (center.coverage == 4u) {
      let separation = abs(depth.closest - center.closest) / max(depth.closest, center.closest);
      weight *= 1. - smoothstep(.025, .12, separation);
    }
    sum += textureLoad(scene, q, 0) * weight;
    weights += weight;
  }
  return sum / weights;
}
`
