import type { SolarFrame } from '../workspace/solarCamera.ts'
import { colorShader } from './color.ts'
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
struct WorldMarkers {
  step: vec4f,
  effects: vec4f,
  mask: vec4f,
  hover: vec4f,
  morph: vec4f,
  selEarth: vec4f,
  selMoon: vec4f,
  selMars: vec4f,
}
@group(0) @binding(0) var<uniform> motion: Motion;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_multisampled_2d;
@group(0) @binding(3) var<uniform> markers: WorldMarkers;
@group(0) @binding(4) var geography: texture_2d<f32>;
${colorShader}

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

fn worldTangent(uv: vec2f, point: vec2f, bodyMorph: f32, unitPerPixel: f32) -> vec2f {
  let angles = (uv - .5) * vec2f(6.283185307, 3.141592654);
  let origin = (point - .5) * vec2f(6.283185307, 3.141592654);
  let direction = vec3f(cos(angles.y) * sin(angles.x), sin(angles.y), cos(angles.y) * cos(angles.x));
  let anchor = vec3f(cos(origin.y) * sin(origin.x), sin(origin.y), cos(origin.y) * cos(origin.x));
  if (bodyMorph > .5 && dot(direction, anchor) < 0.) { return vec2f(1e5, 1e5); }
  let delta = uv.x - point.x;
  let planeDelta = vec2f((delta - floor(delta + .5)) * 6.283185307, direction.y - anchor.y);
  let east = vec3f(cos(origin.x), 0., -sin(origin.x));
  let north = cross(anchor, east);
  let offset = direction - anchor;
  return mix(planeDelta, vec2f(dot(offset, east), dot(offset, north)), bodyMorph) / max(unitPerPixel, 1e-9);
}

// Composite placed world markers into linear HDR taps so they blur along the
// planet's exact 3D velocity with full depth-aware silhouette protection.
fn sampleWorld(pixel: vec2i) -> vec4f {
  let s = textureLoad(scene, pixel, 0);
  let coordinates = textureLoad(geography, pixel, 0).rg;
  if (coordinates.y == -1.) { return s; }
  let bodyId = i32(floor(coordinates.x * .5));
  let sel = select(select(markers.selEarth, markers.selMoon, bodyId == 1), markers.selMars, bodyId == 2);
  if (sel.z <= .005) { return s; }
  let uv = vec2f(coordinates.x - f32(bodyId) * 2., coordinates.y);
  let bodyMorph = select(1., markers.morph.x, bodyId == 0);
  let t = worldTangent(uv, sel.xy, bodyMorph, sel.w);
  let d = length(t);
  if (d > 16.) { return s; }
  let aa = clamp(.65 / max(markers.effects.z, 1.), .15, .75);
  let base = s.rgb / max(s.a, .0001);
  let exposed = base * markers.effects.x;
  let mapped = clamp((exposed * (2.51 * exposed + .03)) / (exposed * (2.43 * exposed + .59) + .14), vec3f(0.), vec3f(1.));
  let bg = dot(select(base, toDisplay(mapped), markers.step.w > .5), vec3f(.2126, .7152, .0722));
  let radius = 8.4;
  let ringDist = abs(d - radius);
  let ax = abs(t.x);
  let ay = abs(t.y);
  let isNorth = t.y > 0. && ax < ay;
  let tickEnd = select(11.3, 12.6, isNorth);
  let tickBand = smoothstep(radius - .4 - aa, radius - .4 + aa, d) * (1. - smoothstep(tickEnd - aa, tickEnd + aa, d));
  let tickCore = max(1. - smoothstep(.58 - aa, .58 + aa, ax), 1. - smoothstep(.58 - aa, .58 + aa, ay)) * tickBand;
  let tickCase = max(1. - smoothstep(1.45 - aa, 1.45 + aa, ax), 1. - smoothstep(1.45 - aa, 1.45 + aa, ay)) * tickBand;
  let ringCore = 1. - smoothstep(.66 - aa, .66 + aa, ringDist);
  let dotCore = 1. - smoothstep(1.35 - aa, 1.35 + aa, d);
  let core = max(max(ringCore, dotCore), tickCore);
  let casing = max(max(1. - smoothstep(1.65 - aa, 1.65 + aa, ringDist), 1. - smoothstep(2.3 - aa, 2.3 + aa, d)), tickCase);
  let inside = 1. - smoothstep(radius - 1. - aa, radius - 1. + aa, d);
  let hdr = markers.step.w > .5;
  var col = mix(base, mix(base * 1.10 + select(.012, .003, hdr), base * .90, smoothstep(.25, .7, bg)), inside * .55);
  let caseAlpha = mix(.62, .92, smoothstep(.1, .65, bg));
  let darkCol = select(vec3f(.02, .028, .045), vec3f(.0016, .0022, .0036), hdr);
  let whiteCol = select(vec3f(.96, .98, 1.), vec3f(.96, .98, 1.), hdr);
  let northCol = select(vec3f(.55, .86, 1.), vec3f(.26, .72, 1.), hdr);
  col = mix(col, darkCol, casing * caseAlpha);
  let accent = mix(whiteCol, northCol, select(0., 1., isNorth) * tickCore * .85);
  col = mix(col, accent, core * .96);
  return vec4f(mix(base, col, sel.z) * s.a, s.a);
}

@fragment fn blurFragment(input: ScreenVertex) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(scene));
  let pixel = vec2i(input.position.xy);
  let color = sampleWorld(pixel);
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
    sum += sampleWorld(q) * weight;
    weights += weight;
  }
  return sum / weights;
}
`
