struct Motion {
  currentToPrevious: mat3x3f,
  eyeDeltaAndNear: vec4f,
  projection: vec4f,
  shutter: vec4f, // fraction of frame interval, maximum trail span, subpixel cutoff, reserved
}

@group(0) @binding(0) var<uniform> motion: Motion;
@group(0) @binding(1) var sceneMsaa: texture_multisampled_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_multisampled_2d;
@group(0) @binding(3) var<uniform> p: PostUniforms;
@group(0) @binding(4) var geography: texture_multisampled_2d<f32>;
@group(0) @binding(5) var orbitAccum: texture_multisampled_2d<f32>;
@group(0) @binding(6) var sceneResolved: texture_2d<f32>;

// Composite placed world markers into linear HDR taps so they blur along the
// planet's exact 3D velocity with full depth-aware silhouette protection.
fn applyWorldMarker(s: vec4f, coordinates: vec2f, anyMarker: bool) -> vec4f {
  if (!anyMarker) { return s; }
  return evalWorldMarkerSample(
    s, coordinates, p.selEarth, p.selMoon, p.selMars,
    p.state.x, p.effects.x, p.effects.z, p.hoverScreen.w > .5,
  );
}

fn sampleWorldSample(pixel: vec2i, k: i32, dk: f32, anyMarker: bool) -> vec4f {
  let s = textureLoad(sceneMsaa, pixel, k);
  if (!anyMarker || dk <= 0.) { return s; }
  let g = textureLoad(geography, pixel, k);
  let coords = vec2f(g.x + g.y, g.z + g.w);
  return applyWorldMarker(s, coords, true);
}

fn nearSelectedMarker(pixel: vec2i) -> bool {
  let g = textureLoad(geography, pixel, 0);
  let coordinates = vec2f(g.x + g.y, g.z + g.w);
  if (coordinates.y == -1.) { return true; }
  let bodyId = i32(floor(coordinates.x * .5));
  let sel = select(select(p.selEarth, p.selMoon, bodyId == 1), p.selMars, bodyId == 2);
  if (sel.z <= .005) { return false; }
  let uv = vec2f(coordinates.x - f32(bodyId) * 2., coordinates.y);
  let bodyMorph = select(1., p.state.x, bodyId == 0);
  // Marker radius (16 px) + max blur trail span (20 px) + safety margin.
  return length(worldTangent(uv, sel.xy, bodyMorph, sel.w)) <= 42.;
}

fn blurUniformResolved(
  pixel: vec2i, dk: f32, uv: vec2f, pos: vec2f, size: vec2f,
) -> vec4f {
  let color = textureLoad(sceneResolved, pixel, 0);
  let ndc = uv * vec2f(2., -2.) + vec2f(-1., 1.);
  let ray = vec3f(ndc * motion.projection.xy, -1.);
  let previous = motion.currentToPrevious * ray +
    motion.eyeDeltaAndNear.xyz * (dk / motion.eyeDeltaAndNear.w);
  if (previous.z >= -0.0001) { return color; }
  let previousNdc = previous.xy / (-previous.z * motion.projection.zw);
  var velocity = (previousNdc - ndc) * vec2f(.5, -.5) * size * motion.shutter.x;
  let speed = length(velocity);
  if (speed < motion.shutter.z) { return color; }
  velocity *= min(1., motion.shutter.y / speed);

  let isSolid = dk > 0.;
  var sum = color;
  var weights = 1.;
  for (var tap = -4; tap <= 4; tap++) {
    if (tap == 0) { continue; }
    let offset = f32(tap) / 8.;
    let q = vec2i(floor(pos + velocity * offset));
    if (any(q < vec2i(0)) || any(q >= vec2i(size))) { continue; }
    let dq0 = textureLoad(sceneDepth, q, 0);
    if ((dq0 > 0.) != isSolid) { continue; }
    let dq1 = textureLoad(sceneDepth, q, 1);
    let dq2 = textureLoad(sceneDepth, q, 2);
    let dq3 = textureLoad(sceneDepth, q, 3);
    if ((dq1 > 0.) != isSolid || (dq2 > 0.) != isSolid || (dq3 > 0.) != isSolid) { continue; }
    var weight = 1. - abs(offset);
    if (isSolid) {
      let dqMax = max(max(dq0, dq1), max(dq2, dq3));
      let dqMin = min(min(dq0, dq1), min(dq2, dq3));
      if (dqMax - dqMin > .035 * dqMax) { continue; }
      let separation = abs(dqMax - dk) / max(dqMax, dk);
      weight *= 1. - smoothstep(.025, .12, separation);
    }
    if (weight <= 1e-4) { continue; }
    sum += textureLoad(sceneResolved, q, 0) * weight;
    weights += weight;
  }
  return sum / weights;
}

fn blurSample(
  pixel: vec2i, k: i32, dk: f32, uv: vec2f, pos: vec2f, size: vec2f, anyMarker: bool,
) -> vec4f {
  let color = sampleWorldSample(pixel, k, dk, anyMarker);
  let ndc = uv * vec2f(2., -2.) + vec2f(-1., 1.);
  let ray = vec3f(ndc * motion.projection.xy, -1.);
  // Reverse-Z stores near / viewDistance. Divide the transformed position by
  // that distance before projecting. Clear sky is at infinity and only rotates.
  let previous = motion.currentToPrevious * ray +
    motion.eyeDeltaAndNear.xyz * (dk / motion.eyeDeltaAndNear.w);
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
    let q = vec2i(floor(pos + velocity * offset));
    if (any(q < vec2i(0)) || any(q >= vec2i(size))) { continue; }
    let dq = textureLoad(sceneDepth, q, k);
    if ((dk > 0.) != (dq > 0.)) { continue; }
    var weight = 1. - abs(offset);
    if (dk > 0.) {
      let separation = abs(dq - dk) / max(dq, dk);
      weight *= 1. - smoothstep(.025, .12, separation);
    }
    if (weight <= 1e-4) { continue; }
    sum += sampleWorldSample(q, k, dq, anyMarker) * weight;
    weights += weight;
  }
  return sum / weights;
}

@fragment fn blurFragment(input: ScreenVertex) -> @location(0) vec4f {
  // Unified per-sample 4x MSAA motion blur + world markers + crisp depth-tested orbits.
  // Evaluating one consistent per-sample kernel regardless of orbit coverage guarantees zero
  // discontinuity when an orbit edge enters a pixel or transitions between partial and full
  // sample visibility, and exact per-sample marker ownership via sample-aligned 4x MSAA geography.
  let size = vec2f(textureDimensions(sceneMsaa));
  let pixel = vec2i(input.position.xy);
  let anyMarker = p.selEarth.z > .005 || p.selMoon.z > .005 || p.selMars.z > .005;
  let hasOrbits = p.state.z > .5;
  let d = array<f32, 4>(
    textureLoad(sceneDepth, pixel, 0),
    textureLoad(sceneDepth, pixel, 1),
    textureLoad(sceneDepth, pixel, 2),
    textureLoad(sceneDepth, pixel, 3),
  );
  let dMax = max(max(d[0], d[1]), max(d[2], d[3]));
  let dMin = min(min(d[0], d[1]), min(d[2], d[3]));
  let covCount = select(0u, 1u, d[0] > 0.) + select(0u, 1u, d[1] > 0.)
    + select(0u, 1u, d[2] > 0.) + select(0u, 1u, d[3] > 0.);
  let isMixed = covCount != 0u && dMax - dMin > .035 * dMax;

  // Fast 1x resolved path for uniform open-sky and uniform-depth interior pixels that have no
  // nearby selected marker and uniform (or absent) orbit coverage across all 4 MSAA samples.
  let uniformDepth = covCount == 0u || (covCount == 4u && dMax - dMin <= .005 * dMax);
  if (!isMixed && uniformDepth) {
    let needsMarker = covCount == 4u && anyMarker && nearSelectedMarker(pixel);
    if (!needsMarker) {
      var o0 = vec4f(0.);
      var uniformOrbit = true;
      if (hasOrbits) {
        o0 = textureLoad(orbitAccum, pixel, 0);
        let o1 = textureLoad(orbitAccum, pixel, 1);
        let o2 = textureLoad(orbitAccum, pixel, 2);
        let o3 = textureLoad(orbitAccum, pixel, 3);
        let oMin = min(min(o0, o1), min(o2, o3));
        let oMax = max(max(o0, o1), max(o2, o3));
        uniformOrbit = all(oMax - oMin <= vec4f(1e-4));
      }
      if (uniformOrbit) {
        let bk = blurUniformResolved(pixel, dMax, input.uv, input.position.xy, size);
        return vec4f(o0.rgb + bk.rgb * (1. - o0.a), o0.a + bk.a * (1. - o0.a));
      }
    }
  }

  var outSum = vec4f(0.);
  for (var k = 0; k < 4; k++) {
    var bk: vec4f;
    if (isMixed) {
      bk = sampleWorldSample(pixel, k, d[k], anyMarker);
    } else {
      bk = blurSample(pixel, k, d[k], input.uv, input.position.xy, size, anyMarker);
    }
    let ok = select(vec4f(0.), textureLoad(orbitAccum, pixel, k), hasOrbits);
    outSum += vec4f(ok.rgb + bk.rgb * (1. - ok.a), ok.a + bk.a * (1. - ok.a));
  }
  return outSum * .25;
}

@fragment fn staticResolveFragment(input: ScreenVertex) -> @location(0) vec4f {
  // Exact zero-velocity counterpart to blurFragment: evaluates per-sample selected world markers
  // on each MSAA sample's own color, composites foreground orbits over markers, and resolves.
  let pixel = vec2i(input.position.xy);
  let bakeMarkers = p.state.w > .5;
  let anyMarker = bakeMarkers && (p.selEarth.z > .005 || p.selMoon.z > .005 || p.selMars.z > .005);
  let hasOrbits = p.state.z > .5;
  var outSum = vec4f(0.);
  for (var k = 0; k < 4; k++) {
    let dk = textureLoad(sceneDepth, pixel, k);
    let bk = sampleWorldSample(pixel, k, dk, anyMarker);
    let ok = select(vec4f(0.), textureLoad(orbitAccum, pixel, k), hasOrbits);
    outSum += vec4f(ok.rgb + bk.rgb * (1. - ok.a), ok.a + bk.a * (1. - ok.a));
  }
  return outSum * .25;
}
