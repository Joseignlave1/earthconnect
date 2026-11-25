@group(0) @binding(0) var sceneDepth: texture_depth_multisampled_2d;

struct OrbitVertex {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) segA: vec4f, // x,y: A in framebuffer px (y-down), z: zA (near/aw), w: halfWidthPx
  @location(1) @interpolate(flat) segB: vec4f, // x,y: B in framebuffer px (y-down), z: zB (near/bw), w: reserved
}

@vertex fn orbitVertex(
  @location(0) position: vec4f,
  @location(1) segA: vec4f,
  @location(2) segB: vec4f,
) -> OrbitVertex {
  return OrbitVertex(position, segA, segB);
}

struct OrbitCovOut {
  @location(0) cov: vec4f,
  @builtin(sample_mask) mask: u32,
}

@fragment fn orbitCoverageFragment(
  input: OrbitVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> OrbitCovOut {
  // Analytically clip each segment to its exact reverse-Z depth-visible sub-interval [t0, t1]
  // against the per-sample scene depth, then evaluate capsule distance in pixel coordinates.
  // Because 1/w (reverse-Z) is strictly affine in screen parameter t, this is exact under
  // collinear subdivision and continuous at every distance scale.
  let basePx = floor(input.position.xy);
  let destCenter = basePx + vec2f(.5);
  let occ = textureLoad(sceneDepth, vec2i(basePx), i32(sampleIdx));
  let a = input.segA.xy;
  let b = input.segB.xy;
  let zA = input.segA.z;
  let zB = input.segB.z;
  let visA = zA > occ;
  let visB = zB > occ;
  if (!visA && !visB) { discard; }
  var t0 = 0.;
  var t1 = 1.;
  if (visA != visB) {
    let dz = zB - zA;
    let tCut = select(select(1., 0., visA), clamp((occ - zA) / dz, 0., 1.), abs(dz) > 0.);
    if (visA) { t1 = tCut; } else { t0 = tCut; }
  }
  let ab = b - a;
  let abLenSq = dot(ab, ab);
  var dist: f32;
  if (abLenSq <= 1e-12) {
    dist = length(destCenter - (a + b) * .5);
  } else {
    let t = clamp(dot(destCenter - a, ab) / abLenSq, t0, t1);
    dist = length(destCenter - (a + ab * t));
  }
  let cov = clamp(input.segA.w + .5 - dist, 0., 1.);
  if (cov <= .002) { discard; }
  return OrbitCovOut(vec4f(cov), 1u << sampleIdx);
}
