struct AnnotationView {
  camera: mat4x4f,
  info: vec4f, // x: body u-offset (2 for moon, 4 for mars), y: reliefFade, z: reference radius (km), w: relief
}

@group(0) @binding(0) var<uniform> view: AnnotationView;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var gridSampler: sampler;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(perspective, sample) local: vec3f,
}

fn cubeFaceDir(face: u32, fx: f32, fy: f32) -> vec3f {
  let qpi = 0.7853981633974483;
  let tx = tan(fx * qpi);
  let ty = tan(fy * qpi);
  switch face {
    case 0u: { return normalize(vec3f(1., ty, -tx)); }
    case 1u: { return normalize(vec3f(-1., ty, tx)); }
    case 2u: { return normalize(vec3f(tx, 1., -ty)); }
    case 3u: { return normalize(vec3f(tx, -1., ty)); }
    case 4u: { return normalize(vec3f(tx, ty, 1.)); }
    default: { return normalize(vec3f(-tx, ty, -1.)); }
  }
}

fn dirToUv(dir: vec3f) -> vec2f {
  let pi = 3.141592653589793;
  let lon = atan2(dir.x, dir.z);
  let lat = atan2(dir.y, length(dir.xz));
  return vec2f(lon / (2. * pi) + .5, .5 - lat / pi);
}

fn decodeHeight(sample: vec4f) -> f32 {
  return dot(sample.rg, vec2f(65280., 255.)) - 32768.;
}

@vertex fn vertexMain(
  @location(0) local: vec3f,
  @location(1) patchRect: vec4f,
  @location(2) patchMeta: vec4f,
) -> Vertex {
  let fx = patchRect.x + local.x * patchRect.z;
  let fy = patchRect.y + local.y * patchRect.w;
  let dir = cubeFaceDir(u32(round(patchMeta.x)), fx, fy);
  let uv = dirToUv(dir);
  let maxLod = max(0., f32(textureNumLevels(heights)) - 1.);
  let demMip = clamp(patchMeta.y, 0., maxLod);
  let elevM = decodeHeight(textureSampleLevel(heights, gridSampler, uv, demMip));
  let skirt = local.z * patchMeta.z;
  let dispR = view.info.z + (elevM * view.info.w * .001) * view.info.y - skirt;
  let pos = dir * dispR;
  return Vertex(view.camera * vec4f(pos, 1.), pos);
}

fn packSplit16(x: f32) -> vec2f {
  let hi = round(x * 256.) / 256.;
  return vec2f(hi, x - hi);
}

fn evalPlanetAnnotation(input: Vertex) -> vec4f {
  let p = normalize(input.local);
  let lon = atan2(p.x, p.z);
  let lat = atan2(p.y, length(p.xz));
  let pi = 3.141592653589793;
  let u = lon / (2. * pi) + .5 + view.info.x;
  let v = lat / pi + .5;
  return vec4f(packSplit16(u), packSplit16(v));
}

@fragment fn fragmentMain(input: Vertex) -> @location(0) vec4f {
  return evalPlanetAnnotation(input);
}

struct CoverageAnnotationOut {
  @location(0) geo: vec4f,
  @location(1) coverage: vec2f,
}

@fragment fn fragmentCoverageMain(input: Vertex) -> CoverageAnnotationOut {
  return CoverageAnnotationOut(evalPlanetAnnotation(input), vec2f(0., 0.));
}
