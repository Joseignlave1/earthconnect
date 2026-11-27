const PLANET_PI: f32 = 3.141592653589793;

struct CubePatchIn {
  @location(0) local: vec3f,
  @location(1) patchOrigin: vec2f,
  @location(2) patchSize: f32,
  @location(3) face: u32,
  @location(4) demMip: f32,
  @location(5) skirtDepthKm: f32,
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
  let lon = atan2(dir.x, dir.z);
  let lat = atan2(dir.y, length(dir.xz));
  return vec2f(lon / (2. * PLANET_PI) + .5, .5 - lat / PLANET_PI);
}

fn decodeHeight(sample: vec4f) -> f32 {
  return dot(sample.rg, vec2f(65280., 255.)) - 32768.;
}

fn packSplit16(x: f32) -> vec2f {
  let hi = round(x * 256.) / 256.;
  return vec2f(hi, x - hi);
}

fn displacedCubePatchPos(
  input: CubePatchIn,
  heightsTex: texture_2d<f32>,
  gridSamp: sampler,
  radiusKm: f32,
  relief: f32,
  reliefFade: f32,
) -> vec3f {
  let fx = input.patchOrigin.x + input.local.x * input.patchSize;
  let fy = input.patchOrigin.y + input.local.y * input.patchSize;
  let dir = cubeFaceDir(input.face, fx, fy);
  let uv = dirToUv(dir);
  let maxLod = max(0., f32(textureNumLevels(heightsTex)) - 1.);
  let demMip = clamp(input.demMip, 0., maxLod);
  let elevM = decodeHeight(textureSampleLevel(heightsTex, gridSamp, uv, demMip));
  let skirt = input.local.z * input.skirtDepthKm;
  let dispR = radiusKm + (elevM * relief * .001) * reliefFade - skirt;
  return dir * dispR;
}
