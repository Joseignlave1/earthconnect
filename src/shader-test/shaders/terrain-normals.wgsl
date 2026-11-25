// Bake a mipmapped rgba8unorm tangent-space normal + multi-scale ridge-AO map once
// from Earth's rg32float terrain pyramid so surfaceFragment does 1 hardware-filtered
// 32-bit fetch instead of 6-11 64-bit rg32float fetches per pixel.
// Channels: rgb = tangent-space normal (east, north, up) calibrated for relief = 20,
//           a   = multi-scale ridge/valley contrast in [0..1] (0.5 = flat).
@group(0) @binding(0) var levelTex: texture_2d<f32>;
@group(0) @binding(1) var fullTerrain: texture_2d<f32>;
const PI: f32 = 3.141592653589793;

fn levelHeight(p: vec2i, size: vec2i) -> f32 {
  let c = vec2i(((p.x % size.x) + size.x) % size.x, clamp(p.y, 0, size.y - 1));
  return textureLoad(levelTex, c, 0).r;
}

fn levelHeightAtDir(dir: vec3f, size: vec2i) -> f32 {
  let lon = atan2(dir.x, dir.z);
  let lat = atan2(dir.y, length(dir.xz));
  let uv = vec2f(lon / (2. * PI) + .5, .5 - lat / PI);
  let pos = uv * vec2f(size) - .5;
  let base = vec2i(floor(pos));
  let f = fract(pos);
  return mix(
    mix(levelHeight(base, size), levelHeight(base + vec2i(1, 0), size), f.x),
    mix(levelHeight(base + vec2i(0, 1), size), levelHeight(base + vec2i(1, 1), size), f.x),
    f.y,
  );
}

fn terrainHeightBilinear(uv: vec2f, lod: i32) -> f32 {
  let size = vec2i(textureDimensions(fullTerrain, lod));
  let pos = uv * vec2f(size) - .5;
  let base = vec2i(floor(pos));
  let f = fract(pos);
  let c00 = textureLoad(fullTerrain, vec2i(((base.x % size.x) + size.x) % size.x, clamp(base.y, 0, size.y - 1)), lod).r;
  let c10 = textureLoad(fullTerrain, vec2i((((base.x + 1) % size.x) + size.x) % size.x, clamp(base.y, 0, size.y - 1)), lod).r;
  let c01 = textureLoad(fullTerrain, vec2i(((base.x % size.x) + size.x) % size.x, clamp(base.y + 1, 0, size.y - 1)), lod).r;
  let c11 = textureLoad(fullTerrain, vec2i((((base.x + 1) % size.x) + size.x) % size.x, clamp(base.y + 1, 0, size.y - 1)), lod).r;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

@fragment fn bakeNormals(input: ScreenVertex) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(levelTex));
  let p = vec2i(input.position.xy);
  if (size.y <= 2 || p.y == 0 || p.y == size.y - 1) {
    return vec4f(.5, .5, 1., .5);
  }
  let fullH = f32(textureDimensions(fullTerrain, 0).y);
  let lodF = max(0., log2(fullH / f32(size.y)));
  let maxLod = i32(textureNumLevels(fullTerrain)) - 1;
  let uv = (vec2f(p) + .5) / vec2f(size);
  let lon = (uv.x - .5) * 2. * PI;
  let lat = (.5 - uv.y) * PI;
  let cosLon = cos(lon);
  let sinLon = sin(lon);
  let cosLat = cos(lat);
  let sinLat = sin(lat);
  let up = vec3f(cosLat * sinLon, sinLat, cosLat * cosLon);
  let east = vec3f(cosLon, 0., -sinLon);
  let north = vec3f(-sinLat * sinLon, cosLat, -sinLat * cosLon);
  let s = min(.25, PI / f32(size.y));
  let hE = levelHeightAtDir(up + east * s, size);
  let hW = levelHeightAtDir(up - east * s, size);
  let hN = levelHeightAtDir(up + north * s, size);
  let hS = levelHeightAtDir(up - north * s, size);
  let center = textureLoad(levelTex, p, 0).rg;
  let landFrac = smoothstep(-.5, .5, center.g);
  let slopeScale = mix(.62, 1., landFrac) * (1. + .12 * min(lodF, 3.)) * (20. / (2. * s * 6378137.));
  let gx = (hE - hW) * slopeScale;
  let gy = (hN - hS) * slopeScale;
  let n = normalize(vec3f(-gx, -gy, 1.));
  let coarseLodF = min(f32(maxLod), lodF + 2.5);
  let lo = i32(floor(coarseLodF));
  let hi = min(lo + 1, maxLod);
  let coarseHeight = mix(terrainHeightBilinear(uv, lo), terrainHeightBilinear(uv, hi), fract(coarseLodF));
  let ridge = clamp((center.r - coarseHeight) / 1400., -1., 1.);
  return vec4f(n.x * .5 + .5, n.y * .5 + .5, n.z, ridge * .5 + .5);
}
