// Full precision terrain still works without float32-filterable. Only this
// sampling helper changes; heights and coast distances are never quantized.
fn terrainTexel(pixel: vec2i, lod: i32) -> vec2f {
  let size = vec2i(textureDimensions(terrain, lod));
  let p = vec2i(((pixel.x % size.x) + size.x) % size.x, clamp(pixel.y, 0, size.y - 1));
  return textureLoad(terrain, p, lod).rg;
}

fn terrainBilinear(uv: vec2f, lod: i32) -> vec2f {
  let p = uv * vec2f(textureDimensions(terrain, lod)) - .5;
  let base = vec2i(floor(p));
  let f = fract(p);
  return mix(
    mix(terrainTexel(base, lod), terrainTexel(base + vec2i(1, 0), lod), f.x),
    mix(terrainTexel(base + vec2i(0, 1), lod), terrainTexel(base + vec2i(1, 1), lod), f.x),
    f.y,
  );
}

fn terrainLevel(uv: vec2f, lod: f32) -> vec2f {
  let level = clamp(lod, 0., f32(textureNumLevels(terrain) - 1u));
  let low = i32(floor(level));
  let high = min(low + 1, i32(textureNumLevels(terrain)) - 1);
  return mix(terrainBilinear(uv, low), terrainBilinear(uv, high), fract(level));
}

fn terrainGradient(uv: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let size = vec2f(textureDimensions(terrain));
  let major = length(a * size);
  let minor = length(b * size);
  let taps = u32(clamp(ceil(major / max(1., minor)), 1., 16.));
  let lod = log2(max(1., max(minor, major / f32(taps))));
  var value = vec2f(0.);
  for (var i = 0u; i < taps; i++) {
    value += terrainLevel(uv + a * ((f32(i) + .5) / f32(taps) - .5), lod);
  }
  return value / f32(taps);
}

fn terrainPair(uv: vec2f, dx: vec2f, dy: vec2f) -> vec2f {
  let sampleUv = vec2f(uv.x, 1. - uv.y);
  let size = vec2f(textureDimensions(terrain));
  let a = vec2f(dx.x, -dx.y) * size;
  let b = vec2f(dy.x, -dy.y) * size;
  let xx = a.x * a.x + b.x * b.x;
  let yy = a.y * a.y + b.y * b.y;
  let xy = a.x * a.y + b.x * b.y;
  let spread = length(vec2f(xx - yy, 2. * xy));
  var angle = 0.;
  if (spread > .000001) { angle = .5 * atan2(2. * xy, xx - yy); }
  let axis = vec2f(cos(angle), sin(angle));
  let major = sqrt(max(.000001, .5 * (xx + yy + spread)));
  let minor = abs(a.x * b.y - a.y * b.x) / major;
  let invSize = 1. / size;
  let perp = vec2f(-axis.y, axis.x);
  return terrainGradient(sampleUv, axis * max(major, 1.) * invSize, perp * max(minor, 1.) * invSize);
}
