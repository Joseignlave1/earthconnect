struct UnpackParams {
  width: u32,
  height: u32,
  northElev: f32,
  northCoast: f32,
  southElev: f32,
  southCoast: f32,
  _pad0: u32,
  _pad1: u32,
}

struct MipParams {
  srcW: u32,
  srcH: u32,
  dstW: u32,
  dstH: u32,
}

struct PoleReduceParams {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var packedTex: texture_2d<u32>;
@group(0) @binding(1) var<uniform> unpackParams: UnpackParams;
@group(0) @binding(2) var srcFloatTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> mipParams: MipParams;
@group(0) @binding(4) var<uniform> poleParams: PoleReduceParams;
@group(0) @binding(5) var<storage, read_write> poleOut: array<vec2f, 2>;
@group(0) @binding(6) var<storage, read> poleRead: array<vec2f, 2>;

@fragment fn unpackRg32Float(@builtin(position) pos: vec4f) -> @location(0) vec2f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (y == 0u) {
    return vec2f(unpackParams.northElev, unpackParams.northCoast);
  }
  if (y == unpackParams.height - 1u) {
    return vec2f(unpackParams.southElev, unpackParams.southCoast);
  }
  let p = textureLoad(packedTex, vec2u(x, y), 0);
  let elev = f32(p.r * 256u + p.g) - 32768.0;
  let coast = (f32(p.b) - 128.0) * 0.25;
  return vec2f(elev, coast);
}

@fragment fn downsampleRg32Float(@builtin(position) pos: vec4f) -> @location(0) vec2f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  let sx0 = (x * mipParams.srcW) / mipParams.dstW;
  let sx1 = ((x + 1u) * mipParams.srcW) / mipParams.dstW;
  let sy0 = (y * mipParams.srcH) / mipParams.dstH;
  let sy1 = ((y + 1u) * mipParams.srcH) / mipParams.dstH;
  var sum = vec2f(0.0);
  var count = 0u;
  for (var sy = sy0; sy < sy1; sy++) {
    for (var sx = sx0; sx < sx1; sx++) {
      sum += textureLoad(srcFloatTex, vec2u(sx, sy), 0).xy;
      count += 1u;
    }
  }
  return sum / f32(count);
}

@fragment fn downsampleRg8Unorm(@builtin(position) pos: vec4f) -> @location(0) vec2f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  let sx0 = (x * mipParams.srcW) / mipParams.dstW;
  let sx1 = ((x + 1u) * mipParams.srcW) / mipParams.dstW;
  let sy0 = (y * mipParams.srcH) / mipParams.dstH;
  let sy1 = ((y + 1u) * mipParams.srcH) / mipParams.dstH;
  var sum = 0u;
  var count = 0u;
  for (var sy = sy0; sy < sy1; sy++) {
    for (var sx = sx0; sx < sx1; sx++) {
      let t = textureLoad(srcFloatTex, vec2u(sx, sy), 0).xy;
      let hi = u32(floor(t.x * 255.0 + 0.5));
      let lo = u32(floor(t.y * 255.0 + 0.5));
      sum += hi * 256u + lo;
      count += 1u;
    }
  }
  let code = clamp(floor(f32(sum) / f32(count) + 0.5), 0.0, 65535.0);
  let hi = floor(code / 256.0);
  let lo = code - hi * 256.0;
  return vec2f(hi / 255.0, lo / 255.0);
}

var<workgroup> sharedVec2: array<vec2f, 256>;

@compute @workgroup_size(256)
fn reducePolesRg32Float(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = select(poleParams.height - 1u, 0u, wg.x == 0u);
  let w = poleParams.width;
  var acc = vec2f(0.0);
  for (var x = lid.x; x < w; x += 256u) {
    acc += textureLoad(srcFloatTex, vec2u(x, row), 0).xy;
  }
  sharedVec2[lid.x] = acc;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lid.x < stride) {
      sharedVec2[lid.x] += sharedVec2[lid.x + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  if (lid.x == 0u) {
    poleOut[wg.x] = sharedVec2[0] / f32(w);
  }
}

var<workgroup> sharedU32: array<u32, 256>;

@compute @workgroup_size(256)
fn reducePolesRg8Unorm(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = select(poleParams.height - 1u, 0u, wg.x == 0u);
  let w = poleParams.width;
  var acc = 0u;
  for (var x = lid.x; x < w; x += 256u) {
    let t = textureLoad(srcFloatTex, vec2u(x, row), 0).xy;
    let hi = u32(floor(t.x * 255.0 + 0.5));
    let lo = u32(floor(t.y * 255.0 + 0.5));
    acc += hi * 256u + lo;
  }
  sharedU32[lid.x] = acc;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lid.x < stride) {
      sharedU32[lid.x] += sharedU32[lid.x + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  if (lid.x == 0u) {
    let code = clamp(floor(f32(sharedU32[0]) / f32(w) + 0.5), 0.0, 65535.0);
    let hi = floor(code / 256.0);
    let lo = code - hi * 256.0;
    poleOut[wg.x] = vec2f(hi / 255.0, lo / 255.0);
  }
}

@fragment fn closePoleRow(@builtin(position) pos: vec4f) -> @location(0) vec2f {
  let idx = select(1u, 0u, u32(pos.y) == 0u);
  return poleRead[idx];
}
