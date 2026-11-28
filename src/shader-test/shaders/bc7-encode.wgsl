// WebGPU compute BC7 Mode 6 encoder (7-bit RGBA endpoints + unique P-bit, 4-bit indices).
// Encodes raw 0..255 RGBA channel bytes from an `rgba8unorm` view (works identically for both
// `bc7-rgba-unorm-srgb` from sRGB bytes and `bc7-rgba-unorm` from linear bytes, preserving
// alpha masks such as Earth 500m tile water/land masks in [128..255]).

struct Bc7Params {
  width: u32,
  height: u32,
  blocksX: u32,
  blocksY: u32,
  rowStrideBlocks: u32,
  outOffsetBlocks: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Bc7Params;
@group(0) @binding(2) var<storage, read_write> outBlocks: array<vec4u>;

fn writeBits(words: ptr<function, array<u32, 4>>, bitPos: ptr<function, u32>, val: u32, count: u32) {
  let pos = *bitPos;
  let wIdx = pos >> 5u;
  let bOff = pos & 31u;
  let masked = val & ((1u << count) - 1u);
  (*words)[wIdx] = (*words)[wIdx] | (masked << bOff);
  if (bOff + count > 32u) {
    (*words)[wIdx + 1u] = (*words)[wIdx + 1u] | (masked >> (32u - bOff));
  }
  *bitPos = pos + count;
}

@compute @workgroup_size(8, 8)
fn encodeBc7Mode6(@builtin(global_invocation_id) gid: vec3u) {
  let bx = gid.x;
  let by = gid.y;
  if (bx >= params.blocksX || by >= params.blocksY) {
    return;
  }

  var px: array<vec4f, 16>;
  var minC = vec4f(255.0, 255.0, 255.0, 255.0);
  var maxC = vec4f(0.0, 0.0, 0.0, 0.0);
  var mean = vec4f(0.0, 0.0, 0.0, 0.0);

  for (var j = 0u; j < 4u; j = j + 1u) {
    let sy = min(by * 4u + j, params.height - 1u);
    for (var i = 0u; i < 4u; i = i + 1u) {
      let sx = min(bx * 4u + i, params.width - 1u);
      let c = textureLoad(srcTex, vec2u(sx, sy), 0) * 255.0;
      let idx = j * 4u + i;
      px[idx] = c;
      minC = min(minC, c);
      maxC = max(maxC, c);
      mean = mean + c;
    }
  }
  mean = mean * 0.0625;

  // Principal axis via 3 power iterations on the 4x4 RGBA covariance matrix.
  var cov0 = vec4f(0.0);
  var cov1 = vec4f(0.0);
  var cov2 = vec4f(0.0);
  var cov3 = vec4f(0.0);
  for (var k = 0u; k < 16u; k = k + 1u) {
    let d = px[k] - mean;
    cov0 = cov0 + d * d.x;
    cov1 = cov1 + d * d.y;
    cov2 = cov2 + d * d.z;
    cov3 = cov3 + d * d.w;
  }

  var axis = maxC - minC;
  if (dot(axis, axis) < 1e-4) {
    axis = vec4f(1.0, 1.0, 1.0, 0.0);
  } else {
    axis = normalize(axis);
    for (var it = 0u; it < 3u; it = it + 1u) {
      let nxt = vec4f(dot(cov0, axis), dot(cov1, axis), dot(cov2, axis), dot(cov3, axis));
      let len = length(nxt);
      if (len > 1e-5) {
        axis = nxt / len;
      }
    }
  }

  var minT: f32 = 1e9;
  var maxT: f32 = -1e9;
  for (var k = 0u; k < 16u; k = k + 1u) {
    let t = dot(px[k] - mean, axis);
    minT = min(minT, t);
    maxT = max(maxT, t);
  }
  var e0 = clamp(floor(mean + axis * minT + 0.5), vec4f(0.0), vec4f(255.0));
  var e1 = clamp(floor(mean + axis * maxT + 0.5), vec4f(0.0), vec4f(255.0));

  var dir = e1 - e0;
  var dirLen2 = dot(dir, dir);

  // One-step least-squares endpoint refinement in 4D RGBA space.
  if (dirLen2 > 0.5) {
    var sumW: f32 = 0.0;
    var sumW2: f32 = 0.0;
    var sumC = vec4f(0.0);
    var sumWC = vec4f(0.0);
    for (var k = 0u; k < 16u; k = k + 1u) {
      let t = clamp(dot(px[k] - e0, dir) / dirLen2, 0.0, 1.0);
      let q = floor(t * 15.0 + 0.5) / 15.0;
      sumW = sumW + q;
      sumW2 = sumW2 + q * q;
      sumC = sumC + px[k];
      sumWC = sumWC + px[k] * q;
    }
    let det = 16.0 * sumW2 - sumW * sumW;
    if (abs(det) > 1e-3) {
      let invDet = 1.0 / det;
      let newE0 = (sumW2 * sumC - sumW * sumWC) * invDet;
      let slope = (16.0 * sumWC - sumW * sumC) * invDet;
      e0 = clamp(floor(newE0 + 0.5), vec4f(0.0), vec4f(255.0));
      e1 = clamp(floor(newE0 + slope + 0.5), vec4f(0.0), vec4f(255.0));
      dir = e1 - e0;
      dirLen2 = dot(dir, dir);
    }
  }

  var indices: array<u32, 16>;
  for (var k = 0u; k < 16u; k = k + 1u) {
    var idx = 0u;
    if (dirLen2 > 0.5) {
      let t = clamp(dot(px[k] - e0, dir) / dirLen2, 0.0, 1.0);
      idx = min(15u, u32(floor(t * 15.0 + 0.5)));
    }
    indices[k] = idx;
  }

  // Mode 6 anchor rule: index[0] must fit in 3 bits [0..7]. Swap endpoints if >= 8.
  if (indices[0] >= 8u) {
    let tmp = e0;
    e0 = e1;
    e1 = tmp;
    for (var k = 0u; k < 16u; k = k + 1u) {
      indices[k] = 15u - indices[k];
    }
  }

  // Mode 6 stores 7-bit RGBA endpoints + 1 unique P-bit per endpoint.
  // When alpha is constant opaque (255), vote P-bits by RGB LSB majority;
  // when alpha varies (e.g. Earth water/land mask), include alpha in the vote.
  let u0 = vec4u(e0);
  let u1 = vec4u(e1);
  let hasAlpha = (maxC.w - minC.w) > 0.5 || minC.w < 254.0;
  let ones0 = (u0.x & 1u) + (u0.y & 1u) + (u0.z & 1u) + select(0u, u0.w & 1u, hasAlpha);
  let ones1 = (u1.x & 1u) + (u1.y & 1u) + (u1.z & 1u) + select(0u, u1.w & 1u, hasAlpha);
  let p0 = select(0u, 1u, ones0 >= 2u);
  let p1 = select(0u, 1u, ones1 >= 2u);
  // Keep water alpha (128) at <= 128 when P-bit is 1 so (a - 0.5) * 2 stays strictly 0.0.
  let a0 = select(u0.w >> 1u, 63u, u0.w == 128u && p0 == 1u);
  let a1 = select(u1.w >> 1u, 63u, u1.w == 128u && p1 == 1u);

  var words = array<u32, 4>(0u, 0u, 0u, 0u);
  var bitPos = 0u;
  writeBits(&words, &bitPos, 0x40u, 7u);      // Mode 6: 0000001
  writeBits(&words, &bitPos, u0.x >> 1u, 7u);
  writeBits(&words, &bitPos, u1.x >> 1u, 7u);
  writeBits(&words, &bitPos, u0.y >> 1u, 7u);
  writeBits(&words, &bitPos, u1.y >> 1u, 7u);
  writeBits(&words, &bitPos, u0.z >> 1u, 7u);
  writeBits(&words, &bitPos, u1.z >> 1u, 7u);
  writeBits(&words, &bitPos, a0, 7u);
  writeBits(&words, &bitPos, a1, 7u);
  writeBits(&words, &bitPos, p0, 1u);
  writeBits(&words, &bitPos, p1, 1u);
  writeBits(&words, &bitPos, indices[0], 3u); // Anchor index (3 bits)
  for (var k = 1u; k < 16u; k = k + 1u) {
    writeBits(&words, &bitPos, indices[k], 4u);
  }

  let outIndex = params.outOffsetBlocks + by * params.rowStrideBlocks + bx;
  outBlocks[outIndex] = vec4u(words[0], words[1], words[2], words[3]);
}
