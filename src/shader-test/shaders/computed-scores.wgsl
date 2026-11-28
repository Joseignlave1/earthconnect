override MISSING: u32 = 65535u;
override SIZE: u32 = 256u;
override TILE_CELLS: u32 = 65536u;
override GROUP_WIDTH: u32 = 16u;
override GROUP_HEIGHT: u32 = 8u;
override HISTOGRAM_SHARDS: u32 = 16u;

struct BatchParams {
  firstSite: u32,
  firstChunkTile: u32,
  packLayerOffset: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> peaks: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> batch: BatchParams;
@group(0) @binding(5) var previous: texture_2d_array<u32>;
@group(0) @binding(6) var<storage, read_write> summary: array<u32>;

var<workgroup> groupPeak: atomic<u32>;
var<workgroup> zeros: atomic<u32>;
var<workgroup> hotCode: u32;
var<workgroup> hotCount: atomic<u32>;

// Two uint16 pixels share a storage word. Copy rows have 256-byte alignment.
fn storePair(id: vec3u, layerOffset: u32, dimensions: vec2u, lo: u32, hi: u32) {
  let wordsPerRow = ((dimensions.x + 127u) / 128u) * 64u;
  packed[((layerOffset + id.z) * dimensions.y + id.y) * wordsPerRow + id.x] = lo | (hi << 16u);
}

fn count(code: u32, amount: u32, shard: u32, hot: u32) {
  if (code == 0u) { atomicAdd(&zeros, amount); }
  else if (hot != 0u && code == hot) { atomicAdd(&hotCount, amount); }
  else if (code < MISSING) { atomicAdd(&histogram[shard * MISSING + code], amount); }
}

@compute @workgroup_size(GROUP_WIDTH, GROUP_HEIGHT)
fn base(
  @builtin(global_invocation_id) id: vec3u,
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_index) local: u32,
) {
  let x = id.x * 2u;
  let site = batch.firstSite + id.z;
  let shard = (group.x + group.y * 7u + site * 11u) % HISTOGRAM_SHARDS;
  let inBounds = x < SIZE && id.y < SIZE;
  var lo = MISSING;
  var hi = MISSING;
  if (inBounds) {
    let row = (batch.firstChunkTile + id.z) * TILE_CELLS + id.y * SIZE + x;
    lo = min(MISSING, scores[row]);
    hi = min(MISSING, scores[row + 1u]);
    storePair(id, batch.packLayerOffset, vec2u(SIZE), lo, hi);
  }
  if (local == 0u) {
    hotCode = select(0u, lo, lo > 0u && lo < MISSING);
  }
  workgroupBarrier();
  let hot = hotCode;
  if (inBounds) {
    if (lo == hi) { count(lo, 2u, shard, hot); }
    else {
      count(lo, 1u, shard, hot);
      count(hi, 1u, shard, hot);
    }
    if (lo < MISSING || hi < MISSING) {
      let useHi = lo == MISSING || (hi < MISSING && hi > lo);
      let code = select(lo, hi, useHi);
      // A group contains 256 pixels. Lower pixel addresses break score ties.
      let pixel = local * 2u + u32(useHi);
      atomicMax(&groupPeak, ((code + 1u) << 8u) | (255u - pixel));
    }
  }
  workgroupBarrier();
  if (local == 0u) {
    let zeroCount = atomicLoad(&zeros);
    if (zeroCount > 0u) { atomicAdd(&histogram[shard * MISSING], zeroCount); }
    let cachedCount = atomicLoad(&hotCount);
    if (cachedCount > 0u) { atomicAdd(&histogram[shard * MISSING + hot], cachedCount); }
    let peak = atomicLoad(&groupPeak);
    if (peak > 0u) {
      let localPixel = 255u - (peak & 255u);
      let px = group.x * GROUP_WIDTH * 2u + localPixel % (GROUP_WIDTH * 2u);
      let py = group.y * GROUP_HEIGHT + localPixel / (GROUP_WIDTH * 2u);
      let pixel = py * SIZE + px;
      // The tile's address needs 17 bits. Each half fits a score and address
      // into one atomic word, with zero reserved for an empty half.
      atomicMax(&peaks[site * 2u + (pixel >> 16u)], ((peak >> 8u) << 16u) | (65535u - (pixel & 65535u)));
    }
  }
}

fn footprint(position: vec2u, layer: u32, dimensions: vec2u, source: vec2u) -> u32 {
  let first = position * source / dimensions;
  let end = (position + 1u) * source / dimensions;
  var best = MISSING;
  for (var y = first.y; y < end.y; y++) {
    for (var x = first.x; x < end.x; x++) {
      let code = textureLoad(previous, vec2i(i32(x), i32(y)), i32(layer), 0).x;
      if (code < MISSING) { best = select(max(best, code), code, best == MISSING); }
    }
  }
  return best;
}

@compute @workgroup_size(GROUP_WIDTH, GROUP_HEIGHT)
fn mip(@builtin(global_invocation_id) id: vec3u) {
  let source = textureDimensions(previous);
  let dimensions = max(vec2u(1u), source / 2u);
  let x = id.x * 2u;
  if (x >= dimensions.x || id.y >= dimensions.y) { return; }
  let lo = footprint(vec2u(x, id.y), id.z, dimensions, source);
  var hi = MISSING;
  if (x + 1u < dimensions.x) { hi = footprint(vec2u(x + 1u, id.y), id.z, dimensions, source); }
  storePair(id, 0u, dimensions, lo, hi);
}

@compute @workgroup_size(256)
fn summarize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= MISSING) { return; }
  var total = 0u;
  for (var shard = 0u; shard < HISTOGRAM_SHARDS; shard++) {
    total += atomicLoad(&histogram[shard * MISSING + id.x]);
  }
  summary[id.x] = total;
}
