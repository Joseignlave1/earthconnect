import { BLOCK_SIZE } from './manifest.ts'

export const EPB1_MAGIC = 0x31425045
export const EPB2_MAGIC = 0x32425045
export const EPB2_HEADER_BYTES = 64

const epb2UnpackWgsl = /* wgsl */ `
// WebGPU compute EPB2 -> EPB1 lossless unpacker.
// Decodes per-128-row-block 1D zigzag deltas (sparse u8 + u16 escape stream),
// reverses cross-column temperature residuals, computes per-block EPB1 descriptors,
// and bit-packs directly into the target EPB1 GPU storage buffer with zero CPU readback.

struct Epb2Params {
  rows: u32,
  fields: u32,
  blocks: u32,
  totalBlocks: u32,
  fieldStridePairs: u32,
  loWordOffset: u32,
  hiWordOffset: u32,
  escWordOffset: u32,
  flags: u32,
  numGroups: u32,
  xcolCold: u32,
  xcolDay: u32,
  xcolHigh: u32,
  xcolLow: u32,
  xcolMean: u32,
  xcolNight: u32,
  xcolRange: u32,
  xcolWarm: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Epb2Params;
@group(0) @binding(1) var<storage, read> srcWords: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratchU16: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockConfigs: array<u32>;
@group(0) @binding(4) var<storage, read_write> blockWordCounts: array<u32>;
@group(0) @binding(5) var<storage, read_write> blockOffsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> groupSums: array<u32>;
@group(0) @binding(7) var<storage, read_write> outEpb1: array<u32>;

var<workgroup> smemA: array<u32, 256>;
var<workgroup> smemB: array<u32, 256>;
var<workgroup> smemC: array<u32, 256>;

fn readLoByte(byteIndex: u32) -> u32 {
  let word = srcWords[params.loWordOffset + (byteIndex >> 2u)];
  return (word >> ((byteIndex & 3u) * 8u)) & 0xffu;
}

fn readHiU16(elemIndex: u32) -> u32 {
  let word = srcWords[params.hiWordOffset + (elemIndex >> 1u)];
  return (word >> ((elemIndex & 1u) * 16u)) & 0xffffu;
}

fn readScratchU16(field: u32, row: u32) -> u32 {
  let word = scratchU16[field * params.fieldStridePairs + (row >> 1u)];
  return (word >> ((row & 1u) * 16u)) & 0xffffu;
}

fn unzigzag16(z: u32) -> u32 {
  let half = z >> 1u;
  let neg = select(0u, 0xffffu, (z & 1u) != 0u);
  return (half ^ neg) & 0xffffu;
}

// Pass 1: Per-(field, block) sparse-u8 decode + 1D 128-row inclusive prefix sum -> scratchU16.
// Dispatch: (blocks, fields, 1)
@compute @workgroup_size(128)
fn undeltaBlocks(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let b = wg.x;
  let f = wg.y;
  let tid = lid.x;
  if (b >= params.blocks || f >= params.fields) {
    return;
  }

  let r = b * 128u + tid;
  let valid = r < params.rows;
  var loByte = 0u;
  if (valid) {
    loByte = readLoByte(f * params.rows + r);
  }
  let isEsc = select(0u, 1u, valid && loByte == 255u);

  // Double-buffered Hillis-Steele inclusive scan of escape flags (7 barriers).
  smemA[tid] = isEsc;
  workgroupBarrier();
  smemB[tid] = smemA[tid] + select(0u, smemA[tid - 1u], tid >= 1u);
  workgroupBarrier();
  smemA[tid] = smemB[tid] + select(0u, smemB[tid - 2u], tid >= 2u);
  workgroupBarrier();
  smemB[tid] = smemA[tid] + select(0u, smemA[tid - 4u], tid >= 4u);
  workgroupBarrier();
  smemA[tid] = smemB[tid] + select(0u, smemB[tid - 8u], tid >= 8u);
  workgroupBarrier();
  smemB[tid] = smemA[tid] + select(0u, smemA[tid - 16u], tid >= 16u);
  workgroupBarrier();
  smemA[tid] = smemB[tid] + select(0u, smemB[tid - 32u], tid >= 32u);
  workgroupBarrier();
  let escIncl = smemA[tid] + select(0u, smemA[tid - 64u], tid >= 64u);
  let escExcl = escIncl - isEsc;
  workgroupBarrier();

  var delta = 0u;
  if (valid) {
    var zz = loByte;
    if (isEsc != 0u) {
      let blockIdx = f * params.blocks + b;
      let escBase = srcWords[params.escWordOffset + blockIdx];
      zz = readHiU16(escBase + escExcl);
    }
    delta = unzigzag16(zz);
  }

  // Double-buffered Hillis-Steele inclusive prefix sum of 16-bit deltas mod 65536 (7 barriers).
  smemA[tid] = delta;
  workgroupBarrier();
  smemB[tid] = (smemA[tid] + select(0u, smemA[tid - 1u], tid >= 1u)) & 0xffffu;
  workgroupBarrier();
  smemA[tid] = (smemB[tid] + select(0u, smemB[tid - 2u], tid >= 2u)) & 0xffffu;
  workgroupBarrier();
  smemB[tid] = (smemA[tid] + select(0u, smemA[tid - 4u], tid >= 4u)) & 0xffffu;
  workgroupBarrier();
  smemA[tid] = (smemB[tid] + select(0u, smemB[tid - 8u], tid >= 8u)) & 0xffffu;
  workgroupBarrier();
  smemB[tid] = (smemA[tid] + select(0u, smemA[tid - 16u], tid >= 16u)) & 0xffffu;
  workgroupBarrier();
  smemA[tid] = (smemB[tid] + select(0u, smemB[tid - 32u], tid >= 32u)) & 0xffffu;
  workgroupBarrier();
  smemB[tid] = (smemA[tid] + select(0u, smemA[tid - 64u], tid >= 64u)) & 0xffffu;
  workgroupBarrier();

  if ((tid & 1u) == 0u && valid) {
    let v0 = smemB[tid];
    let v1 = select(0u, smemB[tid + 1u], r + 1u < params.rows);
    scratchU16[f * params.fieldStridePairs + (r >> 1u)] = v0 | (v1 << 16u);
  }
}

// Pass 2: Undo cross-column temperature residuals in-place in scratchU16 (2 rows per thread).
// Dispatch: (blocks, 1, 1) when (params.flags & 1u) != 0u.
@compute @workgroup_size(64)
fn undoCrossCol(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let pairIdx = wg.x * 64u + lid.x;
  let r0 = pairIdx * 2u;
  if (r0 >= params.rows) {
    return;
  }

  let s = params.fieldStridePairs;
  let dayW = scratchU16[params.xcolDay * s + pairIdx];
  let nightW = scratchU16[params.xcolNight * s + pairIdx];
  let rHighW = scratchU16[params.xcolHigh * s + pairIdx];
  let rLowW = scratchU16[params.xcolLow * s + pairIdx];
  let rMeanW = scratchU16[params.xcolMean * s + pairIdx];
  let rRangeW = scratchU16[params.xcolRange * s + pairIdx];
  let coldW = scratchU16[params.xcolCold * s + pairIdx];
  let rWarmW = scratchU16[params.xcolWarm * s + pairIdx];

  let day0 = dayW & 0xffffu;
  let night0 = nightW & 0xffffu;
  let high0 = (day0 + (rHighW & 0xffffu)) & 0xffffu;
  let low0 = (night0 + (rLowW & 0xffffu)) & 0xffffu;
  let mean0 = (((day0 + night0) >> 1u) + (rMeanW & 0xffffu)) & 0xffffu;
  let range0 = (((high0 - low0) & 0xffffu) + (rRangeW & 0xffffu)) & 0xffffu;
  let warm0 = ((coldW & 0xffffu) + (rWarmW & 0xffffu)) & 0xffffu;

  var high1 = 0u;
  var low1 = 0u;
  var mean1 = 0u;
  var range1 = 0u;
  var warm1 = 0u;
  if (r0 + 1u < params.rows) {
    let day1 = dayW >> 16u;
    let night1 = nightW >> 16u;
    high1 = (day1 + (rHighW >> 16u)) & 0xffffu;
    low1 = (night1 + (rLowW >> 16u)) & 0xffffu;
    mean1 = (((day1 + night1) >> 1u) + (rMeanW >> 16u)) & 0xffffu;
    range1 = (((high1 - low1) & 0xffffu) + (rRangeW >> 16u)) & 0xffffu;
    warm1 = ((coldW >> 16u) + (rWarmW >> 16u)) & 0xffffu;
  }

  scratchU16[params.xcolHigh * s + pairIdx] = high0 | (high1 << 16u);
  scratchU16[params.xcolLow * s + pairIdx] = low0 | (low1 << 16u);
  scratchU16[params.xcolMean * s + pairIdx] = mean0 | (mean1 << 16u);
  scratchU16[params.xcolRange * s + pairIdx] = range0 | (range1 << 16u);
  scratchU16[params.xcolWarm * s + pairIdx] = warm0 | (warm1 << 16u);
}

// Pass 3: Compute EPB1 config (base, bits, has_missing) and word count for each (field, block).
// Dispatch: (blocks, fields, 1)
@compute @workgroup_size(128)
fn measureBlocks(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let b = wg.x;
  let f = wg.y;
  let tid = lid.x;
  if (b >= params.blocks || f >= params.fields) {
    return;
  }

  let r = b * 128u + tid;
  var hm = 0u;
  var mn = 0xffffu;
  var mx = 0u;
  if (r < params.rows) {
    let v = readScratchU16(f, r);
    if (v == 0u) {
      hm = 1u;
    } else {
      mn = v;
      mx = v;
    }
  }

  smemA[tid] = hm;
  smemB[tid] = mn;
  smemC[tid] = mx;
  workgroupBarrier();

  for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
    if (tid < stride) {
      smemA[tid] = smemA[tid] | smemA[tid + stride];
      smemB[tid] = min(smemB[tid], smemB[tid + stride]);
      smemC[tid] = max(smemC[tid], smemC[tid + stride]);
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let blockIdx = f * params.blocks + b;
    let hasMissing = smemA[0];
    let minV = smemB[0];
    let maxV = smemC[0];
    let blockRows = min(128u, params.rows - b * 128u);

    if (minV == 0xffffu && maxV == 0u) {
      blockConfigs[blockIdx] = 0u;
      blockWordCounts[blockIdx] = 0u;
    } else if (hasMissing == 0u) {
      let span = maxV - minV;
      let bits = select(32u - countLeadingZeros(span), 0u, span == 0u);
      blockConfigs[blockIdx] = minV | (bits << 16u);
      blockWordCounts[blockIdx] = select((blockRows * bits + 31u) >> 5u, 0u, bits == 0u);
    } else {
      let span = (maxV - minV) + 1u;
      let bits = 32u - countLeadingZeros(span);
      blockConfigs[blockIdx] = minV | (bits << 16u) | (1u << 21u);
      blockWordCounts[blockIdx] = (blockRows * bits + 31u) >> 5u;
    }
  }
}

// Pass 4a: Local exclusive prefix sum of blockWordCounts in groups of 256.
// Dispatch: (numGroups, 1, 1)
@compute @workgroup_size(256)
fn scanLocal(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let g = wg.x;
  let tid = lid.x;
  let idx = g * 256u + tid;
  let val = select(0u, blockWordCounts[idx], idx < params.totalBlocks);
  smemA[tid] = val;
  workgroupBarrier();

  for (var stride = 1u; stride < 256u; stride = stride << 1u) {
    var addVal = 0u;
    if (tid >= stride) {
      addVal = smemA[tid - stride];
    }
    workgroupBarrier();
    smemA[tid] = smemA[tid] + addVal;
    workgroupBarrier();
  }

  if (idx < params.totalBlocks) {
    blockOffsets[idx] = smemA[tid] - val;
  }
  if (tid == 255u) {
    groupSums[g] = smemA[255];
  }
}

// Pass 4b: Exclusive prefix sum across groupSums (single workgroup, supports arbitrary numGroups).
// Dispatch: (1, 1, 1)
@compute @workgroup_size(256)
fn scanGroups(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  var running = 0u;
  var base = 0u;
  loop {
    if (base >= params.numGroups) {
      break;
    }
    let idx = base + tid;
    let val = select(0u, groupSums[idx], idx < params.numGroups);
    smemA[tid] = val;
    workgroupBarrier();

    for (var stride = 1u; stride < 256u; stride = stride << 1u) {
      var addVal = 0u;
      if (tid >= stride) {
        addVal = smemA[tid - stride];
      }
      workgroupBarrier();
      smemA[tid] = smemA[tid] + addVal;
      workgroupBarrier();
    }

    if (idx < params.numGroups) {
      groupSums[idx] = running + smemA[tid] - val;
    }
    running = running + smemA[255];
    workgroupBarrier();
    base = base + 256u;
  }
}

// Pass 4c: Write final EPB1 header and all (offset, config) descriptors directly into outEpb1.
// Dispatch: (numGroups, 1, 1)
@compute @workgroup_size(256)
fn writeDescriptors(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let g = wg.x;
  let tid = lid.x;
  let idx = g * 256u + tid;
  if (idx == 0u) {
    outEpb1[0] = 0x31425045u;
    outEpb1[1] = 128u;
    outEpb1[2] = params.rows;
    outEpb1[3] = params.fields;
  }
  if (idx >= params.totalBlocks) {
    return;
  }
  let initialOffset = 4u + 2u * params.totalBlocks;
  let finalOffset = initialOffset + groupSums[g] + blockOffsets[idx];
  blockOffsets[idx] = finalOffset;
  let d = 4u + 2u * idx;
  outEpb1[d] = finalOffset;
  outEpb1[d + 1u] = blockConfigs[idx];
}

// Pass 5: Bit-pack 128 values per (field, block) directly into outEpb1 (1 thread per output u32 word).
// Dispatch: (blocks, fields, 1)
@compute @workgroup_size(128)
fn packEpb1(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let b = wg.x;
  let f = wg.y;
  let tid = lid.x;
  if (b >= params.blocks || f >= params.fields) {
    return;
  }

  let blockIdx = f * params.blocks + b;
  let config = blockConfigs[blockIdx];
  let bits = (config >> 16u) & 31u;
  let base = config & 0xffffu;
  let hasMissing = (config & (1u << 21u)) != 0u;
  let r0 = b * 128u;
  let blockRows = min(128u, params.rows - r0);

  // Stage all block codes (0..127) in shared memory before barrier (uniform control flow).
  var code = 0u;
  if (bits != 0u && tid < blockRows) {
    let v = readScratchU16(f, r0 + tid);
    if (hasMissing) {
      code = select(v - base + 1u, 0u, v == 0u);
    } else {
      code = v - base;
    }
  }
  smemA[tid] = code;
  workgroupBarrier();

  if (bits == 0u) {
    return;
  }

  let nwords = (blockRows * bits + 31u) >> 5u;
  if (tid >= nwords) {
    return;
  }

  let bitStart = tid * 32u;
  let bitEnd = bitStart + 32u;
  let firstRow = bitStart / bits;
  let lastRow = min(blockRows - 1u, (bitEnd - 1u) / bits);

  var outWord = 0u;
  for (var k = firstRow; k <= lastRow; k = k + 1u) {
    let rowBit = k * bits;
    let c = smemA[k];
    if (rowBit >= bitStart) {
      outWord = outWord | (c << (rowBit - bitStart));
    } else {
      outWord = outWord | (c >> (bitStart - rowBit));
    }
  }
  outEpb1[blockOffsets[blockIdx] + tid] = outWord;
}
`

type Epb2Pipelines = {
  layout: GPUBindGroupLayout
  undelta: GPUComputePipeline
  undoCrossCol: GPUComputePipeline
  measure: GPUComputePipeline
  scanLocal: GPUComputePipeline
  scanGroups: GPUComputePipeline
  writeDescriptors: GPUComputePipeline
  pack: GPUComputePipeline
}

const pipelineCache = new WeakMap<GPUDevice, Epb2Pipelines>()

function getPipelines(device: GPUDevice): Epb2Pipelines {
  let cached = pipelineCache.get(device)
  if (!cached) {
    const module = device.createShaderModule({ label: 'EPB2 GPU unpacker', code: epb2UnpackWgsl })
    const layout = device.createBindGroupLayout({
      label: 'EPB2 unpack bind group layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    const pipelineLayout = device.createPipelineLayout({
      label: 'EPB2 unpack pipeline layout',
      bindGroupLayouts: [layout],
    })
    const make = (entryPoint: string) =>
      device.createComputePipeline({
        label: `EPB2 ${entryPoint}`,
        layout: pipelineLayout,
        compute: { module, entryPoint },
      })
    cached = {
      layout,
      undelta: make('undeltaBlocks'),
      undoCrossCol: make('undoCrossCol'),
      measure: make('measureBlocks'),
      scanLocal: make('scanLocal'),
      scanGroups: make('scanGroups'),
      writeDescriptors: make('writeDescriptors'),
      pack: make('packEpb1'),
    }
    pipelineCache.set(device, cached)
  }
  return cached
}

export class Epb2GpuUnpacker {
  private readonly device: GPUDevice
  private readonly pipelines: Epb2Pipelines
  private readonly paramsBuffer: GPUBuffer
  private readonly retiredBuffers: GPUBuffer[] = []
  private srcBuffer: GPUBuffer | null = null
  private srcCapacity = 0
  private scratchBuffer: GPUBuffer | null = null
  private scratchCapacity = 0
  private blockConfigsBuffer: GPUBuffer | null = null
  private blockWordCountsBuffer: GPUBuffer | null = null
  private blockOffsetsBuffer: GPUBuffer | null = null
  private blocksCapacity = 0
  private groupSumsBuffer: GPUBuffer | null = null
  private groupsCapacity = 0
  private destroyed = false

  constructor(device: GPUDevice, maxRows: number, fields: number) {
    this.device = device
    this.pipelines = getPipelines(device)
    this.paramsBuffer = device.createBuffer({
      label: 'EPB2 unpack params',
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const maxBlocks = Math.ceil(maxRows / BLOCK_SIZE)
    const maxTotalBlocks = fields * maxBlocks
    const maxGroups = Math.ceil(maxTotalBlocks / 256)
    const maxScratchBytes = fields * ((maxRows + 1) >> 1) * 4
    // Pre-size source buffer generously (header + lo + ~15% escape headroom + blockEscapeOffsets)
    // so typical chunks never trigger a mid-stream buffer reallocation.
    const estSrcBytes = EPB2_HEADER_BYTES + Math.ceil(fields * maxRows * 1.3) + maxTotalBlocks * 4
    this.ensureScratch(estSrcBytes, maxScratchBytes, maxTotalBlocks, maxGroups)
  }

  private retire(buf: GPUBuffer | null) {
    if (buf) this.retiredBuffers.push(buf)
  }

  private ensureScratch(srcBytes: number, scratchBytes: number, totalBlocks: number, numGroups: number) {
    const device = this.device
    if (!this.srcBuffer || srcBytes > this.srcCapacity) {
      this.retire(this.srcBuffer)
      this.srcCapacity = Math.ceil(srcBytes / 4096) * 4096
      this.srcBuffer = device.createBuffer({
        label: 'EPB2 source scratch',
        size: this.srcCapacity,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
    }
    if (!this.scratchBuffer || scratchBytes > this.scratchCapacity) {
      this.retire(this.scratchBuffer)
      this.scratchCapacity = Math.ceil(scratchBytes / 4096) * 4096
      this.scratchBuffer = device.createBuffer({
        label: 'EPB2 u16 scratch',
        size: this.scratchCapacity,
        usage: GPUBufferUsage.STORAGE,
      })
    }
    if (!this.blockConfigsBuffer || totalBlocks > this.blocksCapacity) {
      this.retire(this.blockConfigsBuffer)
      this.retire(this.blockWordCountsBuffer)
      this.retire(this.blockOffsetsBuffer)
      this.blocksCapacity = Math.max(256, Math.ceil(totalBlocks / 1024) * 1024)
      const bytes = this.blocksCapacity * 4
      this.blockConfigsBuffer = device.createBuffer({
        label: 'EPB2 block configs',
        size: bytes,
        usage: GPUBufferUsage.STORAGE,
      })
      this.blockWordCountsBuffer = device.createBuffer({
        label: 'EPB2 block word counts',
        size: bytes,
        usage: GPUBufferUsage.STORAGE,
      })
      this.blockOffsetsBuffer = device.createBuffer({
        label: 'EPB2 block offsets',
        size: bytes,
        usage: GPUBufferUsage.STORAGE,
      })
    }
    if (!this.groupSumsBuffer || numGroups > this.groupsCapacity) {
      this.retire(this.groupSumsBuffer)
      this.groupsCapacity = Math.max(16, Math.ceil(numGroups / 64) * 64)
      this.groupSumsBuffer = device.createBuffer({
        label: 'EPB2 group sums',
        size: this.groupsCapacity * 4,
        usage: GPUBufferUsage.STORAGE,
      })
    }
  }

  /**
   * Unpack an EPB2 buffer (header + lo + hi + blockEscapeOffsets) straight into `outEpb1`
   * on the GPU in a single submitted command buffer.
   */
  unpackToBuffer(epb2Buffer: ArrayBuffer, outEpb1: GPUBuffer) {
    if (this.destroyed) throw new Error('EPB2 GPU unpacker was destroyed.')
    const header = new Uint32Array(epb2Buffer, 0, 16)
    if (header[0] !== EPB2_MAGIC) throw new Error('Invalid EPB2 magic.')
    const blockSize = header[1]
    const rows = header[2]
    const fields = header[3]
    const flags = header[5]
    const loBytes = header[6]
    const escapeCount = header[7]
    if (blockSize !== BLOCK_SIZE) throw new Error(`Unsupported EPB2 block size ${blockSize}.`)

    const blocks = Math.ceil(rows / BLOCK_SIZE)
    const totalBlocks = fields * blocks
    const numGroups = Math.ceil(totalBlocks / 256)
    const fieldStridePairs = (rows + 1) >> 1
    const scratchBytes = fields * fieldStridePairs * 4

    const loAlignedBytes = (loBytes + 3) & ~3
    const hiAlignedBytes = (escapeCount * 2 + 3) & ~3
    const loWordOffset = EPB2_HEADER_BYTES >> 2
    const hiWordOffset = loWordOffset + (loAlignedBytes >> 2)
    const escWordOffset = hiWordOffset + (hiAlignedBytes >> 2)
    const payloadBytes = (escWordOffset + totalBlocks) * 4
    if (epb2Buffer.byteLength < payloadBytes) {
      throw new Error('Truncated EPB2 buffer.')
    }

    this.ensureScratch(payloadBytes, scratchBytes, totalBlocks, numGroups)

    const params = new Uint32Array(20)
    params[0] = rows
    params[1] = fields
    params[2] = blocks
    params[3] = totalBlocks
    params[4] = fieldStridePairs
    params[5] = loWordOffset
    params[6] = hiWordOffset
    params[7] = escWordOffset
    params[8] = flags
    params[9] = numGroups
    for (let i = 0; i < 8; i++) params[10 + i] = header[8 + i]

    const device = this.device
    device.queue.writeBuffer(this.paramsBuffer, 0, params)
    device.queue.writeBuffer(this.srcBuffer!, 0, epb2Buffer, 0, payloadBytes)

    const bindGroup = device.createBindGroup({
      label: 'EPB2 unpack bind group',
      layout: this.pipelines.layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.srcBuffer! } },
        { binding: 2, resource: { buffer: this.scratchBuffer! } },
        { binding: 3, resource: { buffer: this.blockConfigsBuffer! } },
        { binding: 4, resource: { buffer: this.blockWordCountsBuffer! } },
        { binding: 5, resource: { buffer: this.blockOffsetsBuffer! } },
        { binding: 6, resource: { buffer: this.groupSumsBuffer! } },
        { binding: 7, resource: { buffer: outEpb1 } },
      ],
    })

    const encoder = device.createCommandEncoder({ label: 'EPB2 -> EPB1 GPU unpack' })
    const pass = encoder.beginComputePass({ label: 'EPB2 unpack passes' })
    pass.setBindGroup(0, bindGroup)

    pass.setPipeline(this.pipelines.undelta)
    pass.dispatchWorkgroups(blocks, fields, 1)

    if ((flags & 1) !== 0) {
      pass.setPipeline(this.pipelines.undoCrossCol)
      pass.dispatchWorkgroups(blocks, 1, 1)
    }

    pass.setPipeline(this.pipelines.measure)
    pass.dispatchWorkgroups(blocks, fields, 1)

    pass.setPipeline(this.pipelines.scanLocal)
    pass.dispatchWorkgroups(numGroups, 1, 1)
    pass.setPipeline(this.pipelines.scanGroups)
    pass.dispatchWorkgroups(1, 1, 1)
    pass.setPipeline(this.pipelines.writeDescriptors)
    pass.dispatchWorkgroups(numGroups, 1, 1)

    pass.setPipeline(this.pipelines.pack)
    pass.dispatchWorkgroups(blocks, fields, 1)

    pass.end()
    device.queue.submit([encoder.finish()])
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.paramsBuffer.destroy()
    this.srcBuffer?.destroy()
    this.scratchBuffer?.destroy()
    this.blockConfigsBuffer?.destroy()
    this.blockWordCountsBuffer?.destroy()
    this.blockOffsetsBuffer?.destroy()
    this.groupSumsBuffer?.destroy()
    for (const buf of this.retiredBuffers) buf.destroy()
    this.retiredBuffers.length = 0
  }
}
