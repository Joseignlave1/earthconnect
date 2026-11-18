import type { EarthScore, ScoreChunk } from '../earth-score/index.ts'
import { checkedGpu } from '../earth-score/gpu.ts'
import { uploadBuffer } from './gpu.ts'
import { MISSING, SIZE } from './world-grid.ts'

const tileCells = SIZE ** 2
export const scoreMipCount = Math.floor(Math.log2(SIZE)) + 1
const maxBatchLayers = 16
const bins = MISSING
const histogramShards = 16
const histogramBytes = bins * Uint32Array.BYTES_PER_ELEMENT
const groupWidth = 16
const groupHeight = 8

export type ScoreTextures = { lookup: GPUTexture; banks: GPUTexture[] }

export function createScoreTextures(device: GPUDevice, bankSize: number, bankCount: number, sites: number): ScoreTextures {
  return {
    lookup: device.createTexture({ label: 'Sinusoidal tile lookup',size: [36,18],format: 'r16uint',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }),
    banks: Array.from({ length: bankCount },(_,bank) => device.createTexture({
      label: `Native scores ${bank}`,size: [SIZE,SIZE,Math.max(1,Math.min(bankSize,sites - bank * bankSize))],
      mipLevelCount: scoreMipCount,format: 'r16uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    })),
  }
}

export function destroyScoreTextures(map: ScoreTextures) {
  map.lookup.destroy()
  for (const texture of map.banks) texture.destroy()
}

export function writeScoreLookup(device: GPUDevice, map: ScoreTextures, lookup: Uint16Array) {
  device.queue.writeTexture({ texture: map.lookup },lookup,{ bytesPerRow: 72 },[36,18])
}

type Batch = { chunk: ScoreChunk; chunkTile: number; firstSite: number; bank: number; layer: number; layers: number }

export function computedScoreLayout(earth: Pick<EarthScore, 'chunks' | 'layout'>, bankSize: number, bankCount: number) {
  const { layout, chunks } = earth
  if (layout.tileSize !== SIZE || layout.rows !== layout.sites.length * tileCells ||
      !layout.rows || layout.sites.length > bankSize * bankCount) {
    throw new Error('Earth Score does not match the lab’s native score grid.')
  }
  const lookup = new Uint16Array(36 * 18).fill(MISSING)
  layout.sites.forEach((site,index) => {
    const h = Number(site.slice(1,3)),v = Number(site.slice(4,6))
    lookup[v * 36 + h] = index
  })
  const batches: Batch[] = []
  let nextSite = 0
  for (const chunk of chunks) {
    const tiles = chunk.rows / tileCells
    if (chunk.firstAddress !== nextSite * tileCells || !Number.isInteger(tiles) || tiles < 1 ||
        chunk.scores.size < chunk.rows * 4) throw new Error('Invalid Earth Score chunk layout.')
    for (let chunkTile = 0; chunkTile < tiles;) {
      const layer = nextSite % bankSize
      const layers = Math.min(tiles - chunkTile,bankSize - layer,maxBatchLayers)
      batches.push({ chunk,chunkTile,firstSite: nextSite,bank: Math.floor(nextSite / bankSize),layer,layers })
      nextSite += layers
      chunkTile += layers
    }
  }
  if (nextSite !== layout.sites.length) throw new Error('Incomplete Earth Score chunk layout.')
  const bytesPerTile = Array.from({ length: scoreMipCount },(_,lod) => (SIZE >> lod) ** 2 * 2)
    .reduce((total,bytes) => total + bytes,0)
  return { lookup,batches,bytes: bytesPerTile * layout.sites.length }
}

const wgsl = /* wgsl */ `
const missing = ${MISSING}u;
const size = ${SIZE}u;
const tileCells = ${tileCells}u;
const groupWidth = ${groupWidth}u;
const groupHeight = ${groupHeight}u;

@group(0) @binding(0) var<storage,read> scores: array<u32>;
@group(0) @binding(1) var<storage,read_write> packed: array<u32>;
@group(0) @binding(2) var<storage,read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> peaks: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> batch: vec4u; // first manifest site, first chunk tile
@group(0) @binding(5) var previous: texture_2d_array<u32>;
@group(0) @binding(6) var<storage,read_write> summary: array<u32>;

var<workgroup> groupPeak: atomic<u32>;
var<workgroup> zeros: atomic<u32>;

// Two uint16 pixels share a storage word. Copy rows have 256-byte alignment.
fn storePair(id: vec3u, dimensions: vec2u, lo: u32, hi: u32) {
  let wordsPerRow = ((dimensions.x + 127u) / 128u) * 64u;
  packed[(id.z * dimensions.y + id.y) * wordsPerRow + id.x] = lo | (hi << 16u);
}

fn count(code: u32, amount: u32, shard: u32) {
  if (code == 0u) { atomicAdd(&zeros,amount); }
  else if (code < missing) { atomicAdd(&histogram[shard * missing + code],amount); }
}

@compute @workgroup_size(${groupWidth},${groupHeight})
fn base(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) group: vec3u,
        @builtin(local_invocation_index) local: u32) {
  let x = id.x * 2u;
  let site = batch.x + id.z;
  let shard = (group.x + group.y * 7u + site * 11u) % ${histogramShards}u;
  if (x < size && id.y < size) {
    let row = (batch.y + id.z) * tileCells + id.y * size + x;
    let lo = min(missing,scores[row]);
    let hi = min(missing,scores[row + 1u]);
    storePair(id,vec2u(size),lo,hi);
    if (lo == hi) { count(lo,2u,shard); }
    else {
      count(lo,1u,shard);
      count(hi,1u,shard);
    }
    if (lo < missing || hi < missing) {
      let useHi = lo == missing || (hi < missing && hi > lo);
      let code = select(lo,hi,useHi);
      // A group contains 256 pixels. Lower pixel addresses break score ties.
      let pixel = local * 2u + u32(useHi);
      atomicMax(&groupPeak,((code + 1u) << 8u) | (255u - pixel));
    }
  }
  workgroupBarrier();
  if (local == 0u) {
    let zeroCount = atomicLoad(&zeros);
    if (zeroCount > 0u) { atomicAdd(&histogram[shard * missing],zeroCount); }
    let peak = atomicLoad(&groupPeak);
    if (peak > 0u) {
      let localPixel = 255u - (peak & 255u);
      let px = group.x * groupWidth * 2u + localPixel % (groupWidth * 2u);
      let py = group.y * groupHeight + localPixel / (groupWidth * 2u);
      let pixel = py * size + px;
      // The tile's address needs 17 bits. Each half fits a score and address
      // into one atomic word, with zero reserved for an empty half.
      atomicMax(&peaks[site * 2u + (pixel >> 16u)],((peak >> 8u) << 16u) | (65535u - (pixel & 65535u)));
    }
  }
}

fn footprint(position: vec2u, layer: u32, dimensions: vec2u, source: vec2u) -> u32 {
  let first = position * source / dimensions;
  let end = (position + 1u) * source / dimensions;
  var best = missing;
  for (var y = first.y; y < end.y; y++) {
    for (var x = first.x; x < end.x; x++) {
      let code = textureLoad(previous,vec2i(i32(x),i32(y)),i32(layer),0).x;
      if (code < missing) { best = select(max(best,code),code,best == missing); }
    }
  }
  return best;
}

@compute @workgroup_size(${groupWidth},${groupHeight})
fn mip(@builtin(global_invocation_id) id: vec3u) {
  let source = textureDimensions(previous);
  let dimensions = max(vec2u(1u),source / 2u);
  let x = id.x * 2u;
  if (x >= dimensions.x || id.y >= dimensions.y) { return; }
  let lo = footprint(vec2u(x,id.y),id.z,dimensions,source);
  var hi = missing;
  if (x + 1u < dimensions.x) { hi = footprint(vec2u(x + 1u,id.y),id.z,dimensions,source); }
  storePair(id,dimensions,lo,hi);
}

@compute @workgroup_size(256)
fn summarize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= missing) { return; }
  var total = 0u;
  for (var shard = 0u; shard < ${histogramShards}u; shard++) {
    total += atomicLoad(&histogram[shard * missing + id.x]);
  }
  summary[id.x] = total;
}
`

type Pipelines = { base: GPUComputePipeline; mip: GPUComputePipeline; summarize: GPUComputePipeline }
const pipelines = new WeakMap<GPUDevice,Promise<Pipelines>>()

function scorePipelines(device: GPUDevice) {
  const cached = pipelines.get(device)
  if (cached) return cached
  const pending = checkedGpu(device,() => {
    const module = device.createShaderModule({ label: 'Native score tiles and max LODs',code: wgsl })
    const make = (entryPoint: string) => device.createComputePipeline({
      label: `Native scores: ${entryPoint}`,layout: 'auto',compute: { module,entryPoint },
    })
    return { base: make('base'),mip: make('mip'),summarize: make('summarize') }
  },'Preparing the GPU score pipelines')
  pipelines.set(device,pending)
  void pending.catch(() => { pipelines.delete(device) })
  return pending
}

function rowBytes(width: number) { return Math.ceil(width * 2 / 256) * 256 }

type ComputedMap = {
  textures: ScoreTextures
  histogram: Uint32Array<ArrayBuffer>
  bestAddress: number | null
  bytes: number
}

async function readBuffer<T>(buffer: GPUBuffer, read: (data: ArrayBuffer) => T, signal?: AbortSignal) {
  const cancel = () => buffer.destroy()
  signal?.addEventListener('abort',cancel,{ once: true })
  try {
    signal?.throwIfAborted()
    await buffer.mapAsync(GPUMapMode.READ)
    signal?.throwIfAborted()
    return read(buffer.getMappedRange())
  } catch (error) {
    signal?.throwIfAborted()
    throw error
  } finally {
    signal?.removeEventListener('abort',cancel)
    if (buffer.mapState === 'mapped') buffer.unmap()
  }
}

export async function prepareComputedScores(device: GPUDevice, earth: Pick<EarthScore, 'chunks' | 'layout'>, bankSize: number, bankCount: number, signal?: AbortSignal): Promise<ComputedMap> {
  signal?.throwIfAborted()
  const { lookup,batches,bytes } = computedScoreLayout(earth,bankSize,bankCount)
  const { base,mip,summarize } = await scorePipelines(device)
  signal?.throwIfAborted()
  const buffers: GPUBuffer[] = []
  const makeBuffer = (label: string,size: number,usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label,size,usage })
    buffers.push(buffer)
    return buffer
  }
  let textures: ScoreTextures | undefined
  try {
    const staging = await checkedGpu(device,() => {
      const sites = earth.layout.sites.length
      const peakBytes = sites * 2 * 4
      const map = textures = createScoreTextures(device,bankSize,bankCount,sites)
      const storage = GPUBufferUsage.STORAGE
      const pack = makeBuffer('Score tile copy rows',rowBytes(SIZE) * SIZE * Math.max(...batches.map(batch => batch.layers)),storage | GPUBufferUsage.COPY_SRC)
      const histogram = makeBuffer('Score histogram shards',histogramBytes * histogramShards,storage)
      const peaks = makeBuffer('Tile peak addresses',peakBytes,storage | GPUBufferUsage.COPY_SRC)
      const summary = makeBuffer('Score histogram',histogramBytes,storage | GPUBufferUsage.COPY_SRC)
      const readback = makeBuffer('Score statistics readback',histogramBytes + peakBytes,GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
      const stride = device.limits.minUniformBufferOffsetAlignment / 4
      const offsets = new Uint32Array(stride * batches.length)
      batches.forEach(({ firstSite,chunkTile },i) => offsets.set([firstSite,chunkTile],i * stride))
      const params = uploadBuffer(device,'Score tile batch offsets',offsets,GPUBufferUsage.UNIFORM)
      buffers.push(params)
      const encoder = device.createCommandEncoder({ label: 'Publish Earth Score map' })
      const encode = (pipeline: GPUComputePipeline,entries: GPUBindGroupEntry[],x: number,y = 1,z = 1) => {
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),entries })
        const pass = encoder.beginComputePass()
        pass.setPipeline(pipeline)
        pass.setBindGroup(0,group)
        pass.dispatchWorkgroups(x,y,z)
        pass.end()
      }
      for (const [i,batch] of batches.entries()) {
        const { chunk,bank,layer,layers } = batch
        const texture = map.banks[bank]
        for (let lod = 0; lod < scoreMipCount; lod++) {
          const width = SIZE >> lod
          const x = Math.ceil(width / (groupWidth * 2)),y = Math.ceil(width / groupHeight)
          if (lod === 0) {
            encode(base,[
              { binding: 0,resource: { buffer: chunk.scores } },{ binding: 1,resource: { buffer: pack } },
              { binding: 2,resource: { buffer: histogram } },{ binding: 3,resource: { buffer: peaks } },
              { binding: 4,resource: { buffer: params,offset: i * stride * 4,size: 16 } },
            ],x,y,layers)
          } else {
            const previous = texture.createView({ dimension: '2d-array',baseMipLevel: lod - 1,mipLevelCount: 1,baseArrayLayer: layer,arrayLayerCount: layers })
            encode(mip,[{ binding: 1,resource: { buffer: pack } },{ binding: 5,resource: previous }],x,y,layers)
          }
          encoder.copyBufferToTexture({ buffer: pack,bytesPerRow: rowBytes(width),rowsPerImage: width },
            { texture,mipLevel: lod,origin: [0,0,layer] },[width,width,layers])
        }
      }
      encode(summarize,[{ binding: 2,resource: { buffer: histogram } },{ binding: 6,resource: { buffer: summary } }],Math.ceil(bins / 256))
      encoder.copyBufferToBuffer(summary,0,readback,0,histogramBytes)
      encoder.copyBufferToBuffer(peaks,0,readback,histogramBytes,peakBytes)
      writeScoreLookup(device,map,lookup)
      device.queue.submit([encoder.finish()])
      return readback
    },'Building the GPU score map')
    return await readBuffer(staging,data => {
      const counts = new Uint32Array(data,0,bins).slice()
      const peaks = new Uint32Array(data,histogramBytes)
      let bestCode = 0,bestAddress: number | null = null
      for (const [half,peak] of peaks.entries()) {
        if (!peak) continue
        const code = (peak >>> 16) - 1
        // Iterating in address order preserves the first maximum on ties.
        if (bestAddress === null || code > bestCode) {
          bestCode = code
          bestAddress = (half >>> 1) * tileCells + (half % 2) * 65536 + 65535 - (peak & 65535)
        }
      }
      return { textures: textures!,histogram: counts,bestAddress,bytes }
    },signal)
  } catch (error) {
    if (textures) destroyScoreTextures(textures)
    signal?.throwIfAborted()
    throw error
  } finally {
    for (const buffer of buffers) buffer.destroy()
  }
}

export async function readComputedScore(device: GPUDevice, textures: ScoreTextures, bankSize: number, site: number, x: number, y: number) {
  const buffer = device.createBuffer({ label: 'Selected score readback',size: 4,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  try {
    await checkedGpu(device,() => {
      const encoder = device.createCommandEncoder({ label: 'Inspect selected GPU score' })
      encoder.copyTextureToBuffer({
        texture: textures.banks[Math.floor(site / bankSize)],origin: [x,y,site % bankSize],
      },{ buffer },[1,1,1])
      device.queue.submit([encoder.finish()])
    },'Reading the selected GPU score')
    return await readBuffer(buffer,data => new DataView(data).getUint16(0,true))
  } finally {
    buffer.destroy()
  }
}
