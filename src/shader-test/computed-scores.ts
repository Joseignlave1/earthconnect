import type { EarthScore } from '../earth-score/index.ts'
import { checkedGpu } from '../earth-score/gpu.ts'
import {
  computedScoreLayout,
  maxBatchLayers,
  scoreMipCount,
  tileCells,
  type ScoreBatch as Batch,
} from './computed-score-layout.ts'
import { uploadBuffer } from './gpu.ts'
import computedScoresWgsl from './shaders/computed-scores.wgsl?raw'
import { MISSING, SIZE } from './world-grid.ts'

export { computedScoreLayout, scoreMipCount }

const bins = MISSING
const histogramShards = 16
const histogramBytes = bins * Uint32Array.BYTES_PER_ELEMENT
const groupWidth = 16
const groupHeight = 8

export type ScoreTextures = { lookup: GPUTexture; banks: GPUTexture[] }

export function createScoreTextures(
  device: GPUDevice,
  bankSize: number,
  bankCount: number,
  sites: number,
): ScoreTextures {
  return {
    lookup: device.createTexture({
      label: 'Sinusoidal tile lookup',
      size: [36, 18],
      format: 'r16uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }),
    banks: Array.from({ length: bankCount }, (_, bank) =>
      device.createTexture({
        label: `Native scores ${bank}`,
        size: [SIZE, SIZE, Math.max(1, Math.min(bankSize, sites - bank * bankSize))],
        mipLevelCount: scoreMipCount,
        format: 'r16uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      }),
    ),
  }
}

export function scoreTexturesMatch(map: ScoreTextures, bankSize: number, bankCount: number, sites: number) {
  return (
    map.banks.length === bankCount &&
    map.banks.every(
      (texture, bank) => texture.depthOrArrayLayers === Math.max(1, Math.min(bankSize, sites - bank * bankSize)),
    )
  )
}

export function destroyScoreTextures(map: ScoreTextures) {
  map.lookup.destroy()
  for (const texture of map.banks) texture.destroy()
}

export function writeScoreLookup(device: GPUDevice, map: ScoreTextures, lookup: Uint16Array) {
  device.queue.writeTexture({ texture: map.lookup }, lookup, { bytesPerRow: 72 }, [36, 18])
}

type Pipelines = { base: GPUComputePipeline; mip: GPUComputePipeline; summarize: GPUComputePipeline }
const pipelines = new WeakMap<GPUDevice, Promise<Pipelines>>()

type Scratch = {
  pack: GPUBuffer
  histogram: GPUBuffer
  peaks: GPUBuffer
  summary: GPUBuffer
  packBytes: number
  peakBytes: number
}
const scratchPools = new WeakMap<GPUDevice, Scratch[]>()

function destroyScratch(scratch: Scratch) {
  scratch.pack.destroy()
  scratch.histogram.destroy()
  scratch.peaks.destroy()
  scratch.summary.destroy()
}

function getScratchPool(device: GPUDevice) {
  let pool = scratchPools.get(device)
  if (!pool) {
    pool = []
    scratchPools.set(device, pool)
    void device.lost.then(() => {
      for (const item of pool!.splice(0)) destroyScratch(item)
    })
  }
  return pool
}

function acquireScratch(device: GPUDevice, packBytes: number, peakBytes: number): Scratch {
  const pool = getScratchPool(device)
  const index = pool.findIndex(item => item.packBytes >= packBytes && item.peakBytes >= peakBytes)
  if (index >= 0) return pool.splice(index, 1)[0]
  const storage = GPUBufferUsage.STORAGE
  return {
    pack: device.createBuffer({
      label: 'Score tile copy rows',
      size: packBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    }),
    histogram: device.createBuffer({
      label: 'Score histogram shards',
      size: histogramBytes * histogramShards,
      usage: storage | GPUBufferUsage.COPY_DST,
    }),
    peaks: device.createBuffer({
      label: 'Tile peak addresses',
      size: peakBytes,
      usage: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
    summary: device.createBuffer({
      label: 'Score histogram',
      size: histogramBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    }),
    packBytes,
    peakBytes,
  }
}

function scorePipelines(device: GPUDevice) {
  const cached = pipelines.get(device)
  if (cached) return cached
  const pending = checkedGpu(
    device,
    () => {
      const module = device.createShaderModule({ label: 'Native score tiles and max LODs', code: computedScoresWgsl })
      const make = (entryPoint: string, constants: Record<string, number>) =>
        device.createComputePipelineAsync({
          label: `Native scores: ${entryPoint}`,
          layout: 'auto',
          compute: { module, entryPoint, constants },
        })
      return [
        make('base', {
          MISSING,
          SIZE,
          TILE_CELLS: tileCells,
          GROUP_WIDTH: groupWidth,
          GROUP_HEIGHT: groupHeight,
          HISTOGRAM_SHARDS: histogramShards,
        }),
        make('mip', { MISSING, GROUP_WIDTH: groupWidth, GROUP_HEIGHT: groupHeight }),
        make('summarize', { MISSING, HISTOGRAM_SHARDS: histogramShards }),
      ] as const
    },
    'Preparing the GPU score pipelines',
  ).then(async ([base, mip, summarize]) => ({
    base: await base,
    mip: await mip,
    summarize: await summarize,
  }))
  pipelines.set(device, pending)
  void pending.catch(() => {
    pipelines.delete(device)
  })
  return pending
}

export function warmComputedScorePipelines(device: GPUDevice) {
  void scorePipelines(device).catch(() => {})
}

function rowBytes(width: number) {
  return Math.ceil((width * 2) / 256) * 256
}

type ComputedMap = {
  textures: ScoreTextures
  histogram: Uint32Array<ArrayBuffer>
  bestAddress: number | null
  bytes: number
}

async function readBuffer<T>(buffer: GPUBuffer, read: (data: ArrayBuffer) => T, signal?: AbortSignal) {
  const cancel = () => buffer.destroy()
  signal?.addEventListener('abort', cancel, { once: true })
  try {
    signal?.throwIfAborted()
    await buffer.mapAsync(GPUMapMode.READ)
    signal?.throwIfAborted()
    return read(buffer.getMappedRange())
  } catch (error) {
    signal?.throwIfAborted()
    throw error
  } finally {
    signal?.removeEventListener('abort', cancel)
    if (buffer.mapState === 'mapped') buffer.unmap()
  }
}

export async function prepareComputedScores(
  device: GPUDevice,
  earth: Pick<EarthScore, 'chunks' | 'layout'>,
  bankSize: number,
  bankCount: number,
  signal?: AbortSignal,
  target?: ScoreTextures,
): Promise<ComputedMap> {
  signal?.throwIfAborted()
  const { lookup, batches, bytes } = computedScoreLayout(earth, bankSize, bankCount)
  const { base, mip, summarize } = await scorePipelines(device)
  signal?.throwIfAborted()
  const ephemeral: GPUBuffer[] = []
  let scratch: Scratch | undefined
  let textures: ScoreTextures | undefined
  let ownedTextures = false
  let scratchReusable = false
  try {
    const errorFilters = ['internal', 'out-of-memory', 'validation'] as const
    for (const filter of errorFilters) device.pushErrorScope(filter)
    let staging: GPUBuffer
    let gpuErrors: Promise<(GPUError | null)[]>
    try {
      const sites = earth.layout.sites.length
      const peakBytes = sites * 2 * 4
      const lod0RowBytes = rowBytes(SIZE)
      const lod0LayerBytes = lod0RowBytes * SIZE
      const lod1Width = SIZE >> 1
      const lod1LayerBytes = rowBytes(lod1Width) * lod1Width
      const packBytes = Math.min(
        device.limits.maxStorageBufferBindingSize,
        Math.max(Math.min(sites, maxBatchLayers) * lod0LayerBytes, Math.min(sites, bankSize) * lod1LayerBytes),
      )
      const lod0Capacity = Math.max(1, Math.floor(packBytes / lod0LayerBytes))
      const reusable = target && scoreTexturesMatch(target, bankSize, bankCount, sites)
      const map = (textures = reusable ? target : createScoreTextures(device, bankSize, bankCount, sites))
      ownedTextures = !reusable
      scratch = acquireScratch(device, packBytes, peakBytes)
      const { pack, histogram, peaks, summary } = scratch
      const readback = device.createBuffer({
        label: 'Score statistics readback',
        size: histogramBytes + peakBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      ephemeral.push(readback)
      const stride = device.limits.minUniformBufferOffsetAlignment / 4
      const offsets = new Uint32Array(stride * batches.length)
      // Group base-level batches into super-batches that share non-overlapping
      // slices of `pack`, so multiple chunks run concurrently in one pass.
      const superBatches: { index: number; batch: Batch; packLayer: number }[][] = []
      let group: { index: number; batch: Batch; packLayer: number }[] = []
      let usedLayers = 0
      for (const [index, batch] of batches.entries()) {
        if (usedLayers > 0 && usedLayers + batch.layers > lod0Capacity) {
          superBatches.push(group)
          group = []
          usedLayers = 0
        }
        offsets.set([batch.firstSite, batch.chunkTile, usedLayers], index * stride)
        group.push({ index, batch, packLayer: usedLayers })
        usedLayers += batch.layers
      }
      if (group.length) superBatches.push(group)
      const params = uploadBuffer(device, 'Score tile batch offsets', offsets, GPUBufferUsage.UNIFORM)
      ephemeral.push(params)
      const encoder = device.createCommandEncoder({ label: 'Publish Earth Score map' })
      encoder.clearBuffer(histogram, 0, histogramBytes * histogramShards)
      encoder.clearBuffer(peaks, 0, peakBytes)
      const baseLayout = base.getBindGroupLayout(0)
      const mipLayout = mip.getBindGroupLayout(0)
      const lod0X = Math.ceil(SIZE / (groupWidth * 2)),
        lod0Y = Math.ceil(SIZE / groupHeight)
      for (const items of superBatches) {
        const pass = encoder.beginComputePass()
        pass.setPipeline(base)
        for (const { index, batch } of items) {
          pass.setBindGroup(
            0,
            device.createBindGroup({
              layout: baseLayout,
              entries: [
                { binding: 0, resource: { buffer: batch.chunk.scores } },
                { binding: 1, resource: { buffer: pack } },
                { binding: 2, resource: { buffer: histogram } },
                { binding: 3, resource: { buffer: peaks } },
                { binding: 4, resource: { buffer: params, offset: index * stride * 4, size: 16 } },
              ],
            }),
          )
          pass.dispatchWorkgroups(lod0X, lod0Y, batch.layers)
        }
        pass.end()
        for (const { batch, packLayer } of items) {
          encoder.copyBufferToTexture(
            { buffer: pack, offset: packLayer * lod0LayerBytes, bytesPerRow: lod0RowBytes, rowsPerImage: SIZE },
            { texture: map.banks[batch.bank], mipLevel: 0, origin: [0, 0, batch.layer] },
            [SIZE, SIZE, batch.layers],
          )
        }
      }
      // Build higher mip levels bank-wide instead of per chunk batch.
      for (let lod = 1; lod < scoreMipCount; lod++) {
        const width = SIZE >> lod
        const mipRowBytes = rowBytes(width)
        const mipLayerBytes = mipRowBytes * width
        const maxLayers = Math.max(1, Math.floor(packBytes / mipLayerBytes))
        const x = Math.ceil(width / (groupWidth * 2)),
          y = Math.ceil(width / groupHeight)
        for (let bank = 0; bank < map.banks.length; bank++) {
          const bankLayers = Math.min(bankSize, sites - bank * bankSize)
          if (bankLayers <= 0) continue
          const texture = map.banks[bank]
          for (let first = 0; first < bankLayers; first += maxLayers) {
            const layers = Math.min(bankLayers - first, maxLayers)
            const previous = texture.createView({
              dimension: '2d-array',
              baseMipLevel: lod - 1,
              mipLevelCount: 1,
              baseArrayLayer: first,
              arrayLayerCount: layers,
            })
            const pass = encoder.beginComputePass()
            pass.setPipeline(mip)
            pass.setBindGroup(
              0,
              device.createBindGroup({
                layout: mipLayout,
                entries: [
                  { binding: 1, resource: { buffer: pack } },
                  { binding: 5, resource: previous },
                ],
              }),
            )
            pass.dispatchWorkgroups(x, y, layers)
            pass.end()
            encoder.copyBufferToTexture(
              { buffer: pack, bytesPerRow: mipRowBytes, rowsPerImage: width },
              { texture, mipLevel: lod, origin: [0, 0, first] },
              [width, width, layers],
            )
          }
        }
      }
      const summarizePass = encoder.beginComputePass()
      summarizePass.setPipeline(summarize)
      summarizePass.setBindGroup(
        0,
        device.createBindGroup({
          layout: summarize.getBindGroupLayout(0),
          entries: [
            { binding: 2, resource: { buffer: histogram } },
            { binding: 6, resource: { buffer: summary } },
          ],
        }),
      )
      summarizePass.dispatchWorkgroups(Math.ceil(bins / 256))
      summarizePass.end()
      encoder.copyBufferToBuffer(summary, 0, readback, 0, histogramBytes)
      encoder.copyBufferToBuffer(peaks, 0, readback, histogramBytes, peakBytes)
      writeScoreLookup(device, map, lookup)
      device.queue.submit([encoder.finish()])
      staging = readback
    } finally {
      gpuErrors = Promise.all(errorFilters.map(() => device.popErrorScope()))
      void gpuErrors.catch(() => {})
    }
    const readback = readBuffer(
      staging,
      data => {
        const counts = new Uint32Array(data, 0, bins).slice()
        const peaks = new Uint32Array(data, histogramBytes)
        let bestCode = 0,
          bestAddress: number | null = null
        for (const [half, peak] of peaks.entries()) {
          if (!peak) continue
          const code = (peak >>> 16) - 1
          // Iterating in address order preserves the first maximum on ties.
          if (bestAddress === null || code > bestCode) {
            bestCode = code
            bestAddress = (half >>> 1) * tileCells + (half % 2) * 65536 + 65535 - (peak & 65535)
          }
        }
        return { textures: textures!, histogram: counts, bestAddress, bytes }
      },
      signal,
    )
    const [errors, result] = await Promise.all([gpuErrors, readback])
    const gpuError = errors.find(Boolean)
    if (gpuError) throw new Error(`Building the GPU score map: ${gpuError.message}`)
    scratchReusable = true
    return result
  } catch (error) {
    if (textures && ownedTextures) destroyScoreTextures(textures)
    signal?.throwIfAborted()
    throw error
  } finally {
    for (const buffer of ephemeral) buffer.destroy()
    if (scratch) {
      if (scratchReusable) getScratchPool(device).push(scratch)
      else destroyScratch(scratch)
    }
  }
}

export async function readComputedScore(
  device: GPUDevice,
  textures: ScoreTextures,
  bankSize: number,
  site: number,
  x: number,
  y: number,
) {
  const buffer = device.createBuffer({
    label: 'Selected score readback',
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  try {
    await checkedGpu(
      device,
      () => {
        const encoder = device.createCommandEncoder({ label: 'Inspect selected GPU score' })
        encoder.copyTextureToBuffer(
          {
            texture: textures.banks[Math.floor(site / bankSize)],
            origin: [x, y, site % bankSize],
          },
          { buffer },
          [1, 1, 1],
        )
        device.queue.submit([encoder.finish()])
      },
      'Reading the selected GPU score',
    )
    return await readBuffer(buffer, data => new DataView(data).getUint16(0, true))
  } finally {
    buffer.destroy()
  }
}
