import { getPipelineCompileQueue, type DeferredComputePipeline } from './pipeline-queue.ts'
import bc7EncodeWgsl from './shaders/bc7-encode.wgsl?raw'

const LEGACY_GPU_CACHE_DB_NAME = 'earth-connect-gpu-cache'
let legacyGpuCachePurged = false
let legacyGpuCachePurging = false

export function purgeLegacyGpuCacheOnce() {
  if (legacyGpuCachePurged || legacyGpuCachePurging || typeof indexedDB === 'undefined') return
  legacyGpuCachePurging = true
  try {
    const req = indexedDB.deleteDatabase(LEGACY_GPU_CACHE_DB_NAME)
    req.onsuccess = () => {
      legacyGpuCachePurged = true
      legacyGpuCachePurging = false
    }
    req.onerror = () => {
      legacyGpuCachePurging = false
    }
    req.onblocked = () => {
      legacyGpuCachePurging = false
    }
  } catch {
    legacyGpuCachePurging = false
  }
}

type Bc7MipLevel = {
  mip: number
  width: number
  height: number
  blocksX: number
  blocksY: number
  rowStrideBlocks: number
  bytesPerRow: number
  offsetBlocks: number
  offsetBytes: number
  byteLength: number
}

type Bc7MipLayout = {
  width: number
  height: number
  mipCount: number
  totalBlocks: number
  totalBytes: number
  mips: Bc7MipLevel[]
}

const bc7Pipelines = new WeakMap<GPUDevice, DeferredComputePipeline>()

/** Queue async compilation of the BC7 Mode 6 compute encoder so first tile upload never stalls. */
export function warmBc7Pipeline(device: GPUDevice): DeferredComputePipeline {
  purgeLegacyGpuCacheOnce()
  let handle = bc7Pipelines.get(device)
  if (!handle) {
    handle = getPipelineCompileQueue(device).registerCompute(15, () => ({
      label: 'BC7 Mode 6 compute pipeline',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'BC7 Mode 6 compute encoder', code: bc7EncodeWgsl }),
        entryPoint: 'encodeBc7Mode6',
      },
    }))
    bc7Pipelines.set(device, handle)
  }
  return handle
}

function getBc7Pipeline(device: GPUDevice): GPUComputePipeline {
  return warmBc7Pipeline(device).get()
}

export function computeBc7Layout(width: number, height: number, mipCount: number): Bc7MipLayout {
  const mips: Bc7MipLevel[] = []
  let offsetBlocks = 0
  for (let mip = 0; mip < mipCount; mip++) {
    const mw = Math.max(1, width >> mip)
    const mh = Math.max(1, height >> mip)
    const blocksX = Math.ceil(mw / 4)
    const blocksY = Math.ceil(mh / 4)
    // Align row stride to 16 BC7 blocks (256 bytes) so copyBufferToTexture works for every mip.
    const rowStrideBlocks = Math.ceil(blocksX / 16) * 16
    const bytesPerRow = rowStrideBlocks * 16
    const mipBlocks = rowStrideBlocks * blocksY
    mips.push({
      mip,
      width: mw,
      height: mh,
      blocksX,
      blocksY,
      rowStrideBlocks,
      bytesPerRow,
      offsetBlocks,
      offsetBytes: offsetBlocks * 16,
      byteLength: mipBlocks * 16,
    })
    offsetBlocks += mipBlocks
  }
  return {
    width,
    height,
    mipCount,
    totalBlocks: offsetBlocks,
    totalBytes: offsetBlocks * 16,
    mips,
  }
}

/**
 * Encode all mip levels of an `rgba8unorm-srgb` texture (created with `viewFormats: ['rgba8unorm']`)
 * into a new `bc7-rgba-unorm-srgb` texture completely on the GPU (~7–56 ms, zero CPU readback).
 */
export function transcodeSrgbTextureToBc7Gpu(device: GPUDevice, srcTexture: GPUTexture, label: string): GPUTexture {
  const width = srcTexture.width
  const height = srcTexture.height
  const mipCount = srcTexture.mipLevelCount
  const layout = computeBc7Layout(width, height, mipCount)
  const pipeline = getBc7Pipeline(device)

  const uniformStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment)
  const paramBuffer = device.createBuffer({
    label: `${label} BC7 encode params`,
    size: mipCount * uniformStride,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  const paramWords = new Uint32Array(paramBuffer.getMappedRange())
  const strideWords = uniformStride >> 2
  for (const m of layout.mips) {
    const o = m.mip * strideWords
    paramWords[o] = m.width
    paramWords[o + 1] = m.height
    paramWords[o + 2] = m.blocksX
    paramWords[o + 3] = m.blocksY
    paramWords[o + 4] = m.rowStrideBlocks
    paramWords[o + 5] = m.offsetBlocks
  }
  paramBuffer.unmap()

  const blockBuffer = device.createBuffer({
    label: `${label} BC7 encoded blocks`,
    size: layout.totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const bc7Texture = device.createTexture({
    label: `${label} (BC7)`,
    size: [width, height],
    mipLevelCount: mipCount,
    format: 'bc7-rgba-unorm-srgb',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  })

  try {
    const encoder = device.createCommandEncoder({ label: `${label} GPU BC7 transcode` })
    const pass = encoder.beginComputePass({ label: `${label} BC7 Mode 6` })
    pass.setPipeline(pipeline)
    for (const m of layout.mips) {
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: srcTexture.createView({ format: 'rgba8unorm', baseMipLevel: m.mip, mipLevelCount: 1 }),
          },
          { binding: 1, resource: { buffer: paramBuffer, offset: m.mip * uniformStride, size: 32 } },
          { binding: 2, resource: { buffer: blockBuffer } },
        ],
      })
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(Math.ceil(m.blocksX / 8), Math.ceil(m.blocksY / 8))
    }
    pass.end()

    for (const m of layout.mips) {
      encoder.copyBufferToTexture(
        { buffer: blockBuffer, offset: m.offsetBytes, bytesPerRow: m.bytesPerRow, rowsPerImage: m.blocksY },
        { texture: bc7Texture, mipLevel: m.mip },
        [m.blocksX * 4, m.blocksY * 4],
      )
    }

    device.queue.submit([encoder.finish()])
  } catch (error) {
    paramBuffer.destroy()
    blockBuffer.destroy()
    bc7Texture.destroy()
    throw error
  }

  void device.queue
    .onSubmittedWorkDone()
    .catch(() => undefined)
    .finally(() => {
      paramBuffer.destroy()
      blockBuffer.destroy()
    })

  return bc7Texture
}

/**
 * Reusable single-tile BC7 Mode 6 GPU transcoder for virtual tile atlases.
 * Zero per-tile GPU buffer or texture allocations (~0.3 ms per 512x512 3-mip tile).
 */
export class Bc7TileEncoder {
  private readonly pipeline: GPUComputePipeline
  private readonly layout: Bc7MipLayout
  private readonly paramBuffer: GPUBuffer
  private readonly blockBuffer: GPUBuffer
  private readonly scratchTexture: GPUTexture
  private readonly bindGroups: GPUBindGroup[]
  private destroyed = false

  constructor(device: GPUDevice, tileSize: number, mipCount: number, format: 'rgba8unorm-srgb' | 'rgba8unorm') {
    this.pipeline = getBc7Pipeline(device)
    this.layout = computeBc7Layout(tileSize, tileSize, mipCount)
    const uniformStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment)
    this.paramBuffer = device.createBuffer({
      label: 'BC7 tile encode params',
      size: mipCount * uniformStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
    const paramWords = new Uint32Array(this.paramBuffer.getMappedRange())
    const strideWords = uniformStride >> 2
    for (const m of this.layout.mips) {
      const o = m.mip * strideWords
      paramWords[o] = m.width
      paramWords[o + 1] = m.height
      paramWords[o + 2] = m.blocksX
      paramWords[o + 3] = m.blocksY
      paramWords[o + 4] = m.rowStrideBlocks
      paramWords[o + 5] = m.offsetBlocks
    }
    this.paramBuffer.unmap()

    this.blockBuffer = device.createBuffer({
      label: 'BC7 tile encode blocks',
      size: this.layout.totalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    this.scratchTexture = device.createTexture({
      label: 'BC7 tile encode scratch',
      size: [tileSize, tileSize],
      mipLevelCount: mipCount,
      format,
      viewFormats: format === 'rgba8unorm-srgb' ? ['rgba8unorm'] : [],
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.bindGroups = this.layout.mips.map(m =>
      device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: this.scratchTexture.createView({ format: 'rgba8unorm', baseMipLevel: m.mip, mipLevelCount: 1 }),
          },
          { binding: 1, resource: { buffer: this.paramBuffer, offset: m.mip * uniformStride, size: 32 } },
          { binding: 2, resource: { buffer: this.blockBuffer } },
        ],
      }),
    )
  }

  get scratch() {
    return this.scratchTexture
  }

  encodeIntoArrayLayer(encoder: GPUCommandEncoder, targetPool: GPUTexture, layerIndex: number) {
    if (this.destroyed) return
    const pass = encoder.beginComputePass({ label: 'BC7 virtual tile encode' })
    pass.setPipeline(this.pipeline)
    for (let i = 0; i < this.layout.mips.length; i++) {
      const m = this.layout.mips[i]
      pass.setBindGroup(0, this.bindGroups[i])
      pass.dispatchWorkgroups(Math.ceil(m.blocksX / 8), Math.ceil(m.blocksY / 8))
    }
    pass.end()
    for (const m of this.layout.mips) {
      encoder.copyBufferToTexture(
        { buffer: this.blockBuffer, offset: m.offsetBytes, bytesPerRow: m.bytesPerRow, rowsPerImage: m.blocksY },
        { texture: targetPool, mipLevel: m.mip, origin: [0, 0, layerIndex] },
        [m.blocksX * 4, m.blocksY * 4, 1],
      )
    }
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.paramBuffer.destroy()
    this.blockBuffer.destroy()
    this.scratchTexture.destroy()
  }
}
