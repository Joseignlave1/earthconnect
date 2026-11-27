import { checkedGpu } from '../earth-score/gpu.ts'
import type { Bc7WorkerRequest, Bc7WorkerResponse } from './bc7-worker.ts'
import { createBc7Worker } from './bc7-worker-factory.ts'
import fullscreenWgsl from './shaders/fullscreen.wgsl?raw'
import mipBlitWgsl from './shaders/mip-blit.wgsl?raw'

export function joinWgsl(...parts: string[]) {
  return parts.join('\n')
}

// WebGPU owns resource lifetime explicitly; uploads never read numeric data
// through canvas or image color conversion.
export async function shaderModule(device: GPUDevice, label: string, code: string) {
  const module = device.createShaderModule({ label, code })
  const info = await module.getCompilationInfo()
  const errors = info.messages.filter(message => message.type === 'error')
  if (errors.length)
    throw new Error(
      `${label}: ${errors.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n')}`,
    )
  return module
}

export function uploadBuffer(
  device: GPUDevice,
  label: string,
  data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>,
  usage: GPUBufferUsageFlags,
) {
  const size = Math.ceil(data.byteLength / 4) * 4
  const buffer = device.createBuffer({ label, size, usage, mappedAtCreation: true })
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
  buffer.unmap()
  return buffer
}

export function raceAbort<T>(work: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return work
  if (signal.aborted) return Promise.reject(signal.reason)
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(signal.reason)
    signal.addEventListener('abort', abort, { once: true })
    void work.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort))
  })
}

function waitForQueue(queue: GPUQueue, signal?: AbortSignal) {
  return raceAbort(queue.onSubmittedWorkDone(), signal)
}

// Stream large texture mip levels in bounded strips, flushing the GPU queue
// between strips so Dawn reuses a single small IPC shared-memory slab instead
// of growing its MappedMemoryManager pool to tens/hundreds of MiB.
export async function writeTextureChunkedAsync(
  queue: GPUQueue,
  texture: GPUTexture,
  mipLevel: number,
  data: ArrayBufferView<ArrayBuffer>,
  width: number,
  height: number,
  bytesPerPixel: number,
  signal?: AbortSignal,
  maxChunkBytes = 32 * 1024 * 1024,
) {
  const bytesPerRow = width * bytesPerPixel
  const rowsPerChunk = Math.max(1, Math.floor(maxChunkBytes / bytesPerRow))
  if (height <= rowsPerChunk) {
    queue.writeTexture({ texture, mipLevel }, data, { bytesPerRow }, [width, height])
    if (data.byteLength >= maxChunkBytes / 2) await waitForQueue(queue, signal)
    return
  }
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  for (let y = 0; y < height; y += rowsPerChunk) {
    signal?.throwIfAborted()
    const rows = Math.min(rowsPerChunk, height - y)
    const slice = bytes.subarray(y * bytesPerRow, (y + rows) * bytesPerRow)
    queue.writeTexture({ texture, mipLevel, origin: [0, y, 0] }, slice, { bytesPerRow }, [width, rows])
    await waitForQueue(queue, signal)
  }
}

// Stream large ImageBitmaps in bounded row strips, flushing the GPU queue
// between strips so Dawn never retains a 128 MiB IPC shared-memory pool.
export async function copyExternalImageChunked(
  queue: GPUQueue,
  source: ImageBitmap,
  texture: GPUTexture,
  signal?: AbortSignal,
  maxChunkBytes = 32 * 1024 * 1024,
) {
  const bytesPerRow = source.width * 4
  const rowsPerChunk = Math.max(1, Math.floor(maxChunkBytes / bytesPerRow))
  if (source.height <= rowsPerChunk) {
    queue.copyExternalImageToTexture({ source }, { texture }, [source.width, source.height])
    if (source.width * source.height * 4 >= maxChunkBytes / 2) await waitForQueue(queue, signal)
    return
  }
  for (let y = 0; y < source.height; y += rowsPerChunk) {
    signal?.throwIfAborted()
    const rows = Math.min(rowsPerChunk, source.height - y)
    queue.copyExternalImageToTexture({ source, origin: [0, y] }, { texture, origin: [0, y, 0] }, [source.width, rows])
    await waitForQueue(queue, signal)
  }
}

const mipBlitShader = joinWgsl(fullscreenWgsl, mipBlitWgsl)

type DeviceMipCache = {
  module: GPUShaderModule
  pipelines: Map<GPUTextureFormat, GPURenderPipeline>
  samplers: Map<GPUAddressMode, GPUSampler>
}

const mipCaches = new WeakMap<GPUDevice, DeviceMipCache>()

function getMipResources(device: GPUDevice, format: GPUTextureFormat, addressModeU: GPUAddressMode) {
  let cache = mipCaches.get(device)
  if (!cache) {
    cache = {
      module: device.createShaderModule({ label: 'Bilinear mipmap generator', code: mipBlitShader }),
      pipelines: new Map(),
      samplers: new Map(),
    }
    mipCaches.set(device, cache)
  }
  let pipeline = cache.pipelines.get(format)
  if (!pipeline) {
    pipeline = device.createRenderPipeline({
      label: `Bilinear mipmaps (${format})`,
      layout: 'auto',
      vertex: { module: cache.module, entryPoint: 'screenVertex' },
      fragment: { module: cache.module, entryPoint: 'mipFragment', targets: [{ format }] },
    })
    cache.pipelines.set(format, pipeline)
  }
  let sampler = cache.samplers.get(addressModeU)
  if (!sampler) {
    sampler = device.createSampler({
      addressModeU,
      addressModeV: 'clamp-to-edge',
      minFilter: 'linear',
      magFilter: 'linear',
    })
    cache.samplers.set(addressModeU, sampler)
  }
  return { pipeline, sampler }
}

/**
 * Generate bilinear image mipmaps on the GPU for one or more 2D / 2D-array texture layers.
 * Numeric elevation/score textures retain their own averaging/max rules and never use this helper.
 */
export function generateMipmaps(
  device: GPUDevice,
  texture: GPUTexture,
  {
    layers = [0],
    mipLevelCount = texture.mipLevelCount,
    addressModeU = 'clamp-to-edge',
    label = 'Generate mipmaps',
    encoder,
  }: {
    layers?: readonly number[]
    mipLevelCount?: number
    addressModeU?: GPUAddressMode
    label?: string
    encoder?: GPUCommandEncoder
  } = {},
) {
  if (mipLevelCount <= 1) return
  const { pipeline, sampler } = getMipResources(device, texture.format, addressModeU)
  const ownEncoder = !encoder
  const cmd = encoder ?? device.createCommandEncoder({ label })
  const isArray = texture.depthOrArrayLayers > 1
  for (const layer of layers) {
    for (let mip = 1; mip < mipLevelCount; mip++) {
      const viewDesc = (baseMipLevel: number): GPUTextureViewDescriptor =>
        isArray
          ? { dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1, baseMipLevel, mipLevelCount: 1 }
          : { baseMipLevel, mipLevelCount: 1 }
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView(viewDesc(mip - 1)) },
          { binding: 1, resource: sampler },
        ],
      })
      const pass = cmd.beginRenderPass({
        colorAttachments: [{ view: texture.createView(viewDesc(mip)), loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, group)
      pass.draw(3)
      pass.end()
    }
  }
  if (ownEncoder) device.queue.submit([cmd.finish()])
}

export type SrgbTextureLoadOptions = {
  label: string
  mipLabel?: string
  downloadErrorMessage?: string
  maxWidth?: number
  addressModeU?: GPUAddressMode
  validateGpu?: boolean
  createStepLabel?: string
  uploadStepLabel?: string
}

/**
 * Fetch an image, decode without color-space conversion, upload into a mipmapped
 * `rgba8unorm-srgb` texture in bounded strips, and generate GPU mipmaps.
 * Destroys the texture on any error or abort; callers own lifetime on success.
 */
export async function loadSrgbMipmappedTexture(
  device: GPUDevice,
  url: string,
  signal: AbortSignal,
  options: SrgbTextureLoadOptions,
): Promise<GPUTexture> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    throw new Error(options.downloadErrorMessage ?? `${options.label} download failed (${response.status}).`)
  }
  const blob = await response.blob()
  const bitmap = await createImageBitmap(
    blob,
    options.maxWidth
      ? {
          colorSpaceConversion: 'none',
          resizeWidth: options.maxWidth,
          resizeHeight: options.maxWidth >> 1,
          resizeQuality: 'high',
        }
      : { colorSpaceConversion: 'none' },
  )
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const createTexture = () => {
      texture = device.createTexture({
        label: options.label,
        size: [bitmap.width, bitmap.height],
        mipLevelCount: Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1,
        format: 'rgba8unorm-srgb',
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      return texture
    }
    const created = options.validateGpu
      ? await checkedGpu(device, createTexture, options.createStepLabel ?? `Creating ${options.label}`)
      : createTexture()
    signal.throwIfAborted()
    await copyExternalImageChunked(device.queue, bitmap, created, signal)
    signal.throwIfAborted()
    const buildMips = () => {
      generateMipmaps(device, created, {
        label: options.mipLabel ?? `${options.label} mipmaps`,
        addressModeU: options.addressModeU,
      })
      return created
    }
    const finished = options.validateGpu
      ? await checkedGpu(device, buildMips, options.uploadStepLabel ?? `Uploading ${options.label}`)
      : buildMips()
    signal.throwIfAborted()
    return finished
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}

function totalBc7Bytes(width: number, height: number, mipCount: number): number {
  let total = 0
  for (let mip = 0; mip < mipCount; mip++) {
    const mw = Math.max(1, width >> mip)
    const mh = Math.max(1, height >> mip)
    total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * 16
  }
  return total
}

async function decodeBc7OnMainThread(
  url: string,
  signal: AbortSignal,
): Promise<{ width: number; height: number; mipCount: number; data: ArrayBuffer }> {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`BC7 texture HTTP ${response.status}`)
  const raw = await response.arrayBuffer()
  signal.throwIfAborted()
  if (raw.byteLength < 16) throw new Error('Invalid BC7 texture header.')
  const view = new DataView(raw)
  if (view.getUint32(0) !== 0x45434243) throw new Error('Invalid BC7 magic.')
  const width = view.getUint32(4, true)
  const height = view.getUint32(8, true)
  const mipCount = view.getUint32(12, true)
  const { decompressZstd } = await import('../earth-score/decode.ts')
  const decoded = await decompressZstd(raw.slice(16), totalBc7Bytes(width, height, mipCount), signal)
  return { width, height, mipCount, data: decoded.buffer }
}

function decodeBc7InWorker(
  url: string,
  signal: AbortSignal,
): Promise<{ width: number; height: number; mipCount: number; data: ArrayBuffer }> {
  signal.throwIfAborted()
  const resolved = typeof globalThis.location?.href === 'string' ? new URL(url, globalThis.location.href).href : url
  return new Promise((resolve, reject) => {
    let worker: Worker
    try {
      worker = createBc7Worker()
    } catch {
      void decodeBc7OnMainThread(resolved, signal).then(resolve, reject)
      return
    }
    let settled = false
    const cleanup = () => {
      signal.removeEventListener('abort', onAbort)
      worker.terminate()
    }
    const onAbort = () => {
      if (settled) return
      settled = true
      cleanup()
      reject(signal.reason)
    }
    const fallback = () => {
      if (settled) return
      settled = true
      cleanup()
      void decodeBc7OnMainThread(resolved, signal).then(resolve, reject)
    }
    signal.addEventListener('abort', onAbort, { once: true })
    worker.onmessage = ({ data }: MessageEvent<Bc7WorkerResponse>) => {
      if (settled) return
      settled = true
      cleanup()
      if (data.kind === 'error') reject(new Error(data.message))
      else resolve({ width: data.width, height: data.height, mipCount: data.mipCount, data: data.data })
    }
    worker.onerror = event => {
      event.preventDefault()
      fallback()
    }
    worker.onmessageerror = fallback
    try {
      worker.postMessage({ url: resolved } satisfies Bc7WorkerRequest)
    } catch {
      fallback()
    }
  })
}

/**
 * Fetch and Zstd-decompress a pre-mipmapped ECBC BC7 texture off the main thread,
 * then upload all mip levels directly into `bc7-rgba-unorm-srgb` (zero ImageBitmap
 * decode, zero GPU mipmap pass, 4× less VRAM).
 */
export async function loadBc7MipmappedTexture(
  device: GPUDevice,
  url: string,
  signal: AbortSignal,
  label: string,
  maxChunkBytes = 32 * 1024 * 1024,
): Promise<GPUTexture> {
  const { width, height, mipCount, data } = await decodeBc7InWorker(url, signal)
  signal.throwIfAborted()
  const texture = device.createTexture({
    label,
    size: [width, height],
    mipLevelCount: mipCount,
    format: 'bc7-rgba-unorm-srgb',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  })
  try {
    const bytes = new Uint8Array(data)
    let offset = 0
    for (let mip = 0; mip < mipCount; mip++) {
      signal.throwIfAborted()
      const mw = Math.max(1, width >> mip)
      const mh = Math.max(1, height >> mip)
      const blocksX = Math.ceil(mw / 4)
      const blocksY = Math.ceil(mh / 4)
      const bytesPerRow = blocksX * 16
      const mipBytes = bytesPerRow * blocksY
      const blockRowsPerChunk = Math.max(1, Math.floor(maxChunkBytes / bytesPerRow))
      if (blocksY <= blockRowsPerChunk) {
        device.queue.writeTexture(
          { texture, mipLevel: mip },
          bytes.subarray(offset, offset + mipBytes),
          { bytesPerRow, rowsPerImage: blocksY },
          [blocksX * 4, blocksY * 4],
        )
        if (mipBytes >= maxChunkBytes / 2) await waitForQueue(device.queue, signal)
      } else {
        for (let by = 0; by < blocksY; by += blockRowsPerChunk) {
          signal.throwIfAborted()
          const rows = Math.min(blockRowsPerChunk, blocksY - by)
          const start = offset + by * bytesPerRow
          device.queue.writeTexture(
            { texture, mipLevel: mip, origin: [0, by * 4, 0] },
            bytes.subarray(start, start + rows * bytesPerRow),
            { bytesPerRow, rowsPerImage: rows },
            [blocksX * 4, rows * 4],
          )
          await waitForQueue(device.queue, signal)
        }
      }
      offset += mipBytes
    }
    return texture
  } catch (error) {
    texture.destroy()
    throw error
  }
}
