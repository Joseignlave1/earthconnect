import { checkedGpu } from '../earth-score/gpu.ts'
import { purgeLegacyGpuCacheOnce, transcodeSrgbTextureToBc7Gpu } from './bc7-gpu.ts'
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
  bc7?: boolean
}

/**
 * Fetch an image, decode without color-space conversion, upload into a mipmapped
 * texture in bounded strips, and generate GPU mipmaps. When the device supports
 * `texture-compression-bc`, transcodes on the GPU into `bc7-rgba-unorm-srgb`
 * (~7–56 ms, 4× less VRAM, zero CPU readback or disk bloat).
 * Destroys the texture on any error or abort; callers own lifetime on success.
 */
export async function loadSrgbMipmappedTexture(
  device: GPUDevice,
  url: string,
  signal: AbortSignal,
  options: SrgbTextureLoadOptions,
): Promise<GPUTexture> {
  purgeLegacyGpuCacheOnce()
  const useBc7 = options.bc7 !== false && device.features.has('texture-compression-bc')

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
        viewFormats: useBc7 ? ['rgba8unorm'] : [],
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
    if (useBc7) {
      const bc7Texture = transcodeSrgbTextureToBc7Gpu(device, finished, options.label)
      // Release the temporary RGBA texture as soon as the GPU finishes reading it for BC7 encode.
      void device.queue
        .onSubmittedWorkDone()
        .catch(() => undefined)
        .finally(() => finished.destroy())
      texture = bc7Texture
      return bc7Texture
    }
    return finished
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}
