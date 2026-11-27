import bc7EncodeWgsl from './shaders/bc7-encode.wgsl?raw'

const GPU_CACHE_DB_NAME = 'earth-connect-gpu-cache'
const GPU_CACHE_DB_VERSION = 1
const BC7_STORE = 'bc7'
const BC7_CACHE_MAGIC = 0x31474342 // 'BCG1' little-endian
const BC7_CACHE_MAX_ENTRIES = 16

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

type Bc7CacheRecord = {
  magic: number
  width: number
  height: number
  mipCount: number
  updatedAt: number
  data: ArrayBuffer
}

const bc7Pipelines = new WeakMap<GPUDevice, GPUComputePipeline>()

function getBc7Pipeline(device: GPUDevice): GPUComputePipeline {
  let pipeline = bc7Pipelines.get(device)
  if (!pipeline) {
    const module = device.createShaderModule({ label: 'BC7 Mode 6 compute encoder', code: bc7EncodeWgsl })
    pipeline = device.createComputePipeline({
      label: 'BC7 Mode 6 compute pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'encodeBc7Mode6' },
    })
    bc7Pipelines.set(device, pipeline)
  }
  return pipeline
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

let gpuCachePromise: Promise<IDBDatabase | undefined> | undefined

function openGpuCache(): Promise<IDBDatabase | undefined> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(undefined)
  if (!gpuCachePromise) {
    gpuCachePromise = new Promise<IDBDatabase | undefined>(resolve => {
      try {
        let blocked = false
        const request = indexedDB.open(GPU_CACHE_DB_NAME, GPU_CACHE_DB_VERSION)
        request.onupgradeneeded = () => {
          const db = request.result
          if (!db.objectStoreNames.contains(BC7_STORE)) db.createObjectStore(BC7_STORE)
        }
        request.onsuccess = () => {
          const db = request.result
          db.onversionchange = () => {
            db.close()
            gpuCachePromise = undefined
          }
          if (blocked) {
            db.close()
            gpuCachePromise = undefined
          } else {
            resolve(db)
          }
        }
        request.onerror = () => {
          gpuCachePromise = undefined
          resolve(undefined)
        }
        request.onblocked = () => {
          blocked = true
          gpuCachePromise = undefined
          resolve(undefined)
        }
      } catch {
        gpuCachePromise = undefined
        resolve(undefined)
      }
    })
  }
  return gpuCachePromise
}

export async function readCachedBc7(key: string): Promise<Bc7CacheRecord | undefined> {
  const db = await openGpuCache()
  if (!db) return undefined
  return new Promise(resolve => {
    try {
      const tx = db.transaction(BC7_STORE, 'readonly')
      const req = tx.objectStore(BC7_STORE).get(key)
      tx.oncomplete = () => {
        const rec = req.result as Partial<Bc7CacheRecord> | undefined
        if (
          rec &&
          rec.magic === BC7_CACHE_MAGIC &&
          typeof rec.width === 'number' &&
          typeof rec.height === 'number' &&
          typeof rec.mipCount === 'number' &&
          rec.data instanceof ArrayBuffer
        ) {
          const layout = computeBc7Layout(rec.width, rec.height, rec.mipCount)
          if (rec.data.byteLength === layout.totalBytes) {
            resolve(rec as Bc7CacheRecord)
            return
          }
        }
        resolve(undefined)
      }
      tx.onerror = tx.onabort = () => resolve(undefined)
    } catch {
      resolve(undefined)
    }
  })
}

async function writeCachedBc7(key: string, record: Bc7CacheRecord): Promise<void> {
  const db = await openGpuCache()
  if (!db) return
  await new Promise<void>(resolve => {
    try {
      const tx = db.transaction(BC7_STORE, 'readwrite')
      const store = tx.objectStore(BC7_STORE)
      store.put(record, key)
      // Bound cache cardinality by evicting the oldest entries beyond BC7_CACHE_MAX_ENTRIES.
      const entries: { key: IDBValidKey; updatedAt: number }[] = []
      const cursorReq = store.openCursor()
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result
        if (cursor) {
          const val = cursor.value as Partial<Bc7CacheRecord> | undefined
          entries.push({ key: cursor.key, updatedAt: typeof val?.updatedAt === 'number' ? val.updatedAt : 0 })
          cursor.continue()
        } else if (entries.length > BC7_CACHE_MAX_ENTRIES) {
          entries.sort((a, b) => a.updatedAt - b.updatedAt)
          for (let i = 0; i < entries.length - BC7_CACHE_MAX_ENTRIES; i++) {
            if (entries[i].key !== key) store.delete(entries[i].key)
          }
        }
      }
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve()
    } catch {
      resolve()
    }
  })
}

/**
 * Upload raw 256-byte-stride mipmapped BC7 bytes straight into a `bc7-rgba-unorm-srgb`
 * texture in bounded chunks so Dawn stays within a small staging slab.
 */
export async function uploadRawBc7Texture(
  device: GPUDevice,
  record: { width: number; height: number; mipCount: number; data: ArrayBuffer },
  label: string,
  signal: AbortSignal,
  maxChunkBytes = 32 * 1024 * 1024,
): Promise<GPUTexture> {
  signal.throwIfAborted()
  const layout = computeBc7Layout(record.width, record.height, record.mipCount)
  const texture = device.createTexture({
    label,
    size: [layout.width, layout.height],
    mipLevelCount: layout.mipCount,
    format: 'bc7-rgba-unorm-srgb',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  })
  try {
    const bytes = new Uint8Array(record.data)
    for (const m of layout.mips) {
      signal.throwIfAborted()
      const blockRowsPerChunk = Math.max(1, Math.floor(maxChunkBytes / m.bytesPerRow))
      if (m.blocksY <= blockRowsPerChunk) {
        device.queue.writeTexture(
          { texture, mipLevel: m.mip },
          bytes.subarray(m.offsetBytes, m.offsetBytes + m.byteLength),
          { bytesPerRow: m.bytesPerRow, rowsPerImage: m.blocksY },
          [m.blocksX * 4, m.blocksY * 4],
        )
        if (m.byteLength >= maxChunkBytes / 2) await device.queue.onSubmittedWorkDone()
      } else {
        for (let by = 0; by < m.blocksY; by += blockRowsPerChunk) {
          signal.throwIfAborted()
          const rows = Math.min(blockRowsPerChunk, m.blocksY - by)
          const start = m.offsetBytes + by * m.bytesPerRow
          device.queue.writeTexture(
            { texture, mipLevel: m.mip, origin: [0, by * 4, 0] },
            bytes.subarray(start, start + rows * m.bytesPerRow),
            { bytesPerRow: m.bytesPerRow, rowsPerImage: rows },
            [m.blocksX * 4, rows * 4],
          )
          await device.queue.onSubmittedWorkDone()
        }
      }
    }
    return texture
  } catch (error) {
    texture.destroy()
    throw error
  }
}

/**
 * Encode all mip levels of an `rgba8unorm-srgb` texture (created with `viewFormats: ['rgba8unorm']`)
 * into a new `bc7-rgba-unorm-srgb` texture completely on the GPU (~7–56 ms).
 * If `cacheKey` is given, asynchronously reads back the BC7 buffer and persists it to IndexedDB
 * without blocking return of the live GPU texture.
 */
export function transcodeSrgbTextureToBc7Gpu(
  device: GPUDevice,
  srcTexture: GPUTexture,
  label: string,
  cacheKey?: string,
): GPUTexture {
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

  const readbackBuffer = cacheKey
    ? device.createBuffer({
        label: `${label} BC7 IDB staging`,
        size: layout.totalBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : undefined

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

    // Copy directly GPU-to-GPU into the BC7 texture so first frame never stalls on CPU readback.
    for (const m of layout.mips) {
      encoder.copyBufferToTexture(
        { buffer: blockBuffer, offset: m.offsetBytes, bytesPerRow: m.bytesPerRow, rowsPerImage: m.blocksY },
        { texture: bc7Texture, mipLevel: m.mip },
        [m.blocksX * 4, m.blocksY * 4],
      )
    }

    if (readbackBuffer) {
      encoder.copyBufferToBuffer(blockBuffer, 0, readbackBuffer, 0, layout.totalBytes)
    }

    device.queue.submit([encoder.finish()])
  } catch (error) {
    paramBuffer.destroy()
    blockBuffer.destroy()
    readbackBuffer?.destroy()
    bc7Texture.destroy()
    throw error
  }

  // Release encode scratch once GPU finishes; persist raw BC7 to IDB in the background.
  if (readbackBuffer && cacheKey) {
    void readbackBuffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const copy = readbackBuffer.getMappedRange().slice(0)
        readbackBuffer.unmap()
        return writeCachedBc7(cacheKey, {
          magic: BC7_CACHE_MAGIC,
          width,
          height,
          mipCount,
          updatedAt: Date.now(),
          data: copy,
        })
      })
      .catch(() => undefined)
      .finally(() => {
        paramBuffer.destroy()
        blockBuffer.destroy()
        readbackBuffer.destroy()
      })
  } else {
    void device.queue
      .onSubmittedWorkDone()
      .catch(() => undefined)
      .finally(() => {
        paramBuffer.destroy()
        blockBuffer.destroy()
      })
  }

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
