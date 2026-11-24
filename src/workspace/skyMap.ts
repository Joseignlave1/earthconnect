import { copyExternalImageChunked, generateMipmaps, raceAbort } from '../shader-test/gpu.ts'
import { checkedGpu } from '../earth-score/gpu.ts'

type DeviceSkyCache = {
  controller: AbortController
  textures: Map<string, Promise<GPUTexture>>
  created: Set<GPUTexture>
}

const skyCache = new WeakMap<GPUDevice, DeviceSkyCache>()

function getSkyCache(device: GPUDevice): DeviceSkyCache {
  let entry = skyCache.get(device)
  if (!entry) {
    const controller = new AbortController()
    const created = new Set<GPUTexture>()
    entry = { controller, textures: new Map(), created }
    skyCache.set(device, entry)
    void device.lost.then(() => {
      controller.abort()
      for (const texture of created) texture.destroy()
      created.clear()
      skyCache.delete(device)
    })
  }
  return entry
}

async function buildSkyMap(device: GPUDevice, cache: DeviceSkyCache, skyMapUrl: string): Promise<GPUTexture> {
  const { signal } = cache.controller
  const response = await fetch(skyMapUrl, { signal })
  if (!response.ok) throw new Error(`Sky map download failed (${response.status}).`)
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none' })
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const created = await checkedGpu(
      device,
      () => {
        texture = device.createTexture({
          label: 'NASA Deep Star Maps 2020',
          size: [bitmap.width, bitmap.height],
          mipLevelCount: Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1,
          format: 'rgba8unorm-srgb',
          usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        })
        return texture
      },
      'Creating sky map',
    )
    signal.throwIfAborted()
    await copyExternalImageChunked(device.queue, bitmap, created, signal)
    signal.throwIfAborted()
    const atlas = await checkedGpu(
      device,
      () => {
        generateMipmaps(device, created, { label: 'Build sky map mipmaps' })
        return created
      },
      'Uploading sky map',
    )
    if (signal.aborted) {
      atlas.destroy()
      signal.throwIfAborted()
    }
    cache.created.add(atlas)
    return atlas
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}

/**
 * Load and mipmap the star map once per (GPUDevice, URL). The returned texture
 * is shared across all scenes on the device and destroyed on device loss.
 */
export function loadSkyMap(device: GPUDevice, signal: AbortSignal, skyMapUrl: string): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getSkyCache(device)
  let pending = cache.textures.get(skyMapUrl)
  if (!pending) {
    pending = buildSkyMap(device, cache, skyMapUrl)
    cache.textures.set(skyMapUrl, pending)
    void pending.catch(() => {
      if (cache.textures.get(skyMapUrl) === pending) cache.textures.delete(skyMapUrl)
    })
  }
  return raceAbort(pending, signal)
}
