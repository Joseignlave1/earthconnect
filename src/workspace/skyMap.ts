import { loadSrgbMipmappedTexture, raceAbort } from '../shader-test/gpu.ts'

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
  const atlas = await loadSrgbMipmappedTexture(device, skyMapUrl, cache.controller.signal, {
    label: 'NASA Deep Star Maps 2020',
    mipLabel: 'Build sky map mipmaps',
    downloadErrorMessage: 'Sky map download failed.',
    validateGpu: true,
    createStepLabel: 'Creating sky map',
    uploadStepLabel: 'Uploading sky map',
  })
  cache.created.add(atlas)
  return atlas
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
