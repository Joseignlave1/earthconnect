import { loadBc7MipmappedTexture, loadSrgbMipmappedTexture, raceAbort } from '../shader-test/gpu.ts'

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

async function buildSkyMap(
  device: GPUDevice,
  cache: DeviceSkyCache,
  skyMapUrl: string,
  isBc7: boolean,
): Promise<GPUTexture> {
  const atlas = isBc7
    ? await loadBc7MipmappedTexture(device, skyMapUrl, cache.controller.signal, 'NASA Deep Star Maps 2020 (BC7)')
    : await loadSrgbMipmappedTexture(device, skyMapUrl, cache.controller.signal, {
        label: 'NASA Deep Star Maps 2020',
        mipLabel: 'Build sky map mipmaps',
        downloadErrorMessage: 'Sky map download failed.',
      })
  cache.created.add(atlas)
  return atlas
}

/**
 * Load and mipmap the star map once per (GPUDevice, URL). The returned texture
 * is shared across all scenes on the device and destroyed on device loss.
 */
export function loadSkyMap(
  device: GPUDevice,
  signal: AbortSignal,
  skyMapUrl: string,
  isBc7 = false,
): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getSkyCache(device)
  const key = isBc7 ? `bc7:${skyMapUrl}` : skyMapUrl
  let pending = cache.textures.get(key)
  if (!pending) {
    pending = buildSkyMap(device, cache, skyMapUrl, isBc7)
    cache.textures.set(key, pending)
    void pending.catch(() => {
      if (cache.textures.get(key) === pending) cache.textures.delete(key)
    })
  }
  return raceAbort(pending, signal)
}
