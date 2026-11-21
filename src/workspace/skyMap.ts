import { fullscreenVertex, shaderModule } from '../shader-test/gpu.ts'
import { checkedGpu } from '../earth-score/gpu.ts'

const mipShader = `${fullscreenVertex}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@fragment fn mipFragment(input: ScreenVertex) -> @location(0) vec4f {
  return textureSampleLevel(source,linearSampler,input.uv,0.);
}
`

type DeviceSkyCache = {
  controller: AbortController
  pipeline?: Promise<GPURenderPipeline>
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
  const response = await fetch(skyMapUrl,{ signal })
  if (!response.ok) throw new Error(`Sky map download failed (${response.status}).`)
  const bitmap = await createImageBitmap(await response.blob(),{ colorSpaceConversion: 'none' })
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const format = 'rgba8unorm-srgb'
    cache.pipeline ??= shaderModule(device,'Sky map linear mipmaps',mipShader).then(module =>
      device.createRenderPipelineAsync({
        label: 'Sky map linear mipmaps',layout: 'auto',vertex: { module,entryPoint: 'screenVertex' },
        fragment: { module,entryPoint: 'mipFragment',targets: [{ format }] },
      }))
    const pipeline = await cache.pipeline
    signal.throwIfAborted()
    const atlas = await checkedGpu(device,() => {
      const created = texture = device.createTexture({
        label: 'NASA Deep Star Maps 2020',
        size: [bitmap.width,bitmap.height],
        mipLevelCount: Math.floor(Math.log2(Math.max(bitmap.width,bitmap.height))) + 1,
        format,usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      device.queue.copyExternalImageToTexture({ source: bitmap },{ texture: created },[bitmap.width,bitmap.height])
      const sampler = device.createSampler({ minFilter: 'linear',magFilter: 'linear' })
      const encoder = device.createCommandEncoder({ label: 'Build sky map mipmaps' })
      for (let level = 1; level < created.mipLevelCount; level++) {
        const source = created.createView({ baseMipLevel: level - 1,mipLevelCount: 1 })
        const view = created.createView({ baseMipLevel: level,mipLevelCount: 1 })
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),entries: [
          { binding: 0,resource: source },{ binding: 1,resource: sampler },
        ] })
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view,loadOp: 'clear',storeOp: 'store' }] })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0,group)
        pass.draw(3)
        pass.end()
      }
      device.queue.submit([encoder.finish()])
      return created
    },'Uploading sky map')
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
  return new Promise<GPUTexture>((resolve, reject) => {
    const abort = () => reject(signal.reason)
    signal.addEventListener('abort', abort, { once: true })
    if (signal.aborted) return abort()
    void pending.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort))
  })
}
