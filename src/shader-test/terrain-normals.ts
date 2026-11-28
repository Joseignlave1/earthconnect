import { joinWgsl, shaderModule } from './gpu.ts'
import fullscreenWgsl from './shaders/fullscreen.wgsl?raw'
import terrainCommonWgsl from './shaders/terrain-common.wgsl?raw'
import terrainNormalsWgsl from './shaders/terrain-normals.wgsl?raw'

const bakeNormalsShader = joinWgsl(fullscreenWgsl, terrainCommonWgsl, terrainNormalsWgsl)

type DeviceNormalCache = {
  pipeline?: Promise<GPURenderPipeline>
  textures: WeakMap<GPUTexture, Promise<GPUTexture>>
  created: GPUTexture[]
}

const normalCache = new WeakMap<GPUDevice, DeviceNormalCache>()

function getNormalCache(device: GPUDevice): DeviceNormalCache {
  let entry = normalCache.get(device)
  if (!entry) {
    const created: GPUTexture[] = []
    entry = { textures: new WeakMap(), created }
    normalCache.set(device, entry)
    void device.lost.then(() => {
      for (const tex of created) tex.destroy()
      created.length = 0
      normalCache.delete(device)
    })
  }
  return entry
}

function getBakeNormalsPipeline(device: GPUDevice): Promise<GPURenderPipeline> {
  const cache = getNormalCache(device)
  if (!cache.pipeline) {
    const pending = (async () => {
      const module = await shaderModule(device, 'Bake terrain normals WGSL', bakeNormalsShader)
      const bindLayout = device.createBindGroupLayout({
        label: 'Bake terrain normals layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        ],
      })
      return device.createRenderPipelineAsync({
        label: 'Bake Earth terrain normal pyramid',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
        vertex: { module, entryPoint: 'screenVertex' },
        fragment: { module, entryPoint: 'bakeNormals', targets: [{ format: 'rgba8unorm' }] },
      })
    })()
    cache.pipeline = pending
    void pending.catch(() => {
      if (cache.pipeline === pending) cache.pipeline = undefined
    })
  }
  return cache.pipeline
}

export function getSharedTerrainNormals(device: GPUDevice, terrain: GPUTexture): Promise<GPUTexture> {
  const cache = getNormalCache(device)
  let pending = cache.textures.get(terrain)
  if (!pending) {
    pending = (async () => {
      const pipeline = await getBakeNormalsPipeline(device)
      const texture = device.createTexture({
        label: 'Baked Earth terrain normals and ridge AO',
        size: [terrain.width, terrain.height],
        mipLevelCount: terrain.mipLevelCount,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      try {
        const encoder = device.createCommandEncoder({ label: 'Bake Earth terrain normal pyramid' })
        const fullView = terrain.createView()
        const layout = pipeline.getBindGroupLayout(0)
        for (let mip = 0; mip < terrain.mipLevelCount; mip++) {
          const group = device.createBindGroup({
            layout,
            entries: [
              { binding: 0, resource: terrain.createView({ baseMipLevel: mip, mipLevelCount: 1 }) },
              { binding: 1, resource: fullView },
            ],
          })
          const pass = encoder.beginRenderPass({
            label: `Bake terrain normals mip ${mip}`,
            colorAttachments: [
              {
                view: texture.createView({ baseMipLevel: mip, mipLevelCount: 1 }),
                loadOp: 'clear',
                storeOp: 'store',
              },
            ],
          })
          pass.setPipeline(pipeline)
          pass.setBindGroup(0, group)
          pass.draw(3)
          pass.end()
        }
        device.queue.submit([encoder.finish()])
        cache.created.push(texture)
        return texture
      } catch (error) {
        texture.destroy()
        throw error
      }
    })()
    cache.textures.set(terrain, pending)
    void pending.catch(() => {
      if (cache.textures.get(terrain) === pending) cache.textures.delete(terrain)
    })
  }
  return pending
}
