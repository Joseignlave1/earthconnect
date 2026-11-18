import { fullscreenVertex, shaderModule } from '../shader-test/gpu.ts'
import { checkedGpu } from '../earth-score/gpu.ts'

const mipShader = `${fullscreenVertex}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@fragment fn mipFragment(input: ScreenVertex) -> @location(0) vec4f {
  return textureSampleLevel(source,linearSampler,input.uv,0.);
}
`

export async function loadSkyMap(device: GPUDevice,signal: AbortSignal, skyMapUrl: string) {
  const response = await fetch(skyMapUrl,{ signal })
  if (!response.ok) throw new Error(`Sky map download failed (${response.status}).`)
  const bitmap = await createImageBitmap(await response.blob(),{ colorSpaceConversion: 'none' })
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const format = 'rgba8unorm-srgb'
    const module = await shaderModule(device,'Sky map linear mipmaps',mipShader)
    const pipeline = await device.createRenderPipelineAsync({
      label: 'Sky map linear mipmaps',layout: 'auto',vertex: { module,entryPoint: 'screenVertex' },
      fragment: { module,entryPoint: 'mipFragment',targets: [{ format }] },
    })
    signal.throwIfAborted()
    return await checkedGpu(device,() => {
      const atlas = texture = device.createTexture({
        label: 'NASA Deep Star Maps 2020',
        size: [bitmap.width,bitmap.height],
        mipLevelCount: Math.floor(Math.log2(Math.max(bitmap.width,bitmap.height))) + 1,
        format,usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      device.queue.copyExternalImageToTexture({ source: bitmap },{ texture: atlas },[bitmap.width,bitmap.height])
      const sampler = device.createSampler({ minFilter: 'linear',magFilter: 'linear' })
      const encoder = device.createCommandEncoder({ label: 'Build sky map mipmaps' })
      for (let level = 1; level < atlas.mipLevelCount; level++) {
        const source = atlas.createView({ baseMipLevel: level - 1,mipLevelCount: 1 })
        const view = atlas.createView({ baseMipLevel: level,mipLevelCount: 1 })
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
      return atlas
    },'Uploading sky map')
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}
