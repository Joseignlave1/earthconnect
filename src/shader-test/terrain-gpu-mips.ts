import { joinWgsl, writeTextureChunkedAsync } from './gpu.ts'
import fullscreenWgsl from './shaders/fullscreen.wgsl?raw'
import terrainMipsWgsl from './shaders/terrain-mips.wgsl?raw'
import type { Terrain } from './terrain.ts'

const terrainMipShader = joinWgsl(fullscreenWgsl, terrainMipsWgsl)

type TerrainMipPipelines = {
  unpackRg32: GPURenderPipeline
  downsampleRg32: GPURenderPipeline
  downsampleRg8: GPURenderPipeline
  reducePolesRg32: GPUComputePipeline
  reducePolesRg8: GPUComputePipeline
  closePolesRg32: GPURenderPipeline
  closePolesRg8: GPURenderPipeline
}

const pipelineCache = new WeakMap<GPUDevice, TerrainMipPipelines>()

function getTerrainMipPipelines(device: GPUDevice): TerrainMipPipelines {
  let cached = pipelineCache.get(device)
  if (!cached) {
    const module = device.createShaderModule({ label: 'Numeric terrain mipmap generator', code: terrainMipShader })
    const makeRender = (label: string, entryPoint: string, format: GPUTextureFormat) =>
      device.createRenderPipeline({
        label,
        layout: 'auto',
        vertex: { module, entryPoint: 'screenVertex' },
        fragment: { module, entryPoint, targets: [{ format }] },
      })
    cached = {
      unpackRg32: makeRender('Unpack terrain rg32float', 'unpackRg32Float', 'rg32float'),
      downsampleRg32: makeRender('Downsample terrain rg32float', 'downsampleRg32Float', 'rg32float'),
      downsampleRg8: makeRender('Downsample terrain rg8unorm', 'downsampleRg8Unorm', 'rg8unorm'),
      reducePolesRg32: device.createComputePipeline({
        label: 'Reduce terrain poles rg32float',
        layout: 'auto',
        compute: { module, entryPoint: 'reducePolesRg32Float' },
      }),
      reducePolesRg8: device.createComputePipeline({
        label: 'Reduce terrain poles rg8unorm',
        layout: 'auto',
        compute: { module, entryPoint: 'reducePolesRg8Unorm' },
      }),
      closePolesRg32: makeRender('Close terrain poles rg32float', 'closePoleRow', 'rg32float'),
      closePolesRg8: makeRender('Close terrain poles rg8unorm', 'closePoleRow', 'rg8unorm'),
    }
    pipelineCache.set(device, cached)
  }
  return cached
}

function poleAverages(terrain: Terrain): {
  northElev: number
  northCoast: number
  southElev: number
  southCoast: number
} {
  const { pixels, width, height } = terrain
  const stride = terrain.stride ?? 4
  const hasCoast = stride === 4
  const avgRow = (row: number) => {
    let elevSum = 0
    let coastSum = 0
    const rowOff = row * width
    for (let x = 0; x < width; x++) {
      const i = (rowOff + x) * stride
      elevSum += pixels[i] * 256 + pixels[i + 1] - 32768
      if (hasCoast) coastSum += (pixels[i + 2] - 128) * 0.25
    }
    return { elev: elevSum / width, coast: hasCoast ? coastSum / width : 0 }
  }
  const north = avgRow(0)
  const south = height > 1 ? avgRow(height - 1) : north
  return { northElev: north.elev, northCoast: north.coast, southElev: south.elev, southCoast: south.coast }
}

function encodeMipChain(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  texture: GPUTexture,
  width: number,
  height: number,
  mipLevelCount: number,
  downsamplePipeline: GPURenderPipeline,
  reducePolesPipeline: GPUComputePipeline,
  closePolesPipeline: GPURenderPipeline,
): GPUBuffer[] {
  const align = Math.max(256, device.limits.minUniformBufferOffsetAlignment)
  const mipSteps = Math.max(0, mipLevelCount - 1)
  const uniformBuf = device.createBuffer({
    label: 'Terrain mip params',
    size: Math.max(align, mipSteps * 2 * align),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const poleBuf = device.createBuffer({
    label: 'Terrain pole reduction',
    size: 16,
    usage: GPUBufferUsage.STORAGE,
  })
  const uniformData = new ArrayBuffer(Math.max(align, mipSteps * 2 * align))
  const uniformU32 = new Uint32Array(uniformData)

  const dims: { srcW: number; srcH: number; dstW: number; dstH: number }[] = []
  let w = width
  let h = height
  for (let mip = 1; mip < mipLevelCount; mip++) {
    const nextW = Math.max(1, w >> 1)
    const nextH = Math.max(1, h >> 1)
    dims.push({ srcW: w, srcH: h, dstW: nextW, dstH: nextH })
    const mipOff = ((mip - 1) * 2 * align) >> 2
    uniformU32[mipOff] = w
    uniformU32[mipOff + 1] = h
    uniformU32[mipOff + 2] = nextW
    uniformU32[mipOff + 3] = nextH
    const poleOff = (((mip - 1) * 2 + 1) * align) >> 2
    uniformU32[poleOff] = nextW
    uniformU32[poleOff + 1] = nextH
    w = nextW
    h = nextH
  }
  device.queue.writeBuffer(uniformBuf, 0, uniformData)

  for (let mip = 1; mip < mipLevelCount; mip++) {
    const { dstW, dstH } = dims[mip - 1]
    const mipOffset = (mip - 1) * 2 * align
    const poleOffset = ((mip - 1) * 2 + 1) * align
    const srcView = texture.createView({ baseMipLevel: mip - 1, mipLevelCount: 1 })
    const dstView = texture.createView({ baseMipLevel: mip, mipLevelCount: 1 })

    const dsGroup = device.createBindGroup({
      layout: downsamplePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: srcView },
        { binding: 3, resource: { buffer: uniformBuf, offset: mipOffset, size: 16 } },
      ],
    })
    const dsPass = encoder.beginRenderPass({
      colorAttachments: [{ view: dstView, loadOp: 'clear', storeOp: 'store' }],
    })
    dsPass.setPipeline(downsamplePipeline)
    dsPass.setBindGroup(0, dsGroup)
    dsPass.draw(3)
    dsPass.end()

    if (dstW > 1) {
      const reduceGroup = device.createBindGroup({
        layout: reducePolesPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 2, resource: dstView },
          { binding: 4, resource: { buffer: uniformBuf, offset: poleOffset, size: 16 } },
          { binding: 5, resource: { buffer: poleBuf } },
        ],
      })
      const cPass = encoder.beginComputePass()
      cPass.setPipeline(reducePolesPipeline)
      cPass.setBindGroup(0, reduceGroup)
      cPass.dispatchWorkgroups(dstH > 1 ? 2 : 1)
      cPass.end()

      const closeGroup = device.createBindGroup({
        layout: closePolesPipeline.getBindGroupLayout(0),
        entries: [{ binding: 6, resource: { buffer: poleBuf } }],
      })
      const pPass = encoder.beginRenderPass({
        colorAttachments: [{ view: dstView, loadOp: 'load', storeOp: 'store' }],
      })
      pPass.setPipeline(closePolesPipeline)
      pPass.setBindGroup(0, closeGroup)
      pPass.setScissorRect(0, 0, dstW, 1)
      pPass.draw(3)
      if (dstH > 1) {
        pPass.setScissorRect(0, dstH - 1, dstW, 1)
        pPass.draw(3)
      }
      pPass.end()
    }
  }

  return [uniformBuf, poleBuf]
}

/**
 * Upload packed stride-4 terrain level 0 (`rgba8uint`) and unpack + build the full
 * `rg32float` numeric elevation/coast mip pyramid on the GPU in a single command buffer.
 */
export async function buildRg32FloatTerrainTextureGpu(
  device: GPUDevice,
  terrain: Terrain,
  signal?: AbortSignal,
): Promise<GPUTexture> {
  const { width, height } = terrain
  const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1
  const pipelines = getTerrainMipPipelines(device)

  const packedTex = device.createTexture({
    label: 'Terrain packed level 0 staging',
    size: [width, height],
    format: 'rgba8uint',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  })
  const texture = device.createTexture({
    label: 'Numeric elevation and coast distance',
    size: [width, height],
    mipLevelCount,
    format: 'rg32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  })
  const tempBuffers: GPUBuffer[] = []
  try {
    const level0Bytes = new Uint8Array(
      terrain.pixels.buffer as ArrayBuffer,
      terrain.pixels.byteOffset,
      terrain.pixels.byteLength,
    )
    await writeTextureChunkedAsync(device.queue, packedTex, 0, level0Bytes, width, height, 4, signal)
    signal?.throwIfAborted()

    const { northElev, northCoast, southElev, southCoast } = poleAverages(terrain)
    const unpackBuf = device.createBuffer({
      label: 'Terrain unpack params',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    tempBuffers.push(unpackBuf)
    const unpackData = new ArrayBuffer(32)
    const u32 = new Uint32Array(unpackData)
    const f32 = new Float32Array(unpackData)
    u32[0] = width
    u32[1] = height
    f32[2] = northElev
    f32[3] = northCoast
    f32[4] = southElev
    f32[5] = southCoast
    device.queue.writeBuffer(unpackBuf, 0, unpackData)

    const encoder = device.createCommandEncoder({ label: 'Build rg32float terrain pyramid' })
    const unpackGroup = device.createBindGroup({
      layout: pipelines.unpackRg32.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: packedTex.createView() },
        { binding: 1, resource: { buffer: unpackBuf } },
      ],
    })
    const unpackPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' },
      ],
    })
    unpackPass.setPipeline(pipelines.unpackRg32)
    unpackPass.setBindGroup(0, unpackGroup)
    unpackPass.draw(3)
    unpackPass.end()

    tempBuffers.push(
      ...encodeMipChain(
        device,
        encoder,
        texture,
        width,
        height,
        mipLevelCount,
        pipelines.downsampleRg32,
        pipelines.reducePolesRg32,
        pipelines.closePolesRg32,
      ),
    )
    device.queue.submit([encoder.finish()])
    void device.queue.onSubmittedWorkDone().then(
      () => {
        packedTex.destroy()
        for (const buf of tempBuffers) buf.destroy()
      },
      () => {
        packedTex.destroy()
        for (const buf of tempBuffers) buf.destroy()
      },
    )
    return texture
  } catch (error) {
    packedTex.destroy()
    for (const buf of tempBuffers) buf.destroy()
    texture.destroy()
    throw error
  }
}

/**
 * Upload packed stride-2 polar-smoothed terrain level 0 directly to `rg8unorm` mip 0
 * and build all numeric uint16 elevation mips on the GPU in a single command buffer.
 */
export async function buildRg8UnormTerrainTextureGpu(
  device: GPUDevice,
  terrain: Terrain,
  label: string,
  signal?: AbortSignal,
): Promise<GPUTexture> {
  const { width, height } = terrain
  const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1
  const pipelines = getTerrainMipPipelines(device)

  const texture = device.createTexture({
    label,
    size: [width, height],
    mipLevelCount,
    format: 'rg8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  })
  const tempBuffers: GPUBuffer[] = []
  try {
    const level0Bytes = new Uint8Array(
      terrain.pixels.buffer as ArrayBuffer,
      terrain.pixels.byteOffset,
      terrain.pixels.byteLength,
    )
    await writeTextureChunkedAsync(device.queue, texture, 0, level0Bytes, width, height, 2, signal)
    signal?.throwIfAborted()

    const encoder = device.createCommandEncoder({ label: `Build ${label} pyramid` })
    tempBuffers.push(
      ...encodeMipChain(
        device,
        encoder,
        texture,
        width,
        height,
        mipLevelCount,
        pipelines.downsampleRg8,
        pipelines.reducePolesRg8,
        pipelines.closePolesRg8,
      ),
    )
    device.queue.submit([encoder.finish()])
    void device.queue.onSubmittedWorkDone().then(
      () => {
        for (const buf of tempBuffers) buf.destroy()
      },
      () => {
        for (const buf of tempBuffers) buf.destroy()
      },
    )
    return texture
  } catch (error) {
    for (const buf of tempBuffers) buf.destroy()
    texture.destroy()
    throw error
  }
}
