import { fullscreenVertex, shaderModule, uploadBuffer } from '../shader-test/gpu.ts'
import { add, scale } from '../workspace/solarCamera.ts'
import type { Vec3 } from '../workspace/solarSystem.ts'
import ridgeSource from './olympus.wgsl?raw'
import terrainUrl from './assets/olympus-elevation.bin?url'
import orthoUrl from './assets/olympus-ortho.webp?url'
import detailUrl from './assets/olympus-ortho-detail.webp?url'
import lightingUrl from './assets/olympus-lighting.webp?url'
import slabUrl from './assets/olympus-slab.webp?url'
import screeUrl from './assets/olympus-scree.webp?url'
import regionUrl from './assets/olympus-region.bin?url'
import ctxUrl from './assets/olympus-ctx.webp?url'
import earthUrl from './assets/earth-blue-marble.jpg?url'
import './style.css'

const stage = document.querySelector<HTMLElement>('.hero')!
const canvas = stage.querySelector<HTMLCanvasElement>('canvas')!
const status = stage.querySelector<HTMLElement>('.status')!
const motion = matchMedia('(prefers-reduced-motion: reduce)')
const uniform = new Float32Array(52)

async function imageAsset(url: string) {
  const response = await fetch(url)
  if (!response.ok) throw new Error('A planetary image could not load.')
  return createImageBitmap(await response.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
}

async function terrainAsset(url: string) {
  const response = await fetch(url)
  if (!response.ok || !response.body) throw new Error('The Olympus Mons elevation data could not load.')
  const buffer = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
  const header = new DataView(buffer)
  const width = header.getUint32(0, true)
  const height = header.getUint32(4, true)
  const x = header.getFloat32(8, true)
  const z = header.getFloat32(12, true)
  const spacing = header.getFloat32(16, true)
  const base = header.getFloat32(20, true)
  const bytes = new Uint8Array(buffer, 24)
  const count = width * height
  const hasSunlight = bytes.length === count * 3
  if (!count || (!hasSunlight && bytes.length !== count * 2)) throw new Error('Invalid Olympus Mons elevation data.')
  const packed = new Uint8Array(count * 4)
  for (let y = 0; y < height; y++) {
    let elevation = 0
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const zigzag = bytes[i] | bytes[i + count] << 8
      elevation += (zigzag >>> 1) ^ -(zigzag & 1)
      packed[i * 4] = elevation & 255
      packed[i * 4 + 1] = elevation >> 8
      packed[i * 4 + 2] = hasSunlight ? bytes[i + count * 2] : 255
      packed[i * 4 + 3] = 255
    }
  }
  return { width, height, x, z, spacing, base, packed }
}

type Elevation = Awaited<ReturnType<typeof terrainAsset>>

function mesh({ width, height, x, z, spacing }: Elevation) {
  function axis(length: number, start: number, lower: number, upper: number) {
    const extent = (length - 1) * spacing
    const coordinates: number[] = []
    for (let d = 0; d < extent;) {
      coordinates.push(start + d)
      const value = start + d
      // Put the densest grid along the visible section of the escarpment.
      const distance = Math.max(0, lower - value, value - upper)
      d += Math.min(2.2, .17 + distance * .125)
    }
    return [...coordinates, start + extent]
  }
  const u = axis(width, x, -35, 44)
  const v = axis(height, z, -28, 53)
  const columns = u.length - 1
  const rows = v.length - 1
  const points = new Float32Array((columns + 1) * (rows + 1) * 2)
  for (let y = 0; y <= rows; y++) {
    for (let x = 0; x <= columns; x++) {
      points.set([u[x], v[y]], (y * (columns + 1) + x) * 2)
    }
  }
  const indices = new Uint32Array(columns * rows * 6)
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      const a = y * (columns + 1) + x
      const b = a + columns + 1
      indices.set([a, b, a + 1, b, b + 1, a + 1], (y * columns + x) * 6)
    }
  }
  return { points, indices }
}

const mapKilometres = .0010087288906343
const detailBounds = [900, 650, 2150, 1700].map(value => value * mapKilometres)

function frameParameters(width: number, height: number, time: number, dem: Elevation, region: Elevation) {
  const narrow = width / height < .94
  // A high, oblique view looking up the western escarpment.
  const yaw = narrow ? .125 : .335
  const pitch = narrow ? -.18 : -.160
  const roll = -.070
  const origin: Vec3 = [-29, 19, 44]
  const horizontal: Vec3 = [Math.cos(yaw), 0, Math.sin(yaw)]
  const vertical: Vec3 = [-Math.sin(yaw) * Math.sin(pitch), Math.cos(pitch), Math.cos(yaw) * Math.sin(pitch)]
  const right = add(scale(horizontal, Math.cos(roll)), scale(vertical, Math.sin(roll)))
  const up = add(scale(vertical, Math.cos(roll)), scale(horizontal, -Math.sin(roll)))
  const forward: Vec3 = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)]
  const eye = add(origin, scale(right, .12 * Math.sin(time * .10)))
  const tanY = narrow ? .52 : .38
  const tanX = tanY * width / height
  const earth = add(forward, add(scale(right, (narrow ? .61 : .78) * tanX), scale(up, (narrow ? .46 : .76) * tanY)))
  uniform.set([...eye, 0, ...right, 0, ...up, 0, ...forward, 0, tanX, tanY, time, height,
    dem.x, dem.z, dem.spacing, dem.base, (dem.width - 1) * dem.spacing, (dem.height - 1) * dem.spacing, 0, 0,
    ...detailBounds,
    region.x, region.z, region.spacing, region.base,
    (region.width - 1) * region.spacing, (region.height - 1) * region.spacing, 0, 0,
    ...earth, .020,
    -36.19, 0, -48.06, 0,
    -.445, .067, -.893, 0])
}

async function createScene() {
  const context = canvas.getContext('webgpu')
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter || !context) throw new Error('WebGPU is unavailable.')
  const device = await adapter.requestDevice()
  const format = navigator.gpu.getPreferredCanvasFormat()
  const [dem, region, ortho, detail, ctx, baked, slab, scree, earthImage, module, mipShader] = await Promise.all([
    terrainAsset(terrainUrl), terrainAsset(regionUrl), imageAsset(orthoUrl), imageAsset(detailUrl),
    imageAsset(ctxUrl), imageAsset(lightingUrl),
    imageAsset(slabUrl), imageAsset(screeUrl), imageAsset(earthUrl),
    shaderModule(device, 'Olympus Mons scarp', ridgeSource),
    shaderModule(device, 'Terrain texture mipmaps', `${fullscreenVertex}
      @group(0) @binding(0) var image: texture_2d<f32>;
      @group(0) @binding(1) var mipSampler: sampler;
      @fragment fn downsample(v: ScreenVertex) -> @location(0) vec4f {
        return textureSampleLevel(image, mipSampler, v.uv, 0.);
      }`),
  ])
  const mipSampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' })
  const mipPipeline = await device.createRenderPipelineAsync({
    layout: 'auto', vertex: { module: mipShader, entryPoint: 'screenVertex' },
    fragment: { module: mipShader, entryPoint: 'downsample', targets: [{ format: 'rgba8unorm' }] },
  })
  const mipCommands = device.createCommandEncoder({ label: 'Build Olympus terrain mipmaps' })
  const texture = (label: string, image: ImageBitmap) => {
    const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(image.width, image.height)))
    const result = device.createTexture({
      label, size: [image.width, image.height], format: 'rgba8unorm', mipLevelCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.copyExternalImageToTexture({ source: image }, { texture: result }, [image.width, image.height])
    image.close()
    for (let i = 1; i < mipLevelCount; i++) {
      const source = result.createView({ baseMipLevel: i - 1, mipLevelCount: 1 })
      const target = result.createView({ baseMipLevel: i, mipLevelCount: 1 })
      const bind = device.createBindGroup({
        layout: mipPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: source }, { binding: 1, resource: mipSampler }],
      })
      const pass = mipCommands.beginRenderPass({
        colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(mipPipeline)
      pass.setBindGroup(0, bind)
      pass.draw(3)
      pass.end()
    }
    return result
  }
  const orthoTexture = texture('HiRISE RED orthoimage', ortho)
  const detailTexture = texture('HiRISE RED high-resolution ortho crop', detail)
  const ctxTexture = texture('Regional CTX imagery', ctx)
  const lightingTexture = texture('Baked HiRISE normals, sunlight and ambient occlusion', baked)
  const slabTexture = texture('Layered fractured rock detail', slab)
  const screeTexture = texture('Fine rock rubble detail', scree)
  const earthTexture = texture('NASA Blue Marble', earthImage)
  const earthSampler = device.createSampler({
    addressModeU: 'repeat', minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear',
  })
  device.queue.submit([mipCommands.finish()])
  const heightTexture = (data: Elevation, label: string) => {
    const result = device.createTexture({
      label, size: [data.width, data.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture: result }, data.packed, { bytesPerRow: data.width * 4 }, [data.width, data.height])
    return result
  }
  const elevationTexture = heightTexture(dem, 'HiRISE stereo elevation')
  const regionalTexture = heightTexture(region, 'Regional HRSC/MOLA elevation')
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8 })
  const rockSampler = device.createSampler({
    minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear',
    maxAnisotropy: 8, addressModeU: 'repeat', addressModeV: 'repeat',
  })
  const stages = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
  const fragment = GPUShaderStage.FRAGMENT
  const bindings = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: stages, buffer: {} },
    { binding: 1, visibility: fragment, texture: {} },
    { binding: 2, visibility: stages, sampler: {} },
    { binding: 3, visibility: stages, texture: {} },
    { binding: 4, visibility: fragment, texture: {} },
    { binding: 5, visibility: stages, texture: {} },
    { binding: 6, visibility: fragment, texture: {} },
    { binding: 7, visibility: fragment, texture: {} },
    { binding: 8, visibility: fragment, texture: {} },
    { binding: 9, visibility: fragment, texture: {} },
    { binding: 10, visibility: fragment, sampler: {} },
    { binding: 11, visibility: fragment, texture: {} },
    { binding: 12, visibility: fragment, sampler: {} },
  ] })
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindings] })
  const overlayBlend: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
  const pipeline = (kind: 'sky' | 'ground' | 'station' | 'signal') => device.createRenderPipelineAsync({
    label: `Olympus hero ${kind}`, layout,
    vertex: {
      module, entryPoint: `${kind}Vertex`,
      buffers: kind === 'ground' ? [{ arrayStride: 8, attributes: [{ format: 'float32x2', shaderLocation: 0, offset: 0 }] }]
        : [],
    },
    fragment: {
      module, entryPoint: `${kind}Fragment`,
      targets: [{ format, blend: kind === 'signal' ? { color: overlayBlend, alpha: overlayBlend } : undefined }],
    },
    depthStencil: { format: 'depth32float', depthWriteEnabled: kind !== 'signal', depthCompare: kind === 'sky' || kind === 'signal' ? 'always' : 'greater' },
  })
  const [sky, ground, station, signal] = await Promise.all([
    pipeline('sky'), pipeline('ground'), pipeline('station'), pipeline('signal'),
  ])
  const params = device.createBuffer({ size: uniform.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const group = device.createBindGroup({ layout: bindings, entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: orthoTexture.createView() },
    { binding: 2, resource: sampler }, { binding: 3, resource: regionalTexture.createView() },
    { binding: 4, resource: ctxTexture.createView() },
    { binding: 5, resource: elevationTexture.createView() },
    { binding: 6, resource: detailTexture.createView() },
    { binding: 7, resource: lightingTexture.createView() },
    { binding: 8, resource: slabTexture.createView() },
    { binding: 9, resource: screeTexture.createView() },
    { binding: 10, resource: rockSampler }, { binding: 11, resource: earthTexture.createView() },
    { binding: 12, resource: earthSampler },
  ] })
  const { points, indices } = mesh(region)
  const vertices = uploadBuffer(device, 'Olympus scarp geometry', points, GPUBufferUsage.VERTEX)
  const triangles = uploadBuffer(device, 'Olympus scarp triangles', indices, GPUBufferUsage.INDEX)
  context.configure({ device, format })
  let depth: GPUTexture | undefined
  let request = 0
  let last = -Infinity
  let draws = 0
  let destroyed = false
  const started = performance.now()
  const event = new AbortController()

  function submit(now: number) {
    const width = canvas.clientWidth
    const height = canvas.clientHeight
    if (!width || !height) return
    const ratio = Math.min(devicePixelRatio || 1, 1.5)
    const pixelWidth = Math.round(width * ratio)
    const pixelHeight = Math.round(height * ratio)
    if (!depth || canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      depth?.destroy()
      canvas.width = pixelWidth
      canvas.height = pixelHeight
      depth = device.createTexture({ size: [pixelWidth, pixelHeight], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT })
    }
    frameParameters(width, height, motion.matches ? 0 : (now - started) / 1000, dem, region)
    device.queue.writeBuffer(params, 0, uniform)
    const commands = device.createCommandEncoder()
    const pass = commands.beginRenderPass({
      label: 'Draw 3D scarp and dust',
      colorAttachments: [{ view: context!.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 0 },
    })
    pass.setPipeline(sky)
    pass.setBindGroup(0, group)
    pass.draw(3)
    pass.setPipeline(ground)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(triangles, 'uint32')
    pass.drawIndexed(indices.length)
    pass.setPipeline(station)
    pass.draw(36, 5)
    pass.setPipeline(signal)
    pass.draw(3)
    pass.end()
    device.queue.submit([commands.finish()])
    canvas.dataset.draws = String(++draws)
    stage.toggleAttribute('data-ready', true)
    status.hidden = true
  }
  function draw(now: number) {
    request = 0
    if (destroyed) return
    if (motion.matches || now - last >= 31) {
      last = now
      submit(now)
    }
    if (!motion.matches) request = requestAnimationFrame(draw)
  }
  function refresh() {
    last = -Infinity
    if (!request) request = requestAnimationFrame(draw)
  }
  const observer = new ResizeObserver(refresh)
  observer.observe(stage)
  motion.addEventListener('change', refresh, { signal: event.signal })
  device.addEventListener('uncapturederror', error => fail(new Error(error.error.message)), { signal: event.signal })
  void device.lost.then(() => {
    destroyed = true
    observer.disconnect()
    event.abort()
    cancelAnimationFrame(request)
    depth?.destroy()
    params.destroy()
    vertices.destroy()
    triangles.destroy()
    elevationTexture.destroy()
    orthoTexture.destroy()
    detailTexture.destroy()
    ctxTexture.destroy()
    lightingTexture.destroy()
    slabTexture.destroy()
    screeTexture.destroy()
    earthTexture.destroy()
    regionalTexture.destroy()
    stage.removeAttribute('data-ready')
    void createScene().catch(fail)
  })
  refresh()
}

function fail(error: unknown) {
  console.error(error)
  status.hidden = false
  status.textContent = 'The Olympus Mons scene could not load.'
}

void createScene().catch(fail)
