import { contourDetail } from '../shader-test/contours.ts'
import { copyExternalImageChunked, generateMipmaps, joinWgsl, raceAbort, shaderModule, uploadBuffer, writeTextureChunkedAsync } from '../shader-test/gpu.ts'
import colorWgsl from '../shader-test/shaders/color.wgsl?raw'
import contoursWgsl from '../shader-test/shaders/contours.wgsl?raw'
import { elevationPyramid, packRg8Levels, type Terrain } from '../shader-test/terrain.ts'
import { loadPreparedTerrain, type PreparedTerrain } from '../shader-test/terrain-loader.ts'
import type { TerrainLoader } from './earthWorkspaceResources.ts'
import { buildPlanetTerrain, PLANET_TERRAIN_SETTINGS as settings, type PlanetTerrain, type TerrainBody } from './planetTerrainMesh.ts'
import { PLANET_TILE_PAIRS, PlanetTileCache, type PlanetTilePackLoader } from './planetTiles.ts'
import planetAnnotationWgsl from './shaders/planet-annotation.wgsl?raw'
import planetTerrainWgsl from './shaders/planet-terrain.wgsl?raw'
import type { WorkspaceAssets } from './workspaceAssets.ts'

export { buildPlanetTerrain, type PlanetTerrain, type TerrainBody }

const surfaceShader = joinWgsl(colorWgsl, contoursWgsl, planetTerrainWgsl)
const annotationShader = planetAnnotationWgsl

type SharedBodyTerrain = {
  mesh: PlanetTerrain
  vertices: GPUBuffer
  indices: GPUBuffer
  heights: GPUTexture
  sampler: GPUSampler
}

type GpuTerrain = {
  shared: SharedBodyTerrain
  albedo: GPUTexture
  albedoSampler: GPUSampler
  parameters: Float32Array<ArrayBuffer>
  uniform: GPUBuffer
  bindGroup: GPUBindGroup
  loupeParameters: Float32Array<ArrayBuffer>
  loupeUniform: GPUBuffer
  loupeBindGroup: GPUBindGroup
  annotationUniform: GPUBuffer
  annotationBindGroup: GPUBindGroup
  smallAnnotationParameters: Float32Array<ArrayBuffer>
  smallAnnotationUniform: GPUBuffer
  smallAnnotationBindGroup: GPUBindGroup
}

type PlanetPipelines = {
  layout: GPUBindGroupLayout
  annotationLayout: GPUBindGroupLayout
  surface: GPURenderPipeline
  surfaceSmall: GPURenderPipeline
  annotation: GPURenderPipeline
  annotationCoverage: GPURenderPipeline
  annotationSmall: GPURenderPipeline
  annotationSmallCoverage: GPURenderPipeline
}

type DevicePlanetCache = {
  controller: AbortController
  pipelines?: Promise<PlanetPipelines>
  indices?: GPUBuffer
  albedoSampler?: GPUSampler
  fallbackAlbedo?: GPUTexture
  planetTiles?: PlanetTileCache
  bodies: WeakMap<Terrain, Promise<SharedBodyTerrain>>
  albedos: Map<string, Promise<GPUTexture>>
  created: SharedBodyTerrain[]
  createdAlbedos: Set<GPUTexture>
}

const planetCache = new WeakMap<GPUDevice, DevicePlanetCache>()

function getPlanetCache(device: GPUDevice): DevicePlanetCache {
  let entry = planetCache.get(device)
  if (!entry) {
    const controller = new AbortController()
    const created: SharedBodyTerrain[] = []
    const createdAlbedos = new Set<GPUTexture>()
    entry = { controller, bodies: new WeakMap(), albedos: new Map(), created, createdAlbedos }
    planetCache.set(device, entry)
    void device.lost.then(() => {
      controller.abort()
      entry!.indices?.destroy()
      entry!.fallbackAlbedo?.destroy()
      entry!.planetTiles?.destroy()
      for (const item of created) {
        item.vertices.destroy()
        item.heights.destroy()
      }
      for (const texture of createdAlbedos) texture.destroy()
      created.length = 0
      createdAlbedos.clear()
      planetCache.delete(device)
    })
  }
  return entry
}

function getPlanetPipelines(device: GPUDevice): Promise<PlanetPipelines> {
  const cache = getPlanetCache(device)
  if (!cache.pipelines) {
    const vertexBuffers: GPUVertexBufferLayout[] = [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] },
    ]
    const bindLayout = device.createBindGroupLayout({
      label: 'Planetary DEM resources',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] })
    const pending = Promise.all([
      shaderModule(device, 'Planetary DEM terrain shader', surfaceShader).then(module =>
        Promise.all([
          device.createRenderPipelineAsync({
            label: 'Planetary DEM terrain',
            layout,
            vertex: { module, entryPoint: 'vertexMain', buffers: vertexBuffers },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }], constants: { PLANET_TILE_PAIRS } },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM small per-sample sphere',
            layout,
            vertex: { module, entryPoint: 'smallVertex' },
            fragment: { module, entryPoint: 'smallFragment', targets: [{ format: 'rgba16float' }], constants: { PLANET_TILE_PAIRS } },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM small per-sample annotations',
            layout,
            vertex: { module, entryPoint: 'smallVertex' },
            fragment: { module, entryPoint: 'smallAnnotationFragment', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM small per-sample annotations with coverage mask',
            layout,
            vertex: { module, entryPoint: 'smallVertex' },
            fragment: { module, entryPoint: 'smallAnnotationCoverageFragment', targets: [{ format: 'rgba16float' }, { format: 'rg8unorm' }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
        ])),
      shaderModule(device, 'Planetary DEM annotation shader', annotationShader).then(module => {
        const annotationLayout = device.createBindGroupLayout({
          label: 'Planetary DEM annotation resources',
          entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
        })
        const annotationPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [annotationLayout] })
        return Promise.all([
          device.createRenderPipelineAsync({
            label: 'Planetary DEM annotations',
            layout: annotationPipelineLayout,
            vertex: { module, entryPoint: 'vertexMain', buffers: vertexBuffers },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM annotations with coverage mask',
            layout: annotationPipelineLayout,
            vertex: { module, entryPoint: 'vertexMain', buffers: vertexBuffers },
            fragment: { module, entryPoint: 'fragmentCoverageMain', targets: [{ format: 'rgba16float' }, { format: 'rg8unorm' }] },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
        ]).then(([annotation, annotationCoverage]) => ({ annotationLayout, annotation, annotationCoverage }))
      }),
    ]).then(([[surface, surfaceSmall, annotationSmall, annotationSmallCoverage], { annotationLayout, annotation, annotationCoverage }]) => ({
      layout: bindLayout, annotationLayout, surface, surfaceSmall, annotation, annotationCoverage, annotationSmall, annotationSmallCoverage,
    }))
    cache.pipelines = pending
    void pending.catch(() => { if (cache.pipelines === pending) cache.pipelines = undefined })
  }
  return cache.pipelines
}

function getFallbackAlbedo(device: GPUDevice): GPUTexture {
  const cache = getPlanetCache(device)
  if (!cache.fallbackAlbedo) {
    const texture = device.createTexture({
      label: 'Planetary albedo fallback',
      size: [1, 1],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture }, new Uint8Array([255, 255, 255, 0]), {}, [1, 1])
    cache.fallbackAlbedo = texture
  }
  return cache.fallbackAlbedo
}

async function buildBodyAlbedo(device: GPUDevice, cache: DevicePlanetCache, body: TerrainBody, url: string): Promise<GPUTexture> {
  const { signal } = cache.controller
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`${body} albedo download failed (${response.status}).`)
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none' })
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const mipLevelCount = Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1
    texture = device.createTexture({
      label: `${body} global albedo`,
      size: [bitmap.width, bitmap.height],
      mipLevelCount,
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    await copyExternalImageChunked(device.queue, bitmap, texture, signal)
    generateMipmaps(device, texture, { addressModeU: 'repeat', label: `Build ${body} albedo mipmaps` })
    cache.createdAlbedos.add(texture)
    return texture
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}

function loadSharedBodyAlbedo(device: GPUDevice, body: TerrainBody, url: string, signal: AbortSignal): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  let pending = cache.albedos.get(url)
  if (!pending) {
    pending = buildBodyAlbedo(device, cache, body, url)
    cache.albedos.set(url, pending)
    void pending.catch(() => { if (cache.albedos.get(url) === pending) cache.albedos.delete(url) })
  }
  return raceAbort(pending, signal)
}

async function getSharedBodyTerrain(device: GPUDevice, body: TerrainBody, prepared: PreparedTerrain<'rg8unorm'>, signal: AbortSignal): Promise<SharedBodyTerrain> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  let pending = cache.bodies.get(prepared.terrain)
  if (!pending) {
    pending = (async () => {
      const { terrain } = prepared
      const levels = prepared.levels ?? packRg8Levels(elevationPyramid(terrain, true))
      const mesh = buildPlanetTerrain(body, terrain, false)
      const heights = device.createTexture({
        label: `${body} numeric terrain grid`,
        size: [terrain.width, terrain.height], mipLevelCount: levels.length,
        format: 'rg8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      try {
        for (const [mipLevel, { data, width, height }] of levels.entries()) {
          await writeTextureChunkedAsync(device.queue, heights, mipLevel, data, width, height, 2, cache.controller.signal)
        }
      } catch (error) {
        heights.destroy()
        throw error
      }
      const vertices = uploadBuffer(device, `${body} displaced terrain vertices`, mesh.positions, GPUBufferUsage.VERTEX)
      const indices = (cache.indices ??= uploadBuffer(device, 'Planetary displaced terrain indices', mesh.indices, GPUBufferUsage.INDEX))
      const sampler = device.createSampler({
        addressModeU: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      })
      const shared: SharedBodyTerrain = { mesh, heights, vertices, indices, sampler }
      cache.created.push(shared)
      // Mip levels and raw CPU elevation bytes are no longer needed once resident on the GPU.
      // Keeping the terrain object itself as the WeakMap key avoids re-uploading on the same device.
      prepared.levels = null
      if (prepared.terrain.pixels.byteLength) prepared.terrain.pixels = new Uint8ClampedArray(0)
      return shared
    })()
    cache.bodies.set(prepared.terrain, pending)
    void pending.catch(() => { if (cache.bodies.get(prepared.terrain) === pending) cache.bodies.delete(prepared.terrain) })
  }
  return raceAbort(pending, signal)
}

export class PlanetTerrainRenderer {
  private readonly bodies = new Map<TerrainBody, GpuTerrain>()
  private readonly device: GPUDevice
  private readonly bindLayout: GPUBindGroupLayout
  private readonly annotationLayout: GPUBindGroupLayout
  private readonly pipeline: GPURenderPipeline
  private readonly smallPipeline: GPURenderPipeline
  private readonly annotationPipeline: GPURenderPipeline
  private readonly annotationCoveragePipeline: GPURenderPipeline
  private readonly annotationSmallPipeline: GPURenderPipeline
  private readonly annotationSmallCoveragePipeline: GPURenderPipeline
  private readonly tiles: PlanetTileCache
  private readonly unsubscribeTiles: () => void
  private readonly annotationScratch = new Float32Array(20)

  private constructor(device: GPUDevice, pipelines: PlanetPipelines, tiles: PlanetTileCache, onTileUpdate?: () => void) {
    this.device = device
    this.bindLayout = pipelines.layout
    this.annotationLayout = pipelines.annotationLayout
    this.pipeline = pipelines.surface
    this.smallPipeline = pipelines.surfaceSmall
    this.annotationPipeline = pipelines.annotation
    this.annotationCoveragePipeline = pipelines.annotationCoverage
    this.annotationSmallPipeline = pipelines.annotationSmall
    this.annotationSmallCoveragePipeline = pipelines.annotationSmallCoverage
    this.tiles = tiles
    this.unsubscribeTiles = tiles.subscribe(
      () => onTileUpdate?.(),
      () => { this.rebindAll(); onTileUpdate?.() },
    )
  }

  static async create(
    device: GPUDevice,
    signal: AbortSignal,
    assetUrl: WorkspaceAssets['url'],
    loadTerrain?: TerrainLoader,
    onTileUpdate?: () => void,
    loadPlanetTilePack?: PlanetTilePackLoader,
  ) {
    const bodies: TerrainBody[] = ['moon', 'mars']
    const pipelines = await getPlanetPipelines(device)
    signal.throwIfAborted()
    const cache = getPlanetCache(device)
    const tiles = (cache.planetTiles ??= new PlanetTileCache(device))
    if (loadPlanetTilePack) tiles.attachPacks(loadPlanetTilePack)
    const renderer = new PlanetTerrainRenderer(device, pipelines, tiles, onTileUpdate)
    try {
      // Load Moon then Mars sequentially so we never overlap two 8K terrain workers
      // or two 8K RGBA ImageBitmaps (~128 MiB each) in RAM at the same instant.
      for (const body of bodies) {
        const terrainPath = `textures/${body}-terrain.bin`
        let prepared = loadTerrain
          ? await loadTerrain(terrainPath, 'rg8unorm', s => assetUrl(terrainPath, s), signal)
          : await loadPreparedTerrain(await assetUrl(terrainPath, signal), 'rg8unorm', signal)
        if (!prepared.terrain.pixels.byteLength && !cache.bodies.has(prepared.terrain)) {
          // Device recovered after an earlier upload freed CPU pixels; re-decode once.
          prepared = await loadPreparedTerrain(await assetUrl(terrainPath, signal), 'rg8unorm', signal)
        }
        signal.throwIfAborted()
        let albedo: GPUTexture
        try {
          const url = await assetUrl(`textures/${body}-albedo.webp`, signal)
          albedo = await loadSharedBodyAlbedo(device, body, url, signal)
        } catch {
          if (signal.aborted) throw signal.reason
          albedo = getFallbackAlbedo(device)
        }
        signal.throwIfAborted()
        const shared = await getSharedBodyTerrain(device, body, prepared, signal)
        signal.throwIfAborted()
        renderer.attach(body, shared, albedo)
      }
      return renderer
    } catch (error) {
      renderer.destroy()
      throw error
    }
  }

  private createBodyBindGroup(body: TerrainBody, uniform: GPUBuffer, shared: SharedBodyTerrain, albedo: GPUTexture, albedoSampler: GPUSampler) {
    return this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: shared.heights.createView() },
        { binding: 2, resource: shared.sampler },
        { binding: 3, resource: albedo.createView() },
        { binding: 4, resource: albedoSampler },
        { binding: 5, resource: this.tiles.pageTexture(body).createView() },
        { binding: 6, resource: this.tiles.poolTexture.createView({ dimension: '2d-array' }) },
        { binding: 7, resource: this.tiles.sampler },
      ],
    })
  }

  private rebindAll() {
    for (const [body, data] of this.bodies) {
      data.bindGroup = this.createBodyBindGroup(body, data.uniform, data.shared, data.albedo, data.albedoSampler)
      data.loupeBindGroup = this.createBodyBindGroup(body, data.loupeUniform, data.shared, data.albedo, data.albedoSampler)
      data.smallAnnotationBindGroup = this.createBodyBindGroup(body, data.smallAnnotationUniform, data.shared, data.albedo, data.albedoSampler)
    }
  }

  private attach(body: TerrainBody, shared: SharedBodyTerrain, albedo: GPUTexture) {
    const device = this.device
    const cache = getPlanetCache(device)
    const albedoSampler = (cache.albedoSampler ??= device.createSampler({
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    }))
    const parameters = new Float32Array(56)
    const { relief, interval, low, high } = settings[body]
    parameters.set([shared.mesh.radius * 1000, relief, interval, 1], 20)
    parameters.set(low, 24)
    parameters.set(high, 28)
    parameters[39] = body === 'mars' ? 1 : 0
    const uniform = device.createBuffer({
      label: `${body} terrain parameters`,
      size: parameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = this.createBodyBindGroup(body, uniform, shared, albedo, albedoSampler)
    const loupeParameters = new Float32Array(parameters)
    const loupeUniform = device.createBuffer({
      label: `${body} loupe terrain parameters`,
      size: loupeParameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const loupeBindGroup = this.createBodyBindGroup(body, loupeUniform, shared, albedo, albedoSampler)
    const annotationUniform = device.createBuffer({
      label: `${body} annotation camera`,
      size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const annotationBindGroup = device.createBindGroup({
      layout: this.annotationLayout,
      entries: [{ binding: 0, resource: { buffer: annotationUniform } }],
    })
    const smallAnnotationParameters = new Float32Array(parameters)
    const smallAnnotationUniform = device.createBuffer({
      label: `${body} small annotation parameters`,
      size: smallAnnotationParameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const smallAnnotationBindGroup = this.createBodyBindGroup(body, smallAnnotationUniform, shared, albedo, albedoSampler)
    this.bodies.set(body, {
      shared, albedo, albedoSampler, parameters, uniform, bindGroup,
      loupeParameters, loupeUniform, loupeBindGroup,
      annotationUniform, annotationBindGroup,
      smallAnnotationParameters, smallAnnotationUniform, smallAnnotationBindGroup,
    })
  }

  updateTiles(
    body: TerrainBody,
    eyeUnit: readonly [number, number, number],
    right: readonly [number, number, number],
    up: readonly [number, number, number],
    back: readonly [number, number, number],
    aspect: number,
    physicalMetresPerPixel: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    this.tiles.updateView(body, eyeUnit, right, up, back, aspect, data.shared.mesh.radius * 1000, physicalMetresPerPixel, loupe)
  }

  tileStatus() {
    const parts: string[] = []
    for (const body of ['moon', 'mars'] as const) {
      if (this.tiles.hasIndex(body)) {
        parts.push(`${body}:${this.tiles.residentCount(body)}/${this.tiles.indexedCount(body)}`)
      }
    }
    return parts.join(' ')
  }

  getTerrain(body: TerrainBody) {
    return this.bodies.get(body)?.shared.mesh
  }

  drawAnnotation(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, reliefFade = 1,
    includeCoverage = false, smallPad = 0, smallView?: Float32Array, eye?: readonly [number, number, number],
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    if (smallPad > 0 && smallView) {
      const { smallAnnotationParameters, smallAnnotationUniform, smallAnnotationBindGroup } = data
      smallAnnotationParameters.set(matrix, 0)
      smallAnnotationParameters[34] = smallPad
      smallAnnotationParameters[35] = reliefFade
      if (eye) smallAnnotationParameters.set(eye, 36)
      smallAnnotationParameters[39] = body === 'mars' ? 1 : 0
      smallAnnotationParameters.set(smallView, 40)
      this.device.queue.writeBuffer(smallAnnotationUniform, 0, smallAnnotationParameters)
      pass.setPipeline(includeCoverage ? this.annotationSmallCoveragePipeline : this.annotationSmallPipeline)
      pass.setBindGroup(0, smallAnnotationBindGroup)
      pass.draw(6)
      return
    }
    const { shared: { mesh, vertices, indices }, annotationUniform, annotationBindGroup } = data
    this.annotationScratch.set(matrix, 0)
    this.annotationScratch[16] = body === 'moon' ? 2 : 4
    this.annotationScratch[17] = reliefFade
    this.annotationScratch[18] = mesh.radius
    this.annotationScratch[19] = 0
    this.device.queue.writeBuffer(annotationUniform, 0, this.annotationScratch)
    pass.setPipeline(includeCoverage ? this.annotationCoveragePipeline : this.annotationPipeline)
    pass.setBindGroup(0, annotationBindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  draw(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, sunlight: readonly [number, number, number],
    eye: readonly [number, number, number], metresPerPixel: number, hdr: boolean, pixelRatio: number,
    smallPad = 0, reliefFade = 1, smallView?: Float32Array,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const { shared: { mesh, vertices, indices }, parameters, uniform, bindGroup } = data
    const [minor, fine] = contourDetail(metresPerPixel)
    parameters.set(matrix, 0)
    parameters.set(sunlight, 16)
    parameters[19] = Number(hdr)
    parameters[23] = pixelRatio
    parameters[27] = minor
    parameters[31] = fine
    parameters[32] = Number(this.tiles.ready(body))
    parameters[33] = 0
    parameters[34] = smallPad
    parameters[35] = reliefFade
    parameters.set(eye, 36)
    parameters[39] = body === 'mars' ? 1 : 0
    if (smallView) parameters.set(smallView, 40)
    this.device.queue.writeBuffer(uniform, 0, parameters)
    if (smallPad > 0) {
      pass.setPipeline(this.smallPipeline)
      pass.setBindGroup(0, bindGroup)
      pass.draw(6)
    } else {
      pass.setPipeline(this.pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setVertexBuffer(0, vertices)
      pass.setIndexBuffer(indices, 'uint32')
      pass.drawIndexed(mesh.indices.length)
    }
  }

  drawLoupe(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, sunlight: readonly [number, number, number],
    eye: readonly [number, number, number], metresPerPixel: number, hdr: boolean, pixelRatio: number,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const { shared: { mesh, vertices, indices }, loupeParameters, loupeUniform, loupeBindGroup } = data
    const [minor, fine] = contourDetail(metresPerPixel)
    loupeParameters.set(matrix, 0)
    loupeParameters.set(sunlight, 16)
    loupeParameters[19] = Number(hdr)
    loupeParameters[23] = pixelRatio
    loupeParameters[27] = minor
    loupeParameters[31] = fine
    loupeParameters[32] = Number(this.tiles.ready(body))
    loupeParameters[33] = 0
    loupeParameters[34] = 0
    loupeParameters[35] = 1
    loupeParameters.set(eye, 36)
    loupeParameters[39] = body === 'mars' ? 1 : 0
    this.device.queue.writeBuffer(loupeUniform, 0, loupeParameters)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, loupeBindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  destroy() {
    this.unsubscribeTiles()
    for (const { uniform, loupeUniform, annotationUniform, smallAnnotationUniform } of this.bodies.values()) {
      uniform.destroy()
      loupeUniform.destroy()
      annotationUniform.destroy()
      smallAnnotationUniform.destroy()
    }
    this.bodies.clear()
  }
}
