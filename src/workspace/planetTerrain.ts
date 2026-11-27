import { contourDetail } from '../shader-test/contours.ts'
import {
  joinWgsl,
  loadSrgbMipmappedTexture,
  raceAbort,
  shaderModule,
  uploadBuffer,
  writeTextureChunkedAsync,
} from '../shader-test/gpu.ts'
import {
  createPatchMesh,
  CUBE_INSTANCE_FLOATS,
  MAX_PATCH_INSTANCES,
  PlanetCubeQuadtree,
} from '../shader-test/quadtreeLod.ts'
import colorWgsl from '../shader-test/shaders/color.wgsl?raw'
import contoursWgsl from '../shader-test/shaders/contours.wgsl?raw'
import quadWgsl from '../shader-test/shaders/quad.wgsl?raw'
import { elevationPyramid, packRg8Levels, type Terrain } from '../shader-test/terrain.ts'
import { loadPreparedTerrain, type PreparedTerrain } from '../shader-test/terrain-loader.ts'
import type { TerrainLoader } from './earthWorkspaceResources.ts'
import {
  buildPlanetPickingMesh,
  buildPlanetTerrain,
  PLANET_TERRAIN_SETTINGS as settings,
  type PlanetPickingMesh,
  type PlanetTerrain,
  type TerrainBody,
} from './planetTerrainMesh.ts'
import { VIRTUAL_TILE_GUTTER, VIRTUAL_TILE_INNER, VIRTUAL_TILE_SIZE } from '../shader-test/virtual-tiles.ts'
import {
  PLANET_TILE_BASE_COLS,
  PLANET_TILE_BASE_ROWS,
  PLANET_TILE_PAIRS,
  PlanetTileCache,
  type PlanetTilePackLoader,
} from './planetTiles.ts'
import planetAnnotationWgsl from './shaders/planet-annotation.wgsl?raw'
import planetCommonWgsl from './shaders/planet-common.wgsl?raw'
import planetTerrainWgsl from './shaders/planet-terrain.wgsl?raw'
import type { WorkspaceAssets } from './workspaceAssets.ts'

export { buildPlanetPickingMesh, buildPlanetTerrain, type PlanetPickingMesh, type PlanetTerrain, type TerrainBody }

const surfaceShader = joinWgsl(colorWgsl, contoursWgsl, quadWgsl, planetCommonWgsl, planetTerrainWgsl)
const annotationShader = joinWgsl(planetCommonWgsl, planetAnnotationWgsl)
const sharedPatchMesh = createPatchMesh()

type PlanetViewUniformParams = {
  matrix: Float32Array
  sunlight?: readonly [number, number, number]
  hdr?: boolean
  pixelRatio?: number
  minorDetail?: number
  fineDetail?: number
  tilesReady?: boolean
  reliefFade?: number
  eye?: readonly [number, number, number]
  isMars: boolean
  smallView?: Float32Array
}

function writePlanetViewUniform(out: Float32Array, p: PlanetViewUniformParams) {
  out.set(p.matrix, 0)
  if (p.sunlight) out.set(p.sunlight, 16)
  if (p.hdr !== undefined) out[19] = Number(p.hdr)
  if (p.pixelRatio !== undefined) out[23] = p.pixelRatio
  if (p.minorDetail !== undefined) out[27] = p.minorDetail
  if (p.fineDetail !== undefined) out[31] = p.fineDetail
  if (p.tilesReady !== undefined) out[32] = Number(p.tilesReady)
  out[33] = 0
  out[34] = 0
  if (p.reliefFade !== undefined) out[35] = p.reliefFade
  if (p.eye) out.set(p.eye, 36)
  out[39] = p.isMars ? 1 : 0
  if (p.smallView) out.set(p.smallView, 40)
}

type SharedBodyTerrain = {
  mesh: PlanetPickingMesh
  quadtree: PlanetCubeQuadtree
  patchVertices: GPUBuffer
  patchIndices: GPUBuffer
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
  patchInstances: Float32Array<ArrayBuffer>
  patchInstanceBuffer: GPUBuffer
  lastPatchMatrix: Float32Array
  lastPatchEye: [number, number, number]
  lastPatchWidth: number
  lastPatchHeight: number
  lastPatchReliefFade: number
  lastPatchCount: number
  loupePatchInstances: Float32Array<ArrayBuffer>
  loupePatchInstanceBuffer: GPUBuffer
}

type PlanetPipelines = {
  layout: GPUBindGroupLayout
  annotationLayout: GPUBindGroupLayout
  smallAnnotationLayout: GPUBindGroupLayout
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
  patchVertices?: GPUBuffer
  patchIndices?: GPUBuffer
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
      entry!.patchVertices?.destroy()
      entry!.patchIndices?.destroy()
      entry!.fallbackAlbedo?.destroy()
      entry!.planetTiles?.destroy()
      for (const item of created) {
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
      { arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] },
      {
        arrayStride: CUBE_INSTANCE_FLOATS * 4,
        stepMode: 'instance',
        attributes: [
          { shaderLocation: 1, format: 'float32x2', offset: 0 },
          { shaderLocation: 2, format: 'float32', offset: 8 },
          { shaderLocation: 3, format: 'uint32', offset: 12 },
          { shaderLocation: 4, format: 'float32', offset: 16 },
          { shaderLocation: 5, format: 'float32', offset: 20 },
        ],
      },
    ]
    const vf = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
    const bindLayout = device.createBindGroupLayout({
      label: 'Planetary DEM resources',
      entries: [
        { binding: 0, visibility: vf, buffer: { type: 'uniform' } },
        { binding: 1, visibility: vf, texture: { sampleType: 'float' } },
        { binding: 2, visibility: vf, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
        {
          binding: 6,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float', viewDimension: '2d-array' },
        },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })
    const smallAnnotationLayout = device.createBindGroupLayout({
      label: 'Planetary analytic annotation resources',
      entries: [{ binding: 0, visibility: vf, buffer: { type: 'uniform' } }],
    })
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] })
    const smallAnnotationPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [smallAnnotationLayout] })
    const surfaceConstants = {
      PLANET_TILE_PAIRS,
      VIRTUAL_TILE_SIZE,
      VIRTUAL_TILE_INNER,
      VIRTUAL_TILE_GUTTER,
      PLANET_TILE_BASE_COLS,
      PLANET_TILE_BASE_ROWS,
    }
    const pending = Promise.all([
      shaderModule(device, 'Planetary DEM terrain shader', surfaceShader).then(module =>
        Promise.all([
          device.createRenderPipelineAsync({
            label: 'Planetary DEM terrain',
            layout,
            vertex: { module, entryPoint: 'vertexMain', buffers: vertexBuffers },
            fragment: {
              module,
              entryPoint: 'fragmentMain',
              targets: [{ format: 'rgba16float' }],
              constants: surfaceConstants,
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM small per-sample sphere',
            layout,
            vertex: { module, entryPoint: 'smallVertex' },
            fragment: {
              module,
              entryPoint: 'smallFragment',
              targets: [{ format: 'rgba16float' }],
              constants: surfaceConstants,
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM small per-sample annotations',
            layout: smallAnnotationPipelineLayout,
            vertex: { module, entryPoint: 'smallVertex' },
            fragment: { module, entryPoint: 'smallAnnotationFragment', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
          device.createRenderPipelineAsync({
            label: 'Planetary DEM small per-sample annotations with coverage mask',
            layout: smallAnnotationPipelineLayout,
            vertex: { module, entryPoint: 'smallVertex' },
            fragment: {
              module,
              entryPoint: 'smallAnnotationCoverageFragment',
              targets: [{ format: 'rgba16float' }, { format: 'rg8unorm' }],
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
        ]),
      ),
      shaderModule(device, 'Planetary DEM annotation shader', annotationShader).then(module => {
        const annotationLayout = device.createBindGroupLayout({
          label: 'Planetary DEM annotation resources',
          entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, sampler: { type: 'filtering' } },
          ],
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
            fragment: {
              module,
              entryPoint: 'fragmentCoverageMain',
              targets: [{ format: 'rgba16float' }, { format: 'rg8unorm' }],
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            multisample: { count: 4 },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
          }),
        ]).then(([annotation, annotationCoverage]) => ({ annotationLayout, annotation, annotationCoverage }))
      }),
    ]).then(
      ([
        [surface, surfaceSmall, annotationSmall, annotationSmallCoverage],
        { annotationLayout, annotation, annotationCoverage },
      ]) => ({
        layout: bindLayout,
        annotationLayout,
        smallAnnotationLayout,
        surface,
        surfaceSmall,
        annotation,
        annotationCoverage,
        annotationSmall,
        annotationSmallCoverage,
      }),
    )
    cache.pipelines = pending
    void pending.catch(() => {
      if (cache.pipelines === pending) cache.pipelines = undefined
    })
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

async function buildBodyAlbedo(
  device: GPUDevice,
  cache: DevicePlanetCache,
  body: TerrainBody,
  url: string,
  maxWidth?: number,
): Promise<GPUTexture> {
  const texture = await loadSrgbMipmappedTexture(device, url, cache.controller.signal, {
    label: `${body} global albedo`,
    mipLabel: `Build ${body} albedo mipmaps`,
    downloadErrorMessage: `${body} albedo download failed.`,
    maxWidth,
    addressModeU: 'repeat',
  })
  cache.createdAlbedos.add(texture)
  return texture
}

function loadSharedBodyAlbedo(
  device: GPUDevice,
  body: TerrainBody,
  url: string,
  signal: AbortSignal,
  maxWidth?: number,
): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  const key = maxWidth ? `${url}@${maxWidth}` : url
  let pending = cache.albedos.get(key)
  if (!pending) {
    pending = buildBodyAlbedo(device, cache, body, url, maxWidth)
    cache.albedos.set(key, pending)
    void pending.catch(() => {
      if (cache.albedos.get(key) === pending) cache.albedos.delete(key)
    })
  }
  return raceAbort(pending, signal)
}

async function getSharedBodyTerrain(
  device: GPUDevice,
  body: TerrainBody,
  prepared: PreparedTerrain<'rg8unorm'>,
  reloadSource: (signal: AbortSignal) => Promise<PreparedTerrain<'rg8unorm'>>,
  signal: AbortSignal,
): Promise<SharedBodyTerrain> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  // Always key by the original prepared.terrain identity so recovery re-decoding on a replacement
  // device is deduplicated across concurrent and subsequent scenes.
  const cacheKey = prepared.terrain
  let pending = cache.bodies.get(cacheKey)
  if (!pending) {
    pending = (async () => {
      const source = cacheKey.pixels.byteLength ? prepared : await reloadSource(cache.controller.signal)
      const { terrain } = source
      const levels = source.levels ?? packRg8Levels(elevationPyramid(terrain, true))
      const mesh = buildPlanetPickingMesh(body, terrain)
      const quadtree = PlanetCubeQuadtree.fromTerrain(terrain, mesh.radius)
      const heights = device.createTexture({
        label: `${body} numeric terrain grid`,
        size: [terrain.width, terrain.height],
        mipLevelCount: levels.length,
        format: 'rg8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      try {
        for (const [mipLevel, { data, width, height }] of levels.entries()) {
          await writeTextureChunkedAsync(
            device.queue,
            heights,
            mipLevel,
            data,
            width,
            height,
            2,
            cache.controller.signal,
          )
        }
      } catch (error) {
        heights.destroy()
        throw error
      }
      const patchVertices = (cache.patchVertices ??= uploadBuffer(
        device,
        'Planetary quadtree patch vertices',
        sharedPatchMesh.vertices,
        GPUBufferUsage.VERTEX,
      ))
      const patchIndices = (cache.patchIndices ??= uploadBuffer(
        device,
        'Planetary quadtree patch indices',
        sharedPatchMesh.indices,
        GPUBufferUsage.INDEX,
      ))
      const sampler = device.createSampler({
        addressModeU: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
      })
      const shared: SharedBodyTerrain = { mesh, quadtree, heights, patchVertices, patchIndices, sampler }
      cache.created.push(shared)
      // Mip levels and raw CPU elevation bytes are no longer needed once resident on the GPU.
      // Keeping the original terrain object as the WeakMap key avoids re-uploading on the same device.
      source.levels = null
      if (source.terrain.pixels.byteLength) source.terrain.pixels = new Uint8ClampedArray(0)
      return shared
    })()
    cache.bodies.set(cacheKey, pending)
    void pending.catch(() => {
      if (cache.bodies.get(cacheKey) === pending) cache.bodies.delete(cacheKey)
    })
  }
  return raceAbort(pending, signal)
}

export class PlanetTerrainRenderer {
  private readonly bodies = new Map<TerrainBody, GpuTerrain>()
  private readonly device: GPUDevice
  private readonly bindLayout: GPUBindGroupLayout
  private readonly annotationLayout: GPUBindGroupLayout
  private readonly smallAnnotationLayout: GPUBindGroupLayout
  private readonly pipeline: GPURenderPipeline
  private readonly smallPipeline: GPURenderPipeline
  private readonly annotationPipeline: GPURenderPipeline
  private readonly annotationCoveragePipeline: GPURenderPipeline
  private readonly annotationSmallPipeline: GPURenderPipeline
  private readonly annotationSmallCoveragePipeline: GPURenderPipeline
  private readonly tiles: PlanetTileCache
  private readonly unsubscribeTiles: () => void
  private readonly annotationScratch = new Float32Array(20)

  private constructor(
    device: GPUDevice,
    pipelines: PlanetPipelines,
    tiles: PlanetTileCache,
    onTileUpdate?: () => void,
  ) {
    this.device = device
    this.bindLayout = pipelines.layout
    this.annotationLayout = pipelines.annotationLayout
    this.smallAnnotationLayout = pipelines.smallAnnotationLayout
    this.pipeline = pipelines.surface
    this.smallPipeline = pipelines.surfaceSmall
    this.annotationPipeline = pipelines.annotation
    this.annotationCoveragePipeline = pipelines.annotationCoverage
    this.annotationSmallPipeline = pipelines.annotationSmall
    this.annotationSmallCoveragePipeline = pipelines.annotationSmallCoverage
    this.tiles = tiles
    this.unsubscribeTiles = tiles.subscribe(
      () => onTileUpdate?.(),
      () => {
        this.rebindAll()
        onTileUpdate?.()
      },
    )
  }

  static async create(
    device: GPUDevice,
    signal: AbortSignal,
    assetUrl: WorkspaceAssets['url'],
    loadTerrain?: TerrainLoader,
    onTileUpdate?: () => void,
    loadPlanetTilePack?: PlanetTilePackLoader,
    albedoMaxWidths?: Partial<Record<TerrainBody, number>>,
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
        const prepared = loadTerrain
          ? await loadTerrain(terrainPath, 'rg8unorm', s => assetUrl(terrainPath, s), signal)
          : await loadPreparedTerrain(await assetUrl(terrainPath, signal), 'rg8unorm', signal)
        signal.throwIfAborted()
        let albedo: GPUTexture
        try {
          const url = await assetUrl(`textures/${body}-albedo.webp`, signal)
          albedo = await loadSharedBodyAlbedo(device, body, url, signal, albedoMaxWidths?.[body])
        } catch {
          if (signal.aborted) throw signal.reason
          albedo = getFallbackAlbedo(device)
        }
        signal.throwIfAborted()
        const reloadSource = async (reloadSignal: AbortSignal) =>
          loadPreparedTerrain(await assetUrl(terrainPath, reloadSignal), 'rg8unorm', reloadSignal)
        const shared = await getSharedBodyTerrain(device, body, prepared, reloadSource, signal)
        signal.throwIfAborted()
        renderer.attach(body, shared, albedo)
      }
      return renderer
    } catch (error) {
      renderer.destroy()
      throw error
    }
  }

  private createBodyBindGroup(
    body: TerrainBody,
    uniform: GPUBuffer,
    shared: SharedBodyTerrain,
    albedo: GPUTexture,
    albedoSampler: GPUSampler,
  ) {
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
      data.loupeBindGroup = this.createBodyBindGroup(
        body,
        data.loupeUniform,
        data.shared,
        data.albedo,
        data.albedoSampler,
      )
    }
  }

  private attach(body: TerrainBody, shared: SharedBodyTerrain, albedo: GPUTexture) {
    const device = this.device
    const cache = getPlanetCache(device)
    const albedoSampler = (cache.albedoSampler ??= device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    }))
    const parameters = new Float32Array(56)
    const { relief, interval, low, high } = settings[body]
    parameters.set([shared.mesh.radius * 1000, relief, interval, 1], 20)
    parameters.set(low, 24)
    parameters.set(high, 28)
    parameters[39] = body === 'mars' ? 1 : 0
    const uniform = device.createBuffer({
      label: `${body} terrain parameters`,
      size: parameters.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = this.createBodyBindGroup(body, uniform, shared, albedo, albedoSampler)
    const loupeParameters = new Float32Array(parameters)
    const loupeUniform = device.createBuffer({
      label: `${body} loupe terrain parameters`,
      size: loupeParameters.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const loupeBindGroup = this.createBodyBindGroup(body, loupeUniform, shared, albedo, albedoSampler)
    const annotationUniform = device.createBuffer({
      label: `${body} annotation camera`,
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const annotationBindGroup = device.createBindGroup({
      layout: this.annotationLayout,
      entries: [
        { binding: 0, resource: { buffer: annotationUniform } },
        { binding: 1, resource: shared.heights.createView() },
        { binding: 2, resource: shared.sampler },
      ],
    })
    const smallAnnotationParameters = new Float32Array(parameters)
    const smallAnnotationUniform = device.createBuffer({
      label: `${body} small annotation parameters`,
      size: smallAnnotationParameters.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const smallAnnotationBindGroup = device.createBindGroup({
      layout: this.smallAnnotationLayout,
      entries: [{ binding: 0, resource: { buffer: smallAnnotationUniform } }],
    })
    const patchInstances = new Float32Array(MAX_PATCH_INSTANCES * CUBE_INSTANCE_FLOATS)
    const patchInstanceBuffer = device.createBuffer({
      label: `${body} quadtree patch instances`,
      size: patchInstances.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    const loupePatchInstances = new Float32Array(MAX_PATCH_INSTANCES * CUBE_INSTANCE_FLOATS)
    const loupePatchInstanceBuffer = device.createBuffer({
      label: `${body} loupe quadtree patch instances`,
      size: loupePatchInstances.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.bodies.set(body, {
      shared,
      albedo,
      albedoSampler,
      parameters,
      uniform,
      bindGroup,
      loupeParameters,
      loupeUniform,
      loupeBindGroup,
      annotationUniform,
      annotationBindGroup,
      smallAnnotationParameters,
      smallAnnotationUniform,
      smallAnnotationBindGroup,
      patchInstances,
      patchInstanceBuffer,
      lastPatchMatrix: new Float32Array(16),
      lastPatchEye: [NaN, NaN, NaN],
      lastPatchWidth: -1,
      lastPatchHeight: -1,
      lastPatchReliefFade: -1,
      lastPatchCount: 0,
      loupePatchInstances,
      loupePatchInstanceBuffer,
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
    this.tiles.updateView(
      body,
      eyeUnit,
      right,
      up,
      back,
      aspect,
      data.shared.mesh.radius * 1000,
      physicalMetresPerPixel,
      loupe,
    )
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

  private prepareMainPatches(
    data: GpuTerrain,
    body: TerrainBody,
    matrix: Float32Array,
    eyeKm: readonly [number, number, number],
    viewportWidth: number,
    viewportHeight: number,
    reliefFade: number,
  ) {
    let same =
      data.lastPatchWidth === viewportWidth &&
      data.lastPatchHeight === viewportHeight &&
      data.lastPatchReliefFade === reliefFade &&
      data.lastPatchEye[0] === eyeKm[0] &&
      data.lastPatchEye[1] === eyeKm[1] &&
      data.lastPatchEye[2] === eyeKm[2]
    if (same) {
      for (let i = 0; i < 16; i++) {
        if (data.lastPatchMatrix[i] !== matrix[i]) {
          same = false
          break
        }
      }
    }
    if (same) return data.lastPatchCount
    const patchCount = data.shared.quadtree.selectPatches(
      data.patchInstances,
      matrix,
      eyeKm,
      viewportWidth,
      viewportHeight,
      settings[body].relief,
      reliefFade,
    )
    if (patchCount > 0) {
      this.device.queue.writeBuffer(
        data.patchInstanceBuffer,
        0,
        data.patchInstances,
        0,
        patchCount * CUBE_INSTANCE_FLOATS,
      )
    }
    data.lastPatchMatrix.set(matrix)
    data.lastPatchEye[0] = eyeKm[0]
    data.lastPatchEye[1] = eyeKm[1]
    data.lastPatchEye[2] = eyeKm[2]
    data.lastPatchWidth = viewportWidth
    data.lastPatchHeight = viewportHeight
    data.lastPatchReliefFade = reliefFade
    data.lastPatchCount = patchCount
    return patchCount
  }

  drawAnnotation(
    pass: GPURenderPassEncoder,
    body: TerrainBody,
    matrix: Float32Array,
    reliefFade = 1,
    includeCoverage = false,
    viewportWidth = 1024,
    viewportHeight = 1024,
    isSmall = false,
    smallView?: Float32Array,
    eye?: readonly [number, number, number],
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    if (isSmall && smallView) {
      const { smallAnnotationParameters, smallAnnotationUniform, smallAnnotationBindGroup } = data
      writePlanetViewUniform(smallAnnotationParameters, {
        matrix,
        eye,
        isMars: body === 'mars',
        smallView,
      })
      this.device.queue.writeBuffer(smallAnnotationUniform, 0, smallAnnotationParameters)
      pass.setPipeline(includeCoverage ? this.annotationSmallCoveragePipeline : this.annotationSmallPipeline)
      pass.setBindGroup(0, smallAnnotationBindGroup)
      pass.draw(6)
      return
    }
    const {
      shared: { mesh, patchVertices, patchIndices },
      annotationUniform,
      annotationBindGroup,
      patchInstanceBuffer,
    } = data
    const relief = settings[body].relief
    const eyeKm = eye ?? [0, 0, mesh.radius * 4]
    const patchCount = this.prepareMainPatches(data, body, matrix, eyeKm, viewportWidth, viewportHeight, reliefFade)
    if (patchCount === 0) return
    this.annotationScratch.set(matrix, 0)
    this.annotationScratch[16] = body === 'moon' ? 2 : 4
    this.annotationScratch[17] = reliefFade
    this.annotationScratch[18] = mesh.radius
    this.annotationScratch[19] = relief
    this.device.queue.writeBuffer(annotationUniform, 0, this.annotationScratch)
    pass.setPipeline(includeCoverage ? this.annotationCoveragePipeline : this.annotationPipeline)
    pass.setBindGroup(0, annotationBindGroup)
    pass.setVertexBuffer(0, patchVertices)
    pass.setVertexBuffer(1, patchInstanceBuffer)
    pass.setIndexBuffer(patchIndices, 'uint16')
    pass.drawIndexed(sharedPatchMesh.indexCount, patchCount)
  }

  draw(
    pass: GPURenderPassEncoder,
    body: TerrainBody,
    matrix: Float32Array,
    sunlight: readonly [number, number, number],
    eye: readonly [number, number, number],
    metresPerPixel: number,
    hdr: boolean,
    pixelRatio: number,
    viewportWidth: number,
    viewportHeight: number,
    isSmall = false,
    reliefFade = 1,
    smallView?: Float32Array,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const {
      shared: { patchVertices, patchIndices },
      parameters,
      uniform,
      bindGroup,
      patchInstanceBuffer,
    } = data
    const [minor, fine] = contourDetail(metresPerPixel)
    writePlanetViewUniform(parameters, {
      matrix,
      sunlight,
      hdr,
      pixelRatio,
      minorDetail: minor,
      fineDetail: fine,
      tilesReady: this.tiles.ready(body),
      reliefFade,
      eye,
      isMars: body === 'mars',
      smallView,
    })
    this.device.queue.writeBuffer(uniform, 0, parameters)
    if (isSmall) {
      pass.setPipeline(this.smallPipeline)
      pass.setBindGroup(0, bindGroup)
      pass.draw(6)
    } else {
      const patchCount = this.prepareMainPatches(data, body, matrix, eye, viewportWidth, viewportHeight, reliefFade)
      if (patchCount === 0) return
      pass.setPipeline(this.pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setVertexBuffer(0, patchVertices)
      pass.setVertexBuffer(1, patchInstanceBuffer)
      pass.setIndexBuffer(patchIndices, 'uint16')
      pass.drawIndexed(sharedPatchMesh.indexCount, patchCount)
    }
  }

  drawLoupe(
    pass: GPURenderPassEncoder,
    body: TerrainBody,
    matrix: Float32Array,
    sunlight: readonly [number, number, number],
    eye: readonly [number, number, number],
    metresPerPixel: number,
    hdr: boolean,
    pixelRatio: number,
    loupeSize: number,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const {
      shared: { quadtree, patchVertices, patchIndices },
      loupeParameters,
      loupeUniform,
      loupeBindGroup,
      loupePatchInstances,
      loupePatchInstanceBuffer,
    } = data
    const patchCount = quadtree.selectPatches(
      loupePatchInstances,
      matrix,
      eye,
      loupeSize,
      loupeSize,
      settings[body].relief,
      1,
    )
    if (patchCount === 0) return
    const [minor, fine] = contourDetail(metresPerPixel)
    writePlanetViewUniform(loupeParameters, {
      matrix,
      sunlight,
      hdr,
      pixelRatio,
      minorDetail: minor,
      fineDetail: fine,
      tilesReady: this.tiles.ready(body),
      reliefFade: 1,
      eye,
      isMars: body === 'mars',
    })
    this.device.queue.writeBuffer(loupeUniform, 0, loupeParameters)
    this.device.queue.writeBuffer(
      loupePatchInstanceBuffer,
      0,
      loupePatchInstances,
      0,
      patchCount * CUBE_INSTANCE_FLOATS,
    )
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, loupeBindGroup)
    pass.setVertexBuffer(0, patchVertices)
    pass.setVertexBuffer(1, loupePatchInstanceBuffer)
    pass.setIndexBuffer(patchIndices, 'uint16')
    pass.drawIndexed(sharedPatchMesh.indexCount, patchCount)
  }

  destroy() {
    this.unsubscribeTiles()
    for (const {
      uniform,
      loupeUniform,
      annotationUniform,
      smallAnnotationUniform,
      patchInstanceBuffer,
      loupePatchInstanceBuffer,
    } of this.bodies.values()) {
      uniform.destroy()
      loupeUniform.destroy()
      annotationUniform.destroy()
      smallAnnotationUniform.destroy()
      patchInstanceBuffer.destroy()
      loupePatchInstanceBuffer.destroy()
    }
    this.bodies.clear()
  }
}
