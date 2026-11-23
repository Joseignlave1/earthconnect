import type { WorkspaceAssets } from '../workspace/workspaceAssets.ts'
import { TerrainLighting } from './lighting.ts'
import { solarPosition } from './sun.ts'
import { resolveScoreStyle, ScoreDistribution, type ColorMode } from './score-style.ts'
import { LOUPE_TEXTURE_SIZE, PostProcess, type LoupeOverlay } from './postprocess.ts'
import { motionBlurUniforms } from './motion-blur.ts'
import { copyExternalImageChunked, generateMipmaps, raceAbort, shaderModule, uploadBuffer, writeTextureChunkedAsync } from './gpu.ts'
import { surfaceShader } from './surface-shader.ts'
import { cameraBasis, cameraMatrix, createMesh, EARTH_AXIS, ease, EQUAL_EARTH_HALF_HEIGHT, EQUAL_EARTH_HALF_WIDTH, equalEarthProject, FOV, grabCenter, orbitRotation, surfacePoint, SurfacePicker, type Camera, type Rotation, type Selection, type Vec3 } from './geometry.ts'
import { contourDetail } from './contours.ts'
import { MISSING, SIZE, worldTiles } from './world-grid.ts'
import { elevationAt, elevationPyramid, packRg32FloatLevels, type Terrain } from './terrain.ts'
import { loadPreparedTerrain, type PreparedTerrain } from './terrain-loader.ts'
import type { EarthScore, ScoreLayout } from '../earth-score/index.ts'
import { createScoreTextures, destroyScoreTextures, prepareComputedScores, readComputedScore, scoreMipCount, scoreTexturesMatch, writeScoreLookup, type ScoreTextures } from './computed-scores.ts'
import type { SolarCamera, SolarFocus, SolarFrame, SourceBody } from '../workspace/solarCamera.ts'
import type { SolarScene, SourcePoint } from '../workspace/solarScene.ts'
import { createSurfaceDevicePool } from './surface-device.ts'
import type { TerrainLoader } from '../workspace/earthWorkspaceResources.ts'
import { EarthTileCache, type EarthTilePackLoader } from './earth-tiles.ts'
import type { PlanetTilePackLoader } from '../workspace/planetTiles.ts'

type Options = { dataBoundary: boolean; contextFloor: number; contextOpacity: number; minimumScore: number; colorMode: ColorMode; focus: number; morph: number; base: number; emphasis: number; topRange: number; emission: number; opacity: number; exposure: number; bloom: number; hdr: boolean; relief: number; contourInterval: number; depthInterval: number; contourOpacity: number; grid: boolean; peaks: boolean; imagery: boolean }
type ScoreDescription = { name: string; description: string }
type SolarSettings = { createScene: typeof SolarScene.create; assetUrl: WorkspaceAssets['url']; loadTerrain?: TerrainLoader; loadPlanetTilePack?: PlanetTilePackLoader; onSourcePick: (body: SourceBody, point: SourcePoint) => void }
type SurfaceAssets = WorkspaceAssets['url']
type RendererSettings = {
  assets?: SurfaceAssets
  loadTerrain?: TerrainLoader
  loadEarthTilePack?: EarthTilePackLoader
  onDevice?: (device: GPUDevice) => void
  acquireDevice?: ReturnType<typeof createSurfaceDevicePool>
  transparent?: boolean
  signal?: AbortSignal
  initialView?: { latitude: number; longitude: number }
  solar?: SolarSettings
}
type ComputedSource = { layout: ScoreLayout; bestAddress: number | null }
type OrbitView = { rotation: Rotation; zoom: number; center: Vec3 }
type SurfacePipelines = {
  layout: GPUBindGroupLayout
  surface: GPURenderPipeline
  annotation: GPURenderPipeline
  coverage: GPURenderPipeline | null
  coverageDescriptor: GPURenderPipelineDescriptor
}
type SharedMeshGpu = {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  geographicSampler: GPUSampler
}
type DeviceSurfaceCache = {
  controller: AbortController
  pipelines: Map<string, Promise<SurfacePipelines>>
  mesh?: SharedMeshGpu
  earthTiles?: EarthTileCache
  terrains: WeakMap<object, Promise<GPUTexture>>
  earthTextures: Map<string, Promise<GPUTexture>>
  createdTextures: GPUTexture[]
}

const tileCount = worldTiles().length
const sharedMesh = createMesh()
const surfaceCache = new WeakMap<GPUDevice, DeviceSurfaceCache>()

function getSurfaceCache(device: GPUDevice): DeviceSurfaceCache {
  let entry = surfaceCache.get(device)
  if (!entry) {
    const controller = new AbortController()
    const createdTextures: GPUTexture[] = []
    entry = { controller, pipelines: new Map(), terrains: new WeakMap(), earthTextures: new Map(), createdTextures }
    surfaceCache.set(device, entry)
    void device.lost.then(() => {
      controller.abort()
      entry!.mesh?.vertexBuffer.destroy()
      entry!.mesh?.indexBuffer.destroy()
      entry!.earthTiles?.destroy()
      for (const texture of createdTextures) texture.destroy()
      createdTextures.length = 0
      surfaceCache.delete(device)
    })
  }
  return entry
}

async function buildEarthTexture(device: GPUDevice, cache: DeviceSurfaceCache, url: string): Promise<GPUTexture> {
  const { signal } = cache.controller
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error('Could not load the Earth background. Reload to retry.')
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none' })
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const mipLevelCount = Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1
    texture = device.createTexture({
      label: 'NASA Earth imagery',
      size: [bitmap.width, bitmap.height],
      mipLevelCount,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    await copyExternalImageChunked(device.queue, bitmap, texture, signal)
    generateMipmaps(device, texture, { label: 'Earth imagery mipmaps' })
    cache.createdTextures.push(texture)
    return texture
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}

function loadSharedEarthTexture(device: GPUDevice, url: string, signal: AbortSignal): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getSurfaceCache(device)
  let pending = cache.earthTextures.get(url)
  if (!pending) {
    pending = buildEarthTexture(device, cache, url)
    cache.earthTextures.set(url, pending)
    void pending.catch(() => { if (cache.earthTextures.get(url) === pending) cache.earthTextures.delete(url) })
  }
  return raceAbort(pending, signal)
}

function loadSharedTerrainTexture(
  device: GPUDevice,
  terrain: Terrain,
  prepared: PreparedTerrain<'rg32float'> | null,
  signal: AbortSignal,
): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getSurfaceCache(device)
  let pending = cache.terrains.get(terrain)
  if (!pending) {
    pending = (async () => {
      const levels = prepared?.levels ?? packRg32FloatLevels(elevationPyramid(terrain))
      const first = levels[0]
      const texture = device.createTexture({
        label: 'Numeric elevation and coast distance',
        size: [first.width, first.height],
        mipLevelCount: levels.length,
        format: 'rg32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      try {
        for (const [index, level] of levels.entries()) {
          await writeTextureChunkedAsync(device.queue, texture, index, level.data, level.width, level.height, 8, cache.controller.signal)
        }
        cache.createdTextures.push(texture)
        // Drop the ~85 MiB float32 mip pyramid from CPU RAM once uploaded to the GPU.
        if (prepared) prepared.levels = null
        return texture
      } catch (error) {
        texture.destroy()
        throw error
      }
    })()
    cache.terrains.set(terrain, pending)
    void pending.catch(() => { if (cache.terrains.get(terrain) === pending) cache.terrains.delete(terrain) })
  }
  return raceAbort(pending, signal)
}

function getSurfacePipelines(device: GPUDevice, banks: number, bankSize: number, filterable: boolean, reverseDepth: boolean): Promise<SurfacePipelines> {
  const cache = getSurfaceCache(device)
  const key = `${banks}:${bankSize}:${filterable}:${reverseDepth}`
  let pending = cache.pipelines.get(key)
  if (!pending) {
    pending = (async () => {
      const vf = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
      const bindLayout = device.createBindGroupLayout({ label: 'Surface resources',entries: [
        { binding: 0,visibility: vf,buffer: { type: 'uniform' } },
        { binding: 1,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'uint' } },
        { binding: 2,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'float' } },
        { binding: 3,visibility: vf,texture: { sampleType: filterable ? 'float' : 'unfilterable-float' } },
        { binding: 4,visibility: vf,sampler: { type: 'filtering' } },
        { binding: 5,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'float' } },
        ...Array.from({ length: banks },(_,i) => ({ binding: 6 + i,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'uint' as const,viewDimension: '2d-array' as const } })),
        { binding: 6 + banks,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'uint' } },
        { binding: 7 + banks,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'float',viewDimension: '2d-array' } },
        { binding: 8 + banks,visibility: GPUShaderStage.FRAGMENT,sampler: { type: 'filtering' } },
      ] })
      const module = await shaderModule(device,'Earth surface WGSL',surfaceShader(banks,bankSize,filterable))
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] })
      const pipeline = (label: string,entryPoint: string,targets: GPUColorTargetState[],count = 1): GPURenderPipelineDescriptor => ({
        label,layout,
        vertex: { module,entryPoint: 'surfaceVertex',buffers: [{ arrayStride: 8,attributes: [{ shaderLocation: 0,offset: 0,format: 'float32x2' }] }] },
        fragment: { module,entryPoint,targets },
        primitive: { topology: 'triangle-list',cullMode: 'back' },
        depthStencil: { format: reverseDepth ? 'depth32float' : 'depth24plus',depthWriteEnabled: true,depthCompare: reverseDepth ? 'greater' : 'less' },multisample: { count },
      })
      const [surface,annotation] = await Promise.all([
        device.createRenderPipelineAsync(pipeline('4x MSAA Earth surface','surfaceFragment',[{ format: 'rgba16float' }],4)),
        device.createRenderPipelineAsync(pipeline('Geographic annotations','annotationFragment',[{ format: 'rg32float' }])),
      ])
      return { layout: bindLayout,surface,annotation,coverage: null,coverageDescriptor: pipeline('Geographic annotations with coverage','coverageAnnotationFragment',[{ format: 'rg32float' },{ format: 'r8unorm' }]) }
    })()
    cache.pipelines.set(key, pending)
    void pending.catch(() => { if (cache.pipelines.get(key) === pending) cache.pipelines.delete(key) })
  }
  return pending
}

function getSharedMeshGpu(device: GPUDevice): SharedMeshGpu {
  const cache = getSurfaceCache(device)
  if (!cache.mesh) {
    cache.mesh = {
      vertexBuffer: uploadBuffer(device,'Surface UV vertices',sharedMesh.uv,GPUBufferUsage.VERTEX),
      indexBuffer: uploadBuffer(device,'Surface indices',sharedMesh.indices,GPUBufferUsage.INDEX),
      geographicSampler: device.createSampler({ addressModeU: 'repeat',addressModeV: 'clamp-to-edge',magFilter: 'linear',minFilter: 'linear',mipmapFilter: 'linear',maxAnisotropy: 4 }),
    }
  }
  return cache.mesh
}

export class SurfaceRenderer {
  private device!: GPUDevice
  private deviceLease?: ReturnType<ReturnType<typeof createSurfaceDevicePool>>
  private deviceErrors?: AbortController
  private context!: GPUCanvasContext
  private format: GPUTextureFormat = 'bgra8unorm'
  private pipelines: SurfacePipelines | null = null
  private layout!: GPUBindGroupLayout
  private bindings!: GPUBindGroup
  private annotationUniform!: GPUBuffer
  private annotationBindings!: GPUBindGroup
  private loupeUniform!: GPUBuffer
  private loupeBindings!: GPUBindGroup
  private loupeUniformValues = new Float32Array(60)
  private uniformBuffer!: GPUBuffer
  private uniformValues = new Float32Array(60)
  private shiftDown = false
  private loupeZoom: number | null = null
  private lockedLoupe: { rx: number; ry: number; radiusCss: number; zoom: number; dirty: boolean; matrix?: Float32Array } | null = null
  private readonly sun = solarPosition(new Date(performance.timeOrigin))
  private lighting!: TerrainLighting
  private post!: PostProcess
  private cursor: { x: number; y: number } | null = null
  private scoreMap: ScoreTextures | null = null
  private spareScoreMap: ScoreTextures | null = null
  private bankSize = 256
  private earthTexture!: GPUTexture
  private earthOwned = true
  private earthReady = false
  private earthImageUrl: string | null = null
  private earthTiles?: EarthTileCache
  private unsubscribeEarthTiles?: () => void
  private terrainTexture!: GPUTexture
  private terrainOwned = true
  private terrainRequest = new AbortController()
  private terrain: Terrain | null = null
  private poleHeights: [number,number] = [0,0]
  private preparedTerrain: PreparedTerrain<'rg32float'> | null = null
  private vertexBuffer!: GPUBuffer
  private indexBuffer!: GPUBuffer
  private geographicSampler!: GPUSampler
  private mesh = sharedMesh
  private picker = new SurfacePicker(this.mesh)
  private terrainHeight = (longitude: number, latitude: number) => Math.abs(latitude) === Math.PI / 2 ? this.poleHeights[latitude > 0 ? 1 : 0] : elevationAt(this.terrain!,longitude,latitude)
  private computed: ComputedSource | null = null
  private pendingComputed: AbortController | null = null
  private distribution: ScoreDistribution | null = null
  private frame = 0
  private overlayDirty = false
  private displayDirty = false
  private annotationsDirty = false
  private dirty = false
  private rendering = false
  private pendingMove: { id: number; x: number; y: number } | null = null
  private lost = true
  private destroyed = false
  private observer: ResizeObserver
  private pointers = new Map<number, { x: number; y: number }>()
  private gesture: { startX: number; startY: number; moved: boolean } | null = null
  private selected: Selection | null = null
  private grab: { position: Vec3 | null; camera: Camera; x: number; y: number } | null = null
  private center: Vec3 = [0,0,0]
  private rotation: Rotation = orbitRotation([0,0,0,1],0,.1)
  private zoom = 1
  private globeView: { rotation: Rotation; zoom: number } | null = null
  private projectionFrame = 0
  private solar?: SolarScene
  private solarCamera?: SolarCamera
  private pendingSolarProjection?: 'globe' | 'plane'
  private hasSolarLabels = false
  private solarFocus: SolarFocus = 'earth'
  private readonly sourcePoints: Partial<Record<SourceBody,SourcePoint>> = {}
  private readonly solarViews = new Map<SolarFocus,OrbitView>()
  private cameraHistory?: { camera: SolarFrame; time: number; flying: boolean }
  private readonly reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  readonly options: Options = { dataBoundary: false, contextFloor: .1, contextOpacity: .20, minimumScore: 0, colorMode: 'auto', focus: .22, morph: 1, base: .31, emphasis: 1, topRange: .1, emission: 1.5, opacity: .76, exposure: 1, bloom: .24, hdr: true, relief: 20, contourInterval: 500, depthInterval: 1000, contourOpacity: .45, grid: false, peaks: true, imagery: false }

  private style = resolveScoreStyle(null,this.options)
  get matchingScoreCount() { return this.distribution?.atLeast(this.options.minimumScore) ?? 0 }
  /** Load and dispatch Earth Score on this exact device. Device recovery replaces it. */
  get computeDevice(): GPUDevice | null { return this.lost || this.destroyed ? null : this.device }

  private constructor(private canvas: HTMLCanvasElement, private onPick: (selection: Selection | null, score: number | null) => void, private onError: (error: Error) => void, private onStatus: (status: string) => void, private transparent: boolean, private solarSettings?: SolarSettings, private acquireDevice = createSurfaceDevicePool(), assets?: SurfaceAssets, private onDevice?: RendererSettings['onDevice'], loadTerrain?: TerrainLoader, private loadEarthTilePack?: EarthTilePackLoader) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('Could not create the WebGPU surface.')
    this.context = context
    const { signal } = this.terrainRequest
    const assetError = (error: unknown) => {
      if (!signal.aborted) this.onError(error instanceof Error ? error : new Error(String(error)))
    }
    const terrain = assets
      ? (loadTerrain
        ? loadTerrain('textures/earth-terrain.bin', 'rg32float', s => assets('textures/earth-terrain.bin', s), signal)
        : assets('textures/earth-terrain.bin', signal).then(url => loadPreparedTerrain(url, 'rg32float', signal)))
      : loadPreparedTerrain(`${import.meta.env.BASE_URL}textures/earth-terrain.bin`, 'rg32float', signal)
    const imageUrlPromise = assets
      ? assets('textures/earth-blue-marble.jpg', signal)
      : Promise.resolve(`${import.meta.env.BASE_URL}textures/earth-blue-marble.jpg`)
    void imageUrlPromise.then(url => {
      if (this.destroyed || signal.aborted) return
      this.earthImageUrl = url
      if (!this.lost) this.startEarthUpload(url)
    }).catch(assetError)
    void terrain.then(prepared => {
      if (this.destroyed) return
      this.terrain = prepared.terrain
      this.poleHeights = prepared.poles
      this.preparedTerrain = prepared
      this.preparePicker()
      if (!this.lost) this.uploadTerrain()
      this.invalidate()
    }).catch(assetError)
    this.observer = new ResizeObserver(() => {
      if (this.lost || this.destroyed) return
      if (!canvas.clientWidth || !canvas.clientHeight) {
        this.post.suspend()
        this.cameraHistory = undefined
        canvas.width = canvas.height = 1
        return
      }
      this.invalidate()
    })
    this.observer.observe(canvas)
    canvas.addEventListener('pointerdown', this.pointerDown)
    canvas.addEventListener('pointermove', this.pointerMove)
    canvas.addEventListener('pointerleave', this.pointerLeave)
    canvas.addEventListener('pointerup', this.pointerUp)
    canvas.addEventListener('pointercancel', this.pointerCancel)
    canvas.addEventListener('wheel', this.wheel, { passive: false })
    canvas.addEventListener('keydown', this.keyDown)
    window.addEventListener('keydown', this.shiftKey)
    window.addEventListener('keyup', this.shiftKey)
    window.addEventListener('blur', this.pointerBlur)
    this.invalidate()
  }

  static async create(canvas: HTMLCanvasElement,onPick: (selection: Selection | null,score: number | null) => void,onError: (error: Error) => void,onStatus: (status: string) => void,{ transparent = false,signal,initialView,solar,acquireDevice,assets,loadTerrain,loadEarthTilePack,onDevice }: RendererSettings = {}) {
    signal?.throwIfAborted()
    const renderer = new SurfaceRenderer(canvas,onPick,onError,onStatus,transparent,solar,acquireDevice,assets,onDevice,loadTerrain,loadEarthTilePack)
    if (initialView) renderer.rotation = viewRotation(initialView.longitude * Math.PI / 180,initialView.latitude * Math.PI / 180)
    const cancel = () => renderer.destroy()
    signal?.addEventListener('abort',cancel,{ once: true })
    try { await renderer.initialize(); signal?.throwIfAborted(); return renderer }
    catch (error) { renderer.destroy(); throw error }
    finally { signal?.removeEventListener('abort',cancel) }
  }

  private async initialize(restoreScores = false) {
    const lease = this.deviceLease = this.acquireDevice()
    const { adapter, device } = await lease.ready
    if (this.destroyed || this.deviceLease !== lease) return
    const filterable = device.features.has('float32-filterable')
    const adapterInfo = adapter.info
    this.canvas.dataset.adapter = [adapterInfo?.vendor,adapterInfo?.architecture,adapterInfo?.device,adapterInfo?.description].filter(Boolean).join(' · ')
    this.device = device
    const deviceErrors = this.deviceErrors = new AbortController()
    device.addEventListener('uncapturederror',event => {
      if (!this.destroyed && this.device === device) this.onError(new Error(`WebGPU: ${event.error.message}`))
    }, { signal: deviceErrors.signal })
    lease.onLost(info => {
      if (this.destroyed || this.device !== device) return
      deviceErrors.abort()
      lease.release()
      this.lost = true
      this.pendingComputed?.abort(new Error(`The WebGPU device was reset: ${info.message || info.reason}`))
      this.canvas.dataset.ready = 'false'
      delete this.canvas.dataset.frameSource
      this.canvas.classList.remove('custom-cursor')
      cancelAnimationFrame(this.frame); this.frame = 0
      this.releaseGpuResources()
      const hadComputedScores = Boolean(this.computed)
      if (hadComputedScores) this.clearComputedScores()
      this.canvas.dispatchEvent(new Event('computedevicechange'))
      void this.initialize(hadComputedScores || restoreScores).catch(error => { if (!this.destroyed) this.onError(error instanceof Error ? error : new Error(String(error))) })
    })
    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({ device,format: this.format,alphaMode: this.transparent ? 'premultiplied' : 'opaque' })
    this.onDevice?.(device)
    this.bankSize = Math.min(256,device.limits.maxTextureArrayLayers)
    const banks = Math.ceil(tileCount / this.bankSize)
    const reverseDepth = Boolean(this.solarSettings)
    const pipelines = await getSurfacePipelines(device,banks,this.bankSize,filterable,reverseDepth)
    if (this.destroyed || this.device !== device) return
    this.layout = pipelines.layout
    this.pipelines = pipelines
    this.post = new PostProcess(device,this.format,this.transparent,reverseDepth)
    if (this.solarSettings) {
      const solar = await this.solarSettings.createScene(device,new Date(this.sun.timestamp),() => this.invalidate(),this.onError,this.solarCamera,this.solarSettings.assetUrl,this.solarSettings.loadTerrain,this.solarSettings.loadPlanetTilePack)
      if (this.destroyed || this.device !== device) { solar.destroy(); return }
      this.solar = solar
      this.solarCamera = solar.camera
    }
    const sunlight = this.solar?.sunlight('earth') as Vec3 | undefined
    this.lighting = new TerrainLighting(device,sunlight ?? this.sun.direction)
    await Promise.all([this.post.initialize(),this.lighting.initialize()])
    if (this.destroyed || this.device !== device) return
    this.uniformBuffer = device.createBuffer({ label: 'Surface parameters',size: this.uniformValues.byteLength,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.annotationUniform = device.createBuffer({ label: 'Annotation parameters',size: this.uniformValues.byteLength,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.loupeUniform = device.createBuffer({ label: 'Optical loupe surface parameters',size: this.uniformValues.byteLength,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    const sharedGpu = getSharedMeshGpu(device)
    this.vertexBuffer = sharedGpu.vertexBuffer
    this.indexBuffer = sharedGpu.indexBuffer
    this.geographicSampler = sharedGpu.geographicSampler
    this.scoreMap = createScoreTextures(device,this.bankSize,banks,0)
    writeScoreLookup(device,this.scoreMap,new Uint16Array(36 * 18).fill(MISSING))
    this.earthTexture = device.createTexture({ label: 'Earth imagery placeholder',size: [1,1],format: 'rgba8unorm',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    this.earthOwned = true
    device.queue.writeTexture({ texture: this.earthTexture },new Uint8Array([18,30,42,255]),{},[1,1])
    const surfaceDeviceCache = getSurfaceCache(device)
    const earthTiles = (surfaceDeviceCache.earthTiles ??= new EarthTileCache(device))
    this.earthTiles = earthTiles
    this.unsubscribeEarthTiles = earthTiles.subscribe(
      () => { if (!this.destroyed && !this.lost) this.invalidate(false) },
      () => { if (!this.destroyed && !this.lost) { this.bindResources(); this.invalidate(false) } },
    )
    if (this.loadEarthTilePack) earthTiles.attachPacks(this.loadEarthTilePack)
    this.terrainTexture = device.createTexture({ label: 'Terrain placeholder',size: [1,1],format: 'rg32float',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    this.terrainOwned = true
    device.queue.writeTexture({ texture: this.terrainTexture },new Float32Array([0,32]),{},[1,1])
    this.canvas.dataset.sunTime = this.sun.timestamp
    this.canvas.dataset.sunLatitude = (sunlight ? Math.asin(sunlight[1]) * 180 / Math.PI : this.sun.latitude).toFixed(6)
    this.canvas.dataset.sunLongitude = (sunlight ? Math.atan2(sunlight[0],sunlight[2]) * 180 / Math.PI : this.sun.longitude).toFixed(6)
    this.canvas.dataset.backend = 'webgpu'
    this.canvas.dataset.hdr = 'float16'
    this.canvas.dataset.terrainPrecision = 'float32'
    this.canvas.dataset.terrainFiltering = filterable ? 'hardware' : 'shader'
    this.bindResources()
    this.lost = false
    if (this.terrain) this.uploadTerrain()
    if (this.earthImageUrl) this.startEarthUpload(this.earthImageUrl)
    this.canvas.dispatchEvent(new Event('computedevicechange'))
    this.canvas.classList.add('custom-cursor')
    if (restoreScores && !this.distribution) {
      this.onStatus('WebGPU recovered. Run the live query again to restore the score map.')
    }
    this.invalidate()
  }

  private bindResources() {
    const map = this.scoreMap
    const tiles = this.earthTiles
    if (!this.uniformBuffer || !map || !tiles) return
    const banks = map.banks.length
    const bind = (buffer: GPUBuffer) => this.device.createBindGroup({ label: 'Earth surface bindings',layout: this.layout,entries: [
      { binding: 0,resource: { buffer } },{ binding: 1,resource: map.lookup.createView() },
      { binding: 2,resource: this.earthTexture.createView() },{ binding: 3,resource: this.terrainTexture.createView() },
      { binding: 4,resource: this.geographicSampler },{ binding: 5,resource: this.lighting.texture.createView() },
      ...map.banks.map((texture,i) => ({ binding: 6 + i,resource: texture.createView({ dimension: '2d-array' }) })),
      { binding: 6 + banks,resource: tiles.pageTexture.createView() },
      { binding: 7 + banks,resource: tiles.poolTexture.createView({ dimension: '2d-array' }) },
      { binding: 8 + banks,resource: tiles.sampler },
    ] })
    this.bindings = bind(this.uniformBuffer)
    this.annotationBindings = bind(this.annotationUniform)
    this.loupeBindings = bind(this.loupeUniform)
  }

  private uploadTerrain() {
    const device = this.device
    const terrain = this.terrain!
    const { signal } = this.terrainRequest
    void loadSharedTerrainTexture(device, terrain, this.preparedTerrain, signal).then(texture => {
      if (this.destroyed || this.lost || this.device !== device || this.terrain !== terrain) return
      if (this.preparedTerrain) this.preparedTerrain.levels = null
      if (this.terrainOwned) this.terrainTexture.destroy()
      this.terrainTexture = texture
      this.terrainOwned = false
      this.lighting.setTerrain(this.terrainTexture, this.options.relief)
      this.bindResources()
      this.canvas.dataset.terrainReady = 'true'
      this.invalidate()
    }).catch(error => {
      if (!this.destroyed && !this.lost && !signal.aborted && this.device === device && !getSurfaceCache(device).controller.signal.aborted) {
        this.onError(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  private startEarthUpload(url: string) {
    const device = this.device
    const { signal } = this.terrainRequest
    void loadSharedEarthTexture(device, url, signal).then(texture => {
      if (this.destroyed || this.lost || this.device !== device || this.earthImageUrl !== url) return
      if (this.earthOwned) this.earthTexture.destroy()
      this.earthTexture = texture
      this.earthOwned = false
      this.earthReady = true
      this.bindResources()
      this.canvas.dataset.earthReady = 'true'
      this.invalidate(false)
    }).catch(error => {
      if (!this.destroyed && !this.lost && !signal.aborted && this.device === device && !getSurfaceCache(device).controller.signal.aborted) {
        this.onError(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  /**
   * Publish previously dispatched scores from this device. The caller owns
   * EarthScore and serializes dispatches that share its output buffers.
   */
  async setComputedScores(earth: EarthScore, { name,description }: ScoreDescription, signal?: AbortSignal) {
    signal?.throwIfAborted()
    const device = this.computeDevice
    if (!device) throw new Error('WebGPU is still initializing. Try again shortly.')
    this.pendingComputed?.abort()
    const request = new AbortController()
    this.pendingComputed = request
    const active = signal ? AbortSignal.any([signal,request.signal]) : request.signal
    const started = performance.now()
    const bankCount = this.scoreMap!.banks.length
    const sites = earth.layout.sites.length
    let target = this.spareScoreMap
    if (target) {
      this.spareScoreMap = null
      if (!scoreTexturesMatch(target,this.bankSize,bankCount,sites)) {
        destroyScoreTextures(target)
        target = null
      }
    }
    try {
      const { textures,histogram,bestAddress,bytes } = await prepareComputedScores(device,earth,this.bankSize,bankCount,active,target ?? undefined)
      target = null
      if (active.aborted || this.computeDevice !== device) {
        if (this.computeDevice === device && !this.spareScoreMap && scoreTexturesMatch(textures,this.bankSize,bankCount,sites)) {
          this.spareScoreMap = textures
        } else {
          destroyScoreTextures(textures)
        }
        active.throwIfAborted()
        throw new Error('The WebGPU device changed while preparing scores.')
      }
      const previous = this.scoreMap!
      this.scoreMap = textures
      this.computed = { layout: earth.layout,bestAddress }
      this.bindResources()
      if (!this.spareScoreMap && scoreTexturesMatch(previous,this.bankSize,bankCount,sites)) {
        this.spareScoreMap = previous
      } else {
        destroyScoreTextures(previous)
      }
      const entries = earth.layout.rows
      const distribution = new ScoreDistribution(histogram)
      this.publishScoreInfo({ name,description },distribution,entries,bytes)
      this.invalidate()
      this.onStatus(`Live query · ${name}\n${description}\n4 × 4 km · ${distribution.valid.toLocaleString()} valid cells\n${earth.layout.sites.length} × ${SIZE}² tiles · ${(bytes / 2 ** 20).toFixed(1)} MiB GPU incl. LODs\nPrepared in ${((performance.now() - started) / 1000).toFixed(1)} s (WebGPU)`)
    } catch (error) {
      if (target) {
        if (this.computeDevice === device && !this.spareScoreMap) this.spareScoreMap = target
        else destroyScoreTextures(target)
      }
      throw error
    } finally {
      if (this.pendingComputed === request) this.pendingComputed = null
    }
  }

  private publishScoreInfo(score: ScoreDescription, distribution: ScoreDistribution, entries: number, bytes: number) {
    this.distribution = distribution
    this.style = resolveScoreStyle(distribution,this.options,this.style)
    this.selected = null
    Object.assign(this.canvas.dataset,{
      source: 'computed',cells: String(distribution.valid),entries: String(entries),scoreBytes: String(entries * 2),gpuScoreBytes: String(bytes),
      bestScore: String(distribution.best),scoreName: score.name,scoreDescription: score.description,
    })
    this.onPick(null,null)
  }

  private clearComputedScores() {
    this.computed = null
    this.distribution = null
    this.style = resolveScoreStyle(null,this.options,this.style)
    this.selected = null
    for (const key of ['source','cells','entries','scoreBytes','gpuScoreBytes','bestScore','scoreName','scoreDescription']) delete this.canvas.dataset[key]
    this.onPick(null,null)
    this.onStatus('Recovering WebGPU…')
  }

  update(options: Partial<Options>) {
    const keys = (Object.keys(options) as (keyof Options)[]).filter(key => options[key] !== this.options[key])
    if (!keys.length) return
    if (keys.includes('morph')) { this.center = [0,0,0]; this.grab = null }
    if (keys.includes('morph') || keys.includes('relief')) this.cameraHistory = undefined
    Object.assign(this.options,options)
    if (['colorMode','base','focus','topRange','emphasis'].some(key => keys.includes(key as keyof Options))) this.style = resolveScoreStyle(this.distribution,this.options,this.style)
    const surface = keys.some(key => !['exposure','bloom','dataBoundary'].includes(key))
    if (surface) this.invalidate(keys.some(key => ['morph','relief','peaks'].includes(key)))
    else this.invalidateDisplay()
  }

  resetCamera() { this.cameraHistory = undefined; this.lockedLoupe = null; this.rotation = [0,0,0,1]; this.center = [0,0,0]; this.grab = null; this.zoom = 1; this.invalidate() }

  focusSolarBody(focus: SolarFocus) {
    if (!this.solarSettings || this.solarFocus === focus) return
    this.lockedLoupe = null
    this.pendingSolarProjection = undefined
    if (this.options.morph !== 1 || this.projectionFrame) {
      cancelAnimationFrame(this.projectionFrame)
      this.projectionFrame = 0
      if (this.globeView) {
        this.rotation = this.globeView.rotation
        this.zoom = this.globeView.zoom
      }
      this.update({ morph: 1 })
    }
    this.pointerBlur()
    this.solarViews.set(this.solarFocus,{ rotation: this.rotation, zoom: this.zoom, center: this.center })
    this.solarFocus = focus
    const previous = this.solarViews.get(focus)
    const initial = focus === 'mars' ? viewRotation(-75 * Math.PI / 180,.25) : viewRotation(0,0)
    this.rotation = previous?.rotation ?? initial
    this.zoom = previous?.zoom ?? 1
    this.center = previous?.center ?? [0,0,0]
    this.solarCamera?.startFlight(this.reducedMotion.matches)
    if (focus === 'moon' || focus === 'mars') this.solar?.loadTerrain()
    this.invalidate()
  }

  setSourcePoint(body: SourceBody,point: SourcePoint) {
    this.sourcePoints[body] = point
    this.invalidateOverlay()
  }

  retrySolarTerrain() { this.solar?.loadTerrain() }

  setProjection(projection: 'globe' | 'plane') {
    if (this.solarCamera?.transitioning) {
      this.pendingSolarProjection = projection
      return
    }
    this.lockedLoupe = null
    const targetMorph = Number(projection === 'globe')
    if (this.options.morph === targetMorph && !this.projectionFrame) return
    cancelAnimationFrame(this.projectionFrame)
    if (this.options.morph === 1) this.globeView = { rotation: this.rotation, zoom: this.zoom }
    const target = projection === 'globe' && this.globeView ? this.globeView : { rotation: [0,0,0,1] as Rotation, zoom: 1 }
    const from = { morph: this.options.morph, rotation: this.rotation, zoom: this.zoom }
    const duration = this.reducedMotion.matches ? 0 : 480
    const started = performance.now()
    const tick = (now: number) => {
      const progress = duration ? Math.min((now - started) / duration,1) : 1
      const fraction = ease(progress)
      this.rotation = blendRotation(from.rotation,target.rotation,fraction)
      this.zoom = from.zoom * Math.pow(target.zoom / from.zoom,fraction)
      this.update({ morph: from.morph + (targetMorph - from.morph) * fraction })
      this.projectionFrame = progress < 1 ? requestAnimationFrame(tick) : 0
    }
    tick(started)
  }

  focusBest() {
    if (!this.distribution?.valid || this.distribution.best < this.options.minimumScore || !this.computed) return
    const { bestAddress,layout } = this.computed
    const center = bestAddress === null ? null : layout.locate(bestAddress)?.center
    if (center) this.focusScore(center.longitude * Math.PI / 180,center.latitude * Math.PI / 180)
  }

  private focusScore(longitude: number, latitude: number) {
    this.cameraHistory = undefined
    this.lockedLoupe = null
    if (this.options.morph < 1) {
      const [px,py] = equalEarthProject(longitude,latitude)
      this.rotation = [0,0,0,1]
      this.center = [px,py,0]
    } else {
      this.rotation = viewRotation(longitude,latitude)
      this.center = [0,0,0]
    }
    this.grab = null; this.zoom = 12
    const elevation = this.terrain ? this.terrainHeight(longitude,latitude) : 0
    this.selected = { longitude: longitude * 180 / Math.PI,latitude: latitude * 180 / Math.PI,elevation,position: surfacePoint(longitude,latitude,ease(this.options.morph),elevation,this.options.relief) }
    this.onPick(this.selected,this.distribution!.best)
    this.invalidate()
  }

  private camera(): Camera {
    const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight)
    if (this.solarFocus === 'system') {
      const distance = 1 / (Math.tan(FOV / 2) * Math.min(1,aspect) * this.zoom)
      return { yaw: 0,pitch: 0,rotation: this.rotation,distance,aspect }
    }
    const morph = ease(this.options.morph)
    // Fit both endpoints and intermediate shapes, including portrait viewports.
    const halfWidth = EQUAL_EARTH_HALF_WIDTH * (1 - morph) + morph
    const halfHeight = EQUAL_EARTH_HALF_HEIGHT * (1 - morph) + morph
    const margin = 1.06 * (1 - morph) + morph
    const fit = Math.max(halfHeight, halfWidth / aspect) * margin / Math.tan(Math.PI / 8) + 1.4 * morph
    const gap = (fit - morph) / this.zoom
    const clearance = this.solarFocus === 'earth' ? 9000 / EARTH_AXIS * this.options.relief : .035
    return { yaw: 0, pitch: 0, rotation: this.rotation, center: this.center, distance: morph + clearance + gap, aspect, near: Math.max(.000001, Math.min(.05, gap * .08)) }
  }

  private schedule() {
    this.canvas.dataset.idle = 'false'
    if (this.frame || this.rendering || this.lost || this.destroyed) return
    this.frame = requestAnimationFrame(this.renderFrame)
  }

  private invalidate(annotations = true) {
    this.dirty = true
    this.annotationsDirty ||= annotations
    this.displayDirty = true
    if (this.lockedLoupe) this.lockedLoupe.dirty = true
    this.schedule()
  }

  private invalidateDisplay() { this.displayDirty = true; this.schedule() }
  private invalidateOverlay() { this.overlayDirty = true; this.schedule() }

  private renderFrame = async () => {
    this.frame = 0
    if (this.lost || this.destroyed || this.rendering) return
    // Keep the scene state and redraw when the preview becomes visible.
    if (!this.canvas.clientWidth || !this.canvas.clientHeight) return
    this.rendering = true
    try {
      this.applyPendingMove()
      if (this.dirty || this.annotationsDirty || this.displayDirty || this.overlayDirty) {
        const redraw = this.dirty,annotations = this.annotationsDirty,display = this.displayDirty
        this.dirty = false; this.annotationsDirty = false; this.displayDirty = false; this.overlayDirty = false
        this.render(redraw,annotations,display)
      }
    } catch (error) {
      if (!this.destroyed && !this.lost) this.onError(error instanceof Error ? error : new Error(String(error)))
    } finally {
      this.rendering = false
      // Input events schedule work. A stationary pointer or idle view does not
      // need a polling loop, history convergence or additional GPU submissions.
      if (this.dirty || this.annotationsDirty || this.displayDirty || this.overlayDirty) this.schedule()
    }
  }

  private render(redraw = true,annotations = true,display = true) {
    if (this.destroyed || this.lost) return false
    const canvas = this.canvas,device = this.device
    const ratio = Math.min(window.devicePixelRatio || 1,2)
    const maxDim = device.limits.maxTextureDimension2D
    const width = Math.min(maxDim,Math.max(1,Math.round(canvas.clientWidth * ratio)))
    const height = Math.min(maxDim,Math.max(1,Math.round(canvas.clientHeight * ratio)))
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height }
    if (this.post.configure(width,height,ratio)) {
      redraw = true; annotations = true; display = true
      this.cameraHistory = undefined
    }
    const hdr = this.options.hdr
    let camera = this.camera()
    const o = this.options,s = this.style
    const solar = this.solar
    const frameTime = performance.now()
    const sceneFrame = solar?.camera.update(solar.bodies,this.solarFocus,camera,frameTime)
    const flying = solar?.camera.flying ?? false
    const stableCamera = this.projectionFrame ? undefined : sceneFrame
    const prior = this.cameraHistory
    const motion = stableCamera && redraw && !this.reducedMotion.matches && prior?.flying === flying
      ? motionBlurUniforms(prior.camera,stableCamera,frameTime - prior.time,ratio,!flying)
      : undefined
    const arrival = solar?.camera.takeArrival()
    if (arrival) {
      this.rotation = arrival
      this.center = [0,0,0]
      camera = this.camera()
    }
    if (this.pendingSolarProjection && solar && !solar.camera.transitioning) {
      const projection = this.pendingSolarProjection
      this.pendingSolarProjection = undefined
      queueMicrotask(() => { if (!this.destroyed && this.solarFocus === 'earth') this.setProjection(projection) })
    }
    const renderEarth = !sceneFrame || solar!.drawEarth(sceneFrame,height)
    if (o.dataBoundary && !this.post.hasCurrentCoverage) annotations = true
    const cursor = this.cursor && document.elementFromPoint(this.cursor.x,this.cursor.y) === canvas ? this.cursor : null
    const hover = cursor
      ? this.solarFocus === 'earth'
        ? this.pickAt(cursor.x,cursor.y)
        : this.pickSourceAt(cursor.x,cursor.y) ?? null
      : null
    const metresPerPixel = 2 * Math.tan(FOV / 2) * (camera.distance - ease(o.morph)) * EARTH_AXIS / Math.max(1,canvas.clientHeight)
    const detail = contourDetail(metresPerPixel)
    if (redraw || annotations) {
      const u = this.uniformValues
      u.set(sceneFrame ? solar!.earthMatrix(sceneFrame) : cameraMatrix(camera),0)
      u.set(s.anchors,16)
      u.set([s.cutoff,s.contrast,s.topRange,o.emission * s.emissionScale],20)
      u.set([o.contextFloor,o.contextOpacity,o.minimumScore,o.opacity],24)
      u.set([ease(o.morph),o.relief,s.best,scoreMipCount],28)
      u.set([o.contourInterval,o.depthInterval,o.contourOpacity,ratio],32)
      u.set([...detail,0,Number(hdr)],36)
      u.set([Number(s.kind === 'auto'),Number(o.grid),Number(o.peaks),Number(o.imagery)],40)
      u.set([Number(Boolean(this.terrain)),Number(Boolean(sceneFrame)),Number(this.earthReady),Number(Boolean(this.earthTiles?.ready))],44)
      u.set([...this.poleHeights,0,0],48)
      const earthSun = sceneFrame ? solar!.sunlight('earth') : this.sun.direction
      const earthEye = sceneFrame ? solar!.earthEye(sceneFrame) : cameraBasis(camera).eye
      u.set([earthSun[0],earthSun[1],earthSun[2],0],52)
      u.set([earthEye[0],earthEye[1],earthEye[2],0],56)
      if (redraw) device.queue.writeBuffer(this.uniformBuffer,0,u)
      if (annotations) device.queue.writeBuffer(this.annotationUniform,0,u)
    }
    const encoder = device.createCommandEncoder({ label: 'Surface frame' })
    const lighting = this.lighting.encode(encoder,o.relief)
    if (lighting.rebound) this.bindResources()
    const drawMesh = (pass: GPURenderPassEncoder,pipeline: GPURenderPipeline,bindings: GPUBindGroup) => {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0,bindings)
      pass.setVertexBuffer(0,this.vertexBuffer); pass.setIndexBuffer(this.indexBuffer,'uint32')
      pass.drawIndexed(this.mesh.indices.length)
    }
    const pipelines = this.pipelines!
    const clientHeight = Math.max(1,canvas.clientHeight)
    if (redraw) {
      const pass = this.post.begin(encoder,hdr,Boolean(motion))
      if (sceneFrame) {
        const overview = this.solarFocus === 'system'
        const flightTarget = this.solarFocus !== 'system' && solar!.camera.flying ? this.solarFocus : undefined
        solar!.drawSky(pass,sceneFrame,height)
        solar!.draw(
          pass,sceneFrame,height,clientHeight,hdr,ratio,overview,flightTarget,
          renderEarth ? () => drawMesh(pass,pipelines.surface,this.bindings) : undefined,
        )
      } else if (renderEarth) {
        drawMesh(pass,pipelines.surface,this.bindings)
      }
      if (sceneFrame && renderEarth && o.morph > .001) solar!.drawEarthAtmosphere(pass,sceneFrame,ease(o.morph),clientHeight)
      if (sceneFrame) solar!.drawMarsAtmosphere(pass,sceneFrame,height,clientHeight)
      if (sceneFrame) solar!.drawOrbits(pass,sceneFrame,this.solarFocus === 'system')
      if (sceneFrame) solar!.drawHalo(pass)
      pass.end()
    }
    if (annotations) {
      const earthBoundary = o.dataBoundary && this.solarFocus === 'earth'
      const pipeline = earthBoundary
        ? pipelines.coverage ??= device.createRenderPipeline(pipelines.coverageDescriptor)
        : pipelines.annotation
      const pass = this.post.beginAnnotations(encoder,earthBoundary)
      if (renderEarth) drawMesh(pass,pipeline,this.annotationBindings)
      if (sceneFrame && !earthBoundary) solar!.drawAnnotations(pass,sceneFrame,height)
      pass.end()
    }
    const rect = canvas.getBoundingClientRect()
    const locked = !flying ? this.lockedLoupe : null
    const loupeOverlay = this.renderLoupe(
      encoder, drawMesh, camera, sceneFrame, rect, width, height, clientHeight, metresPerPixel, hdr, redraw, flying, cursor, Boolean(hover), locked,
    )
    const spaceCursor: [number,number] | null = cursor && !hover && !loupeOverlay
      ? [(cursor.x - rect.left) * width / Math.max(1,rect.width),(cursor.y - rect.top) * height / Math.max(1,rect.height)]
      : null
    const earthMetrics = sceneFrame && o.morph === 1
      ? solar!.bodyMarkerMetrics(sceneFrame,'earth',clientHeight)
      : { scale: metresPerPixel / EARTH_AXIS,alpha: 1 }
    const moonMetrics = sceneFrame ? solar!.bodyMarkerMetrics(sceneFrame,'moon',clientHeight) : { scale: 1,alpha: 0 }
    const marsMetrics = sceneFrame ? solar!.bodyMarkerMetrics(sceneFrame,'mars',clientHeight) : { scale: 1,alpha: 0 }
    const hoverBody = this.solarFocus === 'earth' ? 0 : this.solarFocus === 'moon' ? 1 : this.solarFocus === 'mars' ? 2 : -1
    const hoverScale = hoverBody === 0 ? earthMetrics.scale : hoverBody === 1 ? moonMetrics.scale : hoverBody === 2 ? marsMetrics.scale : metresPerPixel / EARTH_AXIS
    const moonPoint = this.sourcePoints.moon
    const marsPoint = this.sourcePoints.mars
    const markers = [
      (hover?.longitude ?? 0) / 360 + .5,(hover?.latitude ?? 0) / 180 + .5,Number(Boolean(hover && !loupeOverlay)),hoverScale,
      ease(o.morph),0,0,0,
      (this.selected?.longitude ?? 0) / 360 + .5,(this.selected?.latitude ?? 0) / 180 + .5,this.selected ? earthMetrics.alpha : 0,earthMetrics.scale,
      (moonPoint?.longitude ?? 0) / 360 + .5,(moonPoint?.latitude ?? 0) / 180 + .5,moonPoint ? moonMetrics.alpha : 0,moonMetrics.scale,
      (marsPoint?.longitude ?? 0) / 360 + .5,(marsPoint?.latitude ?? 0) / 180 + .5,marsPoint ? marsMetrics.alpha : 0,marsMetrics.scale,
    ]
    const built = this.post.finish(encoder,this.context.getCurrentTexture().createView(),{
      hdr,exposure: o.exposure,glow: o.bloom,ratio,dataBoundary: o.dataBoundary,cursor: spaceCursor,hoverBody,markers,loupe: loupeOverlay,motion,
    },redraw,display || annotations)
    device.queue.submit([encoder.finish()])
    if (renderEarth && this.solarFocus === 'earth' && !flying && this.earthTiles?.ready && (redraw || loupeOverlay)) {
      const loupePick = loupeOverlay
        ? (locked ? this.pickAt(rect.left + locked.rx, rect.top + locked.ry) : hover)
        : null
      const loupePoint = loupeOverlay && loupePick
        ? { lon: loupePick.longitude * Math.PI / 180, lat: loupePick.latitude * Math.PI / 180, zoom: loupeOverlay.zoom }
        : null
      this.earthTiles.updateView(camera, ease(o.morph), metresPerPixel / ratio, loupePoint)
    } else if (sceneFrame && (this.solarFocus === 'moon' || this.solarFocus === 'mars') && !flying && (redraw || loupeOverlay)) {
      const loupePick = loupeOverlay
        ? (locked ? this.pickSourceAt(rect.left + locked.rx, rect.top + locked.ry) : hover)
        : null
      const loupePoint = loupeOverlay && loupePick
        ? { lon: loupePick.longitude * Math.PI / 180, lat: loupePick.latitude * Math.PI / 180, zoom: loupeOverlay.zoom }
        : null
      solar!.updatePlanetTiles(sceneFrame, this.solarFocus, clientHeight, ratio, loupePoint)
    }
    if (redraw || !prior) this.cameraHistory = stableCamera && { camera: stableCamera,time: frameTime,flying }
    // Input may stop after this frame. Restore the sharp cached scene once,
    // sharing the next frame with any movement that has arrived in the meantime.
    if (built.motion && !flying) this.invalidateDisplay()
    if (lighting.gradients) canvas.dataset.gradientBuilds = String(Number(canvas.dataset.gradientBuilds || 0) + 1)
    if (lighting.lighting) canvas.dataset.lightingBuilds = String(Number(canvas.dataset.lightingBuilds || 0) + 1)
    canvas.dataset.contourDetail = detail.map(value => value.toFixed(3)).join(',')
    canvas.dataset.samples = '4'
    if (sceneFrame) {
      canvas.dataset.cameraBlur = String(built.motion)
      canvas.dataset.flightBlur = String(built.motion && flying)
      if (built.motion) {
        canvas.dataset.cameraBlurBuilds = String(Number(canvas.dataset.cameraBlurBuilds || 0) + 1)
        if (flying) canvas.dataset.flightBlurBuilds = String(Number(canvas.dataset.flightBlurBuilds || 0) + 1)
      }
    }
    canvas.dataset.renderSize = `${width} × ${height}`
    if (annotations) canvas.dataset.annotationDraws = String(Number(canvas.dataset.annotationDraws || 0) + 1)
    if (redraw) {
      canvas.dataset.sceneDraws = String(Number(canvas.dataset.sceneDraws || 0) + 1)
      if (solar) {
        const needsSolarTerrain = this.solarFocus === 'moon' || this.solarFocus === 'mars'
        const settled = renderEarth && this.earthReady && this.terrain &&
          solar.skySettled && (!needsSolarTerrain || solar.terrainReady)
        canvas.dataset.frameSource = settled ? (canvas.dataset.source || 'surface') : ''
        // Defer Moon & Mars terrain/8K albedo loading until Earth has settled its first frame
        // so startup CPU cores and GPU queue fences stay dedicated to Earth + scoring.
        if (settled && !solar.terrainReady) solar.loadTerrain()
      }
    }
    if (built.bloom) canvas.dataset.bloomBuilds = String(Number(canvas.dataset.bloomBuilds || 0) + 1)
    if (built.composition) canvas.dataset.displayBuilds = String(Number(canvas.dataset.displayBuilds || 0) + 1)
    canvas.dataset.idle = String(!this.dirty && !this.annotationsDirty && !this.displayDirty && !this.overlayDirty)
    canvas.dataset.cursor = loupeOverlay ? (loupeOverlay.locked ? 'loupe-locked' : 'loupe') : cursor ? hover ? 'surface' : 'space' : 'hidden'
    canvas.dataset.loupeZoom = loupeOverlay ? loupeOverlay.zoom.toFixed(2) : ''
    canvas.dataset.earthTiles = this.earthTiles?.ready ? `${this.earthTiles.residentCount}/${this.earthTiles.indexedCount}` : ''
    canvas.dataset.planetTiles = solar ? solar.planetTileStatus() : ''
    canvas.dataset.ready = String(Boolean(this.distribution))
    canvas.dataset.draws = String(Number(canvas.dataset.draws || 0) + 1)
    if (sceneFrame) {
      canvas.dataset.solarFocus = this.solarFocus
      canvas.dataset.cameraFlight = String(solar!.camera.flying)
      canvas.dataset.cameraKm = sceneFrame.eye.map(value => value.toFixed(3)).join(',')
      canvas.dataset.solarTerrain = String(solar!.terrainReady)
      const showLabels = this.solarFocus === 'system' && !solar!.camera.flying
      if (showLabels && redraw || !showLabels && this.hasSolarLabels) {
        canvas.dispatchEvent(new CustomEvent('solarlabels',{ detail: showLabels ? solar!.labels(sceneFrame) : [] }))
        this.hasSolarLabels = showLabels
      }
      if (solar!.camera.flying) this.invalidate()
    }
    canvas.dispatchEvent(new Event('renderstatus'))
    return true
  }

  private renderLoupe(
    encoder: GPUCommandEncoder,
    drawMesh: (pass: GPURenderPassEncoder, pipeline: GPURenderPipeline, bindings: GPUBindGroup) => void,
    camera: Camera,
    sceneFrame: SolarFrame | undefined,
    rect: DOMRect,
    width: number,
    height: number,
    clientHeight: number,
    metresPerPixel: number,
    hdr: boolean,
    redraw: boolean,
    flying: boolean,
    cursor: { x: number; y: number } | null,
    hasHover: boolean,
    locked: typeof this.lockedLoupe,
  ): LoupeOverlay | null {
    const scanning = !locked && this.shiftDown && Boolean(cursor) && hasHover && !flying && !this.gesture?.moved && this.pointers.size <= 1
    if (!locked && (!scanning || !cursor)) return null
    const solar = this.solar
    const o = this.options
    const pipelines = this.pipelines!
    const loupeRadiusCss = locked ? locked.radiusCss : 76
    const loupeZoom = locked ? locked.zoom : this.effectiveLoupeZoom()
    const centerClientX = locked ? rect.left + locked.rx : cursor!.x
    const centerClientY = locked ? rect.top + locked.ry : cursor!.y
    const needsRender = !locked || locked.dirty || redraw
    const baseMatrix = needsRender
      ? (this.solarFocus === 'earth'
        ? (sceneFrame ? solar!.earthMatrix(sceneFrame) : cameraMatrix(camera))
        : (sceneFrame && (this.solarFocus === 'moon' || this.solarFocus === 'mars') ? solar!.sourceMatrix(sceneFrame,this.solarFocus) : null))
      : null
    if (needsRender && !baseMatrix) return null

    let loupeMatrix = locked?.matrix ?? null
    if (needsRender && baseMatrix) {
      const [nx,ny] = this.pointerCoordinates(centerClientX,centerClientY)
      const sx = (Math.max(1,rect.width) / (2 * loupeRadiusCss)) * loupeZoom
      const sy = (Math.max(1,rect.height) / (2 * loupeRadiusCss)) * loupeZoom
      loupeMatrix = zoomProjectionMatrix(baseMatrix,nx,ny,sx,sy)
      const loupeRatio = LOUPE_TEXTURE_SIZE / (2 * loupeRadiusCss)
      const loupePass = this.post.beginLoupe(encoder,hdr)
      if (this.solarFocus === 'earth') {
        const lu = this.loupeUniformValues
        lu.set(this.uniformValues)
        lu.set(loupeMatrix,0)
        lu[35] = loupeRatio
        lu.set(contourDetail(metresPerPixel / loupeZoom),36)
        this.device.queue.writeBuffer(this.loupeUniform,0,lu)
        if (sceneFrame) solar!.drawLoupeSky(loupePass,sceneFrame,nx,ny,1 / sx,1 / sy)
        drawMesh(loupePass,pipelines.surface,this.loupeBindings)
        if (sceneFrame && o.morph > .001) solar!.drawLoupeEarthAtmosphere(loupePass,sceneFrame,ease(o.morph))
      } else if (sceneFrame && (this.solarFocus === 'moon' || this.solarFocus === 'mars')) {
        const bodyMetrics = solar!.bodyMarkerMetrics(sceneFrame,this.solarFocus,clientHeight)
        const bodyMpp = bodyMetrics.scale * solar!.bodies[this.solarFocus].radius * 1000 / loupeZoom
        solar!.drawLoupeSky(loupePass,sceneFrame,nx,ny,1 / sx,1 / sy)
        solar!.drawLoupe(loupePass,this.solarFocus,sceneFrame,loupeMatrix,bodyMpp,hdr,loupeRatio)
        if (this.solarFocus === 'mars') solar!.drawLoupeMarsAtmosphere(loupePass,sceneFrame)
      } else if (sceneFrame) {
        solar!.drawLoupeSky(loupePass,sceneFrame,nx,ny,1 / sx,1 / sy)
      }
      loupePass.end()
      if (locked) { locked.dirty = false; locked.matrix = loupeMatrix }
    }

    const selectedPoint = this.solarFocus === 'earth'
      ? this.selected
      : (this.solarFocus === 'moon' || this.solarFocus === 'mars')
        ? this.sourcePoints[this.solarFocus] ?? null
        : null
    let selectedInLoupe: [number,number] | null = null
    if (selectedPoint && loupeMatrix) {
      const pos = this.solarFocus === 'earth'
        ? this.selected?.position ?? null
        : (this.solarFocus === 'moon' || this.solarFocus === 'mars')
          ? solar!.sourcePosition(this.solarFocus,selectedPoint)
          : null
      const eye = this.solarFocus === 'earth'
        ? cameraBasis(camera).eye
        : (sceneFrame && (this.solarFocus === 'moon' || this.solarFocus === 'mars')
          ? solar!.sourceEye(sceneFrame,this.solarFocus)
          : null)
      if (pos && eye) {
        const [px,py,pz] = pos
        const r2 = px * px + py * py + pz * pz
        const frontFacing = (this.solarFocus === 'earth' && o.morph < 1) ||
          (px * (eye[0] - px) + py * (eye[1] - py) + pz * (eye[2] - pz) > -0.02 * r2)
        if (frontFacing) {
          const m = loupeMatrix
          const cw = m[3] * px + m[7] * py + m[11] * pz + m[15]
          if (cw > 0) {
            const lx = (m[0] * px + m[4] * py + m[8] * pz + m[12]) / cw * loupeRadiusCss
            const ly = -(m[1] * px + m[5] * py + m[9] * pz + m[13]) / cw * loupeRadiusCss
            if (Math.hypot(lx,ly) <= loupeRadiusCss + 8) selectedInLoupe = [lx,ly]
          }
        }
      }
    }
    return {
      x: (centerClientX - rect.left) * width / Math.max(1,rect.width),
      y: (centerClientY - rect.top) * height / Math.max(1,rect.height),
      radiusCss: loupeRadiusCss,
      aimX: locked && cursor ? cursor.x - centerClientX : 0,
      aimY: locked && cursor ? cursor.y - centerClientY : 0,
      locked: Boolean(locked),
      zoom: loupeZoom,
      selected: selectedInLoupe,
    }
  }

  private effectiveLoupeZoom() {
    return this.loupeZoom ?? Math.max(2.2,Math.min(5.5,5.5 / Math.pow(Math.max(1,this.zoom),.16)))
  }

  private isInsideLockedLoupe(clientX: number, clientY: number, margin = 0) {
    const locked = this.lockedLoupe
    if (!locked) return false
    const rect = this.canvas.getBoundingClientRect()
    return Math.hypot(clientX - (rect.left + locked.rx), clientY - (rect.top + locked.ry)) <= locked.radiusCss + margin
  }

  private loupeWorldPoint(clientX: number, clientY: number): [number, number] {
    const locked = this.lockedLoupe
    if (!locked) return [clientX, clientY]
    const rect = this.canvas.getBoundingClientRect()
    const cx = rect.left + locked.rx, cy = rect.top + locked.ry
    if (Math.hypot(clientX - cx, clientY - cy) > locked.radiusCss + 4) return [clientX, clientY]
    return [cx + (clientX - cx) / locked.zoom, cy + (clientY - cy) / locked.zoom]
  }

  private shiftKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && this.lockedLoupe) {
      this.lockedLoupe = null
      this.invalidateOverlay()
      return
    }
    const next = event.shiftKey
    if (next === this.shiftDown) return
    this.shiftDown = next
    if (this.cursor) this.invalidateOverlay()
  }

  private trackCursor(event: PointerEvent) {
    this.shiftDown = event.shiftKey
    if (event.pointerType === 'touch') this.cursor = null
    else if (this.cursor) { this.cursor.x = event.clientX; this.cursor.y = event.clientY }
    else this.cursor = { x: event.clientX, y: event.clientY }
    if (this.lockedLoupe && this.pointers.size === 0 && !this.isInsideLockedLoupe(event.clientX, event.clientY, 10)) {
      this.lockedLoupe = null
    }
    this.invalidateOverlay()
  }

  private pointerLeave = () => {
    this.cursor = null
    this.invalidateOverlay()
  }

  private pointerBlur = () => {
    this.cameraHistory = undefined
    this.shiftDown = false
    this.lockedLoupe = null
    this.cursor = null; this.pendingMove = null; this.grab = null; this.gesture = null
    for (const id of this.pointers.keys()) if (this.canvas.hasPointerCapture(id)) this.canvas.releasePointerCapture(id)
    this.pointers.clear()
    this.invalidateOverlay()
  }

  private pointerCoordinates(x: number, y: number): [number,number] {
    const rect = this.canvas.getBoundingClientRect()
    return [(x - rect.left) / rect.width * 2 - 1,1 - (y - rect.top) / rect.height * 2]
  }

  private preparePicker() {
    this.picker.setSurface(ease(this.options.morph),this.terrain ? { heightAt: this.terrainHeight, exaggeration: this.options.relief } : undefined)
  }

  private pickAt(x: number, y: number) {
    if (this.solar?.camera.flying || this.solarFocus !== 'earth') return null
    this.preparePicker()
    const [nx,ny] = this.pointerCoordinates(x,y)
    const camera = this.camera()
    const result = this.picker.pick(nx,ny,camera)
    const frame = this.solar?.camera.frame
    if (result && frame) {
      const eye = cameraBasis(camera).eye
      const distance = Math.hypot(result.position[0] - eye[0],result.position[1] - eye[1],result.position[2] - eye[2]) * EARTH_AXIS / 1000
      if (!this.solar!.visiblePoint(frame,nx,ny,'earth',distance)) return null
    }
    return result
  }

  private clampPlaneCenter(center: Vec3): Vec3 {
    return [
      Math.max(-EQUAL_EARTH_HALF_WIDTH,Math.min(EQUAL_EARTH_HALF_WIDTH,center[0])),
      Math.max(-EQUAL_EARTH_HALF_HEIGHT,Math.min(EQUAL_EARTH_HALF_HEIGHT,center[1])),
      0,
    ]
  }

  private regrab() {
    const pointer = this.pointers.size === 1 ? [...this.pointers.values()][0] : null
    const camera = this.camera()
    let position: Vec3 | null = null
    if (pointer && this.options.morph < 1) {
      position = this.pickAt(pointer.x,pointer.y)?.position ?? null
      if (!position) {
        const [nx,ny] = this.pointerCoordinates(pointer.x,pointer.y)
        const { right,up,back,eye } = cameraBasis(camera)
        const t = Math.tan(FOV / 2)
        const dir = back.map((value,i) => -value + right[i] * nx * t * camera.aspect + up[i] * ny * t) as Vec3
        if (Math.abs(dir[2]) > 1e-6) {
          const hitT = -eye[2] / dir[2]
          if (hitT > 0) position = [eye[0] + dir[0] * hitT,eye[1] + dir[1] * hitT,0]
        }
      }
    }
    this.grab = pointer ? { ...pointer, camera, position } : null
  }

  private pointerDown = (event: PointerEvent) => {
    if (this.solar?.camera.flying) return
    if (event.button === 2 && (this.solarFocus === 'moon' || this.solarFocus === 'mars')) return
    this.trackCursor(event)
    this.applyPendingMove()
    this.canvas.focus(); this.canvas.setPointerCapture(event.pointerId)
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
    this.regrab()
    if (this.pointers.size === 1) this.gesture = { startX: event.clientX, startY: event.clientY, moved: false }
    else if (this.gesture) this.gesture.moved = true
    this.invalidateOverlay()
  }

  private pointerMove = (event: PointerEvent) => {
    this.trackCursor(event)
    const previous = this.pointers.get(event.pointerId)
    if (!previous) return
    if (this.gesture && Math.hypot(event.clientX - this.gesture.startX, event.clientY - this.gesture.startY) > 4) {
      this.gesture.moved = true
      if (this.lockedLoupe) this.lockedLoupe = null
    }
    if (this.pointers.size > 1) {
      if (this.lockedLoupe) this.lockedLoupe = null
      const other = [...this.pointers.entries()].find(([id]) => id !== event.pointerId)![1]
      const old = Math.hypot(previous.x - other.x, previous.y - other.y)
      const next = Math.hypot(event.clientX - other.x, event.clientY - other.y)
      if (old > 0) this.setZoom(this.zoom * next / old)
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
    } else if ((!this.lockedLoupe && !this.shiftDown) || this.gesture?.moved) {
      // Mouse reports can greatly outnumber display frames. Keep the newest
      // position; solve the exact anchor once, immediately before drawing it.
      if (!this.pendingMove) this.pendingMove = { id: event.pointerId, x: event.clientX, y: event.clientY }
      else { this.pendingMove.x = event.clientX; this.pendingMove.y = event.clientY }
    }
    this.invalidate()
  }

  private applyPendingMove() {
    const pending = this.pendingMove
    if (!pending) return
    this.pendingMove = null
    const previous = this.pointers.get(pending.id)
    if (!previous) return
    // Solve against mouse-down, not the previous frame. Returning to a cursor
    // position restores the same orientation regardless of the intervening path.
    const camera = this.grab?.camera ?? this.camera()
    if (this.options.morph < 1) {
      if (this.grab?.position) {
        const [nx,ny] = this.pointerCoordinates(pending.x,pending.y)
        this.center = this.clampPlaneCenter(grabCenter(camera,this.grab.position,nx,ny))
      }
    } else {
      const scale = 2 * Math.tan(FOV / 2) * Math.max(.001,camera.distance - ease(this.options.morph)) / Math.max(1,this.canvas.clientHeight)
      this.rotation = orbitRotation(camera.rotation ?? this.rotation,(pending.x - (this.grab?.x ?? previous.x)) * scale,(pending.y - (this.grab?.y ?? previous.y)) * scale)
    }
    this.pointers.set(pending.id,{ x: pending.x, y: pending.y })
  }

  pickSourceAt(clientX: number, clientY: number) {
    const body = this.solarFocus
    if (body === 'earth' || body === 'system' || this.solar?.camera.flying) return
    const frame = this.solar?.camera.frame
    if (!frame) return
    const [px,py] = this.loupeWorldPoint(clientX,clientY)
    const [nx,ny] = this.pointerCoordinates(px,py)
    const point = this.solar?.pickSource(frame,body,nx,ny)
    if (point) return { body, ...point }
  }

  private pointerUp = (event: PointerEvent) => {
    this.trackCursor(event)
    this.applyPendingMove()
    if (!this.pointers.has(event.pointerId)) return
    if (this.pointers.size === 1 && this.gesture && !this.gesture.moved) {
      if (event.button === 0 && !this.lockedLoupe && this.shiftDown && !this.solar?.camera.flying) {
        const hit = this.solarFocus === 'earth'
          ? this.pickAt(event.clientX,event.clientY)
          : this.pickSourceAt(event.clientX,event.clientY)
        if (hit) {
          const rect = this.canvas.getBoundingClientRect()
          this.lockedLoupe = {
            rx: event.clientX - rect.left,
            ry: event.clientY - rect.top,
            radiusCss: 76,
            zoom: this.effectiveLoupeZoom(),
            dirty: true,
          }
          this.invalidateOverlay()
          this.pointers.delete(event.pointerId)
          this.regrab()
          if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId)
          if (!this.pointers.size) this.gesture = null
          return
        }
      }
      const [px,py] = this.loupeWorldPoint(event.clientX,event.clientY)
      if (this.solarFocus === 'earth') {
        this.selected = this.pickAt(px,py)
        if (this.selected && this.terrain) this.selected.elevation = elevationAt(this.terrain,this.selected.longitude * Math.PI / 180,this.selected.latitude * Math.PI / 180)
        this.reportPickedScore(this.selected)
      } else {
        const point = this.pickSourceAt(event.clientX,event.clientY)
        if (point) this.solarSettings?.onSourcePick(point.body,point)
      }
      this.invalidateOverlay()
    }
    this.pointers.delete(event.pointerId)
    this.regrab()
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId)
    if (!this.pointers.size) this.gesture = null
  }

  private reportPickedScore(selection: Selection | null) {
    if (!selection || !this.computed) this.onPick(selection,null)
    else void this.reportComputedPick(selection,this.computed)
  }

  private async reportComputedPick(selection: Selection,source: ComputedSource) {
    const device = this.computeDevice
    if (!device) return
    const address = source.layout.addressAt(selection.longitude,selection.latitude)
    const current = () => this.selected === selection && this.computed === source && this.computeDevice === device
    try {
      const code = address
        ? await readComputedScore(device,this.scoreMap!,this.bankSize,Math.floor(address.address / SIZE ** 2),address.x,address.y)
        : MISSING
      if (current()) this.onPick(selection,code === MISSING ? null : code / 65534)
    } catch (error) {
      if (current()) this.onError(error instanceof Error ? error : new Error(String(error)))
    }
  }

  private pointerCancel = (event: PointerEvent) => {
    this.cursor = null
    this.invalidate()
    this.applyPendingMove()
    this.pointers.delete(event.pointerId)
    this.regrab()
    if (this.gesture) this.gesture.moved = true
    if (!this.pointers.size) this.gesture = null
  }

  private wheel = (event: WheelEvent) => {
    this.applyPendingMove()
    event.preventDefault()
    if (this.solar?.camera.flying) return
    const deltaScale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.canvas.clientHeight : 1
    if (event.shiftKey || this.isInsideLockedLoupe(event.clientX, event.clientY, 4)) {
      const raw = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
      const current = this.lockedLoupe?.zoom ?? this.effectiveLoupeZoom()
      const next = Math.max(1.5, Math.min(24, current * Math.exp(-raw * deltaScale * .0018)))
      this.loupeZoom = next
      if (this.lockedLoupe) {
        this.lockedLoupe.zoom = next
        this.lockedLoupe.dirty = true
      }
      this.invalidateOverlay()
      return
    }
    if (this.lockedLoupe) this.lockedLoupe = null
    const delta = event.deltaY * deltaScale
    this.setZoom(this.zoom * Math.exp(-delta * .001))
    this.regrab()
    this.invalidate()
  }

  private keyDown = (event: KeyboardEvent) => {
    if (this.solar?.camera.flying) return
    this.applyPendingMove()
    const panStep = .3 / Math.max(.25,this.zoom)
    if (this.options.morph < 1) {
      if (event.key === 'ArrowLeft') this.center = this.clampPlaneCenter([this.center[0] - panStep,this.center[1],0])
      else if (event.key === 'ArrowRight') this.center = this.clampPlaneCenter([this.center[0] + panStep,this.center[1],0])
      else if (event.key === 'ArrowUp') this.center = this.clampPlaneCenter([this.center[0],this.center[1] + panStep,0])
      else if (event.key === 'ArrowDown') this.center = this.clampPlaneCenter([this.center[0],this.center[1] - panStep,0])
      else if (event.key === '+' || event.key === '=') this.setZoom(this.zoom * 1.12)
      else if (event.key === '-') this.setZoom(this.zoom / 1.12)
      else if (event.key === 'Home') this.resetCamera()
      else return
    } else if (event.key === 'ArrowLeft') this.rotation = orbitRotation(this.rotation,.12,0)
    else if (event.key === 'ArrowRight') this.rotation = orbitRotation(this.rotation,-.12,0)
    else if (event.key === 'ArrowUp') this.rotation = orbitRotation(this.rotation,0,.12)
    else if (event.key === 'ArrowDown') this.rotation = orbitRotation(this.rotation,0,-.12)
    else if (event.key === '+' || event.key === '=') this.setZoom(this.zoom * 1.12)
    else if (event.key === '-') this.setZoom(this.zoom / 1.12)
    else if (event.key === 'Home') this.resetCamera()
    else return
    this.regrab()
    event.preventDefault(); this.invalidate()
  }

  private setZoom(value: number) {
    this.zoom = this.solarFocus === 'system' ? Math.max(.04,Math.min(22,value)) : Math.max(.4,Math.min(1000,value))
  }

  private releaseGpuResources() {
    this.cameraHistory = undefined
    this.lockedLoupe = null
    if (this.hasSolarLabels) this.canvas.dispatchEvent(new CustomEvent('solarlabels',{ detail: [] }))
    this.hasSolarLabels = false
    this.solar?.destroy()
    this.solar = undefined
    if (this.earthOwned) this.earthTexture?.destroy()
    if (this.terrainOwned) this.terrainTexture?.destroy()
    this.unsubscribeEarthTiles?.()
    this.unsubscribeEarthTiles = undefined
    this.earthTiles = undefined
    this.earthOwned = this.terrainOwned = false
    this.earthReady = false
    if (this.scoreMap) destroyScoreTextures(this.scoreMap)
    this.scoreMap = null
    if (this.spareScoreMap) destroyScoreTextures(this.spareScoreMap)
    this.spareScoreMap = null
    this.uniformBuffer?.destroy(); this.annotationUniform?.destroy(); this.loupeUniform?.destroy()
    this.lighting?.destroy(); this.post?.destroy()
    this.pipelines = null
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.canvas.dispatchEvent(new Event('computedevicechange'))
    this.terrainRequest.abort()
    this.pendingComputed?.abort()
    this.computed = null
    this.preparedTerrain = null
    cancelAnimationFrame(this.projectionFrame)
    cancelAnimationFrame(this.frame); this.observer.disconnect()
    this.canvas.removeEventListener('pointerdown',this.pointerDown)
    this.canvas.removeEventListener('pointermove',this.pointerMove)
    this.canvas.removeEventListener('pointerleave',this.pointerLeave)
    this.canvas.removeEventListener('pointerup',this.pointerUp)
    this.canvas.removeEventListener('pointercancel',this.pointerCancel)
    this.canvas.removeEventListener('wheel',this.wheel)
    this.canvas.removeEventListener('keydown',this.keyDown)
    window.removeEventListener('keydown',this.shiftKey)
    window.removeEventListener('keyup',this.shiftKey)
    window.removeEventListener('blur',this.pointerBlur)
    this.canvas.classList.remove('custom-cursor')
    this.releaseGpuResources()
    this.context.unconfigure()
    this.deviceErrors?.abort()
    this.deviceLease?.release()
  }
}

function zoomProjectionMatrix(base: Float32Array, nx: number, ny: number, sx: number, sy: number) {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    const o = col * 4
    const x = base[o], y = base[o + 1], z = base[o + 2], w = base[o + 3]
    out[o] = sx * (x - nx * w)
    out[o + 1] = sy * (y - ny * w)
    out[o + 2] = z
    out[o + 3] = w
  }
  return out
}

function viewRotation(longitude: number,latitude: number) {
  return orbitRotation(orbitRotation([0,0,0,1],-longitude,0),0,latitude)
}

function blendRotation(from: Rotation,to: Rotation,fraction: number): Rotation {
  const dot = from.reduce((sum,component,index) => sum + component * to[index],0)
  const direction = dot < 0 ? -1 : 1
  const [x,y,z,w] = from.map((component,index) => (1 - fraction) * component + fraction * direction * to[index])
  const normal = Math.hypot(x,y,z,w)
  return [x / normal,y / normal,z / normal,w / normal]
}
