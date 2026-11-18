import type { WorkspaceAssets } from '../workspace/workspaceAssets.ts'
import { TerrainLighting } from './lighting.ts'
import { solarPosition } from './sun.ts'
import { GpuTimer, GPU_STAGES, type GpuTimings, type TimingResult } from './gpu-timing.ts'
import { resolveScoreStyle, ScoreDistribution, type ColorMode } from './score-style.ts'
import { PostProcess } from './postprocess.ts'
import { motionBlurUniforms } from './motion-blur.ts'
import { shaderModule, uploadBuffer, fullscreenVertex } from './gpu.ts'
import { surfaceShader } from './surface-shader.ts'
import { cameraBasis, cameraMatrix, createMesh, EARTH_AXIS, ease, FOV, grabCenter, grabRotation, orbitRotation, surfacePoint, SurfacePicker, type Camera, type Rotation, type Selection, type Vec3 } from './geometry.ts'
import { contourDetail } from './contours.ts'
import { cellAddress, MISSING, RADIUS, SIZE, tileRow, tileX, worldTiles } from './world-grid.ts'
import type { WorldNoise } from './world-noise.ts'
import DatasetWorker from './noise-worker.ts?worker&inline'
import { prepareDataset, type DataRequest, type DatasetMessage } from './prepare-dataset.ts'
import { elevationAt, elevationPyramid, loadTerrain, poleElevations, type Terrain } from './terrain.ts'
import type { EarthScore, ScoreLayout } from '../earth-score/index.ts'
import { createScoreTextures, destroyScoreTextures, prepareComputedScores, readComputedScore, scoreMipCount, writeScoreLookup, type ScoreTextures } from './computed-scores.ts'
import type { SolarCamera, SolarFocus, SolarFrame, SourceBody } from '../workspace/solarCamera.ts'
import type { SolarScene, SourcePoint } from '../workspace/solarScene.ts'
import { createSurfaceDevicePool } from './surface-device.ts'

export type Options = { dataBoundary: boolean; contextFloor: number; contextOpacity: number; minimumScore: number; colorMode: ColorMode; focus: number; morph: number; base: number; emphasis: number; topRange: number; emission: number; opacity: number; exposure: number; bloom: number; hdr: boolean; relief: number; contourInterval: number; depthInterval: number; contourOpacity: number; grid: boolean; peaks: boolean; imagery: boolean }
type ScoreDescription = { name: string; description: string }
type SolarSettings = { createScene: typeof SolarScene.create; assetUrl: WorkspaceAssets['url']; onSourcePick: (body: SourceBody, point: SourcePoint) => void }
type SurfaceAssets = WorkspaceAssets['url']
type RendererSettings = {
  assets?: SurfaceAssets
  snapshot?: (signal: AbortSignal) => Promise<Blob>
  onDevice?: (device: GPUDevice) => void
  acquireDevice?: ReturnType<typeof createSurfaceDevicePool>
  transparent?: boolean
  signal?: AbortSignal
  initialView?: { latitude: number; longitude: number }
  solar?: SolarSettings
}
type ComputedSource = { layout: ScoreLayout; bestAddress: number | null }
type DatasetJob = { controller: AbortController; worker?: Worker }
type OrbitView = { rotation: Rotation; zoom: number; center: Vec3 }
const tileCount = worldTiles().length

export class SurfaceRenderer {
  private device!: GPUDevice
  private deviceLease?: ReturnType<ReturnType<typeof createSurfaceDevicePool>>
  private deviceErrors?: AbortController
  private context!: GPUCanvasContext
  private format: GPUTextureFormat = 'bgra8unorm'
  private pipelines: { surface: GPURenderPipeline; annotation: GPURenderPipeline; coverage: GPURenderPipeline | null; coverageDescriptor: GPURenderPipelineDescriptor } | null = null
  private layout!: GPUBindGroupLayout
  private bindings!: GPUBindGroup
  private annotationUniform!: GPUBuffer
  private annotationBindings!: GPUBindGroup
  private uniformBuffer!: GPUBuffer
  private uniformValues = new Float32Array(52)
  private readonly sun = solarPosition(new Date(performance.timeOrigin))
  private lighting!: TerrainLighting
  private post!: PostProcess
  private cursor: { x: number; y: number } | null = null
  private scoreMap: ScoreTextures | null = null
  private bankSize = 256
  private earthTexture!: GPUTexture
  private earthImage: HTMLImageElement | null = null
  private terrainTexture!: GPUTexture
  private terrainRequest = new AbortController()
  private terrain: Terrain | null = null
  private poleHeights: [number,number] = [0,0]
  private terrainLevels: ReturnType<typeof elevationPyramid> = []
  private vertexBuffer!: GPUBuffer
  private indexBuffer!: GPUBuffer
  private geographicSampler!: GPUSampler
  private mesh = createMesh()
  private picker = new SurfacePicker(this.mesh)
  private terrainHeight = (longitude: number, latitude: number) => Math.abs(latitude) === Math.PI / 2 ? this.poleHeights[latitude > 0 ? 1 : 0] : elevationAt(this.terrain!,longitude,latitude)
  private dataset: WorldNoise | null = null
  private computed: ComputedSource | null = null
  private pendingComputed: AbortController | null = null
  private pendingDataset: DatasetJob | null = null
  private distribution: ScoreDistribution | null = null
  private frame = 0
  private overlayDirty = false
  private displayDirty = false
  private annotationsDirty = false
  private benchmarking = false
  private timer: GpuTimer | null = null
  private timingSample: Promise<GpuTimings> | null = null
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
  readonly options: Options = { dataBoundary: false, contextFloor: .1, contextOpacity: .4, minimumScore: 0, colorMode: 'auto', focus: .22, morph: 1, base: .31, emphasis: 1, topRange: .1, emission: 1.5, opacity: .76, exposure: 1, bloom: .18, hdr: true, relief: 20, contourInterval: 500, depthInterval: 1000, contourOpacity: .45, grid: false, peaks: true, imagery: false }

  private style = resolveScoreStyle(null,this.options)
  get scoreStyle() { return this.style }
  get matchingScoreCount() { return this.distribution?.atLeast(this.options.minimumScore) ?? 0 }
  /** Load and dispatch Earth Score on this exact device. Device recovery replaces it. */
  get computeDevice(): GPUDevice | null { return this.lost || this.destroyed ? null : this.device }

  private constructor(private canvas: HTMLCanvasElement, private onPick: (selection: Selection | null, score: number | null) => void, private onError: (error: Error) => void, private onStatus: (status: string) => void, private transparent: boolean, private solarSettings?: SolarSettings, private acquireDevice = createSurfaceDevicePool(), assets?: SurfaceAssets, private onDevice?: RendererSettings['onDevice']) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('Could not create the WebGPU surface.')
    this.context = context
    const earthImage = new Image()
    earthImage.onload = () => {
      if (this.destroyed) return
      this.earthImage = earthImage
      if (!this.lost) this.uploadEarth()
      this.invalidate(false)
    }
    earthImage.onerror = () => {
      if (!this.destroyed) this.onError(new Error('Could not load the Earth background. Reload to retry.'))
    }
    const { signal } = this.terrainRequest
    const assetError = (error: unknown) => {
      if (!signal.aborted) this.onError(error instanceof Error ? error : new Error(String(error)))
    }
    const terrain = assets
      ? assets('textures/earth-terrain.bin', signal).then(url => loadTerrain(url, signal))
      : loadTerrain(`${import.meta.env.BASE_URL}textures/earth-terrain.bin?v=f370977a579e35f0-bytes1`, signal)
    if (assets) {
      void assets('textures/earth-blue-marble.jpg', signal).then(url => {
        if (signal.aborted) return
        earthImage.crossOrigin = 'anonymous'
        earthImage.src = url
      }).catch(assetError)
    } else {
      earthImage.src = `${import.meta.env.BASE_URL}textures/earth-blue-marble.jpg`
    }
    void terrain.then(terrain => {
      if (this.destroyed) return
      this.terrain = terrain
      this.poleHeights = poleElevations(this.terrain)
      this.terrainLevels = elevationPyramid(this.terrain)
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
    window.addEventListener('blur', this.pointerBlur)
    this.invalidate()
  }

  static async create(canvas: HTMLCanvasElement,onPick: (selection: Selection | null,score: number | null) => void,onError: (error: Error) => void,onStatus: (status: string) => void,{ transparent = false,signal,initialView,solar,acquireDevice,assets,snapshot,onDevice }: RendererSettings = {}) {
    signal?.throwIfAborted()
    const renderer = new SurfaceRenderer(canvas,onPick,onError,onStatus,transparent,solar,acquireDevice,assets,onDevice)
    if (initialView) renderer.rotation = viewRotation(initialView.longitude * Math.PI / 180,initialView.latitude * Math.PI / 180)
    const cancel = () => renderer.destroy()
    signal?.addEventListener('abort',cancel,{ once: true })
    if (snapshot) {
      const { signal } = renderer.terrainRequest
      void snapshot(signal).then(blob => {
        if (!signal.aborted && !renderer.computed && !renderer.pendingComputed) renderer.loadSnapshot(blob)
      }).catch(error => {
        if (!signal.aborted) onError(error instanceof Error ? error : new Error(String(error)))
      })
    }
    try { await renderer.initialize(); signal?.throwIfAborted(); return renderer }
    catch (error) { renderer.destroy(); throw error }
    finally { signal?.removeEventListener('abort',cancel) }
  }

  private async initialize(restoreScores = false) {
    const lease = this.deviceLease = this.acquireDevice()
    const { adapter, device } = await lease.ready
    if (this.destroyed || this.deviceLease !== lease) return
    const filterable = device.features.has('float32-filterable')
    this.canvas.dataset.adapter = [adapter.info.vendor,adapter.info.architecture,adapter.info.device,adapter.info.description].filter(Boolean).join(' · ')
    this.canvas.dataset.timestampQuery = String(device.features.has('timestamp-query'))
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
    const vf = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
    this.layout = device.createBindGroupLayout({ label: 'Surface resources',entries: [
      { binding: 0,visibility: vf,buffer: { type: 'uniform' } },
      { binding: 1,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'uint' } },
      { binding: 2,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'float' } },
      { binding: 3,visibility: vf,texture: { sampleType: filterable ? 'float' : 'unfilterable-float' } },
      { binding: 4,visibility: vf,sampler: { type: 'filtering' } },
      { binding: 5,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'float' } },
      ...Array.from({ length: banks },(_,i) => ({ binding: 6 + i,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'uint' as const,viewDimension: '2d-array' as const } })),
    ] })
    const module = await shaderModule(device,'Earth surface WGSL',surfaceShader(banks,this.bankSize,filterable))
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] })
    const reverseDepth = Boolean(this.solarSettings)
    const pipeline = (label: string,entryPoint: string,targets: GPUColorTargetState[],count = 1): GPURenderPipelineDescriptor => ({
      label,layout,
      vertex: { module,entryPoint: 'surfaceVertex',buffers: [{ arrayStride: 8,attributes: [{ shaderLocation: 0,offset: 0,format: 'float32x2' }] }] },
      fragment: { module,entryPoint,targets },
      primitive: { topology: 'triangle-list',cullMode: 'none' },
      depthStencil: { format: reverseDepth ? 'depth32float' : 'depth24plus',depthWriteEnabled: true,depthCompare: reverseDepth ? 'greater' : 'less' },multisample: { count },
    })
    const [surface,annotation] = await Promise.all([
      device.createRenderPipelineAsync(pipeline('4x MSAA Earth surface','surfaceFragment',[{ format: 'rgba16float' }],4)),
      device.createRenderPipelineAsync(pipeline('Geographic annotations','annotationFragment',[{ format: 'rg32float' }])),
    ])
    if (this.destroyed || this.device !== device) return
    this.pipelines = { surface,annotation,coverage: null,coverageDescriptor: pipeline('Geographic annotations with coverage','coverageAnnotationFragment',[{ format: 'rg32float' },{ format: 'r8unorm' }]) }
    this.post = new PostProcess(device,this.format,this.transparent,reverseDepth)
    if (this.solarSettings) {
      const solar = await this.solarSettings.createScene(device,new Date(this.sun.timestamp),() => this.invalidate(),this.onError,this.solarCamera,this.solarSettings.assetUrl)
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
    this.vertexBuffer = uploadBuffer(device,'Surface UV vertices',this.mesh.uv,GPUBufferUsage.VERTEX)
    this.indexBuffer = uploadBuffer(device,'Surface indices',this.mesh.indices,GPUBufferUsage.INDEX)
    this.geographicSampler = device.createSampler({ addressModeU: 'repeat',addressModeV: 'clamp-to-edge',magFilter: 'linear',minFilter: 'linear',mipmapFilter: 'linear',maxAnisotropy: 16 })
    this.scoreMap = createScoreTextures(device,this.bankSize,banks,this.dataset?.tiles.length ?? 0)
    writeScoreLookup(device,this.scoreMap,new Uint16Array(36 * 18).fill(MISSING))
    this.earthTexture = device.createTexture({ label: 'Earth imagery placeholder',size: [1,1],format: 'rgba8unorm',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    device.queue.writeTexture({ texture: this.earthTexture },new Uint8Array([18,30,42,255]),{},[1,1])
    this.terrainTexture = device.createTexture({ label: 'Terrain placeholder',size: [1,1],format: 'rg32float',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    device.queue.writeTexture({ texture: this.terrainTexture },new Float32Array([0,32]),{},[1,1])
    this.canvas.dataset.sunTime = this.sun.timestamp
    this.canvas.dataset.sunLatitude = (sunlight ? Math.asin(sunlight[1]) * 180 / Math.PI : this.sun.latitude).toFixed(6)
    this.canvas.dataset.sunLongitude = (sunlight ? Math.atan2(sunlight[0],sunlight[2]) * 180 / Math.PI : this.sun.longitude).toFixed(6)
    this.canvas.dataset.backend = 'webgpu'
    this.canvas.dataset.hdr = 'float16'
    this.canvas.dataset.terrainPrecision = 'float32'
    this.canvas.dataset.terrainFiltering = filterable ? 'hardware' : 'shader'
    if (this.dataset) this.upload()
    if (this.terrain) this.uploadTerrain()
    if (this.earthImage) this.uploadEarth()
    this.bindResources()
    this.lost = false
    this.canvas.dispatchEvent(new Event('computedevicechange'))
    this.canvas.classList.add('custom-cursor')
    if (restoreScores && !this.distribution) {
      if (this.dataset) this.publishCpuScoreInfo(this.dataset)
      else if (!this.pendingDataset) this.onStatus('WebGPU recovered. Run the live query again to restore the score map.')
    }
    this.invalidate()
  }

  private bindResources() {
    const map = this.scoreMap
    if (!this.uniformBuffer || !map) return
    const bind = (buffer: GPUBuffer) => this.device.createBindGroup({ label: 'Earth surface bindings',layout: this.layout,entries: [
      { binding: 0,resource: { buffer } },{ binding: 1,resource: map.lookup.createView() },
      { binding: 2,resource: this.earthTexture.createView() },{ binding: 3,resource: this.terrainTexture.createView() },
      { binding: 4,resource: this.geographicSampler },{ binding: 5,resource: this.lighting.texture.createView() },...map.banks.map((texture,i) => ({ binding: 6 + i,resource: texture.createView({ dimension: '2d-array' }) })),
    ] })
    this.bindings = bind(this.uniformBuffer)
    this.annotationBindings = bind(this.annotationUniform)
  }

  private uploadTerrain() {
    const first = this.terrainLevels[0]
    this.terrainTexture.destroy()
    this.terrainTexture = this.device.createTexture({ label: 'Numeric elevation and coast distance',size: [first.width,first.height],mipLevelCount: this.terrainLevels.length,format: 'rg32float',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    this.terrainLevels.forEach((level,index) => {
      const data = new Float32Array(level.values.length * 2)
      for (let i = 0; i < level.values.length; i++) { data[i * 2] = level.values[i]; data[i * 2 + 1] = level.coast[i] }
      this.device.queue.writeTexture({ texture: this.terrainTexture,mipLevel: index },data,{ bytesPerRow: level.width * 8 },[level.width,level.height])
    })
    this.lighting.setTerrain(this.terrainTexture)
    this.bindResources()
    this.canvas.dataset.terrainReady = 'true'
  }

  private uploadEarth() {
    const image = this.earthImage!,device = this.device
    this.earthTexture.destroy()
    this.earthTexture = device.createTexture({ label: 'NASA Earth imagery',size: [image.naturalWidth,image.naturalHeight],mipLevelCount: Math.floor(Math.log2(Math.max(image.naturalWidth,image.naturalHeight))) + 1,format: 'rgba8unorm',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT })
    device.queue.copyExternalImageToTexture({ source: image },{ texture: this.earthTexture },[image.naturalWidth,image.naturalHeight])
    // Generate image mips directly on the GPU. Elevation/score mips retain their
    // own numeric averaging/max rules and never go through this image filter.
    const module = device.createShaderModule({ code: `${fullscreenVertex}
      @group(0) @binding(0) var source: texture_2d<f32>;
      @group(0) @binding(1) var s: sampler;
      @fragment fn mip(input: ScreenVertex) -> @location(0) vec4f { return textureSampleLevel(source,s,input.uv,0.); }
    ` })
    const pipeline = device.createRenderPipeline({ layout: 'auto',vertex: { module,entryPoint: 'screenVertex' },fragment: { module,entryPoint: 'mip',targets: [{ format: 'rgba8unorm' }] } })
    const sampler = device.createSampler({ minFilter: 'linear',magFilter: 'linear' })
    const encoder = device.createCommandEncoder({ label: 'Earth imagery mipmaps' })
    for (let level = 1; level < this.earthTexture.mipLevelCount; level++) {
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),entries: [{ binding: 0,resource: this.earthTexture.createView({ baseMipLevel: level - 1,mipLevelCount: 1 }) },{ binding: 1,resource: sampler }] })
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: this.earthTexture.createView({ baseMipLevel: level,mipLevelCount: 1 }),loadOp: 'clear',storeOp: 'store' }] })
      pass.setPipeline(pipeline); pass.setBindGroup(0,group); pass.draw(3); pass.end()
    }
    device.queue.submit([encoder.finish()])
    this.bindResources()
    this.canvas.dataset.earthReady = 'true'
  }

  regenerate(frequency: number, seed: number) {
    this.startDataset({ kind: 'noise',frequency,seed })
  }

  loadSnapshot(source: string | Blob) {
    this.startDataset({ kind: 'snapshot',source: typeof source === 'string' ? new URL(source,window.location.href).href : source })
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
    try {
      const { textures,histogram,bestAddress,bytes } = await prepareComputedScores(device,earth,this.bankSize,this.scoreMap!.banks.length,active)
      if (active.aborted || this.computeDevice !== device) {
        destroyScoreTextures(textures)
        active.throwIfAborted()
        throw new Error('The WebGPU device changed while preparing scores.')
      }
      this.stopDataset()
      const previous = this.scoreMap!
      this.scoreMap = textures
      this.computed = { layout: earth.layout,bestAddress }
      this.bindResources()
      destroyScoreTextures(previous)
      const entries = earth.layout.rows
      const distribution = new ScoreDistribution(histogram)
      this.publishScoreInfo('computed',{ name,description },distribution,entries,bytes)
      this.invalidate()
      this.onStatus(`Live query · ${name}\n${description}\n4 × 4 km · ${distribution.valid.toLocaleString()} valid cells\n${earth.layout.sites.length} × ${SIZE}² tiles · ${(bytes / 2 ** 20).toFixed(1)} MiB GPU incl. LODs\nPrepared in ${((performance.now() - started) / 1000).toFixed(1)} s (WebGPU)`)
    } finally {
      if (this.pendingComputed === request) this.pendingComputed = null
    }
  }

  private stopDataset() {
    const job = this.pendingDataset
    this.pendingDataset = null
    job?.worker?.terminate()
    job?.controller.abort()
  }

  private startDataset(request: DataRequest) {
    this.pendingComputed?.abort()
    this.stopDataset()
    const job: DatasetJob = { controller: new AbortController() }
    this.pendingDataset = job
    const action = request.kind === 'snapshot' ? 'Loading real 4 km scores' : 'Generating 4 km noise'
    this.onStatus(`${action}…`)
    const active = () => this.pendingDataset === job && !this.destroyed
    const progress = (done: number,total: number) => {
      if (active()) this.onStatus(`${action} · ${done} / ${total} tiles…`)
    }
    const fail = (error: unknown) => {
      if (!active()) return
      this.stopDataset()
      this.onError(error instanceof Error ? error : new Error(String(error)))
    }
    const complete = (dataset: WorldNoise) => {
      if (!active()) return
      this.stopDataset()
      try {
        this.dataset = dataset
        if (!this.lost) this.upload()
        this.computed = null
        this.publishCpuScoreInfo(dataset)
        this.invalidate()
      } catch (error) {
        this.onError(error instanceof Error ? error : new Error(String(error)))
      }
    }
    const fallback = () => {
      job.worker?.terminate()
      job.worker = undefined
      void prepareDataset(request,progress,job.controller.signal).then(complete,fail)
    }
    let worker: Worker
    try {
      worker = new DatasetWorker()
    } catch {
      fallback()
      return
    }
    job.worker = worker
    worker.onmessage = ({ data }: MessageEvent<DatasetMessage>) => {
      if (!active() || job.worker !== worker) return
      switch (data.kind) {
        case 'progress':
          progress(data.done,data.total)
          break
        case 'complete':
          complete(data.dataset)
          break
        case 'error':
          fail(new Error(data.message))
          break
      }
    }
    worker.onerror = (event) => {
      event.preventDefault()
      if (!active() || job.worker !== worker) return
      fallback()
    }
    worker.onmessageerror = () => {
      if (active() && job.worker === worker) fallback()
    }
    try { worker.postMessage(request) } catch { fallback() }
  }

  private publishScoreInfo(source: 'snapshot' | 'noise' | 'computed', score: ScoreDescription, distribution: ScoreDistribution, entries: number, bytes: number) {
    this.distribution = distribution
    this.style = resolveScoreStyle(distribution,this.options,this.style)
    this.selected = null
    Object.assign(this.canvas.dataset,{
      source,cells: String(distribution.valid),entries: String(entries),scoreBytes: String(entries * 2),gpuScoreBytes: String(bytes),
      bestScore: String(distribution.best),scoreName: score.name,scoreDescription: score.description,
    })
    this.onPick(null,null)
  }

  private publishCpuScoreInfo(dataset: WorldNoise) {
    const entries = dataset.tiles.length * SIZE * SIZE
    const { snapshot } = dataset
    const score = snapshot ?? { name: 'Synthetic Perlin noise',description: 'Synthetic noise on the same native 4 km score grid.' }
    this.publishScoreInfo(snapshot ? 'snapshot' : 'noise',score,new ScoreDistribution(dataset.histogram),entries,dataset.bytes)
    this.onStatus(`${snapshot ? `Real data · ${snapshot.name}\n${snapshot.description}\nCaptured ${snapshot.fetchedAt.slice(0,10)}\n` : 'Synthetic Perlin noise\n'}4 × 4 km · ${dataset.valid.toLocaleString()} valid cells\n${dataset.tiles.length} × ${SIZE}² tiles · ${(entries * 2 / 1e6).toFixed(1)} MB uint16\n${(dataset.bytes / 2 ** 20).toFixed(1)} MiB incl. LODs per CPU / GPU copy\n${snapshot ? 'Prepared' : 'Generated'} in ${(dataset.milliseconds / 1000).toFixed(1)} s`)
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

  private upload() {
    const dataset = this.dataset!, previous = this.scoreMap!
    const tileLayers = previous.banks.reduce((total,texture) => total + texture.depthOrArrayLayers,0)
    const map = tileLayers < dataset.tiles.length ? createScoreTextures(this.device,this.bankSize,previous.banks.length,dataset.tiles.length) : previous
    dataset.tiles.forEach((tile,index) => {
      const bank = Math.floor(index / this.bankSize),layer = index % this.bankSize
      tile.levels.forEach((level,lod) => this.device.queue.writeTexture({ texture: map.banks[bank],mipLevel: lod,origin: [0,0,layer] },level.scores,{ bytesPerRow: level.width * 2 },[level.width,level.height,1]))
    })
    writeScoreLookup(this.device,map,dataset.lookup)
    if (map !== previous) {
      this.scoreMap = map
      this.bindResources()
      destroyScoreTextures(previous)
    }
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

  resetCamera() { this.cameraHistory = undefined; this.rotation = [0,0,0,1]; this.center = [0,0,0]; this.grab = null; this.zoom = 1; this.invalidate() }

  focusSolarBody(focus: SolarFocus) {
    if (!this.solarSettings || this.solarFocus === focus) return
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
    this.invalidate()
  }

  setSourcePoint(body: SourceBody,point: SourcePoint) {
    this.sourcePoints[body] = point
    this.invalidate(false)
  }

  retrySolarTerrain() { this.solar?.loadTerrain() }

  setProjection(projection: 'globe' | 'plane') {
    if (this.solarCamera?.transitioning) {
      this.pendingSolarProjection = projection
      return
    }
    const targetMorph = Number(projection === 'globe')
    if (this.options.morph === targetMorph && !this.projectionFrame) return
    cancelAnimationFrame(this.projectionFrame)
    if (this.options.morph === 1) this.globeView = { rotation: this.rotation, zoom: this.zoom }
    const target = projection === 'globe' && this.globeView ? this.globeView : { rotation: [0,0,0,1] as Rotation, zoom: 1 }
    const from = { morph: this.options.morph, rotation: this.rotation, zoom: this.zoom }
    const duration = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 480
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
    if (!this.distribution?.valid || this.distribution.best < this.options.minimumScore) return
    if (this.computed) {
      const { bestAddress,layout } = this.computed
      const center = bestAddress === null ? null : layout.locate(bestAddress)?.center
      if (center) this.focusScore(center.longitude * Math.PI / 180,center.latitude * Math.PI / 180)
    } else if (this.dataset) {
      for (const tile of this.dataset.tiles) {
        const index = tile.levels[0].scores.indexOf(this.dataset.best)
        if (index < 0) continue
        const { latitude,cos } = tileRow(tile.v,Math.floor(index / SIZE))
        this.focusScore(tileX(tile.h,index % SIZE) / (RADIUS * cos),latitude)
        break
      }
    }
  }

  private focusScore(longitude: number, latitude: number) {
    this.cameraHistory = undefined
    this.rotation = viewRotation(longitude,latitude)
    this.center = [0,0,0]; this.grab = null; this.zoom = 12
    const elevation = this.terrain ? this.terrainHeight(longitude,latitude) : 0
    this.selected = { longitude: longitude * 180 / Math.PI,latitude: latitude * 180 / Math.PI,elevation,position: surfacePoint(longitude,latitude,1,elevation,this.options.relief) }
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
    const halfWidth = Math.PI * (1 - morph) + morph
    const fit = Math.max(1, halfWidth / aspect) / Math.tan(Math.PI / 8) + 1.4
    const gap = (fit - morph) / this.zoom
    const clearance = this.solarFocus === 'earth' ? 9000 / EARTH_AXIS * this.options.relief : .035
    return { yaw: 0, pitch: 0, rotation: this.rotation, center: this.center, distance: morph + clearance + gap, aspect, near: Math.max(.000001, Math.min(.05, gap * .08)) }
  }

  private schedule() {
    this.canvas.dataset.idle = 'false'
    if (this.frame || this.rendering || this.lost || this.destroyed || this.benchmarking) return
    this.frame = requestAnimationFrame(this.renderFrame)
  }

  private invalidate(annotations = true) {
    this.dirty = true
    this.annotationsDirty ||= annotations
    this.displayDirty = true
    this.schedule()
  }

  private invalidateDisplay() { this.displayDirty = true; this.schedule() }
  private invalidateOverlay() { this.overlayDirty = true; this.schedule() }

  private renderFrame = async () => {
    this.frame = 0
    if (this.lost || this.destroyed || this.rendering || this.benchmarking) return
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
    const width = Math.max(1,Math.round(canvas.clientWidth * ratio)),height = Math.max(1,Math.round(canvas.clientHeight * ratio))
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
    const hover = cursor ? this.pickAt(cursor.x,cursor.y) : null
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
      u.set([Number(Boolean(this.terrain)),Number(Boolean(sceneFrame)),0,0],44)
      u.set([...this.poleHeights,0,0],48)
      if (redraw) device.queue.writeBuffer(this.uniformBuffer,0,u)
      if (annotations) device.queue.writeBuffer(this.annotationUniform,0,u)
    }
    this.timer?.beginFrame()
    const encoder = device.createCommandEncoder({ label: 'Surface lab frame' })
    const lighting = this.lighting.encode(encoder,o.relief)
    const drawMesh = (pass: GPURenderPassEncoder,pipeline: GPURenderPipeline,bindings: GPUBindGroup) => {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0,bindings)
      pass.setVertexBuffer(0,this.vertexBuffer); pass.setIndexBuffer(this.indexBuffer,'uint32')
      pass.drawIndexed(this.mesh.indices.length)
    }
    const pipelines = this.pipelines!
    if (redraw) {
      const pass = this.post.begin(encoder,hdr,this.timer?.writes('surface'),Boolean(motion))
      if (sceneFrame) {
        const overview = this.solarFocus === 'system'
        const flightTarget = this.solarFocus !== 'system' && solar!.camera.flying ? this.solarFocus : undefined
        solar!.draw(pass,sceneFrame,width,height,this.sourcePoints,overview,flightTarget)
      }
      if (renderEarth) drawMesh(pass,pipelines.surface,this.bindings)
      if (sceneFrame) solar!.drawOrbits(pass,sceneFrame,this.solarFocus === 'system')
      if (sceneFrame) solar!.drawHalo(pass)
      pass.end()
    }
    if (annotations) {
      const pipeline = o.dataBoundary
        ? pipelines.coverage ??= device.createRenderPipeline(pipelines.coverageDescriptor)
        : pipelines.annotation
      const pass = this.post.beginAnnotations(encoder,o.dataBoundary,this.timer?.writes('annotations'))
      if (renderEarth) drawMesh(pass,pipeline,this.annotationBindings)
      pass.end()
    }
    let spaceCursor: [number,number] | null = null
    if (cursor && !hover) {
      const rect = canvas.getBoundingClientRect()
      spaceCursor = [(cursor.x - rect.left) * width / rect.width,(cursor.y - rect.top) * height / rect.height]
    }
    const markers = [(this.selected?.longitude ?? 0) / 360 + .5,(this.selected?.latitude ?? 0) / 180 + .5,
      (hover?.longitude ?? 0) / 360 + .5,(hover?.latitude ?? 0) / 180 + .5,
      Number(Boolean(this.selected && this.solarFocus === 'earth' && !solar?.camera.flying)),Number(Boolean(hover)),ease(o.morph),metresPerPixel / EARTH_AXIS]
    const built = this.post.finish(encoder,this.context.getCurrentTexture().createView(),{
      hdr,exposure: o.exposure,glow: o.bloom,ratio,dataBoundary: o.dataBoundary,cursor: spaceCursor,markers,motion,
    },redraw,display || annotations,this.timer ?? undefined)
    this.timer?.encode(encoder)
    device.queue.submit([encoder.finish()])
    if (redraw || !prior) this.cameraHistory = stableCamera && { camera: stableCamera,time: frameTime,flying }
    // Input may stop after this frame. Restore the sharp cached scene once,
    // sharing the next frame with any movement that has arrived in the meantime.
    if (built.motion && !flying) this.invalidateDisplay()
    if (this.timer) this.timingSample = this.timer.read()
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
        canvas.dataset.frameSource = renderEarth && this.earthImage && this.terrain &&
          solar.skySettled && solar.terrainReady ? canvas.dataset.source ?? '' : ''
      }
    }
    if (built.bloom) canvas.dataset.bloomBuilds = String(Number(canvas.dataset.bloomBuilds || 0) + 1)
    if (built.composition) canvas.dataset.displayBuilds = String(Number(canvas.dataset.displayBuilds || 0) + 1)
    canvas.dataset.idle = String(!this.dirty && !this.annotationsDirty && !this.displayDirty && !this.overlayDirty)
    canvas.dataset.cursor = cursor ? hover ? 'surface' : 'space' : 'hidden'
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

  async benchmark(onProgress: (message: string) => void = () => {},samples = 60): Promise<TimingResult[]> {
    if (this.benchmarking) throw new Error('GPU profiling is already running.')
    if (!this.distribution || !this.terrain || this.lost) throw new Error('Wait for scores and terrain to finish loading.')
    if (!this.device.features.has('timestamp-query')) throw new Error('This browser does not expose GPU timestamp queries.')
    this.benchmarking = true
    cancelAnimationFrame(this.frame); this.frame = 0
    while (this.rendering) await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
    const rotation = this.rotation,cursor = this.cursor,exposure = this.options.exposure,glow = this.options.bloom
    const device = this.device,results: TimingResult[] = []
    this.timer = new GpuTimer(device)
    const median = (values: number[],fraction = .5) => [...values].sort((a,b) => a - b)[Math.min(values.length - 1,Math.floor(values.length * fraction))]
    try {
      for (const activity of ['orbit','hover','exposure','bloom'] as const) {
        this.rotation = rotation; this.options.exposure = exposure; this.options.bloom = glow; this.cursor = null
        this.render(true,true,true); await this.timingSample
        const times: GpuTimings[] = []
        for (let i = 0; i < samples + 24; i++) {
          if (this.destroyed || this.lost || this.device !== device) throw new Error('GPU profiling interrupted by device loss.')
          if (activity === 'orbit') this.rotation = orbitRotation(rotation,i * .001,0)
          else if (activity === 'hover') {
            const rect = this.canvas.getBoundingClientRect()
            this.cursor = { x: rect.left + rect.width * .5 + 20 * Math.sin(i * .1),y: rect.top + rect.height * .5 }
          } else if (activity === 'exposure') this.options.exposure = exposure * (.9 + .2 * (i % 12) / 11)
          else this.options.bloom = Math.max(.001,glow) * (.9 + .2 * (i % 12) / 11)
          onProgress(`${activity} · ${i < 24 ? 'warming up' : `${i - 23}/${samples}`}`)
          if (!this.render(activity === 'orbit',activity === 'orbit',activity !== 'hover')) throw new Error('GPU profiling interrupted.')
          const timing = await this.timingSample!
          if (i >= 24) times.push(timing)
        }
        const stages = Object.fromEntries([...GPU_STAGES,'total'].map(stage => [stage,median(times.map(t => t[stage as keyof GpuTimings]))])) as GpuTimings
        results.push({ activity,median: stages.total,p95: median(times.map(t => t.total),.95),stages,samples: times.length })
      }
      return results
    } finally {
      this.timer?.destroy(); this.timer = null; this.timingSample = null
      this.options.exposure = exposure; this.options.bloom = glow; this.rotation = rotation; this.cursor = cursor
      this.benchmarking = false
      this.invalidate()
    }
  }

  private trackCursor(event: PointerEvent) {
    if (event.pointerType === 'touch') this.cursor = null
    else if (this.cursor) { this.cursor.x = event.clientX; this.cursor.y = event.clientY }
    else this.cursor = { x: event.clientX, y: event.clientY }
    this.invalidateOverlay()
  }

  private pointerLeave = () => {
    this.cursor = null
    this.invalidateOverlay()
  }

  private pointerBlur = () => {
    this.cameraHistory = undefined
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

  private regrab() {
    const pointer = this.pointers.size === 1 ? [...this.pointers.values()][0] : null
    this.grab = pointer ? { ...pointer, camera: this.camera(), position: this.pickAt(pointer.x,pointer.y)?.position ?? null } : null
  }

  private pointerDown = (event: PointerEvent) => {
    if (this.benchmarking || this.solar?.camera.flying) return
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
    if (this.benchmarking) return
    this.trackCursor(event)
    const previous = this.pointers.get(event.pointerId)
    if (!previous) return
    if (this.gesture && Math.hypot(event.clientX - this.gesture.startX, event.clientY - this.gesture.startY) > 4) this.gesture.moved = true
    if (this.pointers.size > 1) {
      const other = [...this.pointers.entries()].find(([id]) => id !== event.pointerId)![1]
      const old = Math.hypot(previous.x - other.x, previous.y - other.y)
      const next = Math.hypot(event.clientX - other.x, event.clientY - other.y)
      if (old > 0) this.setZoom(this.zoom * next / old)
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
    } else {
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
    if (this.grab?.position) {
      const [nx,ny] = this.pointerCoordinates(pending.x,pending.y)
      if (this.options.morph === 1) this.rotation = grabRotation(camera,this.grab.position,nx,ny)
      else this.center = grabCenter(camera,this.grab.position,nx,ny)
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
    const [nx,ny] = this.pointerCoordinates(clientX,clientY)
    const point = this.solar?.pickSource(frame,body,nx,ny)
    if (point) return { body, ...point }
  }

  private pointerUp = (event: PointerEvent) => {
    if (this.benchmarking) return
    this.trackCursor(event)
    this.applyPendingMove()
    if (!this.pointers.has(event.pointerId)) return
    if (this.pointers.size === 1 && this.gesture && !this.gesture.moved) {
      if (this.solarFocus === 'earth') {
        this.selected = this.pickAt(event.clientX,event.clientY)
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
    if (!selection) {
      this.onPick(null,null)
    } else if (this.computed) {
      void this.reportComputedPick(selection,this.computed)
    } else {
      const cell = cellAddress(selection.longitude,selection.latitude)
      const slot = cell && this.dataset ? this.dataset.lookup[cell.v * 36 + cell.h] : MISSING
      const code = cell && slot !== MISSING ? this.dataset!.tiles[slot].levels[0].scores[cell.y * SIZE + cell.x] : MISSING
      this.onPick(selection,code === MISSING ? null : code / 65534)
    }
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
    if (this.benchmarking) return
    this.cursor = null
    this.invalidate()
    this.applyPendingMove()
    this.pointers.delete(event.pointerId)
    this.regrab()
    if (this.gesture) this.gesture.moved = true
    if (!this.pointers.size) this.gesture = null
  }

  private wheel = (event: WheelEvent) => {
    if (this.benchmarking) return
    this.applyPendingMove()
    event.preventDefault()
    if (this.solar?.camera.flying) return
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.canvas.clientHeight : 1)
    this.setZoom(this.zoom * Math.exp(-delta * .001))
    this.regrab()
    this.invalidate()
  }

  private keyDown = (event: KeyboardEvent) => {
    if (this.benchmarking || this.solar?.camera.flying) return
    this.applyPendingMove()
    if (event.key === 'ArrowLeft') this.rotation = orbitRotation(this.rotation,.12,0)
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
    if (this.hasSolarLabels) this.canvas.dispatchEvent(new CustomEvent('solarlabels',{ detail: [] }))
    this.hasSolarLabels = false
    this.solar?.destroy()
    this.solar = undefined
    this.earthTexture?.destroy(); this.terrainTexture?.destroy()
    if (this.scoreMap) destroyScoreTextures(this.scoreMap)
    this.scoreMap = null
    this.vertexBuffer?.destroy(); this.indexBuffer?.destroy(); this.uniformBuffer?.destroy(); this.annotationUniform?.destroy()
    this.timer?.destroy(); this.timer = null
    this.lighting?.destroy(); this.post?.destroy()
    this.pipelines = null
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.canvas.dispatchEvent(new Event('computedevicechange'))
    this.terrainRequest.abort()
    this.pendingComputed?.abort()
    this.stopDataset()
    this.dataset = null
    this.computed = null
    cancelAnimationFrame(this.projectionFrame)
    cancelAnimationFrame(this.frame); this.observer.disconnect()
    this.canvas.removeEventListener('pointerdown',this.pointerDown)
    this.canvas.removeEventListener('pointermove',this.pointerMove)
    this.canvas.removeEventListener('pointerleave',this.pointerLeave)
    this.canvas.removeEventListener('pointerup',this.pointerUp)
    this.canvas.removeEventListener('pointercancel',this.pointerCancel)
    this.canvas.removeEventListener('wheel',this.wheel)
    this.canvas.removeEventListener('keydown',this.keyDown)
    window.removeEventListener('blur',this.pointerBlur)
    this.canvas.classList.remove('custom-cursor')
    this.releaseGpuResources()
    this.context.unconfigure()
    this.deviceErrors?.abort()
    this.deviceLease?.release()
  }
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
