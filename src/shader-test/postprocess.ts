import { joinWgsl, shaderModule } from './gpu.ts'
import { motionBlurShader, type MotionBlurUniforms } from './motion-blur.ts'
import bloomWgsl from './shaders/bloom.wgsl?raw'
import colorWgsl from './shaders/color.wgsl?raw'
import compositionWgsl from './shaders/composition.wgsl?raw'
import coverageWgsl from './shaders/coverage.wgsl?raw'
import cursorWgsl from './shaders/cursor.wgsl?raw'
import equalEarthWgsl from './shaders/equal-earth.wgsl?raw'
import fullscreenWgsl from './shaders/fullscreen.wgsl?raw'
import loupeWgsl from './shaders/loupe.wgsl?raw'
import postUniformsWgsl from './shaders/post-uniforms.wgsl?raw'
import presentationWgsl from './shaders/presentation.wgsl?raw'
import reticleWgsl from './shaders/reticle.wgsl?raw'

const bloomShader = joinWgsl(fullscreenWgsl, bloomWgsl)
const compositionShader = joinWgsl(fullscreenWgsl, postUniformsWgsl, colorWgsl, coverageWgsl, compositionWgsl)
const presentationShader = joinWgsl(
  fullscreenWgsl,
  postUniformsWgsl,
  colorWgsl,
  equalEarthWgsl,
  reticleWgsl,
  presentationWgsl,
)
const loupeShader = joinWgsl(colorWgsl, loupeWgsl)

type PostPipelines = {
  bloom: GPURenderPipeline
  composition: GPURenderPipeline
  presentation: GPURenderPipeline
  cursor: GPURenderPipeline
  loupe: GPURenderPipeline
  motion?: GPURenderPipeline
  staticResolve?: GPURenderPipeline
}

const postPipelines = new WeakMap<GPUDevice, Map<string, Promise<PostPipelines>>>()

function getPostPipelines(device: GPUDevice, format: GPUTextureFormat, reverseDepth: boolean): Promise<PostPipelines> {
  let byKey = postPipelines.get(device)
  if (!byKey) {
    byKey = new Map()
    postPipelines.set(device, byKey)
  }
  const key = `${format}:${reverseDepth}`
  let pending = byKey.get(key)
  if (!pending) {
    pending = (async () => {
      const [bloom, composition, presentation, cursor, loupe, motion] = await Promise.all([
        shaderModule(device, 'Bloom WGSL', bloomShader),
        shaderModule(device, 'Composition WGSL', compositionShader),
        shaderModule(device, 'Presentation WGSL', presentationShader),
        shaderModule(device, 'Cursor WGSL', cursorWgsl),
        shaderModule(device, 'Optical loupe WGSL', loupeShader),
        reverseDepth ? shaderModule(device, 'Camera motion blur WGSL', motionBlurShader) : undefined,
      ])
      const vertex = (module: GPUShaderModule, entryPoint = 'screenVertex') => ({ module, entryPoint })
      const alphaBlend: GPUBlendState = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      }
      const [bp, cp, pp, sp, lp, mp, srp] = await Promise.all([
        device.createRenderPipelineAsync({
          label: 'HDR bloom',
          layout: 'auto',
          vertex: vertex(bloom),
          fragment: { module: bloom, entryPoint: 'bloomFragment', targets: [{ format: 'rgba16float' }] },
        }),
        device.createRenderPipelineAsync({
          label: 'Cached tone map and coverage',
          layout: 'auto',
          vertex: vertex(composition),
          fragment: { module: composition, entryPoint: 'compositeFragment', targets: [{ format }] },
        }),
        device.createRenderPipelineAsync({
          label: 'Cached image and surface markers',
          layout: 'auto',
          vertex: vertex(presentation),
          fragment: { module: presentation, entryPoint: 'presentFragment', targets: [{ format }] },
        }),
        device.createRenderPipelineAsync({
          label: 'Space cursor',
          layout: 'auto',
          vertex: vertex(cursor, 'cursorVertex'),
          fragment: { module: cursor, entryPoint: 'cursorFragment', targets: [{ format, blend: alphaBlend }] },
        }),
        device.createRenderPipelineAsync({
          label: 'Optical loupe',
          layout: 'auto',
          vertex: vertex(loupe, 'loupeVertex'),
          fragment: { module: loupe, entryPoint: 'loupeFragment', targets: [{ format, blend: alphaBlend }] },
        }),
        motion
          ? device.createRenderPipelineAsync({
              label: 'Depth-aware camera motion blur',
              layout: 'auto',
              vertex: vertex(motion),
              fragment: { module: motion, entryPoint: 'blurFragment', targets: [{ format: 'rgba16float' }] },
            })
          : undefined,
        motion
          ? device.createRenderPipelineAsync({
              label: 'Per-sample static marker and orbit resolve',
              layout: 'auto',
              vertex: vertex(motion),
              fragment: { module: motion, entryPoint: 'staticResolveFragment', targets: [{ format: 'rgba16float' }] },
            })
          : undefined,
      ])
      return { bloom: bp, composition: cp, presentation: pp, cursor: sp, loupe: lp, motion: mp, staticResolve: srp }
    })()
    byKey.set(key, pending)
    void pending.catch(() => {
      if (byKey!.get(key) === pending) byKey!.delete(key)
    })
  }
  return pending
}

export const LOUPE_TEXTURE_SIZE = 320
export type LoupeOverlay = {
  x: number
  y: number
  radiusCss: number
  aimX: number
  aimY: number
  locked: boolean
  zoom: number
  selected?: [number, number] | null
}
export type TileRect = { x: number; y: number; width: number; height: number; index: number; count: number }
export type PrepareOrbits = (
  encoder: GPUCommandEncoder,
  depthView: GPUTextureView,
  width: number,
  height: number,
) => GPUTextureView | null
export type CompositeOrbits = (
  encoder: GPUCommandEncoder,
  multisampleView: GPUTextureView,
  sceneView: GPUTextureView,
) => void
export type PostMotionAtmospheres = (
  encoder: GPUCommandEncoder,
  multisampleView: GPUTextureView,
  sceneView: GPUTextureView,
  motionView: GPUTextureView | null,
  depthView: GPUTextureView,
) => void
export type Presentation = {
  hdr: boolean
  exposure: number
  glow: number
  ratio: number
  dataBoundary: boolean
  cursor: [number, number] | null
  hoverScreen?: [number, number] | null
  hoverBody: number
  markers: number[]
  loupe?: LoupeOverlay | null
  motion?: MotionBlurUniforms
  lowRes?: boolean
  tile?: TileRect
  prepareOrbits?: PrepareOrbits
  compositeOrbits?: CompositeOrbits
  postMotionAtmospheres?: PostMotionAtmospheres
}

type PostUniformValues = {
  hoverScreen: [number, number] | null
  hdr: boolean
  exposure: number
  glow: number
  ratio: number
  dataBoundary: boolean
  displayOnlyHover: boolean
  hoverBody: number
  outWidth: number
  outHeight: number
  markers: readonly number[]
  bakedMode: 0 | 1 | 2
  hasOrbits: boolean
  bakeMarkers: boolean
}

function packPostUniforms(out: Float32Array, v: PostUniformValues) {
  // hoverScreen: (cursorX, cursorY, hasHoverScreen, hdr)
  out[0] = v.hoverScreen ? v.hoverScreen[0] : 0
  out[1] = v.hoverScreen ? v.hoverScreen[1] : 0
  out[2] = Number(Boolean(v.hoverScreen))
  out[3] = Number(v.hdr)
  // effects: (exposure, bloomGlow, pixelRatio, showDataBoundary)
  out[4] = v.exposure
  out[5] = v.hdr ? v.glow : 0
  out[6] = v.ratio
  out[7] = Number(v.dataBoundary)
  // viewport: (displayOnlyHover, hoverBody, outWidth, outHeight)
  out[8] = Number(v.displayOnlyHover)
  out[9] = v.hoverBody
  out[10] = v.outWidth
  out[11] = v.outHeight
  // hover (12..15), state (16..19), selEarth (20..23), selMoon (24..27), selMars (28..31)
  out.set(v.markers, 12)
  out[17] = v.bakedMode
  out[18] = Number(v.hasOrbits)
  out[19] = Number(v.bakeMarkers)
}

type RenderBank = {
  label: string
  width: number
  height: number
  scene: GPUTexture
  sceneView: GPUTextureView
  multisample: GPUTexture
  multisampleView: GPUTextureView
  depth: GPUTexture
  depthOnlyView: GPUTextureView
  geography: GPUTexture
  geographyView: GPUTextureView
  annotationDepth: GPUTexture
  stableCoverage: GPUTexture | null
  coverageMsaa: GPUTexture | null
  boundCoverage: GPUTexture
  coverageValid: boolean
  annotationsValid: boolean
  annotationsUpdated: boolean
  bloomValid: boolean
  hasMultisample: boolean
  lastOrbitView: GPUTextureView | null
  bloom: GPUTexture[]
  bloomGroups: GPUBindGroup[]
  compositionGroup: GPUBindGroup
  presentationGroup: GPUBindGroup
  staticResolveGroup?: GPUBindGroup
  staticResolveOrbitView?: GPUTextureView
  motionTarget?: GPUTexture
  motionTargetView?: GPUTextureView
  motionGroup?: GPUBindGroup
  motionOrbitView?: GPUTextureView
  motionBloomGroup?: GPUBindGroup
  motionCompositionGroup?: GPUBindGroup
}

export class PostProcess {
  private readonly emptyCoverage: GPUTexture
  private readonly emptyOrbitMsaa: GPUTexture
  private readonly emptyOrbitMsaaView: GPUTextureView
  private high!: RenderBank
  private low: RenderBank | null = null
  private activeBank!: RenderBank
  private display!: GPUTexture
  private displayValid = false
  private baseWidth = 0
  private baseHeight = 0
  private outWidth = 0
  private outHeight = 0
  private key = ''
  private displayedMotion = false
  private bakedStaticMarkers = false
  private lastStaticMarkers = new Float32Array(12)
  private motionPipeline?: GPURenderPipeline
  private staticResolvePipeline?: GPURenderPipeline
  private motionUniform?: GPUBuffer
  private bloomPipeline!: GPURenderPipeline
  private compositionPipeline!: GPURenderPipeline
  private presentationPipeline!: GPURenderPipeline
  private cursorPipeline!: GPURenderPipeline
  private loupePipeline!: GPURenderPipeline
  private loupeScene?: GPUTexture
  private loupeMultisample?: GPUTexture
  private loupeDepth?: GPUTexture
  private loupeUniform: GPUBuffer
  private loupeGroup?: GPUBindGroup
  private bloomUniforms: GPUBuffer[]
  private compositeUniform: GPUBuffer
  private cursorUniform: GPUBuffer
  private bloomScratch = new Float32Array(4)
  private compositeScratch = new Float32Array(32)
  private cursorScratch = new Float32Array(8)
  private loupeScratch = new Float32Array(16)
  private sampler: GPUSampler
  private cursorGroup!: GPUBindGroup

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private transparent: boolean,
    private reverseDepth = false,
  ) {
    this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' })
    this.emptyCoverage = device.createTexture({
      label: 'Unused coverage binding',
      size: [1, 1],
      format: 'rg8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    })
    this.emptyOrbitMsaa = device.createTexture({
      label: 'Unused 4x MSAA orbit binding',
      size: [1, 1],
      format: 'rgba16float',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.emptyOrbitMsaaView = this.emptyOrbitMsaa.createView()
    const uniform = (label: string, size: number) =>
      device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.bloomUniforms = [0, 1, 2].map(i => uniform(`Bloom parameters ${i}`, 16))
    this.compositeUniform = uniform('Display parameters', 128)
    this.cursorUniform = uniform('Space cursor parameters', 32)
    this.loupeUniform = uniform('Optical loupe parameters', 64)
  }

  async initialize() {
    const device = this.device
    const { bloom, composition, presentation, cursor, loupe, motion, staticResolve } = await getPostPipelines(
      device,
      this.format,
      this.reverseDepth,
    )
    this.motionPipeline = motion
    this.staticResolvePipeline = staticResolve
    if (motion)
      this.motionUniform = device.createBuffer({
        label: 'Camera motion parameters',
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    this.bloomPipeline = bloom
    this.compositionPipeline = composition
    this.presentationPipeline = presentation
    this.cursorPipeline = cursor
    this.loupePipeline = loupe
    this.cursorGroup = device.createBindGroup({
      layout: cursor.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cursorUniform } }],
    })
  }

  private target(
    label: string,
    width: number,
    height: number,
    format: GPUTextureFormat = 'rgba16float',
    sampleCount = 1,
    sampled = sampleCount === 1,
  ) {
    const usage =
      GPUTextureUsage.RENDER_ATTACHMENT |
      (sampled ? GPUTextureUsage.TEXTURE_BINDING : 0) |
      (sampleCount === 1 ? GPUTextureUsage.COPY_SRC : 0)
    return this.device.createTexture({ label, size: [width, height], format, sampleCount, usage })
  }

  private createBank(label: string, width: number, height: number): RenderBank {
    const scene = this.target(`${label} HDR scene`, width, height)
    const sceneView = scene.createView()
    const multisample = this.target(`${label} 4x MSAA HDR scene`, width, height, 'rgba16float', 4, true)
    const multisampleView = multisample.createView()
    const depthFormat = this.reverseDepth ? 'depth32float' : 'depth24plus'
    const depth = this.target(
      `${label} 4x MSAA surface depth`,
      width,
      height,
      depthFormat,
      4,
      Boolean(this.motionPipeline),
    )
    const depthOnlyView = depth.createView({ aspect: 'depth-only' })
    const geography = this.target(`${label} 4x MSAA Geographic coordinates`, width, height, 'rgba16float', 4, true)
    const geographyView = geography.createView()
    const annotationDepth = this.target(`${label} 4x MSAA Annotation depth`, width, height, depthFormat, 4, false)
    const scale = Math.min(0.5, 1024 / Math.max(width, height))
    const bloom = [0, 1].map(i =>
      this.target(`${label} Bloom ${i}`, Math.max(1, Math.ceil(width * scale)), Math.max(1, Math.ceil(height * scale))),
    )
    const bloomGroups = [scene, bloom[0], bloom[1]].map((texture, i) =>
      this.device.createBindGroup({
        layout: this.bloomPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.bloomUniforms[i] } },
          { binding: 1, resource: texture.createView() },
          { binding: 2, resource: this.sampler },
        ],
      }),
    )
    const compositionGroup = this.compositionBinding(scene, bloom[0], this.emptyCoverage)
    const presentationGroup = this.device.createBindGroup({
      layout: this.presentationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniform } },
        { binding: 1, resource: this.display.createView() },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: geographyView },
        { binding: 4, resource: multisampleView },
        { binding: 5, resource: bloom[0].createView() },
      ],
    })
    return {
      label,
      width,
      height,
      scene,
      sceneView,
      multisample,
      multisampleView,
      depth,
      depthOnlyView,
      geography,
      geographyView,
      annotationDepth,
      stableCoverage: null,
      coverageMsaa: null,
      boundCoverage: this.emptyCoverage,
      coverageValid: false,
      annotationsValid: false,
      annotationsUpdated: false,
      bloomValid: false,
      hasMultisample: false,
      lastOrbitView: null,
      bloom,
      bloomGroups,
      compositionGroup,
      presentationGroup,
    }
  }

  private destroyBank(bank: RenderBank | null | undefined) {
    if (!bank) return
    bank.scene.destroy()
    bank.multisample.destroy()
    bank.depth.destroy()
    bank.geography.destroy()
    bank.annotationDepth.destroy()
    bank.stableCoverage?.destroy()
    bank.coverageMsaa?.destroy()
    bank.bloom.forEach(texture => texture.destroy())
    bank.motionTarget?.destroy()
  }

  private ensureLowBank(): RenderBank {
    if (this.baseWidth === this.outWidth && this.baseHeight === this.outHeight) return this.high
    return (this.low ??= this.createBank('Motion', this.baseWidth, this.baseHeight))
  }

  get hasCurrentCoverage() {
    return this.activeBank.coverageValid
  }
  get isLowResActive() {
    return this.activeBank === this.low
  }
  get hasValidDisplay() {
    return this.displayValid
  }
  hasAnnotations(lowRes = false) {
    return (lowRes ? this.low?.annotationsValid : this.high?.annotationsValid) ?? false
  }

  private compositionBinding(source: GPUTexture, bloom: GPUTexture, coverage: GPUTexture) {
    return this.device.createBindGroup({
      layout: this.compositionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniform } },
        { binding: 1, resource: source.createView() },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: bloom.createView() },
        { binding: 4, resource: coverage.createView() },
      ],
    })
  }

  private bindComposition(bank: RenderBank, coverage: GPUTexture) {
    bank.boundCoverage = coverage
    bank.compositionGroup = this.compositionBinding(bank.scene, bank.bloom[0], coverage)
    bank.motionCompositionGroup =
      bank.motionTarget && this.compositionBinding(bank.motionTarget, bank.bloom[0], coverage)
  }

  private encodeStaticResolve(encoder: GPUCommandEncoder, bank: RenderBank, orbitView: GPUTextureView | null) {
    if (!this.staticResolvePipeline) return false
    const boundOrbit = orbitView ?? this.emptyOrbitMsaaView
    if (!bank.staticResolveGroup || bank.staticResolveOrbitView !== boundOrbit) {
      bank.staticResolveGroup = this.device.createBindGroup({
        layout: this.staticResolvePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: bank.multisampleView },
          { binding: 2, resource: bank.depthOnlyView },
          { binding: 3, resource: { buffer: this.compositeUniform } },
          { binding: 4, resource: bank.geographyView },
          { binding: 5, resource: boundOrbit },
        ],
      })
      bank.staticResolveOrbitView = boundOrbit
    }
    const pass = encoder.beginRenderPass({
      label: 'Per-sample static marker and orbit resolve',
      colorAttachments: [{ view: bank.sceneView, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.staticResolvePipeline)
    pass.setBindGroup(0, bank.staticResolveGroup)
    pass.draw(3)
    pass.end()
    return true
  }

  private encodeMotion(
    encoder: GPUCommandEncoder,
    bank: RenderBank,
    motion: MotionBlurUniforms | undefined,
    orbitView: GPUTextureView | null,
  ) {
    if (!motion || !this.motionPipeline || !this.motionUniform) return false
    if (!bank.motionTarget) {
      bank.motionTarget = this.target(`${bank.label} Motion-filtered HDR scene`, bank.width, bank.height)
      bank.motionTargetView = bank.motionTarget.createView()
      bank.motionBloomGroup = this.device.createBindGroup({
        layout: this.bloomPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.bloomUniforms[0] } },
          { binding: 1, resource: bank.motionTargetView },
          { binding: 2, resource: this.sampler },
        ],
      })
      bank.motionCompositionGroup = this.compositionBinding(bank.motionTarget, bank.bloom[0], bank.boundCoverage)
    }
    const boundOrbit = orbitView ?? this.emptyOrbitMsaaView
    if (!bank.motionGroup || bank.motionOrbitView !== boundOrbit) {
      bank.motionGroup = this.device.createBindGroup({
        layout: this.motionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.motionUniform } },
          { binding: 1, resource: bank.multisampleView },
          { binding: 2, resource: bank.depthOnlyView },
          { binding: 3, resource: { buffer: this.compositeUniform } },
          { binding: 4, resource: bank.geographyView },
          { binding: 5, resource: boundOrbit },
          { binding: 6, resource: bank.sceneView },
        ],
      })
      bank.motionOrbitView = boundOrbit
    }
    this.device.queue.writeBuffer(this.motionUniform, 0, motion)
    const pass = encoder.beginRenderPass({
      label: 'Camera motion blur',
      colorAttachments: [{ view: bank.motionTargetView!, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.motionPipeline)
    pass.setBindGroup(0, bank.motionGroup)
    pass.draw(3)
    pass.end()
    return true
  }

  configure(width: number, height: number, outWidth: number, outHeight: number, ratio: number) {
    const key = `${width},${height},${outWidth},${outHeight},${ratio}`
    if (key === this.key) return false
    this.releaseTargets()
    this.baseWidth = width
    this.baseHeight = height
    this.outWidth = outWidth
    this.outHeight = outHeight
    this.display = this.target('Cached display image', outWidth, outHeight, this.format)
    this.displayValid = false
    this.high = this.createBank('Native', outWidth, outHeight)
    this.activeBank = this.high
    this.key = key
    return true
  }

  beginAnnotations(encoder: GPUCommandEncoder, includeCoverage: boolean, lowRes = false) {
    const bank = lowRes ? this.ensureLowBank() : this.high
    const colorAttachments: GPURenderPassColorAttachment[] = [
      { view: bank.geographyView, clearValue: [0, 0, -1, 0], loadOp: 'clear', storeOp: 'store' },
    ]
    if (includeCoverage) {
      if (!bank.stableCoverage) {
        bank.stableCoverage = this.target(`${bank.label} Score coverage`, bank.width, bank.height, 'rg8unorm')
        bank.coverageMsaa = this.target(
          `${bank.label} 4x MSAA Score coverage`,
          bank.width,
          bank.height,
          'rg8unorm',
          4,
          false,
        )
      }
      if (bank.boundCoverage !== bank.stableCoverage) this.bindComposition(bank, bank.stableCoverage)
      colorAttachments.push({
        view: bank.coverageMsaa!.createView(),
        resolveTarget: bank.stableCoverage.createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'discard',
      })
    } else if (bank.boundCoverage !== this.emptyCoverage) {
      this.bindComposition(bank, this.emptyCoverage)
    }
    bank.coverageValid = includeCoverage
    bank.annotationsValid = true
    bank.annotationsUpdated = true
    return encoder.beginRenderPass({
      label: 'Geographic annotations',
      colorAttachments,
      depthStencilAttachment: {
        view: bank.annotationDepth.createView(),
        depthClearValue: this.reverseDepth ? 0 : 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })
  }

  beginLoupe(encoder: GPUCommandEncoder, hdr: boolean) {
    if (!this.loupeScene) {
      const size = LOUPE_TEXTURE_SIZE
      const depthFormat = this.reverseDepth ? 'depth32float' : 'depth24plus'
      this.loupeScene = this.target('Optical loupe HDR scene', size, size)
      this.loupeMultisample = this.target('Optical loupe 4x MSAA HDR', size, size, 'rgba16float', 4)
      this.loupeDepth = this.target('Optical loupe depth', size, size, depthFormat, 4, false)
      this.loupeGroup = this.device.createBindGroup({
        layout: this.loupePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.loupeUniform } },
          { binding: 1, resource: this.loupeScene.createView() },
          { binding: 2, resource: this.sampler },
        ],
      })
    }
    return encoder.beginRenderPass({
      label: 'Optical loupe mini-pass',
      colorAttachments: [
        {
          view: this.loupeMultisample!.createView(),
          resolveTarget: this.loupeScene.createView(),
          clearValue: hdr ? [0.018 / 12.92, 0.026 / 12.92, 0.039 / 12.92, 1] : [0.018, 0.026, 0.039, 1],
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.loupeDepth!.createView(),
        depthClearValue: this.reverseDepth ? 0 : 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })
  }

  get geography() {
    return this.activeBank.geography
  }
  get multisampleView() {
    return this.activeBank.multisampleView
  }
  get sceneView() {
    return this.activeBank.sceneView
  }
  get depthView() {
    return this.activeBank.depthOnlyView
  }

  begin(
    encoder: GPUCommandEncoder,
    hdr: boolean,
    storeDepth = false,
    keepMultisample = false,
    lowRes = false,
    tile?: TileRect,
  ) {
    const bank = lowRes ? this.ensureLowBank() : this.high
    if (lowRes && bank !== this.high) this.high.annotationsValid = false
    const firstTile = !tile || tile.index === 0
    const lastTile = !tile || tile.index === tile.count - 1
    const keepDepth = storeDepth || !lastTile || Boolean(this.staticResolvePipeline)
    bank.hasMultisample = keepMultisample || Boolean(this.staticResolvePipeline)
    const pass = encoder.beginRenderPass({
      label: '4x MSAA Earth surface',
      colorAttachments: [
        {
          view: bank.multisampleView,
          resolveTarget: lastTile ? bank.sceneView : undefined,
          clearValue: this.transparent
            ? [0, 0, 0, 0]
            : hdr
              ? [0.018 / 12.92, 0.026 / 12.92, 0.039 / 12.92, 1]
              : [0.018, 0.026, 0.039, 1],
          loadOp: firstTile ? 'clear' : 'load',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: bank.depthOnlyView,
        depthClearValue: this.reverseDepth ? 0 : 1,
        depthLoadOp: firstTile ? 'clear' : 'load',
        depthStoreOp: keepDepth ? 'store' : 'discard',
      },
    })
    if (tile) pass.setScissorRect(tile.x, tile.y, tile.width, tile.height)
    return pass
  }

  private drawOverlay(pass: GPURenderPassEncoder, p: Presentation) {
    if (p.loupe && this.loupeGroup) {
      const sel = p.loupe.selected
      const l = this.loupeScratch
      l[0] = p.loupe.x
      l[1] = p.loupe.y
      l[2] = p.loupe.radiusCss
      l[3] = p.ratio
      l[4] = this.outWidth
      l[5] = this.outHeight
      l[6] = p.exposure
      l[7] = Number(p.hdr)
      l[8] = p.loupe.aimX
      l[9] = p.loupe.aimY
      l[10] = Number(p.loupe.locked)
      l[11] = p.loupe.zoom
      l[12] = sel ? sel[0] : 0
      l[13] = sel ? sel[1] : 0
      l[14] = Number(Boolean(sel))
      l[15] = 0
      this.device.queue.writeBuffer(this.loupeUniform, 0, l)
      pass.setPipeline(this.loupePipeline)
      pass.setBindGroup(0, this.loupeGroup)
      pass.draw(6)
    } else if (p.cursor) {
      const cur = this.cursorScratch
      cur[0] = p.cursor[0]
      cur[1] = p.cursor[1]
      cur[2] = this.outWidth
      cur[3] = this.outHeight
      cur[4] = p.ratio
      cur[5] = 0
      cur[6] = 0
      cur[7] = 0
      this.device.queue.writeBuffer(this.cursorUniform, 0, cur)
      pass.setPipeline(this.cursorPipeline)
      pass.setBindGroup(0, this.cursorGroup)
      pass.draw(6)
    }
  }

  finish(
    encoder: GPUCommandEncoder,
    output: GPUTextureView,
    p: Presentation,
    redrawSurface: boolean,
    rebuildDisplay: boolean,
  ) {
    const pendingTile = redrawSurface && Boolean(p.tile && p.tile.index < p.tile.count - 1)
    if (pendingTile) {
      // Intermediate native settle tile: high-res targets are still accumulating, so keep the
      // last completed display image on screen with live hover (using display-space hover when
      // activeBank is high, or full linear hover when activeBank is the untouched low bank).
      packPostUniforms(this.compositeScratch, {
        hoverScreen: p.hoverScreen ?? null,
        hdr: p.hdr,
        exposure: p.exposure,
        glow: p.glow,
        ratio: p.ratio,
        dataBoundary: p.dataBoundary && this.activeBank.coverageValid,
        displayOnlyHover: this.activeBank === this.high,
        hoverBody: p.hoverBody,
        outWidth: this.outWidth,
        outHeight: this.outHeight,
        markers: p.markers,
        bakedMode: this.displayedMotion ? 2 : this.bakedStaticMarkers ? 1 : 0,
        hasOrbits: Boolean(this.activeBank.lastOrbitView),
        bakeMarkers: this.displayedMotion || this.bakedStaticMarkers,
      })
      this.device.queue.writeBuffer(this.compositeUniform, 0, this.compositeScratch)
      const pass = encoder.beginRenderPass({
        label: 'Present cached surface during tile settle',
        colorAttachments: [{ view: output, loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(this.presentationPipeline)
      pass.setBindGroup(0, this.activeBank.presentationGroup)
      pass.draw(3)
      this.drawOverlay(pass, p)
      pass.end()
      return { bloom: false, composition: false, motion: false }
    }
    const bank = redrawSurface ? (p.lowRes ? this.ensureLowBank() : this.high) : this.activeBank
    const bankSwitched = bank !== this.activeBank
    this.activeBank = bank
    const willBlur = redrawSurface && Boolean(p.motion && this.motionPipeline && this.motionUniform)
    const orbitView =
      redrawSurface && p.prepareOrbits ? p.prepareOrbits(encoder, bank.depthOnlyView, bank.width, bank.height) : null
    if (redrawSurface) bank.lastOrbitView = orbitView
    const activeOrbitView = redrawSurface ? orbitView : bank.lastOrbitView
    const canBakeStatic = Boolean(this.staticResolvePipeline && bank.hasMultisample)
    const hasSelMarker = p.markers[10] > 0.005 || p.markers[14] > 0.005 || p.markers[18] > 0.005
    let staticMarkersChanged = false
    for (let i = 0; i < 12; i++) {
      if (this.lastStaticMarkers[i] !== Math.fround(p.markers[8 + i])) {
        staticMarkersChanged = true
        break
      }
    }
    const willBakeStaticMarkers = !willBlur && canBakeStatic && hasSelMarker
    const sceneNeedsStaticResolve =
      !willBlur &&
      Boolean(this.staticResolvePipeline && bank.hasMultisample) &&
      (redrawSurface
        ? willBakeStaticMarkers || Boolean(activeOrbitView)
        : (this.displayedMotion && (willBakeStaticMarkers || Boolean(activeOrbitView))) ||
          this.bakedStaticMarkers !== willBakeStaticMarkers ||
          (willBakeStaticMarkers && (staticMarkersChanged || rebuildDisplay || bank.annotationsUpdated)))
    packPostUniforms(this.compositeScratch, {
      hoverScreen: p.hoverScreen ?? null,
      hdr: p.hdr,
      exposure: p.exposure,
      glow: p.glow,
      ratio: p.ratio,
      dataBoundary: p.dataBoundary && bank.coverageValid,
      displayOnlyHover: false,
      hoverBody: p.hoverBody,
      outWidth: this.outWidth,
      outHeight: this.outHeight,
      markers: p.markers,
      bakedMode: willBlur ? 2 : willBakeStaticMarkers ? 1 : 0,
      hasOrbits: Boolean(activeOrbitView),
      bakeMarkers: willBlur || willBakeStaticMarkers,
    })
    this.device.queue.writeBuffer(this.compositeUniform, 0, this.compositeScratch)
    const motion = redrawSurface && this.encodeMotion(encoder, bank, p.motion, orbitView)
    if (redrawSurface && p.postMotionAtmospheres) {
      p.postMotionAtmospheres(
        encoder,
        bank.multisampleView,
        bank.sceneView,
        motion ? bank.motionTargetView! : null,
        bank.depthOnlyView,
      )
    }
    let staticResolved = false
    if (sceneNeedsStaticResolve) {
      staticResolved = this.encodeStaticResolve(encoder, bank, activeOrbitView)
      this.bakedStaticMarkers = willBakeStaticMarkers
      for (let i = 0; i < 12; i++) this.lastStaticMarkers[i] = Math.fround(p.markers[8 + i])
    } else if (redrawSurface && !motion) {
      this.bakedStaticMarkers = false
      this.lastStaticMarkers.fill(0)
    }
    bank.annotationsUpdated = false
    if (!this.staticResolvePipeline && orbitView && p.compositeOrbits) {
      p.compositeOrbits(encoder, bank.multisampleView, bank.sceneView)
    }
    const motionChanged = motion !== this.displayedMotion
    const sceneUpdated = redrawSurface || staticResolved
    if (sceneUpdated || motionChanged) bank.bloomValid = false
    const rebuildBloom = p.hdr && p.glow > 0 && !bank.bloomValid
    if (rebuildBloom) {
      const bloomRatio = (bank.width / Math.max(1, this.outWidth)) * p.ratio
      for (let i = 0; i < 3; i++) {
        this.bloomScratch[0] = i === 0 ? 1 / bank.width : i === 1 ? (2 * bloomRatio) / bank.width : 0
        this.bloomScratch[1] = i === 0 ? 1 / bank.height : i === 2 ? (2 * bloomRatio) / bank.height : 0
        this.bloomScratch[2] = Number(i === 0)
        this.bloomScratch[3] = 0
        this.device.queue.writeBuffer(this.bloomUniforms[i], 0, this.bloomScratch)
        const pass = encoder.beginRenderPass({
          label: `Bloom pass ${i}`,
          colorAttachments: [{ view: bank.bloom[i % 2].createView(), loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(this.bloomPipeline)
        pass.setBindGroup(0, i === 0 && motion ? bank.motionBloomGroup! : bank.bloomGroups[i])
        pass.draw(3)
        pass.end()
      }
      bank.bloomValid = true
    }
    const composition =
      rebuildDisplay || sceneUpdated || rebuildBloom || motionChanged || bankSwitched || !this.displayValid
    if (composition) {
      const pass = encoder.beginRenderPass({
        label: 'Cache tone map and coverage',
        colorAttachments: [{ view: this.display.createView(), loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(this.compositionPipeline)
      pass.setBindGroup(0, motion ? bank.motionCompositionGroup! : bank.compositionGroup)
      pass.draw(3)
      pass.end()
      this.displayValid = true
    }
    const pass = encoder.beginRenderPass({
      label: 'Present cached surface and markers',
      colorAttachments: [{ view: output, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.presentationPipeline)
    pass.setBindGroup(0, bank.presentationGroup)
    pass.draw(3)
    this.drawOverlay(pass, p)
    pass.end()
    this.displayedMotion = motion
    return { bloom: rebuildBloom, composition, motion }
  }

  private releaseTargets() {
    this.destroyBank(this.high)
    this.destroyBank(this.low)
    this.low = null
    this.display?.destroy()
    this.displayValid = false
    this.displayedMotion = false
    this.bakedStaticMarkers = false
    this.lastStaticMarkers.fill(0)
  }

  suspend() {
    if (!this.key) return
    this.releaseTargets()
    this.key = ''
  }

  destroy() {
    this.suspend()
    this.loupeScene?.destroy()
    this.loupeMultisample?.destroy()
    this.loupeDepth?.destroy()
    this.loupeScene = this.loupeMultisample = this.loupeDepth = undefined
    this.loupeGroup = undefined
    this.emptyCoverage.destroy()
    this.emptyOrbitMsaa.destroy()
    this.motionUniform?.destroy()
    this.bloomUniforms.forEach(buffer => buffer.destroy())
    this.compositeUniform.destroy()
    this.cursorUniform.destroy()
    this.loupeUniform.destroy()
  }
}
