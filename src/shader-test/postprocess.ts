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
import quadWgsl from './shaders/quad.wgsl?raw'
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
const cursorShader = joinWgsl(quadWgsl, cursorWgsl)
const loupeShader = joinWgsl(colorWgsl, quadWgsl, loupeWgsl)
const backdropBlitShader = joinWgsl(
  fullscreenWgsl,
  `
@group(0) @binding(0) var backdropTex: texture_2d<f32>;

@fragment fn backdropBlitFragment(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(backdropTex, vec2i(pos.xy), 0);
}
`,
)

type PostPipelines = {
  bloom: GPURenderPipeline
  composition: GPURenderPipeline
  presentation: GPURenderPipeline
  getCursor: () => GPURenderPipeline
  getLoupe: () => GPURenderPipeline
  getMotion?: () => GPURenderPipeline
  getStaticResolve?: () => GPURenderPipeline
  backdropBlit?: GPURenderPipeline
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
      const [bloom, composition, presentation, backdropBlit] = await Promise.all([
        shaderModule(device, 'Bloom WGSL', bloomShader),
        shaderModule(device, 'Composition WGSL', compositionShader),
        shaderModule(device, 'Presentation WGSL', presentationShader),
        reverseDepth ? shaderModule(device, 'Hero backdrop blit WGSL', backdropBlitShader) : undefined,
      ])
      const vertex = (module: GPUShaderModule, entryPoint = 'screenVertex') => ({ module, entryPoint })
      const alphaBlend: GPUBlendState = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      }
      const [bp, cp, pp, bbp] = await Promise.all([
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
        backdropBlit
          ? device.createRenderPipelineAsync({
              label: 'Hero cached backdrop blit',
              layout: 'auto',
              vertex: vertex(backdropBlit),
              fragment: {
                module: backdropBlit,
                entryPoint: 'backdropBlitFragment',
                targets: [{ format: 'rgba16float' }],
              },
              multisample: { count: 4 },
              depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' },
            })
          : undefined,
      ])
      let cursorPipe: GPURenderPipeline | undefined
      let loupePipe: GPURenderPipeline | undefined
      let motionModule: GPUShaderModule | undefined
      let motionPipe: GPURenderPipeline | undefined
      let staticResolvePipe: GPURenderPipeline | undefined
      const getMotionModule = () =>
        (motionModule ??= device.createShaderModule({ label: 'Camera motion blur WGSL', code: motionBlurShader }))
      return {
        bloom: bp,
        composition: cp,
        presentation: pp,
        getCursor: () => {
          if (!cursorPipe) {
            const mod = device.createShaderModule({ label: 'Cursor WGSL', code: cursorShader })
            cursorPipe = device.createRenderPipeline({
              label: 'Space cursor',
              layout: 'auto',
              vertex: vertex(mod, 'cursorVertex'),
              fragment: { module: mod, entryPoint: 'cursorFragment', targets: [{ format, blend: alphaBlend }] },
            })
          }
          return cursorPipe
        },
        getLoupe: () => {
          if (!loupePipe) {
            const mod = device.createShaderModule({ label: 'Optical loupe WGSL', code: loupeShader })
            loupePipe = device.createRenderPipeline({
              label: 'Optical loupe',
              layout: 'auto',
              vertex: vertex(mod, 'loupeVertex'),
              fragment: { module: mod, entryPoint: 'loupeFragment', targets: [{ format, blend: alphaBlend }] },
            })
          }
          return loupePipe
        },
        getMotion: reverseDepth
          ? () =>
              (motionPipe ??= device.createRenderPipeline({
                label: 'Depth-aware camera motion blur',
                layout: 'auto',
                vertex: vertex(getMotionModule()),
                fragment: {
                  module: getMotionModule(),
                  entryPoint: 'blurFragment',
                  targets: [{ format: 'rgba16float' }],
                },
              }))
          : undefined,
        getStaticResolve: reverseDepth
          ? () =>
              (staticResolvePipe ??= device.createRenderPipeline({
                label: 'Per-sample static marker and orbit resolve',
                layout: 'auto',
                vertex: vertex(getMotionModule()),
                fragment: {
                  module: getMotionModule(),
                  entryPoint: 'staticResolveFragment',
                  targets: [{ format: 'rgba16float' }],
                },
              }))
          : undefined,
        backdropBlit: bbp,
      }
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
export type MarkerUniform = {
  u: number
  v: number
  opacity: number
  unitsPerPixel: number
}
export type PostMarkers = {
  hover: MarkerUniform
  earthMorph: number
  selEarth: MarkerUniform
  selMoon: MarkerUniform
  selMars: MarkerUniform
}
export type TileRect = { x: number; y: number; width: number; height: number; index: number; count: number }
export type PrepareOrbits = (
  encoder: GPUCommandEncoder,
  depthView: GPUTextureView,
  width: number,
  height: number,
) => GPUTextureView | null
export type PostMotionAtmospheres = (
  encoder: GPUCommandEncoder,
  multisampleView: GPUTextureView,
  sceneView: GPUTextureView,
  motionView: GPUTextureView | null,
  depthView: GPUTextureView,
) => void
export type PostSceneEffect = (
  encoder: GPUCommandEncoder,
  sceneView: GPUTextureView,
  width: number,
  height: number,
) => GPUTextureView | null | void
export type Presentation = {
  hdr: boolean
  exposure: number
  glow: number
  ratio: number
  dataBoundary: boolean
  cursor: [number, number] | null
  hoverScreen?: [number, number] | null
  hoverBody: number
  markers: PostMarkers
  loupe?: LoupeOverlay | null
  motion?: MotionBlurUniforms
  lowRes?: boolean
  tile?: TileRect
  prepareOrbits?: PrepareOrbits
  postMotionAtmospheres?: PostMotionAtmospheres
  postScene?: PostSceneEffect
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
  markers: PostMarkers
  bakedMode: 0 | 1 | 2
  hasOrbits: boolean
  bakeMarkers: boolean
}

function writeMarkerRecord(out: Float32Array, offset: number, marker: MarkerUniform) {
  out[offset] = marker.u
  out[offset + 1] = marker.v
  out[offset + 2] = marker.opacity
  out[offset + 3] = marker.unitsPerPixel
}

function packPostUniforms(out: Float32Array, v: PostUniformValues) {
  out[0] = v.hoverScreen ? v.hoverScreen[0] : 0
  out[1] = v.hoverScreen ? v.hoverScreen[1] : 0
  out[2] = Number(Boolean(v.hoverScreen))
  out[3] = Number(v.hdr)
  out[4] = v.exposure
  out[5] = v.hdr ? v.glow : 0
  out[6] = v.ratio
  out[7] = Number(v.dataBoundary)
  out[8] = Number(v.displayOnlyHover)
  out[9] = v.hoverBody
  out[10] = v.outWidth
  out[11] = v.outHeight
  writeMarkerRecord(out, 12, v.markers.hover)
  out[16] = v.markers.earthMorph
  out[17] = v.bakedMode
  out[18] = Number(v.hasOrbits)
  out[19] = Number(v.bakeMarkers)
  writeMarkerRecord(out, 20, v.markers.selEarth)
  writeMarkerRecord(out, 24, v.markers.selMoon)
  writeMarkerRecord(out, 28, v.markers.selMars)
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
  geography: GPUTexture | null
  geographyView: GPUTextureView
  annotationDepth: GPUTexture | null
  annotationDepthView: GPUTextureView | null
  backdropScene?: GPUTexture
  backdropSceneView?: GPUTextureView
  backdropBlitGroup?: GPUBindGroup
  backdropValid: boolean
  stableCoverage: GPUTexture | null
  stableCoverageView: GPUTextureView | null
  coverageMsaa: GPUTexture | null
  coverageMsaaView: GPUTextureView | null
  boundCoverage: GPUTexture
  coverageValid: boolean
  annotationsValid: boolean
  annotationsUpdated: boolean
  bloomValid: boolean
  hasMultisample: boolean
  lastOrbitView: GPUTextureView | null
  bloom: GPUTexture[]
  bloomViews: [GPUTextureView, GPUTextureView]
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
  postSceneSourceView?: GPUTextureView
  postSceneBloomGroup?: GPUBindGroup
  postSceneCompositionGroup?: GPUBindGroup
}

export class PostProcess {
  private readonly emptyCoverage: GPUTexture
  private readonly emptyCoverageView: GPUTextureView
  private readonly emptyOrbitMsaa: GPUTexture
  private readonly emptyOrbitMsaaView: GPUTextureView
  private readonly emptyGeographyMsaa: GPUTexture
  private readonly emptyGeographyMsaaView: GPUTextureView
  private high!: RenderBank
  private low: RenderBank | null = null
  private activeBank!: RenderBank
  private display!: GPUTexture
  private displayView!: GPUTextureView
  private displayValid = false
  private baseWidth = 0
  private baseHeight = 0
  private outWidth = 0
  private outHeight = 0
  private key = ''
  private displayedMotion = false
  private bakedStaticMarkers = false
  private lastStaticMarkers = new Float32Array(12)
  private pipelines!: PostPipelines
  private backdropBlitPipeline?: GPURenderPipeline
  private motionUniform?: GPUBuffer
  private bloomPipeline!: GPURenderPipeline
  private compositionPipeline!: GPURenderPipeline
  private presentationPipeline!: GPURenderPipeline
  private loupeScene?: GPUTexture
  private loupeSceneView?: GPUTextureView
  private loupeMultisample?: GPUTexture
  private loupeMultisampleView?: GPUTextureView
  private loupeDepth?: GPUTexture
  private loupeDepthView?: GPUTextureView
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
  private cursorGroup?: GPUBindGroup

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
    this.emptyCoverageView = this.emptyCoverage.createView()
    this.emptyOrbitMsaa = device.createTexture({
      label: 'Unused 4x MSAA orbit binding',
      size: [1, 1],
      format: 'rgba16float',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.emptyOrbitMsaaView = this.emptyOrbitMsaa.createView()
    this.emptyGeographyMsaa = device.createTexture({
      label: 'Unused 4x MSAA geography binding',
      size: [1, 1],
      format: 'rgba16float',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.emptyGeographyMsaaView = this.emptyGeographyMsaa.createView()
    const uniform = (label: string, size: number) =>
      device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.bloomUniforms = [0, 1, 2].map(i => uniform(`Bloom parameters ${i}`, 16))
    this.compositeUniform = uniform('Display parameters', 128)
    this.cursorUniform = uniform('Space cursor parameters', 32)
    this.loupeUniform = uniform('Optical loupe parameters', 64)
  }

  async initialize() {
    const device = this.device
    const pipelines = await getPostPipelines(device, this.format, this.reverseDepth)
    this.pipelines = pipelines
    this.backdropBlitPipeline = pipelines.backdropBlit
    this.bloomPipeline = pipelines.bloom
    this.compositionPipeline = pipelines.composition
    this.presentationPipeline = pipelines.presentation
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
    const depth = this.target(`${label} 4x MSAA surface depth`, width, height, depthFormat, 4, this.reverseDepth)
    const depthOnlyView = depth.createView({ aspect: 'depth-only' })
    const geographyView = this.emptyGeographyMsaaView
    const scale = Math.min(0.5, 1024 / Math.max(width, height))
    const bloom = [0, 1].map(i =>
      this.target(`${label} Bloom ${i}`, Math.max(1, Math.ceil(width * scale)), Math.max(1, Math.ceil(height * scale))),
    )
    const bloomViews: [GPUTextureView, GPUTextureView] = [bloom[0].createView(), bloom[1].createView()]
    const bloomGroups = [sceneView, bloomViews[0], bloomViews[1]].map((view, i) =>
      this.device.createBindGroup({
        layout: this.bloomPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.bloomUniforms[i] } },
          { binding: 1, resource: view },
          { binding: 2, resource: this.sampler },
        ],
      }),
    )
    const compositionGroup = this.compositionBinding(sceneView, bloomViews[0], this.emptyCoverageView)
    const presentationGroup = this.presentationBinding(geographyView, multisampleView, bloomViews[0])
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
      geography: null,
      geographyView,
      annotationDepth: null,
      annotationDepthView: null,
      backdropValid: false,
      stableCoverage: null,
      stableCoverageView: null,
      coverageMsaa: null,
      coverageMsaaView: null,
      boundCoverage: this.emptyCoverage,
      coverageValid: false,
      annotationsValid: false,
      annotationsUpdated: false,
      bloomValid: false,
      hasMultisample: false,
      lastOrbitView: null,
      bloom,
      bloomViews,
      bloomGroups,
      compositionGroup,
      presentationGroup,
    }
  }

  private presentationBinding(
    geographyView: GPUTextureView,
    multisampleView: GPUTextureView,
    bloom0View: GPUTextureView,
  ) {
    return this.device.createBindGroup({
      layout: this.presentationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniform } },
        { binding: 1, resource: this.displayView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: geographyView },
        { binding: 4, resource: multisampleView },
        { binding: 5, resource: bloom0View },
      ],
    })
  }

  private ensureAnnotationTargets(bank: RenderBank) {
    if (bank.geography && bank.annotationDepth) return
    const depthFormat = this.reverseDepth ? 'depth32float' : 'depth24plus'
    bank.geography = this.target(
      `${bank.label} 4x MSAA Geographic coordinates`,
      bank.width,
      bank.height,
      'rgba16float',
      4,
      true,
    )
    bank.geographyView = bank.geography.createView()
    bank.annotationDepth = this.target(
      `${bank.label} 4x MSAA Annotation depth`,
      bank.width,
      bank.height,
      depthFormat,
      4,
      false,
    )
    bank.annotationDepthView = bank.annotationDepth.createView()
    bank.presentationGroup = this.presentationBinding(bank.geographyView, bank.multisampleView, bank.bloomViews[0])
    bank.staticResolveGroup = undefined
    bank.motionGroup = undefined
  }

  private destroyBank(bank: RenderBank | null | undefined) {
    if (!bank) return
    bank.scene.destroy()
    bank.multisample.destroy()
    bank.depth.destroy()
    bank.geography?.destroy()
    bank.annotationDepth?.destroy()
    bank.backdropScene?.destroy()
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

  private compositionBinding(sourceView: GPUTextureView, bloomView: GPUTextureView, coverageView: GPUTextureView) {
    return this.device.createBindGroup({
      layout: this.compositionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniform } },
        { binding: 1, resource: sourceView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: bloomView },
        { binding: 4, resource: coverageView },
      ],
    })
  }

  private bindComposition(bank: RenderBank, coverage: GPUTexture, coverageView: GPUTextureView) {
    bank.boundCoverage = coverage
    bank.compositionGroup = this.compositionBinding(bank.sceneView, bank.bloomViews[0], coverageView)
    bank.motionCompositionGroup =
      bank.motionTargetView && this.compositionBinding(bank.motionTargetView, bank.bloomViews[0], coverageView)
    bank.postSceneCompositionGroup =
      bank.postSceneSourceView && this.compositionBinding(bank.postSceneSourceView, bank.bloomViews[0], coverageView)
  }

  private encodeStaticResolve(encoder: GPUCommandEncoder, bank: RenderBank, orbitView: GPUTextureView | null) {
    const staticResolvePipeline = this.pipelines.getStaticResolve?.()
    if (!staticResolvePipeline) return false
    const boundOrbit = orbitView ?? this.emptyOrbitMsaaView
    if (!bank.staticResolveGroup || bank.staticResolveOrbitView !== boundOrbit) {
      bank.staticResolveGroup = this.device.createBindGroup({
        layout: staticResolvePipeline.getBindGroupLayout(0),
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
    pass.setPipeline(staticResolvePipeline)
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
    if (!motion || !this.pipelines.getMotion) return false
    const motionPipeline = this.pipelines.getMotion()
    const motionUniform = (this.motionUniform ??= this.device.createBuffer({
      label: 'Camera motion parameters',
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }))
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
      const boundCoverageView =
        bank.boundCoverage === bank.stableCoverage && bank.stableCoverageView
          ? bank.stableCoverageView
          : this.emptyCoverageView
      bank.motionCompositionGroup = this.compositionBinding(
        bank.motionTargetView,
        bank.bloomViews[0],
        boundCoverageView,
      )
    }
    const boundOrbit = orbitView ?? this.emptyOrbitMsaaView
    if (!bank.motionGroup || bank.motionOrbitView !== boundOrbit) {
      bank.motionGroup = this.device.createBindGroup({
        layout: motionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: motionUniform } },
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
    this.device.queue.writeBuffer(motionUniform, 0, motion)
    const pass = encoder.beginRenderPass({
      label: 'Camera motion blur',
      colorAttachments: [{ view: bank.motionTargetView!, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(motionPipeline)
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
    this.displayView = this.display.createView()
    this.displayValid = false
    this.high = this.createBank('Native', outWidth, outHeight)
    this.activeBank = this.high
    this.key = key
    return true
  }

  beginAnnotations(encoder: GPUCommandEncoder, includeCoverage: boolean, lowRes = false) {
    const bank = lowRes ? this.ensureLowBank() : this.high
    this.ensureAnnotationTargets(bank)
    const colorAttachments: GPURenderPassColorAttachment[] = [
      { view: bank.geographyView, clearValue: [0, 0, -1, 0], loadOp: 'clear', storeOp: 'store' },
    ]
    if (includeCoverage) {
      if (!bank.stableCoverage) {
        bank.stableCoverage = this.target(`${bank.label} Score coverage`, bank.width, bank.height, 'rg8unorm')
        bank.stableCoverageView = bank.stableCoverage.createView()
        bank.coverageMsaa = this.target(
          `${bank.label} 4x MSAA Score coverage`,
          bank.width,
          bank.height,
          'rg8unorm',
          4,
          false,
        )
        bank.coverageMsaaView = bank.coverageMsaa.createView()
      }
      if (bank.boundCoverage !== bank.stableCoverage)
        this.bindComposition(bank, bank.stableCoverage, bank.stableCoverageView!)
      colorAttachments.push({
        view: bank.coverageMsaaView!,
        resolveTarget: bank.stableCoverageView!,
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'discard',
      })
    } else if (bank.boundCoverage !== this.emptyCoverage) {
      this.bindComposition(bank, this.emptyCoverage, this.emptyCoverageView)
    }
    bank.coverageValid = includeCoverage
    bank.annotationsValid = true
    bank.annotationsUpdated = true
    return encoder.beginRenderPass({
      label: 'Geographic annotations',
      colorAttachments,
      depthStencilAttachment: {
        view: bank.annotationDepthView!,
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
      this.loupeSceneView = this.loupeScene.createView()
      this.loupeMultisample = this.target('Optical loupe 4x MSAA HDR', size, size, 'rgba16float', 4)
      this.loupeMultisampleView = this.loupeMultisample.createView()
      this.loupeDepth = this.target('Optical loupe depth', size, size, depthFormat, 4, false)
      this.loupeDepthView = this.loupeDepth.createView()
      const loupePipeline = this.pipelines.getLoupe()
      this.loupeGroup = this.device.createBindGroup({
        layout: loupePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.loupeUniform } },
          { binding: 1, resource: this.loupeSceneView },
          { binding: 2, resource: this.sampler },
        ],
      })
    }
    return encoder.beginRenderPass({
      label: 'Optical loupe mini-pass',
      colorAttachments: [
        {
          view: this.loupeMultisampleView!,
          resolveTarget: this.loupeSceneView!,
          clearValue: hdr ? [0.018 / 12.92, 0.026 / 12.92, 0.039 / 12.92, 1] : [0.018, 0.026, 0.039, 1],
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.loupeDepthView!,
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
  get hasValidHeroBackdrop() {
    return Boolean(this.high?.backdropValid && this.high.backdropBlitGroup && this.backdropBlitPipeline)
  }

  /**
   * Render the slow-changing hero planet backdrop (Mars, Earth, Moon, Sky, Atmospheres) into
   * 4x MSAA, resolving to a cached 1x HDR backdrop while preserving 4x MSAA color + depth
   * for the immediately following live overlay pass.
   */
  beginHeroBackdrop(encoder: GPUCommandEncoder, hdr: boolean) {
    const bank = this.high
    if (!bank.backdropScene && this.backdropBlitPipeline) {
      bank.backdropScene = this.target('Hero cached HDR backdrop', bank.width, bank.height)
      bank.backdropSceneView = bank.backdropScene.createView()
      bank.backdropBlitGroup = this.device.createBindGroup({
        layout: this.backdropBlitPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: bank.backdropSceneView }],
      })
    }
    bank.hasMultisample = false
    bank.backdropValid = Boolean(bank.backdropSceneView)
    return encoder.beginRenderPass({
      label: '4x MSAA Earth surface',
      colorAttachments: [
        {
          view: bank.multisampleView,
          resolveTarget: bank.backdropSceneView,
          clearValue: hdr ? [0.018 / 12.92, 0.026 / 12.92, 0.039 / 12.92, 1] : [0.018, 0.026, 0.039, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: bank.depthOnlyView,
        depthClearValue: this.reverseDepth ? 0 : 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
  }

  /**
   * Draw live 60 FPS hero overlays (Bridge arc + Stardust Title/CTA) onto the 4x MSAA target
   * while reusing the exact 4x MSAA planet depth buffer. When `fromCache` is true, blits the
   * cached HDR backdrop in 1 fullscreen triangle instead of re-rendering all planets.
   */
  beginHeroOverlay(encoder: GPUCommandEncoder, fromCache: boolean) {
    const bank = this.high
    bank.hasMultisample = false
    const pass = encoder.beginRenderPass({
      label: fromCache ? 'Hero cached backdrop and overlays' : 'Hero live overlays',
      colorAttachments: [
        {
          view: bank.multisampleView,
          resolveTarget: bank.sceneView,
          clearValue: [0, 0, 0, 1],
          loadOp: fromCache ? 'clear' : 'load',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: bank.depthOnlyView,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      },
    })
    if (fromCache && this.backdropBlitPipeline && bank.backdropBlitGroup) {
      pass.setPipeline(this.backdropBlitPipeline)
      pass.setBindGroup(0, bank.backdropBlitGroup)
      pass.draw(3)
    }
    return pass
  }

  begin(encoder: GPUCommandEncoder, hdr: boolean, storeDepth = false, lowRes = false, tile?: TileRect) {
    const bank = lowRes ? this.ensureLowBank() : this.high
    if (lowRes && bank !== this.high) this.high.annotationsValid = false
    const firstTile = !tile || tile.index === 0
    const lastTile = !tile || tile.index === tile.count - 1
    const keepDepth = storeDepth || !lastTile || this.reverseDepth
    // Always preserve multisample color for non-hero banks because presentation reads sceneMsaa
    // whenever reconstructing selected/hovered reticles (including on later overlay-only frames).
    bank.hasMultisample = true
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
          storeOp: bank.hasMultisample || !lastTile ? 'store' : 'discard',
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
      l[11] = 0
      l[12] = sel ? sel[0] : 0
      l[13] = sel ? sel[1] : 0
      l[14] = Number(Boolean(sel))
      l[15] = 0
      this.device.queue.writeBuffer(this.loupeUniform, 0, l)
      pass.setPipeline(this.pipelines.getLoupe())
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
      const cursorPipeline = this.pipelines.getCursor()
      this.cursorGroup ??= this.device.createBindGroup({
        layout: cursorPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.cursorUniform } }],
      })
      pass.setPipeline(cursorPipeline)
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
    const willBlur = redrawSurface && Boolean(p.motion && this.pipelines.getMotion)
    const orbitView =
      redrawSurface && p.prepareOrbits ? p.prepareOrbits(encoder, bank.depthOnlyView, bank.width, bank.height) : null
    if (redrawSurface) bank.lastOrbitView = orbitView
    const activeOrbitView = redrawSurface ? orbitView : bank.lastOrbitView
    const canBakeStatic = Boolean(this.pipelines.getStaticResolve && bank.hasMultisample)
    const { selEarth, selMoon, selMars } = p.markers
    const hasSelMarker = selEarth.opacity > 0.005 || selMoon.opacity > 0.005 || selMars.opacity > 0.005
    const selectedMarkers = [
      selEarth.u,
      selEarth.v,
      selEarth.opacity,
      selEarth.unitsPerPixel,
      selMoon.u,
      selMoon.v,
      selMoon.opacity,
      selMoon.unitsPerPixel,
      selMars.u,
      selMars.v,
      selMars.opacity,
      selMars.unitsPerPixel,
    ]
    let staticMarkersChanged = false
    for (let i = 0; i < 12; i++) {
      if (this.lastStaticMarkers[i] !== Math.fround(selectedMarkers[i])) {
        staticMarkersChanged = true
        break
      }
    }
    const willBakeStaticMarkers = !willBlur && canBakeStatic && hasSelMarker
    const sceneNeedsStaticResolve =
      !willBlur &&
      canBakeStatic &&
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
      for (let i = 0; i < 12; i++) this.lastStaticMarkers[i] = Math.fround(selectedMarkers[i])
    } else if (redrawSurface && !motion) {
      this.bakedStaticMarkers = false
      this.lastStaticMarkers.fill(0)
    }
    bank.annotationsUpdated = false
    const postSceneOut =
      redrawSurface && p.postScene ? (p.postScene(encoder, bank.sceneView, bank.width, bank.height) ?? null) : null
    if (postSceneOut && bank.postSceneSourceView !== postSceneOut) {
      bank.postSceneSourceView = postSceneOut
      bank.postSceneBloomGroup = this.device.createBindGroup({
        layout: this.bloomPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.bloomUniforms[0] } },
          { binding: 1, resource: postSceneOut },
          { binding: 2, resource: this.sampler },
        ],
      })
      const boundCoverageView =
        bank.boundCoverage === bank.stableCoverage && bank.stableCoverageView
          ? bank.stableCoverageView
          : this.emptyCoverageView
      bank.postSceneCompositionGroup = this.compositionBinding(postSceneOut, bank.bloomViews[0], boundCoverageView)
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
          colorAttachments: [{ view: bank.bloomViews[i % 2], loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(this.bloomPipeline)
        pass.setBindGroup(
          0,
          i === 0 && motion
            ? bank.motionBloomGroup!
            : i === 0 && postSceneOut
              ? bank.postSceneBloomGroup!
              : bank.bloomGroups[i],
        )
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
        colorAttachments: [{ view: this.displayView, loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(this.compositionPipeline)
      pass.setBindGroup(
        0,
        motion ? bank.motionCompositionGroup! : postSceneOut ? bank.postSceneCompositionGroup! : bank.compositionGroup,
      )
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
    this.loupeSceneView = this.loupeMultisampleView = this.loupeDepthView = undefined
    this.loupeGroup = undefined
    this.emptyCoverage.destroy()
    this.emptyOrbitMsaa.destroy()
    this.emptyGeographyMsaa.destroy()
    this.motionUniform?.destroy()
    this.bloomUniforms.forEach(buffer => buffer.destroy())
    this.compositeUniform.destroy()
    this.cursorUniform.destroy()
    this.loupeUniform.destroy()
  }
}
