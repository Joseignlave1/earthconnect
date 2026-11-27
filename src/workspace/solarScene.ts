import type { WorkspaceAssets } from './workspaceAssets.ts'
import type { TerrainLoader } from './earthWorkspaceResources.ts'
import { EARTH_AXIS, FOV, intersectsAabb } from '../shader-test/geometry.ts'
import { joinWgsl, shaderModule } from '../shader-test/gpu.ts'
import { LOUPE_TEXTURE_SIZE } from '../shader-test/postprocess.ts'
import fullscreenWgsl from '../shader-test/shaders/fullscreen.wgsl?raw'
import { PlanetTerrainRenderer, type PlanetPickingMesh } from './planetTerrain.ts'
import type { PlanetTilePackLoader } from './planetTiles.ts'
import solarAtmoCoreWgsl from './shaders/solar-atmo-core.wgsl?raw'
import solarAtmoMotionWgsl from './shaders/solar-atmo-motion.wgsl?raw'
import solarAtmoWgsl from './shaders/solar-atmo.wgsl?raw'
import solarBodyWgsl from './shaders/solar-body.wgsl?raw'
import solarCommonWgsl from './shaders/solar-common.wgsl?raw'
import solarHaloWgsl from './shaders/solar-halo.wgsl?raw'
import solarOrbitAccumWgsl from './shaders/solar-orbit-accum.wgsl?raw'
import solarOrbitWgsl from './shaders/solar-orbit.wgsl?raw'
import solarSkyWgsl from './shaders/solar-sky.wgsl?raw'
import { orbitPolyline, solarSystemAt, type BodyId, type Vec3 } from './solarSystem.ts'
import { loadSkyMap } from './skyMap.ts'
import {
  add,
  bodyMatrix,
  dot,
  length,
  normalized,
  raySphere,
  relativeToEye,
  scale,
  sceneRay,
  SolarCamera,
  subtract,
  toView,
  worldToGeographic,
  type SolarBodies,
  type SolarFrame,
  type SourceBody,
} from './solarCamera.ts'

const EARTH_EQUATORIAL_KM = EARTH_AXIS / 1000
const EARTH_POLAR_STRETCH = 1 / Math.sqrt(1 - 0.0066943799901413165) - 1
const SMALL_PLANET_CSS_PX = 38
const ids: readonly BodyId[] = [
  'sun',
  'mercury',
  'venus',
  'earth',
  'moon',
  'mars',
  'jupiter',
  'saturn',
  'uranus',
  'neptune',
]
const sources: readonly SourceBody[] = ['moon', 'mars']
const colors: Record<BodyId, Vec3> = {
  sun: [10.0, 9.4, 8.6],
  mercury: [0.17, 0.14, 0.12],
  venus: [0.57, 0.46, 0.27],
  earth: [0.1, 0.19, 0.39],
  moon: [0.18, 0.185, 0.2],
  mars: [0.26, 0.11, 0.052],
  jupiter: [0.62, 0.46, 0.35],
  saturn: [0.59, 0.49, 0.36],
  uranus: [0.24, 0.45, 0.5],
  neptune: [0.18, 0.27, 0.57],
}

const skyShader = joinWgsl(fullscreenWgsl, solarCommonWgsl, solarSkyWgsl)
const bodyShader = joinWgsl(solarCommonWgsl, solarBodyWgsl)
const haloShader = joinWgsl(fullscreenWgsl, solarCommonWgsl, solarHaloWgsl)
const earthAtmoCore = joinWgsl(fullscreenWgsl, solarCommonWgsl, solarAtmoCoreWgsl)
const earthAtmoShader = joinWgsl(earthAtmoCore, solarAtmoWgsl)
const earthAtmoMotionShader = joinWgsl(earthAtmoCore, solarAtmoMotionWgsl)
const orbitShader = solarOrbitWgsl
const orbitCompositeShader = joinWgsl(fullscreenWgsl, solarOrbitAccumWgsl)

export type SourcePoint = { latitude: number; longitude: number }
export type SolarLabel = { id: BodyId; x: number; y: number }
export type SolarSceneOptions = {
  bodies?: SolarBodies
  eagerTerrain?: boolean
  albedoMaxWidths?: Partial<Record<'moon' | 'mars', number>>
}

type SolarPipelines = {
  skyPipeline: GPURenderPipeline
  bodyPipeline: GPURenderPipeline
  bodyAnnotationLayout: GPUBindGroupLayout
  bodyAnnotationPipeline: GPURenderPipeline
  bodyAnnotationCoveragePipeline: GPURenderPipeline
  orbitCoveragePipelines: GPURenderPipeline[]
  orbitAccumPipeline: GPURenderPipeline
  haloPipeline: GPURenderPipeline
  lensFlarePipeline: GPURenderPipeline
  earthAtmoPipeline: GPURenderPipeline
  earthAtmoMotionPipeline: GPURenderPipeline
}

const solarPipelines = new WeakMap<GPUDevice, Promise<SolarPipelines>>()

function getSolarPipelines(device: GPUDevice): Promise<SolarPipelines> {
  let pending = solarPipelines.get(device)
  if (!pending) {
    pending = (async () => {
      const [sky, body, orbit, orbitComposite, halo, earthAtmo, earthAtmoMotion] = await Promise.all([
        shaderModule(device, 'Inertial sky', skyShader),
        shaderModule(device, 'Solar bodies', bodyShader),
        shaderModule(device, 'Planetary paths', orbitShader),
        shaderModule(device, 'Planetary path accumulate', orbitCompositeShader),
        shaderModule(device, 'Sunlight halo', haloShader),
        shaderModule(device, 'Earth Rayleigh atmosphere', earthAtmoShader),
        shaderModule(device, 'Earth Rayleigh atmosphere post-motion', earthAtmoMotionShader),
      ])
      const color: GPUColorTargetState = { format: 'rgba16float' }
      const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' }
      const common = { layout: 'auto' as const, multisample: { count: 4 } }
      const additive: GPUBlendState = {
        color: { srcFactor: 'one', dstFactor: 'one' },
        alpha: { srcFactor: 'zero', dstFactor: 'one' },
      }
      const maxBlend: GPUBlendState = {
        color: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
        alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
      }
      const premulOver: GPUBlendState = {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      }
      const orbitCoverageBindGroupLayout = device.createBindGroupLayout({
        label: 'Orbit coverage depth bind group layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', multisampled: true } },
        ],
      })
      const orbitCoverageLayout = device.createPipelineLayout({
        label: 'Orbit coverage pipeline layout',
        bindGroupLayouts: [orbitCoverageBindGroupLayout],
      })
      const orbitAccumBindGroupLayout = device.createBindGroupLayout({
        label: 'Orbit accumulate bind group layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'unfilterable-float', multisampled: true },
          },
        ],
      })
      const orbitAccumLayout = device.createPipelineLayout({
        label: 'Orbit accumulate pipeline layout',
        bindGroupLayouts: [orbitAccumBindGroupLayout],
      })
      const bodyAnnotationLayout = device.createBindGroupLayout({
        label: 'Billboard body annotation bind group layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'read-only-storage' },
          },
        ],
      })
      const bodyAnnotationPipelineLayout = device.createPipelineLayout({
        label: 'Billboard body annotation pipeline layout',
        bindGroupLayouts: [bodyAnnotationLayout],
      })
      const channelMasks = [GPUColorWrite.RED, GPUColorWrite.GREEN, GPUColorWrite.BLUE, GPUColorWrite.ALPHA]
      const [
        skyPipeline,
        bodyPipeline,
        bodyAnnotationPipeline,
        bodyAnnotationCoveragePipeline,
        orbitAccumPipeline,
        haloPipeline,
        lensFlarePipeline,
        earthAtmoPipeline,
        earthAtmoMotionPipeline,
        ...orbitCoveragePipelines
      ] = await Promise.all([
        device.createRenderPipelineAsync({
          ...common,
          label: 'Inertial sky',
          vertex: { module: sky, entryPoint: 'screenVertex' },
          fragment: { module: sky, entryPoint: 'skyFragment', targets: [color] },
          depthStencil: { ...depth, depthWriteEnabled: false, depthCompare: 'equal' },
        }),
        device.createRenderPipelineAsync({
          ...common,
          label: 'Sun and planet disks',
          vertex: { module: body, entryPoint: 'bodyVertex' },
          fragment: { module: body, entryPoint: 'bodyFragment', targets: [color] },
          depthStencil: depth,
        }),
        device.createRenderPipelineAsync({
          layout: bodyAnnotationPipelineLayout,
          multisample: { count: 4 },
          label: 'Billboard body annotation occluders',
          vertex: { module: body, entryPoint: 'bodyVertex' },
          fragment: { module: body, entryPoint: 'bodyAnnotationFragment', targets: [color] },
          depthStencil: depth,
        }),
        device.createRenderPipelineAsync({
          layout: bodyAnnotationPipelineLayout,
          multisample: { count: 4 },
          label: 'Billboard body annotation coverage occluders',
          vertex: { module: body, entryPoint: 'bodyVertex' },
          fragment: {
            module: body,
            entryPoint: 'bodyAnnotationCoverageFragment',
            targets: [color, { format: 'rg8unorm' }],
          },
          depthStencil: depth,
        }),
        device.createRenderPipelineAsync({
          layout: orbitAccumLayout,
          multisample: { count: 4 },
          label: 'Ephemeris orbit accumulate',
          vertex: { module: orbitComposite, entryPoint: 'screenVertex' },
          fragment: {
            module: orbitComposite,
            entryPoint: 'orbitAccumFragment',
            targets: [{ ...color, blend: premulOver }],
          },
        }),
        device.createRenderPipelineAsync({
          ...common,
          label: 'Solar halo with occlusion',
          vertex: { module: halo, entryPoint: 'screenVertex' },
          fragment: { module: halo, entryPoint: 'haloFragment', targets: [{ ...color, blend: additive }] },
          depthStencil: { ...depth, depthWriteEnabled: false },
        }),
        device.createRenderPipelineAsync({
          ...common,
          label: 'Camera lens flare',
          vertex: { module: halo, entryPoint: 'screenVertex' },
          fragment: { module: halo, entryPoint: 'lensFlareFragment', targets: [{ ...color, blend: additive }] },
          depthStencil: { ...depth, depthWriteEnabled: false, depthCompare: 'always' },
        }),
        device.createRenderPipelineAsync({
          ...common,
          label: 'Earth Rayleigh atmosphere',
          vertex: { module: earthAtmo, entryPoint: 'screenVertex' },
          fragment: { module: earthAtmo, entryPoint: 'earthAtmoFragment', targets: [{ ...color, blend: additive }] },
          depthStencil: { ...depth, depthWriteEnabled: false },
        }),
        device.createRenderPipelineAsync({
          layout: 'auto',
          multisample: { count: 1 },
          label: 'Earth Rayleigh atmosphere post-motion',
          vertex: { module: earthAtmoMotion, entryPoint: 'screenVertex' },
          fragment: {
            module: earthAtmoMotion,
            entryPoint: 'earthAtmoMotionFragment',
            targets: [{ ...color, blend: additive }],
          },
        }),
        ...channelMasks.map((writeMask, i) =>
          device.createRenderPipelineAsync({
            layout: orbitCoverageLayout,
            multisample: { count: 4 },
            label: `Ephemeris orbit coverage ${i}`,
            vertex: {
              module: orbit,
              entryPoint: 'orbitVertex',
              buffers: [
                {
                  arrayStride: 48,
                  attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x4' },
                    { shaderLocation: 1, offset: 16, format: 'float32x4' },
                    { shaderLocation: 2, offset: 32, format: 'float32x4' },
                  ],
                },
              ],
            },
            fragment: {
              module: orbit,
              entryPoint: 'orbitCoverageFragment',
              targets: [{ format: 'rgba8unorm', blend: maxBlend, writeMask }],
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
          }),
        ),
      ])
      return {
        skyPipeline,
        bodyPipeline,
        bodyAnnotationLayout,
        bodyAnnotationPipeline,
        bodyAnnotationCoveragePipeline,
        orbitCoveragePipelines,
        orbitAccumPipeline,
        haloPipeline,
        lensFlarePipeline,
        earthAtmoPipeline,
        earthAtmoMotionPipeline,
      }
    })()
    solarPipelines.set(device, pending)
    void pending.catch(() => {
      if (solarPipelines.get(device) === pending) solarPipelines.delete(device)
    })
  }
  return pending
}

export class SolarScene {
  bodies: SolarBodies
  private terrain?: PlanetTerrainRenderer
  private loadingTerrain = false
  skySettled = false
  terrainSettled = false
  private readonly controller = new AbortController()
  private readonly viewBuffer: GPUBuffer
  private readonly annotationViewBuffer: GPUBuffer
  private readonly loupeViewBuffer: GPUBuffer
  private readonly earthAtmoBuffer: GPUBuffer
  private readonly loupeEarthAtmoBuffer: GPUBuffer
  private readonly marsAtmoBuffer: GPUBuffer
  private readonly loupeMarsAtmoBuffer: GPUBuffer
  private readonly haloBuffer: GPUBuffer
  private readonly viewScratch = new Float32Array(24)
  private readonly smallViewScratch = new Float32Array(16)
  private readonly earthAtmoScratch = new Float32Array(12)
  private readonly marsAtmoScratch = new Float32Array(12)
  private readonly haloScratch = new Float32Array(4)
  private haloVisible = false
  private readonly bodyBuffer: GPUBuffer
  private readonly annotationBodyBuffer: GPUBuffer
  private readonly orbitBuffer: GPUBuffer
  private readonly orbitBatchBuffers: GPUBuffer[]
  private readonly orbitBatchScratch = new Float32Array(16)
  private orbitCoverageMsaa?: GPUTexture
  private orbitCoverageMsaaView?: GPUTextureView
  private orbitAccumMsaa?: GPUTexture
  private orbitAccumMsaaView?: GPUTextureView
  private orbitAccumGroups: GPUBindGroup[] = []
  private readonly atmoMotionGroups = new WeakMap<GPUTextureView, { earth: GPUBindGroup; mars: GPUBindGroup }>()
  private readonly orbitDepthGroups = new WeakMap<GPUTextureView, GPUBindGroup>()
  private orbitCoverageKey = ''
  private readonly orbits: { id: BodyId; vertices: readonly Vec3[] }[]
  private readonly bodyData = new Float32Array(ids.length * 12)
  private readonly orbitData: Float32Array<ArrayBuffer>
  private skyGroup: GPUBindGroup
  private loupeSkyGroup: GPUBindGroup
  private skyTexture: GPUTexture
  private skyOwned = true
  private readonly skySampler: GPUSampler
  private readonly haloGroup: GPUBindGroup
  private readonly lensFlareGroup: GPUBindGroup
  private readonly earthAtmoGroup: GPUBindGroup
  private readonly loupeEarthAtmoGroup: GPUBindGroup
  private readonly marsAtmoGroup: GPUBindGroup
  private readonly loupeMarsAtmoGroup: GPUBindGroup
  private readonly bodyGroup: GPUBindGroup
  private readonly bodyAnnotationGroup: GPUBindGroup

  static async create(
    device: GPUDevice,
    date: Date,
    invalidate: () => void,
    onError: (error: Error) => void,
    camera: SolarCamera | undefined,
    assetUrl: WorkspaceAssets['url'],
    terrainLoader?: TerrainLoader,
    loadPlanetTilePack?: PlanetTilePackLoader,
    options?: SolarSceneOptions,
  ) {
    const pipelines = await getSolarPipelines(device)
    return new SolarScene(
      device,
      date,
      pipelines,
      invalidate,
      onError,
      camera ?? new SolarCamera(),
      assetUrl,
      terrainLoader,
      loadPlanetTilePack,
      options,
    )
  }

  private constructor(
    private readonly device: GPUDevice,
    date: Date,
    private readonly pipelines: SolarPipelines,
    private readonly invalidate: () => void,
    private readonly onError: (error: Error) => void,
    readonly camera: SolarCamera,
    private readonly assetUrl: WorkspaceAssets['url'],
    private readonly terrainLoader?: TerrainLoader,
    private readonly loadPlanetTilePack?: PlanetTilePackLoader,
    private readonly sceneOptions?: SolarSceneOptions,
  ) {
    const options = sceneOptions
    this.bodies = options?.bodies ?? solarSystemAt(date)
    this.orbits = options?.bodies
      ? []
      : ids.flatMap(id => {
          const vertices = orbitPolyline(id, date)
          return vertices.length ? [{ id, vertices }] : []
        })
    // Per orbit: one 6-vertex capsule quad per segment (12 floats/vertex).
    this.orbitData = new Float32Array(
      Math.max(
        12,
        this.orbits.reduce((count, path) => count + (path.vertices.length - 1) * 6 * 12, 0),
      ),
    )
    this.orbitBuffer = device.createBuffer({
      label: 'Camera-relative orbit ribbons',
      size: this.orbitData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
    })
    const batchCount = Math.max(1, Math.ceil(this.orbits.length / 4))
    this.orbitBatchBuffers = Array.from({ length: batchCount }, (_, i) =>
      device.createBuffer({
        label: `Orbit batch colors ${i}`,
        size: 64,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
      }),
    )
    this.viewBuffer = device.createBuffer({
      label: 'Heliocentric camera basis',
      size: 96,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.annotationViewBuffer = device.createBuffer({
      label: 'Annotation camera basis',
      size: 96,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.loupeViewBuffer = device.createBuffer({
      label: 'Loupe sky camera basis',
      size: 96,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.earthAtmoBuffer = device.createBuffer({
      label: 'Earth atmosphere sphere',
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.loupeEarthAtmoBuffer = device.createBuffer({
      label: 'Loupe Earth atmosphere sphere',
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.marsAtmoBuffer = device.createBuffer({
      label: 'Mars atmosphere sphere',
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.loupeMarsAtmoBuffer = device.createBuffer({
      label: 'Loupe Mars atmosphere sphere',
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.haloBuffer = device.createBuffer({
      label: 'Solar glare and lens ghosts',
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    })
    this.bodyBuffer = device.createBuffer({
      label: 'Solar system centers in camera space',
      size: this.bodyData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    })
    this.annotationBodyBuffer = device.createBuffer({
      label: 'Annotation billboard bodies in camera space',
      size: this.bodyData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    })
    this.skySampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'linear',
      maxAnisotropy: 4,
    })
    this.skyTexture = device.createTexture({
      label: 'Dark sky placeholder',
      size: [1, 1],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    device.queue.writeTexture({ texture: this.skyTexture }, new Uint8Array([6, 6, 7, 255]), {}, [1, 1])
    this.skyGroup = this.bindSky(this.viewBuffer, this.skyTexture)
    this.loupeSkyGroup = this.bindSky(this.loupeViewBuffer, this.skyTexture)
    this.haloGroup = device.createBindGroup({
      layout: pipelines.haloPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuffer } },
        { binding: 1, resource: { buffer: this.haloBuffer } },
      ],
    })
    this.lensFlareGroup = device.createBindGroup({
      layout: pipelines.lensFlarePipeline.getBindGroupLayout(0),
      entries: [{ binding: 1, resource: { buffer: this.haloBuffer } }],
    })
    this.earthAtmoGroup = device.createBindGroup({
      layout: pipelines.earthAtmoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuffer } },
        { binding: 1, resource: { buffer: this.earthAtmoBuffer } },
      ],
    })
    this.loupeEarthAtmoGroup = device.createBindGroup({
      layout: pipelines.earthAtmoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.loupeViewBuffer } },
        { binding: 1, resource: { buffer: this.loupeEarthAtmoBuffer } },
      ],
    })
    this.marsAtmoGroup = device.createBindGroup({
      layout: pipelines.earthAtmoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuffer } },
        { binding: 1, resource: { buffer: this.marsAtmoBuffer } },
      ],
    })
    this.loupeMarsAtmoGroup = device.createBindGroup({
      layout: pipelines.earthAtmoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.loupeViewBuffer } },
        { binding: 1, resource: { buffer: this.loupeMarsAtmoBuffer } },
      ],
    })
    this.bodyGroup = device.createBindGroup({
      layout: pipelines.bodyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuffer } },
        { binding: 1, resource: { buffer: this.bodyBuffer } },
      ],
    })
    this.bodyAnnotationGroup = device.createBindGroup({
      layout: pipelines.bodyAnnotationLayout,
      entries: [
        { binding: 0, resource: { buffer: this.annotationViewBuffer } },
        { binding: 1, resource: { buffer: this.annotationBodyBuffer } },
      ],
    })
    const { signal } = this.controller
    void this.assetUrl('textures/nasa-deep-star-map.webp', signal)
      .then(url => loadSkyMap(device, signal, url))
      .then(texture => {
        if (signal.aborted) return
        const group = this.bindSky(this.viewBuffer, texture)
        const loupeGroup = this.bindSky(this.loupeViewBuffer, texture)
        if (this.skyOwned) this.skyTexture.destroy()
        this.skyTexture = texture
        this.skyOwned = false
        this.skyGroup = group
        this.loupeSkyGroup = loupeGroup
      })
      .catch(() => {
        // Keep rendering the planets over dark space if the atlas is unavailable.
      })
      .finally(() => {
        if (signal.aborted) return
        this.skySettled = true
        this.invalidate()
      })
    if (options?.eagerTerrain) this.loadTerrain()
  }

  setBodies(bodies: SolarBodies) {
    this.bodies = bodies
    this.invalidate()
  }

  private bindSky(buffer: GPUBuffer, texture: GPUTexture) {
    return this.device.createBindGroup({
      layout: this.pipelines.skyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: texture.createView() },
        { binding: 2, resource: this.skySampler },
      ],
    })
  }

  get terrainReady() {
    return Boolean(this.terrain)
  }

  loadTerrain() {
    if (this.loadingTerrain || this.terrain || this.controller.signal.aborted) return
    this.loadingTerrain = true
    const { signal } = this.controller
    void PlanetTerrainRenderer.create(
      this.device,
      signal,
      this.assetUrl,
      this.terrainLoader,
      this.invalidate,
      this.loadPlanetTilePack,
      this.sceneOptions?.albedoMaxWidths,
    )
      .then(
        terrain => {
          if (signal.aborted) {
            terrain.destroy()
            return
          }
          this.terrain = terrain
          this.invalidate()
        },
        error => {
          if (!signal.aborted) this.onError(error instanceof Error ? error : new Error(String(error)))
        },
      )
      .finally(() => {
        this.loadingTerrain = false
        if (!signal.aborted) {
          this.terrainSettled = true
          this.invalidate()
        }
      })
  }

  updatePlanetTiles(
    frame: SolarFrame,
    body: SourceBody,
    clientHeight: number,
    pixelRatio: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    if (!this.terrain) return
    const state = this.bodies[body]
    const eyeUnit = scale(worldToGeographic(state, subtract(frame.eye, state.center)), 1 / state.radius)
    const right = worldToGeographic(state, frame.right)
    const up = worldToGeographic(state, frame.up)
    const back = worldToGeographic(state, frame.back)
    const metrics = this.bodyMarkerMetrics(frame, body, clientHeight)
    const physicalMpp = (metrics.scale * state.radius * 1000) / Math.max(1, pixelRatio)
    this.terrain.updateTiles(body, eyeUnit, right, up, back, frame.aspect, physicalMpp, loupe)
  }

  planetTileStatus() {
    return this.terrain?.tileStatus() ?? ''
  }

  sunlight(id: BodyId) {
    const body = this.bodies[id]
    return normalized(worldToGeographic(body, subtract(this.bodies.sun.center, body.center)))
  }

  bodyRadiusCss(frame: SolarFrame, id: BodyId, clientHeight: number) {
    const body = this.bodies[id]
    const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
    const distance = Math.max(1e-6, length(subtract(body.center, frame.eye)))
    return ((radius / distance) * Math.max(1, clientHeight)) / (2 * Math.tan(FOV / 2))
  }

  private detailed(frame: SolarFrame, id: BodyId, clientHeight: number) {
    // Use CSS-pixel height consistently across moving (DPR 1) and settled (DPR 2) passes
    // so bodies never flip between billboard disks and 3D meshes during progressive settle.
    return this.bodyRadiusCss(frame, id, clientHeight) >= 4
  }

  drawEarth(frame: SolarFrame, clientHeight: number) {
    return this.detailed(frame, 'earth', clientHeight)
  }

  /** True iff detailed 3D planet should use analytical per-sample raycasting (< SMALL_PLANET_CSS_PX). */
  isSmallBody(frame: SolarFrame, id: 'earth' | SourceBody, clientHeight: number) {
    if (id !== 'earth' && !this.terrain?.getTerrain(id)) return false
    const radiusCss = this.bodyRadiusCss(frame, id, clientHeight)
    return radiusCss >= 4 && radiusCss < SMALL_PLANET_CSS_PX
  }

  /** Smoothly tapers Moon/Mars mesh displacement to zero across [SMALL_PLANET_CSS_PX, 65] CSS px. */
  bodyReliefFade(frame: SolarFrame, id: BodyId, clientHeight: number) {
    const radiusCss = this.bodyRadiusCss(frame, id, clientHeight)
    const t = Math.max(0, Math.min(1, (radiusCss - SMALL_PLANET_CSS_PX) / 27))
    return t * t * (3 - 2 * t)
  }

  earthMatrix(frame: SolarFrame) {
    return bodyMatrix(frame, this.bodies.earth, EARTH_EQUATORIAL_KM)
  }

  earthEye(frame: SolarFrame): Vec3 {
    const earth = this.bodies.earth
    return scale(worldToGeographic(earth, subtract(frame.eye, earth.center)), 1 / EARTH_EQUATORIAL_KM)
  }

  /** Body-local camera basis + CPU f64 center NDC for analytical small-planet raycasting. */
  bodySmallView(frame: SolarFrame, id: 'earth' | SourceBody, outWidth: number, outHeight: number) {
    const body = this.bodies[id]
    const unitScale = id === 'earth' ? 1 / EARTH_EQUATORIAL_KM : 1
    const right = worldToGeographic(body, frame.right)
    const up = worldToGeographic(body, frame.up)
    const back = worldToGeographic(body, frame.back)
    const [cx, cy, cz] = relativeToEye(frame, body.center)
    const tanY = Math.tan(FOV / 2)
    const tanX = tanY * frame.aspect
    const depthKm = -cz
    const invDepth = 1 / Math.max(1e-6, depthKm)
    const centerNdcX = (cx * invDepth) / tanX
    const centerNdcY = (cy * invDepth) / tanY
    const out = this.smallViewScratch
    out.set([right[0], right[1], right[2], tanX], 0)
    out.set([up[0], up[1], up[2], tanY], 4)
    out.set([back[0], back[1], back[2], depthKm * unitScale], 8)
    out.set([centerNdcX, centerNdcY, 2 / Math.max(1, outWidth), 2 / Math.max(1, outHeight)], 12)
    return out
  }

  private writeViewBuffer(
    buffer: GPUBuffer,
    frame: SolarFrame,
    width: number,
    height: number,
    cropCenterX = 0,
    cropCenterY = 0,
    cropScaleX = 1,
    cropScaleY = 1,
  ) {
    const tangent = Math.tan(FOV / 2)
    const sunRel = relativeToEye(frame, this.bodies.sun.center)
    this.viewScratch.set(frame.right, 0)
    this.viewScratch[3] = frame.aspect
    this.viewScratch.set(frame.up, 4)
    this.viewScratch[7] = tangent
    this.viewScratch.set(frame.back, 8)
    this.viewScratch[11] = frame.near
    this.viewScratch.set(sunRel, 12)
    this.viewScratch[15] = (tangent * 2 * Math.min(cropScaleX, cropScaleY)) / Math.max(1, height)
    this.viewScratch.set([cropCenterX, cropCenterY, cropScaleX, cropScaleY], 16)
    this.viewScratch.set([2 / Math.max(1, width), 2 / Math.max(1, height), 0, 0], 20)
    this.device.queue.writeBuffer(buffer, 0, this.viewScratch)
    return { tangent, sunRel }
  }

  private writeView(frame: SolarFrame, width: number, height: number) {
    const queue = this.device.queue
    const { tangent, sunRel } = this.writeViewBuffer(this.viewBuffer, frame, width, height)
    let sunVis = 0
    let sunNdcX = 10,
      sunNdcY = 10
    let haloVisible = false
    if (-sunRel[2] > frame.near) {
      sunNdcX = sunRel[0] / (-sunRel[2] * tangent * frame.aspect)
      sunNdcY = sunRel[1] / (-sunRel[2] * tangent)
      const sunDist = length(sunRel)
      const sunAng = Math.max(695700 / Math.max(1, sunDist), ((tangent * 2) / height) * 2.6)
      const haloNdcY = (24 * sunAng) / tangent + 0.05
      const haloNdcX = haloNdcY / frame.aspect + 0.05
      haloVisible = Math.abs(sunNdcX) <= 1 + haloNdcX && Math.abs(sunNdcY) <= 1 + haloNdcY
      const toSun = subtract(this.bodies.sun.center, frame.eye)
      const toSunDist = length(toSun)
      const sunDir = scale(toSun, 1 / Math.max(1, toSunDist))
      const sunDiskAng = this.bodies.sun.radius / Math.max(1, toSunDist)
      sunVis = 1
      for (const id of ids) {
        if (id === 'sun') continue
        const body = this.bodies[id]
        const rel = subtract(body.center, frame.eye)
        const proj = dot(rel, sunDir)
        if (proj <= 0 || proj >= toSunDist) continue
        const perpDist = length(subtract(rel, scale(sunDir, proj)))
        const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
        const clearAng = (perpDist - radius) / proj
        const t = Math.max(0, Math.min(1, (clearAng + sunDiskAng) / Math.max(1e-6, 2 * sunDiskAng)))
        sunVis = Math.min(sunVis, t * t * (3 - 2 * t))
      }
    }
    this.haloVisible = haloVisible
    this.haloScratch.set([sunNdcX, sunNdcY, sunVis, frame.aspect])
    queue.writeBuffer(this.haloBuffer, 0, this.haloScratch)
  }

  private writeBodies(
    buffer: GPUBuffer,
    frame: SolarFrame,
    clientHeight: number,
    pixelRatio: number,
    overview: boolean,
    flightTarget?: SourceBody | 'earth',
  ) {
    const tangent = Math.tan(FOV / 2)
    // Sort bodies farthest-to-nearest so distant billboard disks composite cleanly over space.
    const ordered = ids
      .map(id => {
        const delta = subtract(this.bodies[id].center, frame.eye)
        return { id, distSq: dot(delta, delta) }
      })
      .sort((a, b) => b.distSq - a.distSq)
    const meshFlags: boolean[] = []
    for (const [index, { id }] of ordered.entries()) {
      const body = this.bodies[id]
      const mesh =
        (id === 'earth' || ((id === 'moon' || id === 'mars') && Boolean(this.terrain?.getTerrain(id)))) &&
        this.detailed(frame, id, clientHeight)
      meshFlags.push(mesh)
      const color = colors[id]
      const position = relativeToEye(frame, body.center)
      let glint = 0
      if (!mesh && id === flightTarget && -position[2] > frame.near) {
        const [x, y, z] = position
        const radiusCss = (body.radius * Math.max(1, clientHeight)) / (-2 * z * tangent)
        const nx = x / (-z * tangent * frame.aspect),
          ny = y / (-z * tangent)
        if (
          Math.abs(nx) < 1 &&
          Math.abs(ny) < 1 &&
          this.visiblePoint(frame, nx, ny, id, length(subtract(body.center, frame.eye)) - body.radius)
        ) {
          glint = Math.max(0, Math.min(1, (3.5 - radiusCss) / 2.5))
        }
      }
      // Scale CSS disk minimum radius by pixelRatio so billboard disks keep identical CSS size
      // between moving DPR-1 frames and settled DPR-2 passes.
      const minDiskPx = id === 'sun' ? 0 : (overview ? 1.9 : 1 + 2.2 * glint) * pixelRatio
      this.bodyData.set(
        [...position, body.radius, ...color, Number(id === 'sun'), Number(!mesh), minDiskPx, glint, 0],
        index * 12,
      )
    }
    this.device.queue.writeBuffer(buffer, 0, this.bodyData)
    return { ordered, meshFlags }
  }

  private drawBillboardRuns(
    pass: GPURenderPassEncoder,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    meshFlags: readonly boolean[],
  ) {
    let runStart = -1
    const flushBodies = (end: number) => {
      if (runStart < 0) return
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.draw(6, end - runStart, 0, runStart)
      runStart = -1
    }
    for (let i = 0; i < meshFlags.length; i++) {
      if (meshFlags[i]) {
        flushBodies(i)
        continue
      }
      if (runStart < 0) runStart = i
    }
    flushBodies(meshFlags.length)
  }

  draw(
    pass: GPURenderPassEncoder,
    frame: SolarFrame,
    width: number,
    height: number,
    clientHeight: number,
    hdr: boolean,
    pixelRatio: number,
    overview: boolean,
    flightTarget?: SourceBody | 'earth',
    drawEarth?: () => void,
  ) {
    this.writeView(frame, width, height)
    const { ordered, meshFlags } = this.writeBodies(
      this.bodyBuffer,
      frame,
      clientHeight,
      pixelRatio,
      overview,
      flightTarget,
    )
    // 1. Draw all detailed 3D planet meshes front-to-back before skyPipeline. Both large meshes
    //    and small per-sample ellipsoid/sphere meshes write exact matching color and depth per
    //    MSAA sample, so skyPipeline cleanly fills uncovered rim samples and early-Z rejects interiors.
    for (let i = ordered.length - 1; i >= 0; i--) {
      if (!meshFlags[i]) continue
      const { id } = ordered[i]
      if (id === 'earth') {
        drawEarth?.()
      } else if ((id === 'moon' || id === 'mars') && this.terrain?.getTerrain(id)) {
        const metrics = this.bodyMarkerMetrics(frame, id, clientHeight)
        const metresPerPixel = metrics.scale * this.bodies[id].radius * 1000
        const isSmall = this.isSmallBody(frame, id, clientHeight)
        const reliefFade = this.bodyReliefFade(frame, id, clientHeight)
        this.terrain.draw(
          pass,
          id,
          bodyMatrix(frame, this.bodies[id], 1),
          this.sunlight(id),
          this.sourceEye(frame, id),
          metresPerPixel,
          hdr,
          pixelRatio,
          width,
          height,
          isSmall,
          reliefFade,
          isSmall ? this.bodySmallView(frame, id, width, height) : undefined,
        )
      }
    }
    // 2. Draw distant per-sample opaque billboard disks with depth write so opaque disk interiors
    //    occlude farther orbits and bodies while uncovered fringe samples discard cleanly for the sky.
    this.drawBillboardRuns(pass, this.pipelines.bodyPipeline, this.bodyGroup, meshFlags)
    // 3. Draw sky backdrop only where reverse-Z depth is still 0.0 (unoccluded space and uncovered rim samples).
    pass.setPipeline(this.pipelines.skyPipeline)
    pass.setBindGroup(0, this.skyGroup)
    pass.draw(3)
  }

  drawHalo(pass: GPURenderPassEncoder) {
    if (this.haloVisible) {
      pass.setPipeline(this.pipelines.haloPipeline)
      pass.setBindGroup(0, this.haloGroup)
      pass.draw(3)
    }
    if (this.haloScratch[2] > 0.001 && Math.max(Math.abs(this.haloScratch[0]), Math.abs(this.haloScratch[1])) < 1.55) {
      pass.setPipeline(this.pipelines.lensFlarePipeline)
      pass.setBindGroup(0, this.lensFlareGroup)
      pass.draw(3)
    }
  }

  private atmosphereVisible(frame: SolarFrame, pos: Vec3, radius: number, maxScale: number) {
    if (-pos[2] + radius * maxScale <= frame.near) return false
    const tan = Math.tan(FOV / 2)
    const tanX = tan * frame.aspect
    const innerSq = (radius * 0.985) ** 2
    const posSq = dot(pos, pos)
    // If every viewport corner ray passes strictly inside the opaque core, the limb halo is offscreen.
    for (const sx of [-1, 1])
      for (const sy of [-1, 1]) {
        const dx = sx * tanX,
          dy = sy * tan,
          dz = -1
        const inv = 1 / Math.hypot(dx, dy, dz)
        const proj = (pos[0] * dx + pos[1] * dy + pos[2] * dz) * inv
        if (proj <= 0 || posSq - proj * proj >= innerSq) return true
      }
    return false
  }

  private atmoFade(frame: SolarFrame, id: 'earth' | 'mars', clientHeight: number) {
    const body = this.bodies[id]
    const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
    const distance = length(subtract(frame.eye, body.center))
    const radiusPx = ((radius / Math.max(radius, distance)) * Math.max(1, clientHeight)) / (2 * Math.tan(FOV / 2))
    const t = Math.max(0, Math.min(1, (radiusPx - 6) / 12))
    return t * t * (3 - 2 * t)
  }

  private writeEarthAtmo(buffer: GPUBuffer, frame: SolarFrame, strength: number, isSmall: boolean) {
    const pos = relativeToEye(frame, this.bodies.earth.center)
    const pole = toView(frame, this.bodies.earth.rotation[2])
    this.earthAtmoScratch.set([
      pos[0],
      pos[1],
      pos[2],
      EARTH_EQUATORIAL_KM,
      strength,
      0,
      Number(isSmall),
      0,
      pole[0],
      pole[1],
      pole[2],
      EARTH_POLAR_STRETCH,
    ])
    this.device.queue.writeBuffer(buffer, 0, this.earthAtmoScratch)
  }

  drawEarthAtmosphere(
    pass: GPURenderPassEncoder,
    frame: SolarFrame,
    morph: number,
    clientHeight: number,
    isSmall = false,
  ) {
    const strength = morph * this.atmoFade(frame, 'earth', clientHeight)
    if (strength <= 0.001) return
    const pos = relativeToEye(frame, this.bodies.earth.center)
    if (!this.atmosphereVisible(frame, pos, EARTH_EQUATORIAL_KM, 1.058)) return
    this.writeEarthAtmo(this.earthAtmoBuffer, frame, strength, isSmall)
    pass.setPipeline(this.pipelines.earthAtmoPipeline)
    pass.setBindGroup(0, this.earthAtmoGroup)
    pass.draw(3)
  }

  drawLoupeEarthAtmosphere(pass: GPURenderPassEncoder, frame: SolarFrame, morph: number) {
    this.writeEarthAtmo(this.loupeEarthAtmoBuffer, frame, morph, false)
    pass.setPipeline(this.pipelines.earthAtmoPipeline)
    pass.setBindGroup(0, this.loupeEarthAtmoGroup)
    pass.draw(3)
  }

  private writeMarsAtmo(buffer: GPUBuffer, frame: SolarFrame, strength: number, isSmall: boolean) {
    const mars = this.bodies.mars
    const pos = relativeToEye(frame, mars.center)
    this.marsAtmoScratch.set([pos[0], pos[1], pos[2], mars.radius, strength, 1, Number(isSmall), 0, 0, 1, 0, 0])
    this.device.queue.writeBuffer(buffer, 0, this.marsAtmoScratch)
  }

  drawMarsAtmosphere(pass: GPURenderPassEncoder, frame: SolarFrame, clientHeight: number, isSmall = false) {
    if (!this.terrain?.getTerrain('mars') || !this.detailed(frame, 'mars', clientHeight)) return
    const strength = this.atmoFade(frame, 'mars', clientHeight)
    if (strength <= 0.001) return
    const mars = this.bodies.mars
    const pos = relativeToEye(frame, mars.center)
    if (!this.atmosphereVisible(frame, pos, mars.radius, 1.032)) return
    this.writeMarsAtmo(this.marsAtmoBuffer, frame, strength, isSmall)
    pass.setPipeline(this.pipelines.earthAtmoPipeline)
    pass.setBindGroup(0, this.marsAtmoGroup)
    pass.draw(3)
  }

  drawLoupeMarsAtmosphere(pass: GPURenderPassEncoder, frame: SolarFrame) {
    if (!this.terrain?.getTerrain('mars')) return
    this.writeMarsAtmo(this.loupeMarsAtmoBuffer, frame, 1, false)
    pass.setPipeline(this.pipelines.earthAtmoPipeline)
    pass.setBindGroup(0, this.loupeMarsAtmoGroup)
    pass.draw(3)
  }

  drawPostMotionAtmospheres(
    encoder: GPUCommandEncoder,
    multisampleView: GPUTextureView,
    sceneView: GPUTextureView,
    motionView: GPUTextureView | null,
    depthView: GPUTextureView,
    frame: SolarFrame,
    renderEarth: boolean,
    earthMorph: number,
    clientHeight: number,
    earthSmall: boolean,
    marsSmall: boolean,
  ) {
    let showEarth = false
    if (renderEarth && earthMorph > 0.001) {
      const strength = earthMorph * this.atmoFade(frame, 'earth', clientHeight)
      if (
        strength > 0.001 &&
        this.atmosphereVisible(frame, relativeToEye(frame, this.bodies.earth.center), EARTH_EQUATORIAL_KM, 1.058)
      ) {
        this.writeEarthAtmo(this.earthAtmoBuffer, frame, strength, earthSmall)
        showEarth = true
      }
    }
    let showMars = false
    if (this.terrain?.getTerrain('mars') && this.detailed(frame, 'mars', clientHeight)) {
      const strength = this.atmoFade(frame, 'mars', clientHeight)
      const mars = this.bodies.mars
      if (strength > 0.001 && this.atmosphereVisible(frame, relativeToEye(frame, mars.center), mars.radius, 1.032)) {
        this.writeMarsAtmo(this.marsAtmoBuffer, frame, strength, marsSmall)
        showMars = true
      }
    }
    if (!showEarth && !showMars) return
    // Keep 4x MSAA scene & resolved HDR scene updated so display-only follow-up frames stay intact.
    const msaaPass = encoder.beginRenderPass({
      label: 'Earth/Mars atmospheres to 4x MSAA scene',
      colorAttachments: [
        {
          view: multisampleView,
          resolveTarget: sceneView,
          loadOp: 'load',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: { view: depthView, depthReadOnly: true },
    })
    msaaPass.setPipeline(this.pipelines.earthAtmoPipeline)
    if (showEarth) {
      msaaPass.setBindGroup(0, this.earthAtmoGroup)
      msaaPass.draw(3)
    }
    if (showMars) {
      msaaPass.setBindGroup(0, this.marsAtmoGroup)
      msaaPass.draw(3)
    }
    msaaPass.end()
    if (!motionView) return
    // Overlay crisp depth-tested atmospheres onto the 1x motion-blurred target after encodeMotion.
    let motionGroups = this.atmoMotionGroups.get(depthView)
    if (!motionGroups) {
      const layout = this.pipelines.earthAtmoMotionPipeline.getBindGroupLayout(0)
      motionGroups = {
        earth: this.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: this.viewBuffer } },
            { binding: 1, resource: { buffer: this.earthAtmoBuffer } },
            { binding: 2, resource: depthView },
          ],
        }),
        mars: this.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: this.viewBuffer } },
            { binding: 1, resource: { buffer: this.marsAtmoBuffer } },
            { binding: 2, resource: depthView },
          ],
        }),
      }
      this.atmoMotionGroups.set(depthView, motionGroups)
    }
    const motionPass = encoder.beginRenderPass({
      label: 'Earth/Mars atmospheres post-motion',
      colorAttachments: [{ view: motionView, loadOp: 'load', storeOp: 'store' }],
    })
    motionPass.setPipeline(this.pipelines.earthAtmoMotionPipeline)
    if (showEarth) {
      motionPass.setBindGroup(0, motionGroups.earth)
      motionPass.draw(3)
    }
    if (showMars) {
      motionPass.setBindGroup(0, motionGroups.mars)
      motionPass.draw(3)
    }
    motionPass.end()
  }

  drawAnnotations(
    pass: GPURenderPassEncoder,
    frame: SolarFrame,
    clientHeight: number,
    width = Math.max(1, Math.round(clientHeight * frame.aspect)),
    height = Math.max(1, clientHeight),
    overview = false,
    flightTarget?: SourceBody | 'earth',
    includeCoverage = false,
    pixelRatio = height / Math.max(1, clientHeight),
  ) {
    for (const body of sources) {
      if (this.terrain?.getTerrain(body) && this.detailed(frame, body, clientHeight)) {
        const isSmall = this.isSmallBody(frame, body, clientHeight)
        this.terrain.drawAnnotation(
          pass,
          body,
          bodyMatrix(frame, this.bodies[body], 1),
          this.bodyReliefFade(frame, body, clientHeight),
          includeCoverage,
          width,
          height,
          isSmall,
          isSmall ? this.bodySmallView(frame, body, width, height) : undefined,
          this.sourceEye(frame, body),
        )
      }
    }
    // Draw opaque non-mesh billboard disks (e.g., Venus, Sun, or unloaded Moon/Mars) with
    // exact per-sample depth and sky-clear geography so foreground bodies occlude markers behind them.
    this.writeViewBuffer(this.annotationViewBuffer, frame, width, height)
    const { meshFlags } = this.writeBodies(
      this.annotationBodyBuffer,
      frame,
      clientHeight,
      pixelRatio,
      overview,
      flightTarget,
    )
    const pipeline = includeCoverage
      ? this.pipelines.bodyAnnotationCoveragePipeline
      : this.pipelines.bodyAnnotationPipeline
    this.drawBillboardRuns(pass, pipeline, this.bodyAnnotationGroup, meshFlags)
  }

  bodyMarkerMetrics(frame: SolarFrame, id: SourceBody | 'earth', clientHeight: number) {
    const body = this.bodies[id]
    const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
    const distance = length(subtract(frame.eye, body.center))
    const tanHalf = Math.tan(FOV / 2)
    const h = Math.max(1, clientHeight)
    const scale = (2 * tanHalf * Math.max(0.001, distance / radius - 1)) / h
    const radiusPx = ((radius / Math.max(radius, distance)) * h) / (2 * tanHalf)
    const t = Math.max(0, Math.min(1, (radiusPx - 12) / 22))
    const alpha = t * t * (3 - 2 * t)
    return { scale, alpha }
  }

  sourceMatrix(frame: SolarFrame, body: SourceBody) {
    return bodyMatrix(frame, this.bodies[body], 1)
  }

  sourcePosition(body: SourceBody, point: SourcePoint): Vec3 | null {
    const terrain = this.terrain?.getTerrain(body)
    if (!terrain) return null
    const rad = Math.PI / 180
    return terrain.positionAt(point.longitude * rad, point.latitude * rad)
  }

  sourceEye(frame: SolarFrame, body: SourceBody): Vec3 {
    const state = this.bodies[body]
    return worldToGeographic(state, subtract(frame.eye, state.center))
  }

  drawLoupeSky(pass: GPURenderPassEncoder, frame: SolarFrame, nx: number, ny: number, invSx: number, invSy: number) {
    this.writeViewBuffer(this.loupeViewBuffer, frame, LOUPE_TEXTURE_SIZE, LOUPE_TEXTURE_SIZE, nx, ny, invSx, invSy)
    pass.setPipeline(this.pipelines.skyPipeline)
    pass.setBindGroup(0, this.loupeSkyGroup)
    pass.draw(3)
  }

  drawLoupe(
    pass: GPURenderPassEncoder,
    body: SourceBody,
    frame: SolarFrame,
    matrix: Float32Array,
    metresPerPixel: number,
    hdr: boolean,
    pixelRatio: number,
  ) {
    if (this.terrain?.getTerrain(body)) {
      this.terrain.drawLoupe(
        pass,
        body,
        matrix,
        this.sunlight(body),
        this.sourceEye(frame, body),
        metresPerPixel,
        hdr,
        pixelRatio,
        LOUPE_TEXTURE_SIZE,
      )
    }
  }

  labels(frame: SolarFrame): SolarLabel[] {
    const tan = Math.tan(FOV / 2)
    return ids.flatMap(id => {
      if (id === 'moon') return []
      const body = this.bodies[id]
      const [x, y, z] = relativeToEye(frame, body.center)
      const px = 0.5 - x / (2 * z * tan * frame.aspect)
      const py = 0.5 + y / (2 * z * tan)
      if (z >= -frame.near || px < 0 || px > 1 || py < 0 || py > 1) return []
      const distance = length(subtract(frame.eye, body.center)) - body.radius
      if (!this.visiblePoint(frame, px * 2 - 1, 1 - py * 2, id, distance)) return []
      return [{ id, x: px, y: py }]
    })
  }

  private ensureOrbitCoverage(width: number, height: number) {
    const key = `${width},${height}`
    if (key === this.orbitCoverageKey && this.orbitCoverageMsaa && this.orbitAccumMsaa) return
    this.orbitCoverageMsaa?.destroy()
    this.orbitAccumMsaa?.destroy()
    this.orbitCoverageMsaa = this.device.createTexture({
      label: 'Orbit coverage 4x MSAA',
      size: [width, height],
      format: 'rgba8unorm',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.orbitAccumMsaa = this.device.createTexture({
      label: 'Orbit accumulated 4x MSAA',
      size: [width, height],
      format: 'rgba16float',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    const covMsaaView = this.orbitCoverageMsaa.createView()
    const accumMsaaView = this.orbitAccumMsaa.createView()
    this.orbitCoverageMsaaView = covMsaaView
    this.orbitAccumMsaaView = accumMsaaView
    const accumLayout = this.pipelines.orbitAccumPipeline.getBindGroupLayout(0)
    this.orbitAccumGroups = this.orbitBatchBuffers.map(buffer =>
      this.device.createBindGroup({
        layout: accumLayout,
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: covMsaaView },
        ],
      }),
    )
    this.orbitCoverageKey = key
  }

  prepareOrbits(
    encoder: GPUCommandEncoder,
    depthView: GPUTextureView,
    width: number,
    height: number,
    frame: SolarFrame,
    pixelRatio: number,
  ): GPUTextureView | null {
    if (!this.orbits.length) return null
    this.ensureOrbitCoverage(width, height)
    const f = 1 / Math.tan(FOV / 2)
    const widthPx = Math.max(1, width)
    const heightPx = Math.max(1, height)
    const ndcPerPxX = 2 / widthPx
    const ndcPerPxY = 2 / heightPx
    const halfWidthPx = 0.55 * pixelRatio
    // Include the maximum MSAA sample offset at every DPR so analytical fringe pixels never drop a sample.
    const geomPadPx = halfWidthPx + 1.65
    const clipMargin = geomPadPx + 2
    const minX = -clipMargin,
      maxX = widthPx + clipMargin
    const minY = -clipMargin,
      maxY = heightPx + clipMargin
    const near = frame.near
    const data = this.orbitData
    let offset = 0
    const emitVert = (
      cx: number,
      cy: number,
      cw: number,
      oxPx: number,
      oyPx: number,
      axPx: number,
      ayPx: number,
      zA: number,
      bxPx: number,
      byPx: number,
      zB: number,
    ) => {
      data[offset] = cx + oxPx * ndcPerPxX * cw
      data[offset + 1] = cy + oyPx * ndcPerPxY * cw
      data[offset + 2] = 0.5 * cw
      data[offset + 3] = cw
      data[offset + 4] = axPx
      data[offset + 5] = ayPx
      data[offset + 6] = zA
      data[offset + 7] = halfWidthPx
      data[offset + 8] = bxPx
      data[offset + 9] = byPx
      data[offset + 10] = zB
      data[offset + 11] = 0
      offset += 12
    }
    const ranges: { id: BodyId; firstVertex: number; vertexCount: number }[] = []
    for (const { id, vertices } of this.orbits) {
      const segCount = vertices.length - 1
      if (segCount <= 0) continue
      const firstVertex = offset / 12
      let prev = relativeToEye(frame, vertices[0])
      for (let i = 0; i < segCount; i++) {
        const next = relativeToEye(frame, vertices[i + 1])
        let [x0, y0, z0] = prev
        let [x1, y1, z1] = next
        prev = next
        if (z0 >= -near && z1 >= -near) continue
        if (z0 >= -near) {
          const t = (-near - z0) / (z1 - z0)
          x0 += (x1 - x0) * t
          y0 += (y1 - y0) * t
          z0 = -near
        } else if (z1 >= -near) {
          const t = (-near - z0) / (z1 - z0)
          x1 = x0 + (x1 - x0) * t
          y1 = y0 + (y1 - y0) * t
          z1 = -near
        }
        const aw0 = -z0,
          bw0 = -z1
        const axPx0 = (((x0 * f) / frame.aspect / aw0) * 0.5 + 0.5) * widthPx
        const ayPx0 = (0.5 - ((y0 * f) / aw0) * 0.5) * heightPx
        const bxPx0 = (((x1 * f) / frame.aspect / bw0) * 0.5 + 0.5) * widthPx
        const byPx0 = (0.5 - ((y1 * f) / bw0) * 0.5) * heightPx
        const zA0 = near / aw0,
          zB0 = near / bw0
        // Clip long offscreen segments in CPU f64 against the stroke-expanded viewport so
        // endpoint coordinates stay O(viewport) with full f32 precision in the fragment shader.
        const sdx = bxPx0 - axPx0,
          sdy = byPx0 - ayPx0
        let t0 = 0,
          t1 = 1
        const clipEdge = (p: number, q: number) => {
          if (p === 0) return q >= 0
          const r = q / p
          if (p < 0) {
            if (r > t1) return false
            if (r > t0) t0 = r
          } else {
            if (r < t0) return false
            if (r < t1) t1 = r
          }
          return true
        }
        if (
          !clipEdge(-sdx, axPx0 - minX) ||
          !clipEdge(sdx, maxX - axPx0) ||
          !clipEdge(-sdy, ayPx0 - minY) ||
          !clipEdge(sdy, maxY - ayPx0)
        )
          continue
        const axPx = axPx0 + sdx * t0
        const ayPx = ayPx0 + sdy * t0
        const bxPx = axPx0 + sdx * t1
        const byPx = ayPx0 + sdy * t1
        const zA = zA0 + (zB0 - zA0) * t0
        const zB = zA0 + (zB0 - zA0) * t1
        const aw = near / Math.max(1e-30, zA)
        const bw = near / Math.max(1e-30, zB)
        const ax = ((axPx / widthPx) * 2 - 1) * aw
        const ay = (1 - (ayPx / heightPx) * 2) * aw
        const bx = ((bxPx / widthPx) * 2 - 1) * bw
        const by = (1 - (byPx / heightPx) * 2) * bw
        // Clip-space Y-up tangent & normal (ayPx - byPx flips framebuffer Y-down to clip Y-up).
        const dx = bxPx - axPx
        const dy = ayPx - byPx
        const len = Math.hypot(dx, dy)
        if (offset + 72 > data.length) break
        if (len < 1e-4) {
          // End-on / degenerate projected segment: emit an axis-aligned padded quad carrying
          // exact (a, zA) and (b, zB) so per-sample depth-interval clipping stays continuous.
          const nearA = zA >= zB
          const cx = nearA ? ax : bx,
            cy = nearA ? ay : by,
            cw = nearA ? aw : bw
          emitVert(cx, cy, cw, -geomPadPx, -geomPadPx, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(cx, cy, cw, geomPadPx, -geomPadPx, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(cx, cy, cw, -geomPadPx, geomPadPx, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(cx, cy, cw, -geomPadPx, geomPadPx, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(cx, cy, cw, geomPadPx, -geomPadPx, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(cx, cy, cw, geomPadPx, geomPadPx, axPx, ayPx, zA, bxPx, byPx, zB)
        } else {
          const tx = dx / len,
            ty = dy / len
          const nx = -ty,
            ny = tx
          const tOx = tx * geomPadPx,
            tOy = ty * geomPadPx
          const nOx = nx * geomPadPx,
            nOy = ny * geomPadPx
          // Finite-capsule quad padded in both longitudinal and perpendicular directions.
          // Under operation:'max' blend, max_i f(dist_capsule(P, visible(S_i))) == f(dist(P, visible(polyline))).
          emitVert(ax, ay, aw, -tOx - nOx, -tOy - nOy, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(bx, by, bw, tOx - nOx, tOy - nOy, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(ax, ay, aw, -tOx + nOx, -tOy + nOy, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(ax, ay, aw, -tOx + nOx, -tOy + nOy, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(bx, by, bw, tOx - nOx, tOy - nOy, axPx, ayPx, zA, bxPx, byPx, zB)
          emitVert(bx, by, bw, tOx + nOx, tOy + nOy, axPx, ayPx, zA, bxPx, byPx, zB)
        }
      }
      ranges.push({ id, firstVertex, vertexCount: offset / 12 - firstVertex })
    }
    if (!offset) return null
    this.device.queue.writeBuffer(this.orbitBuffer, 0, data.subarray(0, offset))
    const covMsaaView = this.orbitCoverageMsaaView!
    const accumMsaaView = this.orbitAccumMsaaView!
    let coverageDepthGroup = this.orbitDepthGroups.get(depthView)
    if (!coverageDepthGroup) {
      coverageDepthGroup = this.device.createBindGroup({
        layout: this.pipelines.orbitCoveragePipelines[0].getBindGroupLayout(0),
        entries: [{ binding: 0, resource: depthView }],
      })
      this.orbitDepthGroups.set(depthView, coverageDepthGroup)
    }
    const activeBatches: { batch: number; slice: typeof ranges }[] = []
    for (let batch = 0; batch < this.orbitBatchBuffers.length; batch++) {
      const slice = ranges.slice(batch * 4, batch * 4 + 4)
      if (slice.some(r => r.vertexCount > 0)) activeBatches.push({ batch, slice })
    }
    if (!activeBatches.length) return null
    for (let bIdx = 0; bIdx < activeBatches.length; bIdx++) {
      const { batch, slice } = activeBatches[bIdx]
      this.orbitBatchScratch.fill(0)
      const covPass = encoder.beginRenderPass({
        label: `Orbit coverage batch ${batch}`,
        colorAttachments: [
          {
            view: covMsaaView,
            clearValue: [0, 0, 0, 0],
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      covPass.setVertexBuffer(0, this.orbitBuffer)
      for (let ch = 0; ch < slice.length; ch++) {
        const { id, firstVertex, vertexCount } = slice[ch]
        if (!vertexCount) continue
        const color = colors[id]
        this.orbitBatchScratch.set([color[0], color[1], color[2], 0.42], ch * 4)
        covPass.setPipeline(this.pipelines.orbitCoveragePipelines[ch])
        covPass.setBindGroup(0, coverageDepthGroup)
        covPass.draw(vertexCount, 1, firstVertex, 0)
      }
      covPass.end()
      this.device.queue.writeBuffer(this.orbitBatchBuffers[batch], 0, this.orbitBatchScratch)
      const accumPass = encoder.beginRenderPass({
        label: `Orbit accumulate batch ${batch}`,
        colorAttachments: [
          {
            view: accumMsaaView,
            clearValue: [0, 0, 0, 0],
            loadOp: bIdx === 0 ? 'clear' : 'load',
            storeOp: 'store',
          },
        ],
      })
      accumPass.setPipeline(this.pipelines.orbitAccumPipeline)
      accumPass.setBindGroup(0, this.orbitAccumGroups[batch])
      accumPass.draw(3)
      accumPass.end()
    }
    return accumMsaaView
  }

  visiblePoint(frame: SolarFrame, nx: number, ny: number, focus: BodyId, distance: number) {
    const ray = sceneRay(frame, nx, ny)
    return !ids.some(id => {
      if (id === focus) return false
      const body = this.bodies[id]
      return raySphere(frame.eye, ray, body.center, body.radius) < distance
    })
  }

  pickSource(frame: SolarFrame, body: SourceBody, nx: number, ny: number, clientHeight = 1000): SourcePoint | null {
    const terrain = this.terrain?.getTerrain(body)
    if (!terrain) return null
    const state = this.bodies[body]
    const ray = sceneRay(frame, nx, ny)
    if (!Number.isFinite(raySphere(frame.eye, ray, state.center, terrain.radius + 100))) return null
    const eye = worldToGeographic(state, subtract(frame.eye, state.center))
    const direction = worldToGeographic(state, ray)
    const radiusCss = this.bodyRadiusCss(frame, body, clientHeight)
    const reliefFade = this.bodyReliefFade(frame, body, clientHeight)
    const distance =
      radiusCss < SMALL_PLANET_CSS_PX
        ? raySphere(eye, direction, [0, 0, 0], terrain.radius)
        : pickTerrain(terrain, eye, direction, reliefFade)
    if (!Number.isFinite(distance) || !this.visiblePoint(frame, nx, ny, body, distance)) return null
    const [x, y, z] = add(eye, scale(direction, distance))
    return {
      longitude: (Math.atan2(x, z) * 180) / Math.PI,
      latitude: (Math.atan2(y, Math.hypot(x, z)) * 180) / Math.PI,
    }
  }

  destroy() {
    this.controller.abort()
    this.terrain?.destroy()
    if (this.skyOwned) this.skyTexture.destroy()
    this.viewBuffer.destroy()
    this.annotationViewBuffer.destroy()
    this.loupeViewBuffer.destroy()
    this.earthAtmoBuffer.destroy()
    this.loupeEarthAtmoBuffer.destroy()
    this.marsAtmoBuffer.destroy()
    this.loupeMarsAtmoBuffer.destroy()
    this.haloBuffer.destroy()
    this.bodyBuffer.destroy()
    this.annotationBodyBuffer.destroy()
    this.orbitBuffer.destroy()
    this.orbitBatchBuffers.forEach(buffer => buffer.destroy())
    this.orbitCoverageMsaa?.destroy()
    this.orbitAccumMsaa?.destroy()
  }
}

function intersectsBlock(block: { min: Vec3; max: Vec3 }, eye: Vec3, ray: Vec3, nearest: number, pad = 0) {
  return intersectsAabb(block.min, block.max, eye, ray, nearest, pad + 1e-6, 1e-12)
}

// Ray/triangle intersection on the CPU picking mesh (including smooth radial relief taper
// across [SMALL_PLANET_CSS_PX, 65] CSS px), accelerated by 8x8-cell blocks.
function pickTerrain(
  { columns, indices, positions, blocks, radius }: PlanetPickingMesh,
  eye: Vec3,
  ray: Vec3,
  reliefFade = 1,
) {
  const [ox, oy, oz] = eye,
    [dx, dy, dz] = ray
  const tapered = reliefFade < 1
  const blockPad = tapered ? 80 : 0
  let nearest = Infinity
  for (const block of blocks) {
    if (!intersectsBlock(block, eye, ray, nearest, blockPad)) continue
    for (let row = block.y; row < block.endY; row++)
      for (let col = block.x; col < block.endX; col++) {
        const base = (row * columns + col) * 6
        for (let tri = 0; tri < 6; tri += 3) {
          const i = base + tri
          const a = indices[i] * 3,
            b = indices[i + 1] * 3,
            c = indices[i + 2] * 3
          let ax = positions[a],
            ay = positions[a + 1],
            az = positions[a + 2]
          let bx = positions[b],
            by = positions[b + 1],
            bz = positions[b + 2]
          let cx = positions[c],
            cy = positions[c + 1],
            cz = positions[c + 2]
          if (tapered) {
            const sa = reliefFade + (1 - reliefFade) * (radius / Math.hypot(ax, ay, az))
            const sb = reliefFade + (1 - reliefFade) * (radius / Math.hypot(bx, by, bz))
            const sc = reliefFade + (1 - reliefFade) * (radius / Math.hypot(cx, cy, cz))
            ax *= sa
            ay *= sa
            az *= sa
            bx *= sb
            by *= sb
            bz *= sb
            cx *= sc
            cy *= sc
            cz *= sc
          }
          const ex = bx - ax,
            ey = by - ay,
            ez = bz - az
          const fx = cx - ax,
            fy = cy - ay,
            fz = cz - az
          const px = dy * fz - dz * fy,
            py = dz * fx - dx * fz,
            pz = dx * fy - dy * fx
          const determinant = ex * px + ey * py + ez * pz
          if (Math.abs(determinant) < 1e-8) continue
          const sx = ox - ax,
            sy = oy - ay,
            sz = oz - az
          const u = (sx * px + sy * py + sz * pz) / determinant
          if (u < -1e-8 || u > 1 + 1e-8) continue
          const qx = sy * ez - sz * ey,
            qy = sz * ex - sx * ez,
            qz = sx * ey - sy * ex
          const v = (dx * qx + dy * qy + dz * qz) / determinant
          if (v < -1e-8 || u + v > 1 + 1e-8) continue
          const distance = (fx * qx + fy * qy + fz * qz) / determinant
          if (distance > 0 && distance < nearest) nearest = distance
        }
      }
  }
  return nearest
}
