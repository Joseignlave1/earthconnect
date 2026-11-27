import { joinWgsl, shaderModule } from '../shader-test/gpu.ts'
import { type BlackHoleGeometry, unlensBlackHoleUv } from './blackHoleLens.ts'
import {
  COMPANION_STARS,
  CONSTELLATION_LETTERS,
  CROSS_SIGHTLINES,
  FIELD_H,
  FIELD_PAD,
  FIELD_W,
  MAX_PARALLAX_X,
  MAX_PARALLAX_Y,
  projectStar,
  STAR_TINT_RGB,
  trimmedSegment,
  VIEW_H,
  VIEW_W,
} from './constellationData.ts'
import titleWgsl from './constellationTitle.wgsl?raw'
import noiseWgsl from './noise.wgsl?raw'
import {
  CTA_EDGES,
  CTA_PAD,
  CTA_STARS,
  CTA_VIEW_H,
  CTA_VIEW_W,
  getCtaDustAssets,
  getTitleDustAssets,
  type TitleDustAssets,
} from './stardustRasterizer.ts'

export { CTA_VIEW_H, CTA_VIEW_W, getTitleDustAssets, type TitleDustAssets }

const titleShader = joinWgsl(noiseWgsl, titleWgsl)

export type TitleStageRect = {
  leftPx: number
  topPx: number
  widthPx: number
  heightPx: number
}

export const WAKE_RADIUS = 72

const VEL_W = Math.ceil(FIELD_W / 4) // 172
const VEL_H = Math.ceil(FIELD_H / 4) // 80
const STAR_FLOATS = 8 // StarGpu: pos(2), mag(1), phase(1), color(3), spiked(1)

type TitlePipelines = {
  velCompute: GPUComputePipeline
  velLayout: GPUBindGroupLayout
  dustCompute: GPUComputePipeline
  dustLayout: GPUBindGroupLayout
  ctaDustCompute: GPUComputePipeline
  ctaDustLayout: GPUBindGroupLayout
  dustRender: GPURenderPipeline
  edgeRender: GPURenderPipeline
  starRender: GPURenderPipeline
  renderLayout: GPUBindGroupLayout
  ctaDustRender: GPURenderPipeline
  ctaEdgeRender: GPURenderPipeline
  ctaStarRender: GPURenderPipeline
  ctaRenderLayout: GPUBindGroupLayout
}

export type TitlePipelineTarget = {
  format?: GPUTextureFormat
  sampleCount?: 1 | 4
  hasDepth?: boolean
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<TitlePipelines>>>()

function getTitlePipelines(device: GPUDevice, target: TitlePipelineTarget = {}): Promise<TitlePipelines> {
  const format = target.format ?? 'rgba16float'
  const sampleCount = target.sampleCount ?? 4
  const hasDepth = target.hasDepth ?? true
  const key = `${format}:${sampleCount}:${hasDepth ? 'd' : 'nd'}`

  let byKey = pipelineCache.get(device)
  if (!byKey) {
    byKey = new Map()
    pipelineCache.set(device, byKey)
  }
  let pending = byKey.get(key)
  if (!pending) {
    pending = (async () => {
      const module = await shaderModule(device, 'Hero constellation title shader', titleShader)

      const velLayout = device.createBindGroupLayout({
        label: 'Hero title velocity compute layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
          },
        ],
      })
      const dustLayout = device.createBindGroupLayout({
        label: 'Hero title dust compute layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        ],
      })
      const ctaDustLayout = device.createBindGroupLayout({
        label: 'Hero CTA dust compute layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      })
      const vf = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
      const renderLayout = device.createBindGroupLayout({
        label: 'Hero title render layout',
        entries: [
          { binding: 0, visibility: vf, buffer: { type: 'uniform' } },
          { binding: 1, visibility: vf, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: vf, sampler: { type: 'filtering' } },
          { binding: 3, visibility: vf, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 6, visibility: vf, texture: { sampleType: 'float' } },
        ],
      })
      const ctaRenderLayout = device.createBindGroupLayout({
        label: 'Hero CTA render layout',
        entries: [
          { binding: 0, visibility: vf, buffer: { type: 'uniform' } },
          { binding: 1, visibility: vf, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: vf, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 7, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        ],
      })

      const additive: GPUBlendState = {
        color: { srcFactor: 'one', dstFactor: 'one' },
        alpha: { srcFactor: 'zero', dstFactor: 'one' },
      }
      const depthStencil: GPUDepthStencilState | undefined = hasDepth
        ? {
            format: 'depth32float',
            depthWriteEnabled: false,
            depthCompare: 'always',
          }
        : undefined
      const renderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] })
      const ctaRenderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [ctaRenderLayout] })
      const makeRender = (label: string, layout: GPUPipelineLayout, vertexEntry: string, fragmentEntry: string) =>
        device.createRenderPipelineAsync({
          label,
          layout,
          multisample: { count: sampleCount },
          vertex: { module, entryPoint: vertexEntry },
          fragment: { module, entryPoint: fragmentEntry, targets: [{ format, blend: additive }] },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil,
        })

      const [
        velCompute,
        dustCompute,
        ctaDustCompute,
        dustRender,
        edgeRender,
        starRender,
        ctaDustRender,
        ctaEdgeRender,
        ctaStarRender,
      ] = await Promise.all([
        device.createComputePipelineAsync({
          label: 'Hero title fluid velocity compute',
          layout: device.createPipelineLayout({ bindGroupLayouts: [velLayout] }),
          compute: { module, entryPoint: 'velCompute' },
        }),
        device.createComputePipelineAsync({
          label: 'Hero title stardust particle compute',
          layout: device.createPipelineLayout({ bindGroupLayouts: [dustLayout] }),
          compute: { module, entryPoint: 'dustCompute' },
        }),
        device.createComputePipelineAsync({
          label: 'Hero CTA stardust particle compute',
          layout: device.createPipelineLayout({ bindGroupLayouts: [ctaDustLayout] }),
          compute: { module, entryPoint: 'ctaDustCompute' },
        }),
        makeRender('Hero title stardust particle render', renderPipelineLayout, 'dustVertex', 'dustFragment'),
        makeRender('Hero title constellation edges render', renderPipelineLayout, 'edgeVertex', 'edgeFragment'),
        makeRender('Hero title HDR stars render', renderPipelineLayout, 'starVertex', 'starFragment'),
        makeRender('Hero CTA stardust particle render', ctaRenderPipelineLayout, 'ctaDustVertex', 'dustFragment'),
        makeRender('Hero CTA constellation edges render', ctaRenderPipelineLayout, 'edgeVertex', 'ctaEdgeFragment'),
        makeRender('Hero CTA HDR stars render', ctaRenderPipelineLayout, 'ctaStarVertex', 'starFragment'),
      ])

      return {
        velCompute,
        velLayout,
        dustCompute,
        dustLayout,
        ctaDustCompute,
        ctaDustLayout,
        dustRender,
        edgeRender,
        starRender,
        renderLayout,
        ctaDustRender,
        ctaEdgeRender,
        ctaStarRender,
        ctaRenderLayout,
      }
    })()
    byKey.set(key, pending)
    void pending.catch(() => {
      if (byKey?.get(key) === pending) byKey.delete(key)
    })
  }
  return pending
}

function initStarRecord(
  out: Float32Array,
  offset: number,
  star: {
    x: number
    y: number
    mag: 1 | 2 | 3
    tint: keyof typeof STAR_TINT_RGB
    delay: number
  },
  spiked: boolean,
): number {
  const rgb = STAR_TINT_RGB[star.tint]
  out[offset] = star.x
  out[offset + 1] = star.y
  out[offset + 2] = star.mag
  out[offset + 3] = star.delay * 1.37
  out[offset + 4] = rgb[0]
  out[offset + 5] = rgb[1]
  out[offset + 6] = rgb[2]
  out[offset + 7] = spiked ? 1 : 0
  return offset + STAR_FLOATS
}

const TOTAL_STARS = CONSTELLATION_LETTERS.reduce((sum, letter) => sum + letter.stars.length, 0) + COMPANION_STARS.length
const MAX_EDGES = CONSTELLATION_LETTERS.reduce((sum, letter) => sum + letter.edges.length, 0) + CROSS_SIGHTLINES.length
const EDGE_FLOATS = 8

export class ConstellationTitleOverlay {
  private readonly uniformData = new Float32Array(36) // matches TitleUniform (144 bytes)
  private readonly uniformBuffer: GPUBuffer
  private readonly starData = new Float32Array(TOTAL_STARS * STAR_FLOATS)
  private readonly starBuffer: GPUBuffer
  private readonly edgeData = new Float32Array(MAX_EDGES * EDGE_FLOATS)
  private readonly edgeBuffer: GPUBuffer
  private readonly dustHomeBuffer: GPUBuffer
  private readonly dustStateBuffer: GPUBuffer
  private readonly dustCount: number
  private readonly velTextures: readonly [GPUTexture, GPUTexture]
  private readonly sampler: GPUSampler

  private readonly velBindGroups: readonly [GPUBindGroup, GPUBindGroup]
  private readonly dustBindGroups: readonly [GPUBindGroup, GPUBindGroup]
  private readonly renderBindGroups: readonly [GPUBindGroup, GPUBindGroup]

  private ping = 0
  private prepared = false
  private destroyed = false

  private readonly projectedLetterStars = CONSTELLATION_LETTERS.map(letter =>
    letter.stars.map(star => ({
      x: star.x,
      y: star.y,
      z: star.z,
      mag: star.mag,
      tint: star.tint,
      delay: star.delay,
      label: star.label,
    })),
  )
  private readonly projectedCompanionStars = COMPANION_STARS.map(star => ({
    x: star.x,
    y: star.y,
    z: star.z,
    mag: star.mag,
    tint: star.tint,
    delay: star.delay,
    label: star.label,
  }))

  private pointerClient: { x: number; y: number } | null = null
  private smoothParallax = { x: 0, y: 0 }
  private prevCursorView: { x: number; y: number } | null = null
  private lastEdgeCount = 0
  private lastTime = -1
  private starsUploaded = false
  private lastStarParallaxX = 0
  private lastStarParallaxY = 0
  private edgesUploaded = false
  private lastEdgeRatio = 0

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: TitlePipelines,
    assets: {
      dustCount: number
      dustHomeData: Float32Array
      dustInitState: Float32Array
    },
  ) {
    this.dustCount = assets.dustCount

    let sOff = 0
    for (const letter of CONSTELLATION_LETTERS) {
      for (const star of letter.stars) {
        sOff = initStarRecord(this.starData, sOff, star, Boolean(star.label))
      }
    }
    for (const star of COMPANION_STARS) {
      sOff = initStarRecord(this.starData, sOff, star, Boolean(star.label))
    }

    this.uniformBuffer = device.createBuffer({
      label: 'Hero title uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.starBuffer = device.createBuffer({
      label: 'Hero title stars storage',
      size: this.starData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.edgeBuffer = device.createBuffer({
      label: 'Hero title edges storage',
      size: this.edgeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.dustHomeBuffer = device.createBuffer({
      label: 'Hero title dust home storage',
      size: assets.dustHomeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustHomeBuffer, 0, assets.dustHomeData)

    this.dustStateBuffer = device.createBuffer({
      label: 'Hero title dust state storage',
      size: assets.dustInitState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustStateBuffer, 0, assets.dustInitState)

    this.velTextures = [
      device.createTexture({
        label: 'Hero title fluid velocity A',
        size: [VEL_W, VEL_H, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }),
      device.createTexture({
        label: 'Hero title fluid velocity B',
        size: [VEL_W, VEL_H, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }),
    ]

    this.sampler = device.createSampler({
      label: 'Hero title linear sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    const velViews = [this.velTextures[0].createView(), this.velTextures[1].createView()] as const

    // For ping = 0: read from 0 -> write to 1; then render reads 1.
    // For ping = 1: read from 1 -> write to 0; then render reads 0.
    this.velBindGroups = [
      device.createBindGroup({
        layout: pipelines.velLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: velViews[0] },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: velViews[1] },
        ],
      }),
      device.createBindGroup({
        layout: pipelines.velLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: velViews[1] },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: velViews[0] },
        ],
      }),
    ]

    this.dustBindGroups = [
      device.createBindGroup({
        layout: pipelines.dustLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.dustHomeBuffer } },
          { binding: 2, resource: { buffer: this.dustStateBuffer } },
          { binding: 3, resource: velViews[1] },
          { binding: 4, resource: this.sampler },
        ],
      }),
      device.createBindGroup({
        layout: pipelines.dustLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.dustHomeBuffer } },
          { binding: 2, resource: { buffer: this.dustStateBuffer } },
          { binding: 3, resource: velViews[0] },
          { binding: 4, resource: this.sampler },
        ],
      }),
    ]

    this.renderBindGroups = [
      device.createBindGroup({
        layout: pipelines.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.starBuffer } },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.edgeBuffer } },
          { binding: 4, resource: { buffer: this.dustHomeBuffer } },
          { binding: 5, resource: { buffer: this.dustStateBuffer } },
          { binding: 6, resource: velViews[1] },
        ],
      }),
      device.createBindGroup({
        layout: pipelines.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.starBuffer } },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.edgeBuffer } },
          { binding: 4, resource: { buffer: this.dustHomeBuffer } },
          { binding: 5, resource: { buffer: this.dustStateBuffer } },
          { binding: 6, resource: velViews[0] },
        ],
      }),
    ]
  }

  static async fromAssets(device: GPUDevice, assets: TitleDustAssets, target?: TitlePipelineTarget) {
    const pipelines = await getTitlePipelines(device, target)
    return new ConstellationTitleOverlay(device, pipelines, assets)
  }

  static async create(device: GPUDevice, target?: TitlePipelineTarget) {
    const [pipelines, assets] = await Promise.all([getTitlePipelines(device, target), getTitleDustAssets()])
    return new ConstellationTitleOverlay(device, pipelines, assets)
  }

  setPointer(clientX: number | null, clientY: number | null) {
    if (clientX === null || clientY === null) {
      this.pointerClient = null
    } else {
      this.pointerClient = { x: clientX, y: clientY }
    }
  }

  /** Run GPU compute passes (fluid velocity + 24k stardust particles) before the HDR render pass. */
  prepare(
    encoder: GPUCommandEncoder,
    stageRect: TitleStageRect,
    canvasRect: { left: number; top: number; width: number; height: number },
    width: number,
    height: number,
    ratio: number,
    timeSeconds: number,
    reducedMotion: boolean,
    opacity = 1,
    blackHole?: BlackHoleGeometry,
  ) {
    this.prepared = false
    if (this.destroyed || opacity <= 0.001 || stageRect.widthPx < 2 || stageRect.heightPx < 2) return

    const dt = this.lastTime < 0 ? 1 / 60 : Math.min(0.05, Math.max(0.001, timeSeconds - this.lastTime))
    this.lastTime = timeSeconds
    const parallaxLerp = reducedMotion ? 1 : 1 - Math.exp(-dt * 3.8)

    let targetParX = 0
    let targetParY = 0
    let cursorView: { x: number; y: number } | null = null

    if (this.pointerClient && canvasRect.width > 0 && canvasRect.height > 0) {
      const screenU = (this.pointerClient.x - canvasRect.left) / canvasRect.width
      const screenV = (this.pointerClient.y - canvasRect.top) / canvasRect.height
      if (!reducedMotion) {
        targetParX = screenU * 2 - 1
        targetParY = screenV * 2 - 1
      }
      // Map screen cursor through the exact inverse black hole lensing equation so hovering
      // distorted letters slashes pixel-exact under the visible cursor (and never inside the void).
      const sceneUv = blackHole ? unlensBlackHoleUv(screenU, screenV, blackHole) : { u: screenU, v: screenV }
      if (sceneUv) {
        const localPxX = sceneUv.u * width - stageRect.leftPx
        const localPxY = sceneUv.v * height - stageRect.topPx
        const viewX = (localPxX / stageRect.widthPx) * VIEW_W
        const viewY = (localPxY / stageRect.heightPx) * VIEW_H
        const margin = WAKE_RADIUS + 42
        if (viewX > -margin && viewX < VIEW_W + margin && viewY > -margin && viewY < VIEW_H + margin) {
          cursorView = { x: viewX, y: viewY }
        }
      }
    }

    this.smoothParallax.x += (targetParX - this.smoothParallax.x) * parallaxLerp
    this.smoothParallax.y += (targetParY - this.smoothParallax.y) * parallaxLerp
    const parallax = reducedMotion ? { x: 0, y: 0 } : this.smoothParallax

    let slashAx = 0
    let slashAy = 0
    let slashBx = 0
    let slashBy = 0
    let slashVx = 0
    let slashVy = 0
    let slashActive = 0
    let brushRadius = WAKE_RADIUS

    if (cursorView) {
      const prev = this.prevCursorView ?? cursorView
      slashAx = prev.x
      slashAy = prev.y
      slashBx = cursorView.x
      slashBy = cursorView.y
      const dx = slashBx - slashAx
      const dy = slashBy - slashAy
      const dist = Math.hypot(dx, dy)
      const speed = Math.min(1800, dist / dt)
      if (dist > 0.01) {
        const clamped = Math.min(720, speed)
        slashVx = (dx / dist) * clamped
        slashVy = (dy / dist) * clamped
      }
      brushRadius = WAKE_RADIUS + Math.min(18, speed * 0.022)
      slashActive = 1
      this.prevCursorView = cursorView
    } else {
      this.prevCursorView = null
    }

    // Only re-project and upload stars & edges when parallax or pixel ratio actually changes.
    const starsDirty =
      !this.starsUploaded ||
      Math.abs(parallax.x - this.lastStarParallaxX) > 1e-6 ||
      Math.abs(parallax.y - this.lastStarParallaxY) > 1e-6
    const edgesDirty = !this.edgesUploaded || starsDirty || Math.abs(ratio - this.lastEdgeRatio) > 1e-6
    if (starsDirty) {
      let sOff = 0
      for (let li = 0; li < CONSTELLATION_LETTERS.length; li++) {
        const srcStars = CONSTELLATION_LETTERS[li].stars
        const dstStars = this.projectedLetterStars[li]
        for (let si = 0; si < srcStars.length; si++) {
          const pos = projectStar(srcStars[si], parallax)
          const dst = dstStars[si]
          dst.x = pos.x
          dst.y = pos.y
          this.starData[sOff] = pos.x
          this.starData[sOff + 1] = pos.y
          sOff += STAR_FLOATS
        }
      }
      for (let ci = 0; ci < COMPANION_STARS.length; ci++) {
        const pos = projectStar(COMPANION_STARS[ci], parallax)
        const dst = this.projectedCompanionStars[ci]
        dst.x = pos.x
        dst.y = pos.y
        this.starData[sOff] = pos.x
        this.starData[sOff + 1] = pos.y
        sOff += STAR_FLOATS
      }
      this.device.queue.writeBuffer(this.starBuffer, 0, this.starData)
      this.starsUploaded = true
      this.lastStarParallaxX = parallax.x
      this.lastStarParallaxY = parallax.y
    }

    if (edgesDirty) {
      let eOff = 0
      let edgeCount = 0
      const pushEdge = (
        x1: number,
        y1: number,
        x2: number,
        y2: number,
        baseOp: number,
        baseWidthPx: number,
        kind: number,
      ) => {
        this.edgeData[eOff++] = x1
        this.edgeData[eOff++] = y1
        this.edgeData[eOff++] = x2
        this.edgeData[eOff++] = y2
        this.edgeData[eOff++] = baseOp
        this.edgeData[eOff++] = baseWidthPx
        this.edgeData[eOff++] = kind
        this.edgeData[eOff++] = 0
        edgeCount++
      }

      for (let li = 0; li < CONSTELLATION_LETTERS.length; li++) {
        const letter = CONSTELLATION_LETTERS[li]
        const stars = this.projectedLetterStars[li]
        for (const [i, j] of letter.edges) {
          const a = stars[i]
          const b = stars[j]
          const seg = trimmedSegment(a, b)
          if (!seg) continue
          const avgDepth = (a.z + b.z) * 0.5
          pushEdge(seg.x1, seg.y1, seg.x2, seg.y2, 0.09 + avgDepth * 0.08, (0.65 + avgDepth * 0.22) * ratio, 0)
        }
      }

      for (const [refA, refB] of CROSS_SIGHTLINES) {
        const a =
          'companion' in refA
            ? this.projectedCompanionStars[refA.companion]
            : this.projectedLetterStars[refA.letter][refA.star]
        const b =
          'companion' in refB
            ? this.projectedCompanionStars[refB.companion]
            : this.projectedLetterStars[refB.letter][refB.star]
        const seg = trimmedSegment(a, b)
        if (!seg) continue
        pushEdge(seg.x1, seg.y1, seg.x2, seg.y2, 0.1, 0.7 * ratio, 1)
      }
      this.lastEdgeCount = edgeCount
      this.edgesUploaded = true
      this.lastEdgeRatio = ratio
      if (edgeCount > 0) {
        this.device.queue.writeBuffer(this.edgeBuffer, 0, this.edgeData.subarray(0, edgeCount * EDGE_FLOATS))
      }
    }

    const outlineShiftX = -parallax.x * 0.35 * MAX_PARALLAX_X
    const outlineShiftY = -parallax.y * 0.35 * MAX_PARALLAX_Y
    this.uniformData.set([width, height, ratio, opacity], 0)
    this.uniformData.set([stageRect.leftPx, stageRect.topPx, stageRect.widthPx, stageRect.heightPx], 4)
    this.uniformData.set([VIEW_W, VIEW_H, FIELD_PAD, 0], 8)
    this.uniformData.set([timeSeconds, dt, outlineShiftX, outlineShiftY], 12)
    this.uniformData.set([this.dustCount, reducedMotion ? 1 : 0, 0, 0], 16)
    this.uniformData.set([slashAx, slashAy, slashBx, slashBy, slashVx, slashVy, slashActive, brushRadius], 20)
    this.uniformData.fill(0, 28)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const compute = encoder.beginComputePass({ label: 'Hero title stardust simulation' })

    // 1. Update 2D fluid velocity & slash wake field.
    compute.setPipeline(this.pipelines.velCompute)
    compute.setBindGroup(0, this.velBindGroups[this.ping])
    compute.dispatchWorkgroups(Math.ceil(VEL_W / 8), Math.ceil(VEL_H / 8))

    // 2. Advance 24k GPU stardust particles forming the typography.
    compute.setPipeline(this.pipelines.dustCompute)
    compute.setBindGroup(0, this.dustBindGroups[this.ping])
    compute.dispatchWorkgroups(Math.ceil(this.dustCount / 128))

    compute.end()
    this.prepared = true
  }

  /** Draw the 24k GPU stardust typography + underlying constellations inside the HDR pass. */
  draw(pass: GPURenderPassEncoder) {
    if (this.destroyed || !this.prepared) return
    this.prepared = false
    const bindGroup = this.renderBindGroups[this.ping]
    pass.setBindGroup(0, bindGroup)

    // 1. 24k GPU stardust particles forming the Cinzel Decorative letterforms.
    pass.setPipeline(this.pipelines.dustRender)
    pass.draw(6, this.dustCount)

    // 2. Constellation hairlines & cross-sightlines emerging cleanly inside the parted rift.
    if (this.lastEdgeCount > 0) {
      pass.setPipeline(this.pipelines.edgeRender)
      pass.draw(6, this.lastEdgeCount)
    }

    // 3. True HDR optical stars with scintillation and diffraction spikes.
    pass.setPipeline(this.pipelines.starRender)
    pass.draw(6, TOTAL_STARS)

    // Swap ping-pong index for next frame's compute pass.
    this.ping = 1 - this.ping
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.uniformBuffer.destroy()
    this.starBuffer.destroy()
    this.edgeBuffer.destroy()
    this.dustHomeBuffer.destroy()
    this.dustStateBuffer.destroy()
    this.velTextures[0].destroy()
    this.velTextures[1].destroy()
  }
}

// ============================================================================
// Companion Bottom-Right Stardust CTA ("Explore" + miniature star-chevron)
// Shares the exact same WebGPU HDR stardust pipelines & material as the title.
// ============================================================================

export class StardustCtaOverlay {
  private readonly uniformData = new Float32Array(36) // matches TitleUniform (144 bytes)
  private readonly uniformBuffer: GPUBuffer
  private readonly starData = new Float32Array(CTA_STARS.length * STAR_FLOATS)
  private readonly starBuffer: GPUBuffer
  private readonly edgeData = new Float32Array(CTA_EDGES.length * EDGE_FLOATS)
  private readonly edgeBuffer: GPUBuffer
  private readonly dustHomeBuffer: GPUBuffer
  private readonly dustStateBuffer: GPUBuffer
  private readonly dustCount: number

  private readonly dustBindGroup: GPUBindGroup
  private readonly renderBindGroup: GPUBindGroup

  private prepared = false
  private destroyed = false
  private pointerClient: { x: number; y: number } | null = null
  private smoothParallax = { x: 0, y: 0 }
  private lastEdgeCount = 0
  private lastTime = -1
  private hovered = false
  private focused = false
  private pressed = false
  private hoverLevel = 0
  private pressedLevel = 0
  private starsUploaded = false
  private lastStarKeyX = 0
  private lastStarKeyY = 0
  private lastArrowAdvance = 0
  private edgesUploaded = false
  private lastEdgeRatio = 0
  private lastHitContext: {
    stageRect: TitleStageRect
    canvasRect: { left: number; top: number; width: number; height: number }
    width: number
    height: number
    blackHole?: BlackHoleGeometry
  } | null = null
  private readonly projectedStars = CTA_STARS.map(st => ({ x: st.x, y: st.y }))

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: TitlePipelines,
    assets: { dustCount: number; dustHomeData: Float32Array; dustInitState: Float32Array },
  ) {
    this.dustCount = assets.dustCount

    let sOff = 0
    for (const st of CTA_STARS) {
      sOff = initStarRecord(this.starData, sOff, st, st.spiked)
    }

    this.uniformBuffer = device.createBuffer({
      label: 'Hero CTA uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.starBuffer = device.createBuffer({
      label: 'Hero CTA stars storage',
      size: this.starData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.edgeBuffer = device.createBuffer({
      label: 'Hero CTA edges storage',
      size: this.edgeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.dustHomeBuffer = device.createBuffer({
      label: 'Hero CTA dust home storage',
      size: assets.dustHomeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustHomeBuffer, 0, assets.dustHomeData)

    this.dustStateBuffer = device.createBuffer({
      label: 'Hero CTA dust state storage',
      size: assets.dustInitState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustStateBuffer, 0, assets.dustInitState)

    this.dustBindGroup = device.createBindGroup({
      layout: pipelines.ctaDustLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.dustHomeBuffer } },
        { binding: 5, resource: { buffer: this.dustStateBuffer } },
      ],
    })
    this.renderBindGroup = device.createBindGroup({
      layout: pipelines.ctaRenderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.starBuffer } },
        { binding: 3, resource: { buffer: this.edgeBuffer } },
        { binding: 4, resource: { buffer: this.dustHomeBuffer } },
        { binding: 7, resource: { buffer: this.dustStateBuffer } },
      ],
    })
  }

  static async create(device: GPUDevice) {
    const [pipelines, assets] = await Promise.all([getTitlePipelines(device), getCtaDustAssets()])
    return new StardustCtaOverlay(device, pipelines, assets)
  }

  setPointer(clientX: number | null, clientY: number | null) {
    this.pointerClient = clientX === null || clientY === null ? null : { x: clientX, y: clientY }
  }

  setHovered(hovered: boolean) {
    this.hovered = hovered
  }

  setFocused(focused: boolean) {
    this.focused = focused
  }

  setPressed(pressed: boolean) {
    this.pressed = pressed
  }

  /** Lensed-aware pointer hit test matching the visible GPU CTA stardust bounds. */
  hitTestClient(clientX: number, clientY: number): boolean {
    const ctx = this.lastHitContext
    if (!ctx || ctx.canvasRect.width <= 0 || ctx.canvasRect.height <= 0) return false
    const screenU = (clientX - ctx.canvasRect.left) / ctx.canvasRect.width
    const screenV = (clientY - ctx.canvasRect.top) / ctx.canvasRect.height
    const sceneUv = ctx.blackHole ? unlensBlackHoleUv(screenU, screenV, ctx.blackHole) : { u: screenU, v: screenV }
    if (!sceneUv) return false
    const localPxX = sceneUv.u * ctx.width - ctx.stageRect.leftPx
    const localPxY = sceneUv.v * ctx.height - ctx.stageRect.topPx
    const viewX = (localPxX / ctx.stageRect.widthPx) * CTA_VIEW_W
    const viewY = (localPxY / ctx.stageRect.heightPx) * CTA_VIEW_H
    return viewX >= -6 && viewX <= CTA_VIEW_W + 6 && viewY >= -6 && viewY <= CTA_VIEW_H + 6
  }

  prepare(
    encoder: GPUCommandEncoder,
    stageRect: TitleStageRect,
    canvasRect: { left: number; top: number; width: number; height: number },
    width: number,
    height: number,
    ratio: number,
    timeSeconds: number,
    reducedMotion: boolean,
    opacity = 1,
    blackHole?: BlackHoleGeometry,
  ) {
    this.prepared = false
    if (this.destroyed || opacity <= 0.001 || stageRect.widthPx < 2 || stageRect.heightPx < 2) {
      this.lastHitContext = null
      return
    }
    this.lastHitContext = { stageRect, canvasRect, width, height, blackHole }

    const dt = this.lastTime < 0 ? 1 / 60 : Math.min(0.05, Math.max(0.001, timeSeconds - this.lastTime))
    this.lastTime = timeSeconds
    const parallaxLerp = reducedMotion ? 1 : 1 - Math.exp(-dt * 3.8)

    let targetParX = 0
    let targetParY = 0
    let pointerInside = false

    if (this.pointerClient && canvasRect.width > 0 && canvasRect.height > 0) {
      const screenU = (this.pointerClient.x - canvasRect.left) / canvasRect.width
      const screenV = (this.pointerClient.y - canvasRect.top) / canvasRect.height
      if (!reducedMotion) {
        targetParX = screenU * 2 - 1
        targetParY = screenV * 2 - 1
      }
      pointerInside = this.hitTestClient(this.pointerClient.x, this.pointerClient.y)
    }

    this.smoothParallax.x += (targetParX - this.smoothParallax.x) * parallaxLerp
    this.smoothParallax.y += (targetParY - this.smoothParallax.y) * parallaxLerp
    const parallax = reducedMotion ? { x: 0, y: 0 } : this.smoothParallax

    // Smooth cohesive whole-button hover & press charge (never flees from cursor!).
    // When pointer coordinates are tracked, use the lensed-aware `pointerInside` check so
    // hovering inside the black hole void never triggers the hidden CTA behind it.
    const pointerHover = this.pointerClient ? pointerInside : this.hovered
    const targetHover = pointerHover || this.focused ? 1 : 0
    const hoverRate = targetHover > this.hoverLevel ? 10.5 : 5.5
    this.hoverLevel += (targetHover - this.hoverLevel) * (reducedMotion ? 1 : 1 - Math.exp(-dt * hoverRate))
    const targetPressed = this.pressed ? 1 : 0
    this.pressedLevel += (targetPressed - this.pressedLevel) * (reducedMotion ? 1 : 1 - Math.exp(-dt * 18))

    // Gently nudge the ↗ constellation arrow tip forward along its travel vector on hover for tactile button affordance.
    const arrowAdvance = this.hoverLevel * 6.5 + this.pressedLevel * 2.5
    const starsDirty =
      !this.starsUploaded ||
      Math.abs(parallax.x - this.lastStarKeyX) > 1e-6 ||
      Math.abs(parallax.y - this.lastStarKeyY) > 1e-6 ||
      Math.abs(arrowAdvance - this.lastArrowAdvance) > 1e-6
    const edgesDirty = !this.edgesUploaded || starsDirty || Math.abs(ratio - this.lastEdgeRatio) > 1e-6
    if (starsDirty) {
      const arrowDx = arrowAdvance * 0.94
      const arrowDy = arrowAdvance * -0.34
      let sOff = 0
      for (let idx = 0; idx < CTA_STARS.length; idx++) {
        const st = CTA_STARS[idx]
        const isArrowStar = idx >= 3
        const px = st.x + parallax.x * (st.z - 0.5) * 2.2 + (isArrowStar ? arrowDx : 0)
        const py = st.y + parallax.y * (st.z - 0.5) * 1.6 + (isArrowStar ? arrowDy : 0)
        this.projectedStars[idx].x = px
        this.projectedStars[idx].y = py
        this.starData[sOff] = px
        this.starData[sOff + 1] = py
        sOff += STAR_FLOATS
      }
      this.device.queue.writeBuffer(this.starBuffer, 0, this.starData)
      this.starsUploaded = true
      this.lastStarKeyX = parallax.x
      this.lastStarKeyY = parallax.y
      this.lastArrowAdvance = arrowAdvance
    }

    if (edgesDirty) {
      let eOff = 0
      let edgeCount = 0
      for (const [i, j] of CTA_EDGES) {
        const a = this.projectedStars[i]
        const b = this.projectedStars[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const len = Math.hypot(dx, dy)
        if (len < 6) continue
        const pad = Math.min(3.5, len * 0.14)
        const ux = dx / len
        const uy = dy / len
        this.edgeData[eOff++] = a.x + ux * pad
        this.edgeData[eOff++] = a.y + uy * pad
        this.edgeData[eOff++] = b.x - ux * pad
        this.edgeData[eOff++] = b.y - uy * pad
        this.edgeData[eOff++] = i >= 2 ? 0.24 : 0.14
        this.edgeData[eOff++] = 0.72 * ratio
        this.edgeData[eOff++] = 0
        this.edgeData[eOff++] = 0
        edgeCount++
      }
      this.lastEdgeCount = edgeCount
      this.edgesUploaded = true
      this.lastEdgeRatio = ratio
      if (edgeCount > 0) {
        this.device.queue.writeBuffer(this.edgeBuffer, 0, this.edgeData.subarray(0, edgeCount * EDGE_FLOATS))
      }
    }

    const sinkPxX = (blackHole ? blackHole.cx : 0.5) * width
    const sinkPxY = (blackHole ? blackHole.cy : 2.155) * height
    const sinkViewX = ((sinkPxX - stageRect.leftPx) / stageRect.widthPx) * CTA_VIEW_W
    const sinkViewY = ((sinkPxY - stageRect.topPx) / stageRect.heightPx) * CTA_VIEW_H
    const horizonViewR = (((blackHole ? blackHole.rh : 1.12) * height) / stageRect.heightPx) * CTA_VIEW_H
    const bottomViewY = ((height - stageRect.topPx) / stageRect.heightPx) * CTA_VIEW_H

    const outlineShiftX = -parallax.x * 0.22 * MAX_PARALLAX_X
    const outlineShiftY = -parallax.y * 0.22 * MAX_PARALLAX_Y
    this.uniformData.set([width, height, ratio, opacity], 0)
    this.uniformData.set([stageRect.leftPx, stageRect.topPx, stageRect.widthPx, stageRect.heightPx], 4)
    this.uniformData.set([CTA_VIEW_W, CTA_VIEW_H, CTA_PAD, 0], 8)
    this.uniformData.set([timeSeconds, dt, outlineShiftX, outlineShiftY], 12)
    this.uniformData.set([this.dustCount, reducedMotion ? 1 : 0, 0, 0], 16)
    this.uniformData.fill(0, 20, 28)
    this.uniformData.set(
      [sinkViewX, sinkViewY, horizonViewR, bottomViewY, this.hoverLevel, this.pressedLevel, 0, 0],
      28,
    )
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const compute = encoder.beginComputePass({ label: 'Hero CTA stardust simulation' })
    compute.setPipeline(this.pipelines.ctaDustCompute)
    compute.setBindGroup(0, this.dustBindGroup)
    compute.dispatchWorkgroups(Math.ceil(this.dustCount / 128))
    compute.end()
    this.prepared = true
  }

  draw(pass: GPURenderPassEncoder) {
    if (this.destroyed || !this.prepared) return
    this.prepared = false
    pass.setBindGroup(0, this.renderBindGroup)

    pass.setPipeline(this.pipelines.ctaDustRender)
    pass.draw(6, this.dustCount)

    if (this.lastEdgeCount > 0) {
      pass.setPipeline(this.pipelines.ctaEdgeRender)
      pass.draw(6, this.lastEdgeCount)
    }

    pass.setPipeline(this.pipelines.ctaStarRender)
    pass.draw(6, CTA_STARS.length)
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.uniformBuffer.destroy()
    this.starBuffer.destroy()
    this.edgeBuffer.destroy()
    this.dustHomeBuffer.destroy()
    this.dustStateBuffer.destroy()
  }
}
