import { shaderModule } from '../shader-test/gpu.ts'
import blackHoleWgsl from './blackHole.wgsl?raw'

export type BlackHoleGeometry = {
  /** Horizontal center in normalized viewport UV (0..1). */
  cx: number
  /** Vertical center in normalized viewport UV (0..1; >1 is below bottom edge). */
  cy: number
  /** Event horizon radius in viewport-height units (1.0 = full viewport height). */
  rh: number
  /** Current viewport aspect ratio (width / height). */
  aspect: number
  /** Overall gravitational lens deflection strength (0..1+). */
  lensStrength: number
  /** Interstellar photon-ring & corona HDR intensity. */
  ringIntensity: number
  /** Normalized scroll progress (0..1). */
  scroll: number
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v))
}

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

function applyBoundedOffset(u: number, delta: number) {
  if (delta > 0) {
    const room = Math.max(0.999 - u, 0)
    return u + room * (delta / (delta + Math.max(room, 0.06)))
  }
  if (delta < 0) {
    const room = Math.max(u - 0.001, 0)
    const ad = -delta
    return u - room * (ad / (ad + Math.max(room, 0.06)))
  }
  return u
}

/**
 * Exact inverse-screen lookup matching `blackHole.wgsl`: given a screen UV under the
 * user's cursor, returns the pre-lens scene UV that is visually rendered at that pixel,
 * or `null` when the cursor lies inside the pitch-black event horizon shadow.
 */
export function unlensBlackHoleUv(uvX: number, uvY: number, geom: BlackHoleGeometry): { u: number; v: number } | null {
  const dx = (uvX - geom.cx) * geom.aspect
  const dy = uvY - geom.cy
  const r = Math.hypot(dx, dy)
  const dr = r - geom.rh

  // Inside the pitch-black event horizon: nothing behind it is visible or interactive.
  if (dr <= 0) return null

  const reach = 0.33 + (0.54 - 0.33) * smoothstep(0, 0.35, geom.scroll)
  const prox = smoothstep(reach, 0, dr)
  if (prox <= 0.0001) return { u: uvX, v: uvY }

  const invR = 1 / Math.max(r, 1e-5)
  const dirX = dx * invR
  const dirY = dy * invR
  const tangX = -dirY
  const tangY = dirX

  const soften = 0.052
  const baseCurve = 1 / (dr + soften) - 1 / (reach + soften)
  const deflect = baseCurve * 0.027 * prox * geom.lensStrength
  const frameDrag = Math.exp(-dr / 0.075) * 0.014 * geom.lensStrength * prox

  const invAspect = 1 / Math.max(geom.aspect, 0.001)
  const offX = -(dirX * deflect + tangX * frameDrag) * invAspect
  const offY = -(dirY * deflect + tangY * frameDrag)

  return {
    u: applyBoundedOffset(uvX, offX),
    v: applyBoundedOffset(uvY, offY),
  }
}

/**
 * Single source of truth for the black hole event horizon geometry as a function of
 * scroll progress (0..1) and viewport aspect ratio — shared between the WebGPU lensing
 * shader and the CSS `clip-path: circle(...)` revealing the dark site inside the horizon.
 */
export function computeBlackHoleGeometry(scrollRaw: number, aspectRaw: number): BlackHoleGeometry {
  const scroll = clamp01(scrollRaw)
  const aspect = Math.max(0.45, Math.min(3.0, aspectRaw))

  // At scroll = 0: massive black hole centered horizontally just below the bottom fold,
  // with its pitch-black event horizon and critical photon ring cresting ~5.4% above the
  // bottom edge so visitors immediately see it is cut off below the fold.
  const rhStart = Math.max(1.12, aspect * 0.7)
  // At scroll = 1: horizon centered on screen with radius comfortably swallowing all 4 corners.
  const rhFull = Math.max(rhStart, Math.hypot(aspect * 0.5, 0.5) + 0.28)

  const rise = clamp01(scroll / 0.92)
  // Balanced linear-eased climb so the horizon sweeps steadily across the full scroll runway:
  // s=0: cresting ~5.4% above bottom; s=0.25: ~26% up; s=0.50: ~58% up; s=0.88+: swallows all 4 corners.
  const ease = rise * 0.65 + rise * rise * (3 - 2 * rise) * 0.35

  const rh = rhStart + (rhFull - rhStart) * (rise * rise)
  const cyStart = rhStart + 0.946
  const cyEnd = 0.5
  const cy = cyStart + (cyEnd - cyStart) * ease

  // Subtle, eerie gravitational warp at rest; amplifies cleanly as the horizon climbs into frame.
  const lensStrength = 0.9 + smoothstep(0.02, 0.45, scroll) * 0.42
  // Crisp Interstellar photon ring already legible on the peeking crest at scroll=0; full intensity on scroll.
  const ringIntensity = 0.68 + smoothstep(0.01, 0.22, scroll) * 0.45

  return {
    cx: 0.5,
    cy,
    rh,
    aspect,
    lensStrength,
    ringIntensity,
    scroll,
  }
}

type BlackHolePipeline = {
  pipeline: GPURenderPipeline
  layout: GPUBindGroupLayout
}

const pipelineCache = new WeakMap<GPUDevice, Promise<BlackHolePipeline>>()

function getBlackHolePipeline(device: GPUDevice): Promise<BlackHolePipeline> {
  let pending = pipelineCache.get(device)
  if (!pending) {
    pending = (async () => {
      const module = await shaderModule(device, 'Hero black hole gravitational lens shader', blackHoleWgsl)
      const layout = device.createBindGroupLayout({
        label: 'Hero black hole lens layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      })
      const pipeline = await device.createRenderPipelineAsync({
        label: 'Hero black hole gravitational lens',
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: 'lensVertex' },
        fragment: {
          module,
          entryPoint: 'lensFragment',
          targets: [{ format: 'rgba16float' }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      })
      return { pipeline, layout }
    })()
    pipelineCache.set(device, pending)
    void pending.catch(() => {
      if (pipelineCache.get(device) === pending) pipelineCache.delete(device)
    })
  }
  return pending
}

export class BlackHoleLensPass {
  private readonly uniformData = new Float32Array(12) // 3 x vec4f
  private readonly uniformBuffer: GPUBuffer
  private readonly linearSampler: GPUSampler
  private scratchTexture: GPUTexture | null = null
  private scratchView: GPUTextureView | null = null
  private scratchW = 0
  private scratchH = 0
  private bindGroup: GPUBindGroup | null = null
  private destroyed = false

  private constructor(
    private readonly device: GPUDevice,
    private readonly gpu: BlackHolePipeline,
  ) {
    this.uniformBuffer = device.createBuffer({
      label: 'Hero black hole lens uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.linearSampler = device.createSampler({
      label: 'Hero black hole linear sampler',
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
  }

  static async create(device: GPUDevice) {
    const gpu = await getBlackHolePipeline(device)
    return new BlackHoleLensPass(device, gpu)
  }

  private ensureScratch(width: number, height: number) {
    const resized = !this.scratchTexture || this.scratchW !== width || this.scratchH !== height
    if (resized) {
      this.scratchTexture?.destroy()
      this.scratchTexture = this.device.createTexture({
        label: 'Hero black hole HDR scratch',
        size: [width, height],
        format: 'rgba16float',
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.scratchView = this.scratchTexture.createView()
      this.scratchW = width
      this.scratchH = height
      this.bindGroup = this.device.createBindGroup({
        layout: this.gpu.layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: this.scratchView },
          { binding: 2, resource: this.linearSampler },
        ],
      })
    }
  }

  apply(
    encoder: GPUCommandEncoder,
    sceneTexture: GPUTexture,
    sceneView: GPUTextureView,
    width: number,
    height: number,
    ratio: number,
    timeSeconds: number,
    geom: BlackHoleGeometry,
  ) {
    if (this.destroyed || width < 2 || height < 2) return
    if (geom.scroll <= 0.0005 && geom.lensStrength <= 0.0005 && geom.ringIntensity <= 0.0005) return
    this.ensureScratch(width, height)

    // Snapshot resolved pre-bloom HDR scene into scratch, then write lensed HDR + photon ring back to sceneView.
    encoder.copyTextureToTexture({ texture: sceneTexture }, { texture: this.scratchTexture! }, [width, height, 1])

    this.uniformData.set([geom.cx, geom.cy, geom.rh, geom.aspect], 0)
    this.uniformData.set([timeSeconds, geom.scroll, geom.lensStrength, geom.ringIntensity], 4)
    this.uniformData.set([width, height, ratio, 0], 8)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const pass = encoder.beginRenderPass({
      label: 'Hero black hole gravitational lens',
      colorAttachments: [{ view: sceneView, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(this.gpu.pipeline)
    pass.setBindGroup(0, this.bindGroup!)
    pass.draw(3)
    pass.end()
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.scratchTexture?.destroy()
    this.uniformBuffer.destroy()
  }
}
