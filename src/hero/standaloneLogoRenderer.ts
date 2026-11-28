import { joinWgsl, shaderModule } from '../shader-test/gpu.ts'
import colorWgsl from '../shader-test/shaders/color.wgsl?raw'
import fullscreenWgsl from '../shader-test/shaders/fullscreen.wgsl?raw'
import { ConstellationTitleOverlay, type TitleDustAssets, type TitleStageRect } from './constellationTitle.ts'

export type LogoLayoutRect = {
  canvasRect: { left: number; top: number; width: number; height: number }
  stageRect: TitleStageRect
}

// Standalone ACES tone-map composite for the 1x rgba16float logo target (matches BridgeHero exposure 1.06).
const LOGO_TONEMAP_WGSL = joinWgsl(
  fullscreenWgsl,
  colorWgsl,
  `
@group(0) @binding(0) var hdrTex: texture_2d<f32>;

@fragment fn logoTonemapFragment(input: ScreenVertex) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(hdrTex));
  let px = clamp(vec2i(input.position.xy), vec2i(0), dims - vec2i(1));
  let core = textureLoad(hdrTex, px, 0).rgb;
  let mapped = toDisplay(acesToneMap(core * 1.06));
  return vec4f(mapped, 1.);
}
`,
)

/**
 * Self-contained WebGPU renderer for the standalone constellation + stardust logo.
 * Runs identically on an OffscreenCanvas inside a dedicated Web Worker or on a
 * main-thread HTMLCanvasElement fallback, using its own isolated GPUDevice.
 */
export class StandaloneLogoRenderer {
  private hdrTex: GPUTexture | null = null
  private hdrView: GPUTextureView | null = null
  private tonemapBindGroup: GPUBindGroup | null = null
  private hdrW = 0
  private hdrH = 0
  private readonly startMs = performance.now()
  private destroyed = false

  private constructor(
    private readonly canvas: HTMLCanvasElement | OffscreenCanvas,
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly overlay: ConstellationTitleOverlay,
    private readonly tonemapPipeline: GPURenderPipeline,
    private readonly tonemapLayout: GPUBindGroupLayout,
  ) {}

  static async create(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    assetsPromise: Promise<TitleDustAssets> | TitleDustAssets,
    initialLayout: LogoLayoutRect,
    isCancelled: () => boolean,
  ): Promise<StandaloneLogoRenderer | null> {
    if (!navigator.gpu) throw new Error('WebGPU unavailable')
    const adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await navigator.gpu.requestAdapter())
    if (isCancelled()) return null
    if (!adapter) throw new Error('No WebGPU adapter available')

    const device = await adapter.requestDevice()
    if (isCancelled()) {
      device.destroy()
      return null
    }

    const context = canvas.getContext('webgpu')
    if (!context) {
      device.destroy()
      throw new Error('WebGPU canvas context unavailable')
    }

    try {
      const swapFormat = navigator.gpu.getPreferredCanvasFormat()
      canvas.width = Math.max(2, Math.round(initialLayout.canvasRect.width))
      canvas.height = Math.max(2, Math.round(initialLayout.canvasRect.height))
      context.configure({ device, format: swapFormat, alphaMode: 'opaque' })

      const module = await shaderModule(device, 'Standalone constellation logo tone map', LOGO_TONEMAP_WGSL)
      if (isCancelled()) {
        context.unconfigure()
        device.destroy()
        return null
      }

      const tonemapLayout = device.createBindGroupLayout({
        label: 'Standalone logo tone map layout',
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
      })

      // Compile the tone-map pipeline in parallel with awaiting dust assets from the main thread.
      const [tonemapPipeline, assets] = await Promise.all([
        device.createRenderPipelineAsync({
          label: 'Standalone logo tone map pipeline',
          layout: device.createPipelineLayout({ bindGroupLayouts: [tonemapLayout] }),
          vertex: { module, entryPoint: 'screenVertex' },
          fragment: { module, entryPoint: 'logoTonemapFragment', targets: [{ format: swapFormat }] },
          primitive: { topology: 'triangle-list' },
        }),
        Promise.resolve(assetsPromise),
      ])
      if (isCancelled()) {
        context.unconfigure()
        device.destroy()
        return null
      }

      const overlay = await ConstellationTitleOverlay.fromAssets(device, assets, {
        format: 'rgba16float',
        sampleCount: 1,
        hasDepth: false,
      })
      if (isCancelled()) {
        overlay.destroy()
        context.unconfigure()
        device.destroy()
        return null
      }

      return new StandaloneLogoRenderer(canvas, device, context, overlay, tonemapPipeline, tonemapLayout)
    } catch (err) {
      try {
        context.unconfigure()
      } catch {
        // Ignore if device was already lost.
      }
      device.destroy()
      throw err
    }
  }

  setPointer(x: number | null, y: number | null) {
    if (!this.destroyed) this.overlay.setPointer(x, y)
  }

  private syncHdrTarget(w: number, h: number) {
    if (this.hdrTex && this.hdrW === w && this.hdrH === h) return
    this.hdrTex?.destroy()
    this.hdrW = w
    this.hdrH = h
    this.hdrTex = this.device.createTexture({
      label: 'Standalone logo HDR target',
      size: [w, h, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.hdrView = this.hdrTex.createView()
    this.tonemapBindGroup = this.device.createBindGroup({
      layout: this.tonemapLayout,
      entries: [{ binding: 0, resource: this.hdrView }],
    })
  }

  /** Render one frame. Returns true if submitted, false if skipped (e.g. swapchain temporarily unavailable). */
  renderFrame(layout: LogoLayoutRect, reducedMotion: boolean): boolean {
    if (this.destroyed) return false
    const { canvasRect, stageRect } = layout
    const w = Math.max(2, Math.round(canvasRect.width))
    const h = Math.max(2, Math.round(canvasRect.height))
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    this.syncHdrTarget(w, h)
    if (!this.hdrView || !this.tonemapBindGroup) return false

    let swapView: GPUTextureView
    try {
      swapView = this.context.getCurrentTexture().createView()
    } catch {
      return false
    }

    const timeSeconds = (performance.now() - this.startMs) * 0.001
    const encoder = this.device.createCommandEncoder({ label: 'Standalone constellation logo frame' })
    this.overlay.prepare(encoder, stageRect, canvasRect, w, h, 1, timeSeconds, reducedMotion, 1)

    const hdrPass = encoder.beginRenderPass({
      label: 'Standalone constellation logo HDR pass',
      colorAttachments: [
        {
          view: this.hdrView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    })
    this.overlay.draw(hdrPass)
    hdrPass.end()

    const blitPass = encoder.beginRenderPass({
      label: 'Standalone constellation logo tone map',
      colorAttachments: [
        {
          view: swapView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    })
    blitPass.setPipeline(this.tonemapPipeline)
    blitPass.setBindGroup(0, this.tonemapBindGroup)
    blitPass.draw(3)
    blitPass.end()

    this.device.queue.submit([encoder.finish()])
    return true
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.overlay.destroy()
    this.hdrTex?.destroy()
    this.hdrTex = null
    this.hdrView = null
    this.tonemapBindGroup = null
    try {
      this.context.unconfigure()
    } catch {
      // Ignore if device was already lost.
    }
    this.device.destroy()
  }
}
