import { joinWgsl, shaderModule } from '../shader-test/gpu.ts'
import colorWgsl from '../shader-test/shaders/color.wgsl?raw'
import fullscreenWgsl from '../shader-test/shaders/fullscreen.wgsl?raw'
import { ConstellationTitleOverlay, type TitleDustAssets, type TitleStageRect } from './constellationTitle.ts'

export type LogoLayoutRect = {
  canvasRect: { left: number; top: number; width: number; height: number }
  stageRect: TitleStageRect
}

export type LogoWorkerInMessage =
  | {
      type: 'boot'
      canvas: OffscreenCanvas
      layout: LogoLayoutRect
      reducedMotion: boolean
    }
  | { type: 'assets'; assets: TitleDustAssets }
  | { type: 'layout'; layout: LogoLayoutRect }
  | { type: 'pointer'; x: number | null; y: number | null }
  | { type: 'reducedMotion'; value: boolean }
  | { type: 'destroy' }

export type LogoWorkerOutMessage = { type: 'ready' } | { type: 'error'; message: string }

const send = globalThis.postMessage as (message: LogoWorkerOutMessage) => void

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

let disposed = false
let raf = 0
let device: GPUDevice | null = null
let context: GPUCanvasContext | null = null
let canvas: OffscreenCanvas | null = null
let overlay: ConstellationTitleOverlay | null = null
let tonemapPipeline: GPURenderPipeline | null = null
let tonemapLayout: GPUBindGroupLayout | null = null
let hdrTex: GPUTexture | null = null
let hdrView: GPUTextureView | null = null
let tonemapBindGroup: GPUBindGroup | null = null
let hdrW = 0
let hdrH = 0
let layout: LogoLayoutRect | null = null
let reducedMotion = false
let startMs = 0
let announcedReady = false
let pendingPointer: { x: number | null; y: number | null } | null = null

let resolveAssets!: (assets: TitleDustAssets) => void
const assetsPromise = new Promise<TitleDustAssets>(resolve => {
  resolveAssets = resolve
})

function syncHdrTarget(w: number, h: number) {
  if (!device || !tonemapLayout) return
  if (hdrTex && hdrW === w && hdrH === h) return
  hdrTex?.destroy()
  hdrW = w
  hdrH = h
  hdrTex = device.createTexture({
    label: 'Worker logo HDR target',
    size: [w, h, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })
  hdrView = hdrTex.createView()
  tonemapBindGroup = device.createBindGroup({
    layout: tonemapLayout,
    entries: [{ binding: 0, resource: hdrView }],
  })
}

function cleanup() {
  if (disposed) return
  disposed = true
  cancelAnimationFrame(raf)
  raf = 0
  overlay?.destroy()
  overlay = null
  hdrTex?.destroy()
  hdrTex = null
  hdrView = null
  tonemapBindGroup = null
  try {
    context?.unconfigure()
  } catch {
    // Ignore if device was already lost.
  }
  context = null
  device?.destroy()
  device = null
}

function frame() {
  raf = 0
  if (disposed || !device || !overlay || !context || !canvas || !tonemapPipeline || !layout) return

  const { canvasRect, stageRect } = layout
  const w = Math.max(2, Math.round(canvasRect.width))
  const h = Math.max(2, Math.round(canvasRect.height))
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w
    canvas.height = h
  }
  syncHdrTarget(w, h)
  if (!hdrView || !tonemapBindGroup) return

  let swapView: GPUTextureView
  try {
    swapView = context.getCurrentTexture().createView()
  } catch {
    raf = requestAnimationFrame(frame)
    return
  }

  const timeSeconds = (performance.now() - startMs) * 0.001
  const encoder = device.createCommandEncoder({ label: 'Worker constellation logo frame' })
  overlay.prepare(encoder, stageRect, canvasRect, w, h, 1, timeSeconds, reducedMotion, 1)

  const hdrPass = encoder.beginRenderPass({
    label: 'Worker constellation logo HDR pass',
    colorAttachments: [
      {
        view: hdrView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  })
  overlay.draw(hdrPass)
  hdrPass.end()

  const blitPass = encoder.beginRenderPass({
    label: 'Worker constellation logo tone map',
    colorAttachments: [
      {
        view: swapView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  })
  blitPass.setPipeline(tonemapPipeline)
  blitPass.setBindGroup(0, tonemapBindGroup)
  blitPass.draw(3)
  blitPass.end()

  device.queue.submit([encoder.finish()])

  if (!announcedReady) {
    announcedReady = true
    requestAnimationFrame(() => {
      if (!disposed) send({ type: 'ready' })
    })
  }
  raf = requestAnimationFrame(frame)
}

async function boot(msg: Extract<LogoWorkerInMessage, { type: 'boot' }>) {
  try {
    if (!navigator.gpu) throw new Error('WebGPU unavailable in worker')
    canvas = msg.canvas
    layout = msg.layout
    reducedMotion = msg.reducedMotion
    startMs = performance.now()

    const adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await navigator.gpu.requestAdapter())
    if (disposed) return
    if (!adapter) throw new Error('No GPU adapter in worker')
    device = await adapter.requestDevice()
    if (disposed) {
      device.destroy()
      device = null
      return
    }
    context = canvas.getContext('webgpu')
    if (!context) throw new Error('WebGPU canvas context unavailable in worker')

    const swapFormat = navigator.gpu.getPreferredCanvasFormat()
    const w = Math.max(2, Math.round(layout.canvasRect.width))
    const h = Math.max(2, Math.round(layout.canvasRect.height))
    canvas.width = w
    canvas.height = h
    context.configure({ device, format: swapFormat, alphaMode: 'opaque' })

    const module = await shaderModule(device, 'Worker constellation logo tone map', LOGO_TONEMAP_WGSL)
    if (disposed) return
    tonemapLayout = device.createBindGroupLayout({
      label: 'Worker logo tone map layout',
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
    })

    // Compile the tone-map pipeline in parallel with waiting for the main thread's font/EDT dust assets.
    const [createdTonemap, assets] = await Promise.all([
      device.createRenderPipelineAsync({
        label: 'Worker logo tone map pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [tonemapLayout] }),
        vertex: { module, entryPoint: 'screenVertex' },
        fragment: { module, entryPoint: 'logoTonemapFragment', targets: [{ format: swapFormat }] },
        primitive: { topology: 'triangle-list' },
      }),
      assetsPromise,
    ])
    if (disposed) return
    tonemapPipeline = createdTonemap

    const createdOverlay = await ConstellationTitleOverlay.fromAssets(device, assets, {
      format: 'rgba16float',
      sampleCount: 1,
      hasDepth: false,
    })
    if (disposed) {
      createdOverlay.destroy()
      return
    }
    overlay = createdOverlay
    if (pendingPointer) {
      overlay.setPointer(pendingPointer.x, pendingPointer.y)
      pendingPointer = null
    }
    raf = requestAnimationFrame(frame)
  } catch (err) {
    if (!disposed) {
      cleanup()
      send({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }
}

globalThis.onmessage = (event: MessageEvent<LogoWorkerInMessage>) => {
  const msg = event.data
  if (msg.type === 'boot') {
    void boot(msg)
  } else if (msg.type === 'assets') {
    resolveAssets(msg.assets)
  } else if (msg.type === 'layout') {
    layout = msg.layout
  } else if (msg.type === 'pointer') {
    if (overlay) overlay.setPointer(msg.x, msg.y)
    else pendingPointer = { x: msg.x, y: msg.y }
  } else if (msg.type === 'reducedMotion') {
    reducedMotion = msg.value
  } else if (msg.type === 'destroy') {
    cleanup()
    globalThis.close()
  }
}
