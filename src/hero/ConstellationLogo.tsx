import { useEffect, useRef, useState } from 'react'
import { joinWgsl, shaderModule } from '../shader-test/gpu.ts'
import colorWgsl from '../shader-test/shaders/color.wgsl?raw'
import fullscreenWgsl from '../shader-test/shaders/fullscreen.wgsl?raw'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import { BAYER_ANCHORS, projectStar, smoothstep, VIEW_H, VIEW_W } from './constellationData.ts'
import type { LogoLayoutRect, LogoWorkerInMessage, LogoWorkerOutMessage } from './constellationLogo.worker.ts'
import {
  ConstellationTitleOverlay,
  getTitleDustAssets,
  type TitleDustAssets,
  WAKE_RADIUS,
} from './constellationTitle.ts'

// Main-thread fallback ACES tone-map composite (used only if OffscreenCanvas WebGPU workers are unavailable).
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

function measureLogoLayout(canvas: HTMLElement, stage: HTMLElement): LogoLayoutRect {
  const cr = canvas.getBoundingClientRect()
  const sr = stage.getBoundingClientRect()
  return {
    canvasRect: { left: cr.left, top: cr.top, width: cr.width, height: cr.height },
    stageRect: {
      leftPx: sr.left - cr.left,
      topPx: sr.top - cr.top,
      widthPx: sr.width,
      heightPx: sr.height,
    },
  }
}

export function ConstellationLogo({
  active = true,
  onReady,
}: {
  devicePool?: ReturnType<typeof createSurfaceDevicePool>
  active?: boolean
  onReady?: () => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const bayerRefs = useRef<(SVGTextElement | null)[]>([])
  const onReadyRef = useRef(onReady)
  const [gpuLive, setGpuLive] = useState(false)
  const [gpuFailed, setGpuFailed] = useState(false)

  useEffect(() => {
    onReadyRef.current = onReady
  }, [onReady])

  useEffect(() => {
    if (!active) return
    const wrap = wrapRef.current
    const stage = stageRef.current
    if (!wrap || !stage) return
    if (!navigator.gpu) {
      queueMicrotask(() => {
        setGpuFailed(true)
        onReadyRef.current?.()
      })
      return
    }

    let disposed = false
    let worker: Worker | null = null
    let fallbackRaf = 0
    let fallbackDevice: GPUDevice | null = null
    let fallbackOverlay: ConstellationTitleOverlay | null = null
    let fallbackHdrTex: GPUTexture | null = null
    let fallbackContext: GPUCanvasContext | null = null
    let bayerRaf = 0

    // Create the canvas element imperatively per effect lifecycle so React StrictMode double-mounts
    // always get a fresh canvas for transferControlToOffscreen().
    const createLogoCanvas = () => {
      const el = document.createElement('canvas')
      el.className = 'startup-constellation-canvas'
      el.setAttribute('aria-hidden', 'true')
      wrap.insertBefore(el, stage)
      return el
    }

    let canvas = createLogoCanvas()
    let cachedLayout = measureLogoLayout(canvas, stage)
    const reducedMedia = window.matchMedia('(prefers-reduced-motion: reduce)')

    // Lightweight Bayer Greek annotation parallax/focus on the main thread (only animates while pointer moves).
    const smoothPar = { x: 0, y: 0 }
    const targetPar = { x: 0, y: 0 }
    const smoothFocus = BAYER_ANCHORS.map(() => 0)
    const targetFocus = BAYER_ANCHORS.map(() => 0)
    let bayerDirty = false

    const scheduleBayer = () => {
      if (!bayerRaf && !disposed) bayerRaf = requestAnimationFrame(stepBayer)
    }

    const stepBayer = () => {
      bayerRaf = 0
      if (disposed) return
      const reduced = reducedMedia.matches
      const parLerp = reduced ? 0 : 0.06
      smoothPar.x += (targetPar.x - smoothPar.x) * parLerp
      smoothPar.y += (targetPar.y - smoothPar.y) * parLerp
      let moving =
        !reduced && (Math.abs(targetPar.x - smoothPar.x) > 0.0015 || Math.abs(targetPar.y - smoothPar.y) > 0.0015)

      for (let i = 0; i < BAYER_ANCHORS.length; i++) {
        const diff = targetFocus[i] - smoothFocus[i]
        const focusLerp = reduced ? 1 : diff > 0 ? 0.25 : 0.04
        if (Math.abs(diff) > 0.003) {
          smoothFocus[i] += diff * focusLerp
          moving = true
        } else {
          smoothFocus[i] = targetFocus[i]
        }
        const el = bayerRefs.current[i]
        if (!el) continue
        const anchor = BAYER_ANCHORS[i]
        const pos = projectStar(anchor, reduced ? { x: 0, y: 0 } : smoothPar)
        el.setAttribute('x', (pos.x + 8).toFixed(2))
        el.setAttribute('y', (pos.y - 7).toFixed(2))
        el.style.opacity = (0.56 + smoothFocus[i] * 0.44).toFixed(3)
      }
      if (moving || bayerDirty) {
        bayerDirty = false
        scheduleBayer()
      }
    }

    const pushLayout = () => {
      if (disposed) return
      cachedLayout = measureLogoLayout(canvas, stage)
      worker?.postMessage({ type: 'layout', layout: cachedLayout } satisfies LogoWorkerInMessage)
    }

    const resizeObserver = new ResizeObserver(pushLayout)
    resizeObserver.observe(wrap)
    resizeObserver.observe(stage)
    window.addEventListener('resize', pushLayout, { passive: true })

    const onReducedChange = () => {
      worker?.postMessage({ type: 'reducedMotion', value: reducedMedia.matches } satisfies LogoWorkerInMessage)
    }
    reducedMedia.addEventListener('change', onReducedChange)

    const onPointerMove = (event: PointerEvent) => {
      const coalesced = event.getCoalescedEvents?.()
      const latest = coalesced && coalesced.length > 0 ? coalesced[coalesced.length - 1] : event
      const cx = latest.clientX
      const cy = latest.clientY
      worker?.postMessage({ type: 'pointer', x: cx, y: cy } satisfies LogoWorkerInMessage)
      fallbackOverlay?.setPointer(cx, cy)

      targetPar.x = (cx / Math.max(1, window.innerWidth)) * 2 - 1
      targetPar.y = (cy / Math.max(1, window.innerHeight)) * 2 - 1
      const { canvasRect, stageRect } = cachedLayout
      const stageLeft = canvasRect.left + stageRect.leftPx
      const stageTop = canvasRect.top + stageRect.topPx
      if (stageRect.widthPx > 0 && stageRect.heightPx > 0) {
        const viewX = ((cx - stageLeft) / stageRect.widthPx) * VIEW_W
        const viewY = ((cy - stageTop) / stageRect.heightPx) * VIEW_H
        for (let i = 0; i < BAYER_ANCHORS.length; i++) {
          const anchor = BAYER_ANCHORS[i]
          const dist = Math.hypot(viewX - anchor.x, viewY - anchor.y)
          targetFocus[i] = smoothstep(WAKE_RADIUS * 1.35, WAKE_RADIUS * 0.3, dist)
        }
      }
      bayerDirty = true
      scheduleBayer()
    }

    const onPointerLeave = () => {
      worker?.postMessage({ type: 'pointer', x: null, y: null } satisfies LogoWorkerInMessage)
      fallbackOverlay?.setPointer(null, null)
      targetPar.x = 0
      targetPar.y = 0
      targetFocus.fill(0)
      bayerDirty = true
      scheduleBayer()
    }

    window.addEventListener('pointermove', onPointerMove, { passive: true })
    window.addEventListener('pointerleave', onPointerLeave)

    const markLive = () => {
      if (disposed) return
      canvas.setAttribute('data-live', 'true')
      setGpuLive(true)
      onReadyRef.current?.()
    }

    // Fallback: dedicated main-thread GPUDevice (still isolated from SurfaceRenderer's GPU queue & zero reflows).
    const startMainThreadFallback = async (assets: TitleDustAssets) => {
      if (disposed || !navigator.gpu) return
      try {
        // Replace canvas in case transferControlToOffscreen had already been called on the first element.
        canvas.remove()
        canvas = createLogoCanvas()
        cachedLayout = measureLogoLayout(canvas, stage)

        const adapter =
          (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
          (await navigator.gpu.requestAdapter())
        if (disposed || !adapter) throw new Error('No WebGPU adapter')
        const device = await adapter.requestDevice()
        if (disposed) {
          device.destroy()
          return
        }
        fallbackDevice = device
        const ctx = canvas.getContext('webgpu')
        if (!ctx) throw new Error('WebGPU canvas context unavailable')
        fallbackContext = ctx
        const swapFormat = navigator.gpu.getPreferredCanvasFormat()
        ctx.configure({ device, format: swapFormat, alphaMode: 'opaque' })

        const module = await shaderModule(device, 'Fallback logo tone map', LOGO_TONEMAP_WGSL)
        if (disposed) return
        const tonemapLayout = device.createBindGroupLayout({
          label: 'Fallback logo tone map layout',
          entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
        })
        const [overlay, tonemapPipeline] = await Promise.all([
          ConstellationTitleOverlay.fromAssets(device, assets, {
            format: 'rgba16float',
            sampleCount: 1,
            hasDepth: false,
          }),
          device.createRenderPipelineAsync({
            label: 'Fallback logo tone map pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [tonemapLayout] }),
            vertex: { module, entryPoint: 'screenVertex' },
            fragment: { module, entryPoint: 'logoTonemapFragment', targets: [{ format: swapFormat }] },
            primitive: { topology: 'triangle-list' },
          }),
        ])
        if (disposed) {
          overlay.destroy()
          return
        }
        fallbackOverlay = overlay

        let hdrView: GPUTextureView | null = null
        let tonemapBindGroup: GPUBindGroup | null = null
        let hdrW = 0
        let hdrH = 0
        let announced = false
        const startMs = performance.now()

        const frame = () => {
          fallbackRaf = 0
          if (disposed) return
          const { canvasRect, stageRect } = cachedLayout
          const w = Math.max(2, Math.round(canvasRect.width))
          const h = Math.max(2, Math.round(canvasRect.height))
          if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w
            canvas.height = h
          }
          if (!fallbackHdrTex || hdrW !== w || hdrH !== h) {
            fallbackHdrTex?.destroy()
            hdrW = w
            hdrH = h
            fallbackHdrTex = device.createTexture({
              label: 'Fallback logo HDR target',
              size: [w, h, 1],
              format: 'rgba16float',
              usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            })
            hdrView = fallbackHdrTex.createView()
            tonemapBindGroup = device.createBindGroup({
              layout: tonemapLayout,
              entries: [{ binding: 0, resource: hdrView }],
            })
          }
          if (!hdrView || !tonemapBindGroup) return

          let swapView: GPUTextureView
          try {
            swapView = ctx.getCurrentTexture().createView()
          } catch {
            fallbackRaf = requestAnimationFrame(frame)
            return
          }

          const timeSeconds = (performance.now() - startMs) * 0.001
          const encoder = device.createCommandEncoder({ label: 'Fallback logo frame' })
          overlay.prepare(encoder, stageRect, canvasRect, w, h, 1, timeSeconds, reducedMedia.matches, 1)

          const hdrPass = encoder.beginRenderPass({
            colorAttachments: [
              { view: hdrView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
            ],
          })
          overlay.draw(hdrPass)
          hdrPass.end()

          const blitPass = encoder.beginRenderPass({
            colorAttachments: [
              { view: swapView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
            ],
          })
          blitPass.setPipeline(tonemapPipeline)
          blitPass.setBindGroup(0, tonemapBindGroup)
          blitPass.draw(3)
          blitPass.end()

          device.queue.submit([encoder.finish()])
          if (!announced) {
            announced = true
            requestAnimationFrame(() => {
              if (!disposed) markLive()
            })
          }
          fallbackRaf = requestAnimationFrame(frame)
        }
        fallbackRaf = requestAnimationFrame(frame)
      } catch {
        if (!disposed) {
          setGpuFailed(true)
          onReadyRef.current?.()
        }
      }
    }

    void (async () => {
      try {
        const canUseWorker = typeof canvas.transferControlToOffscreen === 'function' && typeof Worker === 'function'
        if (!canUseWorker) {
          const assets = await getTitleDustAssets()
          if (!disposed) await startMainThreadFallback(assets)
          return
        }

        // Boot the worker + its dedicated GPUDevice immediately in parallel with font loading & EDT.
        const offscreen = canvas.transferControlToOffscreen()
        const logoWorker = new Worker(new URL('./constellationLogo.worker.ts', import.meta.url), { type: 'module' })
        worker = logoWorker

        const assetsPromise = getTitleDustAssets()
        const fallbackFromWorker = () => {
          logoWorker.terminate()
          if (worker === logoWorker) worker = null
          void assetsPromise.then(
            assets => {
              if (!disposed) void startMainThreadFallback(assets)
            },
            () => {
              if (!disposed) {
                setGpuFailed(true)
                onReadyRef.current?.()
              }
            },
          )
        }

        logoWorker.onmessage = (event: MessageEvent<LogoWorkerOutMessage>) => {
          if (disposed) return
          if (event.data.type === 'ready') {
            markLive()
          } else if (event.data.type === 'error') {
            fallbackFromWorker()
          }
        }
        logoWorker.onerror = event => {
          event.preventDefault()
          if (!disposed) fallbackFromWorker()
        }

        cachedLayout = measureLogoLayout(canvas, stage)
        logoWorker.postMessage(
          {
            type: 'boot',
            canvas: offscreen,
            layout: cachedLayout,
            reducedMotion: reducedMedia.matches,
          } satisfies LogoWorkerInMessage,
          [offscreen],
        )

        const assets = await assetsPromise
        if (disposed || worker !== logoWorker) return

        // Clone Float32Arrays for zero-copy transfer so the cached arrays stay intact for BridgeHero.
        const homeCopy = assets.dustHomeData.slice()
        const stateCopy = assets.dustInitState.slice()
        logoWorker.postMessage(
          {
            type: 'assets',
            assets: {
              dustCount: assets.dustCount,
              dustHomeData: homeCopy,
              dustInitState: stateCopy,
            },
          } satisfies LogoWorkerInMessage,
          [homeCopy.buffer, stateCopy.buffer],
        )
      } catch {
        if (!disposed) {
          setGpuFailed(true)
          onReadyRef.current?.()
        }
      }
    })()

    return () => {
      disposed = true
      cancelAnimationFrame(bayerRaf)
      cancelAnimationFrame(fallbackRaf)
      resizeObserver.disconnect()
      window.removeEventListener('resize', pushLayout)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerleave', onPointerLeave)
      reducedMedia.removeEventListener('change', onReducedChange)
      if (worker) {
        try {
          worker.postMessage({ type: 'destroy' } satisfies LogoWorkerInMessage)
        } catch {
          // Worker may already be terminated.
        }
        worker.terminate()
        worker = null
      }
      fallbackOverlay?.destroy()
      fallbackHdrTex?.destroy()
      try {
        fallbackContext?.unconfigure()
      } catch {
        // Ignore if device was already lost.
      }
      fallbackDevice?.destroy()
      canvas.remove()
    }
  }, [active])

  return (
    <div ref={wrapRef} className="startup-constellation-wrap" aria-label="Earth Connect">
      <div ref={stageRef} className="startup-constellation-stage">
        {gpuFailed && !gpuLive && (
          <div className="startup-constellation-fallback" aria-hidden="true">
            <span className="startup-constellation-row startup-constellation-row-1">Earth</span>
            <span className="startup-constellation-row startup-constellation-row-2">Connect</span>
          </div>
        )}
        <svg
          className="startup-constellation-bayer"
          data-live={gpuLive || undefined}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="presentation"
          aria-hidden="true"
        >
          {BAYER_ANCHORS.map((anchor, index) => (
            <text
              key={anchor.id}
              ref={node => {
                bayerRefs.current[index] = node
              }}
              x={anchor.x + 8}
              y={anchor.y - 7}
              className="startup-bayer-label"
            >
              {anchor.label}
            </text>
          ))}
        </svg>
      </div>
    </div>
  )
}
