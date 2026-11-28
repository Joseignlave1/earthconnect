import { useEffect, useRef, useState } from 'react'
import { BayerAnnotations } from './BayerAnnotations.tsx'
import type { LogoLayoutRect, LogoWorkerInMessage, LogoWorkerOutMessage } from './constellationLogo.worker.ts'
import { getTitleDustAssets, type TitleDustAssets } from './constellationTitle.ts'
import { StandaloneLogoRenderer } from './standaloneLogoRenderer.ts'

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

export function ConstellationLogo({ active = true, onReady }: { active?: boolean; onReady?: () => void }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
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
    let fallbackRenderer: StandaloneLogoRenderer | null = null
    let scheduleFallbackFrame: (() => void) | null = null

    // Create the canvas element imperatively per effect lifecycle so React StrictMode double-mounts
    // always get a fresh canvas for transferControlToOffscreen().
    let canvasGeneration = 0
    const createLogoCanvas = () => {
      canvasGeneration++
      const el = document.createElement('canvas')
      el.className = 'startup-constellation-canvas'
      el.setAttribute('aria-hidden', 'true')
      wrap.insertBefore(el, stage)
      return el
    }

    let canvas = createLogoCanvas()
    let cachedLayout = measureLogoLayout(canvas, stage)
    const reducedMedia = window.matchMedia('(prefers-reduced-motion: reduce)')

    const pushLayout = () => {
      if (disposed) return
      cachedLayout = measureLogoLayout(canvas, stage)
      worker?.postMessage({ type: 'layout', layout: cachedLayout } satisfies LogoWorkerInMessage)
      scheduleFallbackFrame?.()
    }

    const resizeObserver = new ResizeObserver(pushLayout)
    resizeObserver.observe(wrap)
    resizeObserver.observe(stage)
    resizeObserver.observe(canvas)
    window.addEventListener('resize', pushLayout, { passive: true })

    const onReducedChange = () => {
      worker?.postMessage({ type: 'reducedMotion', value: reducedMedia.matches } satisfies LogoWorkerInMessage)
      if (reducedMedia.matches && fallbackRaf) {
        cancelAnimationFrame(fallbackRaf)
        fallbackRaf = 0
      }
      scheduleFallbackFrame?.()
    }
    reducedMedia.addEventListener('change', onReducedChange)

    const onPointerMove = (event: PointerEvent) => {
      const coalesced = event.getCoalescedEvents?.()
      const latest = coalesced && coalesced.length > 0 ? coalesced[coalesced.length - 1] : event
      worker?.postMessage({ type: 'pointer', x: latest.clientX, y: latest.clientY } satisfies LogoWorkerInMessage)
      fallbackRenderer?.setPointer(latest.clientX, latest.clientY)
    }
    const onPointerLeave = () => {
      worker?.postMessage({ type: 'pointer', x: null, y: null } satisfies LogoWorkerInMessage)
      fallbackRenderer?.setPointer(null, null)
    }

    window.addEventListener('pointermove', onPointerMove, { passive: true })
    window.addEventListener('pointerleave', onPointerLeave)

    const retireWorker = (target?: Worker | null) => {
      const w = target ?? worker
      if (!w) return
      w.terminate()
      if (worker === w) worker = null
    }

    const markLive = (gen: number) => {
      if (disposed || gen !== canvasGeneration) return
      canvas.setAttribute('data-live', 'true')
      setGpuFailed(false)
      setGpuLive(true)
      onReadyRef.current?.()
    }

    const failToFallbackText = () => {
      if (disposed) return
      retireWorker()
      setGpuLive(false)
      setGpuFailed(true)
      onReadyRef.current?.()
    }

    // Fallback: dedicated main-thread GPUDevice via the shared StandaloneLogoRenderer.
    const startMainThreadFallback = async (assets: TitleDustAssets) => {
      if (disposed || !navigator.gpu) {
        failToFallbackText()
        return
      }
      try {
        canvas.remove()
        canvas = createLogoCanvas()
        resizeObserver.observe(canvas)
        const gen = canvasGeneration
        setGpuLive(false)
        cachedLayout = measureLogoLayout(canvas, stage)

        const created = await StandaloneLogoRenderer.create(canvas, assets, cachedLayout, () => disposed)
        if (!created || disposed) return
        fallbackRenderer = created

        let announced = false
        const frame = () => {
          fallbackRaf = 0
          if (disposed || !fallbackRenderer) return
          if (fallbackRenderer.renderFrame(cachedLayout, reducedMedia.matches) && !announced) {
            announced = true
            requestAnimationFrame(() => {
              if (!disposed) markLive(gen)
            })
          }
          if (!reducedMedia.matches) {
            fallbackRaf = requestAnimationFrame(frame)
          }
        }
        scheduleFallbackFrame = () => {
          if (!disposed && fallbackRenderer && !fallbackRaf) {
            fallbackRaf = requestAnimationFrame(frame)
          }
        }
        scheduleFallbackFrame()
      } catch {
        failToFallbackText()
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
        const workerGen = canvasGeneration
        const offscreen = canvas.transferControlToOffscreen()
        const logoWorker = new Worker(new URL('./constellationLogo.worker.ts', import.meta.url), { type: 'module' })
        worker = logoWorker

        const assetsPromise = getTitleDustAssets()
        let fellBack = false
        const fallbackFromWorker = (reason?: string) => {
          if (fellBack || disposed) return
          fellBack = true
          if (reason) console.warn('Constellation logo worker fallback:', reason)
          retireWorker(logoWorker)
          void assetsPromise.then(assets => {
            if (!disposed) void startMainThreadFallback(assets)
          }, failToFallbackText)
        }

        logoWorker.onmessage = (event: MessageEvent<LogoWorkerOutMessage>) => {
          if (disposed) return
          if (event.data.type === 'ready') markLive(workerGen)
          else if (event.data.type === 'error') fallbackFromWorker(event.data.message)
        }
        logoWorker.onerror = event => {
          event.preventDefault()
          if (!disposed) fallbackFromWorker(event.message || 'Worker error')
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
        failToFallbackText()
      }
    })()

    return () => {
      disposed = true
      cancelAnimationFrame(fallbackRaf)
      resizeObserver.disconnect()
      window.removeEventListener('resize', pushLayout)
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerleave', onPointerLeave)
      reducedMedia.removeEventListener('change', onReducedChange)
      scheduleFallbackFrame = null
      retireWorker()
      fallbackRenderer?.destroy()
      fallbackRenderer = null
      canvas.remove()
    }
  }, [active])

  return (
    <div ref={wrapRef} className="startup-constellation-wrap" aria-label="Earth Connect">
      <div ref={stageRef} className="startup-constellation-stage">
        {gpuFailed && !gpuLive && (
          <div className="startup-constellation-fallback" aria-hidden="true">
            <span className="startup-constellation-row">Earth</span>
            <span className="startup-constellation-row startup-constellation-row-2">Connect</span>
          </div>
        )}
        <BayerAnnotations
          stageRef={stageRef}
          svgClassName="startup-constellation-bayer"
          labelClassName="startup-bayer-label"
          active={active}
          live={gpuLive}
        />
      </div>
    </div>
  )
}
