import { useEffect, useRef, useState } from 'react'
import { SurfaceRenderer } from '../shader-test/renderer.ts'
import type { EarthWorkspaceResources } from '../workspace/earthWorkspaceResources.ts'
import type { SolarBodies } from '../workspace/solarCamera.ts'
import { SolarScene } from '../workspace/solarScene.ts'
import { scoreManifestUrl, type WorkspaceAssets } from '../workspace/workspaceAssets.ts'
import { BlackHoleLensPass, computeBlackHoleGeometry } from './blackHoleLens.ts'
import { BridgeOverlay } from './bridgeOverlay.ts'
import {
  bridgeEndpoints,
  buildBridgeBodies,
  buildBridgeFrame,
  HERO_BRIDGE_SCORE_SHADER,
  HERO_MARS_SITE,
  projectSkyBox,
} from './bridgeScene.ts'
import { ConstellationTitleOverlay, StardustCtaOverlay } from './constellationTitle.ts'
import { HeroTitle } from './HeroTitle.tsx'
import { LandingEditorial } from './LandingEditorial.tsx'
import './style.css'

type Coordinates = { latitude: number; longitude: number }

const HERO_SCROLL_CUE_MS = 1950

export function BridgeHero({
  assets,
  resources,
  leaving,
  onOpenWorkspace,
  onReady,
}: {
  assets: WorkspaceAssets
  resources: EarthWorkspaceResources
  leaving: boolean
  onOpenWorkspace: () => void
  onReady: () => void
}) {
  const scrollerRef = useRef<HTMLElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const stickyRef = useRef<HTMLDivElement>(null)
  const siteShellRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const titleStageRef = useRef<HTMLDivElement>(null)
  const ctaStageRef = useRef<HTMLAnchorElement>(null)
  const scrollCueRef = useRef<HTMLButtonElement>(null)
  const cancelScrollAnimRef = useRef<(() => void) | null>(null)
  const rendererRef = useRef<SurfaceRenderer | null>(null)
  const ctaOverlayRef = useRef<StardustCtaOverlay | null>(null)
  const scrollProgressRef = useRef(0)
  const scrollPausedRef = useRef(false)
  const leavingRef = useRef(leaving)
  const onReadyRef = useRef(onReady)
  const [error, setError] = useState('')

  useEffect(() => {
    leavingRef.current = leaving
    rendererRef.current?.setPaused(leaving || scrollPausedRef.current)
  }, [leaving])

  useEffect(() => {
    onReadyRef.current = onReady
  }, [onReady])

  useEffect(() => {
    const scroller = scrollerRef.current
    const track = trackRef.current
    const sticky = stickyRef.current
    const siteShell = siteShellRef.current
    if (!scroller || !track || !sticky || !siteShell) return

    const syncScroll = () => {
      // Measure the sticky hero viewport and actual CSS runway so DOM clip and GPU lens stay aligned.
      const vh = Math.max(1, sticky.clientHeight)
      const vw = Math.max(1, sticky.clientWidth)
      const runway = Math.max(1, track.offsetHeight - sticky.offsetHeight)
      const progress = Math.max(0, Math.min(1, scroller.scrollTop / runway))
      const prevProgress = scrollProgressRef.current
      scrollProgressRef.current = progress

      const geom = computeBlackHoleGeometry(progress, vw / vh)
      const cxPx = geom.cx * vw
      const cyPx = geom.cy * vh
      // Match blackHole.wgsl's aspect-scaled distance metric exactly, including outside the aspect clamp.
      const rxPx = (geom.rh / geom.aspect) * vw
      const ryPx = geom.rh * vh

      // Clip the dark site interior to the exact WebGPU event horizon until it swallows the screen.
      if (progress >= 0.96) {
        siteShell.style.clipPath = 'none'
        siteShell.style.opacity = '1'
        siteShell.style.pointerEvents = 'auto'
        siteShell.inert = false
      } else if (progress <= 0.08) {
        siteShell.style.clipPath = `ellipse(0px 0px at ${cxPx.toFixed(1)}px ${cyPx.toFixed(1)}px)`
        siteShell.style.opacity = '0'
        siteShell.style.pointerEvents = 'none'
        siteShell.inert = true
      } else {
        // Slightly inset from the photon ring so DOM content stays cleanly inside the pitch-black shadow.
        const innerRx = Math.max(0, rxPx - 6)
        const innerRy = Math.max(0, ryPx - 6)
        const siteTopInViewport = siteShell.getBoundingClientRect().top
        const localCyPx = cyPx - siteTopInViewport
        siteShell.style.clipPath = `ellipse(${innerRx.toFixed(1)}px ${innerRy.toFixed(1)}px at ${cxPx.toFixed(1)}px ${localCyPx.toFixed(1)}px)`
        siteShell.style.opacity = String(Math.min(1, (progress - 0.14) / 0.35))
        const interactive = progress > 0.55
        siteShell.style.pointerEvents = interactive ? 'auto' : 'none'
        siteShell.inert = !interactive
      }

      // Keep WebGPU title & CTA at full luminosity so the black hole event horizon physically
      // lenses and swallows them (just like Earth & Mars). Fade only non-lensed DOM micro-labels
      // (Bayer Greek letters) early so they don't float unwarped over the rising photon ring.
      const domLabelOpacity = Math.max(0, Math.min(1, 1 - (progress - 0.06) / 0.18))
      sticky.style.setProperty('--hero-bayer-opacity', domLabelOpacity.toFixed(3))
      const scrollCueOpacity = Math.max(0, Math.min(1, 1 - progress / 0.1))
      sticky.style.setProperty('--hero-scroll-cue-opacity', scrollCueOpacity.toFixed(3))
      if (scrollCueRef.current) {
        const cueHidden = progress > 0.06
        scrollCueRef.current.style.pointerEvents = cueHidden ? 'none' : 'auto'
        scrollCueRef.current.tabIndex = cueHidden ? -1 : 0
        scrollCueRef.current.ariaHidden = cueHidden ? 'true' : null
      }
      if (ctaStageRef.current) {
        const ctaHidden = progress > 0.36
        ctaStageRef.current.style.pointerEvents = ctaHidden ? 'none' : 'auto'
        ctaStageRef.current.tabIndex = ctaHidden ? -1 : 0
        ctaStageRef.current.ariaHidden = ctaHidden ? 'true' : null
      }

      // Pause the hero WebGPU renderer once completely inside the black hole void (0% GPU while reading the site).
      const shouldPause = progress >= 0.99
      const pauseChanged = scrollPausedRef.current !== shouldPause
      if (pauseChanged) {
        scrollPausedRef.current = shouldPause
        rendererRef.current?.setPaused(leavingRef.current || shouldPause)
      }
      // Under prefers-reduced-motion, the renderer stops scheduling animation frames once settled,
      // so explicitly request a frame whenever scroll progress moves (including the last visible frame).
      if (Math.abs(progress - prevProgress) > 1e-4 && !leavingRef.current && (!shouldPause || pauseChanged)) {
        rendererRef.current?.invalidateHeroFrame()
      }
    }

    syncScroll()
    scroller.addEventListener('scroll', syncScroll, { passive: true })
    window.addEventListener('resize', syncScroll, { passive: true })
    return () => {
      cancelScrollAnimRef.current?.()
      scroller.removeEventListener('scroll', syncScroll)
      window.removeEventListener('resize', syncScroll)
    }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const controller = new AbortController()
    const { signal } = controller
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const pointerTarget = { x: 0, y: 0 }
    const pointerSmooth = { x: 0, y: 0 }

    let earthTarget: Coordinates = { latitude: 30, longitude: 88 }
    let currentAspect = Math.max(0.55, canvas.clientWidth / Math.max(1, canvas.clientHeight))
    let currentBodies: SolarBodies = buildBridgeBodies(currentAspect, earthTarget)
    let renderer: SurfaceRenderer | undefined
    let overlay: BridgeOverlay | undefined
    let titleOverlay: ConstellationTitleOverlay | undefined
    let ctaOverlay: StardustCtaOverlay | undefined
    let blackHoleLens: BlackHoleLensPass | undefined
    let world: ReturnType<EarthWorkspaceResources['acquireWorld']> | undefined
    let notifiedReady = false
    let scoreReady = false
    let readyArmed = false
    let deviceGeneration = 0
    let fencedGeneration = -1
    let deviceController: AbortController | null = null

    const signalReady = () => {
      if (notifiedReady || signal.aborted) return
      notifiedReady = true
      onReadyRef.current()
    }

    const onPointerMove = (event: PointerEvent) => {
      titleOverlay?.setPointer(event.clientX, event.clientY)
      ctaOverlay?.setPointer(event.clientX, event.clientY)
      if (reducedMotion.matches) return
      const width = Math.max(1, window.innerWidth)
      const height = Math.max(1, window.innerHeight)
      pointerTarget.x = (event.clientX / width) * 2 - 1
      pointerTarget.y = 1 - (event.clientY / height) * 2
    }
    const onPointerLeave = () => {
      titleOverlay?.setPointer(null, null)
      ctaOverlay?.setPointer(null, null)
      pointerTarget.x = 0
      pointerTarget.y = 0
    }
    window.addEventListener('pointermove', onPointerMove, { passive: true, signal })
    window.addEventListener('pointerleave', onPointerLeave, { signal })

    const layoutOffsetInSticky = (el: HTMLElement | null, root: HTMLElement | null) => {
      if (!el || !root) return null
      let left = 0
      let top = 0
      let node: HTMLElement | null = el
      while (node && node !== root) {
        left += node.offsetLeft
        top += node.offsetTop
        node = node.offsetParent as HTMLElement | null
      }
      return { left, top, width: el.offsetWidth, height: el.offsetHeight }
    }

    let restTitleBox = layoutOffsetInSticky(titleStageRef.current, stickyRef.current)
    let restCtaBox = layoutOffsetInSticky(ctaStageRef.current, stickyRef.current)
    let cachedCanvasRect = canvas.getBoundingClientRect()
    const syncRestLayout = () => {
      restTitleBox = layoutOffsetInSticky(titleStageRef.current, stickyRef.current)
      restCtaBox = layoutOffsetInSticky(ctaStageRef.current, stickyRef.current)
      cachedCanvasRect = canvas.getBoundingClientRect()
    }
    const layoutObserver = new ResizeObserver(syncRestLayout)
    if (stickyRef.current) layoutObserver.observe(stickyRef.current)
    if (titleStageRef.current) layoutObserver.observe(titleStageRef.current)
    if (ctaStageRef.current) layoutObserver.observe(ctaStageRef.current)
    window.addEventListener('resize', syncRestLayout, { passive: true, signal })

    const onRenderStatus = () => {
      if (notifiedReady || !readyArmed || signal.aborted || !renderer) return
      if (fencedGeneration === deviceGeneration) return
      const solar = renderer.solarScene
      if (
        canvas.dataset.frameSource !== 'computed' ||
        canvas.dataset.terrainReady !== 'true' ||
        !scoreReady ||
        (solar && (!solar.skySettled || !solar.terrainSettled))
      ) {
        return
      }
      const device = renderer.computeDevice
      if (!device) return
      const gen = deviceGeneration
      fencedGeneration = gen
      void device.queue.onSubmittedWorkDone().then(
        () => {
          if (!signal.aborted && gen === deviceGeneration) signalReady()
        },
        () => {
          if (!signal.aborted && gen === deviceGeneration) signalReady()
        },
      )
    }
    canvas.addEventListener('renderstatus', onRenderStatus, { signal })

    const releaseHeroDeviceResources = () => {
      deviceController?.abort()
      deviceController = null
      ctaOverlayRef.current = null
      blackHoleLens?.destroy()
      blackHoleLens = undefined
      ctaOverlay?.destroy()
      ctaOverlay = undefined
      titleOverlay?.destroy()
      titleOverlay = undefined
      overlay?.destroy()
      overlay = undefined
      world?.release()
      world = undefined
      scoreReady = false
      readyArmed = false
    }

    const mountHeroDevice = async (instance: SurfaceRenderer, device: GPUDevice) => {
      releaseHeroDeviceResources()
      const gen = ++deviceGeneration
      const localController = new AbortController()
      deviceController = localController
      const devSignal = AbortSignal.any([signal, localController.signal])

      let bridge: BridgeOverlay | undefined
      let title: ConstellationTitleOverlay | undefined
      let cta: StardustCtaOverlay | undefined
      let lens: BlackHoleLensPass | undefined
      let localWorld: ReturnType<EarthWorkspaceResources['acquireWorld']> | undefined

      try {
        const results = await Promise.allSettled([
          BridgeOverlay.create(device),
          ConstellationTitleOverlay.create(device),
          StardustCtaOverlay.create(device),
          BlackHoleLensPass.create(device),
        ])
        if (results[0].status === 'fulfilled') bridge = results[0].value
        if (results[1].status === 'fulfilled') title = results[1].value
        if (results[2].status === 'fulfilled') cta = results[2].value
        if (results[3].status === 'fulfilled') lens = results[3].value
        const firstRejected = results.find(r => r.status === 'rejected')
        if (firstRejected && firstRejected.status === 'rejected') throw firstRejected.reason
        devSignal.throwIfAborted()

        overlay = bridge
        titleOverlay = title
        ctaOverlay = cta
        blackHoleLens = lens
        ctaOverlayRef.current = cta!
        bridge = title = cta = lens = undefined

        localWorld = resources.acquireWorld(device, {
          manifestUrl: scoreManifestUrl,
          readAsset: assets.readAsset,
        })
        world = localWorld
        try {
          const earth = await localWorld.loading
          devSignal.throwIfAborted()
          await localWorld.publish(async () => {
            devSignal.throwIfAborted()
            await earth.dispatchShader(HERO_BRIDGE_SCORE_SHADER, devSignal)
            devSignal.throwIfAborted()
            await instance.setComputedScores(
              earth,
              {
                name: 'Jezero Crater analogue',
                description: 'Arid basin floors enclosed by rugged regional relief',
              },
              devSignal,
            )
          })
          devSignal.throwIfAborted()
          // Scores are now baked into SurfaceRenderer's Native scores textures; wait for GPU submission
          // before releasing the ~530 MB EarthScore compute buffers while on the hero.
          await device.queue.onSubmittedWorkDone()
          devSignal.throwIfAborted()
        } finally {
          localWorld.release()
          if (world === localWorld) world = undefined
          localWorld = undefined
        }

        const best = instance.bestScoreLocation
        if (best) {
          earthTarget = { latitude: best.latitude, longitude: best.longitude }
          currentBodies = buildBridgeBodies(currentAspect, earthTarget)
          instance.setHeroBodies(currentBodies)
        } else {
          instance.invalidateHeroFrame()
        }
        if (gen === deviceGeneration && !devSignal.aborted) {
          scoreReady = true
          // Arm readiness for the next submitted frame that reflects final staging.
          readyArmed = true
        }
      } catch (cause) {
        bridge?.destroy()
        title?.destroy()
        cta?.destroy()
        lens?.destroy()
        if (!devSignal.aborted) {
          console.error('Could not initialize hero scene:', cause)
          setError(cause instanceof Error ? cause.message : String(cause))
          signalReady()
        }
      }
    }

    const onComputeDeviceChange = () => {
      if (signal.aborted || !renderer) return
      const nextDevice = renderer.computeDevice
      if (!nextDevice) {
        releaseHeroDeviceResources()
        return
      }
      void mountHeroDevice(renderer, nextDevice)
    }
    canvas.addEventListener('computedevicechange', onComputeDeviceChange, { signal })

    void (async () => {
      try {
        const instance = await SurfaceRenderer.create(
          canvas,
          () => undefined,
          err => {
            if (!signal.aborted) {
              setError(err.message)
              signalReady()
            }
          },
          () => undefined,
          {
            signal,
            acquireDevice: resources.acquireDevice,
            assets: assets.url,
            loadTerrain: resources.acquireTerrain,
            solar: {
              createScene: SolarScene.create,
              assetUrl: assets.url,
              loadTerrain: resources.acquireTerrain,
              onSourcePick: () => undefined,
            },
            hero: {
              bodies: currentBodies,
              animate: true,
              options: {
                hdr: true,
                exposure: 1.06,
                bloom: 0.42,
                emission: 1.85,
                opacity: 0.84,
                contextFloor: 0.08,
                contextOpacity: 0.24,
                contourOpacity: 0.16,
                relief: 22,
                imagery: false,
              },
              frame: (aspect, timeSeconds) => {
                if (Math.abs(aspect - currentAspect) > 0.004) {
                  currentAspect = aspect
                  currentBodies = buildBridgeBodies(aspect, earthTarget)
                  const solar = renderer?.solarScene
                  if (solar) solar.bodies = currentBodies
                }
                pointerSmooth.x += (pointerTarget.x - pointerSmooth.x) * 0.06
                pointerSmooth.y += (pointerTarget.y - pointerSmooth.y) * 0.06
                return buildBridgeFrame(
                  aspect,
                  timeSeconds,
                  pointerSmooth,
                  reducedMotion.matches,
                  scrollProgressRef.current,
                )
              },
              prepareOverlay: (encoder, frame, width, height, ratio, timeSeconds) => {
                const canvasRect = cachedCanvasRect
                if (canvasRect.width <= 1 || canvasRect.height <= 1) return
                const scaleX = width / canvasRect.width
                const scaleY = height / canvasRect.height
                const bhGeom = computeBlackHoleGeometry(scrollProgressRef.current, width / Math.max(1, height))
                const prepareSkyOverlay = (
                  target: ConstellationTitleOverlay | StardustCtaOverlay | undefined,
                  restBox: { left: number; top: number; width: number; height: number } | null,
                  domEl: HTMLElement | null,
                ) => {
                  if (!target || !restBox) return
                  const skyBox = projectSkyBox(restBox, canvasRect.width, canvasRect.height, frame)
                  if (!skyBox) return
                  if (domEl)
                    domEl.style.transform = `translate3d(${skyBox.dx.toFixed(2)}px, ${skyBox.dy.toFixed(2)}px, 0)`
                  target.prepare(
                    encoder,
                    {
                      leftPx: skyBox.left * scaleX,
                      topPx: skyBox.top * scaleY,
                      widthPx: skyBox.width * scaleX,
                      heightPx: skyBox.height * scaleY,
                    },
                    canvasRect,
                    width,
                    height,
                    ratio,
                    timeSeconds,
                    reducedMotion.matches,
                    1,
                    bhGeom,
                  )
                }
                prepareSkyOverlay(titleOverlay, restTitleBox, titleStageRef.current)
                prepareSkyOverlay(ctaOverlay, restCtaBox, ctaStageRef.current)
              },
              drawOverlay: (pass, frame, width, height, ratio, timeSeconds) => {
                if (overlay && scoreReady) {
                  const marsLocal = renderer?.solarScene?.sourcePosition('mars', HERO_MARS_SITE)
                  const endpoints = bridgeEndpoints(currentBodies, earthTarget, marsLocal)
                  overlay.draw(pass, frame, endpoints, width, height, ratio, timeSeconds, reducedMotion.matches, 1)
                }
                titleOverlay?.draw(pass)
                ctaOverlay?.draw(pass)
              },
              postScene: (encoder, sceneView, width, height, timeSeconds) => {
                if (!blackHoleLens) return null
                const geom = computeBlackHoleGeometry(scrollProgressRef.current, width / Math.max(1, height))
                return blackHoleLens.apply(encoder, sceneView, width, height, timeSeconds, geom)
              },
            },
          },
        )
        if (signal.aborted) {
          instance.destroy()
          return
        }
        renderer = instance
        rendererRef.current = instance
        instance.setPaused(leavingRef.current || scrollPausedRef.current)

        const device = instance.computeDevice
        if (!device) throw new Error('WebGPU is unavailable.')
        await mountHeroDevice(instance, device)
      } catch (cause) {
        if (!signal.aborted) {
          console.error('Could not initialize hero scene:', cause)
          setError(cause instanceof Error ? cause.message : String(cause))
          signalReady()
        }
      }
    })()

    return () => {
      controller.abort()
      layoutObserver.disconnect()
      releaseHeroDeviceResources()
      rendererRef.current = null
      renderer?.destroy()
    }
  }, [assets, resources])

  return (
    <section
      ref={scrollerRef}
      className="bridge-hero"
      aria-label="Earth Connect overview"
      data-leaving={leaving ? '' : undefined}
      inert={leaving}
    >
      <div ref={trackRef} className="bridge-hero-scroll-track">
        <div ref={stickyRef} className="bridge-hero-sticky">
          <canvas
            ref={canvasRef}
            className="bridge-hero-canvas"
            role="img"
            aria-label="Mars in the foreground at twilight linked by an illuminated arc across space to its glowing terrain analogue on Earth, above a gravitational lensing horizon."
          />
          <HeroTitle
            stageRef={titleStageRef}
            boundsRef={canvasRef}
            ctaRef={ctaStageRef}
            onCtaHoverChange={hovered => {
              ctaOverlayRef.current?.setHovered(hovered)
              rendererRef.current?.invalidateHeroFrame()
            }}
            onCtaFocusChange={focused => {
              ctaOverlayRef.current?.setFocused(focused)
              rendererRef.current?.invalidateHeroFrame()
            }}
            onCtaPressChange={pressed => {
              ctaOverlayRef.current?.setPressed(pressed)
              rendererRef.current?.invalidateHeroFrame()
            }}
            canActivateCta={(x, y) => ctaOverlayRef.current?.hitTestClient(x, y) ?? true}
            onOpenWorkspace={onOpenWorkspace}
          />
          <button
            ref={scrollCueRef}
            type="button"
            className="bridge-hero-scroll-cue"
            aria-label="Scroll down into Earth Connect"
            onClick={() => {
              const scroller = scrollerRef.current
              const track = trackRef.current
              const sticky = stickyRef.current
              if (!scroller || !track || !sticky) return
              const targetTop = Math.max(0, Math.round(track.offsetHeight - sticky.offsetHeight))
              const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
              if (reduced) {
                scroller.scrollTop = targetTop
                return
              }
              cancelScrollAnimRef.current?.()
              const startTop = scroller.scrollTop
              const delta = targetTop - startTop
              if (Math.abs(delta) < 1) return
              const startMs = performance.now()
              let rafId = 0
              const cancel = () => {
                if (rafId) {
                  cancelAnimationFrame(rafId)
                  rafId = 0
                }
                scroller.removeEventListener('wheel', cancel)
                scroller.removeEventListener('touchstart', cancel)
                window.removeEventListener('keydown', cancel)
                if (cancelScrollAnimRef.current === cancel) cancelScrollAnimRef.current = null
              }
              cancelScrollAnimRef.current = cancel
              scroller.addEventListener('wheel', cancel, { passive: true, once: true })
              scroller.addEventListener('touchstart', cancel, { passive: true, once: true })
              window.addEventListener('keydown', cancel, { once: true })
              const step = (now: number) => {
                const t = Math.min(1, (now - startMs) / HERO_SCROLL_CUE_MS)
                // Smooth cubic ease-in-out so the black hole swells gently, sweeps cleanly across the planets, and settles softly.
                const eased = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2
                scroller.scrollTop = startTop + delta * eased
                if (t < 1) {
                  rafId = requestAnimationFrame(step)
                } else {
                  cancel()
                }
              }
              rafId = requestAnimationFrame(step)
            }}
          >
            <svg viewBox="0 0 28 14" width="28" height="14" aria-hidden="true">
              <path
                d="M4.5 3.5L14 11L23.5 3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.35"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          {error && (
            <p className="bridge-hero-error" role="alert">
              {error}
            </p>
          )}
        </div>
      </div>

      {/* Site content emerging inside the pitch-black event horizon as the user scrolls down */}
      <main ref={siteShellRef} className="bridge-site-shell">
        <LandingEditorial
          scrollerRef={scrollerRef}
          assets={assets}
          resources={resources}
          onOpenWorkspace={onOpenWorkspace}
        />
      </main>
    </section>
  )
}
