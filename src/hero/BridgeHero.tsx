import { useEffect, useRef, useState } from 'react'
import { SurfaceRenderer } from '../shader-test/renderer.ts'
import { createBootGraph } from '../workspace/bootGraph.ts'
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
    }

    type HeroOverlayBundle = {
      bridge: BridgeOverlay
      title: ConstellationTitleOverlay
      cta: StardustCtaOverlay
      lens: BlackHoleLensPass
    }

    const createOverlays = async (device: GPUDevice, devSignal: AbortSignal): Promise<HeroOverlayBundle> => {
      const results = await Promise.allSettled([
        BridgeOverlay.create(device),
        ConstellationTitleOverlay.create(device),
        StardustCtaOverlay.create(device),
        BlackHoleLensPass.create(device),
      ])
      const bridge = results[0].status === 'fulfilled' ? results[0].value : undefined
      const title = results[1].status === 'fulfilled' ? results[1].value : undefined
      const cta = results[2].status === 'fulfilled' ? results[2].value : undefined
      const lens = results[3].status === 'fulfilled' ? results[3].value : undefined
      const firstRejected = results.find(r => r.status === 'rejected')
      if (firstRejected && firstRejected.status === 'rejected') {
        bridge?.destroy()
        title?.destroy()
        cta?.destroy()
        lens?.destroy()
        throw firstRejected.reason
      }
      if (devSignal.aborted) {
        bridge?.destroy()
        title?.destroy()
        cta?.destroy()
        lens?.destroy()
        devSignal.throwIfAborted()
      }
      return { bridge: bridge!, title: title!, cta: cta!, lens: lens! }
    }

    const createHeroRenderer = (devSignal: AbortSignal) =>
      SurfaceRenderer.create(
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
          signal: devSignal,
          acquireDevice: resources.acquireDevice,
          assets: assets.url,
          loadTerrain: resources.acquireTerrain,
          initialPhase: 'prepare',
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

    /**
     * Declarative hero boot DAG (`prepare` -> `settle` -> `live`):
     * - Runs renderer creation, hero overlays, world loading, score dispatch,
     *   sky map, and planetary terrain maximally in parallel while `SurfaceRenderer`
     *   stays in `prepare` (zero hidden swapchain draws).
     * - Once all critical tasks finish, renders one complete GPU-fenced frame,
     *   releases the ~530 MB compute buffers, switches to `live`, and signals ready.
     */
    const bootHeroScene = async (existing?: { instance: SurfaceRenderer; device: GPUDevice }) => {
      releaseHeroDeviceResources()
      const localController = new AbortController()
      deviceController = localController
      const devSignal = AbortSignal.any([signal, localController.signal])
      const boot = createBootGraph(devSignal, 'prepare')
      const isInitial = !existing
      let bootDeviceLease: ReturnType<EarthWorkspaceResources['acquireDevice']> | undefined
      let localWorld: ReturnType<EarthWorkspaceResources['acquireWorld']> | undefined
      let createdOverlays: HeroOverlayBundle | undefined

      try {
        existing?.instance.setPhase('prepare')

        boot.task('hero-device', {
          recordTiming: false,
          run: async () => {
            if (existing) return existing.device
            bootDeviceLease = resources.acquireDevice()
            const { device } = await bootDeviceLease.ready
            return device
          },
        })

        boot.task('hero-renderer', {
          recordTiming: isInitial,
          run: async ({ signal: s }) => {
            if (existing) return existing.instance
            const instance = await createHeroRenderer(s)
            if (s.aborted) {
              instance.destroy()
              s.throwIfAborted()
            }
            renderer = instance
            rendererRef.current = instance
            instance.setPaused(leavingRef.current || scrollPausedRef.current)
            return instance
          },
        })

        boot.task('hero-overlays', {
          after: ['hero-device'],
          recordTiming: isInitial,
          run: async ({ get, signal: s }) => {
            const bundle = await createOverlays(get<GPUDevice>('hero-device'), s)
            createdOverlays = bundle
            return bundle
          },
        })

        boot.task('hero-world', {
          after: ['hero-device'],
          recordTiming: false,
          run: async ({ get }) => {
            const dev = get<GPUDevice>('hero-device')
            localWorld = resources.acquireWorld(dev, {
              manifestUrl: scoreManifestUrl,
              readAsset: assets.readAsset,
            })
            world = localWorld
            const earth = await localWorld.loading
            return { lease: localWorld, earth }
          },
        })

        boot.task('hero-score', {
          after: ['hero-renderer', 'hero-world'],
          recordTiming: isInitial,
          run: async ({ get, signal: s }) => {
            const instance = get<SurfaceRenderer>('hero-renderer')
            const { lease, earth } = get<{
              lease: ReturnType<EarthWorkspaceResources['acquireWorld']>
              earth: Awaited<ReturnType<EarthWorkspaceResources['acquireWorld']>['loading']>
            }>('hero-world')
            await lease.publish(async () => {
              s.throwIfAborted()
              await earth.dispatchShader(HERO_BRIDGE_SCORE_SHADER, s)
              s.throwIfAborted()
              await instance.setComputedScores(
                earth,
                {
                  name: 'Jezero Crater analogue',
                  description: 'Arid basin floors enclosed by rugged regional relief',
                },
                s,
              )
            })
          },
        })

        boot.task('hero-terrain', {
          after: ['hero-renderer'],
          recordTiming: false,
          run: ({ get, signal: s }) => get<SurfaceRenderer>('hero-renderer').waitForTerrainReady(s),
        })

        boot.task('hero-sky', {
          after: ['hero-renderer'],
          recordTiming: isInitial,
          run: ({ get, signal: s }) => get<SurfaceRenderer>('hero-renderer').solarScene?.whenSkySettled(s),
        })

        boot.task('hero-planets', {
          after: ['hero-renderer'],
          recordTiming: isInitial,
          run: ({ get, signal: s }) => get<SurfaceRenderer>('hero-renderer').solarScene?.whenTerrainSettled(s),
        })

        await boot.ready('critical')
        devSignal.throwIfAborted()

        const instance = boot.get<SurfaceRenderer>('hero-renderer')
        const bundle = boot.get<HeroOverlayBundle>('hero-overlays')
        overlay = bundle.bridge
        titleOverlay = bundle.title
        ctaOverlay = bundle.cta
        blackHoleLens = bundle.lens
        ctaOverlayRef.current = bundle.cta
        createdOverlays = undefined

        const best = instance.bestScoreLocation
        if (best) {
          earthTarget = { latitude: best.latitude, longitude: best.longitude }
          currentBodies = buildBridgeBodies(currentAspect, earthTarget)
          instance.setHeroBodies(currentBodies)
        }
        scoreReady = true

        // Render exactly one settled frame and wait for its GPU fence before releasing compute buffers and revealing.
        await boot.task('hero-settle', {
          phase: 'settle',
          recordTiming: isInitial,
          run: ({ signal: s }) => instance.settleFrame(s),
        })
        devSignal.throwIfAborted()

        if (localWorld) {
          localWorld.release()
          if (world === localWorld) world = undefined
          localWorld = undefined
        }

        instance.setPhase('live')
        signalReady()
        boot.startBackground()
      } catch (cause) {
        if (createdOverlays) {
          createdOverlays.bridge.destroy()
          createdOverlays.title.destroy()
          createdOverlays.cta.destroy()
          createdOverlays.lens.destroy()
        }
        if (localWorld) {
          localWorld.release()
          if (world === localWorld) world = undefined
          localWorld = undefined
        }
        if (!devSignal.aborted) {
          console.error('Could not initialize hero scene:', cause)
          setError(cause instanceof Error ? cause.message : String(cause))
          renderer?.setPhase('live')
          signalReady()
        }
      } finally {
        bootDeviceLease?.release()
      }
    }

    const onComputeDeviceChange = () => {
      if (signal.aborted || !renderer) return
      const nextDevice = renderer.computeDevice
      if (!nextDevice) {
        releaseHeroDeviceResources()
        return
      }
      void bootHeroScene({ instance: renderer, device: nextDevice })
    }
    canvas.addEventListener('computedevicechange', onComputeDeviceChange, { signal })

    void bootHeroScene()

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
