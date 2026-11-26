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

const HERO_RUNWAY_VH = 2.2
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
  const stickyRef = useRef<HTMLDivElement>(null)
  const siteShellRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const titleStageRef = useRef<HTMLDivElement>(null)
  const ctaStageRef = useRef<HTMLAnchorElement>(null)
  const scrollCueRef = useRef<HTMLButtonElement>(null)
  const scrollAnimRef = useRef<number | null>(null)
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
    const sticky = stickyRef.current
    const siteShell = siteShellRef.current
    if (!scroller || !sticky || !siteShell) return

    const syncScroll = () => {
      const vh = Math.max(1, scroller.clientHeight)
      const vw = Math.max(1, scroller.clientWidth)
      // Scroll runway: 2.2x viewport height drives the black hole from peeking crest to full pitch black.
      const runway = Math.max(1, vh * HERO_RUNWAY_VH)
      const progress = Math.max(0, Math.min(1, scroller.scrollTop / runway))
      scrollProgressRef.current = progress

      const geom = computeBlackHoleGeometry(progress, vw / vh)
      const cxPx = geom.cx * vw
      const cyPx = geom.cy * vh
      const rhPx = geom.rh * vh

      // Clip the dark site interior to the exact WebGPU event horizon circle until it swallows the screen.
      if (progress >= 0.96) {
        siteShell.style.clipPath = 'none'
        siteShell.style.opacity = '1'
        siteShell.style.pointerEvents = 'auto'
      } else if (progress <= 0.08) {
        siteShell.style.clipPath = `circle(0px at ${cxPx.toFixed(1)}px ${cyPx.toFixed(1)}px)`
        siteShell.style.opacity = '0'
        siteShell.style.pointerEvents = 'none'
      } else {
        // Slightly inset from the photon ring so DOM content stays cleanly inside the pitch-black shadow.
        const innerR = Math.max(0, rhPx - 6)
        const siteTopInViewport = siteShell.getBoundingClientRect().top
        const localCyPx = cyPx - siteTopInViewport
        siteShell.style.clipPath = `circle(${innerR.toFixed(1)}px at ${cxPx.toFixed(1)}px ${localCyPx.toFixed(1)}px)`
        siteShell.style.opacity = String(Math.min(1, (progress - 0.14) / 0.35))
        siteShell.style.pointerEvents = progress > 0.55 ? 'auto' : 'none'
      }

      // Keep WebGPU title & CTA at full luminosity so the black hole event horizon physically
      // lenses and swallows them (just like Earth & Mars). Fade only non-lensed DOM micro-labels
      // (Bayer Greek letters) early so they don't float unwarped over the rising photon ring.
      const domLabelOpacity = Math.max(0, Math.min(1, 1 - (progress - 0.06) / 0.18))
      sticky.style.setProperty('--hero-bayer-opacity', domLabelOpacity.toFixed(3))
      const scrollCueOpacity = Math.max(0, Math.min(1, 1 - progress / 0.1))
      sticky.style.setProperty('--hero-scroll-cue-opacity', scrollCueOpacity.toFixed(3))
      if (scrollCueRef.current) {
        scrollCueRef.current.style.pointerEvents = progress > 0.06 ? 'none' : 'auto'
      }
      if (ctaStageRef.current) {
        ctaStageRef.current.style.pointerEvents = progress > 0.36 ? 'none' : 'auto'
      }

      // Pause the hero WebGPU renderer once completely inside the black hole void (0% GPU while reading the site).
      const shouldPause = progress >= 0.99
      if (scrollPausedRef.current !== shouldPause) {
        scrollPausedRef.current = shouldPause
        rendererRef.current?.setPaused(leavingRef.current || shouldPause)
      }
    }

    syncScroll()
    scroller.addEventListener('scroll', syncScroll, { passive: true })
    window.addEventListener('resize', syncScroll, { passive: true })
    return () => {
      if (scrollAnimRef.current !== null) {
        cancelAnimationFrame(scrollAnimRef.current)
        scrollAnimRef.current = null
      }
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
      if (notifiedReady || signal.aborted || !renderer) return
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
      void device.queue.onSubmittedWorkDone().then(
        () => {
          if (!signal.aborted) signalReady()
        },
        () => {
          if (!signal.aborted) signalReady()
        },
      )
    }
    canvas.addEventListener('renderstatus', onRenderStatus, { signal })

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
                if (titleOverlay && restTitleBox) {
                  const skyBox = projectSkyBox(restTitleBox, canvasRect.width, canvasRect.height, frame)
                  if (skyBox) {
                    if (titleStageRef.current) {
                      titleStageRef.current.style.transform = `translate3d(${skyBox.dx.toFixed(2)}px, ${skyBox.dy.toFixed(2)}px, 0)`
                    }
                    titleOverlay.prepare(
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
                }
                if (ctaOverlay && restCtaBox) {
                  const skyBox = projectSkyBox(restCtaBox, canvasRect.width, canvasRect.height, frame)
                  if (skyBox) {
                    if (ctaStageRef.current) {
                      ctaStageRef.current.style.transform = `translate3d(${skyBox.dx.toFixed(2)}px, ${skyBox.dy.toFixed(2)}px, 0)`
                    }
                    ctaOverlay.prepare(
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
                }
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
              postScene: (encoder, sceneTexture, sceneView, width, height, ratio, timeSeconds) => {
                if (!blackHoleLens) return
                const geom = computeBlackHoleGeometry(scrollProgressRef.current, width / Math.max(1, height))
                blackHoleLens.apply(encoder, sceneTexture, sceneView, width, height, ratio, timeSeconds, geom)
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

        const device = instance.computeDevice
        if (!device) throw new Error('WebGPU is unavailable.')
        const [bridge, title, cta, lens] = await Promise.all([
          BridgeOverlay.create(device),
          ConstellationTitleOverlay.create(device),
          StardustCtaOverlay.create(device),
          BlackHoleLensPass.create(device),
        ])
        if (signal.aborted) {
          bridge.destroy()
          title.destroy()
          cta.destroy()
          lens.destroy()
          return
        }
        overlay = bridge
        titleOverlay = title
        ctaOverlay = cta
        blackHoleLens = lens
        ctaOverlayRef.current = cta

        world = resources.acquireWorld(device, {
          manifestUrl: scoreManifestUrl,
          readAsset: assets.readAsset,
        })
        const earth = await world.loading
        signal.throwIfAborted()
        await world.publish(async () => {
          signal.throwIfAborted()
          await earth.dispatchShader(HERO_BRIDGE_SCORE_SHADER, signal)
          signal.throwIfAborted()
          await instance.setComputedScores(
            earth,
            {
              name: 'Jezero Crater analogue',
              description: 'Arid basin floors enclosed by rugged regional relief',
            },
            signal,
          )
        })
        signal.throwIfAborted()
        // Scores are now baked into SurfaceRenderer's Native scores textures; release the ~530 MB
        // EarthScore input/output compute buffers while on the hero (re-acquired on demand in /find).
        await device.queue.onSubmittedWorkDone()
        signal.throwIfAborted()
        world.release()
        world = undefined

        const best = instance.bestScoreLocation
        if (best) {
          earthTarget = { latitude: best.latitude, longitude: best.longitude }
          currentBodies = buildBridgeBodies(currentAspect, earthTarget)
          instance.setHeroBodies(currentBodies)
        }
        scoreReady = true
        onRenderStatus()
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
      rendererRef.current = null
      ctaOverlayRef.current = null
      blackHoleLens?.destroy()
      ctaOverlay?.destroy()
      titleOverlay?.destroy()
      overlay?.destroy()
      world?.release()
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
      <div className="bridge-hero-scroll-track">
        <div ref={stickyRef} className="bridge-hero-sticky">
          <canvas
            ref={canvasRef}
            className="bridge-hero-canvas"
            role="img"
            aria-label="Mars in the foreground at twilight linked by an illuminated arc across space to its glowing terrain analogue on Earth, above a gravitational lensing horizon."
          />
          <HeroTitle
            stageRef={titleStageRef}
            ctaRef={ctaStageRef}
            onCtaHoverChange={hovered => ctaOverlayRef.current?.setHovered(hovered)}
            onCtaFocusChange={focused => ctaOverlayRef.current?.setFocused(focused)}
            onCtaPressChange={pressed => ctaOverlayRef.current?.setPressed(pressed)}
            onOpenWorkspace={onOpenWorkspace}
          />
          <button
            ref={scrollCueRef}
            type="button"
            className="bridge-hero-scroll-cue"
            aria-label="Scroll down into Earth Connect"
            onClick={() => {
              const scroller = scrollerRef.current
              if (!scroller) return
              const targetTop = Math.round(scroller.clientHeight * HERO_RUNWAY_VH)
              const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
              if (reduced) {
                scroller.scrollTop = targetTop
                return
              }
              if (scrollAnimRef.current !== null) cancelAnimationFrame(scrollAnimRef.current)
              const startTop = scroller.scrollTop
              const delta = targetTop - startTop
              if (Math.abs(delta) < 1) return
              const startMs = performance.now()
              const cancel = () => {
                if (scrollAnimRef.current !== null) {
                  cancelAnimationFrame(scrollAnimRef.current)
                  scrollAnimRef.current = null
                }
                scroller.removeEventListener('wheel', cancel)
                scroller.removeEventListener('touchstart', cancel)
                window.removeEventListener('keydown', cancel)
              }
              scroller.addEventListener('wheel', cancel, { passive: true, once: true })
              scroller.addEventListener('touchstart', cancel, { passive: true, once: true })
              window.addEventListener('keydown', cancel, { once: true })
              const step = (now: number) => {
                const t = Math.min(1, (now - startMs) / HERO_SCROLL_CUE_MS)
                // Smooth cubic ease-in-out so the black hole swells gently, sweeps cleanly across the planets, and settles softly.
                const eased = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2
                scroller.scrollTop = startTop + delta * eased
                if (t < 1) {
                  scrollAnimRef.current = requestAnimationFrame(step)
                } else {
                  cancel()
                }
              }
              scrollAnimRef.current = requestAnimationFrame(step)
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
