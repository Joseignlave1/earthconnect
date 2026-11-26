import { type RefObject, useEffect, useRef } from 'react'
import { BAYER_ANCHORS, projectStar, smoothstep, VIEW_H, VIEW_W } from './constellationData.ts'
import { WAKE_RADIUS } from './constellationTitle.ts'

export function HeroTitle({
  stageRef,
  ctaRef,
  onCtaHoverChange,
  onCtaFocusChange,
  onCtaPressChange,
  onOpenWorkspace,
}: {
  stageRef: RefObject<HTMLDivElement | null>
  ctaRef: RefObject<HTMLAnchorElement | null>
  onCtaHoverChange: (hovered: boolean) => void
  onCtaFocusChange: (focused: boolean) => void
  onCtaPressChange: (pressed: boolean) => void
  onOpenWorkspace: () => void
}) {
  const bayerRefs = useRef<(SVGTextElement | null)[]>([])

  // Keep the italic Greek Bayer annotations subtly synced with the GPU star parallax & wake.
  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
    let raf = 0
    const target = { x: 0, y: 0 }
    const smooth = { x: 0, y: 0 }
    const targetFocus = BAYER_ANCHORS.map(() => 0)
    const smoothFocus = BAYER_ANCHORS.map(() => 0)
    let dirty = false

    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(step)
    }

    const step = () => {
      raf = 0
      const parLerp = reduced.matches ? 0 : 0.06
      smooth.x += (target.x - smooth.x) * parLerp
      smooth.y += (target.y - smooth.y) * parLerp

      let active =
        !reduced.matches && (Math.abs(target.x - smooth.x) > 0.0015 || Math.abs(target.y - smooth.y) > 0.0015)

      for (let i = 0; i < BAYER_ANCHORS.length; i++) {
        const diff = targetFocus[i] - smoothFocus[i]
        const focusLerp = reduced.matches ? 1 : diff > 0 ? 0.25 : 0.04
        if (Math.abs(diff) > 0.003) {
          smoothFocus[i] += diff * focusLerp
          active = true
        } else {
          smoothFocus[i] = targetFocus[i]
        }
        const el = bayerRefs.current[i]
        if (!el) continue
        const anchor = BAYER_ANCHORS[i]
        const pos = projectStar(anchor, reduced.matches ? { x: 0, y: 0 } : smooth)
        el.setAttribute('x', (pos.x + 8).toFixed(2))
        el.setAttribute('y', (pos.y - 7).toFixed(2))
        el.style.opacity = (0.56 + smoothFocus[i] * 0.44).toFixed(3)
      }

      if (active || dirty) {
        dirty = false
        schedule()
      }
    }

    const onMove = (event: PointerEvent) => {
      target.x = (event.clientX / Math.max(1, window.innerWidth)) * 2 - 1
      target.y = (event.clientY / Math.max(1, window.innerHeight)) * 2 - 1
      const stage = stageRef.current
      if (stage) {
        const rect = stage.getBoundingClientRect()
        if (rect.width > 0 && rect.height > 0) {
          const viewX = ((event.clientX - rect.left) / rect.width) * VIEW_W
          const viewY = ((event.clientY - rect.top) / rect.height) * VIEW_H
          for (let i = 0; i < BAYER_ANCHORS.length; i++) {
            const anchor = BAYER_ANCHORS[i]
            const dist = Math.hypot(viewX - anchor.x, viewY - anchor.y)
            targetFocus[i] = smoothstep(WAKE_RADIUS * 1.35, WAKE_RADIUS * 0.3, dist)
          }
        }
      }
      dirty = true
      schedule()
    }

    const onLeave = () => {
      target.x = 0
      target.y = 0
      targetFocus.fill(0)
      dirty = true
      schedule()
    }

    window.addEventListener('pointermove', onMove, { passive: true })
    window.addEventListener('pointerleave', onLeave)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerleave', onLeave)
      cancelAnimationFrame(raf)
    }
  }, [stageRef])

  return (
    <>
      <div className="bridge-hero-copy">
        <div ref={stageRef} className="constellation-stage">
          {/* Invisible semantic heading overlaid on the WebGPU shader title for copy/paste & accessibility */}
          <h1 className="constellation-a11y-heading">
            <span className="constellation-a11y-row constellation-a11y-row-1">Earth</span>{' '}
            <span className="constellation-a11y-row constellation-a11y-row-2">Connect</span>
          </h1>
          <svg
            className="constellation-bayer-svg"
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
                className="hero-bayer-label"
              >
                {anchor.label}
              </text>
            ))}
          </svg>
        </div>
      </div>

      {/* Accessible hit target overlaid on the bottom-right WebGPU Stardust "Explore" constellation CTA */}
      <a
        ref={ctaRef}
        className="bridge-hero-cta"
        href="/find"
        aria-label="Explore workspace"
        onPointerEnter={() => onCtaHoverChange(true)}
        onPointerLeave={() => {
          onCtaHoverChange(false)
          onCtaPressChange(false)
        }}
        onPointerDown={() => onCtaPressChange(true)}
        onPointerUp={() => onCtaPressChange(false)}
        onFocus={() => onCtaFocusChange(true)}
        onBlur={() => {
          onCtaFocusChange(false)
          onCtaPressChange(false)
        }}
        onClick={event => {
          if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
          event.preventDefault()
          onOpenWorkspace()
        }}
      >
        <span className="bridge-hero-cta-a11y">Explore</span>
      </a>
    </>
  )
}
