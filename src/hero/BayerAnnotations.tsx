import { type RefObject, useEffect, useRef } from 'react'
import { BAYER_ANCHORS, projectStar, smoothstep, VIEW_H, VIEW_W } from './constellationData.ts'
import { WAKE_RADIUS } from './constellationTitle.ts'

/**
 * Subtle italic Greek Bayer star annotations (α β γ δ ε) synced to pointer
 * parallax and wake focus over the constellation wordmark.
 */
export function BayerAnnotations({
  stageRef,
  boundsRef,
  svgClassName,
  labelClassName,
  active = true,
  live,
}: {
  stageRef: RefObject<HTMLDivElement | null>
  boundsRef?: RefObject<HTMLElement | null>
  svgClassName: string
  labelClassName: string
  active?: boolean
  live?: boolean
}) {
  const bayerRefs = useRef<(SVGTextElement | null)[]>([])

  useEffect(() => {
    if (!active) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
    let raf = 0
    let lastMs = -1
    const target = { x: 0, y: 0 }
    const smooth = { x: 0, y: 0 }
    const targetFocus = BAYER_ANCHORS.map(() => 0)
    const smoothFocus = BAYER_ANCHORS.map(() => 0)
    let dirty = false

    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(step)
    }

    const step = (nowMs: number) => {
      raf = 0
      const dt = lastMs < 0 ? 1 / 60 : Math.min(0.05, Math.max(0.001, (nowMs - lastMs) / 1000))
      lastMs = nowMs
      // Match ConstellationTitleOverlay's exact time-based exponential parallax smoothing (rate 3.8/s).
      const parLerp = reduced.matches ? 1 : 1 - Math.exp(-dt * 3.8)
      smooth.x += ((reduced.matches ? 0 : target.x) - smooth.x) * parLerp
      smooth.y += ((reduced.matches ? 0 : target.y) - smooth.y) * parLerp

      let moving =
        !reduced.matches && (Math.abs(target.x - smooth.x) > 0.0015 || Math.abs(target.y - smooth.y) > 0.0015)

      for (let i = 0; i < BAYER_ANCHORS.length; i++) {
        const diff = targetFocus[i] - smoothFocus[i]
        const focusLerp = reduced.matches ? 1 : 1 - Math.exp(-dt * (diff > 0 ? 17 : 2.5))
        if (Math.abs(diff) > 0.003) {
          smoothFocus[i] += diff * focusLerp
          moving = true
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

      if (moving || dirty) {
        dirty = false
        schedule()
      } else {
        lastMs = -1
      }
    }

    const onMove = (event: PointerEvent) => {
      const boundsEl = boundsRef?.current ?? stageRef.current
      if (boundsEl) {
        const bRect = boundsEl.getBoundingClientRect()
        if (bRect.width > 0 && bRect.height > 0) {
          target.x = ((event.clientX - bRect.left) / bRect.width) * 2 - 1
          target.y = ((event.clientY - bRect.top) / bRect.height) * 2 - 1
        }
      } else {
        target.x = (event.clientX / Math.max(1, window.innerWidth)) * 2 - 1
        target.y = (event.clientY / Math.max(1, window.innerHeight)) * 2 - 1
      }
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
  }, [stageRef, boundsRef, active])

  return (
    <svg
      className={svgClassName}
      data-live={live ? 'true' : undefined}
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
          className={labelClassName}
        >
          {anchor.label}
        </text>
      ))}
    </svg>
  )
}
