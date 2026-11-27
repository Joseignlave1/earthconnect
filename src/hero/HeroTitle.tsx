import type { RefObject } from 'react'
import { BayerAnnotations } from './BayerAnnotations.tsx'

export function HeroTitle({
  stageRef,
  boundsRef,
  ctaRef,
  onCtaHoverChange,
  onCtaFocusChange,
  onCtaPressChange,
  canActivateCta,
  ctaBusy = false,
  onOpenWorkspace,
}: {
  stageRef: RefObject<HTMLDivElement | null>
  boundsRef?: RefObject<HTMLElement | null>
  ctaRef: RefObject<HTMLAnchorElement | null>
  onCtaHoverChange: (hovered: boolean) => void
  onCtaFocusChange: (focused: boolean) => void
  onCtaPressChange: (pressed: boolean) => void
  canActivateCta?: (clientX: number, clientY: number) => boolean
  ctaBusy?: boolean
  onOpenWorkspace: () => void
}) {
  return (
    <>
      <div className="bridge-hero-copy">
        <div ref={stageRef} className="constellation-stage">
          {/* Invisible semantic heading overlaid on the WebGPU shader title for copy/paste & accessibility */}
          <h1 className="constellation-a11y-heading">
            <span className="constellation-a11y-row">Earth</span>{' '}
            <span className="constellation-a11y-row constellation-a11y-row-2">Connect</span>
          </h1>
          <BayerAnnotations
            stageRef={stageRef}
            boundsRef={boundsRef}
            svgClassName="constellation-bayer-svg"
            labelClassName="hero-bayer-label"
          />
        </div>
      </div>

      {/* Accessible hit target overlaid on the bottom-right WebGPU Stardust "Explore" constellation CTA */}
      <a
        ref={ctaRef}
        className="bridge-hero-cta"
        href="/find"
        aria-label="Explore workspace"
        aria-busy={ctaBusy || undefined}
        aria-disabled={ctaBusy || undefined}
        onPointerEnter={() => onCtaHoverChange(true)}
        onPointerLeave={() => {
          onCtaHoverChange(false)
          onCtaPressChange(false)
        }}
        onPointerDown={() => {
          if (!ctaBusy) onCtaPressChange(true)
        }}
        onPointerUp={() => onCtaPressChange(false)}
        onFocus={() => onCtaFocusChange(true)}
        onBlur={() => {
          onCtaFocusChange(false)
          onCtaPressChange(false)
        }}
        onClick={event => {
          if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
          event.preventDefault()
          if (ctaBusy) return
          // For pointer clicks (detail > 0), honor the exact lensed GPU hit test so clicks
          // inside the swallowed black-hole void do not activate the hidden CTA behind it.
          if (event.detail > 0 && canActivateCta && !canActivateCta(event.clientX, event.clientY)) return
          event.currentTarget.blur()
          onOpenWorkspace()
        }}
      >
        <span className="bridge-hero-cta-a11y">Explore</span>
      </a>
    </>
  )
}
