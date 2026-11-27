import { useEffect, useRef, useState } from 'react'
import type { EarthWorkspaceResources } from '../workspace/earthWorkspaceResources.ts'
import { CrewAstronautStage } from './crewAstronaut.ts'
import { formatCoord, TEAM_MEMBERS } from './editorialData.ts'

export function CrewSection({
  scrollerRef,
  resources,
}: {
  scrollerRef: React.RefObject<HTMLElement | null>
  resources: EarthWorkspaceResources
}) {
  const [crewIndex, setCrewIndex] = useState<number>(() => Math.floor(Math.random() * TEAM_MEMBERS.length))
  const [activated, setActivated] = useState(() => typeof IntersectionObserver === 'undefined')

  const sectionRef = useRef<HTMLElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<CrewAstronautStage | null>(null)
  const visibleRef = useRef(typeof IntersectionObserver === 'undefined')
  const swipeStartRef = useRef<number | null>(null)

  const activeMember = TEAM_MEMBERS[crewIndex] ?? TEAM_MEMBERS[0]
  const activeMemberIdRef = useRef(activeMember.id)
  useEffect(() => {
    activeMemberIdRef.current = activeMember.id
  }, [activeMember.id])

  useEffect(() => {
    const el = sectionRef.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.target === el) {
            visibleRef.current = entry.isIntersecting
            if (entry.isIntersecting) setActivated(true)
            stageRef.current?.setActive(entry.isIntersecting)
          }
        }
      },
      { root: scrollerRef.current, rootMargin: '20px 0px', threshold: 0.01 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [scrollerRef])

  useEffect(() => {
    if (!activated) return
    const canvas = canvasRef.current
    if (!canvas) return
    const controller = new AbortController()
    let stage: CrewAstronautStage | null = null

    void (async () => {
      try {
        const created = await CrewAstronautStage.create(
          canvas,
          resources.acquireDevice,
          TEAM_MEMBERS,
          activeMemberIdRef.current,
          controller.signal,
        )
        if (controller.signal.aborted) {
          created.destroy()
          return
        }
        stage = created
        stageRef.current = created
        created.setMember(activeMemberIdRef.current)
        created.setActive(visibleRef.current)
      } catch (err) {
        if (!controller.signal.aborted) console.error('Could not initialize crew astronaut stage:', err)
      }
    })()

    return () => {
      controller.abort()
      stageRef.current = null
      stage?.destroy()
    }
  }, [activated, resources])

  useEffect(() => {
    stageRef.current?.setMember(activeMember.id)
  }, [activeMember.id])

  const stepCrew = (delta: number) => {
    setCrewIndex(prev => (prev + delta + TEAM_MEMBERS.length) % TEAM_MEMBERS.length)
  }

  return (
    <footer
      ref={sectionRef}
      className="credits-section"
      aria-label="Team and data credits"
      onKeyDown={event => {
        if (event.key === 'ArrowLeft') stepCrew(-1)
        else if (event.key === 'ArrowRight') stepCrew(1)
      }}
    >
      <div className="crew-top-bar">
        <h2 className="credits-heading">Made by</h2>

        <div className="crew-name-strip" role="tablist" aria-label="Team members">
          {TEAM_MEMBERS.map((member, idx) => {
            const selected = idx === crewIndex
            return (
              <button
                key={member.id}
                type="button"
                role="tab"
                aria-selected={selected}
                className="crew-name-btn"
                onClick={() => setCrewIndex(idx)}
              >
                {member.name}
              </button>
            )
          })}
        </div>

        <div className="crew-arrows" aria-label="Carousel navigation">
          <button
            type="button"
            className="crew-arrow-btn"
            aria-label="Previous team member"
            onClick={() => stepCrew(-1)}
          >
            ←
          </button>
          <button type="button" className="crew-arrow-btn" aria-label="Next team member" onClick={() => stepCrew(1)}>
            →
          </button>
        </div>
      </div>

      <div
        className="crew-carousel-stage"
        role="tabpanel"
        aria-label={activeMember.name}
        onTouchStart={event => {
          swipeStartRef.current = event.touches[0]?.clientX ?? null
        }}
        onTouchEnd={event => {
          const startX = swipeStartRef.current
          swipeStartRef.current = null
          const endX = event.changedTouches[0]?.clientX
          if (startX === null || endX === undefined) return
          const dx = endX - startX
          if (Math.abs(dx) > 42) stepCrew(dx < 0 ? 1 : -1)
        }}
      >
        <div className="crew-visual-col">
          <canvas
            ref={canvasRef}
            className="crew-astronaut-canvas"
            role="img"
            aria-label={`3D stylized astronaut representing ${activeMember.name} being stretched toward the black hole`}
            onPointerMove={event => {
              const rect = event.currentTarget.getBoundingClientRect()
              const nx = ((event.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1
              const ny = ((event.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1
              stageRef.current?.setPointer(nx, ny, true)
            }}
            onPointerLeave={() => stageRef.current?.setPointer(0, 0, false)}
          />
        </div>

        <div className="crew-bio-col">
          <span className="crew-counter">
            {String(crewIndex + 1).padStart(2, '0')} / {String(TEAM_MEMBERS.length).padStart(2, '0')}
          </span>
          <h3 className="crew-person-name">{activeMember.name}</h3>
          <p className="crew-person-role">{activeMember.role}</p>
          <p className="crew-person-bio">{activeMember.bio}</p>
          <p className="crew-person-coord">Born at: {formatCoord(activeMember.lat, activeMember.lon, 4)}</p>

          <nav className="crew-socials" aria-label={`${activeMember.name} links`}>
            {activeMember.socials.map(social => (
              <a
                key={social.label}
                className="crew-social-link"
                href={social.href}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span>{social.label}</span>
                <span className="crew-social-arrow" aria-hidden="true">
                  ↗
                </span>
              </a>
            ))}
          </nav>
        </div>
      </div>

      <div className="credits-colophon">
        <p className="credits-origin">
          Created for NASA Space Apps Challenge 2026 · Special Mention · Open-source Apache-2.0
        </p>
        <p className="credits-sources">
          Terrain &amp; sky data from NASA MOLA / CTX, Lunar Reconnaissance Orbiter, GEBCO 2026, MODIS 4 km grid, NASA
          SVS Deep Star Map, and Earth Index.
        </p>
      </div>
    </footer>
  )
}
