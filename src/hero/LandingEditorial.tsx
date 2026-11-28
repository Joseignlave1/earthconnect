import type { EarthWorkspaceResources } from '../workspace/earthWorkspaceResources.ts'
import type { WorkspaceAssets } from '../workspace/workspaceAssets.ts'
import { AtlasSection } from './AtlasSection.tsx'
import { CopilotSection } from './CopilotSection.tsx'
import { CrewSection } from './CrewSection.tsx'

export function LandingEditorial({
  scrollerRef,
  assets,
  resources,
  onOpenWorkspace,
}: {
  scrollerRef: React.RefObject<HTMLElement | null>
  assets: WorkspaceAssets
  resources: EarthWorkspaceResources
  onOpenWorkspace: () => void
}) {
  return (
    <div className="editorial-root">
      {/* Opening · What Earth Connect is & why planetary-scale search matters */}
      <section className="intro-section" aria-label="About Earth Connect">
        <h2 className="intro-headline">Find where Earth matches the Moon and Mars.</h2>
        <p className="intro-body">
          Training crews, testing rovers, and reading orbital geology all rely on terrestrial analogues—places on Earth
          shaped by comparable slopes, basins, and aridity. Searching the whole planet at 4 km resolution means
          evaluating <span className="intro-emphasis">31.8 million cells</span> across elevation, regional relief,
          rainfall, and land cover. Earth Connect runs those comparisons live on your GPU, turning a global GIS search
          into an interactive atlas in the browser.
        </p>
      </section>

      <AtlasSection scrollerRef={scrollerRef} assets={assets} resources={resources} onOpenWorkspace={onOpenWorkspace} />
      <CopilotSection />
      <CrewSection scrollerRef={scrollerRef} resources={resources} />
    </div>
  )
}
