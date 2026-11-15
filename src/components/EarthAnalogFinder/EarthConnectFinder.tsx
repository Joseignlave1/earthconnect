import './EarthConnectFinder.css'

const starPositions = Array.from({ length: 60 }, (_, i) => ({
  left: `${(i * 37 + 7) % 100}%`,
  top: `${(i * 53 + 11) % 98}%`,
}))

function Arrow() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 12h13M14 7l5 5-5 5" />
    </svg>
  )
}

type EarthConnectFinderProps = {
  onFind: () => void
}

export default function EarthConnectFinder({ onFind }: EarthConnectFinderProps) {
  return (
    <main className="eaf">
      <div className="stars" aria-hidden="true">
        {starPositions.map((p, i) => (
          <i key={i} style={p} />
        ))}
      </div>

      <div className="mars" aria-hidden="true" />
      <div className="earth-wrap" aria-hidden="true">
        <div className="earth" />
        <div className="moon-orbit">
          <div className="moon-path">
            <div className="moon" />
          </div>
        </div>
      </div>

      <div className="scrim" aria-hidden="true" />

      <section className="hero">
        <h1>
          Earth Connect
          <br />
          <em>Finder</em>
        </h1>
        <p>
          Compare worlds. Understand environments. Find the planets most like
          our own.
        </p>
        <a
          className="enter glass"
          href="/find"
          onClick={(event) => {
            if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
            event.preventDefault()
            onFind()
          }}
        >
          <span>Find</span>
          <Arrow />
        </a>
      </section>

      <footer className="nasa-signature">
        <span>Created for</span>
        <img src="/nasa-logo.svg" alt="NASA" />
      </footer>
    </main>
  )
}