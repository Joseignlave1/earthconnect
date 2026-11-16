import './App.css'
import PlanetViewer from './PlanetViewer'

const starPositions = Array.from({ length: 48 }, (_, index) => ({
  left: `${(index * 37 + 7) % 100}%`,
  top: `${(index * 53 + 11) % 96}%`,
}))

function NasaMark() {
  return (
    <a className="nasa-signature" href="https://www.spaceappschallenge.org/" rel="noreferrer" target="_blank">
      <span>Created for</span>
      <img src="https://assets.spaceappschallenge.org/media/images/Space_Apps_Logo_Color_and_White.width-440.jpegquality-60.png" alt="NASA International Space Apps Challenge" />
    </a>
  )
}

function App() {
  if (window.location.pathname.startsWith('/find')) {
    return <PlanetViewer />
  }

  return (
    <main className="page-shell">
      <div className="stars" aria-hidden="true">
        {starPositions.map((position, index) => <i key={index} style={position} />)}
      </div>

      <header className="topbar">
        <a className="brand" href="#top" aria-label="Earth Connect home">
          <span className="brand-symbol" aria-hidden="true"><i /></span>
          <span>EARTH</span> CONNECT
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-halo" aria-hidden="true"><i /><i /><i /></div>
        <p className="hero-eyebrow"><span />Moon and Mars site finder<span /></p>
        <h1>Find your best<br /><em>spot on Earth.</em></h1>
        <p>
          Pick a point on the Moon or Mars. We’ll rank the closest matches on
          Earth.
        </p>
        <a className="find-button" href="/find">
          <span>Find your spot</span>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 12h13M14 7l5 5-5 5" />
          </svg>
        </a>
        <div className="hero-data" aria-label="Earth Connect capabilities">
          <span><strong>Moon + Mars</strong><small>Reference worlds</small></span>
          <i aria-hidden="true" />
          <span><strong>19 variables</strong><small>Compared fields</small></span>
          <i aria-hidden="true" />
          <span><strong>35.5M cells</strong><small>Earth coverage</small></span>
        </div>
      </section>

      <footer className="page-footer">
        <NasaMark />
      </footer>
    </main>
  )
}

export default App
