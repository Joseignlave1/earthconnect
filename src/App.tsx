import './App.css'
import PlanetViewer from './PlanetViewer'

const starPositions = Array.from({ length: 48 }, (_, index) => ({
  left: `${(index * 37 + 7) % 100}%`,
  top: `${(index * 53 + 11) % 96}%`,
}))

function NasaMark() {
  return (
    <div className="nasa-signature">
      <span>Created for</span>
      <img src="/nasa-logo.svg" alt="NASA" />
    </div>
  )
}

function Orbit() {
  return (
    <div className="orbit-scene" aria-label="Earth, Mars and the Moon orbiting in space">
      <div className="orbit-glow" aria-hidden="true" />
      <div className="orbit-ring orbit-ring--outer" aria-hidden="true" />
      <div className="orbit-ring orbit-ring--inner" aria-hidden="true" />
      <div className="orbit-axis">
        <div className="planet-slot planet-slot--mars">
          <div className="planet-lock">
            <div className="planet mars" role="img" aria-label="Mars" />
            <span className="planet-name">Mars</span>
          </div>
        </div>
        <div className="planet-slot planet-slot--moon">
          <div className="planet-lock">
            <div className="planet moon" role="img" aria-label="The Moon" />
            <span className="planet-name">Moon</span>
          </div>
        </div>
        <div className="planet-slot planet-slot--earth">
          <div className="planet-lock">
            <div className="planet earth" role="img" aria-label="Earth" />
            <span className="planet-name">Earth</span>
          </div>
        </div>
      </div>
      <div className="orbit-center" aria-hidden="true"><span /></div>
    </div>
  )
}

function App() {
  if (window.location.pathname === '/find') {
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
        <h1>Find your best<br /><em>spot on Earth</em></h1>
        <p>
          Compare worlds. Understand environments. Discover the place on our
          planet that feels made for you.
        </p>
        <a className="find-button" href="/find">
          <span>Find your spot</span>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 12h13M14 7l5 5-5 5" />
          </svg>
        </a>
        <Orbit />
      </section>

      <footer className="page-footer">
        <NasaMark />
      </footer>
    </main>
  )
}

export default App
