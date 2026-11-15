import { useEffect, useState } from 'react'
import './App.css'
import PlanetViewer from './PlanetViewer'
import EarthConnectFinder from './components/EarthAnalogFinder/EarthConnectFinder'

export default function App() {
  const [activeView, setActiveView] = useState<'home' | 'find'>(() =>
    window.location.pathname === '/find' ? 'find' : 'home',
  )

  useEffect(() => {
    const syncViewWithUrl = () => {
      setActiveView(window.location.pathname === '/find' ? 'find' : 'home')
    }

    window.addEventListener('popstate', syncViewWithUrl)
    return () => window.removeEventListener('popstate', syncViewWithUrl)
  }, [])

  const navigate = (view: 'home' | 'find') => {
    if (view === activeView) return

    window.history.pushState({}, '', view === 'find' ? '/find' : '/')
    setActiveView(view)
  }

  return (
    <>
      <div
        className={`app-view${activeView === 'home' ? ' is-active' : ''}`}
        aria-hidden={activeView !== 'home'}
        inert={activeView !== 'home'}
      >
        <EarthConnectFinder onFind={() => navigate('find')} />
      </div>
      <div
        className={`app-view${activeView === 'find' ? ' is-active' : ''}`}
        aria-hidden={activeView !== 'find'}
        inert={activeView !== 'find'}
      >
        <PlanetViewer active={activeView === 'find'} onHome={() => navigate('home')} />
      </div>
    </>
  )
}