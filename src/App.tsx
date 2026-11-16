import { useEffect, useState } from 'react'
import './App.css'
import PlanetViewer from './PlanetViewer'
import EarthConnectFinder from './components/EarthAnalogFinder/EarthConnectFinder'

type View = 'home' | 'find'

function viewFromLocation(): View {
  return window.location.pathname.startsWith('/find') ? 'find' : 'home'
}

export default function App() {
  const [activeView, setActiveView] = useState<View>(viewFromLocation)

  useEffect(() => {
    const syncViewWithUrl = () => setActiveView(viewFromLocation())
    window.addEventListener('popstate', syncViewWithUrl)
    return () => window.removeEventListener('popstate', syncViewWithUrl)
  }, [])

  const navigate = (view: View) => {
    if (view === activeView) return
    window.history.pushState({}, '', view === 'find' ? '/find' : '/')
    setActiveView(view)
  }

  return <>
    <div className={`app-view${activeView === 'home' ? ' is-active' : ''}`} aria-hidden={activeView !== 'home'} inert={activeView !== 'home'}>
      <EarthConnectFinder onFind={() => navigate('find')} />
    </div>
    <div className={`app-view${activeView === 'find' ? ' is-active' : ''}`} aria-hidden={activeView !== 'find'} inert={activeView !== 'find'}>
      <PlanetViewer active={activeView === 'find'} onHome={() => navigate('home')} />
    </div>
  </>
}
