import PlanetViewer from './PlanetViewer'
import EarthConnectFinder from './components/EarthAnalogFinder/EarthConnectFinder'

export default function App() {
  if (window.location.pathname === '/find') {
    return <PlanetViewer />
  }

  return <EarthConnectFinder />
}