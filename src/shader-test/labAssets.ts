import release from '../../cdn/pinned-release.json'

const cdn = 'https://earth-connect-assets.pages.dev/'
export const earthImageUrl = import.meta.env.DEV
  ? `${import.meta.env.BASE_URL}textures/earth-blue-marble.jpg`
  : cdn + release.assets['textures/earth-blue-marble.jpg'].key
export const terrainDataUrl = import.meta.env.DEV
  ? `${import.meta.env.BASE_URL}textures/earth-terrain.bin`
  : cdn + release.assets['textures/earth-terrain.bin'].key
export const scoreManifestUrl = cdn + release.scoreManifest.key
