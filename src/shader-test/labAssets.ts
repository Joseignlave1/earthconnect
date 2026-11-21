import release from '../../cdn/pinned-release.json'

const cdn = 'https://earth-connect-assets.pages.dev/'
export const earthImageUrl = cdn + release.assets['textures/earth-blue-marble.jpg'].key
export const terrainDataUrl = cdn + release.assets['textures/earth-terrain.bin'].key
export const scoreManifestUrl = cdn + release.scoreManifest.key
