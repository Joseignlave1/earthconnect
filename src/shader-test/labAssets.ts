import release from '../../cdn/pinned-release.json'
import flatSnapshotUrl from '/data/surface-scores.bin?url'

const cdn = 'https://earth-connect-assets.pages.dev/'
export const scoreSnapshotUrl = cdn + release.assets['data/surface-basins.bin'].key
export const earthImageUrl = cdn + release.assets['textures/earth-blue-marble.jpg'].key
export const terrainDataUrl = cdn + release.assets['textures/earth-terrain.bin'].key
export const scoreManifestUrl = cdn + release.scoreManifest.key
export { flatSnapshotUrl }
