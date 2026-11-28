import { Bc7TileEncoder, warmBc7Pipeline } from '../shader-test/bc7-gpu.ts'
import {
  Av1Height10Decoder,
  ECV3_PREFETCH_PRIORITY,
  Ecv3ChunkCache,
  ecv3TileKey,
  HeightToNormalEncoder,
  loadEcv3PackIndex,
  reassembleEcv3AvifBlob,
  unpackEcv3HeightBounds,
  warmHeightPipelines,
  type DecodedHeight10,
  type Ecv3Pack,
  type Ecv3TileEntry,
} from '../shader-test/ecv3.ts'
import { FOV } from '../shader-test/geometry.ts'
import { generateMipmaps } from '../shader-test/gpu.ts'
import {
  createTileSampler,
  findLruSlot,
  intersectUnitSphere,
  paintTilePageTable,
  planetTileKey,
  TILE_PAGE_EMPTY,
  toPackHandle,
  VIRTUAL_TILE_INNER,
  VIRTUAL_TILE_SIZE,
  type TileVec3,
} from '../shader-test/virtual-tiles.ts'
import { PLANET_TERRAIN_SETTINGS } from './planetTerrainMesh.ts'
import { bodyRadius } from './solarSystem.ts'

export { planetTileKey }

export type PlanetTileBody = 'moon' | 'mars'

export const PLANET_TILE_PAIRS = 48
const PLANET_TILE_PAIRS_BC7 = 96

export function planetTilePairCapacity(device: GPUDevice): number {
  return device.features.has('texture-compression-bc') ? PLANET_TILE_PAIRS_BC7 : PLANET_TILE_PAIRS
}

const ECV3_CDN_BASE = 'https://earth-connect-assets.pages.dev'
const PLANET_TILE_PACKS = 48 // 12 latitude bands x 4 90° quarters
export const PLANET_TILE_BASE_COLS = 24
export const PLANET_TILE_BASE_ROWS = 12
const LEVEL_COUNT = 5 // L0 (24x12) .. L4 (384x192)
export const PLANET_TILE_PAGE_W = PLANET_TILE_BASE_COLS << (LEVEL_COUNT - 1) // 384
export const PLANET_TILE_PAGE_H = PLANET_TILE_BASE_ROWS << (LEVEL_COUNT - 1) // 192
const PLANET_TILE_MIPS = 5 // 512 -> 256 -> 128 -> 64 -> 32
const MAX_CONCURRENT_ECV3_DECODES = 24
const MAX_CONCURRENT_NETWORK_JOBS = 14
const MAX_UPLOADS_PER_FRAME = 6
const MAX_READY_UPLOADS = 16
const TILE_FADE_MS = 180
const ZOOM_SETTLE_DELAY_MS = 90
const CONE_PREFETCH_IDLE_MS = 220

type TileSlot = {
  body: PlanetTileBody
  kind: 0 | 1
  key: number // ecv3TileKey(kind, level, tx, ty)
  level: number
  tx: number
  ty: number
  lastUsed: number
  readyAt: number
}

type ReadyUpload = {
  job: TileJob
  entry: Ecv3TileEntry
  bitmap?: ImageBitmap
  decodedHeight10?: DecodedHeight10
}

type TileJob = {
  body: PlanetTileBody
  kind: 0 | 1
  key: number
  level: number
  tx: number
  ty: number
  priority: number
}

export type PlanetTilePackLoader = (
  body: PlanetTileBody,
  packId: number,
  signal: AbortSignal,
) => Promise<Blob | string | null>

export function planetTilePath(body: PlanetTileBody, packId: number) {
  return `textures/${body}-tiles-${packId}.bin`
}

export function ecv3PlanetPackUrl(body: PlanetTileBody, packId: number) {
  return `${ECV3_CDN_BASE}/ecv3/${body}/${body}-ecv3-p${String(packId).padStart(2, '0')}.ecv3?v=q92h`
}

function packIdForTile(level: number, tx: number, ty: number): number {
  const band = ty >>> level
  const qCols = 6 << level
  const quarter = Math.min(3, Math.floor(tx / qCols))
  return band * 4 + quarter
}

type BodyState = {
  packs: (Ecv3Pack | null)[]
  loadingPacks: Set<number>
  failedPacks: Set<number>
  index: Map<number, Ecv3TileEntry>
  residentAlbedo: Map<number, number> // ecv3Key -> albedo slotIndex [0..pairCapacity-1]
  residentNormal: Map<number, number> // ecv3Key -> normal slotIndex [0..pairCapacity-1]
  inFlight: Set<number>
  // Two stacked 384x192 page tables inside one 384x384 rgba8uint texture:
  //   top half    [0..192)   = Albedo (kind 0)
  //   bottom half [192..384) = Normal (kind 1)
  pageData: Uint8Array<ArrayBuffer>
  pageAlbedoView: Uint8Array<ArrayBuffer>
  pageNormalView: Uint8Array<ArrayBuffer>
  pageTexture: GPUTexture
  albedoPageDirty: boolean
  normalPageDirty: boolean
  viewMaxLevel: number
  indexed: boolean
  lastEye: [number, number, number] | null
  lastMpp: number
}

export class PlanetTileCache {
  private readonly device: GPUDevice
  private readonly useBc7: boolean
  readonly pairCapacity: number
  private readonly controller = new AbortController()
  private readonly bodies: Record<PlanetTileBody, BodyState>
  private readonly albedoSlots: (TileSlot | null)[]
  private readonly normalSlots: (TileSlot | null)[]
  private bc7Encoder: Bc7TileEncoder | null = null
  private heightEncoder: HeightToNormalEncoder | null = null
  private readonly height10Decoder = new Av1Height10Decoder(VIRTUAL_TILE_SIZE, 4)
  private readonly chunkCache = new Ecv3ChunkCache()
  private readonly slotFadeValues = new Float32Array(PLANET_TILE_PAIRS_BC7 * 2)
  private hasActiveFades = false
  private activeDemand = new Set<string>()
  private pendingQueue: TileJob[] = []
  private readyUploads: ReadyUpload[] = []
  private flushFrame = 0
  private zoomSettleTimer: ReturnType<typeof setTimeout> | null = null
  private zoomSettleUntil = 0
  private conePrefetchTimer: ReturnType<typeof setTimeout> | null = null
  private coneSettled = false
  private lastConeCenterKey = ''
  private activeDecodes = 0
  private activeNetworkJobs = 0
  private frameCounter = 0
  private _revision = 0
  private pausedForFlight = false
  private destroyed = false
  private poolAllocated = false
  private packsAttached = false
  private readonly tileListeners = new Set<() => void>()
  private readonly textureListeners = new Set<() => void>()

  poolTexture: GPUTexture
  readonly slotFadeBuffer: GPUBuffer
  readonly sampler: GPUSampler

  constructor(device: GPUDevice) {
    this.device = device
    this.useBc7 = device.features.has('texture-compression-bc')
    if (this.useBc7) warmBc7Pipeline(device)
    warmHeightPipelines(device)
    this.pairCapacity = planetTilePairCapacity(device)
    this.albedoSlots = Array(this.pairCapacity).fill(null)
    this.normalSlots = Array(this.pairCapacity).fill(null)
    this.slotFadeValues.fill(1)
    this.slotFadeBuffer = device.createBuffer({
      label: 'Planetary ECV3 per-slot fade weights',
      size: this.slotFadeValues.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.slotFadeBuffer, 0, this.slotFadeValues)
    this.sampler = createTileSampler(device, 'Planetary ECV3 virtual tile sampler')
    // Tiny placeholders until the user actually zooms into Moon or Mars (0 VRAM on Earth).
    this.poolTexture = device.createTexture({
      label: 'Planetary tile pool placeholder',
      size: [1, 1, 2],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture(
      { texture: this.poolTexture },
      new Uint8Array([128, 128, 128, 255, 128, 128, 255, 255]),
      { bytesPerRow: 4, rowsPerImage: 1 },
      [1, 1, 2],
    )
    const halfBytes = PLANET_TILE_PAGE_W * PLANET_TILE_PAGE_H * 4
    const makeBody = (body: PlanetTileBody): BodyState => {
      const pageData = new Uint8Array(halfBytes * 2)
      pageData.fill(TILE_PAGE_EMPTY)
      const pageTexture = device.createTexture({
        label: `${body} tile page placeholder`,
        size: [1, 2],
        format: 'rgba8uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      device.queue.writeTexture(
        { texture: pageTexture },
        new Uint8Array([255, 0, 255, 0, 255, 0, 255, 0]),
        { bytesPerRow: 4 },
        [1, 2],
      )
      return {
        packs: Array(PLANET_TILE_PACKS).fill(null),
        loadingPacks: new Set(),
        failedPacks: new Set(),
        index: new Map(),
        residentAlbedo: new Map(),
        residentNormal: new Map(),
        inFlight: new Set(),
        pageData,
        pageAlbedoView: pageData.subarray(0, halfBytes),
        pageNormalView: pageData.subarray(halfBytes, halfBytes * 2),
        pageTexture,
        albedoPageDirty: false,
        normalPageDirty: false,
        viewMaxLevel: LEVEL_COUNT - 1,
        indexed: false,
        lastEye: null,
        lastMpp: Infinity,
      }
    }
    this.bodies = { moon: makeBody('moon'), mars: makeBody('mars') }
  }

  pageTexture(body: PlanetTileBody) {
    return this.bodies[body].pageTexture
  }
  ready(body: PlanetTileBody) {
    return this.poolAllocated && this.bodies[body].indexed
  }
  hasIndex(body: PlanetTileBody) {
    return this.bodies[body].indexed
  }
  residentCount(body: PlanetTileBody) {
    const state = this.bodies[body]
    return state.residentAlbedo.size + state.residentNormal.size
  }
  indexedCount(body: PlanetTileBody) {
    return this.bodies[body].index.size
  }
  get revision() {
    return this._revision
  }

  subscribe(onTileReady: () => void, onTexturesChanged: () => void) {
    this.tileListeners.add(onTileReady)
    this.textureListeners.add(onTexturesChanged)
    return () => {
      this.tileListeners.delete(onTileReady)
      this.textureListeners.delete(onTexturesChanged)
    }
  }

  private notifyTileReady() {
    for (const listener of this.tileListeners) listener()
  }

  private notifyTexturesChanged() {
    for (const listener of this.textureListeners) listener()
  }

  attachPacks() {
    if (this.packsAttached || this.destroyed) return
    this.packsAttached = true
  }

  /** Pause queued and in-flight tile decodes/uploads while flying between bodies so flights never stutter. */
  pauseForFlight() {
    this.pausedForFlight = true
    this.pendingQueue.length = 0
    this.activeDemand.clear()
    this.clearReadyUploads()
    this.chunkCache.cancelQueued()
    if (this.zoomSettleTimer !== null) {
      clearTimeout(this.zoomSettleTimer)
      this.zoomSettleTimer = null
    }
    this.zoomSettleUntil = 0
    if (this.conePrefetchTimer !== null) {
      clearTimeout(this.conePrefetchTimer)
      this.conePrefetchTimer = null
    }
    this.coneSettled = false
    this.lastConeCenterKey = ''
    for (const body of ['moon', 'mars'] as const) {
      this.bodies[body].lastEye = null
    }
  }

  private clearReadyUploads() {
    for (const item of this.readyUploads) {
      item.bitmap?.close()
      this.bodies[item.job.body].inFlight.delete(item.job.key)
    }
    this.readyUploads.length = 0
  }

  private pruneStaleReadyUploads() {
    if (this.readyUploads.length === 0) return
    let write = 0
    for (let i = 0; i < this.readyUploads.length; i++) {
      const item = this.readyUploads[i]
      if (this.activeDemand.has(`${item.job.body}:${item.job.key}`)) {
        this.readyUploads[write++] = item
      } else {
        item.bitmap?.close()
        this.bodies[item.job.body].inFlight.delete(item.job.key)
      }
    }
    this.readyUploads.length = write
  }

  /** Advance per-slot smooth fade-in weights; returns true while any tile is still crossfading. */
  updateFades(now: number): boolean {
    if (!this.hasActiveFades || this.destroyed) return false
    let anyFading = false
    for (let i = 0; i < this.pairCapacity; i++) {
      const a = this.albedoSlots[i]
      if (a) {
        const r = Math.max(0, Math.min(1, (now - a.readyAt) / TILE_FADE_MS))
        this.slotFadeValues[i] = r * r * (3 - 2 * r)
        if (r < 1) anyFading = true
      } else {
        this.slotFadeValues[i] = 1
      }
      const n = this.normalSlots[i]
      if (n) {
        const r = Math.max(0, Math.min(1, (now - n.readyAt) / TILE_FADE_MS))
        this.slotFadeValues[PLANET_TILE_PAIRS_BC7 + i] = r * r * (3 - 2 * r)
        if (r < 1) anyFading = true
      } else {
        this.slotFadeValues[PLANET_TILE_PAIRS_BC7 + i] = 1
      }
    }
    this.device.queue.writeBuffer(this.slotFadeBuffer, 0, this.slotFadeValues)
    this.hasActiveFades = anyFading
    this._revision++
    return anyFading
  }

  private ensurePackLoaded(body: PlanetTileBody, packId: number) {
    const state = this.bodies[body]
    if (
      this.destroyed ||
      packId < 0 ||
      packId >= PLANET_TILE_PACKS ||
      state.packs[packId] ||
      state.loadingPacks.has(packId) ||
      state.failedPacks.has(packId)
    ) {
      return
    }
    state.loadingPacks.add(packId)
    void this.loadPackIndex(body, packId, this.controller.signal)
  }

  private async loadPackIndex(body: PlanetTileBody, packId: number, signal: AbortSignal) {
    const state = this.bodies[body]
    try {
      const handle = toPackHandle(ecv3PlanetPackUrl(body, packId))
      const loaded = await loadEcv3PackIndex(handle, packId, signal)
      if (!loaded || signal.aborted || this.destroyed) {
        state.loadingPacks.delete(packId)
        if (!signal.aborted && !this.destroyed) state.failedPacks.add(packId)
        return
      }
      state.packs[packId] = loaded.pack
      if (loaded.chunk0) this.chunkCache.seedChunk(`${body}:${packId}`, 0, loaded.chunk0)
      state.loadingPacks.delete(packId)
      for (const [key, entry] of loaded.entries) state.index.set(key, entry)
      if (!state.indexed && state.index.size > 0) state.indexed = true
      if (state.index.size > 0) this.notifyTileReady()
    } catch (error) {
      state.loadingPacks.delete(packId)
      if (!signal.aborted && !this.destroyed && error instanceof Error && error.message.includes('(404)')) {
        state.failedPacks.add(packId)
      }
    }
  }

  private ensureGpuPool() {
    if (this.poolAllocated || this.destroyed) return
    this.poolAllocated = true
    const device = this.device
    const oldPool = this.poolTexture
    this.poolTexture = device.createTexture({
      label: this.useBc7
        ? 'Planetary ECV3 albedo + normal virtual tile cache (BC7)'
        : 'Planetary ECV3 albedo + normal virtual tile cache',
      size: [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE, this.pairCapacity * 2],
      mipLevelCount: PLANET_TILE_MIPS,
      format: this.useBc7 ? 'bc7-rgba-unorm' : 'rgba8unorm',
      usage: this.useBc7
        ? GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    if (this.useBc7) {
      this.bc7Encoder = new Bc7TileEncoder(device, VIRTUAL_TILE_SIZE, PLANET_TILE_MIPS, 'rgba8unorm')
    }
    this.heightEncoder = new HeightToNormalEncoder(device, VIRTUAL_TILE_SIZE)

    const oldTextures: GPUTexture[] = [oldPool]
    for (const body of ['moon', 'mars'] as const) {
      const state = this.bodies[body]
      const oldPage = state.pageTexture
      state.pageTexture = device.createTexture({
        label: `${body} ECV3 virtual tile page table`,
        size: [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H * 2],
        format: 'rgba8uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      state.pageData.fill(TILE_PAGE_EMPTY)
      device.queue.writeTexture(
        { texture: state.pageTexture },
        state.pageData,
        { bytesPerRow: PLANET_TILE_PAGE_W * 4 },
        [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H * 2],
      )
      oldTextures.push(oldPage)
    }
    this.notifyTexturesChanged()
    void device.queue
      .onSubmittedWorkDone()
      .then(() => {
        for (const tex of oldTextures) tex.destroy()
      })
      .catch(() => undefined)
  }

  /**
   * Prefetch the entire center zoom cone (`startLevel .. L4`) into the 32 MiB RAM chunk cache
   * (and browser immutable cache) once the user stops panning/rotating. Uses zero GPU pool slots
   * and strictly low-priority background streams so zooming straight in hits RAM (~10 ms).
   */
  private prefetchZoomCone(body: PlanetTileBody, lon: number, lat: number, startLevel: number) {
    const state = this.bodies[body]
    const u = (((lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
    const v = Math.max(0, Math.min(0.999999, 0.5 - lat / Math.PI))
    const maxLevel = LEVEL_COUNT - 1
    const centerKey = `${body}:${Math.floor(u * (PLANET_TILE_BASE_COLS << maxLevel))}:${Math.floor(v * (PLANET_TILE_BASE_ROWS << maxLevel))}:${startLevel}`
    const packSet = new Set<number>()
    let anyPackPending = false

    for (let level = Math.max(0, startLevel); level <= maxLevel; level++) {
      const cols = PLANET_TILE_BASE_COLS << level
      const rows = PLANET_TILE_BASE_ROWS << level
      const isDeepest = level === maxLevel
      const x0 = isDeepest ? Math.floor(u * cols) - 1 : Math.floor(u * cols - 0.5)
      const y0 = isDeepest ? Math.floor(v * rows) - 1 : Math.floor(v * rows - 0.5)
      const span = isDeepest ? 3 : 2
      for (let dy = 0; dy < span; dy++) {
        const ty = y0 + dy
        if (ty < 0 || ty >= rows) continue
        for (let dx = 0; dx < span; dx++) {
          const tx = (x0 + dx + cols) % cols
          const pid = packIdForTile(level, tx, ty)
          if (!state.packs[pid]) {
            if (!packSet.has(pid)) {
              packSet.add(pid)
              this.ensurePackLoaded(body, pid)
            }
            anyPackPending = true
            continue
          }
          if (this.lastConeCenterKey === centerKey) continue
          const pack = state.packs[pid]!
          const packKey = `${body}:${pid}`
          const prio = ECV3_PREFETCH_PRIORITY + (level - startLevel) * 10 + (dx === 1 && dy === 1 ? 0 : 2)
          for (const kind of [0, 1] as const) {
            let lv = level
            let cx = tx
            let cy = ty
            let key = ecv3TileKey(kind, lv, cx, cy)
            while (lv > 0 && !state.index.has(key)) {
              lv--
              cx >>>= 1
              cy >>>= 1
              key = ecv3TileKey(kind, lv, cx, cy)
            }
            const entry = state.index.get(key)
            if (!entry) continue
            this.chunkCache.prefetchSlice(pack.handle, packKey, entry.offset, entry.length, prio + kind)
          }
        }
      }
    }
    if (!anyPackPending) this.lastConeCenterKey = centerKey
  }

  private scheduleConePrefetch(cameraMoved: boolean) {
    if (cameraMoved) {
      this.coneSettled = false
      this.lastConeCenterKey = ''
      this.chunkCache.cancelLowPriority(ECV3_PREFETCH_PRIORITY)
      if (this.conePrefetchTimer !== null) clearTimeout(this.conePrefetchTimer)
      this.conePrefetchTimer = setTimeout(() => {
        this.conePrefetchTimer = null
        if (this.destroyed || this.pausedForFlight) return
        this.coneSettled = true
        this.notifyTileReady()
      }, CONE_PREFETCH_IDLE_MS)
    } else if (!this.coneSettled && this.conePrefetchTimer === null) {
      this.conePrefetchTimer = setTimeout(() => {
        this.conePrefetchTimer = null
        if (this.destroyed || this.pausedForFlight) return
        this.coneSettled = true
        this.notifyTileReady()
      }, CONE_PREFETCH_IDLE_MS)
    }
  }

  /**
   * Determine visible Moon or Mars tiles in body-local unit-sphere camera space,
   * lazily load visible 15°x90° ECV3 packs over HTTP, and stream decoupled
   * AVIF albedo + 10-bit height-derived normal tiles across L0..L4 with full-depth
   * settled zoom-cone prefetching.
   */
  updateView(
    body: PlanetTileBody,
    eyeUnit: TileVec3,
    right: TileVec3,
    up: TileVec3,
    back: TileVec3,
    aspect: number,
    radiusMetres: number,
    physicalMetresPerPixel: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    if (!this.packsAttached || this.destroyed) return
    this.pausedForFlight = false
    const state = this.bodies[body]

    // L0 = (2*pi*R)/(24*508) -> Moon: 895 m/px .. L4 56 m/px; Mars: 1747 m/px .. L4 109 m/px.
    // Match the shader's L0 fadeEnd (tileMip = 1.15 -> 2.22x l0TexelM) with margin so every pixel
    // with non-zero tileWeight has its tile requested before becoming visible.
    const l0TexelM = (2 * Math.PI * radiusMetres) / (PLANET_TILE_BASE_COLS * VIRTUAL_TILE_INNER)
    const entryThreshold = l0TexelM * 2.4

    const prevEye = state.lastEye
    const prevMpp = state.lastMpp
    const dEyeX = prevEye ? eyeUnit[0] - prevEye[0] : 0
    const dEyeY = prevEye ? eyeUnit[1] - prevEye[1] : 0
    const dEyeZ = prevEye ? eyeUnit[2] - prevEye[2] : 0
    const panSpeed = Math.hypot(dEyeX, dEyeY, dEyeZ)
    const zoomRatio = Number.isFinite(prevMpp) && physicalMetresPerPixel > 0 ? prevMpp / physicalMetresPerPixel : 1
    const cameraMoved = prevEye === null || panSpeed > 1e-4 || Math.abs(zoomRatio - 1) > 0.015
    this.scheduleConePrefetch(cameraMoved)

    state.lastEye = [eyeUnit[0], eyeUnit[1], eyeUnit[2]]
    state.lastMpp = physicalMetresPerPixel

    const loupeMpp = loupe ? physicalMetresPerPixel / Math.max(1, loupe.zoom) : Infinity
    if (Math.min(physicalMetresPerPixel, loupeMpp) > entryThreshold) {
      this.pendingQueue.length = 0
      // Prewarm visible hemisphere pack indices AND L0 tiles while still at orbital globe distance
      // so the instant the user starts zooming in, pack indices & base L0 are already resident (0 ms).
      if (physicalMetresPerPixel <= entryThreshold * 5.0) {
        const tanHalf = Math.tan(FOV / 2)
        const prewarmJobs: TileJob[] = []
        const seenPrewarm = new Set<number>()
        for (let gy = -2; gy <= 2; gy++) {
          for (let gx = -2; gx <= 2; gx++) {
            const hit = intersectUnitSphere(eyeUnit, right, up, back, gx * 0.35 * tanHalf * aspect, gy * 0.35 * tanHalf)
            if (!hit) continue
            const u = (((hit.lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
            const v = Math.max(0, Math.min(0.999999, 0.5 - hit.lat / Math.PI))
            const tx = Math.floor(u * PLANET_TILE_BASE_COLS) % PLANET_TILE_BASE_COLS
            const ty = Math.max(0, Math.min(PLANET_TILE_BASE_ROWS - 1, Math.floor(v * PLANET_TILE_BASE_ROWS)))
            const pid = packIdForTile(0, tx, ty)
            if (!state.packs[pid]) {
              this.ensurePackLoaded(body, pid)
              continue
            }
            const distSq = gx * gx + gy * gy
            for (const kind of [0, 1] as const) {
              const key = ecv3TileKey(kind, 0, tx, ty)
              if (seenPrewarm.has(key) || !state.index.has(key) || state.inFlight.has(key)) continue
              const resMap = kind === 0 ? state.residentAlbedo : state.residentNormal
              if (resMap.has(key)) continue
              seenPrewarm.add(key)
              prewarmJobs.push({ body, kind, key, level: 0, tx, ty, priority: distSq + kind * 0.1 })
            }
          }
        }
        if (prewarmJobs.length > 0) {
          prewarmJobs.sort((a, b) => a.priority - b.priority)
          const capped = prewarmJobs.slice(0, 16)
          this.activeDemand = new Set(capped.map(j => `${j.body}:${j.key}`))
          this.pruneStaleReadyUploads()
          this.pendingQueue = capped
          this.pumpQueue()
        } else {
          this.activeDemand.clear()
          this.pruneStaleReadyUploads()
        }
      } else {
        this.activeDemand.clear()
        this.pruneStaleReadyUploads()
      }
      if (this.coneSettled) {
        const centerHit = intersectUnitSphere(eyeUnit, right, up, back, 0, 0)
        if (centerHit) this.prefetchZoomCone(body, centerHit.lon, centerHit.lat, 0)
      }
      return
    }

    if (state.indexed) this.ensureGpuPool()

    const frame = ++this.frameCounter
    const desiredAlbedo = new Map<number, TileJob>()
    const desiredNormal = new Map<number, TileJob>()
    let maxSeenLevel = 0
    let onscreenTargetMissing = 0
    const now = performance.now()
    const isFastZooming = zoomRatio > 1.08 || zoomRatio < 0.92
    if (isFastZooming) {
      this.zoomSettleUntil = now + ZOOM_SETTLE_DELAY_MS
      if (this.zoomSettleTimer !== null) clearTimeout(this.zoomSettleTimer)
      this.zoomSettleTimer = setTimeout(() => {
        this.zoomSettleTimer = null
        if (!this.destroyed && !this.pausedForFlight) this.notifyTileReady()
      }, ZOOM_SETTLE_DELAY_MS)
    }
    const zoomSuppressing = isFastZooming || now < this.zoomSettleUntil

    const requestKindTile = (
      kind: 0 | 1,
      level: number,
      tx: number,
      ty: number,
      distSq: number,
      offsetPenalty: number,
      levelBias: number,
      isOnscreenTarget = false,
    ) => {
      const packId = packIdForTile(level, tx, ty)
      if (!state.packs[packId]) {
        if (packId < 0 || packId >= PLANET_TILE_PACKS || state.failedPacks.has(packId)) return
        this.ensurePackLoaded(body, packId)
        if (isOnscreenTarget) onscreenTargetMissing++
        return
      }
      // Walk up from requested level to find the finest non-pruned / available tile
      // (handles Moon L3 albedo vs L4 DEM, Mars L4 albedo vs L3 DEM, pruned flat tiles, and polar caps).
      let lv = level
      let cx = tx
      let cy = ty
      let key = ecv3TileKey(kind, lv, cx, cy)
      while (lv > 0 && !state.index.has(key)) {
        lv--
        cx >>>= 1
        cy >>>= 1
        key = ecv3TileKey(kind, lv, cx, cy)
      }
      if (!state.index.has(key)) return

      const resMap = kind === 0 ? state.residentAlbedo : state.residentNormal
      if (!resMap.has(key) && isOnscreenTarget) onscreenTargetMissing++
      const priority = levelBias + distSq + offsetPenalty + kind * 0.08
      const desired = kind === 0 ? desiredAlbedo : desiredNormal
      const prev = desired.get(key)
      if (!prev || priority < prev.priority) {
        desired.set(key, { body, kind, key, level: lv, tx: cx, ty: cy, priority })
      }
    }

    const requestPoint = (
      lon: number,
      lat: number,
      mpp: number,
      distSq: number,
      halo: number,
      includeParent = false,
      onscreen = false,
    ) => {
      if (mpp > entryThreshold) return
      const u = (((lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
      const v = Math.max(0, Math.min(0.999999, 0.5 - lat / Math.PI))
      const targetLevel = Math.max(
        0,
        Math.min(LEVEL_COUNT - 1, Math.floor(Math.log2((l0TexelM * 1.55) / Math.max(1, mpp)))),
      )
      if (targetLevel > maxSeenLevel) maxSeenLevel = targetLevel

      // Primary pass requests L0 anchor + targetLevel directly; parent fallback waits until
      // all visible on-screen target tiles are resident so 100% of HTTP/2 streams serve visible pixels.
      const levelSpecs: [number, number][] = [[0, -10]]
      if (targetLevel >= 1) levelSpecs.push([targetLevel, 0])
      if (includeParent && targetLevel >= 2) levelSpecs.push([targetLevel - 1, 6])

      for (const [level, bias] of levelSpecs) {
        const cols = PLANET_TILE_BASE_COLS << level
        const rows = PLANET_TILE_BASE_ROWS << level
        const cx = Math.floor(u * cols) % cols
        const cy = Math.max(0, Math.min(rows - 1, Math.floor(v * rows)))
        const radius = level === targetLevel ? halo : 0
        const isPrimary = onscreen && (level === 0 || level === targetLevel)
        for (let dy = -radius; dy <= radius; dy++) {
          const ty = cy + dy
          if (ty < 0 || ty >= rows) continue
          for (let dx = -radius; dx <= radius; dx++) {
            const tx = (cx + dx + cols) % cols
            const pen = (dx * dx + dy * dy) * 0.25
            requestKindTile(0, level, tx, ty, distSq, pen, bias, isPrimary)
            requestKindTile(1, level, tx, ty, distSq, pen, bias, isPrimary)
          }
        }
      }
    }

    const onscreenRays: { lon: number; lat: number; effectiveMpp: number; prio: number }[] = []
    const offscreenRays: { lon: number; lat: number; effectiveMpp: number; prio: number }[] = []
    if (physicalMetresPerPixel <= entryThreshold) {
      const tanHalf = Math.tan(FOV / 2)
      const gridN = 15
      for (let gy = 0; gy < gridN; gy++) {
        const ndcY = ((gy / (gridN - 1)) * 2 - 1) * 1.28
        for (let gx = 0; gx < gridN; gx++) {
          const ndcX = ((gx / (gridN - 1)) * 2 - 1) * 1.28
          const hit = intersectUnitSphere(eyeUnit, right, up, back, ndcX * tanHalf * aspect, ndcY * tanHalf)
          if (!hit) continue
          const effectiveMpp = physicalMetresPerPixel / Math.max(0.55, hit.cosIncidence)
          const isOffscreen = Math.abs(ndcX) > 1.02 || Math.abs(ndcY) > 1.02
          if (isOffscreen) {
            offscreenRays.push({
              lon: hit.lon,
              lat: hit.lat,
              effectiveMpp,
              prio: ndcX * ndcX + ndcY * ndcY + 4.0,
            })
          } else {
            onscreenRays.push({
              lon: hit.lon,
              lat: hit.lat,
              effectiveMpp,
              prio: ndcX * ndcX + ndcY * ndcY,
            })
            requestPoint(hit.lon, hit.lat, effectiveMpp, ndcX * ndcX + ndcY * ndcY, 0, false, true)
          }
        }
      }
    }

    if (loupe && loupeMpp <= entryThreshold) {
      requestPoint(loupe.lon, loupe.lat, loupeMpp, -2, 1, false, true)
    }

    // --- Secondary warming (strictly gated until every visible on-screen target tile is resident) ---
    if (physicalMetresPerPixel <= entryThreshold && !zoomSuppressing && onscreenTargetMissing === 0) {
      for (const r of onscreenRays) {
        requestPoint(r.lon, r.lat, r.effectiveMpp, r.prio + 2, 0, true, false)
      }
      for (const r of offscreenRays) {
        requestPoint(r.lon, r.lat, r.effectiveMpp, r.prio + 4, 0, false, false)
      }
      const tanHalf = Math.tan(FOV / 2)
      const isZoomingIn = Number.isFinite(prevMpp) && physicalMetresPerPixel < prevMpp * 0.985
      // 1. Zoom-in velocity lookahead: fetch 1 mip level deeper at screen center while scrolling in.
      if (isZoomingIn && maxSeenLevel < LEVEL_COUNT - 1) {
        const nextLevel = maxSeenLevel + 1
        const cols = PLANET_TILE_BASE_COLS << nextLevel
        const rows = PLANET_TILE_BASE_ROWS << nextLevel
        for (let gy = -1; gy <= 1; gy++) {
          for (let gx = -1; gx <= 1; gx++) {
            const hit = intersectUnitSphere(eyeUnit, right, up, back, gx * 0.35 * tanHalf * aspect, gy * 0.35 * tanHalf)
            if (!hit) continue
            const u = (((hit.lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
            const v = Math.max(0, Math.min(0.999999, 0.5 - hit.lat / Math.PI))
            const tx = Math.floor(u * cols) % cols
            const ty = Math.max(0, Math.min(rows - 1, Math.floor(v * rows)))
            requestKindTile(0, nextLevel, tx, ty, gx * gx + gy * gy, 0, 5)
            requestKindTile(1, nextLevel, tx, ty, gx * gx + gy * gy, 0, 5)
          }
        }
      }

      // 2. Pan-velocity lookahead: shift rays along the leading edge of camera motion.
      const velR = dEyeX * right[0] + dEyeY * right[1] + dEyeZ * right[2]
      const velU = dEyeX * up[0] + dEyeY * up[1] + dEyeZ * up[2]
      const velMag = Math.hypot(velR, velU)
      if (velMag > 1e-4) {
        const dirX = velR / velMag
        const dirY = velU / velMag
        for (let i = -1; i <= 1; i++) {
          const ndcX = dirX * 1.45 - dirY * (i * 0.55)
          const ndcY = dirY * 1.45 + dirX * (i * 0.55)
          const hit = intersectUnitSphere(eyeUnit, right, up, back, ndcX * tanHalf * aspect, ndcY * tanHalf)
          if (!hit) continue
          const effectiveMpp = physicalMetresPerPixel / Math.max(0.55, hit.cosIncidence)
          requestPoint(hit.lon, hit.lat, effectiveMpp, 8 + i * i, 0, false, false)
        }
      }

      // 3. Settled center lookahead: once the current viewport is 100% loaded, GPU-warm one level
      //    deeper at center and RAM-warm the full remaining zoom cone all the way down to L4.
      const hasMissingSecondary =
        Array.from(desiredAlbedo.keys()).some(k => !state.residentAlbedo.has(k)) ||
        Array.from(desiredNormal.keys()).some(k => !state.residentNormal.has(k))
      if (!hasMissingSecondary && maxSeenLevel < LEVEL_COUNT - 1) {
        const nextLevel = maxSeenLevel + 1
        const cols = PLANET_TILE_BASE_COLS << nextLevel
        const rows = PLANET_TILE_BASE_ROWS << nextLevel
        for (let gy = -1; gy <= 1; gy++) {
          for (let gx = -1; gx <= 1; gx++) {
            const hit = intersectUnitSphere(eyeUnit, right, up, back, gx * 0.28 * tanHalf * aspect, gy * 0.28 * tanHalf)
            if (!hit) continue
            const u = (((hit.lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
            const v = Math.max(0, Math.min(0.999999, 0.5 - hit.lat / Math.PI))
            const tx = Math.floor(u * cols) % cols
            const ty = Math.max(0, Math.min(rows - 1, Math.floor(v * rows)))
            requestKindTile(0, nextLevel, tx, ty, gx * gx + gy * gy, 0, 12)
            requestKindTile(1, nextLevel, tx, ty, gx * gx + gy * gy, 0, 12)
          }
        }
        if (this.coneSettled) {
          const centerHit = intersectUnitSphere(eyeUnit, right, up, back, 0, 0)
          if (centerHit) this.prefetchZoomCone(body, centerHit.lon, centerHit.lat, nextLevel)
        }
      }
    }

    if (state.viewMaxLevel !== maxSeenLevel) {
      if (maxSeenLevel > state.viewMaxLevel) this.chunkCache.cancelQueued()
      state.viewMaxLevel = maxSeenLevel
      state.albedoPageDirty = true
      state.normalPageDirty = true
      this.flushPageTables()
    }

    // Admit a stable, pool-capacity-bounded set per kind by screen priority so wide/4K
    // viewports never evict and reload visible tiles in an endless stationary loop.
    const nextDemand = new Set<string>()
    const nextQueue: TileJob[] = []
    for (const [kind, desired] of [[0, desiredAlbedo] as const, [1, desiredNormal] as const]) {
      const resMap = kind === 0 ? state.residentAlbedo : state.residentNormal
      const slots = kind === 0 ? this.albedoSlots : this.normalSlots
      const ranked = Array.from(desired.values()).sort((a, b) => a.priority - b.priority)
      const admitted = ranked.length > this.pairCapacity ? ranked.slice(0, this.pairCapacity) : ranked
      for (const job of admitted) {
        nextDemand.add(`${job.body}:${job.key}`)
        const slotIndex = resMap.get(job.key)
        if (slotIndex !== undefined) {
          const slot = slots[slotIndex]
          if (slot) slot.lastUsed = frame
          continue
        }
        if (state.inFlight.has(job.key)) continue
        // During rapid scroll-zooming (or within the 90 ms settle window), skip network misses
        // for transient intermediate levels (L1..L3) unless already in RAM cache.
        if (zoomSuppressing && job.level > 0 && job.level < LEVEL_COUNT - 1) {
          const entry = state.index.get(job.key)
          if (!entry || !this.chunkCache.hasSlice(`${body}:${entry.packId}`, entry.offset, entry.length)) continue
        }
        nextQueue.push(job)
      }
    }
    this.activeDemand = nextDemand
    this.pruneStaleReadyUploads()

    if (nextQueue.length === 0) {
      this.pendingQueue.length = 0
      return
    }
    nextQueue.sort((a, b) => a.priority - b.priority)
    this.pendingQueue = nextQueue
    this.pumpQueue()
  }

  private pumpQueue() {
    for (
      let i = 0;
      i < this.pendingQueue.length &&
      this.activeDecodes < MAX_CONCURRENT_ECV3_DECODES &&
      this.readyUploads.length < MAX_READY_UPLOADS &&
      !this.destroyed;
    ) {
      const job = this.pendingQueue[i]
      const state = this.bodies[job.body]
      const resMap = job.kind === 0 ? state.residentAlbedo : state.residentNormal
      if (resMap.has(job.key) || state.inFlight.has(job.key)) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const entry = state.index.get(job.key)
      if (!entry) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const pack = state.packs[entry.packId]
      if (!pack) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const tpl = pack.templates.get(entry.tplId)
      if (!tpl) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const isRamHit = this.chunkCache.hasSlice(`${job.body}:${entry.packId}`, entry.offset, entry.length)
      if (!isRamHit && this.activeNetworkJobs >= MAX_CONCURRENT_NETWORK_JOBS) {
        i++
        continue
      }
      this.pendingQueue.splice(i, 1)
      state.inFlight.add(job.key)
      this.activeDecodes++
      if (!isRamHit) this.activeNetworkJobs++
      void this.decodeAndUpload(job, entry, pack, !isRamHit)
    }
  }

  private allocateSlot(kind: 0 | 1): number {
    const slots = kind === 0 ? this.albedoSlots : this.normalSlots
    const slotIndex = findLruSlot(slots)
    const evicted = slots[slotIndex]
    if (evicted) {
      const evictedBody = this.bodies[evicted.body]
      const resMap = kind === 0 ? evictedBody.residentAlbedo : evictedBody.residentNormal
      resMap.delete(evicted.key)
      if (kind === 0) evictedBody.albedoPageDirty = true
      else evictedBody.normalPageDirty = true
      slots[slotIndex] = null
    }
    return slotIndex
  }

  private async decodeAndUpload(job: TileJob, entry: Ecv3TileEntry, pack: Ecv3Pack, holdsNetworkSlot: boolean) {
    const { signal } = this.controller
    const state = this.bodies[job.body]
    let bitmap: ImageBitmap | undefined
    let transferredToUpload = false
    let networkSlotHeld = holdsNetworkSlot
    const releaseNetworkSlot = () => {
      if (!networkSlotHeld) return
      networkSlotHeld = false
      this.activeNetworkJobs--
      this.pumpQueue()
    }
    try {
      const tpl = pack.templates.get(entry.tplId)
      if (!tpl) return
      const obuBytes = await this.chunkCache.readSlice(
        pack.handle,
        `${job.body}:${entry.packId}`,
        entry.offset,
        entry.length,
        signal,
        job.priority,
      )
      releaseNetworkSlot()
      if (signal.aborted || this.destroyed || this.pausedForFlight) return
      let decodedHeight10: DecodedHeight10 | null = null
      if (job.kind === 1) {
        // Decode raw 10-bit AV1 OBU via WebCodecs VideoDecoder to preserve all 1,024 elevation steps.
        decodedHeight10 = await this.height10Decoder.decode(obuBytes, signal)
        if (signal.aborted || this.destroyed || this.pausedForFlight) return
      }
      if (!decodedHeight10) {
        const avifBlob = reassembleEcv3AvifBlob(tpl, obuBytes)
        bitmap = await createImageBitmap(avifBlob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
        if (signal.aborted || this.destroyed || this.pausedForFlight) return
      }

      this.readyUploads.push({ job, entry, bitmap, decodedHeight10: decodedHeight10 ?? undefined })
      bitmap = undefined
      transferredToUpload = true
      this.scheduleFlush()
    } catch {
      // Transient aborts/decode errors degrade gracefully to coarser ECV3 levels or 8K base textures.
    } finally {
      bitmap?.close()
      if (networkSlotHeld) {
        networkSlotHeld = false
        this.activeNetworkJobs--
      }
      if (!transferredToUpload) state.inFlight.delete(job.key)
      this.activeDecodes--
      this.pumpQueue()
    }
  }

  private touchParentSlot(body: PlanetTileBody, kind: 0 | 1, level: number, tx: number, ty: number) {
    const state = this.bodies[body]
    const resMap = kind === 0 ? state.residentAlbedo : state.residentNormal
    const slots = kind === 0 ? this.albedoSlots : this.normalSlots
    let lv = level - 1
    let cx = tx >>> 1
    let cy = ty >>> 1
    while (lv >= 0) {
      const idx = resMap.get(ecv3TileKey(kind, lv, cx, cy))
      if (idx !== undefined) {
        const slot = slots[idx]
        if (slot) slot.lastUsed = this.frameCounter
        return
      }
      lv--
      cx >>>= 1
      cy >>>= 1
    }
  }

  private uploadDecodedTile(item: ReadyUpload, now: number) {
    const { job, entry, bitmap, decodedHeight10 } = item
    const state = this.bodies[job.body]
    try {
      if (!this.activeDemand.has(`${job.body}:${job.key}`)) return
      this.ensureGpuPool()
      if (job.level > 0) this.touchParentSlot(job.body, job.kind, job.level, job.tx, job.ty)
      const slotIndex = this.allocateSlot(job.kind)
      const layerIndex = job.kind === 0 ? slotIndex : slotIndex + this.pairCapacity
      const device = this.device
      if (job.kind === 0) {
        if (!bitmap) return
        if (this.bc7Encoder) {
          device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.bc7Encoder.scratch }, [
            VIRTUAL_TILE_SIZE,
            VIRTUAL_TILE_SIZE,
          ])
          const encoder = device.createCommandEncoder({ label: 'ECV3 albedo BC7 upload' })
          generateMipmaps(device, this.bc7Encoder.scratch, {
            mipLevelCount: PLANET_TILE_MIPS,
            label: 'ECV3 albedo mipmaps',
            encoder,
          })
          this.bc7Encoder.encodeIntoArrayLayer(encoder, this.poolTexture, layerIndex)
          device.queue.submit([encoder.finish()])
        } else {
          device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: this.poolTexture, origin: [0, 0, layerIndex] },
            [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
          )
          generateMipmaps(device, this.poolTexture, {
            layers: [layerIndex],
            mipLevelCount: PLANET_TILE_MIPS,
            label: 'ECV3 albedo mipmaps',
          })
        }
      } else {
        if (!this.heightEncoder || (!decodedHeight10 && !bitmap)) return
        const { spanM } = unpackEcv3HeightBounds(entry.aux)
        const relief = PLANET_TERRAIN_SETTINGS[job.body].relief
        const radiusMeters = bodyRadius(job.body) * 1000
        const cols = PLANET_TILE_BASE_COLS << job.level
        const rows = PLANET_TILE_BASE_ROWS << job.level
        const destripe = false
        const encoder = device.createCommandEncoder({ label: 'ECV3 height->normal upload' })
        if (decodedHeight10) {
          this.heightEncoder.encodeFromHeight10(
            encoder,
            decodedHeight10,
            spanM,
            relief,
            radiusMeters,
            cols,
            rows,
            job.ty,
            destripe,
          )
        } else {
          this.heightEncoder.encodeFromBitmap(
            encoder,
            bitmap!,
            spanM,
            relief,
            radiusMeters,
            cols,
            rows,
            job.ty,
            destripe,
          )
        }
        if (this.bc7Encoder) {
          encoder.copyTextureToTexture(
            { texture: this.heightEncoder.outputTexture },
            { texture: this.bc7Encoder.scratch },
            [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
          )
          generateMipmaps(device, this.bc7Encoder.scratch, {
            mipLevelCount: PLANET_TILE_MIPS,
            label: 'ECV3 normal mipmaps',
            encoder,
          })
          this.bc7Encoder.encodeIntoArrayLayer(encoder, this.poolTexture, layerIndex)
        } else {
          encoder.copyTextureToTexture(
            { texture: this.heightEncoder.outputTexture },
            { texture: this.poolTexture, origin: [0, 0, layerIndex] },
            [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
          )
          generateMipmaps(device, this.poolTexture, {
            layers: [layerIndex],
            mipLevelCount: PLANET_TILE_MIPS,
            label: 'ECV3 normal mipmaps',
            encoder,
          })
        }
        device.queue.submit([encoder.finish()])
      }

      const slots = job.kind === 0 ? this.albedoSlots : this.normalSlots
      const resMap = job.kind === 0 ? state.residentAlbedo : state.residentNormal
      slots[slotIndex] = {
        body: job.body,
        kind: job.kind,
        key: job.key,
        level: job.level,
        tx: job.tx,
        ty: job.ty,
        lastUsed: this.frameCounter,
        readyAt: now,
      }
      resMap.set(job.key, slotIndex)
      this.slotFadeValues[job.kind === 0 ? slotIndex : PLANET_TILE_PAIRS_BC7 + slotIndex] = 0
      this.hasActiveFades = true
      if (job.kind === 0) state.albedoPageDirty = true
      else state.normalPageDirty = true
    } catch {
      // Transient upload error degrades gracefully to coarser ECV3 levels or 8K base textures.
    } finally {
      bitmap?.close()
      state.inFlight.delete(job.key)
    }
  }

  private scheduleFlush() {
    if (this.flushFrame || this.destroyed) return
    this.flushFrame = requestAnimationFrame(() => {
      this.flushFrame = 0
      if (this.destroyed) return
      if (this.pausedForFlight) {
        this.clearReadyUploads()
        return
      }
      if (this.readyUploads.length > 0) {
        if (this.readyUploads.length > MAX_UPLOADS_PER_FRAME) {
          this.readyUploads.sort((a, b) => a.job.priority - b.job.priority)
        }
        const count = Math.min(MAX_UPLOADS_PER_FRAME, this.readyUploads.length)
        const now = performance.now()
        for (let i = 0; i < count; i++) {
          this.uploadDecodedTile(this.readyUploads.shift()!, now)
        }
      }
      if (this.poolAllocated) {
        this.flushPageTables()
        this.notifyTileReady()
      }
      this.pumpQueue()
      if (this.readyUploads.length > 0) this.scheduleFlush()
    })
  }

  private flushPageTables() {
    if (!this.poolAllocated || this.destroyed) return
    const rowBytes = PLANET_TILE_PAGE_W * 4
    let wrote = false
    for (const body of ['moon', 'mars'] as const) {
      const state = this.bodies[body]
      if (!state.albedoPageDirty && !state.normalPageDirty) continue
      const maxPaintLevel = Math.min(LEVEL_COUNT - 1, state.viewMaxLevel + 1)
      if (state.albedoPageDirty && state.normalPageDirty) {
        state.albedoPageDirty = false
        state.normalPageDirty = false
        paintTilePageTable(
          state.pageAlbedoView,
          PLANET_TILE_PAGE_W,
          LEVEL_COUNT,
          this.albedoSlots,
          slot => slot.body === body && slot.level <= maxPaintLevel,
        )
        paintTilePageTable(
          state.pageNormalView,
          PLANET_TILE_PAGE_W,
          LEVEL_COUNT,
          this.normalSlots,
          slot => slot.body === body && slot.level <= maxPaintLevel,
        )
        this.device.queue.writeTexture({ texture: state.pageTexture }, state.pageData, { bytesPerRow: rowBytes }, [
          PLANET_TILE_PAGE_W,
          PLANET_TILE_PAGE_H * 2,
        ])
        wrote = true
      } else if (state.albedoPageDirty) {
        state.albedoPageDirty = false
        paintTilePageTable(
          state.pageAlbedoView,
          PLANET_TILE_PAGE_W,
          LEVEL_COUNT,
          this.albedoSlots,
          slot => slot.body === body && slot.level <= maxPaintLevel,
        )
        this.device.queue.writeTexture(
          { texture: state.pageTexture, origin: [0, 0, 0] },
          state.pageAlbedoView,
          { bytesPerRow: rowBytes },
          [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H],
        )
        wrote = true
      } else if (state.normalPageDirty) {
        state.normalPageDirty = false
        paintTilePageTable(
          state.pageNormalView,
          PLANET_TILE_PAGE_W,
          LEVEL_COUNT,
          this.normalSlots,
          slot => slot.body === body && slot.level <= maxPaintLevel,
        )
        this.device.queue.writeTexture(
          { texture: state.pageTexture, origin: [0, PLANET_TILE_PAGE_H, 0] },
          state.pageNormalView,
          { bytesPerRow: rowBytes },
          [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H],
        )
        wrote = true
      }
    }
    if (wrote) this._revision++
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    if (this.flushFrame) {
      cancelAnimationFrame(this.flushFrame)
      this.flushFrame = 0
    }
    if (this.zoomSettleTimer !== null) {
      clearTimeout(this.zoomSettleTimer)
      this.zoomSettleTimer = null
    }
    if (this.conePrefetchTimer !== null) {
      clearTimeout(this.conePrefetchTimer)
      this.conePrefetchTimer = null
    }
    this.clearReadyUploads()
    this.controller.abort()
    this.pendingQueue.length = 0
    this.tileListeners.clear()
    this.textureListeners.clear()
    this.bc7Encoder?.destroy()
    this.bc7Encoder = null
    this.heightEncoder?.destroy()
    this.heightEncoder = null
    this.height10Decoder.destroy()
    this.chunkCache.clear()
    this.poolTexture.destroy()
    this.slotFadeBuffer.destroy()
    for (const body of ['moon', 'mars'] as const) {
      const state = this.bodies[body]
      state.residentAlbedo.clear()
      state.residentNormal.clear()
      state.inFlight.clear()
      state.index.clear()
      state.pageTexture.destroy()
    }
  }
}
