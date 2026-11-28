import { Bc7TileEncoder, warmBc7Pipeline } from './bc7-gpu.ts'
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
} from './ecv3.ts'
import { cameraBasis, equalEarthInverse, FOV, type Camera, type Vec3 } from './geometry.ts'
import { generateMipmaps } from './gpu.ts'
import {
  createTileSampler,
  findLruSlot,
  intersectUnitSphere,
  paintTilePageTable,
  TILE_PAGE_EMPTY,
  tileKey,
  toPackHandle,
  VIRTUAL_TILE_INNER,
  VIRTUAL_TILE_SIZE,
} from './virtual-tiles.ts'

export { tileKey }

const ECV3_CDN_BASE = 'https://earth-connect-assets.pages.dev'
const EARTH_RADIUS_M = 6_371_008.8
const EARTH_TILE_PACKS = 512 // 16 latitude bands x 32 longitude blocks (11.25° x 11.25° squares)
// Exact 512-bit presence mask of all 384 Earth land packs on R2 (128 pure-ocean squares are omitted
// on R2 and skipped client-side so browsing oceans never fires 404 requests or console errors).
const EARTH_LAND_PACK_MASK = new Uint32Array([
  0x03ff7fc0, 0xffffffff, 0xffffffff, 0xffff8ff7, 0x3fffa7f0, 0x3fffc7e3, 0xdfffe7e7, 0xffdfcfc7, 0xff7f7f8f,
  0xfd3ebe0f, 0xfc1e1f5d, 0xfc46cf01, 0xc0593600, 0x5fb03e00, 0xffffffff, 0xffffffff,
])

function hasEarthLandPack(packId: number): boolean {
  if (packId < 0 || packId >= EARTH_TILE_PACKS) return false
  return ((EARTH_LAND_PACK_MASK[packId >>> 5] >>> (packId & 31)) & 1) === 1
}
const EARTH_TILE_SLOTS_BC7 = 160
const EARTH_TILE_SLOTS_RGBA = 64
export const EARTH_TILE_BASE_COLS = 32
export const EARTH_TILE_BASE_ROWS = 16
const LEVEL_COUNT = 6 // L0 (32x16) .. L5 (1024x512, ~61 m global land)
export const EARTH_TILE_PAGE_W = EARTH_TILE_BASE_COLS << (LEVEL_COUNT - 1) // 1024
export const EARTH_TILE_PAGE_H = EARTH_TILE_BASE_ROWS << (LEVEL_COUNT - 1) // 512
const EARTH_TILE_MIPS = 5 // 512 -> 256 -> 128 -> 64 -> 32
const MAX_CONCURRENT_ECV3_DECODES = 24
const MAX_CONCURRENT_NETWORK_JOBS = 14
const MAX_UPLOADS_PER_FRAME = 6
const MAX_READY_UPLOADS = 16
const TILE_FADE_MS = 180
const ZOOM_SETTLE_DELAY_MS = 90
const CONE_PREFETCH_IDLE_MS = 220
const EARTH_DEM_RELIEF = 2.6

type TileSlot = {
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
  decoded10?: DecodedHeight10
}

type TileJob = {
  kind: 0 | 1
  key: number
  level: number
  tx: number
  ty: number
  priority: number
}

export type EarthTilePackLoader = (packId: number, signal: AbortSignal) => Promise<Blob | string | null>

export function earthTilePath(packId: number) {
  return `ecv3/earth/earth-ecv3-p${String(packId).padStart(3, '0')}.ecv3`
}

export function ecv3EarthPackUrl(packId: number) {
  return `${ECV3_CDN_BASE}/ecv3/earth/earth-ecv3-p${String(packId).padStart(3, '0')}.ecv3?v=e1h`
}

function packIdForTile(level: number, tx: number, ty: number): number {
  const band = ty >>> level
  const lonBlk = tx >>> level
  return band * 32 + lonBlk
}

export class EarthTileCache {
  private readonly controller = new AbortController()
  private readonly useBc7: boolean
  private readonly slotCapacity: number
  private readonly packs: (Ecv3Pack | null)[] = Array(EARTH_TILE_PACKS).fill(null)
  private readonly loadingPacks = new Set<number>()
  private readonly failedPacks = new Set<number>()
  private bc7SrgbEncoder: Bc7TileEncoder | null = null
  private bc7LinearEncoder: Bc7TileEncoder | null = null
  private heightEncoder: HeightToNormalEncoder | null = null
  private readonly height10Decoder = new Av1Height10Decoder(VIRTUAL_TILE_SIZE, 4)
  private readonly chunkCache = new Ecv3ChunkCache()
  private readonly index = new Map<number, Ecv3TileEntry>()
  private readonly residentAlbedo = new Map<number, number>()
  private readonly residentNormal = new Map<number, number>()
  private readonly inFlight = new Set<number>()
  private readonly albedoSlots: (TileSlot | null)[]
  private readonly normalSlots: (TileSlot | null)[]
  // Stacked 1024 x 1024 rgba8uint page table:
  //   top half    [0..512)    = Albedo (kind 0, Sentinel-2 RGB + 30 m Copernicus land/water alpha)
  //   bottom half [512..1024) = Normal (kind 1, 60 m Copernicus 10-bit DEM tangent-space normals)
  private readonly pageData: Uint8Array<ArrayBuffer>
  private readonly pageAlbedoView: Uint8Array<ArrayBuffer>
  private readonly pageNormalView: Uint8Array<ArrayBuffer>
  private readonly slotFadeValues = new Float32Array(EARTH_TILE_SLOTS_BC7 * 2)
  private hasActiveFades = false
  private albedoPageDirty = false
  private normalPageDirty = false
  private flushFrame = 0
  private zoomSettleTimer: ReturnType<typeof setTimeout> | null = null
  private zoomSettleUntil = 0
  private conePrefetchTimer: ReturnType<typeof setTimeout> | null = null
  private coneSettled = false
  private lastConeCenterKey = ''
  private activeDemand = new Set<number>()
  private pendingQueue: TileJob[] = []
  private readyUploads: ReadyUpload[] = []
  private activeDecodes = 0
  private activeNetworkJobs = 0
  private frameCounter = 0
  private _revision = 0
  private viewMaxLevel = LEVEL_COUNT - 1
  private lastEye: [number, number, number] | null = null
  private lastMpp = Infinity
  private pausedForFlight = false
  private destroyed = false
  private activated = false
  private packsAttached = false
  private readonly tileListeners = new Set<() => void>()
  private readonly textureListeners = new Set<() => void>()

  private readonly device: GPUDevice
  pageTexture: GPUTexture
  poolTexture: GPUTexture
  normalPoolTexture: GPUTexture
  readonly slotFadeBuffer: GPUBuffer
  readonly sampler: GPUSampler

  constructor(device: GPUDevice) {
    this.device = device
    this.useBc7 = device.features.has('texture-compression-bc')
    if (this.useBc7) warmBc7Pipeline(device)
    warmHeightPipelines(device)
    this.slotCapacity = this.useBc7 ? EARTH_TILE_SLOTS_BC7 : EARTH_TILE_SLOTS_RGBA
    this.albedoSlots = Array(this.slotCapacity).fill(null)
    this.normalSlots = Array(this.slotCapacity).fill(null)
    this.slotFadeValues.fill(1)
    this.slotFadeBuffer = device.createBuffer({
      label: 'Earth ECV3 per-slot fade weights',
      size: this.slotFadeValues.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.slotFadeBuffer, 0, this.slotFadeValues)
    this.sampler = createTileSampler(device, 'Earth ECV3 virtual tile sampler')
    const halfBytes = EARTH_TILE_PAGE_W * EARTH_TILE_PAGE_H * 4
    this.pageData = new Uint8Array(halfBytes * 2)
    this.pageData.fill(TILE_PAGE_EMPTY)
    this.pageAlbedoView = this.pageData.subarray(0, halfBytes)
    this.pageNormalView = this.pageData.subarray(halfBytes, halfBytes * 2)

    // Tiny placeholders until the user zooms in past the 8K Blue Marble detail threshold.
    this.pageTexture = device.createTexture({
      label: 'Earth ECV3 tile page placeholder',
      size: [1, 2],
      format: 'rgba8uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture(
      { texture: this.pageTexture },
      new Uint8Array([255, 0, 255, 0, 255, 0, 255, 0]),
      { bytesPerRow: 4 },
      [1, 2],
    )
    this.poolTexture = device.createTexture({
      label: 'Earth ECV3 albedo pool placeholder',
      size: [1, 1, 1],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture: this.poolTexture }, new Uint8Array([18, 30, 42, 255]), {}, [1, 1, 1])
    this.normalPoolTexture = device.createTexture({
      label: 'Earth ECV3 normal pool placeholder',
      size: [1, 1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture: this.normalPoolTexture }, new Uint8Array([128, 128, 255, 255]), {}, [1, 1, 1])
  }

  get ready() {
    return this.activated
  }
  get attached() {
    return this.packsAttached
  }
  get residentCount() {
    return this.residentAlbedo.size + this.residentNormal.size
  }
  get indexedCount() {
    return this.index.size
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

  /** Pause queued and in-flight tile decodes/uploads while a camera flight is running so flights stay stutter-free. */
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
    this.lastEye = null
  }

  private clearReadyUploads() {
    for (const item of this.readyUploads) {
      item.bitmap?.close()
      this.inFlight.delete(item.job.key)
    }
    this.readyUploads.length = 0
  }

  private pruneStaleReadyUploads() {
    if (this.readyUploads.length === 0) return
    let write = 0
    for (let i = 0; i < this.readyUploads.length; i++) {
      const item = this.readyUploads[i]
      if (this.activeDemand.has(item.job.key)) {
        this.readyUploads[write++] = item
      } else {
        item.bitmap?.close()
        this.inFlight.delete(item.job.key)
      }
    }
    this.readyUploads.length = write
  }

  /** Advance per-slot smooth fade-in weights; returns true while any visible slot is still crossfading. */
  updateFades(now: number): boolean {
    if (!this.hasActiveFades || this.destroyed) return false
    let anyFading = false
    for (let i = 0; i < this.slotCapacity; i++) {
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
        this.slotFadeValues[EARTH_TILE_SLOTS_BC7 + i] = r * r * (3 - 2 * r)
        if (r < 1) anyFading = true
      } else {
        this.slotFadeValues[EARTH_TILE_SLOTS_BC7 + i] = 1
      }
    }
    this.device.queue.writeBuffer(this.slotFadeBuffer, 0, this.slotFadeValues)
    this.hasActiveFades = anyFading
    this._revision++
    return anyFading
  }

  private ensurePackLoaded(packId: number) {
    if (
      this.destroyed ||
      !hasEarthLandPack(packId) ||
      this.packs[packId] ||
      this.loadingPacks.has(packId) ||
      this.failedPacks.has(packId)
    ) {
      return
    }
    this.loadingPacks.add(packId)
    void this.loadPackIndex(packId, this.controller.signal)
  }

  private async loadPackIndex(packId: number, signal: AbortSignal) {
    try {
      const handle = toPackHandle(ecv3EarthPackUrl(packId))
      const loaded = await loadEcv3PackIndex(handle, packId, signal)
      if (!loaded || signal.aborted || this.destroyed) {
        this.loadingPacks.delete(packId)
        if (!signal.aborted && !this.destroyed) this.failedPacks.add(packId)
        return
      }
      this.packs[packId] = loaded.pack
      if (loaded.chunk0) this.chunkCache.seedChunk(`earth:${packId}`, 0, loaded.chunk0)
      this.loadingPacks.delete(packId)
      for (const [key, entry] of loaded.entries) this.index.set(key, entry)
      if (this.index.size > 0) this.notifyTileReady()
    } catch (error) {
      this.loadingPacks.delete(packId)
      // Only permanently mark 404 (pure-ocean) packs; transient network blips can retry on next view update.
      if (!signal.aborted && !this.destroyed && error instanceof Error && error.message.includes('(404)')) {
        this.failedPacks.add(packId)
      }
    }
  }

  private activateGpuPool() {
    if (this.activated || this.destroyed) return
    this.activated = true
    const device = this.device
    const oldPage = this.pageTexture
    const oldPool = this.poolTexture
    const oldNormalPool = this.normalPoolTexture
    this.pageTexture = device.createTexture({
      label: 'Earth ECV3 virtual tile page table',
      size: [EARTH_TILE_PAGE_W, EARTH_TILE_PAGE_H * 2],
      format: 'rgba8uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.pageData.fill(TILE_PAGE_EMPTY)
    device.queue.writeTexture({ texture: this.pageTexture }, this.pageData, { bytesPerRow: EARTH_TILE_PAGE_W * 4 }, [
      EARTH_TILE_PAGE_W,
      EARTH_TILE_PAGE_H * 2,
    ])
    this.poolTexture = device.createTexture({
      label: this.useBc7
        ? 'Earth ECV3 61m albedo virtual tile cache (BC7)'
        : 'Earth ECV3 61m albedo virtual tile cache',
      size: [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE, this.slotCapacity],
      mipLevelCount: EARTH_TILE_MIPS,
      format: this.useBc7 ? 'bc7-rgba-unorm-srgb' : 'rgba8unorm-srgb',
      usage: this.useBc7
        ? GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.normalPoolTexture = device.createTexture({
      label: this.useBc7
        ? 'Earth ECV3 60m DEM normal virtual tile cache (BC7)'
        : 'Earth ECV3 60m DEM normal virtual tile cache',
      size: [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE, this.slotCapacity],
      mipLevelCount: EARTH_TILE_MIPS,
      format: this.useBc7 ? 'bc7-rgba-unorm' : 'rgba8unorm',
      usage: this.useBc7
        ? GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    if (this.useBc7) {
      this.bc7SrgbEncoder = new Bc7TileEncoder(device, VIRTUAL_TILE_SIZE, EARTH_TILE_MIPS, 'rgba8unorm-srgb')
      this.bc7LinearEncoder = new Bc7TileEncoder(device, VIRTUAL_TILE_SIZE, EARTH_TILE_MIPS, 'rgba8unorm')
    }
    this.heightEncoder = new HeightToNormalEncoder(device, VIRTUAL_TILE_SIZE)
    this.notifyTexturesChanged()
    void device.queue
      .onSubmittedWorkDone()
      .then(() => {
        oldPage.destroy()
        oldPool.destroy()
        oldNormalPool.destroy()
      })
      .catch(() => undefined)
  }

  /**
   * Prefetch the entire center zoom cone (`startLevel .. L5`) into the 32 MiB RAM chunk cache
   * (and browser immutable cache) once the user stops panning/rotating. Uses zero GPU pool slots
   * and strictly low-priority background streams so zooming all the way in hits RAM (~10 ms).
   */
  private prefetchZoomCone(lon: number, lat: number, startLevel: number) {
    const u = (((lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
    const v = Math.max(0, Math.min(0.999999, 0.5 - lat / Math.PI))
    const maxLevel = LEVEL_COUNT - 1
    const centerKey = `${Math.floor(u * (EARTH_TILE_BASE_COLS << maxLevel))}:${Math.floor(v * (EARTH_TILE_BASE_ROWS << maxLevel))}:${startLevel}`
    const packSet = new Set<number>()
    let anyPackPending = false

    for (let level = Math.max(0, startLevel); level <= maxLevel; level++) {
      const cols = EARTH_TILE_BASE_COLS << level
      const rows = EARTH_TILE_BASE_ROWS << level
      const isDeepest = level === maxLevel
      // 2x2 centered around exact fractional center for intermediate levels (handles tile-border centers),
      // 3x3 at L5 so a full deep-zoom screen is already warm in RAM.
      const x0 = isDeepest ? Math.floor(u * cols) - 1 : Math.floor(u * cols - 0.5)
      const y0 = isDeepest ? Math.floor(v * rows) - 1 : Math.floor(v * rows - 0.5)
      const span = isDeepest ? 3 : 2
      for (let dy = 0; dy < span; dy++) {
        const ty = y0 + dy
        if (ty < 0 || ty >= rows) continue
        for (let dx = 0; dx < span; dx++) {
          const tx = (x0 + dx + cols) % cols
          const pid = packIdForTile(level, tx, ty)
          if (!hasEarthLandPack(pid)) continue
          if (!this.packs[pid]) {
            if (!packSet.has(pid)) {
              packSet.add(pid)
              this.ensurePackLoaded(pid)
            }
            anyPackPending = true
            continue
          }
          if (this.lastConeCenterKey === centerKey) continue
          const pack = this.packs[pid]!
          const packKey = `earth:${pid}`
          const prio = ECV3_PREFETCH_PRIORITY + (level - startLevel) * 10 + (dx === 1 && dy === 1 ? 0 : 2)
          for (const kind of [0, 1] as const) {
            let lv = level
            let cx = tx
            let cy = ty
            let key = ecv3TileKey(kind, lv, cx, cy)
            while (lv > 0 && !this.index.has(key)) {
              lv--
              cx >>>= 1
              cy >>>= 1
              key = ecv3TileKey(kind, lv, cx, cy)
            }
            const entry = this.index.get(key)
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
   * Determine which Earth ECV3 tiles (61 m Sentinel-2 RGB+Alpha + 60 m Copernicus 10-bit DEM)
   * intersect the current camera (and optional loupe), lazily load visible 11.25°×11.25° packs,
   * and stream tiles with orbital prewarming + settled full-depth zoom-cone prefetching.
   */
  updateView(
    camera: Camera,
    morph: number,
    metresPerPixel: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    if (!this.packsAttached || this.destroyed) return
    this.pausedForFlight = false
    const l0TexelM = (2 * Math.PI * EARTH_RADIUS_M) / (EARTH_TILE_BASE_COLS * VIRTUAL_TILE_INNER) // ~2461 m/px
    const entryThreshold = l0TexelM * 2.75 // ~6770 m/px (matches shader tileMip 1.45 fade-in)
    const prewarmThreshold = l0TexelM * 3.8 // ~9350 m/px (warms L0 packs before fade-in)

    const { right, up, back, eye } = cameraBasis(camera)
    let panSpeed = 0
    let velX = 0
    let velY = 0
    let velZ = 0
    if (this.lastEye) {
      velX = eye[0] - this.lastEye[0]
      velY = eye[1] - this.lastEye[1]
      velZ = eye[2] - this.lastEye[2]
      panSpeed = Math.hypot(velX, velY, velZ)
    }
    const zoomRatio = Number.isFinite(this.lastMpp) && metresPerPixel > 0 ? this.lastMpp / metresPerPixel : 1
    const cameraMoved = this.lastEye === null || panSpeed > 1e-4 || Math.abs(zoomRatio - 1) > 0.015
    this.scheduleConePrefetch(cameraMoved)

    const loupeMpp = loupe ? metresPerPixel / Math.max(1, loupe.zoom) : Infinity
    const minMpp = Math.min(metresPerPixel, loupeMpp)
    if (minMpp > prewarmThreshold) {
      this.lastEye = [eye[0], eye[1], eye[2]]
      this.lastMpp = metresPerPixel
      this.pendingQueue.length = 0
      this.activeDemand.clear()
      this.pruneStaleReadyUploads()
      // Even from full-globe orbital altitude, once the user stops rotating for 220 ms, warm the
      // center L0..L5 zoom cone into RAM so zooming straight in has near-zero network wait.
      if (this.coneSettled) {
        const centerHit = intersectEarth(eye, right, up, back, 0, 0, morph)
        if (centerHit) this.prefetchZoomCone(centerHit.lon, centerHit.lat, 0)
      }
      return
    }

    const tanHalf = Math.tan(FOV / 2)
    const frame = ++this.frameCounter
    const desiredAlbedo = new Map<number, TileJob>()
    const desiredNormal = new Map<number, TileJob>()
    let maxRequestedLevel = 0

    const now = performance.now()
    const isZoomingIn = zoomRatio > 1.02
    const isFastZooming = zoomRatio > 1.08 || zoomRatio < 0.92
    const isSettled = !cameraMoved
    this.lastEye = [eye[0], eye[1], eye[2]]
    this.lastMpp = metresPerPixel

    if (isFastZooming) {
      this.zoomSettleUntil = now + ZOOM_SETTLE_DELAY_MS
      if (this.zoomSettleTimer !== null) clearTimeout(this.zoomSettleTimer)
      this.zoomSettleTimer = setTimeout(() => {
        this.zoomSettleTimer = null
        if (!this.destroyed && !this.pausedForFlight) this.notifyTileReady()
      }, ZOOM_SETTLE_DELAY_MS)
    }
    const zoomSuppressing = isFastZooming || now < this.zoomSettleUntil

    let onscreenTargetMissing = 0

    const enqueueKind = (
      kind: 0 | 1,
      level: number,
      tx: number,
      ty: number,
      prio: number,
      isOnscreenTarget: boolean,
    ) => {
      const packId = packIdForTile(level, tx, ty)
      if (!this.packs[packId]) {
        if (!hasEarthLandPack(packId) || this.failedPacks.has(packId)) return
        this.ensurePackLoaded(packId)
        if (isOnscreenTarget) onscreenTargetMissing++
        return
      }
      // Walk up from requested level to the finest non-pruned ancestor so flat/pruned regions
      // always pin and display their finest available tile instead of getting evicted by LRU.
      let lv = level
      let cx = tx
      let cy = ty
      let key = ecv3TileKey(kind, lv, cx, cy)
      while (lv > 0 && !this.index.has(key)) {
        lv--
        cx >>>= 1
        cy >>>= 1
        key = ecv3TileKey(kind, lv, cx, cy)
      }
      if (!this.index.has(key)) return
      const resMap = kind === 0 ? this.residentAlbedo : this.residentNormal
      if (!resMap.has(key) && isOnscreenTarget) onscreenTargetMissing++
      const desired = kind === 0 ? desiredAlbedo : desiredNormal
      const prev = desired.get(key)
      if (!prev || prio < prev.priority) {
        desired.set(key, { kind, key, level: lv, tx: cx, ty: cy, priority: prio })
      }
    }

    const requestPoint = (
      lon: number,
      lat: number,
      mpp: number,
      distSq: number,
      halo: number,
      prioBias = 0,
      maxLevelCap = LEVEL_COUNT - 1,
      includeParent = false,
      onscreen = false,
    ) => {
      if (mpp > prewarmThreshold) return
      const u = (((lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
      const v = Math.max(0, Math.min(0.999999, 0.5 - lat / Math.PI))
      const rawLevel = mpp > entryThreshold ? 0 : Math.ceil(Math.log2(l0TexelM / Math.max(mpp, 1)) + 0.55)
      const targetLevel = Math.max(0, Math.min(maxLevelCap, rawLevel))
      if (prioBias === 0 && targetLevel > maxRequestedLevel) maxRequestedLevel = targetLevel

      // Primary pass requests L0 anchor + targetLevel directly; parent fallback is deferred until
      // all on-screen target tiles are resident so 100% of HTTP/2 slots serve visible detail first.
      const levelSpecs: [number, number][] = [[0, -10]]
      if (targetLevel >= 1) levelSpecs.push([targetLevel, 0])
      if (includeParent && targetLevel >= 2) levelSpecs.push([targetLevel - 1, 6])

      for (const [level, levelBias] of levelSpecs) {
        const cols = EARTH_TILE_BASE_COLS << level
        const rows = EARTH_TILE_BASE_ROWS << level
        const cx = Math.floor(u * cols) % cols
        const cy = Math.max(0, Math.min(rows - 1, Math.floor(v * rows)))
        const radius = level === targetLevel ? halo : 0
        const isPrimary = onscreen && prioBias === 0 && (level === 0 || level === targetLevel)
        for (let dy = -radius; dy <= radius; dy++) {
          const ty = cy + dy
          if (ty < 0 || ty >= rows) continue
          for (let dx = -radius; dx <= radius; dx++) {
            const tx = (cx + dx + cols) % cols
            const basePrio = prioBias + levelBias + distSq + (dx * dx + dy * dy) * 0.35
            enqueueKind(0, level, tx, ty, basePrio, isPrimary)
            enqueueKind(1, level, tx, ty, basePrio + 0.4, isPrimary)
          }
        }
      }
    }

    if (metresPerPixel <= prewarmThreshold) {
      const prewarmOnly = metresPerPixel > entryThreshold
      const gridN = prewarmOnly ? 5 : 15
      const extent = prewarmOnly ? 0.65 : 1.24
      const offscreenRays: { lon: number; lat: number; effectiveMpp: number; prio: number }[] = []
      const onscreenRays: { lon: number; lat: number; effectiveMpp: number; prio: number }[] = []
      for (let gy = 0; gy < gridN; gy++) {
        const ndcY = ((gy / (gridN - 1)) * 2 - 1) * extent
        for (let gx = 0; gx < gridN; gx++) {
          const ndcX = ((gx / (gridN - 1)) * 2 - 1) * extent
          const hit = intersectEarth(eye, right, up, back, ndcX * tanHalf * camera.aspect, ndcY * tanHalf, morph)
          if (!hit) continue
          const effectiveMpp = metresPerPixel / Math.max(0.45, hit.cosIncidence)
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
            requestPoint(
              hit.lon,
              hit.lat,
              effectiveMpp,
              ndcX * ndcX + ndcY * ndcY,
              0,
              prewarmOnly ? 80 : 0,
              prewarmOnly ? 0 : LEVEL_COUNT - 1,
              false,
              true,
            )
          }
        }
      }

      // Only once every visible on-screen target tile is resident do we warm parent fallbacks,
      // the offscreen 1.24x margin ring, and predictive pan/zoom lookahead.
      if (!prewarmOnly && !zoomSuppressing && onscreenTargetMissing === 0) {
        for (const r of onscreenRays) {
          requestPoint(r.lon, r.lat, r.effectiveMpp, r.prio, 0, 20, LEVEL_COUNT - 1, true, false)
        }
        for (const r of offscreenRays) {
          requestPoint(r.lon, r.lat, r.effectiveMpp, r.prio, 0, 40, LEVEL_COUNT - 1, false, false)
        }
        if (morph >= 0.5 && panSpeed > 0.0004) {
          const lookAhead = Math.min(18, 0.35 / Math.max(panSpeed, 0.001))
          const predEye: Vec3 = [eye[0] + velX * lookAhead, eye[1] + velY * lookAhead, eye[2] + velZ * lookAhead]
          const hit = intersectUnitSphere(predEye, right, up, back, 0, 0)
          if (hit) requestPoint(hit.lon, hit.lat, metresPerPixel, 0.25, 1, 90, LEVEL_COUNT - 1, false, false)
        }
        if (isZoomingIn || isSettled) {
          const centerHit = intersectEarth(eye, right, up, back, 0, 0, morph)
          if (centerHit) {
            const deeperMpp = metresPerPixel * 0.48
            requestPoint(
              centerHit.lon,
              centerHit.lat,
              deeperMpp,
              0,
              0,
              isZoomingIn ? 75 : 115,
              LEVEL_COUNT - 1,
              false,
              false,
            )
            if (this.coneSettled && maxRequestedLevel < LEVEL_COUNT - 1) {
              this.prefetchZoomCone(centerHit.lon, centerHit.lat, maxRequestedLevel + 1)
            }
          }
        }
      } else if (prewarmOnly && this.coneSettled && onscreenTargetMissing === 0) {
        const centerHit = intersectEarth(eye, right, up, back, 0, 0, morph)
        if (centerHit) this.prefetchZoomCone(centerHit.lon, centerHit.lat, 1)
      }
    }

    if (loupe && loupeMpp <= entryThreshold) {
      requestPoint(loupe.lon, loupe.lat, loupeMpp, -2, 1, 0, LEVEL_COUNT - 1, false, true)
    }

    if (this.viewMaxLevel !== maxRequestedLevel) {
      if (maxRequestedLevel > this.viewMaxLevel) this.chunkCache.cancelQueued()
      this.viewMaxLevel = maxRequestedLevel
      this.albedoPageDirty = true
      this.normalPageDirty = true
    }
    if (this.albedoPageDirty || this.normalPageDirty) this.flushPageTable()

    // Admit a stable, pool-capacity-bounded set per kind by screen priority so wide/4K
    // viewports never evict and reload visible tiles in an endless stationary loop.
    const nextDemand = new Set<number>()
    const nextQueue: TileJob[] = []
    for (const [kind, desired] of [[0, desiredAlbedo] as const, [1, desiredNormal] as const]) {
      const resMap = kind === 0 ? this.residentAlbedo : this.residentNormal
      const slots = kind === 0 ? this.albedoSlots : this.normalSlots
      const ranked = Array.from(desired.values()).sort((a, b) => a.priority - b.priority)
      const admitted = ranked.length > this.slotCapacity ? ranked.slice(0, this.slotCapacity) : ranked
      for (const job of admitted) {
        nextDemand.add(job.key)
        const slotIndex = resMap.get(job.key)
        if (slotIndex !== undefined) {
          const slot = slots[slotIndex]
          if (slot) slot.lastUsed = frame
          continue
        }
        if (this.inFlight.has(job.key)) continue
        // During rapid scroll-zooming (or within the 90 ms settle window), skip network misses
        // for transient intermediate levels (L1..L4) unless already in RAM cache.
        if (zoomSuppressing && job.level > 0 && job.level < LEVEL_COUNT - 1) {
          const entry = this.index.get(job.key)
          if (!entry || !this.chunkCache.hasSlice(`earth:${entry.packId}`, entry.offset, entry.length)) continue
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
      const resMap = job.kind === 0 ? this.residentAlbedo : this.residentNormal
      if (resMap.has(job.key) || this.inFlight.has(job.key)) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const entry = this.index.get(job.key)
      if (!entry) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const pack = this.packs[entry.packId]
      if (!pack) {
        this.pendingQueue.splice(i, 1)
        continue
      }
      const isRamHit = this.chunkCache.hasSlice(`earth:${pack.packId}`, entry.offset, entry.length)
      if (!isRamHit && this.activeNetworkJobs >= MAX_CONCURRENT_NETWORK_JOBS) {
        i++
        continue
      }
      this.pendingQueue.splice(i, 1)
      this.inFlight.add(job.key)
      this.activeDecodes++
      if (!isRamHit) this.activeNetworkJobs++
      void this.decodeAndUpload(job, entry, pack, !isRamHit)
    }
  }

  private allocateSlot(kind: 0 | 1): number {
    const slots = kind === 0 ? this.albedoSlots : this.normalSlots
    const resMap = kind === 0 ? this.residentAlbedo : this.residentNormal
    const slotIndex = findLruSlot(slots)
    const evicted = slots[slotIndex]
    if (evicted) {
      resMap.delete(evicted.key)
      slots[slotIndex] = null
      if (kind === 0) this.albedoPageDirty = true
      else this.normalPageDirty = true
    }
    return slotIndex
  }

  private async decodeAndUpload(job: TileJob, entry: Ecv3TileEntry, pack: Ecv3Pack, holdsNetworkSlot: boolean) {
    const { signal } = this.controller
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
      const obu = await this.chunkCache.readSlice(
        pack.handle,
        `earth:${pack.packId}`,
        entry.offset,
        entry.length,
        signal,
        job.priority,
      )
      releaseNetworkSlot()
      if (signal.aborted || this.destroyed || this.pausedForFlight) return

      if (job.kind === 0) {
        const blob = reassembleEcv3AvifBlob(tpl, obu)
        bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
        if (signal.aborted || this.destroyed || this.pausedForFlight) return
        this.readyUploads.push({ job, entry, bitmap })
        bitmap = undefined
        transferredToUpload = true
      } else {
        // Kind 1: 10-bit Copernicus DEM heightfield -> GPU-derived tangent-space normal map
        const decoded10 = await this.height10Decoder.decode(obu, signal)
        if (!decoded10) {
          const blob = reassembleEcv3AvifBlob(tpl, obu)
          bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
        }
        if (signal.aborted || this.destroyed || this.pausedForFlight) return
        this.readyUploads.push({ job, entry, bitmap, decoded10: decoded10 ?? undefined })
        bitmap = undefined
        transferredToUpload = true
      }

      this.scheduleFlush()
    } catch {
      // Transient aborts/decode errors degrade gracefully to parent/8K texture & 4K DEM.
    } finally {
      bitmap?.close()
      if (networkSlotHeld) {
        networkSlotHeld = false
        this.activeNetworkJobs--
      }
      if (!transferredToUpload) this.inFlight.delete(job.key)
      this.activeDecodes--
      this.pumpQueue()
    }
  }

  private touchParentSlot(kind: 0 | 1, level: number, tx: number, ty: number) {
    const resMap = kind === 0 ? this.residentAlbedo : this.residentNormal
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
    const { job, entry, bitmap, decoded10 } = item
    const device = this.device
    try {
      if (!this.activeDemand.has(job.key)) return
      this.activateGpuPool()
      if (job.kind === 0) {
        if (!bitmap) return
        if (job.level > 0) this.touchParentSlot(0, job.level, job.tx, job.ty)
        const slotIndex = this.allocateSlot(0)
        if (this.bc7SrgbEncoder) {
          device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.bc7SrgbEncoder.scratch }, [
            VIRTUAL_TILE_SIZE,
            VIRTUAL_TILE_SIZE,
          ])
          const encoder = device.createCommandEncoder({ label: 'Earth ECV3 albedo BC7 upload' })
          generateMipmaps(device, this.bc7SrgbEncoder.scratch, {
            mipLevelCount: EARTH_TILE_MIPS,
            label: 'Earth ECV3 albedo mipmaps',
            encoder,
          })
          this.bc7SrgbEncoder.encodeIntoArrayLayer(encoder, this.poolTexture, slotIndex)
          device.queue.submit([encoder.finish()])
        } else {
          device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: this.poolTexture, origin: [0, 0, slotIndex] },
            [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
          )
          generateMipmaps(device, this.poolTexture, {
            layers: [slotIndex],
            mipLevelCount: EARTH_TILE_MIPS,
            label: 'Earth ECV3 albedo mipmaps',
          })
        }
        this.albedoSlots[slotIndex] = {
          kind: 0,
          key: job.key,
          level: job.level,
          tx: job.tx,
          ty: job.ty,
          lastUsed: this.frameCounter,
          readyAt: now,
        }
        this.residentAlbedo.set(job.key, slotIndex)
        this.slotFadeValues[slotIndex] = 0
        this.hasActiveFades = true
        this.albedoPageDirty = true
      } else {
        if (!this.heightEncoder || (!decoded10 && !bitmap)) return
        if (job.level > 0) this.touchParentSlot(1, job.level, job.tx, job.ty)
        const { spanM } = unpackEcv3HeightBounds(entry.aux)
        const cols = EARTH_TILE_BASE_COLS << job.level
        const rows = EARTH_TILE_BASE_ROWS << job.level
        const slotIndex = this.allocateSlot(1)
        const encoder = device.createCommandEncoder({ label: 'Earth ECV3 DEM->Normal upload' })
        if (decoded10) {
          this.heightEncoder.encodeFromHeight10(
            encoder,
            decoded10,
            spanM,
            EARTH_DEM_RELIEF,
            EARTH_RADIUS_M,
            cols,
            rows,
            job.ty,
            false,
          )
        } else {
          this.heightEncoder.encodeFromBitmap(
            encoder,
            bitmap!,
            spanM,
            EARTH_DEM_RELIEF,
            EARTH_RADIUS_M,
            cols,
            rows,
            job.ty,
            false,
          )
        }
        const normalSrc = this.heightEncoder.outputTexture
        if (this.bc7LinearEncoder) {
          encoder.copyTextureToTexture({ texture: normalSrc }, { texture: this.bc7LinearEncoder.scratch }, [
            VIRTUAL_TILE_SIZE,
            VIRTUAL_TILE_SIZE,
          ])
          generateMipmaps(device, this.bc7LinearEncoder.scratch, {
            mipLevelCount: EARTH_TILE_MIPS,
            label: 'Earth ECV3 normal mipmaps',
            encoder,
          })
          this.bc7LinearEncoder.encodeIntoArrayLayer(encoder, this.normalPoolTexture, slotIndex)
          device.queue.submit([encoder.finish()])
        } else {
          encoder.copyTextureToTexture(
            { texture: normalSrc },
            { texture: this.normalPoolTexture, origin: [0, 0, slotIndex] },
            [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
          )
          generateMipmaps(device, this.normalPoolTexture, {
            layers: [slotIndex],
            mipLevelCount: EARTH_TILE_MIPS,
            label: 'Earth ECV3 normal mipmaps',
            encoder,
          })
          device.queue.submit([encoder.finish()])
        }
        this.normalSlots[slotIndex] = {
          kind: 1,
          key: job.key,
          level: job.level,
          tx: job.tx,
          ty: job.ty,
          lastUsed: this.frameCounter,
          readyAt: now,
        }
        this.residentNormal.set(job.key, slotIndex)
        this.slotFadeValues[EARTH_TILE_SLOTS_BC7 + slotIndex] = 0
        this.hasActiveFades = true
        this.normalPageDirty = true
      }
    } catch {
      // Transient upload error degrades gracefully to parent/8K texture.
    } finally {
      bitmap?.close()
      this.inFlight.delete(job.key)
    }
  }

  /** Pace GPU tile uploads (<=6/frame) and coalesce dirty page-table halves to once per rAF. */
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
      if (this.albedoPageDirty || this.normalPageDirty) {
        this.flushPageTable()
        this.notifyTileReady()
      }
      this.pumpQueue()
      if (this.readyUploads.length > 0) this.scheduleFlush()
    })
  }

  private flushPageTable() {
    if (!this.activated || this.destroyed) return
    const cap = this.viewMaxLevel
    const rowBytes = EARTH_TILE_PAGE_W * 4
    let wrote = false
    if (this.albedoPageDirty && this.normalPageDirty) {
      this.albedoPageDirty = false
      this.normalPageDirty = false
      paintTilePageTable(this.pageAlbedoView, EARTH_TILE_PAGE_W, LEVEL_COUNT, this.albedoSlots, s => s.level <= cap)
      paintTilePageTable(this.pageNormalView, EARTH_TILE_PAGE_W, LEVEL_COUNT, this.normalSlots, s => s.level <= cap)
      this.device.queue.writeTexture({ texture: this.pageTexture }, this.pageData, { bytesPerRow: rowBytes }, [
        EARTH_TILE_PAGE_W,
        EARTH_TILE_PAGE_H * 2,
      ])
      wrote = true
    } else if (this.albedoPageDirty) {
      this.albedoPageDirty = false
      paintTilePageTable(this.pageAlbedoView, EARTH_TILE_PAGE_W, LEVEL_COUNT, this.albedoSlots, s => s.level <= cap)
      this.device.queue.writeTexture(
        { texture: this.pageTexture, origin: [0, 0, 0] },
        this.pageAlbedoView,
        { bytesPerRow: rowBytes },
        [EARTH_TILE_PAGE_W, EARTH_TILE_PAGE_H],
      )
      wrote = true
    } else if (this.normalPageDirty) {
      this.normalPageDirty = false
      paintTilePageTable(this.pageNormalView, EARTH_TILE_PAGE_W, LEVEL_COUNT, this.normalSlots, s => s.level <= cap)
      this.device.queue.writeTexture(
        { texture: this.pageTexture, origin: [0, EARTH_TILE_PAGE_H, 0] },
        this.pageNormalView,
        { bytesPerRow: rowBytes },
        [EARTH_TILE_PAGE_W, EARTH_TILE_PAGE_H],
      )
      wrote = true
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
    this.controller.abort()
    this.chunkCache.clear()
    this.height10Decoder.destroy()
    this.heightEncoder?.destroy()
    this.heightEncoder = null
    this.clearReadyUploads()
    this.pendingQueue.length = 0
    this.residentAlbedo.clear()
    this.residentNormal.clear()
    this.inFlight.clear()
    this.index.clear()
    this.tileListeners.clear()
    this.textureListeners.clear()
    this.bc7SrgbEncoder?.destroy()
    this.bc7SrgbEncoder = null
    this.bc7LinearEncoder?.destroy()
    this.bc7LinearEncoder = null
    this.pageTexture.destroy()
    this.poolTexture.destroy()
    this.normalPoolTexture.destroy()
    this.slotFadeBuffer.destroy()
  }
}

function intersectEarth(
  eye: Vec3,
  right: Vec3,
  up: Vec3,
  back: Vec3,
  sx: number,
  sy: number,
  morph: number,
): { lon: number; lat: number; cosIncidence: number } | null {
  if (morph >= 0.5) return intersectUnitSphere(eye, right, up, back, sx, sy)
  const rx = right[0] * sx + up[0] * sy - back[0]
  const ry = right[1] * sx + up[1] * sy - back[1]
  const rz = right[2] * sx + up[2] * sy - back[2]
  const invLen = 1 / Math.hypot(rx, ry, rz)
  const dx = rx * invLen,
    dy = ry * invLen,
    dz = rz * invLen
  // Flat Equal Earth plane at z = 0
  if (Math.abs(dz) < 1e-6) return null
  const t = -eye[2] / dz
  if (t <= 0) return null
  const hit = equalEarthInverse(eye[0] + dx * t, eye[1] + dy * t)
  return hit ? { lon: hit.lon, lat: hit.lat, cosIncidence: Math.abs(dz) } : null
}
