import { FOV } from '../shader-test/geometry.ts'
import { generateMipmaps } from '../shader-test/gpu.ts'
import {
  createTileSampler, findLruSlot, intersectUnitSphere, loadPackIndexEntries, MAX_CONCURRENT_TILE_DECODES,
  paintTilePageTable, readTileWebpBlob, toPackHandle, VIRTUAL_TILE_INNER, VIRTUAL_TILE_MIPS, VIRTUAL_TILE_SIZE,
  type PackHandle, type TileEntry, type TileVec3,
} from '../shader-test/virtual-tiles.ts'

export type PlanetTileBody = 'moon' | 'mars'

export const PLANET_TILE_PAIRS = 48

const PLANET_TILE_PACKS = 4
const PLANET_TILE_BASE_COLS = 24
const PLANET_TILE_BASE_ROWS = 12
const PLANET_TILE_PAGE_W = 48
const PLANET_TILE_PAGE_H = 24
const LEVEL_COUNT = 2

type PairSlot = {
  body: PlanetTileBody
  key: number
  level: number
  tx: number
  ty: number
  lastUsed: number
}

type TileJob = { body: PlanetTileBody; key: number; level: number; tx: number; ty: number; priority: number }

export type PlanetTilePackLoader = (body: PlanetTileBody, packId: number, signal: AbortSignal) => Promise<Blob | string | null>

export function planetTilePath(body: PlanetTileBody, packId: number) {
  return `textures/${body}-tiles-${packId}.bin`
}

export function planetTileKey(kind: 0 | 1, level: number, tx: number, ty: number) {
  return (kind << 20) | (level << 16) | (ty << 8) | tx
}

function pairKey(level: number, tx: number, ty: number) {
  return (level << 16) | (ty << 8) | tx
}

function keyLevel(key: number) { return (key >>> 16) & 0xf }
function keyTy(key: number) { return (key >>> 8) & 0xff }
function keyTx(key: number) { return key & 0xff }

type BodyState = {
  packs: (PackHandle | null)[]
  index: Map<number, TileEntry>
  resident: Map<number, number> // pairKey -> pairSlotIndex (0..47)
  inFlight: Set<number>
  pageData: Uint8Array<ArrayBuffer>
  pageTexture: GPUTexture
  pageDirty: boolean
  indexed: boolean
}

export class PlanetTileCache {
  private readonly device: GPUDevice
  private readonly controller = new AbortController()
  private readonly bodies: Record<PlanetTileBody, BodyState>
  private readonly slots: (PairSlot | null)[] = Array(PLANET_TILE_PAIRS).fill(null)
  private pendingQueue: TileJob[] = []
  private activeDecodes = 0
  private frameCounter = 0
  private destroyed = false
  private poolAllocated = false
  private packsAttached = false
  private readonly tileListeners = new Set<() => void>()
  private readonly textureListeners = new Set<() => void>()

  poolTexture: GPUTexture
  readonly sampler: GPUSampler

  constructor(device: GPUDevice) {
    this.device = device
    this.sampler = createTileSampler(device, 'Planetary virtual tile sampler')
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
    const makeBody = (body: PlanetTileBody): BodyState => {
      const pageData = new Uint8Array(PLANET_TILE_PAGE_W * PLANET_TILE_PAGE_H * 4)
      pageData.fill(255)
      const pageTexture = device.createTexture({
        label: `${body} tile page placeholder`,
        size: [1, 1],
        format: 'rgba8uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      device.queue.writeTexture({ texture: pageTexture }, new Uint8Array([255, 0, 255, 0]), {}, [1, 1])
      return {
        packs: Array(PLANET_TILE_PACKS).fill(null),
        index: new Map(),
        resident: new Map(),
        inFlight: new Set(),
        pageData,
        pageTexture,
        pageDirty: false,
        indexed: false,
      }
    }
    this.bodies = { moon: makeBody('moon'), mars: makeBody('mars') }
  }

  pageTexture(body: PlanetTileBody) { return this.bodies[body].pageTexture }
  ready(body: PlanetTileBody) { return this.poolAllocated && this.bodies[body].indexed }
  hasIndex(body: PlanetTileBody) { return this.bodies[body].indexed }
  residentCount(body: PlanetTileBody) { return this.bodies[body].resident.size }
  indexedCount(body: PlanetTileBody) { return this.bodies[body].index.size >> 1 }

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

  attachPacks(loadPack: PlanetTilePackLoader) {
    if (this.packsAttached || this.destroyed) return
    this.packsAttached = true
    const { signal } = this.controller
    for (const body of ['moon', 'mars'] as const) {
      for (let packId = 0; packId < PLANET_TILE_PACKS; packId++) {
        void this.loadPackIndex(body, packId, loadPack, signal)
      }
    }
  }

  private async loadPackIndex(body: PlanetTileBody, packId: number, loadPack: PlanetTilePackLoader, signal: AbortSignal) {
    try {
      const resource = await loadPack(body, packId, signal)
      if (!resource || signal.aborted || this.destroyed) return
      const handle = toPackHandle(resource)
      const entries = await loadPackIndexEntries(handle, packId, signal)
      if (!entries || signal.aborted || this.destroyed) return
      const state = this.bodies[body]
      state.packs[packId] = handle
      for (const [key, entry] of entries) state.index.set(key, entry)
      if (!state.indexed && state.index.size > 0) state.indexed = true
      if (state.index.size > 0) this.notifyTileReady()
    } catch {
      // Optional Moon/Mars high-res packs degrade cleanly to 8K base albedo + 4K DEM.
    }
  }

  private ensureGpuPool() {
    if (this.poolAllocated || this.destroyed) return
    this.poolAllocated = true
    const device = this.device
    const oldPool = this.poolTexture
    this.poolTexture = device.createTexture({
      label: 'Planetary paired albedo + normal virtual tile cache',
      size: [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE, PLANET_TILE_PAIRS * 2],
      mipLevelCount: VIRTUAL_TILE_MIPS,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    const oldTextures: GPUTexture[] = [oldPool]
    for (const body of ['moon', 'mars'] as const) {
      const state = this.bodies[body]
      const oldPage = state.pageTexture
      state.pageTexture = device.createTexture({
        label: `${body} virtual tile page table`,
        size: [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H],
        format: 'rgba8uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      state.pageData.fill(255)
      device.queue.writeTexture(
        { texture: state.pageTexture },
        state.pageData,
        { bytesPerRow: PLANET_TILE_PAGE_W * 4 },
        [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H],
      )
      oldTextures.push(oldPage)
    }
    this.notifyTexturesChanged()
    void device.queue.onSubmittedWorkDone().then(() => {
      for (const tex of oldTextures) tex.destroy()
    }).catch(() => undefined)
  }

  /**
   * Determine visible Moon or Mars tiles in body-local unit-sphere camera space
   * and stream paired albedo + relief-normal tiles into the shared GPU LRU pool.
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
    if (this.destroyed) return
    const state = this.bodies[body]
    if (!state.indexed) return

    // Native 8K texel size = (2*pi*R) / 8192; L0 = (2*pi*R)/12192; L1 = (2*pi*R)/24384
    const l0TexelM = (2 * Math.PI * radiusMetres) / (PLANET_TILE_BASE_COLS * VIRTUAL_TILE_INNER)
    const l1TexelM = l0TexelM * 0.5
    const entryThreshold = l0TexelM * 1.45 // activate just before surpassing 8K (~1,300m Moon, ~2,540m Mars)
    const l1Threshold = l1TexelM * 1.55    // refine to L1 (~695m Moon, ~1,355m Mars)

    const loupeMpp = loupe ? physicalMetresPerPixel / Math.max(1, loupe.zoom) : Infinity
    if (Math.min(physicalMetresPerPixel, loupeMpp) > entryThreshold) return

    // Lazily allocate the shared Moon/Mars pool on first actual zoom-in demand.
    this.ensureGpuPool()

    const frame = ++this.frameCounter
    const needed = new Map<number, number>()

    const requestPoint = (lon: number, lat: number, mpp: number, distSq: number, halo: number) => {
      if (mpp > entryThreshold) return
      const u = ((lon / (2 * Math.PI) + 0.5) % 1 + 1) % 1
      const v = Math.max(0, Math.min(0.999999, 0.5 - lat / Math.PI))
      const targetLevel = mpp < l1Threshold ? 1 : 0

      for (let level = 0; level <= targetLevel; level++) {
        const cols = PLANET_TILE_BASE_COLS << level
        const rows = PLANET_TILE_BASE_ROWS << level
        const cx = Math.floor(u * cols) % cols
        const cy = Math.max(0, Math.min(rows - 1, Math.floor(v * rows)))
        const radius = level === targetLevel ? halo : 0
        for (let dy = -radius; dy <= radius; dy++) {
          const ty = cy + dy
          if (ty < 0 || ty >= rows) continue
          for (let dx = -radius; dx <= radius; dx++) {
            const tx = (cx + dx + cols) % cols
            const pk = pairKey(level, tx, ty)
            if (!state.index.has(planetTileKey(0, level, tx, ty)) || !state.index.has(planetTileKey(1, level, tx, ty))) continue
            const slotIndex = state.resident.get(pk)
            if (slotIndex !== undefined) {
              const slot = this.slots[slotIndex]
              if (slot) slot.lastUsed = frame
              continue
            }
            const prio = level * 10 + distSq + (dx * dx + dy * dy) * 0.25
            const prev = needed.get(pk)
            if (prev === undefined || prio < prev) needed.set(pk, prio)
          }
        }
      }
    }

    if (physicalMetresPerPixel <= entryThreshold) {
      const tanHalf = Math.tan(FOV / 2)
      const gridN = 11
      for (let gy = 0; gy < gridN; gy++) {
        const ndcY = ((gy / (gridN - 1)) * 2 - 1) * 1.08
        for (let gx = 0; gx < gridN; gx++) {
          const ndcX = ((gx / (gridN - 1)) * 2 - 1) * 1.08
          const hit = intersectUnitSphere(eyeUnit, right, up, back, ndcX * tanHalf * aspect, ndcY * tanHalf)
          if (!hit) continue
          const effectiveMpp = physicalMetresPerPixel / Math.max(0.35, hit.cosIncidence)
          requestPoint(hit.lon, hit.lat, effectiveMpp, ndcX * ndcX + ndcY * ndcY, 0)
        }
      }
    }

    if (loupe && loupeMpp <= entryThreshold) {
      requestPoint(loupe.lon, loupe.lat, loupeMpp, -2, 1)
    }

    if (needed.size === 0) return
    const nextQueue: TileJob[] = []
    for (const [key, priority] of needed) {
      if (state.inFlight.has(key)) continue
      nextQueue.push({ body, key, level: keyLevel(key), tx: keyTx(key), ty: keyTy(key), priority })
    }
    nextQueue.sort((a, b) => a.priority - b.priority)
    this.pendingQueue = nextQueue
    this.pumpQueue()
  }

  private pumpQueue() {
    while (this.activeDecodes < MAX_CONCURRENT_TILE_DECODES && this.pendingQueue.length > 0 && !this.destroyed) {
      const job = this.pendingQueue.shift()!
      const state = this.bodies[job.body]
      if (state.resident.has(job.key) || state.inFlight.has(job.key)) continue
      const albedoEntry = state.index.get(planetTileKey(0, job.level, job.tx, job.ty))
      const normalEntry = state.index.get(planetTileKey(1, job.level, job.tx, job.ty))
      if (!albedoEntry || !normalEntry) continue
      const albedoHandle = state.packs[albedoEntry.packId]
      const normalHandle = state.packs[normalEntry.packId]
      if (!albedoHandle || !normalHandle) continue
      state.inFlight.add(job.key)
      this.activeDecodes++
      void this.decodeAndUpload(job, albedoEntry, albedoHandle, normalEntry, normalHandle)
    }
  }

  private allocateSlot(): number {
    const slotIndex = findLruSlot(this.slots)
    const evicted = this.slots[slotIndex]
    if (evicted) {
      const evictedBody = this.bodies[evicted.body]
      evictedBody.resident.delete(evicted.key)
      evictedBody.pageDirty = true
      this.slots[slotIndex] = null
    }
    return slotIndex
  }

  private async decodeAndUpload(
    job: { body: PlanetTileBody; key: number; level: number; tx: number; ty: number },
    albedoEntry: TileEntry,
    albedoHandle: PackHandle,
    normalEntry: TileEntry,
    normalHandle: PackHandle,
  ) {
    const { signal } = this.controller
    const state = this.bodies[job.body]
    let albedoBitmap: ImageBitmap | undefined
    let normalBitmap: ImageBitmap | undefined
    try {
      const [albedoBlob, normalBlob] = await Promise.all([
        readTileWebpBlob(albedoHandle, albedoEntry.offset, albedoEntry.length, signal),
        readTileWebpBlob(normalHandle, normalEntry.offset, normalEntry.length, signal),
      ])
      ;[albedoBitmap, normalBitmap] = await Promise.all([
        createImageBitmap(albedoBlob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }),
        createImageBitmap(normalBlob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }),
      ])
      if (signal.aborted || this.destroyed) return
      const slotIndex = this.allocateSlot()
      const normalSlice = slotIndex + PLANET_TILE_PAIRS
      const device = this.device
      device.queue.copyExternalImageToTexture(
        { source: albedoBitmap },
        { texture: this.poolTexture, origin: [0, 0, slotIndex] },
        [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
      )
      device.queue.copyExternalImageToTexture(
        { source: normalBitmap },
        { texture: this.poolTexture, origin: [0, 0, normalSlice] },
        [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
      )
      generateMipmaps(device, this.poolTexture, {
        layers: [slotIndex, normalSlice],
        mipLevelCount: VIRTUAL_TILE_MIPS,
        label: 'Planetary tile mipmaps',
      })
      this.slots[slotIndex] = { body: job.body, key: job.key, level: job.level, tx: job.tx, ty: job.ty, lastUsed: this.frameCounter }
      state.resident.set(job.key, slotIndex)
      state.pageDirty = true
      this.flushPageTables()
      this.notifyTileReady()
    } catch {
      // Transient aborts/decode errors degrade gracefully to 8K base textures.
    } finally {
      albedoBitmap?.close()
      normalBitmap?.close()
      state.inFlight.delete(job.key)
      this.activeDecodes--
      this.pumpQueue()
    }
  }

  private flushPageTables() {
    if (!this.poolAllocated || this.destroyed) return
    for (const body of ['moon', 'mars'] as const) {
      const state = this.bodies[body]
      if (!state.pageDirty) continue
      state.pageDirty = false
      paintTilePageTable(state.pageData, PLANET_TILE_PAGE_W, LEVEL_COUNT, this.slots, slot => slot.body === body)
      this.device.queue.writeTexture(
        { texture: state.pageTexture },
        state.pageData,
        { bytesPerRow: PLANET_TILE_PAGE_W * 4 },
        [PLANET_TILE_PAGE_W, PLANET_TILE_PAGE_H],
      )
    }
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.controller.abort()
    this.pendingQueue.length = 0
    this.tileListeners.clear()
    this.textureListeners.clear()
    this.poolTexture.destroy()
    for (const body of ['moon', 'mars'] as const) {
      const state = this.bodies[body]
      state.resident.clear()
      state.inFlight.clear()
      state.index.clear()
      state.pageTexture.destroy()
    }
  }
}
