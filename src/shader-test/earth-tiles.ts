import { Bc7TileEncoder } from './bc7-gpu.ts'
import { cameraBasis, equalEarthInverse, FOV, type Camera, type Vec3 } from './geometry.ts'
import { generateMipmaps } from './gpu.ts'
import {
  createTileSampler,
  findLruSlot,
  intersectUnitSphere,
  loadPackIndexEntries,
  MAX_CONCURRENT_TILE_DECODES,
  paintTilePageTable,
  readTileWebpBlob,
  tileKey,
  toPackHandle,
  VIRTUAL_TILE_MIPS,
  VIRTUAL_TILE_SIZE,
  type PackHandle,
  type TileEntry,
} from './virtual-tiles.ts'

export { tileKey }

const EARTH_TILE_PACKS = 8
// BC7 uses 4× less VRAM than RGBA8, so we double the resident tile cache (192 vs 96 slots)
// while still cutting tile pool VRAM in half (66 MB vs 132 MB).
const EARTH_TILE_SLOTS_BC7 = 192
const EARTH_TILE_SLOTS_RGBA = 96
export const EARTH_TILE_PAGE_W = 160
export const EARTH_TILE_PAGE_H = 80
export const EARTH_TILE_BASE_COLS = 40
export const EARTH_TILE_BASE_ROWS = 20
const LEVEL_COUNT = 3

type TileSlot = { key: number; level: number; tx: number; ty: number; lastUsed: number }
type TileJob = { key: number; level: number; tx: number; ty: number; priority: number }

export type EarthTilePackLoader = (packId: number, signal: AbortSignal) => Promise<Blob | string | null>

function keyLevel(key: number) {
  return key >>> 16
}
function keyTy(key: number) {
  return (key >>> 8) & 0xff
}
function keyTx(key: number) {
  return key & 0xff
}

export class EarthTileCache {
  private readonly controller = new AbortController()
  private readonly useBc7: boolean
  private readonly slotCapacity: number
  private readonly packs: (PackHandle | null)[] = Array(EARTH_TILE_PACKS).fill(null)
  private readonly loadingPacks = new Set<number>()
  private packLoader: EarthTilePackLoader | null = null
  private bc7Encoder: Bc7TileEncoder | null = null
  private readonly index = new Map<number, TileEntry>()
  private readonly resident = new Map<number, number>()
  private readonly inFlight = new Set<number>()
  private readonly slots: (TileSlot | null)[]
  private readonly pageData = new Uint8Array(EARTH_TILE_PAGE_W * EARTH_TILE_PAGE_H * 4)
  private pendingQueue: TileJob[] = []
  private activeDecodes = 0
  private frameCounter = 0
  private destroyed = false
  private activated = false
  private packsAttached = false
  private readonly tileListeners = new Set<() => void>()
  private readonly textureListeners = new Set<() => void>()

  private readonly device: GPUDevice
  pageTexture: GPUTexture
  poolTexture: GPUTexture
  readonly sampler: GPUSampler

  constructor(device: GPUDevice) {
    this.device = device
    this.useBc7 = device.features.has('texture-compression-bc')
    this.slotCapacity = this.useBc7 ? EARTH_TILE_SLOTS_BC7 : EARTH_TILE_SLOTS_RGBA
    this.slots = Array(this.slotCapacity).fill(null)
    this.sampler = createTileSampler(device, 'Earth virtual tile sampler')
    // Tiny 4-byte placeholders until the user zooms in and a regional ECVT1 pack loads.
    this.pageTexture = device.createTexture({
      label: 'Earth tile page placeholder',
      size: [1, 1],
      format: 'rgba8uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture: this.pageTexture }, new Uint8Array([255, 0, 255, 0]), {}, [1, 1])
    this.poolTexture = device.createTexture({
      label: 'Earth tile pool placeholder',
      size: [1, 1, 1],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture: this.poolTexture }, new Uint8Array([18, 30, 42, 255]), {}, [1, 1, 1])
    this.pageData.fill(255)
  }

  get ready() {
    return this.activated
  }
  get attached() {
    return this.packsAttached
  }
  get residentCount() {
    return this.resident.size
  }
  get indexedCount() {
    return this.index.size
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

  attachPacks(loadPack: EarthTilePackLoader) {
    if (this.packsAttached || this.destroyed) return
    this.packsAttached = true
    this.packLoader = loadPack
  }

  private ensurePackLoaded(packId: number) {
    if (
      this.destroyed ||
      !this.packLoader ||
      packId < 0 ||
      packId >= EARTH_TILE_PACKS ||
      this.packs[packId] ||
      this.loadingPacks.has(packId)
    ) {
      return
    }
    this.loadingPacks.add(packId)
    void this.loadPackIndex(packId, this.packLoader, this.controller.signal)
  }

  private async loadPackIndex(packId: number, loadPack: EarthTilePackLoader, signal: AbortSignal) {
    try {
      const resource = await loadPack(packId, signal)
      if (!resource || signal.aborted || this.destroyed) {
        this.loadingPacks.delete(packId)
        return
      }
      const handle = toPackHandle(resource)
      const entries = await loadPackIndexEntries(handle, packId, signal)
      if (!entries || signal.aborted || this.destroyed) {
        this.loadingPacks.delete(packId)
        return
      }
      this.packs[packId] = handle
      this.loadingPacks.delete(packId)
      for (const [key, entry] of entries) this.index.set(key, entry)
      if (!this.activated && this.index.size > 0) this.activateGpuPool()
      this.notifyTileReady()
    } catch {
      this.loadingPacks.delete(packId)
      // Optional 500 m packs degrade seamlessly to the 8K base texture.
    }
  }

  private activateGpuPool() {
    if (this.activated || this.destroyed) return
    this.activated = true
    const device = this.device
    const oldPage = this.pageTexture
    const oldPool = this.poolTexture
    this.pageTexture = device.createTexture({
      label: 'Earth virtual tile page table',
      size: [EARTH_TILE_PAGE_W, EARTH_TILE_PAGE_H],
      format: 'rgba8uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.pageData.fill(255)
    device.queue.writeTexture({ texture: this.pageTexture }, this.pageData, { bytesPerRow: EARTH_TILE_PAGE_W * 4 }, [
      EARTH_TILE_PAGE_W,
      EARTH_TILE_PAGE_H,
    ])
    this.poolTexture = device.createTexture({
      label: this.useBc7 ? 'Earth 500m virtual tile cache (BC7)' : 'Earth 500m virtual tile cache',
      size: [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE, this.slotCapacity],
      mipLevelCount: VIRTUAL_TILE_MIPS,
      format: this.useBc7 ? 'bc7-rgba-unorm-srgb' : 'rgba8unorm-srgb',
      usage: this.useBc7
        ? GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        : GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    if (this.useBc7) {
      this.bc7Encoder = new Bc7TileEncoder(device, VIRTUAL_TILE_SIZE, VIRTUAL_TILE_MIPS, 'rgba8unorm-srgb')
    }
    this.notifyTexturesChanged()
    void device.queue
      .onSubmittedWorkDone()
      .then(() => {
        oldPage.destroy()
        oldPool.destroy()
      })
      .catch(() => undefined)
  }

  /**
   * Determine which 500m tiles intersect the current camera (and optional loupe),
   * lazily load only the visible 90°×90° octant pack(s), and stream missing tiles
   * into the GPU LRU tile pool.
   */
  updateView(
    camera: Camera,
    morph: number,
    metresPerPixel: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    if (!this.packsAttached || this.destroyed) return
    const loupeMpp = loupe ? metresPerPixel / Math.max(1, loupe.zoom) : Infinity
    const minMpp = Math.min(metresPerPixel, loupeMpp)
    // Stay on the already-resident 8K texture until the viewport exceeds ~8K detail.
    if (minMpp > 3900) {
      this.pendingQueue.length = 0
      return
    }

    const frame = ++this.frameCounter
    const needed = new Map<number, number>() // key -> priority (lower is higher priority)

    const requestPoint = (lon: number, lat: number, mpp: number, distSq: number, halo: number) => {
      if (mpp > 3900) return
      const u = (((lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1
      const v = Math.max(0, Math.min(0.999999, 0.5 - lat / Math.PI))
      let targetLevel = 0
      if (mpp < 750) targetLevel = 2
      else if (mpp < 1500) targetLevel = 1

      // Touch and enqueue target level + parent levels so refinement is instantaneous.
      for (let level = 0; level <= targetLevel; level++) {
        const cols = EARTH_TILE_BASE_COLS << level
        const rows = EARTH_TILE_BASE_ROWS << level
        const cx = Math.floor(u * cols) % cols
        const cy = Math.max(0, Math.min(rows - 1, Math.floor(v * rows)))
        const radius = level === targetLevel ? halo : 0
        for (let dy = -radius; dy <= radius; dy++) {
          const ty = cy + dy
          if (ty < 0 || ty >= rows) continue
          const packRow = ty >= rows >> 1 ? 4 : 0
          for (let dx = -radius; dx <= radius; dx++) {
            const tx = (cx + dx + cols) % cols
            const packId = packRow + Math.floor((tx * 4) / cols)
            if (!this.packs[packId]) {
              this.ensurePackLoaded(packId)
              continue
            }
            const key = tileKey(level, tx, ty)
            if (!this.index.has(key)) continue
            const slotIndex = this.resident.get(key)
            if (slotIndex !== undefined) {
              const slot = this.slots[slotIndex]
              if (slot) slot.lastUsed = frame
              continue
            }
            // Coarser levels load first (level * 10), then center-most tiles first.
            const prio = level * 10 + distSq + (dx * dx + dy * dy) * 0.25
            const prev = needed.get(key)
            if (prev === undefined || prio < prev) needed.set(key, prio)
          }
        }
      }
    }

    if (metresPerPixel <= 3900) {
      const { right, up, back, eye } = cameraBasis(camera)
      const tanHalf = Math.tan(FOV / 2)
      const gridN = 11
      for (let gy = 0; gy < gridN; gy++) {
        const ndcY = ((gy / (gridN - 1)) * 2 - 1) * 1.08
        for (let gx = 0; gx < gridN; gx++) {
          const ndcX = ((gx / (gridN - 1)) * 2 - 1) * 1.08
          const hit = intersectEarth(eye, right, up, back, ndcX * tanHalf * camera.aspect, ndcY * tanHalf, morph)
          if (!hit) continue
          // Scale ground mpp at grazing angles by 1/cos(incidence) so the limb doesn't over-request L2.
          const effectiveMpp = metresPerPixel / Math.max(0.35, hit.cosIncidence)
          requestPoint(hit.lon, hit.lat, effectiveMpp, ndcX * ndcX + ndcY * ndcY, 0)
        }
      }
    }

    if (loupe && loupeMpp <= 3900) {
      requestPoint(loupe.lon, loupe.lat, loupeMpp, -2, 1)
    }

    if (needed.size === 0) {
      this.pendingQueue.length = 0
      return
    }
    const nextQueue: TileJob[] = []
    for (const [key, priority] of needed) {
      if (this.inFlight.has(key)) continue
      nextQueue.push({ key, level: keyLevel(key), tx: keyTx(key), ty: keyTy(key), priority })
    }
    nextQueue.sort((a, b) => a.priority - b.priority)
    this.pendingQueue = nextQueue
    this.pumpQueue()
  }

  private pumpQueue() {
    while (this.activeDecodes < MAX_CONCURRENT_TILE_DECODES && this.pendingQueue.length > 0 && !this.destroyed) {
      const job = this.pendingQueue.shift()!
      if (this.resident.has(job.key) || this.inFlight.has(job.key)) continue
      const entry = this.index.get(job.key)
      if (!entry) continue
      const handle = this.packs[entry.packId]
      if (!handle) continue
      this.inFlight.add(job.key)
      this.activeDecodes++
      void this.decodeAndUpload(job, entry, handle)
    }
  }

  private allocateSlot(): number {
    const slotIndex = findLruSlot(this.slots)
    const evicted = this.slots[slotIndex]
    if (evicted) {
      this.resident.delete(evicted.key)
      this.slots[slotIndex] = null
    }
    return slotIndex
  }

  private async decodeAndUpload(
    job: { key: number; level: number; tx: number; ty: number },
    entry: TileEntry,
    handle: PackHandle,
  ) {
    const { signal } = this.controller
    let bitmap: ImageBitmap | undefined
    try {
      const blob = await readTileWebpBlob(handle, entry.offset, entry.length, signal)
      bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
      if (signal.aborted || this.destroyed) return
      const slotIndex = this.allocateSlot()
      try {
        const device = this.device
        if (this.bc7Encoder) {
          device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: this.bc7Encoder.scratch }, [
            VIRTUAL_TILE_SIZE,
            VIRTUAL_TILE_SIZE,
          ])
          const encoder = device.createCommandEncoder({ label: 'Earth tile BC7 upload' })
          generateMipmaps(device, this.bc7Encoder.scratch, {
            mipLevelCount: VIRTUAL_TILE_MIPS,
            label: 'Earth tile mipmaps',
            encoder,
          })
          this.bc7Encoder.encodeIntoArrayLayer(encoder, this.poolTexture, slotIndex)
          device.queue.submit([encoder.finish()])
        } else {
          device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: this.poolTexture, origin: [0, 0, slotIndex] },
            [VIRTUAL_TILE_SIZE, VIRTUAL_TILE_SIZE],
          )
          generateMipmaps(device, this.poolTexture, {
            layers: [slotIndex],
            mipLevelCount: VIRTUAL_TILE_MIPS,
            label: 'Earth tile mipmaps',
          })
        }
        this.slots[slotIndex] = { key: job.key, level: job.level, tx: job.tx, ty: job.ty, lastUsed: this.frameCounter }
        this.resident.set(job.key, slotIndex)
        this.flushPageTable()
        this.notifyTileReady()
      } catch (uploadError) {
        this.flushPageTable()
        throw uploadError
      }
    } catch {
      // Transient aborts/decode errors degrade gracefully to parent/8K texture.
    } finally {
      bitmap?.close()
      this.inFlight.delete(job.key)
      this.activeDecodes--
      this.pumpQueue()
    }
  }

  private flushPageTable() {
    if (!this.activated || this.destroyed) return
    paintTilePageTable(this.pageData, EARTH_TILE_PAGE_W, LEVEL_COUNT, this.slots)
    this.device.queue.writeTexture(
      { texture: this.pageTexture },
      this.pageData,
      { bytesPerRow: EARTH_TILE_PAGE_W * 4 },
      [EARTH_TILE_PAGE_W, EARTH_TILE_PAGE_H],
    )
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.controller.abort()
    this.pendingQueue.length = 0
    this.resident.clear()
    this.inFlight.clear()
    this.index.clear()
    this.tileListeners.clear()
    this.textureListeners.clear()
    this.bc7Encoder?.destroy()
    this.bc7Encoder = null
    this.pageTexture.destroy()
    this.poolTexture.destroy()
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

export function earthTilePath(packId: number) {
  return `textures/earth-tiles-${packId}.bin`
}
