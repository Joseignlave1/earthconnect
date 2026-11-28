export type TileEntry = { packId: number; offset: number; length: number }

export type PackHandle =
  | { kind: 'blob'; blob: Blob }
  | { kind: 'url'; url: string; fullBlob?: Blob; fullBlobPending?: Promise<Blob> }

export type TileVec3 = readonly [number, number, number]

export const VIRTUAL_TILE_SIZE = 512
export const VIRTUAL_TILE_INNER = 508
export const VIRTUAL_TILE_GUTTER = 2
export const VIRTUAL_TILE_MIPS = 3
export const MAX_CONCURRENT_TILE_DECODES = 4
export const TILE_PAGE_EMPTY = 255

const PAGE_FINEST_SLOT_OFFSET = 0
const PAGE_FINEST_LEVEL_OFFSET = 1
const PAGE_FALLBACK_SLOT_OFFSET = 2
const PAGE_FALLBACK_LEVEL_OFFSET = 3

export function tileKey(level: number, tx: number, ty: number) {
  return (level << 16) | (ty << 8) | tx
}

export function planetTileKey(kind: 0 | 1, level: number, tx: number, ty: number) {
  return (kind << 20) | (level << 16) | (ty << 8) | tx
}

export function parseTilePackHeader(buffer: ArrayBuffer): { packId: number; count: number } | null {
  if (buffer.byteLength < 16) return null
  const head = new DataView(buffer)
  // 'ECVT' little-endian = 0x54564345
  if (head.getUint32(0, true) !== 0x54564345 || head.getUint32(4, true) !== 1) return null
  const packId = head.getUint32(8, true)
  const count = head.getUint32(12, true)
  if (!count || count > 32768) return null
  return { packId, count }
}

export function parseTilePackIndex(buffer: ArrayBuffer, packId: number, count: number): Map<number, TileEntry> {
  const entries = new Map<number, TileEntry>()
  if (buffer.byteLength < count * 12) return entries
  const table = new DataView(buffer)
  for (let i = 0; i < count; i++) {
    const base = i * 12
    const key = table.getUint32(base, true)
    const offset = table.getUint32(base + 4, true)
    const length = table.getUint32(base + 8, true)
    if (length > 0) entries.set(key, { packId, offset, length })
  }
  return entries
}

export function toPackHandle(resource: Blob | string): PackHandle {
  return typeof resource === 'string' ? { kind: 'url', url: resource } : { kind: 'blob', blob: resource }
}

async function sliceHandleBlob(blob: Blob, start: number, length: number, signal: AbortSignal): Promise<ArrayBuffer> {
  const buf = await blob.slice(start, start + length).arrayBuffer()
  signal.throwIfAborted()
  return buf
}

export async function readPackRange(
  handle: PackHandle,
  start: number,
  length: number,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  if (handle.kind === 'blob') return sliceHandleBlob(handle.blob, start, length, signal)
  if (handle.fullBlob) return sliceHandleBlob(handle.fullBlob, start, length, signal)
  if (handle.fullBlobPending) {
    const blob = await handle.fullBlobPending
    signal.throwIfAborted()
    return sliceHandleBlob(blob, start, length, signal)
  }
  const response = await fetch(handle.url, {
    signal,
    headers: { Range: `bytes=${start}-${start + length - 1}` },
  })
  if (!response.ok) throw new Error(`Tile range fetch failed (${response.status}).`)
  // If a server ignored Range and returned the full file (HTTP 200), cache it as a Blob
  // so subsequent range reads slice locally without re-downloading the entire pack.
  if (response.status === 200) {
    const pending = (handle.fullBlobPending ??= response.blob().then(
      blob => {
        handle.fullBlob = blob
        handle.fullBlobPending = undefined
        return blob
      },
      error => {
        handle.fullBlobPending = undefined
        throw error
      },
    ))
    const blob = await pending
    signal.throwIfAborted()
    return sliceHandleBlob(blob, start, length, signal)
  }
  const buf = await response.arrayBuffer()
  signal.throwIfAborted()
  return buf
}

export async function readTileWebpBlob(
  handle: PackHandle,
  offset: number,
  length: number,
  signal: AbortSignal,
): Promise<Blob> {
  if (handle.kind === 'blob') {
    signal.throwIfAborted()
    return handle.blob.slice(offset, offset + length, 'image/webp')
  }
  if (handle.fullBlob) {
    signal.throwIfAborted()
    return handle.fullBlob.slice(offset, offset + length, 'image/webp')
  }
  const buf = await readPackRange(handle, offset, length, signal)
  return new Blob([buf], { type: 'image/webp' })
}

export async function loadPackIndexEntries(
  handle: PackHandle,
  packId: number,
  signal: AbortSignal,
): Promise<Map<number, TileEntry> | null> {
  const headBuf = await readPackRange(handle, 0, 16, signal)
  if (signal.aborted) return null
  const header = parseTilePackHeader(headBuf)
  if (!header || header.packId !== packId) return null
  const tableBuf = await readPackRange(handle, 16, header.count * 12, signal)
  if (signal.aborted || tableBuf.byteLength < header.count * 12) return null
  return parseTilePackIndex(tableBuf, packId, header.count)
}

export function createTileSampler(device: GPUDevice, label: string): GPUSampler {
  return device.createSampler({
    label,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
  })
}

export function paintTilePageTable<S extends { level: number; tx: number; ty: number }>(
  data: Uint8Array,
  pageWidth: number,
  levelCount: number,
  slots: readonly (S | null)[],
  matches: (slot: S) => boolean = () => true,
) {
  data.fill(TILE_PAGE_EMPTY)
  // Paint coarser levels first so (r,g) holds finest resident tile and (b,a) holds coarser fallback.
  for (let level = 0; level < levelCount; level++) {
    const span = 1 << (levelCount - 1 - level)
    for (let slotIndex = 0; slotIndex < slots.length; slotIndex++) {
      const slot = slots[slotIndex]
      if (!slot || slot.level !== level || !matches(slot)) continue
      const x0 = slot.tx * span
      const y0 = slot.ty * span
      for (let dy = 0; dy < span; dy++) {
        const rowOff = (y0 + dy) * pageWidth
        for (let dx = 0; dx < span; dx++) {
          const idx = (rowOff + x0 + dx) * 4
          if (data[idx + PAGE_FINEST_SLOT_OFFSET] < TILE_PAGE_EMPTY) {
            data[idx + PAGE_FALLBACK_SLOT_OFFSET] = data[idx + PAGE_FINEST_SLOT_OFFSET]
            data[idx + PAGE_FALLBACK_LEVEL_OFFSET] = data[idx + PAGE_FINEST_LEVEL_OFFSET]
          }
          data[idx + PAGE_FINEST_SLOT_OFFSET] = slotIndex
          data[idx + PAGE_FINEST_LEVEL_OFFSET] = level
        }
      }
    }
  }
}

export function findLruSlot(slots: readonly ({ lastUsed: number } | null)[]): number {
  let oldestSlot = 0
  let oldestFrame = Infinity
  for (let i = 0; i < slots.length; i++) {
    const slot = slots[i]
    if (!slot) return i
    if (slot.lastUsed < oldestFrame) {
      oldestFrame = slot.lastUsed
      oldestSlot = i
    }
  }
  return oldestSlot
}

export function intersectUnitSphere(
  eye: TileVec3,
  right: TileVec3,
  up: TileVec3,
  back: TileVec3,
  sx: number,
  sy: number,
): { lon: number; lat: number; cosIncidence: number } | null {
  const rx = right[0] * sx + up[0] * sy - back[0]
  const ry = right[1] * sx + up[1] * sy - back[1]
  const rz = right[2] * sx + up[2] * sy - back[2]
  const invLen = 1 / Math.hypot(rx, ry, rz)
  const dx = rx * invLen,
    dy = ry * invLen,
    dz = rz * invLen

  const b = eye[0] * dx + eye[1] * dy + eye[2] * dz
  const c = eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2] - 1
  const disc = b * b - c
  if (disc < 0) return null
  const t = -b - Math.sqrt(disc)
  if (t <= 0) return null
  const px = eye[0] + dx * t
  const py = eye[1] + dy * t
  const pz = eye[2] + dz * t
  const lon = Math.atan2(px, pz)
  const lat = Math.asin(Math.max(-1, Math.min(1, py)))
  const cosIncidence = Math.max(0, -(px * dx + py * dy + pz * dz))
  return { lon, lat, cosIncidence }
}
