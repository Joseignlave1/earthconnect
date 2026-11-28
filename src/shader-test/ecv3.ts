import { getPipelineCompileQueue, type DeferredComputePipeline } from './pipeline-queue.ts'
import {
  readPackRange,
  VIRTUAL_TILE_GUTTER,
  VIRTUAL_TILE_INNER,
  VIRTUAL_TILE_SIZE,
  type PackHandle,
} from './virtual-tiles.ts'

export const ECV3_MAGIC = 0x33564345 // 'ECV3' little-endian
export const ECV3_VERSION = 3

export const ECV3_TPL_GRAY8 = 0
export const ECV3_TPL_RGB8 = 1
export const ECV3_TPL_RGBA8 = 2
export const ECV3_TPL_HEIGHT10 = 3

export type Ecv3Template = {
  id: number
  bytes: Uint8Array<ArrayBuffer>
  ilocOff: number
  mdatOff: number
}

export type Ecv3TileEntry = {
  packId: number
  offset: number
  length: number
  tplId: number
  grainAmp: number
  quality: number
  flags: number
  aux: number
}

export type Ecv3Pack = {
  handle: PackHandle
  bodyId: number
  packId: number
  tileSize: number
  gutter: number
  baseCols: number
  baseRows: number
  levelCount: number
  templates: Map<number, Ecv3Template>
}

export function ecv3TileKey(kind: 0 | 1, level: number, tx: number, ty: number): number {
  return (((kind & 0xff) << 24) | ((level & 0xf) << 20) | ((ty & 0x3ff) << 10) | (tx & 0x3ff)) >>> 0
}

export function decodeEcv3Key(key: number): { kind: 0 | 1; level: number; tx: number; ty: number } {
  return {
    kind: ((key >>> 24) & 0xff) === 1 ? 1 : 0,
    level: (key >>> 20) & 0xf,
    ty: (key >>> 10) & 0x3ff,
    tx: key & 0x3ff,
  }
}

export function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1
  const exp = (bits >>> 10) & 0x1f
  const frac = bits & 0x03ff
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024)
  if (exp === 31) return frac === 0 ? sign * Infinity : Number.NaN
  return sign * 2 ** (exp - 15) * (1 + frac / 1024)
}

export function unpackEcv3HeightBounds(aux: number): { loM: number; spanM: number } {
  const loM = decodeFloat16(aux & 0xffff)
  const spanM = Math.max(1, decodeFloat16((aux >>> 16) & 0xffff))
  return { loM, spanM }
}

/**
 * Reassemble a full, browser-decodable AVIF file from an ECV3 shared ISOBMFF header
 * template and a raw stripped AV1 OBU payload.
 */
export function reassembleEcv3AvifBlob(tpl: Ecv3Template, obuBytes: Uint8Array): Blob {
  const hdrLen = tpl.bytes.byteLength
  if (hdrLen === 0) {
    if (obuBytes.buffer instanceof ArrayBuffer) {
      return new Blob([obuBytes as Uint8Array<ArrayBuffer>], { type: 'image/avif' })
    }
    const copy = new Uint8Array(obuBytes.byteLength)
    copy.set(obuBytes)
    return new Blob([copy], { type: 'image/avif' })
  }
  const obuLen = obuBytes.byteLength
  const out = new Uint8Array(hdrLen + obuLen)
  out.set(tpl.bytes, 0)
  const view = new DataView(out.buffer)
  view.setUint32(tpl.ilocOff, obuLen, false)
  view.setUint32(tpl.mdatOff, obuLen + 8, false)
  out.set(obuBytes, hdrLen)
  return new Blob([out], { type: 'image/avif' })
}

const INITIAL_HEADER_PROBE_BYTES = 65536

/**
 * Append `?r=start-length` for our ECV3 CDN worker so each 64 KiB chunk has a unique URL:
 * - Eliminates Chromium's per-URL HTTP disk-cache writer lock that serializes concurrent `Range:` requests
 *   to the same pack file.
 * - Eliminates CORS `OPTIONS` preflight round-trips (no custom `Range` request header).
 * - Enables lock-free `HTTP 200 OK` + `immutable` browser disk caching while sharing the exact same
 *   Cloudflare edge cache entry as `Range:` requests.
 */
export function ecv3ChunkRangeUrl(url: string, start: number, length: number): string {
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}r=${start}-${length}`
}

export async function readEcv3PackRange(
  handle: PackHandle,
  start: number,
  length: number,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  if (handle.kind === 'url' && !handle.fullBlob && !handle.fullBlobPending && handle.url.includes('/ecv3/')) {
    const response = await fetch(ecv3ChunkRangeUrl(handle.url, start, length), { signal })
    if (!response.ok) throw new Error(`Tile range fetch failed (${response.status}).`)
    const buf = await response.arrayBuffer()
    signal.throwIfAborted()
    return buf
  }
  return readPackRange(handle, start, length, signal)
}

/**
 * Load an ECV3 pack header, shared AVIF templates, and 20-byte tile index table
 * in 1 HTTP request (2 if the index exceeds 64 KiB).
 */
export async function loadEcv3PackIndex(
  handle: PackHandle,
  expectedPackId: number,
  signal: AbortSignal,
): Promise<{ pack: Ecv3Pack; entries: Map<number, Ecv3TileEntry>; chunk0?: Uint8Array<ArrayBuffer> } | null> {
  const probeBuf = await readEcv3PackRange(handle, 0, INITIAL_HEADER_PROBE_BYTES, signal)
  if (signal.aborted || probeBuf.byteLength < 32) return null
  const head = new DataView(probeBuf)
  if (head.getUint32(0, true) !== ECV3_MAGIC || head.getUint32(4, true) !== ECV3_VERSION) return null

  const bodyId = head.getUint16(8, true)
  const packId = head.getUint16(10, true)
  const tileSize = head.getUint16(12, true)
  const gutter = head.getUint16(14, true)
  const baseCols = head.getUint16(16, true)
  const baseRows = head.getUint16(18, true)
  const levelCount = head.getUint16(20, true)
  const tplCount = head.getUint16(22, true)
  const tileCount = head.getUint32(24, true)
  if (packId !== expectedPackId || tileCount === 0 || tileCount > 65536 || tplCount > 16) return null

  const probeBytes = new Uint8Array(probeBuf)
  let pos = 32
  const templates = new Map<number, Ecv3Template>()
  for (let i = 0; i < tplCount; i++) {
    if (pos + 8 > probeBytes.byteLength) return null
    const id = head.getUint16(pos, true)
    const len = head.getUint16(pos + 2, true)
    const ilocOff = head.getUint16(pos + 4, true)
    const mdatOff = head.getUint16(pos + 6, true)
    pos += 8
    if (pos + len > probeBytes.byteLength) return null
    const bytes = new Uint8Array(len)
    bytes.set(probeBytes.subarray(pos, pos + len))
    pos += len
    const pad = -(pos - 32) & 3
    pos += pad
    templates.set(id, { id, bytes, ilocOff, mdatOff })
  }

  const indexStart = pos
  const indexBytes = tileCount * 20
  const indexEnd = indexStart + indexBytes

  let tableView: DataView
  let tableOffset = indexStart
  if (indexEnd <= probeBuf.byteLength) {
    tableView = head
  } else {
    const tailLen = indexEnd - probeBuf.byteLength
    const tailBuf = await readEcv3PackRange(handle, probeBuf.byteLength, tailLen, signal)
    if (signal.aborted || tailBuf.byteLength < tailLen) return null
    const combined = new Uint8Array(indexBytes)
    const firstPart = probeBuf.byteLength - indexStart
    if (firstPart > 0) combined.set(probeBytes.subarray(indexStart, probeBuf.byteLength), 0)
    combined.set(new Uint8Array(tailBuf), Math.max(0, firstPart))
    tableView = new DataView(combined.buffer)
    tableOffset = 0
  }

  const entries = new Map<number, Ecv3TileEntry>()
  for (let i = 0; i < tileCount; i++) {
    const o = tableOffset + i * 20
    const key = tableView.getUint32(o, true)
    const offset = tableView.getUint32(o + 4, true)
    const length = tableView.getUint32(o + 8, true)
    const meta = tableView.getUint32(o + 12, true)
    const aux = tableView.getUint32(o + 16, true)
    if (length === 0) continue
    entries.set(key, {
      packId,
      offset,
      length,
      tplId: (meta >>> 24) & 0xff,
      grainAmp: ((meta >>> 16) & 0xff) / 10,
      quality: (meta >>> 8) & 0xff,
      flags: meta & 0xff,
      aux,
    })
  }

  return {
    pack: { handle, bodyId, packId, tileSize, gutter, baseCols, baseRows, levelCount, templates },
    entries,
    chunk0: probeBytes.byteLength <= ECV3_CHUNK_BYTES ? probeBytes : undefined,
  }
}

export type DecodedHeight10 = {
  yPlane: Uint8Array<ArrayBuffer>
  bytesPerRow: number
  maxVal: number
}

export const ECV3_CHUNK_SHIFT = 16 // 64 KiB aligned HTTP Range chunks (matches header probe, cuts row-major over-fetch ~2.5-3x)
export const ECV3_CHUNK_BYTES = 1 << ECV3_CHUNK_SHIFT
const ECV3_CHUNK_CACHE_MAX = 512 // 32 MiB bounded LRU in-RAM chunk cache
const DEFAULT_MAX_CONCURRENT_FETCHES = 12
const DEFAULT_MAX_CONCURRENT_PREFETCHES = 6
export const ECV3_PREFETCH_PRIORITY = 200

type ChunkWaiter = {
  signal: AbortSignal
  resolve: (data: Uint8Array<ArrayBuffer>) => void
  reject: (err: unknown) => void
  cleanup: () => void
}

type QueuedChunk = {
  key: string
  handle: PackHandle
  chunkIndex: number
  priority: number
  seq: number
  prefetch: boolean
  waiters: Set<ChunkWaiter>
}

export type ChunkRangeReader = (
  handle: PackHandle,
  start: number,
  length: number,
  signal: AbortSignal,
) => Promise<ArrayBuffer>

/**
 * Priority-gated 64 KiB aligned Range chunk cache for ECV3 packs.
 *
 * - Shares identical 64 KiB aligned edge-cache keys on Cloudflare CDN across users while avoiding
 *   the 6-8x row-major over-fetch of larger chunks.
 * - Caps concurrent network fetches (`maxConcurrentFetches`, default 6) in strict priority order so
 *   on-screen center tiles never starve behind offscreen margin or predictive prefetch chunks.
 * - RAM cache hits (`hasSlice` / `cache.get`) return synchronously without consuming a fetch slot.
 */
export class Ecv3ChunkCache {
  private readonly maxConcurrentFetches: number
  private readonly maxConcurrentPrefetches: number
  private readonly maxCachedChunks: number
  private readonly readRange: ChunkRangeReader
  private readonly cache = new Map<string, Uint8Array<ArrayBuffer>>()
  private readonly inflight = new Map<string, Promise<Uint8Array<ArrayBuffer>>>()
  private readonly queued = new Map<string, QueuedChunk>()
  private controller = new AbortController()
  private generation = 0
  private activeFetches = 0
  private activePrefetches = 0
  private seq = 0

  constructor({
    maxConcurrentFetches = DEFAULT_MAX_CONCURRENT_FETCHES,
    maxConcurrentPrefetches = DEFAULT_MAX_CONCURRENT_PREFETCHES,
    maxCachedChunks = ECV3_CHUNK_CACHE_MAX,
    readRange = readEcv3PackRange,
  }: {
    maxConcurrentFetches?: number
    maxConcurrentPrefetches?: number
    maxCachedChunks?: number
    readRange?: ChunkRangeReader
  } = {}) {
    this.maxConcurrentFetches = Math.max(1, maxConcurrentFetches)
    this.maxConcurrentPrefetches = Math.max(1, Math.min(this.maxConcurrentFetches, maxConcurrentPrefetches))
    this.maxCachedChunks = Math.max(1, maxCachedChunks)
    this.readRange = readRange
  }

  private touch(key: string, data: Uint8Array<ArrayBuffer>) {
    this.cache.delete(key)
    this.cache.set(key, data)
    while (this.cache.size > this.maxCachedChunks) {
      const oldest = this.cache.keys().next().value
      if (oldest === undefined) break
      this.cache.delete(oldest)
    }
  }

  /** Seed an already-fetched aligned chunk (e.g. chunk 0 from `loadEcv3PackIndex`) into RAM cache. */
  seedChunk(packKey: string, chunkIndex: number, bytes: Uint8Array<ArrayBuffer>) {
    const key = `${packKey}:${chunkIndex}`
    if (!this.cache.has(key)) this.touch(key, bytes)
  }

  /** True if every 64 KiB chunk covering `[offset, offset + length)` is already resident in RAM. */
  hasSlice(packKey: string, offset: number, length: number): boolean {
    if (length <= 0) return true
    const firstChunk = Math.floor(offset / ECV3_CHUNK_BYTES)
    const lastChunk = Math.floor((offset + length - 1) / ECV3_CHUNK_BYTES)
    for (let c = firstChunk; c <= lastChunk; c++) {
      if (!this.cache.has(`${packKey}:${c}`)) return false
    }
    return true
  }

  private attachInflight(
    promise: Promise<Uint8Array<ArrayBuffer>>,
    signal: AbortSignal,
  ): Promise<Uint8Array<ArrayBuffer>> {
    if (signal.aborted) return Promise.reject(signal.reason)
    return new Promise<Uint8Array<ArrayBuffer>>((resolve, reject) => {
      const onAbort = () => reject(signal.reason)
      signal.addEventListener('abort', onAbort, { once: true })
      promise.then(
        val => {
          signal.removeEventListener('abort', onAbort)
          resolve(val)
        },
        err => {
          signal.removeEventListener('abort', onAbort)
          reject(err)
        },
      )
    })
  }

  private pumpFetches() {
    while (this.activeFetches < this.maxConcurrentFetches && this.queued.size > 0) {
      let best: QueuedChunk | undefined
      for (const entry of this.queued.values()) {
        if (entry.waiters.size === 0 && !entry.prefetch) {
          this.queued.delete(entry.key)
          continue
        }
        const isPrefetchOnly = entry.waiters.size === 0 && entry.prefetch
        if (isPrefetchOnly && this.activePrefetches >= this.maxConcurrentPrefetches) continue
        if (!best || entry.priority < best.priority || (entry.priority === best.priority && entry.seq < best.seq)) {
          best = entry
        }
      }
      if (!best) return
      this.queued.delete(best.key)
      const isPrefetchOnly = best.waiters.size === 0 && best.prefetch
      this.activeFetches++
      if (isPrefetchOnly) this.activePrefetches++
      const { key, handle, chunkIndex, waiters } = best
      const gen = this.generation
      const start = chunkIndex * ECV3_CHUNK_BYTES
      const promise = this.readRange(handle, start, ECV3_CHUNK_BYTES, this.controller.signal).then(
        buf => {
          const bytes = new Uint8Array(buf)
          if (this.inflight.get(key) === promise) this.inflight.delete(key)
          if (gen === this.generation) this.touch(key, bytes)
          return bytes
        },
        err => {
          if (this.inflight.get(key) === promise) this.inflight.delete(key)
          throw err
        },
      )
      this.inflight.set(key, promise)
      promise
        .then(
          bytes => {
            for (const w of waiters) {
              w.cleanup()
              if (!w.signal.aborted) w.resolve(bytes)
            }
            waiters.clear()
          },
          err => {
            for (const w of waiters) {
              w.cleanup()
              if (!w.signal.aborted) w.reject(err)
            }
            waiters.clear()
          },
        )
        .finally(() => {
          this.activeFetches--
          if (isPrefetchOnly) this.activePrefetches--
          this.pumpFetches()
        })
    }
  }

  private getChunk(
    handle: PackHandle,
    packKey: string,
    chunkIndex: number,
    signal: AbortSignal,
    priority: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    if (signal.aborted) return Promise.reject(signal.reason)
    const key = `${packKey}:${chunkIndex}`
    const hit = this.cache.get(key)
    if (hit) {
      this.touch(key, hit)
      return Promise.resolve(hit)
    }
    const active = this.inflight.get(key)
    if (active) return this.attachInflight(active, signal)

    let entry = this.queued.get(key)
    if (entry) {
      if (priority < entry.priority) entry.priority = priority
      entry.prefetch = false
    } else {
      entry = { key, handle, chunkIndex, priority, seq: this.seq++, prefetch: false, waiters: new Set() }
      this.queued.set(key, entry)
    }
    const queuedEntry = entry
    const result = new Promise<Uint8Array<ArrayBuffer>>((resolve, reject) => {
      const onAbort = () => {
        queuedEntry.waiters.delete(waiter)
        if (queuedEntry.waiters.size === 0 && !queuedEntry.prefetch && this.queued.get(key) === queuedEntry) {
          this.queued.delete(key)
        }
        reject(signal.reason)
      }
      const waiter: ChunkWaiter = {
        signal,
        resolve,
        reject,
        cleanup: () => signal.removeEventListener('abort', onAbort),
      }
      queuedEntry.waiters.add(waiter)
      signal.addEventListener('abort', onAbort, { once: true })
    })
    this.pumpFetches()
    return result
  }

  /**
   * Warm the 64 KiB chunks covering `[offset, offset + length)` into RAM (and browser immutable cache)
   * in the background at low priority, without allocating GPU tile slots or blocking foreground reads.
   */
  prefetchSlice(
    handle: PackHandle,
    packKey: string,
    offset: number,
    length: number,
    priority = ECV3_PREFETCH_PRIORITY,
  ) {
    if (length <= 0 || handle.kind === 'blob' || handle.fullBlob) return
    const firstChunk = Math.floor(offset / ECV3_CHUNK_BYTES)
    const lastChunk = Math.floor((offset + length - 1) / ECV3_CHUNK_BYTES)
    let added = false
    for (let c = firstChunk; c <= lastChunk; c++) {
      const key = `${packKey}:${c}`
      const hit = this.cache.get(key)
      if (hit) {
        this.touch(key, hit)
        continue
      }
      if (this.inflight.has(key)) continue
      const existing = this.queued.get(key)
      if (existing) {
        if (priority < existing.priority) existing.priority = priority
        continue
      }
      this.queued.set(key, {
        key,
        handle,
        chunkIndex: c,
        priority,
        seq: this.seq++,
        prefetch: true,
        waiters: new Set(),
      })
      added = true
    }
    if (added) this.pumpFetches()
  }

  /** Drop queued low-priority background prefetch chunks (e.g. when the user starts panning/rotating). */
  cancelLowPriority(minPriority = ECV3_PREFETCH_PRIORITY) {
    if (this.queued.size === 0) return
    for (const [key, entry] of this.queued) {
      if (entry.waiters.size === 0 && entry.prefetch && entry.priority >= minPriority) {
        this.queued.delete(key)
      }
    }
  }

  async readSlice(
    handle: PackHandle,
    packKey: string,
    offset: number,
    length: number,
    signal: AbortSignal,
    priority = 0,
  ): Promise<Uint8Array<ArrayBuffer>> {
    if (length <= 0) return new Uint8Array(0)
    if (handle.kind === 'blob' || handle.fullBlob) {
      const buf = await this.readRange(handle, offset, length, signal)
      if (buf.byteLength !== length) {
        throw new Error(`Truncated ECV3 tile slice (${buf.byteLength} < ${length}).`)
      }
      return new Uint8Array(buf)
    }
    const firstChunk = Math.floor(offset / ECV3_CHUNK_BYTES)
    const lastChunk = Math.floor((offset + length - 1) / ECV3_CHUNK_BYTES)
    if (firstChunk === lastChunk) {
      const chunk = await this.getChunk(handle, packKey, firstChunk, signal, priority)
      const localOff = offset - firstChunk * ECV3_CHUNK_BYTES
      if (localOff < 0 || localOff + length > chunk.byteLength) {
        throw new Error(`Truncated ECV3 chunk (${chunk.byteLength} < ${localOff + length}).`)
      }
      return chunk.subarray(localOff, localOff + length)
    }
    const chunks = await Promise.all(
      Array.from({ length: lastChunk - firstChunk + 1 }, (_, i) =>
        this.getChunk(handle, packKey, firstChunk + i, signal, priority),
      ),
    )
    const out = new Uint8Array(length)
    let written = 0
    for (let i = 0; i < chunks.length; i++) {
      const chunkStart = (firstChunk + i) * ECV3_CHUNK_BYTES
      const sliceStart = Math.max(0, offset - chunkStart)
      const sliceEnd = Math.min(chunks[i].byteLength, offset + length - chunkStart)
      if (sliceEnd <= sliceStart) continue
      const part = chunks[i].subarray(sliceStart, sliceEnd)
      out.set(part, written)
      written += part.byteLength
    }
    if (written !== length) {
      throw new Error(`Truncated ECV3 multi-chunk slice (${written} < ${length}).`)
    }
    return out
  }

  /** Drop queued (not-yet-dispatched) network fetches without aborting already in-flight chunks. */
  cancelQueued() {
    if (this.queued.size === 0) return
    const err = new DOMException('Chunk fetch cancelled', 'AbortError')
    for (const entry of this.queued.values()) {
      for (const w of entry.waiters) {
        w.cleanup()
        if (!w.signal.aborted) w.reject(err)
      }
      entry.waiters.clear()
    }
    this.queued.clear()
  }

  clear() {
    this.generation++
    this.cancelQueued()
    this.controller.abort()
    this.controller = new AbortController()
    this.cache.clear()
    this.inflight.clear()
  }
}

class SingleAv1Height10Decoder {
  private decoder: VideoDecoder | null = null
  private supported: boolean | null = null
  private activeJob: { deliver: (frame: VideoFrame | null) => void } | null = null
  private chain: Promise<void> = Promise.resolve()
  private ts = 0
  private destroyed = false
  private readonly tileSize: number

  constructor(tileSize = VIRTUAL_TILE_SIZE) {
    this.tileSize = tileSize
  }

  private ensureDecoder(): VideoDecoder | null {
    if (this.destroyed || this.supported === false) return null
    if (typeof VideoDecoder === 'undefined' || typeof EncodedVideoChunk === 'undefined') {
      this.supported = false
      return null
    }
    if (!this.decoder || this.decoder.state === 'closed') {
      try {
        const vd = new VideoDecoder({
          output: frame => {
            const job = this.activeJob
            if (job) {
              this.activeJob = null
              job.deliver(frame)
            } else {
              frame.close()
            }
          },
          error: () => {
            const job = this.activeJob
            this.activeJob = null
            job?.deliver(null)
            if (this.decoder === vd) this.decoder = null
          },
        })
        vd.configure({
          codec: 'av01.0.04M.10',
          codedWidth: this.tileSize,
          codedHeight: this.tileSize,
          // Hardware AV1 decoders often reject monochrome YUV400 10-bit or return opaque GPU VideoFrames;
          // prefer-software routes directly to Chromium's libdav1d (<0.4 ms/tile, exact 10-bit I420P10).
          hardwareAcceleration: 'prefer-software',
        })
        this.decoder = vd
        this.supported = true
      } catch {
        this.supported = false
        return null
      }
    }
    return this.decoder
  }

  async decode(obuBytes: Uint8Array, signal: AbortSignal): Promise<DecodedHeight10 | null> {
    if (this.destroyed || signal.aborted || this.supported === false) return null
    let result: DecodedHeight10 | null = null
    const run = async () => {
      if (this.destroyed || signal.aborted) return
      const vd = this.ensureDecoder()
      if (!vd) return
      let frame: VideoFrame | null = null
      let closed = false
      const job = {
        deliver: (next: VideoFrame | null) => {
          if (!next) return
          if (closed || this.destroyed || signal.aborted || frame) {
            next.close()
            return
          }
          frame = next
        },
      }
      try {
        this.activeJob = job
        vd.decode(new EncodedVideoChunk({ type: 'key', timestamp: this.ts++, data: obuBytes }))
        // WebCodecs guarantees all outputs for prior decode() calls are emitted before flush() resolves.
        // Never await a separate unbounded output promise after flush(), since a zero-output flush is valid.
        await vd.flush()
        if (this.activeJob === job) this.activeJob = null
        if (!frame || this.destroyed || signal.aborted) return
        const decodedFrame: VideoFrame = frame
        const fmt: string = decodedFrame.format ?? ''
        const is10Bit = fmt.includes('P10') || fmt.includes('10')
        if (!is10Bit) return
        const maxVal = fmt === 'P010' ? 65472 : 1023
        const allocSize = decodedFrame.allocationSize()
        const buf = new ArrayBuffer(allocSize)
        const layout = await decodedFrame.copyTo(buf)
        if (this.destroyed || signal.aborted || !layout.length) return
        const stride = layout[0].stride
        const off = layout[0].offset
        const rowBytes = this.tileSize * 2
        if (stride >= rowBytes && stride % 256 === 0) {
          // Zero-copy view over our freshly allocated ArrayBuffer when stride is already 256B-aligned.
          result = { yPlane: new Uint8Array(buf, off, stride * this.tileSize), bytesPerRow: stride, maxVal }
        } else {
          const alignedStride = Math.ceil(rowBytes / 256) * 256
          const owned = new Uint8Array(alignedStride * this.tileSize)
          const src = new Uint8Array(buf)
          for (let y = 0; y < this.tileSize; y++) {
            owned.set(src.subarray(off + y * stride, off + y * stride + rowBytes), y * alignedStride)
          }
          result = { yPlane: owned, bytesPerRow: alignedStride, maxVal }
        }
      } catch {
        // Degrade gracefully to 8-bit AVIF fallback on WebCodecs flush/decode error.
      } finally {
        closed = true
        if (this.activeJob === job) this.activeJob = null
        const toClose = frame as VideoFrame | null
        frame = null
        toClose?.close()
      }
    }
    const next = this.chain.then(run, run)
    this.chain = next
    await next
    return result
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    const job = this.activeJob
    this.activeJob = null
    job?.deliver(null)
    if (this.decoder && this.decoder.state !== 'closed') {
      try {
        this.decoder.close()
      } catch {
        // ignore close errors on shutdown
      }
    }
    this.decoder = null
  }
}

/**
 * Parallel pool of WebCodecs VideoDecoder('av01.0.04M.10') instances so 10-bit elevation
 * tiles decode concurrently without serializing behind flush().
 */
export class Av1Height10Decoder {
  private readonly workers: SingleAv1Height10Decoder[]
  private cursor = 0

  constructor(tileSize = VIRTUAL_TILE_SIZE, poolSize = 4) {
    this.workers = Array.from({ length: poolSize }, () => new SingleAv1Height10Decoder(tileSize))
  }

  decode(obuBytes: Uint8Array, signal: AbortSignal): Promise<DecodedHeight10 | null> {
    const worker = this.workers[this.cursor++ % this.workers.length]
    return worker.decode(obuBytes, signal)
  }

  destroy() {
    for (const w of this.workers) w.destroy()
  }
}

const UNPACK_U16_WGSL = /* wgsl */ `
struct UnpackParams {
  spanM: f32,
  invMaxVal: f32,
  tileSize: u32,
  _pad0: u32,
}
@group(0) @binding(0) var srcTex: texture_2d<u32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> p: UnpackParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = i32(p.tileSize);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= s || y >= s) { return; }
  let h = f32(textureLoad(srcTex, vec2i(x, y), 0).r) * p.invMaxVal * p.spanM;
  textureStore(dstTex, vec2i(x, y), vec4f(h, 0.0, 0.0, 0.0));
}
`

const UNPACK_F32_WGSL = /* wgsl */ `
struct UnpackParams {
  spanM: f32,
  invMaxVal: f32,
  tileSize: u32,
  _pad0: u32,
}
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> p: UnpackParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = i32(p.tileSize);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= s || y >= s) { return; }
  let h = textureLoad(srcTex, vec2i(x, y), 0).r * p.spanM;
  textureStore(dstTex, vec2i(x, y), vec4f(h, 0.0, 0.0, 0.0));
}
`

// Pass A of directional LOLA destriper: soft-clamped horizontal high-pass (isolates narrow N-S tracks
// without letting steep crater rims create vertical halos).
const DESTRIPE_HP_X_WGSL = /* wgsl */ `
struct DestripeParams {
  radius: i32,
  tileSize: i32,
  clampM: f32,
  _pad0: f32,
}
@group(0) @binding(0) var heightM: texture_2d<f32>;
@group(0) @binding(1) var hpOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> p: DestripeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= p.tileSize || y >= p.tileSize) { return; }
  let maxI = p.tileSize - 1;
  let c = textureLoad(heightM, vec2i(x, y), 0).r;
  var sum = 0.0;
  for (var dx = -p.radius; dx <= p.radius; dx++) {
    sum += textureLoad(heightM, vec2i(clamp(x + dx, 0, maxI), y), 0).r;
  }
  let hp = c - sum / f32(2 * p.radius + 1);
  let clamped = p.clampM * tanh(hp / max(p.clampM, 1e-3));
  textureStore(hpOut, vec2i(x, y), vec4f(clamped, 0.0, 0.0, 0.0));
}
`

// Pass B of directional LOLA destriper: subtract vertical column mean of clamped horizontal high-pass.
const DESTRIPE_SUB_Y_WGSL = /* wgsl */ `
struct DestripeParams {
  radius: i32,
  tileSize: i32,
  clampM: f32,
  _pad0: f32,
}
@group(0) @binding(0) var heightM: texture_2d<f32>;
@group(0) @binding(1) var hpIn: texture_2d<f32>;
@group(0) @binding(2) var cleanOut: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> p: DestripeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= p.tileSize || y >= p.tileSize) { return; }
  let maxI = p.tileSize - 1;
  var sum = 0.0;
  for (var dy = -p.radius; dy <= p.radius; dy++) {
    sum += textureLoad(hpIn, vec2i(x, clamp(y + dy, 0, maxI)), 0).r;
  }
  let stripe = sum / f32(2 * p.radius + 1);
  // Smoothly taper destripe subtraction to 0 at tile edges so horizontal/vertical clamp windows
  // never diverge across adjacent tile seams (GUTTER = 2).
  let edgeX = smoothstep(2.0, 10.0, f32(min(x, maxI - x)));
  let edgeY = smoothstep(2.0, 16.0, f32(min(y, maxI - y)));
  let clean = textureLoad(heightM, vec2i(x, y), 0).r - stripe * edgeX * edgeY;
  textureStore(cleanOut, vec2i(x, y), vec4f(clean, 0.0, 0.0, 0.0));
}
`

const HEIGHT_METERS_TO_NORMAL_WGSL = /* wgsl */ `
struct HeightNormalParams {
  relief: f32,
  dyM: f32,
  dxEqM: f32,
  latTopRad: f32,
  dLatPerPx: f32,
  tileSize: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var heightM: texture_2d<f32>;
@group(0) @binding(1) var outNormal: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> p: HeightNormalParams;

// 5x5 Gaussian-derivative kernel (wSmooth=[1,4,6,4,1]/16, wDeriv=[-1,-2,0,2,1]/8 => scale 1/128).
// Fits strictly within GUTTER=2 for all inner pixels [2..509] (zero tile seams).
const W_SMOOTH = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
const W_DERIV  = array<f32, 5>(-1.0, -2.0, 0.0, 2.0, 1.0);

fn readH(x: i32, y: i32, maxI: i32) -> f32 {
  return textureLoad(heightM, vec2i(clamp(x, 0, maxI), clamp(y, 0, maxI)), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let sizeI = i32(p.tileSize);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= sizeI || y >= sizeI) { return; }
  let maxI = sizeI - 1;

  var gx = 0.0;
  var gy = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    let wsY = W_SMOOTH[dy + 2];
    let wdY = W_DERIV[dy + 2];
    for (var dx = -2; dx <= 2; dx++) {
      let h = readH(x + dx, y + dy, maxI);
      gx += h * W_DERIV[dx + 2] * wsY;
      gy += h * W_SMOOTH[dx + 2] * wdY;
    }
  }
  let dhEast  = gx * 0.0078125 * p.relief;
  let dhSouth = gy * 0.0078125 * p.relief;

  let lat = p.latTopRad - (f32(y) + 0.5) * p.dLatPerPx;
  let dxM = max(p.dxEqM * max(cos(lat), 0.08), 1.0);
  let dyM = max(p.dyM, 1.0);

  let tn = normalize(vec3f(-dhEast / dxM, dhSouth / dyM, 1.0));
  textureStore(outNormal, vec2i(x, y), vec4f(tn.x * 0.5 + 0.5, tn.y * 0.5 + 0.5, max(0.0, tn.z), 1.0));
}
`

type DeferredHeightPipelines = {
  unpackU16: DeferredComputePipeline
  unpackF32: DeferredComputePipeline
  destripeHpX: DeferredComputePipeline
  destripeSubY: DeferredComputePipeline
  toNormal: DeferredComputePipeline
}

type HeightPipelines = {
  unpackU16: GPUComputePipeline
  unpackF32: GPUComputePipeline
  destripeHpX: GPUComputePipeline
  destripeSubY: GPUComputePipeline
  toNormal: GPUComputePipeline
}

const deferredHeightPipelinesCache = new WeakMap<GPUDevice, DeferredHeightPipelines>()

/** Queue async compilation of ECV3 height-to-normal compute pipelines before the first tile arrives. */
export function warmHeightPipelines(device: GPUDevice): DeferredHeightPipelines {
  let handles = deferredHeightPipelinesCache.get(device)
  if (!handles) {
    const queue = getPipelineCompileQueue(device)
    const reg = (order: number, label: string, code: string) =>
      queue.registerCompute(order, () => ({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
      }))
    handles = {
      unpackU16: reg(16, 'ECV3 unpack r16uint height', UNPACK_U16_WGSL),
      toNormal: reg(17, 'ECV3 height-meters to normal', HEIGHT_METERS_TO_NORMAL_WGSL),
      unpackF32: reg(72, 'ECV3 unpack f32 height', UNPACK_F32_WGSL),
      destripeHpX: reg(73, 'ECV3 LOLA destripe hpX', DESTRIPE_HP_X_WGSL),
      destripeSubY: reg(74, 'ECV3 LOLA destripe subY', DESTRIPE_SUB_Y_WGSL),
    }
    deferredHeightPipelinesCache.set(device, handles)
  }
  return handles
}

function getHeightPipelines(device: GPUDevice): HeightPipelines {
  const handles = warmHeightPipelines(device)
  return {
    unpackU16: handles.unpackU16.get(),
    unpackF32: handles.unpackF32.get(),
    destripeHpX: handles.destripeHpX.get(),
    destripeSubY: handles.destripeSubY.get(),
    toNormal: handles.toNormal.get(),
  }
}

/**
 * Reusable GPU compute converter: unpacks a true 10-bit r16uint height plane (or fallback
 * ImageBitmap) into meters, optionally runs a two-scale directional notch destriper to remove
 * LOLA polar-orbit laser tracks on the Moon without blurring craters, and emits a physically
 * scaled, seam-free 512x512 tangent-space normal map in ~0.12 ms.
 */
export class HeightToNormalEncoder {
  private readonly device: GPUDevice
  private readonly pipelines: HeightPipelines
  private readonly tileSize: number
  private readonly heightU16Scratch: GPUTexture
  private readonly heightF32Scratch: GPUTexture
  private readonly workA: GPUTexture
  private readonly workB: GPUTexture
  private readonly workC: GPUTexture
  private readonly normalScratch: GPUTexture
  private readonly unpackParamBuffer: GPUBuffer
  private readonly normalParamBuffer: GPUBuffer
  private readonly destripeParamBuffer: GPUBuffer
  private readonly unpackU16BG: GPUBindGroup
  private readonly unpackF32BG: GPUBindGroup
  private readonly hpX1BG: GPUBindGroup
  private readonly subY1BG: GPUBindGroup
  private readonly hpX2BG: GPUBindGroup
  private readonly subY2BG: GPUBindGroup
  private readonly toNormalBG: GPUBindGroup
  private readonly unpackData = new ArrayBuffer(16)
  private readonly unpackF32 = new Float32Array(this.unpackData)
  private readonly unpackU32 = new Uint32Array(this.unpackData)
  private readonly normalData = new ArrayBuffer(32)
  private readonly normalF32 = new Float32Array(this.normalData)
  private readonly normalU32 = new Uint32Array(this.normalData)
  private destroyed = false

  constructor(device: GPUDevice, tileSize = VIRTUAL_TILE_SIZE) {
    this.device = device
    this.tileSize = tileSize
    this.pipelines = getHeightPipelines(device)
    this.heightU16Scratch = device.createTexture({
      label: 'ECV3 height10 r16uint scratch',
      size: [tileSize, tileSize],
      format: 'r16uint',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.heightF32Scratch = device.createTexture({
      label: 'ECV3 height fallback bitmap scratch',
      size: [tileSize, tileSize],
      format: 'rgba16float',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    const makeWork = (label: string) =>
      device.createTexture({
        label,
        size: [tileSize, tileSize],
        format: 'r32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      })
    this.workA = makeWork('ECV3 height meters workA')
    this.workB = makeWork('ECV3 destripe hp scratch')
    this.workC = makeWork('ECV3 height meters workC')
    this.normalScratch = device.createTexture({
      label: 'ECV3 derived normal scratch',
      size: [tileSize, tileSize],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.unpackParamBuffer = device.createBuffer({
      label: 'ECV3 height unpack params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.normalParamBuffer = device.createBuffer({
      label: 'ECV3 height-to-normal params',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Static two-scale LOLA destripe params (4 uniform slots aligned to minUniformBufferOffsetAlignment).
    const stride = Math.max(256, device.limits.minUniformBufferOffsetAlignment)
    this.destripeParamBuffer = device.createBuffer({
      label: 'ECV3 LOLA destripe params',
      size: stride * 4,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    })
    const mapped = this.destripeParamBuffer.getMappedRange()
    const writeSlot = (slot: number, radius: number, clampM: number) => {
      const i32 = new Int32Array(mapped, slot * stride, 4)
      const f32 = new Float32Array(mapped, slot * stride, 4)
      i32[0] = radius
      i32[1] = tileSize
      f32[2] = clampM
      f32[3] = 0
    }
    writeSlot(0, 6, 18.0) // Scale 1 hpX: catches 4-10px track steps
    writeSlot(1, 20, 18.0) // Scale 1 subY: 41px vertical track window
    writeSlot(2, 3, 10.0) // Scale 2 hpX: catches fine 1-3px laser scratches
    writeSlot(3, 12, 10.0) // Scale 2 subY: 25px vertical track window
    this.destripeParamBuffer.unmap()

    const vA = this.workA.createView()
    const vB = this.workB.createView()
    const vC = this.workC.createView()

    this.unpackU16BG = device.createBindGroup({
      layout: this.pipelines.unpackU16.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.heightU16Scratch.createView() },
        { binding: 1, resource: vA },
        { binding: 2, resource: { buffer: this.unpackParamBuffer } },
      ],
    })
    this.unpackF32BG = device.createBindGroup({
      layout: this.pipelines.unpackF32.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.heightF32Scratch.createView() },
        { binding: 1, resource: vA },
        { binding: 2, resource: { buffer: this.unpackParamBuffer } },
      ],
    })
    this.hpX1BG = device.createBindGroup({
      layout: this.pipelines.destripeHpX.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: vA },
        { binding: 1, resource: vB },
        { binding: 2, resource: { buffer: this.destripeParamBuffer, offset: 0, size: 16 } },
      ],
    })
    this.subY1BG = device.createBindGroup({
      layout: this.pipelines.destripeSubY.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: vA },
        { binding: 1, resource: vB },
        { binding: 2, resource: vC },
        { binding: 3, resource: { buffer: this.destripeParamBuffer, offset: stride, size: 16 } },
      ],
    })
    this.hpX2BG = device.createBindGroup({
      layout: this.pipelines.destripeHpX.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: vC },
        { binding: 1, resource: vB },
        { binding: 2, resource: { buffer: this.destripeParamBuffer, offset: stride * 2, size: 16 } },
      ],
    })
    this.subY2BG = device.createBindGroup({
      layout: this.pipelines.destripeSubY.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: vC },
        { binding: 1, resource: vB },
        { binding: 2, resource: vA },
        { binding: 3, resource: { buffer: this.destripeParamBuffer, offset: stride * 3, size: 16 } },
      ],
    })
    this.toNormalBG = device.createBindGroup({
      layout: this.pipelines.toNormal.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: vA },
        { binding: 1, resource: this.normalScratch.createView() },
        { binding: 2, resource: { buffer: this.normalParamBuffer } },
      ],
    })
  }

  get outputTexture(): GPUTexture {
    return this.normalScratch
  }

  private runTail(
    encoder: GPUCommandEncoder,
    relief: number,
    radiusMeters: number,
    cols: number,
    rows: number,
    ty: number,
    destripe: boolean,
  ) {
    const wg = Math.ceil(this.tileSize / 8)
    if (destripe) {
      const p1 = encoder.beginComputePass({ label: 'ECV3 LOLA destripe scale1 hpX' })
      p1.setPipeline(this.pipelines.destripeHpX)
      p1.setBindGroup(0, this.hpX1BG)
      p1.dispatchWorkgroups(wg, wg)
      p1.end()

      const p2 = encoder.beginComputePass({ label: 'ECV3 LOLA destripe scale1 subY' })
      p2.setPipeline(this.pipelines.destripeSubY)
      p2.setBindGroup(0, this.subY1BG)
      p2.dispatchWorkgroups(wg, wg)
      p2.end()

      const p3 = encoder.beginComputePass({ label: 'ECV3 LOLA destripe scale2 hpX' })
      p3.setPipeline(this.pipelines.destripeHpX)
      p3.setBindGroup(0, this.hpX2BG)
      p3.dispatchWorkgroups(wg, wg)
      p3.end()

      const p4 = encoder.beginComputePass({ label: 'ECV3 LOLA destripe scale2 subY' })
      p4.setPipeline(this.pipelines.destripeSubY)
      p4.setBindGroup(0, this.subY2BG)
      p4.dispatchWorkgroups(wg, wg)
      p4.end()
    }

    const totalY = rows * VIRTUAL_TILE_INNER
    const totalX = cols * VIRTUAL_TILE_INNER
    const dLatPerPx = Math.PI / totalY
    const latTopRad = Math.PI * 0.5 - (ty * VIRTUAL_TILE_INNER - VIRTUAL_TILE_GUTTER) * dLatPerPx
    this.normalF32[0] = relief
    this.normalF32[1] = (Math.PI * radiusMeters) / totalY
    this.normalF32[2] = (2 * Math.PI * radiusMeters) / totalX
    this.normalF32[3] = latTopRad
    this.normalF32[4] = dLatPerPx
    this.normalU32[5] = this.tileSize
    this.normalU32[6] = 0
    this.normalU32[7] = 0
    this.device.queue.writeBuffer(this.normalParamBuffer, 0, this.normalData)

    const pn = encoder.beginComputePass({ label: 'ECV3 height-to-normal' })
    pn.setPipeline(this.pipelines.toNormal)
    pn.setBindGroup(0, this.toNormalBG)
    pn.dispatchWorkgroups(wg, wg)
    pn.end()
  }

  /** True 10-bit path: upload native r16uint Y-plane from WebCodecs VideoDecoder. */
  encodeFromHeight10(
    encoder: GPUCommandEncoder,
    decoded: DecodedHeight10,
    spanM: number,
    relief: number,
    radiusMeters: number,
    cols: number,
    rows: number,
    ty: number,
    destripe = false,
  ) {
    if (this.destroyed) return
    this.device.queue.writeTexture(
      { texture: this.heightU16Scratch },
      decoded.yPlane,
      { bytesPerRow: decoded.bytesPerRow, rowsPerImage: this.tileSize },
      [this.tileSize, this.tileSize],
    )
    this.unpackF32[0] = spanM
    this.unpackF32[1] = 1 / Math.max(1, decoded.maxVal)
    this.unpackU32[2] = this.tileSize
    this.unpackU32[3] = 0
    this.device.queue.writeBuffer(this.unpackParamBuffer, 0, this.unpackData)

    const wg = Math.ceil(this.tileSize / 8)
    const pu = encoder.beginComputePass({ label: 'ECV3 unpack height10' })
    pu.setPipeline(this.pipelines.unpackU16)
    pu.setBindGroup(0, this.unpackU16BG)
    pu.dispatchWorkgroups(wg, wg)
    pu.end()

    this.runTail(encoder, relief, radiusMeters, cols, rows, ty, destripe)
  }

  /** Fallback path when WebCodecs VideoDecoder is unavailable. */
  encodeFromBitmap(
    encoder: GPUCommandEncoder,
    heightBitmap: ImageBitmap,
    spanM: number,
    relief: number,
    radiusMeters: number,
    cols: number,
    rows: number,
    ty: number,
    destripe = false,
  ) {
    if (this.destroyed) return
    this.device.queue.copyExternalImageToTexture({ source: heightBitmap }, { texture: this.heightF32Scratch }, [
      this.tileSize,
      this.tileSize,
    ])
    this.unpackF32[0] = spanM
    this.unpackF32[1] = 1
    this.unpackU32[2] = this.tileSize
    this.unpackU32[3] = 0
    this.device.queue.writeBuffer(this.unpackParamBuffer, 0, this.unpackData)

    const wg = Math.ceil(this.tileSize / 8)
    const pu = encoder.beginComputePass({ label: 'ECV3 unpack heightBitmap' })
    pu.setPipeline(this.pipelines.unpackF32)
    pu.setBindGroup(0, this.unpackF32BG)
    pu.dispatchWorkgroups(wg, wg)
    pu.end()

    this.runTail(encoder, relief, radiusMeters, cols, rows, ty, destripe)
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.heightU16Scratch.destroy()
    this.heightF32Scratch.destroy()
    this.workA.destroy()
    this.workB.destroy()
    this.workC.destroy()
    this.normalScratch.destroy()
    this.unpackParamBuffer.destroy()
    this.normalParamBuffer.destroy()
    this.destripeParamBuffer.destroy()
  }
}
