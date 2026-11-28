import { raceAbort } from '../shader-test/gpu.ts'
import { workspaceAssets, type WorkspaceAssets } from './workspaceAssets.ts'

const manifestUrl = 'https://earth-connect-assets.pages.dev/manifest.json'
const scorePath = 'compute-data/world/manifest.json'
const world = 'compute-data/world/'
const renderPaths = [
  'textures/earth-blue-marble.jpg',
  'textures/earth-terrain.bin',
  'textures/moon-terrain.bin',
  'textures/mars-terrain.bin',
  'textures/nasa-deep-star-map.webp',
]
const legacyTilePaths = [
  ...Array.from({ length: 8 }, (_, index) => `textures/earth-tiles-${index}.bin`),
  ...Array.from({ length: 4 }, (_, index) => `textures/moon-tiles-${index}.bin`),
  ...Array.from({ length: 4 }, (_, index) => `textures/mars-tiles-${index}.bin`),
]
const optionalRenderPaths = new Set(['textures/moon-albedo.webp', 'textures/mars-albedo.webp'])
// Legacy CDN releases still list removed basins, pre-baked BC7 textures, and superseded ECVT1
// zoom tile packs (replaced by HTTP-streamed ECV3 packs on R2). Ignoring them lets pruneCache()
// reclaim up to ~175 MB of dead tile blobs from existing IndexedDB caches.
const ignoredPaths = new Set([
  'data/surface-basins.bin',
  'textures/nasa-deep-star-map.bc7',
  'textures/mars-albedo-4096.bc7',
  'textures/moon-albedo-512.bc7',
  ...legacyTilePaths,
])
const hashPattern = /^[a-f0-9]{64}$/
type Asset = { key: string; bytes: number; sha256: string; contentType: string }
type Release = { etag: string | null; files: Record<string, Asset>; bytes: number }
export type DownloadProgress = { downloaded: number; total: number; fallback: boolean }

function parseRelease(value: unknown, etag: string | null): Release {
  if (!value || typeof value !== 'object') throw new Error('Invalid Earth data release.')
  const { format, runtimeFileCount, runtimeBytes, scoreManifest, assets } = value as {
    format?: unknown
    runtimeFileCount?: unknown
    runtimeBytes?: unknown
    scoreManifest?: Partial<Asset>
    assets?: Record<string, unknown>
  }
  if (
    format !== 'earth-connect-assets-v1' ||
    !assets ||
    typeof assets !== 'object' ||
    !renderPaths.every(path => Object.hasOwn(assets, path)) ||
    !Object.hasOwn(assets, scorePath)
  ) {
    throw new Error('The Earth data release is missing required files.')
  }
  const files: Record<string, Asset> = Object.create(null)
  let bytes = 0
  let manifestFiles = 0
  let manifestBytes = 0
  for (const [path, value] of Object.entries(assets)) {
    const item = value as Partial<Asset> | undefined
    const ignored = ignoredPaths.has(path)
    if (
      !ignored &&
      !renderPaths.includes(path) &&
      !optionalRenderPaths.has(path) &&
      path !== scorePath &&
      !/^compute-data\/world\/chunk-\d+\.zst$/.test(path)
    ) {
      throw new Error(`Unexpected Earth data file ${path}.`)
    }
    if (
      !item ||
      typeof item.sha256 !== 'string' ||
      !hashPattern.test(item.sha256) ||
      typeof item.bytes !== 'number' ||
      !Number.isSafeInteger(item.bytes) ||
      item.bytes <= 0 ||
      typeof item.contentType !== 'string'
    ) {
      throw new Error(`Invalid Earth data metadata for ${path}.`)
    }
    const dot = path.lastIndexOf('.')
    const prefix = path.startsWith(world) ? '' : 'assets/'
    if (item.key !== `${prefix}${path.slice(0, dot)}-${item.sha256}${path.slice(dot)}`) {
      throw new Error(`Invalid Earth data URL for ${path}.`)
    }
    manifestFiles++
    manifestBytes += item.bytes
    if (!ignored) {
      files[path] = { key: item.key, bytes: item.bytes, sha256: item.sha256, contentType: item.contentType }
      bytes += item.bytes
    }
  }
  if (
    scoreManifest?.key !== files[scorePath].key ||
    scoreManifest.sha256 !== files[scorePath].sha256 ||
    scoreManifest.bytes !== files[scorePath].bytes
  ) {
    throw new Error('The Earth scoring manifest does not match the release.')
  }
  if (manifestFiles !== runtimeFileCount || manifestBytes !== runtimeBytes || !Number.isSafeInteger(manifestBytes)) {
    throw new Error('The Earth data release has inconsistent file totals.')
  }
  return { etag, files, bytes }
}

function openCache() {
  return new Promise<IDBDatabase | undefined>(resolve => {
    try {
      let blocked = false
      const request = indexedDB.open('earth-connect-release-data', 1)
      request.onupgradeneeded = () => {
        request.result.createObjectStore('files')
        request.result.createObjectStore('releases')
      }
      request.onsuccess = () => {
        const database = request.result
        database.onversionchange = () => database.close()
        if (blocked) database.close()
        else resolve(database)
      }
      request.onerror = () => resolve(undefined)
      request.onblocked = () => {
        blocked = true
        resolve(undefined)
      }
    } catch {
      resolve(undefined)
    }
  })
}

function accessCache<T>(db: IDBDatabase | undefined, store: 'files' | 'releases', key: string, value?: T) {
  return new Promise<T | undefined>(resolve => {
    try {
      if (!db) return resolve(undefined)
      const transaction = db.transaction(store, value === undefined ? 'readonly' : 'readwrite')
      const objects = transaction.objectStore(store)
      const request = value === undefined ? objects.get(key) : objects.put(value, key)
      transaction.oncomplete = () => resolve(value ?? request.result)
      transaction.onerror = transaction.onabort = () => resolve(undefined)
    } catch {
      resolve(undefined)
    }
  })
}

function pruneCache(db: IDBDatabase, release: Release) {
  const keep = new Set(Object.values(release.files).map(asset => asset.sha256))
  return new Promise<void>(resolve => {
    try {
      const transaction = db.transaction('files', 'readwrite')
      const files = transaction.objectStore('files')
      const keys = files.openKeyCursor()
      keys.onsuccess = () => {
        const cursor = keys.result
        if (!cursor) return
        if (!keep.has(String(cursor.key))) files.delete(cursor.key)
        cursor.continue()
      }
      transaction.oncomplete = transaction.onerror = transaction.onabort = () => resolve()
    } catch {
      resolve()
    }
  })
}

async function downloadBytes(asset: Asset, signal: AbortSignal, count: (bytes: number) => void) {
  const lifetime = AbortSignal.any([signal, AbortSignal.timeout(300_000)])
  const response = await fetch(new URL(asset.key, manifestUrl), {
    signal: lifetime,
    cache: 'no-store',
    credentials: 'omit',
  })
  if (!response.ok || !response.body) throw new Error(`${asset.key} returned HTTP ${response.status}.`)
  const bytes = new Uint8Array(asset.bytes)
  const reader = response.body.getReader()
  let received = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (received + value.byteLength > bytes.length) throw new Error(`Unexpected size for ${asset.key}.`)
      bytes.set(value, received)
      received += value.byteLength
      count(value.byteLength)
    }
    if (received !== asset.bytes) throw new Error(`Incomplete download for ${asset.key}.`)
  } finally {
    await reader.cancel().catch(() => undefined)
    reader.releaseLock()
  }
  lifetime.throwIfAborted()
  return bytes
}

async function verifyAsset(asset: Asset, bytes: Uint8Array<ArrayBuffer>, signal: AbortSignal) {
  signal.throwIfAborted()
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
  if (hex !== asset.sha256) throw new Error(`Integrity check failed for ${asset.key}.`)
  signal.throwIfAborted()
  return new Blob([bytes], { type: asset.contentType })
}

async function validateScoreManifest(blob: Blob, worldFiles: ReadonlyMap<string, Asset>) {
  const score = await new Response(blob).json()
  const remaining = new Map(worldFiles)
  if (
    !Array.isArray(score?.chunks) ||
    score.chunks.length !== remaining.size ||
    score.chunks.length === 0 ||
    !score.chunks.every((chunk: { file?: unknown; compressedBytes?: unknown } | null) => {
      if (typeof chunk?.file !== 'string') return false
      const path = world + chunk.file
      const asset = remaining.get(path)
      if (!asset || asset.bytes !== chunk.compressedBytes) return false
      remaining.delete(path)
      return true
    })
  ) {
    throw new Error('The Earth scoring release is missing world chunks.')
  }
}

function releaseSource(release: Release, lifetime: AbortSignal) {
  const slots = new Map<string, { blob?: Blob; pending?: Promise<Blob>; resolve?: (blob: Blob) => void }>()
  const worldFiles = new Map<string, Asset>()
  for (const [path, asset] of Object.entries(release.files)) {
    if (ignoredPaths.has(path)) continue
    let resolve!: (blob: Blob) => void
    const pending = new Promise<Blob>(ready => {
      resolve = ready
    })
    const slot = { pending, resolve }
    slots.set(path, slot)
    if (path.startsWith(world) && path !== scorePath) {
      slots.set(asset.key, slot)
      worldFiles.set(asset.key, asset)
    }
  }
  const localDevUrlCache = import.meta.env.DEV ? new Map<string, Promise<string | undefined>>() : undefined
  const resolveLocalDevTextureUrl = (path: string, signal: AbortSignal) => {
    if (!localDevUrlCache || !path.startsWith('textures/')) return Promise.resolve(undefined)
    let pending = localDevUrlCache.get(path)
    if (!pending) {
      const url = `${import.meta.env.BASE_URL}${path}`
      pending = fetch(url, { method: 'HEAD', cache: 'no-store' })
        .then(res => (res.ok ? url : undefined))
        .catch(() => undefined)
      localDevUrlCache.set(path, pending)
    }
    return raceAbort(pending, signal)
  }
  const assets = workspaceAssets(
    async (path, signal) => {
      lifetime.throwIfAborted()
      const combined = AbortSignal.any([signal, lifetime])
      if (localDevUrlCache && path.startsWith('textures/')) {
        const direct = await resolveLocalDevTextureUrl(path, combined)
        if (direct) {
          const res = await fetch(direct, { signal: combined, cache: 'no-store' })
          if (res.ok) return await res.blob()
        }
      }
      const slot = slots.get(path)
      if (!slot) throw new Error(`Workspace data is missing ${path}.`)
      if (slot.blob) return slot.blob
      return raceAbort(slot.pending!, combined)
    },
    localDevUrlCache
      ? (path, signal) => resolveLocalDevTextureUrl(path, AbortSignal.any([signal, lifetime]))
      : undefined,
    path => release.files[path]?.sha256,
  )
  lifetime.addEventListener(
    'abort',
    () => {
      assets.dispose()
      slots.clear()
    },
    { once: true },
  )
  return {
    assets,
    async publish(path: string, blob: Blob) {
      if (path === scorePath) await validateScoreManifest(blob, worldFiles)
      lifetime.throwIfAborted()
      const slot = slots.get(path)!
      slot.blob = blob
      slot.resolve?.(blob)
      slot.pending = undefined
      slot.resolve = undefined
    },
    replaceBlob(path: string, blob: Blob) {
      const slot = slots.get(path)
      if (slot && slot.blob) slot.blob = blob
    },
  }
}

type ReleaseOptions = {
  refresh?: boolean
  onAssets?: (assets: WorkspaceAssets) => void
}

function readCachedBlobs(db: IDBDatabase | undefined, entries: readonly (readonly [string, Asset])[]) {
  return new Promise<Map<string, Blob>>(resolve => {
    const found = new Map<string, Blob>()
    if (!db || !entries.length) return resolve(found)
    try {
      const transaction = db.transaction('files', 'readonly')
      const store = transaction.objectStore('files')
      for (const [path, asset] of entries) {
        const request = store.get(asset.sha256)
        request.onsuccess = () => {
          const blob = request.result
          if (blob instanceof Blob && blob.size === asset.bytes) found.set(path, blob)
        }
      }
      transaction.oncomplete = () => resolve(found)
      transaction.onerror = transaction.onabort = () => resolve(found)
    } catch {
      resolve(found)
    }
  })
}

const downloadConcurrency = 8
const verificationBacklog = 6
const writeBacklog = 8

function orderedPaths(files: Record<string, Asset>) {
  // Prioritize the score manifest and terrain/imagery needed immediately.
  // High-resolution zoom tile packs (earth/moon/mars-tiles-*.bin) are fetched lazily
  // per octant/quadrant on demand when the user zooms in, saving ~175 MB upfront.
  const priority = [
    scorePath,
    'textures/earth-blue-marble.jpg',
    'textures/nasa-deep-star-map.webp',
    'textures/earth-terrain.bin',
    'textures/moon-albedo.webp',
    'textures/mars-albedo.webp',
    'textures/moon-terrain.bin',
    'textures/mars-terrain.bin',
  ].filter(path => Object.hasOwn(files, path))
  const chunks = Object.keys(files).filter(path => !priority.includes(path) && !ignoredPaths.has(path))
  return [...priority, ...chunks]
}

function stripIgnoredFiles(release: Release): { release: Release; changed: boolean } {
  let changed = false
  const files: Record<string, Asset> = Object.create(null)
  let bytes = 0
  for (const [path, asset] of Object.entries(release.files)) {
    if (ignoredPaths.has(path)) {
      changed = true
      continue
    }
    files[path] = asset
    bytes += asset.bytes
  }
  return changed ? { release: { etag: release.etag, files, bytes }, changed: true } : { release, changed: false }
}

function sameRelease(a: Release | undefined, b: Release | undefined): boolean {
  if (!a || !b) return a === b
  if (a.etag && b.etag) return a.etag === b.etag
  const aKeys = Object.keys(a.files)
  const bKeys = Object.keys(b.files)
  if (aKeys.length !== bKeys.length) return false
  return aKeys.every(path => {
    const af = a.files[path]
    const bf = b.files[path]
    return Boolean(bf) && af.sha256 === bf.sha256 && af.bytes === bf.bytes && af.key === bf.key
  })
}

async function revalidateInBackground(saved: Release) {
  const controller = new AbortController()
  const { signal } = controller
  try {
    const response = await fetch(manifestUrl, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(12_000)]),
      cache: 'no-store',
      credentials: 'omit',
      headers: saved.etag ? { 'If-None-Match': saved.etag } : undefined,
    })
    if (response.status === 304 || !response.ok) return
    const next = parseRelease(await response.json(), response.headers.get('ETag'))
    if (sameRelease(next, saved)) return
    const worldFiles = new Map<string, Asset>()
    for (const [path, asset] of Object.entries(next.files)) {
      if (path.startsWith(world) && path !== scorePath) worldFiles.set(asset.key, asset)
    }
    const update = async (exclusive: boolean) => {
      const cache = await openCache()
      if (!cache) return
      try {
        // Re-check under the release lock so a slower tab never overwrites a newer release.
        const active = await accessCache<Release>(cache, 'releases', manifestUrl)
        if (!sameRelease(active, saved)) return

        const entries = orderedPaths(next.files).map(path => [path, next.files[path]] as const)
        const existing = await readCachedBlobs(cache, entries)
        let scoreBlob = existing.get(scorePath)
        const missing = entries.filter(([path]) => !existing.has(path))
        let cursor = 0
        let persisted = true
        const workers = Array.from({ length: Math.min(4, missing.length) }, async () => {
          while (!signal.aborted && cursor < missing.length) {
            const [path, asset] = missing[cursor++]
            const bytes = await downloadBytes(asset, signal, () => {})
            const blob = await verifyAsset(asset, bytes, signal)
            if (path === scorePath) scoreBlob = blob
            if (!(await accessCache(cache, 'files', asset.sha256, blob))) persisted = false
          }
        })
        try {
          await Promise.all(workers)
        } catch (error) {
          controller.abort()
          await Promise.allSettled(workers)
          throw error
        }
        if (!scoreBlob) throw new Error('The Earth scoring release is missing its score manifest.')
        await validateScoreManifest(scoreBlob, worldFiles)
        if (persisted && (await accessCache(cache, 'releases', manifestUrl, next)) && exclusive) {
          await pruneCache(cache, next)
        }
      } finally {
        cache.close()
      }
    }
    if (navigator.locks) {
      await navigator.locks.request('earth-connect-release-data', () => update(true))
    } else {
      await update(false)
    }
  } catch {
    // Background revalidation is best-effort; the active saved release stays usable.
  }
}

async function prepareRelease(
  signal: AbortSignal,
  progress: (state: DownloadProgress) => void,
  exclusive: boolean,
  { refresh, onAssets }: ReleaseOptions,
) {
  const cache = await openCache()
  try {
    signal.throwIfAborted()
    // Keep the saved release on disk so ordinary navigation can still use it
    // if a forced download fails before the replacement is committed.
    const rawSaved = refresh ? undefined : await accessCache<Release>(cache, 'releases', manifestUrl)
    const cleanedSaved = rawSaved ? stripIgnoredFiles(rawSaved) : undefined
    const saved = cleanedSaved?.release
    if (saved) {
      if (cleanedSaved?.changed && cache && exclusive) {
        if (await accessCache(cache, 'releases', manifestUrl, saved)) {
          await pruneCache(cache, saved)
        }
      }
      // Stale-while-revalidate: if every file of the saved release is already in
      // IndexedDB, open from disk immediately without blocking on a network RTT.
      const entries = orderedPaths(saved.files).map(path => [path, saved.files[path]] as const)
      const existing = await readCachedBlobs(cache, entries)
      signal.throwIfAborted()
      if (existing.size === entries.length) {
        const source = releaseSource(saved, signal)
        onAssets?.(source.assets)
        progress({ downloaded: 0, total: 0, fallback: false })
        try {
          await Promise.all(entries.map(([path]) => source.publish(path, existing.get(path)!)))
          signal.throwIfAborted()
          void revalidateInBackground(saved)
          return { assets: source.assets, saved: true, downloadedBytes: 0 }
        } catch {
          signal.throwIfAborted()
        }
      }
    }

    let release: Release
    let fallback = false
    try {
      const response = await fetch(manifestUrl, {
        signal: AbortSignal.any([signal, AbortSignal.timeout(12_000)]),
        cache: 'no-store',
        credentials: 'omit',
        headers: saved?.etag ? { 'If-None-Match': saved.etag } : undefined,
      })
      if (response.status === 304 && saved) {
        release = saved
      } else {
        if (!response.ok) throw new Error(`The Earth data manifest returned HTTP ${response.status}.`)
        release = parseRelease(await response.json(), response.headers.get('ETag'))
      }
    } catch (error) {
      signal.throwIfAborted()
      if (!saved) throw error
      release = saved
      fallback = true
    }

    async function prepare(current: Release, cachedOnly: boolean, reloading: boolean) {
      const controller = new AbortController()
      const transfer = AbortSignal.any([signal, controller.signal])
      const source = releaseSource(current, transfer)
      const verifications = new Set<Promise<void>>()
      const writes = new Set<Promise<void>>()
      let persisted = true
      onAssets?.(source.assets)
      async function persist(path: string, asset: Asset, blob: Blob) {
        // Bound completed files waiting on IndexedDB without blocking
        // consumers that already received a verified copy.
        while (writes.size >= writeBacklog) await Promise.race(writes)
        transfer.throwIfAborted()
        const write = accessCache(cache, 'files', asset.sha256, blob)
          .then(async savedBlob => {
            if (!savedBlob) {
              persisted = false
              return
            }
            // Replace the RAM-backed Blob with the disk-backed IndexedDB Blob so
            // Chromium's browser process can release downloaded bytes immediately.
            const diskBlob = await accessCache<Blob>(cache, 'files', asset.sha256)
            if (diskBlob instanceof Blob && diskBlob.size === asset.bytes) {
              source.replaceBlob(path, diskBlob)
            }
          })
          .finally(() => writes.delete(write))
        writes.add(write)
      }
      try {
        const ordered = orderedPaths(current.files)
        const entries = ordered.map(path => [path, current.files[path]] as const)
        const existing = reloading ? new Map<string, Blob>() : await readCachedBlobs(cache, entries)
        transfer.throwIfAborted()
        const missing: (readonly [string, Asset])[] = []
        const publications: Promise<void>[] = []
        for (const entry of entries) {
          const blob = existing.get(entry[0])
          if (blob) publications.push(source.publish(entry[0], blob))
          else missing.push(entry)
        }
        await Promise.all(publications)
        transfer.throwIfAborted()
        if (cachedOnly && missing.length)
          throw new Error('The saved Earth data is incomplete. Connect to the internet and retry.')
        const total = missing.reduce((sum, [, asset]) => sum + asset.bytes, 0)
        let downloaded = 0
        progress({ downloaded, total, fallback: cachedOnly })
        let next = 0
        async function worker() {
          while (next < missing.length) {
            transfer.throwIfAborted()
            const [path, asset] = missing[next++]
            const bytes = await downloadBytes(asset, transfer, count => {
              downloaded += count
              progress({ downloaded, total, fallback: cachedOnly })
            })
            // Refill the network slot while verification proceeds. Once the
            // small verification backlog is full, pause until it drains.
            while (verifications.size >= verificationBacklog) await Promise.race(verifications)
            transfer.throwIfAborted()
            const verification = verifyAsset(asset, bytes, transfer)
              .then(async blob => {
                await source.publish(path, blob)
                await persist(path, asset, blob)
              })
              .catch(error => controller.abort(error))
              .finally(() => verifications.delete(verification))
            verifications.add(verification)
          }
        }
        const jobs = Array.from({ length: Math.min(downloadConcurrency, missing.length) }, worker)
        try {
          await Promise.all(jobs)
        } catch (error) {
          controller.abort(error)
          await Promise.allSettled(jobs)
          throw error
        }
        await Promise.all(verifications)
        await Promise.all(writes)
        transfer.throwIfAborted()
        if (persisted && current !== saved) {
          persisted = !!(await accessCache(cache, 'releases', manifestUrl, current))
          // A staged or failed update must leave the previous release intact.
          if (persisted && cache && exclusive) await pruneCache(cache, current)
        }
        transfer.throwIfAborted()
        return { assets: source.assets, saved: persisted, downloadedBytes: total }
      } catch (error) {
        controller.abort(error)
        await Promise.allSettled(verifications)
        await Promise.allSettled(writes)
        throw error
      }
    }

    try {
      return await prepare(release, fallback, !!refresh)
    } catch (error) {
      signal.throwIfAborted()
      if (!saved || saved === release) throw error
      try {
        return await prepare(saved, true, false)
      } catch {
        signal.throwIfAborted()
        throw error
      }
    }
  } finally {
    cache?.close()
  }
}

export async function loadWorkspaceRelease(
  signal: AbortSignal,
  progress: (state: DownloadProgress) => void,
  options: ReleaseOptions = {},
) {
  signal.throwIfAborted()
  let entered = false
  try {
    if (navigator.locks) {
      return await navigator.locks.request('earth-connect-release-data', { signal }, () => {
        entered = true
        return prepareRelease(signal, progress, true, options)
      })
    }
  } catch (error) {
    signal.throwIfAborted()
    if (entered) throw error
    // Some origins expose the API but deny access to a lock.
  }
  return prepareRelease(signal, progress, false, options)
}
