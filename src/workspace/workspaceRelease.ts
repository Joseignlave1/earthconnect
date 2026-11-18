import { workspaceAssets, type WorkspaceAssets } from './workspaceAssets.ts'

const manifestUrl = 'https://earth-connect-assets.pages.dev/manifest.json'
const scorePath = 'compute-data/world/manifest.json'
const world = 'compute-data/world/'
const renderPaths = [
  'data/surface-basins.bin',
  'textures/earth-blue-marble.jpg',
  'textures/earth-terrain.bin',
  'textures/moon-terrain.bin',
  'textures/mars-terrain.bin',
  'textures/nasa-deep-star-map.webp',
]
const hashPattern = /^[a-f0-9]{64}$/
type Asset = { key: string; bytes: number; sha256: string; contentType: string }
type Release = { etag: string | null; files: Record<string, Asset>; bytes: number }
export type DownloadProgress = { downloaded: number; total: number; fallback: boolean }

function parseRelease(value: unknown, etag: string | null): Release {
  if (!value || typeof value !== 'object') throw new Error('Invalid Earth data release.')
  const { format, runtimeFileCount, runtimeBytes, scoreManifest, assets } = value as {
    format?: unknown; runtimeFileCount?: unknown; runtimeBytes?: unknown
    scoreManifest?: Partial<Asset>; assets?: Record<string, unknown>
  }
  if (format !== 'earth-connect-assets-v1' || !assets || typeof assets !== 'object' ||
      !renderPaths.every(path => Object.hasOwn(assets, path)) || !Object.hasOwn(assets, scorePath)) {
    throw new Error('The Earth data release is missing required files.')
  }
  const files: Record<string, Asset> = Object.create(null)
  let bytes = 0
  for (const [path, value] of Object.entries(assets)) {
    const item = value as Partial<Asset> | undefined
    if (!renderPaths.includes(path) && path !== scorePath && !/^compute-data\/world\/chunk-\d+\.zst$/.test(path)) {
      throw new Error(`Unexpected Earth data file ${path}.`)
    }
    if (!item || typeof item.sha256 !== 'string' || !hashPattern.test(item.sha256) ||
        typeof item.bytes !== 'number' || !Number.isSafeInteger(item.bytes) || item.bytes <= 0 ||
        typeof item.contentType !== 'string') {
      throw new Error(`Invalid Earth data metadata for ${path}.`)
    }
    const dot = path.lastIndexOf('.')
    const prefix = path.startsWith(world) ? '' : 'assets/'
    if (item.key !== `${prefix}${path.slice(0, dot)}-${item.sha256}${path.slice(dot)}`) {
      throw new Error(`Invalid Earth data URL for ${path}.`)
    }
    files[path] = { key: item.key, bytes: item.bytes, sha256: item.sha256, contentType: item.contentType }
    bytes += item.bytes
  }
  if (scoreManifest?.key !== files[scorePath].key ||
      scoreManifest.sha256 !== files[scorePath].sha256 || scoreManifest.bytes !== files[scorePath].bytes) {
    throw new Error('The Earth scoring manifest does not match the release.')
  }
  if (Object.keys(files).length !== runtimeFileCount || bytes !== runtimeBytes || !Number.isSafeInteger(bytes)) {
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
    signal: lifetime, cache: 'no-store', credentials: 'omit',
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
  if (!Array.isArray(score?.chunks) || score.chunks.length !== remaining.size ||
      score.chunks.length === 0 || !score.chunks.every((chunk: { file?: unknown; compressedBytes?: unknown } | null) => {
        if (typeof chunk?.file !== 'string') return false
        const path = world + chunk.file
        const asset = remaining.get(path)
        if (!asset || asset.bytes !== chunk.compressedBytes) return false
        remaining.delete(path)
        return true
      })) {
    throw new Error('The Earth scoring release is missing world chunks.')
  }
}

function waitForBlob(pending: Promise<Blob>, signal: AbortSignal) {
  return new Promise<Blob>((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason)
    const abort = () => reject(signal.reason)
    signal.addEventListener('abort', abort, { once: true })
    void pending.then(blob => {
      signal.removeEventListener('abort', abort)
      resolve(blob)
    })
  })
}

function releaseSource(release: Release, lifetime: AbortSignal) {
  const slots = new Map<string, { blob?: Blob; pending: Promise<Blob>; resolve: (blob: Blob) => void }>()
  const worldFiles = new Map<string, Asset>()
  for (const [path, asset] of Object.entries(release.files)) {
    let resolve!: (blob: Blob) => void
    const pending = new Promise<Blob>(ready => { resolve = ready })
    const slot = { pending, resolve }
    slots.set(path, slot)
    if (path.startsWith(world) && path !== scorePath) {
      slots.set(asset.key, slot)
      worldFiles.set(asset.key, asset)
    }
  }
  const assets = workspaceAssets((path, signal) => {
    lifetime.throwIfAborted()
    const slot = slots.get(path)
    if (!slot) throw new Error(`Workspace data is missing ${path}.`)
    return slot.blob ?? waitForBlob(slot.pending, AbortSignal.any([signal, lifetime]))
  })
  lifetime.addEventListener('abort', () => {
    assets.dispose()
    slots.clear()
  }, { once: true })
  return {
    assets,
    async publish(path: string, blob: Blob) {
      if (path === scorePath) await validateScoreManifest(blob, worldFiles)
      lifetime.throwIfAborted()
      const slot = slots.get(path)!
      slot.blob = blob
      slot.resolve(blob)
    },
  }
}

type ReleaseOptions = {
  refresh?: boolean
  onAssets?: (assets: WorkspaceAssets) => void
}

async function prepareRelease(signal: AbortSignal, progress: (state: DownloadProgress) => void, exclusive: boolean, { refresh, onAssets }: ReleaseOptions) {
  const cache = await openCache()
  try {
    signal.throwIfAborted()
    // Keep the saved release on disk so ordinary navigation can still use it
    // if a forced download fails before the replacement is committed.
    const saved = refresh ? undefined : await accessCache<Release>(cache, 'releases', manifestUrl)
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
      async function readCached(path: string, asset: Asset) {
        const blob = reloading ? undefined : await accessCache<Blob>(cache, 'files', asset.sha256)
        if (!(blob instanceof Blob) || blob.size !== asset.bytes) return [path, asset] as const
        await source.publish(path, blob)
      }
      async function persist(asset: Asset, blob: Blob) {
        // Bound completed files waiting on IndexedDB without blocking
        // consumers that already received a verified copy.
        while (writes.size >= 4) await Promise.race(writes)
        transfer.throwIfAborted()
        const write = accessCache(cache, 'files', asset.sha256, blob).then(savedBlob => {
          if (!savedBlob) persisted = false
        }).finally(() => writes.delete(write))
        writes.add(write)
      }
      try {
        const paths = [scorePath, ...renderPaths]
        const ordered = [...paths, ...Object.keys(current.files).filter(path => !paths.includes(path))]
        const cached = await Promise.allSettled(ordered.map(path => readCached(path, current.files[path])))
        const missing = cached.flatMap(result => {
          if (result.status === 'rejected') throw result.reason
          return result.value ? [result.value] : []
        })
        transfer.throwIfAborted()
        if (cachedOnly && missing.length) throw new Error('The saved Earth data is incomplete. Connect to the internet and retry.')
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
            while (verifications.size >= 3) await Promise.race(verifications)
            transfer.throwIfAborted()
            const verification = verifyAsset(asset, bytes, transfer).then(async blob => {
              await source.publish(path, blob)
              await persist(asset, blob)
            }).catch(error => controller.abort(error)).finally(() => verifications.delete(verification))
            verifications.add(verification)
          }
        }
        const jobs = Array.from({ length: Math.min(3, missing.length) }, worker)
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
          persisted = !!await accessCache(cache, 'releases', manifestUrl, current)
          // A staged or failed update must leave the previous release intact.
          if (persisted && cache && exclusive) await pruneCache(cache, current)
        }
        transfer.throwIfAborted()
        return { assets: source.assets, saved: persisted }
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

export async function loadWorkspaceRelease(signal: AbortSignal, progress: (state: DownloadProgress) => void, options: ReleaseOptions = {}) {
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
