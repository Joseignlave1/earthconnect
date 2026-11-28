const versionedKey =
  /^(?:(?:assets\/(?:data|textures)|compute-data\/world)\/[a-zA-Z0-9_-]+-[a-f0-9]{64}\.[a-zA-Z0-9]+|releases\/[a-f0-9]{64}\.json)$/
const ecv3Key = /^ecv3\/(?:earth|moon|mars)\/[a-z0-9_-]+\.ecv3$/
const validVersionParam = /^[a-zA-Z0-9_-]{1,32}$/
const maxCacheableChunkBytes = 4 * 1024 * 1024
const readMethods = 'GET, HEAD, OPTIONS'
const cors = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Expose-Headers': 'ETag, Content-Range, Content-Length, Accept-Ranges',
}

function parseByteRange(header) {
  if (!header) return null
  const match = /^bytes=(\d+)-(\d*)$/.exec(header.trim())
  if (!match) return 'invalid'
  const offset = Number(match[1])
  const end = match[2] ? Number(match[2]) : undefined
  if (!Number.isSafeInteger(offset) || (end !== undefined && (!Number.isSafeInteger(end) || end < offset))) {
    return 'invalid'
  }
  return end !== undefined ? { offset, length: end - offset + 1 } : { offset }
}

function parseQueryRange(param) {
  if (param === null) return null
  const match = /^(\d+)-(\d+)$/.exec(param.trim())
  if (!match) return 'invalid'
  const offset = Number(match[1])
  const length = Number(match[2])
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || length <= 0 || length > 4 * 1024 * 1024) {
    return 'invalid'
  }
  return { offset, length }
}

function message(request, status, text, headers = {}) {
  return new Response(request.method === 'HEAD' ? null : text, {
    status,
    headers: { ...cors, 'Cache-Control': 'no-store', ...headers },
  })
}

function matchesEtag(request, etag) {
  return request.headers
    .get('If-None-Match')
    ?.split(',')
    .some(value => {
      const candidate = value.trim()
      return candidate === '*' || candidate.replace(/^W\//, '') === etag
    })
}

function sendObject(request, headers, body) {
  if (matchesEtag(request, headers.get('ETag'))) {
    headers.delete('Content-Length')
    return new Response(null, { status: 304, headers })
  }
  return new Response(request.method === 'HEAD' ? null : body, { headers })
}

export default {
  async fetch(request, env, context) {
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          ...cors,
          'Access-Control-Allow-Methods': readMethods,
          'Access-Control-Allow-Headers': 'If-None-Match, Range',
          'Access-Control-Max-Age': '86400',
        },
      })
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return message(request, 405, 'Method not allowed', { Allow: readMethods })
    }

    const url = new URL(request.url)
    const key = url.pathname.slice(1)
    const isEcv3 = ecv3Key.test(key)
    const immutable = versionedKey.test(key)
    if (key !== 'manifest.json' && !immutable && !isEcv3) {
      return message(request, 404, 'Asset not found')
    }

    const bucket = env.EARTH_ASSETS
    const headOnly = request.method === 'HEAD'

    if (isEcv3) {
      const rangeHeader = request.headers.get('Range')
      const queryRangeParam = url.searchParams.get('r')
      // Support ?r=offset-length query ranges (returns HTTP 200 with unique per-chunk URL) so browsers
      // bypass Chromium's per-URL HTTP cache writer lock on 206 Partial Content and avoid CORS preflight.
      const viaQuery = !rangeHeader && queryRangeParam !== null
      const parsedRange = viaQuery ? parseQueryRange(queryRangeParam) : parseByteRange(rangeHeader)
      if (parsedRange === 'invalid') {
        return message(request, 416, 'Invalid range')
      }
      // Versioned ECV3 URLs (?v=q92, ?v=e1) are immutable so browser disk cache and CF edge cache
      // can retain 64 KiB chunks long-term instead of expiring every hour.
      const ecv3CacheControl = url.searchParams.has('v')
        ? 'public, max-age=31536000, immutable'
        : 'public, max-age=3600'
      if (headOnly) {
        const head = await bucket.head(key)
        if (!head) return message(request, 404, 'Asset not found')
        const headers = new Headers(cors)
        headers.set('Content-Type', 'application/octet-stream')
        headers.set('ETag', head.httpEtag)
        headers.set('Accept-Ranges', 'bytes')
        headers.set('Cache-Control', ecv3CacheControl)
        if (parsedRange) {
          if (parsedRange.offset >= head.size) {
            return message(request, 416, 'Range not satisfiable', { 'Content-Range': `bytes */${head.size}` })
          }
          const rLen =
            parsedRange.length !== undefined
              ? Math.min(parsedRange.length, head.size - parsedRange.offset)
              : head.size - parsedRange.offset
          headers.set('Content-Length', String(rLen))
          if (!viaQuery) {
            headers.set('Content-Range', `bytes ${parsedRange.offset}-${parsedRange.offset + rLen - 1}/${head.size}`)
            return new Response(null, { status: 206, headers })
          }
          return new Response(null, { status: 200, headers })
        }
        headers.set('Content-Length', String(head.size))
        return sendObject(request, headers, null)
      }
      if (parsedRange) {
        const v = url.searchParams.get('v') ?? ''
        const cacheableRange =
          parsedRange.length !== undefined &&
          parsedRange.length <= maxCacheableChunkBytes &&
          (v === '' || validVersionParam.test(v))
        let rangeCacheKey = null
        if (cacheableRange) {
          const rangeCacheUrl = new URL(url)
          rangeCacheUrl.search = `?v=${encodeURIComponent(v)}&r=${parsedRange.offset}-${parsedRange.length}`
          rangeCacheKey = new Request(rangeCacheUrl)
          const cached = await caches.default.match(rangeCacheKey)
          if (cached) {
            const headers = new Headers(cached.headers)
            headers.set('Cache-Control', ecv3CacheControl)
            headers.set('Cache-Status', 'EarthConnect; hit')
            if (viaQuery) headers.delete('Content-Range')
            return new Response(cached.body, { status: viaQuery ? 200 : 206, headers })
          }
        }
        let object
        try {
          object = await bucket.get(key, { range: parsedRange })
        } catch {
          const head = await bucket.head(key)
          if (!head) return message(request, 404, 'Asset not found')
          return message(request, 416, 'Range not satisfiable', { 'Content-Range': `bytes */${head.size}` })
        }
        if (!object) return message(request, 404, 'Asset not found')
        const r = object.range ?? { offset: 0, length: object.size }
        const rOff = r.offset ?? 0
        const rLen = r.length ?? Math.max(0, object.size - rOff)
        if (rOff >= object.size || rLen <= 0) {
          if (object.body) context.waitUntil(object.body.cancel().catch(() => {}))
          return message(request, 416, 'Range not satisfiable', { 'Content-Range': `bytes */${object.size}` })
        }
        const cacheHeaders = new Headers(cors)
        cacheHeaders.set('Content-Type', 'application/octet-stream')
        cacheHeaders.set('ETag', object.httpEtag)
        cacheHeaders.set('Accept-Ranges', 'bytes')
        cacheHeaders.set('Content-Range', `bytes ${rOff}-${rOff + rLen - 1}/${object.size}`)
        cacheHeaders.set('Content-Length', String(rLen))
        cacheHeaders.set('Cache-Control', ecv3CacheControl)
        if (rangeCacheKey) {
          const [clientStream, cacheStream] = object.body.tee()
          context.waitUntil(
            caches.default
              .put(rangeCacheKey, new Response(cacheStream, { status: 200, headers: cacheHeaders }))
              .catch(() => {}),
          )
          const clientHeaders = new Headers(cacheHeaders)
          if (viaQuery) clientHeaders.delete('Content-Range')
          clientHeaders.set('Cache-Status', 'EarthConnect; fwd=miss')
          return new Response(clientStream, { status: viaQuery ? 200 : 206, headers: clientHeaders })
        }
        const clientHeaders = new Headers(cacheHeaders)
        if (viaQuery) clientHeaders.delete('Content-Range')
        return new Response(object.body, { status: viaQuery ? 200 : 206, headers: clientHeaders })
      }
      const object = await bucket.get(key)
      if (!object) return message(request, 404, 'Asset not found')
      const headers = new Headers(cors)
      headers.set('Content-Type', 'application/octet-stream')
      headers.set('ETag', object.httpEtag)
      headers.set('Accept-Ranges', 'bytes')
      headers.set('Content-Length', String(object.size))
      headers.set('Cache-Control', ecv3CacheControl)
      return sendObject(request, headers, object.body)
    }

    // Object identity is determined by its path. Queries cannot fragment the cache.
    url.search = ''
    const cacheKey = new Request(url)
    if (immutable) {
      const cached = await caches.default.match(cacheKey)
      if (cached) {
        const headers = new Headers(cached.headers)
        headers.set('Cache-Status', 'EarthConnect; hit')
        if (cached.body && (request.method === 'HEAD' || matchesEtag(request, headers.get('ETag')))) {
          context.waitUntil(cached.body.cancel().catch(() => {}))
        }
        return sendObject(request, headers, cached.body)
      }
    }

    const ifNoneMatch = request.headers.has('If-None-Match')
    let object = headOnly || ifNoneMatch ? await bucket.head(key) : await bucket.get(key)
    if (object && !headOnly && ifNoneMatch && !matchesEtag(request, object.httpEtag)) {
      object = await bucket.get(key)
    }
    if (!object) return message(request, 404, 'Asset not found')

    const headers = new Headers(cors)
    object.writeHttpMetadata(headers)
    headers.set('ETag', object.httpEtag)
    headers.set('Content-Length', String(object.size))
    headers.set(
      'Cache-Control',
      immutable ? 'public, max-age=31536000, immutable' : 'public, max-age=0, must-revalidate',
    )
    if (immutable) headers.set('Cache-Status', 'EarthConnect; fwd=miss')

    const response = sendObject(request, headers, object.body)
    if (response.status === 304 && object.body) {
      context.waitUntil(object.body.cancel().catch(() => {}))
    }
    if (immutable && !headOnly && response.status === 200) {
      context.waitUntil(caches.default.put(cacheKey, response.clone()).catch(() => {}))
    }
    return response
  },
}
