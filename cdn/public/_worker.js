const versionedKey = /^(?:(?:assets\/(?:data|textures)|compute-data\/world)\/[a-zA-Z0-9_-]+-[a-f0-9]{64}\.[a-zA-Z0-9]+|releases\/[a-f0-9]{64}\.json)$/
const readMethods = 'GET, HEAD, OPTIONS'
const cors = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Expose-Headers': 'ETag',
}

function message(request, status, text, headers = {}) {
  return new Response(request.method === 'HEAD' ? null : text, {
    status,
    headers: { ...cors, 'Cache-Control': 'no-store', ...headers },
  })
}

function matchesEtag(request, etag) {
  return request.headers.get('If-None-Match')?.split(',').some(value => {
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
          'Access-Control-Allow-Headers': 'If-None-Match',
          'Access-Control-Max-Age': '86400',
        },
      })
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return message(request, 405, 'Method not allowed', { Allow: readMethods })
    }

    const url = new URL(request.url)
    const key = url.pathname.slice(1)
    const immutable = versionedKey.test(key)
    if (key !== 'manifest.json' && !immutable) {
      return message(request, 404, 'Asset not found')
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

    const bucket = env.EARTH_ASSETS
    const headOnly = request.method === 'HEAD'
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
    headers.set('Cache-Control', immutable
      ? 'public, max-age=31536000, immutable'
      : 'public, max-age=0, must-revalidate')
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
