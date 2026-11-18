import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { zstdDecompressSync } from 'node:zlib'
import { chromium } from 'playwright-core'
import { createServer } from 'vite'

const root = fileURLToPath(new URL('../../', import.meta.url))
const world = new URL('../../compute-data/world/', import.meta.url)

test('Earth Score decodes chunks when isolation prevents its workers from starting', { timeout: 30000 }, async () => {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', world), 'utf8'))
  const chunk = manifest.chunks[0]
  const expected = createHash('sha256')
    .update(zstdDecompressSync(await readFile(new URL(chunk.file, world))))
    .digest('hex')
  const server = await createServer({
    root,
    logLevel: 'silent',
    server: { host: '127.0.0.1', port: 0 },
    plugins: [{
      name: 'decoder-test-page',
      configureServer(server) {
        server.middlewares.use('/decoder-test', (_request, response) => {
          response.setHeader('Content-Type', 'text/html')
          response.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
          response.setHeader('Cross-Origin-Embedder-Policy', 'credentialless')
          response.end('<!doctype html><title>Earth Score decoder verification</title>')
        })
      },
    }],
  })
  let browser
  try {
    await server.listen()
    browser = await chromium.launch({
      executablePath: process.env.EARTH_SCORE_CHROMIUM,
      channel: 'chromium',
      headless: true,
    })
    for (const omitWorkerIsolation of [false, true]) {
      const context = await browser.newContext()
      let intercepted = 0
      if (omitWorkerIsolation) {
        await context.route(/\/decoder-entry\.worker\.ts\?worker_file/, async route => {
          const response = await route.fetch()
          const headers = response.headers()
          assert.equal(headers['cross-origin-embedder-policy'], 'credentialless')
          delete headers['cross-origin-embedder-policy']
          intercepted++
          await route.fulfill({ response, headers })
        })
      }
      try {
        const page = await context.newPage()
        await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/decoder-test`)
        const result = await page.evaluate(async ({ chunk, fields }) => {
          const { createDecoderPool } = await import('/src/earth-score/decoder-pool.ts')
          const pool = createDecoderPool(4, new AbortController().signal)
          const frame = await (await fetch(`/compute-data/world/${chunk.file}`)).arrayBuffer()
          const corrupt = frame.slice(0)
          new Uint8Array(corrupt)[0] ^= 255
          const { packedBytes, rows } = chunk
          try {
            const [packed, error] = await Promise.all([
              pool.decode({ frame, packedBytes, rows, fields }),
              pool.decode({ frame: corrupt, packedBytes, rows, fields }).then(() => null, error => error.message),
            ])
            const hash = await crypto.subtle.digest('SHA-256', packed)
            return {
              isolated: crossOriginIsolated,
              transferred: frame.byteLength === 0,
              hash: Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join(''),
              error,
            }
          } finally {
            pool.close()
          }
        }, { chunk, fields: manifest.columns.length })
        if (omitWorkerIsolation) assert.ok(intercepted > 0)
        assert.deepEqual(result, {
          isolated: true,
          transferred: !omitWorkerIsolation,
          hash: expected,
          error: 'Incorrect Zstd frame content size',
        })
      } finally {
        await context.close()
      }
    }
  } finally {
    await browser?.close()
    await server.close()
  }
})
