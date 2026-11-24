import { createReadStream } from 'node:fs'
import { readFile, stat } from 'node:fs/promises'
import { createServer } from 'node:http'
import { fileURLToPath } from 'node:url'
import { zstdDecompressSync } from 'node:zlib'
import { chromium } from 'playwright-core'

const output = new URL('../../exports/earth-score/', import.meta.url)
const here = new URL('./', import.meta.url)
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://localhost').pathname
    if (path === '/favicon.ico') {
      res.writeHead(204).end()
      return
    }
    if (path === '/') {
      res.setHeader('Content-Type', 'text/html')
      res.end('<!doctype html><title>Earth Score GPU verification</title>')
      return
    }
    if (/^\/reference\/chunk-\d{2}\.zst$/.test(path)) {
      const data = zstdDecompressSync(await readFile(new URL(path.slice(1), here)))
      res.setHeader('Content-Length', data.length)
      res.end(data)
      return
    }
    const testFiles = ['/suite.mjs', '/reference/manifest.json']
    if (!testFiles.includes(path) && !/^\/(?:earth-score\.mjs|(?:assets|world)\/[\w.-]+)$/.test(path)) {
      res.writeHead(404).end()
      return
    }
    const file = fileURLToPath(new URL(path.slice(1), testFiles.includes(path) ? here : output))
    res.setHeader('Content-Length', (await stat(file)).size)
    if (/\.m?js$/.test(file)) res.setHeader('Content-Type', 'text/javascript')
    if (file.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm')
    if (file.endsWith('.json')) res.setHeader('Content-Type', 'application/json')
    const stream = createReadStream(file)
    stream.on('error', () => res.destroy())
    res.on('close', () => stream.destroy())
    stream.pipe(res)
  } catch (error) {
    if (!res.headersSent) res.writeHead(error.code === 'ENOENT' ? 404 : 500)
    res.end()
  }
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
let browser
try {
  browser = await chromium.launch({
    executablePath: process.env.EARTH_SCORE_CHROMIUM,
    channel: 'chromium',
    headless: true,
    args: JSON.parse(
      process.env.EARTH_SCORE_CHROMIUM_ARGS ??
        JSON.stringify([
          '--enable-unsafe-webgpu',
          ...(process.platform === 'linux'
            ? [
                '--ignore-gpu-blocklist',
                '--use-gl=angle',
                '--use-angle=vulkan',
                '--enable-features=Vulkan',
                '--disable-vulkan-surface',
              ]
            : []),
        ]),
    ),
  })
  const page = await browser.newPage()
  page.on('console', message => console.log(message.text()))
  page.on('pageerror', error => console.error(error))
  await page.goto(`http://127.0.0.1:${server.address().port}/`)
  const result = await page.evaluate(async () => (await import('/suite.mjs')).verify())
  console.log(JSON.stringify(result, null, 2))
} finally {
  await browser?.close()
  server.close()
}
