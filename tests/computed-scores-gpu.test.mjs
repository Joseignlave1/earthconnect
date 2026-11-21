import assert from 'node:assert/strict'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'
import { createServer } from 'vite'
import { hydrateAssets } from '../scripts/hydrate-assets.mjs'

await hydrateAssets()

test('compute score textures, distribution, peak, selection, cancellation and recovery on WebGPU', { timeout: 120000 }, async () => {
  const root = fileURLToPath(new URL('../',import.meta.url))
  const server = await createServer({
    root,configFile: false,logLevel: 'silent',
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { host: '127.0.0.1',port: 0,hmr: false,watch: null },
    plugins: [{
      name: 'gpu-test-page',
      configureServer(server) {
        server.middlewares.use('/gpu-test',(_request,response) => {
          response.setHeader('Content-Type','text/html')
          response.end('<!doctype html><canvas id="surface" style="width:400px;height:400px"></canvas>')
        })
      },
    }],
  })
  let browser
  try {
    await server.listen()
    browser = await chromium.launch({
      executablePath: process.env.EARTH_SCORE_CHROMIUM,channel: 'chromium',headless: true,
      args: JSON.parse(process.env.EARTH_SCORE_CHROMIUM_ARGS ?? JSON.stringify([
        '--enable-unsafe-webgpu',
        ...(process.platform === 'linux' ? [
          '--ignore-gpu-blocklist','--use-gl=angle','--use-angle=vulkan',
          '--enable-features=Vulkan','--disable-vulkan-surface',
        ] : []),
      ])),
    })
    const page = await browser.newPage()
    const errors = []
    page.on('pageerror',error => errors.push(error.message))
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/gpu-test`)
    const summary = await page.evaluate(async () => {
      const [{ SurfaceRenderer }, { prepareComputedScores,destroyScoreTextures,readComputedScore }, { createLayout }, { maxPyramid }, { MISSING,SIZE }] = await Promise.all([
        import('/src/shader-test/renderer.ts'),import('/src/shader-test/computed-scores.ts'),
        import('/src/earth-score/layout.ts'),import('/src/shader-test/noise.ts'),import('/src/shader-test/world-grid.ts'),
      ])
      const check = (condition,message) => { if (!condition) throw new Error(message) }
      const canvas = document.querySelector('canvas')
      const renderErrors = []
      const picks = []
      let status = ''
      const renderer = await SurfaceRenderer.create(canvas,(selection,score) => picks.push({ selection,score }),error => renderErrors.push(error.message),value => { status = value })
      renderer.update({ hdr: false,relief: 0 })
      const makeFixture = device => {
        const tileSize = SIZE ** 2
        const sites = ['h18v08','h17v09','h19v08','h17v08','h18v09','h19v09']
        const layout = createLayout({
          sites,rows: sites.length * tileSize,
          chunks: [{ rows: tileSize }, { rows: tileSize * (sites.length - 1) }],
        })
        const tiles = sites.map(() => new Uint16Array(tileSize).fill(MISSING))
        tiles[0][65539] = 65534
        tiles[0][tileSize - 1] = 65534
        tiles[1].fill(0)
        tiles[2][tileSize - 1] = 256
        tiles[3].fill(27500)
        tiles[4][5] = 65534
        const chunks = layout.chunks.map(chunk => {
          const source = new Uint32Array(chunk.rows)
          const firstSite = chunk.firstAddress / tileSize
          for (let row = 0; row < chunk.rows; row += tileSize) source.set(tiles[firstSite + row / tileSize],row)
          const scores = device.createBuffer({ size: source.byteLength,usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
          device.queue.writeBuffer(scores,0,source)
          return { ...chunk,scores }
        })
        return { layout,chunks,tiles,destroy: () => chunks.forEach(({ scores }) => scores.destroy()) }
      }
      const device = renderer.computeDevice
      const fixture = makeFixture(device)
      const map = await prepareComputedScores(device,fixture,2,3)
      const readMip = async (texture,lod,layer,width) => {
        const bytesPerRow = Math.ceil(width * 2 / 256) * 256
        const buffer = device.createBuffer({ size: bytesPerRow * width,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
        try {
          const encoder = device.createCommandEncoder()
          encoder.copyTextureToBuffer({ texture,mipLevel: lod,origin: [0,0,layer] },{ buffer,bytesPerRow },[width,width,1])
          device.queue.submit([encoder.finish()])
          await buffer.mapAsync(GPUMapMode.READ)
          const view = new Uint16Array(buffer.getMappedRange())
          const scores = new Uint16Array(width * width)
          for (let y = 0; y < width; y++) scores.set(view.subarray(y * bytesPerRow / 2,y * bytesPerRow / 2 + width),y * width)
          return scores
        } finally { buffer.destroy() }
      }
      let compared = 0
      try {
        const expected = new Uint32Array(MISSING)
        for (const [site,tile] of fixture.tiles.entries()) {
          for (const score of tile) if (score !== MISSING) expected[score]++
          for (const [lod,{ scores,width }] of maxPyramid(tile,SIZE,SIZE).entries()) {
            const actual = await readMip(map.textures.banks[Math.floor(site / 2)],lod,site % 2,width)
            check(actual.every((code,index) => code === scores[index]),`Texture mismatch at site ${site}, LOD ${lod}`)
            compared += actual.length
          }
        }
        check(map.histogram.every((count,index) => count === expected[index]),'Score histogram mismatch')
        check(map.bestAddress === 65539,'Packed peak address or tie-breaking mismatch')
        check(await readComputedScore(device,map.textures,2,1,0,0) === 0,'Valid zero was lost')
        check(await readComputedScore(device,map.textures,2,0,0,0) === MISSING,'Missing score was lost')
      } finally { destroyScoreTextures(map.textures) }

      const description = name => ({ name,description: name })
      const stale = renderer.setComputedScores(fixture,description('stale')).then(() => 'published',error => error.name)
      await renderer.setComputedScores(fixture,description('fixture'))
      check(await stale === 'AbortError','An outdated request was allowed to publish')
      const controller = new AbortController()
      const createBuffer = device.createBuffer
      device.createBuffer = function (options) {
        const buffer = createBuffer.call(this,options)
        if (options.label === 'Score statistics readback') {
          const mapAsync = buffer.mapAsync
          buffer.mapAsync = function (...args) {
            const pending = mapAsync.apply(this,args)
            controller.abort()
            return pending
          }
        }
        return buffer
      }
      try {
        const canceled = await renderer.setComputedScores(fixture,description('canceled'),controller.signal).then(() => 'published',error => error.name)
        check(canceled === 'AbortError','A canceled GPU readback was allowed to publish')
      } finally { device.createBuffer = createBuffer }
      check(canvas.dataset.scoreName === 'fixture','Canceled publication replaced the active map')
      check(renderer.matchingScoreCount === 2 * SIZE ** 2 + 4,'Renderer score statistics mismatch')
      renderer.update({ minimumScore: 1 })
      check(renderer.matchingScoreCount === 3,'Renderer score filter mismatch')
      renderer.update({ minimumScore: 0 })
      fixture.destroy()
      renderer.focusBest()
      const best = picks.at(-1)
      const expected = fixture.layout.locate(65539).center
      check(best.score === 1 && Math.abs(best.selection.latitude - expected.latitude) < 1e-6 &&
        Math.abs(best.selection.longitude - expected.longitude) < 1e-6,'Renderer peak navigation mismatch')
      window.gpuTest = { renderer,device,fixture,makeFixture,canvas,picks,renderErrors,description,getStatus: () => status }
      return compared
    })
    assert.ok(summary > 500000)
    await page.waitForFunction(() => window.gpuTest.canvas.dataset.ready === 'true')
    const { x,y,width,height } = await page.locator('canvas').boundingBox()
    await page.mouse.click(x + width / 2,y + height / 2)
    await page.waitForFunction(() => window.gpuTest.picks.length >= 3)
    const picked = await page.evaluate(() => {
      const { picks,fixture } = window.gpuTest
      const { selection,score } = picks.at(-1)
      const address = fixture.layout.addressAt(selection.longitude,selection.latitude)
      const cell = fixture.tiles[Math.floor(address.address / fixture.tiles[0].length)][address.pixel]
      return { actual: score,expected: cell === 65535 ? null : cell / 65534 }
    })
    assert.deepEqual(picked.actual,picked.expected)
    await page.evaluate(() => window.gpuTest.device.destroy())
    await page.waitForFunction(() => {
      const { renderer,device,canvas } = window.gpuTest
      return renderer.computeDevice && renderer.computeDevice !== device && !canvas.dataset.scoreName
    })
    await page.evaluate(() => window.gpuTest.renderer.regenerate(6,42))
    await page.waitForFunction(() => window.gpuTest.canvas.dataset.source === 'noise')
    await page.evaluate(async () => {
      const { renderer,makeFixture,canvas,description } = window.gpuTest
      const noiseName = canvas.dataset.scoreName
      const fixture = makeFixture(renderer.computeDevice)
      const device = renderer.computeDevice
      try {
        await renderer.setComputedScores(fixture,description('restored'))
        if (canvas.dataset.scoreName !== 'restored') throw new Error('The recovered device did not publish scores')
        const interrupted = renderer.setComputedScores(fixture,description('interrupted')).then(() => false,() => true)
        device.destroy()
        if (!await interrupted) throw new Error('A query published after its device was lost')
        Object.assign(window.gpuTest,{ noiseName,device })
      } finally {
        fixture.destroy()
      }
    })
    await page.waitForFunction(() => {
      const { renderer,device,canvas,noiseName } = window.gpuTest
      return renderer.computeDevice && renderer.computeDevice !== device && canvas.dataset.scoreName === noiseName &&
        canvas.dataset.ready === 'true' && canvas.dataset.idle === 'true'
    })
    let blockedWorkers = 0
    await page.route(/\/noise-worker\.ts\?worker_file/, route => {
      blockedWorkers++
      return route.fulfill({ contentType: 'text/html',body: '<!doctype html><title>Unexpected HTML</title>' })
    })
    await page.evaluate(() => {
      const { renderer,canvas } = window.gpuTest
      window.gpuTest.noise = { cells: canvas.dataset.cells,bestScore: canvas.dataset.bestScore }
      renderer.regenerate(6,42)
    })
    await page.waitForFunction(() => /Generating 4 km noise · \d+ \/ \d+ tiles/.test(window.gpuTest.getStatus()))
    await page.waitForFunction(() => window.gpuTest.getStatus().startsWith('Synthetic Perlin noise'))
    const restored = await page.evaluate(() => {
      const { canvas,noise } = window.gpuTest
      return { expected: noise,actual: { cells: canvas.dataset.cells,bestScore: canvas.dataset.bestScore } }
    })
    assert.equal(blockedWorkers,1)
    assert.deepEqual(restored.actual,restored.expected)
    const renderErrors = await page.evaluate(() => {
      window.gpuTest.renderer.destroy()
      return window.gpuTest.renderErrors
    })
    assert.deepEqual(renderErrors,[])
    const canvases = await page.evaluate(async () => {
      const [{ SurfaceRenderer }, { createEarthWorkspaceResources }] = await Promise.all([
        import('/src/shader-test/renderer.ts'), import('/src/workspace/earthWorkspaceResources.ts'),
      ])
      const { acquireDevice, acquireWorld } = createEarthWorkspaceResources()
      const canvases = [0, 1].map(() => {
        const canvas = document.createElement('canvas')
        canvas.style.cssText = 'width:240px;height:240px'
        document.body.append(canvas)
        return canvas
      })
      const failures = []
      const renderers = await Promise.all(canvases.map(canvas => SurfaceRenderer.create(
        canvas, () => {}, error => failures.push(error.message), () => {}, { acquireDevice },
      )))
      const [first, second] = renderers
      const device = first.computeDevice
      if (!device || second.computeDevice !== device) throw new Error('Previews received different GPU devices')
      let manifestLoads = 0
      const options = {
        manifestUrl: new URL('/compute-data/world/manifest.json', location.href),
        readAsset(url, signal) {
          if (url.pathname.endsWith('/manifest.json')) manifestLoads++
          return fetch(url, { signal })
        },
      }
      const worlds = renderers.map(() => acquireWorld(device, options))
      try {
        const [earth, otherEarth] = await Promise.all(worlds.map(world => world.loading))
        if (earth !== otherEarth || manifestLoads !== 1) throw new Error('Previews loaded separate Earth datasets')
        const publish = (index, value) => worlds[index].publish(async () => {
          await earth.dispatchShader(`fn score(cell: EarthCell) -> f32 { return ${value}; }`)
          await renderers[index].setComputedScores(earth, { name: String(value), description: 'Constant score' })
        })
        const [firstScore, cancelled, secondScore] = await Promise.allSettled([
          publish(0, 0.2),
          worlds[1].publish(() => Promise.reject(new DOMException('Preview closed', 'AbortError'))),
          publish(1, 0.7),
        ])
        if (firstScore.status === 'rejected') throw firstScore.reason
        if (secondScore.status === 'rejected') throw secondScore.reason
        if (cancelled.status !== 'rejected') throw new Error('Cancellation was swallowed')
        const scores = canvases.map(canvas => Number(canvas.dataset.bestScore))
        worlds[0].release()
        first.destroy()
        await publish(1, 0.9)
        return { scores, survivingScore: Number(canvases[1].dataset.bestScore), failures }
      } finally {
        worlds.forEach(world => world.release())
        renderers.forEach(renderer => renderer.destroy())
        canvases.forEach(canvas => canvas.remove())
      }
    })
    assert.ok(Math.abs(canvases.scores[0] - 0.2) < 0.001)
    assert.ok(Math.abs(canvases.scores[1] - 0.7) < 0.001)
    assert.ok(Math.abs(canvases.survivingScore - 0.9) < 0.001)
    assert.deepEqual(canvases.failures,[])
    assert.deepEqual(errors,[])
  } finally {
    await browser?.close()
    await server.close()
  }
})
