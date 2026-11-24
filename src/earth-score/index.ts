import { createDecoderPool } from './decoder-pool.ts'
import { makeScorePipeline, prepareScoreShader } from './custom-shader.ts'
import { checkDeviceLimits, checkedGpu, deviceLifetime } from './gpu.ts'
import { planetShaderBytes, packPlanetaryShaderPoints, type PlanetaryShaderData } from './planetShaderSchema.ts'
import { createLayout, PIXELS_PER_SITE, type ChunkLayout, type ScoreLayout } from './layout.ts'
import { parseManifest, WORKGROUP_SIZE, type FieldEncoding, type ScoreFeature } from './manifest.ts'
import { createQueryShaderSource, type EarthQuery } from './query.ts'
import { createLookup } from './scoring.ts'

export {
  DEFAULT_SCORE_SHADER,
  SCORE_SHADER_EXAMPLES,
  SCORE_SHADER_FIELDS,
  SCORE_SHADER_PRELUDE,
  ScoreShaderError,
} from './custom-shader.ts'
export type { ScoreShaderDiagnostic } from './custom-shader.ts'
export { freezePlanetaryShaderPoint } from './planetShaderSchema.ts'
export type {
  PlanetaryShaderCall,
  PlanetaryShaderData,
  PlanetaryShaderMeasurement,
  PlanetaryShaderSample,
} from './planetShaderSchema.ts'
export { buildEarthQuery } from './query.ts'
export type { EarthQuery, EarthQueryCriterion } from './query.ts'
export { SCORE_MAX, SCORE_MISSING, SOURCE_MISSING, sourceCodeWGSL } from './scoring.ts'
export type { ScoreAddress, ChunkLayout, ScoreLayout } from './layout.ts'
export type { FieldEncoding, ScoreFeature } from './manifest.ts'

export type ScoreChunk = Readonly<
  ChunkLayout & {
    packedInput: GPUBuffer
    scores: GPUBuffer
  }
>

export type EarthScore = Readonly<{
  layout: ScoreLayout
  fields: readonly Readonly<{ name: string; encoding: FieldEncoding }>[]
  baseline: readonly ScoreFeature[]
  lookup: GPUBuffer
  chunks: readonly ScoreChunk[]
  /** Generate editable WGSL for a validated criteria query on this dataset. */
  queryShaderSource(query: EarthQuery): string
  /** Compile and dispatch a WGSL module with an EarthCell scoring function. */
  dispatchShader(source: string, signal?: AbortSignal, points?: PlanetaryShaderData): Promise<void>
  /** Invalidate this instance's buffers. The caller still owns its GPUDevice. */
  destroy(): void
}>

export type EarthScoreOptions = Readonly<{
  manifestUrl: string | URL
  signal?: AbortSignal
  readAsset?: (url: string | URL, signal: AbortSignal) => Promise<Response>
}>

const workerCount = Math.min(6, Math.max(3, Math.floor((globalThis.navigator?.hardwareConcurrency || 4) * 0.6)))
const jobSlots = workerCount + 4
const shaderChunksPerFence = 4

// A caller can cancel while the shared queue is waiting on unrelated GPU work.
function waitForLoad<T>(work: Promise<T>, signal: AbortSignal) {
  return new Promise<T>((resolve, reject) => {
    function abort() {
      reject(signal.reason)
    }
    signal.addEventListener('abort', abort, { once: true })
    if (signal.aborted) abort()
    void work.then(resolve, reject).then(() => signal.removeEventListener('abort', abort))
  })
}

export async function loadEarthScore(
  device: GPUDevice,
  { manifestUrl, signal, readAsset = (url, signal) => fetch(url, { signal }) }: EarthScoreOptions,
): Promise<EarthScore> {
  signal?.throwIfAborted()
  const { controller, state: lifetime } = deviceLifetime(device)
  const loading = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal
  const buffers = new Set<GPUBuffer>()

  async function fetchAsset(url: string | URL) {
    const response = await readAsset(url, loading)
    if (!response.ok) throw new Error(`Earth Score asset ${url}: HTTP ${response.status}`)
    return response
  }

  function buffer(label: string, size: number, usage: number) {
    const gpuBuffer = device.createBuffer({ label, size, usage })
    buffers.add(gpuBuffer)
    return gpuBuffer
  }

  function releaseBuffers() {
    for (const gpuBuffer of buffers) gpuBuffer.destroy()
    buffers.clear()
  }

  try {
    loading.throwIfAborted()
    const response = await fetchAsset(manifestUrl)
    const manifest = parseManifest(await response.json())
    const base = response.url || new URL(manifestUrl, globalThis.location?.href).href
    checkDeviceLimits(device, manifest)
    const layout = createLayout(manifest)
    const fields = Object.freeze(
      manifest.columns.map((name, i) => Object.freeze({ name, encoding: manifest.encodings[i] })),
    )
    const queue = device.queue

    const { shaderLayout, shaderBindings, lookup, emptyPoints, firstSites, firstSiteStride } = await waitForLoad(
      checkedGpu(
        device,
        () => {
          const shaderBindings = device.createBindGroupLayout({
            label: 'Custom Earth score inputs',
            entries: [
              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
              { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
              { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
          })
          const shaderLayout = device.createPipelineLayout({ bindGroupLayouts: [shaderBindings] })
          const values = createLookup(manifest)
          const lookup = buffer(
            'Earth Score float lookup',
            values.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          )
          queue.writeBuffer(lookup, 0, values)
          const emptyPoints = buffer('WGSL empty planetary samples', 4, GPUBufferUsage.STORAGE)
          const firstSiteStride = device.limits.minUniformBufferOffsetAlignment
          const siteWords = new Uint32Array((firstSiteStride / 4) * layout.chunks.length)
          layout.chunks.forEach((range, i) => {
            siteWords[i * (firstSiteStride / 4)] = range.firstAddress / PIXELS_PER_SITE
          })
          const firstSites = buffer(
            'Earth Score first sites',
            siteWords.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          )
          queue.writeBuffer(firstSites, 0, siteWords)
          return { shaderLayout, shaderBindings, lookup, emptyPoints, firstSites, firstSiteStride }
        },
        'Earth Score GPU setup',
      ),
      loading,
    )

    const residents: { output: ScoreChunk; defaultGroup: GPUBindGroup; firstSiteOffset: number }[] = []
    const decoder = createDecoderPool(Math.min(workerCount, manifest.chunks.length), loading)
    try {
      const jobs = new Map<number, Promise<{ index: number; packed: ArrayBuffer } | { error: unknown }>>()
      let next = 0

      function launch() {
        if (next >= manifest.chunks.length) return
        const index = next++
        const { file, rows, compressedBytes, packedBytes } = manifest.chunks[index]
        const job = (async () => {
          const frame = await (await fetchAsset(new URL(file, base))).blob()
          loading.throwIfAborted()
          if (frame.size !== compressedBytes) throw new Error('Incorrect compressed length')
          const packed = await decoder.decode({ frame, packedBytes, rows, fields: fields.length })
          return { index, packed }
        })().catch(error => ({
          error: new Error(`Earth Score chunk ${file}: ${error instanceof Error ? error.message : String(error)}`, {
            cause: error,
          }),
        }))
        jobs.set(index, job)
      }

      for (let i = 0; i < jobSlots; i++) launch()
      const checks: Promise<unknown>[] = []
      let uploaded = 0
      while (jobs.size) {
        const ready = await Promise.race(jobs.values())
        loading.throwIfAborted()
        if ('error' in ready) throw ready.error
        const { index, packed } = ready
        const range = layout.chunks[index]
        jobs.delete(index)
        launch()
        const check = checkedGpu(
          device,
          () => {
            const inputBuffer = device.createBuffer({
              label: `Earth Score input ${index}`,
              size: packed.byteLength,
              usage: GPUBufferUsage.STORAGE,
              mappedAtCreation: true,
            })
            buffers.add(inputBuffer)
            new Uint8Array(inputBuffer.getMappedRange()).set(new Uint8Array(packed))
            inputBuffer.unmap()
            const scores = buffer(
              `Earth Score output ${index}`,
              range.rows * 4,
              GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            )
            const firstSiteOffset = index * firstSiteStride
            const defaultGroup = device.createBindGroup({
              layout: shaderBindings,
              entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: scores } },
                { binding: 2, resource: { buffer: lookup } },
                { binding: 3, resource: { buffer: emptyPoints } },
                { binding: 4, resource: { buffer: firstSites, offset: firstSiteOffset, size: 4 } },
              ],
            })
            residents[index] = {
              output: Object.freeze({ ...range, packedInput: inputBuffer, scores }),
              defaultGroup,
              firstSiteOffset,
            }
          },
          `Earth Score upload ${index}`,
        )
        checks.push(
          check.then(
            () => undefined,
            error => error,
          ),
        )
        // Flush pending unmap copies every 4 chunks without stalling on a CPU/GPU fence
        // so Dawn reclaims staging memory incrementally at full decode speed.
        if ((++uploaded & 3) === 0) queue.submit([])
      }
      if (uploaded & 3) queue.submit([])
      const completed = await waitForLoad(Promise.all(checks), loading)
      const error = completed.find(Boolean)
      if (error) throw error
    } finally {
      decoder.close()
    }
    loading.throwIfAborted()

    const chunks = Object.freeze(residents.map(({ output }) => output))
    const ready = { residents }
    let work: typeof ready | undefined = ready
    let shaderRevision = 0
    const shaderPipelines = new Map<string, Promise<GPUComputePipeline>>()

    function activeWork() {
      if (!work) throw new Error('Earth Score instance was destroyed')
      if (lifetime.lost) throw lifetime.lost
      return work
    }

    function submit(
      active: typeof residents,
      compute: GPUComputePipeline,
      bind: (resident: (typeof residents)[number]) => GPUBindGroup,
    ) {
      const encoder = device.createCommandEncoder({ label: 'Earth Score query' })
      const pass = encoder.beginComputePass()
      pass.setPipeline(compute)
      for (const resident of active) {
        pass.setBindGroup(0, bind(resident))
        pass.dispatchWorkgroups(Math.ceil(resident.output.rows / WORKGROUP_SIZE))
      }
      pass.end()
      queue.submit([encoder.finish()])
    }

    async function dispatchShader(source: string, signal?: AbortSignal, points?: PlanetaryShaderData) {
      signal?.throwIfAborted()
      const active = activeWork()
      const limits = device.limits
      const hasPoints = Boolean(points?.samples.length)
      if (
        planetShaderBytes(points?.samples.length ?? 0) >
        Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize)
      ) {
        throw new Error('The planetary point data exceeds the device’s storage binding limit.')
      }
      const shader = prepareScoreShader(source, manifest, points)
      const pointData = hasPoints ? packPlanetaryShaderPoints(points!.samples) : undefined
      const revision = ++shaderRevision
      function current() {
        signal?.throwIfAborted()
        if (revision !== shaderRevision) throw new DOMException('The score shader was superseded.', 'AbortError')
        activeWork()
      }

      let compilation = shaderPipelines.get(shader.code)
      if (compilation) {
        shaderPipelines.delete(shader.code)
        shaderPipelines.set(shader.code, compilation)
      } else {
        compilation = makeScorePipeline(device, shaderLayout, shader)
        shaderPipelines.set(shader.code, compilation)
        const oldest = shaderPipelines.keys().next().value
        if (shaderPipelines.size > 4 && oldest !== undefined) shaderPipelines.delete(oldest)
        void compilation.catch(() => {
          if (shaderPipelines.get(shader.code) === compilation) shaderPipelines.delete(shader.code)
        })
      }
      const pipeline = await (signal ? waitForLoad(compilation, signal) : compilation)
      current()
      let samplesBuffer: GPUBuffer | undefined
      try {
        // Start with a small probe batch so slow user shaders stay cancellable
        // and avoid GPU TDR, then widen to a single submit for fast shaders.
        let batchSize = shaderChunksPerFence
        for (let first = 0; first < active.residents.length; ) {
          current()
          const batch = active.residents.slice(first, first + batchSize)
          const batchStart = performance.now()
          const dispatched = checkedGpu(
            device,
            () => {
              if (pointData && !samplesBuffer) {
                samplesBuffer = buffer(
                  'WGSL planetary samples',
                  pointData.byteLength,
                  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                )
                queue.writeBuffer(samplesBuffer, 0, pointData)
              }
              const input = samplesBuffer
              submit(batch, pipeline, resident =>
                input
                  ? device.createBindGroup({
                      layout: shaderBindings,
                      entries: [
                        { binding: 0, resource: { buffer: resident.output.packedInput } },
                        { binding: 1, resource: { buffer: resident.output.scores } },
                        { binding: 2, resource: { buffer: lookup } },
                        { binding: 3, resource: { buffer: input } },
                        { binding: 4, resource: { buffer: firstSites, offset: resident.firstSiteOffset, size: 4 } },
                      ],
                    })
                  : resident.defaultGroup,
              )
            },
            'Custom Earth score dispatch',
          )
          const completed = await Promise.allSettled([dispatched, queue.onSubmittedWorkDone()])
          const failure = completed.find(result => result.status === 'rejected')
          if (failure) throw failure.reason
          first += batch.length
          const elapsed = performance.now() - batchStart
          batchSize =
            elapsed < 15 ? active.residents.length - first : Math.max(1, Math.floor((batch.length * 50) / elapsed))
        }
        current()
      } finally {
        if (samplesBuffer && buffers.delete(samplesBuffer)) samplesBuffer.destroy()
      }
    }

    return Object.freeze({
      fields,
      baseline: manifest.features,
      layout,
      lookup,
      chunks,
      dispatchShader,
      queryShaderSource(query) {
        const limits = device.limits
        return createQueryShaderSource(
          manifest,
          query,
          Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize),
        )
      },
      destroy() {
        work = undefined
        shaderPipelines.clear()
        releaseBuffers()
      },
    })
  } catch (error) {
    releaseBuffers()
    throw error
  } finally {
    lifetime.loading = undefined
    controller.abort()
  }
}
