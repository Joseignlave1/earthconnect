import { createDecoderPool } from './decoder-pool.ts'
import { makeScorePipeline, prepareScoreShader } from './custom-shader.ts'
import { checkDeviceLimits, checkedGpu, deviceLifetime } from './gpu.ts'
import { planetShaderBytes, packPlanetaryShaderPoints, type PlanetaryShaderData } from './planetShaderSchema.ts'
import { createLayout, PIXELS_PER_SITE, type ChunkLayout, type ScoreLayout } from './layout.ts'
import { parseManifest, WORKGROUP_SIZE, type FieldEncoding, type ScoreFeature } from './manifest.ts'
import { createQueryParameters, createQueryShaderSource, queryShader, type EarthQuery } from './query.ts'
import { createLookup, createParameters, PARAMETER_BYTES, scoreShader, type ScoreOverrides } from './scoring.ts'

export { DEFAULT_SCORE_SHADER, SCORE_SHADER_EXAMPLES, SCORE_SHADER_FIELDS, SCORE_SHADER_PRELUDE, ScoreShaderError } from './custom-shader.ts'
export type { ScoreShaderDiagnostic } from './custom-shader.ts'
export { freezePlanetaryShaderPoint } from './planetShaderSchema.ts'
export type { PlanetaryShaderCall, PlanetaryShaderData, PlanetaryShaderMeasurement, PlanetaryShaderSample } from './planetShaderSchema.ts'
export { buildEarthQuery } from './query.ts'
export type { EarthQuery, EarthQueryCriterion } from './query.ts'
export { SCORE_MAX, SCORE_MISSING, SOURCE_MISSING, sourceCodeWGSL } from './scoring.ts'
export type { ScoreOverrides } from './scoring.ts'
export type { ScoreAddress, ChunkLayout, ScoreLayout } from './layout.ts'
export type { FieldEncoding, ScoreFeature } from './manifest.ts'

export type ScoreChunk = Readonly<ChunkLayout & {
  packedInput: GPUBuffer
  scores: GPUBuffer
}>

export type EarthScore = Readonly<{
  layout: ScoreLayout
  fields: readonly Readonly<{ name: string; encoding: FieldEncoding }>[]
  baseline: readonly ScoreFeature[]
  lookup: GPUBuffer
  chunks: readonly ScoreChunk[]
  /** Enqueue a new full-world query immediately, then resolve on GPU completion. */
  dispatch(overrides?: ScoreOverrides): Promise<void>
  /** Score the caller's criteria into the shared chunk score buffers. */
  dispatchQuery(query: EarthQuery): Promise<void>
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

const workerCount = 4
const jobSlots = 8
const uploadsPerFence = 8
const shaderChunksPerFence = 4

// A caller can cancel while the shared queue is waiting on unrelated GPU work.
function waitForLoad<T>(work: Promise<T>, signal: AbortSignal) {
  return new Promise<T>((resolve, reject) => {
    function abort() { reject(signal.reason) }
    signal.addEventListener('abort', abort, { once: true })
    if (signal.aborted) abort()
    void work.then(resolve, reject).then(() => signal.removeEventListener('abort', abort))
  })
}

export async function loadEarthScore(device: GPUDevice, {
  manifestUrl, signal, readAsset = (url, signal) => fetch(url, { signal }),
}: EarthScoreOptions): Promise<EarthScore> {
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
    const fields = Object.freeze(manifest.columns.map((name, i) => Object.freeze({ name, encoding: manifest.encodings[i] })))
    const baselineParameters = createParameters(manifest)
    const queue = device.queue

    const { pipeline, bindings, queryPipeline, queryBindings, lookup } = await waitForLoad(checkedGpu(device, () => {
      const module = device.createShaderModule({ label: 'Earth Score shader', code: scoreShader(manifest) })
      const pipeline = device.createComputePipeline({
        label: 'Earth Score', layout: 'auto', compute: { module, entryPoint: 'score' },
      })
      const queryModule = device.createShaderModule({ label: 'Earth Score criteria shader', code: queryShader })
      const queryPipeline = device.createComputePipeline({
        label: 'Earth Score criteria', layout: 'auto', compute: { module: queryModule, entryPoint: 'score' },
      })
      const values = createLookup(manifest)
      const lookup = buffer('Earth Score float lookup', values.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
      queue.writeBuffer(lookup, 0, values)
      return { pipeline, bindings: pipeline.getBindGroupLayout(0), queryPipeline,
        queryBindings: queryPipeline.getBindGroupLayout(0), lookup }
    }, 'Earth Score GPU setup'), loading)

    const residents: { output: ScoreChunk; parameters: GPUBuffer; firstSite: GPUBuffer; binding: GPUBindGroup }[] = []
    const decoder = createDecoderPool(Math.min(workerCount, manifest.chunks.length), loading)
    try {
      const jobs = new Map<number, Promise<{ index: number; packed: ArrayBuffer } | { error: unknown }>>()
      let next = 0

      function launch() {
        if (next >= manifest.chunks.length) return
        const index = next++
        const { file, rows, compressedBytes, packedBytes } = manifest.chunks[index]
        const job = (async () => {
          const frame = await (await fetchAsset(new URL(file, base))).arrayBuffer()
          loading.throwIfAborted()
          if (frame.byteLength !== compressedBytes) throw new Error('Incorrect compressed length')
          const packed = await decoder.decode({ frame, packedBytes, rows, fields: fields.length })
          return { index, packed }
        })().catch(error => ({ error: new Error(`Earth Score chunk ${file}: ${error instanceof Error ? error.message : String(error)}`, { cause: error }) }))
        jobs.set(index, job)
      }

      for (let i = 0; i < jobSlots; i++) launch()
      const checks: Promise<unknown>[] = []
      while (jobs.size) {
        const ready = await Promise.race(jobs.values())
        loading.throwIfAborted()
        if ('error' in ready) throw ready.error
        const { index, packed } = ready
        const range = layout.chunks[index]
        jobs.delete(index)
        const check = checkedGpu(device, () => {
          const packedInput = buffer(`Earth Score input ${index}`, packed.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
          const scores = buffer(`Earth Score output ${index}`, range.rows * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)
          const parameters = buffer(`Earth Score parameters ${index}`, PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
          const firstSite = buffer(`Earth Score first site ${index}`, 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
          queue.writeBuffer(firstSite, 0, new Uint32Array([range.firstAddress / PIXELS_PER_SITE]))
          const binding = device.createBindGroup({
            layout: bindings,
            entries: [packedInput, scores, parameters, lookup].map((buffer, binding) => ({ binding, resource: { buffer } })),
          })
          queue.writeBuffer(packedInput, 0, packed)
          residents[index] = { output: Object.freeze({ ...range, packedInput, scores }), parameters, firstSite, binding }
        }, `Earth Score upload ${index}`)
        checks.push(check.then(() => undefined, error => error))
        launch()
        // Already admitted fetches/decodes can finish while this fence is pending.
        if (checks.length === uploadsPerFence || !jobs.size) {
          const fence = Promise.all([Promise.all(checks.splice(0)), queue.onSubmittedWorkDone()])
          const [completed] = await waitForLoad(fence, loading)
          const error = completed.find(Boolean)
          if (error) throw error
        }
      }
    } finally {
      decoder.close()
    }
    loading.throwIfAborted()

    const chunks = Object.freeze(residents.map(({ output }) => output))
    const ready = { residents, pipeline, queryPipeline, queryBindings }
    let work: typeof ready | undefined = ready
    let shaderRevision = 0
    let shaderBindings: Promise<{ layout: GPUPipelineLayout; bindings: GPUBindGroupLayout }> | undefined
    const shaderPipelines = new Map<string, Promise<GPUComputePipeline>>()

    function activeWork() {
      if (!work) throw new Error('Earth Score instance was destroyed')
      if (lifetime.lost) throw lifetime.lost
      return work
    }

    function submit(active: typeof residents, compute: GPUComputePipeline, bind: (resident: typeof residents[number]) => GPUBindGroup) {
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

    async function dispatch(overrides?: ScoreOverrides) {
      const active = activeWork()
      shaderRevision++
      const words = overrides ? createParameters(manifest, overrides) : baselineParameters
      const dispatched = checkedGpu(device, () => {
        submit(active.residents, active.pipeline, ({ output, parameters, binding }) => {
          words[0] = output.rows
          queue.writeBuffer(parameters, 0, words)
          return binding
        })
      }, 'Earth Score dispatch')
      await Promise.all([dispatched, queue.onSubmittedWorkDone()])
      if (lifetime.lost) throw lifetime.lost
    }

    async function dispatchQuery(query: EarthQuery) {
      const active = activeWork()
      shaderRevision++
      const limits = device.limits
      const words = createQueryParameters(manifest, query, Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize))
      let parameters: GPUBuffer | undefined
      try {
        const dispatched = checkedGpu(device, () => {
          const params = buffer('Earth Score criteria parameters', words.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
          parameters = params
          queue.writeBuffer(params, 0, words)
          submit(active.residents, active.queryPipeline, ({ output }) => device.createBindGroup({
            layout: active.queryBindings,
            entries: [output.packedInput, output.scores, params].map((buffer, binding) => ({ binding, resource: { buffer } })),
          }))
        }, 'Earth Score criteria dispatch')
        await Promise.all([dispatched, queue.onSubmittedWorkDone()])
        if (lifetime.lost) throw lifetime.lost
      } finally {
        if (parameters && buffers.delete(parameters)) parameters.destroy()
      }
    }

    async function dispatchShader(source: string, signal?: AbortSignal, points?: PlanetaryShaderData) {
      signal?.throwIfAborted()
      const active = activeWork()
      const limits = device.limits
      if (planetShaderBytes(points?.samples.length ?? 0) > Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize)) {
        throw new Error('The planetary point data exceeds the device’s storage binding limit.')
      }
      const shader = prepareScoreShader(source, manifest, points)
      const pointData = packPlanetaryShaderPoints(points?.samples ?? [])
      const revision = ++shaderRevision
      function current() {
        signal?.throwIfAborted()
        if (revision !== shaderRevision) throw new DOMException('The score shader was superseded.', 'AbortError')
        activeWork()
      }

      shaderBindings ??= checkedGpu(device, () => {
        if (limits.maxStorageBuffersPerShaderStage < 4) {
          throw new Error('Planetary WGSL scoring requires four WebGPU storage bindings.')
        }
        const bindings = device.createBindGroupLayout({
          label: 'Custom Earth score inputs',
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          ],
        })
        return { layout: device.createPipelineLayout({ bindGroupLayouts: [bindings] }), bindings }
      }, 'Custom Earth score GPU setup').catch(error => {
        shaderBindings = undefined
        throw error
      })
      const { layout, bindings } = await (signal ? waitForLoad(shaderBindings, signal) : shaderBindings)
      current()
      let compilation = shaderPipelines.get(shader.code)
      if (!compilation) {
        compilation = makeScorePipeline(device, layout, shader)
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
        // An aborted run stops submitting after the current batch finishes.
        for (let first = 0; first < active.residents.length; first += shaderChunksPerFence) {
          current()
          const batch = active.residents.slice(first, first + shaderChunksPerFence)
          const dispatched = checkedGpu(device, () => {
            if (!samplesBuffer) {
              samplesBuffer = buffer('WGSL planetary samples', pointData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
              queue.writeBuffer(samplesBuffer, 0, pointData)
            }
            const input = samplesBuffer
            submit(batch, pipeline, ({ output, firstSite }) => device.createBindGroup({
              layout: bindings,
              entries: [output.packedInput, output.scores, lookup, input, firstSite].map((buffer, binding) => ({ binding, resource: { buffer } })),
            }))
          }, 'Custom Earth score dispatch')
          const completed = await Promise.allSettled([dispatched, queue.onSubmittedWorkDone()])
          const failure = completed.find(result => result.status === 'rejected')
          if (failure) throw failure.reason
        }
        current()
      } finally {
        if (samplesBuffer && buffers.delete(samplesBuffer)) samplesBuffer.destroy()
      }
    }

    return Object.freeze({
      fields, baseline: manifest.features, layout, lookup, chunks, dispatch, dispatchQuery, dispatchShader,
      queryShaderSource(query) {
        const limits = device.limits
        return createQueryShaderSource(manifest, query, Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize))
      },
      destroy() {
        work = undefined
        shaderBindings = undefined
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
