export type DeferredPipeline = {
  /** Return the compiled pipeline if ready; otherwise promote it to the priority queue and compile synchronously. */
  get: () => GPURenderPipeline
  /** Return the compiled pipeline if already ready without stalling; promotes it to the priority queue if pending. */
  peek: () => GPURenderPipeline | undefined
  /** Move a pending shader to the back of the urgent priority queue (drained before the normal queue). */
  promote: () => void
  /** True once the pipeline has finished compiling. */
  readonly ready: boolean
  /** True if async compilation failed or the queue was disposed before completion. */
  readonly failed: boolean
  /** Failure reason when `failed` is true. */
  readonly error: unknown
}

export type DeferredComputePipeline = {
  /** Return the compiled compute pipeline if ready; otherwise compile synchronously once. */
  get: () => GPUComputePipeline
  /** Return the compiled compute pipeline if ready without stalling; promotes it if pending. */
  peek: () => GPUComputePipeline | undefined
  /** Move a pending compute shader to the back of the urgent priority queue. */
  promote: () => void
  /** True once the compute pipeline has finished compiling. */
  readonly ready: boolean
  /** True if async compilation failed or the queue was disposed before completion. */
  readonly failed: boolean
  /** Failure reason when `failed` is true. */
  readonly error: unknown
}

type QueueEntry =
  | {
      kind: 'render'
      order: number
      seq: number
      build: () => GPURenderPipelineDescriptor
      descriptor?: GPURenderPipelineDescriptor
      pipeline?: GPURenderPipeline
      error?: unknown
      state: 'normal' | 'priority' | 'compiling' | 'ready' | 'failed'
    }
  | {
      kind: 'compute'
      order: number
      seq: number
      build: () => GPUComputePipelineDescriptor
      descriptor?: GPUComputePipelineDescriptor
      pipeline?: GPUComputePipeline
      error?: unknown
      state: 'normal' | 'priority' | 'compiling' | 'ready' | 'failed'
    }

const DEFAULT_CONCURRENCY = 2

/**
 * Per-GPUDevice eager shader compilation queue with priority promotion.
 *
 * Critical first-frame pipelines compile directly at startup; all secondary pipelines register here
 * in a sensible default interaction order (`order`, lower compiles first). They compile eagerly in
 * the background via `createRenderPipelineAsync` with bounded concurrency so startup stays fast and
 * every shader finishes compiling shortly after load. Requesting/promoting a not-yet-compiled shader
 * moves it to the back of `priorityQueue`, which is always drained ahead of `normalQueue`.
 */
export class PipelineCompileQueue {
  private readonly device: GPUDevice
  private readonly maxConcurrent: number
  private readonly yieldTurn: (fn: () => void) => void
  private readonly priorityQueue: QueueEntry[] = []
  private readonly normalQueue: QueueEntry[] = []
  private active = 0
  private seq = 0
  private scheduled = false
  private disposed = false

  constructor(
    device: GPUDevice,
    {
      maxConcurrent = DEFAULT_CONCURRENCY,
      yieldTurn = fn => {
        setTimeout(fn, 0)
      },
    }: { maxConcurrent?: number; yieldTurn?: (fn: () => void) => void } = {},
  ) {
    this.device = device
    this.maxConcurrent = Math.max(1, maxConcurrent)
    this.yieldTurn = yieldTurn
    void device.lost?.then(() => this.dispose())
  }

  register(order: number, build: () => GPURenderPipelineDescriptor): DeferredPipeline {
    const entry: Extract<QueueEntry, { kind: 'render' }> = {
      kind: 'render',
      order,
      seq: this.seq++,
      build,
      state: this.disposed ? 'failed' : 'normal',
      error: this.disposed ? new Error('PipelineCompileQueue has been disposed.') : undefined,
    }
    if (!this.disposed) {
      this.insertNormal(entry)
      this.schedulePump()
    }

    const promote = () => this.promoteEntry(entry)

    const get = (): GPURenderPipeline => {
      if (entry.pipeline) return entry.pipeline
      if (this.disposed) throw entry.error ?? new Error('PipelineCompileQueue has been disposed.')
      this.dequeueWaiting(entry)
      const desc = (entry.descriptor ??= entry.build())
      const pipeline = this.device.createRenderPipeline(desc)
      entry.pipeline = pipeline
      entry.error = undefined
      entry.state = 'ready'
      return pipeline
    }

    const peek = (): GPURenderPipeline | undefined => {
      if (entry.pipeline) return entry.pipeline
      promote()
      return undefined
    }

    return {
      get,
      peek,
      promote,
      get ready() {
        return entry.pipeline !== undefined
      },
      get failed() {
        return entry.state === 'failed'
      },
      get error() {
        return entry.error
      },
    }
  }

  registerCompute(order: number, build: () => GPUComputePipelineDescriptor): DeferredComputePipeline {
    const entry: Extract<QueueEntry, { kind: 'compute' }> = {
      kind: 'compute',
      order,
      seq: this.seq++,
      build,
      state: this.disposed ? 'failed' : 'normal',
      error: this.disposed ? new Error('PipelineCompileQueue has been disposed.') : undefined,
    }
    if (!this.disposed) {
      this.insertNormal(entry)
      this.schedulePump()
    }

    const promote = () => this.promoteEntry(entry)

    const get = (): GPUComputePipeline => {
      if (entry.pipeline) return entry.pipeline
      if (this.disposed) throw entry.error ?? new Error('PipelineCompileQueue has been disposed.')
      this.dequeueWaiting(entry)
      const desc = (entry.descriptor ??= entry.build())
      const pipeline = this.device.createComputePipeline(desc)
      entry.pipeline = pipeline
      entry.error = undefined
      entry.state = 'ready'
      return pipeline
    }

    const peek = (): GPUComputePipeline | undefined => {
      if (entry.pipeline) return entry.pipeline
      promote()
      return undefined
    }

    return {
      get,
      peek,
      promote,
      get ready() {
        return entry.pipeline !== undefined
      },
      get failed() {
        return entry.state === 'failed'
      },
      get error() {
        return entry.error
      },
    }
  }

  private promoteEntry(entry: QueueEntry) {
    if (this.disposed || entry.state !== 'normal') return
    const idx = this.normalQueue.indexOf(entry)
    if (idx >= 0) this.normalQueue.splice(idx, 1)
    entry.state = 'priority'
    this.priorityQueue.push(entry)
    this.schedulePump()
  }

  private dequeueWaiting(entry: QueueEntry) {
    if (entry.state === 'normal') {
      const idx = this.normalQueue.indexOf(entry)
      if (idx >= 0) this.normalQueue.splice(idx, 1)
    } else if (entry.state === 'priority') {
      const idx = this.priorityQueue.indexOf(entry)
      if (idx >= 0) this.priorityQueue.splice(idx, 1)
    }
  }

  private insertNormal(entry: QueueEntry) {
    let i = this.normalQueue.length
    while (
      i > 0 &&
      (this.normalQueue[i - 1].order > entry.order ||
        (this.normalQueue[i - 1].order === entry.order && this.normalQueue[i - 1].seq > entry.seq))
    ) {
      i--
    }
    this.normalQueue.splice(i, 0, entry)
  }

  private schedulePump() {
    if (this.disposed || this.scheduled || this.active >= this.maxConcurrent) return
    if (!this.priorityQueue.length && !this.normalQueue.length) return
    this.scheduled = true
    this.yieldTurn(() => {
      this.scheduled = false
      this.pump()
    })
  }

  private popNext(): QueueEntry | undefined {
    if (this.priorityQueue.length > 0) return this.priorityQueue.shift()
    return this.normalQueue.shift()
  }

  private pump() {
    if (this.disposed) return
    while (this.active < this.maxConcurrent) {
      const entry = this.popNext()
      if (!entry) return
      if (entry.state === 'ready' || entry.pipeline) continue
      entry.state = 'compiling'
      this.active++
      let compilePromise: Promise<void>
      try {
        if (entry.kind === 'render') {
          const desc = (entry.descriptor ??= entry.build())
          compilePromise = this.device.createRenderPipelineAsync(desc).then(
            pipeline => {
              if (this.disposed) return
              if (!entry.pipeline) entry.pipeline = pipeline
              entry.error = undefined
              entry.state = 'ready'
            },
            err => {
              if (this.disposed) return
              if (!entry.pipeline) {
                entry.error = err
                entry.state = 'failed'
              }
            },
          )
        } else {
          const desc = (entry.descriptor ??= entry.build())
          compilePromise = this.device.createComputePipelineAsync(desc).then(
            pipeline => {
              if (this.disposed) return
              if (!entry.pipeline) entry.pipeline = pipeline
              entry.error = undefined
              entry.state = 'ready'
            },
            err => {
              if (this.disposed) return
              if (!entry.pipeline) {
                entry.error = err
                entry.state = 'failed'
              }
            },
          )
        }
      } catch (err) {
        entry.error = err
        entry.state = 'failed'
        this.active--
        continue
      }
      void compilePromise.finally(() => {
        this.active--
        this.schedulePump()
      })
    }
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.priorityQueue.length = 0
    this.normalQueue.length = 0
  }
}

const deviceQueues = new WeakMap<GPUDevice, PipelineCompileQueue>()

export function getPipelineCompileQueue(device: GPUDevice): PipelineCompileQueue {
  let queue = deviceQueues.get(device)
  if (!queue) {
    queue = new PipelineCompileQueue(device)
    deviceQueues.set(device, queue)
  }
  return queue
}
