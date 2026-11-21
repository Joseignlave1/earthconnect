import type { createDecoder, DecodeRequest } from './decode.ts'
import type { DecodeResult } from './decode.worker.ts'
import { createDecoderWorker } from './decoder-worker.ts'

type Job = {
  request: DecodeRequest
  resolve: (packed: ArrayBuffer) => void
  reject: (error: unknown) => void
}

type Slot = { worker: Worker; ready: boolean; sentModule: boolean; job?: Job }

export function createDecoderPool(count: number, signal: AbortSignal) {
  const slots: Slot[] = []
  const pending: Job[] = []
  let mainThreadDecoder: Promise<ReturnType<typeof createDecoder>> | undefined
  let fallbackJob: Job | undefined
  let closed: unknown
  let compiledModule: WebAssembly.Module | undefined

  function close(reason: unknown = new Error('Earth Score decoder pool closed')) {
    if (closed !== undefined) return
    closed = reason
    signal.removeEventListener('abort', abort)
    for (const { worker, job } of slots) {
      worker.terminate()
      job?.reject(reason)
    }
    slots.length = 0
    fallbackJob?.reject(reason)
    fallbackJob = undefined
    for (const job of pending.splice(0)) job.reject(reason)
  }

  function abort() {
    close(signal.reason)
  }

  function start(slot: Slot) {
    const job = pending.shift()
    if (!job) return
    slot.job = job
    try {
      const message = !slot.sentModule && compiledModule
        ? { ...job.request, module: compiledModule }
        : job.request
      if (compiledModule) slot.sentModule = true
      const transfer = job.request.frame instanceof ArrayBuffer ? [job.request.frame] : []
      slot.worker.postMessage(message, transfer)
    } catch (error) {
      close(error)
    }
  }

  function schedule() {
    if (closed !== undefined || !pending.length) return
    while (pending.length) {
      const idle = slots.find(slot => slot.ready && !slot.job)
      if (!idle) break
      start(idle)
    }
    if (!slots.length && !fallbackJob && pending.length) {
      const job = pending.shift()!
      fallbackJob = job
      // Let the browser update the UI between chunks when workers cannot start.
      setTimeout(() => {
        if (fallbackJob !== job) return
        mainThreadDecoder ??= import('./decode.ts').then(({ createDecoder }) => createDecoder(signal))
        const request = compiledModule ? { ...job.request, module: compiledModule } : job.request
        void mainThreadDecoder.then(decode => decode(request)).then(job.resolve, job.reject).finally(() => {
          fallbackJob = undefined
          schedule()
        })
      }, 0)
    }
  }

  signal.throwIfAborted()
  signal.addEventListener('abort', abort, { once: true })
  // Compile zstd.wasm once on the main thread while workers boot; if ready
  // before a worker receives its first job, share the compiled Module to avoid
  // redundant fetches and compilations (without ever blocking worker dispatch).
  void import('./decode.ts').then(({ compileZstdModule }) => compileZstdModule(signal)).then(module => {
    compiledModule = module
  }, () => undefined)
  for (let i = 0; i < count; i++) {
    let worker: Worker
    try {
      worker = createDecoderWorker()
    } catch {
      continue
    }
    const slot: Slot = { worker, ready: false, sentModule: false }
    slots.push(slot)
    worker.onmessage = ({ data }: MessageEvent<DecodeResult>) => {
      if ('ready' in data) {
        slot.ready = true
        schedule()
        return
      }
      const job = slot.job
      slot.job = undefined
      if (!job) return
      if ('error' in data) job.reject(new Error(data.error))
      else job.resolve(data.packed)
      schedule()
    }
    worker.onerror = (event) => {
      event.preventDefault()
      if (closed !== undefined) return
      if (slot.ready) {
        close(new Error(event.message || 'Earth Score decoder worker failed'))
        return
      }
      // Keep queued frames intact until a worker confirms it can accept them.
      worker.terminate()
      slots.splice(slots.indexOf(slot), 1)
      schedule()
    }
    worker.onmessageerror = () => close(new Error('Earth Score decoder message could not be read'))
  }

  return {
    decode(request: DecodeRequest) {
      if (closed !== undefined) return Promise.reject<ArrayBuffer>(closed)
      return new Promise<ArrayBuffer>((resolve, reject) => {
        pending.push({ request, resolve, reject })
        schedule()
      })
    },
    close,
  }
}
