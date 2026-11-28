import type { TitleDustAssets } from './constellationTitle.ts'
import { type LogoLayoutRect, StandaloneLogoRenderer } from './standaloneLogoRenderer.ts'

export type { LogoLayoutRect }

export type LogoWorkerInMessage =
  | {
      type: 'boot'
      canvas: OffscreenCanvas
      layout: LogoLayoutRect
      reducedMotion: boolean
    }
  | { type: 'assets'; assets: TitleDustAssets }
  | { type: 'layout'; layout: LogoLayoutRect }
  | { type: 'pointer'; x: number | null; y: number | null }
  | { type: 'reducedMotion'; value: boolean }
  | { type: 'destroy' }

export type LogoWorkerOutMessage = { type: 'ready' } | { type: 'error'; message: string }

const send = globalThis.postMessage as (message: LogoWorkerOutMessage) => void

let disposed = false
let raf = 0
let renderer: StandaloneLogoRenderer | null = null
let layout: LogoLayoutRect | null = null
let reducedMotion = false
let announcedReady = false
let pendingPointer: { x: number | null; y: number | null } | null = null

let resolveAssets!: (assets: TitleDustAssets) => void
const assetsPromise = new Promise<TitleDustAssets>(resolve => {
  resolveAssets = resolve
})

function cleanup() {
  if (disposed) return
  disposed = true
  cancelAnimationFrame(raf)
  raf = 0
  renderer?.destroy()
  renderer = null
}

function scheduleFrame() {
  if (disposed || !renderer || !layout || raf) return
  raf = requestAnimationFrame(frame)
}

function frame() {
  raf = 0
  if (disposed || !renderer || !layout) return
  if (renderer.renderFrame(layout, reducedMotion) && !announcedReady) {
    announcedReady = true
    requestAnimationFrame(() => {
      if (!disposed) send({ type: 'ready' })
    })
  }
  if (!reducedMotion) {
    raf = requestAnimationFrame(frame)
  }
}

async function boot(msg: Extract<LogoWorkerInMessage, { type: 'boot' }>) {
  try {
    layout = msg.layout
    reducedMotion = msg.reducedMotion
    const created = await StandaloneLogoRenderer.create(msg.canvas, assetsPromise, msg.layout, () => disposed)
    if (!created || disposed) return
    renderer = created
    if (pendingPointer) {
      renderer.setPointer(pendingPointer.x, pendingPointer.y)
      pendingPointer = null
    }
    scheduleFrame()
  } catch (err) {
    if (!disposed) {
      cleanup()
      send({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }
}

globalThis.onmessage = (event: MessageEvent<LogoWorkerInMessage>) => {
  const msg = event.data
  if (msg.type === 'boot') {
    void boot(msg)
  } else if (msg.type === 'assets') {
    resolveAssets(msg.assets)
  } else if (msg.type === 'layout') {
    layout = msg.layout
    scheduleFrame()
  } else if (msg.type === 'pointer') {
    if (renderer) {
      renderer.setPointer(msg.x, msg.y)
      if (!reducedMotion) scheduleFrame()
    } else {
      pendingPointer = { x: msg.x, y: msg.y }
    }
  } else if (msg.type === 'reducedMotion') {
    reducedMotion = msg.value
    if (reducedMotion && raf) {
      cancelAnimationFrame(raf)
      raf = 0
    }
    scheduleFrame()
  } else if (msg.type === 'destroy') {
    cleanup()
    globalThis.close()
  }
}
