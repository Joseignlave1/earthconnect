type SurfaceDevice = { device: GPUDevice; adapter: GPUAdapter }
type DeviceEntry = { ready: Promise<SurfaceDevice>; users: number; lost: Set<(info: GPUDeviceLostInfo) => void> }

async function createSurfaceDevice(): Promise<SurfaceDevice> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('Could not initialize a WebGPU adapter.')
  const requiredFeatures = (['float32-filterable', 'timestamp-query'] as const)
    .filter(feature => adapter.features.has(feature))
  return { adapter, device: await adapter.requestDevice({ requiredFeatures }) }
}

export function createSurfaceDevicePool() {
  let current: DeviceEntry | undefined

  return function acquireDevice() {
    if (!current) {
      const entry: DeviceEntry = { ready: createSurfaceDevice(), users: 0, lost: new Set() }
      current = entry
      const retire = () => {
        if (current === entry) current = undefined
      }
      void entry.ready.then(({ device }) => device.lost.then(info => {
        retire()
        for (const listener of entry.lost) listener(info)
      }), retire)
    }
    const entry = current
    entry.users++
    let listener: ((info: GPUDeviceLostInfo) => void) | undefined
    let released = false
    return {
      ready: entry.ready,
      onLost(callback: (info: GPUDeviceLostInfo) => void) {
        if (released) return
        if (listener) entry.lost.delete(listener)
        listener = callback
        entry.lost.add(callback)
      },
      release() {
        if (released) return
        released = true
        if (listener) entry.lost.delete(listener)
        if (--entry.users) return
        if (current === entry) current = undefined
        void entry.ready.then(({ device }) => device.destroy(), () => {})
      },
    }
  }
}
