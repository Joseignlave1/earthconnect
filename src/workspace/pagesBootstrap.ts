import type { WorkspaceAssets } from './workspaceAssets.ts'
import { loadWorkspaceRelease, type DownloadProgress } from './workspaceRelease.ts'
import { createSurfaceDevicePool, WEBGPU_UNAVAILABLE_MESSAGE } from '../shader-test/surface-device.ts'
import type { WorkspacePrewarm } from './workspacePrewarm.ts'
import { endStartupSpan, markStartup, startStartupSpan } from './startupTiming.ts'

markStartup('bootstrap')

if (!navigator.gpu) {
  const initial = document.querySelector<HTMLElement>('#initial')
  const status = document.querySelector<HTMLElement>('#initial-status')
  if (status) {
    status.setAttribute('role', 'alert')
    status.style.color = '#f7f9ff'
    status.textContent = WEBGPU_UNAVAILABLE_MESSAGE
  }
  initial?.setAttribute('data-visible', 'true')
} else {
  // Start WebGPU adapter/device acquisition immediately in parallel with IDB and JS loading.
  const devicePool = createSurfaceDevicePool()
  startStartupSpan('gpu-device')
  const initialDeviceLease = devicePool()
  void initialDeviceLease.ready.then(
    () => endStartupSpan('gpu-device'),
    () => undefined,
  )
  let deviceClaimed = false
  function releaseUnclaimedDevice() {
    if (deviceClaimed) return
    deviceClaimed = true
    initialDeviceLease.release()
  }

  // Load the lightweight prewarm module immediately so Earth terrain + 36 Zstd
  // world chunks start decoding the instant IDB + WebGPU settle—before Monaco
  // mounts and outside React StrictMode double-effects.
  const prewarmModule = import('./workspacePrewarm.ts')
  let prewarmPromise: Promise<WorkspacePrewarm> | undefined
  function startBootstrapPrewarm(assets: WorkspaceAssets) {
    if (prewarmPromise) return
    const lease = deviceClaimed ? undefined : initialDeviceLease
    deviceClaimed = true
    prewarmPromise = prewarmModule.then(({ startWorkspacePrewarm }) => startWorkspacePrewarm(assets, devicePool, lease))
    void prewarmPromise.catch(() => lease?.release())
  }

  // Start the release check while the UI bundle loads. Keep the latest events
  // for React's first render and let retries use the ordinary release loader.
  let latestAssets: WorkspaceAssets | undefined
  let latestProgress: DownloadProgress | undefined
  let listener:
    | {
        onAssets?: (assets: WorkspaceAssets) => void
        onProgress: (progress: DownloadProgress) => void
      }
    | undefined
  const refresh = new URLSearchParams(location.search).get('refreshCache') === '1'
  const initialRelease = loadWorkspaceRelease(
    new AbortController().signal,
    progress => {
      latestProgress = progress
      listener?.onProgress(progress)
    },
    {
      refresh,
      onAssets: assets => {
        latestAssets = assets
        startBootstrapPrewarm(assets)
        listener?.onAssets?.(assets)
      },
    },
  )
  // A failed network check can settle before the UI starts observing it.
  void initialRelease.catch(() => undefined)

  const loadInitialRelease: typeof loadWorkspaceRelease = async (signal, onProgress, options) => {
    signal.throwIfAborted()
    const subscription = { onAssets: options?.onAssets, onProgress }
    listener = subscription
    function unsubscribe() {
      if (listener === subscription) listener = undefined
    }
    signal.addEventListener('abort', unsubscribe, { once: true })
    try {
      if (latestProgress) onProgress(latestProgress)
      if (latestAssets) subscription.onAssets?.(latestAssets)
      return await initialRelease
    } finally {
      unsubscribe()
      signal.removeEventListener('abort', unsubscribe)
    }
  }

  // Only eager-fetch the heavy Monaco / WGSL workspace bundle when landing directly on /find.
  // On the landing page (/), defer it until the user opens the workspace so startup stays lean.
  const path = location.pathname.replace(/\/+$/, '') || '/'
  if (path === '/find' || path.startsWith('/find/')) {
    startStartupSpan('workspace-js')
    void import('./SharedSkyWorkspace.tsx').then(
      () => endStartupSpan('workspace-js'),
      () => undefined,
    )
  }
  startStartupSpan('app-bundle')
  void import('./pagesApp.tsx')
    .then(({ mount }) => {
      endStartupSpan('app-bundle')
      mount(loadInitialRelease, devicePool, releaseUnclaimedDevice, () => prewarmPromise)
    })
    .catch(() => {
      releaseUnclaimedDevice()
      document.querySelector('#initial')?.setAttribute('data-visible', 'true')
      document.querySelector('#initial-status')!.textContent = 'Could not open Earth Connect. Reload to retry.'
    })
}
