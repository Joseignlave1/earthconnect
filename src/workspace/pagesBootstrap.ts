import type { WorkspaceAssets } from './workspaceAssets.ts'
import { loadWorkspaceRelease, type DownloadProgress } from './workspaceRelease.ts'

// Start the release check while the UI bundle loads. Keep the latest events
// for React's first render and let retries use the ordinary release loader.
let latestAssets: WorkspaceAssets | undefined
let latestProgress: DownloadProgress | undefined
let listener: {
  onAssets?: (assets: WorkspaceAssets) => void
  onProgress: (progress: DownloadProgress) => void
} | undefined
const refresh = new URLSearchParams(location.search).get('refreshCache') === '1'
const initialRelease = loadWorkspaceRelease(new AbortController().signal, progress => {
  latestProgress = progress
  listener?.onProgress(progress)
}, {
  refresh,
  onAssets: assets => {
    latestAssets = assets
    listener?.onAssets?.(assets)
  },
})
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

void import('./pagesApp.tsx').then(({ mount }) => mount(loadInitialRelease)).catch(() => {
  document.querySelector('#initial-status')!.textContent = 'Could not open Earth Connect. Reload to retry.'
})
