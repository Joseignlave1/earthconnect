import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { ConstellationLogo } from '../hero/ConstellationLogo.tsx'
import { WEBGPU_UNAVAILABLE_MESSAGE } from '../shader-test/surface-device.ts'
import { endStartupSpan, reportStartupTiming, startStartupSpan } from './startupTiming.ts'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import { type DownloadProgress, loadWorkspaceRelease } from './workspaceRelease.ts'
import './WorkspaceStartup.css'

const decimal = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })
const mb = (bytes: number) => decimal.format(bytes / 1_000_000)

export function WorkspaceStartup({
  open,
  embedded,
  loadInitialRelease,
}: {
  open(assets: WorkspaceAssets, onReady: (ready?: boolean) => void, signal: AbortSignal): Promise<ReactNode>
  embedded?: () => WorkspaceAssets
  loadInitialRelease?: typeof loadWorkspaceRelease
}) {
  const [attempt, setAttempt] = useState(0)
  const [progress, setProgress] = useState<DownloadProgress>()
  const [unsaved, setUnsaved] = useState(false)
  const [opening, setOpening] = useState(false)
  const [error, setError] = useState('')
  const [workspace, setWorkspace] = useState<{ id: number; view: ReactNode }>()
  const [ready, setReady] = useState(false)
  const [transitioning, setTransitioning] = useState(false)
  const [logoCycle, setLogoCycle] = useState(0)
  const [logoReady, setLogoReady] = useState(false)
  const [logoDismissed, setLogoDismissed] = useState(false)
  const [slow, setSlow] = useState(false)
  const nextView = useRef(0)
  const readyToken = useRef(0)
  const openView = useCallback(
    async (assets: WorkspaceAssets, signal: AbortSignal) => {
      const id = ++nextView.current
      const isCurrent = () => id === nextView.current && !signal.aborted
      setReady(false)
      setTransitioning(false)
      setWorkspace(undefined)
      setError('')
      let initialReported = false
      const onReady = (isReady = true) => {
        if (!isCurrent()) return
        const token = ++readyToken.current
        if (!isReady) {
          setTransitioning(true)
          setReady(false)
          setLogoReady(false)
          setLogoDismissed(false)
          setLogoCycle(value => value + 1)
          return
        }
        // Leave a full paint opportunity for Monaco, the preview controls and
        // the already completed WebGPU canvas before removing the overlay.
        if (!initialReported) startStartupSpan('reveal', 'settle')
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!isCurrent() || token !== readyToken.current) return
            if (!initialReported) {
              initialReported = true
              endStartupSpan('reveal')
              reportStartupTiming()
            }
            setTransitioning(false)
            setReady(true)
          }),
        )
      }
      try {
        const view = await open(assets, onReady, signal)
        if (isCurrent()) setWorkspace({ id, view })
      } catch (cause) {
        if (isCurrent()) setError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [open],
  )

  const revealed = opening && ready && !error
  const finalizing = !!progress?.total && progress.downloaded === progress.total
  // Keep the WebGPU logo alive through the 220ms fade-out so it never snaps back to blank,
  // then tear down its GPU resources & stop rAF completely once hidden.
  const handleLogoReady = useCallback(() => setLogoReady(true), [])
  useEffect(() => {
    if (logoReady || revealed || error) return
    const safety = window.setTimeout(() => setLogoReady(true), 2000)
    return () => window.clearTimeout(safety)
  }, [logoReady, revealed, error])
  useEffect(() => {
    if (!revealed) return
    const timer = window.setTimeout(() => setLogoDismissed(true), 280)
    return () => window.clearTimeout(timer)
  }, [revealed])
  const logoMounted = !revealed || !logoDismissed
  const startupContentVisible = logoReady || Boolean(error)

  useEffect(() => {
    if ((!opening && !finalizing) || revealed || error || unsaved) return
    let timeout: number | undefined
    function watchVisibleStartup() {
      window.clearTimeout(timeout)
      if (!document.hidden) timeout = window.setTimeout(() => setSlow(true), 120_000)
    }
    watchVisibleStartup()
    document.addEventListener('visibilitychange', watchVisibleStartup)
    return () => {
      window.clearTimeout(timeout)
      document.removeEventListener('visibilitychange', watchVisibleStartup)
    }
  }, [opening, finalizing, revealed, error, unsaved])

  useEffect(() => {
    const controller = new AbortController()
    const { signal } = controller
    const refresh = new URLSearchParams(window.location.search).get('refreshCache') === '1'
    async function start() {
      if (!navigator.gpu) {
        setError(WEBGPU_UNAVAILABLE_MESSAGE)
        return
      }
      // Let the browser paint the archive splash before decoding embedded files.
      if (embedded) {
        await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
        signal.throwIfAborted()
        setOpening(true)
        const assets = embedded()
        signal.addEventListener('abort', assets.dispose, { once: true })
        await openView(assets, signal)
        return
      }
      const load = attempt === 0 && loadInitialRelease ? loadInitialRelease : loadWorkspaceRelease
      let lastProgressFlush = 0
      let pendingProgress: DownloadProgress | undefined
      let progressTimer: number | undefined
      const flushProgress = (state: DownloadProgress) => {
        pendingProgress = undefined
        lastProgressFlush = performance.now()
        if (!signal.aborted) setProgress(state)
      }
      const onProgress = (state: DownloadProgress) => {
        if (signal.aborted) return
        const now = performance.now()
        const isEdge = state.downloaded === 0 || (state.total > 0 && state.downloaded >= state.total) || state.fallback
        if (isEdge || now - lastProgressFlush >= 50) {
          window.clearTimeout(progressTimer)
          progressTimer = undefined
          flushProgress(state)
        } else {
          pendingProgress = state
          if (progressTimer === undefined) {
            progressTimer = window.setTimeout(() => {
              progressTimer = undefined
              if (pendingProgress) flushProgress(pendingProgress)
            }, 50)
          }
        }
      }
      signal.addEventListener('abort', () => window.clearTimeout(progressTimer), { once: true })
      const release = await load(signal, onProgress, {
        refresh,
        onAssets: assets => {
          if (!signal.aborted) void openView(assets, signal)
        },
      })
      signal.throwIfAborted()
      if (release.saved) {
        setOpening(true)
      } else {
        setUnsaved(true)
      }
    }
    void start().catch(cause => {
      if (signal.aborted) return
      nextView.current++
      setWorkspace(undefined)
      setError(cause instanceof Error ? cause.message : String(cause))
    })
    return () => controller.abort()
  }, [attempt, embedded, loadInitialRelease, openView])

  function retry() {
    nextView.current++
    readyToken.current++
    setWorkspace(undefined)
    setReady(false)
    setTransitioning(false)
    setLogoReady(false)
    setLogoDismissed(false)
    setLogoCycle(value => value + 1)
    setSlow(false)
    setError('')
    setProgress(undefined)
    setUnsaved(false)
    setOpening(false)
    setAttempt(value => value + 1)
  }

  const offline = progress?.fallback
  const transferring = !!progress?.total && !opening && !unsaved && !error
  let status = 'Checking Earth data…'
  if (error) status = 'Could not open Earth Connect.'
  else if (slow) status = 'Earth Connect is taking longer to open.'
  else if (transitioning) status = 'Opening Earth Connect…'
  else if (opening)
    status = embedded
      ? 'Opening embedded data…'
      : unsaved || progress?.total
        ? 'Opening Earth Connect…'
        : 'Opening saved data…'
  else if (unsaved) status = 'Your browser could not save the Earth data.'
  else if (embedded) status = 'Opening embedded data…'
  else if (finalizing) status = 'Verifying and saving Earth data…'
  else if (progress?.total) status = `${mb(progress.downloaded)} / ${mb(progress.total)} MB downloaded`
  else if (progress) status = 'Opening saved data…'
  return (
    <>
      {workspace && (
        <div className="workspace-content" key={workspace.id} inert={!revealed}>
          {workspace.view}
        </div>
      )}
      <main
        className="workspace-startup"
        aria-label="Earth Connect startup"
        inert={revealed}
        data-revealed={revealed || undefined}
        data-offline={offline || undefined}
      >
        <div className="workspace-startup-content" data-ready={startupContentVisible || undefined}>
          {logoMounted && <ConstellationLogo key={logoCycle} onReady={handleLogoReady} />}
          {!error && (!unsaved || opening) && (
            <progress
              aria-label="Earth data download"
              max={progress?.total || 1}
              value={progress ? (progress.total ? progress.downloaded : 1) : undefined}
            />
          )}
          <p role={error || (unsaved && !opening) ? 'alert' : 'status'} aria-live={transferring ? 'off' : 'polite'}>
            {status}
          </p>
          {slow && !error && (
            <button type="button" onClick={() => window.location.reload()}>
              Retry
            </button>
          )}
          {error && (
            <>
              <p className="workspace-startup-error">{error}</p>
              <button type="button" onClick={retry}>
                Retry
              </button>
            </>
          )}
          {unsaved && !opening && !error && (
            <button
              type="button"
              onClick={() => {
                setOpening(true)
              }}
            >
              Continue without saving
            </button>
          )}
        </div>
      </main>
    </>
  )
}
