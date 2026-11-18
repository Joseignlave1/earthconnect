import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { loadWorkspaceRelease, type DownloadProgress } from './workspaceRelease.ts'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import './WorkspaceStartup.css'

const decimal = new Intl.NumberFormat(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })
const mb = (bytes: number) => decimal.format(bytes / 1_000_000)

export function WorkspaceStartup({ open, embedded, loadInitialRelease }: {
  open(assets: WorkspaceAssets, onReady: () => void): Promise<ReactNode>
  embedded?: () => WorkspaceAssets
  loadInitialRelease?: typeof loadWorkspaceRelease
}) {
  const [attempt, setAttempt] = useState(0)
  const [progress, setProgress] = useState<DownloadProgress>()
  const [unsaved, setUnsaved] = useState(false)
  const [authorized, setAuthorized] = useState(false)
  const [opening, setOpening] = useState(false)
  const [error, setError] = useState('')
  const [workspace, setWorkspace] = useState<{ id: number; view: ReactNode }>()
  const [ready, setReady] = useState(false)
  const [slow, setSlow] = useState(false)
  const nextView = useRef(0)
  const openView = useCallback(async (assets: WorkspaceAssets, signal: AbortSignal) => {
    const id = ++nextView.current
    const isCurrent = () => id === nextView.current && !signal.aborted
    setReady(false)
    setWorkspace(undefined)
    setError('')
    let reported = false
    const onReady = () => {
      if (reported) return
      reported = true
      // Leave a full paint opportunity for Monaco, the preview controls and
      // the already completed WebGPU canvas before removing the overlay.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (isCurrent()) setReady(true)
      }))
    }
    try {
      const view = await open(assets, onReady)
      if (isCurrent()) setWorkspace({ id, view })
    } catch (cause) {
      if (isCurrent()) setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [open])

  const revealed = authorized && ready && !error
  const finalizing = !!progress?.total && progress.downloaded === progress.total

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
      // Let the browser paint the archive splash before decoding embedded files.
      if (embedded) {
        await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
        signal.throwIfAborted()
        setOpening(true)
        setAuthorized(true)
        const assets = embedded()
        signal.addEventListener('abort', assets.dispose, { once: true })
        await openView(assets, signal)
        return
      }
      const load = attempt === 0 && loadInitialRelease ? loadInitialRelease : loadWorkspaceRelease
      const release = await load(signal, state => {
        if (!signal.aborted) setProgress(state)
      }, {
        refresh,
        onAssets: assets => {
          if (!signal.aborted) void openView(assets, signal)
        },
      })
      signal.throwIfAborted()
      if (release.saved) {
        setOpening(true)
        setAuthorized(true)
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
    setWorkspace(undefined)
    setReady(false)
    setSlow(false)
    setError('')
    setProgress(undefined)
    setUnsaved(false)
    setAuthorized(false)
    setOpening(false)
    setAttempt(value => value + 1)
  }

  const offline = progress?.fallback
  const transferring = !!progress?.total && !opening && !unsaved && !error
  let status = 'Checking Earth data…'
  if (error) status = 'Could not open Earth Connect.'
  else if (slow) status = 'Earth Connect is taking longer to open.'
  else if (opening) status = embedded ? 'Opening embedded data…' :
    unsaved || progress?.total ? 'Opening Earth Connect…' : 'Opening saved data…'
  else if (unsaved) status = 'Your browser could not save the Earth data.'
  else if (embedded) status = 'Opening embedded data…'
  else if (finalizing) status = 'Verifying and saving Earth data…'
  else if (progress?.total) status = `${mb(progress.downloaded)} / ${mb(progress.total)} MB downloaded`
  else if (progress) status = 'Opening saved data…'
  return (
    <>
      {workspace && <div className="workspace-content" key={workspace.id} inert={!revealed}>{workspace.view}</div>}
      <main className="workspace-startup" aria-label="Earth Connect startup" inert={revealed}
        data-revealed={revealed || undefined} data-offline={offline || undefined}>
        <div className="workspace-startup-content">
          <div className="brand" aria-label="Earth Connect">
            <span className="brand-symbol" aria-hidden="true"><i /></span>
            <span>EARTH</span> CONNECT
          </div>
          {!error && (!unsaved || opening) && <progress aria-label="Earth data download" max={progress?.total || 1}
            value={progress ? (progress.total ? progress.downloaded : 1) : undefined} />}
          <p role={error || (unsaved && !opening) ? 'alert' : 'status'} aria-live={transferring ? 'off' : 'polite'}>{status}</p>
          {slow && !error && <button type="button" onClick={() => window.location.reload()}>Retry</button>}
          {error && <><p className="workspace-startup-error">{error}</p><button type="button" onClick={retry}>Retry</button></>}
          {unsaved && !opening && !error && <button type="button" onClick={() => {
            setOpening(true)
            setAuthorized(true)
          }}>Continue without saving</button>}
        </div>
      </main>
    </>
  )
}
