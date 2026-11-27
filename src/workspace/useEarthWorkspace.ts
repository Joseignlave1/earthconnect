import { useCallback, useEffect, useEffectEvent, useRef, useState, type RefObject } from 'react'
import {
  freezePlanetaryShaderPoint,
  type PlanetaryShaderData,
  type PlanetaryShaderSample,
} from '../earth-score/index.ts'
import { ScoreShaderError } from '../earth-score/custom-shader.ts'
import { friendlyApiError } from '../earthIndexApi.ts'
import type { BodyId } from '../explorerData.ts'
import { SurfaceRenderer } from '../shader-test/renderer.ts'
import { earthTilePath } from '../shader-test/earth-tiles.ts'
import { planetTilePath, type PlanetTileBody } from './planetTiles.ts'
import { loadWorkspaceTilePack, scoreManifestUrl, type WorkspaceAssets } from './workspaceAssets.ts'
import type { SolarFocus } from './solarCamera.ts'
import { SolarScene } from './solarScene.ts'
import { canonicalPoint, coordinatesValid, pointKey } from './planetaryPointCache.ts'
import { createEarthWorkspaceResources, type EarthWorkspaceResources } from './earthWorkspaceResources.ts'
import type { PlanetaryCall } from './planetaryShaderAnalysis.ts'
import { endStartupSpan, startStartupSpan } from './startupTiming.ts'

type Reference = { body: BodyId; latitude: number; longitude: number }
type Inspection = { latitude: number; longitude: number; score: number | null; elevation: number | null }
type Job = { source: string }
type World = ReturnType<EarthWorkspaceResources['acquireWorld']>
type Commands = {
  place: (reference: Reference) => void
  pickSourceAt: (clientX: number, clientY: number) => Reference | undefined
  freezePoint: (reference: Reference, signal: AbortSignal) => Promise<{ code: string; partial: boolean }>
  retry: () => void
  setProjection: (projection: 'globe' | 'plane') => void
  focus: (body: SolarFocus) => void
  runShader: (source: string) => void
  cancelShader: () => void
}
type WorkspaceState = {
  busy: boolean
  status: string
  error: string
  inspection: Inspection | null
  ready: boolean
  frameSource?: string
  surfaceFailed: boolean
  worldSettled: boolean
  mapName: string
  shaderActive: boolean
  shaderPartial: boolean
  shaderDiagnostics: ScoreShaderError['diagnostics']
  shaderSource: string | null
}
const initial: WorkspaceState = {
  busy: false,
  status: 'Preparing the Earth surface…',
  error: '',
  inspection: null,
  ready: false,
  surfaceFailed: false,
  worldSettled: false,
  mapName: '',
  shaderActive: false,
  shaderPartial: false,
  shaderDiagnostics: [],
  shaderSource: null,
}
const coordinate = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 })
function referenceName({ body, latitude, longitude }: Reference) {
  return `${body === 'moon' ? 'Moon' : 'Mars'} · ${coordinate.format(latitude)}°, ${coordinate.format(longitude)}°`
}
function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

const analyzeShader = import.meta.env.VITE_WORKSPACE_STANDALONE
  ? undefined
  : async (source: string, signal: AbortSignal) => {
      const { analyzePlanetaryShader, PlanetaryAnalysisError } = await import('./planetaryShaderAnalysis.ts')
      signal.throwIfAborted()
      try {
        return await analyzePlanetaryShader(source, signal)
      } catch (error) {
        if (error instanceof PlanetaryAnalysisError) throw new ScoreShaderError(error.diagnostics)
        throw error
      }
    }

class PointFetchError extends ScoreShaderError {}
const defaultInitialView = { latitude: 36, longitude: 95 }

function pointDiagnostic(call: PlanetaryCall, message: string) {
  return { line: call.line, column: call.column, length: call.length, severity: 'error' as const, message }
}

export function useEarthWorkspace(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  assets: WorkspaceAssets,
  onSourcePick?: (body: BodyId, latitude: number, longitude: number) => void,
  resources?: EarthWorkspaceResources,
  initialView = defaultInitialView,
) {
  const [state, setState] = useState(initial)
  const commands = useRef<Commands>(null)
  const place = useCallback<Commands['place']>(reference => commands.current?.place(reference), [])
  const pickSourceAt = useCallback<Commands['pickSourceAt']>((...args) => commands.current?.pickSourceAt(...args), [])
  const freezePoint = useCallback<Commands['freezePoint']>(
    (...args) =>
      commands.current?.freezePoint(...args) ?? Promise.reject(new Error('The point sampler is still starting.')),
    [],
  )
  const retry = useCallback(() => commands.current?.retry(), [])
  const setProjection = useCallback<Commands['setProjection']>(
    projection => commands.current?.setProjection(projection),
    [],
  )
  const focus = useCallback<Commands['focus']>(body => commands.current?.focus(body), [])
  const runShader = useCallback<Commands['runShader']>(source => commands.current?.runShader(source), [])
  const cancelShader = useCallback(() => commands.current?.cancelShader(), [])
  const reportSourcePick = useEffectEvent((body: BodyId, point: Pick<Reference, 'latitude' | 'longitude'>) => {
    onSourcePick?.(body, point.latitude, point.longitude)
  })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const lifecycle = new AbortController()
    const shared = resources ?? createEarthWorkspaceResources(undefined, lifecycle.signal)
    let renderer: SurfaceRenderer | undefined
    let creating = false
    let surfaceFailed = false
    let world: World | undefined
    let current: { controller: AbortController; job: Job } | undefined
    let failed: Job | undefined
    let resume: Job | undefined
    let shownJob: Job | undefined
    let wasRendered = false
    let completedFrame: string | undefined
    let pendingFrame: string | undefined
    let displayedMap = ''
    let projection: 'globe' | 'plane' = 'globe'
    let focusBody: SolarFocus = 'earth'
    const sourcePoints = new Map<BodyId, Reference>()
    const { readPoint } = shared

    function update(changes: Partial<WorkspaceState>) {
      if (!lifecycle.signal.aborted) setState(previous => ({ ...previous, ...changes }))
    }

    function reportFailure(message: string) {
      update({ error: message, status: '' })
    }

    function placePoint(reference: Reference) {
      const { body, latitude, longitude } = reference
      if (!coordinatesValid(latitude, longitude)) return
      sourcePoints.set(body, reference)
      renderer?.setSourcePoint(body, reference)
    }

    async function resolveShaderPoints(
      calls: readonly PlanetaryCall[],
      signal: AbortSignal,
    ): Promise<PlanetaryShaderData | undefined> {
      if (!calls.length) return
      const slots = new Map<string, number>()
      const unique: PlanetaryCall[] = []
      for (const call of calls) {
        const key = pointKey(canonicalPoint(call))
        if (!slots.has(key)) {
          slots.set(key, unique.length)
          unique.push(call)
        }
      }
      update({
        status: `Resolving ${unique.length === 1 ? '1 planetary point' : `${unique.length} planetary points`}…`,
      })
      const batch = new AbortController()
      const batchSignal = AbortSignal.any([signal, batch.signal])
      const samples: PlanetaryShaderSample[] = []
      let next = 0
      try {
        await Promise.all(
          Array.from({ length: Math.min(4, unique.length) }, async () => {
            while (next < unique.length) {
              batchSignal.throwIfAborted()
              const index = next++
              const call = unique[index]
              const point = canonicalPoint(call)
              try {
                const sample = await readPoint(point, batchSignal)
                samples[index] = { ...point, measurements: sample.measurements, partial: sample.partial }
              } catch (error) {
                batchSignal.throwIfAborted()
                throw new PointFetchError([
                  pointDiagnostic(call, `${referenceName(point)}: ${friendlyApiError(error)}`),
                ])
              }
            }
          }),
        )
        return {
          samples,
          calls: calls.map(call => ({
            body: call.body,
            offset: call.offset,
            length: call.endOffset - call.offset,
            slot: slots.get(pointKey(canonicalPoint(call)))!,
          })),
        }
      } finally {
        batch.abort()
      }
    }

    function releaseWorld() {
      const previous = world
      world = undefined
      previous?.release()
    }

    function prepareWorld(device: GPUDevice) {
      if (world?.device === device) return world
      releaseWorld()
      const entry = shared.acquireWorld(device, {
        manifestUrl: scoreManifestUrl,
        readAsset: assets.readAsset,
      })
      world = entry
      update({ worldSettled: false, ...(!current && { status: 'Loading Earth data for live scoring…' }) })
      void entry.loading.then(
        () => {
          if (world === entry) update({ worldSettled: true, ...(!current && { status: '' }) })
        },
        error => {
          if (world !== entry || lifecycle.signal.aborted) return
          releaseWorld()
          update({ worldSettled: true })
          if (!current) reportFailure(`Could not load the Earth dataset. ${errorMessage(error)}`)
        },
      )
      return entry
    }

    function mapStatus() {
      const mapName = canvas!.dataset.scoreName ?? ''
      if (displayedMap === mapName) return
      displayedMap = mapName
      update({ mapName })
    }

    function rendered() {
      if (!wasRendered) {
        wasRendered = true
        update({ ready: true })
      }
      const source = canvas!.dataset.frameSource
      const size = canvas!.dataset.renderSize
      const { clientWidth, clientHeight } = canvas!
      const device = renderer?.computeDevice
      if (!source || !device) return
      const frame = `${source}/${size}`
      if (frame === completedFrame || frame === pendingFrame) return
      pendingFrame = frame
      void device.queue
        .onSubmittedWorkDone()
        .then(
          () => {
            if (
              renderer?.computeDevice !== device ||
              pendingFrame !== frame ||
              canvas!.dataset.frameSource !== source ||
              canvas!.dataset.renderSize !== size ||
              canvas!.clientWidth !== clientWidth ||
              canvas!.clientHeight !== clientHeight
            )
              return
            completedFrame = frame
            update({ frameSource: source })
          },
          error => {
            if (renderer?.computeDevice === device) {
              reportFailure(errorMessage(error))
              update({ surfaceFailed: true })
            }
          },
        )
        .finally(() => {
          if (pendingFrame === frame) pendingFrame = undefined
        })
    }

    function deviceChanged() {
      if (!renderer || lifecycle.signal.aborted) return
      const device = renderer.computeDevice
      if (!device) {
        resume = current?.job ?? resume ?? shownJob
        current?.controller.abort()
        current = undefined
        releaseWorld()
        wasRendered = false
        completedFrame = pendingFrame = undefined
        displayedMap = ''
        update({
          ready: false,
          frameSource: undefined,
          surfaceFailed: false,
          worldSettled: false,
          mapName: '',
          busy: Boolean(resume),
          status: 'Recovering the Earth surface…',
          error: '',
        })
        return
      }
      prepareWorld(device)
      mapStatus()
      if (resume) {
        const job = resume
        resume = undefined
        void run(job)
      }
    }

    async function startRenderer() {
      if (creating) return
      creating = true
      surfaceFailed = false
      try {
        renderer?.destroy()
        renderer = undefined
        delete canvas!.dataset.scoreName
        delete canvas!.dataset.source
        delete canvas!.dataset.frameSource
        const loadTilePack = (path: string, signal: AbortSignal) => loadWorkspaceTilePack(assets, path, signal)
        const instance = await SurfaceRenderer.create(
          canvas!,
          (selection, score) => {
            const inspection = selection && {
              latitude: selection.latitude,
              longitude: selection.longitude,
              elevation: selection.elevation ?? null,
              score,
            }
            update({ inspection })
          },
          error => {
            reportFailure(error.message)
            update({ surfaceFailed: true })
            if (!renderer?.computeDevice) update({ busy: false })
          },
          mapStatus,
          {
            signal: lifecycle.signal,
            acquireDevice: shared.acquireDevice,
            solar: {
              createScene: SolarScene.create,
              assetUrl: assets.url,
              loadTerrain: shared.acquireTerrain,
              loadPlanetTilePack: (body: PlanetTileBody, packId: number, signal: AbortSignal) =>
                loadTilePack(planetTilePath(body, packId), signal),
              onSourcePick: (body, point) => reportSourcePick(body, point),
            },
            assets: assets.url,
            loadTerrain: shared.acquireTerrain,
            loadEarthTilePack: (packId, signal) => loadTilePack(earthTilePath(packId), signal),
            onDevice: prepareWorld,
            initialView,
          },
        )
        if (lifecycle.signal.aborted) {
          instance.destroy()
          return
        }
        renderer = instance
        instance.focusSolarBody(focusBody)
        instance.setProjection(projection)
        for (const [body, point] of sourcePoints) instance.setSourcePoint(body, point)
        deviceChanged()
      } catch (error) {
        releaseWorld()
        if (!lifecycle.signal.aborted) {
          surfaceFailed = true
          console.error('Could not initialize WebGPU surface:', error)
          reportFailure(errorMessage(error))
          update({ busy: false, surfaceFailed: true })
        }
      } finally {
        creating = false
      }
    }

    async function run(job: Job) {
      current?.controller.abort()
      const controller = new AbortController()
      const { signal } = controller
      const attempt = { controller, job }
      const instance = renderer
      const device = instance?.computeDevice
      update({ shaderSource: job.source, shaderDiagnostics: [] })
      if (!instance || !device) {
        resume = job
        if (!surfaceFailed) update({ busy: true, error: '', status: 'Waiting for WebGPU…' })
        return
      }
      resume = undefined
      current = attempt
      failed = undefined
      update({ busy: true, error: '' })
      const entry = prepareWorld(device)
      try {
        const calls = (await analyzeShader?.(job.source, signal)) ?? []
        signal.throwIfAborted()
        const points = await resolveShaderPoints(calls, signal)
        signal.throwIfAborted()
        const earth = await entry.loading.catch(error => {
          throw new Error(`Could not load the Earth dataset. ${errorMessage(error)}`)
        })
        signal.throwIfAborted()
        await entry.publish(async () => {
          signal.throwIfAborted()
          startStartupSpan('first-score')
          update({ status: 'Scoring the Earth grid…' })
          await earth.dispatchShader(job.source, signal, points)
          signal.throwIfAborted()
          await instance.setComputedScores(
            earth,
            {
              name: 'WGSL score map',
              description: 'Earth cells scored with the WGSL query.',
            },
            signal,
          )
          signal.throwIfAborted()
          endStartupSpan('first-score')
          shownJob = job
          update({
            shaderActive: true,
            shaderPartial: Boolean(points?.samples.some(point => point.partial)),
            status: '',
          })
        })
      } catch (error) {
        if (!signal.aborted) {
          const shaderDiagnostics: ScoreShaderError['diagnostics'] =
            error instanceof ScoreShaderError
              ? error.diagnostics
              : [{ line: 0, column: 0, length: 0, severity: 'error', message: errorMessage(error) }]
          if (error instanceof PointFetchError) {
            failed = job
            reportFailure('Could not resolve a planetary point in the query.')
          }
          update({ shaderDiagnostics, status: '' })
        }
      } finally {
        if (current === attempt) {
          current = undefined
          update({ busy: false })
        }
        controller.abort()
      }
    }

    commands.current = {
      place: placePoint,
      async freezePoint(reference, signal) {
        const sample = await readPoint(reference, signal).catch(error => {
          signal.throwIfAborted()
          throw new Error(friendlyApiError(error))
        })
        signal.throwIfAborted()
        return {
          code: freezePlanetaryShaderPoint({
            ...canonicalPoint(reference),
            measurements: sample.measurements,
            partial: sample.partial,
          }),
          partial: sample.partial,
        }
      },
      focus(body) {
        focusBody = body
        if (body !== 'earth') projection = 'globe'
        renderer?.focusSolarBody(body)
      },
      setProjection(value) {
        projection = value
        renderer?.setProjection(value)
      },
      pickSourceAt: (x, y) => renderer?.pickSourceAt(x, y),
      runShader(source) {
        void run({ source })
      },
      cancelShader() {
        if (failed) {
          failed = undefined
          update({ error: '' })
        }
        if (resume) {
          resume = undefined
          update({ busy: false, status: '' })
        }
        if (current) {
          current.controller.abort()
          current = undefined
          update({ busy: false, status: '' })
        }
      },
      retry() {
        if (!renderer?.computeDevice) {
          resume ??= failed ?? shownJob
          wasRendered = false
          completedFrame = pendingFrame = undefined
          update({
            ready: false,
            frameSource: undefined,
            surfaceFailed: false,
            worldSettled: false,
            mapName: '',
            error: '',
            status: 'Preparing the Earth surface…',
          })
          void startRenderer()
        } else if (failed) {
          void run(failed)
        } else {
          update({ error: '' })
          renderer.retrySolarTerrain()
          prepareWorld(renderer.computeDevice)
        }
      },
    }
    canvas.addEventListener('renderstatus', rendered)
    canvas.addEventListener('computedevicechange', deviceChanged)
    void startRenderer()

    return () => {
      commands.current = null
      canvas.removeEventListener('renderstatus', rendered)
      canvas.removeEventListener('computedevicechange', deviceChanged)
      lifecycle.abort()
      current?.controller.abort()
      releaseWorld()
      renderer?.destroy()
    }
  }, [canvasRef, assets, resources, initialView])

  return { ...state, place, pickSourceAt, freezePoint, retry, setProjection, focus, runShader, cancelShader }
}
