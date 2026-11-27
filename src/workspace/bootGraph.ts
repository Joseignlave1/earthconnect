import { endStartupSpan, type StartupPhase, startStartupSpan } from './startupTiming.ts'

export type BootTier = 'critical' | 'background'

export type BootContext = {
  readonly signal: AbortSignal
  get<T>(name: string): T
}

export type BootTaskSpec<T> = {
  after?: readonly string[]
  tier?: BootTier
  phase?: StartupPhase
  detail?: string | ((value: T) => string | undefined)
  /** Record this task in startupTiming (defaults to true for critical tasks). */
  recordTiming?: boolean
  run: (ctx: BootContext) => Promise<T> | T
}

/**
 * Lightweight dependency graph for startup tasks.
 * - Critical tasks run maximally in parallel as soon as their `after` dependencies resolve.
 * - Background tasks remain parked until `startBackground()` is called (after reveal).
 * - Critical tasks automatically report phase-grouped timing spans to `startupTiming.ts`.
 */
export class BootGraph {
  private readonly controller = new AbortController()
  private readonly signal: AbortSignal
  private readonly defaultPhase: StartupPhase
  private readonly tasks = new Map<string, { tier: BootTier; promise: Promise<unknown> }>()
  private readonly values = new Map<string, unknown>()
  private readonly backgroundGate: Promise<void>
  private releaseBackground!: () => void
  private backgroundStarted = false

  constructor(parentSignal?: AbortSignal, defaultPhase: StartupPhase = 'prepare') {
    this.signal = parentSignal ? AbortSignal.any([parentSignal, this.controller.signal]) : this.controller.signal
    this.defaultPhase = defaultPhase
    this.backgroundGate = new Promise<void>(resolve => {
      this.releaseBackground = resolve
    })
  }

  task<T>(name: string, spec: BootTaskSpec<T>): Promise<T> {
    if (this.tasks.has(name)) throw new Error(`Duplicate boot task "${name}".`)
    const tier: BootTier = spec.tier ?? 'critical'
    const phase: StartupPhase = spec.phase ?? this.defaultPhase
    const recordTiming = spec.recordTiming ?? tier === 'critical'
    const deps = spec.after ?? []
    for (const dep of deps) {
      if (!this.tasks.has(dep)) {
        throw new Error(`Boot task "${name}" depends on unregistered task "${dep}".`)
      }
    }

    const promise = (async (): Promise<T> => {
      if (tier === 'background') await this.backgroundGate
      this.signal.throwIfAborted()
      if (deps.length > 0) {
        await Promise.all(deps.map(dep => this.tasks.get(dep)!.promise))
      }
      this.signal.throwIfAborted()
      const ctx: BootContext = {
        signal: this.signal,
        get: <R>(depName: string) => {
          if (!this.values.has(depName)) {
            throw new Error(`Boot task "${name}" read unresolved dependency "${depName}".`)
          }
          return this.values.get(depName) as R
        },
      }
      const initialDetail = typeof spec.detail === 'string' ? spec.detail : undefined
      if (recordTiming) startStartupSpan(name, phase, initialDetail)
      try {
        const result = await spec.run(ctx)
        this.values.set(name, result)
        if (recordTiming) {
          const finalDetail = typeof spec.detail === 'function' ? spec.detail(result) : initialDetail
          endStartupSpan(name, finalDetail)
        }
        return result
      } catch (err) {
        if (recordTiming) endStartupSpan(name)
        throw err
      }
    })()

    // Avoid unhandled rejection noise when callers await `ready()` instead of individual tasks.
    void promise.catch(() => undefined)
    this.tasks.set(name, { tier, promise })
    return promise
  }

  /** Wait until every registered task in the given tier finishes. */
  async ready(tier: BootTier = 'critical'): Promise<void> {
    const matching = [...this.tasks.values()].filter(t => t.tier === tier).map(t => t.promise)
    await Promise.all(matching)
    this.signal.throwIfAborted()
  }

  /** Read a resolved task value after awaiting `ready()` or the task promise. */
  get<T>(name: string): T {
    if (!this.values.has(name)) throw new Error(`Boot task "${name}" has not resolved yet.`)
    return this.values.get(name) as T
  }

  /** Unblock background-tier tasks (typically after first-frame reveal). */
  startBackground() {
    if (this.backgroundStarted || this.signal.aborted) return
    this.backgroundStarted = true
    this.releaseBackground()
  }

  abort(reason?: unknown) {
    this.controller.abort(reason)
  }
}

export function createBootGraph(signal?: AbortSignal, defaultPhase: StartupPhase = 'prepare') {
  return new BootGraph(signal, defaultPhase)
}
